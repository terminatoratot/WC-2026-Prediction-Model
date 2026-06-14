#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import poisson

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except Exception:
    CatBoostClassifier = None
    CatBoostRegressor = None


RNG = np.random.default_rng(7)
MEN_WORLD_CUP_FINAL_YEARS = {2002, 2006, 2010, 2014, 2018, 2022}
WOMEN_WORLD_CUP_FINAL_YEARS = {2003, 2007, 2011, 2015, 2019, 2023}
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

TEAM_ALIASES = {
    "usa": "United States",
    "united states of america": "United States",
    "iran": "IR Iran",
    "south korea": "Korea Republic",
    "korea republic": "Korea Republic",
    "north korea": "Korea DPR",
    "ivory coast": "Côte d'Ivoire",
    "cote d'ivoire": "Côte d'Ivoire",
    "côte d’ivoire": "Côte d'Ivoire",
    "czech republic": "Czechia",
    "türkiye": "Turkey",
    "curacao": "Curaçao",
    "serbia and montenegro": "Serbia and Montenegro",
}


def canon_team(x: str) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    key = s.lower().replace("&", "and")
    return TEAM_ALIASES.get(key, s)


def poisson_score_matrix(lam_a: float, lam_b: float, max_goals: int = 10) -> Dict[Tuple[int, int], float]:
    pa = poisson.pmf(np.arange(max_goals + 1), lam_a)
    pb = poisson.pmf(np.arange(max_goals + 1), lam_b)
    mat = {(i, j): float(pa[i] * pb[j]) for i in range(max_goals + 1) for j in range(max_goals + 1)}
    total = sum(mat.values())
    return {k: v / total for k, v in mat.items()}


def result_probs(score_probs: Dict[Tuple[int, int], float]) -> Dict[str, float]:
    a = sum(p for (i, j), p in score_probs.items() if i > j)
    d = sum(p for (i, j), p in score_probs.items() if i == j)
    b = sum(p for (i, j), p in score_probs.items() if i < j)
    return {"team_a_win": a, "draw": d, "team_b_win": b}


def apply_dixon_coles_adjustment(
    score_probs: Dict[Tuple[int, int], float],
    lam_a: float,
    lam_b: float,
    rho: float = -0.08,
) -> Dict[Tuple[int, int], float]:
    """Apply Dixon-Coles dependency corrections to low scorelines."""
    adjusted = dict(score_probs)
    for (i, j), p in score_probs.items():
        if (i, j) == (0, 0):
            tau = 1.0 - lam_a * lam_b * rho
        elif (i, j) == (0, 1):
            tau = 1.0 + lam_a * rho
        elif (i, j) == (1, 0):
            tau = 1.0 + lam_b * rho
        elif (i, j) == (1, 1):
            tau = 1.0 - rho
        else:
            continue
        adjusted[(i, j)] = p * max(tau, 1e-6)
    total = sum(adjusted.values())
    return {k: v / total for k, v in adjusted.items()}


def temperature_smooth_result_probs(res: Dict[str, float], temperature: float = 1.08) -> Dict[str, float]:
    """Conservative probability smoothing.

    temperature > 1 flattens overconfident result probabilities.
    """
    if temperature <= 0:
        return res
    arr = np.array([res["team_a_win"], res["draw"], res["team_b_win"]], dtype=float)
    arr = np.clip(arr, 1e-12, 1.0)
    arr = arr ** (1.0 / temperature)
    arr = arr / arr.sum()
    return {"team_a_win": float(arr[0]), "draw": float(arr[1]), "team_b_win": float(arr[2])}


def reweight_score_matrix_to_results(
    score_probs: Dict[Tuple[int, int], float],
    target: Dict[str, float],
) -> Dict[Tuple[int, int], float]:
    """Keep exact-score probabilities coherent with final W/D/L probabilities."""
    current = result_probs(score_probs)
    adjusted = {}
    for (goals_a, goals_b), probability in score_probs.items():
        if goals_a > goals_b:
            outcome = "team_a_win"
        elif goals_a == goals_b:
            outcome = "draw"
        else:
            outcome = "team_b_win"
        adjusted[(goals_a, goals_b)] = (
            probability * target[outcome] / max(current[outcome], 1e-12)
        )
    total = sum(adjusted.values())
    return {key: value / total for key, value in adjusted.items()}


def build_year_recency_weights(
    frame: pd.DataFrame,
    half_life_years: float = 16.0,
    min_weight: float = 0.10,
) -> pd.Series:
    """Return normalized exponential weights based on each match year."""
    if half_life_years <= 0:
        raise ValueError("recency half-life must be greater than zero")
    if not 0 <= min_weight <= 1:
        raise ValueError("minimum recency weight must be between 0 and 1")
    if frame.empty:
        return pd.Series(dtype=float, index=frame.index)

    years = pd.to_datetime(frame["date"], errors="coerce").dt.year.astype(float)
    reference_year = float(years.max())
    ages = (reference_year - years).clip(lower=0).fillna(0.0)
    weights = np.maximum(
        np.power(0.5, ages / float(half_life_years)),
        float(min_weight),
    )
    weights = weights / max(float(np.mean(weights)), 1e-12)
    return pd.Series(weights, index=frame.index, dtype=float)


def fit_with_sample_weight(
    model: Any,
    X: pd.DataFrame,
    y: Any,
    sample_weight: pd.Series | np.ndarray,
) -> Any:
    """Fit estimators and sklearn pipelines with a common sample-weight API."""
    weights = np.asarray(sample_weight, dtype=float)
    if isinstance(model, Pipeline):
        final_step = model.steps[-1][0]
        model.fit(X, y, **{f"{final_step}__sample_weight": weights})
    else:
        model.fit(X, y, sample_weight=weights)
    return model


@dataclass
class DataBundle:
    matches: pd.DataFrame
    team_current: pd.DataFrame
    training_frame: pd.DataFrame
    event_columns: List[str]
    box_frame: pd.DataFrame


class WorldCupSAILoader:
    def __init__(self, zip_path: str | Path, workdir: str | Path = "_worldcupsai_extracted"):
        self.zip_path = Path(zip_path)
        self.workdir = Path(workdir)

    def extract(self) -> Path:
        self.workdir.mkdir(parents=True, exist_ok=True)
        marker = self.workdir / "curated" / "matches_curated.csv"
        if not marker.exists():
            with zipfile.ZipFile(self.zip_path) as z:
                z.extractall(self.workdir)
        return self.workdir / "curated"

    def read(self, name: str) -> pd.DataFrame:
        curated = self.extract()
        return pd.read_csv(curated / name, low_memory=False)

    def load_matches(self) -> pd.DataFrame:
        m = self.read("matches_curated.csv")
        goals = self.read("goals_curated.csv")
        bookings = self.read("bookings_curated.csv")
        pens = self.read("penalty_kicks_curated.csv")
        subs = self.read("substitutions_curated.csv")
        hosts = self.read("host_countries_curated.csv")
        teams = self.read("teams_curated.csv")

        # Men and women share team names in this database. Mixing the two
        # competitions corrupts form, Elo, and draw histories for men's games.
        m = m[
            m["tournament_name"].astype(str).str.contains(
                "FIFA Men's World Cup",
                case=False,
                na=False,
            )
        ].copy()

        df = m[
            [
                "tournament_id",
                "tournament_name",
                "match_id",
                "match_date",
                "stage_name",
                "group_name",
                "group_stage",
                "knockout_stage",
                "stadium_name",
                "city_name",
                "country_name",
                "home_team_name",
                "away_team_name",
                "home_team_score",
                "away_team_score",
                "extra_time",
                "penalty_shootout",
            ]
        ].copy()

        df["team_a"] = df["home_team_name"].map(canon_team)
        df["team_b"] = df["away_team_name"].map(canon_team)
        df["goals_a"] = pd.to_numeric(df["home_team_score"], errors="coerce")
        df["goals_b"] = pd.to_numeric(df["away_team_score"], errors="coerce")
        df["date"] = pd.to_datetime(df["match_date"], errors="coerce")
        df["year"] = df["date"].dt.year
        df["is_group_stage"] = df["group_stage"].astype(bool).astype(int)
        df["is_knockout"] = df["knockout_stage"].astype(bool).astype(int)
        df["extra_time"] = df["extra_time"].fillna(False).astype(bool).astype(int)
        df["penalty_shootout"] = df["penalty_shootout"].fillna(False).astype(bool).astype(int)

        host_map = hosts.groupby("tournament_id")["team_name"].apply(lambda s: set(canon_team(x) for x in s)).to_dict()
        df["host_a"] = [int(a in host_map.get(t, set())) for a, t in zip(df.team_a, df.tournament_id)]
        df["host_b"] = [int(b in host_map.get(t, set())) for b, t in zip(df.team_b, df.tournament_id)]

        team_meta = teams[["team_name", "confederation_code"]].copy()
        team_meta["team_name"] = team_meta["team_name"].map(canon_team)
        conf = dict(zip(team_meta.team_name, team_meta.confederation_code))
        df["confed_a_code"] = df.team_a.map(conf).fillna("UNK")
        df["confed_b_code"] = df.team_b.map(conf).fillna("UNK")
        df["same_confed"] = (df.confed_a_code == df.confed_b_code).astype(int)

        if len(goals):
            g = goals.copy()
            g["team_name"] = g["team_name"].map(canon_team)
            g["own_goal"] = g["own_goal"].fillna(False).astype(bool).astype(int)
            g["penalty"] = g["penalty"].fillna(False).astype(bool).astype(int)
            agg = (
                g.groupby(["match_id", "team_name"])
                .agg(event_goals=("goal_id", "count"), own_goals=("own_goal", "sum"), penalty_goals=("penalty", "sum"))
                .reset_index()
            )
            for side, col in [("a", "team_a"), ("b", "team_b")]:
                df = df.merge(
                    agg.rename(
                        columns={
                            "team_name": col,
                            "event_goals": f"event_goals_{side}",
                            "own_goals": f"own_goals_{side}",
                            "penalty_goals": f"penalty_goals_{side}",
                        }
                    ),
                    on=["match_id", col],
                    how="left",
                )

        if len(bookings):
            bk = bookings.copy()
            bk["team_name"] = bk["team_name"].map(canon_team)
            for c in ["yellow_card", "red_card", "second_yellow_card", "sending_off"]:
                bk[c] = bk[c].fillna(False).astype(bool).astype(int)
            agg = (
                bk.groupby(["match_id", "team_name"])
                .agg(
                    yellow_cards=("yellow_card", "sum"),
                    red_cards=("red_card", "sum"),
                    second_yellow_cards=("second_yellow_card", "sum"),
                    sending_offs=("sending_off", "sum"),
                )
                .reset_index()
            )
            for side, col in [("a", "team_a"), ("b", "team_b")]:
                df = df.merge(
                    agg.rename(
                        columns={
                            "team_name": col,
                            "yellow_cards": f"yellow_cards_{side}",
                            "red_cards": f"red_cards_{side}",
                            "second_yellow_cards": f"second_yellow_cards_{side}",
                            "sending_offs": f"sending_offs_{side}",
                        }
                    ),
                    on=["match_id", col],
                    how="left",
                )

        if len(pens):
            p = pens.copy()
            p["team_name"] = p["team_name"].map(canon_team)
            p["converted"] = p["converted"].fillna(False).astype(bool).astype(int)
            agg = (
                p.groupby(["match_id", "team_name"])
                .agg(penalty_kicks=("penalty_kick_id", "count"), penalty_kicks_converted=("converted", "sum"))
                .reset_index()
            )
            for side, col in [("a", "team_a"), ("b", "team_b")]:
                df = df.merge(
                    agg.rename(
                        columns={
                            "team_name": col,
                            "penalty_kicks": f"penalty_kicks_{side}",
                            "penalty_kicks_converted": f"penalty_kicks_converted_{side}",
                        }
                    ),
                    on=["match_id", col],
                    how="left",
                )

        if len(subs):
            s = subs.copy()
            s["team_name"] = s["team_name"].map(canon_team)
            agg = s.groupby(["match_id", "team_name"]).agg(substitutions=("substitution_id", "count")).reset_index()
            for side, col in [("a", "team_a"), ("b", "team_b")]:
                df = df.merge(
                    agg.rename(columns={"team_name": col, "substitutions": f"substitutions_{side}"}),
                    on=["match_id", col],
                    how="left",
                )

        for c in df.columns:
            if c.endswith("_a") or c.endswith("_b"):
                if c not in ["team_a", "team_b"] and pd.api.types.is_numeric_dtype(df[c]):
                    df[c] = df[c].fillna(0)

        return df.dropna(subset=["team_a", "team_b", "goals_a", "goals_b", "date"]).sort_values("date").reset_index(drop=True)


def load_current_team_features(train_csv: Optional[str], test_csv: Optional[str]) -> pd.DataFrame:
    frames = []
    for path in [train_csv, test_csv]:
        if path and Path(path).exists():
            d = pd.read_csv(path)
            if "team" in d.columns:
                d["team"] = d["team"].map(canon_team)
                frames.append(d)
    if not frames:
        return pd.DataFrame(columns=["team"])

    cur = pd.concat(frames, ignore_index=True)
    if "version" in cur.columns:
        cur = cur.sort_values("version").groupby("team", as_index=False).tail(1)
    return cur.reset_index(drop=True)



def load_kaggle_box_data(box_csv: Optional[str]) -> pd.DataFrame:
    """Load the Kaggle FIFAallMatchBoxData.csv file.

    This file gives match box-score statistics such as shots, shots on target,
    possession, fouls, saves, yellow cards, and red cards.
    """
    if not box_csv or not Path(box_csv).exists():
        return pd.DataFrame()

    raw = pd.read_csv(box_csv)
    required = {"year", "hname", "aname"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Box data is missing required columns: {sorted(missing)}")

    # Paired zeros in these continuous fields represent unrecorded statistics.
    for home_col, away_col in [
        ("hPossesion", "aPossesion"),
        ("hshotsOnTarget", "ashotsOnTarget"),
        ("hshots", "ashots"),
        ("hfouls", "afouls"),
        ("hsaves", "asaves"),
    ]:
        if home_col in raw.columns and away_col in raw.columns:
            missing_pair = (
                pd.to_numeric(raw[home_col], errors="coerce").eq(0)
                & pd.to_numeric(raw[away_col], errors="coerce").eq(0)
            )
            raw.loc[missing_pair, [home_col, away_col]] = np.nan

    mapping = {
        "goals": ("hgoals", "agoals"),
        "possession": ("hPossesion", "aPossesion"),
        "shots_on_target": ("hshotsOnTarget", "ashotsOnTarget"),
        "shots": ("hshots", "ashots"),
        "yellow_cards": ("hyellowCards", "ayellowCards"),
        "red_cards": ("hredCards", "aredCards"),
        "fouls": ("hfouls", "afouls"),
        "saves": ("hsaves", "asaves"),
    }

    rows = []
    for match_index, r in raw.iterrows():
        home_team = canon_team(r["hname"])
        away_team = canon_team(r["aname"])
        year = pd.to_numeric(r["year"], errors="coerce")

        home = {
            "box_match_id": match_index,
            "box_year": year,
            "team": home_team,
            "opponent": away_team,
            "is_home": 1,
        }
        away = {
            "box_match_id": match_index,
            "box_year": year,
            "team": away_team,
            "opponent": home_team,
            "is_home": 0,
        }

        for target, (home_col, away_col) in mapping.items():
            if home_col in raw.columns:
                home[target] = pd.to_numeric(r[home_col], errors="coerce")
            if away_col in raw.columns:
                away[target] = pd.to_numeric(r[away_col], errors="coerce")

        rows.append(home)
        rows.append(away)

    df = pd.DataFrame(rows)
    for c in df.columns:
        if c not in ["team", "opponent"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.reset_index(drop=True)


def load_world_cup_qualification_results(
    results_csv: Optional[str],
    former_names_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Load explicit World Cup qualification matches from results.csv."""
    if not results_csv or not Path(results_csv).exists():
        return pd.DataFrame()

    raw = pd.read_csv(results_csv)
    required = {
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "neutral",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(
            f"results.csv is missing required columns: {sorted(missing)}"
        )

    name_map: Dict[str, str] = {}
    if former_names_csv and Path(former_names_csv).exists():
        former = pd.read_csv(former_names_csv)
        if {"current", "former"}.issubset(former.columns):
            for _, row in former.iterrows():
                name_map[canon_team(row["former"])] = canon_team(row["current"])

    def normalize_name(name: str) -> str:
        canonical = canon_team(name)
        return name_map.get(canonical, canonical)

    qualifiers = raw[
        raw["tournament"].astype(str).eq("FIFA World Cup qualification")
    ].copy()
    qualifiers["date"] = pd.to_datetime(qualifiers["date"], errors="coerce")
    qualifiers["home_score"] = pd.to_numeric(
        qualifiers["home_score"],
        errors="coerce",
    )
    qualifiers["away_score"] = pd.to_numeric(
        qualifiers["away_score"],
        errors="coerce",
    )
    qualifiers = qualifiers.dropna(
        subset=[
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        ]
    ).sort_values("date").reset_index(drop=True)

    rows = []
    for match_index, row in qualifiers.iterrows():
        home = normalize_name(row["home_team"])
        away = normalize_name(row["away_team"])
        neutral = str(row.get("neutral", False)).strip().lower() in {
            "1",
            "true",
            "yes",
        }
        common = {
            "box_match_id": match_index,
            "box_year": int(row["date"].year),
            "date": row["date"],
            "source": "results_fifa_world_cup_qualification",
        }
        rows.append(
            {
                **common,
                "team": home,
                "opponent": away,
                "is_home": 0 if neutral else 1,
                "goals": float(row["home_score"]),
            }
        )
        rows.append(
            {
                **common,
                "team": away,
                "opponent": home,
                "is_home": 0,
                "goals": float(row["away_score"]),
            }
        )

    return pd.DataFrame(rows)


def build_box_team_profiles(box: pd.DataFrame) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """Build recent team-level box-stat profiles for immediate predictions.

    The WorldCupSAI historical backbone and the Kaggle box-score file do not have
    a clean common match_id, so this uses the Kaggle file as an event-stat layer.
    """
    if box.empty or "team" not in box.columns:
        return {}, []

    candidate_targets = [
        "shots",
        "shots_on_target",
        "possession",
        "fouls",
        "saves",
        "yellow_cards",
        "red_cards",
    ]
    targets = [c for c in candidate_targets if c in box.columns and box[c].notna().sum() >= 20]
    if not targets:
        return {}, []

    df = box.copy()
    if "box_year" in df.columns:
        df = df.sort_values("box_year")
        recent = df.groupby("team", group_keys=False).tail(20)
    else:
        recent = df

    global_means = {c: float(df[c].mean()) for c in targets}
    profiles: Dict[str, Dict[str, float]] = {}

    for team, g in recent.groupby("team"):
        profile = {}
        for c in targets:
            profile[c] = float(g[c].mean()) if g[c].notna().any() else global_means[c]
        profile["box_matches_seen"] = int(len(g))
        profiles[canon_team(team)] = profile

    return profiles, targets


def qualifier_rows(box: pd.DataFrame) -> pd.DataFrame:
    """Select likely qualifiers from the Kaggle file, which has no stage field."""
    if box.empty or "box_year" not in box.columns:
        return box.iloc[0:0].copy()
    if (
        "source" in box.columns
        and box["source"]
        .astype(str)
        .eq("results_fifa_world_cup_qualification")
        .any()
    ):
        return box[
            box["source"]
            .astype(str)
            .eq("results_fifa_world_cup_qualification")
        ].copy()
    finals_years = MEN_WORLD_CUP_FINAL_YEARS | WOMEN_WORLD_CUP_FINAL_YEARS
    return box[~box["box_year"].isin(finals_years)].copy()


def build_qualifier_team_profiles(
    box: pd.DataFrame,
    before_year: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """Build recent qualification form using only years before the cutoff."""
    qualifying = qualifier_rows(box)
    if before_year is not None:
        qualifying = qualifying[qualifying["box_year"] < before_year]
    if qualifying.empty or "goals" not in qualifying.columns:
        return {}

    qualifying = qualifying.sort_values(["box_year", "box_match_id"])
    goal_lookup = qualifying.set_index(["box_match_id", "team"])["goals"]
    recent = qualifying.groupby("team", group_keys=False).tail(12)
    profiles: Dict[str, Dict[str, float]] = {}
    for team, group in recent.groupby("team"):
        goals_for = pd.to_numeric(group["goals"], errors="coerce")
        goals_against = pd.Series(
            [
                pd.to_numeric(
                    goal_lookup.get(
                        (row["box_match_id"], row["opponent"]),
                        np.nan,
                    ),
                    errors="coerce",
                )
                for _, row in group.iterrows()
            ],
            index=group.index,
            dtype=float,
        )
        valid = goals_for.notna() & goals_against.notna()
        if not valid.any():
            continue
        goals_for = goals_for[valid]
        goals_against = goals_against[valid]
        profiles[canon_team(team)] = {
            "gf_avg": float(goals_for.mean()),
            "ga_avg": float(goals_against.mean()),
            "gd_avg": float((goals_for - goals_against).mean()),
            "draw_rate": float((goals_for == goals_against).mean()),
            "clean_sheet_rate": float((goals_against == 0).mean()),
            "matches_seen": int(valid.sum()),
        }
    return profiles


def qualifier_influence_for_year(
    year: int,
    start_year: int = 2014,
    full_weight_year: int = 2022,
    minimum_influence: float = 0.0,
) -> float:
    """Blend explicit qualifiers in gradually across tournament eras."""
    if full_weight_year <= start_year:
        raise ValueError(
            "qualifier full-weight year must be greater than start year"
        )
    if not 0 <= minimum_influence <= 1:
        raise ValueError("qualifier minimum influence must be between 0 and 1")
    progress = (float(year) - start_year) / (full_weight_year - start_year)
    progress = float(np.clip(progress, 0.0, 1.0))
    return minimum_influence + (1.0 - minimum_influence) * progress


def qualifier_pair_features(
    team_a: str,
    team_b: str,
    profiles: Dict[str, Dict[str, float]],
    fallback_profiles: Optional[Dict[str, Dict[str, float]]] = None,
    influence: float = 1.0,
) -> Dict[str, float]:
    default = {
        "gf_avg": 1.25,
        "ga_avg": 1.25,
        "gd_avg": 0.0,
        "draw_rate": 0.25,
        "clean_sheet_rate": 0.25,
        "matches_seen": 0,
    }
    fallback_profiles = fallback_profiles or {}

    def blended_profile(team: str) -> Dict[str, float]:
        name = canon_team(team)
        fallback = fallback_profiles.get(name, default)
        explicit = profiles.get(name, fallback)
        return {
            key: (1.0 - influence) * float(fallback[key])
            + influence * float(explicit[key])
            for key in default
        }

    a = blended_profile(team_a)
    b = blended_profile(team_b)
    return {
        "qual_a_gf_avg": a["gf_avg"],
        "qual_a_ga_avg": a["ga_avg"],
        "qual_a_gd_avg": a["gd_avg"],
        "qual_a_draw_rate": a["draw_rate"],
        "qual_a_clean_sheet_rate": a["clean_sheet_rate"],
        "qual_a_matches_seen": a["matches_seen"],
        "qual_b_gf_avg": b["gf_avg"],
        "qual_b_ga_avg": b["ga_avg"],
        "qual_b_gd_avg": b["gd_avg"],
        "qual_b_draw_rate": b["draw_rate"],
        "qual_b_clean_sheet_rate": b["clean_sheet_rate"],
        "qual_b_matches_seen": b["matches_seen"],
        "qual_abs_gd_diff": abs(a["gd_avg"] - b["gd_avg"]),
        "qual_mean_draw_rate": (a["draw_rate"] + b["draw_rate"]) / 2.0,
        "qual_abs_draw_rate_diff": abs(a["draw_rate"] - b["draw_rate"]),
        "qual_expected_total": (
            a["gf_avg"] + a["ga_avg"] + b["gf_avg"] + b["ga_avg"]
        )
        / 2.0,
        "qual_min_matches_seen": min(a["matches_seen"], b["matches_seen"]),
    }


def add_qualifier_features(
    frame: pd.DataFrame,
    box: pd.DataFrame,
    fallback_box: Optional[pd.DataFrame] = None,
    blend_start_year: int = 2010,
    full_weight_year: int = 2022,
    minimum_influence: float = 0.0,
) -> pd.DataFrame:
    """Attach leakage-free qualification form to historical training rows."""
    if frame.empty:
        return frame
    result = frame.reset_index(drop=True).copy()
    years = pd.to_datetime(result["date"], errors="coerce").dt.year
    explicit_cache = {
        int(year): build_qualifier_team_profiles(box, before_year=int(year))
        for year in years.dropna().unique()
    }
    fallback_source = fallback_box if fallback_box is not None else box
    fallback_cache = {
        int(year): build_qualifier_team_profiles(
            fallback_source,
            before_year=int(year),
        )
        for year in years.dropna().unique()
    }
    feature_rows = []
    for (_, row), year in zip(result.iterrows(), years):
        if pd.notna(year):
            match_year = int(year)
            profiles = explicit_cache.get(match_year, {})
            fallback_profiles = fallback_cache.get(match_year, {})
            influence = qualifier_influence_for_year(
                match_year,
                start_year=blend_start_year,
                full_weight_year=full_weight_year,
                minimum_influence=minimum_influence,
            )
        else:
            profiles = {}
            fallback_profiles = {}
            influence = minimum_influence
        feature_rows.append(
            qualifier_pair_features(
                row.team_a,
                row.team_b,
                profiles,
                fallback_profiles=fallback_profiles,
                influence=influence,
            )
        )
    return pd.concat([result, pd.DataFrame(feature_rows)], axis=1)


def elo_expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-(ra - rb) / 400.0))


def elo_margin_multiplier(gd: float) -> float:
    margin = abs(float(gd))
    if margin <= 1:
        return 1.0
    return math.log1p(margin) * 1.25


def build_rolling_features(
    matches: pd.DataFrame,
    current: pd.DataFrame,
    qualifier_box: Optional[pd.DataFrame] = None,
    qualifier_fallback_box: Optional[pd.DataFrame] = None,
    qualifier_blend_start_year: int = 2014,
    qualifier_full_weight_year: int = 2022,
    qualifier_minimum_influence: float = 0.0,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    rows = []
    team_hist: Dict[str, List[dict]] = {}
    elo: Dict[str, float] = {}

    current_idx = current.set_index("team") if len(current) and "team" in current.columns else pd.DataFrame()
    drop_current_targets = {
        "winner",
        "finalist",
        "semi_finalist",
        "quarter_finalist",
        "round_reached",
        "group_position",
        "points_current_tournament",
        "goals_current_tournament",
        "wins_current_tournament",
    }
    current_numeric = [
        c
        for c in current.columns
        if c not in ["team", "continent"]
        and c not in drop_current_targets
        and pd.api.types.is_numeric_dtype(current[c])
    ] if len(current) else []

    event_bases = [
        "yellow_cards",
        "red_cards",
        "second_yellow_cards",
        "sending_offs",
        "penalty_goals",
        "penalty_kicks",
        "penalty_kicks_converted",
        "own_goals",
        "substitutions",
    ]

    for _, r in matches.iterrows():
        a, b = r.team_a, r.team_b
        elo_a = float(elo.get(a, 1500.0))
        elo_b = float(elo.get(b, 1500.0))
        elo_prob_a = elo_expected(elo_a, elo_b)

        ha, hb = team_hist.get(a, []), team_hist.get(b, [])

        def stats(hist: List[dict]) -> Dict[str, float]:
            recent = hist[-12:]
            if not recent:
                return {"gf_avg": 1.25, "ga_avg": 1.25, "gd_avg": 0.0, "win_rate": 0.33, "draw_rate": 0.25, "matches_seen": 0}
            gf = np.array([x["gf"] for x in recent], dtype=float)
            ga = np.array([x["ga"] for x in recent], dtype=float)
            return {
                "gf_avg": float(gf.mean()),
                "ga_avg": float(ga.mean()),
                "gd_avg": float((gf - ga).mean()),
                "win_rate": float((gf > ga).mean()),
                "draw_rate": float((gf == ga).mean()),
                "matches_seen": len(hist),
            }

        sa, sb = stats(ha), stats(hb)

        feat = {
            "match_id": r.match_id,
            "date": r.date,
            "team_a": a,
            "team_b": b,
            "goals_a": r.goals_a,
            "goals_b": r.goals_b,
            "goal_diff": r.goals_a - r.goals_b,
            "is_group_stage": r.is_group_stage,
            "is_knockout": r.is_knockout,
            "host_a": r.host_a,
            "host_b": r.host_b,
            "host_diff": r.host_a - r.host_b,
            "same_confed": r.same_confed,
            "elo_a": elo_a,
            "elo_b": elo_b,
            "elo_diff": elo_a - elo_b,
            "elo_prob_a": elo_prob_a,
            "a_gf_avg": sa["gf_avg"],
            "a_ga_avg": sa["ga_avg"],
            "a_gd_avg": sa["gd_avg"],
            "a_win_rate": sa["win_rate"],
            "a_draw_rate": sa["draw_rate"],
            "a_matches_seen": sa["matches_seen"],
            "b_gf_avg": sb["gf_avg"],
            "b_ga_avg": sb["ga_avg"],
            "b_gd_avg": sb["gd_avg"],
            "b_win_rate": sb["win_rate"],
            "b_draw_rate": sb["draw_rate"],
            "b_matches_seen": sb["matches_seen"],
            "gf_diff": sa["gf_avg"] - sb["gf_avg"],
            "ga_diff": sa["ga_avg"] - sb["ga_avg"],
            "gd_diff": sa["gd_avg"] - sb["gd_avg"],
            "win_rate_diff": sa["win_rate"] - sb["win_rate"],
            "abs_host_diff": abs(r.host_a - r.host_b),
            "abs_elo_diff": abs(elo_a - elo_b),
            "abs_gf_diff": abs(sa["gf_avg"] - sb["gf_avg"]),
            "abs_ga_diff": abs(sa["ga_avg"] - sb["ga_avg"]),
            "abs_gd_diff": abs(sa["gd_avg"] - sb["gd_avg"]),
            "mean_draw_rate": (sa["draw_rate"] + sb["draw_rate"]) / 2.0,
            "abs_draw_rate_diff": abs(sa["draw_rate"] - sb["draw_rate"]),
            "form_expected_total": (
                sa["gf_avg"] + sa["ga_avg"] + sb["gf_avg"] + sb["ga_avg"]
            )
            / 2.0,
        }

        for c in current_numeric:
            av = current_idx.loc[a, c] if len(current_idx) and a in current_idx.index and c in current_idx.columns else np.nan
            bv = current_idx.loc[b, c] if len(current_idx) and b in current_idx.index and c in current_idx.columns else np.nan
            if isinstance(av, pd.Series):
                av = av.iloc[-1]
            if isinstance(bv, pd.Series):
                bv = bv.iloc[-1]
            feat[f"cur_a_{c}"] = av
            feat[f"cur_b_{c}"] = bv
            feat[f"cur_diff_{c}"] = (av - bv) if pd.notna(av) and pd.notna(bv) else np.nan

        for base in event_bases:
            ca, cb = f"{base}_a", f"{base}_b"
            if ca in matches.columns:
                feat[ca] = r.get(ca, 0)
                feat[cb] = r.get(cb, 0)

        rows.append(feat)

        # Update rolling stats and Elo after creating the pre-match feature row.
        team_hist.setdefault(a, []).append({"gf": r.goals_a, "ga": r.goals_b})
        team_hist.setdefault(b, []).append({"gf": r.goals_b, "ga": r.goals_a})

        actual_a = 1.0 if r.goals_a > r.goals_b else 0.5 if r.goals_a == r.goals_b else 0.0
        k = 24.0 * elo_margin_multiplier(r.goals_a - r.goals_b)
        elo[a] = elo_a + k * (actual_a - elo_prob_a)
        elo[b] = elo_b + k * ((1.0 - actual_a) - (1.0 - elo_prob_a))

    frame = pd.DataFrame(rows)
    frame = add_qualifier_features(
        frame,
        qualifier_box if qualifier_box is not None else pd.DataFrame(),
        fallback_box=qualifier_fallback_box,
        blend_start_year=qualifier_blend_start_year,
        full_weight_year=qualifier_full_weight_year,
        minimum_influence=qualifier_minimum_influence,
    )

    feature_cols = [
        c
        for c in frame.columns
        if c not in ["match_id", "date", "team_a", "team_b", "goals_a", "goals_b", "goal_diff"]
        and pd.api.types.is_numeric_dtype(frame[c])
        and not (c.endswith("_a") or c.endswith("_b"))
    ]
    feature_cols = [
        c
        for c in feature_cols
        if not any(c == f"{base}_{side}" for base in event_bases for side in ["a", "b"])
    ]
    feature_cols = [c for c in feature_cols if frame[c].notna().mean() > 0.20]

    event_cols = [base for base in event_bases if f"{base}_a" in frame.columns and f"{base}_b" in frame.columns]
    return frame, feature_cols, event_cols


def build_current_strength_table(current: pd.DataFrame) -> Dict[str, float]:
    """Build a compact current-strength prior from Kaggle current team features."""
    if current.empty or "team" not in current.columns:
        return {}

    df = current.copy()
    out = pd.Series(0.0, index=df.index)
    used = 0

    def add_col(name: str, sign: float = 1.0, log: bool = False):
        nonlocal out, used
        if name not in df.columns:
            return
        x = pd.to_numeric(df[name], errors="coerce")
        if log:
            x = np.log1p(x.clip(lower=0))
        if x.notna().sum() < 3:
            return
        z = (x - x.mean()) / (x.std(ddof=0) + 1e-9)
        out += sign * z.fillna(0)
        used += 1

    # Higher is better.
    for c in [
        "fifa_points_pre_tournament",
        "squad_total_market_value_eur",
        "goals_scored_last_4y",
        "wins_last_4y",
        "world_cup_titles_before",
        "world_cup_participations_before",
    ]:
        add_col(c, sign=1.0, log=c == "squad_total_market_value_eur")

    # Lower is better.
    for c in ["fifa_rank_pre_tournament", "goals_received_last_4y", "losses_last_4y"]:
        add_col(c, sign=-1.0)

    if used == 0:
        return {}

    strength = out / used
    return dict(zip(df["team"].map(canon_team), strength.astype(float)))


class StrongWorldCupModel:
    def __init__(
        self,
        model_type: str = "ensemble",
        recency_half_life_years: float = 16.0,
        recency_min_weight: float = 0.10,
    ):
        self.model_type = model_type
        self.recency_half_life_years = float(recency_half_life_years)
        self.recency_min_weight = float(recency_min_weight)
        self.recency_weight_summary: Dict[str, float] = {}
        self.feature_cols: List[str] = []
        self.event_cols: List[str] = []

        self.goal_a = None
        self.goal_b = None
        self.goal_a_models: List[Tuple[str, Any, float]] = []
        self.goal_b_models: List[Tuple[str, Any, float]] = []

        self.goal_diff_model = None
        self.goal_diff_models: List[Tuple[str, Any, float]] = []

        self.result_model = None
        self.result_models: List[Tuple[str, Any, float]] = []
        self.draw_model = None
        self.draw_calibrator = None
        self.draw_feature_cols: List[str] = []
        self.event_models = {}

        self.train_frame = None
        self.current = pd.DataFrame()
        self.latest_team_stats = {}
        self.latest_elo: Dict[str, float] = {}
        self.current_strength: Dict[str, float] = {}
        self.box_profiles: Dict[str, Dict[str, float]] = {}
        self.box_targets: List[str] = []
        self.qualifier_profiles: Dict[str, Dict[str, float]] = {}
        self.qualifier_fallback_profiles: Dict[str, Dict[str, float]] = {}
        self.qualifier_source = ""
        self.qualifier_source_rows = 0
        self.qualifier_prediction_year = 2026
        self.qualifier_blend_start_year = 2014
        self.qualifier_full_weight_year = 2022
        self.qualifier_minimum_influence = 0.0

        self.temperature = 1.08
        self.current_strength_k = 0.10
        self.goal_diff_blend = 0.30
        self.dixon_coles_rho = -0.08
        self.draw_model_weight = 0.75

    def _regressor(self):
        if self.model_type == "lightgbm" and lgb is not None:
            return lgb.LGBMRegressor(n_estimators=30, learning_rate=0.035, max_depth=3, num_leaves=15, random_state=7, verbose=-1)
        if self.model_type == "xgboost" and xgb is not None:
            return xgb.XGBRegressor(n_estimators=30, learning_rate=0.035, max_depth=3, subsample=.85, colsample_bytree=.85, random_state=7, objective="count:poisson")
        if self.model_type == "catboost" and CatBoostRegressor is not None:
            return CatBoostRegressor(iterations=120, depth=4, learning_rate=.035, loss_function="Poisson", verbose=False, random_seed=7)
        if self.model_type == "rf":
            return RandomForestRegressor(n_estimators=30, min_samples_leaf=5, random_state=7, n_jobs=-1)
        if self.model_type == "poisson":
            return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("m", PoissonRegressor(alpha=.3, max_iter=1000))])
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("m", RandomForestRegressor(n_estimators=30, min_samples_leaf=5, random_state=7, n_jobs=-1))])

    def _diff_regressor(self):
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", Ridge(alpha=2.0)),
        ])

    def _named_regressors(self) -> List[Tuple[str, Any, float]]:
        models: List[Tuple[str, Any, float]] = [
            ("rf", RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=7, n_jobs=-1), 0.25),
            ("hgb", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("m", HistGradientBoostingRegressor(max_iter=300, learning_rate=0.035, max_leaf_nodes=15, l2_regularization=0.08, random_state=7)),
            ]), 0.20),
            ("poisson", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
                ("m", PoissonRegressor(alpha=0.25, max_iter=2000)),
            ]), 0.15),
        ]
        if lgb is not None:
            models.append(("lightgbm", lgb.LGBMRegressor(
                n_estimators=350, learning_rate=0.025, max_depth=4, num_leaves=15,
                subsample=0.90, colsample_bytree=0.90, reg_lambda=1.0,
                random_state=7, verbose=-1
            ), 0.15))
        if xgb is not None:
            models.append(("xgboost", xgb.XGBRegressor(
                n_estimators=350, learning_rate=0.025, max_depth=4,
                subsample=0.90, colsample_bytree=0.90, reg_lambda=1.0,
                random_state=7, objective="count:poisson"
            ), 0.15))
        if CatBoostRegressor is not None:
            models.append(("catboost", CatBoostRegressor(
                iterations=350, depth=5, learning_rate=0.025,
                loss_function="Poisson", l2_leaf_reg=5.0,
                verbose=False, random_seed=7
            ), 0.20))
        total = sum(w for _, _, w in models)
        return [(name, model, w / total) for name, model, w in models]

    def _named_diff_regressors(self) -> List[Tuple[str, Any, float]]:
        models: List[Tuple[str, Any, float]] = [
            ("ridge", self._diff_regressor(), 0.30),
            ("rf", RandomForestRegressor(n_estimators=250, min_samples_leaf=4, random_state=7, n_jobs=-1), 0.30),
            ("hgb", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("m", HistGradientBoostingRegressor(max_iter=250, learning_rate=0.035, max_leaf_nodes=15, l2_regularization=0.10, random_state=7)),
            ]), 0.25),
        ]
        if lgb is not None:
            models.append(("lightgbm", lgb.LGBMRegressor(
                n_estimators=250, learning_rate=0.025, max_depth=3, num_leaves=12,
                reg_lambda=1.0, random_state=7, verbose=-1
            ), 0.15))
        total = sum(w for _, _, w in models)
        return [(name, model, w / total) for name, model, w in models]

    def _classifier(self):
        if self.model_type == "lightgbm" and lgb is not None:
            return lgb.LGBMClassifier(n_estimators=30, learning_rate=.035, max_depth=3, num_leaves=15, random_state=7, verbose=-1)
        if self.model_type == "xgboost" and xgb is not None:
            return xgb.XGBClassifier(n_estimators=30, learning_rate=.035, max_depth=3, subsample=.85, colsample_bytree=.85, random_state=7, eval_metric="mlogloss")
        if self.model_type == "catboost" and CatBoostClassifier is not None:
            return CatBoostClassifier(iterations=100, depth=4, learning_rate=.035, loss_function="MultiClass", verbose=False, random_seed=7)
        if self.model_type == "rf":
            return RandomForestClassifier(n_estimators=30, min_samples_leaf=5, random_state=7, n_jobs=-1)
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("m", RandomForestClassifier(n_estimators=30, min_samples_leaf=5, random_state=7, n_jobs=-1))])

    def _named_classifiers(self) -> List[Tuple[str, Any, float]]:
        models: List[Tuple[str, Any, float]] = [
            ("rf", RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=7, n_jobs=-1), 0.35),
            ("hgb", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("m", HistGradientBoostingClassifier(max_iter=250, learning_rate=0.035, max_leaf_nodes=15, l2_regularization=0.08, random_state=7)),
            ]), 0.20),
            ("logistic", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
                ("m", LogisticRegression(max_iter=2000)),
            ]), 0.10),
        ]
        if lgb is not None:
            models.append(("lightgbm", lgb.LGBMClassifier(
                n_estimators=300, learning_rate=0.025, max_depth=4, num_leaves=15,
                subsample=0.90, colsample_bytree=0.90, reg_lambda=1.0,
                random_state=7, verbose=-1
            ), 0.15))
        if xgb is not None:
            models.append(("xgboost", xgb.XGBClassifier(
                n_estimators=300, learning_rate=0.025, max_depth=4,
                subsample=0.90, colsample_bytree=0.90, reg_lambda=1.0,
                random_state=7, eval_metric="mlogloss"
            ), 0.15))
        if CatBoostClassifier is not None:
            models.append(("catboost", CatBoostClassifier(
                iterations=300, depth=5, learning_rate=0.025,
                loss_function="MultiClass", l2_leaf_reg=5.0,
                verbose=False, random_seed=7
            ), 0.20))
        total = sum(w for _, _, w in models)
        return [(name, model, w / total) for name, model, w in models]

    @staticmethod
    def _weighted_regression_prediction(models: List[Tuple[str, Any, float]], X: pd.DataFrame) -> float:
        preds, weights = [], []
        for _, model, weight in models:
            pred = float(model.predict(X)[0])
            if math.isfinite(pred):
                preds.append(pred)
                weights.append(weight)
        if not preds:
            return 1.25
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()
        return float(np.dot(np.asarray(preds, dtype=float), w))

    @staticmethod
    def _weighted_classification_prediction(models: List[Tuple[str, Any, float]], X: pd.DataFrame) -> Dict[str, float]:
        out = {"team_a_win": 0.0, "draw": 0.0, "team_b_win": 0.0}
        total_weight = 0.0
        for _, model, weight in models:
            if not hasattr(model, "predict_proba"):
                continue
            probs = model.predict_proba(X)[0]
            classes = list(model.classes_) if hasattr(model, "classes_") else [0, 1, 2]
            class_map = {int(c): float(p) for c, p in zip(classes, probs)}
            out["team_a_win"] += weight * class_map.get(2, 0.0)
            out["draw"] += weight * class_map.get(1, 0.0)
            out["team_b_win"] += weight * class_map.get(0, 0.0)
            total_weight += weight
        if total_weight <= 0:
            return out
        out = {k: v / total_weight for k, v in out.items()}
        s = sum(out.values())
        return {k: v / s for k, v in out.items()} if s > 0 else out

    def set_box_data(self, box: pd.DataFrame):
        self.box_profiles, self.box_targets = build_box_team_profiles(box)
        return self

    def set_qualifier_data(
        self,
        box: pd.DataFrame,
        fallback_box: Optional[pd.DataFrame] = None,
        prediction_year: int = 2026,
        blend_start_year: int = 2014,
        full_weight_year: int = 2022,
        minimum_influence: float = 0.0,
    ):
        self.qualifier_profiles = build_qualifier_team_profiles(box)
        self.qualifier_fallback_profiles = build_qualifier_team_profiles(
            fallback_box if fallback_box is not None else box
        )
        self.qualifier_source_rows = int(len(box))
        self.qualifier_prediction_year = int(prediction_year)
        self.qualifier_blend_start_year = int(blend_start_year)
        self.qualifier_full_weight_year = int(full_weight_year)
        self.qualifier_minimum_influence = float(minimum_influence)
        self.qualifier_source = (
            "results.csv FIFA World Cup qualification"
            if "source" in box.columns
            and box["source"]
            .astype(str)
            .eq("results_fifa_world_cup_qualification")
            .any()
            else "FIFAallMatchBoxData.csv heuristic"
        )
        return self

    @staticmethod
    def _new_draw_model():
        return Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
                (
                    "m",
                    LogisticRegression(
                        class_weight="balanced",
                        C=0.6,
                        max_iter=2000,
                        random_state=7,
                    ),
                ),
            ]
        )

    def _fit_draw_model(self, frame: pd.DataFrame) -> None:
        candidates = [
            "is_group_stage",
            "is_knockout",
            "same_confed",
            "abs_host_diff",
            "abs_elo_diff",
            "abs_gf_diff",
            "abs_ga_diff",
            "abs_gd_diff",
            "mean_draw_rate",
            "abs_draw_rate_diff",
            "form_expected_total",
            "qual_abs_gd_diff",
            "qual_mean_draw_rate",
            "qual_abs_draw_rate_diff",
            "qual_expected_total",
            "qual_min_matches_seen",
        ]
        self.draw_feature_cols = [column for column in candidates if column in frame]
        if not self.draw_feature_cols:
            return

        ordered = frame.sort_values("date").reset_index(drop=True)
        target = (ordered["goals_a"] == ordered["goals_b"]).astype(int)
        weights = build_year_recency_weights(
            ordered,
            self.recency_half_life_years,
            self.recency_min_weight,
        )
        split = max(int(len(ordered) * 0.80), 1)
        if (
            split < len(ordered)
            and target.iloc[:split].nunique() == 2
            and target.iloc[split:].nunique() == 2
        ):
            calibration_model = self._new_draw_model()
            fit_with_sample_weight(
                calibration_model,
                ordered.iloc[:split][self.draw_feature_cols],
                target.iloc[:split],
                weights.iloc[:split],
            )
            raw = calibration_model.predict_proba(
                ordered.iloc[split:][self.draw_feature_cols]
            )[:, 1]
            logits = np.log(
                np.clip(raw, 1e-6, 1 - 1e-6)
                / np.clip(1 - raw, 1e-6, 1 - 1e-6)
            ).reshape(-1, 1)
            self.draw_calibrator = LogisticRegression(
                C=0.5,
                max_iter=1000,
                random_state=7,
            ).fit(
                logits,
                target.iloc[split:],
                sample_weight=weights.iloc[split:].to_numpy(),
            )

        self.draw_model = self._new_draw_model()
        fit_with_sample_weight(
            self.draw_model,
            ordered[self.draw_feature_cols],
            target,
            weights,
        )

    def _predict_draw_probability(self, features: pd.DataFrame) -> float:
        if self.draw_model is None or not self.draw_feature_cols:
            return 0.20
        raw = float(
            self.draw_model.predict_proba(features[self.draw_feature_cols])[0, 1]
        )
        if self.draw_calibrator is None:
            return raw
        logit = math.log(max(raw, 1e-6) / max(1.0 - raw, 1e-6))
        return float(self.draw_calibrator.predict_proba([[logit]])[0, 1])

    def fit(self, frame: pd.DataFrame, feature_cols: List[str], event_cols: List[str], current: pd.DataFrame):
        self.train_frame = frame.copy()
        self.feature_cols = feature_cols
        self.event_cols = event_cols
        self.current = current.copy()
        self.current_strength = build_current_strength_table(current)

        X = frame[feature_cols]
        yres = np.where(frame.goals_a > frame.goals_b, 2, np.where(frame.goals_a == frame.goals_b, 1, 0))
        sample_weight = build_year_recency_weights(
            frame,
            self.recency_half_life_years,
            self.recency_min_weight,
        )
        weight_array = sample_weight.to_numpy()
        self.recency_weight_summary = {
            "half_life_years": self.recency_half_life_years,
            "minimum_raw_weight": self.recency_min_weight,
            "reference_year": float(
                pd.to_datetime(frame["date"], errors="coerce").dt.year.max()
            ),
            "normalized_min_weight": float(sample_weight.min()),
            "normalized_max_weight": float(sample_weight.max()),
            "effective_sample_size": float(
                weight_array.sum() ** 2 / max(np.square(weight_array).sum(), 1e-12)
            ),
        }

        if self.model_type == "ensemble":
            self.goal_a_models = self._named_regressors()
            self.goal_b_models = self._named_regressors()
            for _, model, _ in self.goal_a_models:
                fit_with_sample_weight(model, X, frame.goals_a, sample_weight)
            for _, model, _ in self.goal_b_models:
                fit_with_sample_weight(model, X, frame.goals_b, sample_weight)

            self.goal_diff_models = self._named_diff_regressors()
            for _, model, _ in self.goal_diff_models:
                fit_with_sample_weight(model, X, frame.goal_diff, sample_weight)

            self.result_models = self._named_classifiers()
            for _, model, _ in self.result_models:
                fit_with_sample_weight(model, X, yres, sample_weight)
        else:
            self.goal_a = self._regressor()
            self.goal_b = self._regressor()
            fit_with_sample_weight(self.goal_a, X, frame.goals_a, sample_weight)
            fit_with_sample_weight(self.goal_b, X, frame.goals_b, sample_weight)

            self.goal_diff_model = self._diff_regressor()
            fit_with_sample_weight(
                self.goal_diff_model,
                X,
                frame.goal_diff,
                sample_weight,
            )

            self.result_model = self._classifier()
            fit_with_sample_weight(self.result_model, X, yres, sample_weight)

        self._fit_draw_model(frame)

        for ev in event_cols:
            if self.model_type == "ensemble":
                ma_models = self._named_regressors()
                mb_models = self._named_regressors()
                for _, model, _ in ma_models:
                    fit_with_sample_weight(
                        model,
                        X,
                        frame[f"{ev}_a"],
                        sample_weight,
                    )
                for _, model, _ in mb_models:
                    fit_with_sample_weight(
                        model,
                        X,
                        frame[f"{ev}_b"],
                        sample_weight,
                    )
                self.event_models[ev] = (ma_models, mb_models)
            else:
                ma, mb = self._regressor(), self._regressor()
                fit_with_sample_weight(ma, X, frame[f"{ev}_a"], sample_weight)
                fit_with_sample_weight(mb, X, frame[f"{ev}_b"], sample_weight)
                self.event_models[ev] = (ma, mb)

        self._cache_latest_team_stats(frame)
        return self

    def _cache_latest_team_stats(self, frame: pd.DataFrame):
        self.latest_elo = {}
        if "elo_a" in frame.columns:
            for _, r in frame.sort_values("date").iterrows():
                self.latest_elo[r.team_a] = float(r.elo_a)
                self.latest_elo[r.team_b] = float(r.elo_b)

        for team in sorted(set(frame.team_a) | set(frame.team_b)):
            hist = []
            arows = frame[frame.team_a == team].tail(12)
            for _, r in arows.iterrows():
                hist.append((r.goals_a, r.goals_b))
            brows = frame[frame.team_b == team].tail(12)
            for _, r in brows.iterrows():
                hist.append((r.goals_b, r.goals_a))
            if hist:
                gf = np.array([x[0] for x in hist])
                ga = np.array([x[1] for x in hist])
                self.latest_team_stats[team] = {
                    "gf_avg": float(gf.mean()),
                    "ga_avg": float(ga.mean()),
                    "gd_avg": float((gf - ga).mean()),
                    "win_rate": float((gf > ga).mean()),
                    "draw_rate": float((gf == ga).mean()),
                    "matches_seen": len(hist),
                }

    def make_features(self, team_a, team_b, host_a=False, host_b=False, knockout=False):
        a, b = canon_team(team_a), canon_team(team_b)

        def s(team):
            return self.latest_team_stats.get(
                team,
                {"gf_avg": 1.25, "ga_avg": 1.25, "gd_avg": 0.0, "win_rate": 0.33, "draw_rate": 0.25, "matches_seen": 0},
            )

        sa, sb = s(a), s(b)
        elo_a = float(self.latest_elo.get(a, 1500.0))
        elo_b = float(self.latest_elo.get(b, 1500.0))

        row = {
            "is_group_stage": int(not knockout),
            "is_knockout": int(knockout),
            "host_a": int(host_a),
            "host_b": int(host_b),
            "host_diff": int(host_a) - int(host_b),
            "abs_host_diff": abs(int(host_a) - int(host_b)),
            "same_confed": 0,
            "elo_a": elo_a,
            "elo_b": elo_b,
            "elo_diff": elo_a - elo_b,
            "elo_prob_a": elo_expected(elo_a, elo_b),
            "a_gf_avg": sa["gf_avg"],
            "a_ga_avg": sa["ga_avg"],
            "a_gd_avg": sa["gd_avg"],
            "a_win_rate": sa["win_rate"],
            "a_draw_rate": sa["draw_rate"],
            "a_matches_seen": sa["matches_seen"],
            "b_gf_avg": sb["gf_avg"],
            "b_ga_avg": sb["ga_avg"],
            "b_gd_avg": sb["gd_avg"],
            "b_win_rate": sb["win_rate"],
            "b_draw_rate": sb["draw_rate"],
            "b_matches_seen": sb["matches_seen"],
            "gf_diff": sa["gf_avg"] - sb["gf_avg"],
            "ga_diff": sa["ga_avg"] - sb["ga_avg"],
            "gd_diff": sa["gd_avg"] - sb["gd_avg"],
            "win_rate_diff": sa["win_rate"] - sb["win_rate"],
            "abs_elo_diff": abs(elo_a - elo_b),
            "abs_gf_diff": abs(sa["gf_avg"] - sb["gf_avg"]),
            "abs_ga_diff": abs(sa["ga_avg"] - sb["ga_avg"]),
            "abs_gd_diff": abs(sa["gd_avg"] - sb["gd_avg"]),
            "mean_draw_rate": (sa["draw_rate"] + sb["draw_rate"]) / 2.0,
            "abs_draw_rate_diff": abs(sa["draw_rate"] - sb["draw_rate"]),
            "form_expected_total": (
                sa["gf_avg"] + sa["ga_avg"] + sb["gf_avg"] + sb["ga_avg"]
            )
            / 2.0,
        }
        qualifier_influence = qualifier_influence_for_year(
            self.qualifier_prediction_year,
            start_year=self.qualifier_blend_start_year,
            full_weight_year=self.qualifier_full_weight_year,
            minimum_influence=self.qualifier_minimum_influence,
        )
        row.update(
            qualifier_pair_features(
                a,
                b,
                self.qualifier_profiles,
                fallback_profiles=self.qualifier_fallback_profiles,
                influence=qualifier_influence,
            )
        )

        cur = self.current.set_index("team") if len(self.current) and "team" in self.current else pd.DataFrame()
        excluded = {"team", "continent", "winner", "finalist", "semi_finalist", "quarter_finalist"}
        for c in [
            c
            for c in self.current.columns
            if len(self.current)
            and c not in excluded
            and pd.api.types.is_numeric_dtype(self.current[c])
        ]:
            av = cur.loc[a, c] if len(cur) and a in cur.index else np.nan
            bv = cur.loc[b, c] if len(cur) and b in cur.index else np.nan
            if isinstance(av, pd.Series):
                av = av.iloc[-1]
            if isinstance(bv, pd.Series):
                bv = bv.iloc[-1]
            row[f"cur_a_{c}"] = av
            row[f"cur_b_{c}"] = bv
            row[f"cur_diff_{c}"] = av - bv if pd.notna(av) and pd.notna(bv) else np.nan

        return pd.DataFrame([{c: row.get(c, np.nan) for c in self.feature_cols}])

    def _apply_current_strength_correction(self, lam_a: float, lam_b: float, team_a: str, team_b: str) -> Tuple[float, float, float]:
        a, b = canon_team(team_a), canon_team(team_b)
        diff = float(self.current_strength.get(a, 0.0) - self.current_strength.get(b, 0.0))
        diff = float(np.clip(diff, -3.0, 3.0))
        factor_a = math.exp(self.current_strength_k * diff)
        factor_b = math.exp(-self.current_strength_k * diff)
        return lam_a * factor_a, lam_b * factor_b, diff

    def _apply_goal_difference_blend(self, lam_a: float, lam_b: float, diff_pred: float) -> Tuple[float, float]:
        total = max(lam_a + lam_b, 0.30)
        poisson_diff = lam_a - lam_b
        target_diff = (1.0 - self.goal_diff_blend) * poisson_diff + self.goal_diff_blend * float(diff_pred)
        target_diff = float(np.clip(target_diff, -total + 0.15, total - 0.15))
        new_a = (total + target_diff) / 2.0
        new_b = (total - target_diff) / 2.0
        return float(new_a), float(new_b)

    def predict(self, team_a, team_b, host_a=False, host_b=False, knockout=False, max_goals=10):
        X = self.make_features(team_a, team_b, host_a, host_b, knockout)

        if self.model_type == "ensemble":
            raw_lam_a = self._weighted_regression_prediction(self.goal_a_models, X)
            raw_lam_b = self._weighted_regression_prediction(self.goal_b_models, X)
            diff_pred = self._weighted_regression_prediction(self.goal_diff_models, X)
        else:
            raw_lam_a = float(self.goal_a.predict(X)[0])
            raw_lam_b = float(self.goal_b.predict(X)[0])
            diff_pred = float(self.goal_diff_model.predict(X)[0])

        raw_lam_a = max(raw_lam_a, 0.001)
        raw_lam_b = max(raw_lam_b, 0.001)

        corrected_a, corrected_b, current_strength_diff = self._apply_current_strength_correction(raw_lam_a, raw_lam_b, team_a, team_b)
        blended_a, blended_b = self._apply_goal_difference_blend(corrected_a, corrected_b, diff_pred)

        lam_a = float(np.clip(blended_a, 0.15, 4.5))
        lam_b = float(np.clip(blended_b, 0.15, 4.5))

        score_probs = poisson_score_matrix(lam_a, lam_b, max_goals)
        score_probs = apply_dixon_coles_adjustment(
            score_probs,
            lam_a,
            lam_b,
            rho=self.dixon_coles_rho,
        )
        res = result_probs(score_probs)

        # Light classifier blend. Scoreline layer remains dominant.
        if self.model_type == "ensemble":
            cls_res = self._weighted_classification_prediction(self.result_models, X)
            if sum(cls_res.values()) > 0:
                res = {k: 0.86 * res[k] + 0.14 * cls_res[k] for k in res}
                s = sum(res.values())
                res = {k: v / s for k, v in res.items()}
        elif hasattr(self.result_model, "predict_proba"):
            cp = self.result_model.predict_proba(X)[0]
            classes = list(self.result_model.classes_) if hasattr(self.result_model, "classes_") else [0, 1, 2]
            class_map = {int(c): float(p) for c, p in zip(classes, cp)}
            cls_res = {"team_a_win": class_map.get(2, 0), "draw": class_map.get(1, 0), "team_b_win": class_map.get(0, 0)}
            res = {k: 0.84 * res[k] + 0.16 * cls_res[k] for k in res}
            s = sum(res.values())
            res = {k: v / s for k, v in res.items()}

        res = temperature_smooth_result_probs(res, self.temperature)
        draw_model_probability = self._predict_draw_probability(X)
        draw_probability = (
            self.draw_model_weight * draw_model_probability
            + (1.0 - self.draw_model_weight) * res["draw"]
        )
        draw_probability = float(np.clip(draw_probability, 0.05, 0.55))
        non_draw_total = max(res["team_a_win"] + res["team_b_win"], 1e-12)
        final_results = {
            "team_a_win": (1.0 - draw_probability)
            * res["team_a_win"]
            / non_draw_total,
            "draw": draw_probability,
            "team_b_win": (1.0 - draw_probability)
            * res["team_b_win"]
            / non_draw_total,
        }
        score_probs = reweight_score_matrix_to_results(
            score_probs,
            final_results,
        )
        res = result_probs(score_probs)

        top = sorted(
            [{"team_a_goals": i, "team_b_goals": j, "probability": p} for (i, j), p in score_probs.items()],
            key=lambda x: x["probability"],
            reverse=True,
        )[:15]
        spreads = {str(d): sum(p for (i, j), p in score_probs.items() if i - j == d) for d in range(-max_goals, max_goals + 1)}
        totals = {str(t): sum(p for (i, j), p in score_probs.items() if i + j == t) for t in range(0, 2 * max_goals + 1)}
        ou = {}
        for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
            under = sum(p for (i, j), p in score_probs.items() if i + j < line)
            ou[f"over_{line}"] = 1 - under
            ou[f"under_{line}"] = under

        events = {}
        for ev, (ma, mb) in self.event_models.items():
            if self.model_type == "ensemble":
                ea_raw = self._weighted_regression_prediction(ma, X)
                eb_raw = self._weighted_regression_prediction(mb, X)
            else:
                ea_raw = float(ma.predict(X)[0])
                eb_raw = float(mb.predict(X)[0])
            ea = float(np.clip(ea_raw, 0.001, 8.0))
            eb = float(np.clip(eb_raw, 0.001, 8.0))
            events[ev] = {"expected_" + canon_team(team_a): ea, "expected_" + canon_team(team_b): eb}

        box_events = {}
        a_name, b_name = canon_team(team_a), canon_team(team_b)
        pa = self.box_profiles.get(a_name, {})
        pb = self.box_profiles.get(b_name, {})
        global_box_means = {}
        for target in self.box_targets:
            vals = [profile.get(target, np.nan) for profile in self.box_profiles.values()]
            global_box_means[target] = float(np.nanmean(vals)) if len(vals) else np.nan

        for target in self.box_targets:
            aval = pa.get(target, global_box_means.get(target, np.nan))
            bval = pb.get(target, global_box_means.get(target, np.nan))
            if pd.isna(aval) and pd.isna(bval):
                continue
            if pd.isna(aval):
                aval = global_box_means.get(target, 0.0)
            if pd.isna(bval):
                bval = global_box_means.get(target, 0.0)

            if target == "possession":
                total_poss = max(float(aval) + float(bval), 1e-9)
                aval = 100.0 * float(aval) / total_poss
                bval = 100.0 - aval

            box_events[target] = {
                "expected_" + a_name: float(aval),
                "expected_" + b_name: float(bval),
            }

        events.update({f"box_{k}": v for k, v in box_events.items()})

        return {
            "team_a": canon_team(team_a),
            "team_b": canon_team(team_b),
            "lambda_a": lam_a,
            "lambda_b": lam_b,
            "raw_lambda_a": float(raw_lam_a),
            "raw_lambda_b": float(raw_lam_b),
            "goal_difference_model_prediction": float(diff_pred),
            "current_strength_diff": float(current_strength_diff),
            "result_probabilities": res,
            "draw_model_probability": float(draw_model_probability),
            "top_scorelines": top,
            "scoreline_probabilities": [
                {"team_a_goals": i, "team_b_goals": j, "probability": p}
                for (i, j), p in sorted(score_probs.items())
            ],
            "spread_probabilities": spreads,
            "total_goal_probabilities": totals,
            "over_under_probabilities": ou,
            "event_predictions": events,
            "calibration_notes": {
                "draw_calibration": "balanced binary draw model with chronological holdout calibration",
                "draw_model_weight": self.draw_model_weight,
                "dixon_coles_rho": self.dixon_coles_rho,
                "temperature": self.temperature,
                "current_strength_k": self.current_strength_k,
                "goal_diff_blend": self.goal_diff_blend,
                "recency_half_life_years": self.recency_half_life_years,
                "recency_min_weight": self.recency_min_weight,
                "qualification_influence": qualifier_influence_for_year(
                    self.qualifier_prediction_year,
                    start_year=self.qualifier_blend_start_year,
                    full_weight_year=self.qualifier_full_weight_year,
                    minimum_influence=self.qualifier_minimum_influence,
                ),
                "exact_score_policy": "derived from calibrated expected goals; not optimized directly",
            },
        }


def build_from_zip(
    zip_path,
    train_csv=None,
    test_csv=None,
    model_type="ensemble",
    box_csv=None,
    results_csv=None,
    former_names_csv=None,
    prediction_year=2026,
    qualifier_blend_start_year=2014,
    qualifier_full_weight_year=2022,
    qualifier_minimum_influence=0.0,
    recency_half_life_years=16.0,
    recency_min_weight=0.10,
):
    loader = WorldCupSAILoader(zip_path, Path(str(zip_path) + "_extracted"))
    matches = loader.load_matches()
    current = load_current_team_features(train_csv, test_csv)
    box = load_kaggle_box_data(box_csv)
    qualification_results = load_world_cup_qualification_results(
        results_csv,
        former_names_csv,
    )
    qualifier_source = (
        qualification_results if not qualification_results.empty else box
    )
    # Current rankings/form describe 2026 strength. Attaching them to historical
    # World Cup rows would leak future information into model fitting. They are
    # retained on the fitted model only for the explicit live correction.
    historical_current = pd.DataFrame(columns=["team"])
    frame, features, events = build_rolling_features(
        matches,
        historical_current,
        qualifier_box=qualifier_source,
        qualifier_fallback_box=box,
        qualifier_blend_start_year=qualifier_blend_start_year,
        qualifier_full_weight_year=qualifier_full_weight_year,
        qualifier_minimum_influence=qualifier_minimum_influence,
    )
    model = (
        StrongWorldCupModel(
            model_type=model_type,
            recency_half_life_years=recency_half_life_years,
            recency_min_weight=recency_min_weight,
        )
        .fit(frame, features, events, current)
        .set_box_data(box)
        .set_qualifier_data(
            qualifier_source,
            fallback_box=box,
            prediction_year=prediction_year,
            blend_start_year=qualifier_blend_start_year,
            full_weight_year=qualifier_full_weight_year,
            minimum_influence=qualifier_minimum_influence,
        )
    )
    return model, DataBundle(matches=matches, team_current=current, training_frame=frame, event_columns=events, box_frame=box)



def brier_score_3way(actual: str, probs: Dict[str, float]) -> float:
    labels = ["team_a_win", "draw", "team_b_win"]
    return float(sum((probs[l] - (1.0 if l == actual else 0.0)) ** 2 for l in labels))


def actual_result_label(goals_a: float, goals_b: float) -> str:
    if goals_a > goals_b:
        return "team_a_win"
    if goals_a == goals_b:
        return "draw"
    return "team_b_win"


def safe_log_loss(prob: float) -> float:
    return -math.log(max(float(prob), 1e-12))


def chronological_backtest(
    zip_path: str,
    train_csv: Optional[str] = None,
    test_csv: Optional[str] = None,
    model_type: str = "ensemble",
    test_years: Optional[List[int]] = None,
    min_train_year: int = 1930,
    max_goals: int = 10,
    box_csv: Optional[str] = None,
    results_csv: Optional[str] = None,
    former_names_csv: Optional[str] = None,
    qualifier_blend_start_year: int = 2014,
    qualifier_full_weight_year: int = 2022,
    qualifier_minimum_influence: float = 0.0,
    recency_half_life_years: float = 16.0,
    recency_min_weight: float = 0.10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run expanding-window World Cup backtests.

    For each test year:
    - train on matches before that year
    - test on matches in that year

    This is the correct overfitting check because it prevents the model from
    learning from future tournaments.
    """
    loader = WorldCupSAILoader(zip_path, Path(str(zip_path) + "_extracted"))
    all_matches = loader.load_matches()
    current = load_current_team_features(train_csv, test_csv)
    box = load_kaggle_box_data(box_csv)
    qualification_results = load_world_cup_qualification_results(
        results_csv,
        former_names_csv,
    )
    qualifier_source = (
        qualification_results if not qualification_results.empty else box
    )

    all_years = sorted(int(y) for y in all_matches["year"].dropna().unique())
    if test_years is None:
        test_years = [y for y in all_years if y >= 2010]
    else:
        test_years = [int(y) for y in test_years]

    pred_rows = []
    summary_rows = []

    for year in test_years:
        train_matches = all_matches[(all_matches["year"] < year) & (all_matches["year"] >= min_train_year)].copy()
        test_matches = all_matches[all_matches["year"] == year].copy()

        if len(train_matches) < 80 or len(test_matches) == 0:
            continue

        historical_box = box[box["box_year"] < year].copy() if not box.empty else box
        if not qualification_results.empty:
            historical_qualifiers = qualifier_source[
                qualifier_source["box_year"] < year
            ].copy()
        else:
            historical_qualifiers = historical_box
        historical_current = pd.DataFrame(columns=["team"])
        train_frame, features, events = build_rolling_features(
            train_matches,
            historical_current,
            qualifier_box=historical_qualifiers,
            qualifier_fallback_box=historical_box,
            qualifier_blend_start_year=qualifier_blend_start_year,
            qualifier_full_weight_year=qualifier_full_weight_year,
            qualifier_minimum_influence=qualifier_minimum_influence,
        )
        model = (
            StrongWorldCupModel(
                model_type=model_type,
                recency_half_life_years=recency_half_life_years,
                recency_min_weight=recency_min_weight,
            )
            .fit(train_frame, features, events, historical_current)
            .set_box_data(historical_box)
            .set_qualifier_data(
                historical_qualifiers,
                fallback_box=historical_box,
                prediction_year=year,
                blend_start_year=qualifier_blend_start_year,
                full_weight_year=qualifier_full_weight_year,
                minimum_influence=qualifier_minimum_influence,
            )
        )

        for _, r in test_matches.iterrows():
            pred = model.predict(
                r.team_a,
                r.team_b,
                host_a=bool(r.host_a),
                host_b=bool(r.host_b),
                knockout=bool(r.is_knockout),
                max_goals=max_goals,
            )

            actual = actual_result_label(r.goals_a, r.goals_b)
            actual_prob = pred["result_probabilities"][actual]
            predicted_result = max(pred["result_probabilities"], key=pred["result_probabilities"].get)

            score_probs = {
                (int(s["team_a_goals"]), int(s["team_b_goals"])): float(s["probability"])
                for s in pred["scoreline_probabilities"]
            }
            exact_prob = score_probs.get((int(r.goals_a), int(r.goals_b)), 0.0)

            pred_rows.append(
                {
                    "test_year": year,
                    "match_id": r.match_id,
                    "date": r.date,
                    "stage": r.stage_name,
                    "team_a": r.team_a,
                    "team_b": r.team_b,
                    "actual_score": f"{int(r.goals_a)}-{int(r.goals_b)}",
                    "lambda_a": pred["lambda_a"],
                    "lambda_b": pred["lambda_b"],
                    "predicted_result": predicted_result,
                    "actual_result": actual,
                    "correct_result": int(predicted_result == actual),
                    "team_a_win_prob": pred["result_probabilities"]["team_a_win"],
                    "draw_prob": pred["result_probabilities"]["draw"],
                    "team_b_win_prob": pred["result_probabilities"]["team_b_win"],
                    "actual_result_probability": actual_prob,
                    "result_log_loss": safe_log_loss(actual_prob),
                    "result_brier": brier_score_3way(actual, pred["result_probabilities"]),
                    "goal_mae": (abs(pred["lambda_a"] - r.goals_a) + abs(pred["lambda_b"] - r.goals_b)) / 2.0,
                    "goal_diff_abs_error": abs((pred["lambda_a"] - pred["lambda_b"]) - (r.goals_a - r.goals_b)),
                    "exact_score_probability": exact_prob,
                    "exact_score_log_loss": safe_log_loss(exact_prob),
                }
            )

        year_df = pd.DataFrame([x for x in pred_rows if x["test_year"] == year])
        summary_rows.append(
            {
                "test_year": year,
                "train_matches": int(len(train_matches)),
                "test_matches": int(len(test_matches)),
                "result_accuracy": float(year_df["correct_result"].mean()),
                "mean_result_log_loss": float(year_df["result_log_loss"].mean()),
                "mean_result_brier": float(year_df["result_brier"].mean()),
                "mean_goal_mae": float(year_df["goal_mae"].mean()),
                "mean_goal_diff_abs_error": float(year_df["goal_diff_abs_error"].mean()),
                "mean_exact_score_log_loss": float(year_df["exact_score_log_loss"].mean()),
                "mean_actual_result_probability": float(year_df["actual_result_probability"].mean()),
                "mean_exact_score_probability": float(year_df["exact_score_probability"].mean()),
                "features_used": len(features),
                "event_targets": ",".join(events),
                "model_type": model_type,
                "recency_half_life_years": recency_half_life_years,
                "recency_min_weight": recency_min_weight,
                "qualifier_influence": qualifier_influence_for_year(
                    year,
                    start_year=qualifier_blend_start_year,
                    full_weight_year=qualifier_full_weight_year,
                    minimum_influence=qualifier_minimum_influence,
                ),
            }
        )

    pred_df = pd.DataFrame(pred_rows)
    summary_df = pd.DataFrame(summary_rows)

    if len(pred_df):
        overall = {
            "test_year": "overall",
            "train_matches": np.nan,
            "test_matches": int(len(pred_df)),
            "result_accuracy": float(pred_df["correct_result"].mean()),
            "mean_result_log_loss": float(pred_df["result_log_loss"].mean()),
            "mean_result_brier": float(pred_df["result_brier"].mean()),
            "mean_goal_mae": float(pred_df["goal_mae"].mean()),
            "mean_goal_diff_abs_error": float(pred_df["goal_diff_abs_error"].mean()),
            "mean_exact_score_log_loss": float(pred_df["exact_score_log_loss"].mean()),
            "mean_actual_result_probability": float(pred_df["actual_result_probability"].mean()),
            "mean_exact_score_probability": float(pred_df["exact_score_probability"].mean()),
            "features_used": np.nan,
            "event_targets": "",
            "model_type": model_type,
            "recency_half_life_years": recency_half_life_years,
            "recency_min_weight": recency_min_weight,
            "qualifier_influence": np.nan,
        }
        summary_df = pd.concat([summary_df, pd.DataFrame([overall])], ignore_index=True)

    return pred_df, summary_df


def run_model_comparison_backtest(
    zip_path: str,
    train_csv: Optional[str],
    test_csv: Optional[str],
    models: List[str],
    test_years: Optional[List[int]],
    outdir: Path,
    box_csv: Optional[str] = None,
    results_csv: Optional[str] = None,
    former_names_csv: Optional[str] = None,
    qualifier_blend_start_year: int = 2014,
    qualifier_full_weight_year: int = 2022,
    qualifier_minimum_influence: float = 0.0,
    recency_half_life_years: float = 16.0,
    recency_min_weight: float = 0.10,
) -> pd.DataFrame:
    rows = []
    for model_name in models:
        pred_df, summary_df = chronological_backtest(
            zip_path=zip_path,
            train_csv=train_csv,
            test_csv=test_csv,
            model_type=model_name,
            test_years=test_years,
            box_csv=box_csv,
            results_csv=results_csv,
            former_names_csv=former_names_csv,
            qualifier_blend_start_year=qualifier_blend_start_year,
            qualifier_full_weight_year=qualifier_full_weight_year,
            qualifier_minimum_influence=qualifier_minimum_influence,
            recency_half_life_years=recency_half_life_years,
            recency_min_weight=recency_min_weight,
        )
        pred_df.to_csv(outdir / f"backtest_predictions_{model_name}.csv", index=False)
        summary_df.to_csv(outdir / f"backtest_summary_{model_name}.csv", index=False)
        if len(summary_df):
            overall = summary_df[summary_df["test_year"].astype(str) == "overall"]
            if len(overall):
                row = overall.iloc[0].to_dict()
                row["model_type"] = model_name
                rows.append(row)
    comparison = pd.DataFrame(rows)
    if len(comparison):
        comparison = comparison.sort_values(["mean_result_log_loss", "mean_result_brier"], ascending=True)
    comparison.to_csv(outdir / "backtest_model_comparison.csv", index=False)
    return comparison


def _require_matplotlib() -> None:
    if plt is None:
        raise ImportError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        )


def _save_plot(fig: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_result_probabilities(pred: Dict[str, Any], outdir: Path) -> Path:
    _require_matplotlib()
    res = pred["result_probabilities"]
    labels = [f"{pred['team_a']} win", "Draw", f"{pred['team_b']} win"]
    values = [res["team_a_win"], res["draw"], res["team_b_win"]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(
        labels,
        values,
        color=["#2ecc71", "#f39c12", "#e74c3c"],
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_title(
        f"Match Result Probabilities: {pred['team_a']} vs {pred['team_b']}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim(0, max(values) * 1.18)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.1%}",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
    return _save_plot(fig, outdir / "result_probabilities.png")


def plot_top_scorelines(
    pred: Dict[str, Any],
    outdir: Path,
    top_n: int = 12,
) -> Path:
    _require_matplotlib()
    top = pred["top_scorelines"][:top_n][::-1]
    labels = [f"{row['team_a_goals']}-{row['team_b_goals']}" for row in top]
    values = [row["probability"] for row in top]
    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.3 * len(labels))))
    ax.barh(
        labels,
        values,
        color="#3498db",
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )
    ax.set_xlabel("Probability", fontsize=12)
    ax.set_title(
        f"Top Exact Scores: {pred['team_a']} vs {pred['team_b']}",
        fontsize=14,
        fontweight="bold",
    )
    for index, value in enumerate(values):
        ax.text(value + 0.003, index, f"{value:.1%}", va="center", fontsize=10)
    return _save_plot(fig, outdir / "top_scorelines.png")


def plot_scoreline_heatmap(
    pred: Dict[str, Any],
    outdir: Path,
    max_goals: int = 7,
) -> Path:
    _require_matplotlib()
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for row in pred["scoreline_probabilities"]:
        goals_a = int(row["team_a_goals"])
        goals_b = int(row["team_b_goals"])
        if goals_a <= max_goals and goals_b <= max_goals:
            matrix[goals_b, goals_a] = float(row["probability"])

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xlabel(f"{pred['team_a']} Goals", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"{pred['team_b']} Goals", fontsize=12, fontweight="bold")
    ax.set_title("Scoreline Probability Heatmap", fontsize=14, fontweight="bold")
    ax.set_xticks(range(max_goals + 1))
    ax.set_yticks(range(max_goals + 1))
    fig.colorbar(image, ax=ax, label="Probability")
    threshold = matrix.max() * 0.55
    for goals_b in range(max_goals + 1):
        for goals_a in range(max_goals + 1):
            value = matrix[goals_b, goals_a]
            ax.text(
                goals_a,
                goals_b,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=9,
            )
    return _save_plot(fig, outdir / "scoreline_heatmap.png")


def plot_goal_spread(pred: Dict[str, Any], outdir: Path) -> Path:
    _require_matplotlib()
    spreads = sorted(
        (int(spread), float(probability))
        for spread, probability in pred["spread_probabilities"].items()
    )
    labels = [f"{spread:+d}" if spread else "0" for spread, _ in spreads]
    values = [probability for _, probability in spreads]
    colors = [
        "#e74c3c" if spread < 0 else "#2ecc71" if spread > 0 else "#f39c12"
        for spread, _ in spreads
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        labels,
        values,
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )
    ax.set_xlabel(f"Goal Spread ({pred['team_a']} - {pred['team_b']})", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_title("Goal Spread Distribution", fontsize=14, fontweight="bold")
    return _save_plot(fig, outdir / "goal_spread.png")


def plot_total_goals(pred: Dict[str, Any], outdir: Path) -> Path:
    _require_matplotlib()
    totals = sorted(
        (int(total), float(probability))
        for total, probability in pred["total_goal_probabilities"].items()
        if float(probability) > 0.001
    )
    labels = [str(total) for total, _ in totals]
    values = [probability for _, probability in totals]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        labels,
        values,
        color="#9b59b6",
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )
    ax.set_xlabel("Total Goals", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_title("Total Goals Distribution", fontsize=14, fontweight="bold")
    return _save_plot(fig, outdir / "total_goals.png")


def plot_over_under(pred: Dict[str, Any], outdir: Path) -> Path:
    _require_matplotlib()
    probabilities = pred["over_under_probabilities"]
    lines = sorted(
        float(key.split("_", 1)[1])
        for key in probabilities
        if key.startswith("over_")
    )
    positions = np.arange(len(lines))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        positions - width / 2,
        [probabilities[f"over_{line}"] for line in lines],
        width,
        label="Over",
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )
    ax.bar(
        positions + width / 2,
        [probabilities[f"under_{line}"] for line in lines],
        width,
        label="Under",
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )
    ax.set_xlabel("Goal Line", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_title("Over/Under Probabilities", fontsize=14, fontweight="bold")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{line:.1f}" for line in lines])
    ax.legend(fontsize=11)
    return _save_plot(fig, outdir / "over_under.png")


def plot_event_expectations(
    pred: Dict[str, Any],
    outdir: Path,
) -> Optional[Path]:
    _require_matplotlib()
    events = pred.get("event_predictions", {})
    if not events:
        return None

    event_names = list(events)
    team_a_values = [
        events[event].get(f"expected_{pred['team_a']}", 0.0)
        for event in event_names
    ]
    team_b_values = [
        events[event].get(f"expected_{pred['team_b']}", 0.0)
        for event in event_names
    ]
    positions = np.arange(len(event_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(
        positions - width / 2,
        team_a_values,
        width,
        label=pred["team_a"],
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )
    ax.bar(
        positions + width / 2,
        team_b_values,
        width,
        label=pred["team_b"],
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )
    ax.set_xlabel("Event Type", fontsize=12)
    ax.set_ylabel("Expected Count", fontsize=12)
    ax.set_title("Expected Event Predictions", fontsize=14, fontweight="bold")
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [event.replace("_", " ").title() for event in event_names],
        rotation=45,
        ha="right",
    )
    ax.legend(fontsize=11)
    return _save_plot(fig, outdir / "event_expectations.png")


def plot_prediction_outputs(
    pred: Dict[str, Any],
    outdir: Path,
) -> List[Path]:
    """Create the complete separate-plot suite used by test_output."""
    _require_matplotlib()
    plot_dir = outdir / "plots"
    paths = [
        plot_result_probabilities(pred, plot_dir),
        plot_top_scorelines(pred, plot_dir),
        plot_scoreline_heatmap(pred, plot_dir),
        plot_goal_spread(pred, plot_dir),
        plot_total_goals(pred, plot_dir),
        plot_over_under(pred, plot_dir),
    ]
    event_path = plot_event_expectations(pred, plot_dir)
    if event_path is not None:
        paths.append(event_path)
    return paths


def plot_prediction_dashboard(pred: Dict[str, Any], outdir: Path) -> Path:
    """Create the older combined dashboard on demand."""
    _require_matplotlib()
    outdir.mkdir(parents=True, exist_ok=True)
    team_a = pred["team_a"]
    team_b = pred["team_b"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    result_probabilities = pred["result_probabilities"]
    result_labels = [f"{team_a} win", "Draw", f"{team_b} win"]
    result_values = [
        result_probabilities["team_a_win"],
        result_probabilities["draw"],
        result_probabilities["team_b_win"],
    ]
    bars = ax.bar(
        result_labels,
        result_values,
        color=["#2e86de", "#f5b041", "#e74c3c"],
    )
    ax.set_title("Result Probabilities")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    for bar, value in zip(bars, result_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.1%}",
            ha="center",
        )

    ax = axes[0, 1]
    top_scores = pred["top_scorelines"][:10][::-1]
    score_labels = [
        f"{score['team_a_goals']}-{score['team_b_goals']}"
        for score in top_scores
    ]
    score_values = [score["probability"] for score in top_scores]
    ax.barh(score_labels, score_values, color="#7dcea0")
    ax.set_title("Most Likely Scorelines")
    ax.set_xlabel("Probability")
    for index, value in enumerate(score_values):
        ax.text(value + 0.002, index, f"{value:.1%}", va="center", fontsize=9)

    ax = axes[1, 0]
    expected_goals = [pred["lambda_a"], pred["lambda_b"]]
    goal_bars = ax.bar(
        [team_a, team_b],
        expected_goals,
        color=["#5dade2", "#f1948a"],
    )
    ax.set_title("Expected Goals")
    ax.set_ylabel("Goals")
    ax.set_ylim(0, max(expected_goals) * 1.25)
    for bar, value in zip(goal_bars, expected_goals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.05,
            f"{value:.2f}",
            ha="center",
        )

    ax = axes[1, 1]
    event_predictions = pred.get("event_predictions", {})
    preferred_events = [
        "yellow_cards",
        "red_cards",
        "penalty_goals",
        "own_goals",
        "substitutions",
    ]
    event_names = [name for name in preferred_events if name in event_predictions]
    if event_names:
        positions = np.arange(len(event_names))
        width = 0.36
        values_a = [
            event_predictions[name].get(f"expected_{team_a}", np.nan)
            for name in event_names
        ]
        values_b = [
            event_predictions[name].get(f"expected_{team_b}", np.nan)
            for name in event_names
        ]
        ax.bar(positions - width / 2, values_a, width, label=team_a, color="#5dade2")
        ax.bar(positions + width / 2, values_b, width, label=team_b, color="#f1948a")
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [name.replace("_", " ").title() for name in event_names],
            rotation=25,
            ha="right",
        )
        ax.set_title("Expected Match Events")
        ax.set_ylabel("Expected count")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No event predictions available", ha="center", va="center")
        ax.set_axis_off()

    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.25)

    top_score = pred["top_scorelines"][0]
    fig.suptitle(
        f"{team_a} vs {team_b} Prediction\n"
        f"Most likely score: {top_score['team_a_goals']}-{top_score['team_b_goals']}",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plot_path = outdir / "single_match_prediction_dashboard.png"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def unique_output_dir(requested: str | Path) -> Path:
    """Return a new output directory without overwriting an earlier run."""
    requested_path = Path(requested)
    if not requested_path.exists():
        return requested_path

    counter = 2
    while True:
        candidate = requested_path.with_name(
            f"{requested_path.name}_{counter}"
        )
        if not candidate.exists():
            return candidate
        counter += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--worldcupsai-zip",
        default=str(DATA_DIR / "worldcupsai.zip"),
        help="WorldCupSAI archive (default: data/worldcupsai.zip).",
    )
    ap.add_argument(
        "--team-train",
        default=str(DATA_DIR / "current_team_features_2026.csv"),
        help=(
            "Current team-feature CSV (default: "
            "data/current_team_features_2026.csv)."
        ),
    )
    ap.add_argument("--team-test", help="Optional second current team-feature CSV.")
    ap.add_argument(
        "--box-data",
        default=str(DATA_DIR / "FIFAallMatchBoxData.csv"),
        help="FIFA box-score data (default: data/FIFAallMatchBoxData.csv).",
    )
    ap.add_argument(
        "--results-data",
        default=str(DATA_DIR / "results.csv"),
        help="International results; only FIFA World Cup qualification is used.",
    )
    ap.add_argument(
        "--former-names",
        default=str(DATA_DIR / "former_names.csv"),
        help="Country-name normalization (default: data/former_names.csv).",
    )
    ap.add_argument(
        "--prediction-year",
        type=int,
        default=2026,
        help="Year used for live qualification-feature influence.",
    )
    ap.add_argument(
        "--qualifier-blend-start-year",
        type=int,
        default=2014,
        help="Year when explicit results.csv qualifier influence starts.",
    )
    ap.add_argument(
        "--qualifier-full-weight-year",
        type=int,
        default=2022,
        help="Year when explicit qualifier features reach full influence.",
    )
    ap.add_argument(
        "--qualifier-minimum-influence",
        type=float,
        default=0.0,
        help="Minimum explicit qualifier influence before the blend start year.",
    )
    ap.add_argument("--team-a")
    ap.add_argument("--team-b")
    ap.add_argument("--host-a", action="store_true")
    ap.add_argument("--host-b", action="store_true")
    ap.add_argument("--knockout", action="store_true")
    ap.add_argument("--model", default="ensemble", choices=["ensemble", "hgb", "rf", "poisson", "lightgbm", "xgboost", "catboost"])
    ap.add_argument("--outdir", default="outputs_v11_wcq_v9_base")
    ap.add_argument(
        "--recency-half-life-years",
        type=float,
        default=16.0,
        help="Years for a historical match's training weight to halve.",
    )
    ap.add_argument(
        "--recency-min-weight",
        type=float,
        default=0.10,
        help="Minimum raw training weight retained for old matches.",
    )

    ap.add_argument("--backtest", action="store_true", help="Run chronological expanding-window World Cup backtest.")
    ap.add_argument("--test-years", nargs="*", type=int, default=None, help="Specific World Cup years to test, e.g. --test-years 2014 2018 2022.")
    ap.add_argument("--compare-models", action="store_true", help="Backtest several model types and create a comparison table.")
    ap.add_argument("--comparison-models", nargs="*", default=["poisson", "rf", "ensemble"], help="Models to compare in backtest mode.")
    ap.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not generate the single-match prediction dashboard.",
    )
    args = ap.parse_args()

    out = unique_output_dir(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    if out != Path(args.outdir):
        print(f"Output directory exists; using: {out}")

    if args.backtest:
        if args.compare_models:
            comparison = run_model_comparison_backtest(
                zip_path=args.worldcupsai_zip,
                train_csv=args.team_train,
                test_csv=args.team_test,
                models=args.comparison_models,
                test_years=args.test_years,
                outdir=out,
                box_csv=args.box_data,
                results_csv=args.results_data,
                former_names_csv=args.former_names,
                qualifier_blend_start_year=args.qualifier_blend_start_year,
                qualifier_full_weight_year=args.qualifier_full_weight_year,
                qualifier_minimum_influence=args.qualifier_minimum_influence,
                recency_half_life_years=args.recency_half_life_years,
                recency_min_weight=args.recency_min_weight,
            )
            print(comparison.to_string(index=False))
            return

        pred_df, summary_df = chronological_backtest(
            zip_path=args.worldcupsai_zip,
            train_csv=args.team_train,
            test_csv=args.team_test,
            model_type=args.model,
            test_years=args.test_years,
            box_csv=args.box_data,
            results_csv=args.results_data,
            former_names_csv=args.former_names,
            qualifier_blend_start_year=args.qualifier_blend_start_year,
            qualifier_full_weight_year=args.qualifier_full_weight_year,
            qualifier_minimum_influence=args.qualifier_minimum_influence,
            recency_half_life_years=args.recency_half_life_years,
            recency_min_weight=args.recency_min_weight,
        )
        pred_df.to_csv(out / "backtest_predictions.csv", index=False)
        summary_df.to_csv(out / "backtest_summary.csv", index=False)
        print(summary_df.to_string(index=False))
        return

    if not args.team_a or not args.team_b:
        raise SystemExit("For single-match prediction, provide --team-a and --team-b. For backtesting, use --backtest.")

    model, data = build_from_zip(
        zip_path=args.worldcupsai_zip,
        train_csv=args.team_train,
        test_csv=args.team_test,
        model_type=args.model,
        box_csv=args.box_data,
        results_csv=args.results_data,
        former_names_csv=args.former_names,
        prediction_year=args.prediction_year,
        qualifier_blend_start_year=args.qualifier_blend_start_year,
        qualifier_full_weight_year=args.qualifier_full_weight_year,
        qualifier_minimum_influence=args.qualifier_minimum_influence,
        recency_half_life_years=args.recency_half_life_years,
        recency_min_weight=args.recency_min_weight,
    )
    pred = model.predict(args.team_a, args.team_b, args.host_a, args.host_b, args.knockout)

    (out / "single_match_prediction.json").write_text(json.dumps(pred, indent=2))

    pd.DataFrame(pred["top_scorelines"]).to_csv(out / "scoreline_probabilities_top.csv", index=False)
    pd.DataFrame(pred["scoreline_probabilities"]).to_csv(out / "scoreline_probabilities.csv", index=False)

    data.matches.to_csv(out / "normalized_worldcupsai_matches.csv", index=False)
    data.training_frame.to_csv(out / "training_frame.csv", index=False)
    plot_paths = []
    if not args.no_plots:
        plot_paths = plot_prediction_outputs(pred, out)

    report = {
        "version": "v11-wcq-v9-base-blended",
        "base_model": "v9_today_predictions.py",
        "n_matches": int(len(data.matches)),
        "n_training_rows": int(len(data.training_frame)),
        "n_box_team_rows": int(len(data.box_frame)),
        "n_current_team_rows": int(len(data.team_current)),
        "current_strength_profile_teams": len(
            getattr(model, "current_strength", {})
        ),
        "current_strength_source": args.team_train,
        "qualification_source": getattr(model, "qualifier_source", ""),
        "qualification_team_rows": getattr(
            model,
            "qualifier_source_rows",
            0,
        ),
        "qualification_profile_teams": len(
            getattr(model, "qualifier_profiles", {})
        ),
        "qualification_blend": {
            "prediction_year": args.prediction_year,
            "start_year": args.qualifier_blend_start_year,
            "full_weight_year": args.qualifier_full_weight_year,
            "minimum_influence": args.qualifier_minimum_influence,
            "live_influence": qualifier_influence_for_year(
                args.prediction_year,
                start_year=args.qualifier_blend_start_year,
                full_weight_year=args.qualifier_full_weight_year,
                minimum_influence=args.qualifier_minimum_influence,
            ),
        },
        "n_features": len(model.feature_cols),
        "features": model.feature_cols,
        "event_targets": data.event_columns,
        "unavailable_event_targets": ["corners", "passes", "pass_accuracy", "xG", "offsides"],
        "box_data_targets": getattr(model, "box_targets", []),
        "competition_filter": "FIFA Men's World Cup only",
        "draw_model_features": getattr(model, "draw_feature_cols", []),
        "recency_weighting": getattr(model, "recency_weight_summary", {}),
        "model_type": args.model,
        "ensemble_goal_models": [name for name, _, _ in model.goal_a_models] if args.model == "ensemble" else [],
        "ensemble_goal_difference_models": [name for name, _, _ in model.goal_diff_models] if args.model == "ensemble" else [],
        "ensemble_result_models": [name for name, _, _ in model.result_models] if args.model == "ensemble" else [],
        "added_vs_v7": [
            "men's World Cup training data only",
            "balanced binary draw classifier with chronological calibration",
            "two-stage draw versus conditional winner probabilities",
            "Dixon-Coles low-score correction",
            "recency-aware blend of explicit results.csv and fallback qualification features",
            "year-based exponential recency weighting",
            "chronological Elo features",
            "temperature smoothing of result probabilities",
            "explicit current-strength correction",
            "goal-difference ensemble model",
            "exact-score output treated as derived, not primary objective",
            "chronological expanding-window backtesting",
            "Kaggle box-score event-stat layer for shots, shots on target, possession, fouls, saves, yellow cards, and red cards",
        ],
        "metric_priority": [
            "result log loss",
            "Brier score",
            "result accuracy",
            "goal difference MAE",
            "over/under Brier score",
            "exact-score probability",
        ],
    }
    (out / "training_report.json").write_text(json.dumps(report, indent=2))

    print(json.dumps({"prediction": pred, "report": report}, indent=2))
    for plot_path in plot_paths:
        print(f"\nWrote: {plot_path}")


if __name__ == "__main__":
    main()

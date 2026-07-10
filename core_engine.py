"""Namespace-isolated module bundle.

Each embedded source block below is one original file's content, byte-for-byte
unchanged. `_load_submodule` executes it into its own `types.ModuleType` and
registers it in `sys.modules` under its original filename-derived name, so
every `import vNN_x as vNN` statement elsewhere in this project keeps working
exactly as it did when these were separate files.
"""
from __future__ import annotations

import sys
import types


def _load_submodule(name: str, source: str, filename: str) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    mod.__file__ = filename
    sys.modules[name] = mod
    exec(compile(source, filename, "exec"), mod.__dict__)
    return mod


# ======================================================================
# v11_wcq_results_model.py  (bundled as an isolated sub-module)
# ======================================================================
_V11_WCQ_RESULTS_MODEL_SOURCE = r'''
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
    "bosnia–herz": "Bosnia and Herzegovina",
    "bosnia-herz": "Bosnia and Herzegovina",
    "czech republic": "Czechia",
    "türkiye": "Turkey",
    "curacao": "Curaçao",
    "bosnia": "Bosnia and Herzegovina",
    "bosnia-herzegovina": "Bosnia and Herzegovina",
    "bosnia herzegovina": "Bosnia and Herzegovina",
    "bosnia and herzegovina": "Bosnia and Herzegovina",
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


def combine_training_weights(
    frame: pd.DataFrame,
    recency_weights: pd.Series,
) -> pd.Series:
    """Combine recency with an optional per-row tournament prestige."""
    weights = recency_weights.astype(float).copy()
    if "prestige_weight" in frame.columns:
        prestige = pd.to_numeric(
            frame["prestige_weight"],
            errors="coerce",
        ).fillna(1.0)
        weights = weights * prestige.clip(lower=0.01)
    mean_weight = float(weights.mean())
    if not np.isfinite(mean_weight) or mean_weight <= 0:
        return pd.Series(1.0, index=frame.index, dtype=float)
    return weights / mean_weight


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


# Tournament importance tiers, following the well-established "World Football
# Elo Ratings" K-factor convention (eloratings.net): World Cup finals = 60,
# major continental championship finals = 50, World Cup/continental
# qualifiers and Nations Leagues = 40, other recognized tournaments = 30,
# friendlies = 20. Used as `prestige_weight` so a World Cup match still
# dominates a friendly even though both are "just one row" in training data.
MAJOR_CONTINENTAL_FINALS = {
    "UEFA Euro",
    "Copa América",
    "African Cup of Nations",
    "AFC Asian Cup",
    "Confederations Cup",
    "Gold Cup",
    "CONCACAF Championship",
}


def _prestige_weight_for_tournament(tournament: str) -> float:
    t = str(tournament).strip()
    if t == "FIFA World Cup":
        return 60.0
    if t in MAJOR_CONTINENTAL_FINALS:
        return 50.0
    if t == "Friendly":
        return 20.0
    if "qualification" in t.lower() or "Nations League" in t:
        return 40.0
    return 30.0


# --- BEGIN volume-normalized prestige weighting (opt-in, off by default) ------------------
# Self-contained addition: safe to delete this block, the `prestige_tier` column
# assignments in load_expanded_competition_matches()/chronological_backtest()/
# chronological_backtest_pytorch_focal(), and the use_volume_normalized_weighting
# parameter (plus its one call site per function) to fully revert to the flat
# wc_prestige_weight scheme, with no other code depending on it.
#
# The flat scheme (_prestige_weight_for_tournament) sets a fixed per-row weight
# per tier, but combine_training_weights() normalizes by the *overall* mean
# weight across all rows, not per tier -- so a tier's actual aggregate
# contribution to training is row_count x avg_prestige, not just avg_prestige.
# World Cup rows are outnumbered ~47:1 by everything else, so a fixed
# wc_prestige_weight has to be tuned very high (600, ~10x the naive per-row
# tier ordering) to reach a meaningful aggregate share, and that tuned value
# would need to be re-derived if the pool's composition changes. This scheme
# instead targets an aggregate *share* directly and self-adjusts to whatever
# row counts are actually present in a given training fold.
DEFAULT_PRESTIGE_TIER_TARGET_SHARES: Dict[str, float] = {
    "world_cup": 0.35,
    "continental_final": 0.25,
    "qualifier_or_nations_league": 0.20,
    "other": 0.12,
    "friendly": 0.08,
}


def _prestige_tier_for_tournament(tournament: str) -> str:
    t = str(tournament).strip()
    if t == "FIFA World Cup":
        return "world_cup"
    if t in MAJOR_CONTINENTAL_FINALS:
        return "continental_final"
    if t == "Friendly":
        return "friendly"
    if "qualification" in t.lower() or "Nations League" in t:
        return "qualifier_or_nations_league"
    return "other"


def assign_volume_normalized_weights(
    frame: pd.DataFrame,
    target_shares: Dict[str, float] = DEFAULT_PRESTIGE_TIER_TARGET_SHARES,
) -> pd.Series:
    """Per-row weight = target_shares[tier] / (row count of that tier in `frame`),
    so each tier's aggregate contribution to the *normalized* training weight
    approximates its target share regardless of how many rows it has. Must be
    called on the already-year-filtered per-fold training frame (row counts
    are fold-specific, since the expanding window changes tier composition
    every year), not on the full match pool once upfront.
    """
    tier_counts = frame["prestige_tier"].value_counts()
    weights = frame["prestige_tier"].map(
        lambda tier: target_shares.get(tier, min(target_shares.values())) / max(tier_counts.get(tier, 1), 1)
    )
    return weights.astype(float)
# --- END volume-normalized prestige weighting ----------------------------------------------


def load_expanded_competition_matches(
    results_csv: Optional[str],
    former_names_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Every non-World-Cup-finals match in results.csv (Euro, Copa América,
    AFCON, Asian Cup, their qualifiers, Nations Leagues, friendlies, and the
    long tail of regional/minor tournaments), reshaped to the same schema
    WorldCupSAILoader.load_matches() produces so it can be concatenated onto
    the curated World Cup match history as additional training rows -- not
    just an auxiliary feature -- each tagged with a tournament `prestige_weight`
    that combine_training_weights() already knows how to apply.

    World Cup finals matches are excluded here (tournament == "FIFA World
    Cup") since the curated WorldCupSAILoader dataset already provides those
    with richer metadata (real host flags, knockout stage, etc.); mixing in
    results.csv's own FIFA World Cup rows would double-count them.

    Simplifications, since results.csv has no stage/confederation detail:
    is_knockout is always False and same_confed is always 0 for these rows --
    an accepted precision loss, not a correctness bug (the model still uses
    every other feature; these two are just uninformative for this slice).
    """
    if not results_csv or not Path(results_csv).exists():
        return pd.DataFrame()

    raw = pd.read_csv(results_csv)
    raw = raw[raw["tournament"].astype(str) != "FIFA World Cup"]
    raw = raw[~raw["tournament"].astype(str).str.contains("CONIFA", na=False)]
    raw = raw.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"]).copy()

    name_map: Dict[str, str] = {}
    if former_names_csv and Path(former_names_csv).exists():
        former = pd.read_csv(former_names_csv)
        if {"current", "former"}.issubset(former.columns):
            for _, row in former.iterrows():
                name_map[canon_team(row["former"])] = canon_team(row["current"])

    def normalize_name(name: str) -> str:
        canonical = canon_team(name)
        return name_map.get(canonical, canonical)

    date = pd.to_datetime(raw["date"], errors="coerce")
    neutral = raw["neutral"].astype(str).str.strip().str.lower().isin({"1", "true", "yes"})

    df = pd.DataFrame(
        {
            "match_id": ["results_" + str(i) for i in raw.index],
            "date": date,
            "year": date.dt.year,
            "team_a": raw["home_team"].map(normalize_name),
            "team_b": raw["away_team"].map(normalize_name),
            "goals_a": pd.to_numeric(raw["home_score"], errors="coerce"),
            "goals_b": pd.to_numeric(raw["away_score"], errors="coerce"),
            "host_a": (~neutral).astype(int),
            "host_b": 0,
            "is_group_stage": 1,
            "is_knockout": 0,
            "same_confed": 0,
            "prestige_weight": raw["tournament"].map(_prestige_weight_for_tournament),
            "prestige_tier": raw["tournament"].map(_prestige_tier_for_tournament),
        }
    )
    return df.dropna(subset=["date", "goals_a", "goals_b"]).reset_index(drop=True)


def load_fbref_world_cup_matches(csv_path: Optional[str]) -> pd.DataFrame:
    """Reshape a pull_fbref_world_cup_matches.py CSV into the same schema
    WorldCupSAILoader.load_matches() produces, so it can extend `all_matches`
    itself (not the generic expanded pool) for World Cup years the curated
    worldcupsai.zip dataset doesn't have -- currently 2026, since that
    dataset stops at 2022 and the tournament is live. Concatenating onto
    `all_matches` (rather than only the training pool) makes these rows
    available both as training signal for later folds and as real
    `test_matches` when a caller passes 2026 as a test year.

    Simplifications, since FBref's schedule table has no host/confederation
    detail beyond a venue string: host_a/host_b are always 0 (every 2026
    match is explicitly at a "(Neutral Site)" venue -- true co-hosting, not
    an omission) and same_confed is always 0, matching the same accepted
    precision loss load_expanded_competition_matches() already makes.
    """
    if not csv_path or not Path(csv_path).exists():
        return pd.DataFrame()

    raw = pd.read_csv(csv_path)
    raw = raw.dropna(subset=["score", "home_team", "away_team", "date"]).copy()
    if raw.empty:
        return pd.DataFrame()

    # Penalty shootouts are formatted as "(3) 1–1 (4)" (regulation/ET score in
    # the middle, penalty counts in parens) -- strip the parenthesized parts
    # so goals_a/goals_b are always the regulation+ET score, not penalties.
    score_clean = raw["score"].astype(str).str.replace(r"\(\d+\)", "", regex=True).str.strip()
    goals = score_clean.str.split(r"[–‒-]", n=1, expand=True)
    raw["goals_a"] = pd.to_numeric(goals[0], errors="coerce")
    raw["goals_b"] = pd.to_numeric(goals[1], errors="coerce")

    date = pd.to_datetime(raw["date"], errors="coerce")
    is_knockout = raw["round"].astype(str).str.strip().ne("Group stage")

    df = pd.DataFrame(
        {
            "match_id": raw.get("game_id", pd.Series(range(len(raw)))).astype(str),
            "date": date,
            "year": date.dt.year,
            "team_a": raw["home_team"].map(canon_team),
            "team_b": raw["away_team"].map(canon_team),
            "goals_a": raw["goals_a"],
            "goals_b": raw["goals_b"],
            "host_a": 0,
            "host_b": 0,
            "is_group_stage": (~is_knockout).astype(int),
            "is_knockout": is_knockout.astype(int),
            "same_confed": 0,
            "stage_name": raw["round"],
        }
    )
    return df.dropna(subset=["date", "goals_a", "goals_b"]).reset_index(drop=True)


# Maps pull_fbref_international_matches.py's FBref league codes onto the same
# canonical tournament-name vocabulary results.csv uses, so this reuses
# _prestige_tier_for_tournament()/_prestige_weight_for_tournament() directly
# instead of duplicating the tier logic.
FBREF_LEAGUE_TO_TOURNAMENT_NAME = {
    "INT-European Championship": "UEFA Euro",
    "INT-Copa America": "Copa América",
    "INT-Africa Cup of Nations": "African Cup of Nations",
    "INT-CONCACAF Gold Cup": "Gold Cup",
    "INT-AFC Asian Cup": "AFC Asian Cup",
    "INT-Nations League": "UEFA Nations League",
    "INT-Friendlies": "Friendly",
}

_KNOCKOUT_ROUND_PATTERN = r"final|round of|quarter|semi|third-place|play-?off"


def load_fbref_international_matches(csv_path: Optional[str]) -> pd.DataFrame:
    """Reshape a pull_fbref_international_matches.py CSV (Euro, Copa América,
    AFCON, Gold Cup, Asian Cup, Nations League, and men's friendlies,
    2022-2026) into the same schema load_expanded_competition_matches()
    produces, so it extends the same "everything else" training pool --
    this is a fresher, more current equivalent of what results.csv already
    provides for those competitions in earlier years.

    Simplifications, matching load_expanded_competition_matches(): same_confed
    is always 0. is_knockout is inferred from the `round` label (Nations
    League's "League A/B/C/D" group phase and "Group stage" elsewhere count
    as non-knockout; "Quarter-finals"/"Semi-finals"/"Final"/"Round of N"/
    "Third-place match"/"play-off" count as knockout).
    """
    if not csv_path or not Path(csv_path).exists():
        return pd.DataFrame()

    raw = pd.read_csv(csv_path)
    raw = raw.dropna(subset=["score", "home_team", "away_team", "date"]).copy()
    if raw.empty:
        return pd.DataFrame()

    score_clean = raw["score"].astype(str).str.replace(r"\(\d+\)", "", regex=True).str.strip()
    goals = score_clean.str.split(r"[–‒-]", n=1, expand=True)
    raw["goals_a"] = pd.to_numeric(goals[0], errors="coerce")
    raw["goals_b"] = pd.to_numeric(goals[1], errors="coerce")

    date = pd.to_datetime(raw["date"], errors="coerce")
    is_knockout = raw["round"].astype(str).str.contains(_KNOCKOUT_ROUND_PATTERN, case=False, regex=True)
    neutral = raw["venue"].astype(str).str.contains("Neutral Site", case=False)
    tournament_name = raw["source_league"].map(FBREF_LEAGUE_TO_TOURNAMENT_NAME).fillna("Friendly")

    df = pd.DataFrame(
        {
            "match_id": "fbref_" + raw.get("game_id", pd.Series(range(len(raw)))).astype(str),
            "date": date,
            "year": date.dt.year,
            "team_a": raw["home_team"].map(canon_team),
            "team_b": raw["away_team"].map(canon_team),
            "goals_a": raw["goals_a"],
            "goals_b": raw["goals_b"],
            "host_a": (~neutral).astype(int),
            "host_b": 0,
            "is_group_stage": (~is_knockout).astype(int),
            "is_knockout": is_knockout.astype(int),
            "same_confed": 0,
            "prestige_weight": tournament_name.map(_prestige_weight_for_tournament),
            "prestige_tier": tournament_name.map(_prestige_tier_for_tournament),
        }
    )
    return df.dropna(subset=["date", "goals_a", "goals_b"]).reset_index(drop=True)


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

        if "prestige_weight" in matches.columns:
            feat["prestige_weight"] = r.get("prestige_weight", 1.0)

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
        if c not in ["match_id", "date", "team_a", "team_b", "goals_a", "goals_b", "goal_diff", "prestige_weight"]
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
        score_matrix_variant: str = "poisson_dc",
        score_matrix_r: float = 25.0,
    ):
        self.model_type = model_type
        self.recency_half_life_years = float(recency_half_life_years)
        self.recency_min_weight = float(recency_min_weight)
        self.score_matrix_variant = score_matrix_variant
        self.score_matrix_r = float(score_matrix_r)
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
            "continental_abs_gd_diff",
            "continental_mean_draw_rate",
            "continental_abs_draw_rate_diff",
            "continental_expected_total",
            "continental_min_matches_seen",
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
        weights = combine_training_weights(ordered, weights)
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
        sample_weight = combine_training_weights(frame, sample_weight)
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
            valid = frame[f"{ev}_a"].notna() & frame[f"{ev}_b"].notna()
            if not valid.any():
                continue
            event_X = X.loc[valid]
            event_weight = sample_weight.loc[valid]
            if self.model_type == "ensemble":
                ma_models = self._named_regressors()
                mb_models = self._named_regressors()
                for _, model, _ in ma_models:
                    fit_with_sample_weight(
                        model,
                        event_X,
                        frame.loc[valid, f"{ev}_a"],
                        event_weight,
                    )
                for _, model, _ in mb_models:
                    fit_with_sample_weight(
                        model,
                        event_X,
                        frame.loc[valid, f"{ev}_b"],
                        event_weight,
                    )
                self.event_models[ev] = (ma_models, mb_models)
            else:
                ma, mb = self._regressor(), self._regressor()
                fit_with_sample_weight(
                    ma,
                    event_X,
                    frame.loc[valid, f"{ev}_a"],
                    event_weight,
                )
                fit_with_sample_weight(
                    mb,
                    event_X,
                    frame.loc[valid, f"{ev}_b"],
                    event_weight,
                )
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

        if self.score_matrix_variant == "bivariate_negbin_dc":
            from v49_bivariate_negbin_model import build_score_matrix as _v49_build_score_matrix

            score_probs = _v49_build_score_matrix(
                lam_a,
                lam_b,
                r=self.score_matrix_r,
                dc_rho=self.dixon_coles_rho,
                max_goals=max_goals,
            )
        else:
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


# --- BEGIN top3 / top3+outlier accuracy instrumentation -----------------------------------
# Self-contained addition: safe to delete this block plus its two call sites in
# chronological_backtest() (the top3_hit/top3_plus_outlier_hit computation and the
# corresponding summary columns) to fully revert, with no other code depending on it.
def _top3_and_outlier_hits(
    top_scorelines: List[Dict[str, Any]],
    score_probs: Dict[Tuple[int, int], float],
    lambda_a: float,
    lambda_b: float,
    actual_key: Tuple[int, int],
) -> Tuple[int, int]:
    """Returns (top3_hit, top3_plus_outlier_hit) as 0/1 ints.

    top3_hit: was the actual scoreline among the 3 highest-probability predicted
    scorelines. top3_plus_outlier_hit: same, plus one "coverage outlier" hedge
    scoreline (v39_coverage_outlier_model.select_coverage_outlier -- a
    higher-total-goals scoreline added only when lambda_a+lambda_b implies a
    plausibly higher-scoring game than the top-3 already covers). Lazy import
    to avoid a circular import (v39_coverage_outlier_model imports this module
    at its top level).
    """
    from v39_coverage_outlier_model import select_coverage_outlier

    top3 = top_scorelines[:3]
    top3_keys = {(int(s["team_a_goals"]), int(s["team_b_goals"])) for s in top3}
    top3_hit = int(actual_key in top3_keys)

    outlier, _ = select_coverage_outlier(
        score_probs,
        top3,
        lambda_a,
        lambda_b,
        use_observed_game_state_priors=False,
    )
    plus_keys = set(top3_keys)
    if outlier is not None:
        plus_keys.add((int(outlier["team_a_goals"]), int(outlier["team_b_goals"])))
    top3_plus_outlier_hit = int(actual_key in plus_keys)

    return top3_hit, top3_plus_outlier_hit
# --- END top3 / top3+outlier accuracy instrumentation -------------------------------------


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
    score_matrix_variant: str = "poisson_dc",
    score_matrix_r: float = 25.0,
    use_expanded_training_pool: bool = True,
    wc_prestige_weight: float = 60.0,
    use_volume_normalized_weighting: bool = False,
    prestige_tier_target_shares: Dict[str, float] = DEFAULT_PRESTIGE_TIER_TARGET_SHARES,
    fbref_world_cup_csv: Optional[str] = None,
    fbref_international_csv: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run expanding-window World Cup backtests.

    For each test year:
    - train on matches before that year
    - test on matches in that year

    This is the correct overfitting check because it prevents the model from
    learning from future tournaments.

    Test matches are always World Cup finals matches only (that's the thing
    being predicted). Training matches, when `use_expanded_training_pool` is
    True (the default), are the union of prior World Cup finals matches
    *plus* every other competition in results.csv (Euro, Copa América, AFCON,
    Asian Cup, their qualifiers, Nations Leagues, friendlies, etc. -- see
    load_expanded_competition_matches()), weighted by tournament prestige via
    combine_training_weights()'s existing `prestige_weight` column support.
    Set it to False to reproduce the old World-Cup-finals-only training set.

    Note combine_training_weights() normalizes by the overall mean weight
    across *all* rows, not per tournament tier -- so a tier's aggregate
    contribution to training is row_count x avg_prestige, not just
    avg_prestige. World Cup rows are outnumbered by everything else roughly
    47:1 in the expanded pool, so wc_prestige_weight=60 (the default, same as
    a continental-final-adjacent value) leaves World Cup matches at only ~4%
    of the total weighted training signal despite nominally having the
    highest per-row weight. Raise this (e.g. ~600 for a ~30% aggregate share
    at 2010-2022 pool sizes) to test whether that dilution actually matters.

    Set `use_volume_normalized_weighting=True` to replace the flat
    `wc_prestige_weight` scheme with per-fold target-aggregate-share weighting
    (see assign_volume_normalized_weights()) instead -- self-adjusting to each
    fold's actual tier composition rather than a single tuned constant.
    """
    loader = WorldCupSAILoader(zip_path, Path(str(zip_path) + "_extracted"))
    all_matches = loader.load_matches()
    # Extend with World Cup years the curated dataset doesn't have yet (2026
    # is live and ongoing) -- concatenated onto all_matches itself, not just
    # the training pool, so these rows are available both as training signal
    # and as real test_matches when 2026 is passed as a test year.
    fbref_wc_matches = load_fbref_world_cup_matches(fbref_world_cup_csv)
    if not fbref_wc_matches.empty:
        all_matches = pd.concat([all_matches, fbref_wc_matches], ignore_index=True, sort=False)
    all_matches["prestige_weight"] = wc_prestige_weight
    all_matches["prestige_tier"] = "world_cup"
    current = load_current_team_features(train_csv, test_csv)
    box = load_kaggle_box_data(box_csv)
    qualification_results = load_world_cup_qualification_results(
        results_csv,
        former_names_csv,
    )
    qualifier_source = (
        qualification_results if not qualification_results.empty else box
    )

    if use_expanded_training_pool:
        expanded_matches = load_expanded_competition_matches(results_csv, former_names_csv)
        fbref_intl_matches = load_fbref_international_matches(fbref_international_csv)
        train_pool = pd.concat(
            [all_matches, expanded_matches, fbref_intl_matches], ignore_index=True, sort=False
        )
    else:
        train_pool = all_matches

    all_years = sorted(int(y) for y in all_matches["year"].dropna().unique())
    if test_years is None:
        test_years = [y for y in all_years if y >= 2010]
    else:
        test_years = [int(y) for y in test_years]

    pred_rows = []
    summary_rows = []

    for year in test_years:
        train_matches = train_pool[(train_pool["year"] < year) & (train_pool["year"] >= min_train_year)].copy()
        test_matches = all_matches[all_matches["year"] == year].copy()

        if len(train_matches) < 80 or len(test_matches) == 0:
            continue

        if use_volume_normalized_weighting:
            train_matches["prestige_weight"] = assign_volume_normalized_weights(
                train_matches, prestige_tier_target_shares
            )

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
                score_matrix_variant=score_matrix_variant,
                score_matrix_r=score_matrix_r,
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
            actual_key = (int(r.goals_a), int(r.goals_b))
            top3_hit, top3_plus_outlier_hit = _top3_and_outlier_hits(
                pred["top_scorelines"], score_probs, pred["lambda_a"], pred["lambda_b"], actual_key
            )

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
                    "top3_hit": top3_hit,
                    "top3_plus_outlier_hit": top3_plus_outlier_hit,
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
                "top3_accuracy": float(year_df["top3_hit"].mean()),
                "top3_plus_outlier_accuracy": float(year_df["top3_plus_outlier_hit"].mean()),
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
                "score_matrix_variant": score_matrix_variant,
                "score_matrix_r": score_matrix_r,
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
            "top3_accuracy": float(pred_df["top3_hit"].mean()),
            "top3_plus_outlier_accuracy": float(pred_df["top3_plus_outlier_hit"].mean()),
            "features_used": np.nan,
            "event_targets": "",
            "model_type": model_type,
            "recency_half_life_years": recency_half_life_years,
            "recency_min_weight": recency_min_weight,
            "qualifier_influence": np.nan,
            "score_matrix_variant": score_matrix_variant,
            "score_matrix_r": score_matrix_r,
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


def run_score_matrix_variant_backtest(
    zip_path: str,
    train_csv: Optional[str],
    test_csv: Optional[str],
    outdir: Path,
    model_type: str = "ensemble",
    variants: Optional[List[str]] = None,
    score_matrix_r: float = 25.0,
    test_years: Optional[List[int]] = None,
    box_csv: Optional[str] = None,
    results_csv: Optional[str] = None,
    former_names_csv: Optional[str] = None,
    qualifier_blend_start_year: int = 2014,
    qualifier_full_weight_year: int = 2022,
    qualifier_minimum_influence: float = 0.0,
    recency_half_life_years: float = 16.0,
    recency_min_weight: float = 0.10,
) -> pd.DataFrame:
    """Ablation: same lambda_a/lambda_b, same weighting, same expanding-window
    train/test splits as chronological_backtest -- only the final scoreline
    matrix construction (independent-Poisson+Dixon-Coles vs. v49's
    shared-frailty bivariate NegBin+Dixon-Coles) changes between rows."""
    variants = variants if variants is not None else ["poisson_dc", "bivariate_negbin_dc"]
    rows = []
    for variant in variants:
        pred_df, summary_df = chronological_backtest(
            zip_path=zip_path,
            train_csv=train_csv,
            test_csv=test_csv,
            model_type=model_type,
            test_years=test_years,
            box_csv=box_csv,
            results_csv=results_csv,
            former_names_csv=former_names_csv,
            qualifier_blend_start_year=qualifier_blend_start_year,
            qualifier_full_weight_year=qualifier_full_weight_year,
            qualifier_minimum_influence=qualifier_minimum_influence,
            recency_half_life_years=recency_half_life_years,
            recency_min_weight=recency_min_weight,
            score_matrix_variant=variant,
            score_matrix_r=score_matrix_r,
        )
        pred_df.to_csv(outdir / f"backtest_predictions_{variant}.csv", index=False)
        summary_df.to_csv(outdir / f"backtest_summary_{variant}.csv", index=False)
        if len(summary_df):
            overall = summary_df[summary_df["test_year"].astype(str) == "overall"]
            if len(overall):
                row = overall.iloc[0].to_dict()
                row["score_matrix_variant"] = variant
                rows.append(row)

    comparison = pd.DataFrame(rows)
    comparison.to_csv(outdir / "variant_comparison.csv", index=False)

    delta_rows = []
    if len(comparison) == 2:
        baseline = comparison[comparison["score_matrix_variant"] == "poisson_dc"].iloc[0]
        challenger = comparison[comparison["score_matrix_variant"] == "bivariate_negbin_dc"].iloc[0]
        metric_cols = [
            "mean_exact_score_log_loss",
            "mean_exact_score_probability",
            "mean_result_log_loss",
            "mean_result_brier",
            "result_accuracy",
            "top3_accuracy",
            "top3_plus_outlier_accuracy",
            "mean_goal_mae",
            "mean_goal_diff_abs_error",
        ]
        for metric in metric_cols:
            delta_rows.append(
                {
                    "metric": metric,
                    "poisson_dc": baseline[metric],
                    "bivariate_negbin_dc": challenger[metric],
                    "delta_bivariate_minus_poisson": float(challenger[metric]) - float(baseline[metric]),
                }
            )
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(outdir / "pairwise_deltas.csv", index=False)

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
    ap.add_argument("--outdir", default="outputs/outputs_v11_wcq_v9_base")
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
        "--score-matrix-variant",
        default="poisson_dc",
        choices=["poisson_dc", "bivariate_negbin_dc"],
        help="Scoreline matrix construction: independent Poisson+Dixon-Coles (default) or v49's shared-frailty bivariate NegBin+Dixon-Coles.",
    )
    ap.add_argument(
        "--score-matrix-r",
        type=float,
        default=25.0,
        help="Shared dispersion shape for --score-matrix-variant bivariate_negbin_dc (default 25.0, calibrated -- see calibrate_v49_dispersion.py).",
    )
    ap.add_argument(
        "--compare-score-matrix-variants",
        action="store_true",
        help="Backtest both score-matrix variants (same lambdas/weights/splits) and create a comparison table.",
    )
    ap.add_argument(
        "--wc-prestige-weight",
        type=float,
        default=60.0,
        help="Prestige weight for World Cup finals rows in the expanded training pool (default 60.0). World Cup rows are outnumbered ~47:1 by other competitions, so this stays only about 4 percent of the total weighted training signal at the default -- raise substantially (~600) to test a higher World-Cup-specific share.",
    )
    ap.add_argument(
        "--use-volume-normalized-weighting",
        action="store_true",
        help="Replace the flat wc-prestige-weight with per-fold volume-normalized tier weights (see assign_volume_normalized_weights()).",
    )
    ap.add_argument(
        "--fbref-world-cup-csv",
        default=None,
        help="Optional path to FBref-pulled World Cup matches CSV (e.g. data/fbref_world_cup_matches.csv) to extend WC coverage through the live tournament.",
    )
    ap.add_argument(
        "--fbref-international-csv",
        default=None,
        help="Optional path to FBref-pulled broader international matches CSV (e.g. data/fbref_international_matches.csv) to extend the expanded training pool through 2022-2026.",
    )
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

        if args.compare_score_matrix_variants:
            comparison = run_score_matrix_variant_backtest(
                zip_path=args.worldcupsai_zip,
                train_csv=args.team_train,
                test_csv=args.team_test,
                outdir=out,
                model_type=args.model,
                score_matrix_r=args.score_matrix_r,
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
            score_matrix_variant=args.score_matrix_variant,
            score_matrix_r=args.score_matrix_r,
            wc_prestige_weight=args.wc_prestige_weight,
            use_volume_normalized_weighting=args.use_volume_normalized_weighting,
            fbref_world_cup_csv=args.fbref_world_cup_csv,
            fbref_international_csv=args.fbref_international_csv,
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
'''
v11_wcq_results_model = _load_submodule("v11_wcq_results_model", _V11_WCQ_RESULTS_MODEL_SOURCE, "core_engine.py:v11_wcq_results_model")

# ======================================================================
# v49_bivariate_negbin_model.py  (bundled as an isolated sub-module)
# ======================================================================
_V49_BIVARIATE_NEGBIN_MODEL_SOURCE = r'''
"""V49: true bivariate Negative Binomial scoreline model.

Standalone module. Does not import or modify v11/v39/v42/v43/v46 -- it can be
dropped into any of those pipelines later by calling `score_matrix_from_prediction`
or `build_score_matrix`, but for now it only reads/writes its own outputs.

Why this exists
----------------
The pipeline that currently produces buy cards (v11 -> v39 -> v42 -> v46_4) builds
scorelines from *independent* Poisson marginals and then patches only the
(0,0)/(1,0)/(0,1)/(1,1) corner with a Dixon-Coles tau term. Every other cell --
every 2-3, 3-1, 4-2, etc. -- is exactly what independent Poisson says it is, with
zero overdispersion and zero cross-team correlation. v43 already swaps in
independent Negative Binomial marginals (fixing each team's own overdispersion),
but the two teams are still combined as pa[i] * pb[j]: no correlation, so a
"both teams get dragged into a chaotic, high-event match together" scenario still
isn't modeled anywhere in the tails.

This module fixes both gaps at once with a shared-frailty construction:

    Z ~ Gamma(shape=r, scale=1/r)      # E[Z] = 1, Var[Z] = 1/r ("match volatility")
    A | Z ~ Poisson(lambda_a * Z)
    B | Z ~ Poisson(lambda_b * Z)      # conditionally independent given Z

Both teams' goal counts share the same draw of Z, so a volatile match (large Z)
pushes both teams' scoring up together, and a shut-down match (small Z) pushes
both down together. Marginalizing Z out analytically (it's a standard
Poisson-Gamma mixture) gives a closed-form bivariate Negative Binomial --
sometimes called a "negative trinomial" -- with no numerical integration needed:

    P(A=i, B=j) = [Gamma(i+j+r) / (i! j! Gamma(r))] * theta^r * p_a^i * p_b^j

    where theta = r / (lambda_a + lambda_b + r)
          p_a   = lambda_a / (lambda_a + lambda_b + r)
          p_b   = lambda_b / (lambda_a + lambda_b + r)

This recovers NB marginals (Var(A) = lambda_a + lambda_a^2 / r, matching v43's
per-team overdispersion form when r == alpha) *and* a genuine positive
covariance Cov(A, B) = lambda_a * lambda_b / r across the whole grid, not just
a hand-patched 2x2 corner. As r -> inf, Var[Z] -> 0 and this collapses back to
plain independent Poisson; smaller r means more shared volatility and fatter,
more correlated tails in both directions (high-combined-score blowouts and
low-scoring mutual shutdowns).

Dixon-Coles is layered on top, unchanged in form, because it corrects a
different, well-documented effect (tactical behavior at low scores, e.g. sides
playing for a 1-1) that shared-frailty overdispersion doesn't capture.

Calibration of `r`
-------------------
DEFAULT_R = 25.0 is fit from data, not guessed -- see calibrate_v49_dispersion.py
and its output at outputs/v49_dispersion_calibration/calibration_summary.json.
That script fits a standard multiplicative attack/defense Poisson model
(attack_i * defense_j * home_advantage) to data/results.csv (2010-present,
~15k matches) to get a match-specific lambda_home/lambda_away, then estimates
`r` from the *residuals* against those lambdas -- naively plugging raw
(mean, variance) of goals scored into implied_r_from_moments would conflate
between-team strength heterogeneity with genuine overdispersion and imply a
much smaller (fatter-tailed) r than the data supports.

On the World Cup finals subset specifically (n=260, the actual target
domain -- qualifiers/friendlies include lopsided mismatches that behave
differently): r_marginal_pooled=27.4 and r_correlation=26.6 agree closely;
r_marginal_away_only=9.1 is noisier (n=130) and close to the old guessed
default of 8.0, which is probably a coincidence rather than validation, since
home-goals overdispersion comes out essentially infinite (Poisson-like, no
excess variance once attack/defense is controlled for) in both the finals
subset and the broader 2010-present sample across all competitions. r=25
follows the two numbers that agree with each other.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

ScoreKey = tuple[int, int]
ScoreMatrix = dict[ScoreKey, float]

DEFAULT_R = 25.0
DEFAULT_DC_RHO = -0.08


def normalize_matrix(matrix: ScoreMatrix) -> ScoreMatrix:
    total = sum(matrix.values())
    if total <= 0:
        return matrix
    return {k: v / total for k, v in matrix.items()}


def _log_pmf_shift(i: int, j: int, r: float) -> float:
    return (
        math.lgamma(i + j + r)
        - math.lgamma(i + 1)
        - math.lgamma(j + 1)
        - math.lgamma(r)
    )


def bivariate_negbin_matrix(
    lambda_a: float,
    lambda_b: float,
    *,
    r: float = DEFAULT_R,
    max_goals: int = 10,
) -> ScoreMatrix:
    """Shared-gamma-frailty bivariate Negative Binomial scoreline grid.

    `r` is the shared dispersion shape: r -> inf recovers independent Poisson;
    small r produces fat, positively-correlated tails in both directions.
    """
    lambda_a = max(float(lambda_a), 1e-9)
    lambda_b = max(float(lambda_b), 1e-9)
    r = max(float(r), 1e-6)
    denom = lambda_a + lambda_b + r
    log_theta = math.log(r) - math.log(denom)
    log_pa = math.log(lambda_a) - math.log(denom)
    log_pb = math.log(lambda_b) - math.log(denom)

    matrix: ScoreMatrix = {}
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            log_p = _log_pmf_shift(i, j, r) + r * log_theta + i * log_pa + j * log_pb
            matrix[(i, j)] = math.exp(log_p)
    return normalize_matrix(matrix)


def apply_dixon_coles_adjustment(
    score_probs: ScoreMatrix,
    lambda_a: float,
    lambda_b: float,
    rho: float = DEFAULT_DC_RHO,
) -> ScoreMatrix:
    """Low-score tactical correction, layered on top of the bivariate NB grid."""
    adjusted = dict(score_probs)
    for (i, j), p in score_probs.items():
        if (i, j) == (0, 0):
            tau = 1.0 - lambda_a * lambda_b * rho
        elif (i, j) == (0, 1):
            tau = 1.0 + lambda_a * rho
        elif (i, j) == (1, 0):
            tau = 1.0 + lambda_b * rho
        elif (i, j) == (1, 1):
            tau = 1.0 - rho
        else:
            continue
        adjusted[(i, j)] = p * max(tau, 1e-6)
    return normalize_matrix(adjusted)


def build_score_matrix(
    lambda_a: float,
    lambda_b: float,
    *,
    r: float = DEFAULT_R,
    dc_rho: float | None = DEFAULT_DC_RHO,
    max_goals: int = 10,
) -> ScoreMatrix:
    matrix = bivariate_negbin_matrix(lambda_a, lambda_b, r=r, max_goals=max_goals)
    if dc_rho is not None:
        matrix = apply_dixon_coles_adjustment(matrix, lambda_a, lambda_b, rho=dc_rho)
    return matrix


def score_matrix_from_prediction(
    prediction: dict[str, Any],
    *,
    r: float = DEFAULT_R,
    dc_rho: float | None = None,
    max_goals: int = 10,
) -> ScoreMatrix:
    """Convenience entry point matching the `score_matrix_from_prediction` shape
    used by v11/v39/v42/v43, so this module is a drop-in replacement if wired in
    later -- reads `lambda_a`/`lambda_b`/`calibration_notes.dixon_coles_rho` off
    an existing prediction dict without needing to import those modules.
    """
    lambda_a = float(prediction["lambda_a"])
    lambda_b = float(prediction["lambda_b"])
    rho = (
        float(dc_rho)
        if dc_rho is not None
        else float(prediction.get("calibration_notes", {}).get("dixon_coles_rho", DEFAULT_DC_RHO))
    )
    return build_score_matrix(lambda_a, lambda_b, r=r, dc_rho=rho, max_goals=max_goals)


def result_probs(score_probs: ScoreMatrix) -> dict[str, float]:
    a = sum(p for (i, j), p in score_probs.items() if i > j)
    d = sum(p for (i, j), p in score_probs.items() if i == j)
    b = sum(p for (i, j), p in score_probs.items() if i < j)
    return {"team_a_win": a, "draw": d, "team_b_win": b}


def matrix_moments(score_probs: ScoreMatrix) -> dict[str, float]:
    mean_a = sum(i * p for (i, j), p in score_probs.items())
    mean_b = sum(j * p for (i, j), p in score_probs.items())
    var_a = sum(((i - mean_a) ** 2) * p for (i, j), p in score_probs.items())
    var_b = sum(((j - mean_b) ** 2) * p for (i, j), p in score_probs.items())
    cov_ab = sum((i - mean_a) * (j - mean_b) * p for (i, j), p in score_probs.items())
    corr_ab = cov_ab / math.sqrt(var_a * var_b) if var_a > 0 and var_b > 0 else 0.0
    return {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "var_a": var_a,
        "var_b": var_b,
        "cov_ab": cov_ab,
        "corr_ab": corr_ab,
    }


def implied_r_from_moments(mean: float, variance: float) -> float:
    """Method-of-moments estimate of r from an observed (mean, variance) pair,
    for calibrating against real scoring data: Var = mean + mean^2 / r."""
    excess = variance - mean
    if excess <= 0:
        return float("inf")
    return (mean * mean) / excess


def score_matrix_to_rows(matrix: ScoreMatrix) -> list[dict[str, Any]]:
    return [
        {"team_a_goals": i, "team_b_goals": j, "probability": p}
        for (i, j), p in sorted(matrix.items(), key=lambda kv: -kv[1])
    ]


def compare_to_independent_poisson(
    lambda_a: float, lambda_b: float, *, r: float, max_goals: int = 10
) -> list[dict[str, Any]]:
    """Diagnostic: which scorelines gain/lose the most probability mass versus
    plain independent Poisson at the same lambdas."""

    def poisson_pmf(k: int, lam: float) -> float:
        return math.exp(-lam) * lam**k / math.factorial(k)

    poisson_matrix = normalize_matrix(
        {
            (i, j): poisson_pmf(i, lambda_a) * poisson_pmf(j, lambda_b)
            for i in range(max_goals + 1)
            for j in range(max_goals + 1)
        }
    )
    nb_matrix = bivariate_negbin_matrix(lambda_a, lambda_b, r=r, max_goals=max_goals)
    rows = []
    for key in poisson_matrix:
        p_poisson = poisson_matrix[key]
        p_nb = nb_matrix[key]
        rows.append(
            {
                "scoreline": f"{key[0]}-{key[1]}",
                "poisson_probability": p_poisson,
                "bivariate_negbin_probability": p_nb,
                "absolute_lift": p_nb - p_poisson,
                "relative_lift": (p_nb / p_poisson - 1.0) if p_poisson > 0 else float("inf"),
            }
        )
    rows.sort(key=lambda row: -row["absolute_lift"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "V49: standalone true bivariate Negative Binomial scoreline model "
            "(shared-gamma-frailty overdispersion + correlation) for comparing "
            "against the independent-Poisson+Dixon-Coles pipeline used elsewhere."
        )
    )
    parser.add_argument("--lambda-a", type=float, required=True, help="Team A expected goals")
    parser.add_argument("--lambda-b", type=float, required=True, help="Team B expected goals")
    parser.add_argument(
        "--r",
        type=float,
        default=DEFAULT_R,
        help="Shared dispersion shape: lower = fatter, more correlated tails (default 25.0, calibrated -- see calibrate_v49_dispersion.py)",
    )
    parser.add_argument(
        "--dc-rho",
        type=float,
        default=DEFAULT_DC_RHO,
        help="Dixon-Coles low-score rho applied on top; pass 0 to disable",
    )
    parser.add_argument("--max-goals", type=int, default=10)
    parser.add_argument("--top", type=int, default=15, help="How many scorelines to print in the comparison table")
    parser.add_argument("--outdir", default=None, help="Optional directory to write score_matrix.json + comparison CSV")
    args = parser.parse_args()

    matrix = build_score_matrix(
        args.lambda_a, args.lambda_b, r=args.r, dc_rho=args.dc_rho, max_goals=args.max_goals
    )
    moments = matrix_moments(matrix)
    results = result_probs(matrix)
    comparison = compare_to_independent_poisson(args.lambda_a, args.lambda_b, r=args.r, max_goals=args.max_goals)

    print(f"lambda_a={args.lambda_a}, lambda_b={args.lambda_b}, r={args.r}, dc_rho={args.dc_rho}")
    print(f"result_probabilities: {results}")
    print(f"moments: {moments}")
    print(f"\nTop {args.top} scorelines gaining the most probability vs independent Poisson:")
    for row in comparison[: args.top]:
        print(
            f"  {row['scoreline']:>5}  poisson={row['poisson_probability']:.4f}  "
            f"bivar_negbin={row['bivariate_negbin_probability']:.4f}  "
            f"lift={row['absolute_lift']:+.4f} ({row['relative_lift']:+.1%})"
        )

    if args.outdir:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "score_matrix.json").write_text(json.dumps(score_matrix_to_rows(matrix), indent=2))
        with (outdir / "poisson_vs_bivariate_negbin_comparison.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(comparison[0].keys()))
            writer.writeheader()
            writer.writerows(comparison)
        print(f"\nWrote {outdir / 'score_matrix.json'} and comparison CSV to {outdir}.")


if __name__ == "__main__":
    main()
'''
v49_bivariate_negbin_model = _load_submodule("v49_bivariate_negbin_model", _V49_BIVARIATE_NEGBIN_MODEL_SOURCE, "core_engine.py:v49_bivariate_negbin_model")

# ======================================================================
# v13_live_signal_model.py  (bundled as an isolated sub-module)
# ======================================================================
_V13_LIVE_SIGNAL_MODEL_SOURCE = r'''
#!/usr/bin/env python3
"""V13 W/D/L decisions with V11 Poisson/Dixon-Coles exact scores."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

import v11_wcq_results_model as v11


canon_team = v11.canon_team


class V13SklearnWorldCupModel(v11.StrongWorldCupModel):
    """Preserve V13's original ensemble when optional boosters are installed."""

    def _named_regressors(self):
        return [
            item
            for item in super()._named_regressors()
            if item[0] in {"rf", "hgb", "poisson"}
        ]

    def _named_diff_regressors(self):
        return [
            item
            for item in super()._named_diff_regressors()
            if item[0] in {"ridge", "rf", "hgb"}
        ]

    def _named_classifiers(self):
        return [
            item
            for item in super()._named_classifiers()
            if item[0] in {"rf", "hgb", "logistic"}
        ]


@dataclass(frozen=True)
class V13Config:
    draw_decision_threshold: float = 0.2147
    close_elo_gap: float = 100.0
    close_match_draw_target: float = 0.218045
    # Retained for callers using the earlier V13 config. Score widening is
    # disabled while the hybrid model uses V11 exact-score probabilities.
    large_elo_gap: float = 200.0
    large_mismatch_goal_std_scale: float = 1.10
    live_elo_k: float = 24.0


def _redistribute_draw_probability(
    result_probabilities: Dict[str, float],
    target_draw_probability: float,
) -> Dict[str, float]:
    current_draw = float(result_probabilities["draw"])
    target_draw = float(np.clip(target_draw_probability, current_draw, 0.55))
    non_draw_total = max(
        result_probabilities["team_a_win"]
        + result_probabilities["team_b_win"],
        1e-12,
    )
    return {
        "team_a_win": (1.0 - target_draw)
        * result_probabilities["team_a_win"]
        / non_draw_total,
        "draw": target_draw,
        "team_b_win": (1.0 - target_draw)
        * result_probabilities["team_b_win"]
        / non_draw_total,
    }


class V13LiveSignalModel:
    def __init__(
        self,
        base_model: v11.StrongWorldCupModel,
        config: V13Config | None = None,
    ):
        self.base_model = base_model
        self.config = config or V13Config()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_model, name)

    def predict(
        self,
        team_a: str,
        team_b: str,
        host_a: bool = False,
        host_b: bool = False,
        knockout: bool = False,
        max_goals: int = 10,
    ) -> Dict[str, Any]:
        features = self.base_model.make_features(
            team_a,
            team_b,
            host_a,
            host_b,
            knockout,
        )
        elo_gap = abs(float(features.iloc[0]["elo_diff"]))
        prediction = self.base_model.predict(
            team_a,
            team_b,
            host_a,
            host_b,
            knockout,
            max_goals,
        )

        adjusted_results = dict(prediction["result_probabilities"])
        draw_boost_applied = elo_gap < self.config.close_elo_gap
        if draw_boost_applied:
            adjusted_results = _redistribute_draw_probability(
                adjusted_results,
                self.config.close_match_draw_target,
            )

        # Keep V11's full score policy because it produced better observed
        # top-two coverage than the experimental unreweighted score matrix.
        prediction["result_probabilities"] = adjusted_results

        draw_signal = float(prediction["draw_model_probability"])
        if draw_signal >= self.config.draw_decision_threshold:
            predicted_result = "draw"
        elif (
            prediction["result_probabilities"]["team_a_win"]
            >= prediction["result_probabilities"]["team_b_win"]
        ):
            predicted_result = "team_a_win"
        else:
            predicted_result = "team_b_win"

        prediction["predicted_result"] = predicted_result
        prediction["v13_adjustments"] = {
            "pre_match_elo_gap": elo_gap,
            "draw_signal": draw_signal,
            "draw_decision_threshold": self.config.draw_decision_threshold,
            "draw_boost_applied": draw_boost_applied,
            "close_match_draw_target": self.config.close_match_draw_target,
            "variance_widened": False,
            "goal_std_scale": 1.0,
            "live_elo_k": self.config.live_elo_k,
            "wdl_model": "v13",
            "exact_score_model": "v11_poisson_dixon_coles_reweighted",
            "exact_score_dixon_coles_rho": prediction.get(
                "calibration_notes",
                {},
            ).get("dixon_coles_rho"),
            "exact_score_result_reweighting": True,
        }
        prediction["calibration_notes"] = {
            **prediction.get("calibration_notes", {}),
            "v13": prediction["v13_adjustments"],
            "hybrid_model_policy": (
                "V13 supplies W/D/L probabilities and the result decision; "
                "V11 supplies its calibrated Poisson/Dixon-Coles exact-score "
                "distribution and all score-derived markets."
            ),
        }
        return prediction

    def update_after_match(
        self,
        team_a: str,
        team_b: str,
        goals_a: int,
        goals_b: int,
    ) -> Dict[str, float]:
        """Update live Elo state after an observed match."""
        name_a = canon_team(team_a)
        name_b = canon_team(team_b)
        elo_a = float(self.base_model.latest_elo.get(name_a, 1500.0))
        elo_b = float(self.base_model.latest_elo.get(name_b, 1500.0))
        expected_a = v11.elo_expected(elo_a, elo_b)
        score_a = 1.0 if goals_a > goals_b else 0.5 if goals_a == goals_b else 0.0
        delta = self.config.live_elo_k * (score_a - expected_a)
        self.base_model.latest_elo[name_a] = elo_a + delta
        self.base_model.latest_elo[name_b] = elo_b - delta
        return {
            "team_a_elo_before": elo_a,
            "team_b_elo_before": elo_b,
            "team_a_elo_after": elo_a + delta,
            "team_b_elo_after": elo_b - delta,
            "elo_delta_a": delta,
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
    loader = v11.WorldCupSAILoader(
        zip_path,
        Path(str(zip_path) + "_extracted"),
    )
    matches = loader.load_matches()
    current = v11.load_current_team_features(train_csv, test_csv)
    box = v11.load_kaggle_box_data(box_csv)
    qualification_results = v11.load_world_cup_qualification_results(
        results_csv,
        former_names_csv,
    )
    qualifier_source = (
        qualification_results if not qualification_results.empty else box
    )
    historical_current = pd.DataFrame(columns=["team"])
    frame, features, events = v11.build_rolling_features(
        matches,
        historical_current,
        qualifier_box=qualifier_source,
        qualifier_fallback_box=box,
        qualifier_blend_start_year=qualifier_blend_start_year,
        qualifier_full_weight_year=qualifier_full_weight_year,
        qualifier_minimum_influence=qualifier_minimum_influence,
    )
    model_class = (
        V13SklearnWorldCupModel
        if model_type == "ensemble"
        else v11.StrongWorldCupModel
    )
    base_model = (
        model_class(
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
    data = v11.DataBundle(
        matches=matches,
        team_current=current,
        training_frame=frame,
        event_columns=events,
        box_frame=box,
    )
    return V13LiveSignalModel(base_model), data


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    parser = argparse.ArgumentParser(
        description="Run a V13 W/D/L and V11 exact-score match prediction."
    )
    parser.add_argument("--team-a", required=True)
    parser.add_argument("--team-b", required=True)
    parser.add_argument("--host-a", action="store_true")
    parser.add_argument("--host-b", action="store_true")
    parser.add_argument("--knockout", action="store_true")
    parser.add_argument(
        "--model",
        default="ensemble",
        choices=[
            "ensemble",
            "hgb",
            "rf",
            "poisson",
            "lightgbm",
            "xgboost",
            "catboost",
        ],
    )
    parser.add_argument("--outdir", default="outputs/outputs_v13_prediction")
    parser.add_argument(
        "--worldcupsai-zip",
        default=str(data_dir / "worldcupsai.zip"),
    )
    parser.add_argument(
        "--team-train",
        default=str(data_dir / "current_team_features_2026.csv"),
    )
    parser.add_argument("--team-test")
    parser.add_argument(
        "--box-data",
        default=str(data_dir / "FIFAallMatchBoxData.csv"),
    )
    parser.add_argument(
        "--results-data",
        default=str(data_dir / "results.csv"),
    )
    parser.add_argument(
        "--former-names",
        default=str(data_dir / "former_names.csv"),
    )
    parser.add_argument("--prediction-year", type=int, default=2026)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    output_dir = v11.unique_output_dir(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, _ = build_from_zip(
        args.worldcupsai_zip,
        train_csv=args.team_train,
        test_csv=args.team_test,
        model_type=args.model,
        box_csv=args.box_data,
        results_csv=args.results_data,
        former_names_csv=args.former_names,
        prediction_year=args.prediction_year,
    )
    prediction = model.predict(
        args.team_a,
        args.team_b,
        host_a=args.host_a,
        host_b=args.host_b,
        knockout=args.knockout,
    )

    (output_dir / "single_match_prediction.json").write_text(
        json.dumps(prediction, indent=2)
    )
    pd.DataFrame(prediction["top_scorelines"]).to_csv(
        output_dir / "scoreline_probabilities_top.csv",
        index=False,
    )
    pd.DataFrame(prediction["scoreline_probabilities"]).to_csv(
        output_dir / "scoreline_probabilities.csv",
        index=False,
    )
    (output_dir / "model_summary.json").write_text(
        json.dumps(
            {
                "wdl_model": "v13",
                "exact_score_model": (
                    "v11_poisson_dixon_coles_reweighted"
                ),
                "model_type": args.model,
                "team_a": prediction["team_a"],
                "team_b": prediction["team_b"],
            },
            indent=2,
        )
    )
    if not args.no_plots:
        v11.plot_prediction_outputs(prediction, output_dir)

    print(
        json.dumps(
            {
                "result_probabilities": prediction["result_probabilities"],
                "predicted_result": prediction["predicted_result"],
                "lambda_a": prediction["lambda_a"],
                "lambda_b": prediction["lambda_b"],
                "top_scorelines": prediction["top_scorelines"][:5],
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
'''
v13_live_signal_model = _load_submodule("v13_live_signal_model", _V13_LIVE_SIGNAL_MODEL_SOURCE, "core_engine.py:v13_live_signal_model")

# ======================================================================
# v15_catboost_model.py  (bundled as an isolated sub-module)
# ======================================================================
_V15_CATBOOST_MODEL_SOURCE = r'''
#!/usr/bin/env python3
"""V15: CatBoost-enhanced ensemble with V13 live decision logic."""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

import v11_wcq_results_model as v11
from v13_live_signal_model import V13LiveSignalModel

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except Exception:
    CatBoostClassifier = None
    CatBoostRegressor = None


canon_team = v11.canon_team

PLAYER_PROFILE_METRICS = (
    "overall_mean",
    "top11_overall_mean",
    "top5_overall_mean",
    "gk_overall",
    "defense_overall",
    "midfield_overall",
    "attack_overall",
    "potential_mean",
    "value_log_sum",
    "age_mean",
    "rating_count",
)

PLAYER_TEAM_ALIASES = {
    "bosnia and herzegovina": "Bosnia and Herzegovina",
    "congo dr": "Congo DR",
    "cote d ivoire": "Côte d'Ivoire",
    "ivory coast": "Côte d'Ivoire",
    "korea republic": "Korea Republic",
    "south korea": "Korea Republic",
    "turkiye": "Turkey",
    "usa": "United States",
}

CONTINENTAL_TOURNAMENTS = {
    "UEFA Euro": ("EURO", 0.75, 40.0),
    "Copa América": ("COPA", 0.75, 40.0),
    "African Cup of Nations": ("AFCON", 0.55, 30.0),
    "AFC Asian Cup": ("AFCCUP", 0.55, 30.0),
    "Gold Cup": ("GOLD", 0.35, 22.0),
    "CONCACAF Championship": ("GOLD", 0.35, 22.0),
    "Oceania Nations Cup": ("OTHER_CONTINENTAL", 0.35, 22.0),
}

TOURNAMENT_FEATURE_TYPES = (
    "WC",
    "EURO",
    "COPA",
    "AFCON",
    "AFCCUP",
    "GOLD",
    "OTHER_CONTINENTAL",
)

DEFAULT_RESULTS_AS_OF = "latest"
CURRENT_WORLD_CUP_YEAR = 2026

PAIR_DIFFERENCE_FEATURES = (
    "elo_diff",
    "gf_diff",
    "ga_diff",
    "gd_diff",
    "win_rate_diff",
    "continental_gf_diff",
    "continental_ga_diff",
    "continental_gd_diff",
    "continental_win_rate_diff",
)


def tournament_metadata(name: Any) -> Dict[str, Any]:
    tournament = str(name).strip()
    if tournament in CONTINENTAL_TOURNAMENTS:
        tournament_type, prestige_weight, k_factor = (
            CONTINENTAL_TOURNAMENTS[tournament]
        )
        return {
            "tournament_type": tournament_type,
            "prestige_weight": prestige_weight,
            "k_factor": k_factor,
            "is_continental_final": True,
        }
    lower = tournament.lower()
    if tournament == "FIFA World Cup":
        return {
            "tournament_type": "WC",
            "prestige_weight": 1.0,
            "k_factor": 50.0,
            "is_continental_final": False,
        }
    if "qualification" in lower or "qualifier" in lower:
        return {
            "tournament_type": "QUALIFIER",
            "prestige_weight": 0.35,
            "k_factor": 25.0,
            "is_continental_final": False,
        }
    if tournament == "Friendly":
        return {
            "tournament_type": "FRIENDLY",
            "prestige_weight": 0.15,
            "k_factor": 10.0,
            "is_continental_final": False,
        }
    return {
        "tournament_type": "OTHER",
        "prestige_weight": 0.25,
        "k_factor": 15.0,
        "is_continental_final": False,
    }


def load_international_results(
    path: str | Path | None,
    former_names_csv: str | Path | None = None,
    as_of: str | pd.Timestamp = DEFAULT_RESULTS_AS_OF,
) -> pd.DataFrame:
    """Load completed internationals without leaking current World Cup finals."""
    if not path or not Path(path).exists():
        return pd.DataFrame()
    raw = pd.read_csv(path)
    required = {
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "neutral",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"results.csv is missing columns: {missing}")

    former_names: Dict[str, str] = {}
    if former_names_csv and Path(former_names_csv).exists():
        former = pd.read_csv(former_names_csv)
        if {"current", "former"}.issubset(former.columns):
            former_names = {
                canon_team(row["former"]): canon_team(row["current"])
                for _, row in former.iterrows()
            }

    def team_name(value: Any) -> str:
        name = canon_team(value)
        return former_names.get(name, name)

    results = raw.copy()
    results["date"] = pd.to_datetime(results["date"], errors="coerce")
    results["goals_a"] = pd.to_numeric(
        results["home_score"],
        errors="coerce",
    )
    results["goals_b"] = pd.to_numeric(
        results["away_score"],
        errors="coerce",
    )
    results["team_a"] = results["home_team"].map(team_name)
    results["team_b"] = results["away_team"].map(team_name)
    results["neutral"] = (
        results["neutral"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "yes"})
    )
    metadata = results["tournament"].map(tournament_metadata)
    for column in (
        "tournament_type",
        "prestige_weight",
        "k_factor",
        "is_continental_final",
    ):
        results[column] = metadata.map(lambda value: value[column])

    completed = results.dropna(
        subset=["date", "team_a", "team_b", "goals_a", "goals_b"]
    ).copy()
    current_world_cup = (
        completed["tournament"].astype(str).eq("FIFA World Cup")
        & completed["date"].dt.year.eq(CURRENT_WORLD_CUP_YEAR)
    )
    non_current_world_cup = completed.loc[~current_world_cup].copy()
    if str(as_of).strip().lower() in {"latest", "max", "latest_non_world_cup"}:
        if non_current_world_cup.empty:
            cutoff = completed["date"].max().normalize()
        else:
            cutoff = non_current_world_cup["date"].max().normalize()
    else:
        cutoff = pd.Timestamp(as_of).normalize()
    output = (
        non_current_world_cup.loc[lambda frame: frame["date"] <= cutoff]
        .sort_values(["date"], kind="stable")
        .reset_index(drop=True)
    )
    output.attrs["resolved_as_of"] = str(cutoff.date())
    output.attrs["excluded_current_world_cup_matches"] = int(current_world_cup.sum())
    return output


def _rolling_team_stats(
    history: list[tuple[float, float]],
    limit: int = 12,
) -> Dict[str, float]:
    recent = history[-limit:]
    if not recent:
        return {
            "gf_avg": 1.25,
            "ga_avg": 1.25,
            "gd_avg": 0.0,
            "win_rate": 0.33,
            "draw_rate": 0.25,
            "matches_seen": 0.0,
        }
    gf = np.asarray([value[0] for value in recent], dtype=float)
    ga = np.asarray([value[1] for value in recent], dtype=float)
    return {
        "gf_avg": float(gf.mean()),
        "ga_avg": float(ga.mean()),
        "gd_avg": float((gf - ga).mean()),
        "win_rate": float((gf > ga).mean()),
        "draw_rate": float((gf == ga).mean()),
        "matches_seen": float(len(history)),
    }


def international_pair_features(
    state: Dict[str, Any],
    team_a: str,
    team_b: str,
    match_date: pd.Timestamp,
    tournament_type: str = "WC",
) -> Dict[str, float]:
    a = canon_team(team_a)
    b = canon_team(team_b)
    elo_a = float(state["elo"].get(a, 1500.0))
    elo_b = float(state["elo"].get(b, 1500.0))
    sa = _rolling_team_stats(state["team_history"].get(a, []))
    sb = _rolling_team_stats(state["team_history"].get(b, []))
    ca = _rolling_team_stats(state["continental_history"].get(a, []))
    cb = _rolling_team_stats(state["continental_history"].get(b, []))

    def months_since(team: str) -> float:
        last_date = state["last_continental"].get(team)
        if last_date is None:
            return 120.0
        return max(
            float((pd.Timestamp(match_date) - last_date).days) / 30.4375,
            0.0,
        )

    features = {
        "elo_a": elo_a,
        "elo_b": elo_b,
        "elo_diff": elo_a - elo_b,
        "elo_prob_a": v11.elo_expected(elo_a, elo_b),
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
        "abs_draw_rate_diff": abs(
            sa["draw_rate"] - sb["draw_rate"]
        ),
        "form_expected_total": (
            sa["gf_avg"] + sa["ga_avg"] + sb["gf_avg"] + sb["ga_avg"]
        )
        / 2.0,
        "continental_a_gf_avg": ca["gf_avg"],
        "continental_a_ga_avg": ca["ga_avg"],
        "continental_a_gd_avg": ca["gd_avg"],
        "continental_a_win_rate": ca["win_rate"],
        "continental_a_draw_rate": ca["draw_rate"],
        "continental_a_matches_seen": ca["matches_seen"],
        "continental_a_months_since": months_since(a),
        "continental_b_gf_avg": cb["gf_avg"],
        "continental_b_ga_avg": cb["ga_avg"],
        "continental_b_gd_avg": cb["gd_avg"],
        "continental_b_win_rate": cb["win_rate"],
        "continental_b_draw_rate": cb["draw_rate"],
        "continental_b_matches_seen": cb["matches_seen"],
        "continental_b_months_since": months_since(b),
        "continental_gf_diff": ca["gf_avg"] - cb["gf_avg"],
        "continental_ga_diff": ca["ga_avg"] - cb["ga_avg"],
        "continental_gd_diff": ca["gd_avg"] - cb["gd_avg"],
        "continental_win_rate_diff": (
            ca["win_rate"] - cb["win_rate"]
        ),
        "continental_abs_gd_diff": abs(
            ca["gd_avg"] - cb["gd_avg"]
        ),
        "continental_mean_draw_rate": (
            ca["draw_rate"] + cb["draw_rate"]
        )
        / 2.0,
        "continental_abs_draw_rate_diff": abs(
            ca["draw_rate"] - cb["draw_rate"]
        ),
        "continental_expected_total": (
            ca["gf_avg"] + ca["ga_avg"] + cb["gf_avg"] + cb["ga_avg"]
        )
        / 2.0,
        "continental_min_matches_seen": min(
            ca["matches_seen"],
            cb["matches_seen"],
        ),
    }
    for name in TOURNAMENT_FEATURE_TYPES:
        features[f"tournament_is_{name.lower()}"] = float(
            tournament_type == name
        )
    return features


def build_international_timeline(
    results: pd.DataFrame,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    state: Dict[str, Any] = {
        "elo": {},
        "team_history": {},
        "continental_history": {},
        "last_continental": {},
    }
    rows = []
    for source_index, row in results.iterrows():
        features = international_pair_features(
            state,
            row["team_a"],
            row["team_b"],
            row["date"],
            row["tournament_type"],
        )
        rows.append(
            {
                "source_index": source_index,
                "date": row["date"],
                "team_a": row["team_a"],
                "team_b": row["team_b"],
                "goals_a": float(row["goals_a"]),
                "goals_b": float(row["goals_b"]),
                "tournament": row["tournament"],
                "tournament_type": row["tournament_type"],
                "prestige_weight": float(row["prestige_weight"]),
                "is_continental_final": bool(
                    row["is_continental_final"]
                ),
                "neutral": bool(row["neutral"]),
                "country": row.get("country", ""),
                **features,
            }
        )

        a = row["team_a"]
        b = row["team_b"]
        goals_a = float(row["goals_a"])
        goals_b = float(row["goals_b"])
        expected_a = features["elo_prob_a"]
        actual_a = (
            1.0 if goals_a > goals_b else 0.5 if goals_a == goals_b else 0.0
        )
        k_factor = float(row["k_factor"]) * v11.elo_margin_multiplier(
            goals_a - goals_b
        )
        delta = k_factor * (actual_a - expected_a)
        state["elo"][a] = features["elo_a"] + delta
        state["elo"][b] = features["elo_b"] - delta
        state["team_history"].setdefault(a, []).append((goals_a, goals_b))
        state["team_history"].setdefault(b, []).append((goals_b, goals_a))
        if row["is_continental_final"]:
            state["continental_history"].setdefault(a, []).append(
                (goals_a, goals_b)
            )
            state["continental_history"].setdefault(b, []).append(
                (goals_b, goals_a)
            )
            state["last_continental"][a] = pd.Timestamp(row["date"])
            state["last_continental"][b] = pd.Timestamp(row["date"])
    return pd.DataFrame(rows), state


def _reverse_pair_features(features: Dict[str, Any]) -> Dict[str, Any]:
    reversed_features = dict(features)
    swap_pairs = [
        ("elo_a", "elo_b"),
        *[
            (f"a_{metric}", f"b_{metric}")
            for metric in (
                "gf_avg",
                "ga_avg",
                "gd_avg",
                "win_rate",
                "draw_rate",
                "matches_seen",
            )
        ],
        *[
            (f"continental_a_{metric}", f"continental_b_{metric}")
            for metric in (
                "gf_avg",
                "ga_avg",
                "gd_avg",
                "win_rate",
                "draw_rate",
                "matches_seen",
                "months_since",
            )
        ],
    ]
    for left, right in swap_pairs:
        reversed_features[left], reversed_features[right] = (
            features[right],
            features[left],
        )
    reversed_features["elo_prob_a"] = 1.0 - float(
        features["elo_prob_a"]
    )
    for column in PAIR_DIFFERENCE_FEATURES:
        reversed_features[column] = -float(features[column])
    return reversed_features


def build_expanded_training_frame(
    world_cup_frame: pd.DataFrame,
    timeline: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], Dict[str, int]]:
    """Add continental targets and all-match pre-game features."""
    wc = world_cup_frame.copy()
    wc["prestige_weight"] = 1.0
    wc["tournament_type"] = "WC"
    wc["training_source"] = "world_cup"

    feature_columns = [
        column
        for column in timeline.columns
        if column
        not in {
            "source_index",
            "date",
            "team_a",
            "team_b",
            "goals_a",
            "goals_b",
            "tournament",
            "tournament_type",
            "prestige_weight",
            "is_continental_final",
            "neutral",
            "country",
        }
    ]
    direct_lookup = {}
    for _, row in timeline.iterrows():
        key = (
            pd.Timestamp(row["date"]).normalize(),
            row["team_a"],
            row["team_b"],
        )
        direct_lookup[key] = {column: row[column] for column in feature_columns}

    matched = 0
    for index, row in wc.iterrows():
        date = pd.Timestamp(row["date"]).normalize()
        key = (date, row["team_a"], row["team_b"])
        reverse_key = (date, row["team_b"], row["team_a"])
        values = direct_lookup.get(key)
        if values is None and reverse_key in direct_lookup:
            values = _reverse_pair_features(direct_lookup[reverse_key])
        if values is None:
            continue
        matched += 1
        for column, value in values.items():
            wc.at[index, column] = value

    continental = timeline[timeline["is_continental_final"]].copy()
    regional = pd.DataFrame(
        {
            "match_id": continental["source_index"].map(
                lambda value: f"continental_{value}"
            ),
            "date": continental["date"],
            "team_a": continental["team_a"],
            "team_b": continental["team_b"],
            "goals_a": continental["goals_a"],
            "goals_b": continental["goals_b"],
            "goal_diff": (
                continental["goals_a"] - continental["goals_b"]
            ),
            "is_group_stage": 0,
            "is_knockout": 0,
            "host_a": 0,
            "host_b": 0,
            "host_diff": 0,
            "abs_host_diff": 0,
            "same_confed": 1,
            "prestige_weight": continental["prestige_weight"],
            "tournament_type": continental["tournament_type"],
            "training_source": "continental",
        }
    )
    for column in feature_columns:
        regional[column] = continental[column].to_numpy()

    combined = pd.concat([wc, regional], ignore_index=True, sort=False)
    qualifier_columns = [
        column for column in combined if column.startswith("qual_")
    ]
    combined = combined.drop(columns=qualifier_columns)
    combined = combined.sort_values("date", kind="stable").reset_index(
        drop=True
    )

    excluded = {
        "match_id",
        "date",
        "team_a",
        "team_b",
        "goals_a",
        "goals_b",
        "goal_diff",
        "prestige_weight",
    }
    event_targets = {
        f"{event}_{side}"
        for event in (
            "yellow_cards",
            "red_cards",
            "second_yellow_cards",
            "sending_offs",
            "penalty_goals",
            "penalty_kicks",
            "penalty_kicks_converted",
            "own_goals",
            "substitutions",
        )
        for side in ("a", "b")
    }
    features = [
        column
        for column in combined.columns
        if column not in excluded
        and column not in event_targets
        and pd.api.types.is_numeric_dtype(combined[column])
        and combined[column].notna().mean() > 0.20
    ]
    summary = {
        "world_cup_rows": int(len(wc)),
        "continental_rows": int(len(regional)),
        "world_cup_external_matches": int(matched),
        "international_timeline_rows": int(len(timeline)),
    }
    return combined, features, summary


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def normalize_player_team(value: Any) -> str:
    normalized = normalize_text(value)
    return canon_team(PLAYER_TEAM_ALIASES.get(normalized, str(value).strip()))


def player_role(value: Any) -> str:
    position = str(value).upper().split(",")[0].strip()
    if position in {"GK"}:
        return "gk"
    if position in {"CB", "LB", "RB", "LWB", "RWB", "DF"}:
        return "defense"
    if position in {"CM", "CDM", "CAM", "LM", "RM", "MF"}:
        return "midfield"
    return "attack"


def load_player_ratings(path: str | Path | None) -> pd.DataFrame:
    if not path or not Path(path).exists():
        return pd.DataFrame()
    ratings = pd.read_csv(path)
    required = {
        "fifa_update_date",
        "nationality_name",
        "short_name",
        "long_name",
        "player_positions",
        "overall",
        "potential",
        "value_eur",
        "age",
    }
    missing = sorted(required - set(ratings.columns))
    if missing:
        raise ValueError(f"Player ratings are missing columns: {missing}")
    ratings = ratings.copy()
    ratings["rating_date"] = pd.to_datetime(
        ratings["fifa_update_date"],
        errors="coerce",
    )
    ratings["team"] = ratings["nationality_name"].map(normalize_player_team)
    ratings["role"] = ratings["player_positions"].map(player_role)
    ratings["long_name_key"] = ratings["long_name"].map(normalize_text)
    ratings["short_name_key"] = ratings["short_name"].map(normalize_text)
    for column in ("overall", "potential", "value_eur", "age"):
        ratings[column] = pd.to_numeric(ratings[column], errors="coerce")
    return ratings.dropna(
        subset=["rating_date", "team", "overall"]
    ).reset_index(drop=True)


def load_declared_squads(path: str | Path | None) -> pd.DataFrame:
    if not path or not Path(path).exists():
        return pd.DataFrame()
    raw = Path(path).read_bytes()
    text = raw.decode("utf-8-sig", errors="replace").replace("\x00", "")
    squads = pd.read_csv(io.StringIO(text))
    required = {
        "team",
        "position",
        "player_name",
        "first_names",
        "last_names",
        "date_of_birth",
        "caps",
        "goals",
    }
    missing = sorted(required - set(squads.columns))
    if missing:
        raise ValueError(f"Declared squads are missing columns: {missing}")
    squads = squads.copy()
    squads["team"] = squads["team"].map(normalize_player_team)
    squads["role"] = squads["position"].map(player_role)
    squads["date_of_birth"] = pd.to_datetime(
        squads["date_of_birth"],
        format="%d/%m/%Y",
        errors="coerce",
    )
    squads["caps"] = pd.to_numeric(squads["caps"], errors="coerce")
    squads["goals"] = pd.to_numeric(squads["goals"], errors="coerce")
    return squads


def aggregate_player_profile(players: pd.DataFrame) -> Dict[str, float]:
    if players.empty:
        return {metric: float("nan") for metric in PLAYER_PROFILE_METRICS}
    ordered = players.sort_values("overall", ascending=False)

    def role_mean(role: str, count: int) -> float:
        values = ordered.loc[ordered["role"] == role, "overall"].head(count)
        return float(values.mean()) if not values.empty else float("nan")

    values = np.clip(
        pd.to_numeric(ordered["value_eur"], errors="coerce").fillna(0.0),
        0.0,
        None,
    )
    return {
        "overall_mean": float(ordered["overall"].mean()),
        "top11_overall_mean": float(ordered["overall"].head(11).mean()),
        "top5_overall_mean": float(ordered["overall"].head(5).mean()),
        "gk_overall": role_mean("gk", 1),
        "defense_overall": role_mean("defense", 4),
        "midfield_overall": role_mean("midfield", 4),
        "attack_overall": role_mean("attack", 3),
        "potential_mean": float(ordered["potential"].mean()),
        "value_log_sum": float(math.log1p(values.sum())),
        "age_mean": float(ordered["age"].mean()),
        "rating_count": float(len(ordered)),
    }


def build_historical_player_profiles(
    ratings: pd.DataFrame,
) -> Dict[str, list[tuple[pd.Timestamp, Dict[str, float]]]]:
    profiles: Dict[str, list[tuple[pd.Timestamp, Dict[str, float]]]] = {}
    if ratings.empty:
        return profiles
    for (team, rating_date), frame in ratings.groupby(
        ["team", "rating_date"],
        sort=True,
    ):
        profile = aggregate_player_profile(
            frame.sort_values("overall", ascending=False).head(26)
        )
        profiles.setdefault(team, []).append((rating_date, profile))
    return profiles


def latest_historical_profile(
    profiles: Dict[str, list[tuple[pd.Timestamp, Dict[str, float]]]],
    team: str,
    match_date: pd.Timestamp,
) -> Dict[str, float] | None:
    candidates = profiles.get(normalize_player_team(team), [])
    available = [
        profile
        for rating_date, profile in candidates
        if rating_date <= match_date
    ]
    return available[-1] if available else None


def add_historical_player_features(
    frame: pd.DataFrame,
    ratings: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    enriched = frame.copy()
    profiles = build_historical_player_profiles(ratings)
    player_features = []
    for metric in PLAYER_PROFILE_METRICS:
        for prefix in ("player_a", "player_b"):
            column = f"{prefix}_{metric}"
            enriched[column] = np.nan
            player_features.append(column)
        diff_column = f"player_diff_{metric}"
        enriched[diff_column] = np.nan
        player_features.append(diff_column)

    for index, row in enriched.iterrows():
        match_date = pd.Timestamp(row["date"])
        profile_a = latest_historical_profile(
            profiles,
            row["team_a"],
            match_date,
        )
        profile_b = latest_historical_profile(
            profiles,
            row["team_b"],
            match_date,
        )
        for metric in PLAYER_PROFILE_METRICS:
            value_a = profile_a.get(metric, np.nan) if profile_a else np.nan
            value_b = profile_b.get(metric, np.nan) if profile_b else np.nan
            enriched.at[index, f"player_a_{metric}"] = value_a
            enriched.at[index, f"player_b_{metric}"] = value_b
            if pd.notna(value_a) and pd.notna(value_b):
                enriched.at[index, f"player_diff_{metric}"] = value_a - value_b
    return enriched, [*feature_columns, *player_features]


def squad_name_candidates(row: pd.Series) -> set[str]:
    return {
        normalize_text(row.get("player_name", "")),
        normalize_text(
            f"{row.get('first_names', '')} {row.get('last_names', '')}"
        ),
        normalize_text(
            f"{row.get('last_names', '')} {row.get('first_names', '')}"
        ),
        normalize_text(row.get("name_on_shirt", "")),
    } - {""}


def best_rating_match(
    squad_row: pd.Series,
    candidates: pd.DataFrame,
) -> tuple[pd.Series | None, float]:
    names = squad_name_candidates(squad_row)
    exact = candidates[
        candidates["long_name_key"].isin(names)
        | candidates["short_name_key"].isin(names)
    ]
    if not exact.empty:
        return exact.sort_values("overall", ascending=False).iloc[0], 1.0

    best_index = None
    best_score = 0.0
    for index, candidate in candidates.iterrows():
        candidate_names = {
            candidate["long_name_key"],
            candidate["short_name_key"],
        }
        score = max(
            SequenceMatcher(None, left, right).ratio()
            for left in names
            for right in candidate_names
            if left and right
        )
        if score > best_score:
            best_index = index
            best_score = score
    if best_index is not None and best_score >= 0.84:
        return candidates.loc[best_index], best_score
    return None, best_score


def build_current_squad_profiles(
    squads: pd.DataFrame,
    ratings: pd.DataFrame,
) -> Dict[str, Dict[str, Any]]:
    if squads.empty or ratings.empty:
        return {}
    latest_date = ratings["rating_date"].max()
    latest = ratings[ratings["rating_date"] == latest_date].copy()
    global_role_medians = latest.groupby("role")[
        ["overall", "potential", "value_eur", "age"]
    ].median()
    profiles: Dict[str, Dict[str, Any]] = {}
    for team, squad in squads.groupby("team"):
        national = latest[latest["team"] == team]
        national_role_medians = national.groupby("role")[
            ["overall", "potential", "value_eur", "age"]
        ].median()
        player_rows = []
        exact_or_fuzzy = 0
        match_scores = []
        for _, player in squad.iterrows():
            matched, score = best_rating_match(player, national)
            if matched is not None:
                exact_or_fuzzy += 1
                match_scores.append(score)
                player_rows.append(
                    {
                        "overall": matched["overall"],
                        "potential": matched["potential"],
                        "value_eur": matched["value_eur"],
                        "age": matched["age"],
                        "role": player["role"],
                    }
                )
                continue
            role = player["role"]
            fallback = (
                national_role_medians.loc[role]
                if role in national_role_medians.index
                else global_role_medians.loc[role]
            )
            age = (
                (latest_date - player["date_of_birth"]).days / 365.25
                if pd.notna(player["date_of_birth"])
                else fallback["age"]
            )
            player_rows.append(
                {
                    "overall": fallback["overall"],
                    "potential": fallback["potential"],
                    "value_eur": fallback["value_eur"],
                    "age": age,
                    "role": role,
                }
            )
        profile = aggregate_player_profile(pd.DataFrame(player_rows))
        profile.update(
            {
                "team": team,
                "squad_size": int(len(squad)),
                "matched_players": int(exact_or_fuzzy),
                "match_coverage": float(exact_or_fuzzy / max(len(squad), 1)),
                "mean_name_match_score": float(np.mean(match_scores))
                if match_scores
                else 0.0,
                "caps_mean": float(squad["caps"].mean()),
                "caps_sum": float(squad["caps"].sum()),
                "international_goals_sum": float(squad["goals"].sum()),
                "rating_date": str(latest_date.date()),
            }
        )
        profiles[team] = profile
    return profiles


def score_outputs(
    score_probabilities: Dict[Tuple[int, int], float],
    max_goals: int,
) -> Dict[str, Any]:
    top = sorted(
        [
            {
                "team_a_goals": goals_a,
                "team_b_goals": goals_b,
                "probability": probability,
            }
            for (goals_a, goals_b), probability in score_probabilities.items()
        ],
        key=lambda item: item["probability"],
        reverse=True,
    )[:15]
    spreads = {
        str(goal_difference): sum(
            probability
            for (goals_a, goals_b), probability in score_probabilities.items()
            if goals_a - goals_b == goal_difference
        )
        for goal_difference in range(-max_goals, max_goals + 1)
    }
    totals = {
        str(total_goals): sum(
            probability
            for (goals_a, goals_b), probability in score_probabilities.items()
            if goals_a + goals_b == total_goals
        )
        for total_goals in range(2 * max_goals + 1)
    }
    over_under = {}
    for line in (0.5, 1.5, 2.5, 3.5, 4.5):
        under = sum(
            probability
            for (goals_a, goals_b), probability in score_probabilities.items()
            if goals_a + goals_b < line
        )
        over_under[f"over_{line}"] = 1.0 - under
        over_under[f"under_{line}"] = under
    return {
        "top_scorelines": top,
        "scoreline_probabilities": [
            {
                "team_a_goals": goals_a,
                "team_b_goals": goals_b,
                "probability": probability,
            }
            for (goals_a, goals_b), probability in sorted(
                score_probabilities.items()
            )
        ],
        "spread_probabilities": spreads,
        "total_goal_probabilities": totals,
        "over_under_probabilities": over_under,
    }


class FeatureSubsetEstimator:
    """Restrict an estimator to stable non-player result features."""

    def __init__(self, estimator: Any, columns: list[str]):
        self.estimator = estimator
        self.columns = columns

    @property
    def classes_(self):
        return self.estimator.classes_

    def fit(self, X, y, sample_weight=None):
        v11.fit_with_sample_weight(
            self.estimator,
            X[self.columns],
            y,
            sample_weight,
        )
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(X[self.columns])


def require_catboost() -> None:
    if CatBoostClassifier is None or CatBoostRegressor is None:
        raise RuntimeError(
            "V15 requires CatBoost. Install it with "
            "`.venv/bin/pip install catboost`."
        )


class V15CatBoostWorldCupModel(v11.StrongWorldCupModel):
    """Add CatBoost conservatively to the proven V13 ensemble."""

    def __init__(
        self,
        recency_half_life_years: float = 16.0,
        recency_min_weight: float = 0.10,
    ):
        require_catboost()
        super().__init__(
            model_type="ensemble",
            recency_half_life_years=recency_half_life_years,
            recency_min_weight=recency_min_weight,
        )
        self.current_squad_profiles: Dict[str, Dict[str, Any]] = {}
        self.current_international_state: Dict[str, Any] = {}
        self.international_as_of = pd.Timestamp("2026-06-10")

    def set_current_squad_profiles(
        self,
        profiles: Dict[str, Dict[str, Any]],
    ):
        self.current_squad_profiles = profiles
        return self

    def set_current_international_state(
        self,
        state: Dict[str, Any],
        as_of: str | pd.Timestamp,
    ):
        self.current_international_state = state
        self.international_as_of = pd.Timestamp(as_of)
        self.latest_elo = dict(state.get("elo", {}))
        return self

    def make_features(
        self,
        team_a,
        team_b,
        host_a=False,
        host_b=False,
        knockout=False,
    ):
        features = super().make_features(
            team_a,
            team_b,
            host_a,
            host_b,
            knockout,
        )
        if self.current_international_state:
            international = international_pair_features(
                self.current_international_state,
                team_a,
                team_b,
                self.international_as_of + pd.Timedelta(days=1),
                "WC",
            )
            name_a = canon_team(team_a)
            name_b = canon_team(team_b)
            elo_a = float(self.latest_elo.get(name_a, 1500.0))
            elo_b = float(self.latest_elo.get(name_b, 1500.0))
            international.update(
                {
                    "elo_a": elo_a,
                    "elo_b": elo_b,
                    "elo_diff": elo_a - elo_b,
                    "elo_prob_a": v11.elo_expected(elo_a, elo_b),
                    "abs_elo_diff": abs(elo_a - elo_b),
                }
            )
            for column, value in international.items():
                if column in features:
                    features.loc[:, column] = value
        profile_a = self.current_squad_profiles.get(
            normalize_player_team(team_a)
        )
        profile_b = self.current_squad_profiles.get(
            normalize_player_team(team_b)
        )
        for metric in PLAYER_PROFILE_METRICS:
            value_a = profile_a.get(metric, np.nan) if profile_a else np.nan
            value_b = profile_b.get(metric, np.nan) if profile_b else np.nan
            column_a = f"player_a_{metric}"
            column_b = f"player_b_{metric}"
            diff_column = f"player_diff_{metric}"
            if column_a in features:
                features.loc[:, column_a] = value_a
            if column_b in features:
                features.loc[:, column_b] = value_b
            if diff_column in features:
                features.loc[:, diff_column] = (
                    value_a - value_b
                    if pd.notna(value_a) and pd.notna(value_b)
                    else np.nan
                )
        return features

    @staticmethod
    def _catboost_common() -> Dict[str, Any]:
        return {
            "iterations": 300,
            "depth": 5,
            "learning_rate": 0.025,
            "l2_leaf_reg": 5.0,
            "random_seed": 15,
            "verbose": False,
            "allow_writing_files": False,
            "thread_count": -1,
        }

    @staticmethod
    def _normalize(models):
        total = sum(weight for _, _, weight in models)
        return [
            (name, model, weight / total)
            for name, model, weight in models
        ]

    def _named_regressors(self):
        models = [
            (
                "rf",
                v11.RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=3,
                    random_state=7,
                    n_jobs=-1,
                ),
                0.25,
            ),
            (
                "hgb",
                v11.Pipeline(
                    [
                        ("imp", v11.SimpleImputer(strategy="median")),
                        (
                            "m",
                            v11.HistGradientBoostingRegressor(
                                max_iter=300,
                                learning_rate=0.035,
                                max_leaf_nodes=15,
                                l2_regularization=0.08,
                                random_state=7,
                            ),
                        ),
                    ]
                ),
                0.20,
            ),
            (
                "poisson",
                v11.Pipeline(
                    [
                        ("imp", v11.SimpleImputer(strategy="median")),
                        ("sc", v11.StandardScaler()),
                        (
                            "m",
                            v11.PoissonRegressor(
                                alpha=0.25,
                                max_iter=2000,
                            ),
                        ),
                    ]
                ),
                0.15,
            ),
            (
                "catboost",
                CatBoostRegressor(
                    **self._catboost_common(),
                    loss_function="Poisson",
                ),
                0.10,
            ),
        ]
        return self._normalize(models)

    def _named_diff_regressors(self):
        models = [
            ("ridge", super()._diff_regressor(), 0.30),
            (
                "rf",
                v11.RandomForestRegressor(
                    n_estimators=250,
                    min_samples_leaf=4,
                    random_state=7,
                    n_jobs=-1,
                ),
                0.30,
            ),
            (
                "hgb",
                v11.Pipeline(
                    [
                        ("imp", v11.SimpleImputer(strategy="median")),
                        (
                            "m",
                            v11.HistGradientBoostingRegressor(
                                max_iter=250,
                                learning_rate=0.035,
                                max_leaf_nodes=15,
                                l2_regularization=0.10,
                                random_state=7,
                            ),
                        ),
                    ]
                ),
                0.25,
            ),
            (
                "catboost",
                CatBoostRegressor(
                    **self._catboost_common(),
                    loss_function="RMSE",
                ),
                0.10,
            ),
        ]
        return self._normalize(models)

    def _named_classifiers(self):
        result_columns = [
            column
            for column in self.feature_cols
            if not column.startswith("player_")
        ]
        models = [
            (
                "rf",
                FeatureSubsetEstimator(
                    v11.RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=3,
                        random_state=7,
                        n_jobs=-1,
                    ),
                    result_columns,
                ),
                0.35,
            ),
            (
                "hgb",
                FeatureSubsetEstimator(
                    v11.Pipeline(
                        [
                            ("imp", v11.SimpleImputer(strategy="median")),
                            (
                                "m",
                                v11.HistGradientBoostingClassifier(
                                    max_iter=250,
                                    learning_rate=0.035,
                                    max_leaf_nodes=15,
                                    l2_regularization=0.08,
                                    random_state=7,
                                ),
                            ),
                        ]
                    ),
                    result_columns,
                ),
                0.20,
            ),
            (
                "logistic",
                FeatureSubsetEstimator(
                    v11.Pipeline(
                        [
                            ("imp", v11.SimpleImputer(strategy="median")),
                            ("sc", v11.StandardScaler()),
                            (
                                "m",
                                v11.LogisticRegression(max_iter=2000),
                            ),
                        ]
                    ),
                    result_columns,
                ),
                0.10,
            ),
            (
                "catboost",
                FeatureSubsetEstimator(
                    CatBoostClassifier(
                        **self._catboost_common(),
                        loss_function="MultiClass",
                    ),
                    result_columns,
                ),
                0.10,
            ),
        ]
        return self._normalize(models)


class V15CatBoostModel(V13LiveSignalModel):
    player_profile_goal_weight = 0.25

    def __init__(
        self,
        player_model: V15CatBoostWorldCupModel,
        outcome_model: V15CatBoostWorldCupModel,
    ):
        super().__init__(player_model)
        self.outcome_model = V13LiveSignalModel(outcome_model)

    def predict(self, *args, **kwargs) -> Dict[str, Any]:
        team_a = kwargs.get("team_a", args[0] if args else "")
        team_b = kwargs.get("team_b", args[1] if len(args) > 1 else "")
        max_goals = kwargs.get(
            "max_goals",
            args[5] if len(args) > 5 else 10,
        )
        player_prediction = super().predict(*args, **kwargs)
        player_lambda_a = float(player_prediction["lambda_a"])
        player_lambda_b = float(player_prediction["lambda_b"])
        baseline_prediction = self.outcome_model.predict(*args, **kwargs)

        weight = self.player_profile_goal_weight
        lambda_a = (
            (1.0 - weight) * baseline_prediction["lambda_a"]
            + weight * player_lambda_a
        )
        lambda_b = (
            (1.0 - weight) * baseline_prediction["lambda_b"]
            + weight * player_lambda_b
        )
        score_matrix = v11.poisson_score_matrix(
            lambda_a,
            lambda_b,
            max_goals,
        )
        rho = baseline_prediction.get("calibration_notes", {}).get(
            "dixon_coles_rho",
            -0.08,
        )
        score_matrix = v11.apply_dixon_coles_adjustment(
            score_matrix,
            lambda_a,
            lambda_b,
            rho=rho,
        )
        score_matrix = v11.reweight_score_matrix_to_results(
            score_matrix,
            baseline_prediction["result_probabilities"],
        )

        prediction = player_prediction
        prediction["lambda_a"] = float(lambda_a)
        prediction["lambda_b"] = float(lambda_b)
        prediction.update(score_outputs(score_matrix, max_goals))
        prediction["result_probabilities"] = baseline_prediction[
            "result_probabilities"
        ]
        prediction["predicted_result"] = max(
            prediction["result_probabilities"],
            key=prediction["result_probabilities"].get,
        )
        adjustments = dict(prediction.pop("v13_adjustments"))
        adjustments.update(
            {
                "wdl_model": "v15_catboost",
                "learned_model_family": "sklearn_catboost_ensemble",
                "catboost_goal_models": True,
                "catboost_goal_difference_model": True,
                "catboost_result_model": True,
                "catboost_draw_model": False,
                "catboost_event_models": True,
                "player_profile_features": True,
                "player_profile_goal_weight": weight,
                "baseline_lambda_a": baseline_prediction["lambda_a"],
                "baseline_lambda_b": baseline_prediction["lambda_b"],
                "player_lambda_a": player_lambda_a,
                "player_lambda_b": player_lambda_b,
                "player_profiles_affect_wdl": False,
                "result_decision_rule": "probability_argmax",
                "v13_draw_threshold_decision": baseline_prediction[
                    "predicted_result"
                ],
                "expanded_training_data": getattr(
                    self,
                    "training_data_summary",
                    {},
                ),
            }
        )
        prediction["player_profiles"] = {
            "team_a": self.base_model.current_squad_profiles.get(
                normalize_player_team(team_a),
                {},
            ),
            "team_b": self.base_model.current_squad_profiles.get(
                normalize_player_team(team_b),
                {},
            ),
        }
        prediction["v15_adjustments"] = adjustments
        prediction["calibration_notes"].pop("v13", None)
        prediction["calibration_notes"]["v15"] = adjustments
        prediction["calibration_notes"]["hybrid_model_policy"] = (
            "V15 uses the highest CatBoost-enhanced W/D/L probability as its "
            "result decision and blends a separate player-aware CatBoost goal "
            "head into expected goals and exact scores. Player profiles do not "
            "alter W/D/L probabilities."
        )
        return prediction

    def update_after_match(
        self,
        team_a: str,
        team_b: str,
        goals_a: int,
        goals_b: int,
    ) -> Dict[str, float]:
        details = super().update_after_match(
            team_a,
            team_b,
            goals_a,
            goals_b,
        )
        self.outcome_model.update_after_match(
            team_a,
            team_b,
            goals_a,
            goals_b,
        )
        return details


def build_from_zip(
    zip_path,
    train_csv=None,
    test_csv=None,
    model_type="catboost",
    box_csv=None,
    results_csv=None,
    former_names_csv=None,
    prediction_year=2026,
    qualifier_blend_start_year=2014,
    qualifier_full_weight_year=2022,
    qualifier_minimum_influence=0.0,
    recency_half_life_years=16.0,
    recency_min_weight=0.10,
    player_ratings_csv=None,
    declared_squads_csv=None,
    results_as_of=DEFAULT_RESULTS_AS_OF,
):
    require_catboost()
    loader = v11.WorldCupSAILoader(
        zip_path,
        Path(str(zip_path) + "_extracted"),
    )
    matches = loader.load_matches()
    current = v11.load_current_team_features(train_csv, test_csv)
    data_dir = Path(__file__).resolve().parent / "data"
    player_ratings_csv = player_ratings_csv or (
        data_dir / "player_ratings_international.csv"
    )
    declared_squads_csv = declared_squads_csv or (
        data_dir / "world_cup_2026_declared_squads.csv"
    )
    player_ratings = load_player_ratings(player_ratings_csv)
    declared_squads = load_declared_squads(declared_squads_csv)
    current_squad_profiles = build_current_squad_profiles(
        declared_squads,
        player_ratings,
    )
    box = v11.load_kaggle_box_data(box_csv)
    qualification_results = v11.load_world_cup_qualification_results(
        results_csv,
        former_names_csv,
    )
    qualifier_source = (
        qualification_results if not qualification_results.empty else box
    )
    historical_current = pd.DataFrame(columns=["team"])
    frame, features, events = v11.build_rolling_features(
        matches,
        historical_current,
        qualifier_box=qualifier_source,
        qualifier_fallback_box=box,
        qualifier_blend_start_year=qualifier_blend_start_year,
        qualifier_full_weight_year=qualifier_full_weight_year,
        qualifier_minimum_influence=qualifier_minimum_influence,
    )
    international_results = load_international_results(
        results_csv,
        former_names_csv=former_names_csv,
        as_of=results_as_of,
    )
    timeline, international_state = build_international_timeline(
        international_results
    )
    resolved_results_as_of = international_results.attrs.get(
        "resolved_as_of",
        str(pd.Timestamp(results_as_of).date())
        if str(results_as_of).strip().lower()
        not in {"latest", "max", "latest_non_world_cup"}
        else DEFAULT_RESULTS_AS_OF,
    )
    expanded_frame, expanded_features, expansion_summary = (
        build_expanded_training_frame(
            frame,
            timeline,
        )
    )
    player_frame, player_features = add_historical_player_features(
        expanded_frame,
        player_ratings,
        expanded_features,
    )
    outcome_model = (
        V15CatBoostWorldCupModel(
            recency_half_life_years=recency_half_life_years,
            recency_min_weight=recency_min_weight,
        )
        .fit(expanded_frame, expanded_features, [], current)
        .set_box_data(box)
        .set_qualifier_data(
            qualifier_source,
            fallback_box=box,
            prediction_year=prediction_year,
            blend_start_year=qualifier_blend_start_year,
            full_weight_year=qualifier_full_weight_year,
            minimum_influence=qualifier_minimum_influence,
        )
        .set_current_international_state(
            international_state,
            resolved_results_as_of,
        )
    )
    player_model = (
        V15CatBoostWorldCupModel(
            recency_half_life_years=recency_half_life_years,
            recency_min_weight=recency_min_weight,
        )
        .fit(player_frame, player_features, events, current)
        .set_box_data(box)
        .set_qualifier_data(
            qualifier_source,
            fallback_box=box,
            prediction_year=prediction_year,
            blend_start_year=qualifier_blend_start_year,
            full_weight_year=qualifier_full_weight_year,
            minimum_influence=qualifier_minimum_influence,
        )
        .set_current_squad_profiles(current_squad_profiles)
        .set_current_international_state(
            international_state,
            resolved_results_as_of,
        )
    )
    data = v11.DataBundle(
        matches=matches,
        team_current=current,
        training_frame=player_frame,
        event_columns=events,
        box_frame=box,
    )
    model = V15CatBoostModel(player_model, outcome_model)
    model.training_data_summary = {
        **expansion_summary,
        "results_as_of": str(resolved_results_as_of),
        "results_as_of_requested": str(results_as_of),
        "excluded_current_world_cup_matches": int(
            international_results.attrs.get("excluded_current_world_cup_matches", 0)
        ),
        "continental_stage_features": False,
        "continental_stage_feature_reason": (
            "results.csv has no round or stage column"
        ),
    }
    return model, data


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    parser = argparse.ArgumentParser(
        description="Run the CatBoost-enhanced V15 World Cup ensemble."
    )
    parser.add_argument("--team-a", required=True)
    parser.add_argument("--team-b", required=True)
    parser.add_argument("--host-a", action="store_true")
    parser.add_argument("--host-b", action="store_true")
    parser.add_argument("--knockout", action="store_true")
    parser.add_argument(
        "--outdir",
        default="outputs/outputs_v15_catboost_prediction",
    )
    parser.add_argument(
        "--worldcupsai-zip",
        default=str(data_dir / "worldcupsai.zip"),
    )
    parser.add_argument(
        "--team-train",
        default=str(data_dir / "current_team_features_2026.csv"),
    )
    parser.add_argument("--team-test")
    parser.add_argument(
        "--box-data",
        default=str(data_dir / "FIFAallMatchBoxData.csv"),
    )
    parser.add_argument(
        "--results-data",
        default=str(data_dir / "results.csv"),
    )
    parser.add_argument(
        "--results-as-of",
        default=DEFAULT_RESULTS_AS_OF,
        help=(
            "Use results on or before this date for live international "
            "state. Use 'latest' to use the latest completed non-2026-World-Cup "
            "matches while excluding current World Cup finals."
        ),
    )
    parser.add_argument(
        "--former-names",
        default=str(data_dir / "former_names.csv"),
    )
    parser.add_argument("--prediction-year", type=int, default=2026)
    parser.add_argument(
        "--player-ratings",
        default=str(data_dir / "player_ratings_international.csv"),
    )
    parser.add_argument(
        "--declared-squads",
        default=str(data_dir / "world_cup_2026_declared_squads.csv"),
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    output_dir = v11.unique_output_dir(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, _ = build_from_zip(
        args.worldcupsai_zip,
        train_csv=args.team_train,
        test_csv=args.team_test,
        box_csv=args.box_data,
        results_csv=args.results_data,
        former_names_csv=args.former_names,
        prediction_year=args.prediction_year,
        player_ratings_csv=args.player_ratings,
        declared_squads_csv=args.declared_squads,
        results_as_of=args.results_as_of,
    )
    prediction = model.predict(
        args.team_a,
        args.team_b,
        host_a=args.host_a,
        host_b=args.host_b,
        knockout=args.knockout,
    )
    (output_dir / "single_match_prediction.json").write_text(
        json.dumps(prediction, indent=2)
    )
    (output_dir / "player_profiles.json").write_text(
        json.dumps(prediction["player_profiles"], indent=2)
    )
    pd.DataFrame(prediction["top_scorelines"]).to_csv(
        output_dir / "scoreline_probabilities_top.csv",
        index=False,
    )
    pd.DataFrame(prediction["scoreline_probabilities"]).to_csv(
        output_dir / "scoreline_probabilities.csv",
        index=False,
    )
    (output_dir / "model_summary.json").write_text(
        json.dumps(
            {
                "version": "v15-catboost",
                "learned_model_family": "sklearn_catboost_ensemble",
                "wdl_model": "v15_catboost",
                "exact_score_model": (
                    "catboost_enhanced_goals_with_v11_score_conversion"
                ),
                "team_a": prediction["team_a"],
                "team_b": prediction["team_b"],
                "player_ratings_source": args.player_ratings,
                "declared_squads_source": args.declared_squads,
                "expanded_training_data": model.training_data_summary,
                "player_profile_match_coverage": {
                    "team_a": prediction["player_profiles"]["team_a"].get(
                        "match_coverage"
                    ),
                    "team_b": prediction["player_profiles"]["team_b"].get(
                        "match_coverage"
                    ),
                },
            },
            indent=2,
        )
    )
    if not args.no_plots:
        v11.plot_prediction_outputs(prediction, output_dir)
    print(
        json.dumps(
            {
                "result_probabilities": prediction["result_probabilities"],
                "predicted_result": prediction["predicted_result"],
                "lambda_a": prediction["lambda_a"],
                "lambda_b": prediction["lambda_b"],
                "top_scorelines": prediction["top_scorelines"][:5],
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
'''
v15_catboost_model = _load_submodule("v15_catboost_model", _V15_CATBOOST_MODEL_SOURCE, "core_engine.py:v15_catboost_model")

# ======================================================================
# v18_hybrid_elo_form_player_model.py  (bundled as an isolated sub-module)
# ======================================================================
_V18_HYBRID_ELO_FORM_PLAYER_MODEL_SOURCE = r'''
#!/usr/bin/env python3
"""V18: V15 CatBoost base with current FC Ratings squad-strength adjustment."""

from __future__ import annotations

import argparse
import json
import math
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd

import v11_wcq_results_model as v11
import v15_catboost_model as v15


canon_team = v11.canon_team

V18_TEAM_ALIASES = {
    "cabo verde": "Cape Verde",
    "cape verde islands": "Cape Verde",
    "curacao": "Curaçao",
    "cote d ivoire": "Côte d'Ivoire",
    "congo dr": "DR Congo",
    "dr congo": "DR Congo",
    "korea republic": "South Korea",
    "south korea": "South Korea",
    "turkiye": "Turkey",
    "usa": "United States",
}

ATTACK_POSITIONS = {"ST", "CF", "LW", "RW", "CAM", "LM", "RM"}
MIDFIELD_POSITIONS = {"CM", "CDM", "CAM", "LM", "RM"}
DEFENSE_POSITIONS = {"CB", "LB", "RB", "LWB", "RWB", "CDM", "GK"}

SQUAD_PROFILE_COMPONENTS = (
    "attack_raw",
    "defense_raw",
    "midfield_raw",
    "keeper_raw",
)

COMPONENT_POSITIONS = {
    "attack": ATTACK_POSITIONS,
    "defense": DEFENSE_POSITIONS,
    "midfield": MIDFIELD_POSITIONS,
    "keeper": {"GK"},
}

POSITIONAL_IMPUTATION_PRIOR_WEIGHT = 6.0
UNMATCHED_PLAYER_DISCOUNT = 1.5

DEFAULT_BETA_ATTACK = 0.11
DEFAULT_BETA_MIDFIELD = 0.035
DEFAULT_BETA_KEEPER = 0.02
DEFAULT_MAX_LOG_ADJUSTMENT = 0.16
DEFAULT_RESULT_BLEND_WEIGHT = 0.22
DEFAULT_PLAYER_RATINGS_AFFECT_WDL = False


def normalize_team(value: Any) -> str:
    key = v15.normalize_text(value)
    return canon_team(V18_TEAM_ALIASES.get(key, str(value).strip()))


def position_set(value: Any) -> set[str]:
    return {
        item.strip().upper()
        for item in str(value or "").split(",")
        if item.strip()
    }


def _tokens(value: str) -> list[str]:
    return [token for token in v15.normalize_text(value).split() if token]


def compact_name_variants(value: Any) -> set[str]:
    tokens = _tokens(str(value or ""))
    if not tokens:
        return set()
    variants = {" ".join(tokens)}
    if len(tokens) >= 2:
        variants.add(f"{tokens[0]} {tokens[-1]}")
        variants.add(f"{tokens[0][0]} {tokens[-1]}")
        variants.add(tokens[-1])
    return variants


def squad_name_variants(row: pd.Series) -> set[str]:
    variants: set[str] = set()
    for column in ("long_name", "short_name"):
        variants.update(compact_name_variants(row.get(column, "")))
    return variants - {""}


def name_match_score(squad_row: pd.Series, player_name: str) -> float:
    variants = squad_name_variants(squad_row)
    player_variants = compact_name_variants(player_name)
    if not variants or not player_variants:
        return 0.0
    if variants & player_variants:
        return 1.0

    long_tokens = set(_tokens(str(squad_row.get("long_name", ""))))
    player_ordered_tokens = _tokens(player_name)
    player_tokens = set(player_ordered_tokens)
    if len(player_ordered_tokens) >= 2 and player_tokens.issubset(long_tokens):
        return 0.96
    if (
        len(player_ordered_tokens) >= 2
        and player_ordered_tokens[-1:] == _tokens(str(squad_row.get("short_name", "")))[-1:]
    ):
        squad_short = _tokens(str(squad_row.get("short_name", "")))
        if squad_short and squad_short[0][:1] == player_ordered_tokens[0][:1]:
            return 0.94

    return max(
        SequenceMatcher(None, left, right).ratio()
        for left in variants
        for right in player_variants
    )


def load_current_squad_names(
    path: str | Path | None,
    fifa_version: int = 26,
) -> pd.DataFrame:
    if not path or not Path(path).exists():
        return pd.DataFrame()
    squads = pd.read_csv(path)
    required = {
        "fifa_version",
        "nationality_name",
        "short_name",
        "long_name",
        "player_positions",
    }
    missing = sorted(required - set(squads.columns))
    if missing:
        raise ValueError(f"Squad ratings file is missing columns: {missing}")
    squads = squads.loc[squads["fifa_version"].eq(fifa_version)].copy()
    squads["team"] = squads["nationality_name"].map(normalize_team)
    squads["positions"] = squads["player_positions"].map(position_set)
    squads["role"] = squads["player_positions"].map(v15.player_role)
    return squads.reset_index(drop=True)


def load_fcratings_players(path: str | Path | None) -> pd.DataFrame:
    if not path or not Path(path).exists():
        return pd.DataFrame()
    players = pd.read_csv(path)
    required = {
        "country",
        "player_name",
        "position",
        "ovr",
        "pac",
        "sho",
        "pas",
        "dri",
        "def",
        "phy",
    }
    missing = sorted(required - set(players.columns))
    if missing:
        raise ValueError(f"FC Ratings file is missing columns: {missing}")
    players = players.copy()
    players["team"] = players["country"].map(normalize_team)
    players["positions"] = players["position"].map(position_set)
    for column in ("ovr", "pac", "sho", "pas", "dri", "def", "phy"):
        players[column] = pd.to_numeric(players[column], errors="coerce")
    return players.dropna(subset=["team", "player_name", "ovr"]).reset_index(
        drop=True
    )


def attacking_value(row: pd.Series) -> float:
    return float(
        0.40 * row["sho"]
        + 0.25 * row["dri"]
        + 0.20 * row["pac"]
        + 0.15 * row["pas"]
    )


def defensive_value(row: pd.Series) -> float:
    return float(
        0.45 * row["def"]
        + 0.25 * row["phy"]
        + 0.15 * row["pac"]
        + 0.15 * row["pas"]
    )


def midfield_value(row: pd.Series) -> float:
    return float(
        0.35 * row["pas"]
        + 0.25 * row["dri"]
        + 0.20 * row["def"]
        + 0.20 * row["phy"]
    )


def _mean_top(values: Iterable[float], count: int) -> float:
    clean = sorted(
        [float(value) for value in values if pd.notna(value)],
        reverse=True,
    )
    if not clean:
        return float("nan")
    return float(np.mean(clean[:count]))


def component_value(row: pd.Series, component: str) -> float:
    if component == "attack":
        return attacking_value(row)
    if component == "defense":
        return defensive_value(row)
    if component == "midfield":
        return midfield_value(row)
    if component == "keeper":
        return float(row["ovr"])
    raise ValueError(f"Unknown rating component: {component}")


def component_values(
    frame: pd.DataFrame,
    component: str,
    positions_column: str = "positions",
) -> list[float]:
    values = []
    wanted_positions = COMPONENT_POSITIONS[component]
    for _, row in frame.iterrows():
        positions = row.get(positions_column, set())
        if not isinstance(positions, set):
            positions = position_set(positions)
        if positions & wanted_positions:
            values.append(component_value(row, component))
    return values


def median_or_nan(values: Iterable[float]) -> float:
    clean = [float(value) for value in values if pd.notna(value)]
    return float(np.median(clean)) if clean else float("nan")


def global_component_baselines(fcratings: pd.DataFrame) -> Dict[str, float]:
    return {
        component: median_or_nan(component_values(fcratings, component))
        for component in COMPONENT_POSITIONS
    }


def shrunken_component_baseline(
    team_values: list[float],
    global_value: float,
) -> float:
    if not team_values and pd.isna(global_value):
        return float("nan")
    if not team_values:
        return float(global_value)
    team_value = median_or_nan(team_values)
    if pd.isna(global_value):
        return float(team_value)
    support = len(team_values)
    team_weight = support / (support + POSITIONAL_IMPUTATION_PRIOR_WEIGHT)
    estimate = team_weight * team_value + (1.0 - team_weight) * global_value
    discount = UNMATCHED_PLAYER_DISCOUNT * (
        POSITIONAL_IMPUTATION_PRIOR_WEIGHT
        / (support + POSITIONAL_IMPUTATION_PRIOR_WEIGHT)
    )
    return float(estimate - discount)


def imputed_component_values_for_player(
    squad_row: pd.Series,
    team_component_values: Dict[str, list[float]],
    global_baselines: Dict[str, float],
) -> Dict[str, float]:
    positions = squad_row["positions"]
    imputed = {}
    for component, wanted_positions in COMPONENT_POSITIONS.items():
        if positions & wanted_positions:
            imputed[component] = shrunken_component_baseline(
                team_component_values.get(component, []),
                global_baselines.get(component, float("nan")),
            )
    return imputed


def best_fcratings_match(
    squad_row: pd.Series,
    candidates: pd.DataFrame,
    threshold: float = 0.84,
) -> tuple[pd.Series | None, float]:
    best_index: Any = None
    best_score = 0.0
    for index, candidate in candidates.iterrows():
        score = name_match_score(squad_row, str(candidate["player_name"]))
        if score > best_score:
            best_index = index
            best_score = score
    if best_index is not None and best_score >= threshold:
        return candidates.loc[best_index], float(best_score)
    return None, float(best_score)


def _aggregate_matched_players(
    players: pd.DataFrame,
    imputed_rows: list[Dict[str, Any]] | None = None,
) -> Dict[str, float]:
    imputed_rows = imputed_rows or []
    if players.empty and not imputed_rows:
        return {
            "attack_raw": float("nan"),
            "defense_raw": float("nan"),
            "midfield_raw": float("nan"),
            "keeper_raw": float("nan"),
        }

    attack_values = []
    defense_values = []
    midfield_values = []
    keeper_values = []
    for _, row in players.iterrows():
        positions = row["squad_positions"]
        if positions & ATTACK_POSITIONS:
            attack_values.append(attacking_value(row))
        if positions & DEFENSE_POSITIONS:
            defense_values.append(defensive_value(row))
        if positions & MIDFIELD_POSITIONS:
            midfield_values.append(midfield_value(row))
        if "GK" in positions:
            keeper_values.append(float(row["ovr"]))

    for row in imputed_rows:
        positions = row["positions"]
        values = row["component_values"]
        if positions & ATTACK_POSITIONS and pd.notna(values.get("attack")):
            attack_values.append(float(values["attack"]))
        if positions & DEFENSE_POSITIONS and pd.notna(values.get("defense")):
            defense_values.append(float(values["defense"]))
        if positions & MIDFIELD_POSITIONS and pd.notna(values.get("midfield")):
            midfield_values.append(float(values["midfield"]))
        if "GK" in positions and pd.notna(values.get("keeper")):
            keeper_values.append(float(values["keeper"]))

    return {
        "attack_raw": _mean_top(attack_values, 5),
        "defense_raw": _mean_top(defense_values, 6),
        "midfield_raw": _mean_top(midfield_values, 5),
        "keeper_raw": _mean_top(keeper_values, 1),
    }


def build_current_fcratings_squad_profiles(
    squad_names: pd.DataFrame,
    fcratings: pd.DataFrame,
    match_threshold: float = 0.84,
) -> Dict[str, Dict[str, Any]]:
    if squad_names.empty or fcratings.empty:
        return {}

    global_baselines = global_component_baselines(fcratings)
    profiles: Dict[str, Dict[str, Any]] = {}
    for team, squad in squad_names.groupby("team", sort=True):
        candidates = fcratings[fcratings["team"].eq(team)]
        team_component_values = {
            component: component_values(candidates, component)
            for component in COMPONENT_POSITIONS
        }
        matched_rows = []
        imputed_rows = []
        scores = []
        unmatched = []
        used_urls: set[str] = set()
        for _, squad_row in squad.iterrows():
            available = candidates
            if "player_url" in available.columns:
                available = available.loc[
                    ~available["player_url"].astype(str).isin(used_urls)
                ]
            matched, score = best_fcratings_match(
                squad_row,
                available,
                threshold=match_threshold,
            )
            if matched is None:
                unmatched.append(str(squad_row.get("long_name", "")))
                imputed_rows.append(
                    {
                        "squad_name": str(squad_row.get("long_name", "")),
                        "positions": squad_row["positions"],
                        "component_values": imputed_component_values_for_player(
                            squad_row,
                            team_component_values,
                            global_baselines,
                        ),
                    }
                )
                continue
            matched_copy = matched.copy()
            matched_copy["squad_positions"] = squad_row["positions"]
            matched_copy["squad_role"] = squad_row["role"]
            matched_copy["squad_short_name"] = squad_row.get("short_name", "")
            matched_copy["squad_long_name"] = squad_row.get("long_name", "")
            matched_copy["match_score"] = score
            if matched_copy.get("player_url"):
                used_urls.add(str(matched_copy.get("player_url")))
            scores.append(score)
            matched_rows.append(matched_copy)

        matched_frame = (
            pd.DataFrame(matched_rows)
            if matched_rows
            else pd.DataFrame(columns=list(fcratings.columns))
        )
        profile = _aggregate_matched_players(matched_frame, imputed_rows)
        match_coverage = float(len(matched_frame) / max(len(squad), 1))
        imputed_share = float(len(imputed_rows) / max(len(squad), 1))
        team_data_support = float(np.clip(len(candidates) / 20.0, 0.0, 1.0))
        rating_confidence = float(
            np.clip(match_coverage + 0.25 * imputed_share * team_data_support, 0.0, 1.0)
        )
        profile.update(
            {
                "team": team,
                "squad_size": int(len(squad)),
                "fcratings_country_rows": int(len(candidates)),
                "matched_players": int(len(matched_frame)),
                "imputed_players": int(len(imputed_rows)),
                "match_coverage": match_coverage,
                "rating_confidence": rating_confidence,
                "mean_name_match_score": float(np.mean(scores)) if scores else 0.0,
                "imputation_policy": (
                    "Unmatched squad players are assigned shrunken positional "
                    "FC Ratings estimates: team component median blended toward "
                    "the global component median, with a sparse-team discount. "
                    "Actual squad-file rating values are never used."
                ),
                "unmatched_players": unmatched[:12],
                "imputed_player_examples": [
                    {
                        "squad_name": row["squad_name"],
                        "positions": sorted(row["positions"]),
                        **{
                            f"imputed_{key}": float(value)
                            for key, value in row["component_values"].items()
                            if pd.notna(value)
                        },
                    }
                    for row in imputed_rows[:12]
                ],
                "matched_player_examples": [
                    {
                        "squad_name": str(row.get("squad_long_name", "")),
                        "fcratings_name": str(row.get("player_name", "")),
                        "position": str(row.get("position", "")),
                        "ovr": float(row.get("ovr", np.nan)),
                        "match_score": float(row.get("match_score", np.nan)),
                    }
                    for _, row in matched_frame.head(12).iterrows()
                ],
            }
        )
        profiles[team] = profile

    return normalize_squad_profile_components(profiles)


def normalize_squad_profile_components(
    profiles: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    if not profiles:
        return profiles
    for component in SQUAD_PROFILE_COMPONENTS:
        values = np.asarray(
            [
                float(profile.get(component, np.nan))
                for profile in profiles.values()
                if pd.notna(profile.get(component, np.nan))
            ],
            dtype=float,
        )
        mean = float(np.mean(values)) if len(values) else float("nan")
        std = float(np.std(values, ddof=0)) if len(values) else float("nan")
        if not np.isfinite(std) or std < 1e-9:
            std = 1.0
        for profile in profiles.values():
            raw_value = profile.get(component, np.nan)
            z_column = component.replace("_raw", "_z")
            if pd.notna(raw_value) and np.isfinite(mean):
                profile[z_column] = float((float(raw_value) - mean) / std)
            else:
                profile[z_column] = 0.0
    return profiles


def result_probs_from_score_matrix(
    score_matrix: Dict[Tuple[int, int], float],
) -> Dict[str, float]:
    return {
        "team_a_win": float(
            sum(prob for (a, b), prob in score_matrix.items() if a > b)
        ),
        "draw": float(
            sum(prob for (a, b), prob in score_matrix.items() if a == b)
        ),
        "team_b_win": float(
            sum(prob for (a, b), prob in score_matrix.items() if a < b)
        ),
    }


def blend_result_probabilities(
    base: Dict[str, float],
    adjusted: Dict[str, float],
    adjusted_weight: float,
) -> Dict[str, float]:
    result = {
        key: (1.0 - adjusted_weight) * float(base[key])
        + adjusted_weight * float(adjusted[key])
        for key in ("team_a_win", "draw", "team_b_win")
    }
    total = sum(result.values())
    return {key: float(value / total) for key, value in result.items()}


class V18HybridSquadModel:
    """Wrap V15 and apply current FC Ratings squad quality on log-goals."""

    def __init__(
        self,
        base_model: v15.V15CatBoostModel,
        squad_profiles: Dict[str, Dict[str, Any]],
        beta_attack: float = DEFAULT_BETA_ATTACK,
        beta_midfield: float = DEFAULT_BETA_MIDFIELD,
        beta_keeper: float = DEFAULT_BETA_KEEPER,
        max_log_adjustment: float = DEFAULT_MAX_LOG_ADJUSTMENT,
        result_blend_weight: float = DEFAULT_RESULT_BLEND_WEIGHT,
        player_ratings_affect_wdl: bool = DEFAULT_PLAYER_RATINGS_AFFECT_WDL,
    ):
        self.base_model = base_model
        self.squad_profiles = squad_profiles
        self.beta_attack = float(beta_attack)
        self.beta_midfield = float(beta_midfield)
        self.beta_keeper = float(beta_keeper)
        self.max_log_adjustment = float(max_log_adjustment)
        self.result_blend_weight = float(result_blend_weight)
        self.player_ratings_affect_wdl = bool(player_ratings_affect_wdl)
        self.training_data_summary = getattr(
            base_model,
            "training_data_summary",
            {},
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_model, name)

    def profile_for_team(self, team: str) -> Dict[str, Any]:
        return self.squad_profiles.get(normalize_team(team), {})

    @staticmethod
    def _coverage(profile: Dict[str, Any]) -> float:
        return float(
            np.clip(
                profile.get(
                    "rating_confidence",
                    profile.get("match_coverage", 0.0),
                ),
                0.0,
                1.0,
            )
        )

    def _log_adjustments(
        self,
        profile_a: Dict[str, Any],
        profile_b: Dict[str, Any],
    ) -> tuple[float, float, Dict[str, float]]:
        coverage_a = self._coverage(profile_a)
        coverage_b = self._coverage(profile_b)
        pair_coverage = math.sqrt(max(coverage_a * coverage_b, 0.0))

        attack_edge_a = float(profile_a.get("attack_z", 0.0)) - float(
            profile_b.get("defense_z", 0.0)
        )
        attack_edge_b = float(profile_b.get("attack_z", 0.0)) - float(
            profile_a.get("defense_z", 0.0)
        )
        midfield_edge_a = float(profile_a.get("midfield_z", 0.0)) - float(
            profile_b.get("midfield_z", 0.0)
        )
        keeper_edge_a = float(profile_a.get("keeper_z", 0.0)) - float(
            profile_b.get("keeper_z", 0.0)
        )

        raw_a = (
            self.beta_attack * attack_edge_a
            + self.beta_midfield * midfield_edge_a
            - self.beta_keeper * keeper_edge_a
        )
        raw_b = (
            self.beta_attack * attack_edge_b
            - self.beta_midfield * midfield_edge_a
            + self.beta_keeper * keeper_edge_a
        )
        log_a = float(
            np.clip(
                pair_coverage * raw_a,
                -self.max_log_adjustment,
                self.max_log_adjustment,
            )
        )
        log_b = float(
            np.clip(
                pair_coverage * raw_b,
                -self.max_log_adjustment,
                self.max_log_adjustment,
            )
        )
        details = {
            "coverage_a": coverage_a,
            "coverage_b": coverage_b,
            "raw_match_coverage_a": float(
                np.clip(profile_a.get("match_coverage", 0.0), 0.0, 1.0)
            ),
            "raw_match_coverage_b": float(
                np.clip(profile_b.get("match_coverage", 0.0), 0.0, 1.0)
            ),
            "pair_coverage_shrink": pair_coverage,
            "attack_edge_a": attack_edge_a,
            "attack_edge_b": attack_edge_b,
            "midfield_edge_a": midfield_edge_a,
            "keeper_edge_a": keeper_edge_a,
            "raw_log_adjustment_a": raw_a,
            "raw_log_adjustment_b": raw_b,
            "applied_log_adjustment_a": log_a,
            "applied_log_adjustment_b": log_b,
        }
        return log_a, log_b, details

    def predict(self, *args, **kwargs) -> Dict[str, Any]:
        team_a = kwargs.get("team_a", args[0] if args else "")
        team_b = kwargs.get("team_b", args[1] if len(args) > 1 else "")
        max_goals = int(kwargs.get("max_goals", args[5] if len(args) > 5 else 10))
        prediction = self.base_model.predict(*args, **kwargs)
        base_lambda_a = float(prediction["lambda_a"])
        base_lambda_b = float(prediction["lambda_b"])
        base_results = dict(prediction["result_probabilities"])

        profile_a = self.profile_for_team(str(team_a))
        profile_b = self.profile_for_team(str(team_b))
        log_a, log_b, adjustment_details = self._log_adjustments(
            profile_a,
            profile_b,
        )
        lambda_a = float(np.clip(base_lambda_a * math.exp(log_a), 0.15, 4.5))
        lambda_b = float(np.clip(base_lambda_b * math.exp(log_b), 0.15, 4.5))
        score_matrix = v11.poisson_score_matrix(lambda_a, lambda_b, max_goals)
        rho = prediction.get("calibration_notes", {}).get(
            "dixon_coles_rho",
            -0.08,
        )
        score_matrix = v11.apply_dixon_coles_adjustment(
            score_matrix,
            lambda_a,
            lambda_b,
            rho=rho,
        )
        adjusted_results = result_probs_from_score_matrix(score_matrix)
        if self.player_ratings_affect_wdl:
            result_probabilities = blend_result_probabilities(
                base_results,
                adjusted_results,
                self.result_blend_weight,
            )
        else:
            result_probabilities = base_results
        score_matrix = v11.reweight_score_matrix_to_results(
            score_matrix,
            result_probabilities,
        )

        prediction["lambda_a"] = lambda_a
        prediction["lambda_b"] = lambda_b
        prediction.update(v15.score_outputs(score_matrix, max_goals))
        prediction["result_probabilities"] = result_probabilities
        if self.player_ratings_affect_wdl:
            prediction["predicted_result"] = max(
                result_probabilities,
                key=result_probabilities.get,
            )
        prediction["v18_squad_profiles"] = {
            "team_a": profile_a,
            "team_b": profile_b,
        }
        prediction["v18_adjustments"] = {
            "base_model": "v15_catboost",
            "ratings_source": "fcratings_top50_worldcup2026",
            "squad_name_source": "player_ratings_international",
            "squad_file_rating_values_used": False,
            "player_ratings_affect_wdl": self.player_ratings_affect_wdl,
            "player_ratings_affect_scorelines": True,
            "result_blend_weight": self.result_blend_weight,
            "beta_attack": self.beta_attack,
            "beta_midfield": self.beta_midfield,
            "beta_keeper": self.beta_keeper,
            "max_log_adjustment": self.max_log_adjustment,
            "base_lambda_a": base_lambda_a,
            "base_lambda_b": base_lambda_b,
            "adjusted_poisson_result_probabilities": adjusted_results,
            "base_result_probabilities": base_results,
            **adjustment_details,
        }
        prediction["calibration_notes"] = {
            **prediction.get("calibration_notes", {}),
            "v18": prediction["v18_adjustments"],
            "hybrid_model_policy": (
                "V18 uses V15 CatBoost as the base model, then applies a "
                "tempered, confidence-shrunk FC Ratings squad-strength "
                "adjustment on the log-goal scale. The squad file supplies "
                "player names and positions only; numeric player ratings come "
                "from FC Ratings. Unmatched squad players are imputed from "
                "shrunken positional FC Ratings baselines. By default the "
                "squad adjustment changes expected goals and exact-score "
                "probabilities only, while W/D/L probabilities and the result "
                "decision stay with V15."
            ),
        }
        return prediction

    def update_after_match(
        self,
        team_a: str,
        team_b: str,
        goals_a: int,
        goals_b: int,
    ) -> Dict[str, float]:
        return self.base_model.update_after_match(
            team_a,
            team_b,
            goals_a,
            goals_b,
        )


def build_from_zip(
    zip_path,
    train_csv=None,
    test_csv=None,
    model_type="catboost",
    box_csv=None,
    results_csv=None,
    former_names_csv=None,
    prediction_year=2026,
    qualifier_blend_start_year=2014,
    qualifier_full_weight_year=2022,
    qualifier_minimum_influence=0.0,
    recency_half_life_years=16.0,
    recency_min_weight=0.10,
    player_ratings_csv=None,
    declared_squads_csv=None,
    fcratings_csv=None,
    results_as_of=v15.DEFAULT_RESULTS_AS_OF,
    beta_attack=DEFAULT_BETA_ATTACK,
    beta_midfield=DEFAULT_BETA_MIDFIELD,
    beta_keeper=DEFAULT_BETA_KEEPER,
    max_log_adjustment=DEFAULT_MAX_LOG_ADJUSTMENT,
    result_blend_weight=DEFAULT_RESULT_BLEND_WEIGHT,
    player_ratings_affect_wdl=DEFAULT_PLAYER_RATINGS_AFFECT_WDL,
    match_threshold=0.84,
):
    data_dir = Path(__file__).resolve().parent / "data"
    player_ratings_csv = player_ratings_csv or (
        data_dir / "player_ratings_international.csv"
    )
    fcratings_csv = fcratings_csv or (data_dir / "fcratings_top50_worldcup2026.csv")
    base_model, data = v15.build_from_zip(
        zip_path,
        train_csv=train_csv,
        test_csv=test_csv,
        model_type=model_type,
        box_csv=box_csv,
        results_csv=results_csv,
        former_names_csv=former_names_csv,
        prediction_year=prediction_year,
        qualifier_blend_start_year=qualifier_blend_start_year,
        qualifier_full_weight_year=qualifier_full_weight_year,
        qualifier_minimum_influence=qualifier_minimum_influence,
        recency_half_life_years=recency_half_life_years,
        recency_min_weight=recency_min_weight,
        player_ratings_csv=player_ratings_csv,
        declared_squads_csv=declared_squads_csv,
        results_as_of=results_as_of,
    )
    squad_names = load_current_squad_names(player_ratings_csv)
    fcratings = load_fcratings_players(fcratings_csv)
    squad_profiles = build_current_fcratings_squad_profiles(
        squad_names,
        fcratings,
        match_threshold=match_threshold,
    )
    model = V18HybridSquadModel(
        base_model,
        squad_profiles,
        beta_attack=beta_attack,
        beta_midfield=beta_midfield,
        beta_keeper=beta_keeper,
        max_log_adjustment=max_log_adjustment,
        result_blend_weight=result_blend_weight,
        player_ratings_affect_wdl=player_ratings_affect_wdl,
    )
    model.training_data_summary = {
        **getattr(base_model, "training_data_summary", {}),
        "v18_squad_profile_teams": len(squad_profiles),
        "v18_fcratings_rows": int(len(fcratings)),
        "v18_squad_name_rows": int(len(squad_names)),
    }
    return model, data


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    parser = argparse.ArgumentParser(
        description="Run V18: V15 CatBoost with FC Ratings squad adjustment."
    )
    parser.add_argument("--team-a", required=True)
    parser.add_argument("--team-b", required=True)
    parser.add_argument("--host-a", action="store_true")
    parser.add_argument("--host-b", action="store_true")
    parser.add_argument("--knockout", action="store_true")
    parser.add_argument(
        "--outdir",
        default="outputs/outputs_v18_hybrid_prediction",
    )
    parser.add_argument(
        "--worldcupsai-zip",
        default=str(data_dir / "worldcupsai.zip"),
    )
    parser.add_argument(
        "--team-train",
        default=str(data_dir / "current_team_features_2026.csv"),
    )
    parser.add_argument("--team-test")
    parser.add_argument(
        "--box-data",
        default=str(data_dir / "FIFAallMatchBoxData.csv"),
    )
    parser.add_argument(
        "--results-data",
        default=str(data_dir / "results.csv"),
    )
    parser.add_argument("--results-as-of", default=v15.DEFAULT_RESULTS_AS_OF)
    parser.add_argument(
        "--former-names",
        default=str(data_dir / "former_names.csv"),
    )
    parser.add_argument("--prediction-year", type=int, default=2026)
    parser.add_argument(
        "--player-ratings",
        default=str(data_dir / "player_ratings_international.csv"),
        help="Used for current squad player names/positions only in V18.",
    )
    parser.add_argument(
        "--declared-squads",
        default=str(data_dir / "world_cup_2026_declared_squads.csv"),
        help="Passed through to the V15 base model.",
    )
    parser.add_argument(
        "--fcratings",
        default=str(data_dir / "fcratings_top50_worldcup2026.csv"),
    )
    parser.add_argument("--beta-attack", type=float, default=DEFAULT_BETA_ATTACK)
    parser.add_argument(
        "--beta-midfield",
        type=float,
        default=DEFAULT_BETA_MIDFIELD,
    )
    parser.add_argument("--beta-keeper", type=float, default=DEFAULT_BETA_KEEPER)
    parser.add_argument(
        "--max-log-adjustment",
        type=float,
        default=DEFAULT_MAX_LOG_ADJUSTMENT,
    )
    parser.add_argument(
        "--result-blend-weight",
        type=float,
        default=DEFAULT_RESULT_BLEND_WEIGHT,
    )
    parser.add_argument(
        "--player-ratings-affect-wdl",
        action="store_true",
        default=DEFAULT_PLAYER_RATINGS_AFFECT_WDL,
        help=(
            "Let the FC Ratings squad layer blend W/D/L probabilities. By "
            "default, player ratings affect lambdas and exact scorelines only."
        ),
    )
    parser.add_argument("--match-threshold", type=float, default=0.84)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    output_dir = v11.unique_output_dir(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, _ = build_from_zip(
        args.worldcupsai_zip,
        train_csv=args.team_train,
        test_csv=args.team_test,
        box_csv=args.box_data,
        results_csv=args.results_data,
        former_names_csv=args.former_names,
        prediction_year=args.prediction_year,
        player_ratings_csv=args.player_ratings,
        declared_squads_csv=args.declared_squads,
        fcratings_csv=args.fcratings,
        results_as_of=args.results_as_of,
        beta_attack=args.beta_attack,
        beta_midfield=args.beta_midfield,
        beta_keeper=args.beta_keeper,
        max_log_adjustment=args.max_log_adjustment,
        result_blend_weight=args.result_blend_weight,
        player_ratings_affect_wdl=args.player_ratings_affect_wdl,
        match_threshold=args.match_threshold,
    )
    prediction = model.predict(
        args.team_a,
        args.team_b,
        host_a=args.host_a,
        host_b=args.host_b,
        knockout=args.knockout,
    )
    (output_dir / "single_match_prediction.json").write_text(
        json.dumps(prediction, indent=2)
    )
    (output_dir / "v18_squad_profiles.json").write_text(
        json.dumps(prediction["v18_squad_profiles"], indent=2)
    )
    pd.DataFrame(prediction["top_scorelines"]).to_csv(
        output_dir / "scoreline_probabilities_top.csv",
        index=False,
    )
    pd.DataFrame(prediction["scoreline_probabilities"]).to_csv(
        output_dir / "scoreline_probabilities.csv",
        index=False,
    )
    (output_dir / "model_summary.json").write_text(
        json.dumps(
            {
                "version": "v18-hybrid-elo-form-fcratings",
                "base_model": "v15_catboost",
                "team_a": prediction["team_a"],
                "team_b": prediction["team_b"],
                "player_ratings_source": args.fcratings,
                "squad_name_source": args.player_ratings,
                "squad_file_rating_values_used": False,
                "expanded_training_data": model.training_data_summary,
                "v18_adjustments": prediction["v18_adjustments"],
                "coverage": {
                    "team_a": prediction["v18_adjustments"]["coverage_a"],
                    "team_b": prediction["v18_adjustments"]["coverage_b"],
                },
            },
            indent=2,
        )
    )
    if not args.no_plots:
        v11.plot_prediction_outputs(prediction, output_dir)
    print(
        json.dumps(
            {
                "result_probabilities": prediction["result_probabilities"],
                "predicted_result": prediction["predicted_result"],
                "lambda_a": prediction["lambda_a"],
                "lambda_b": prediction["lambda_b"],
                "top_scorelines": prediction["top_scorelines"][:5],
                "v18_adjustments": prediction["v18_adjustments"],
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
'''
v18_hybrid_elo_form_player_model = _load_submodule("v18_hybrid_elo_form_player_model", _V18_HYBRID_ELO_FORM_PLAYER_MODEL_SOURCE, "core_engine.py:v18_hybrid_elo_form_player_model")

# ======================================================================
# v23_no_player_scoreline_model.py  (bundled as an isolated sub-module)
# ======================================================================
_V23_NO_PLAYER_SCORELINE_MODEL_SOURCE = r'''
#!/usr/bin/env python3
"""V23: no-player scoreline layer on top of the V15 outcome model."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd

import v11_wcq_results_model as v11
import v15_catboost_model as v15


canon_team = v11.canon_team
ScoreMatrix = Dict[Tuple[int, int], float]

DEFAULT_SCORELINE_LAYER_WEIGHT = 0.55
DEFAULT_FAVORITE_TAIL_STRENGTH = 0.32
DEFAULT_FAVORITE_TAIL_THRESHOLD = 0.60
DEFAULT_RERANKER_STRENGTH = 0.18
DEFAULT_DIVERSITY_RELATIVE_FLOOR = 0.42


def normalize_matrix(score_matrix: ScoreMatrix) -> ScoreMatrix:
    total = float(sum(score_matrix.values()))
    if total <= 0:
        raise ValueError("score matrix has no probability mass")
    return {key: float(value) / total for key, value in score_matrix.items()}


def score_matrix_from_prediction(prediction: Dict[str, Any]) -> ScoreMatrix:
    return {
        (int(item["team_a_goals"]), int(item["team_b_goals"])): float(
            item["probability"]
        )
        for item in prediction["scoreline_probabilities"]
    }


def blend_score_matrices(
    base_matrix: ScoreMatrix,
    adjusted_matrix: ScoreMatrix,
    adjusted_weight: float,
) -> ScoreMatrix:
    weight = float(np.clip(adjusted_weight, 0.0, 1.0))
    keys = set(base_matrix) | set(adjusted_matrix)
    return normalize_matrix(
        {
            key: (1.0 - weight) * base_matrix.get(key, 0.0)
            + weight * adjusted_matrix.get(key, 0.0)
            for key in keys
        }
    )


def result_label(goals_a: int, goals_b: int) -> str:
    if goals_a > goals_b:
        return "team_a_win"
    if goals_a < goals_b:
        return "team_b_win"
    return "draw"


def favorite_context(
    result_probabilities: Dict[str, float],
    lambda_a: float,
    lambda_b: float,
) -> Dict[str, Any]:
    win_probs = {
        "team_a_win": float(result_probabilities["team_a_win"]),
        "team_b_win": float(result_probabilities["team_b_win"]),
    }
    favorite_result = max(win_probs, key=win_probs.get)
    underdog_result = (
        "team_b_win" if favorite_result == "team_a_win" else "team_a_win"
    )
    favorite_is_a = favorite_result == "team_a_win"
    favorite_lambda = float(lambda_a if favorite_is_a else lambda_b)
    underdog_lambda = float(lambda_b if favorite_is_a else lambda_a)
    return {
        "favorite_result": favorite_result,
        "underdog_result": underdog_result,
        "favorite_is_a": favorite_is_a,
        "favorite_probability": win_probs[favorite_result],
        "underdog_probability": win_probs[underdog_result],
        "draw_probability": float(result_probabilities["draw"]),
        "favorite_lambda": favorite_lambda,
        "underdog_lambda": underdog_lambda,
        "lambda_gap": favorite_lambda - underdog_lambda,
        "probability_gap": win_probs[favorite_result] - win_probs[underdog_result],
    }


def _favorite_goals(goals_a: int, goals_b: int, favorite_is_a: bool) -> int:
    return goals_a if favorite_is_a else goals_b


def _underdog_goals(goals_a: int, goals_b: int, favorite_is_a: bool) -> int:
    return goals_b if favorite_is_a else goals_a


def favorite_tail_multiplier(
    goals_a: int,
    goals_b: int,
    context: Dict[str, Any],
    strength: float,
    threshold: float,
) -> float:
    if result_label(goals_a, goals_b) != context["favorite_result"]:
        return 1.0
    favorite_goals = _favorite_goals(
        goals_a,
        goals_b,
        bool(context["favorite_is_a"]),
    )
    underdog_goals = _underdog_goals(
        goals_a,
        goals_b,
        bool(context["favorite_is_a"]),
    )
    if favorite_goals < 3:
        return 1.0

    probability_gate = np.clip(
        (float(context["favorite_probability"]) - float(threshold))
        / max(0.82 - float(threshold), 1e-6),
        0.0,
        1.0,
    )
    lambda_gate = np.clip((float(context["lambda_gap"]) - 0.15) / 0.95, 0.0, 1.0)
    gate = max(float(probability_gate), float(lambda_gate) * 0.8)
    if gate <= 0:
        return 1.0

    margin = favorite_goals - underdog_goals
    goal_shape = 0.75 + 0.18 * (favorite_goals - 3) + 0.14 * max(margin - 1, 0)
    return float(min(1.0 + float(strength) * gate * goal_shape, 1.85))


def apply_favorite_tail_boost(
    score_matrix: ScoreMatrix,
    result_probabilities: Dict[str, float],
    lambda_a: float,
    lambda_b: float,
    strength: float,
    threshold: float,
) -> tuple[ScoreMatrix, Dict[str, float]]:
    context = favorite_context(result_probabilities, lambda_a, lambda_b)
    adjusted = {
        key: value
        * favorite_tail_multiplier(
            key[0],
            key[1],
            context,
            strength=strength,
            threshold=threshold,
        )
        for key, value in score_matrix.items()
    }
    adjusted = v11.reweight_score_matrix_to_results(
        normalize_matrix(adjusted),
        result_probabilities,
    )
    return adjusted, {
        "favorite_probability": float(context["favorite_probability"]),
        "favorite_lambda": float(context["favorite_lambda"]),
        "underdog_lambda": float(context["underdog_lambda"]),
        "lambda_gap": float(context["lambda_gap"]),
    }


def reranker_multiplier(
    goals_a: int,
    goals_b: int,
    context: Dict[str, Any],
    lambda_a: float,
    lambda_b: float,
    strength: float,
) -> float:
    label = result_label(goals_a, goals_b)
    total_goals = goals_a + goals_b
    multiplier = 1.0
    expected_total = float(lambda_a) + float(lambda_b)

    if label == context["favorite_result"]:
        favorite_goals = _favorite_goals(
            goals_a,
            goals_b,
            bool(context["favorite_is_a"]),
        )
        underdog_goals = _underdog_goals(
            goals_a,
            goals_b,
            bool(context["favorite_is_a"]),
        )
        if favorite_goals >= 3 and float(context["favorite_probability"]) >= 0.60:
            multiplier += float(strength) * (0.75 + 0.12 * (favorite_goals - 3))
        if favorite_goals == 1 and underdog_goals <= 1 and float(
            context["favorite_probability"]
        ) >= 0.62:
            multiplier -= float(strength) * 0.35

    if label == "draw":
        draw_prob = float(context["draw_probability"])
        if goals_a == goals_b == 1 and expected_total >= 2.35:
            multiplier -= float(strength) * 0.30
        if goals_a == goals_b == 2 and draw_prob >= 0.23 and expected_total >= 2.20:
            multiplier += float(strength) * 0.95
        if goals_a == goals_b == 0 and draw_prob >= 0.22 and expected_total <= 2.15:
            multiplier += float(strength) * 0.80

    if total_goals >= 5 and max(float(lambda_a), float(lambda_b)) < 1.70:
        multiplier -= float(strength) * 0.25
    return float(max(multiplier, 0.35))


def apply_scoreline_reranker(
    score_matrix: ScoreMatrix,
    result_probabilities: Dict[str, float],
    lambda_a: float,
    lambda_b: float,
    strength: float,
) -> ScoreMatrix:
    context = favorite_context(result_probabilities, lambda_a, lambda_b)
    adjusted = {
        key: value
        * reranker_multiplier(
            key[0],
            key[1],
            context,
            lambda_a=lambda_a,
            lambda_b=lambda_b,
            strength=strength,
        )
        for key, value in score_matrix.items()
    }
    return v11.reweight_score_matrix_to_results(
        normalize_matrix(adjusted),
        result_probabilities,
    )


def score_item(key: Tuple[int, int], probability: float) -> Dict[str, Any]:
    return {
        "team_a_goals": int(key[0]),
        "team_b_goals": int(key[1]),
        "probability": float(probability),
    }


def sorted_score_items(score_matrix: ScoreMatrix) -> list[Dict[str, Any]]:
    return [
        score_item(key, probability)
        for key, probability in sorted(
            score_matrix.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]


def _item_key(item: Dict[str, Any]) -> Tuple[int, int]:
    return int(item["team_a_goals"]), int(item["team_b_goals"])


def _best_candidate(
    score_matrix: ScoreMatrix,
    candidates: Iterable[Tuple[int, int]],
) -> Tuple[int, int] | None:
    available = [key for key in candidates if key in score_matrix]
    if not available:
        return None
    return max(available, key=lambda key: score_matrix.get(key, 0.0))


def diversity_candidates(
    score_matrix: ScoreMatrix,
    result_probabilities: Dict[str, float],
    lambda_a: float,
    lambda_b: float,
) -> list[Tuple[int, int]]:
    context = favorite_context(result_probabilities, lambda_a, lambda_b)
    candidates: list[Tuple[int, int]] = []
    favorite_is_a = bool(context["favorite_is_a"])
    favorite_probability = float(context["favorite_probability"])
    expected_total = float(lambda_a) + float(lambda_b)

    if favorite_probability >= 0.60:
        if favorite_is_a:
            tail = [(3, 0), (3, 1), (4, 0), (4, 1)]
        else:
            tail = [(0, 3), (1, 3), (0, 4), (1, 4)]
        best_tail = _best_candidate(score_matrix, tail)
        if best_tail is not None:
            candidates.append(best_tail)

    if (
        float(result_probabilities["draw"]) >= 0.23
        and expected_total >= 2.20
        and max(result_probabilities.values()) <= 0.55
    ):
        candidates.append((2, 2))
    if (
        float(result_probabilities["draw"]) >= 0.22
        and expected_total <= 2.15
        and max(result_probabilities.values()) <= 0.62
    ):
        candidates.append((0, 0))
    return candidates


def diversify_top_scorelines(
    score_matrix: ScoreMatrix,
    result_probabilities: Dict[str, float],
    lambda_a: float,
    lambda_b: float,
    top_n: int = 15,
    relative_floor: float = DEFAULT_DIVERSITY_RELATIVE_FLOOR,
) -> list[Dict[str, Any]]:
    ranked = sorted_score_items(score_matrix)
    selected = ranked[:top_n]
    if top_n < 3 or len(selected) < 3:
        return selected

    selected_keys = [_item_key(item) for item in selected]
    top_three = selected_keys[:3]
    top_probability = max(float(selected[0]["probability"]), 1e-12)
    floor = top_probability * float(relative_floor)

    for candidate in diversity_candidates(
        score_matrix,
        result_probabilities,
        lambda_a,
        lambda_b,
    ):
        if candidate in top_three:
            continue
        if float(score_matrix.get(candidate, 0.0)) < floor:
            continue
        top_three[-1] = candidate

    rebuilt_keys = []
    for key in [*top_three, *selected_keys]:
        if key not in rebuilt_keys:
            rebuilt_keys.append(key)
        if len(rebuilt_keys) >= top_n:
            break
    return [score_item(key, score_matrix[key]) for key in rebuilt_keys]


def postprocess_score_matrix(
    baseline_matrix: ScoreMatrix,
    result_probabilities: Dict[str, float],
    lambda_a: float,
    lambda_b: float,
    scoreline_layer_weight: float,
    favorite_tail_strength: float,
    favorite_tail_threshold: float,
    reranker_strength: float,
) -> tuple[ScoreMatrix, Dict[str, Any]]:
    adjusted, tail_diagnostics = apply_favorite_tail_boost(
        baseline_matrix,
        result_probabilities,
        lambda_a,
        lambda_b,
        strength=favorite_tail_strength,
        threshold=favorite_tail_threshold,
    )
    adjusted = apply_scoreline_reranker(
        adjusted,
        result_probabilities,
        lambda_a,
        lambda_b,
        strength=reranker_strength,
    )
    blended = blend_score_matrices(
        baseline_matrix,
        adjusted,
        adjusted_weight=scoreline_layer_weight,
    )
    blended = v11.reweight_score_matrix_to_results(blended, result_probabilities)
    diagnostics = {
        "scoreline_layer_weight": float(np.clip(scoreline_layer_weight, 0.0, 1.0)),
        "favorite_tail_strength": float(favorite_tail_strength),
        "favorite_tail_threshold": float(favorite_tail_threshold),
        "reranker_strength": float(reranker_strength),
        **tail_diagnostics,
    }
    return blended, diagnostics


class V23NoPlayerScorelineModel:
    """Use V15 W/D/L without player-profile scoring, then rerank exact scores."""

    def __init__(
        self,
        base_model: v15.V15CatBoostModel,
        scoreline_layer_weight: float = DEFAULT_SCORELINE_LAYER_WEIGHT,
        favorite_tail_strength: float = DEFAULT_FAVORITE_TAIL_STRENGTH,
        favorite_tail_threshold: float = DEFAULT_FAVORITE_TAIL_THRESHOLD,
        reranker_strength: float = DEFAULT_RERANKER_STRENGTH,
        diversity_relative_floor: float = DEFAULT_DIVERSITY_RELATIVE_FLOOR,
    ):
        self.base_model = base_model
        self.outcome_model = getattr(base_model, "outcome_model", base_model)
        self.scoreline_layer_weight = float(np.clip(scoreline_layer_weight, 0.0, 1.0))
        self.favorite_tail_strength = float(max(favorite_tail_strength, 0.0))
        self.favorite_tail_threshold = float(
            np.clip(favorite_tail_threshold, 0.0, 1.0)
        )
        self.reranker_strength = float(max(reranker_strength, 0.0))
        self.diversity_relative_floor = float(max(diversity_relative_floor, 0.0))
        self.training_data_summary = getattr(base_model, "training_data_summary", {})

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_model, name)

    def predict(self, *args, **kwargs) -> Dict[str, Any]:
        max_goals = int(kwargs.get("max_goals", args[5] if len(args) > 5 else 10))
        base_prediction = copy.deepcopy(self.outcome_model.predict(*args, **kwargs))
        result_probabilities = dict(base_prediction["result_probabilities"])
        baseline_matrix = score_matrix_from_prediction(base_prediction)
        lambda_a = float(base_prediction["lambda_a"])
        lambda_b = float(base_prediction["lambda_b"])

        score_matrix, diagnostics = postprocess_score_matrix(
            baseline_matrix,
            result_probabilities,
            lambda_a,
            lambda_b,
            scoreline_layer_weight=self.scoreline_layer_weight,
            favorite_tail_strength=self.favorite_tail_strength,
            favorite_tail_threshold=self.favorite_tail_threshold,
            reranker_strength=self.reranker_strength,
        )
        prediction = base_prediction
        prediction.update(v15.score_outputs(score_matrix, max_goals))
        prediction["top_scorelines"] = diversify_top_scorelines(
            score_matrix,
            result_probabilities,
            lambda_a,
            lambda_b,
            top_n=15,
            relative_floor=self.diversity_relative_floor,
        )
        prediction["result_probabilities"] = result_probabilities
        prediction["predicted_result"] = max(
            result_probabilities,
            key=result_probabilities.get,
        )
        prediction["v23_adjustments"] = {
            "base_model": "v15_catboost_outcome_head",
            "scoreline_policy": (
                "no_player_favorite_tail_reranker_top3_diversity"
            ),
            "player_or_squad_data_used": False,
            "scoreline_layer_affects_wdl": False,
            "top3_diversity_rule": True,
            **diagnostics,
        }
        prediction["calibration_notes"] = {
            **prediction.get("calibration_notes", {}),
            "v23": prediction["v23_adjustments"],
            "exact_score_policy": (
                "V23 uses the V15 no-player outcome head for W/D/L and lambdas, "
                "then applies a capped favorite-tail boost, a scoreline reranker, "
                "and a top-3 diversity rule. Player, squad, and FC ratings data "
                "do not affect the V23 prediction path."
            ),
        }
        return prediction

    def update_after_match(
        self,
        team_a: str,
        team_b: str,
        goals_a: int,
        goals_b: int,
    ) -> Dict[str, float]:
        update_after_match = getattr(self.outcome_model, "update_after_match", None)
        if callable(update_after_match):
            return update_after_match(team_a, team_b, goals_a, goals_b)
        return {}


def build_from_zip(
    zip_path,
    train_csv=None,
    test_csv=None,
    model_type="catboost",
    box_csv=None,
    results_csv=None,
    former_names_csv=None,
    prediction_year=2026,
    qualifier_blend_start_year=2014,
    qualifier_full_weight_year=2022,
    qualifier_minimum_influence=0.0,
    recency_half_life_years=16.0,
    recency_min_weight=0.10,
    player_ratings_csv=None,
    declared_squads_csv=None,
    results_as_of=v15.DEFAULT_RESULTS_AS_OF,
    scoreline_layer_weight=DEFAULT_SCORELINE_LAYER_WEIGHT,
    favorite_tail_strength=DEFAULT_FAVORITE_TAIL_STRENGTH,
    favorite_tail_threshold=DEFAULT_FAVORITE_TAIL_THRESHOLD,
    reranker_strength=DEFAULT_RERANKER_STRENGTH,
    diversity_relative_floor=DEFAULT_DIVERSITY_RELATIVE_FLOOR,
):
    v15.require_catboost()
    loader = v11.WorldCupSAILoader(
        zip_path,
        Path(str(zip_path) + "_extracted"),
    )
    matches = loader.load_matches()
    current = v11.load_current_team_features(train_csv, test_csv)
    box = v11.load_kaggle_box_data(box_csv)
    qualification_results = v11.load_world_cup_qualification_results(
        results_csv,
        former_names_csv,
    )
    qualifier_source = (
        qualification_results if not qualification_results.empty else box
    )
    historical_current = pd.DataFrame(columns=["team"])
    frame, features, events = v11.build_rolling_features(
        matches,
        historical_current,
        qualifier_box=qualifier_source,
        qualifier_fallback_box=box,
        qualifier_blend_start_year=qualifier_blend_start_year,
        qualifier_full_weight_year=qualifier_full_weight_year,
        qualifier_minimum_influence=qualifier_minimum_influence,
    )
    international_results = v15.load_international_results(
        results_csv,
        former_names_csv=former_names_csv,
        as_of=results_as_of,
    )
    timeline, international_state = v15.build_international_timeline(
        international_results
    )
    resolved_results_as_of = international_results.attrs.get(
        "resolved_as_of",
        str(pd.Timestamp(results_as_of).date())
        if str(results_as_of).strip().lower()
        not in {"latest", "max", "latest_non_world_cup"}
        else v15.DEFAULT_RESULTS_AS_OF,
    )
    expanded_frame, expanded_features, expansion_summary = (
        v15.build_expanded_training_frame(
            frame,
            timeline,
        )
    )
    outcome_model = (
        v15.V15CatBoostWorldCupModel(
            recency_half_life_years=recency_half_life_years,
            recency_min_weight=recency_min_weight,
        )
        .fit(expanded_frame, expanded_features, [], current)
        .set_box_data(box)
        .set_qualifier_data(
            qualifier_source,
            fallback_box=box,
            prediction_year=prediction_year,
            blend_start_year=qualifier_blend_start_year,
            full_weight_year=qualifier_full_weight_year,
            minimum_influence=qualifier_minimum_influence,
        )
        .set_current_international_state(
            international_state,
            resolved_results_as_of,
        )
    )
    data = v11.DataBundle(
        matches=matches,
        team_current=current,
        training_frame=expanded_frame,
        event_columns=events,
        box_frame=box,
    )
    model = V23NoPlayerScorelineModel(
        outcome_model,
        scoreline_layer_weight=scoreline_layer_weight,
        favorite_tail_strength=favorite_tail_strength,
        favorite_tail_threshold=favorite_tail_threshold,
        reranker_strength=reranker_strength,
        diversity_relative_floor=diversity_relative_floor,
    )
    model.training_data_summary = {
        **expansion_summary,
        "results_as_of": str(resolved_results_as_of),
        "results_as_of_requested": str(results_as_of),
        "excluded_current_world_cup_matches": int(
            international_results.attrs.get("excluded_current_world_cup_matches", 0)
        ),
        "v23_outcome_head_only": True,
        "v23_scoreline_policy": (
            "no_player_favorite_tail_reranker_top3_diversity"
        ),
        "v23_player_or_squad_data_used": False,
        "v23_scoreline_layer_weight": model.scoreline_layer_weight,
        "v23_favorite_tail_strength": model.favorite_tail_strength,
        "v23_favorite_tail_threshold": model.favorite_tail_threshold,
        "v23_reranker_strength": model.reranker_strength,
        "v23_diversity_relative_floor": model.diversity_relative_floor,
    }
    return model, data


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    parser = argparse.ArgumentParser(
        description="Run V23: no-player exact-score reranker."
    )
    parser.add_argument("--team-a", required=True)
    parser.add_argument("--team-b", required=True)
    parser.add_argument("--host-a", action="store_true")
    parser.add_argument("--host-b", action="store_true")
    parser.add_argument("--knockout", action="store_true")
    parser.add_argument(
        "--outdir",
        default="outputs/outputs_v23_no_player_scoreline_prediction",
    )
    parser.add_argument(
        "--worldcupsai-zip",
        default=str(data_dir / "worldcupsai.zip"),
    )
    parser.add_argument(
        "--team-train",
        default=str(data_dir / "current_team_features_2026.csv"),
    )
    parser.add_argument("--team-test")
    parser.add_argument(
        "--box-data",
        default=str(data_dir / "FIFAallMatchBoxData.csv"),
    )
    parser.add_argument(
        "--results-data",
        default=str(data_dir / "results.csv"),
    )
    parser.add_argument("--results-as-of", default=v15.DEFAULT_RESULTS_AS_OF)
    parser.add_argument(
        "--former-names",
        default=str(data_dir / "former_names.csv"),
    )
    parser.add_argument("--prediction-year", type=int, default=2026)
    parser.add_argument(
        "--scoreline-layer-weight",
        type=float,
        default=DEFAULT_SCORELINE_LAYER_WEIGHT,
    )
    parser.add_argument(
        "--favorite-tail-strength",
        type=float,
        default=DEFAULT_FAVORITE_TAIL_STRENGTH,
    )
    parser.add_argument(
        "--favorite-tail-threshold",
        type=float,
        default=DEFAULT_FAVORITE_TAIL_THRESHOLD,
    )
    parser.add_argument(
        "--reranker-strength",
        type=float,
        default=DEFAULT_RERANKER_STRENGTH,
    )
    parser.add_argument(
        "--diversity-relative-floor",
        type=float,
        default=DEFAULT_DIVERSITY_RELATIVE_FLOOR,
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    output_dir = v11.unique_output_dir(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, _ = build_from_zip(
        args.worldcupsai_zip,
        train_csv=args.team_train,
        test_csv=args.team_test,
        box_csv=args.box_data,
        results_csv=args.results_data,
        former_names_csv=args.former_names,
        prediction_year=args.prediction_year,
        results_as_of=args.results_as_of,
        scoreline_layer_weight=args.scoreline_layer_weight,
        favorite_tail_strength=args.favorite_tail_strength,
        favorite_tail_threshold=args.favorite_tail_threshold,
        reranker_strength=args.reranker_strength,
        diversity_relative_floor=args.diversity_relative_floor,
    )
    prediction = model.predict(
        args.team_a,
        args.team_b,
        host_a=args.host_a,
        host_b=args.host_b,
        knockout=args.knockout,
    )
    (output_dir / "single_match_prediction.json").write_text(
        json.dumps(prediction, indent=2)
    )
    pd.DataFrame(prediction["top_scorelines"]).to_csv(
        output_dir / "scoreline_probabilities_top.csv",
        index=False,
    )
    pd.DataFrame(prediction["scoreline_probabilities"]).to_csv(
        output_dir / "scoreline_probabilities.csv",
        index=False,
    )
    (output_dir / "model_summary.json").write_text(
        json.dumps(
            {
                "version": "v23-no-player-scoreline",
                "base_model": "v15-catboost-outcome-head",
                "wdl_model": "v15_catboost_preserved_no_player_head",
                "exact_score_model": (
                    "favorite_tail_reranker_top3_diversity_no_player"
                ),
                "team_a": prediction["team_a"],
                "team_b": prediction["team_b"],
                "v23_adjustments": prediction["v23_adjustments"],
                "expanded_training_data": model.training_data_summary,
            },
            indent=2,
        )
    )
    if not args.no_plots:
        v11.plot_prediction_outputs(prediction, output_dir)
    print(
        json.dumps(
            {
                "result_probabilities": prediction["result_probabilities"],
                "predicted_result": prediction["predicted_result"],
                "lambda_a": prediction["lambda_a"],
                "lambda_b": prediction["lambda_b"],
                "top_scorelines": prediction["top_scorelines"][:5],
                "v23_adjustments": {
                    "scoreline_layer_weight": prediction["v23_adjustments"][
                        "scoreline_layer_weight"
                    ],
                    "favorite_tail_strength": prediction["v23_adjustments"][
                        "favorite_tail_strength"
                    ],
                    "reranker_strength": prediction["v23_adjustments"][
                        "reranker_strength"
                    ],
                    "player_or_squad_data_used": prediction["v23_adjustments"][
                        "player_or_squad_data_used"
                    ],
                },
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
'''
v23_no_player_scoreline_model = _load_submodule("v23_no_player_scoreline_model", _V23_NO_PLAYER_SCORELINE_MODEL_SOURCE, "core_engine.py:v23_no_player_scoreline_model")

# ======================================================================
# v20_scoreline_ensemble_model.py  (bundled as an isolated sub-module)
# ======================================================================
_V20_SCORELINE_ENSEMBLE_MODEL_SOURCE = r'''
#!/usr/bin/env python3
"""V20: V15 W/D/L with blended V15 and V18 scoreline-only exact scores."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

import v11_wcq_results_model as v11
import v15_catboost_model as v15
import v18_hybrid_elo_form_player_model as v18


canon_team = v11.canon_team
ScoreMatrix = Dict[Tuple[int, int], float]

DEFAULT_V18_SCORELINE_WEIGHT = 0.35


def score_matrix_from_prediction(prediction: Dict[str, Any]) -> ScoreMatrix:
    return {
        (int(item["team_a_goals"]), int(item["team_b_goals"])): float(
            item["probability"]
        )
        for item in prediction["scoreline_probabilities"]
    }


def normalize_matrix(score_matrix: ScoreMatrix) -> ScoreMatrix:
    total = float(sum(score_matrix.values()))
    if total <= 0:
        raise ValueError("score matrix has no probability mass")
    return {key: float(value) / total for key, value in score_matrix.items()}


def blend_score_matrices(
    base_matrix: ScoreMatrix,
    adjusted_matrix: ScoreMatrix,
    adjusted_weight: float,
) -> ScoreMatrix:
    weight = float(np.clip(adjusted_weight, 0.0, 1.0))
    keys = set(base_matrix) | set(adjusted_matrix)
    blended = {
        key: (1.0 - weight) * base_matrix.get(key, 0.0)
        + weight * adjusted_matrix.get(key, 0.0)
        for key in keys
    }
    return normalize_matrix(blended)


class V20ScorelineEnsembleModel:
    """Preserve V15 W/D/L while blending V15 and V18 exact-score matrices."""

    def __init__(
        self,
        base_model: v15.V15CatBoostModel,
        squad_profiles: Dict[str, Dict[str, Any]],
        v18_scoreline_weight: float = DEFAULT_V18_SCORELINE_WEIGHT,
        beta_attack: float = v18.DEFAULT_BETA_ATTACK,
        beta_midfield: float = v18.DEFAULT_BETA_MIDFIELD,
        beta_keeper: float = v18.DEFAULT_BETA_KEEPER,
        max_log_adjustment: float = v18.DEFAULT_MAX_LOG_ADJUSTMENT,
    ):
        self.base_model = base_model
        self.squad_profiles = squad_profiles
        self.v18_scoreline_weight = float(np.clip(v18_scoreline_weight, 0.0, 1.0))
        self.v18_scoreline_model = v18.V18HybridSquadModel(
            base_model,
            squad_profiles,
            beta_attack=beta_attack,
            beta_midfield=beta_midfield,
            beta_keeper=beta_keeper,
            max_log_adjustment=max_log_adjustment,
            player_ratings_affect_wdl=False,
        )
        self.training_data_summary = getattr(
            base_model,
            "training_data_summary",
            {},
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_model, name)

    def _v18_scoreline_prediction(
        self,
        base_prediction: Dict[str, Any],
        team_a: str,
        team_b: str,
        max_goals: int,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        prediction = copy.deepcopy(base_prediction)
        base_lambda_a = float(prediction["lambda_a"])
        base_lambda_b = float(prediction["lambda_b"])
        base_results = dict(prediction["result_probabilities"])

        profile_a = self.v18_scoreline_model.profile_for_team(str(team_a))
        profile_b = self.v18_scoreline_model.profile_for_team(str(team_b))
        log_a, log_b, adjustment_details = self.v18_scoreline_model._log_adjustments(
            profile_a,
            profile_b,
        )
        lambda_a = float(np.clip(base_lambda_a * math.exp(log_a), 0.15, 4.5))
        lambda_b = float(np.clip(base_lambda_b * math.exp(log_b), 0.15, 4.5))
        score_matrix = v11.poisson_score_matrix(lambda_a, lambda_b, max_goals)
        rho = prediction.get("calibration_notes", {}).get(
            "dixon_coles_rho",
            -0.08,
        )
        score_matrix = v11.apply_dixon_coles_adjustment(
            score_matrix,
            lambda_a,
            lambda_b,
            rho=rho,
        )
        adjusted_poisson_results = v11.result_probs(score_matrix)
        score_matrix = v11.reweight_score_matrix_to_results(
            score_matrix,
            base_results,
        )

        prediction["lambda_a"] = lambda_a
        prediction["lambda_b"] = lambda_b
        prediction.update(v15.score_outputs(score_matrix, max_goals))
        prediction["result_probabilities"] = base_results
        diagnostics = {
            "v18_base_lambda_a": base_lambda_a,
            "v18_base_lambda_b": base_lambda_b,
            "v18_lambda_a": lambda_a,
            "v18_lambda_b": lambda_b,
            "v18_adjusted_poisson_result_probabilities": adjusted_poisson_results,
            "v18_player_ratings_affect_wdl": False,
            **adjustment_details,
        }
        return prediction, diagnostics

    def predict(self, *args, **kwargs) -> Dict[str, Any]:
        team_a = kwargs.get("team_a", args[0] if args else "")
        team_b = kwargs.get("team_b", args[1] if len(args) > 1 else "")
        max_goals = int(kwargs.get("max_goals", args[5] if len(args) > 5 else 10))

        v15_prediction = self.base_model.predict(*args, **kwargs)
        v18_prediction, v18_diagnostics = self._v18_scoreline_prediction(
            v15_prediction,
            str(team_a),
            str(team_b),
            max_goals,
        )
        v15_matrix = score_matrix_from_prediction(v15_prediction)
        v18_matrix = score_matrix_from_prediction(v18_prediction)
        blended_matrix = blend_score_matrices(
            v15_matrix,
            v18_matrix,
            self.v18_scoreline_weight,
        )

        result_probabilities = dict(v15_prediction["result_probabilities"])
        prediction = v15_prediction
        blended_lambda_a = (
            (1.0 - self.v18_scoreline_weight) * float(v15_prediction["lambda_a"])
            + self.v18_scoreline_weight * float(v18_prediction["lambda_a"])
        )
        blended_lambda_b = (
            (1.0 - self.v18_scoreline_weight) * float(v15_prediction["lambda_b"])
            + self.v18_scoreline_weight * float(v18_prediction["lambda_b"])
        )
        prediction["lambda_a"] = float(blended_lambda_a)
        prediction["lambda_b"] = float(blended_lambda_b)
        prediction.update(v15.score_outputs(blended_matrix, max_goals))
        prediction["result_probabilities"] = result_probabilities
        prediction["predicted_result"] = max(
            result_probabilities,
            key=result_probabilities.get,
        )
        prediction["v20_adjustments"] = {
            "base_model": "v15_catboost",
            "scoreline_policy": "linear_blend_v15_v18_scoreline_only",
            "v18_scoreline_weight": self.v18_scoreline_weight,
            "v15_scoreline_weight": 1.0 - self.v18_scoreline_weight,
            "scoreline_blend_affects_wdl": False,
            "rank_stabilizer": False,
            "v15_result_probabilities": result_probabilities,
            "v15_lambda_a": float(v15_prediction["lambda_a"]),
            "v15_lambda_b": float(v15_prediction["lambda_b"]),
            **v18_diagnostics,
        }
        prediction["calibration_notes"] = {
            **prediction.get("calibration_notes", {}),
            "v20": prediction["v20_adjustments"],
            "exact_score_policy": (
                "V20 preserves V15 W/D/L probabilities and blends the V15 "
                "exact-score matrix with the V18 scoreline-only matrix. No "
                "rank stabilizer or scoreline ordering override is applied."
            ),
        }
        return prediction

    def update_after_match(
        self,
        team_a: str,
        team_b: str,
        goals_a: int,
        goals_b: int,
    ) -> Dict[str, float]:
        return self.base_model.update_after_match(
            team_a,
            team_b,
            goals_a,
            goals_b,
        )


def build_from_zip(
    zip_path,
    train_csv=None,
    test_csv=None,
    model_type="catboost",
    box_csv=None,
    results_csv=None,
    former_names_csv=None,
    prediction_year=2026,
    qualifier_blend_start_year=2014,
    qualifier_full_weight_year=2022,
    qualifier_minimum_influence=0.0,
    recency_half_life_years=16.0,
    recency_min_weight=0.10,
    player_ratings_csv=None,
    declared_squads_csv=None,
    fcratings_csv=None,
    results_as_of=v15.DEFAULT_RESULTS_AS_OF,
    beta_attack=v18.DEFAULT_BETA_ATTACK,
    beta_midfield=v18.DEFAULT_BETA_MIDFIELD,
    beta_keeper=v18.DEFAULT_BETA_KEEPER,
    max_log_adjustment=v18.DEFAULT_MAX_LOG_ADJUSTMENT,
    match_threshold=0.84,
    v18_scoreline_weight=DEFAULT_V18_SCORELINE_WEIGHT,
):
    data_dir = Path(__file__).resolve().parent / "data"
    player_ratings_csv = player_ratings_csv or (
        data_dir / "player_ratings_international.csv"
    )
    fcratings_csv = fcratings_csv or (data_dir / "fcratings_top50_worldcup2026.csv")
    base_model, data = v15.build_from_zip(
        zip_path,
        train_csv=train_csv,
        test_csv=test_csv,
        model_type=model_type,
        box_csv=box_csv,
        results_csv=results_csv,
        former_names_csv=former_names_csv,
        prediction_year=prediction_year,
        qualifier_blend_start_year=qualifier_blend_start_year,
        qualifier_full_weight_year=qualifier_full_weight_year,
        qualifier_minimum_influence=qualifier_minimum_influence,
        recency_half_life_years=recency_half_life_years,
        recency_min_weight=recency_min_weight,
        player_ratings_csv=player_ratings_csv,
        declared_squads_csv=declared_squads_csv,
        results_as_of=results_as_of,
    )
    squad_names = v18.load_current_squad_names(player_ratings_csv)
    fcratings = v18.load_fcratings_players(fcratings_csv)
    squad_profiles = v18.build_current_fcratings_squad_profiles(
        squad_names,
        fcratings,
        match_threshold=match_threshold,
    )
    model = V20ScorelineEnsembleModel(
        base_model,
        squad_profiles,
        v18_scoreline_weight=v18_scoreline_weight,
        beta_attack=beta_attack,
        beta_midfield=beta_midfield,
        beta_keeper=beta_keeper,
        max_log_adjustment=max_log_adjustment,
    )
    model.training_data_summary = {
        **getattr(base_model, "training_data_summary", {}),
        "v20_scoreline_policy": "linear_blend_v15_v18_scoreline_only",
        "v20_v18_scoreline_weight": model.v18_scoreline_weight,
        "v20_rank_stabilizer": False,
        "v20_squad_profile_teams": len(squad_profiles),
        "v20_fcratings_rows": int(len(fcratings)),
        "v20_squad_name_rows": int(len(squad_names)),
    }
    return model, data


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    parser = argparse.ArgumentParser(
        description="Run V20: blended V15/V18 scoreline ensemble."
    )
    parser.add_argument("--team-a", required=True)
    parser.add_argument("--team-b", required=True)
    parser.add_argument("--host-a", action="store_true")
    parser.add_argument("--host-b", action="store_true")
    parser.add_argument("--knockout", action="store_true")
    parser.add_argument(
        "--outdir",
        default="outputs/outputs_v20_scoreline_ensemble_prediction",
    )
    parser.add_argument(
        "--worldcupsai-zip",
        default=str(data_dir / "worldcupsai.zip"),
    )
    parser.add_argument(
        "--team-train",
        default=str(data_dir / "current_team_features_2026.csv"),
    )
    parser.add_argument("--team-test")
    parser.add_argument(
        "--box-data",
        default=str(data_dir / "FIFAallMatchBoxData.csv"),
    )
    parser.add_argument(
        "--results-data",
        default=str(data_dir / "results.csv"),
    )
    parser.add_argument("--results-as-of", default=v15.DEFAULT_RESULTS_AS_OF)
    parser.add_argument(
        "--former-names",
        default=str(data_dir / "former_names.csv"),
    )
    parser.add_argument("--prediction-year", type=int, default=2026)
    parser.add_argument(
        "--player-ratings",
        default=str(data_dir / "player_ratings_international.csv"),
    )
    parser.add_argument(
        "--declared-squads",
        default=str(data_dir / "world_cup_2026_declared_squads.csv"),
    )
    parser.add_argument(
        "--fcratings",
        default=str(data_dir / "fcratings_top50_worldcup2026.csv"),
    )
    parser.add_argument(
        "--v18-scoreline-weight",
        type=float,
        default=DEFAULT_V18_SCORELINE_WEIGHT,
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    output_dir = v11.unique_output_dir(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, _ = build_from_zip(
        args.worldcupsai_zip,
        train_csv=args.team_train,
        test_csv=args.team_test,
        box_csv=args.box_data,
        results_csv=args.results_data,
        former_names_csv=args.former_names,
        prediction_year=args.prediction_year,
        player_ratings_csv=args.player_ratings,
        declared_squads_csv=args.declared_squads,
        fcratings_csv=args.fcratings,
        results_as_of=args.results_as_of,
        v18_scoreline_weight=args.v18_scoreline_weight,
    )
    prediction = model.predict(
        args.team_a,
        args.team_b,
        host_a=args.host_a,
        host_b=args.host_b,
        knockout=args.knockout,
    )
    (output_dir / "single_match_prediction.json").write_text(
        json.dumps(prediction, indent=2)
    )
    pd.DataFrame(prediction["top_scorelines"]).to_csv(
        output_dir / "scoreline_probabilities_top.csv",
        index=False,
    )
    pd.DataFrame(prediction["scoreline_probabilities"]).to_csv(
        output_dir / "scoreline_probabilities.csv",
        index=False,
    )
    (output_dir / "model_summary.json").write_text(
        json.dumps(
            {
                "version": "v20-scoreline-ensemble",
                "base_model": "v15-catboost",
                "wdl_model": "v15_catboost_preserved",
                "exact_score_model": "linear_blend_v15_v18_scoreline_only",
                "team_a": prediction["team_a"],
                "team_b": prediction["team_b"],
                "v20_adjustments": prediction["v20_adjustments"],
                "expanded_training_data": model.training_data_summary,
            },
            indent=2,
        )
    )
    if not args.no_plots:
        v11.plot_prediction_outputs(prediction, output_dir)
    print(
        json.dumps(
            {
                "result_probabilities": prediction["result_probabilities"],
                "predicted_result": prediction["predicted_result"],
                "lambda_a": prediction["lambda_a"],
                "lambda_b": prediction["lambda_b"],
                "top_scorelines": prediction["top_scorelines"][:5],
                "v20_adjustments": {
                    "v18_scoreline_weight": prediction["v20_adjustments"][
                        "v18_scoreline_weight"
                    ],
                    "rank_stabilizer": prediction["v20_adjustments"][
                        "rank_stabilizer"
                    ],
                },
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
'''
v20_scoreline_ensemble_model = _load_submodule("v20_scoreline_ensemble_model", _V20_SCORELINE_ENSEMBLE_MODEL_SOURCE, "core_engine.py:v20_scoreline_ensemble_model")

# ======================================================================
# v24_scoreline_reranker_model.py  (bundled as an isolated sub-module)
# ======================================================================
_V24_SCORELINE_RERANKER_MODEL_SOURCE = r'''
#!/usr/bin/env python3
"""V24: V23 no-player model with a supervised exact-score reranker."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

import v11_wcq_results_model as v11
import v15_catboost_model as v15
import v23_no_player_scoreline_model as v23


ScoreMatrix = Dict[Tuple[int, int], float]

DEFAULT_CANDIDATE_POOL_SIZE = 12
DEFAULT_MAX_RERANKER_TRAIN_MATCHES = 1400
DEFAULT_RERANKER_BLEND = 0.35
DEFAULT_RERANKER_POWER = 0.65
DEFAULT_RERANKER_MODEL = "hgb"

RERANKER_FEATURES = [
    "base_probability",
    "log_base_probability",
    "base_rank",
    "rank_inverse",
    "goals_a",
    "goals_b",
    "total_goals",
    "margin",
    "abs_margin",
    "is_draw_score",
    "is_team_a_win_score",
    "is_team_b_win_score",
    "is_low_score",
    "is_clean_sheet",
    "lambda_a",
    "lambda_b",
    "lambda_total",
    "lambda_diff",
    "abs_lambda_diff",
    "candidate_lambda_error",
    "candidate_total_error",
    "team_a_win_probability",
    "draw_probability",
    "team_b_win_probability",
    "max_result_probability",
    "favorite_probability",
    "underdog_probability",
    "candidate_result_probability",
    "score_matches_predicted_result",
    "favorite_scoreline",
    "upset_scoreline",
    "is_group_stage",
    "is_knockout",
    "host_a",
    "host_b",
    "same_confed",
]


def outcome_label(goals_a: int, goals_b: int) -> str:
    if goals_a > goals_b:
        return "team_a_win"
    if goals_a < goals_b:
        return "team_b_win"
    return "draw"


def normalize_matrix(matrix: ScoreMatrix) -> ScoreMatrix:
    total = float(sum(matrix.values()))
    if total <= 0:
        raise ValueError("score matrix has no probability mass")
    return {key: float(value) / total for key, value in matrix.items()}


def score_matrix_from_prediction(prediction: Dict[str, Any]) -> ScoreMatrix:
    return {
        (int(item["team_a_goals"]), int(item["team_b_goals"])): float(
            item["probability"]
        )
        for item in prediction["scoreline_probabilities"]
    }


def sorted_matrix_items(matrix: ScoreMatrix) -> list[tuple[Tuple[int, int], float]]:
    return sorted(matrix.items(), key=lambda item: item[1], reverse=True)


def candidate_keys(
    matrix: ScoreMatrix,
    pool_size: int,
    actual_score: Tuple[int, int] | None = None,
) -> list[Tuple[int, int]]:
    keys = [key for key, _ in sorted_matrix_items(matrix)[: max(pool_size, 1)]]
    if actual_score is not None and actual_score in matrix and actual_score not in keys:
        keys.append(actual_score)
    return keys


def model_prediction_from_feature_row(
    model: v11.StrongWorldCupModel,
    row: pd.Series,
    max_goals: int,
) -> Dict[str, Any]:
    """Predict from an already-built historical feature row.

    This avoids using current/live team state while constructing reranker
    training candidates.
    """
    X = pd.DataFrame([{column: row.get(column, np.nan) for column in model.feature_cols}])
    if model.model_type == "ensemble":
        raw_lam_a = model._weighted_regression_prediction(model.goal_a_models, X)
        raw_lam_b = model._weighted_regression_prediction(model.goal_b_models, X)
        diff_pred = model._weighted_regression_prediction(model.goal_diff_models, X)
    else:
        raw_lam_a = float(model.goal_a.predict(X)[0])
        raw_lam_b = float(model.goal_b.predict(X)[0])
        diff_pred = float(model.goal_diff_model.predict(X)[0])

    raw_lam_a = max(float(raw_lam_a), 0.001)
    raw_lam_b = max(float(raw_lam_b), 0.001)
    blended_a, blended_b = model._apply_goal_difference_blend(
        raw_lam_a,
        raw_lam_b,
        diff_pred,
    )
    lambda_a = float(np.clip(blended_a, 0.15, 4.5))
    lambda_b = float(np.clip(blended_b, 0.15, 4.5))

    matrix = v11.poisson_score_matrix(lambda_a, lambda_b, max_goals)
    matrix = v11.apply_dixon_coles_adjustment(
        matrix,
        lambda_a,
        lambda_b,
        rho=model.dixon_coles_rho,
    )
    result_probabilities = v11.result_probs(matrix)

    if model.model_type == "ensemble":
        cls_res = model._weighted_classification_prediction(model.result_models, X)
        if sum(cls_res.values()) > 0:
            result_probabilities = {
                key: 0.86 * result_probabilities[key] + 0.14 * cls_res[key]
                for key in result_probabilities
            }
            total = sum(result_probabilities.values())
            result_probabilities = {
                key: value / total for key, value in result_probabilities.items()
            }
    elif hasattr(model.result_model, "predict_proba"):
        class_probs = model.result_model.predict_proba(X)[0]
        classes = (
            list(model.result_model.classes_)
            if hasattr(model.result_model, "classes_")
            else [0, 1, 2]
        )
        class_map = {int(label): float(prob) for label, prob in zip(classes, class_probs)}
        cls_res = {
            "team_a_win": class_map.get(2, 0.0),
            "draw": class_map.get(1, 0.0),
            "team_b_win": class_map.get(0, 0.0),
        }
        result_probabilities = {
            key: 0.84 * result_probabilities[key] + 0.16 * cls_res[key]
            for key in result_probabilities
        }
        total = sum(result_probabilities.values())
        result_probabilities = {
            key: value / total for key, value in result_probabilities.items()
        }

    result_probabilities = v11.temperature_smooth_result_probs(
        result_probabilities,
        model.temperature,
    )
    draw_model_probability = model._predict_draw_probability(X)
    draw_probability = (
        model.draw_model_weight * draw_model_probability
        + (1.0 - model.draw_model_weight) * result_probabilities["draw"]
    )
    draw_probability = float(np.clip(draw_probability, 0.05, 0.55))
    non_draw_total = max(
        result_probabilities["team_a_win"] + result_probabilities["team_b_win"],
        1e-12,
    )
    final_results = {
        "team_a_win": (1.0 - draw_probability)
        * result_probabilities["team_a_win"]
        / non_draw_total,
        "draw": draw_probability,
        "team_b_win": (1.0 - draw_probability)
        * result_probabilities["team_b_win"]
        / non_draw_total,
    }
    matrix = v11.reweight_score_matrix_to_results(matrix, final_results)
    final_results = v11.result_probs(matrix)
    return {
        "team_a": row.get("team_a", ""),
        "team_b": row.get("team_b", ""),
        "lambda_a": lambda_a,
        "lambda_b": lambda_b,
        "result_probabilities": final_results,
        **v15.score_outputs(matrix, max_goals),
    }


def _weighted_regression_predictions(models, X: pd.DataFrame) -> np.ndarray:
    predictions = []
    weights = []
    for _, estimator, weight in models:
        values = np.asarray(estimator.predict(X), dtype=float)
        predictions.append(values)
        weights.append(float(weight))
    if not predictions:
        return np.full(len(X), 1.25, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    weight_array = weight_array / max(float(weight_array.sum()), 1e-12)
    return np.average(np.vstack(predictions), axis=0, weights=weight_array)


def _weighted_classification_predictions(models, X: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(
        0.0,
        index=X.index,
        columns=["team_a_win", "draw", "team_b_win"],
    )
    total_weight = 0.0
    for _, estimator, weight in models:
        if not hasattr(estimator, "predict_proba"):
            continue
        probabilities = np.asarray(estimator.predict_proba(X), dtype=float)
        classes = list(estimator.classes_) if hasattr(estimator, "classes_") else [0, 1, 2]
        class_positions = {int(label): index for index, label in enumerate(classes)}
        output["team_a_win"] += float(weight) * probabilities[:, class_positions.get(2, 0)]
        output["draw"] += float(weight) * probabilities[:, class_positions.get(1, 0)]
        output["team_b_win"] += float(weight) * probabilities[:, class_positions.get(0, 0)]
        total_weight += float(weight)
    if total_weight <= 0:
        return output
    output = output / total_weight
    totals = output.sum(axis=1).replace(0.0, np.nan)
    return output.div(totals, axis=0).fillna(0.0)


def _draw_probabilities(model: v11.StrongWorldCupModel, frame: pd.DataFrame) -> np.ndarray:
    if model.draw_model is None or not model.draw_feature_cols:
        return np.full(len(frame), 0.20, dtype=float)
    raw = np.asarray(
        model.draw_model.predict_proba(frame[model.draw_feature_cols])[:, 1],
        dtype=float,
    )
    if model.draw_calibrator is None:
        return raw
    logits = np.log(
        np.clip(raw, 1e-6, 1 - 1e-6)
        / np.clip(1.0 - raw, 1e-6, 1 - 1e-6)
    ).reshape(-1, 1)
    return np.asarray(model.draw_calibrator.predict_proba(logits)[:, 1], dtype=float)


def model_predictions_from_feature_frame(
    model: v11.StrongWorldCupModel,
    frame: pd.DataFrame,
    max_goals: int,
) -> list[Dict[str, Any]]:
    """Vectorized historical base predictions for reranker training."""
    feature_frame = frame.reset_index(drop=True).copy()
    X = feature_frame[model.feature_cols]
    if model.model_type == "ensemble":
        raw_lam_a = _weighted_regression_predictions(model.goal_a_models, X)
        raw_lam_b = _weighted_regression_predictions(model.goal_b_models, X)
        diff_pred = _weighted_regression_predictions(model.goal_diff_models, X)
        cls_res = _weighted_classification_predictions(model.result_models, X)
    else:
        raw_lam_a = np.asarray(model.goal_a.predict(X), dtype=float)
        raw_lam_b = np.asarray(model.goal_b.predict(X), dtype=float)
        diff_pred = np.asarray(model.goal_diff_model.predict(X), dtype=float)
        cls_res = pd.DataFrame(
            0.0,
            index=X.index,
            columns=["team_a_win", "draw", "team_b_win"],
        )
        if hasattr(model.result_model, "predict_proba"):
            probabilities = np.asarray(model.result_model.predict_proba(X), dtype=float)
            classes = (
                list(model.result_model.classes_)
                if hasattr(model.result_model, "classes_")
                else [0, 1, 2]
            )
            class_positions = {int(label): index for index, label in enumerate(classes)}
            cls_res["team_a_win"] = probabilities[:, class_positions.get(2, 0)]
            cls_res["draw"] = probabilities[:, class_positions.get(1, 0)]
            cls_res["team_b_win"] = probabilities[:, class_positions.get(0, 0)]

    raw_lam_a = np.maximum(raw_lam_a, 0.001)
    raw_lam_b = np.maximum(raw_lam_b, 0.001)
    draw_model_probs = _draw_probabilities(model, X)
    predictions: list[Dict[str, Any]] = []
    for index, row in feature_frame.iterrows():
        blended_a, blended_b = model._apply_goal_difference_blend(
            float(raw_lam_a[index]),
            float(raw_lam_b[index]),
            float(diff_pred[index]),
        )
        lambda_a = float(np.clip(blended_a, 0.15, 4.5))
        lambda_b = float(np.clip(blended_b, 0.15, 4.5))
        matrix = v11.poisson_score_matrix(lambda_a, lambda_b, max_goals)
        matrix = v11.apply_dixon_coles_adjustment(
            matrix,
            lambda_a,
            lambda_b,
            rho=model.dixon_coles_rho,
        )
        result_probabilities = v11.result_probs(matrix)
        if model.model_type == "ensemble":
            classifier_weight = 0.14
        else:
            classifier_weight = 0.16
        if float(cls_res.iloc[index].sum()) > 0:
            result_probabilities = {
                key: (1.0 - classifier_weight) * result_probabilities[key]
                + classifier_weight * float(cls_res.iloc[index][key])
                for key in result_probabilities
            }
            total = sum(result_probabilities.values())
            result_probabilities = {
                key: value / total for key, value in result_probabilities.items()
            }
        result_probabilities = v11.temperature_smooth_result_probs(
            result_probabilities,
            model.temperature,
        )
        draw_probability = (
            model.draw_model_weight * float(draw_model_probs[index])
            + (1.0 - model.draw_model_weight) * result_probabilities["draw"]
        )
        draw_probability = float(np.clip(draw_probability, 0.05, 0.55))
        non_draw_total = max(
            result_probabilities["team_a_win"] + result_probabilities["team_b_win"],
            1e-12,
        )
        final_results = {
            "team_a_win": (1.0 - draw_probability)
            * result_probabilities["team_a_win"]
            / non_draw_total,
            "draw": draw_probability,
            "team_b_win": (1.0 - draw_probability)
            * result_probabilities["team_b_win"]
            / non_draw_total,
        }
        matrix = v11.reweight_score_matrix_to_results(matrix, final_results)
        predictions.append(
            {
                "team_a": row.get("team_a", ""),
                "team_b": row.get("team_b", ""),
                "lambda_a": lambda_a,
                "lambda_b": lambda_b,
                "result_probabilities": v11.result_probs(matrix),
                **v15.score_outputs(matrix, max_goals),
            }
        )
    return predictions


def candidate_feature_row(
    key: Tuple[int, int],
    probability: float,
    rank: int,
    lambda_a: float,
    lambda_b: float,
    result_probabilities: Dict[str, float],
    context: Dict[str, Any] | None = None,
) -> Dict[str, float]:
    context = context or {}
    goals_a, goals_b = int(key[0]), int(key[1])
    total_goals = goals_a + goals_b
    margin = goals_a - goals_b
    label = outcome_label(goals_a, goals_b)
    predicted_result = max(result_probabilities, key=result_probabilities.get)
    favorite_result = (
        "team_a_win"
        if result_probabilities["team_a_win"] >= result_probabilities["team_b_win"]
        else "team_b_win"
    )
    underdog_result = "team_b_win" if favorite_result == "team_a_win" else "team_a_win"
    favorite_probability = float(result_probabilities[favorite_result])
    underdog_probability = float(result_probabilities[underdog_result])
    favorite_scoreline = (
        label == favorite_result
        and favorite_probability >= result_probabilities["draw"]
    )
    return {
        "base_probability": float(probability),
        "log_base_probability": math.log(max(float(probability), 1e-12)),
        "base_rank": float(rank),
        "rank_inverse": 1.0 / float(rank),
        "goals_a": float(goals_a),
        "goals_b": float(goals_b),
        "total_goals": float(total_goals),
        "margin": float(margin),
        "abs_margin": float(abs(margin)),
        "is_draw_score": float(goals_a == goals_b),
        "is_team_a_win_score": float(goals_a > goals_b),
        "is_team_b_win_score": float(goals_a < goals_b),
        "is_low_score": float(total_goals <= 2),
        "is_clean_sheet": float(goals_a == 0 or goals_b == 0),
        "lambda_a": float(lambda_a),
        "lambda_b": float(lambda_b),
        "lambda_total": float(lambda_a + lambda_b),
        "lambda_diff": float(lambda_a - lambda_b),
        "abs_lambda_diff": float(abs(lambda_a - lambda_b)),
        "candidate_lambda_error": float(
            abs(goals_a - lambda_a) + abs(goals_b - lambda_b)
        ),
        "candidate_total_error": float(abs(total_goals - (lambda_a + lambda_b))),
        "team_a_win_probability": float(result_probabilities["team_a_win"]),
        "draw_probability": float(result_probabilities["draw"]),
        "team_b_win_probability": float(result_probabilities["team_b_win"]),
        "max_result_probability": float(max(result_probabilities.values())),
        "favorite_probability": favorite_probability,
        "underdog_probability": underdog_probability,
        "candidate_result_probability": float(result_probabilities[label]),
        "score_matches_predicted_result": float(label == predicted_result),
        "favorite_scoreline": float(favorite_scoreline),
        "upset_scoreline": float(label == underdog_result),
        "is_group_stage": float(context.get("is_group_stage", 0.0)),
        "is_knockout": float(context.get("is_knockout", 0.0)),
        "host_a": float(context.get("host_a", 0.0)),
        "host_b": float(context.get("host_b", 0.0)),
        "same_confed": float(context.get("same_confed", 0.0)),
    }


def candidate_feature_frame(
    matrix: ScoreMatrix,
    prediction: Dict[str, Any],
    keys: Iterable[Tuple[int, int]],
    context: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    ranks = {key: rank for rank, (key, _) in enumerate(sorted_matrix_items(matrix), start=1)}
    rows = [
        candidate_feature_row(
            key,
            matrix.get(key, 0.0),
            ranks.get(key, len(ranks) + 1),
            float(prediction["lambda_a"]),
            float(prediction["lambda_b"]),
            prediction["result_probabilities"],
            context=context,
        )
        for key in keys
    ]
    return pd.DataFrame(rows, columns=RERANKER_FEATURES)


def build_reranker_estimator(model_type: str):
    if model_type == "logistic":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.8,
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=24,
                    ),
                ),
            ]
        )
    return HistGradientBoostingClassifier(
        max_iter=160,
        learning_rate=0.045,
        max_leaf_nodes=15,
        l2_regularization=0.15,
        random_state=24,
    )


def predict_positive_probability(estimator: Any, features: pd.DataFrame) -> np.ndarray:
    if estimator is None or features.empty:
        return np.zeros(len(features), dtype=float)
    probabilities = estimator.predict_proba(features[RERANKER_FEATURES])
    classes = list(estimator.classes_) if hasattr(estimator, "classes_") else [0, 1]
    if 1 not in classes:
        return np.zeros(len(features), dtype=float)
    return probabilities[:, classes.index(1)]


def build_reranker_training_data(
    base_model: v23.V23NoPlayerScorelineModel,
    max_goals: int,
    candidate_pool_size: int,
    max_train_matches: int,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray, Dict[str, Any]]:
    outcome_model = base_model.outcome_model
    train_frame = getattr(outcome_model, "train_frame", pd.DataFrame())
    if train_frame is None or train_frame.empty:
        return pd.DataFrame(columns=RERANKER_FEATURES), pd.Series(dtype=int), np.array([]), {
            "training_rows": 0,
            "candidate_rows": 0,
            "positive_rows": 0,
        }

    recency_weights = v11.build_year_recency_weights(
        train_frame,
        outcome_model.recency_half_life_years,
        outcome_model.recency_min_weight,
    )
    sample_weights = v11.combine_training_weights(train_frame, recency_weights)
    ordered = train_frame.reset_index(drop=True)
    if len(ordered) > max_train_matches > 0:
        rng = np.random.default_rng(24)
        probabilities = sample_weights.to_numpy(dtype=float)
        probabilities = probabilities / max(float(probabilities.sum()), 1e-12)
        chosen = rng.choice(
            np.arange(len(ordered)),
            size=int(max_train_matches),
            replace=False,
            p=probabilities,
        )
        ordered = ordered.iloc[np.sort(chosen)].reset_index(drop=True)
        sample_weights = sample_weights.iloc[np.sort(chosen)].reset_index(drop=True)
    else:
        sample_weights = sample_weights.reset_index(drop=True)

    base_predictions = model_predictions_from_feature_frame(
        outcome_model,
        ordered,
        max_goals=max_goals,
    )
    feature_rows = []
    labels = []
    weights = []
    used_matches = 0
    for index, match in ordered.iterrows():
        if pd.isna(match.get("goals_a")) or pd.isna(match.get("goals_b")):
            continue
        base_prediction = base_predictions[index]
        base_matrix = score_matrix_from_prediction(base_prediction)
        score_matrix, _ = v23.postprocess_score_matrix(
            base_matrix,
            base_prediction["result_probabilities"],
            float(base_prediction["lambda_a"]),
            float(base_prediction["lambda_b"]),
            scoreline_layer_weight=base_model.scoreline_layer_weight,
            favorite_tail_strength=base_model.favorite_tail_strength,
            favorite_tail_threshold=base_model.favorite_tail_threshold,
            reranker_strength=base_model.reranker_strength,
        )
        actual = (int(match["goals_a"]), int(match["goals_b"]))
        if actual not in score_matrix:
            continue
        keys = candidate_keys(
            score_matrix,
            pool_size=candidate_pool_size,
            actual_score=actual,
        )
        context = {
            "is_group_stage": match.get("is_group_stage", 0.0),
            "is_knockout": match.get("is_knockout", 0.0),
            "host_a": match.get("host_a", 0.0),
            "host_b": match.get("host_b", 0.0),
            "same_confed": match.get("same_confed", 0.0),
        }
        candidates = candidate_feature_frame(
            score_matrix,
            base_prediction,
            keys,
            context=context,
        )
        for key, (_, row) in zip(keys, candidates.iterrows()):
            feature_rows.append(row.to_dict())
            labels.append(int(key == actual))
            base_weight = float(sample_weights.iloc[index])
            weights.append(base_weight * (8.0 if key == actual else 1.0))
        used_matches += 1

    X = pd.DataFrame(feature_rows, columns=RERANKER_FEATURES)
    y = pd.Series(labels, dtype=int)
    weight_array = np.asarray(weights, dtype=float)
    diagnostics = {
        "training_rows": int(used_matches),
        "source_training_rows": int(len(train_frame)),
        "candidate_rows": int(len(X)),
        "positive_rows": int(y.sum()) if len(y) else 0,
        "candidate_pool_size": int(candidate_pool_size),
        "max_train_matches": int(max_train_matches),
        "positive_rate": float(y.mean()) if len(y) else 0.0,
    }
    return X, y, weight_array, diagnostics


def fit_scoreline_reranker(
    base_model: v23.V23NoPlayerScorelineModel,
    max_goals: int,
    candidate_pool_size: int,
    max_train_matches: int,
    reranker_model: str,
) -> tuple[Any | None, Dict[str, Any]]:
    X, y, sample_weight, diagnostics = build_reranker_training_data(
        base_model,
        max_goals=max_goals,
        candidate_pool_size=candidate_pool_size,
        max_train_matches=max_train_matches,
    )
    diagnostics["reranker_model"] = reranker_model
    if X.empty or y.nunique() < 2:
        diagnostics["enabled"] = False
        return None, diagnostics
    estimator = build_reranker_estimator(reranker_model)
    estimator.fit(X[RERANKER_FEATURES], y, sample_weight=sample_weight)
    diagnostics["enabled"] = True
    diagnostics["top_feature_importances"] = []
    return estimator, diagnostics


def apply_reranker_to_matrix(
    matrix: ScoreMatrix,
    prediction: Dict[str, Any],
    estimator: Any | None,
    blend: float,
    power: float,
    context: Dict[str, Any] | None = None,
) -> tuple[ScoreMatrix, Dict[str, Any]]:
    if estimator is None:
        return dict(matrix), {"reranker_enabled": False}
    keys = [key for key, _ in sorted_matrix_items(matrix)]
    features = candidate_feature_frame(matrix, prediction, keys, context=context)
    learned = predict_positive_probability(estimator, features)
    if len(learned) != len(keys):
        return dict(matrix), {"reranker_enabled": False}
    learned = np.clip(learned, 1e-6, 1.0)
    learned = learned / max(float(np.mean(learned)), 1e-12)
    weights = np.power(learned, float(power))
    adjusted = {
        key: matrix[key] * float(weight)
        for key, weight in zip(keys, weights)
    }
    adjusted = normalize_matrix(adjusted)
    adjusted = v11.reweight_score_matrix_to_results(
        adjusted,
        prediction["result_probabilities"],
    )
    blended = v23.blend_score_matrices(matrix, adjusted, adjusted_weight=blend)
    blended = v11.reweight_score_matrix_to_results(
        blended,
        prediction["result_probabilities"],
    )
    return blended, {
        "reranker_enabled": True,
        "reranker_blend": float(np.clip(blend, 0.0, 1.0)),
        "reranker_power": float(power),
        "mean_learned_score": float(np.mean(learned)),
        "max_learned_score": float(np.max(learned)),
    }


class V24ScorelineRerankerModel(v23.V23NoPlayerScorelineModel):
    """V23 no-player exact-score matrix plus supervised scoreline reranking."""

    def __init__(
        self,
        base_model: v23.V23NoPlayerScorelineModel,
        reranker: Any | None,
        reranker_diagnostics: Dict[str, Any] | None = None,
        reranker_blend: float = DEFAULT_RERANKER_BLEND,
        reranker_power: float = DEFAULT_RERANKER_POWER,
    ):
        super().__init__(
            base_model.outcome_model,
            scoreline_layer_weight=base_model.scoreline_layer_weight,
            favorite_tail_strength=base_model.favorite_tail_strength,
            favorite_tail_threshold=base_model.favorite_tail_threshold,
            reranker_strength=base_model.reranker_strength,
            diversity_relative_floor=base_model.diversity_relative_floor,
        )
        self.base_v23_model = base_model
        self.reranker = reranker
        self.reranker_diagnostics = reranker_diagnostics or {}
        self.reranker_blend = float(np.clip(reranker_blend, 0.0, 1.0))
        self.reranker_power = float(max(reranker_power, 0.0))
        self.training_data_summary = getattr(base_model, "training_data_summary", {})

    def predict(self, *args, **kwargs) -> Dict[str, Any]:
        max_goals = int(kwargs.get("max_goals", args[5] if len(args) > 5 else 10))
        prediction = self.base_v23_model.predict(*args, **kwargs)
        base_matrix = score_matrix_from_prediction(prediction)
        context = {
            "is_group_stage": float(not bool(kwargs.get("knockout", False))),
            "is_knockout": float(bool(kwargs.get("knockout", False))),
            "host_a": float(bool(kwargs.get("host_a", False))),
            "host_b": float(bool(kwargs.get("host_b", False))),
            "same_confed": 0.0,
        }
        score_matrix, diagnostics = apply_reranker_to_matrix(
            base_matrix,
            prediction,
            self.reranker,
            blend=self.reranker_blend,
            power=self.reranker_power,
            context=context,
        )
        prediction.update(v15.score_outputs(score_matrix, max_goals))
        prediction["top_scorelines"] = v23.diversify_top_scorelines(
            score_matrix,
            prediction["result_probabilities"],
            float(prediction["lambda_a"]),
            float(prediction["lambda_b"]),
            top_n=15,
            relative_floor=self.diversity_relative_floor,
        )
        prediction["v24_adjustments"] = {
            "base_model": "v23_no_player_scoreline",
            "scoreline_policy": "supervised_candidate_scoreline_reranker",
            "scoreline_layer_affects_wdl": False,
            **diagnostics,
            "training_diagnostics": self.reranker_diagnostics,
        }
        prediction["calibration_notes"] = {
            **prediction.get("calibration_notes", {}),
            "v24": prediction["v24_adjustments"],
            "exact_score_policy": (
                "V24 preserves V23/V15 W/D/L probabilities, then applies a "
                "supervised candidate-scoreline reranker trained on historical "
                "World Cup/continental training rows with recency and prestige "
                "weights."
            ),
        }
        return prediction


def build_from_zip(
    zip_path,
    train_csv=None,
    test_csv=None,
    model_type="catboost",
    box_csv=None,
    results_csv=None,
    former_names_csv=None,
    prediction_year=2026,
    qualifier_blend_start_year=2014,
    qualifier_full_weight_year=2022,
    qualifier_minimum_influence=0.0,
    recency_half_life_years=16.0,
    recency_min_weight=0.10,
    player_ratings_csv=None,
    declared_squads_csv=None,
    results_as_of=v15.DEFAULT_RESULTS_AS_OF,
    scoreline_layer_weight=v23.DEFAULT_SCORELINE_LAYER_WEIGHT,
    favorite_tail_strength=v23.DEFAULT_FAVORITE_TAIL_STRENGTH,
    favorite_tail_threshold=v23.DEFAULT_FAVORITE_TAIL_THRESHOLD,
    reranker_strength=v23.DEFAULT_RERANKER_STRENGTH,
    diversity_relative_floor=v23.DEFAULT_DIVERSITY_RELATIVE_FLOOR,
    max_goals=10,
    candidate_pool_size=DEFAULT_CANDIDATE_POOL_SIZE,
    max_reranker_train_matches=DEFAULT_MAX_RERANKER_TRAIN_MATCHES,
    scoreline_reranker_blend=DEFAULT_RERANKER_BLEND,
    scoreline_reranker_power=DEFAULT_RERANKER_POWER,
    scoreline_reranker_model=DEFAULT_RERANKER_MODEL,
):
    base_model, data = v23.build_from_zip(
        zip_path,
        train_csv=train_csv,
        test_csv=test_csv,
        model_type=model_type,
        box_csv=box_csv,
        results_csv=results_csv,
        former_names_csv=former_names_csv,
        prediction_year=prediction_year,
        qualifier_blend_start_year=qualifier_blend_start_year,
        qualifier_full_weight_year=qualifier_full_weight_year,
        qualifier_minimum_influence=qualifier_minimum_influence,
        recency_half_life_years=recency_half_life_years,
        recency_min_weight=recency_min_weight,
        player_ratings_csv=player_ratings_csv,
        declared_squads_csv=declared_squads_csv,
        results_as_of=results_as_of,
        scoreline_layer_weight=scoreline_layer_weight,
        favorite_tail_strength=favorite_tail_strength,
        favorite_tail_threshold=favorite_tail_threshold,
        reranker_strength=reranker_strength,
        diversity_relative_floor=diversity_relative_floor,
    )
    reranker, diagnostics = fit_scoreline_reranker(
        base_model,
        max_goals=max_goals,
        candidate_pool_size=candidate_pool_size,
        max_train_matches=max_reranker_train_matches,
        reranker_model=scoreline_reranker_model,
    )
    model = V24ScorelineRerankerModel(
        base_model,
        reranker,
        reranker_diagnostics=diagnostics,
        reranker_blend=scoreline_reranker_blend,
        reranker_power=scoreline_reranker_power,
    )
    model.training_data_summary = {
        **getattr(base_model, "training_data_summary", {}),
        "v24_scoreline_policy": "supervised_candidate_scoreline_reranker",
        "v24_reranker_blend": model.reranker_blend,
        "v24_reranker_power": model.reranker_power,
        "v24_reranker_diagnostics": diagnostics,
    }
    return model, data


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    parser = argparse.ArgumentParser(
        description="Run V24: V23 with supervised exact-score reranker."
    )
    parser.add_argument("--team-a", required=True)
    parser.add_argument("--team-b", required=True)
    parser.add_argument("--host-a", action="store_true")
    parser.add_argument("--host-b", action="store_true")
    parser.add_argument("--knockout", action="store_true")
    parser.add_argument("--outdir", default="outputs/outputs_v24_scoreline_reranker")
    parser.add_argument("--worldcupsai-zip", default=str(data_dir / "worldcupsai.zip"))
    parser.add_argument("--team-train", default=str(data_dir / "current_team_features_2026.csv"))
    parser.add_argument("--team-test")
    parser.add_argument("--box-data", default=str(data_dir / "FIFAallMatchBoxData.csv"))
    parser.add_argument("--results-data", default=str(data_dir / "results.csv"))
    parser.add_argument("--results-as-of", default=v15.DEFAULT_RESULTS_AS_OF)
    parser.add_argument("--former-names", default=str(data_dir / "former_names.csv"))
    parser.add_argument("--prediction-year", type=int, default=2026)
    parser.add_argument("--candidate-pool-size", type=int, default=DEFAULT_CANDIDATE_POOL_SIZE)
    parser.add_argument(
        "--max-reranker-train-matches",
        type=int,
        default=DEFAULT_MAX_RERANKER_TRAIN_MATCHES,
    )
    parser.add_argument("--scoreline-reranker-blend", type=float, default=DEFAULT_RERANKER_BLEND)
    parser.add_argument("--scoreline-reranker-power", type=float, default=DEFAULT_RERANKER_POWER)
    parser.add_argument(
        "--scoreline-reranker-model",
        choices=["hgb", "logistic"],
        default=DEFAULT_RERANKER_MODEL,
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    output_dir = v11.unique_output_dir(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, _ = build_from_zip(
        args.worldcupsai_zip,
        train_csv=args.team_train,
        test_csv=args.team_test,
        box_csv=args.box_data,
        results_csv=args.results_data,
        former_names_csv=args.former_names,
        prediction_year=args.prediction_year,
        results_as_of=args.results_as_of,
        candidate_pool_size=args.candidate_pool_size,
        max_reranker_train_matches=args.max_reranker_train_matches,
        scoreline_reranker_blend=args.scoreline_reranker_blend,
        scoreline_reranker_power=args.scoreline_reranker_power,
        scoreline_reranker_model=args.scoreline_reranker_model,
    )
    prediction = model.predict(
        args.team_a,
        args.team_b,
        host_a=args.host_a,
        host_b=args.host_b,
        knockout=args.knockout,
    )
    (output_dir / "single_match_prediction.json").write_text(
        json.dumps(prediction, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(prediction["top_scorelines"]).to_csv(
        output_dir / "scoreline_probabilities_top.csv",
        index=False,
    )
    pd.DataFrame(prediction["scoreline_probabilities"]).to_csv(
        output_dir / "scoreline_probabilities.csv",
        index=False,
    )
    (output_dir / "model_summary.json").write_text(
        json.dumps(
            {
                "version": "v24-scoreline-reranker",
                "base_model": "v23-no-player-scoreline",
                "wdl_model": "v15_catboost_preserved",
                "exact_score_model": "supervised_candidate_scoreline_reranker",
                "team_a": prediction["team_a"],
                "team_b": prediction["team_b"],
                "v24_adjustments": prediction["v24_adjustments"],
                "expanded_training_data": model.training_data_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not args.no_plots:
        v11.plot_prediction_outputs(prediction, output_dir)
    print(
        json.dumps(
            {
                "result_probabilities": prediction["result_probabilities"],
                "predicted_result": prediction["predicted_result"],
                "lambda_a": prediction["lambda_a"],
                "lambda_b": prediction["lambda_b"],
                "top_scorelines": prediction["top_scorelines"][:5],
                "v24_adjustments": {
                    "reranker_enabled": prediction["v24_adjustments"][
                        "reranker_enabled"
                    ],
                    "reranker_blend": prediction["v24_adjustments"].get(
                        "reranker_blend"
                    ),
                    "reranker_power": prediction["v24_adjustments"].get(
                        "reranker_power"
                    ),
                    "training_diagnostics": prediction["v24_adjustments"][
                        "training_diagnostics"
                    ],
                },
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
'''
v24_scoreline_reranker_model = _load_submodule("v24_scoreline_reranker_model", _V24_SCORELINE_RERANKER_MODEL_SOURCE, "core_engine.py:v24_scoreline_reranker_model")

# ======================================================================
# v26_top3_coverage_model.py  (bundled as an isolated sub-module)
# ======================================================================
_V26_TOP3_COVERAGE_MODEL_SOURCE = r'''
#!/usr/bin/env python3
"""V26: V20 with a probability-gated Top-3 scoreline coverage selector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

import v11_wcq_results_model as v11
import v15_catboost_model as v15
import v20_scoreline_ensemble_model as v20


ScoreMatrix = Dict[Tuple[int, int], float]

DEFAULT_TAIL_RELATIVE_FLOOR = 1.0
DEFAULT_FAVORITE_WIN_GATE = 0.55
DEFAULT_TOTAL_LAMBDA_GATE = 2.45
DEFAULT_FAVORITE_LAMBDA_GATE = 1.55
DEFAULT_DRAW_CEILING = 0.30


def score_matrix_from_prediction(prediction: Dict[str, Any]) -> ScoreMatrix:
    return {
        (int(item["team_a_goals"]), int(item["team_b_goals"])): float(
            item["probability"]
        )
        for item in prediction["scoreline_probabilities"]
    }


def score_item(key: Tuple[int, int], probability: float) -> Dict[str, Any]:
    return {
        "team_a_goals": int(key[0]),
        "team_b_goals": int(key[1]),
        "probability": float(probability),
    }


def sorted_score_keys(score_matrix: ScoreMatrix) -> list[Tuple[int, int]]:
    return [
        key
        for key, _ in sorted(
            score_matrix.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]


def favorite_side(result_probabilities: Dict[str, float]) -> str | None:
    decisive = {
        "team_a": float(result_probabilities.get("team_a_win", 0.0)),
        "team_b": float(result_probabilities.get("team_b_win", 0.0)),
    }
    side = max(decisive, key=decisive.get)
    if decisive[side] <= float(result_probabilities.get("draw", 0.0)):
        return None
    return side


def is_favorite_win_score(key: Tuple[int, int], side: str) -> bool:
    goals_a, goals_b = key
    return goals_a > goals_b if side == "team_a" else goals_b > goals_a


def is_high_total_favorite_score(key: Tuple[int, int], side: str) -> bool:
    goals_a, goals_b = key
    if not is_favorite_win_score(key, side):
        return False
    winner_goals = goals_a if side == "team_a" else goals_b
    loser_goals = goals_b if side == "team_a" else goals_a
    total_goals = goals_a + goals_b
    return winner_goals >= 3 and loser_goals >= 1 and total_goals >= 4


def best_available(
    score_matrix: ScoreMatrix,
    candidates: list[Tuple[int, int]],
    selected: set[Tuple[int, int]],
) -> Tuple[int, int] | None:
    available = [key for key in candidates if key in score_matrix and key not in selected]
    if not available:
        return None
    return max(available, key=lambda key: score_matrix.get(key, 0.0))


def high_total_favorite_candidates(
    score_matrix: ScoreMatrix,
    side: str,
    max_winner_goals: int = 5,
) -> list[Tuple[int, int]]:
    candidates = [
        key
        for key in score_matrix
        if is_high_total_favorite_score(key, side)
        and max(key) <= max_winner_goals
    ]
    return sorted(candidates, key=lambda key: score_matrix.get(key, 0.0), reverse=True)


def select_top_scorelines_with_coverage(
    score_matrix: ScoreMatrix,
    result_probabilities: Dict[str, float],
    lambda_a: float,
    lambda_b: float,
    top_n: int = 15,
    tail_relative_floor: float = DEFAULT_TAIL_RELATIVE_FLOOR,
    favorite_win_gate: float = DEFAULT_FAVORITE_WIN_GATE,
    total_lambda_gate: float = DEFAULT_TOTAL_LAMBDA_GATE,
    favorite_lambda_gate: float = DEFAULT_FAVORITE_LAMBDA_GATE,
    draw_ceiling: float = DEFAULT_DRAW_CEILING,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    ranked = sorted_score_keys(score_matrix)
    selected = ranked[:top_n]
    diagnostics: Dict[str, Any] = {
        "coverage_selector_enabled": True,
        "coverage_applied": False,
        "tail_relative_floor": float(tail_relative_floor),
        "favorite_win_gate": float(favorite_win_gate),
        "total_lambda_gate": float(total_lambda_gate),
        "favorite_lambda_gate": float(favorite_lambda_gate),
        "draw_ceiling": float(draw_ceiling),
    }
    if len(selected) < 3:
        return [score_item(key, score_matrix[key]) for key in selected], diagnostics

    side = favorite_side(result_probabilities)
    favorite_probability = (
        float(result_probabilities.get("team_a_win", 0.0))
        if side == "team_a"
        else float(result_probabilities.get("team_b_win", 0.0))
        if side == "team_b"
        else 0.0
    )
    favorite_lambda = float(lambda_a) if side == "team_a" else float(lambda_b)
    total_lambda = float(lambda_a) + float(lambda_b)
    top_probability = max(float(score_matrix[selected[0]]), 1e-12)
    floor = top_probability * float(tail_relative_floor)
    diagnostics.update(
        {
            "favorite_side": side,
            "favorite_probability": favorite_probability,
            "favorite_lambda": favorite_lambda,
            "total_lambda": total_lambda,
            "probability_floor": floor,
        }
    )

    qualifies = (
        side is not None
        and favorite_probability >= float(favorite_win_gate)
        and total_lambda >= float(total_lambda_gate)
        and favorite_lambda >= float(favorite_lambda_gate)
        and float(result_probabilities.get("draw", 0.0)) <= float(draw_ceiling)
    )
    if not qualifies:
        diagnostics["skip_reason"] = "gates_not_met"
        return [score_item(key, score_matrix[key]) for key in selected], diagnostics

    top_three = selected[:3]
    selected_set = set(top_three)
    candidates = high_total_favorite_candidates(score_matrix, side)
    candidate = best_available(score_matrix, candidates, selected_set)
    if candidate is None:
        diagnostics["skip_reason"] = "no_candidate"
        return [score_item(key, score_matrix[key]) for key in selected], diagnostics
    candidate_probability = float(score_matrix[candidate])
    diagnostics["candidate_scoreline"] = f"{candidate[0]}-{candidate[1]}"
    diagnostics["candidate_probability"] = candidate_probability
    if candidate_probability < floor:
        diagnostics["skip_reason"] = "candidate_below_floor"
        return [score_item(key, score_matrix[key]) for key in selected], diagnostics

    top_three[-1] = candidate
    rebuilt: list[Tuple[int, int]] = []
    for key in [*top_three, *selected]:
        if key not in rebuilt:
            rebuilt.append(key)
        if len(rebuilt) >= top_n:
            break
    diagnostics["coverage_applied"] = True
    diagnostics["replaced_third_scoreline"] = f"{selected[2][0]}-{selected[2][1]}"
    return [score_item(key, score_matrix[key]) for key in rebuilt], diagnostics


class V26Top3CoverageModel:
    """Wrap V20 and select Top-3 as a small coverage portfolio."""

    def __init__(
        self,
        base_model: v20.V20ScorelineEnsembleModel,
        tail_relative_floor: float = DEFAULT_TAIL_RELATIVE_FLOOR,
        favorite_win_gate: float = DEFAULT_FAVORITE_WIN_GATE,
        total_lambda_gate: float = DEFAULT_TOTAL_LAMBDA_GATE,
        favorite_lambda_gate: float = DEFAULT_FAVORITE_LAMBDA_GATE,
        draw_ceiling: float = DEFAULT_DRAW_CEILING,
    ):
        self.base_model = base_model
        self.tail_relative_floor = float(max(tail_relative_floor, 0.0))
        self.favorite_win_gate = float(np.clip(favorite_win_gate, 0.0, 1.0))
        self.total_lambda_gate = float(max(total_lambda_gate, 0.0))
        self.favorite_lambda_gate = float(max(favorite_lambda_gate, 0.0))
        self.draw_ceiling = float(np.clip(draw_ceiling, 0.0, 1.0))
        self.training_data_summary = getattr(base_model, "training_data_summary", {})

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_model, name)

    def predict(self, *args, **kwargs) -> Dict[str, Any]:
        prediction = self.base_model.predict(*args, **kwargs)
        score_matrix = score_matrix_from_prediction(prediction)
        top_scorelines, diagnostics = select_top_scorelines_with_coverage(
            score_matrix,
            prediction["result_probabilities"],
            float(prediction["lambda_a"]),
            float(prediction["lambda_b"]),
            top_n=15,
            tail_relative_floor=self.tail_relative_floor,
            favorite_win_gate=self.favorite_win_gate,
            total_lambda_gate=self.total_lambda_gate,
            favorite_lambda_gate=self.favorite_lambda_gate,
            draw_ceiling=self.draw_ceiling,
        )
        prediction["top_scorelines"] = top_scorelines
        prediction["v26_adjustments"] = {
            "base_model": "v20_scoreline_ensemble",
            "scoreline_policy": "top3_coverage_selector_only",
            "probability_matrix_changed": False,
            "scoreline_layer_affects_wdl": False,
            **diagnostics,
        }
        prediction["calibration_notes"] = {
            **prediction.get("calibration_notes", {}),
            "v26": prediction["v26_adjustments"],
            "top_scoreline_policy": (
                "V26 leaves V20 probabilities unchanged and only reorders the "
                "displayed Top-3/Top-15 list when a high-total favorite-win "
                "script clears conservative probability gates."
            ),
        }
        return prediction


def build_from_zip(
    zip_path,
    train_csv=None,
    test_csv=None,
    model_type="catboost",
    box_csv=None,
    results_csv=None,
    former_names_csv=None,
    prediction_year=2026,
    qualifier_blend_start_year=2014,
    qualifier_full_weight_year=2022,
    qualifier_minimum_influence=0.0,
    recency_half_life_years=16.0,
    recency_min_weight=0.10,
    player_ratings_csv=None,
    declared_squads_csv=None,
    fcratings_csv=None,
    results_as_of=v15.DEFAULT_RESULTS_AS_OF,
    tail_relative_floor=DEFAULT_TAIL_RELATIVE_FLOOR,
    favorite_win_gate=DEFAULT_FAVORITE_WIN_GATE,
    total_lambda_gate=DEFAULT_TOTAL_LAMBDA_GATE,
    favorite_lambda_gate=DEFAULT_FAVORITE_LAMBDA_GATE,
    draw_ceiling=DEFAULT_DRAW_CEILING,
):
    base_model, data = v20.build_from_zip(
        zip_path,
        train_csv=train_csv,
        test_csv=test_csv,
        model_type=model_type,
        box_csv=box_csv,
        results_csv=results_csv,
        former_names_csv=former_names_csv,
        prediction_year=prediction_year,
        qualifier_blend_start_year=qualifier_blend_start_year,
        qualifier_full_weight_year=qualifier_full_weight_year,
        qualifier_minimum_influence=qualifier_minimum_influence,
        recency_half_life_years=recency_half_life_years,
        recency_min_weight=recency_min_weight,
        player_ratings_csv=player_ratings_csv,
        declared_squads_csv=declared_squads_csv,
        fcratings_csv=fcratings_csv,
        results_as_of=results_as_of,
    )
    model = V26Top3CoverageModel(
        base_model,
        tail_relative_floor=tail_relative_floor,
        favorite_win_gate=favorite_win_gate,
        total_lambda_gate=total_lambda_gate,
        favorite_lambda_gate=favorite_lambda_gate,
        draw_ceiling=draw_ceiling,
    )
    model.training_data_summary = {
        **getattr(base_model, "training_data_summary", {}),
        "v26_scoreline_policy": "top3_coverage_selector_only",
        "v26_probability_matrix_changed": False,
        "v26_tail_relative_floor": model.tail_relative_floor,
        "v26_favorite_win_gate": model.favorite_win_gate,
        "v26_total_lambda_gate": model.total_lambda_gate,
        "v26_favorite_lambda_gate": model.favorite_lambda_gate,
        "v26_draw_ceiling": model.draw_ceiling,
    }
    return model, data


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    parser = argparse.ArgumentParser(
        description="Run V26: V20 with Top-3 coverage selection."
    )
    parser.add_argument("--team-a", required=True)
    parser.add_argument("--team-b", required=True)
    parser.add_argument("--host-a", action="store_true")
    parser.add_argument("--host-b", action="store_true")
    parser.add_argument("--knockout", action="store_true")
    parser.add_argument("--outdir", default="outputs/outputs_v26_top3_coverage")
    parser.add_argument("--worldcupsai-zip", default=str(data_dir / "worldcupsai.zip"))
    parser.add_argument("--team-train", default=str(data_dir / "current_team_features_2026.csv"))
    parser.add_argument("--team-test")
    parser.add_argument("--box-data", default=str(data_dir / "FIFAallMatchBoxData.csv"))
    parser.add_argument("--results-data", default=str(data_dir / "results.csv"))
    parser.add_argument("--results-as-of", default=v15.DEFAULT_RESULTS_AS_OF)
    parser.add_argument("--former-names", default=str(data_dir / "former_names.csv"))
    parser.add_argument("--prediction-year", type=int, default=2026)
    parser.add_argument("--tail-relative-floor", type=float, default=DEFAULT_TAIL_RELATIVE_FLOOR)
    parser.add_argument("--favorite-win-gate", type=float, default=DEFAULT_FAVORITE_WIN_GATE)
    parser.add_argument("--total-lambda-gate", type=float, default=DEFAULT_TOTAL_LAMBDA_GATE)
    parser.add_argument("--favorite-lambda-gate", type=float, default=DEFAULT_FAVORITE_LAMBDA_GATE)
    parser.add_argument("--draw-ceiling", type=float, default=DEFAULT_DRAW_CEILING)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    output_dir = v11.unique_output_dir(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, _ = build_from_zip(
        args.worldcupsai_zip,
        train_csv=args.team_train,
        test_csv=args.team_test,
        box_csv=args.box_data,
        results_csv=args.results_data,
        former_names_csv=args.former_names,
        prediction_year=args.prediction_year,
        results_as_of=args.results_as_of,
        tail_relative_floor=args.tail_relative_floor,
        favorite_win_gate=args.favorite_win_gate,
        total_lambda_gate=args.total_lambda_gate,
        favorite_lambda_gate=args.favorite_lambda_gate,
        draw_ceiling=args.draw_ceiling,
    )
    prediction = model.predict(
        args.team_a,
        args.team_b,
        host_a=args.host_a,
        host_b=args.host_b,
        knockout=args.knockout,
    )
    (output_dir / "single_match_prediction.json").write_text(
        json.dumps(prediction, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(prediction["top_scorelines"]).to_csv(
        output_dir / "scoreline_probabilities_top.csv",
        index=False,
    )
    pd.DataFrame(prediction["scoreline_probabilities"]).to_csv(
        output_dir / "scoreline_probabilities.csv",
        index=False,
    )
    (output_dir / "model_summary.json").write_text(
        json.dumps(
            {
                "version": "v26-top3-coverage",
                "base_model": "v20-scoreline-ensemble",
                "team_a": prediction["team_a"],
                "team_b": prediction["team_b"],
                "result_probabilities": prediction["result_probabilities"],
                "v26_adjustments": prediction["v26_adjustments"],
                "expanded_training_data": model.training_data_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not args.no_plots:
        v11.plot_prediction_outputs(prediction, output_dir)
    print(
        json.dumps(
            {
                "result_probabilities": prediction["result_probabilities"],
                "predicted_result": prediction["predicted_result"],
                "lambda_a": prediction["lambda_a"],
                "lambda_b": prediction["lambda_b"],
                "top_scorelines": prediction["top_scorelines"][:5],
                "v26_adjustments": prediction["v26_adjustments"],
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
'''
v26_top3_coverage_model = _load_submodule("v26_top3_coverage_model", _V26_TOP3_COVERAGE_MODEL_SOURCE, "core_engine.py:v26_top3_coverage_model")

# ======================================================================
# v27_total_goals_calibrated_model.py  (bundled as an isolated sub-module)
# ======================================================================
_V27_TOTAL_GOALS_CALIBRATED_MODEL_SOURCE = r'''
#!/usr/bin/env python3
"""V27: V20 with historical total-goals calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd

import v11_wcq_results_model as v11
import v15_catboost_model as v15
import v20_scoreline_ensemble_model as v20
import v24_scoreline_reranker_model as v24


ScoreMatrix = Dict[Tuple[int, int], float]

DEFAULT_TOTAL_CALIBRATION_STRENGTH = 0.30
DEFAULT_MULTIPLIER_CLIP_LOW = 0.82
DEFAULT_MULTIPLIER_CLIP_HIGH = 1.18
DEFAULT_TOTAL_SMOOTHING = 0.35
DEFAULT_MIN_BIN_SUPPORT = 80.0
DEFAULT_MAX_TRAIN_MATCHES = 0
DEFAULT_TOTAL_BIN_EDGES = (2.20, 2.80)
DEFAULT_CALIBRATION_BLEND = 0.0


def score_matrix_from_prediction(prediction: Dict[str, Any]) -> ScoreMatrix:
    return {
        (int(item["team_a_goals"]), int(item["team_b_goals"])): float(
            item["probability"]
        )
        for item in prediction["scoreline_probabilities"]
    }


def total_distribution(score_matrix: ScoreMatrix, max_total: int) -> np.ndarray:
    distribution = np.zeros(max_total + 1, dtype=float)
    for (goals_a, goals_b), probability in score_matrix.items():
        total = int(goals_a) + int(goals_b)
        if total <= max_total:
            distribution[total] += float(probability)
    total_mass = float(distribution.sum())
    if total_mass <= 0:
        distribution[0] = 1.0
        return distribution
    return distribution / total_mass


def normalize_matrix(score_matrix: ScoreMatrix) -> ScoreMatrix:
    total = float(sum(score_matrix.values()))
    if total <= 0:
        raise ValueError("score matrix has no probability mass")
    return {key: float(value) / total for key, value in score_matrix.items()}


def predicted_total_bin(lambda_total: float, edges: Tuple[float, float]) -> str:
    low, high = edges
    if float(lambda_total) < float(low):
        return "low"
    if float(lambda_total) < float(high):
        return "mid"
    return "high"


def phase_key_from_row(row: pd.Series) -> str:
    if float(row.get("is_knockout", 0.0) or 0.0) > 0.5:
        return "knockout"
    return "group"


class TotalGoalsCalibrationModel:
    def __init__(
        self,
        multipliers: Dict[str, Dict[str, list[float]]],
        support: Dict[str, Dict[str, float]],
        max_total: int,
        bin_edges: Tuple[float, float] = DEFAULT_TOTAL_BIN_EDGES,
        min_bin_support: float = DEFAULT_MIN_BIN_SUPPORT,
    ):
        self.multipliers = multipliers
        self.support = support
        self.max_total = int(max_total)
        self.bin_edges = (float(bin_edges[0]), float(bin_edges[1]))
        self.min_bin_support = float(min_bin_support)

    def lookup_key(self, lambda_total: float, knockout: bool) -> tuple[str, str]:
        phase = "knockout" if knockout else "group"
        total_bin = predicted_total_bin(lambda_total, self.bin_edges)
        if self.support.get(phase, {}).get(total_bin, 0.0) >= self.min_bin_support:
            return phase, total_bin
        if self.support.get("overall", {}).get(total_bin, 0.0) >= self.min_bin_support:
            return "overall", total_bin
        return "overall", "all"

    def multipliers_for(self, lambda_total: float, knockout: bool) -> np.ndarray:
        phase, total_bin = self.lookup_key(lambda_total, knockout)
        values = self.multipliers.get(phase, {}).get(total_bin)
        if values is None:
            values = self.multipliers["overall"]["all"]
        return np.asarray(values, dtype=float)

    def apply(
        self,
        score_matrix: ScoreMatrix,
        result_probabilities: Dict[str, float],
        lambda_total: float,
        knockout: bool,
    ) -> tuple[ScoreMatrix, Dict[str, Any]]:
        multipliers = self.multipliers_for(lambda_total, knockout)
        adjusted = {}
        for key, probability in score_matrix.items():
            total = int(key[0]) + int(key[1])
            multiplier = multipliers[min(total, self.max_total)]
            adjusted[key] = float(probability) * float(multiplier)
        adjusted = normalize_matrix(adjusted)
        adjusted = v11.reweight_score_matrix_to_results(
            adjusted,
            result_probabilities,
        )
        phase, total_bin = self.lookup_key(lambda_total, knockout)
        return adjusted, {
            "total_calibration_enabled": True,
            "lookup_phase": phase,
            "lookup_total_bin": total_bin,
            "lambda_total": float(lambda_total),
            "multipliers": {
                str(total): float(multipliers[total])
                for total in range(len(multipliers))
            },
            "support": self.support.get(phase, {}).get(total_bin, 0.0),
        }

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "max_total": self.max_total,
            "bin_edges": list(self.bin_edges),
            "min_bin_support": self.min_bin_support,
            "support": self.support,
            "overall_all_multipliers": self.multipliers.get("overall", {}).get("all", []),
        }


def weighted_calibration_counts(
    rows: list[Dict[str, Any]],
    max_total: int,
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    actual = np.full(max_total + 1, float(smoothing), dtype=float)
    predicted = np.full(max_total + 1, float(smoothing), dtype=float)
    support = 0.0
    for row in rows:
        weight = float(row["weight"])
        actual_total = min(int(row["actual_total"]), max_total)
        actual[actual_total] += weight
        predicted += weight * np.asarray(row["predicted_total_distribution"], dtype=float)
        support += weight
    return actual, predicted, support


def make_multipliers(
    actual: np.ndarray,
    predicted: np.ndarray,
    strength: float,
    clip_low: float,
    clip_high: float,
) -> list[float]:
    actual_share = actual / max(float(actual.sum()), 1e-12)
    predicted_share = predicted / max(float(predicted.sum()), 1e-12)
    ratio = actual_share / np.clip(predicted_share, 1e-9, None)
    multiplier = np.power(ratio, float(strength))
    multiplier = np.clip(multiplier, float(clip_low), float(clip_high))
    return [float(value) for value in multiplier]


def fit_total_goals_calibration(
    base_model: v20.V20ScorelineEnsembleModel,
    max_goals: int,
    recency_half_life_years: float,
    recency_min_weight: float,
    strength: float,
    clip_low: float,
    clip_high: float,
    smoothing: float,
    min_bin_support: float,
    max_train_matches: int,
    bin_edges: Tuple[float, float] = DEFAULT_TOTAL_BIN_EDGES,
) -> TotalGoalsCalibrationModel:
    outcome_model = getattr(base_model.base_model, "outcome_model", base_model.base_model)
    train_frame = getattr(outcome_model, "train_frame", pd.DataFrame())
    max_total = int(max_goals * 2)
    if train_frame is None or train_frame.empty:
        identity = [1.0 for _ in range(max_total + 1)]
        return TotalGoalsCalibrationModel(
            {"overall": {"all": identity}},
            {"overall": {"all": 0.0}},
            max_total=max_total,
            bin_edges=bin_edges,
            min_bin_support=min_bin_support,
        )

    frame = train_frame.dropna(subset=["goals_a", "goals_b"]).reset_index(drop=True)
    sample_weights = v11.build_year_recency_weights(
        frame,
        recency_half_life_years,
        recency_min_weight,
    )
    sample_weights = v11.combine_training_weights(frame, sample_weights).reset_index(drop=True)
    if len(frame) > max_train_matches > 0:
        rng = np.random.default_rng(27)
        probabilities = sample_weights.to_numpy(dtype=float)
        probabilities = probabilities / max(float(probabilities.sum()), 1e-12)
        chosen = rng.choice(
            np.arange(len(frame)),
            size=int(max_train_matches),
            replace=False,
            p=probabilities,
        )
        chosen = np.sort(chosen)
        frame = frame.iloc[chosen].reset_index(drop=True)
        sample_weights = sample_weights.iloc[chosen].reset_index(drop=True)

    predictions = v24.model_predictions_from_feature_frame(
        outcome_model,
        frame,
        max_goals=max_goals,
    )
    rows: list[Dict[str, Any]] = []
    for index, match in frame.iterrows():
        prediction = predictions[index]
        matrix = score_matrix_from_prediction(prediction)
        lambda_total = float(prediction["lambda_a"]) + float(prediction["lambda_b"])
        rows.append(
            {
                "phase": phase_key_from_row(match),
                "total_bin": predicted_total_bin(lambda_total, bin_edges),
                "actual_total": int(match["goals_a"]) + int(match["goals_b"]),
                "predicted_total_distribution": total_distribution(matrix, max_total),
                "weight": float(sample_weights.iloc[index]),
            }
        )

    phases = ["overall", "group", "knockout"]
    bins = ["all", "low", "mid", "high"]
    multipliers: Dict[str, Dict[str, list[float]]] = {}
    support: Dict[str, Dict[str, float]] = {}
    for phase in phases:
        multipliers[phase] = {}
        support[phase] = {}
        for total_bin in bins:
            selected = [
                row
                for row in rows
                if (phase == "overall" or row["phase"] == phase)
                and (total_bin == "all" or row["total_bin"] == total_bin)
            ]
            actual, predicted, bin_support = weighted_calibration_counts(
                selected,
                max_total=max_total,
                smoothing=smoothing,
            )
            multipliers[phase][total_bin] = make_multipliers(
                actual,
                predicted,
                strength=strength,
                clip_low=clip_low,
                clip_high=clip_high,
            )
            support[phase][total_bin] = float(bin_support)

    return TotalGoalsCalibrationModel(
        multipliers,
        support,
        max_total=max_total,
        bin_edges=bin_edges,
        min_bin_support=min_bin_support,
    )


class V27TotalGoalsCalibratedModel:
    """Wrap V20 and apply conservative total-goal calibration."""

    def __init__(
        self,
        base_model: v20.V20ScorelineEnsembleModel,
        calibration_model: TotalGoalsCalibrationModel,
        calibration_blend: float = DEFAULT_CALIBRATION_BLEND,
    ):
        self.base_model = base_model
        self.calibration_model = calibration_model
        self.calibration_blend = float(np.clip(calibration_blend, 0.0, 1.0))
        self.training_data_summary = getattr(base_model, "training_data_summary", {})

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_model, name)

    def predict(self, *args, **kwargs) -> Dict[str, Any]:
        max_goals = int(kwargs.get("max_goals", args[5] if len(args) > 5 else 10))
        knockout = bool(kwargs.get("knockout", False))
        prediction = self.base_model.predict(*args, **kwargs)
        base_matrix = score_matrix_from_prediction(prediction)
        calibrated_matrix, diagnostics = self.calibration_model.apply(
            base_matrix,
            prediction["result_probabilities"],
            float(prediction["lambda_a"]) + float(prediction["lambda_b"]),
            knockout=knockout,
        )
        score_matrix = v20.blend_score_matrices(
            base_matrix,
            calibrated_matrix,
            adjusted_weight=self.calibration_blend,
        )
        score_matrix = v11.reweight_score_matrix_to_results(
            score_matrix,
            prediction["result_probabilities"],
        )
        prediction.update(v15.score_outputs(score_matrix, max_goals))
        prediction["v27_adjustments"] = {
            "base_model": "v20_scoreline_ensemble",
            "scoreline_policy": "historical_total_goals_calibration",
            "scoreline_layer_affects_wdl": False,
            "calibration_blend": self.calibration_blend,
            **diagnostics,
        }
        prediction["calibration_notes"] = {
            **prediction.get("calibration_notes", {}),
            "v27": prediction["v27_adjustments"],
            "exact_score_policy": (
                "V27 preserves V20 W/D/L probabilities, then applies a "
                "conservative historical total-goals calibration by predicted "
                "total-goal bin and tournament phase."
            ),
        }
        return prediction


def build_from_zip(
    zip_path,
    train_csv=None,
    test_csv=None,
    model_type="catboost",
    box_csv=None,
    results_csv=None,
    former_names_csv=None,
    prediction_year=2026,
    qualifier_blend_start_year=2014,
    qualifier_full_weight_year=2022,
    qualifier_minimum_influence=0.0,
    recency_half_life_years=16.0,
    recency_min_weight=0.10,
    player_ratings_csv=None,
    declared_squads_csv=None,
    fcratings_csv=None,
    results_as_of=v15.DEFAULT_RESULTS_AS_OF,
    total_calibration_strength=DEFAULT_TOTAL_CALIBRATION_STRENGTH,
    total_multiplier_clip_low=DEFAULT_MULTIPLIER_CLIP_LOW,
    total_multiplier_clip_high=DEFAULT_MULTIPLIER_CLIP_HIGH,
    total_smoothing=DEFAULT_TOTAL_SMOOTHING,
    min_bin_support=DEFAULT_MIN_BIN_SUPPORT,
    max_train_matches=DEFAULT_MAX_TRAIN_MATCHES,
    calibration_blend=DEFAULT_CALIBRATION_BLEND,
):
    base_model, data = v20.build_from_zip(
        zip_path,
        train_csv=train_csv,
        test_csv=test_csv,
        model_type=model_type,
        box_csv=box_csv,
        results_csv=results_csv,
        former_names_csv=former_names_csv,
        prediction_year=prediction_year,
        qualifier_blend_start_year=qualifier_blend_start_year,
        qualifier_full_weight_year=qualifier_full_weight_year,
        qualifier_minimum_influence=qualifier_minimum_influence,
        recency_half_life_years=recency_half_life_years,
        recency_min_weight=recency_min_weight,
        player_ratings_csv=player_ratings_csv,
        declared_squads_csv=declared_squads_csv,
        fcratings_csv=fcratings_csv,
        results_as_of=results_as_of,
    )
    calibration_model = fit_total_goals_calibration(
        base_model,
        max_goals=10,
        recency_half_life_years=recency_half_life_years,
        recency_min_weight=recency_min_weight,
        strength=total_calibration_strength,
        clip_low=total_multiplier_clip_low,
        clip_high=total_multiplier_clip_high,
        smoothing=total_smoothing,
        min_bin_support=min_bin_support,
        max_train_matches=max_train_matches,
    )
    model = V27TotalGoalsCalibratedModel(
        base_model,
        calibration_model,
        calibration_blend=calibration_blend,
    )
    model.training_data_summary = {
        **getattr(base_model, "training_data_summary", {}),
        "v27_scoreline_policy": "historical_total_goals_calibration",
        "v27_calibration_blend": model.calibration_blend,
        "v27_total_calibration_strength": float(total_calibration_strength),
        "v27_total_multiplier_clip_low": float(total_multiplier_clip_low),
        "v27_total_multiplier_clip_high": float(total_multiplier_clip_high),
        "v27_total_smoothing": float(total_smoothing),
        "v27_min_bin_support": float(min_bin_support),
        "v27_calibration_diagnostics": calibration_model.diagnostics(),
    }
    return model, data


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    parser = argparse.ArgumentParser(
        description="Run V27: V20 with total-goals calibration."
    )
    parser.add_argument("--team-a", required=True)
    parser.add_argument("--team-b", required=True)
    parser.add_argument("--host-a", action="store_true")
    parser.add_argument("--host-b", action="store_true")
    parser.add_argument("--knockout", action="store_true")
    parser.add_argument("--outdir", default="outputs/outputs_v27_total_goals_calibrated")
    parser.add_argument("--worldcupsai-zip", default=str(data_dir / "worldcupsai.zip"))
    parser.add_argument("--team-train", default=str(data_dir / "current_team_features_2026.csv"))
    parser.add_argument("--team-test")
    parser.add_argument("--box-data", default=str(data_dir / "FIFAallMatchBoxData.csv"))
    parser.add_argument("--results-data", default=str(data_dir / "results.csv"))
    parser.add_argument("--results-as-of", default=v15.DEFAULT_RESULTS_AS_OF)
    parser.add_argument("--former-names", default=str(data_dir / "former_names.csv"))
    parser.add_argument("--prediction-year", type=int, default=2026)
    parser.add_argument("--total-calibration-strength", type=float, default=DEFAULT_TOTAL_CALIBRATION_STRENGTH)
    parser.add_argument("--total-multiplier-clip-low", type=float, default=DEFAULT_MULTIPLIER_CLIP_LOW)
    parser.add_argument("--total-multiplier-clip-high", type=float, default=DEFAULT_MULTIPLIER_CLIP_HIGH)
    parser.add_argument("--total-smoothing", type=float, default=DEFAULT_TOTAL_SMOOTHING)
    parser.add_argument("--min-bin-support", type=float, default=DEFAULT_MIN_BIN_SUPPORT)
    parser.add_argument("--calibration-blend", type=float, default=DEFAULT_CALIBRATION_BLEND)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    output_dir = v11.unique_output_dir(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, _ = build_from_zip(
        args.worldcupsai_zip,
        train_csv=args.team_train,
        test_csv=args.team_test,
        box_csv=args.box_data,
        results_csv=args.results_data,
        former_names_csv=args.former_names,
        prediction_year=args.prediction_year,
        results_as_of=args.results_as_of,
        total_calibration_strength=args.total_calibration_strength,
        total_multiplier_clip_low=args.total_multiplier_clip_low,
        total_multiplier_clip_high=args.total_multiplier_clip_high,
        total_smoothing=args.total_smoothing,
        min_bin_support=args.min_bin_support,
        calibration_blend=args.calibration_blend,
    )
    prediction = model.predict(
        args.team_a,
        args.team_b,
        host_a=args.host_a,
        host_b=args.host_b,
        knockout=args.knockout,
    )
    (output_dir / "single_match_prediction.json").write_text(
        json.dumps(prediction, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(prediction["top_scorelines"]).to_csv(
        output_dir / "scoreline_probabilities_top.csv",
        index=False,
    )
    pd.DataFrame(prediction["scoreline_probabilities"]).to_csv(
        output_dir / "scoreline_probabilities.csv",
        index=False,
    )
    (output_dir / "model_summary.json").write_text(
        json.dumps(
            {
                "version": "v27-total-goals-calibrated",
                "base_model": "v20-scoreline-ensemble",
                "team_a": prediction["team_a"],
                "team_b": prediction["team_b"],
                "result_probabilities": prediction["result_probabilities"],
                "v27_adjustments": prediction["v27_adjustments"],
                "expanded_training_data": model.training_data_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not args.no_plots:
        v11.plot_prediction_outputs(prediction, output_dir)
    print(
        json.dumps(
            {
                "result_probabilities": prediction["result_probabilities"],
                "predicted_result": prediction["predicted_result"],
                "lambda_a": prediction["lambda_a"],
                "lambda_b": prediction["lambda_b"],
                "top_scorelines": prediction["top_scorelines"][:5],
                "v27_adjustments": prediction["v27_adjustments"],
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
'''
v27_total_goals_calibrated_model = _load_submodule("v27_total_goals_calibrated_model", _V27_TOTAL_GOALS_CALIBRATED_MODEL_SOURCE, "core_engine.py:v27_total_goals_calibrated_model")

if __name__ == "__main__":
    # Preserves `python v11_wcq_results_model.py ...` CLI behavior now that
    # v11 lives as a bundled sub-module rather than its own top-level file.
    v11_wcq_results_model.main()

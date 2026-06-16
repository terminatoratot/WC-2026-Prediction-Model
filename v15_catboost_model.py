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
    as_of: str | pd.Timestamp = "2026-06-10",
) -> pd.DataFrame:
    """Load completed internationals up to an explicit leakage cutoff."""
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
    cutoff = pd.Timestamp(as_of).normalize()
    return (
        results.dropna(
            subset=["date", "team_a", "team_b", "goals_a", "goals_b"]
        )
        .loc[lambda frame: frame["date"] <= cutoff]
        .sort_values(["date"], kind="stable")
        .reset_index(drop=True)
    )


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
    results_as_of="2026-06-10",
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
            results_as_of,
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
            results_as_of,
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
        "results_as_of": str(pd.Timestamp(results_as_of).date()),
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
        default="2026-06-10",
        help=(
            "Use results on or before this date for live international "
            "state; defaults to the day before the observed WC 2026 sample."
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

#!/usr/bin/env python3
"""V30: V29 plus conservative player-role form and matchup adjustments.

Run:
    .venv/bin/python v30_player_role_form_model.py --team-a "Switzerland" --team-b "Bosnia and Herzegovina"
    .venv/bin/python v30_player_role_form_model.py --team-a "Switzerland" --team-b "Bosnia and Herzegovina" --no-plots
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

import v11_wcq_results_model as v11
import v15_catboost_model as v15
import v20_scoreline_ensemble_model as v20
import v26_top3_coverage_model as v26
import v27_total_goals_calibrated_model as v27
import v28_current_worldcup_form_model as v28
import v29_tail_risk_scoreline_model as v29


ScoreMatrix = Dict[Tuple[int, int], float]

DEFAULT_ROLE_FORM_WDL_BLEND = 0.14
DEFAULT_ROLE_FORM_SCORELINE_BLEND = 0.30
DEFAULT_BETA_ROLE_ATTACK_EDGE = 0.055
DEFAULT_BETA_SET_PIECE_EDGE = 0.020
DEFAULT_BETA_KEEPER_FRAGILITY = 0.025
DEFAULT_BETA_DISCIPLINE_EDGE = 0.015
DEFAULT_MAX_ROLE_LOG_ADJUSTMENT = 0.10

ATTACKER_STATS = {
    "Top scorer": 0.65,
    "Goals per 90": 0.65,
    "Goals per match": 0.65,
    "Expected goals (xG)": 1.00,
    "xG per 90": 1.00,
    "Expected goals on target (xGOT)": 1.00,
    "Shots per 90": 0.55,
    "Shots per match": 0.55,
    "Shots on target per 90": 0.75,
    "Shots on target per match": 0.75,
    "Big chances": 0.80,
    "Touches in opposition box": 0.55,
    "Penalties awarded": 0.25,
    "Big chances missed": -0.35,
}
CREATOR_STATS = {
    "Assists": 0.45,
    "Expected assists (xA)": 1.00,
    "xA per 90": 1.00,
    "xG + xA per 90": 0.80,
    "Chances created": 0.80,
    "Big chances created": 0.95,
    "Accurate crosses per match": 0.40,
    "Accurate long balls per 90": 0.35,
    "Accurate long balls per match": 0.35,
    "Accurate passes per 90": 0.25,
    "Accurate passes per match": 0.25,
    "Successful dribbles per 90": 0.45,
}
DEFENDER_STATS = {
    "Clean sheets": 0.45,
    "Defensive contributions per 90": 0.85,
    "Tackles per 90": 0.60,
    "Tackles per match": 0.60,
    "Interceptions per 90": 0.65,
    "Interceptions per match": 0.65,
    "Clearances per 90": 0.50,
    "Clearances per match": 0.50,
    "Blocks per 90": 0.55,
    "Recoveries per 90": 0.45,
    "Possession won final 3rd per 90": 0.40,
    "Possession won final 3rd per match": 0.40,
}
KEEPER_STATS = {
    "Save percentage": 0.70,
    "Goals prevented": 1.00,
    "Saves per 90": 0.45,
    "Saves per match": 0.45,
}
FRAGILITY_STATS = {
    "xG conceded": 1.00,
    "Goals conceded per 90": 0.55,
    "Goals conceded per match": 0.55,
    "Penalties conceded": 0.35,
}
SET_PIECE_FOR_STATS = {
    "Set piece goals": 0.80,
    "Corners": 0.45,
    "Accurate crosses per match": 0.30,
}
SET_PIECE_AGAINST_STATS = {
    "Set piece goals conceded": 0.80,
}
DISCIPLINE_STATS = {
    "Yellow cards": 0.40,
    "Red cards": 0.90,
    "Fouls committed per 90": 0.45,
    "Fouls per match": 0.45,
}
MINUTES_STATS = {"Minutes played", "FotMob rating"}


@dataclass
class PlayerRoleProfile:
    team: str
    attacker: float = 0.0
    creator: float = 0.0
    defender: float = 0.0
    keeper: float = 0.0
    defensive_fragility: float = 0.0
    set_piece_for: float = 0.0
    set_piece_against: float = 0.0
    discipline_risk: float = 0.0
    finishing_delta: float = 0.0
    coverage: float = 0.0
    rows: int = 0
    minutes_rows: int = 0

    @property
    def attack_unit(self) -> float:
        return float(0.60 * self.attacker + 0.40 * self.creator)

    @property
    def defense_unit(self) -> float:
        return float(0.65 * self.defender + 0.35 * self.keeper)

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "team": self.team,
            "attacker": self.attacker,
            "creator": self.creator,
            "defender": self.defender,
            "keeper": self.keeper,
            "defensive_fragility": self.defensive_fragility,
            "set_piece_for": self.set_piece_for,
            "set_piece_against": self.set_piece_against,
            "discipline_risk": self.discipline_risk,
            "finishing_delta": self.finishing_delta,
            "attack_unit": self.attack_unit,
            "defense_unit": self.defense_unit,
            "coverage": self.coverage,
            "rows": int(self.rows),
            "minutes_rows": int(self.minutes_rows),
        }


def parse_number(value: object) -> float:
    return v28.parse_number(value)


def zscore(series: pd.Series) -> pd.Series:
    return v28.zscore(series)


def normalized_stat_contribution(frame: pd.DataFrame, stat: str) -> pd.Series:
    return v28.normalized_stat_contribution(frame, stat)


def stat_bucket(stat: str) -> tuple[str | None, float]:
    for bucket_name, mapping in (
        ("attacker", ATTACKER_STATS),
        ("creator", CREATOR_STATS),
        ("defender", DEFENDER_STATS),
        ("keeper", KEEPER_STATS),
        ("defensive_fragility", FRAGILITY_STATS),
        ("set_piece_for", SET_PIECE_FOR_STATS),
        ("set_piece_against", SET_PIECE_AGAINST_STATS),
        ("discipline_risk", DISCIPLINE_STATS),
    ):
        if stat in mapping:
            return bucket_name, float(mapping[stat])
    return None, 0.0


def normalize_player_name(name: object) -> str:
    text = str(name).lower().strip()
    text = "".join(
        char
        for char in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(char)
    )
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def player_match_score(player_name: object, roster_name: object) -> float:
    player = normalize_player_name(player_name)
    roster = normalize_player_name(roster_name)
    if not player or not roster:
        return 0.0
    player_tokens = player.split()
    roster_tokens = roster.split()
    if player == roster:
        return 100.0
    if roster in player or player in roster:
        return 90.0
    if player_tokens[-1] == roster_tokens[-1]:
        return 80.0
    overlap = len(set(player_tokens) & set(roster_tokens))
    return 50.0 + 10.0 * overlap if overlap else 0.0


def attach_roster_context(
    player_stats: pd.DataFrame,
    lineups: pd.DataFrame,
    substitutions: pd.DataFrame,
    min_score: float = 70.0,
) -> pd.DataFrame:
    roster_columns = [
        "match_id",
        "team",
        "player",
        "position",
        "squad_role",
        "minute_on",
        "minute_off",
    ]
    roster_parts = []
    for frame in (lineups, substitutions):
        if not frame.empty and set(roster_columns).issubset(frame.columns):
            roster_parts.append(frame[roster_columns])
    if not roster_parts:
        enriched = player_stats.copy()
        enriched["team"] = ""
        enriched["position"] = ""
        enriched["roster_match_score"] = 0.0
        return enriched

    roster = pd.concat(roster_parts, ignore_index=True).drop_duplicates()
    rows = []
    for match_id, group in player_stats.groupby("match_id", dropna=False):
        choices = roster[roster["match_id"].eq(match_id)]
        for _, row in group.iterrows():
            best = None
            best_score = 0.0
            for _, candidate in choices.iterrows():
                score = player_match_score(row["player"], candidate["player"])
                if score > best_score:
                    best = candidate
                    best_score = score
            record = row.to_dict()
            if best is not None and best_score >= min_score:
                record["team"] = best["team"]
                record["position"] = best["position"]
                record["squad_role"] = best.get("squad_role", "")
                record["lineup_minute_on"] = best.get("minute_on", np.nan)
                record["lineup_minute_off"] = best.get("minute_off", np.nan)
                record["roster_player"] = best["player"]
                record["roster_match_score"] = float(best_score)
            else:
                record["team"] = ""
                record["position"] = ""
                record["squad_role"] = ""
                record["lineup_minute_on"] = np.nan
                record["lineup_minute_off"] = np.nan
                record["roster_player"] = ""
                record["roster_match_score"] = float(best_score)
            rows.append(record)
    return pd.DataFrame(rows)


def aggregate_player_match_stats(player_stats: pd.DataFrame) -> pd.DataFrame:
    if player_stats.empty:
        return player_stats
    numeric_columns = [
        column
        for column in player_stats.columns
        if column
        not in {
            "match_id",
            "match_slug",
            "match_url",
            "title",
            "home_team",
            "away_team",
            "kickoff",
            "round",
            "venue",
            "referee",
            "status",
            "category",
            "player",
        }
        and pd.api.types.is_numeric_dtype(player_stats[column])
    ]
    aggregations = {column: "max" for column in numeric_columns}
    for column in (
        "match_slug",
        "match_url",
        "title",
        "home_team",
        "away_team",
        "kickoff",
        "round",
        "status",
    ):
        if column in player_stats:
            aggregations[column] = "first"
    return (
        player_stats.groupby(["match_id", "player"], as_index=False)
        .agg(aggregations)
        .copy()
    )


def build_match_player_role_profiles(
    player_stats_csv: str | Path | None,
    lineups_csv: str | Path | None,
    substitutions_csv: str | Path | None,
    keeper_stats_csv: str | Path | None,
) -> dict[str, PlayerRoleProfile]:
    paths = [player_stats_csv, lineups_csv, substitutions_csv]
    if not all(path and Path(path).exists() for path in paths):
        return {}
    player_stats = pd.read_csv(player_stats_csv)
    lineups = pd.read_csv(lineups_csv)
    substitutions = pd.read_csv(substitutions_csv)
    if player_stats.empty or lineups.empty:
        return {}

    player_stats = aggregate_player_match_stats(player_stats)
    enriched = attach_roster_context(player_stats, lineups, substitutions)
    enriched = enriched[enriched["team"].astype(str).ne("")]
    if enriched.empty:
        return {}

    numeric = [
        "minutes_played",
        "fotmob_rating",
        "goals",
        "assists",
        "xg",
        "xa",
        "xgot",
        "total_shots",
        "shots_on_target",
        "touches_in_opposition_box",
        "successful_dribbles",
        "big_chances_missed",
        "chances_created",
        "passes_into_final_third",
        "accurate_crosses",
        "accurate_long_balls",
        "defensive_contributions",
        "tackles",
        "interceptions",
        "blocks",
        "recoveries",
        "clearances",
        "fouls_committed",
    ]
    for column in numeric:
        if column not in enriched:
            enriched[column] = 0.0
        enriched[column] = pd.to_numeric(enriched[column], errors="coerce").fillna(0.0)

    minutes = enriched["minutes_played"].where(enriched["minutes_played"] > 0, 90.0)
    minute_weight = np.sqrt(np.clip(minutes / 90.0, 0.05, 1.25))
    position = enriched["position"].astype(str).str.upper()
    attacker_position = position.isin(["FWD", "FW", "ST", "CF"])
    creator_position = position.isin(["AM", "MID", "MF", "CM", "DM", "W"])
    defender_position = position.isin(["DEF", "DF", "CB", "LB", "RB", "WB"])
    keeper_position = position.isin(["GK", "KEEPER"])

    enriched["attacker_raw"] = minute_weight * (
        (1.00 * enriched["xg"] + 0.80 * enriched["xgot"])
        + 0.22 * enriched["shots_on_target"]
        + 0.12 * enriched["total_shots"]
        + 0.08 * enriched["touches_in_opposition_box"]
        + 0.05 * (enriched["fotmob_rating"] - 6.5).clip(lower=-1.5)
        - 0.25 * enriched["big_chances_missed"]
    ) * np.where(attacker_position | creator_position, 1.0, 0.40)
    enriched["creator_raw"] = minute_weight * (
        1.00 * enriched["xa"]
        + 0.28 * enriched["chances_created"]
        + 0.08 * enriched["passes_into_final_third"]
        + 0.10 * enriched["accurate_crosses"]
        + 0.08 * enriched["successful_dribbles"]
        + 0.04 * (enriched["fotmob_rating"] - 6.5).clip(lower=-1.5)
    ) * np.where(attacker_position | creator_position, 1.0, 0.45)
    enriched["defender_raw"] = minute_weight * (
        0.20 * enriched["defensive_contributions"]
        + 0.22 * enriched["tackles"]
        + 0.28 * enriched["interceptions"]
        + 0.28 * enriched["blocks"]
        + 0.08 * enriched["recoveries"]
        + 0.12 * enriched["clearances"]
        + 0.04 * (enriched["fotmob_rating"] - 6.5).clip(lower=-1.5)
    ) * np.where(defender_position | creator_position, 1.0, 0.35)
    enriched["set_piece_for_raw"] = minute_weight * (
        0.16 * enriched["accurate_crosses"]
        + 0.10 * enriched["accurate_long_balls"]
        + 0.30 * enriched["xa"]
    )
    enriched["discipline_raw"] = minute_weight * (0.20 * enriched["fouls_committed"])

    team = (
        enriched.groupby("team", as_index=False)
        .agg(
            rows=("player", "size"),
            matches=("match_id", "nunique"),
            attacker=("attacker_raw", "sum"),
            creator=("creator_raw", "sum"),
            defender=("defender_raw", "sum"),
            set_piece_for=("set_piece_for_raw", "sum"),
            discipline_risk=("discipline_raw", "sum"),
            minutes_rows=("minutes_played", lambda s: int((s > 0).sum())),
        )
        .copy()
    )

    for column in [
        "attacker",
        "creator",
        "defender",
        "set_piece_for",
        "discipline_risk",
    ]:
        team[column] = zscore(team[column])
    team["keeper"] = 0.0
    team["defensive_fragility"] = 0.0
    team["set_piece_against"] = 0.0

    if keeper_stats_csv and Path(keeper_stats_csv).exists():
        keeper = pd.read_csv(keeper_stats_csv)
        if not keeper.empty and {"match_id", "player"}.issubset(keeper.columns):
            keeper_agg = aggregate_player_match_stats(keeper)
            keeper_enriched = attach_roster_context(keeper_agg, lineups, substitutions)
            keeper_enriched = keeper_enriched[keeper_enriched["team"].astype(str).ne("")]
            for column in [
                "saves",
                "goals_conceded",
                "xgot_faced",
                "goals_prevented",
                "acted_as_sweeper",
                "high_claim",
            ]:
                if column not in keeper_enriched:
                    keeper_enriched[column] = 0.0
                keeper_enriched[column] = pd.to_numeric(
                    keeper_enriched[column], errors="coerce"
                ).fillna(0.0)
            if not keeper_enriched.empty:
                keeper_team = (
                    keeper_enriched.groupby("team", as_index=False)
                    .agg(
                        keeper=(
                            "goals_prevented",
                            "sum",
                        ),
                        xgot_faced=("xgot_faced", "sum"),
                        goals_conceded=("goals_conceded", "sum"),
                        saves=("saves", "sum"),
                    )
                    .copy()
                )
                keeper_team["keeper"] = zscore(
                    keeper_team["keeper"] + 0.08 * keeper_team["saves"]
                )
                keeper_team["defensive_fragility"] = zscore(
                    keeper_team["xgot_faced"]
                    + 0.45 * keeper_team["goals_conceded"]
                    - 0.35 * keeper_team["saves"]
                )
                team = team.merge(
                    keeper_team[["team", "keeper", "defensive_fragility"]],
                    on="team",
                    how="left",
                    suffixes=("", "_keeper"),
                )
                team["keeper"] = team["keeper_keeper"].fillna(team["keeper"])
                team["defensive_fragility"] = team[
                    "defensive_fragility_keeper"
                ].fillna(team["defensive_fragility"])
                team = team.drop(
                    columns=["keeper_keeper", "defensive_fragility_keeper"]
                )

    coverage = np.sqrt(team["rows"] / (team["rows"] + 8.0))
    coverage *= np.sqrt(team["matches"] / (team["matches"] + 2.0))
    team["coverage"] = np.clip(coverage, 0.0, 1.0)
    team["finishing_delta"] = zscore(team["attacker"] - 0.65 * team["creator"])

    profiles: dict[str, PlayerRoleProfile] = {}
    for row in team.to_dict(orient="records"):
        coverage_value = float(row.get("coverage", 0.0) or 0.0)
        profiles[str(row["team"])] = PlayerRoleProfile(
            team=str(row["team"]),
            attacker=float(row.get("attacker", 0.0)) * coverage_value,
            creator=float(row.get("creator", 0.0)) * coverage_value,
            defender=float(row.get("defender", 0.0)) * coverage_value,
            keeper=float(row.get("keeper", 0.0)) * coverage_value,
            defensive_fragility=float(row.get("defensive_fragility", 0.0))
            * coverage_value,
            set_piece_for=float(row.get("set_piece_for", 0.0)) * coverage_value,
            set_piece_against=float(row.get("set_piece_against", 0.0))
            * coverage_value,
            discipline_risk=float(row.get("discipline_risk", 0.0))
            * coverage_value,
            finishing_delta=float(row.get("finishing_delta", 0.0)) * coverage_value,
            coverage=coverage_value,
            rows=int(row.get("rows", 0) or 0),
            minutes_rows=int(row.get("minutes_rows", 0) or 0),
        )
    return profiles


def merge_role_profiles(
    leaderboard_profiles: dict[str, PlayerRoleProfile],
    match_profiles: dict[str, PlayerRoleProfile],
) -> dict[str, PlayerRoleProfile]:
    teams = set(leaderboard_profiles) | set(match_profiles)
    merged: dict[str, PlayerRoleProfile] = {}
    fields = [
        "attacker",
        "creator",
        "defender",
        "keeper",
        "defensive_fragility",
        "set_piece_for",
        "set_piece_against",
        "discipline_risk",
        "finishing_delta",
    ]
    for team in teams:
        leader = leaderboard_profiles.get(team, PlayerRoleProfile(team=team))
        match = match_profiles.get(team, PlayerRoleProfile(team=team))
        match_weight = 0.70 * min(max(match.coverage, 0.0), 1.0)
        values = {
            field: (1.0 - match_weight) * getattr(leader, field)
            + match_weight * getattr(match, field)
            for field in fields
        }
        merged[team] = PlayerRoleProfile(
            team=team,
            **values,
            coverage=max(leader.coverage, match.coverage),
            rows=leader.rows + match.rows,
            minutes_rows=leader.minutes_rows + match.minutes_rows,
        )
    return merged


def build_player_role_profiles(path: str | Path | None) -> dict[str, PlayerRoleProfile]:
    if not path or not Path(path).exists():
        return {}
    frame = pd.read_csv(path)
    required = {"stat", "country_or_team", "value"}
    if frame.empty or not required.issubset(frame.columns):
        return {}

    clean = frame.copy()
    clean["team"] = clean["country_or_team"].map(v28.canon_team)
    clean = clean[clean["team"].ne("")]
    if clean.empty:
        return {}
    clean["numeric_value"] = clean["value"].map(parse_number)

    records: list[dict[str, Any]] = []
    for stat in sorted(clean["stat"].dropna().unique()):
        bucket, weight = stat_bucket(str(stat))
        if bucket is None:
            continue
        contributions = normalized_stat_contribution(clean, str(stat))
        if contributions.empty:
            continue
        stat_frame = clean.loc[contributions.index]
        for idx, contribution in contributions.items():
            records.append(
                {
                    "team": stat_frame.loc[idx, "team"],
                    "bucket": bucket,
                    "contribution": float(contribution) * weight,
                }
            )

    if not records:
        return {}
    contrib = pd.DataFrame(records)
    pivot = contrib.pivot_table(
        index="team",
        columns="bucket",
        values="contribution",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()
    buckets = [
        "attacker",
        "creator",
        "defender",
        "keeper",
        "defensive_fragility",
        "set_piece_for",
        "set_piece_against",
        "discipline_risk",
    ]
    for bucket in buckets:
        if bucket not in pivot:
            pivot[bucket] = 0.0
        pivot[bucket] = zscore(pivot[bucket])

    counts = contrib.groupby("team").size().rename("rows").reset_index()
    minutes = (
        clean[clean["stat"].isin(MINUTES_STATS)]
        .groupby("team")
        .size()
        .rename("minutes_rows")
        .reset_index()
    )
    pivot = pivot.merge(counts, on="team", how="left")
    pivot = pivot.merge(minutes, on="team", how="left")
    pivot["minutes_rows"] = pivot["minutes_rows"].fillna(0)
    base_coverage = np.sqrt(pivot["rows"] / (pivot["rows"] + 5.0))
    minutes_coverage = 1.0 + 0.08 * np.minimum(pivot["minutes_rows"], 3.0)
    pivot["coverage"] = np.clip(base_coverage * minutes_coverage, 0.0, 1.0)

    if {"attacker", "defensive_fragility"}.issubset(pivot.columns):
        pivot["finishing_delta"] = zscore(pivot["attacker"] - 0.45 * pivot["creator"])
    else:
        pivot["finishing_delta"] = 0.0

    profiles: dict[str, PlayerRoleProfile] = {}
    for row in pivot.to_dict(orient="records"):
        coverage = float(row.get("coverage", 0.0) or 0.0)
        profiles[str(row["team"])] = PlayerRoleProfile(
            team=str(row["team"]),
            attacker=float(row.get("attacker", 0.0)) * coverage,
            creator=float(row.get("creator", 0.0)) * coverage,
            defender=float(row.get("defender", 0.0)) * coverage,
            keeper=float(row.get("keeper", 0.0)) * coverage,
            defensive_fragility=float(row.get("defensive_fragility", 0.0)) * coverage,
            set_piece_for=float(row.get("set_piece_for", 0.0)) * coverage,
            set_piece_against=float(row.get("set_piece_against", 0.0)) * coverage,
            discipline_risk=float(row.get("discipline_risk", 0.0)) * coverage,
            finishing_delta=float(row.get("finishing_delta", 0.0)) * coverage,
            coverage=coverage,
            rows=int(row.get("rows", 0) or 0),
            minutes_rows=int(row.get("minutes_rows", 0) or 0),
        )
    return profiles


def score_matrix_from_prediction(prediction: Dict[str, Any]) -> ScoreMatrix:
    return v29.score_matrix_from_prediction(prediction)


def blend_result_probabilities(
    base: Dict[str, float],
    adjusted: Dict[str, float],
    adjusted_weight: float,
) -> Dict[str, float]:
    return v28.blend_result_probabilities(base, adjusted, adjusted_weight)


def expected_goals(score_matrix: ScoreMatrix) -> tuple[float, float]:
    return v28.expected_goals(score_matrix)


class V30PlayerRoleFormModel:
    """Wrap V29 with small player-role matchup adjustments."""

    def __init__(
        self,
        base_model: v29.V29TailRiskScorelineModel,
        role_profiles: dict[str, PlayerRoleProfile],
        role_form_wdl_blend: float = DEFAULT_ROLE_FORM_WDL_BLEND,
        role_form_scoreline_blend: float = DEFAULT_ROLE_FORM_SCORELINE_BLEND,
        beta_role_attack_edge: float = DEFAULT_BETA_ROLE_ATTACK_EDGE,
        beta_set_piece_edge: float = DEFAULT_BETA_SET_PIECE_EDGE,
        beta_keeper_fragility: float = DEFAULT_BETA_KEEPER_FRAGILITY,
        beta_discipline_edge: float = DEFAULT_BETA_DISCIPLINE_EDGE,
        max_role_log_adjustment: float = DEFAULT_MAX_ROLE_LOG_ADJUSTMENT,
    ):
        self.base_model = base_model
        self.role_profiles = role_profiles
        self.role_form_wdl_blend = float(np.clip(role_form_wdl_blend, 0.0, 1.0))
        self.role_form_scoreline_blend = float(
            np.clip(role_form_scoreline_blend, 0.0, 1.0)
        )
        self.beta_role_attack_edge = float(beta_role_attack_edge)
        self.beta_set_piece_edge = float(beta_set_piece_edge)
        self.beta_keeper_fragility = float(beta_keeper_fragility)
        self.beta_discipline_edge = float(beta_discipline_edge)
        self.max_role_log_adjustment = float(max(max_role_log_adjustment, 0.0))
        self.training_data_summary = getattr(base_model, "training_data_summary", {})

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_model, name)

    def role_for_team(self, team: object) -> PlayerRoleProfile:
        canonical = v28.canon_team(team)
        return self.role_profiles.get(canonical, PlayerRoleProfile(team=canonical))

    def _role_adjusted_matrix(
        self,
        prediction: Dict[str, Any],
        role_a: PlayerRoleProfile,
        role_b: PlayerRoleProfile,
        max_goals: int,
    ) -> tuple[ScoreMatrix, Dict[str, Any]]:
        base_lambda_a = float(prediction["lambda_a"])
        base_lambda_b = float(prediction["lambda_b"])
        attack_edge_a = role_a.attack_unit - role_b.defense_unit
        attack_edge_b = role_b.attack_unit - role_a.defense_unit
        set_piece_edge_a = role_a.set_piece_for + role_b.set_piece_against
        set_piece_edge_b = role_b.set_piece_for + role_a.set_piece_against
        fragility_edge_a = role_b.defensive_fragility - 0.35 * role_b.keeper
        fragility_edge_b = role_a.defensive_fragility - 0.35 * role_a.keeper
        discipline_edge_a = role_b.discipline_risk - 0.25 * role_a.discipline_risk
        discipline_edge_b = role_a.discipline_risk - 0.25 * role_b.discipline_risk

        log_a = (
            self.beta_role_attack_edge * attack_edge_a
            + self.beta_set_piece_edge * set_piece_edge_a
            + self.beta_keeper_fragility * fragility_edge_a
            + self.beta_discipline_edge * discipline_edge_a
        )
        log_b = (
            self.beta_role_attack_edge * attack_edge_b
            + self.beta_set_piece_edge * set_piece_edge_b
            + self.beta_keeper_fragility * fragility_edge_b
            + self.beta_discipline_edge * discipline_edge_b
        )
        log_a = float(
            np.clip(log_a, -self.max_role_log_adjustment, self.max_role_log_adjustment)
        )
        log_b = float(
            np.clip(log_b, -self.max_role_log_adjustment, self.max_role_log_adjustment)
        )
        lambda_a = float(np.clip(base_lambda_a * math.exp(log_a), 0.15, 5.5))
        lambda_b = float(np.clip(base_lambda_b * math.exp(log_b), 0.15, 5.5))
        matrix = v11.poisson_score_matrix(lambda_a, lambda_b, max_goals)
        rho = prediction.get("calibration_notes", {}).get("dixon_coles_rho", -0.08)
        matrix = v11.apply_dixon_coles_adjustment(matrix, lambda_a, lambda_b, rho=rho)
        return matrix, {
            "role_base_lambda_a": base_lambda_a,
            "role_base_lambda_b": base_lambda_b,
            "role_lambda_a": lambda_a,
            "role_lambda_b": lambda_b,
            "role_log_adjustment_a": log_a,
            "role_log_adjustment_b": log_b,
            "role_attack_edge_a": float(attack_edge_a),
            "role_attack_edge_b": float(attack_edge_b),
            "set_piece_edge_a": float(set_piece_edge_a),
            "set_piece_edge_b": float(set_piece_edge_b),
            "fragility_edge_a": float(fragility_edge_a),
            "fragility_edge_b": float(fragility_edge_b),
            "discipline_edge_a": float(discipline_edge_a),
            "discipline_edge_b": float(discipline_edge_b),
        }

    def predict(self, *args, **kwargs) -> Dict[str, Any]:
        team_a = kwargs.get("team_a", args[0] if args else "")
        team_b = kwargs.get("team_b", args[1] if len(args) > 1 else "")
        max_goals = int(kwargs.get("max_goals", args[5] if len(args) > 5 else 10))
        prediction = self.base_model.predict(*args, **kwargs)
        base_matrix = score_matrix_from_prediction(prediction)
        base_result_probabilities = dict(prediction["result_probabilities"])
        role_a = self.role_for_team(team_a)
        role_b = self.role_for_team(team_b)

        role_matrix, role_diagnostics = self._role_adjusted_matrix(
            prediction,
            role_a,
            role_b,
            max_goals=max_goals,
        )
        role_result_probabilities = v11.result_probs(role_matrix)
        final_result_probabilities = blend_result_probabilities(
            base_result_probabilities,
            role_result_probabilities,
            self.role_form_wdl_blend,
        )
        score_matrix = v20.blend_score_matrices(
            base_matrix,
            role_matrix,
            adjusted_weight=self.role_form_scoreline_blend,
        )
        score_matrix = v11.reweight_score_matrix_to_results(
            score_matrix,
            final_result_probabilities,
        )
        lambda_a, lambda_b = expected_goals(score_matrix)
        prediction["lambda_a"] = lambda_a
        prediction["lambda_b"] = lambda_b
        prediction["result_probabilities"] = final_result_probabilities
        prediction["predicted_result"] = max(
            final_result_probabilities,
            key=final_result_probabilities.get,
        )
        prediction.update(v15.score_outputs(score_matrix, max_goals))
        top_scorelines, tail_diagnostics = v29.select_top_scorelines_with_tail_risk(
            score_matrix,
            final_result_probabilities,
            lambda_a,
            lambda_b,
            current_top_scorelines=prediction.get("top_scorelines", []),
            top_n=15,
            favorite_win_gate=self.base_model.favorite_win_gate,
            extreme_favorite_win_gate=self.base_model.extreme_favorite_win_gate,
            draw_ceiling=self.base_model.draw_ceiling,
            favorite_lambda_gate=self.base_model.favorite_lambda_gate,
            extreme_lambda_gate=self.base_model.extreme_lambda_gate,
            lambda_gap_gate=self.base_model.lambda_gap_gate,
            total_lambda_gate=self.base_model.total_lambda_gate,
            relative_floor=self.base_model.relative_floor,
            absolute_floor=self.base_model.absolute_floor,
            max_winner_goals=self.base_model.max_winner_goals,
        )
        prediction["top_scorelines"] = top_scorelines
        prediction["v30_adjustments"] = {
            "base_model": "v29_tail_risk_scoreline",
            "scoreline_policy": "player_role_form_plus_tail_risk",
            "role_layer_affects_wdl": True,
            "role_form_wdl_blend": self.role_form_wdl_blend,
            "role_form_scoreline_blend": self.role_form_scoreline_blend,
            "beta_role_attack_edge": self.beta_role_attack_edge,
            "beta_set_piece_edge": self.beta_set_piece_edge,
            "beta_keeper_fragility": self.beta_keeper_fragility,
            "beta_discipline_edge": self.beta_discipline_edge,
            "max_role_log_adjustment": self.max_role_log_adjustment,
            "team_a_role_profile": role_a.diagnostics(),
            "team_b_role_profile": role_b.diagnostics(),
            "base_result_probabilities": base_result_probabilities,
            "role_result_probabilities": role_result_probabilities,
            **role_diagnostics,
            "tail_risk": tail_diagnostics,
        }
        prediction["calibration_notes"] = {
            **prediction.get("calibration_notes", {}),
            "v30": prediction["v30_adjustments"],
            "player_role_policy": (
                "V30 starts from V29, adds small coverage-shrunk player-role "
                "matchup adjustments from FotMob match-level player, lineup, "
                "keeper, and leaderboard data, then re-applies the V29 "
                "tail-risk Top-3 selector."
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
    observed_matches_csv=None,
    fotmob_leaders_csv=None,
    fotmob_player_stats_csv=None,
    fotmob_lineups_csv=None,
    fotmob_substitutions_csv=None,
    fotmob_keeper_stats_csv=None,
    include_observed_goals=True,
    include_fotmob_goal_stats=True,
    include_group_score_context=True,
    role_form_wdl_blend=DEFAULT_ROLE_FORM_WDL_BLEND,
    role_form_scoreline_blend=DEFAULT_ROLE_FORM_SCORELINE_BLEND,
    beta_role_attack_edge=DEFAULT_BETA_ROLE_ATTACK_EDGE,
    beta_set_piece_edge=DEFAULT_BETA_SET_PIECE_EDGE,
    beta_keeper_fragility=DEFAULT_BETA_KEEPER_FRAGILITY,
    beta_discipline_edge=DEFAULT_BETA_DISCIPLINE_EDGE,
    max_role_log_adjustment=DEFAULT_MAX_ROLE_LOG_ADJUSTMENT,
):
    data_dir = Path(__file__).resolve().parent / "data"
    fotmob_leaders_csv = fotmob_leaders_csv or (
        data_dir / "fotmob_stat_leaders_clean.csv"
    )
    fotmob_player_stats_csv = fotmob_player_stats_csv or (
        data_dir / "fotmob_match_player_stats_clean.csv"
    )
    fotmob_lineups_csv = fotmob_lineups_csv or (
        data_dir / "fotmob_match_lineups_clean.csv"
    )
    fotmob_substitutions_csv = fotmob_substitutions_csv or (
        data_dir / "fotmob_match_substitutions_clean.csv"
    )
    fotmob_keeper_stats_csv = fotmob_keeper_stats_csv or (
        data_dir / "fotmob_match_keeper_stats_clean.csv"
    )
    base_model, data = v29.build_from_zip(
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
        observed_matches_csv=observed_matches_csv,
        fotmob_leaders_csv=fotmob_leaders_csv,
        include_observed_goals=include_observed_goals,
        include_fotmob_goal_stats=include_fotmob_goal_stats,
        include_group_score_context=include_group_score_context,
    )
    leaderboard_profiles = build_player_role_profiles(fotmob_leaders_csv)
    match_profiles = build_match_player_role_profiles(
        fotmob_player_stats_csv,
        fotmob_lineups_csv,
        fotmob_substitutions_csv,
        fotmob_keeper_stats_csv,
    )
    role_profiles = merge_role_profiles(leaderboard_profiles, match_profiles)
    model = V30PlayerRoleFormModel(
        base_model,
        role_profiles,
        role_form_wdl_blend=role_form_wdl_blend,
        role_form_scoreline_blend=role_form_scoreline_blend,
        beta_role_attack_edge=beta_role_attack_edge,
        beta_set_piece_edge=beta_set_piece_edge,
        beta_keeper_fragility=beta_keeper_fragility,
        beta_discipline_edge=beta_discipline_edge,
        max_role_log_adjustment=max_role_log_adjustment,
    )
    model.training_data_summary = {
        **getattr(base_model, "training_data_summary", {}),
        "v30_role_profile_teams": len(role_profiles),
        "v30_leaderboard_role_profile_teams": len(leaderboard_profiles),
        "v30_match_role_profile_teams": len(match_profiles),
        "v30_fotmob_leaders_csv": str(fotmob_leaders_csv),
        "v30_fotmob_player_stats_csv": str(fotmob_player_stats_csv),
        "v30_fotmob_lineups_csv": str(fotmob_lineups_csv),
        "v30_fotmob_substitutions_csv": str(fotmob_substitutions_csv),
        "v30_fotmob_keeper_stats_csv": str(fotmob_keeper_stats_csv),
        "v30_role_form_wdl_blend": model.role_form_wdl_blend,
        "v30_role_form_scoreline_blend": model.role_form_scoreline_blend,
        "v30_beta_role_attack_edge": model.beta_role_attack_edge,
        "v30_beta_set_piece_edge": model.beta_set_piece_edge,
        "v30_beta_keeper_fragility": model.beta_keeper_fragility,
        "v30_beta_discipline_edge": model.beta_discipline_edge,
        "v30_max_role_log_adjustment": model.max_role_log_adjustment,
    }
    return model, data


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    parser = argparse.ArgumentParser(
        description="Run V30: V29 with player-role form adjustments."
    )
    parser.add_argument("--team-a", required=True)
    parser.add_argument("--team-b", required=True)
    parser.add_argument("--host-a", action="store_true")
    parser.add_argument("--host-b", action="store_true")
    parser.add_argument("--knockout", action="store_true")
    parser.add_argument("--outdir", default="outputs/outputs_v30_player_role_form")
    parser.add_argument("--worldcupsai-zip", default=str(data_dir / "worldcupsai.zip"))
    parser.add_argument("--team-train", default=str(data_dir / "current_team_features_2026.csv"))
    parser.add_argument("--team-test")
    parser.add_argument("--box-data", default=str(data_dir / "FIFAallMatchBoxData.csv"))
    parser.add_argument("--results-data", default=str(data_dir / "results.csv"))
    parser.add_argument("--results-as-of", default=v15.DEFAULT_RESULTS_AS_OF)
    parser.add_argument("--former-names", default=str(data_dir / "former_names.csv"))
    parser.add_argument("--prediction-year", type=int, default=2026)
    parser.add_argument("--player-ratings", default=str(data_dir / "player_ratings_international.csv"))
    parser.add_argument("--declared-squads", default=str(data_dir / "world_cup_2026_declared_squads.csv"))
    parser.add_argument("--fcratings", default=str(data_dir / "fcratings_top50_worldcup2026.csv"))
    parser.add_argument("--observed-matches", default=str(data_dir / "wc2026_observed_matches_from_screenshots.csv"))
    parser.add_argument("--fotmob-leaders", default=str(data_dir / "fotmob_stat_leaders_clean.csv"))
    parser.add_argument("--fotmob-player-stats", default=str(data_dir / "fotmob_match_player_stats_clean.csv"))
    parser.add_argument("--fotmob-lineups", default=str(data_dir / "fotmob_match_lineups_clean.csv"))
    parser.add_argument("--fotmob-substitutions", default=str(data_dir / "fotmob_match_substitutions_clean.csv"))
    parser.add_argument("--fotmob-keeper-stats", default=str(data_dir / "fotmob_match_keeper_stats_clean.csv"))
    parser.add_argument("--blind-current-goals", action="store_true")
    parser.add_argument("--ignore-observed-goals", action="store_true")
    parser.add_argument("--ignore-fotmob-goal-stats", action="store_true")
    parser.add_argument("--disable-group-score-context", action="store_true")
    parser.add_argument("--role-form-wdl-blend", type=float, default=DEFAULT_ROLE_FORM_WDL_BLEND)
    parser.add_argument("--role-form-scoreline-blend", type=float, default=DEFAULT_ROLE_FORM_SCORELINE_BLEND)
    parser.add_argument("--beta-role-attack-edge", type=float, default=DEFAULT_BETA_ROLE_ATTACK_EDGE)
    parser.add_argument("--beta-set-piece-edge", type=float, default=DEFAULT_BETA_SET_PIECE_EDGE)
    parser.add_argument("--beta-keeper-fragility", type=float, default=DEFAULT_BETA_KEEPER_FRAGILITY)
    parser.add_argument("--beta-discipline-edge", type=float, default=DEFAULT_BETA_DISCIPLINE_EDGE)
    parser.add_argument("--max-role-log-adjustment", type=float, default=DEFAULT_MAX_ROLE_LOG_ADJUSTMENT)
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
        observed_matches_csv=args.observed_matches,
        fotmob_leaders_csv=args.fotmob_leaders,
        fotmob_player_stats_csv=args.fotmob_player_stats,
        fotmob_lineups_csv=args.fotmob_lineups,
        fotmob_substitutions_csv=args.fotmob_substitutions,
        fotmob_keeper_stats_csv=args.fotmob_keeper_stats,
        include_observed_goals=not (
            args.blind_current_goals or args.ignore_observed_goals
        ),
        include_fotmob_goal_stats=not (
            args.blind_current_goals or args.ignore_fotmob_goal_stats
        ),
        include_group_score_context=not (
            args.blind_current_goals or args.disable_group_score_context
        ),
        role_form_wdl_blend=args.role_form_wdl_blend,
        role_form_scoreline_blend=args.role_form_scoreline_blend,
        beta_role_attack_edge=args.beta_role_attack_edge,
        beta_set_piece_edge=args.beta_set_piece_edge,
        beta_keeper_fragility=args.beta_keeper_fragility,
        beta_discipline_edge=args.beta_discipline_edge,
        max_role_log_adjustment=args.max_role_log_adjustment,
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
                "version": "v30-player-role-form",
                "base_model": "v29-tail-risk-scoreline",
                "team_a": prediction["team_a"],
                "team_b": prediction["team_b"],
                "result_probabilities": prediction["result_probabilities"],
                "v30_adjustments": prediction["v30_adjustments"],
                "expanded_training_data": model.training_data_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not args.no_plots:
        v11.plot_prediction_outputs(prediction, output_dir)
        v28.plot_top3_scorelines(prediction, output_dir / "plots")
    print(
        json.dumps(
            {
                "result_probabilities": prediction["result_probabilities"],
                "predicted_result": prediction["predicted_result"],
                "lambda_a": prediction["lambda_a"],
                "lambda_b": prediction["lambda_b"],
                "top_scorelines": prediction["top_scorelines"][:5],
                "v30_adjustments": {
                    "role_log_adjustment_a": prediction["v30_adjustments"][
                        "role_log_adjustment_a"
                    ],
                    "role_log_adjustment_b": prediction["v30_adjustments"][
                        "role_log_adjustment_b"
                    ],
                    "tail_risk": prediction["v30_adjustments"]["tail_risk"],
                },
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

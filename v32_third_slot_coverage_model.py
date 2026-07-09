#!/usr/bin/env python3
"""V32: V29 base with a coverage-oriented third-slot selector.

Run:
    MPLCONFIGDIR=.matplotlib_cache .venv/bin/python v32_third_slot_coverage_model.py --team-a "Switzerland" --team-b "Bosnia and Herzegovina" --outdir outputs/outputs_v32_switzerland_bosnia
    MPLCONFIGDIR=.matplotlib_cache .venv/bin/python v32_third_slot_coverage_model.py --eval-observed --eval-outdir observed_eval/observed_eval_v32_third_slot_top3
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd

import v11_wcq_results_model as v11
import v15_catboost_model as v15
import v26_top3_coverage_model as v26
import v27_total_goals_calibrated_model as v27
import v28_current_worldcup_form_model as v28
import v29_tail_risk_scoreline_model as v29
import v30_player_role_form_model as v30
import v31_gated_role_selector_model as v31


ScoreMatrix = Dict[Tuple[int, int], float]

DEFAULT_THIRD_SLOT_RELATIVE_FLOOR = 0.34
DEFAULT_THIRD_SLOT_ABSOLUTE_FLOOR = 0.026
DEFAULT_THIRD_SLOT_MIN_UTILITY_GAIN = 0.09
DEFAULT_THIRD_SLOT_PROBABILITY_WEIGHT = 0.84
DEFAULT_THIRD_SLOT_SIGNAL_WEIGHT = 0.28
DEFAULT_THIRD_SLOT_INCUMBENT_BONUS = 0.035
DEFAULT_THIRD_SLOT_TAIL_INCUMBENT_BONUS = 0.16
DEFAULT_THIRD_SLOT_EXTREME_TAIL_INCUMBENT_BONUS = 0.42
DEFAULT_THIRD_SLOT_RANKED_CANDIDATES = 9


@dataclass
class CurrentXGProfile:
    team: str
    attack_pressure: float = 0.0
    creative_pressure: float = 0.0
    finishing_pressure: float = 0.0
    defensive_leakiness: float = 0.0
    keeper_form: float = 0.0
    coverage: float = 0.0
    matches: int = 0
    rows: int = 0

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "team": self.team,
            "attack_pressure": self.attack_pressure,
            "creative_pressure": self.creative_pressure,
            "finishing_pressure": self.finishing_pressure,
            "defensive_leakiness": self.defensive_leakiness,
            "keeper_form": self.keeper_form,
            "coverage": self.coverage,
            "matches": int(self.matches),
            "rows": int(self.rows),
        }


def score_matrix_from_prediction(prediction: Dict[str, Any]) -> ScoreMatrix:
    return v29.score_matrix_from_prediction(prediction)


def score_item(key: Tuple[int, int], probability: float) -> Dict[str, Any]:
    return v29.score_item(key, probability)


def _score_key(item: Dict[str, Any]) -> Tuple[int, int]:
    return int(item["team_a_goals"]), int(item["team_b_goals"])


def _clip_signal(value: float, low: float = -1.0, high: float = 1.75) -> float:
    return float(np.clip(value, low, high))


def build_current_xg_profiles(
    player_stats_csv: str | Path | None,
    lineups_csv: str | Path | None,
    substitutions_csv: str | Path | None,
    keeper_stats_csv: str | Path | None,
) -> dict[str, CurrentXGProfile]:
    paths = [player_stats_csv, lineups_csv, substitutions_csv]
    if not all(path and Path(path).exists() for path in paths):
        return {}
    player_stats = pd.read_csv(player_stats_csv)
    lineups = pd.read_csv(lineups_csv)
    substitutions = pd.read_csv(substitutions_csv)
    if player_stats.empty or lineups.empty:
        return {}

    player_stats = v30.aggregate_player_match_stats(player_stats)
    enriched = v30.attach_roster_context(player_stats, lineups, substitutions)
    enriched = enriched[enriched["team"].astype(str).ne("")]
    if enriched.empty:
        return {}

    numeric = [
        "xg",
        "xa",
        "xgot",
        "total_shots",
        "shots_on_target",
        "touches_in_opposition_box",
        "chances_created",
    ]
    for column in numeric:
        if column not in enriched:
            enriched[column] = 0.0
        enriched[column] = pd.to_numeric(enriched[column], errors="coerce").fillna(0.0)

    team = (
        enriched.groupby("team", as_index=False)
        .agg(
            rows=("player", "size"),
            matches=("match_id", "nunique"),
            xg=("xg", "sum"),
            xa=("xa", "sum"),
            xgot=("xgot", "sum"),
            total_shots=("total_shots", "sum"),
            shots_on_target=("shots_on_target", "sum"),
            box_touches=("touches_in_opposition_box", "sum"),
            chances_created=("chances_created", "sum"),
        )
        .copy()
    )
    match_denominator = team["matches"].replace(0, 1)
    for column in [
        "xg",
        "xa",
        "xgot",
        "total_shots",
        "shots_on_target",
        "box_touches",
        "chances_created",
    ]:
        team[f"{column}_pm"] = team[column] / match_denominator

    team["attack_pressure"] = v28.zscore(
        team["xg_pm"]
        + 0.45 * team["xgot_pm"]
        + 0.10 * team["shots_on_target_pm"]
        + 0.025 * team["box_touches_pm"]
    )
    team["creative_pressure"] = v28.zscore(
        team["xa_pm"] + 0.16 * team["chances_created_pm"]
    )
    team["finishing_pressure"] = v28.zscore(team["xgot_pm"] - 0.65 * team["xg_pm"])
    team["defensive_leakiness"] = 0.0
    team["keeper_form"] = 0.0

    if keeper_stats_csv and Path(keeper_stats_csv).exists():
        keeper = pd.read_csv(keeper_stats_csv)
        if not keeper.empty and {"match_id", "player"}.issubset(keeper.columns):
            keeper_agg = v30.aggregate_player_match_stats(keeper)
            keeper_enriched = v30.attach_roster_context(
                keeper_agg,
                lineups,
                substitutions,
            )
            keeper_enriched = keeper_enriched[
                keeper_enriched["team"].astype(str).ne("")
            ]
            for column in [
                "saves",
                "goals_conceded",
                "xgot_faced",
                "goals_prevented",
            ]:
                if column not in keeper_enriched:
                    keeper_enriched[column] = 0.0
                keeper_enriched[column] = pd.to_numeric(
                    keeper_enriched[column],
                    errors="coerce",
                ).fillna(0.0)
            if not keeper_enriched.empty:
                keeper_team = (
                    keeper_enriched.groupby("team", as_index=False)
                    .agg(
                        keeper_matches=("match_id", "nunique"),
                        saves=("saves", "sum"),
                        goals_conceded=("goals_conceded", "sum"),
                        xgot_faced=("xgot_faced", "sum"),
                        goals_prevented=("goals_prevented", "sum"),
                    )
                    .copy()
                )
                keeper_denominator = keeper_team["keeper_matches"].replace(0, 1)
                keeper_team["defensive_leakiness"] = v28.zscore(
                    keeper_team["xgot_faced"] / keeper_denominator
                    + 0.35 * keeper_team["goals_conceded"] / keeper_denominator
                    - 0.15 * keeper_team["saves"] / keeper_denominator
                    - 0.25
                    * keeper_team["goals_prevented"]
                    / keeper_denominator
                )
                keeper_team["keeper_form"] = v28.zscore(
                    keeper_team["goals_prevented"] / keeper_denominator
                    + 0.10 * keeper_team["saves"] / keeper_denominator
                )
                team = team.merge(
                    keeper_team[["team", "defensive_leakiness", "keeper_form"]],
                    on="team",
                    how="left",
                    suffixes=("", "_keeper"),
                )
                team["defensive_leakiness"] = team[
                    "defensive_leakiness_keeper"
                ].fillna(team["defensive_leakiness"])
                team["keeper_form"] = team["keeper_form_keeper"].fillna(
                    team["keeper_form"]
                )
                team = team.drop(
                    columns=["defensive_leakiness_keeper", "keeper_form_keeper"]
                )

    coverage = np.sqrt(team["rows"] / (team["rows"] + 8.0))
    coverage *= np.sqrt(team["matches"] / (team["matches"] + 2.0))
    team["coverage"] = np.clip(coverage, 0.0, 1.0)

    profiles: dict[str, CurrentXGProfile] = {}
    for row in team.to_dict(orient="records"):
        coverage_value = float(row.get("coverage", 0.0) or 0.0)
        profiles[str(row["team"])] = CurrentXGProfile(
            team=str(row["team"]),
            attack_pressure=float(row.get("attack_pressure", 0.0)) * coverage_value,
            creative_pressure=float(row.get("creative_pressure", 0.0))
            * coverage_value,
            finishing_pressure=float(row.get("finishing_pressure", 0.0))
            * coverage_value,
            defensive_leakiness=float(row.get("defensive_leakiness", 0.0))
            * coverage_value,
            keeper_form=float(row.get("keeper_form", 0.0)) * coverage_value,
            coverage=coverage_value,
            matches=int(row.get("matches", 0) or 0),
            rows=int(row.get("rows", 0) or 0),
        )
    return profiles


def _favorite_xg_metrics(
    side: str,
    xg_a: CurrentXGProfile,
    xg_b: CurrentXGProfile,
) -> Dict[str, float]:
    favorite = xg_a if side == "team_a" else xg_b
    underdog = xg_b if side == "team_a" else xg_a
    coverage = math.sqrt(
        max(float(favorite.coverage), 0.0) * max(float(underdog.coverage), 0.0)
    )
    favorite_pressure = (
        0.62 * favorite.attack_pressure
        + 0.38 * favorite.creative_pressure
        + 0.35 * underdog.defensive_leakiness
    )
    underdog_score_pressure = (
        0.70 * underdog.attack_pressure
        + 0.30 * underdog.creative_pressure
        + 0.35 * favorite.defensive_leakiness
        - 0.25 * favorite.keeper_form
    )
    clean_sheet_xg_signal = -underdog_score_pressure
    favorite_by_three_xg_signal = (
        0.70 * favorite_pressure + 0.30 * clean_sheet_xg_signal
    )
    return {
        "xg_coverage": float(coverage),
        "favorite_pressure": float(favorite_pressure),
        "underdog_score_pressure": float(underdog_score_pressure),
        "clean_sheet_xg_signal": float(clean_sheet_xg_signal),
        "favorite_by_three_xg_signal": float(favorite_by_three_xg_signal),
    }


def _favorite_score(side: str, favorite_goals: int, underdog_goals: int) -> Tuple[int, int]:
    return v31._favorite_score(side, favorite_goals, underdog_goals)


def _score_outcome(key: Tuple[int, int]) -> str:
    if key[0] > key[1]:
        return "team_a_win"
    if key[1] > key[0]:
        return "team_b_win"
    return "draw"


def _candidate_pool(
    ranked_keys: list[Tuple[int, int]],
    selected: list[Tuple[int, int]],
    result_probabilities: Dict[str, float],
    ranked_limit: int,
) -> list[Tuple[int, int]]:
    pool: list[Tuple[int, int]] = []
    for key in [*selected[:3], *ranked_keys[:ranked_limit]]:
        if key not in pool:
            pool.append(key)

    side = v26.favorite_side(result_probabilities)
    if side is not None:
        structured = [
            _favorite_score(side, 1, 0),
            _favorite_score(side, 2, 0),
            _favorite_score(side, 2, 1),
            _favorite_score(side, 3, 0),
            _favorite_score(side, 3, 1),
            _favorite_score(side, 4, 0),
            _favorite_score(side, 4, 1),
        ]
        for key in structured:
            if key not in pool:
                pool.append(key)

    if max(result_probabilities.values()) < 0.52 or result_probabilities.get(
        "draw",
        0.0,
    ) >= 0.24:
        for key in [(0, 0), (1, 1), (2, 2), (1, 0), (0, 1), (2, 1), (1, 2)]:
            if key not in pool:
                pool.append(key)
    return pool


def _favorite_goal_view(
    key: Tuple[int, int],
    side: str | None,
) -> tuple[int, int, int]:
    if side == "team_a":
        favorite_goals, underdog_goals = key
    elif side == "team_b":
        underdog_goals, favorite_goals = key
    else:
        favorite_goals, underdog_goals = max(key), min(key)
    return favorite_goals, underdog_goals, favorite_goals - underdog_goals


def candidate_coverage_utility(
    key: Tuple[int, int],
    score_matrix: ScoreMatrix,
    selected: list[Tuple[int, int]],
    result_probabilities: Dict[str, float],
    role_metrics: Dict[str, float],
    xg_metrics: Dict[str, float],
    probability_weight: float = DEFAULT_THIRD_SLOT_PROBABILITY_WEIGHT,
    signal_weight: float = DEFAULT_THIRD_SLOT_SIGNAL_WEIGHT,
    incumbent_bonus: float = DEFAULT_THIRD_SLOT_INCUMBENT_BONUS,
    tail_incumbent_bonus: float = DEFAULT_THIRD_SLOT_TAIL_INCUMBENT_BONUS,
    extreme_tail_incumbent_bonus: float = DEFAULT_THIRD_SLOT_EXTREME_TAIL_INCUMBENT_BONUS,
) -> Dict[str, float]:
    side = v26.favorite_side(result_probabilities)
    top_probability = max(float(score_matrix.get(selected[0], 0.0)), 1e-12)
    probability_ratio = float(score_matrix.get(key, 0.0)) / top_probability
    probability_component = float(probability_weight) * math.sqrt(
        max(probability_ratio, 0.0)
    )
    favorite_goals, underdog_goals, margin = _favorite_goal_view(key, side)
    predicted_outcome = max(result_probabilities, key=result_probabilities.get)
    outcome_component = 0.06 if _score_outcome(key) == predicted_outcome else 0.0
    normal_total_component = 0.035 if 2 <= sum(key) <= 4 else 0.0
    if key == selected[2]:
        outcome_component += float(incumbent_bonus)
        if side is not None and v29.is_favorite_tail_score(key, side):
            outcome_component += float(tail_incumbent_bonus)
            if max(key) >= 5:
                outcome_component += float(extreme_tail_incumbent_bonus)

    role_tail = role_metrics.get("favorite_by_three_signal", 0.0)
    role_clean = role_metrics.get("clean_sheet_signal", 0.0)
    xg_tail = xg_metrics.get("favorite_by_three_xg_signal", 0.0)
    xg_clean = xg_metrics.get("clean_sheet_xg_signal", 0.0)
    role_coverage = role_metrics.get("role_coverage", 0.0)
    xg_coverage = xg_metrics.get("xg_coverage", 0.0)
    signal_coverage = math.sqrt(max(role_coverage, 0.0) * max(xg_coverage, 0.0))
    tail_signal = signal_coverage * (
        0.55 * _clip_signal(role_tail) + 0.45 * _clip_signal(xg_tail)
    )
    clean_signal = signal_coverage * (
        0.48 * _clip_signal(role_clean) + 0.52 * _clip_signal(xg_clean)
    )
    both_score_signal = signal_coverage * _clip_signal(
        -0.45 * role_clean
        - 0.55 * xg_clean
        + 0.18 * xg_metrics.get("underdog_score_pressure", 0.0)
    )

    signal_component = 0.0
    if side is not None and margin > 0:
        if favorite_goals >= 3 and margin >= 2:
            signal_component += float(signal_weight) * tail_signal
        if underdog_goals == 0:
            signal_component += float(signal_weight) * 0.82 * clean_signal
        elif underdog_goals == 1:
            signal_component += float(signal_weight) * 0.46 * both_score_signal
        if favorite_goals == 2 and underdog_goals == 0:
            signal_component += float(signal_weight) * 0.24 * clean_signal

    utility = (
        probability_component
        + outcome_component
        + normal_total_component
        + signal_component
    )
    return {
        "utility": float(utility),
        "probability_ratio": float(probability_ratio),
        "probability_component": float(probability_component),
        "signal_component": float(signal_component),
        "outcome_component": float(outcome_component),
        "normal_total_component": float(normal_total_component),
        "tail_signal": float(tail_signal),
        "clean_signal": float(clean_signal),
        "both_score_signal": float(both_score_signal),
    }


def _rebuild_top_scorelines(
    selected: list[Tuple[int, int]],
    ranked_keys: list[Tuple[int, int]],
    candidate: Tuple[int, int],
    score_matrix: ScoreMatrix,
    top_n: int,
) -> list[Dict[str, Any]]:
    top_three = [selected[0], selected[1], candidate]
    rebuilt: list[Tuple[int, int]] = []
    for key in [*top_three, *selected, *ranked_keys]:
        if key not in rebuilt and key in score_matrix:
            rebuilt.append(key)
        if len(rebuilt) >= top_n:
            break
    return [score_item(key, score_matrix[key]) for key in rebuilt]


def select_top_scorelines_with_third_slot_coverage(
    score_matrix: ScoreMatrix,
    result_probabilities: Dict[str, float],
    role_a: v30.PlayerRoleProfile,
    role_b: v30.PlayerRoleProfile,
    xg_a: CurrentXGProfile,
    xg_b: CurrentXGProfile,
    current_top_scorelines: list[Dict[str, Any]] | None = None,
    top_n: int = 15,
    relative_floor: float = DEFAULT_THIRD_SLOT_RELATIVE_FLOOR,
    absolute_floor: float = DEFAULT_THIRD_SLOT_ABSOLUTE_FLOOR,
    min_utility_gain: float = DEFAULT_THIRD_SLOT_MIN_UTILITY_GAIN,
    probability_weight: float = DEFAULT_THIRD_SLOT_PROBABILITY_WEIGHT,
    signal_weight: float = DEFAULT_THIRD_SLOT_SIGNAL_WEIGHT,
    incumbent_bonus: float = DEFAULT_THIRD_SLOT_INCUMBENT_BONUS,
    tail_incumbent_bonus: float = DEFAULT_THIRD_SLOT_TAIL_INCUMBENT_BONUS,
    extreme_tail_incumbent_bonus: float = DEFAULT_THIRD_SLOT_EXTREME_TAIL_INCUMBENT_BONUS,
    ranked_candidates: int = DEFAULT_THIRD_SLOT_RANKED_CANDIDATES,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    ranked_keys = [
        key
        for key, _ in sorted(
            score_matrix.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]
    selected = (
        [_score_key(item) for item in current_top_scorelines]
        if current_top_scorelines
        else ranked_keys[:top_n]
    )
    diagnostics: Dict[str, Any] = {
        "third_slot_selector_enabled": True,
        "third_slot_changed": False,
        "relative_floor": float(relative_floor),
        "absolute_floor": float(absolute_floor),
        "min_utility_gain": float(min_utility_gain),
        "probability_weight": float(probability_weight),
        "signal_weight": float(signal_weight),
        "incumbent_bonus": float(incumbent_bonus),
        "tail_incumbent_bonus": float(tail_incumbent_bonus),
        "extreme_tail_incumbent_bonus": float(extreme_tail_incumbent_bonus),
        "ranked_candidates": int(ranked_candidates),
    }
    if len(selected) < 3:
        return [score_item(key, score_matrix[key]) for key in selected], diagnostics

    side = v26.favorite_side(result_probabilities)
    role_metrics = (
        v31._favorite_role_metrics(side, role_a, role_b) if side else {}
    )
    xg_metrics = _favorite_xg_metrics(side, xg_a, xg_b) if side else {}
    diagnostics.update(
        {
            "favorite_side": side,
            "base_top_3": [f"{key[0]}-{key[1]}" for key in selected[:3]],
            "role_metrics": role_metrics,
            "xg_metrics": xg_metrics,
        }
    )
    pool = [
        key
        for key in _candidate_pool(
            ranked_keys,
            selected,
            result_probabilities,
            int(ranked_candidates),
        )
        if key in score_matrix and key not in set(selected[:2])
    ]
    top_probability = max(float(score_matrix.get(selected[0], 0.0)), 1e-12)
    floor = max(float(absolute_floor), top_probability * float(relative_floor))
    pool = [key for key in pool if float(score_matrix.get(key, 0.0)) >= floor]
    if selected[2] not in pool and selected[2] in score_matrix:
        pool.append(selected[2])
    if not pool:
        diagnostics["skip_reason"] = "no_candidate_above_floor"
        return [score_item(key, score_matrix[key]) for key in selected], diagnostics

    utilities = {
        key: candidate_coverage_utility(
            key,
            score_matrix,
            selected,
            result_probabilities,
            role_metrics,
            xg_metrics,
            probability_weight=probability_weight,
            signal_weight=signal_weight,
            incumbent_bonus=incumbent_bonus,
            tail_incumbent_bonus=tail_incumbent_bonus,
            extreme_tail_incumbent_bonus=extreme_tail_incumbent_bonus,
        )
        for key in pool
    }
    incumbent = selected[2]
    best = max(pool, key=lambda key: utilities[key]["utility"])
    incumbent_utility = utilities.get(
        incumbent,
        candidate_coverage_utility(
            incumbent,
            score_matrix,
            selected,
            result_probabilities,
            role_metrics,
            xg_metrics,
            probability_weight=probability_weight,
            signal_weight=signal_weight,
            incumbent_bonus=incumbent_bonus,
            tail_incumbent_bonus=tail_incumbent_bonus,
            extreme_tail_incumbent_bonus=extreme_tail_incumbent_bonus,
        ),
    )["utility"]
    best_utility = utilities[best]["utility"]
    diagnostics.update(
        {
            "candidate_count": len(pool),
            "incumbent_third_scoreline": f"{incumbent[0]}-{incumbent[1]}",
            "incumbent_utility": float(incumbent_utility),
            "best_candidate_scoreline": f"{best[0]}-{best[1]}",
            "best_candidate_probability": float(score_matrix.get(best, 0.0)),
            "best_candidate_utility": float(best_utility),
            "best_candidate_details": utilities[best],
        }
    )
    if best == incumbent:
        diagnostics["skip_reason"] = "incumbent_best"
        return [score_item(key, score_matrix[key]) for key in selected], diagnostics
    if best_utility < incumbent_utility + float(min_utility_gain):
        diagnostics["skip_reason"] = "utility_gain_too_small"
        return [score_item(key, score_matrix[key]) for key in selected], diagnostics

    diagnostics["third_slot_changed"] = True
    diagnostics["replaced_third_scoreline"] = f"{incumbent[0]}-{incumbent[1]}"
    diagnostics["new_third_scoreline"] = f"{best[0]}-{best[1]}"
    return _rebuild_top_scorelines(selected, ranked_keys, best, score_matrix, top_n), diagnostics


class V32ThirdSlotCoverageModel:
    """Wrap V29 and choose the third Top-3 slot by coverage utility."""

    def __init__(
        self,
        base_model: v29.V29TailRiskScorelineModel,
        role_profiles: dict[str, v30.PlayerRoleProfile],
        xg_profiles: dict[str, CurrentXGProfile],
        relative_floor: float = DEFAULT_THIRD_SLOT_RELATIVE_FLOOR,
        absolute_floor: float = DEFAULT_THIRD_SLOT_ABSOLUTE_FLOOR,
        min_utility_gain: float = DEFAULT_THIRD_SLOT_MIN_UTILITY_GAIN,
        probability_weight: float = DEFAULT_THIRD_SLOT_PROBABILITY_WEIGHT,
        signal_weight: float = DEFAULT_THIRD_SLOT_SIGNAL_WEIGHT,
        incumbent_bonus: float = DEFAULT_THIRD_SLOT_INCUMBENT_BONUS,
        tail_incumbent_bonus: float = DEFAULT_THIRD_SLOT_TAIL_INCUMBENT_BONUS,
        extreme_tail_incumbent_bonus: float = DEFAULT_THIRD_SLOT_EXTREME_TAIL_INCUMBENT_BONUS,
        ranked_candidates: int = DEFAULT_THIRD_SLOT_RANKED_CANDIDATES,
    ):
        self.base_model = base_model
        self.role_profiles = role_profiles
        self.xg_profiles = xg_profiles
        self.relative_floor = float(max(relative_floor, 0.0))
        self.absolute_floor = float(max(absolute_floor, 0.0))
        self.min_utility_gain = float(max(min_utility_gain, 0.0))
        self.probability_weight = float(max(probability_weight, 0.0))
        self.signal_weight = float(signal_weight)
        self.incumbent_bonus = float(max(incumbent_bonus, 0.0))
        self.tail_incumbent_bonus = float(max(tail_incumbent_bonus, 0.0))
        self.extreme_tail_incumbent_bonus = float(
            max(extreme_tail_incumbent_bonus, 0.0)
        )
        self.ranked_candidates = int(max(ranked_candidates, 3))
        self.training_data_summary = getattr(base_model, "training_data_summary", {})

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_model, name)

    def role_for_team(self, team: object) -> v30.PlayerRoleProfile:
        canonical = v28.canon_team(team)
        return self.role_profiles.get(canonical, v30.PlayerRoleProfile(team=canonical))

    def xg_for_team(self, team: object) -> CurrentXGProfile:
        canonical = v28.canon_team(team)
        return self.xg_profiles.get(canonical, CurrentXGProfile(team=canonical))

    def predict(self, *args, **kwargs) -> Dict[str, Any]:
        team_a = kwargs.get("team_a", args[0] if args else "")
        team_b = kwargs.get("team_b", args[1] if len(args) > 1 else "")
        prediction = self.base_model.predict(*args, **kwargs)
        score_matrix = score_matrix_from_prediction(prediction)
        role_a = self.role_for_team(team_a)
        role_b = self.role_for_team(team_b)
        xg_a = self.xg_for_team(team_a)
        xg_b = self.xg_for_team(team_b)
        top_scorelines, diagnostics = select_top_scorelines_with_third_slot_coverage(
            score_matrix,
            prediction["result_probabilities"],
            role_a,
            role_b,
            xg_a,
            xg_b,
            current_top_scorelines=prediction.get("top_scorelines", []),
            top_n=15,
            relative_floor=self.relative_floor,
            absolute_floor=self.absolute_floor,
            min_utility_gain=self.min_utility_gain,
            probability_weight=self.probability_weight,
            signal_weight=self.signal_weight,
            incumbent_bonus=self.incumbent_bonus,
            tail_incumbent_bonus=self.tail_incumbent_bonus,
            extreme_tail_incumbent_bonus=self.extreme_tail_incumbent_bonus,
            ranked_candidates=self.ranked_candidates,
        )
        prediction["top_scorelines"] = top_scorelines
        prediction["v32_adjustments"] = {
            "base_model": "v29_tail_risk_scoreline",
            "scoreline_policy": "third_slot_coverage_selector_only",
            "probability_matrix_changed": False,
            "scoreline_layer_affects_wdl": False,
            "team_a_role_profile": role_a.diagnostics(),
            "team_b_role_profile": role_b.diagnostics(),
            "team_a_xg_profile": xg_a.diagnostics(),
            "team_b_xg_profile": xg_b.diagnostics(),
            **diagnostics,
        }
        prediction["calibration_notes"] = {
            **prediction.get("calibration_notes", {}),
            "v32": prediction["v32_adjustments"],
            "third_slot_policy": (
                "V32 leaves V29 probabilities and W/D/L unchanged. It keeps "
                "the first two displayed scorelines and ranks only the third "
                "slot using probability plus player-role, xG/xA, xGOT, and "
                "keeper/concession coverage signals."
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
    third_slot_relative_floor=DEFAULT_THIRD_SLOT_RELATIVE_FLOOR,
    third_slot_absolute_floor=DEFAULT_THIRD_SLOT_ABSOLUTE_FLOOR,
    third_slot_min_utility_gain=DEFAULT_THIRD_SLOT_MIN_UTILITY_GAIN,
    third_slot_probability_weight=DEFAULT_THIRD_SLOT_PROBABILITY_WEIGHT,
    third_slot_signal_weight=DEFAULT_THIRD_SLOT_SIGNAL_WEIGHT,
    third_slot_incumbent_bonus=DEFAULT_THIRD_SLOT_INCUMBENT_BONUS,
    third_slot_tail_incumbent_bonus=DEFAULT_THIRD_SLOT_TAIL_INCUMBENT_BONUS,
    third_slot_extreme_tail_incumbent_bonus=DEFAULT_THIRD_SLOT_EXTREME_TAIL_INCUMBENT_BONUS,
    third_slot_ranked_candidates=DEFAULT_THIRD_SLOT_RANKED_CANDIDATES,
    total_calibration_strength=v27.DEFAULT_TOTAL_CALIBRATION_STRENGTH,
    total_multiplier_clip_low=v27.DEFAULT_MULTIPLIER_CLIP_LOW,
    total_multiplier_clip_high=v27.DEFAULT_MULTIPLIER_CLIP_HIGH,
    total_smoothing=v27.DEFAULT_TOTAL_SMOOTHING,
    min_bin_support=v27.DEFAULT_MIN_BIN_SUPPORT,
    max_train_matches=v27.DEFAULT_MAX_TRAIN_MATCHES,
    current_wdl_blend=v28.DEFAULT_CURRENT_WDL_BLEND,
    current_scoreline_blend=v28.DEFAULT_CURRENT_SCORELINE_BLEND,
    beta_attack_edge=v28.DEFAULT_BETA_ATTACK_EDGE,
    beta_tempo=v28.DEFAULT_BETA_TEMPO,
    beta_group_pressure=v28.DEFAULT_BETA_GROUP_PRESSURE,
    max_log_adjustment=v28.DEFAULT_MAX_LOG_ADJUSTMENT,
    total_calibration_blend=v28.DEFAULT_TOTAL_CALIBRATION_BLEND,
    tail_relative_floor=v28.DEFAULT_TAIL_RELATIVE_FLOOR,
    favorite_win_gate=v26.DEFAULT_FAVORITE_WIN_GATE,
    total_lambda_gate=v26.DEFAULT_TOTAL_LAMBDA_GATE,
    favorite_lambda_gate=v26.DEFAULT_FAVORITE_LAMBDA_GATE,
    draw_ceiling=v26.DEFAULT_DRAW_CEILING,
    tail_favorite_win_gate=v29.DEFAULT_TAIL_FAVORITE_WIN_GATE,
    tail_extreme_favorite_win_gate=v29.DEFAULT_TAIL_EXTREME_FAVORITE_WIN_GATE,
    tail_draw_ceiling=v29.DEFAULT_TAIL_DRAW_CEILING,
    tail_favorite_lambda_gate=v29.DEFAULT_TAIL_FAVORITE_LAMBDA_GATE,
    tail_extreme_lambda_gate=v29.DEFAULT_TAIL_EXTREME_LAMBDA_GATE,
    tail_lambda_gap_gate=v29.DEFAULT_TAIL_LAMBDA_GAP_GATE,
    tail_total_lambda_gate=v29.DEFAULT_TAIL_TOTAL_LAMBDA_GATE,
    tail_selector_relative_floor=v29.DEFAULT_TAIL_RELATIVE_FLOOR,
    tail_selector_absolute_floor=v29.DEFAULT_TAIL_ABSOLUTE_FLOOR,
    tail_max_winner_goals=v29.DEFAULT_TAIL_MAX_WINNER_GOALS,
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
        total_calibration_strength=total_calibration_strength,
        total_multiplier_clip_low=total_multiplier_clip_low,
        total_multiplier_clip_high=total_multiplier_clip_high,
        total_smoothing=total_smoothing,
        min_bin_support=min_bin_support,
        max_train_matches=max_train_matches,
        current_wdl_blend=current_wdl_blend,
        current_scoreline_blend=current_scoreline_blend,
        beta_attack_edge=beta_attack_edge,
        beta_tempo=beta_tempo,
        beta_group_pressure=beta_group_pressure,
        max_log_adjustment=max_log_adjustment,
        total_calibration_blend=total_calibration_blend,
        tail_relative_floor=tail_relative_floor,
        favorite_win_gate=favorite_win_gate,
        total_lambda_gate=total_lambda_gate,
        favorite_lambda_gate=favorite_lambda_gate,
        draw_ceiling=draw_ceiling,
        tail_favorite_win_gate=tail_favorite_win_gate,
        tail_extreme_favorite_win_gate=tail_extreme_favorite_win_gate,
        tail_draw_ceiling=tail_draw_ceiling,
        tail_favorite_lambda_gate=tail_favorite_lambda_gate,
        tail_extreme_lambda_gate=tail_extreme_lambda_gate,
        tail_lambda_gap_gate=tail_lambda_gap_gate,
        tail_total_lambda_gate=tail_total_lambda_gate,
        tail_selector_relative_floor=tail_selector_relative_floor,
        tail_selector_absolute_floor=tail_selector_absolute_floor,
        tail_max_winner_goals=tail_max_winner_goals,
    )
    leaderboard_profiles = v30.build_player_role_profiles(fotmob_leaders_csv)
    match_profiles = v30.build_match_player_role_profiles(
        fotmob_player_stats_csv,
        fotmob_lineups_csv,
        fotmob_substitutions_csv,
        fotmob_keeper_stats_csv,
    )
    role_profiles = v30.merge_role_profiles(leaderboard_profiles, match_profiles)
    xg_profiles = build_current_xg_profiles(
        fotmob_player_stats_csv,
        fotmob_lineups_csv,
        fotmob_substitutions_csv,
        fotmob_keeper_stats_csv,
    )
    model = V32ThirdSlotCoverageModel(
        base_model,
        role_profiles,
        xg_profiles,
        relative_floor=third_slot_relative_floor,
        absolute_floor=third_slot_absolute_floor,
        min_utility_gain=third_slot_min_utility_gain,
        probability_weight=third_slot_probability_weight,
        signal_weight=third_slot_signal_weight,
        incumbent_bonus=third_slot_incumbent_bonus,
        tail_incumbent_bonus=third_slot_tail_incumbent_bonus,
        extreme_tail_incumbent_bonus=third_slot_extreme_tail_incumbent_bonus,
        ranked_candidates=third_slot_ranked_candidates,
    )
    model.training_data_summary = {
        **getattr(base_model, "training_data_summary", {}),
        "v32_scoreline_policy": "third_slot_coverage_selector_only",
        "v32_probability_matrix_changed": False,
        "v32_role_profile_teams": len(role_profiles),
        "v32_match_role_profile_teams": len(match_profiles),
        "v32_xg_profile_teams": len(xg_profiles),
        "v32_fotmob_leaders_csv": str(fotmob_leaders_csv),
        "v32_fotmob_player_stats_csv": str(fotmob_player_stats_csv),
        "v32_fotmob_lineups_csv": str(fotmob_lineups_csv),
        "v32_fotmob_substitutions_csv": str(fotmob_substitutions_csv),
        "v32_fotmob_keeper_stats_csv": str(fotmob_keeper_stats_csv),
        "v32_third_slot_relative_floor": model.relative_floor,
        "v32_third_slot_absolute_floor": model.absolute_floor,
        "v32_third_slot_min_utility_gain": model.min_utility_gain,
        "v32_third_slot_probability_weight": model.probability_weight,
        "v32_third_slot_signal_weight": model.signal_weight,
        "v32_third_slot_incumbent_bonus": model.incumbent_bonus,
        "v32_third_slot_tail_incumbent_bonus": model.tail_incumbent_bonus,
        "v32_third_slot_extreme_tail_incumbent_bonus": model.extreme_tail_incumbent_bonus,
        "v32_third_slot_ranked_candidates": model.ranked_candidates,
    }
    return model, data


def evaluate_observed_matches(
    model: V32ThirdSlotCoverageModel,
    observed_matches_csv: str | Path,
    outdir: str | Path,
    limit: int = 20,
) -> Dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    observed = pd.read_csv(observed_matches_csv).head(limit)
    rows = []
    for idx, row in observed.iterrows():
        prediction = model.predict(row["team_a"], row["team_b"])
        score_matrix = score_matrix_from_prediction(prediction)
        actual = f"{int(row['goals_a'])}-{int(row['goals_b'])}"
        actual_key = (int(row["goals_a"]), int(row["goals_b"]))
        top = [
            f"{int(item['team_a_goals'])}-{int(item['team_b_goals'])}"
            for item in prediction["top_scorelines"][:3]
        ]
        probs = [
            float(item["probability"])
            for item in prediction["top_scorelines"][:3]
        ]
        base_top = prediction["v32_adjustments"].get("base_top_3", top)
        base_hit = actual in base_top
        final_hit = actual in top
        actual_result = (
            "team_a_win"
            if row["goals_a"] > row["goals_b"]
            else "team_b_win"
            if row["goals_b"] > row["goals_a"]
            else "draw"
        )
        result_probabilities = prediction["result_probabilities"]
        predicted_result = prediction["predicted_result"]
        result_label = {
            "team_a_win": row["team_a"],
            "team_b_win": row["team_b"],
            "draw": "Draw",
        }
        changed = bool(prediction["v32_adjustments"].get("third_slot_changed", False))
        rows.append(
            {
                "observed_order": idx + 1,
                "match_id": row["match_id"],
                "team_a": row["team_a"],
                "team_b": row["team_b"],
                "actual_score": actual,
                "base_top_1_scoreline": base_top[0],
                "base_top_2_scoreline": base_top[1],
                "base_top_3_scoreline": base_top[2],
                "top_1_scoreline": top[0],
                "top_1_probability": probs[0],
                "top_2_scoreline": top[1],
                "top_2_probability": probs[1],
                "top_3_scoreline": top[2],
                "top_3_probability": probs[2],
                "actual_is_top_1": actual == top[0],
                "actual_is_top_2": actual == top[1],
                "actual_is_top_3": actual == top[2],
                "actual_in_top_3": final_hit,
                "actual_score_probability": float(score_matrix.get(actual_key, 0.0)),
                "base_actual_in_top_3": base_hit,
                "third_slot_changed": changed,
                "change_gained_hit": changed and final_hit and not base_hit,
                "change_lost_hit": changed and base_hit and not final_hit,
                "change_kept_hit": changed and final_hit and base_hit,
                "change_kept_miss": changed and not final_hit and not base_hit,
                "replaced_third_scoreline": prediction["v32_adjustments"].get(
                    "replaced_third_scoreline",
                    "",
                ),
                "new_third_scoreline": prediction["v32_adjustments"].get(
                    "new_third_scoreline",
                    "",
                ),
                "best_candidate_utility": prediction["v32_adjustments"].get(
                    "best_candidate_utility",
                    "",
                ),
                "incumbent_utility": prediction["v32_adjustments"].get(
                    "incumbent_utility",
                    "",
                ),
                "actual_result": actual_result,
                "actual_result_label": result_label[actual_result],
                "predicted_result": predicted_result,
                "predicted_result_label": result_label[predicted_result],
                "outcome_correct": predicted_result == actual_result,
                "predicted_result_probability": float(
                    result_probabilities[predicted_result]
                ),
                "actual_result_probability": float(result_probabilities[actual_result]),
                "team_a_win_probability": float(result_probabilities["team_a_win"]),
                "draw_probability": float(result_probabilities["draw"]),
                "team_b_win_probability": float(result_probabilities["team_b_win"]),
                "v29_tail_applied": bool(
                    prediction.get("v29_adjustments", {}).get(
                        "tail_risk_applied",
                        False,
                    )
                ),
            }
        )
    eval_df = pd.DataFrame(rows)
    csv_path = outdir / "v32_third_slot_top_three_scoreline_comparison_20_matches.csv"
    eval_df.to_csv(csv_path, index=False)
    normal = eval_df[
        eval_df["actual_score"].str.split("-").apply(
            lambda parts: int(parts[0]) <= 3 and int(parts[1]) <= 3
        )
    ]
    summary = {
        "model": "v32_third_slot_coverage",
        "n_matches": int(len(eval_df)),
        "top_1_exact_score_hits": int(eval_df["actual_is_top_1"].sum()),
        "top_2_exact_score_hits": int(
            (eval_df["actual_is_top_1"] | eval_df["actual_is_top_2"]).sum()
        ),
        "top_3_exact_score_hits": int(eval_df["actual_in_top_3"].sum()),
        "top_3_exact_score_accuracy": float(eval_df["actual_in_top_3"].mean()),
        "outcome_correct": int(eval_df["outcome_correct"].sum()),
        "normal_range_matches": int(len(normal)),
        "normal_range_top_3_hits": int(normal["actual_in_top_3"].sum()),
        "normal_range_top_3_accuracy": float(normal["actual_in_top_3"].mean()),
        "tail_risk_applied_matches": int(eval_df["v29_tail_applied"].sum()),
        "third_slot_changed_matches": int(eval_df["third_slot_changed"].sum()),
        "changed_matches_gained_hits": int(eval_df["change_gained_hit"].sum()),
        "changed_matches_lost_hits": int(eval_df["change_lost_hit"].sum()),
        "changed_matches_kept_hits": int(eval_df["change_kept_hit"].sum()),
        "changed_matches_kept_misses": int(eval_df["change_kept_miss"].sum()),
        "csv": str(csv_path),
    }
    summary_path = outdir / "v32_third_slot_top3_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    try:
        import compare_v11_top_scorelines as scoreline_chart

        png_path = outdir / "v32_third_slot_top_three_scoreline_comparison_20_matches.png"
        chart_df = eval_df.copy()
        chart_df.attrs["model_label"] = "V32 third-slot coverage"
        chart_df.attrs["top_n"] = 3
        chart_df.attrs["excluded_count"] = 0
        chart_df.attrs["max_observed_goals_per_team"] = None
        scoreline_chart.draw_scoreline_chart(chart_df, png_path)
        summary["png"] = str(png_path)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except Exception as exc:
        summary["plot_error"] = str(exc)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    parser = argparse.ArgumentParser(
        description="Run V32: V29 with a role/xG third-slot selector."
    )
    parser.add_argument("--team-a")
    parser.add_argument("--team-b")
    parser.add_argument("--host-a", action="store_true")
    parser.add_argument("--host-b", action="store_true")
    parser.add_argument("--knockout", action="store_true")
    parser.add_argument("--outdir", default="outputs/outputs_v32_third_slot")
    parser.add_argument("--eval-observed", action="store_true")
    parser.add_argument("--eval-outdir", default="observed_eval/observed_eval_v32_third_slot_top3")
    parser.add_argument("--eval-limit", type=int, default=20)
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
    parser.add_argument("--third-slot-relative-floor", type=float, default=DEFAULT_THIRD_SLOT_RELATIVE_FLOOR)
    parser.add_argument("--third-slot-absolute-floor", type=float, default=DEFAULT_THIRD_SLOT_ABSOLUTE_FLOOR)
    parser.add_argument("--third-slot-min-utility-gain", type=float, default=DEFAULT_THIRD_SLOT_MIN_UTILITY_GAIN)
    parser.add_argument("--third-slot-probability-weight", type=float, default=DEFAULT_THIRD_SLOT_PROBABILITY_WEIGHT)
    parser.add_argument("--third-slot-signal-weight", type=float, default=DEFAULT_THIRD_SLOT_SIGNAL_WEIGHT)
    parser.add_argument("--third-slot-incumbent-bonus", type=float, default=DEFAULT_THIRD_SLOT_INCUMBENT_BONUS)
    parser.add_argument("--third-slot-tail-incumbent-bonus", type=float, default=DEFAULT_THIRD_SLOT_TAIL_INCUMBENT_BONUS)
    parser.add_argument("--third-slot-extreme-tail-incumbent-bonus", type=float, default=DEFAULT_THIRD_SLOT_EXTREME_TAIL_INCUMBENT_BONUS)
    parser.add_argument("--third-slot-ranked-candidates", type=int, default=DEFAULT_THIRD_SLOT_RANKED_CANDIDATES)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

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
        third_slot_relative_floor=args.third_slot_relative_floor,
        third_slot_absolute_floor=args.third_slot_absolute_floor,
        third_slot_min_utility_gain=args.third_slot_min_utility_gain,
        third_slot_probability_weight=args.third_slot_probability_weight,
        third_slot_signal_weight=args.third_slot_signal_weight,
        third_slot_incumbent_bonus=args.third_slot_incumbent_bonus,
        third_slot_tail_incumbent_bonus=args.third_slot_tail_incumbent_bonus,
        third_slot_extreme_tail_incumbent_bonus=args.third_slot_extreme_tail_incumbent_bonus,
        third_slot_ranked_candidates=args.third_slot_ranked_candidates,
    )
    if args.eval_observed:
        summary = evaluate_observed_matches(
            model,
            args.observed_matches,
            args.eval_outdir,
            limit=args.eval_limit,
        )
        print(json.dumps(summary, indent=2))
        return
    if not args.team_a or not args.team_b:
        parser.error("--team-a and --team-b are required unless --eval-observed is set")

    output_dir = v11.unique_output_dir(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
                "version": "v32-third-slot-coverage",
                "base_model": "v29-tail-risk-scoreline",
                "team_a": prediction["team_a"],
                "team_b": prediction["team_b"],
                "result_probabilities": prediction["result_probabilities"],
                "v29_adjustments": prediction.get("v29_adjustments", {}),
                "v32_adjustments": prediction["v32_adjustments"],
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
                "v29_adjustments": prediction.get("v29_adjustments", {}),
                "v32_adjustments": {
                    key: value
                    for key, value in prediction["v32_adjustments"].items()
                    if key
                    not in {
                        "team_a_role_profile",
                        "team_b_role_profile",
                        "team_a_xg_profile",
                        "team_b_xg_profile",
                    }
                },
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

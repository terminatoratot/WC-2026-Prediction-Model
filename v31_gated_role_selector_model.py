#!/usr/bin/env python3
"""V31: V29 base selector plus gated V30 role-data Top-3 overrides.

Run:
    MPLCONFIGDIR=.matplotlib_cache .venv/bin/python v31_gated_role_selector_model.py --team-a "Switzerland" --team-b "Bosnia and Herzegovina" --outdir outputs/outputs_v31_switzerland_bosnia
    MPLCONFIGDIR=.matplotlib_cache .venv/bin/python v31_gated_role_selector_model.py --team-a "Argentina" --team-b "France" --no-plots
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

import v11_wcq_results_model as v11
import v15_catboost_model as v15
import v26_top3_coverage_model as v26
import v27_total_goals_calibrated_model as v27
import v28_current_worldcup_form_model as v28
import v29_tail_risk_scoreline_model as v29
import v30_player_role_form_model as v30


ScoreMatrix = Dict[Tuple[int, int], float]

DEFAULT_ROLE_COVERAGE_GATE = 0.50
DEFAULT_ROLE_SIGNAL_GATE = 0.72
DEFAULT_ROLE_FAVORITE_WIN_GATE = 0.58
DEFAULT_ROLE_DRAW_CEILING = 0.31
DEFAULT_ROLE_RELATIVE_FLOOR = 0.42
DEFAULT_ROLE_ABSOLUTE_FLOOR = 0.035
DEFAULT_ROLE_MAX_REPLACEMENT_INDEX = 2


def score_matrix_from_prediction(prediction: Dict[str, Any]) -> ScoreMatrix:
    return v29.score_matrix_from_prediction(prediction)


def score_item(key: Tuple[int, int], probability: float) -> Dict[str, Any]:
    return v29.score_item(key, probability)


def _score_key(item: Dict[str, Any]) -> Tuple[int, int]:
    return int(item["team_a_goals"]), int(item["team_b_goals"])


def _favorite_role_metrics(
    side: str,
    role_a: v30.PlayerRoleProfile,
    role_b: v30.PlayerRoleProfile,
) -> Dict[str, float]:
    favorite = role_a if side == "team_a" else role_b
    underdog = role_b if side == "team_a" else role_a
    coverage = math.sqrt(
        max(float(favorite.coverage), 0.0) * max(float(underdog.coverage), 0.0)
    )
    attack_mismatch = favorite.attack_unit - underdog.defense_unit
    weak_underdog_defense = (
        underdog.defensive_fragility
        - underdog.defense_unit
        - 0.20 * underdog.keeper
    )
    clean_sheet_edge = (
        favorite.defense_unit
        - underdog.attack_unit
        - 0.25 * underdog.set_piece_for
        - 0.20 * underdog.finishing_delta
    )
    creator_attacker_mismatch = (
        0.55 * favorite.attacker
        + 0.45 * favorite.creator
        - 0.70 * underdog.defender
        - 0.20 * underdog.keeper
        + 0.15 * underdog.defensive_fragility
    )
    favorite_by_three_signal = (
        0.45 * attack_mismatch
        + 0.30 * weak_underdog_defense
        + 0.25 * creator_attacker_mismatch
    )
    clean_sheet_signal = 0.65 * clean_sheet_edge + 0.35 * weak_underdog_defense
    high_goal_clean_sheet_signal = (
        0.55 * favorite_by_three_signal + 0.45 * clean_sheet_signal
    )
    return {
        "role_coverage": float(coverage),
        "attack_mismatch": float(attack_mismatch),
        "weak_underdog_defense": float(weak_underdog_defense),
        "clean_sheet_edge": float(clean_sheet_edge),
        "creator_attacker_mismatch": float(creator_attacker_mismatch),
        "favorite_by_three_signal": float(favorite_by_three_signal),
        "clean_sheet_signal": float(clean_sheet_signal),
        "high_goal_clean_sheet_signal": float(high_goal_clean_sheet_signal),
    }


def _favorite_score(
    side: str,
    favorite_goals: int,
    underdog_goals: int,
) -> Tuple[int, int]:
    if side == "team_a":
        return favorite_goals, underdog_goals
    return underdog_goals, favorite_goals


def _available_candidate(
    score_matrix: ScoreMatrix,
    current_top: list[Tuple[int, int]],
    candidates: list[Tuple[int, int]],
    relative_floor: float,
    absolute_floor: float,
) -> Tuple[int, int] | None:
    current_set = set(current_top)
    top_probability = max(
        [float(score_matrix.get(key, 0.0)) for key in current_top[:3]] or [0.0]
    )
    floor = max(float(absolute_floor), top_probability * float(relative_floor))
    available = [
        key
        for key in candidates
        if key in score_matrix
        and key not in current_set
        and float(score_matrix.get(key, 0.0)) >= floor
    ]
    if not available:
        return None
    return max(available, key=lambda key: score_matrix.get(key, 0.0))


def _replace_third_scoreline(
    selected: list[Tuple[int, int]],
    ranked_keys: list[Tuple[int, int]],
    candidate: Tuple[int, int],
    score_matrix: ScoreMatrix,
    top_n: int,
) -> list[Dict[str, Any]]:
    selected_top = list(selected[:3])
    selected_top[DEFAULT_ROLE_MAX_REPLACEMENT_INDEX] = candidate
    rebuilt: list[Tuple[int, int]] = []
    for key in [*selected_top, *selected, *ranked_keys]:
        if key not in rebuilt and key in score_matrix:
            rebuilt.append(key)
        if len(rebuilt) >= top_n:
            break
    return [score_item(key, score_matrix[key]) for key in rebuilt]


def select_top_scorelines_with_gated_roles(
    score_matrix: ScoreMatrix,
    result_probabilities: Dict[str, float],
    role_a: v30.PlayerRoleProfile,
    role_b: v30.PlayerRoleProfile,
    current_top_scorelines: list[Dict[str, Any]] | None = None,
    top_n: int = 15,
    role_coverage_gate: float = DEFAULT_ROLE_COVERAGE_GATE,
    role_signal_gate: float = DEFAULT_ROLE_SIGNAL_GATE,
    favorite_win_gate: float = DEFAULT_ROLE_FAVORITE_WIN_GATE,
    draw_ceiling: float = DEFAULT_ROLE_DRAW_CEILING,
    relative_floor: float = DEFAULT_ROLE_RELATIVE_FLOOR,
    absolute_floor: float = DEFAULT_ROLE_ABSOLUTE_FLOOR,
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
        "role_selector_enabled": True,
        "role_selector_applied": False,
        "role_coverage_gate": float(role_coverage_gate),
        "role_signal_gate": float(role_signal_gate),
        "favorite_win_gate": float(favorite_win_gate),
        "draw_ceiling": float(draw_ceiling),
        "relative_floor": float(relative_floor),
        "absolute_floor": float(absolute_floor),
    }
    if len(selected) < 3:
        return [score_item(key, score_matrix[key]) for key in selected], diagnostics

    side = v26.favorite_side(result_probabilities)
    favorite_probability = (
        float(result_probabilities.get("team_a_win", 0.0))
        if side == "team_a"
        else float(result_probabilities.get("team_b_win", 0.0))
        if side == "team_b"
        else 0.0
    )
    diagnostics.update(
        {
            "favorite_side": side,
            "favorite_probability": favorite_probability,
            "draw_probability": float(result_probabilities.get("draw", 0.0)),
        }
    )
    if (
        side is None
        or favorite_probability < float(favorite_win_gate)
        or float(result_probabilities.get("draw", 0.0)) > float(draw_ceiling)
    ):
        diagnostics["skip_reason"] = "result_gates_not_met"
        return [score_item(key, score_matrix[key]) for key in selected], diagnostics

    metrics = _favorite_role_metrics(side, role_a, role_b)
    diagnostics.update(metrics)
    if metrics["role_coverage"] < float(role_coverage_gate):
        diagnostics["skip_reason"] = "coverage_gate_not_met"
        return [score_item(key, score_matrix[key]) for key in selected], diagnostics

    candidates: list[tuple[str, float, list[Tuple[int, int]]]] = []
    if metrics["favorite_by_three_signal"] >= float(role_signal_gate):
        candidates.append(
            (
                "favorite_by_three",
                metrics["favorite_by_three_signal"],
                [
                    _favorite_score(side, 3, 0),
                    _favorite_score(side, 3, 1),
                    _favorite_score(side, 4, 1),
                    _favorite_score(side, 4, 0),
                ],
            )
        )
    if metrics["clean_sheet_signal"] >= float(role_signal_gate):
        candidates.append(
            (
                "clean_sheet_protection",
                metrics["clean_sheet_signal"],
                [
                    _favorite_score(side, 2, 0),
                    _favorite_score(side, 1, 0),
                    _favorite_score(side, 3, 0),
                ],
            )
        )
    if metrics["high_goal_clean_sheet_signal"] >= float(role_signal_gate):
        candidates.append(
            (
                "creator_attacker_clean_sheet_mismatch",
                metrics["high_goal_clean_sheet_signal"],
                [
                    _favorite_score(side, 3, 0),
                    _favorite_score(side, 2, 0),
                    _favorite_score(side, 4, 0),
                ],
            )
        )

    if not candidates:
        diagnostics["skip_reason"] = "role_signal_gate_not_met"
        return [score_item(key, score_matrix[key]) for key in selected], diagnostics

    candidates = sorted(candidates, key=lambda item: item[1], reverse=True)
    for reason, signal, candidate_list in candidates:
        candidate = _available_candidate(
            score_matrix,
            selected[:3],
            candidate_list,
            relative_floor=relative_floor,
            absolute_floor=absolute_floor,
        )
        if candidate is None:
            continue
        diagnostics.update(
            {
                "role_selector_applied": True,
                "role_selector_reason": reason,
                "role_selector_signal": float(signal),
                "candidate_scoreline": f"{candidate[0]}-{candidate[1]}",
                "candidate_probability": float(score_matrix.get(candidate, 0.0)),
                "replaced_third_scoreline": f"{selected[2][0]}-{selected[2][1]}",
            }
        )
        return (
            _replace_third_scoreline(selected, ranked_keys, candidate, score_matrix, top_n),
            diagnostics,
        )

    diagnostics["skip_reason"] = "no_candidate_above_probability_floor"
    return [score_item(key, score_matrix[key]) for key in selected], diagnostics


class V31GatedRoleSelectorModel:
    """Wrap V29 and let V30 role data make only gated Top-3 substitutions."""

    def __init__(
        self,
        base_model: v29.V29TailRiskScorelineModel,
        role_profiles: dict[str, v30.PlayerRoleProfile],
        role_coverage_gate: float = DEFAULT_ROLE_COVERAGE_GATE,
        role_signal_gate: float = DEFAULT_ROLE_SIGNAL_GATE,
        favorite_win_gate: float = DEFAULT_ROLE_FAVORITE_WIN_GATE,
        draw_ceiling: float = DEFAULT_ROLE_DRAW_CEILING,
        relative_floor: float = DEFAULT_ROLE_RELATIVE_FLOOR,
        absolute_floor: float = DEFAULT_ROLE_ABSOLUTE_FLOOR,
    ):
        self.base_model = base_model
        self.role_profiles = role_profiles
        self.role_coverage_gate = float(np.clip(role_coverage_gate, 0.0, 1.0))
        self.role_signal_gate = float(role_signal_gate)
        self.favorite_win_gate = float(np.clip(favorite_win_gate, 0.0, 1.0))
        self.draw_ceiling = float(np.clip(draw_ceiling, 0.0, 1.0))
        self.relative_floor = float(max(relative_floor, 0.0))
        self.absolute_floor = float(max(absolute_floor, 0.0))
        self.training_data_summary = getattr(base_model, "training_data_summary", {})

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_model, name)

    def role_for_team(self, team: object) -> v30.PlayerRoleProfile:
        canonical = v28.canon_team(team)
        return self.role_profiles.get(canonical, v30.PlayerRoleProfile(team=canonical))

    def predict(self, *args, **kwargs) -> Dict[str, Any]:
        team_a = kwargs.get("team_a", args[0] if args else "")
        team_b = kwargs.get("team_b", args[1] if len(args) > 1 else "")
        prediction = self.base_model.predict(*args, **kwargs)
        score_matrix = score_matrix_from_prediction(prediction)
        role_a = self.role_for_team(team_a)
        role_b = self.role_for_team(team_b)
        top_scorelines, role_diagnostics = select_top_scorelines_with_gated_roles(
            score_matrix,
            prediction["result_probabilities"],
            role_a,
            role_b,
            current_top_scorelines=prediction.get("top_scorelines", []),
            top_n=15,
            role_coverage_gate=self.role_coverage_gate,
            role_signal_gate=self.role_signal_gate,
            favorite_win_gate=self.favorite_win_gate,
            draw_ceiling=self.draw_ceiling,
            relative_floor=self.relative_floor,
            absolute_floor=self.absolute_floor,
        )
        prediction["top_scorelines"] = top_scorelines
        prediction["v31_adjustments"] = {
            "base_model": "v29_tail_risk_scoreline",
            "scoreline_policy": "gated_role_top3_selector_only",
            "probability_matrix_changed": False,
            "scoreline_layer_affects_wdl": False,
            "team_a_role_profile": role_a.diagnostics(),
            "team_b_role_profile": role_b.diagnostics(),
            **role_diagnostics,
        }
        prediction["calibration_notes"] = {
            **prediction.get("calibration_notes", {}),
            "v31": prediction["v31_adjustments"],
            "gated_role_policy": (
                "V31 leaves V29 probabilities and W/D/L unchanged. V30 role "
                "data can only replace the third displayed Top-3 scoreline "
                "when coverage, result, role-signal, and probability-floor "
                "gates all pass."
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
    role_coverage_gate=DEFAULT_ROLE_COVERAGE_GATE,
    role_signal_gate=DEFAULT_ROLE_SIGNAL_GATE,
    role_favorite_win_gate=DEFAULT_ROLE_FAVORITE_WIN_GATE,
    role_draw_ceiling=DEFAULT_ROLE_DRAW_CEILING,
    role_relative_floor=DEFAULT_ROLE_RELATIVE_FLOOR,
    role_absolute_floor=DEFAULT_ROLE_ABSOLUTE_FLOOR,
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
    model = V31GatedRoleSelectorModel(
        base_model,
        role_profiles,
        role_coverage_gate=role_coverage_gate,
        role_signal_gate=role_signal_gate,
        favorite_win_gate=role_favorite_win_gate,
        draw_ceiling=role_draw_ceiling,
        relative_floor=role_relative_floor,
        absolute_floor=role_absolute_floor,
    )
    model.training_data_summary = {
        **getattr(base_model, "training_data_summary", {}),
        "v31_scoreline_policy": "gated_role_top3_selector_only",
        "v31_probability_matrix_changed": False,
        "v31_role_profile_teams": len(role_profiles),
        "v31_leaderboard_role_profile_teams": len(leaderboard_profiles),
        "v31_match_role_profile_teams": len(match_profiles),
        "v31_fotmob_leaders_csv": str(fotmob_leaders_csv),
        "v31_fotmob_player_stats_csv": str(fotmob_player_stats_csv),
        "v31_fotmob_lineups_csv": str(fotmob_lineups_csv),
        "v31_fotmob_substitutions_csv": str(fotmob_substitutions_csv),
        "v31_fotmob_keeper_stats_csv": str(fotmob_keeper_stats_csv),
        "v31_role_coverage_gate": model.role_coverage_gate,
        "v31_role_signal_gate": model.role_signal_gate,
        "v31_role_favorite_win_gate": model.favorite_win_gate,
        "v31_role_draw_ceiling": model.draw_ceiling,
        "v31_role_relative_floor": model.relative_floor,
        "v31_role_absolute_floor": model.absolute_floor,
    }
    return model, data


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    parser = argparse.ArgumentParser(
        description="Run V31: V29 with gated V30 role-data Top-3 overrides."
    )
    parser.add_argument("--team-a", required=True)
    parser.add_argument("--team-b", required=True)
    parser.add_argument("--host-a", action="store_true")
    parser.add_argument("--host-b", action="store_true")
    parser.add_argument("--knockout", action="store_true")
    parser.add_argument("--outdir", default="outputs/outputs_v31_gated_role_selector")
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
    parser.add_argument("--role-coverage-gate", type=float, default=DEFAULT_ROLE_COVERAGE_GATE)
    parser.add_argument("--role-signal-gate", type=float, default=DEFAULT_ROLE_SIGNAL_GATE)
    parser.add_argument("--role-favorite-win-gate", type=float, default=DEFAULT_ROLE_FAVORITE_WIN_GATE)
    parser.add_argument("--role-draw-ceiling", type=float, default=DEFAULT_ROLE_DRAW_CEILING)
    parser.add_argument("--role-relative-floor", type=float, default=DEFAULT_ROLE_RELATIVE_FLOOR)
    parser.add_argument("--role-absolute-floor", type=float, default=DEFAULT_ROLE_ABSOLUTE_FLOOR)
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
        role_coverage_gate=args.role_coverage_gate,
        role_signal_gate=args.role_signal_gate,
        role_favorite_win_gate=args.role_favorite_win_gate,
        role_draw_ceiling=args.role_draw_ceiling,
        role_relative_floor=args.role_relative_floor,
        role_absolute_floor=args.role_absolute_floor,
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
                "version": "v31-gated-role-selector",
                "base_model": "v29-tail-risk-scoreline",
                "team_a": prediction["team_a"],
                "team_b": prediction["team_b"],
                "result_probabilities": prediction["result_probabilities"],
                "v29_adjustments": prediction.get("v29_adjustments", {}),
                "v31_adjustments": prediction["v31_adjustments"],
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
                "v31_adjustments": {
                    key: value
                    for key, value in prediction["v31_adjustments"].items()
                    if key
                    not in {
                        "team_a_role_profile",
                        "team_b_role_profile",
                    }
                },
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

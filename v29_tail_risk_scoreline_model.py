#!/usr/bin/env python3
"""V29: V28 plus a conservative blowout/tail-risk Top-3 selector.

Full feature run:
    MPLCONFIGDIR=.matplotlib_cache .venv/bin/python v29_tail_risk_scoreline_model.py --team-a "Argentina" --team-b "France" --knockout --outdir outputs/outputs_v29_argentina_france --observed-matches data/wc2026_observed_matches_from_screenshots.csv --fotmob-leaders data/fotmob_stat_leaders_clean.csv --tail-favorite-win-gate 0.66 --tail-extreme-favorite-win-gate 0.78 --tail-draw-ceiling 0.27 --tail-favorite-lambda-gate 1.75 --tail-extreme-lambda-gate 2.40 --tail-lambda-gap-gate 0.75 --tail-total-lambda-gate 2.45 --tail-selector-relative-floor 0.12 --tail-selector-absolute-floor 0.008 --tail-max-winner-goals 7

All current completed matches:
    .venv/bin/python -c "from pathlib import Path; import v36_fotmob_current_form_model as v36; v36.completed_fotmob_facts_to_observed(Path('data/fotmob_match_facts_clean.csv'), Path('data/fotmob_completed_matches_observed_schema_current.csv'))"
    MPLCONFIGDIR=.matplotlib_cache LOKY_MAX_CPU_COUNT=8 .venv/bin/python eval_v29_v36_completed_worldcup.py --models v29_tail_risk --observed data/fotmob_completed_matches_observed_schema_current.csv --outdir observed_eval/observed_eval_v29_all_current_matches_no_score_leak
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
VERSIONS_DIR = PROJECT_DIR / "versions"
if str(VERSIONS_DIR) not in sys.path:
    sys.path.insert(0, str(VERSIONS_DIR))

import v11_wcq_results_model as v11
import v15_catboost_model as v15
import v26_top3_coverage_model as v26
import v27_total_goals_calibrated_model as v27
import v28_current_worldcup_form_model as v28


ScoreMatrix = Dict[Tuple[int, int], float]

DEFAULT_TAIL_FAVORITE_WIN_GATE = 0.66
DEFAULT_TAIL_EXTREME_FAVORITE_WIN_GATE = 0.78
DEFAULT_TAIL_DRAW_CEILING = 0.27
DEFAULT_TAIL_FAVORITE_LAMBDA_GATE = 1.75
DEFAULT_TAIL_EXTREME_LAMBDA_GATE = 2.40
DEFAULT_TAIL_LAMBDA_GAP_GATE = 0.75
DEFAULT_TAIL_TOTAL_LAMBDA_GATE = 2.45
DEFAULT_TAIL_RELATIVE_FLOOR = 0.12
DEFAULT_TAIL_ABSOLUTE_FLOOR = 0.008
DEFAULT_TAIL_MAX_WINNER_GOALS = 7


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


def is_favorite_tail_score(key: Tuple[int, int], side: str) -> bool:
    goals_a, goals_b = key
    winner_goals = goals_a if side == "team_a" else goals_b
    loser_goals = goals_b if side == "team_a" else goals_a
    margin = winner_goals - loser_goals
    return winner_goals >= 3 and margin >= 2 and loser_goals <= 2


def is_existing_tail_coverage(top_three: list[Tuple[int, int]], side: str) -> bool:
    return any(is_favorite_tail_score(key, side) and max(key) <= 3 for key in top_three)


def tail_candidates(
    score_matrix: ScoreMatrix,
    side: str,
    max_winner_goals: int,
) -> list[Tuple[int, int]]:
    candidates = [
        key
        for key in score_matrix
        if is_favorite_tail_score(key, side)
        and (key[0] if side == "team_a" else key[1]) <= max_winner_goals
    ]
    return sorted(candidates, key=lambda key: score_matrix.get(key, 0.0), reverse=True)


def tail_utility(
    key: Tuple[int, int],
    probability: float,
    side: str,
    extreme: bool,
) -> float:
    goals_a, goals_b = key
    winner_goals = goals_a if side == "team_a" else goals_b
    loser_goals = goals_b if side == "team_a" else goals_a
    if not extreme:
        return float(probability)
    tail_bonus = np.exp(0.75 * max(winner_goals - 3, 0) + 0.75 * loser_goals)
    return float(probability) * float(tail_bonus)


def select_top_scorelines_with_tail_risk(
    score_matrix: ScoreMatrix,
    result_probabilities: Dict[str, float],
    lambda_a: float,
    lambda_b: float,
    current_top_scorelines: list[Dict[str, Any]] | None = None,
    top_n: int = 15,
    favorite_win_gate: float = DEFAULT_TAIL_FAVORITE_WIN_GATE,
    extreme_favorite_win_gate: float = DEFAULT_TAIL_EXTREME_FAVORITE_WIN_GATE,
    draw_ceiling: float = DEFAULT_TAIL_DRAW_CEILING,
    favorite_lambda_gate: float = DEFAULT_TAIL_FAVORITE_LAMBDA_GATE,
    extreme_lambda_gate: float = DEFAULT_TAIL_EXTREME_LAMBDA_GATE,
    lambda_gap_gate: float = DEFAULT_TAIL_LAMBDA_GAP_GATE,
    total_lambda_gate: float = DEFAULT_TAIL_TOTAL_LAMBDA_GATE,
    relative_floor: float = DEFAULT_TAIL_RELATIVE_FLOOR,
    absolute_floor: float = DEFAULT_TAIL_ABSOLUTE_FLOOR,
    max_winner_goals: int = DEFAULT_TAIL_MAX_WINNER_GOALS,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    ranked_keys = [
        key
        for key, _ in sorted(
            score_matrix.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]
    current_keys = (
        [
            (int(item["team_a_goals"]), int(item["team_b_goals"]))
            for item in current_top_scorelines
        ]
        if current_top_scorelines
        else ranked_keys
    )
    selected = current_keys[:top_n]
    diagnostics: Dict[str, Any] = {
        "tail_risk_selector_enabled": True,
        "tail_risk_applied": False,
        "favorite_win_gate": float(favorite_win_gate),
        "extreme_favorite_win_gate": float(extreme_favorite_win_gate),
        "draw_ceiling": float(draw_ceiling),
        "favorite_lambda_gate": float(favorite_lambda_gate),
        "extreme_lambda_gate": float(extreme_lambda_gate),
        "lambda_gap_gate": float(lambda_gap_gate),
        "total_lambda_gate": float(total_lambda_gate),
        "relative_floor": float(relative_floor),
        "absolute_floor": float(absolute_floor),
        "max_winner_goals": int(max_winner_goals),
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
    favorite_lambda = float(lambda_a) if side == "team_a" else float(lambda_b)
    underdog_lambda = float(lambda_b) if side == "team_a" else float(lambda_a)
    lambda_gap = favorite_lambda - underdog_lambda
    total_lambda = float(lambda_a) + float(lambda_b)
    top_probability = max(float(score_matrix.get(selected[0], 0.0)), 1e-12)
    probability_floor = max(
        float(absolute_floor),
        top_probability * float(relative_floor),
    )
    extreme = (
        favorite_probability >= float(extreme_favorite_win_gate)
        and favorite_lambda >= float(extreme_lambda_gate)
    )
    diagnostics.update(
        {
            "favorite_side": side,
            "favorite_probability": favorite_probability,
            "favorite_lambda": favorite_lambda,
            "underdog_lambda": underdog_lambda,
            "lambda_gap": float(lambda_gap),
            "total_lambda": total_lambda,
            "probability_floor": probability_floor,
            "extreme_tail_mode": bool(extreme),
        }
    )

    qualifies = (
        side is not None
        and favorite_probability >= float(favorite_win_gate)
        and float(result_probabilities.get("draw", 0.0)) <= float(draw_ceiling)
        and favorite_lambda >= float(favorite_lambda_gate)
        and lambda_gap >= float(lambda_gap_gate)
        and total_lambda >= float(total_lambda_gate)
    )
    if not qualifies:
        diagnostics["skip_reason"] = "gates_not_met"
        return [score_item(key, score_matrix[key]) for key in selected], diagnostics

    top_three = selected[:3]
    if not extreme and is_existing_tail_coverage(top_three, side):
        diagnostics["skip_reason"] = "tail_already_covered"
        return [score_item(key, score_matrix[key]) for key in selected], diagnostics

    selected_set = set(top_three)
    available = [
        key
        for key in tail_candidates(score_matrix, side, max_winner_goals)
        if key not in selected_set
    ]
    if not available:
        diagnostics["skip_reason"] = "no_candidate"
        return [score_item(key, score_matrix[key]) for key in selected], diagnostics

    candidate = max(
        available,
        key=lambda key: tail_utility(
            key,
            score_matrix.get(key, 0.0),
            side,
            extreme=extreme,
        ),
    )
    candidate_probability = float(score_matrix.get(candidate, 0.0))
    diagnostics["candidate_scoreline"] = f"{candidate[0]}-{candidate[1]}"
    diagnostics["candidate_probability"] = candidate_probability
    diagnostics["candidate_utility"] = tail_utility(
        candidate,
        candidate_probability,
        side,
        extreme=extreme,
    )
    if candidate_probability < probability_floor:
        diagnostics["skip_reason"] = "candidate_below_floor"
        return [score_item(key, score_matrix[key]) for key in selected], diagnostics

    top_three[-1] = candidate
    rebuilt: list[Tuple[int, int]] = []
    for key in [*top_three, *selected, *ranked_keys]:
        if key not in rebuilt and key in score_matrix:
            rebuilt.append(key)
        if len(rebuilt) >= top_n:
            break
    diagnostics["tail_risk_applied"] = True
    diagnostics["replaced_third_scoreline"] = f"{selected[2][0]}-{selected[2][1]}"
    return [score_item(key, score_matrix[key]) for key in rebuilt], diagnostics


class V29TailRiskScorelineModel:
    """Wrap V28 and add one conservative blowout candidate to the Top-3."""

    def __init__(
        self,
        base_model: v28.V28CurrentWorldCupFormModel,
        favorite_win_gate: float = DEFAULT_TAIL_FAVORITE_WIN_GATE,
        extreme_favorite_win_gate: float = DEFAULT_TAIL_EXTREME_FAVORITE_WIN_GATE,
        draw_ceiling: float = DEFAULT_TAIL_DRAW_CEILING,
        favorite_lambda_gate: float = DEFAULT_TAIL_FAVORITE_LAMBDA_GATE,
        extreme_lambda_gate: float = DEFAULT_TAIL_EXTREME_LAMBDA_GATE,
        lambda_gap_gate: float = DEFAULT_TAIL_LAMBDA_GAP_GATE,
        total_lambda_gate: float = DEFAULT_TAIL_TOTAL_LAMBDA_GATE,
        relative_floor: float = DEFAULT_TAIL_RELATIVE_FLOOR,
        absolute_floor: float = DEFAULT_TAIL_ABSOLUTE_FLOOR,
        max_winner_goals: int = DEFAULT_TAIL_MAX_WINNER_GOALS,
    ):
        self.base_model = base_model
        self.favorite_win_gate = float(np.clip(favorite_win_gate, 0.0, 1.0))
        self.extreme_favorite_win_gate = float(
            np.clip(extreme_favorite_win_gate, 0.0, 1.0)
        )
        self.draw_ceiling = float(np.clip(draw_ceiling, 0.0, 1.0))
        self.favorite_lambda_gate = float(max(favorite_lambda_gate, 0.0))
        self.extreme_lambda_gate = float(max(extreme_lambda_gate, 0.0))
        self.lambda_gap_gate = float(max(lambda_gap_gate, 0.0))
        self.total_lambda_gate = float(max(total_lambda_gate, 0.0))
        self.relative_floor = float(max(relative_floor, 0.0))
        self.absolute_floor = float(max(absolute_floor, 0.0))
        self.max_winner_goals = int(max(max_winner_goals, 3))
        self.training_data_summary = getattr(base_model, "training_data_summary", {})

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_model, name)

    def predict(self, *args, **kwargs) -> Dict[str, Any]:
        prediction = self.base_model.predict(*args, **kwargs)
        score_matrix = score_matrix_from_prediction(prediction)
        top_scorelines, diagnostics = select_top_scorelines_with_tail_risk(
            score_matrix,
            prediction["result_probabilities"],
            float(prediction["lambda_a"]),
            float(prediction["lambda_b"]),
            current_top_scorelines=prediction.get("top_scorelines", []),
            top_n=15,
            favorite_win_gate=self.favorite_win_gate,
            extreme_favorite_win_gate=self.extreme_favorite_win_gate,
            draw_ceiling=self.draw_ceiling,
            favorite_lambda_gate=self.favorite_lambda_gate,
            extreme_lambda_gate=self.extreme_lambda_gate,
            lambda_gap_gate=self.lambda_gap_gate,
            total_lambda_gate=self.total_lambda_gate,
            relative_floor=self.relative_floor,
            absolute_floor=self.absolute_floor,
            max_winner_goals=self.max_winner_goals,
        )
        prediction["top_scorelines"] = top_scorelines
        prediction["v29_adjustments"] = {
            "base_model": "v28_current_worldcup_form",
            "scoreline_policy": "tail_risk_top3_selector_only",
            "probability_matrix_changed": False,
            "scoreline_layer_affects_wdl": False,
            **diagnostics,
        }
        prediction["calibration_notes"] = {
            **prediction.get("calibration_notes", {}),
            "v29": prediction["v29_adjustments"],
            "tail_risk_policy": (
                "V29 leaves V28 probabilities unchanged and only replaces the "
                "third displayed Top-3 scoreline with one blowout candidate "
                "when favorite, lambda, draw, and probability-floor gates pass."
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
    include_observed_goals=True,
    include_fotmob_goal_stats=True,
    include_group_score_context=True,
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
    tail_favorite_win_gate=DEFAULT_TAIL_FAVORITE_WIN_GATE,
    tail_extreme_favorite_win_gate=DEFAULT_TAIL_EXTREME_FAVORITE_WIN_GATE,
    tail_draw_ceiling=DEFAULT_TAIL_DRAW_CEILING,
    tail_favorite_lambda_gate=DEFAULT_TAIL_FAVORITE_LAMBDA_GATE,
    tail_extreme_lambda_gate=DEFAULT_TAIL_EXTREME_LAMBDA_GATE,
    tail_lambda_gap_gate=DEFAULT_TAIL_LAMBDA_GAP_GATE,
    tail_total_lambda_gate=DEFAULT_TAIL_TOTAL_LAMBDA_GATE,
    tail_selector_relative_floor=DEFAULT_TAIL_RELATIVE_FLOOR,
    tail_selector_absolute_floor=DEFAULT_TAIL_ABSOLUTE_FLOOR,
    tail_max_winner_goals=DEFAULT_TAIL_MAX_WINNER_GOALS,
):
    base_model, data = v28.build_from_zip(
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
    )
    model = V29TailRiskScorelineModel(
        base_model,
        favorite_win_gate=tail_favorite_win_gate,
        extreme_favorite_win_gate=tail_extreme_favorite_win_gate,
        draw_ceiling=tail_draw_ceiling,
        favorite_lambda_gate=tail_favorite_lambda_gate,
        extreme_lambda_gate=tail_extreme_lambda_gate,
        lambda_gap_gate=tail_lambda_gap_gate,
        total_lambda_gate=tail_total_lambda_gate,
        relative_floor=tail_selector_relative_floor,
        absolute_floor=tail_selector_absolute_floor,
        max_winner_goals=tail_max_winner_goals,
    )
    model.training_data_summary = {
        **getattr(base_model, "training_data_summary", {}),
        "v29_scoreline_policy": "tail_risk_top3_selector_only",
        "v29_probability_matrix_changed": False,
        "v29_tail_favorite_win_gate": model.favorite_win_gate,
        "v29_tail_extreme_favorite_win_gate": model.extreme_favorite_win_gate,
        "v29_tail_draw_ceiling": model.draw_ceiling,
        "v29_tail_favorite_lambda_gate": model.favorite_lambda_gate,
        "v29_tail_extreme_lambda_gate": model.extreme_lambda_gate,
        "v29_tail_lambda_gap_gate": model.lambda_gap_gate,
        "v29_tail_total_lambda_gate": model.total_lambda_gate,
        "v29_tail_selector_relative_floor": model.relative_floor,
        "v29_tail_selector_absolute_floor": model.absolute_floor,
        "v29_tail_max_winner_goals": model.max_winner_goals,
    }
    return model, data


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    parser = argparse.ArgumentParser(
        description="Run V29: V28 with conservative tail-risk Top-3 selection."
    )
    parser.add_argument("--team-a", required=True)
    parser.add_argument("--team-b", required=True)
    parser.add_argument("--host-a", action="store_true")
    parser.add_argument("--host-b", action="store_true")
    parser.add_argument("--knockout", action="store_true")
    parser.add_argument("--outdir", default="outputs/outputs_v29_tail_risk")
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
    parser.add_argument("--blind-current-goals", action="store_true")
    parser.add_argument("--ignore-observed-goals", action="store_true")
    parser.add_argument("--ignore-fotmob-goal-stats", action="store_true")
    parser.add_argument("--disable-group-score-context", action="store_true")
    parser.add_argument("--tail-favorite-win-gate", type=float, default=DEFAULT_TAIL_FAVORITE_WIN_GATE)
    parser.add_argument("--tail-extreme-favorite-win-gate", type=float, default=DEFAULT_TAIL_EXTREME_FAVORITE_WIN_GATE)
    parser.add_argument("--tail-draw-ceiling", type=float, default=DEFAULT_TAIL_DRAW_CEILING)
    parser.add_argument("--tail-favorite-lambda-gate", type=float, default=DEFAULT_TAIL_FAVORITE_LAMBDA_GATE)
    parser.add_argument("--tail-extreme-lambda-gate", type=float, default=DEFAULT_TAIL_EXTREME_LAMBDA_GATE)
    parser.add_argument("--tail-lambda-gap-gate", type=float, default=DEFAULT_TAIL_LAMBDA_GAP_GATE)
    parser.add_argument("--tail-total-lambda-gate", type=float, default=DEFAULT_TAIL_TOTAL_LAMBDA_GATE)
    parser.add_argument("--tail-selector-relative-floor", type=float, default=DEFAULT_TAIL_RELATIVE_FLOOR)
    parser.add_argument("--tail-selector-absolute-floor", type=float, default=DEFAULT_TAIL_ABSOLUTE_FLOOR)
    parser.add_argument("--tail-max-winner-goals", type=int, default=DEFAULT_TAIL_MAX_WINNER_GOALS)
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
        include_observed_goals=not (
            args.blind_current_goals or args.ignore_observed_goals
        ),
        include_fotmob_goal_stats=not (
            args.blind_current_goals or args.ignore_fotmob_goal_stats
        ),
        include_group_score_context=not (
            args.blind_current_goals or args.disable_group_score_context
        ),
        tail_favorite_win_gate=args.tail_favorite_win_gate,
        tail_extreme_favorite_win_gate=args.tail_extreme_favorite_win_gate,
        tail_draw_ceiling=args.tail_draw_ceiling,
        tail_favorite_lambda_gate=args.tail_favorite_lambda_gate,
        tail_extreme_lambda_gate=args.tail_extreme_lambda_gate,
        tail_lambda_gap_gate=args.tail_lambda_gap_gate,
        tail_total_lambda_gate=args.tail_total_lambda_gate,
        tail_selector_relative_floor=args.tail_selector_relative_floor,
        tail_selector_absolute_floor=args.tail_selector_absolute_floor,
        tail_max_winner_goals=args.tail_max_winner_goals,
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
                "version": "v29-tail-risk-scoreline",
                "base_model": "v28-current-worldcup-form",
                "team_a": prediction["team_a"],
                "team_b": prediction["team_b"],
                "result_probabilities": prediction["result_probabilities"],
                "v29_adjustments": prediction["v29_adjustments"],
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
                "v29_adjustments": prediction["v29_adjustments"],
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

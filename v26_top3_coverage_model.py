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

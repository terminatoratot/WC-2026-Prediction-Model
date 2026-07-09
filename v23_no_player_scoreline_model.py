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

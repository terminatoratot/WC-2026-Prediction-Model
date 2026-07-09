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

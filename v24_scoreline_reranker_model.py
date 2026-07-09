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

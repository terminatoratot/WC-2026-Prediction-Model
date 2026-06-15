#!/usr/bin/env python3
"""V11 with live Elo updates and calibrated prediction post-processing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

import v11_wcq_results_model as v11


canon_team = v11.canon_team


@dataclass(frozen=True)
class V13Config:
    draw_decision_threshold: float = 0.2147
    close_elo_gap: float = 100.0
    close_match_draw_target: float = 0.218045
    large_elo_gap: float = 200.0
    large_mismatch_goal_std_scale: float = 1.10
    live_elo_k: float = 24.0


def _negative_binomial_pmf(k: int, mean: float, std_scale: float) -> float:
    """Return an overdispersed count PMF with the requested Poisson SD scale."""
    mean = max(float(mean), 1e-9)
    if std_scale <= 1.0:
        return math.exp(-mean) * mean**k / math.factorial(k)

    variance = std_scale**2 * mean
    shape = mean**2 / max(variance - mean, 1e-12)
    success_probability = shape / (shape + mean)
    return math.exp(
        math.lgamma(k + shape)
        - math.lgamma(shape)
        - math.lgamma(k + 1)
        + shape * math.log(success_probability)
        + k * math.log1p(-success_probability)
    )


def _overdispersed_score_matrix(
    lambda_a: float,
    lambda_b: float,
    max_goals: int,
    std_scale: float,
) -> Dict[Tuple[int, int], float]:
    probabilities_a = [
        _negative_binomial_pmf(goals, lambda_a, std_scale)
        for goals in range(max_goals + 1)
    ]
    probabilities_b = [
        _negative_binomial_pmf(goals, lambda_b, std_scale)
        for goals in range(max_goals + 1)
    ]
    matrix = {
        (goals_a, goals_b): probabilities_a[goals_a] * probabilities_b[goals_b]
        for goals_a in range(max_goals + 1)
        for goals_b in range(max_goals + 1)
    }
    total = sum(matrix.values())
    return {score: probability / total for score, probability in matrix.items()}


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


def _score_outputs(
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

        variance_widened = elo_gap > self.config.large_elo_gap
        if variance_widened:
            score_matrix = _overdispersed_score_matrix(
                prediction["lambda_a"],
                prediction["lambda_b"],
                max_goals,
                self.config.large_mismatch_goal_std_scale,
            )
        else:
            score_matrix = {
                (
                    int(item["team_a_goals"]),
                    int(item["team_b_goals"]),
                ): float(item["probability"])
                for item in prediction["scoreline_probabilities"]
            }

        score_matrix = v11.reweight_score_matrix_to_results(
            score_matrix,
            adjusted_results,
        )
        prediction.update(_score_outputs(score_matrix, max_goals))
        prediction["result_probabilities"] = v11.result_probs(score_matrix)

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
            "variance_widened": variance_widened,
            "goal_std_scale": (
                self.config.large_mismatch_goal_std_scale
                if variance_widened
                else 1.0
            ),
            "live_elo_k": self.config.live_elo_k,
        }
        prediction["calibration_notes"] = {
            **prediction.get("calibration_notes", {}),
            "v13": prediction["v13_adjustments"],
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
    base_model, data = v11.build_from_zip(
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
    )
    return V13LiveSignalModel(base_model), data

from __future__ import annotations

import unittest

import pandas as pd

import market_edge  # noqa: F401  (bundles v11/v13-v27/v28-v38/v39; populates sys.modules)
import v11_wcq_results_model as v11
import v27_total_goals_calibrated_model as v27


class V27TotalGoalsCalibratedModelTest(unittest.TestCase):
    def test_calibration_apply_preserves_wdl_marginals(self):
        matrix = {
            (0, 0): 0.12,
            (1, 1): 0.18,
            (1, 0): 0.25,
            (2, 0): 0.18,
            (3, 1): 0.05,
            (0, 1): 0.14,
            (0, 2): 0.08,
        }
        matrix = v27.normalize_matrix(matrix)
        target = v11.result_probs(matrix)
        model = v27.TotalGoalsCalibrationModel(
            multipliers={
                "overall": {
                    "all": [0.90, 0.95, 1.00, 1.08, 1.12, 1.0, 1.0],
                    "mid": [0.90, 0.95, 1.00, 1.08, 1.12, 1.0, 1.0],
                }
            },
            support={"overall": {"all": 100.0, "mid": 100.0}},
            max_total=6,
            min_bin_support=1.0,
        )

        adjusted, diagnostics = model.apply(
            matrix,
            target,
            lambda_total=2.5,
            knockout=False,
        )

        self.assertAlmostEqual(sum(adjusted.values()), 1.0)
        results = v11.result_probs(adjusted)
        for label, probability in target.items():
            self.assertAlmostEqual(results[label], probability)
        self.assertTrue(diagnostics["total_calibration_enabled"])

    def test_fitted_multipliers_can_learn_high_total_bias(self):
        rows = [
            {
                "actual_total": 4,
                "predicted_total_distribution": [0.10, 0.20, 0.40, 0.20, 0.10],
                "weight": 1.0,
            },
            {
                "actual_total": 4,
                "predicted_total_distribution": [0.10, 0.20, 0.40, 0.20, 0.10],
                "weight": 1.0,
            },
            {
                "actual_total": 3,
                "predicted_total_distribution": [0.10, 0.20, 0.40, 0.20, 0.10],
                "weight": 1.0,
            },
        ]

        actual, predicted, support = v27.weighted_calibration_counts(
            rows,
            max_total=4,
            smoothing=0.01,
        )
        multipliers = v27.make_multipliers(
            actual,
            predicted,
            strength=0.50,
            clip_low=0.80,
            clip_high=1.25,
        )

        self.assertEqual(support, 3.0)
        self.assertGreater(multipliers[4], 1.0)
        self.assertLess(multipliers[2], 1.0)

    def test_wrapper_keeps_result_probabilities(self):
        base_prediction = {
            "team_a": "Alpha",
            "team_b": "Beta",
            "lambda_a": 1.5,
            "lambda_b": 1.0,
            "result_probabilities": {
                "team_a_win": 0.50,
                "draw": 0.27,
                "team_b_win": 0.23,
            },
            "predicted_result": "team_a_win",
            "scoreline_probabilities": [
                {"team_a_goals": 0, "team_b_goals": 0, "probability": 0.12},
                {"team_a_goals": 1, "team_b_goals": 1, "probability": 0.15},
                {"team_a_goals": 1, "team_b_goals": 0, "probability": 0.25},
                {"team_a_goals": 2, "team_b_goals": 0, "probability": 0.15},
                {"team_a_goals": 2, "team_b_goals": 1, "probability": 0.10},
                {"team_a_goals": 0, "team_b_goals": 1, "probability": 0.14},
                {"team_a_goals": 0, "team_b_goals": 2, "probability": 0.09},
            ],
            "top_scorelines": [],
            "calibration_notes": {},
        }

        class BaseModel:
            training_data_summary = {}

            def predict(self, *args, **kwargs):
                return {
                    **base_prediction,
                    "result_probabilities": dict(base_prediction["result_probabilities"]),
                    "scoreline_probabilities": list(base_prediction["scoreline_probabilities"]),
                    "calibration_notes": {},
                }

        calibration = v27.TotalGoalsCalibrationModel(
            multipliers={
                "overall": {
                    "all": [0.90, 0.95, 1.00, 1.08, 1.12, 1.0, 1.0],
                    "mid": [0.90, 0.95, 1.00, 1.08, 1.12, 1.0, 1.0],
                }
            },
            support={"overall": {"all": 100.0, "mid": 100.0}},
            max_total=6,
            min_bin_support=1.0,
        )
        model = v27.V27TotalGoalsCalibratedModel(BaseModel(), calibration)

        prediction = model.predict("Alpha", "Beta", max_goals=3)

        self.assertEqual(
            prediction["result_probabilities"],
            base_prediction["result_probabilities"],
        )
        self.assertFalse(prediction["v27_adjustments"]["scoreline_layer_affects_wdl"])
        matrix = v27.score_matrix_from_prediction(prediction)
        results = v11.result_probs(matrix)
        for label, probability in base_prediction["result_probabilities"].items():
            self.assertAlmostEqual(results[label], probability)


if __name__ == "__main__":
    unittest.main()

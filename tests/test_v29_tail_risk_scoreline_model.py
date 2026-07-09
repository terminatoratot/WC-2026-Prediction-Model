from __future__ import annotations

import unittest

import v29_tail_risk_scoreline_model as v29


class V29TailRiskScorelineModelTest(unittest.TestCase):
    def test_selector_promotes_extreme_tail_candidate(self):
        matrix = {
            (3, 0): 0.100,
            (4, 0): 0.095,
            (3, 1): 0.061,
            (2, 0): 0.080,
            (1, 0): 0.070,
            (7, 1): 0.014,
            (6, 1): 0.018,
            (5, 1): 0.027,
        }
        current_top = [
            {"team_a_goals": 3, "team_b_goals": 0, "probability": 0.100},
            {"team_a_goals": 4, "team_b_goals": 0, "probability": 0.095},
            {"team_a_goals": 3, "team_b_goals": 1, "probability": 0.061},
        ]

        top, diagnostics = v29.select_top_scorelines_with_tail_risk(
            matrix,
            {"team_a_win": 0.82, "draw": 0.14, "team_b_win": 0.04},
            lambda_a=3.2,
            lambda_b=0.7,
            current_top_scorelines=current_top,
        )

        self.assertTrue(diagnostics["tail_risk_applied"])
        self.assertTrue(diagnostics["extreme_tail_mode"])
        self.assertEqual(
            [(item["team_a_goals"], item["team_b_goals"]) for item in top[:3]],
            [(3, 0), (4, 0), (7, 1)],
        )

    def test_selector_skips_non_extreme_when_tail_already_covered(self):
        matrix = {
            (2, 0): 0.12,
            (3, 0): 0.10,
            (3, 1): 0.07,
            (4, 1): 0.035,
        }
        current_top = [
            {"team_a_goals": 2, "team_b_goals": 0, "probability": 0.12},
            {"team_a_goals": 3, "team_b_goals": 0, "probability": 0.10},
            {"team_a_goals": 3, "team_b_goals": 1, "probability": 0.07},
        ]

        top, diagnostics = v29.select_top_scorelines_with_tail_risk(
            matrix,
            {"team_a_win": 0.70, "draw": 0.21, "team_b_win": 0.09},
            lambda_a=2.3,
            lambda_b=0.8,
            current_top_scorelines=current_top,
        )

        self.assertFalse(diagnostics["tail_risk_applied"])
        self.assertEqual(diagnostics["skip_reason"], "tail_already_covered")
        self.assertEqual(
            [(item["team_a_goals"], item["team_b_goals"]) for item in top[:3]],
            [(2, 0), (3, 0), (3, 1)],
        )

    def test_wrapper_only_changes_displayed_top_scorelines(self):
        base_prediction = {
            "team_a": "Alpha",
            "team_b": "Beta",
            "lambda_a": 2.8,
            "lambda_b": 0.7,
            "result_probabilities": {
                "team_a_win": 0.80,
                "draw": 0.15,
                "team_b_win": 0.05,
            },
            "predicted_result": "team_a_win",
            "scoreline_probabilities": [
                {"team_a_goals": 3, "team_b_goals": 0, "probability": 0.10},
                {"team_a_goals": 4, "team_b_goals": 0, "probability": 0.09},
                {"team_a_goals": 3, "team_b_goals": 1, "probability": 0.06},
                {"team_a_goals": 6, "team_b_goals": 1, "probability": 0.02},
            ],
            "top_scorelines": [
                {"team_a_goals": 3, "team_b_goals": 0, "probability": 0.10},
                {"team_a_goals": 4, "team_b_goals": 0, "probability": 0.09},
                {"team_a_goals": 3, "team_b_goals": 1, "probability": 0.06},
            ],
            "calibration_notes": {},
        }

        class BaseModel:
            training_data_summary = {}

            def predict(self, *args, **kwargs):
                return {
                    **base_prediction,
                    "result_probabilities": dict(
                        base_prediction["result_probabilities"]
                    ),
                    "scoreline_probabilities": list(
                        base_prediction["scoreline_probabilities"]
                    ),
                    "top_scorelines": list(base_prediction["top_scorelines"]),
                    "calibration_notes": {},
                }

        model = v29.V29TailRiskScorelineModel(BaseModel())
        prediction = model.predict("Alpha", "Beta")

        self.assertEqual(
            prediction["scoreline_probabilities"],
            base_prediction["scoreline_probabilities"],
        )
        self.assertEqual(
            prediction["result_probabilities"],
            base_prediction["result_probabilities"],
        )
        self.assertFalse(prediction["v29_adjustments"]["probability_matrix_changed"])
        self.assertTrue(prediction["v29_adjustments"]["tail_risk_applied"])


if __name__ == "__main__":
    unittest.main()

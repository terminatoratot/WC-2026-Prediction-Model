from __future__ import annotations

import unittest

import market_edge  # noqa: F401  (bundles v11/v13-v27/v28-v38/v39; populates sys.modules)
import v26_top3_coverage_model as v26


class V26Top3CoverageModelTest(unittest.TestCase):
    def test_selector_promotes_high_total_favorite_candidate_when_gated(self):
        matrix = {
            (1, 1): 0.12,
            (2, 0): 0.11,
            (1, 0): 0.10,
            (2, 1): 0.08,
            (3, 0): 0.07,
            (3, 1): 0.055,
            (4, 1): 0.03,
            (3, 2): 0.02,
            (0, 0): 0.09,
            (0, 1): 0.06,
        }

        top, diagnostics = v26.select_top_scorelines_with_coverage(
            matrix,
            {"team_a_win": 0.57, "draw": 0.26, "team_b_win": 0.17},
            lambda_a=1.9,
            lambda_b=0.8,
            top_n=5,
            tail_relative_floor=0.35,
            favorite_win_gate=0.55,
            total_lambda_gate=2.45,
            favorite_lambda_gate=1.55,
            draw_ceiling=0.30,
        )

        top_three = [(item["team_a_goals"], item["team_b_goals"]) for item in top[:3]]
        self.assertEqual(top_three, [(1, 1), (2, 0), (3, 1)])
        self.assertTrue(diagnostics["coverage_applied"])
        self.assertEqual(diagnostics["candidate_scoreline"], "3-1")

    def test_selector_skips_candidate_below_probability_floor(self):
        matrix = {
            (1, 1): 0.12,
            (2, 0): 0.11,
            (1, 0): 0.10,
            (3, 1): 0.03,
        }

        top, diagnostics = v26.select_top_scorelines_with_coverage(
            matrix,
            {"team_a_win": 0.57, "draw": 0.26, "team_b_win": 0.17},
            lambda_a=1.9,
            lambda_b=0.8,
            top_n=4,
            tail_relative_floor=0.35,
        )

        top_three = [(item["team_a_goals"], item["team_b_goals"]) for item in top[:3]]
        self.assertEqual(top_three, [(1, 1), (2, 0), (1, 0)])
        self.assertFalse(diagnostics["coverage_applied"])
        self.assertEqual(diagnostics["skip_reason"], "candidate_below_floor")

    def test_wrapper_only_changes_top_scorelines(self):
        base_prediction = {
            "team_a": "Alpha",
            "team_b": "Beta",
            "lambda_a": 1.9,
            "lambda_b": 0.8,
            "result_probabilities": {
                "team_a_win": 0.57,
                "draw": 0.26,
                "team_b_win": 0.17,
            },
            "predicted_result": "team_a_win",
            "scoreline_probabilities": [
                {"team_a_goals": 1, "team_b_goals": 1, "probability": 0.12},
                {"team_a_goals": 2, "team_b_goals": 0, "probability": 0.11},
                {"team_a_goals": 1, "team_b_goals": 0, "probability": 0.10},
                {"team_a_goals": 3, "team_b_goals": 1, "probability": 0.055},
            ],
            "top_scorelines": [
                {"team_a_goals": 1, "team_b_goals": 1, "probability": 0.12},
                {"team_a_goals": 2, "team_b_goals": 0, "probability": 0.11},
                {"team_a_goals": 1, "team_b_goals": 0, "probability": 0.10},
                {"team_a_goals": 3, "team_b_goals": 1, "probability": 0.055},
            ],
            "calibration_notes": {},
        }

        class BaseModel:
            training_data_summary = {}

            def predict(self, *args, **kwargs):
                return {
                    **base_prediction,
                    "result_probabilities": dict(base_prediction["result_probabilities"]),
                    "scoreline_probabilities": list(base_prediction["scoreline_probabilities"]),
                    "top_scorelines": list(base_prediction["top_scorelines"]),
                    "calibration_notes": {},
                }

        model = v26.V26Top3CoverageModel(BaseModel(), tail_relative_floor=0.35)
        prediction = model.predict("Alpha", "Beta")

        self.assertEqual(
            prediction["scoreline_probabilities"],
            base_prediction["scoreline_probabilities"],
        )
        self.assertEqual(
            prediction["result_probabilities"],
            base_prediction["result_probabilities"],
        )
        self.assertFalse(prediction["v26_adjustments"]["probability_matrix_changed"])
        self.assertEqual(
            [
                (item["team_a_goals"], item["team_b_goals"])
                for item in prediction["top_scorelines"][:3]
            ],
            [(1, 1), (2, 0), (3, 1)],
        )


if __name__ == "__main__":
    unittest.main()

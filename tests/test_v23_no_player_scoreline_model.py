from __future__ import annotations

import unittest
from unittest.mock import Mock

import market_edge  # noqa: F401  (bundles v11/v13-v27/v28-v38/v39; populates sys.modules)
import v11_wcq_results_model as v11
import v23_no_player_scoreline_model as v23


class V23NoPlayerScorelineModelTest(unittest.TestCase):
    def test_postprocessor_preserves_wdl_marginals(self):
        base = {
            (0, 0): 0.12,
            (1, 1): 0.18,
            (1, 0): 0.23,
            (2, 0): 0.19,
            (3, 0): 0.05,
            (0, 1): 0.12,
            (0, 2): 0.11,
        }
        target = v11.result_probs(base)

        adjusted, diagnostics = v23.postprocess_score_matrix(
            base,
            target,
            lambda_a=1.55,
            lambda_b=0.80,
            scoreline_layer_weight=0.55,
            favorite_tail_strength=0.32,
            favorite_tail_threshold=0.56,
            reranker_strength=0.18,
        )

        self.assertEqual(set(adjusted), set(base))
        self.assertAlmostEqual(sum(adjusted.values()), 1.0)
        results = v11.result_probs(adjusted)
        for label, probability in target.items():
            self.assertAlmostEqual(results[label], probability)
        self.assertIn("scoreline_layer_weight", diagnostics)

    def test_wrapper_uses_outcome_model_and_marks_player_data_disabled(self):
        base_prediction = {
            "team_a": "Alpha",
            "team_b": "Beta",
            "lambda_a": 1.4,
            "lambda_b": 0.8,
            "result_probabilities": {
                "team_a_win": 0.62,
                "draw": 0.23,
                "team_b_win": 0.15,
            },
            "predicted_result": "team_a_win",
            "scoreline_probabilities": [
                {"team_a_goals": 0, "team_b_goals": 0, "probability": 0.10},
                {"team_a_goals": 1, "team_b_goals": 0, "probability": 0.24},
                {"team_a_goals": 2, "team_b_goals": 0, "probability": 0.20},
                {"team_a_goals": 3, "team_b_goals": 0, "probability": 0.08},
                {"team_a_goals": 1, "team_b_goals": 1, "probability": 0.13},
                {"team_a_goals": 0, "team_b_goals": 1, "probability": 0.08},
                {"team_a_goals": 0, "team_b_goals": 2, "probability": 0.07},
                {"team_a_goals": 2, "team_b_goals": 1, "probability": 0.10},
            ],
            "top_scorelines": [],
            "calibration_notes": {},
        }
        outcome_model = Mock()
        outcome_model.predict.return_value = base_prediction
        player_model = Mock()
        base_model = Mock(outcome_model=outcome_model)
        base_model.predict = player_model
        model = v23.V23NoPlayerScorelineModel(base_model)

        prediction = model.predict("Alpha", "Beta", max_goals=3)

        outcome_model.predict.assert_called_once()
        player_model.assert_not_called()
        self.assertEqual(
            prediction["result_probabilities"],
            base_prediction["result_probabilities"],
        )
        self.assertFalse(prediction["v23_adjustments"]["player_or_squad_data_used"])
        self.assertFalse(prediction["v23_adjustments"]["scoreline_layer_affects_wdl"])

    def test_diversity_can_promote_favorite_tail_into_top_three(self):
        matrix = {
            (1, 0): 0.18,
            (2, 0): 0.16,
            (1, 1): 0.15,
            (3, 0): 0.10,
            (2, 1): 0.09,
            (0, 0): 0.08,
            (0, 1): 0.07,
            (0, 2): 0.06,
            (4, 0): 0.05,
            (3, 1): 0.04,
            (1, 2): 0.02,
        }
        matrix = v23.normalize_matrix(matrix)

        top = v23.diversify_top_scorelines(
            matrix,
            {
                "team_a_win": 0.65,
                "draw": 0.22,
                "team_b_win": 0.13,
            },
            lambda_a=1.75,
            lambda_b=0.65,
            top_n=5,
            relative_floor=0.42,
        )

        top_three = {
            (item["team_a_goals"], item["team_b_goals"]) for item in top[:3]
        }
        self.assertIn((3, 0), top_three)


if __name__ == "__main__":
    unittest.main()

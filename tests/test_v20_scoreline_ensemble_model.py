from __future__ import annotations

import unittest
from unittest.mock import Mock

import market_edge  # noqa: F401  (bundles v11/v13-v27/v28-v38/v39; populates sys.modules)
import v11_wcq_results_model as v11
import v20_scoreline_ensemble_model as v20


class V20ScorelineEnsembleModelTest(unittest.TestCase):
    def test_blend_score_matrices_uses_requested_weight(self):
        base = {(1, 0): 0.70, (1, 1): 0.30}
        adjusted = {(1, 0): 0.10, (1, 1): 0.90}

        blended = v20.blend_score_matrices(base, adjusted, 0.25)

        self.assertAlmostEqual(blended[(1, 0)], 0.55)
        self.assertAlmostEqual(blended[(1, 1)], 0.45)
        self.assertAlmostEqual(sum(blended.values()), 1.0)

    def test_blending_preserves_matching_result_marginals(self):
        base = {(1, 0): 0.50, (1, 1): 0.25, (0, 1): 0.25}
        adjusted = {(2, 1): 0.50, (0, 0): 0.25, (1, 2): 0.25}

        blended = v20.blend_score_matrices(base, adjusted, 0.35)

        self.assertEqual(v11.result_probs(blended), v11.result_probs(base))

    def test_wrapper_preserves_v15_wdl_and_disables_rank_stabilizer(self):
        base_prediction = {
            "team_a": "Alpha",
            "team_b": "Beta",
            "lambda_a": 1.2,
            "lambda_b": 0.8,
            "result_probabilities": {
                "team_a_win": 0.50,
                "draw": 0.25,
                "team_b_win": 0.25,
            },
            "predicted_result": "team_a_win",
            "scoreline_probabilities": [
                {"team_a_goals": 0, "team_b_goals": 0, "probability": 0.10},
                {"team_a_goals": 0, "team_b_goals": 1, "probability": 0.25},
                {"team_a_goals": 1, "team_b_goals": 0, "probability": 0.40},
                {"team_a_goals": 1, "team_b_goals": 1, "probability": 0.15},
                {"team_a_goals": 2, "team_b_goals": 1, "probability": 0.10},
            ],
            "top_scorelines": [],
            "calibration_notes": {"dixon_coles_rho": -0.08},
        }
        base_model = Mock()
        base_model.predict.return_value = dict(base_prediction)
        model = v20.V20ScorelineEnsembleModel(
            base_model,
            squad_profiles={},
            v18_scoreline_weight=0.35,
        )

        prediction = model.predict("Alpha", "Beta", max_goals=2)

        self.assertEqual(
            prediction["result_probabilities"],
            base_prediction["result_probabilities"],
        )
        self.assertEqual(prediction["predicted_result"], "team_a_win")
        self.assertFalse(prediction["v20_adjustments"]["scoreline_blend_affects_wdl"])
        self.assertFalse(prediction["v20_adjustments"]["rank_stabilizer"])
        self.assertAlmostEqual(
            sum(item["probability"] for item in prediction["scoreline_probabilities"]),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()

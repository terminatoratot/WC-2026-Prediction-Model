from __future__ import annotations

import copy
import unittest

import pandas as pd

from v13_live_signal_model import V13LiveSignalModel


class StubV11Model:
    def __init__(self, elo_gap: float = 50.0):
        self.elo_gap = elo_gap
        self.latest_elo = {}
        self.last_prediction = None

    def make_features(self, *args, **kwargs):
        return pd.DataFrame([{"elo_diff": self.elo_gap}])

    def predict(self, *args, **kwargs):
        prediction = {
            "team_a": "Alpha",
            "team_b": "Beta",
            "lambda_a": 1.4,
            "lambda_b": 1.0,
            "result_probabilities": {
                "team_a_win": 0.55,
                "draw": 0.15,
                "team_b_win": 0.30,
            },
            "draw_model_probability": 0.23,
            "top_scorelines": [
                {"team_a_goals": 1, "team_b_goals": 0, "probability": 0.19},
                {"team_a_goals": 1, "team_b_goals": 1, "probability": 0.16},
            ],
            "scoreline_probabilities": [
                {"team_a_goals": 0, "team_b_goals": 0, "probability": 0.12},
                {"team_a_goals": 1, "team_b_goals": 0, "probability": 0.19},
                {"team_a_goals": 1, "team_b_goals": 1, "probability": 0.16},
                {"team_a_goals": 2, "team_b_goals": 1, "probability": 0.13},
                {"team_a_goals": 0, "team_b_goals": 1, "probability": 0.11},
                {"team_a_goals": 0, "team_b_goals": 2, "probability": 0.09},
                {"team_a_goals": 2, "team_b_goals": 0, "probability": 0.20},
            ],
            "spread_probabilities": {"0": 0.28, "1": 0.32},
            "total_goal_probabilities": {"0": 0.12, "1": 0.30},
            "over_under_probabilities": {
                "over_2.5": 0.44,
                "under_2.5": 0.56,
            },
            "calibration_notes": {"dixon_coles_rho": -0.08},
        }
        self.last_prediction = copy.deepcopy(prediction)
        return prediction


class V13HybridModelTest(unittest.TestCase):
    def test_v13_wdl_keeps_v11_exact_score_outputs_unchanged(self):
        base_model = StubV11Model()
        model = V13LiveSignalModel(base_model)

        prediction = model.predict("Alpha", "Beta")

        self.assertAlmostEqual(
            prediction["result_probabilities"]["draw"],
            model.config.close_match_draw_target,
        )
        self.assertEqual(prediction["predicted_result"], "draw")
        for key in (
            "top_scorelines",
            "scoreline_probabilities",
            "spread_probabilities",
            "total_goal_probabilities",
            "over_under_probabilities",
        ):
            self.assertEqual(prediction[key], base_model.last_prediction[key])

        adjustments = prediction["v13_adjustments"]
        self.assertEqual(adjustments["wdl_model"], "v13")
        self.assertEqual(
            adjustments["exact_score_model"],
            "v11_poisson_dixon_coles_reweighted",
        )
        self.assertTrue(adjustments["exact_score_result_reweighting"])
        self.assertEqual(adjustments["exact_score_dixon_coles_rho"], -0.08)
        self.assertFalse(adjustments["variance_widened"])


if __name__ == "__main__":
    unittest.main()

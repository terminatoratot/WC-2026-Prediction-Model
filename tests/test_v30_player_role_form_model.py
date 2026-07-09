from __future__ import annotations

import unittest

import v11_wcq_results_model as v11
import v15_catboost_model as v15
import v30_player_role_form_model as v30


class V30PlayerRoleFormModelTest(unittest.TestCase):
    def test_role_profiles_load_from_fotmob_leaders(self):
        profiles = v30.build_player_role_profiles("data/fotmob_stat_leaders_clean.csv")
        match_profiles = v30.build_match_player_role_profiles(
            "data/fotmob_match_player_stats_clean.csv",
            "data/fotmob_match_lineups_clean.csv",
            "data/fotmob_match_substitutions_clean.csv",
            "data/fotmob_match_keeper_stats_clean.csv",
        )

        self.assertGreater(len(profiles), 0)
        self.assertGreaterEqual(len(match_profiles), 40)
        self.assertIn("Argentina", profiles)
        self.assertIn("Switzerland", match_profiles)
        self.assertGreater(profiles["Argentina"].rows, 0)
        self.assertGreater(match_profiles["Switzerland"].rows, 0)
        self.assertGreaterEqual(profiles["Argentina"].coverage, 0.0)
        self.assertLessEqual(profiles["Argentina"].coverage, 1.0)

    def test_wrapper_keeps_score_matrix_coherent_with_final_wdl(self):
        base_matrix = v11.apply_dixon_coles_adjustment(
            v11.poisson_score_matrix(1.5, 1.0, max_goals=3),
            1.5,
            1.0,
        )
        result_probabilities = v11.result_probs(base_matrix)
        base_prediction = {
            "team_a": "Alpha",
            "team_b": "Beta",
            "lambda_a": 1.5,
            "lambda_b": 1.0,
            "result_probabilities": result_probabilities,
            "predicted_result": max(result_probabilities, key=result_probabilities.get),
            "calibration_notes": {},
            **v15.score_outputs(base_matrix, max_goals=3),
            "v29_adjustments": {},
        }

        class BaseModel:
            training_data_summary = {}
            favorite_win_gate = 0.99
            extreme_favorite_win_gate = 0.99
            draw_ceiling = 0.01
            favorite_lambda_gate = 99.0
            extreme_lambda_gate = 99.0
            lambda_gap_gate = 99.0
            total_lambda_gate = 99.0
            relative_floor = 1.0
            absolute_floor = 1.0
            max_winner_goals = 7

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

        profiles = {
            "Alpha": v30.PlayerRoleProfile(
                team="Alpha",
                attacker=1.0,
                creator=0.5,
                coverage=1.0,
                rows=4,
            ),
            "Beta": v30.PlayerRoleProfile(
                team="Beta",
                defender=-0.6,
                defensive_fragility=0.6,
                coverage=1.0,
                rows=4,
            ),
        }
        model = v30.V30PlayerRoleFormModel(BaseModel(), profiles)

        prediction = model.predict("Alpha", "Beta", max_goals=3)

        self.assertTrue(prediction["v30_adjustments"]["role_layer_affects_wdl"])
        self.assertGreater(
            prediction["v30_adjustments"]["role_lambda_a"],
            base_prediction["lambda_a"],
        )
        matrix = v30.score_matrix_from_prediction(prediction)
        result_from_matrix = v11.result_probs(matrix)
        for label, probability in prediction["result_probabilities"].items():
            self.assertAlmostEqual(result_from_matrix[label], probability)


if __name__ == "__main__":
    unittest.main()

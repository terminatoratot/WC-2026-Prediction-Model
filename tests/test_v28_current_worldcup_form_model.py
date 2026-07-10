from __future__ import annotations

import unittest

import market_edge  # noqa: F401  (bundles v11/v13-v27/v28-v38/v39; populates sys.modules)
import v11_wcq_results_model as v11
import v15_catboost_model as v15
import v27_total_goals_calibrated_model as v27
import v28_current_worldcup_form_model as v28


class V28CurrentWorldCupFormModelTest(unittest.TestCase):
    def test_current_feature_builders_load_local_sources(self):
        observed = v28.build_observed_form(
            "data/wc2026_observed_matches_from_screenshots.csv"
        )
        fotmob = v28.build_fotmob_form("data/fotmob_stat_leaders_clean.csv")
        group_context = v28.build_group_context(
            "data/wc2026_observed_matches_from_screenshots.csv"
        )
        forms = v28.merge_current_forms(observed, fotmob, group_context)
        blind_observed = v28.build_observed_form(
            "data/wc2026_observed_matches_from_screenshots.csv",
            include_goals=False,
        )
        blind_fotmob = v28.build_fotmob_form(
            "data/fotmob_stat_leaders_clean.csv",
            include_goal_stats=False,
        )
        blind_group_context = v28.build_group_context(
            "data/wc2026_observed_matches_from_screenshots.csv",
            include_score_context=False,
        )

        self.assertGreater(len(observed), 0)
        self.assertGreater(len(fotmob), 0)
        self.assertGreater(len(group_context), 0)
        self.assertEqual(len(blind_observed), len(observed))
        self.assertLess(len(blind_fotmob), len(fotmob))
        self.assertEqual(len(blind_group_context), 0)
        self.assertIn("Argentina", forms)
        self.assertGreater(forms["Argentina"].fotmob_rows, 0)

    def test_wrapper_keeps_score_matrix_coherent_with_blended_wdl(self):
        base_matrix = v11.apply_dixon_coles_adjustment(
            v11.poisson_score_matrix(1.5, 1.0, max_goals=3),
            1.5,
            1.0,
        )
        base_result_probabilities = v11.result_probs(base_matrix)
        base_prediction = {
            "team_a": "Alpha",
            "team_b": "Beta",
            "lambda_a": 1.5,
            "lambda_b": 1.0,
            "result_probabilities": base_result_probabilities,
            "predicted_result": max(
                base_result_probabilities,
                key=base_result_probabilities.get,
            ),
            "calibration_notes": {},
            **v15.score_outputs(base_matrix, max_goals=3),
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

        identity_calibration = v27.TotalGoalsCalibrationModel(
            multipliers={"overall": {"all": [1.0 for _ in range(7)]}},
            support={"overall": {"all": 100.0}},
            max_total=6,
            min_bin_support=1.0,
        )
        forms = {
            "Alpha": v28.CurrentTeamForm(
                team="Alpha",
                observed_attack=1.0,
                observed_defense=0.2,
                observed_tempo=0.5,
                group_pressure=0.2,
            ),
            "Beta": v28.CurrentTeamForm(
                team="Beta",
                observed_attack=-0.5,
                observed_defense=-0.6,
                observed_tempo=0.1,
                group_pressure=0.0,
            ),
        }
        model = v28.V28CurrentWorldCupFormModel(
            BaseModel(),
            forms,
            identity_calibration,
            current_wdl_blend=0.4,
            current_scoreline_blend=0.8,
            total_calibration_blend=0.0,
        )

        prediction = model.predict("Alpha", "Beta", max_goals=3)

        self.assertTrue(prediction["v28_adjustments"]["current_layer_affects_wdl"])
        self.assertAlmostEqual(sum(prediction["result_probabilities"].values()), 1.0)
        score_matrix = v28.score_matrix_from_prediction(prediction)
        result_probabilities = v11.result_probs(score_matrix)
        for label, probability in prediction["result_probabilities"].items():
            self.assertAlmostEqual(result_probabilities[label], probability)
        self.assertGreater(
            prediction["v28_adjustments"]["current_lambda_a"],
            base_prediction["lambda_a"],
        )


if __name__ == "__main__":
    unittest.main()

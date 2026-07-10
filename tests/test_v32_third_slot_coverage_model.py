from __future__ import annotations

import unittest

import market_edge  # noqa: F401  (bundles v11/v13-v27/v28-v38/v39; populates sys.modules)
import v30_player_role_form_model as v30
import v32_third_slot_coverage_model as v32


class V32ThirdSlotCoverageModelTest(unittest.TestCase):
    def test_current_xg_profiles_load_from_match_data(self):
        profiles = v32.build_current_xg_profiles(
            "data/fotmob_match_player_stats_clean.csv",
            "data/fotmob_match_lineups_clean.csv",
            "data/fotmob_match_substitutions_clean.csv",
            "data/fotmob_match_keeper_stats_clean.csv",
        )

        self.assertGreaterEqual(len(profiles), 40)
        self.assertIn("Switzerland", profiles)
        self.assertGreater(profiles["Switzerland"].rows, 0)
        self.assertGreaterEqual(profiles["Switzerland"].coverage, 0.0)
        self.assertLessEqual(profiles["Switzerland"].coverage, 1.0)

    def test_selector_uses_role_and_xg_to_pick_third_slot(self):
        matrix = {
            (2, 0): 0.130,
            (1, 0): 0.115,
            (2, 1): 0.090,
            (3, 0): 0.078,
            (1, 1): 0.070,
        }
        current_top = [
            {"team_a_goals": 2, "team_b_goals": 0, "probability": 0.130},
            {"team_a_goals": 1, "team_b_goals": 0, "probability": 0.115},
            {"team_a_goals": 2, "team_b_goals": 1, "probability": 0.090},
        ]
        role_a = v30.PlayerRoleProfile(
            team="Alpha",
            attacker=1.1,
            creator=1.0,
            defender=1.0,
            keeper=0.4,
            coverage=0.9,
        )
        role_b = v30.PlayerRoleProfile(
            team="Beta",
            attacker=-0.7,
            defender=-0.9,
            keeper=-0.4,
            defensive_fragility=0.9,
            coverage=0.9,
        )
        xg_a = v32.CurrentXGProfile(
            team="Alpha",
            attack_pressure=1.1,
            creative_pressure=0.8,
            keeper_form=0.4,
            coverage=0.9,
        )
        xg_b = v32.CurrentXGProfile(
            team="Beta",
            attack_pressure=-0.9,
            creative_pressure=-0.7,
            defensive_leakiness=1.0,
            coverage=0.9,
        )

        top, diagnostics = v32.select_top_scorelines_with_third_slot_coverage(
            matrix,
            {"team_a_win": 0.68, "draw": 0.22, "team_b_win": 0.10},
            role_a,
            role_b,
            xg_a,
            xg_b,
            current_top_scorelines=current_top,
            min_utility_gain=0.0,
        )

        self.assertTrue(diagnostics["third_slot_changed"])
        self.assertEqual(
            [(item["team_a_goals"], item["team_b_goals"]) for item in top[:3]],
            [(2, 0), (1, 0), (3, 0)],
        )

    def test_wrapper_keeps_probabilities_and_wdl_unchanged(self):
        base_prediction = {
            "team_a": "Alpha",
            "team_b": "Beta",
            "lambda_a": 2.1,
            "lambda_b": 0.8,
            "result_probabilities": {
                "team_a_win": 0.68,
                "draw": 0.22,
                "team_b_win": 0.10,
            },
            "predicted_result": "team_a_win",
            "scoreline_probabilities": [
                {"team_a_goals": 2, "team_b_goals": 0, "probability": 0.13},
                {"team_a_goals": 1, "team_b_goals": 0, "probability": 0.11},
                {"team_a_goals": 2, "team_b_goals": 1, "probability": 0.09},
                {"team_a_goals": 3, "team_b_goals": 0, "probability": 0.08},
            ],
            "top_scorelines": [
                {"team_a_goals": 2, "team_b_goals": 0, "probability": 0.13},
                {"team_a_goals": 1, "team_b_goals": 0, "probability": 0.11},
                {"team_a_goals": 2, "team_b_goals": 1, "probability": 0.09},
            ],
            "calibration_notes": {},
            "v29_adjustments": {},
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
                    "v29_adjustments": {},
                }

        role_profiles = {
            "Alpha": v30.PlayerRoleProfile(
                team="Alpha",
                attacker=1.0,
                creator=0.8,
                defender=0.9,
                keeper=0.4,
                coverage=1.0,
            ),
            "Beta": v30.PlayerRoleProfile(
                team="Beta",
                attacker=-0.7,
                defender=-0.8,
                defensive_fragility=0.9,
                coverage=1.0,
            ),
        }
        xg_profiles = {
            "Alpha": v32.CurrentXGProfile(
                team="Alpha",
                attack_pressure=1.0,
                creative_pressure=0.8,
                keeper_form=0.4,
                coverage=1.0,
            ),
            "Beta": v32.CurrentXGProfile(
                team="Beta",
                attack_pressure=-0.8,
                defensive_leakiness=1.0,
                coverage=1.0,
            ),
        }
        model = v32.V32ThirdSlotCoverageModel(BaseModel(), role_profiles, xg_profiles)
        prediction = model.predict("Alpha", "Beta")

        self.assertEqual(
            prediction["scoreline_probabilities"],
            base_prediction["scoreline_probabilities"],
        )
        self.assertEqual(
            prediction["result_probabilities"],
            base_prediction["result_probabilities"],
        )
        self.assertFalse(prediction["v32_adjustments"]["probability_matrix_changed"])
        self.assertFalse(prediction["v32_adjustments"]["scoreline_layer_affects_wdl"])


if __name__ == "__main__":
    unittest.main()

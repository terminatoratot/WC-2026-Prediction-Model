from __future__ import annotations

import unittest

import market_edge  # noqa: F401  (bundles v11/v13-v27/v28-v38/v39; populates sys.modules)
import v31_gated_role_selector_model as v31
import v30_player_role_form_model as v30


class V31GatedRoleSelectorModelTest(unittest.TestCase):
    def test_role_selector_promotes_clean_sheet_candidate_when_confident(self):
        matrix = {
            (1, 1): 0.130,
            (2, 1): 0.120,
            (3, 1): 0.080,
            (2, 0): 0.070,
            (3, 0): 0.060,
        }
        current_top = [
            {"team_a_goals": 1, "team_b_goals": 1, "probability": 0.130},
            {"team_a_goals": 2, "team_b_goals": 1, "probability": 0.120},
            {"team_a_goals": 3, "team_b_goals": 1, "probability": 0.080},
        ]
        role_a = v30.PlayerRoleProfile(
            team="Alpha",
            attacker=1.2,
            creator=1.0,
            defender=1.1,
            keeper=0.6,
            coverage=0.9,
        )
        role_b = v30.PlayerRoleProfile(
            team="Beta",
            attacker=-0.8,
            creator=-0.4,
            defender=-0.6,
            keeper=-0.4,
            defensive_fragility=0.8,
            coverage=0.9,
        )

        top, diagnostics = v31.select_top_scorelines_with_gated_roles(
            matrix,
            {"team_a_win": 0.64, "draw": 0.24, "team_b_win": 0.12},
            role_a,
            role_b,
            current_top_scorelines=current_top,
            relative_floor=0.40,
        )

        self.assertTrue(diagnostics["role_selector_applied"])
        self.assertEqual(
            [(item["team_a_goals"], item["team_b_goals"]) for item in top[:2]],
            [(1, 1), (2, 1)],
        )
        self.assertEqual(top[2]["team_b_goals"], 0)

    def test_role_selector_skips_when_coverage_is_low(self):
        matrix = {
            (2, 1): 0.120,
            (1, 1): 0.110,
            (3, 1): 0.080,
            (3, 0): 0.060,
        }
        current_top = [
            {"team_a_goals": 2, "team_b_goals": 1, "probability": 0.120},
            {"team_a_goals": 1, "team_b_goals": 1, "probability": 0.110},
            {"team_a_goals": 3, "team_b_goals": 1, "probability": 0.080},
        ]
        role_a = v30.PlayerRoleProfile(
            team="Alpha",
            attacker=2.0,
            creator=2.0,
            defender=2.0,
            keeper=1.0,
            coverage=0.2,
        )
        role_b = v30.PlayerRoleProfile(
            team="Beta",
            attacker=-1.0,
            defender=-1.0,
            defensive_fragility=1.0,
            coverage=0.9,
        )

        top, diagnostics = v31.select_top_scorelines_with_gated_roles(
            matrix,
            {"team_a_win": 0.70, "draw": 0.20, "team_b_win": 0.10},
            role_a,
            role_b,
            current_top_scorelines=current_top,
        )

        self.assertFalse(diagnostics["role_selector_applied"])
        self.assertEqual(diagnostics["skip_reason"], "coverage_gate_not_met")
        self.assertEqual(
            [(item["team_a_goals"], item["team_b_goals"]) for item in top[:3]],
            [(2, 1), (1, 1), (3, 1)],
        )

    def test_wrapper_keeps_v29_probabilities_unchanged(self):
        base_prediction = {
            "team_a": "Alpha",
            "team_b": "Beta",
            "lambda_a": 2.0,
            "lambda_b": 0.8,
            "result_probabilities": {
                "team_a_win": 0.66,
                "draw": 0.22,
                "team_b_win": 0.12,
            },
            "predicted_result": "team_a_win",
            "scoreline_probabilities": [
                {"team_a_goals": 2, "team_b_goals": 1, "probability": 0.12},
                {"team_a_goals": 1, "team_b_goals": 1, "probability": 0.11},
                {"team_a_goals": 3, "team_b_goals": 1, "probability": 0.08},
                {"team_a_goals": 3, "team_b_goals": 0, "probability": 0.06},
            ],
            "top_scorelines": [
                {"team_a_goals": 2, "team_b_goals": 1, "probability": 0.12},
                {"team_a_goals": 1, "team_b_goals": 1, "probability": 0.11},
                {"team_a_goals": 3, "team_b_goals": 1, "probability": 0.08},
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

        profiles = {
            "Alpha": v30.PlayerRoleProfile(
                team="Alpha",
                attacker=1.4,
                creator=1.2,
                defender=1.0,
                keeper=0.5,
                coverage=1.0,
            ),
            "Beta": v30.PlayerRoleProfile(
                team="Beta",
                attacker=-0.8,
                defender=-0.8,
                keeper=-0.4,
                defensive_fragility=1.0,
                coverage=1.0,
            ),
        }
        model = v31.V31GatedRoleSelectorModel(BaseModel(), profiles)
        prediction = model.predict("Alpha", "Beta")

        self.assertEqual(
            prediction["scoreline_probabilities"],
            base_prediction["scoreline_probabilities"],
        )
        self.assertEqual(
            prediction["result_probabilities"],
            base_prediction["result_probabilities"],
        )
        self.assertFalse(prediction["v31_adjustments"]["probability_matrix_changed"])
        self.assertFalse(prediction["v31_adjustments"]["scoreline_layer_affects_wdl"])


if __name__ == "__main__":
    unittest.main()

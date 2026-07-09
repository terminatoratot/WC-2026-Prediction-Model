from __future__ import annotations

import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd

import v18_hybrid_elo_form_player_model as v18


class V18HybridSquadModelTest(unittest.TestCase):
    def test_name_match_handles_initials_and_extra_surnames(self):
        squad_row = pd.Series(
            {
                "short_name": "K. Mbappé",
                "long_name": "Kylian Mbappé Lottin",
            }
        )

        self.assertGreaterEqual(
            v18.name_match_score(squad_row, "Kylian Mbappé"),
            0.94,
        )

    def test_squad_file_ratings_are_not_used_in_fc_profile(self):
        squads = pd.DataFrame(
            [
                {
                    "fifa_version": 26,
                    "nationality_name": "France",
                    "short_name": "K. Mbappé",
                    "long_name": "Kylian Mbappé Lottin",
                    "player_positions": "ST, LW",
                    "overall": 1,
                }
            ]
        )
        fcratings = pd.DataFrame(
            [
                {
                    "country": "France",
                    "player_name": "Kylian Mbappé",
                    "position": "ST",
                    "ovr": 91,
                    "pac": 96,
                    "sho": 91,
                    "pas": 81,
                    "dri": 92,
                    "def": 37,
                    "phy": 76,
                }
            ]
        )
        squads["team"] = squads["nationality_name"].map(v18.normalize_team)
        squads["positions"] = squads["player_positions"].map(v18.position_set)
        squads["role"] = squads["player_positions"].map(v18.v15.player_role)
        fcratings["team"] = fcratings["country"].map(v18.normalize_team)
        fcratings["positions"] = fcratings["position"].map(v18.position_set)

        profiles = v18.build_current_fcratings_squad_profiles(
            squads,
            fcratings,
        )
        profile = profiles["France"]

        expected_attack = 0.40 * 91 + 0.25 * 92 + 0.20 * 96 + 0.15 * 81
        self.assertAlmostEqual(profile["attack_raw"], expected_attack)
        self.assertNotEqual(profile["attack_raw"], 1)
        self.assertEqual(profile["matched_players"], 1)

    def test_unmatched_players_get_shrunken_positional_imputation(self):
        squads = pd.DataFrame(
            [
                {
                    "fifa_version": 26,
                    "nationality_name": "France",
                    "short_name": "Known",
                    "long_name": "Known Striker",
                    "player_positions": "ST",
                },
                {
                    "fifa_version": 26,
                    "nationality_name": "France",
                    "short_name": "Missing",
                    "long_name": "Ghost Winger",
                    "player_positions": "ST",
                },
            ]
        )
        fcratings = pd.DataFrame(
            [
                {
                    "country": "France",
                    "player_name": "Known Striker",
                    "position": "ST",
                    "ovr": 80,
                    "pac": 80,
                    "sho": 80,
                    "pas": 80,
                    "dri": 80,
                    "def": 30,
                    "phy": 80,
                },
                {
                    "country": "France",
                    "player_name": "Reserve Striker",
                    "position": "ST",
                    "ovr": 70,
                    "pac": 70,
                    "sho": 70,
                    "pas": 70,
                    "dri": 70,
                    "def": 25,
                    "phy": 70,
                },
            ]
        )
        squads["team"] = squads["nationality_name"].map(v18.normalize_team)
        squads["positions"] = squads["player_positions"].map(v18.position_set)
        squads["role"] = squads["player_positions"].map(v18.v15.player_role)
        fcratings["team"] = fcratings["country"].map(v18.normalize_team)
        fcratings["positions"] = fcratings["position"].map(v18.position_set)

        profiles = v18.build_current_fcratings_squad_profiles(
            squads,
            fcratings,
        )
        profile = profiles["France"]

        self.assertEqual(profile["matched_players"], 1)
        self.assertEqual(profile["imputed_players"], 1)
        self.assertGreater(profile["rating_confidence"], profile["match_coverage"])
        self.assertIn("imputed_player_examples", profile)

    def test_log_adjustment_is_coverage_shrunk_and_capped(self):
        model = v18.V18HybridSquadModel(
            base_model=object(),
            squad_profiles={},
            beta_attack=1.0,
            beta_midfield=0.0,
            beta_keeper=0.0,
            max_log_adjustment=0.2,
        )
        profile_a = {
            "match_coverage": 0.25,
            "attack_z": 5.0,
            "defense_z": 0.0,
            "midfield_z": 0.0,
            "keeper_z": 0.0,
        }
        profile_b = {
            "match_coverage": 0.25,
            "attack_z": 0.0,
            "defense_z": -5.0,
            "midfield_z": 0.0,
            "keeper_z": 0.0,
        }

        log_a, log_b, details = model._log_adjustments(profile_a, profile_b)

        self.assertTrue(np.isfinite(log_a))
        self.assertEqual(log_a, 0.2)
        self.assertEqual(log_b, 0.0)
        self.assertAlmostEqual(details["pair_coverage_shrink"], 0.25)

    def test_default_prediction_keeps_base_wdl_probabilities(self):
        base_prediction = {
            "team_a": "Alpha",
            "team_b": "Beta",
            "lambda_a": 1.0,
            "lambda_b": 1.0,
            "result_probabilities": {
                "team_a_win": 0.40,
                "draw": 0.30,
                "team_b_win": 0.30,
            },
            "predicted_result": "team_a_win",
            "scoreline_probabilities": [],
            "top_scorelines": [],
            "calibration_notes": {"dixon_coles_rho": -0.08},
        }
        base_model = Mock()
        base_model.predict.return_value = dict(base_prediction)
        model = v18.V18HybridSquadModel(
            base_model=base_model,
            squad_profiles={
                "Alpha": {
                    "rating_confidence": 1.0,
                    "match_coverage": 1.0,
                    "attack_z": 2.0,
                    "defense_z": 0.0,
                    "midfield_z": 0.0,
                    "keeper_z": 0.0,
                },
                "Beta": {
                    "rating_confidence": 1.0,
                    "match_coverage": 1.0,
                    "attack_z": 0.0,
                    "defense_z": -2.0,
                    "midfield_z": 0.0,
                    "keeper_z": 0.0,
                },
            },
            beta_attack=0.1,
        )

        prediction = model.predict("Alpha", "Beta", max_goals=4)

        self.assertEqual(
            prediction["result_probabilities"],
            base_prediction["result_probabilities"],
        )
        self.assertEqual(prediction["predicted_result"], "team_a_win")
        self.assertFalse(prediction["v18_adjustments"]["player_ratings_affect_wdl"])
        self.assertTrue(
            prediction["v18_adjustments"]["player_ratings_affect_scorelines"]
        )


if __name__ == "__main__":
    unittest.main()

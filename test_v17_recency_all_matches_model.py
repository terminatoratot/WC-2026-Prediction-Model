from __future__ import annotations

import unittest

import pandas as pd

import v15_catboost_model as v15
from v17_recency_all_matches_model import (
    DEFAULT_RECENCY_HALF_LIFE_YEARS,
    DEFAULT_RECENCY_MIN_WEIGHT,
    V17CatBoostWorldCupModel,
    build_all_match_training_frame,
    scoreline_text,
)


class V17RecencyAllMatchesModelTest(unittest.TestCase):
    def test_all_non_world_cup_internationals_become_training_rows(self):
        world_cup_frame = pd.DataFrame(
            [
                {
                    "match_id": "wc_2002_alpha_beta",
                    "date": pd.Timestamp("2002-06-01"),
                    "team_a": "Alpha",
                    "team_b": "Beta",
                    "goals_a": 1,
                    "goals_b": 0,
                    "goal_diff": 1,
                    "is_group_stage": 1,
                    "is_knockout": 0,
                    "host_a": 0,
                    "host_b": 0,
                    "host_diff": 0,
                    "abs_host_diff": 0,
                    "same_confed": 0,
                }
            ]
        )
        rows = []
        for date, tournament, goals_a, goals_b in (
            ("2001-01-01", "Friendly", 0, 0),
            ("2001-06-01", "UEFA Euro", 2, 1),
            ("2002-06-01", "FIFA World Cup", 1, 0),
            ("2003-07-01", "Copa América", 1, 1),
        ):
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "team_a": "Alpha",
                    "team_b": "Beta",
                    "goals_a": goals_a,
                    "goals_b": goals_b,
                    "tournament": tournament,
                    "neutral": True,
                    "country": "",
                    **v15.tournament_metadata(tournament),
                }
            )
        timeline, _ = v15.build_international_timeline(pd.DataFrame(rows))

        frame, features, summary = build_all_match_training_frame(
            world_cup_frame,
            timeline,
        )

        self.assertEqual(summary["world_cup_rows"], 1)
        self.assertEqual(summary["international_training_rows"], 3)
        self.assertEqual(len(frame), 4)
        self.assertEqual(
            frame["training_source"].value_counts().to_dict(),
            {"international": 3, "world_cup": 1},
        )
        self.assertEqual(
            set(frame.loc[frame["training_source"] == "international", "tournament_type"]),
            {"FRIENDLY", "EURO", "COPA"},
        )
        self.assertNotIn("WC", set(frame.loc[frame["training_source"] == "international", "tournament_type"]))
        self.assertIn("elo_diff", features)

    def test_v17_defaults_use_stronger_recency_than_v15(self):
        model = V17CatBoostWorldCupModel()

        self.assertEqual(model.recency_half_life_years, DEFAULT_RECENCY_HALF_LIFE_YEARS)
        self.assertEqual(model.recency_min_weight, DEFAULT_RECENCY_MIN_WEIGHT)
        self.assertLess(model.recency_half_life_years, 16.0)

    def test_scoreline_text_formats_model_output(self):
        self.assertEqual(
            scoreline_text(
                {
                    "team_a_goals": 2.0,
                    "team_b_goals": 1.0,
                    "probability": 0.12,
                }
            ),
            "2-1",
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.pipeline import Pipeline

from v13_live_signal_model import V13SklearnWorldCupModel
from v15_catboost_model import (
    PLAYER_PROFILE_METRICS,
    V15CatBoostWorldCupModel,
    add_historical_player_features,
    aggregate_player_profile,
    build_international_timeline,
    tournament_metadata,
)


class V15CatBoostModelTest(unittest.TestCase):
    def test_v15_uses_catboost_for_each_learned_layer(self):
        model = V15CatBoostWorldCupModel()

        regressors = dict(
            (name, estimator)
            for name, estimator, _ in model._named_regressors()
        )
        diff_regressors = dict(
            (name, estimator)
            for name, estimator, _ in model._named_diff_regressors()
        )
        classifiers = dict(
            (name, estimator)
            for name, estimator, _ in model._named_classifiers()
        )

        self.assertIsInstance(regressors["catboost"], CatBoostRegressor)
        self.assertIsInstance(diff_regressors["catboost"], CatBoostRegressor)
        self.assertIsInstance(
            classifiers["catboost"].estimator,
            CatBoostClassifier,
        )
        self.assertIsInstance(model._new_draw_model(), Pipeline)

    def test_v13_default_ensemble_remains_sklearn_only(self):
        model = V13SklearnWorldCupModel(model_type="ensemble")

        self.assertEqual(
            [name for name, _, _ in model._named_regressors()],
            ["rf", "hgb", "poisson"],
        )
        self.assertEqual(
            [name for name, _, _ in model._named_diff_regressors()],
            ["ridge", "rf", "hgb"],
        )
        self.assertEqual(
            [name for name, _, _ in model._named_classifiers()],
            ["rf", "hgb", "logistic"],
        )

    def test_historical_player_features_do_not_use_future_ratings(self):
        frame = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2022-06-01"),
                    "team_a": "Alpha",
                    "team_b": "Beta",
                },
                {
                    "date": pd.Timestamp("2022-12-01"),
                    "team_a": "Alpha",
                    "team_b": "Beta",
                },
            ]
        )
        ratings = pd.DataFrame(
            [
                {
                    "team": team,
                    "rating_date": pd.Timestamp(date),
                    "overall": overall,
                    "potential": overall,
                    "value_eur": 1_000_000,
                    "age": 25,
                    "role": role,
                }
                for team in ("Alpha", "Beta")
                for date, overall in (
                    ("2022-09-01", 80),
                    ("2021-08-01", 70),
                )
                for role in ("gk", "defense", "midfield", "attack")
            ]
        )

        enriched, features = add_historical_player_features(
            frame,
            ratings,
            [],
        )

        self.assertIn("player_diff_overall_mean", features)
        self.assertEqual(enriched.loc[0, "player_a_overall_mean"], 70)
        self.assertEqual(enriched.loc[1, "player_a_overall_mean"], 80)

    def test_player_profile_contains_all_model_metrics(self):
        players = pd.DataFrame(
            [
                {
                    "overall": 80 + index,
                    "potential": 82 + index,
                    "value_eur": 1_000_000 * (index + 1),
                    "age": 24 + index,
                    "role": role,
                }
                for index, role in enumerate(
                    ["gk", "defense", "defense", "midfield", "attack"]
                )
            ]
        )

        profile = aggregate_player_profile(players)

        self.assertEqual(set(profile), set(PLAYER_PROFILE_METRICS))
        self.assertEqual(profile["rating_count"], 5)

    def test_tournament_prestige_ladder(self):
        self.assertEqual(tournament_metadata("FIFA World Cup")["prestige_weight"], 1.0)
        self.assertEqual(tournament_metadata("UEFA Euro")["prestige_weight"], 0.75)
        self.assertEqual(
            tournament_metadata("African Cup of Nations")["prestige_weight"],
            0.55,
        )
        self.assertEqual(tournament_metadata("Gold Cup")["prestige_weight"], 0.35)

    def test_international_features_are_strictly_pre_match(self):
        rows = []
        for date, tournament, goals_a, goals_b in (
            ("2024-01-01", "Friendly", 2, 0),
            ("2024-06-01", "UEFA Euro", 1, 1),
            ("2024-06-05", "UEFA Euro", 0, 1),
        ):
            metadata = tournament_metadata(tournament)
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
                    **metadata,
                }
            )

        timeline, state = build_international_timeline(pd.DataFrame(rows))

        self.assertEqual(timeline.loc[0, "a_matches_seen"], 0)
        self.assertGreater(timeline.loc[1, "a_matches_seen"], 0)
        self.assertEqual(timeline.loc[1, "continental_a_matches_seen"], 0)
        self.assertEqual(timeline.loc[2, "continental_a_matches_seen"], 1)
        self.assertNotEqual(state["elo"]["Alpha"], 1500.0)
        self.assertTrue(np.isfinite(timeline["elo_diff"]).all())


if __name__ == "__main__":
    unittest.main()

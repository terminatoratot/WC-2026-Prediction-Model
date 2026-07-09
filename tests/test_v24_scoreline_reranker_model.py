from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

import v11_wcq_results_model as v11
import v24_scoreline_reranker_model as v24


class FakeReranker:
    classes_ = np.array([0, 1])

    def predict_proba(self, features):
        scores = []
        for _, row in features.iterrows():
            if int(row["goals_a"]) == 2 and int(row["goals_b"]) == 0:
                scores.append(0.90)
            else:
                scores.append(0.10)
        return np.asarray([[1.0 - score, score] for score in scores], dtype=float)


class V24ScorelineRerankerModelTest(unittest.TestCase):
    def test_apply_reranker_preserves_wdl_marginals(self):
        matrix = {
            (1, 0): 0.30,
            (2, 0): 0.20,
            (1, 1): 0.25,
            (0, 1): 0.15,
            (0, 0): 0.10,
        }
        result_probabilities = v11.result_probs(matrix)
        prediction = {
            "lambda_a": 1.4,
            "lambda_b": 0.8,
            "result_probabilities": result_probabilities,
        }

        adjusted, diagnostics = v24.apply_reranker_to_matrix(
            matrix,
            prediction,
            FakeReranker(),
            blend=1.0,
            power=1.0,
        )

        self.assertTrue(diagnostics["reranker_enabled"])
        self.assertAlmostEqual(sum(adjusted.values()), 1.0)
        results = v11.result_probs(adjusted)
        for label, probability in result_probabilities.items():
            self.assertAlmostEqual(results[label], probability)
        self.assertGreater(adjusted[(2, 0)], matrix[(2, 0)])

    def test_candidate_features_capture_match_context(self):
        row = v24.candidate_feature_row(
            (2, 1),
            probability=0.12,
            rank=3,
            lambda_a=1.6,
            lambda_b=0.9,
            result_probabilities={
                "team_a_win": 0.58,
                "draw": 0.24,
                "team_b_win": 0.18,
            },
            context={"is_group_stage": 1, "host_a": 1},
        )

        self.assertEqual(row["total_goals"], 3.0)
        self.assertEqual(row["is_team_a_win_score"], 1.0)
        self.assertEqual(row["score_matches_predicted_result"], 1.0)
        self.assertEqual(row["is_group_stage"], 1.0)
        self.assertEqual(row["host_a"], 1.0)

    def test_wrapper_records_reranker_adjustments(self):
        base_prediction = {
            "team_a": "Alpha",
            "team_b": "Beta",
            "lambda_a": 1.4,
            "lambda_b": 0.8,
            "result_probabilities": {
                "team_a_win": 0.55,
                "draw": 0.25,
                "team_b_win": 0.20,
            },
            "predicted_result": "team_a_win",
            "scoreline_probabilities": [
                {"team_a_goals": 0, "team_b_goals": 0, "probability": 0.10},
                {"team_a_goals": 1, "team_b_goals": 0, "probability": 0.30},
                {"team_a_goals": 2, "team_b_goals": 0, "probability": 0.20},
                {"team_a_goals": 1, "team_b_goals": 1, "probability": 0.25},
                {"team_a_goals": 0, "team_b_goals": 1, "probability": 0.15},
            ],
            "top_scorelines": [],
            "calibration_notes": {},
        }

        class BaseV23:
            outcome_model = object()
            scoreline_layer_weight = 0.55
            favorite_tail_strength = 0.32
            favorite_tail_threshold = 0.60
            reranker_strength = 0.18
            diversity_relative_floor = 0.42
            training_data_summary = {}

            def predict(self, *args, **kwargs):
                return dict(base_prediction)

        model = v24.V24ScorelineRerankerModel(
            BaseV23(),
            FakeReranker(),
            reranker_diagnostics={"enabled": True},
            reranker_blend=1.0,
            reranker_power=1.0,
        )

        prediction = model.predict("Alpha", "Beta", max_goals=2)

        self.assertEqual(
            prediction["result_probabilities"],
            base_prediction["result_probabilities"],
        )
        self.assertTrue(prediction["v24_adjustments"]["reranker_enabled"])
        self.assertEqual(
            prediction["v24_adjustments"]["training_diagnostics"],
            {"enabled": True},
        )


if __name__ == "__main__":
    unittest.main()

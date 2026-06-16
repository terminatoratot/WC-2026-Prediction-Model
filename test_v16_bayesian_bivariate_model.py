from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import v11_wcq_results_model as v11
from v16_bayesian_bivariate_model import (
    DEFAULT_BAYES_GOAL_BLEND_GRID,
    DEFAULT_GROUP_DRAW_SCALE_GRID,
    apply_draw_score_scaling,
    apply_zero_inflation,
    apply_result_mass_correction,
    bivariate_poisson_score_matrix,
    hierarchy_pair_features,
    fit_temperature,
    normalize_matrix,
    score_parameter_metrics,
    temperature_scale,
)


class V16BayesianBivariateTest(unittest.TestCase):
    def test_default_blend_grid_includes_baseline_and_extends_boundary(self):
        self.assertEqual(DEFAULT_BAYES_GOAL_BLEND_GRID[0], 0.0)
        self.assertEqual(DEFAULT_BAYES_GOAL_BLEND_GRID[-1], 0.7)
        self.assertEqual(len(DEFAULT_BAYES_GOAL_BLEND_GRID), 15)
        np.testing.assert_allclose(
            np.diff(DEFAULT_BAYES_GOAL_BLEND_GRID),
            0.05,
        )

    def test_group_draw_scale_grid_matches_requested_values(self):
        self.assertEqual(
            DEFAULT_GROUP_DRAW_SCALE_GRID,
            (1.0, 1.05, 1.10, 1.15, 1.20, 1.25),
        )

    def test_bivariate_matrix_is_normalized_and_correlated(self):
        independent = bivariate_poisson_score_matrix(1.6, 0.9, 0.0)
        correlated = bivariate_poisson_score_matrix(1.6, 0.9, 0.15)

        self.assertAlmostEqual(sum(correlated.values()), 1.0, places=12)
        independent_product = sum(
            goals_a * goals_b * probability
            for (goals_a, goals_b), probability in independent.items()
        )
        correlated_product = sum(
            goals_a * goals_b * probability
            for (goals_a, goals_b), probability in correlated.items()
        )
        self.assertGreater(correlated_product, independent_product)

    def test_zero_inflation_only_adds_structural_zero_mass(self):
        matrix = bivariate_poisson_score_matrix(1.4, 1.1, 0.1)
        inflated = apply_zero_inflation(matrix, 0.04)

        self.assertAlmostEqual(sum(inflated.values()), 1.0, places=12)
        self.assertGreater(inflated[(0, 0)], matrix[(0, 0)])
        self.assertLess(inflated[(1, 0)], matrix[(1, 0)])

    def test_draw_score_scaling_only_boosts_draw_scorelines(self):
        matrix = bivariate_poisson_score_matrix(1.4, 1.1, 0.1)
        scaled = apply_draw_score_scaling(matrix, 1.20)

        self.assertAlmostEqual(sum(scaled.values()), 1.0, places=12)
        self.assertGreater(scaled[(1, 1)], matrix[(1, 1)])
        self.assertGreater(scaled[(0, 0)], matrix[(0, 0)])
        self.assertLess(scaled[(1, 0)], matrix[(1, 0)])

    def test_unit_draw_score_scale_preserves_matrix(self):
        matrix = bivariate_poisson_score_matrix(1.4, 1.1, 0.1)
        self.assertEqual(apply_draw_score_scaling(matrix, 1.0), matrix)

    def test_zero_reweight_strength_preserves_score_matrix(self):
        matrix = bivariate_poisson_score_matrix(1.5, 0.9, 0.1)
        corrected = apply_result_mass_correction(
            matrix,
            {
                "team_a_win": 0.2,
                "draw": 0.2,
                "team_b_win": 0.6,
            },
            strength=0.0,
        )

        self.assertEqual(matrix, corrected)

    def test_full_reweight_strength_matches_target_outcomes(self):
        matrix = bivariate_poisson_score_matrix(1.5, 0.9, 0.1)
        target = {
            "team_a_win": 0.2,
            "draw": 0.2,
            "team_b_win": 0.6,
        }
        corrected = apply_result_mass_correction(
            matrix,
            target,
            strength=1.0,
        )
        outcomes = v11.result_probs(corrected)

        for label, probability in target.items():
            self.assertAlmostEqual(outcomes[label], probability, places=10)

    def test_temperature_scaling_is_normalized(self):
        scaled = temperature_scale(
            {
                "team_a_win": 0.70,
                "draw": 0.20,
                "team_b_win": 0.10,
            },
            1.4,
        )

        self.assertAlmostEqual(sum(scaled.values()), 1.0, places=12)
        self.assertLess(scaled["team_a_win"], 0.70)

    def test_hierarchy_features_use_attack_defense_and_host(self):
        profile = {
            "mu": 0.2,
            "home_advantage": 0.1,
            "team_profiles": {
                "Alpha": {"attack": 0.3, "defense": 0.2},
                "Beta": {"attack": -0.1, "defense": -0.2},
            },
        }
        features = hierarchy_pair_features(
            profile,
            "Alpha",
            "Beta",
            host_a=True,
        )

        self.assertAlmostEqual(features["bayes_log_goal_a"], 0.8)
        self.assertAlmostEqual(features["bayes_log_goal_b"], -0.1)
        self.assertGreater(
            features["bayes_log_goal_diff"],
            0.0,
        )

    def test_normalize_matrix_rejects_empty_mass(self):
        with self.assertRaises(ValueError):
            normalize_matrix({(0, 0): np.nan})

    def test_temperature_fit_returns_finite_positive_value(self):
        frame = pd.DataFrame(
            {
                "goals_a": [2, 0, 1, 1],
                "goals_b": [0, 1, 1, 0],
                "team_a_win": [0.8, 0.5, 0.4, 0.7],
                "draw": [0.1, 0.2, 0.4, 0.2],
                "team_b_win": [0.1, 0.3, 0.2, 0.1],
            }
        )

        temperature = fit_temperature(frame)

        self.assertTrue(np.isfinite(temperature))
        self.assertGreater(temperature, 0.0)

    def test_score_parameter_metrics_count_top_two_hits(self):
        frame = pd.DataFrame(
            {
                "goals_a": [1, 4],
                "goals_b": [0, 3],
                "v15_lambda_a": [1.2, 1.2],
                "v15_lambda_b": [0.7, 0.7],
                "bayes_lambda_a": [1.2, 1.2],
                "bayes_lambda_b": [0.7, 0.7],
                "team_a_win": [0.55, 0.55],
                "draw": [0.25, 0.25],
                "team_b_win": [0.20, 0.20],
            }
        )

        metrics = score_parameter_metrics(
            frame,
            temperature=1.0,
            bayes_goal_weight=0.0,
            covariance=0.0,
            zero_inflation=0.0,
            draw_score_scale=1.0,
            result_reweight_strength=0.0,
        )

        self.assertEqual(metrics["exact_score_top2_hits"], 1)
        self.assertEqual(metrics["exact_score_top2_rate"], 0.5)
        self.assertTrue(np.isfinite(metrics["exact_score_log_loss"]))


if __name__ == "__main__":
    unittest.main()

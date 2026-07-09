#!/usr/bin/env python3
"""V38: V35 plus one shrunk global total-lambda correction.

This is intentionally small: one estimated multiplier, no hand-built extra
thresholds. It targets the under-bracketing diagnosis by lifting or lowering
the total-goals scale while preserving the base model's W/D/L probabilities.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

import v11_wcq_results_model as v11
import v15_catboost_model as v15
import v20_scoreline_ensemble_model as v20
import v28_current_worldcup_form_model as v28
import v29_tail_risk_scoreline_model as v29
import v35_game_state_late_mutation_model as v35
import v36_fotmob_current_form_model as v36


ScoreMatrix = Dict[Tuple[int, int], float]

DEFAULT_BOOTSTRAP_SAMPLES = 2500
DEFAULT_SHRINKAGE_PRIOR_SD = 0.08
DEFAULT_RANDOM_SEED = 20260623


@dataclass
class TotalLambdaCalibration:
    raw_multiplier: float = 1.0
    shrunk_multiplier: float = 1.0
    shrinkage_weight: float = 0.0
    bootstrap_mean: float = 1.0
    bootstrap_std: float = 0.0
    bootstrap_ci_low: float = 1.0
    bootstrap_ci_high: float = 1.0
    mean_actual_total: float = 0.0
    mean_predicted_lambda_sum: float = 0.0
    n_matches: int = 0
    prior_sd: float = DEFAULT_SHRINKAGE_PRIOR_SD

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "raw_multiplier": float(self.raw_multiplier),
            "shrunk_multiplier": float(self.shrunk_multiplier),
            "shrinkage_weight": float(self.shrinkage_weight),
            "bootstrap_mean": float(self.bootstrap_mean),
            "bootstrap_std": float(self.bootstrap_std),
            "bootstrap_ci_low": float(self.bootstrap_ci_low),
            "bootstrap_ci_high": float(self.bootstrap_ci_high),
            "mean_actual_total": float(self.mean_actual_total),
            "mean_predicted_lambda_sum": float(self.mean_predicted_lambda_sum),
            "n_matches": int(self.n_matches),
            "prior_sd": float(self.prior_sd),
        }


def result_label(goals_a: int, goals_b: int) -> str:
    if goals_a > goals_b:
        return "team_a_win"
    if goals_b > goals_a:
        return "team_b_win"
    return "draw"


def estimate_total_lambda_calibration(
    base_model: Any,
    observed_matches_csv: str | Path | None,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    shrinkage_prior_sd: float = DEFAULT_SHRINKAGE_PRIOR_SD,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> TotalLambdaCalibration:
    if not observed_matches_csv or not Path(observed_matches_csv).exists():
        return TotalLambdaCalibration(prior_sd=shrinkage_prior_sd)
    observed = pd.read_csv(observed_matches_csv)
    required = {"team_a", "team_b", "goals_a", "goals_b"}
    if observed.empty or not required.issubset(observed.columns):
        return TotalLambdaCalibration(prior_sd=shrinkage_prior_sd)

    rows = []
    for row in observed.to_dict(orient="records"):
        team_a = str(row["team_a"])
        team_b = str(row["team_b"])
        prediction = base_model.predict(
            team_a,
            team_b,
            host_a=team_a in {"Canada", "Mexico", "USA", "United States"},
            host_b=team_b in {"Canada", "Mexico", "USA", "United States"},
            knockout=str(row.get("stage", "")).strip().lower() != "group stage",
        )
        lambda_sum = float(prediction.get("lambda_a", 0.0)) + float(
            prediction.get("lambda_b", 0.0)
        )
        actual_total = int(row["goals_a"]) + int(row["goals_b"])
        if lambda_sum > 1e-9:
            rows.append({"actual_total": actual_total, "lambda_sum": lambda_sum})
    if not rows:
        return TotalLambdaCalibration(prior_sd=shrinkage_prior_sd)

    frame = pd.DataFrame(rows)
    actual = frame["actual_total"].to_numpy(dtype=float)
    predicted = frame["lambda_sum"].to_numpy(dtype=float)
    raw_multiplier = float(actual.sum() / max(predicted.sum(), 1e-9))

    rng = np.random.default_rng(int(random_seed))
    ratios = []
    n = len(frame)
    for _ in range(max(int(bootstrap_samples), 1)):
        idx = rng.integers(0, n, size=n)
        denom = float(predicted[idx].sum())
        ratios.append(float(actual[idx].sum() / max(denom, 1e-9)))
    boot = np.asarray(ratios, dtype=float)
    boot_std = float(np.std(boot, ddof=1)) if len(boot) > 1 else 0.0
    ci_low, ci_high = np.quantile(boot, [0.025, 0.975])

    prior_sd = max(float(shrinkage_prior_sd), 1e-9)
    shrinkage_weight = float((prior_sd * prior_sd) / (prior_sd * prior_sd + boot_std * boot_std))
    shrunk_multiplier = float(1.0 + shrinkage_weight * (raw_multiplier - 1.0))
    return TotalLambdaCalibration(
        raw_multiplier=raw_multiplier,
        shrunk_multiplier=shrunk_multiplier,
        shrinkage_weight=shrinkage_weight,
        bootstrap_mean=float(np.mean(boot)),
        bootstrap_std=boot_std,
        bootstrap_ci_low=float(ci_low),
        bootstrap_ci_high=float(ci_high),
        mean_actual_total=float(np.mean(actual)),
        mean_predicted_lambda_sum=float(np.mean(predicted)),
        n_matches=int(n),
        prior_sd=prior_sd,
    )


class V38TotalLambdaCalibratedModel:
    """Wrap V35 and apply a single shrunk total-goals multiplier."""

    def __init__(
        self,
        base_model: v35.V35GameStateLateMutationModel,
        calibration: TotalLambdaCalibration,
    ):
        self.base_model = base_model
        self.calibration = calibration
        self.training_data_summary = getattr(base_model, "training_data_summary", {})

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_model, name)

    @property
    def total_lambda_multiplier(self) -> float:
        return float(self.calibration.shrunk_multiplier)

    def _adjusted_score_matrix(
        self,
        prediction: Dict[str, Any],
        max_goals: int,
    ) -> tuple[ScoreMatrix, Dict[str, Any]]:
        base_lambda_a = float(prediction["lambda_a"])
        base_lambda_b = float(prediction["lambda_b"])
        multiplier = self.total_lambda_multiplier
        lambda_a = float(np.clip(base_lambda_a * multiplier, 0.05, 7.5))
        lambda_b = float(np.clip(base_lambda_b * multiplier, 0.05, 7.5))
        matrix = v11.poisson_score_matrix(lambda_a, lambda_b, max_goals)
        rho = prediction.get("calibration_notes", {}).get("dixon_coles_rho", -0.08)
        matrix = v11.apply_dixon_coles_adjustment(matrix, lambda_a, lambda_b, rho=rho)
        result_probabilities = dict(prediction["result_probabilities"])
        matrix = v11.reweight_score_matrix_to_results(matrix, result_probabilities)
        adjusted_lambda_a, adjusted_lambda_b = v28.expected_goals(matrix)
        return matrix, {
            "base_lambda_a": base_lambda_a,
            "base_lambda_b": base_lambda_b,
            "raw_scaled_lambda_a": lambda_a,
            "raw_scaled_lambda_b": lambda_b,
            "adjusted_lambda_a": adjusted_lambda_a,
            "adjusted_lambda_b": adjusted_lambda_b,
            "base_lambda_sum": base_lambda_a + base_lambda_b,
            "adjusted_lambda_sum": adjusted_lambda_a + adjusted_lambda_b,
            "total_lambda_multiplier": multiplier,
            "wdl_preserved": True,
        }

    def predict(self, *args, **kwargs) -> Dict[str, Any]:
        team_a = kwargs.get("team_a", args[0] if args else "")
        team_b = kwargs.get("team_b", args[1] if len(args) > 1 else "")
        max_goals = int(kwargs.get("max_goals", args[5] if len(args) > 5 else 10))
        prediction = self.base_model.predict(*args, **kwargs)
        result_probabilities = dict(prediction["result_probabilities"])
        matrix, diagnostics = self._adjusted_score_matrix(prediction, max_goals)
        lambda_a, lambda_b = v28.expected_goals(matrix)

        prediction["lambda_a"] = lambda_a
        prediction["lambda_b"] = lambda_b
        prediction["result_probabilities"] = result_probabilities
        prediction["predicted_result"] = max(result_probabilities, key=result_probabilities.get)
        prediction.update(v15.score_outputs(matrix, max_goals))
        top_scorelines, tail_diagnostics = v29.select_top_scorelines_with_tail_risk(
            matrix,
            result_probabilities,
            lambda_a,
            lambda_b,
            current_top_scorelines=prediction.get("top_scorelines", []),
            top_n=15,
        )
        prediction["top_scorelines"] = top_scorelines

        outlier, outlier_diagnostics = v35.select_game_state_late_outlier(
            matrix,
            result_probabilities,
            lambda_a,
            lambda_b,
            prediction.get("top_scorelines", []),
            self.base_model.tournament_late_profile,
            self.base_model.late_profile_for_team(team_a),
            self.base_model.late_profile_for_team(team_b),
            self.base_model.sub_profile_for_team(team_a),
            self.base_model.sub_profile_for_team(team_b),
            self.base_model.mutation_table,
            relative_floor=self.base_model.relative_floor,
            absolute_floor=self.base_model.absolute_floor,
            source_limit=self.base_model.source_limit,
            max_goals=self.base_model.max_goals,
        )
        prediction["game_state_late_outlier"] = outlier
        prediction["late_instability_outlier"] = outlier
        prediction["outlier_scoreline"] = outlier
        prediction["top_scorelines_plus_outlier"] = [
            *prediction.get("top_scorelines", [])[:3],
            *([outlier] if outlier else []),
        ]
        prediction["v38_adjustments"] = {
            "base_model": "v35_game_state_late_mutation",
            "scoreline_policy": "single_shrunk_total_lambda_multiplier",
            "calibration": self.calibration.diagnostics(),
            "score_matrix_changed": True,
            "scoreline_layer_affects_wdl": False,
            "tail_risk_selector": tail_diagnostics,
            "game_state_outlier": outlier_diagnostics,
            **diagnostics,
        }
        prediction["calibration_notes"] = {
            **prediction.get("calibration_notes", {}),
            "v38": prediction["v38_adjustments"],
        }
        return prediction


def build_from_zip(
    zip_path,
    train_csv=None,
    test_csv=None,
    model_type="catboost",
    box_csv=None,
    results_csv=None,
    former_names_csv=None,
    prediction_year=2026,
    qualifier_blend_start_year=2014,
    qualifier_full_weight_year=2022,
    qualifier_minimum_influence=0.0,
    recency_half_life_years=16.0,
    recency_min_weight=0.10,
    player_ratings_csv=None,
    declared_squads_csv=None,
    fcratings_csv=None,
    results_as_of=v15.DEFAULT_RESULTS_AS_OF,
    observed_matches_csv=None,
    fotmob_leaders_csv=None,
    fotmob_player_stats_csv=None,
    fotmob_lineups_csv=None,
    fotmob_substitutions_csv=None,
    fotmob_keeper_stats_csv=None,
    fotmob_match_facts_csv=None,
    fotmob_goal_events_csv=None,
    bootstrap_samples=DEFAULT_BOOTSTRAP_SAMPLES,
    shrinkage_prior_sd=DEFAULT_SHRINKAGE_PRIOR_SD,
    random_seed=DEFAULT_RANDOM_SEED,
    **kwargs,
):
    data_dir = Path(__file__).resolve().parent / "data"
    fotmob_match_facts_csv = fotmob_match_facts_csv or (
        data_dir / "fotmob_match_facts_clean.csv"
    )
    if observed_matches_csv is None:
        generated_observed = v36.completed_fotmob_facts_to_observed(
            fotmob_match_facts_csv,
            data_dir / "fotmob_completed_matches_observed_schema.csv",
        )
        observed_matches_csv = str(generated_observed) if generated_observed else None

    base_model, data = v35.build_from_zip(
        zip_path,
        train_csv=train_csv,
        test_csv=test_csv,
        model_type=model_type,
        box_csv=box_csv,
        results_csv=results_csv,
        former_names_csv=former_names_csv,
        prediction_year=prediction_year,
        qualifier_blend_start_year=qualifier_blend_start_year,
        qualifier_full_weight_year=qualifier_full_weight_year,
        qualifier_minimum_influence=qualifier_minimum_influence,
        recency_half_life_years=recency_half_life_years,
        recency_min_weight=recency_min_weight,
        player_ratings_csv=player_ratings_csv,
        declared_squads_csv=declared_squads_csv,
        fcratings_csv=fcratings_csv,
        results_as_of=results_as_of,
        observed_matches_csv=observed_matches_csv,
        fotmob_leaders_csv=fotmob_leaders_csv,
        fotmob_player_stats_csv=fotmob_player_stats_csv,
        fotmob_lineups_csv=fotmob_lineups_csv,
        fotmob_substitutions_csv=fotmob_substitutions_csv,
        fotmob_keeper_stats_csv=fotmob_keeper_stats_csv,
        fotmob_match_facts_csv=fotmob_match_facts_csv,
        fotmob_goal_events_csv=fotmob_goal_events_csv,
        **kwargs,
    )
    calibration = estimate_total_lambda_calibration(
        base_model,
        observed_matches_csv,
        bootstrap_samples=bootstrap_samples,
        shrinkage_prior_sd=shrinkage_prior_sd,
        random_seed=random_seed,
    )
    model = V38TotalLambdaCalibratedModel(base_model, calibration)
    model.training_data_summary = {
        **getattr(base_model, "training_data_summary", {}),
        "v38_scoreline_policy": "single_shrunk_total_lambda_multiplier",
        "v38_observed_matches_csv": str(observed_matches_csv),
        "v38_total_lambda_calibration": calibration.diagnostics(),
    }
    return model, data


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    parser = argparse.ArgumentParser(description="Run V38: total-lambda calibrated V35.")
    parser.add_argument("--team-a", required=True)
    parser.add_argument("--team-b", required=True)
    parser.add_argument("--host-a", action="store_true")
    parser.add_argument("--host-b", action="store_true")
    parser.add_argument("--knockout", action="store_true")
    parser.add_argument("--outdir", default="outputs/outputs_v38_total_lambda_calibrated")
    parser.add_argument("--worldcupsai-zip", default=str(data_dir / "worldcupsai.zip"))
    parser.add_argument("--team-train", default=str(data_dir / "current_team_features_2026.csv"))
    parser.add_argument("--team-test")
    parser.add_argument("--box-data", default=str(data_dir / "FIFAallMatchBoxData.csv"))
    parser.add_argument("--results-data", default=str(data_dir / "results.csv"))
    parser.add_argument("--results-as-of", default=v15.DEFAULT_RESULTS_AS_OF)
    parser.add_argument("--former-names", default=str(data_dir / "former_names.csv"))
    parser.add_argument("--prediction-year", type=int, default=2026)
    parser.add_argument("--player-ratings", default=str(data_dir / "player_ratings_international.csv"))
    parser.add_argument("--declared-squads", default=str(data_dir / "world_cup_2026_declared_squads.csv"))
    parser.add_argument("--fcratings", default=str(data_dir / "fcratings_top50_worldcup2026.csv"))
    parser.add_argument("--observed-matches")
    parser.add_argument("--fotmob-leaders", default=str(data_dir / "fotmob_full_stat_tables_clean.csv"))
    parser.add_argument("--fotmob-player-stats", default=str(data_dir / "fotmob_match_player_stats_clean.csv"))
    parser.add_argument("--fotmob-lineups", default=str(data_dir / "fotmob_match_lineups_clean.csv"))
    parser.add_argument("--fotmob-substitutions", default=str(data_dir / "fotmob_match_substitutions_clean.csv"))
    parser.add_argument("--fotmob-keeper-stats", default=str(data_dir / "fotmob_match_keeper_stats_clean.csv"))
    parser.add_argument("--fotmob-match-facts", default=str(data_dir / "fotmob_match_facts_clean.csv"))
    parser.add_argument("--fotmob-goal-events", default=str(data_dir / "fotmob_match_goal_events_clean.csv"))
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--shrinkage-prior-sd", type=float, default=DEFAULT_SHRINKAGE_PRIOR_SD)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    model, _ = build_from_zip(
        args.worldcupsai_zip,
        train_csv=args.team_train,
        test_csv=args.team_test,
        box_csv=args.box_data,
        results_csv=args.results_data,
        former_names_csv=args.former_names,
        prediction_year=args.prediction_year,
        player_ratings_csv=args.player_ratings,
        declared_squads_csv=args.declared_squads,
        fcratings_csv=args.fcratings,
        results_as_of=args.results_as_of,
        observed_matches_csv=args.observed_matches,
        fotmob_leaders_csv=args.fotmob_leaders,
        fotmob_player_stats_csv=args.fotmob_player_stats,
        fotmob_lineups_csv=args.fotmob_lineups,
        fotmob_substitutions_csv=args.fotmob_substitutions,
        fotmob_keeper_stats_csv=args.fotmob_keeper_stats,
        fotmob_match_facts_csv=args.fotmob_match_facts,
        fotmob_goal_events_csv=args.fotmob_goal_events,
        bootstrap_samples=args.bootstrap_samples,
        shrinkage_prior_sd=args.shrinkage_prior_sd,
    )
    output_dir = v11.unique_output_dir(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction = model.predict(
        args.team_a,
        args.team_b,
        host_a=args.host_a,
        host_b=args.host_b,
        knockout=args.knockout,
    )
    (output_dir / "single_match_prediction.json").write_text(
        json.dumps(prediction, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(prediction["top_scorelines"]).to_csv(output_dir / "scoreline_probabilities_top.csv", index=False)
    pd.DataFrame(prediction["scoreline_probabilities"]).to_csv(output_dir / "scoreline_probabilities.csv", index=False)
    pd.DataFrame(prediction["top_scorelines_plus_outlier"]).to_csv(
        output_dir / "scoreline_probabilities_top_plus_game_state_outlier.csv",
        index=False,
    )
    (output_dir / "model_summary.json").write_text(
        json.dumps(
            {
                "version": "v38-total-lambda-calibrated",
                "base_model": "v35-game-state-late-mutation",
                "team_a": prediction["team_a"],
                "team_b": prediction["team_b"],
                "result_probabilities": prediction["result_probabilities"],
                "top_3": prediction["top_scorelines"][:3],
                "game_state_late_outlier": prediction["game_state_late_outlier"],
                "v38_adjustments": prediction["v38_adjustments"],
                "expanded_training_data": model.training_data_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not args.no_plots:
        v11.plot_prediction_outputs(prediction, output_dir)
        v28.plot_top3_scorelines(prediction, output_dir / "plots")
        v35.plot_top3_plus_game_state_outlier(prediction, output_dir)
    print(
        json.dumps(
            {
                "result_probabilities": prediction["result_probabilities"],
                "predicted_result": prediction["predicted_result"],
                "lambda_a": prediction["lambda_a"],
                "lambda_b": prediction["lambda_b"],
                "top_3": prediction["top_scorelines"][:3],
                "game_state_late_outlier": prediction["game_state_late_outlier"],
                "v38_calibration": prediction["v38_adjustments"]["calibration"],
                "v38_multiplier": prediction["v38_adjustments"]["total_lambda_multiplier"],
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

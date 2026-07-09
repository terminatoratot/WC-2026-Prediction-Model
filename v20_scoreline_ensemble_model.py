#!/usr/bin/env python3
"""V20: V15 W/D/L with blended V15 and V18 scoreline-only exact scores."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

import v11_wcq_results_model as v11
import v15_catboost_model as v15
import v18_hybrid_elo_form_player_model as v18


canon_team = v11.canon_team
ScoreMatrix = Dict[Tuple[int, int], float]

DEFAULT_V18_SCORELINE_WEIGHT = 0.35


def score_matrix_from_prediction(prediction: Dict[str, Any]) -> ScoreMatrix:
    return {
        (int(item["team_a_goals"]), int(item["team_b_goals"])): float(
            item["probability"]
        )
        for item in prediction["scoreline_probabilities"]
    }


def normalize_matrix(score_matrix: ScoreMatrix) -> ScoreMatrix:
    total = float(sum(score_matrix.values()))
    if total <= 0:
        raise ValueError("score matrix has no probability mass")
    return {key: float(value) / total for key, value in score_matrix.items()}


def blend_score_matrices(
    base_matrix: ScoreMatrix,
    adjusted_matrix: ScoreMatrix,
    adjusted_weight: float,
) -> ScoreMatrix:
    weight = float(np.clip(adjusted_weight, 0.0, 1.0))
    keys = set(base_matrix) | set(adjusted_matrix)
    blended = {
        key: (1.0 - weight) * base_matrix.get(key, 0.0)
        + weight * adjusted_matrix.get(key, 0.0)
        for key in keys
    }
    return normalize_matrix(blended)


class V20ScorelineEnsembleModel:
    """Preserve V15 W/D/L while blending V15 and V18 exact-score matrices."""

    def __init__(
        self,
        base_model: v15.V15CatBoostModel,
        squad_profiles: Dict[str, Dict[str, Any]],
        v18_scoreline_weight: float = DEFAULT_V18_SCORELINE_WEIGHT,
        beta_attack: float = v18.DEFAULT_BETA_ATTACK,
        beta_midfield: float = v18.DEFAULT_BETA_MIDFIELD,
        beta_keeper: float = v18.DEFAULT_BETA_KEEPER,
        max_log_adjustment: float = v18.DEFAULT_MAX_LOG_ADJUSTMENT,
    ):
        self.base_model = base_model
        self.squad_profiles = squad_profiles
        self.v18_scoreline_weight = float(np.clip(v18_scoreline_weight, 0.0, 1.0))
        self.v18_scoreline_model = v18.V18HybridSquadModel(
            base_model,
            squad_profiles,
            beta_attack=beta_attack,
            beta_midfield=beta_midfield,
            beta_keeper=beta_keeper,
            max_log_adjustment=max_log_adjustment,
            player_ratings_affect_wdl=False,
        )
        self.training_data_summary = getattr(
            base_model,
            "training_data_summary",
            {},
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_model, name)

    def _v18_scoreline_prediction(
        self,
        base_prediction: Dict[str, Any],
        team_a: str,
        team_b: str,
        max_goals: int,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        prediction = copy.deepcopy(base_prediction)
        base_lambda_a = float(prediction["lambda_a"])
        base_lambda_b = float(prediction["lambda_b"])
        base_results = dict(prediction["result_probabilities"])

        profile_a = self.v18_scoreline_model.profile_for_team(str(team_a))
        profile_b = self.v18_scoreline_model.profile_for_team(str(team_b))
        log_a, log_b, adjustment_details = self.v18_scoreline_model._log_adjustments(
            profile_a,
            profile_b,
        )
        lambda_a = float(np.clip(base_lambda_a * math.exp(log_a), 0.15, 4.5))
        lambda_b = float(np.clip(base_lambda_b * math.exp(log_b), 0.15, 4.5))
        score_matrix = v11.poisson_score_matrix(lambda_a, lambda_b, max_goals)
        rho = prediction.get("calibration_notes", {}).get(
            "dixon_coles_rho",
            -0.08,
        )
        score_matrix = v11.apply_dixon_coles_adjustment(
            score_matrix,
            lambda_a,
            lambda_b,
            rho=rho,
        )
        adjusted_poisson_results = v11.result_probs(score_matrix)
        score_matrix = v11.reweight_score_matrix_to_results(
            score_matrix,
            base_results,
        )

        prediction["lambda_a"] = lambda_a
        prediction["lambda_b"] = lambda_b
        prediction.update(v15.score_outputs(score_matrix, max_goals))
        prediction["result_probabilities"] = base_results
        diagnostics = {
            "v18_base_lambda_a": base_lambda_a,
            "v18_base_lambda_b": base_lambda_b,
            "v18_lambda_a": lambda_a,
            "v18_lambda_b": lambda_b,
            "v18_adjusted_poisson_result_probabilities": adjusted_poisson_results,
            "v18_player_ratings_affect_wdl": False,
            **adjustment_details,
        }
        return prediction, diagnostics

    def predict(self, *args, **kwargs) -> Dict[str, Any]:
        team_a = kwargs.get("team_a", args[0] if args else "")
        team_b = kwargs.get("team_b", args[1] if len(args) > 1 else "")
        max_goals = int(kwargs.get("max_goals", args[5] if len(args) > 5 else 10))

        v15_prediction = self.base_model.predict(*args, **kwargs)
        v18_prediction, v18_diagnostics = self._v18_scoreline_prediction(
            v15_prediction,
            str(team_a),
            str(team_b),
            max_goals,
        )
        v15_matrix = score_matrix_from_prediction(v15_prediction)
        v18_matrix = score_matrix_from_prediction(v18_prediction)
        blended_matrix = blend_score_matrices(
            v15_matrix,
            v18_matrix,
            self.v18_scoreline_weight,
        )

        result_probabilities = dict(v15_prediction["result_probabilities"])
        prediction = v15_prediction
        blended_lambda_a = (
            (1.0 - self.v18_scoreline_weight) * float(v15_prediction["lambda_a"])
            + self.v18_scoreline_weight * float(v18_prediction["lambda_a"])
        )
        blended_lambda_b = (
            (1.0 - self.v18_scoreline_weight) * float(v15_prediction["lambda_b"])
            + self.v18_scoreline_weight * float(v18_prediction["lambda_b"])
        )
        prediction["lambda_a"] = float(blended_lambda_a)
        prediction["lambda_b"] = float(blended_lambda_b)
        prediction.update(v15.score_outputs(blended_matrix, max_goals))
        prediction["result_probabilities"] = result_probabilities
        prediction["predicted_result"] = max(
            result_probabilities,
            key=result_probabilities.get,
        )
        prediction["v20_adjustments"] = {
            "base_model": "v15_catboost",
            "scoreline_policy": "linear_blend_v15_v18_scoreline_only",
            "v18_scoreline_weight": self.v18_scoreline_weight,
            "v15_scoreline_weight": 1.0 - self.v18_scoreline_weight,
            "scoreline_blend_affects_wdl": False,
            "rank_stabilizer": False,
            "v15_result_probabilities": result_probabilities,
            "v15_lambda_a": float(v15_prediction["lambda_a"]),
            "v15_lambda_b": float(v15_prediction["lambda_b"]),
            **v18_diagnostics,
        }
        prediction["calibration_notes"] = {
            **prediction.get("calibration_notes", {}),
            "v20": prediction["v20_adjustments"],
            "exact_score_policy": (
                "V20 preserves V15 W/D/L probabilities and blends the V15 "
                "exact-score matrix with the V18 scoreline-only matrix. No "
                "rank stabilizer or scoreline ordering override is applied."
            ),
        }
        return prediction

    def update_after_match(
        self,
        team_a: str,
        team_b: str,
        goals_a: int,
        goals_b: int,
    ) -> Dict[str, float]:
        return self.base_model.update_after_match(
            team_a,
            team_b,
            goals_a,
            goals_b,
        )


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
    beta_attack=v18.DEFAULT_BETA_ATTACK,
    beta_midfield=v18.DEFAULT_BETA_MIDFIELD,
    beta_keeper=v18.DEFAULT_BETA_KEEPER,
    max_log_adjustment=v18.DEFAULT_MAX_LOG_ADJUSTMENT,
    match_threshold=0.84,
    v18_scoreline_weight=DEFAULT_V18_SCORELINE_WEIGHT,
):
    data_dir = Path(__file__).resolve().parent / "data"
    player_ratings_csv = player_ratings_csv or (
        data_dir / "player_ratings_international.csv"
    )
    fcratings_csv = fcratings_csv or (data_dir / "fcratings_top50_worldcup2026.csv")
    base_model, data = v15.build_from_zip(
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
        results_as_of=results_as_of,
    )
    squad_names = v18.load_current_squad_names(player_ratings_csv)
    fcratings = v18.load_fcratings_players(fcratings_csv)
    squad_profiles = v18.build_current_fcratings_squad_profiles(
        squad_names,
        fcratings,
        match_threshold=match_threshold,
    )
    model = V20ScorelineEnsembleModel(
        base_model,
        squad_profiles,
        v18_scoreline_weight=v18_scoreline_weight,
        beta_attack=beta_attack,
        beta_midfield=beta_midfield,
        beta_keeper=beta_keeper,
        max_log_adjustment=max_log_adjustment,
    )
    model.training_data_summary = {
        **getattr(base_model, "training_data_summary", {}),
        "v20_scoreline_policy": "linear_blend_v15_v18_scoreline_only",
        "v20_v18_scoreline_weight": model.v18_scoreline_weight,
        "v20_rank_stabilizer": False,
        "v20_squad_profile_teams": len(squad_profiles),
        "v20_fcratings_rows": int(len(fcratings)),
        "v20_squad_name_rows": int(len(squad_names)),
    }
    return model, data


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    parser = argparse.ArgumentParser(
        description="Run V20: blended V15/V18 scoreline ensemble."
    )
    parser.add_argument("--team-a", required=True)
    parser.add_argument("--team-b", required=True)
    parser.add_argument("--host-a", action="store_true")
    parser.add_argument("--host-b", action="store_true")
    parser.add_argument("--knockout", action="store_true")
    parser.add_argument(
        "--outdir",
        default="outputs/outputs_v20_scoreline_ensemble_prediction",
    )
    parser.add_argument(
        "--worldcupsai-zip",
        default=str(data_dir / "worldcupsai.zip"),
    )
    parser.add_argument(
        "--team-train",
        default=str(data_dir / "current_team_features_2026.csv"),
    )
    parser.add_argument("--team-test")
    parser.add_argument(
        "--box-data",
        default=str(data_dir / "FIFAallMatchBoxData.csv"),
    )
    parser.add_argument(
        "--results-data",
        default=str(data_dir / "results.csv"),
    )
    parser.add_argument("--results-as-of", default=v15.DEFAULT_RESULTS_AS_OF)
    parser.add_argument(
        "--former-names",
        default=str(data_dir / "former_names.csv"),
    )
    parser.add_argument("--prediction-year", type=int, default=2026)
    parser.add_argument(
        "--player-ratings",
        default=str(data_dir / "player_ratings_international.csv"),
    )
    parser.add_argument(
        "--declared-squads",
        default=str(data_dir / "world_cup_2026_declared_squads.csv"),
    )
    parser.add_argument(
        "--fcratings",
        default=str(data_dir / "fcratings_top50_worldcup2026.csv"),
    )
    parser.add_argument(
        "--v18-scoreline-weight",
        type=float,
        default=DEFAULT_V18_SCORELINE_WEIGHT,
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    output_dir = v11.unique_output_dir(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
        v18_scoreline_weight=args.v18_scoreline_weight,
    )
    prediction = model.predict(
        args.team_a,
        args.team_b,
        host_a=args.host_a,
        host_b=args.host_b,
        knockout=args.knockout,
    )
    (output_dir / "single_match_prediction.json").write_text(
        json.dumps(prediction, indent=2)
    )
    pd.DataFrame(prediction["top_scorelines"]).to_csv(
        output_dir / "scoreline_probabilities_top.csv",
        index=False,
    )
    pd.DataFrame(prediction["scoreline_probabilities"]).to_csv(
        output_dir / "scoreline_probabilities.csv",
        index=False,
    )
    (output_dir / "model_summary.json").write_text(
        json.dumps(
            {
                "version": "v20-scoreline-ensemble",
                "base_model": "v15-catboost",
                "wdl_model": "v15_catboost_preserved",
                "exact_score_model": "linear_blend_v15_v18_scoreline_only",
                "team_a": prediction["team_a"],
                "team_b": prediction["team_b"],
                "v20_adjustments": prediction["v20_adjustments"],
                "expanded_training_data": model.training_data_summary,
            },
            indent=2,
        )
    )
    if not args.no_plots:
        v11.plot_prediction_outputs(prediction, output_dir)
    print(
        json.dumps(
            {
                "result_probabilities": prediction["result_probabilities"],
                "predicted_result": prediction["predicted_result"],
                "lambda_a": prediction["lambda_a"],
                "lambda_b": prediction["lambda_b"],
                "top_scorelines": prediction["top_scorelines"][:5],
                "v20_adjustments": {
                    "v18_scoreline_weight": prediction["v20_adjustments"][
                        "v18_scoreline_weight"
                    ],
                    "rank_stabilizer": prediction["v20_adjustments"][
                        "rank_stabilizer"
                    ],
                },
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

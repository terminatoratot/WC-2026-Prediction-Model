#!/usr/bin/env python3
"""V13 W/D/L decisions with V11 Poisson/Dixon-Coles exact scores."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

import v11_wcq_results_model as v11


canon_team = v11.canon_team


@dataclass(frozen=True)
class V13Config:
    draw_decision_threshold: float = 0.2147
    close_elo_gap: float = 100.0
    close_match_draw_target: float = 0.218045
    # Retained for callers using the earlier V13 config. Score widening is
    # disabled while the hybrid model uses V11 exact-score probabilities.
    large_elo_gap: float = 200.0
    large_mismatch_goal_std_scale: float = 1.10
    live_elo_k: float = 24.0


def _redistribute_draw_probability(
    result_probabilities: Dict[str, float],
    target_draw_probability: float,
) -> Dict[str, float]:
    current_draw = float(result_probabilities["draw"])
    target_draw = float(np.clip(target_draw_probability, current_draw, 0.55))
    non_draw_total = max(
        result_probabilities["team_a_win"]
        + result_probabilities["team_b_win"],
        1e-12,
    )
    return {
        "team_a_win": (1.0 - target_draw)
        * result_probabilities["team_a_win"]
        / non_draw_total,
        "draw": target_draw,
        "team_b_win": (1.0 - target_draw)
        * result_probabilities["team_b_win"]
        / non_draw_total,
    }


class V13LiveSignalModel:
    def __init__(
        self,
        base_model: v11.StrongWorldCupModel,
        config: V13Config | None = None,
    ):
        self.base_model = base_model
        self.config = config or V13Config()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_model, name)

    def predict(
        self,
        team_a: str,
        team_b: str,
        host_a: bool = False,
        host_b: bool = False,
        knockout: bool = False,
        max_goals: int = 10,
    ) -> Dict[str, Any]:
        features = self.base_model.make_features(
            team_a,
            team_b,
            host_a,
            host_b,
            knockout,
        )
        elo_gap = abs(float(features.iloc[0]["elo_diff"]))
        prediction = self.base_model.predict(
            team_a,
            team_b,
            host_a,
            host_b,
            knockout,
            max_goals,
        )

        adjusted_results = dict(prediction["result_probabilities"])
        draw_boost_applied = elo_gap < self.config.close_elo_gap
        if draw_boost_applied:
            adjusted_results = _redistribute_draw_probability(
                adjusted_results,
                self.config.close_match_draw_target,
            )

        # Keep V11's full score policy because it produced better observed
        # top-two coverage than the experimental unreweighted score matrix.
        prediction["result_probabilities"] = adjusted_results

        draw_signal = float(prediction["draw_model_probability"])
        if draw_signal >= self.config.draw_decision_threshold:
            predicted_result = "draw"
        elif (
            prediction["result_probabilities"]["team_a_win"]
            >= prediction["result_probabilities"]["team_b_win"]
        ):
            predicted_result = "team_a_win"
        else:
            predicted_result = "team_b_win"

        prediction["predicted_result"] = predicted_result
        prediction["v13_adjustments"] = {
            "pre_match_elo_gap": elo_gap,
            "draw_signal": draw_signal,
            "draw_decision_threshold": self.config.draw_decision_threshold,
            "draw_boost_applied": draw_boost_applied,
            "close_match_draw_target": self.config.close_match_draw_target,
            "variance_widened": False,
            "goal_std_scale": 1.0,
            "live_elo_k": self.config.live_elo_k,
            "wdl_model": "v13",
            "exact_score_model": "v11_poisson_dixon_coles_reweighted",
            "exact_score_dixon_coles_rho": prediction.get(
                "calibration_notes",
                {},
            ).get("dixon_coles_rho"),
            "exact_score_result_reweighting": True,
        }
        prediction["calibration_notes"] = {
            **prediction.get("calibration_notes", {}),
            "v13": prediction["v13_adjustments"],
            "hybrid_model_policy": (
                "V13 supplies W/D/L probabilities and the result decision; "
                "V11 supplies its calibrated Poisson/Dixon-Coles exact-score "
                "distribution and all score-derived markets."
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
        """Update live Elo state after an observed match."""
        name_a = canon_team(team_a)
        name_b = canon_team(team_b)
        elo_a = float(self.base_model.latest_elo.get(name_a, 1500.0))
        elo_b = float(self.base_model.latest_elo.get(name_b, 1500.0))
        expected_a = v11.elo_expected(elo_a, elo_b)
        score_a = 1.0 if goals_a > goals_b else 0.5 if goals_a == goals_b else 0.0
        delta = self.config.live_elo_k * (score_a - expected_a)
        self.base_model.latest_elo[name_a] = elo_a + delta
        self.base_model.latest_elo[name_b] = elo_b - delta
        return {
            "team_a_elo_before": elo_a,
            "team_b_elo_before": elo_b,
            "team_a_elo_after": elo_a + delta,
            "team_b_elo_after": elo_b - delta,
            "elo_delta_a": delta,
        }


def build_from_zip(
    zip_path,
    train_csv=None,
    test_csv=None,
    model_type="ensemble",
    box_csv=None,
    results_csv=None,
    former_names_csv=None,
    prediction_year=2026,
    qualifier_blend_start_year=2014,
    qualifier_full_weight_year=2022,
    qualifier_minimum_influence=0.0,
    recency_half_life_years=16.0,
    recency_min_weight=0.10,
):
    base_model, data = v11.build_from_zip(
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
    )
    return V13LiveSignalModel(base_model), data


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    parser = argparse.ArgumentParser(
        description="Run a V13 W/D/L and V11 exact-score match prediction."
    )
    parser.add_argument("--team-a", required=True)
    parser.add_argument("--team-b", required=True)
    parser.add_argument("--host-a", action="store_true")
    parser.add_argument("--host-b", action="store_true")
    parser.add_argument("--knockout", action="store_true")
    parser.add_argument(
        "--model",
        default="ensemble",
        choices=[
            "ensemble",
            "hgb",
            "rf",
            "poisson",
            "lightgbm",
            "xgboost",
            "catboost",
        ],
    )
    parser.add_argument("--outdir", default="outputs_v13_prediction")
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
    parser.add_argument(
        "--former-names",
        default=str(data_dir / "former_names.csv"),
    )
    parser.add_argument("--prediction-year", type=int, default=2026)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    output_dir = v11.unique_output_dir(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, _ = build_from_zip(
        args.worldcupsai_zip,
        train_csv=args.team_train,
        test_csv=args.team_test,
        model_type=args.model,
        box_csv=args.box_data,
        results_csv=args.results_data,
        former_names_csv=args.former_names,
        prediction_year=args.prediction_year,
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
                "wdl_model": "v13",
                "exact_score_model": (
                    "v11_poisson_dixon_coles_reweighted"
                ),
                "model_type": args.model,
                "team_a": prediction["team_a"],
                "team_b": prediction["team_b"],
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
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

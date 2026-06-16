#!/usr/bin/env python3
"""V17: V15 CatBoost model with all-match recency training and WC-only tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import v11_wcq_results_model as v11
import v15_catboost_model as v15


canon_team = v11.canon_team

DEFAULT_RECENCY_HALF_LIFE_YEARS = 6.0
DEFAULT_RECENCY_MIN_WEIGHT = 0.03


def scoreline_text(item: dict[str, Any]) -> str:
    return f"{int(item['team_a_goals'])}-{int(item['team_b_goals'])}"


def _timeline_feature_columns(timeline: pd.DataFrame) -> list[str]:
    excluded = {
        "source_index",
        "date",
        "team_a",
        "team_b",
        "goals_a",
        "goals_b",
        "tournament",
        "tournament_type",
        "prestige_weight",
        "is_continental_final",
        "neutral",
        "country",
    }
    return [column for column in timeline.columns if column not in excluded]


def build_all_match_training_frame(
    world_cup_frame: pd.DataFrame,
    timeline: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Use prior World Cups plus all non-WC internationals as training rows.

    The World Cup rows come from the curated WorldCupSAI data. International
    rows come from results.csv and are used for supervised training only when
    they are not FIFA World Cup matches, preventing duplicate WC targets.
    """
    wc = world_cup_frame.copy()
    wc["prestige_weight"] = 1.0
    wc["tournament_type"] = "WC"
    wc["training_source"] = "world_cup"

    feature_columns = _timeline_feature_columns(timeline)
    direct_lookup: dict[tuple[pd.Timestamp, str, str], dict[str, Any]] = {}
    for _, row in timeline.iterrows():
        key = (
            pd.Timestamp(row["date"]).normalize(),
            row["team_a"],
            row["team_b"],
        )
        direct_lookup[key] = {column: row[column] for column in feature_columns}

    matched = 0
    for index, row in wc.iterrows():
        date = pd.Timestamp(row["date"]).normalize()
        key = (date, row["team_a"], row["team_b"])
        reverse_key = (date, row["team_b"], row["team_a"])
        values = direct_lookup.get(key)
        if values is None and reverse_key in direct_lookup:
            values = v15._reverse_pair_features(direct_lookup[reverse_key])
        if values is None:
            continue
        matched += 1
        for column, value in values.items():
            wc.at[index, column] = value

    row_columns = [
        "match_id",
        "date",
        "team_a",
        "team_b",
        "goals_a",
        "goals_b",
        "goal_diff",
        "is_group_stage",
        "is_knockout",
        "host_a",
        "host_b",
        "host_diff",
        "abs_host_diff",
        "same_confed",
        "prestige_weight",
        "tournament_type",
        "training_source",
    ]
    if timeline.empty:
        all_match_rows = pd.DataFrame(columns=row_columns + feature_columns)
    else:
        international = timeline[timeline["tournament_type"] != "WC"].copy()
        neutral = international["neutral"].astype(bool)
        all_match_rows = pd.DataFrame(
            {
                "match_id": international["source_index"].map(
                    lambda value: f"international_{value}"
                ),
                "date": international["date"],
                "team_a": international["team_a"],
                "team_b": international["team_b"],
                "goals_a": international["goals_a"],
                "goals_b": international["goals_b"],
                "goal_diff": (
                    international["goals_a"] - international["goals_b"]
                ),
                "is_group_stage": 0,
                "is_knockout": 0,
                "host_a": (~neutral).astype(int),
                "host_b": 0,
                "host_diff": (~neutral).astype(int),
                "abs_host_diff": (~neutral).astype(int),
                "same_confed": international["is_continental_final"].astype(int),
                "prestige_weight": international["prestige_weight"],
                "tournament_type": international["tournament_type"],
                "training_source": "international",
            }
        )
        for column in feature_columns:
            all_match_rows[column] = international[column].to_numpy()

    combined = pd.concat([wc, all_match_rows], ignore_index=True, sort=False)
    qualifier_columns = [
        column for column in combined if column.startswith("qual_")
    ]
    combined = combined.drop(columns=qualifier_columns)
    combined = combined.sort_values("date", kind="stable").reset_index(
        drop=True
    )

    excluded = {
        "match_id",
        "date",
        "team_a",
        "team_b",
        "goals_a",
        "goals_b",
        "goal_diff",
        "prestige_weight",
    }
    event_targets = {
        f"{event}_{side}"
        for event in (
            "yellow_cards",
            "red_cards",
            "second_yellow_cards",
            "sending_offs",
            "penalty_goals",
            "penalty_kicks",
            "penalty_kicks_converted",
            "own_goals",
            "substitutions",
        )
        for side in ("a", "b")
    }
    features = [
        column
        for column in combined.columns
        if column not in excluded
        and column not in event_targets
        and pd.api.types.is_numeric_dtype(combined[column])
        and combined[column].notna().mean() > 0.20
    ]

    tournament_counts = (
        combined["tournament_type"].value_counts(dropna=False).to_dict()
    )
    summary = {
        "world_cup_rows": int(len(wc)),
        "international_training_rows": int(len(all_match_rows)),
        "world_cup_external_matches": int(matched),
        "international_timeline_rows": int(len(timeline)),
        "training_rows_by_tournament_type": {
            str(key): int(value) for key, value in tournament_counts.items()
        },
        "training_policy": (
            "Train on curated prior World Cups plus every non-WC international "
            "available before the cutoff; evaluate/backtest only World Cups."
        ),
    }
    return combined, features, summary


class V17CatBoostWorldCupModel(v15.V15CatBoostWorldCupModel):
    """V15 model with v17's stronger default recency weighting."""

    def __init__(
        self,
        recency_half_life_years: float = DEFAULT_RECENCY_HALF_LIFE_YEARS,
        recency_min_weight: float = DEFAULT_RECENCY_MIN_WEIGHT,
    ):
        super().__init__(
            recency_half_life_years=recency_half_life_years,
            recency_min_weight=recency_min_weight,
        )

    @staticmethod
    def _catboost_common() -> dict[str, Any]:
        params = v15.V15CatBoostWorldCupModel._catboost_common()
        params["random_seed"] = 17
        return params


class V17CatBoostModel(v15.V15CatBoostModel):
    """Prediction wrapper that reports v17's training policy."""

    def predict(self, *args, **kwargs) -> dict[str, Any]:
        prediction = super().predict(*args, **kwargs)
        adjustments = dict(prediction.pop("v15_adjustments", {}))
        adjustments.update(
            {
                "wdl_model": "v17_catboost",
                "version": "v17-recency-all-matches",
                "recency_all_match_training": True,
                "world_cup_only_validation": True,
                "recency_half_life_years": self.base_model.recency_half_life_years,
                "recency_min_weight": self.base_model.recency_min_weight,
            }
        )
        prediction["v17_adjustments"] = adjustments
        prediction["calibration_notes"].pop("v15", None)
        prediction["calibration_notes"]["v17"] = adjustments
        prediction["calibration_notes"]["hybrid_model_policy"] = (
            "V17 keeps the V15 CatBoost/player-profile architecture, trains "
            "the supervised heads on all pre-cutoff non-WC internationals plus "
            "curated prior World Cups, applies stronger recency weighting, "
            "and validates with World Cup-only forward splits."
        )
        return prediction


def _training_inputs(
    zip_path,
    train_csv,
    test_csv,
    box_csv,
    results_csv,
    former_names_csv,
    results_as_of,
    qualifier_blend_start_year,
    qualifier_full_weight_year,
    qualifier_minimum_influence,
):
    loader = v11.WorldCupSAILoader(
        zip_path,
        Path(str(zip_path) + "_extracted"),
    )
    matches = loader.load_matches()
    current = v11.load_current_team_features(train_csv, test_csv)
    box = v11.load_kaggle_box_data(box_csv)
    qualification_results = v11.load_world_cup_qualification_results(
        results_csv,
        former_names_csv,
    )
    qualifier_source = (
        qualification_results if not qualification_results.empty else box
    )
    historical_current = pd.DataFrame(columns=["team"])
    frame, _, events = v11.build_rolling_features(
        matches,
        historical_current,
        qualifier_box=qualifier_source,
        qualifier_fallback_box=box,
        qualifier_blend_start_year=qualifier_blend_start_year,
        qualifier_full_weight_year=qualifier_full_weight_year,
        qualifier_minimum_influence=qualifier_minimum_influence,
    )
    international_results = v15.load_international_results(
        results_csv,
        former_names_csv=former_names_csv,
        as_of=results_as_of,
    )
    timeline, international_state = v15.build_international_timeline(
        international_results
    )
    expanded_frame, expanded_features, expansion_summary = (
        build_all_match_training_frame(frame, timeline)
    )
    return {
        "matches": matches,
        "current": current,
        "box": box,
        "qualifier_source": qualifier_source,
        "events": events,
        "international_state": international_state,
        "expanded_frame": expanded_frame,
        "expanded_features": expanded_features,
        "expansion_summary": expansion_summary,
    }


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
    recency_half_life_years=DEFAULT_RECENCY_HALF_LIFE_YEARS,
    recency_min_weight=DEFAULT_RECENCY_MIN_WEIGHT,
    player_ratings_csv=None,
    declared_squads_csv=None,
    results_as_of="2026-06-10",
):
    v15.require_catboost()
    data_dir = Path(__file__).resolve().parent / "data"
    player_ratings_csv = player_ratings_csv or (
        data_dir / "player_ratings_international.csv"
    )
    declared_squads_csv = declared_squads_csv or (
        data_dir / "world_cup_2026_declared_squads.csv"
    )
    inputs = _training_inputs(
        zip_path,
        train_csv,
        test_csv,
        box_csv,
        results_csv,
        former_names_csv,
        results_as_of,
        qualifier_blend_start_year,
        qualifier_full_weight_year,
        qualifier_minimum_influence,
    )
    player_ratings = v15.load_player_ratings(player_ratings_csv)
    declared_squads = v15.load_declared_squads(declared_squads_csv)
    current_squad_profiles = v15.build_current_squad_profiles(
        declared_squads,
        player_ratings,
    )
    player_frame, player_features = v15.add_historical_player_features(
        inputs["expanded_frame"],
        player_ratings,
        inputs["expanded_features"],
    )
    outcome_model = (
        V17CatBoostWorldCupModel(
            recency_half_life_years=recency_half_life_years,
            recency_min_weight=recency_min_weight,
        )
        .fit(
            inputs["expanded_frame"],
            inputs["expanded_features"],
            [],
            inputs["current"],
        )
        .set_box_data(inputs["box"])
        .set_qualifier_data(
            inputs["qualifier_source"],
            fallback_box=inputs["box"],
            prediction_year=prediction_year,
            blend_start_year=qualifier_blend_start_year,
            full_weight_year=qualifier_full_weight_year,
            minimum_influence=qualifier_minimum_influence,
        )
        .set_current_international_state(
            inputs["international_state"],
            results_as_of,
        )
    )
    player_model = (
        V17CatBoostWorldCupModel(
            recency_half_life_years=recency_half_life_years,
            recency_min_weight=recency_min_weight,
        )
        .fit(player_frame, player_features, inputs["events"], inputs["current"])
        .set_box_data(inputs["box"])
        .set_qualifier_data(
            inputs["qualifier_source"],
            fallback_box=inputs["box"],
            prediction_year=prediction_year,
            blend_start_year=qualifier_blend_start_year,
            full_weight_year=qualifier_full_weight_year,
            minimum_influence=qualifier_minimum_influence,
        )
        .set_current_squad_profiles(current_squad_profiles)
        .set_current_international_state(
            inputs["international_state"],
            results_as_of,
        )
    )
    data = v11.DataBundle(
        matches=inputs["matches"],
        team_current=inputs["current"],
        training_frame=player_frame,
        event_columns=inputs["events"],
        box_frame=inputs["box"],
    )
    model = V17CatBoostModel(player_model, outcome_model)
    model.training_data_summary = {
        **inputs["expansion_summary"],
        "results_as_of": str(pd.Timestamp(results_as_of).date()),
        "recency_half_life_years": recency_half_life_years,
        "recency_min_weight": recency_min_weight,
        "continental_stage_features": False,
        "continental_stage_feature_reason": (
            "results.csv has tournament names but no round or stage column"
        ),
    }
    return model, data


def chronological_world_cup_backtest(
    zip_path: str,
    train_csv: str | None = None,
    test_csv: str | None = None,
    test_years: list[int] | None = None,
    min_train_year: int = 1930,
    max_goals: int = 10,
    box_csv: str | None = None,
    results_csv: str | None = None,
    former_names_csv: str | None = None,
    qualifier_blend_start_year: int = 2014,
    qualifier_full_weight_year: int = 2022,
    qualifier_minimum_influence: float = 0.0,
    recency_half_life_years: float = DEFAULT_RECENCY_HALF_LIFE_YEARS,
    recency_min_weight: float = DEFAULT_RECENCY_MIN_WEIGHT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on final results before each cutoff and test only World Cups."""
    loader = v11.WorldCupSAILoader(zip_path, Path(str(zip_path) + "_extracted"))
    all_matches = loader.load_matches()
    all_years = sorted(int(y) for y in all_matches["year"].dropna().unique())
    if test_years is None:
        test_years = [year for year in all_years if year >= 2010]
    else:
        test_years = [int(year) for year in test_years]

    pred_rows = []
    summary_rows = []
    for year in test_years:
        train_matches = all_matches[
            (all_matches["year"] < year)
            & (all_matches["year"] >= min_train_year)
        ].copy()
        test_matches = all_matches[all_matches["year"] == year].copy()
        if len(train_matches) < 80 or test_matches.empty:
            continue

        cutoff = f"{year - 1}-12-31"
        inputs = _training_inputs(
            zip_path,
            train_csv,
            test_csv,
            box_csv,
            results_csv,
            former_names_csv,
            cutoff,
            qualifier_blend_start_year,
            qualifier_full_weight_year,
            qualifier_minimum_influence,
        )
        inputs["matches"] = train_matches
        historical_box = (
            inputs["box"][inputs["box"]["box_year"] < year].copy()
            if not inputs["box"].empty and "box_year" in inputs["box"]
            else inputs["box"]
        )
        if not inputs["qualifier_source"].empty and "box_year" in inputs["qualifier_source"]:
            historical_qualifiers = inputs["qualifier_source"][
                inputs["qualifier_source"]["box_year"] < year
            ].copy()
        else:
            historical_qualifiers = historical_box

        historical_current = pd.DataFrame(columns=["team"])
        train_frame, _, events = v11.build_rolling_features(
            train_matches,
            historical_current,
            qualifier_box=historical_qualifiers,
            qualifier_fallback_box=historical_box,
            qualifier_blend_start_year=qualifier_blend_start_year,
            qualifier_full_weight_year=qualifier_full_weight_year,
            qualifier_minimum_influence=qualifier_minimum_influence,
        )
        international_results = v15.load_international_results(
            results_csv,
            former_names_csv=former_names_csv,
            as_of=cutoff,
        )
        timeline, international_state = v15.build_international_timeline(
            international_results
        )
        expanded_frame, expanded_features, expansion_summary = (
            build_all_match_training_frame(train_frame, timeline)
        )
        model = (
            V17CatBoostWorldCupModel(
                recency_half_life_years=recency_half_life_years,
                recency_min_weight=recency_min_weight,
            )
            .fit(expanded_frame, expanded_features, [], historical_current)
            .set_box_data(historical_box)
            .set_qualifier_data(
                historical_qualifiers,
                fallback_box=historical_box,
                prediction_year=year,
                blend_start_year=qualifier_blend_start_year,
                full_weight_year=qualifier_full_weight_year,
                minimum_influence=qualifier_minimum_influence,
            )
            .set_current_international_state(international_state, cutoff)
        )

        for _, row in test_matches.iterrows():
            prediction = model.predict(
                row.team_a,
                row.team_b,
                host_a=bool(row.host_a),
                host_b=bool(row.host_b),
                knockout=bool(row.is_knockout),
                max_goals=max_goals,
            )
            actual = v11.actual_result_label(row.goals_a, row.goals_b)
            result_probs = prediction["result_probabilities"]
            predicted_result = max(result_probs, key=result_probs.get)
            score_probs = {
                (int(item["team_a_goals"]), int(item["team_b_goals"])): float(
                    item["probability"]
                )
                for item in prediction["scoreline_probabilities"]
            }
            exact_prob = score_probs.get(
                (int(row.goals_a), int(row.goals_b)),
                0.0,
            )
            actual_score = f"{int(row.goals_a)}-{int(row.goals_b)}"
            top_three = prediction["top_scorelines"][:3]
            top_scorelines = [scoreline_text(item) for item in top_three]
            top_probabilities = [
                float(item["probability"]) for item in top_three
            ]
            while len(top_scorelines) < 3:
                top_scorelines.append("")
                top_probabilities.append(0.0)
            pred_rows.append(
                {
                    "test_year": year,
                    "match_id": row.match_id,
                    "date": row.date,
                    "stage": row.stage_name,
                    "team_a": row.team_a,
                    "team_b": row.team_b,
                    "actual_score": actual_score,
                    "lambda_a": prediction["lambda_a"],
                    "lambda_b": prediction["lambda_b"],
                    "predicted_result": predicted_result,
                    "actual_result": actual,
                    "correct_result": int(predicted_result == actual),
                    "team_a_win_prob": result_probs["team_a_win"],
                    "draw_prob": result_probs["draw"],
                    "team_b_win_prob": result_probs["team_b_win"],
                    "actual_result_probability": result_probs[actual],
                    "result_log_loss": v11.safe_log_loss(result_probs[actual]),
                    "result_brier": v11.brier_score_3way(actual, result_probs),
                    "goal_mae": (
                        abs(prediction["lambda_a"] - row.goals_a)
                        + abs(prediction["lambda_b"] - row.goals_b)
                    )
                    / 2.0,
                    "goal_diff_abs_error": abs(
                        (prediction["lambda_a"] - prediction["lambda_b"])
                        - (row.goals_a - row.goals_b)
                    ),
                    "exact_score_probability": exact_prob,
                    "exact_score_log_loss": v11.safe_log_loss(exact_prob),
                    "top_1_scoreline": top_scorelines[0],
                    "top_1_probability": top_probabilities[0],
                    "top_2_scoreline": top_scorelines[1],
                    "top_2_probability": top_probabilities[1],
                    "top_3_scoreline": top_scorelines[2],
                    "top_3_probability": top_probabilities[2],
                    "actual_is_top_1": actual_score == top_scorelines[0],
                    "actual_is_top_2": actual_score == top_scorelines[1],
                    "actual_is_top_3": actual_score == top_scorelines[2],
                    "actual_in_top_3": actual_score in set(top_scorelines),
                }
            )

        year_df = pd.DataFrame(
            [item for item in pred_rows if item["test_year"] == year]
        )
        summary_rows.append(
            {
                "test_year": year,
                "train_world_cup_matches": int(len(train_matches)),
                "train_total_matches": int(len(expanded_frame)),
                "train_international_matches": int(
                    expansion_summary["international_training_rows"]
                ),
                "test_world_cup_matches": int(len(test_matches)),
                "result_accuracy": float(year_df["correct_result"].mean()),
                "mean_result_log_loss": float(
                    year_df["result_log_loss"].mean()
                ),
                "mean_result_brier": float(year_df["result_brier"].mean()),
                "mean_goal_mae": float(year_df["goal_mae"].mean()),
                "mean_goal_diff_abs_error": float(
                    year_df["goal_diff_abs_error"].mean()
                ),
                "mean_exact_score_log_loss": float(
                    year_df["exact_score_log_loss"].mean()
                ),
                "mean_actual_result_probability": float(
                    year_df["actual_result_probability"].mean()
                ),
                "mean_exact_score_probability": float(
                    year_df["exact_score_probability"].mean()
                ),
                "exact_score_top_1_accuracy": float(
                    year_df["actual_is_top_1"].mean()
                ),
                "exact_score_top_3_accuracy": float(
                    year_df["actual_in_top_3"].mean()
                ),
                "exact_score_top_3_hits": int(
                    year_df["actual_in_top_3"].sum()
                ),
                "features_used": len(expanded_features),
                "event_targets": "",
                "model_type": "v17_catboost",
                "training_target_policy": (
                    "final match score/result only; no box-event targets"
                ),
                "recency_half_life_years": recency_half_life_years,
                "recency_min_weight": recency_min_weight,
                "results_cutoff": cutoff,
            }
        )

    pred_df = pd.DataFrame(pred_rows)
    summary_df = pd.DataFrame(summary_rows)
    if not pred_df.empty:
        overall = {
            "test_year": "overall",
            "train_world_cup_matches": np.nan,
            "train_total_matches": np.nan,
            "train_international_matches": np.nan,
            "test_world_cup_matches": int(len(pred_df)),
            "result_accuracy": float(pred_df["correct_result"].mean()),
            "mean_result_log_loss": float(pred_df["result_log_loss"].mean()),
            "mean_result_brier": float(pred_df["result_brier"].mean()),
            "mean_goal_mae": float(pred_df["goal_mae"].mean()),
            "mean_goal_diff_abs_error": float(
                pred_df["goal_diff_abs_error"].mean()
            ),
            "mean_exact_score_log_loss": float(
                pred_df["exact_score_log_loss"].mean()
            ),
            "mean_actual_result_probability": float(
                pred_df["actual_result_probability"].mean()
            ),
            "mean_exact_score_probability": float(
                pred_df["exact_score_probability"].mean()
            ),
            "exact_score_top_1_accuracy": float(
                pred_df["actual_is_top_1"].mean()
            ),
            "exact_score_top_3_accuracy": float(
                pred_df["actual_in_top_3"].mean()
            ),
            "exact_score_top_3_hits": int(pred_df["actual_in_top_3"].sum()),
            "features_used": np.nan,
            "event_targets": "",
            "model_type": "v17_catboost",
            "training_target_policy": (
                "final match score/result only; no box-event targets"
            ),
            "recency_half_life_years": recency_half_life_years,
            "recency_min_weight": recency_min_weight,
            "results_cutoff": "",
        }
        summary_df = pd.concat(
            [summary_df, pd.DataFrame([overall])],
            ignore_index=True,
        )
    return pred_df, summary_df


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    parser = argparse.ArgumentParser(
        description=(
            "Run V17: V15 CatBoost architecture with all-match recency "
            "training and World Cup-only forward tests."
        )
    )
    parser.add_argument("--team-a")
    parser.add_argument("--team-b")
    parser.add_argument("--host-a", action="store_true")
    parser.add_argument("--host-b", action="store_true")
    parser.add_argument("--knockout", action="store_true")
    parser.add_argument(
        "--outdir",
        default="outputs/outputs_v17_recency_all_matches_prediction",
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
    parser.add_argument("--results-data", default=str(data_dir / "results.csv"))
    parser.add_argument(
        "--results-as-of",
        default="2026-06-10",
        help="Use internationals on or before this date for live fitting.",
    )
    parser.add_argument("--former-names", default=str(data_dir / "former_names.csv"))
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
        "--recency-half-life-years",
        type=float,
        default=DEFAULT_RECENCY_HALF_LIFE_YEARS,
    )
    parser.add_argument(
        "--recency-min-weight",
        type=float,
        default=DEFAULT_RECENCY_MIN_WEIGHT,
    )
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--test-years", nargs="*", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    output_dir = v11.unique_output_dir(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.backtest:
        pred_df, summary_df = chronological_world_cup_backtest(
            zip_path=args.worldcupsai_zip,
            train_csv=args.team_train,
            test_csv=args.team_test,
            test_years=args.test_years,
            box_csv=args.box_data,
            results_csv=args.results_data,
            former_names_csv=args.former_names,
            recency_half_life_years=args.recency_half_life_years,
            recency_min_weight=args.recency_min_weight,
        )
        pred_df.to_csv(output_dir / "backtest_predictions.csv", index=False)
        summary_df.to_csv(output_dir / "backtest_summary.csv", index=False)
        print(summary_df.to_string(index=False))
        return

    if not args.team_a or not args.team_b:
        raise SystemExit(
            "For single-match prediction, provide --team-a and --team-b. "
            "For World Cup-only forward testing, use --backtest."
        )

    model, data = build_from_zip(
        args.worldcupsai_zip,
        train_csv=args.team_train,
        test_csv=args.team_test,
        box_csv=args.box_data,
        results_csv=args.results_data,
        former_names_csv=args.former_names,
        prediction_year=args.prediction_year,
        recency_half_life_years=args.recency_half_life_years,
        recency_min_weight=args.recency_min_weight,
        player_ratings_csv=args.player_ratings,
        declared_squads_csv=args.declared_squads,
        results_as_of=args.results_as_of,
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
    (output_dir / "player_profiles.json").write_text(
        json.dumps(prediction["player_profiles"], indent=2)
    )
    pd.DataFrame(prediction["top_scorelines"]).to_csv(
        output_dir / "scoreline_probabilities_top.csv",
        index=False,
    )
    pd.DataFrame(prediction["scoreline_probabilities"]).to_csv(
        output_dir / "scoreline_probabilities.csv",
        index=False,
    )
    data.training_frame.to_csv(output_dir / "training_frame.csv", index=False)
    (output_dir / "model_summary.json").write_text(
        json.dumps(
            {
                "version": "v17-recency-all-matches",
                "learned_model_family": "sklearn_catboost_ensemble",
                "wdl_model": "v17_catboost",
                "exact_score_model": (
                    "v15_catboost_goals_with_all_match_recency_training"
                ),
                "team_a": prediction["team_a"],
                "team_b": prediction["team_b"],
                "player_ratings_source": args.player_ratings,
                "declared_squads_source": args.declared_squads,
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
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

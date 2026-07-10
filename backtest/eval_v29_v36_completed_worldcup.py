#!/usr/bin/env python3
"""Evaluate V29 and V36 on completed 2026 World Cup matches.

The completed-match CSV is used only as the answer key. Models are built with
empty observed-score inputs, and V36 also receives empty FotMob match-fact and
goal-event inputs, so final scores and goal timelines cannot leak into the
prediction layer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
VERSIONS_DIR = PROJECT_DIR / "versions"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(VERSIONS_DIR) not in sys.path:
    sys.path.insert(0, str(VERSIONS_DIR))

import market_edge  # noqa: F401  (bundles v11/v13-v27/v28-v38/v39; populates sys.modules)
import v15_catboost_model as v15
import v29_tail_risk_scoreline_model as v29
import v36_fotmob_current_form_model as v36


DATA_DIR = PROJECT_DIR / "data"
HOSTS_2026 = {"Canada", "Mexico", "USA", "United States"}


def score_text(item: dict[str, Any]) -> str:
    return f"{int(item['team_a_goals'])}-{int(item['team_b_goals'])}"


def result_label(goals_a: int, goals_b: int) -> str:
    if goals_a > goals_b:
        return "team_a_win"
    if goals_b > goals_a:
        return "team_b_win"
    return "draw"


def add_rank_fields(
    output: dict[str, Any],
    prefix: str,
    items: list[dict[str, Any]],
    actual: str,
    limit: int,
) -> None:
    for rank in range(1, limit + 1):
        item = items[rank - 1] if rank <= len(items) else None
        if item:
            scoreline = score_text(item)
            output[f"{prefix}_{rank}_scoreline"] = scoreline
            output[f"{prefix}_{rank}_probability"] = float(item.get("probability", 0.0) or 0.0)
            output[f"actual_is_{prefix}_{rank}"] = actual == scoreline
        else:
            output[f"{prefix}_{rank}_scoreline"] = ""
            output[f"{prefix}_{rank}_probability"] = 0.0
            output[f"actual_is_{prefix}_{rank}"] = False


def write_no_score_inputs(outdir: Path) -> dict[str, Path]:
    no_score_dir = outdir / "no_score_model_inputs"
    no_score_dir.mkdir(parents=True, exist_ok=True)

    observed_path = no_score_dir / "empty_observed_matches_no_scores.csv"
    pd.DataFrame(
        columns=[
            "match_id",
            "date_label",
            "stage",
            "group",
            "team_a",
            "team_b",
            "goals_a",
            "goals_b",
            "source",
        ]
    ).to_csv(observed_path, index=False)

    match_facts_path = no_score_dir / "empty_fotmob_match_facts_no_scores.csv"
    pd.DataFrame(
        columns=[
            "match_id",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "status",
            "raw_text",
        ]
    ).to_csv(match_facts_path, index=False)

    goal_events_path = no_score_dir / "empty_fotmob_goal_events_no_scores.csv"
    pd.DataFrame(
        columns=[
            "match_id",
            "match_slug",
            "home_team",
            "away_team",
            "goal_index",
            "elapsed_minute",
            "stoppage_minute",
            "scoring_country",
            "opponent_country",
            "scorer",
            "home_score_after",
            "away_score_after",
        ]
    ).to_csv(goal_events_path, index=False)

    return {
        "observed": observed_path,
        "match_facts": match_facts_path,
        "goal_events": goal_events_path,
    }


def build_v29_model(args: argparse.Namespace, no_score_inputs: dict[str, Path]) -> Any:
    model, _ = v29.build_from_zip(
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
        observed_matches_csv=str(no_score_inputs["observed"]),
        fotmob_leaders_csv=args.fotmob_leaders_v29,
        include_observed_goals=False,
        include_fotmob_goal_stats=False,
        include_group_score_context=False,
    )
    return model


def build_v36_model(args: argparse.Namespace, no_score_inputs: dict[str, Path]) -> Any:
    model, _ = v36.build_from_zip(
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
        observed_matches_csv=str(no_score_inputs["observed"]),
        fotmob_leaders_csv=args.fotmob_leaders_v36,
        fotmob_player_stats_csv=args.fotmob_player_stats,
        fotmob_lineups_csv=args.fotmob_lineups,
        fotmob_substitutions_csv=args.fotmob_substitutions,
        fotmob_keeper_stats_csv=args.fotmob_keeper_stats,
        fotmob_match_facts_csv=str(no_score_inputs["match_facts"]),
        fotmob_goal_events_csv=str(no_score_inputs["goal_events"]),
        include_observed_goals=False,
        include_group_score_context=False,
        include_fotmob_goal_stats=False,
    )
    return model


def evaluate(model: Any, observed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, row in observed.reset_index(drop=True).iterrows():
        team_a = str(row["team_a"])
        team_b = str(row["team_b"])
        goals_a = int(row["goals_a"])
        goals_b = int(row["goals_b"])
        actual = f"{goals_a}-{goals_b}"
        prediction = model.predict(
            team_a,
            team_b,
            host_a=team_a in HOSTS_2026,
            host_b=team_b in HOSTS_2026,
            knockout=str(row.get("stage", "")).strip().lower() != "group stage",
        )

        top3_items = prediction.get("top_scorelines", [])[:3]
        plus_items = prediction.get("top_scorelines_plus_outlier") or top3_items
        plus_items = plus_items[:4]
        top3 = [score_text(item) for item in top3_items]
        plus = [score_text(item) for item in plus_items]
        result_probabilities = prediction["result_probabilities"]
        actual_result = result_label(goals_a, goals_b)
        predicted_result = prediction.get(
            "predicted_result",
            max(result_probabilities, key=result_probabilities.get),
        )
        actual_score_probability = next(
            (
                float(item["probability"])
                for item in prediction.get("scoreline_probabilities", [])
                if score_text(item) == actual
            ),
            0.0,
        )
        output = {
            "observed_order": int(idx) + 1,
            "match_id": row["match_id"],
            "team_a": team_a,
            "team_b": team_b,
            "actual_score": actual,
            "actual_total_goals": goals_a + goals_b,
            "actual_result": actual_result,
            "predicted_result": predicted_result,
            "outcome_correct": predicted_result == actual_result,
            "actual_score_probability": actual_score_probability,
            "predicted_result_probability": float(result_probabilities[predicted_result]),
            "actual_result_probability": float(result_probabilities[actual_result]),
            "team_a_win_probability": float(result_probabilities["team_a_win"]),
            "draw_probability": float(result_probabilities["draw"]),
            "team_b_win_probability": float(result_probabilities["team_b_win"]),
            "lambda_a": float(prediction["lambda_a"]),
            "lambda_b": float(prediction["lambda_b"]),
        }
        add_rank_fields(output, "top", top3_items, actual, 3)
        output["actual_in_top_3"] = actual in set(top3)
        add_rank_fields(output, "plus", plus_items, actual, 4)
        output["actual_in_top_3_plus_outlier"] = actual in set(plus)
        output["outlier_gained_hit"] = (
            output["actual_in_top_3_plus_outlier"] and not output["actual_in_top_3"]
        )
        rows.append(output)
    return pd.DataFrame(rows)


def summarize(model_name: str, comparison: pd.DataFrame, observed_path: str, no_score_inputs: dict[str, Path], csv_path: Path) -> dict[str, Any]:
    n = int(len(comparison))
    top3_hits = int(comparison["actual_in_top_3"].sum())
    plus_hits = int(comparison["actual_in_top_3_plus_outlier"].sum())
    return {
        "model": model_name,
        "evaluation_mode": "completed_worldcup_no_score_leak_inputs_no_plots",
        "n_matches": n,
        "top_1_exact_score_hits": int(comparison["actual_is_top_1"].sum()),
        "top_2_exact_score_hits": int(
            (comparison["actual_is_top_1"] | comparison["actual_is_top_2"]).sum()
        ),
        "top_3_exact_score_hits": top3_hits,
        "top_3_exact_score_accuracy": top3_hits / n if n else 0.0,
        "top_3_plus_outlier_hits": plus_hits,
        "top_3_plus_outlier_accuracy": plus_hits / n if n else 0.0,
        "outlier_gained_hits": int(comparison["outlier_gained_hit"].sum()),
        "outcome_correct": int(comparison["outcome_correct"].sum()),
        "outcome_accuracy": float(comparison["outcome_correct"].mean()) if n else 0.0,
        "mean_lambda_sum": float((comparison["lambda_a"] + comparison["lambda_b"]).mean()) if n else 0.0,
        "mean_actual_total": float(comparison["actual_total_goals"].mean()) if n else 0.0,
        "observed_answer_key_csv": observed_path,
        "model_score_inputs": {key: str(value) for key, value in no_score_inputs.items()},
        "comparison_csv": str(csv_path),
    }


def run_one(model_name: str, args: argparse.Namespace, observed: pd.DataFrame, outdir: Path) -> dict[str, Any]:
    model_outdir = outdir / model_name
    model_outdir.mkdir(parents=True, exist_ok=True)
    no_score_inputs = write_no_score_inputs(model_outdir)
    model = build_v29_model(args, no_score_inputs) if model_name == "v29_tail_risk" else build_v36_model(args, no_score_inputs)
    comparison = evaluate(model, observed)
    csv_path = model_outdir / f"{model_name}_completed_worldcup_top3_comparison_no_score_leak.csv"
    comparison.to_csv(csv_path, index=False)
    summary = summarize(model_name, comparison, args.observed, no_score_inputs, csv_path)
    summary_path = model_outdir / f"{model_name}_completed_worldcup_summary_no_score_leak.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate V29 and V36 completed World Cup accuracy without score leakage.")
    parser.add_argument("--outdir", default="observed_eval/observed_eval_v29_v36_completed_worldcup_no_score_leak")
    parser.add_argument("--observed", default=str(DATA_DIR / "fotmob_completed_matches_observed_schema.csv"))
    parser.add_argument("--models", nargs="+", choices=["v29_tail_risk", "v36_fotmob_current_form"], default=["v29_tail_risk", "v36_fotmob_current_form"])
    parser.add_argument("--worldcupsai-zip", default=str(DATA_DIR / "worldcupsai.zip"))
    parser.add_argument("--team-train", default=str(DATA_DIR / "current_team_features_2026.csv"))
    parser.add_argument("--team-test")
    parser.add_argument("--box-data", default=str(DATA_DIR / "FIFAallMatchBoxData.csv"))
    parser.add_argument("--results-data", default=str(DATA_DIR / "results.csv"))
    parser.add_argument("--results-as-of", default=v15.DEFAULT_RESULTS_AS_OF)
    parser.add_argument("--former-names", default=str(DATA_DIR / "former_names.csv"))
    parser.add_argument("--prediction-year", type=int, default=2026)
    parser.add_argument("--player-ratings", default=str(DATA_DIR / "player_ratings_international.csv"))
    parser.add_argument("--declared-squads", default=str(DATA_DIR / "world_cup_2026_declared_squads.csv"))
    parser.add_argument("--fcratings", default=str(DATA_DIR / "fcratings_top50_worldcup2026.csv"))
    parser.add_argument("--fotmob-leaders-v29", default=str(DATA_DIR / "fotmob_stat_leaders_clean.csv"))
    parser.add_argument("--fotmob-leaders-v36", default=str(DATA_DIR / "fotmob_full_stat_tables_clean.csv"))
    parser.add_argument("--fotmob-player-stats", default=str(DATA_DIR / "fotmob_match_player_stats_clean.csv"))
    parser.add_argument("--fotmob-lineups", default=str(DATA_DIR / "fotmob_match_lineups_clean.csv"))
    parser.add_argument("--fotmob-substitutions", default=str(DATA_DIR / "fotmob_match_substitutions_clean.csv"))
    parser.add_argument("--fotmob-keeper-stats", default=str(DATA_DIR / "fotmob_match_keeper_stats_clean.csv"))
    args = parser.parse_args()

    observed = pd.read_csv(args.observed)
    required = {"match_id", "team_a", "team_b", "goals_a", "goals_b"}
    missing = sorted(required - set(observed.columns))
    if missing:
        raise ValueError(f"Observed file is missing columns: {missing}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summaries = [run_one(model_name, args, observed, outdir) for model_name in args.models]
    combined_path = outdir / "v29_v36_completed_worldcup_summary_no_score_leak.json"
    combined_path.write_text(json.dumps({"summaries": summaries}, indent=2), encoding="utf-8")
    print(json.dumps({"summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate V42 on completed 2026 World Cup matches without score leakage.

The completed-match CSV is used only as the answer key. Model construction uses
empty observed-score, match-fact, and goal-event inputs so final scores and goal
timelines from FotMob cannot feed V42's prediction layer.
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

import compare_v11_top_scorelines as chart
import v15_catboost_model as v15
import v36_fotmob_current_form_model as v36
import v42_fotmob_market_edge_model as v42


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


def result_text(result: str, team_a: str, team_b: str) -> str:
    return {
        "team_a_win": team_a,
        "draw": "Draw",
        "team_b_win": team_b,
    }[result]


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


def build_v42_model(args: argparse.Namespace, no_score_inputs: dict[str, Path]) -> v42.V42FotmobCoverageModel:
    base_model, _ = v36.build_from_zip(
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
        fotmob_leaders_csv=args.fotmob_leaders,
        fotmob_player_stats_csv=args.fotmob_player_stats,
        fotmob_lineups_csv=args.fotmob_lineups,
        fotmob_substitutions_csv=args.fotmob_substitutions,
        fotmob_keeper_stats_csv=args.fotmob_keeper_stats,
        fotmob_match_facts_csv=str(no_score_inputs["match_facts"]),
        fotmob_goal_events_csv=str(no_score_inputs["goal_events"]),
        include_observed_goals=False,
        include_group_score_context=False,
        include_fotmob_goal_stats=False,
        fotmob_wdl_blend=args.fotmob_wdl_blend,
        fotmob_scoreline_blend=args.fotmob_scoreline_blend,
    )
    return v42.V42FotmobCoverageModel(base_model, coverage_margin=args.coverage_margin)


def evaluate(model: v42.V42FotmobCoverageModel, observed: pd.DataFrame) -> pd.DataFrame:
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
        plus_items = prediction.get("top_scorelines_plus_outlier", top3_items)[:4]
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
        outlier = prediction.get("coverage_total_outlier") or prediction.get("game_state_late_outlier") or {}
        output = {
            "observed_order": int(idx) + 1,
            "match_id": row["match_id"],
            "team_a": team_a,
            "team_b": team_b,
            "actual_score": actual,
            "actual_total_goals": goals_a + goals_b,
            "actual_result": actual_result,
            "actual_result_label": result_text(actual_result, team_a, team_b),
            "predicted_result": predicted_result,
            "predicted_result_label": result_text(predicted_result, team_a, team_b),
            "outcome_correct": predicted_result == actual_result,
            "actual_score_probability": actual_score_probability,
            "predicted_result_probability": float(result_probabilities[predicted_result]),
            "actual_result_probability": float(result_probabilities[actual_result]),
            "team_a_win_probability": float(result_probabilities["team_a_win"]),
            "draw_probability": float(result_probabilities["draw"]),
            "team_b_win_probability": float(result_probabilities["team_b_win"]),
            "lambda_a": float(prediction["lambda_a"]),
            "lambda_b": float(prediction["lambda_b"]),
            "outlier_scoreline": score_text(outlier) if outlier else "",
            "outlier_probability": float(outlier.get("probability", 0.0) or 0.0),
            "outlier_kind": outlier.get("kind", "") if outlier else "",
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


def draw_charts(comparison: pd.DataFrame, outdir: Path) -> dict[str, str]:
    top3_chart = comparison.copy()
    top3_chart.attrs["model_label"] = "V42 Top-3 (no score-leak inputs)"
    top3_chart.attrs["top_n"] = 3
    top3_chart.attrs["excluded_count"] = 0
    top3_chart.attrs["max_observed_goals_per_team"] = None
    top3_png = outdir / "v42_fotmob_coverage_top3_completed_worldcup_no_score_leak.png"
    chart.draw_scoreline_chart(top3_chart, top3_png)

    plus_chart = comparison.copy()
    for rank in range(1, 5):
        plus_chart[f"top_{rank}_scoreline"] = plus_chart[f"plus_{rank}_scoreline"]
        plus_chart[f"top_{rank}_probability"] = plus_chart[f"plus_{rank}_probability"]
        plus_chart[f"actual_is_top_{rank}"] = plus_chart[f"actual_is_plus_{rank}"]
    plus_chart["actual_in_top_4"] = plus_chart["actual_in_top_3_plus_outlier"]
    plus_chart.attrs["model_label"] = "V42 Top-3 + outlier (no score-leak inputs)"
    plus_chart.attrs["top_n"] = 4
    plus_chart.attrs["excluded_count"] = 0
    plus_chart.attrs["max_observed_goals_per_team"] = None
    plus_png = outdir / "v42_fotmob_coverage_top3_plus_outlier_completed_worldcup_no_score_leak.png"
    chart.draw_scoreline_chart(plus_chart, plus_png)
    return {"top3_png": str(top3_png), "top3_plus_outlier_png": str(plus_png)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate V42 completed World Cup top-3 + outlier accuracy.")
    parser.add_argument("--outdir", default="observed_eval/observed_eval_v42_fotmob_completed_worldcup_no_score_leak")
    parser.add_argument("--observed", default=str(DATA_DIR / "fotmob_completed_matches_observed_schema.csv"))
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
    parser.add_argument("--fotmob-leaders", default=str(DATA_DIR / "fotmob_full_stat_tables_clean.csv"))
    parser.add_argument("--fotmob-player-stats", default=str(DATA_DIR / "fotmob_match_player_stats_clean.csv"))
    parser.add_argument("--fotmob-lineups", default=str(DATA_DIR / "fotmob_match_lineups_clean.csv"))
    parser.add_argument("--fotmob-substitutions", default=str(DATA_DIR / "fotmob_match_substitutions_clean.csv"))
    parser.add_argument("--fotmob-keeper-stats", default=str(DATA_DIR / "fotmob_match_keeper_stats_clean.csv"))
    parser.add_argument("--fotmob-wdl-blend", type=float, default=v36.DEFAULT_FOTMOB_WDL_BLEND)
    parser.add_argument("--fotmob-scoreline-blend", type=float, default=v36.DEFAULT_FOTMOB_SCORELINE_BLEND)
    parser.add_argument("--coverage-margin", type=float, default=v42.DEFAULT_COVERAGE_MARGIN)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    observed = pd.read_csv(args.observed)
    required = {"match_id", "team_a", "team_b", "goals_a", "goals_b"}
    missing = sorted(required - set(observed.columns))
    if missing:
        raise ValueError(f"Observed file is missing columns: {missing}")

    no_score_inputs = write_no_score_inputs(outdir)
    model = build_v42_model(args, no_score_inputs)
    comparison = evaluate(model, observed)
    csv_path = outdir / "v42_fotmob_completed_worldcup_top3_comparison_no_score_leak.csv"
    comparison.to_csv(csv_path, index=False)
    plot_paths = draw_charts(comparison, outdir)

    n = int(len(comparison))
    top3_hits = int(comparison["actual_in_top_3"].sum())
    plus_hits = int(comparison["actual_in_top_3_plus_outlier"].sum())
    summary = {
        "model": "v42_fotmob_coverage",
        "evaluation_mode": "completed_worldcup_no_score_leak_inputs",
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
        "observed_answer_key_csv": str(args.observed),
        "model_score_inputs": {key: str(value) for key, value in no_score_inputs.items()},
        "comparison_csv": str(csv_path),
        **plot_paths,
    }
    summary_path = outdir / "v42_fotmob_completed_worldcup_summary_no_score_leak.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

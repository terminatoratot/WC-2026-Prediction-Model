#!/usr/bin/env python3
"""Strict no-leak V29/V36 evaluation on completed current World Cup matches.

This differs from the older no-score-leak harness:

- V29 is built with no current World Cup observed-score inputs.
- V36's base model is also built with no current World Cup observed/FotMob inputs.
- For each evaluated match, V36 receives a FotMob form profile built only from
  rows whose kickoff is earlier than the evaluated match kickoff.

So the evaluated match and future matches cannot affect that prediction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
VERSIONS_DIR = PROJECT_DIR / "versions"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(VERSIONS_DIR) not in sys.path:
    sys.path.insert(0, str(VERSIONS_DIR))

import v15_catboost_model as v15
import v29_tail_risk_scoreline_model as v29
import v36_fotmob_current_form_model as v36


DATA_DIR = PROJECT_DIR / "data"
HOSTS_2026 = {"Canada", "Mexico", "USA", "United States"}


def score_text(item: dict[str, Any]) -> str:
    return f"{int(item['team_a_goals'])}-{int(item['team_b_goals'])}"


def result_label(a: int, b: int) -> str:
    if a > b:
        return "team_a_win"
    if b > a:
        return "team_b_win"
    return "draw"


def add_rank_fields(out: dict[str, Any], prefix: str, items: list[dict[str, Any]], actual: str, n: int) -> None:
    for rank in range(1, n + 1):
        item = items[rank - 1] if rank <= len(items) else None
        scoreline = score_text(item) if item else ""
        out[f"{prefix}_{rank}_scoreline"] = scoreline
        out[f"{prefix}_{rank}_{'probability'}"] = float(item.get("probability", 0.0) or 0.0) if item else 0.0
        out[f"actual_is_{prefix}_{rank}"] = actual == scoreline


def write_empty_like(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.exists():
        pd.read_csv(source, nrows=0).to_csv(target, index=False)
    else:
        pd.DataFrame().to_csv(target, index=False)
    return target


def write_empty_model_inputs(outdir: Path) -> dict[str, Path]:
    empty = outdir / "strict_empty_model_inputs"
    empty.mkdir(parents=True, exist_ok=True)
    observed = empty / "empty_observed_matches.csv"
    pd.DataFrame(
        columns=["match_id", "date_label", "stage", "group", "team_a", "team_b", "goals_a", "goals_b", "source"]
    ).to_csv(observed, index=False)
    paths = {
        "observed": observed,
        "match_facts": write_empty_like(DATA_DIR / "fotmob_match_facts_clean.csv", empty / "empty_fotmob_match_facts.csv"),
        "goal_events": write_empty_like(DATA_DIR / "fotmob_match_goal_events_clean.csv", empty / "empty_fotmob_goal_events.csv"),
        "player_stats": write_empty_like(DATA_DIR / "fotmob_match_player_stats_clean.csv", empty / "empty_fotmob_player_stats.csv"),
        "lineups": write_empty_like(DATA_DIR / "fotmob_match_lineups_clean.csv", empty / "empty_fotmob_lineups.csv"),
        "substitutions": write_empty_like(DATA_DIR / "fotmob_match_substitutions_clean.csv", empty / "empty_fotmob_substitutions.csv"),
        "keeper_stats": write_empty_like(DATA_DIR / "fotmob_match_keeper_stats_clean.csv", empty / "empty_fotmob_keeper_stats.csv"),
    }
    return paths


def build_v29(args: argparse.Namespace, empty: dict[str, Path]) -> Any:
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
        observed_matches_csv=str(empty["observed"]),
        fotmob_leaders_csv="",
        include_observed_goals=False,
        include_fotmob_goal_stats=False,
        include_group_score_context=False,
    )
    return model


def build_v36(args: argparse.Namespace, empty: dict[str, Path]) -> Any:
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
        observed_matches_csv=str(empty["observed"]),
        fotmob_leaders_csv="",
        fotmob_player_stats_csv=str(empty["player_stats"]),
        fotmob_lineups_csv=str(empty["lineups"]),
        fotmob_substitutions_csv=str(empty["substitutions"]),
        fotmob_keeper_stats_csv=str(empty["keeper_stats"]),
        fotmob_match_facts_csv=str(empty["match_facts"]),
        fotmob_goal_events_csv=str(empty["goal_events"]),
        include_observed_goals=False,
        include_group_score_context=False,
        include_fotmob_goal_stats=False,
    )
    return model


def filter_by_kickoff(source: Path, target: Path, cutoff: pd.Timestamp) -> tuple[Path, int, str | None]:
    df = pd.read_csv(source)
    if df.empty or "kickoff" not in df.columns:
        df.iloc[0:0].to_csv(target, index=False)
        return target, 0, None
    k = pd.to_datetime(df["kickoff"], errors="coerce", utc=True)
    keep = df[k.notna() & (k < cutoff)].copy()
    keep.to_csv(target, index=False)
    max_seen = k[k < cutoff].max()
    max_text = max_seen.isoformat() if pd.notna(max_seen) else None
    return target, int(len(keep)), max_text


def prior_profile_paths(outdir: Path, cutoff: pd.Timestamp, order: int) -> tuple[dict[str, Path], dict[str, Any]]:
    pdir = outdir / "walkforward_prior_inputs" / f"match_{order:03d}"
    pdir.mkdir(parents=True, exist_ok=True)
    specs = {
        "player_stats": DATA_DIR / "fotmob_match_player_stats_clean.csv",
        "lineups": DATA_DIR / "fotmob_match_lineups_clean.csv",
        "substitutions": DATA_DIR / "fotmob_match_substitutions_clean.csv",
        "keeper_stats": DATA_DIR / "fotmob_match_keeper_stats_clean.csv",
    }
    paths = {}
    counts = {}
    max_kickoffs = {}
    for key, source in specs.items():
        target, count, max_seen = filter_by_kickoff(source, pdir / source.name, cutoff)
        paths[key] = target
        counts[key] = count
        max_kickoffs[key] = max_seen
    diagnostics = {
        "cutoff": cutoff.isoformat(),
        "row_counts": counts,
        "max_input_kickoffs": max_kickoffs,
    }
    return paths, diagnostics


def exact_probability(prediction: dict[str, Any], actual: str) -> float:
    for item in prediction.get("scoreline_probabilities", []):
        if score_text(item) == actual:
            return float(item.get("probability", 0.0) or 0.0)
    return 0.0


def prediction_row(model_name: str, order: int, match: pd.Series, prediction: dict[str, Any], leakage_diag: dict[str, Any]) -> dict[str, Any]:
    goals_a = int(match["goals_a"])
    goals_b = int(match["goals_b"])
    actual = f"{goals_a}-{goals_b}"
    result_probs = prediction["result_probabilities"]
    actual_result = result_label(goals_a, goals_b)
    predicted_result = prediction.get("predicted_result", max(result_probs, key=result_probs.get))
    top3_items = prediction.get("top_scorelines", [])[:3]
    plus_items = (prediction.get("top_scorelines_plus_outlier") or top3_items)[:4]
    top3 = {score_text(item) for item in top3_items}
    plus = {score_text(item) for item in plus_items}
    out = {
        "model": model_name,
        "observed_order": order,
        "match_id": match["match_id"],
        "date_label": match["date_label"],
        "stage": match["stage"],
        "team_a": match["team_a"],
        "team_b": match["team_b"],
        "actual_score": actual,
        "actual_total_goals": goals_a + goals_b,
        "actual_result": actual_result,
        "predicted_result": predicted_result,
        "outcome_correct": predicted_result == actual_result,
        "actual_score_probability": exact_probability(prediction, actual),
        "predicted_result_probability": float(result_probs[predicted_result]),
        "actual_result_probability": float(result_probs[actual_result]),
        "team_a_win_probability": float(result_probs["team_a_win"]),
        "draw_probability": float(result_probs["draw"]),
        "team_b_win_probability": float(result_probs["team_b_win"]),
        "lambda_a": float(prediction["lambda_a"]),
        "lambda_b": float(prediction["lambda_b"]),
        "actual_in_top_3": actual in top3,
        "actual_in_top_3_plus_outlier": actual in plus,
        "outlier_gained_hit": actual in plus and actual not in top3,
        "leakage_cutoff": leakage_diag.get("cutoff", ""),
        "max_input_kickoffs_json": json.dumps(leakage_diag.get("max_input_kickoffs", {}), sort_keys=True),
        "prior_row_counts_json": json.dumps(leakage_diag.get("row_counts", {}), sort_keys=True),
    }
    add_rank_fields(out, "top", top3_items, actual, 3)
    add_rank_fields(out, "plus", plus_items, actual, 4)
    return out


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, g in frame.groupby("model"):
        n = len(g)
        rows.append(
            {
                "model": model,
                "matches": n,
                "outcome_accuracy": float(g["outcome_correct"].mean()) if n else 0.0,
                "top1_exact_hits": int(g["actual_is_top_1"].sum()),
                "top2_exact_hits": int((g["actual_is_top_1"] | g["actual_is_top_2"]).sum()),
                "top3_exact_hits": int(g["actual_in_top_3"].sum()),
                "top3_exact_accuracy": float(g["actual_in_top_3"].mean()) if n else 0.0,
                "top3_plus_outlier_hits": int(g["actual_in_top_3_plus_outlier"].sum()),
                "top3_plus_outlier_accuracy": float(g["actual_in_top_3_plus_outlier"].mean()) if n else 0.0,
                "outlier_gained_hits": int(g["outlier_gained_hit"].sum()),
                "mean_actual_score_probability": float(g["actual_score_probability"].mean()) if n else 0.0,
                "mean_lambda_sum": float((g["lambda_a"] + g["lambda_b"]).mean()) if n else 0.0,
                "mean_actual_total": float(g["actual_total_goals"].mean()) if n else 0.0,
            }
        )
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, outdir: Path) -> None:
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("outcome_accuracy", "Outcome"),
        ("top3_exact_accuracy", "Top-3 exact"),
        ("top3_plus_outlier_accuracy", "Top-3+outlier"),
    ]
    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
    colors = {"v29_tail_risk": "#94A3B8", "v36_fotmob_walkforward": "#2563EB"}
    for idx, (_, row) in enumerate(summary.iterrows()):
        vals = [float(row[col]) for col, _ in metrics]
        ax.bar(x + (idx - 0.5) * width, vals, width=width, label=row["model"], color=colors.get(row["model"], "#64748B"))
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylim(0, max(0.55, float(summary[[m[0] for m in metrics]].max().max()) + 0.08))
    ax.yaxis.set_major_formatter(lambda v, pos: f"{v:.0%}")
    ax.set_title("Strict No-Leak V29 vs V36 Current World Cup Evaluation", fontweight="bold", loc="left")
    ax.set_ylabel("Hit rate")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_dir / "v29_v36_strict_no_leak_accuracy.png", dpi=180)
    plt.close(fig)


def plot_cumulative(frame: pd.DataFrame, outdir: Path) -> None:
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor="white")
    colors = {"v29_tail_risk": "#64748B", "v36_fotmob_walkforward": "#2563EB"}
    for model, g in frame.sort_values("observed_order").groupby("model"):
        y = g["actual_in_top_3"].astype(int).cumsum() / np.arange(1, len(g) + 1)
        ax.plot(g["observed_order"], y, linewidth=2.5, label=model, color=colors.get(model, None))
    ax.yaxis.set_major_formatter(lambda v, pos: f"{v:.0%}")
    ax.set_title("Cumulative Top-3 Exact Score Accuracy", fontweight="bold", loc="left")
    ax.set_xlabel("Observed match order")
    ax.set_ylabel("Cumulative hit rate")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_dir / "v29_v36_strict_no_leak_cumulative_top3.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="observed_eval/observed_eval_v29_v36_walkforward_no_leak_latest")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    empty = write_empty_model_inputs(outdir)
    observed = pd.read_csv(args.observed)
    observed["kickoff_dt"] = pd.to_datetime(observed["date_label"], errors="coerce", utc=True)
    observed = observed.sort_values(["kickoff_dt", "match_id"]).reset_index(drop=True)
    if observed["kickoff_dt"].isna().any():
        raise ValueError("Some observed rows have invalid date_label values.")

    v29_model = build_v29(args, empty)
    v36_model = build_v36(args, empty)

    rows: list[dict[str, Any]] = []
    for idx, match in observed.iterrows():
        order = idx + 1
        knockout = str(match.get("stage", "")).strip().lower() != "group stage"
        common = {
            "host_a": str(match["team_a"]) in HOSTS_2026,
            "host_b": str(match["team_b"]) in HOSTS_2026,
            "knockout": knockout,
        }
        pred29 = v29_model.predict(match["team_a"], match["team_b"], **common)
        rows.append(
            prediction_row(
                "v29_tail_risk",
                order,
                match,
                pred29,
                {"cutoff": match["kickoff_dt"].isoformat(), "row_counts": {}, "max_input_kickoffs": {}},
            )
        )

        prior_paths, diag = prior_profile_paths(outdir, match["kickoff_dt"], order)
        profiles = v36.build_fotmob_team_form_profiles(
            prior_paths["player_stats"],
            prior_paths["lineups"],
            prior_paths["substitutions"],
            prior_paths["keeper_stats"],
            prior_matches=v36.DEFAULT_PROFILE_PRIOR_MATCHES,
        )
        v36_model.team_form_profiles = profiles
        pred36 = v36_model.predict(match["team_a"], match["team_b"], **common)
        rows.append(prediction_row("v36_fotmob_walkforward", order, match, pred36, diag))

    comparison = pd.DataFrame(rows)
    summary = summarize(comparison)
    comparison_csv = outdir / "v29_v36_strict_no_leak_comparison.csv"
    summary_csv = outdir / "v29_v36_strict_no_leak_summary.csv"
    summary_json = outdir / "v29_v36_strict_no_leak_summary.json"
    comparison.to_csv(comparison_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    plot_summary(summary, outdir)
    plot_cumulative(comparison, outdir)

    payload = {
        "evaluation_mode": "strict_walkforward_no_current_match_or_future_fotmob_rows",
        "observed_rows": int(len(observed)),
        "comparison_csv": str(comparison_csv),
        "summary_csv": str(summary_csv),
        "plots": [
            str(outdir / "plots" / "v29_v36_strict_no_leak_accuracy.png"),
            str(outdir / "plots" / "v29_v36_strict_no_leak_cumulative_top3.png"),
        ],
        "summary": summary.to_dict(orient="records"),
        "leakage_controls": {
            "v29_current_wc_observed_inputs": "empty",
            "v36_base_current_wc_observed_and_fotmob_inputs": "empty",
            "v36_form_profile_rule": "only FotMob rows with kickoff < evaluated match kickoff",
            "answer_key_only": args.observed,
        },
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

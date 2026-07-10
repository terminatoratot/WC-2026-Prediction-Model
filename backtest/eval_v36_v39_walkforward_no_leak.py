#!/usr/bin/env python3
"""Walk-forward V36/V39 evaluation with no current-match score leakage."""

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

import market_edge  # noqa: F401  (bundles v11/v13-v27/v28-v38/v39; populates sys.modules)
import eval_v29_v36_walkforward_no_leak as strict
import v15_catboost_model as v15
import v36_fotmob_current_form_model as v36
import v39_withbetterdata as v39bd


DATA_DIR = PROJECT_DIR / "data"
HOSTS_2026 = {"Canada", "Mexico", "USA", "United States"}


def build_v39(args: argparse.Namespace, empty: dict[str, Path], empty_profile: Path) -> Any:
    model, _ = v39bd.build_betterdata_model_from_zip(
        args.worldcupsai_zip,
        betterdata_profile_csv=empty_profile,
        betterdata_scoreline_blend=args.betterdata_scoreline_blend,
        betterdata_wdl_blend=args.betterdata_wdl_blend,
        betterdata_max_log_adjustment=args.betterdata_max_log_adjustment,
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
        use_observed_game_state_priors=False,
    )
    return model


def prior_betterdata_profile(team_frame: pd.DataFrame, cutoff: pd.Timestamp, out_path: Path) -> tuple[Path, dict[str, Any]]:
    frame = team_frame.copy()
    frame["kickoff_dt"] = pd.to_datetime(frame["db_match_date"], errors="coerce", utc=True)
    prior = frame[frame["kickoff_dt"].notna() & (frame["kickoff_dt"] < cutoff)].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if prior.empty:
        pd.DataFrame(columns=["team", "matches"]).to_csv(out_path, index=False)
        return out_path, {"betterdata_prior_rows": 0, "betterdata_prior_matches": 0}

    grouped = prior.groupby("team", as_index=False)
    profile = grouped.agg(
        matches=("match_id", "nunique"),
        goals_for=("goals_for", "sum"),
        goals_against=("goals_against", "sum"),
        xg_for=("xg_for", "sum"),
        xg_against=("xg_against", "sum"),
        db_xg_for_ft=("db_xg_for_ft", "sum"),
        db_xg_against_ft=("db_xg_against_ft", "sum"),
        db_sot_for_ft=("db_sot_for_ft", "sum"),
        db_shots_for_ft=("db_shots_for_ft", "sum"),
        db_corners_for_ft=("db_corners_for_ft", "sum"),
        db_corners_against_ft=("db_corners_against_ft", "sum"),
        db_yellow_cards_against_ft=("db_yellow_cards_against_ft", "sum"),
        avg_db_team_win_prob=("db_team_win_prob", "mean"),
        avg_match_over25_prob=("db_implied_over25_prob", "mean"),
        avg_set_piece_pressure=("set_piece_pressure_index", "mean"),
        avg_instability=("discipline_instability_index", "mean"),
        avg_second_half_net_surge=("second_half_net_surge_index", "mean"),
    )
    profile["goal_diff"] = profile["goals_for"] - profile["goals_against"]
    profile["fotmob_xg_diff"] = profile["xg_for"] - profile["xg_against"]
    profile["db_xg_diff"] = profile["db_xg_for_ft"] - profile["db_xg_against_ft"]
    profile["db_corner_diff"] = profile["db_corners_for_ft"] - profile["db_corners_against_ft"]
    profile["db_sot_rate"] = profile["db_sot_for_ft"] / profile["db_shots_for_ft"].replace(0, np.nan)
    profile["betterdata_attack_index"] = (
        v39bd.zscore(profile["db_xg_diff"].fillna(0))
        + 0.55 * v39bd.zscore(profile["db_sot_rate"].fillna(0))
        + 0.45 * v39bd.zscore(profile["avg_set_piece_pressure"].fillna(0))
        + 0.35 * v39bd.zscore(profile["avg_second_half_net_surge"].fillna(0))
    )
    profile["betterdata_risk_index"] = (
        v39bd.zscore(-profile["goal_diff"].fillna(0))
        + 0.50 * v39bd.zscore(profile["avg_instability"].fillna(0))
        + 0.40 * v39bd.zscore(-profile["db_corner_diff"].fillna(0))
        + 0.35 * v39bd.zscore(profile["db_yellow_cards_against_ft"].fillna(0))
    )
    profile["style_cluster"] = "walkforward_prior"
    profile.to_csv(out_path, index=False)
    return out_path, {
        "betterdata_prior_rows": int(len(prior)),
        "betterdata_prior_matches": int(prior["match_id"].nunique()),
        "betterdata_profile_teams": int(profile["team"].nunique()),
        "betterdata_max_input_kickoff": prior["kickoff_dt"].max().isoformat(),
    }


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, g in frame.groupby("model"):
        n = len(g)
        rows.append({
            "model": model,
            "matches": n,
            "outcome_accuracy": float(g["outcome_correct"].mean()),
            "top1_hits": int(g["actual_is_top_1"].sum()),
            "top3_hits": int(g["actual_in_top_3"].sum()),
            "top3_accuracy": float(g["actual_in_top_3"].mean()),
            "top3_plus_outlier_hits": int(g["actual_in_top_3_plus_outlier"].sum()),
            "top3_plus_outlier_accuracy": float(g["actual_in_top_3_plus_outlier"].mean()),
            "outlier_gained_hits": int(g["outlier_gained_hit"].sum()),
            "mean_actual_score_probability": float(g["actual_score_probability"].mean()),
        })
    return pd.DataFrame(rows)


def plot_accuracy(summary: pd.DataFrame, outdir: Path) -> None:
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    metrics = [("outcome_accuracy", "Outcome"), ("top3_accuracy", "Top 3"), ("top3_plus_outlier_accuracy", "Top 3 + outlier")]
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
    for idx, (_, row) in enumerate(summary.iterrows()):
        ax.bar(x + (idx - 0.5) * width, [row[m] for m, _ in metrics], width, label=row["model"])
    ax.set_xticks(x, [label for _, label in metrics])
    ax.yaxis.set_major_formatter(lambda v, pos: f"{v:.0%}")
    ax.set_ylim(0, max(0.55, float(summary[[m for m, _ in metrics]].max().max()) + 0.08))
    ax.set_title("V36 vs V39 no-leak walk-forward accuracy", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_dir / "v36_v39_no_leak_accuracy.png", dpi=180)
    plt.close(fig)


def plot_cumulative(frame: pd.DataFrame, outdir: Path, column: str, filename: str, title: str) -> None:
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    for model, g in frame.sort_values("observed_order").groupby("model"):
        y = g[column].astype(int).cumsum() / np.arange(1, len(g) + 1)
        ax.plot(g["observed_order"], y, linewidth=2.2, label=model)
    ax.yaxis.set_major_formatter(lambda v, pos: f"{v:.0%}")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Match order")
    ax.set_ylabel("Cumulative hit rate")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_dir / filename, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="observed_eval/observed_eval_v36_v39_walkforward_no_leak_latest")
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
    parser.add_argument("--betterdata-frame", default="analysis/v39_withbetterdata_latest/v39_withbetterdata_team_match_model_frame.csv")
    parser.add_argument("--betterdata-scoreline-blend", type=float, default=v39bd.DEFAULT_BETTERDATA_SCORELINE_BLEND)
    parser.add_argument("--betterdata-wdl-blend", type=float, default=v39bd.DEFAULT_BETTERDATA_WDL_BLEND)
    parser.add_argument("--betterdata-max-log-adjustment", type=float, default=v39bd.DEFAULT_BETTERDATA_MAX_LOG_ADJUSTMENT)
    parser.add_argument("--start-order", type=int, default=1)
    parser.add_argument("--end-order", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    empty = strict.write_empty_model_inputs(outdir)
    observed_path = v36.completed_fotmob_facts_to_observed(
        DATA_DIR / "fotmob_match_facts_clean.csv",
        outdir / "fotmob_completed_matches_observed_schema_current.csv",
    )
    observed = pd.read_csv(observed_path)
    observed["kickoff_dt"] = pd.to_datetime(observed["date_label"], errors="coerce", utc=True)
    observed = observed[observed["kickoff_dt"].notna()].sort_values(["kickoff_dt", "match_id"]).reset_index(drop=True)
    observed["observed_order"] = np.arange(1, len(observed) + 1)
    if args.start_order > 1:
        observed = observed[observed["observed_order"] >= args.start_order].copy()
    if args.end_order > 0:
        observed = observed[observed["observed_order"] <= args.end_order].copy()

    empty_profile = outdir / "strict_empty_model_inputs" / "empty_betterdata_profiles.csv"
    pd.DataFrame(columns=["team", "matches"]).to_csv(empty_profile, index=False)
    v36_model = strict.build_v36(args, empty)
    v39_model = build_v39(args, empty, empty_profile)
    better_frame = pd.read_csv(args.betterdata_frame)

    rows: list[dict[str, Any]] = []
    for _, match in observed.iterrows():
        order = int(match["observed_order"])
        knockout = str(match.get("stage", "")).strip().lower() != "group stage"
        common = {
            "host_a": str(match["team_a"]) in HOSTS_2026,
            "host_b": str(match["team_b"]) in HOSTS_2026,
            "knockout": knockout,
        }
        prior_paths, diag36 = strict.prior_profile_paths(outdir, match["kickoff_dt"], order)
        v36_model.team_form_profiles = v36.build_fotmob_team_form_profiles(
            prior_paths["player_stats"],
            prior_paths["lineups"],
            prior_paths["substitutions"],
            prior_paths["keeper_stats"],
            prior_matches=v36.DEFAULT_PROFILE_PRIOR_MATCHES,
        )
        pred36 = v36_model.predict(match["team_a"], match["team_b"], **common)
        rows.append(strict.prediction_row("v36_walkforward", order, match, pred36, diag36))

        profile_path, diag39 = prior_betterdata_profile(
            better_frame,
            match["kickoff_dt"],
            outdir / "walkforward_betterdata_profiles" / f"match_{order:03d}_profiles.csv",
        )
        v39_model.better_priors = v39bd.BetterDataPriors(profile_path)
        pred39 = v39_model.predict(match["team_a"], match["team_b"], **common)
        diag = {**diag36, **diag39, "cutoff": match["kickoff_dt"].isoformat()}
        rows.append(strict.prediction_row("v39_betterdata_walkforward", order, match, pred39, diag))

    comparison = pd.DataFrame(rows)
    summary = summarize(comparison)
    comparison_csv = outdir / "v36_v39_no_leak_comparison.csv"
    summary_csv = outdir / "v36_v39_no_leak_summary.csv"
    summary_json = outdir / "v36_v39_no_leak_summary.json"
    comparison.to_csv(comparison_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    plot_accuracy(summary, outdir)
    plot_cumulative(comparison, outdir, "actual_in_top_3", "v36_v39_no_leak_cumulative_top3.png", "Cumulative top-3 exact score hit rate")
    plot_cumulative(comparison, outdir, "actual_in_top_3_plus_outlier", "v36_v39_no_leak_cumulative_top3_plus_outlier.png", "Cumulative top-3 + outlier exact score hit rate")
    payload = {
        "mode": "walkforward_no_current_or_future_match_inputs",
        "matches": int(observed["match_id"].nunique()),
        "comparison_csv": str(comparison_csv),
        "summary_csv": str(summary_csv),
        "summary": summary.to_dict(orient="records"),
        "plots": [
            str(outdir / "plots" / "v36_v39_no_leak_accuracy.png"),
            str(outdir / "plots" / "v36_v39_no_leak_cumulative_top3.png"),
            str(outdir / "plots" / "v36_v39_no_leak_cumulative_top3_plus_outlier.png"),
        ],
        "leakage_controls": {
            "answer_key": str(observed_path),
            "base_model_current_wc_observed_inputs": "empty",
            "base_model_fotmob_match_facts_goal_events": "empty",
            "v36_profile_rows": "kickoff < evaluated kickoff only",
            "v39_betterdata_profiles": "quantitative/stat rows with db_match_date < evaluated kickoff only",
            "v39_observed_game_state_priors": "disabled",
        },
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

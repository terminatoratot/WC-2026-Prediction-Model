#!/usr/bin/env python3
"""Common-denominator audit of the base scoreline models.

Evaluates v11, v15, v29, v36, v39_betterdata, and the v42 production stack on
the SAME completed WC2026 matches with proper scoring rules (exact-score
log-loss, result log-loss, Brier, RPS), the legacy top-3 coverage metrics for
continuity, calibration diagnostics, and a Polymarket prematch benchmark.

Leak controls follow the strictest existing harnesses:
- v11/v15/v29 are pre-tournament models: built once with empty current-WC
  observed inputs (results_as_of already excludes 2026 WC rows).
- v36/v39/v42 get per-match walk-forward form profiles (only rows with
  kickoff strictly before the evaluated match), reusing
  eval_v29_v36_walkforward_no_leak.py and eval_v36_v39_walkforward_no_leak.py.
- The v42 row reproduces the production stack recipe from
  v42_fotmob_market_edge_model.py main(): v36 base (fotmob_wdl_blend=0.20)
  -> V39CoverageOutlierModel (game-state priors on) -> V39BetterDataModel
  (wdl_blend=0.25), with both form layers walk-forward.

Usage:
    MPLCONFIGDIR=.matplotlib_cache LOKY_MAX_CPU_COUNT=1 \
    .venv/bin/python audit_base_models.py --outdir outputs/base_model_audit
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
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
import eval_v36_v39_walkforward_no_leak as wf39
import v11_wcq_results_model as v11
import v15_catboost_model as v15
import v36_fotmob_current_form_model as v36
import v39_coverage_outlier_model as v39co
import v39_withbetterdata as v39bd

DATA_DIR = PROJECT_DIR / "data"
HOSTS_2026 = {"Canada", "Mexico", "USA", "United States"}
PROB_FLOOR = 1e-4

MODEL_ORDER = ["v11", "v15", "v29", "v36", "v39_betterdata", "v42_production", "market"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="outputs/base_model_audit")
    parser.add_argument("--observed", default=str(DATA_DIR / "fotmob_completed_matches_observed_schema_current.csv"))
    parser.add_argument("--market-snapshots-dir", default=str(DATA_DIR / "polymarket_prematch_snapshots_from_fotmob"))
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
    parser.add_argument("--v42-fotmob-wdl-blend", type=float, default=0.20)
    parser.add_argument("--v42-fotmob-scoreline-blend", type=float, default=0.15)
    parser.add_argument("--skip-models", default="", help="Comma-separated model names to skip.")
    parser.add_argument("--max-matches", type=int, default=0)
    return parser.parse_args()


# -----------------------------
# Model construction
# -----------------------------

def pre_tournament_results_csv(results_path: str, outdir: Path) -> Path:
    """v11.build_from_zip has NO results_as_of filter, and data/results.csv
    contains 2026 WC finals rows (the answer key). Feed v11 a copy truncated
    strictly before the tournament start so it stays a pre-tournament model."""
    target = outdir / "results_pre_tournament.csv"
    if not target.exists():
        df = pd.read_csv(results_path)
        dates = pd.to_datetime(df["date"], errors="coerce")
        df[dates < pd.Timestamp("2026-06-11")].to_csv(target, index=False)
    return target


def build_static_v11(args: argparse.Namespace) -> Any:
    model, _ = v11.build_from_zip(
        args.worldcupsai_zip,
        train_csv=args.team_train,
        test_csv=args.team_test,
        model_type="ensemble",
        box_csv=args.box_data,
        results_csv=str(pre_tournament_results_csv(args.results_data, Path(args.outdir))),
        former_names_csv=args.former_names,
        prediction_year=args.prediction_year,
    )
    return model


def build_static_v15(args: argparse.Namespace) -> Any:
    model, _ = v15.build_from_zip(
        args.worldcupsai_zip,
        train_csv=args.team_train,
        test_csv=args.team_test,
        box_csv=args.box_data,
        results_csv=args.results_data,
        former_names_csv=args.former_names,
        prediction_year=args.prediction_year,
        player_ratings_csv=args.player_ratings,
        declared_squads_csv=args.declared_squads,
        results_as_of=args.results_as_of,
    )
    return model


def build_v42_production(args: argparse.Namespace, empty: dict[str, Path], empty_profile: Path) -> Any:
    """Reproduce v42_fotmob_market_edge_model.py's model stack, leak-controlled."""
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
        fotmob_wdl_blend=args.v42_fotmob_wdl_blend,
        fotmob_scoreline_blend=args.v42_fotmob_scoreline_blend,
    )
    v39_base = v39co.V39CoverageOutlierModel(
        base_model,
        coverage_margin=v39co.DEFAULT_COVERAGE_MARGIN,
        observed_game_state_priors=v39co.default_observed_game_state_priors(),
        use_observed_game_state_priors=True,
    )
    model = v39bd.V39BetterDataModel(
        v39_base,
        better_priors=v39bd.BetterDataPriors(empty_profile),
        betterdata_scoreline_blend=v39bd.DEFAULT_BETTERDATA_SCORELINE_BLEND,
        betterdata_wdl_blend=v39bd.DEFAULT_BETTERDATA_WDL_BLEND,
        max_abs_log_adjustment=v39bd.DEFAULT_BETTERDATA_MAX_LOG_ADJUSTMENT,
    )
    return model, base_model


# -----------------------------
# Prediction capture
# -----------------------------

def score_text(item: dict[str, Any]) -> str:
    return f"{int(item['team_a_goals'])}-{int(item['team_b_goals'])}"


def capture_prediction(model_name: str, match: pd.Series, order: int, prediction: dict[str, Any]) -> dict[str, Any]:
    """Flatten one prediction into a cacheable record with the full matrix."""
    matrix = {
        score_text(item): float(item.get("probability", 0.0) or 0.0)
        for item in prediction.get("scoreline_probabilities", [])
    }
    result_probs = prediction["result_probabilities"]
    top3 = [score_text(x) for x in prediction.get("top_scorelines", [])[:3]]
    plus = [score_text(x) for x in (prediction.get("top_scorelines_plus_outlier") or prediction.get("top_scorelines", []))[:4]]
    return {
        "model": model_name,
        "observed_order": order,
        "match_id": str(match["match_id"]),
        "kickoff": match["kickoff_dt"].isoformat(),
        "team_a": match["team_a"],
        "team_b": match["team_b"],
        "actual_score": f"{int(match['goals_a'])}-{int(match['goals_b'])}",
        "stage": match.get("stage", ""),
        "team_a_win_probability": float(result_probs["team_a_win"]),
        "draw_probability": float(result_probs["draw"]),
        "team_b_win_probability": float(result_probs["team_b_win"]),
        "predicted_result": prediction.get("predicted_result", max(result_probs, key=result_probs.get)),
        "lambda_a": float(prediction.get("lambda_a", 0.0) or 0.0),
        "lambda_b": float(prediction.get("lambda_b", 0.0) or 0.0),
        "top3_json": json.dumps(top3),
        "plus_json": json.dumps(plus),
        "matrix_json": json.dumps(matrix),
    }


def load_market_predictions(snapshots_dir: Path, observed: pd.DataFrame) -> list[dict[str, Any]]:
    """Polymarket prematch quotes -> normalized pseudo-model rows."""
    rows: list[dict[str, Any]] = []
    for _, match in observed.iterrows():
        f = snapshots_dir / str(match["match_id"]) / "prematch_prices.csv"
        if not f.exists():
            continue
        quotes = pd.read_csv(f)
        quotes = quotes[quotes["yes_price"].notna() & (quotes["yes_price"] > 0)]
        if quotes.empty:
            continue
        raw = dict(zip(quotes["scoreline"].astype(str), quotes["yes_price"].astype(float)))
        total = sum(raw.values())
        other_mass = max(0.0, 1.0 - total)
        denom = total + other_mass
        matrix = {k: v / denom for k, v in raw.items()}

        result_sums = {"team_a_win": 0.0, "draw": 0.0, "team_b_win": 0.0}
        for scoreline, p in matrix.items():
            a, b = scoreline.split("-")
            a, b = int(a), int(b)
            key = "team_a_win" if a > b else ("team_b_win" if b > a else "draw")
            result_sums[key] += p
        quoted_result_total = sum(result_sums.values())
        if quoted_result_total > 0:
            result_probs = {k: v / quoted_result_total for k, v in result_sums.items()}
        else:
            result_probs = {k: 1 / 3 for k in result_sums}

        top = sorted(matrix, key=matrix.get, reverse=True)
        rows.append(
            {
                "model": "market",
                "observed_order": int(match["observed_order"]),
                "match_id": str(match["match_id"]),
                "kickoff": match["kickoff_dt"].isoformat(),
                "team_a": match["team_a"],
                "team_b": match["team_b"],
                "actual_score": f"{int(match['goals_a'])}-{int(match['goals_b'])}",
                "stage": match.get("stage", ""),
                "team_a_win_probability": result_probs["team_a_win"],
                "draw_probability": result_probs["draw"],
                "team_b_win_probability": result_probs["team_b_win"],
                "predicted_result": max(result_probs, key=result_probs.get),
                "lambda_a": 0.0,
                "lambda_b": 0.0,
                "top3_json": json.dumps(top[:3]),
                "plus_json": json.dumps(top[:4]),
                "matrix_json": json.dumps(matrix),
                "market_other_mass": other_mass,
            }
        )
    return rows


# -----------------------------
# Metrics
# -----------------------------

def result_label(a: int, b: int) -> str:
    return "team_a_win" if a > b else ("team_b_win" if b > a else "draw")


def is_clean_sheet(scoreline: str) -> bool:
    a, b = scoreline.split("-")
    return int(a) == 0 or int(b) == 0


def per_match_metrics(record: dict[str, Any]) -> dict[str, Any]:
    matrix = json.loads(record["matrix_json"])
    actual = record["actual_score"]
    a, b = (int(x) for x in actual.split("-"))
    actual_result = result_label(a, b)
    p_result = {
        "team_a_win": record["team_a_win_probability"],
        "draw": record["draw_probability"],
        "team_b_win": record["team_b_win_probability"],
    }
    top3 = set(json.loads(record["top3_json"]))
    plus = set(json.loads(record["plus_json"]))

    p_exact = max(matrix.get(actual, 0.0), PROB_FLOOR)
    p_act_result = max(p_result[actual_result], PROB_FLOOR)

    # Brier over three outcomes; RPS with ordering (A win, draw, B win)
    outcomes = ["team_a_win", "draw", "team_b_win"]
    onehot = np.array([1.0 if o == actual_result else 0.0 for o in outcomes])
    probs = np.array([p_result[o] for o in outcomes])
    brier = float(np.sum((probs - onehot) ** 2))
    rps = float(np.sum((np.cumsum(probs) - np.cumsum(onehot))[:-1] ** 2) / (len(outcomes) - 1))

    pred_total = sum(p * (int(s.split("-")[0]) + int(s.split("-")[1])) for s, p in matrix.items())
    pred_clean_sheet = sum(p for s, p in matrix.items() if is_clean_sheet(s))
    top_pick, top_pick_prob = ("", 0.0)
    if matrix:
        top_pick = max(matrix, key=matrix.get)
        top_pick_prob = matrix[top_pick]

    return {
        "model": record["model"],
        "match_id": record["match_id"],
        "observed_order": record["observed_order"],
        "stage": record["stage"],
        "actual_score": actual,
        "actual_result": actual_result,
        "actual_total": a + b,
        "actual_clean_sheet": is_clean_sheet(actual),
        "exact_log_loss": -np.log(p_exact),
        "actual_in_matrix": actual in matrix,
        "actual_score_probability": matrix.get(actual, 0.0),
        "result_log_loss": -np.log(p_act_result),
        "brier": brier,
        "rps": rps,
        "outcome_correct": record["predicted_result"] == actual_result,
        "actual_in_top3": actual in top3,
        "actual_in_plus": actual in plus,
        "top_pick_scoreline": top_pick,
        "top_pick_probability": top_pick_prob,
        "top_pick_hit": top_pick == actual,
        "predicted_total_goals": pred_total,
        "predicted_clean_sheet_prob": pred_clean_sheet,
        "team_a_win_probability": p_result["team_a_win"],
        "draw_probability": p_result["draw"],
        "team_b_win_probability": p_result["team_b_win"],
        "actual_result_probability": p_result[actual_result],
        "matrix_mass": sum(matrix.values()),
    }


def bootstrap_delta_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 4000, seed: int = 7) -> tuple[float, float, float]:
    """CI for mean(a - b) over paired matches."""
    rng = np.random.default_rng(seed)
    d = a - b
    means = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return float(d.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize_models(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, g in metrics.groupby("model"):
        rows.append(
            {
                "model": model,
                "matches": len(g),
                "exact_log_loss": g["exact_log_loss"].mean(),
                "result_log_loss": g["result_log_loss"].mean(),
                "brier": g["brier"].mean(),
                "rps": g["rps"].mean(),
                "outcome_accuracy": g["outcome_correct"].mean(),
                "top3_accuracy": g["actual_in_top3"].mean(),
                "top3_plus_outlier_accuracy": g["actual_in_plus"].mean(),
                "top_pick_hit_rate": g["top_pick_hit"].mean(),
                "mean_actual_score_prob": g["actual_score_probability"].mean(),
                "mean_predicted_total": g["predicted_total_goals"].mean(),
                "mean_actual_total": g["actual_total"].mean(),
                "predicted_clean_sheet_rate": g["predicted_clean_sheet_prob"].mean(),
                "actual_clean_sheet_rate": g["actual_clean_sheet"].mean(),
                "actual_missing_from_matrix": int((~g["actual_in_matrix"]).sum()),
            }
        )
    df = pd.DataFrame(rows)
    df["order"] = df["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    return df.sort_values("order").drop(columns="order").reset_index(drop=True)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    pred_dir = outdir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    skip = {s.strip() for s in args.skip_models.split(",") if s.strip()}

    observed = pd.read_csv(args.observed)
    observed["kickoff_dt"] = pd.to_datetime(observed["date_label"], errors="coerce", utc=True)
    observed = observed[observed["kickoff_dt"].notna()].sort_values(["kickoff_dt", "match_id"]).reset_index(drop=True)
    observed["observed_order"] = np.arange(1, len(observed) + 1)
    if args.max_matches > 0:
        observed = observed.head(args.max_matches).copy()
    print(f"[audit] {len(observed)} matches | {observed['kickoff_dt'].min()} .. {observed['kickoff_dt'].max()}", file=sys.stderr)

    empty = strict.write_empty_model_inputs(outdir)
    empty_profile = outdir / "strict_empty_model_inputs" / "empty_betterdata_profiles.csv"
    pd.DataFrame(columns=["team", "matches"]).to_csv(empty_profile, index=False)

    all_records: list[dict[str, Any]] = []

    def cache_path(name: str) -> Path:
        return pred_dir / f"{name}.csv"

    def run_static(name: str, build_fn) -> None:
        if name in skip:
            return
        if cache_path(name).exists():
            print(f"[audit] {name}: using cached predictions", file=sys.stderr)
            all_records.extend(pd.read_csv(cache_path(name)).to_dict("records"))
            return
        print(f"[audit] building {name} ...", file=sys.stderr)
        try:
            model = build_fn(args)
        except Exception:
            print(f"[audit] ERROR building {name}:\n{traceback.format_exc()}", file=sys.stderr)
            return
        records = []
        for _, match in observed.iterrows():
            common = {
                "host_a": str(match["team_a"]) in HOSTS_2026,
                "host_b": str(match["team_b"]) in HOSTS_2026,
                "knockout": str(match.get("stage", "")).strip().lower() != "group stage",
            }
            try:
                pred = model.predict(match["team_a"], match["team_b"], **common)
                records.append(capture_prediction(name, match, int(match["observed_order"]), pred))
            except Exception:
                print(f"[audit] ERROR {name} predict {match['team_a']} vs {match['team_b']}:\n{traceback.format_exc()}", file=sys.stderr)
            if int(match["observed_order"]) % 10 == 0:
                print(f"[audit] {name}: {int(match['observed_order'])}/{len(observed)} matches", file=sys.stderr)
        pd.DataFrame(records).to_csv(cache_path(name), index=False)
        all_records.extend(records)
        print(f"[audit] {name}: DONE ({len(records)} predictions)", file=sys.stderr)

    run_static("v11", build_static_v11)
    run_static("v15", build_static_v15)
    run_static("v29", lambda a: strict.build_v29(a, empty))

    # ---- walk-forward models: v36, v39_betterdata, v42_production ----
    wf_models: dict[str, Any] = {}
    if "v36" not in skip and not cache_path("v36").exists():
        print("[audit] building v36 (walk-forward base) ...", file=sys.stderr)
        wf_models["v36"] = strict.build_v36(args, empty)
    if "v39_betterdata" not in skip and not cache_path("v39_betterdata").exists():
        print("[audit] building v39_betterdata (walk-forward base) ...", file=sys.stderr)
        wf_models["v39_betterdata"] = wf39.build_v39(args, empty, empty_profile)
    v42_base = None
    if "v42_production" not in skip and not cache_path("v42_production").exists():
        print("[audit] building v42_production stack ...", file=sys.stderr)
        wf_models["v42_production"], v42_base = build_v42_production(args, empty, empty_profile)

    for name in ("v36", "v39_betterdata", "v42_production"):
        if name in skip or name not in wf_models:
            if cache_path(name).exists() and name not in skip:
                print(f"[audit] {name}: using cached predictions", file=sys.stderr)
                all_records.extend(pd.read_csv(cache_path(name)).to_dict("records"))
            continue

    better_frame = None
    if any(n in wf_models for n in ("v39_betterdata", "v42_production")):
        better_frame = pd.read_csv(args.betterdata_frame)

    if wf_models:
        wf_records: dict[str, list[dict[str, Any]]] = {name: [] for name in wf_models}
        for _, match in observed.iterrows():
            order = int(match["observed_order"])
            common = {
                "host_a": str(match["team_a"]) in HOSTS_2026,
                "host_b": str(match["team_b"]) in HOSTS_2026,
                "knockout": str(match.get("stage", "")).strip().lower() != "group stage",
            }
            prior_paths, _diag = strict.prior_profile_paths(outdir, match["kickoff_dt"], order)
            profiles = v36.build_fotmob_team_form_profiles(
                prior_paths["player_stats"],
                prior_paths["lineups"],
                prior_paths["substitutions"],
                prior_paths["keeper_stats"],
                prior_matches=v36.DEFAULT_PROFILE_PRIOR_MATCHES,
            )
            profile_path = None
            if better_frame is not None:
                profile_path, _d39 = wf39.prior_betterdata_profile(
                    better_frame,
                    match["kickoff_dt"],
                    outdir / "walkforward_betterdata_profiles" / f"match_{order:03d}_profiles.csv",
                )

            for name, model in wf_models.items():
                try:
                    if name == "v36":
                        model.team_form_profiles = profiles
                    elif name == "v39_betterdata":
                        model.team_form_profiles = profiles
                        model.better_priors = v39bd.BetterDataPriors(profile_path)
                    elif name == "v42_production":
                        v42_base.team_form_profiles = profiles
                        model.better_priors = v39bd.BetterDataPriors(profile_path)
                    pred = model.predict(match["team_a"], match["team_b"], **common)
                    wf_records[name].append(capture_prediction(name, match, order, pred))
                except Exception:
                    print(f"[audit] ERROR {name} predict {match['team_a']} vs {match['team_b']}:\n{traceback.format_exc()}", file=sys.stderr)
            if order % 10 == 0:
                print(f"[audit] walk-forward: {order}/{len(observed)} matches", file=sys.stderr)

        for name, records in wf_records.items():
            pd.DataFrame(records).to_csv(cache_path(name), index=False)
            all_records.extend(records)
            print(f"[audit] {name}: DONE ({len(records)} predictions)", file=sys.stderr)

    # ---- market benchmark ----
    if "market" not in skip:
        if cache_path("market").exists():
            all_records.extend(pd.read_csv(cache_path("market")).to_dict("records"))
        else:
            market_rows = load_market_predictions(Path(args.market_snapshots_dir), observed)
            pd.DataFrame(market_rows).to_csv(cache_path("market"), index=False)
            all_records.extend(market_rows)
            print(f"[audit] market: {len(market_rows)} matches with prematch quotes", file=sys.stderr)

    # ---- metrics ----
    metrics = pd.DataFrame([per_match_metrics(r) for r in all_records])
    metrics.to_csv(outdir / "per_match_metrics.csv", index=False)
    summary = summarize_models(metrics)
    summary.to_csv(outdir / "model_summary.csv", index=False)

    # Pairwise bootstrap deltas on exact log-loss (common matches only)
    deltas = []
    models = [m for m in MODEL_ORDER if m in set(metrics["model"])]
    by_model = {m: metrics[metrics["model"] == m].set_index("match_id") for m in models}
    for i, m1 in enumerate(models):
        for m2 in models[i + 1:]:
            common_ids = by_model[m1].index.intersection(by_model[m2].index)
            if len(common_ids) < 10:
                continue
            a = by_model[m1].loc[common_ids, "exact_log_loss"].to_numpy()
            b = by_model[m2].loc[common_ids, "exact_log_loss"].to_numpy()
            mean_d, lo, hi = bootstrap_delta_ci(a, b)
            deltas.append(
                {
                    "model_a": m1,
                    "model_b": m2,
                    "n_common": len(common_ids),
                    "exact_log_loss_delta": mean_d,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "significant": (lo > 0) or (hi < 0),
                }
            )
    pd.DataFrame(deltas).to_csv(outdir / "pairwise_deltas.csv", index=False)

    payload = {
        "matches": int(len(observed)),
        "summary": summary.to_dict("records"),
        "leakage_controls": {
            "v11_v15_v29": "static pre-tournament build, empty current-WC observed inputs, results_as_of excludes 2026 WC",
            "v36_v39_v42": "per-match walk-forward FotMob + betterdata profiles (kickoff < evaluated match only)",
            "v42_production_recipe": "v36(fotmob_wdl_blend=%.2f) -> V39CoverageOutlier(game-state priors on) -> V39BetterData(wdl=%.2f)"
            % (args.v42_fotmob_wdl_blend, v39bd.DEFAULT_BETTERDATA_WDL_BLEND),
            "market": "Polymarket prematch quotes T-60min, sum-normalized with residual 'other' mass",
        },
    }
    (outdir / "audit_summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({"summary": payload["summary"]}, indent=2))
    print(f"[audit] COMPLETE -> {outdir}", file=sys.stderr)


if __name__ == "__main__":
    main()

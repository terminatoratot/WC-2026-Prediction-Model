#!/usr/bin/env python3
"""Evaluate the v14 player model on the REAL World Cup 2026 matches played so far.

Compares the plain v13 baseline (variant="none", no player data) against the
lineup variant (variant="lineup", FIFA-designated starting XI) on the observed
WC2026 results in data/wc2026_observed_matches_from_screenshots.csv.

Metrics are copied EXACTLY from evaluate_headtohead.py so results are comparable:
  * adjusted confidence (higher better): P(actual outcome) / rank-of-actual-outcome
  * individual goal difference (lower better): |pa-ga| + |pb-gb| on most-likely scoreline
  * accuracy, ranked probability score (RPS, lower better), log-loss (lower better)

The lineup layer only applies when BOTH teams are player-covered at the latest
FIFA edition; otherwise it falls back to baseline (identical prediction on that
row). We report how many of the matches were actually player-covered.

NOTE: the sample is tiny (12 matches), so all metrics are noisy.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import pandas as pd

import v14_player_model as v14

OUTDIR = Path(__file__).resolve().parent / "wc2026_eval"
OUTCOMES = ["team_a_win", "draw", "team_b_win"]
OBSERVED_CSV = v14.DATA_DIR / "wc2026_observed_matches_from_screenshots.csv"

# Knockout-stage labels (observed CSV is all "Group Stage", but be robust).
KNOCKOUT_STAGES = {
    "round of 32", "round of 16", "quarter-final", "quarter final",
    "quarter-finals", "semi-final", "semi final", "semi-finals",
    "final", "third place", "third-place", "knockout",
}


# ---- metric functions copied EXACTLY from evaluate_headtohead.py ------------
def actual_label(ga: float, gb: float) -> str:
    return "team_a_win" if ga > gb else ("draw" if ga == gb else "team_b_win")


def adjusted_confidence(probs: Dict[str, float], actual: str) -> float:
    ranked = sorted(OUTCOMES, key=lambda k: probs[k], reverse=True)
    rank = ranked.index(actual) + 1
    return probs[actual] / rank


def individual_goal_diff(pred, ga: int, gb: int) -> float:
    top = pred["top_scorelines"][0]
    return abs(top["team_a_goals"] - ga) + abs(top["team_b_goals"] - gb)


def rps(probs: Dict[str, float], actual: str) -> float:
    # ordinal categories: a_win < draw < b_win
    p = [probs[o] for o in OUTCOMES]
    obs = [1.0 if o == actual else 0.0 for o in OUTCOMES]
    cum_p, cum_o, total = 0.0, 0.0, 0.0
    for i in range(len(OUTCOMES) - 1):
        cum_p += p[i]; cum_o += obs[i]
        total += (cum_p - cum_o) ** 2
    return total / (len(OUTCOMES) - 1)
# ---------------------------------------------------------------------------


def is_knockout(stage: object) -> bool:
    return str(stage).strip().lower() in KNOCKOUT_STAGES


def call_str(probs: Dict[str, float]) -> str:
    return {"team_a_win": "A win", "draw": "draw", "team_b_win": "B win"}[
        max(probs, key=probs.get)
    ]


def top_scoreline_str(pred) -> str:
    top = pred["top_scorelines"][0]
    return f"{top['team_a_goals']}-{top['team_b_goals']}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", default=str(v14.DATA_DIR / "worldcupsai.zip"))
    ap.add_argument("--results", default=str(v14.DATA_DIR / "results.csv"))
    ap.add_argument("--observed", default=str(OBSERVED_CSV))
    ap.add_argument("--outdir", default=str(OUTDIR))
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    print(">>> building base model + lineup player layer (one ensemble train)...")
    lineup, _ = v14.build_from_zip(args.zip, variant="lineup", results_csv=args.results)
    # baseline REUSES the same trained engine (no player layer).
    baseline = v14.V14PlayerModel(lineup.base_model, "none", {}, {}, None)
    models = {"baseline": baseline, "lineup": lineup}

    matches = pd.read_csv(args.observed)
    print(f">>> scoring {len(matches)} observed WC2026 matches")

    rows: List[dict] = []
    covered_count = 0
    for _, m in matches.iterrows():
        a, b = m["team_a"], m["team_b"]
        ga, gb = int(m["goals_a"]), int(m["goals_b"])
        actual = actual_label(ga, gb)
        ko = is_knockout(m.get("stage"))

        preds = {}
        for name, model in models.items():
            # Default match_date/edition -> latest FIFA edition (per task).
            preds[name] = model.predict(a, b, knockout=ko)

        lineup_cov = bool(preds["lineup"].get("player_adjustment", {}).get("covered"))
        if lineup_cov:
            covered_count += 1

        for name in ("baseline", "lineup"):
            pred = preds[name]
            probs = pred["result_probabilities"]
            rows.append({
                "match_id": m.get("match_id", ""),
                "team_a": a, "team_b": b,
                "stage": m.get("stage", ""),
                "score": f"{ga}-{gb}", "actual": actual,
                "model": name,
                "lineup_covered": lineup_cov,
                "pred_top_scoreline": top_scoreline_str(pred),
                "pred_call": call_str(probs),
                "p_a_win": probs["team_a_win"],
                "p_draw": probs["draw"],
                "p_b_win": probs["team_b_win"],
                "adj_conf": adjusted_confidence(probs, actual),
                "igd": individual_goal_diff(pred, ga, gb),
                "correct": int(max(probs, key=probs.get) == actual),
                "rps": rps(probs, actual),
                "logloss": -math.log(max(probs[actual], 1e-12)),
            })

    per_match = pd.DataFrame(rows)
    per_match.to_csv(outdir / "per_match_predictions.csv", index=False)

    summary = (
        per_match.groupby("model")
        .agg(n=("adj_conf", "size"),
             adjusted_confidence=("adj_conf", "mean"),
             individual_goal_diff=("igd", "mean"),
             accuracy=("correct", "mean"),
             rps=("rps", "mean"),
             logloss=("logloss", "mean"))
        .reindex(["baseline", "lineup"])
    )
    summary.to_csv(outdir / "summary.csv")

    winners = {
        "n_matches": int(len(matches)),
        "n_lineup_covered": int(covered_count),
        "adjusted_confidence (higher better)": summary["adjusted_confidence"].idxmax(),
        "individual_goal_diff (lower better)": summary["individual_goal_diff"].idxmin(),
        "accuracy (higher better)": summary["accuracy"].idxmax(),
        "rps (lower better)": summary["rps"].idxmin(),
        "logloss (lower better)": summary["logloss"].idxmin(),
    }
    (outdir / "winners.json").write_text(json.dumps(winners, indent=2))

    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n================  baseline vs lineup summary (WC2026)  ================")
    print(summary.to_string())
    print(f"\nplayer-covered matches (lineup layer applied): "
          f"{covered_count}/{len(matches)}  "
          f"(the rest fall back to baseline => identical rows)")

    print("\n----------------  per-match detail  ----------------")
    for _, m in matches.iterrows():
        a, b = m["team_a"], m["team_b"]
        sub = per_match[(per_match["team_a"] == a) & (per_match["team_b"] == b)]
        base = sub[sub["model"] == "baseline"].iloc[0]
        lin = sub[sub["model"] == "lineup"].iloc[0]
        cov = "covered" if bool(lin["lineup_covered"]) else "fallback"
        print(f"  {a} vs {b}  actual {base['score']} ({base['actual']}) [{cov}]")
        print(f"      baseline: top {base['pred_top_scoreline']:>4}  call {base['pred_call']:<6}")
        print(f"      lineup:   top {lin['pred_top_scoreline']:>4}  call {lin['pred_call']:<6}")

    print("\nWinners by metric:")
    for k, v in winners.items():
        print(f"  {k:38} -> {v}")
    print(f"\nWrote: {outdir}/summary.csv, per_match_predictions.csv, winners.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Out-of-sample head-to-head evaluation for the v14 player model.

Tests on UEFA Euro + Copa América matches (nation vs nation, NOT in the World Cup
training set) using only pre-kickoff data: each match is scored with the FIFA
edition whose snapshot pre-dates it (point-in-time, leak-free).

Three models are compared on the SAME match set:
  * baseline -- plain v13 (no player data)
  * squad    -- v14 with whole-national-squad aggregates
  * lineup   -- v14 with the FIFA-designated starting XI

Metrics
  * adjusted confidence (higher better): P(actual outcome) / rank-of-actual-outcome
  * individual goal difference (lower better): |pa-ga| + |pb-gb| on the most-likely
    scoreline
  * accuracy, ranked probability score (RPS, lower better), log-loss (lower better)

Only matches where BOTH teams have player data at the relevant edition are scored,
so the comparison is apples-to-apples.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import pandas as pd

import v14_player_model as v14

OUTDIR = Path(__file__).resolve().parent / "headtohead_eval"
OUTCOMES = ["team_a_win", "draw", "team_b_win"]
TEST_TOURNAMENTS = ["UEFA Euro", "Copa América"]


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


def load_test_matches(results_csv: Path, edition_dates) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[
        df["tournament"].isin(TEST_TOURNAMENTS)
        & df["date"].notna()
        & df["home_score"].notna()
        & df["away_score"].notna()
    ].copy()
    earliest = min(edition_dates.values())
    df = df[df["date"] >= earliest]
    df["edition"] = df["date"].map(lambda d: v14.date_to_edition(d, edition_dates))
    return df[df["edition"].notna()].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", default=str(v14.DATA_DIR / "worldcupsai.zip"))
    ap.add_argument("--results", default=str(v14.DATA_DIR / "results.csv"))
    ap.add_argument("--outdir", default=str(OUTDIR))
    ap.add_argument("--limit", type=int, default=0, help="cap #matches (debug)")
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    print(">>> building base model + player layers (one ensemble train, shared)...")
    squad, _ = v14.build_from_zip(args.zip, variant="squad", results_csv=args.results)
    # lineup + baseline reuse the SAME trained engine.
    lin_lookup, lin_eds = v14.load_player_tables("lineup")
    lin_impact = v14.fit_player_impact(lin_lookup, lin_eds, Path(args.results))
    lineup = v14.V14PlayerModel(squad.base_model, "lineup", lin_lookup, lin_eds, lin_impact)
    baseline = v14.V14PlayerModel(squad.base_model, "none", {}, {}, None)
    models = {"baseline": baseline, "squad": squad, "lineup": lineup}

    matches = load_test_matches(Path(args.results), squad.edition_dates)
    # keep only matches both teams are covered at that edition (fair comparison)
    def covered(row):
        a = v14.norm_team(row["home_team"])
        b = v14.norm_team(row["away_team"])
        e = int(row["edition"])
        return (a, e) in squad.lookup and (b, e) in squad.lookup
    matches = matches[matches.apply(covered, axis=1)].reset_index(drop=True)
    if args.limit:
        matches = matches.head(args.limit)
    print(f">>> scoring {len(matches)} covered Euro/Copa matches "
          f"({matches['date'].min().date()}..{matches['date'].max().date()})")

    rows: List[dict] = []
    for _, m in matches.iterrows():
        a, b = m["home_team"], m["away_team"]
        ga, gb = int(m["home_score"]), int(m["away_score"])
        actual = actual_label(ga, gb)
        edition = int(m["edition"])
        for name, model in models.items():
            pred = model.predict(a, b, match_date=m["date"], edition=(edition if name != "baseline" else None))
            probs = pred["result_probabilities"]
            rows.append({
                "date": m["date"].date(), "team_a": a, "team_b": b,
                "score": f"{ga}-{gb}", "actual": actual, "model": name,
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
        .reindex(["baseline", "squad", "lineup"])
    )
    summary.to_csv(outdir / "summary.csv")

    winners = {
        "adjusted_confidence (higher better)": summary["adjusted_confidence"].idxmax(),
        "individual_goal_diff (lower better)": summary["individual_goal_diff"].idxmin(),
        "accuracy (higher better)": summary["accuracy"].idxmax(),
        "rps (lower better)": summary["rps"].idxmin(),
        "logloss (lower better)": summary["logloss"].idxmin(),
    }
    (outdir / "winners.json").write_text(json.dumps(winners, indent=2))

    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n================  A/B/baseline summary  ================")
    print(summary.to_string())
    print("\nWinners by metric:")
    for k, v in winners.items():
        print(f"  {k:38} -> {v}")
    print(f"\nWrote: {outdir}/summary.csv, per_match_predictions.csv, winners.json")


if __name__ == "__main__":
    main()

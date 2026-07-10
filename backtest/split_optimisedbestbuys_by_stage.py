"""Re-run tune_v46_4_tiers.py's in-memory grid+random search separately for
group-stage vs. knockout matches, using the already-cached candidate rows
from the v46_4_optimisedbestbuys tuning run (outputs/old_outputs/v46_4_tier_optimization_final/cached_training_rows.csv).

Mirrors split_tier_weights_by_stage.py (which did this for the basev51 cache)
so the two models' group/knockout ROI splits are directly comparable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import v11_wcq_results_model as v11
from tune_v46_4_tiers import BetRow, run_search

CACHE_CSV = Path("outputs/old_outputs/v46_4_tier_optimization_final/cached_training_rows.csv")
FBREF_CSV = Path("data/fbref_world_cup_matches.csv")

KNOCKOUT_ROUNDS = {
    "Round of 32",
    "Round of 16",
    "Quarter-finals",
    "Semi-finals",
    "Third place",
    "Final",
}

TARGET_STAKE = 5.0
MIN_ORDER_SIZE = 1.0
ROUNDING = 0.05
BANKROLL = 100.0
CAP_MODE = "fraction"
FOLD_COUNT = 5
RANDOM_TRIALS = 1000
TOP_K = 20
SEED = 46
HARD_FILTER_ARGS = {
    "min_behr": 0.92,
    "min_value_share": 0.65,
    "max_drawdown_allowed": 0.40,
    "min_avg_bets": 4.0,
    "max_avg_bets": 5.0,
    "max_cover_leakage_ratio": 0.15,
}


def load_round_by_pair() -> dict[frozenset, str]:
    fbref = pd.read_csv(FBREF_CSV).dropna(subset=["score"])
    pairs = fbref.apply(lambda r: frozenset([v11.canon_team(r["home_team"]), v11.canon_team(r["away_team"])]), axis=1)
    return dict(zip(pairs, fbref["round"]))


def main() -> None:
    df = pd.read_csv(CACHE_CSV)
    round_by_pair = load_round_by_pair()
    df["pair"] = df.apply(lambda r: frozenset([v11.canon_team(r["team_a"]), v11.canon_team(r["team_b"])]), axis=1)
    df["round"] = df["pair"].map(round_by_pair)
    df["is_knockout"] = df["round"].isin(KNOCKOUT_ROUNDS)

    unmatched = df[df["round"].isna()]["pair"].nunique()
    if unmatched:
        print(f"[warn] {unmatched} match pairs did not match an fbref round", file=sys.stderr)

    for label, outdir, subset in [
        ("group-stage", Path("outputs/v46_4_optimisedbestbuys_tier_optimization_group_stage"), df[~df["is_knockout"]]),
        ("knockout", Path("outputs/v46_4_optimisedbestbuys_tier_optimization_knockout"), df[df["is_knockout"]]),
    ]:
        bets = [
            BetRow(
                match_id=str(r["match_id"]),
                kickoff=str(r["kickoff"]),
                team_a=str(r["team_a"]),
                team_b=str(r["team_b"]),
                final_score=str(r["final_score"]),
                scoreline=str(r["scoreline"]),
                role=str(r["role"]),
                tier=str(r["tier"]),
                model_probability=float(r["model_probability"]),
                market_price=float(r["market_price"]),
                raw_edge=float(r["raw_edge"]),
                model_rank=None if pd.isna(r["model_rank"]) else int(r["model_rank"]),
            )
            for _, r in subset.iterrows()
        ]
        match_count = len({b.match_id for b in bets})
        print(f"[{label}] {match_count} matches, {len(bets)} candidate bets")

        outdir.mkdir(parents=True, exist_ok=True)
        subset[["match_id", "kickoff", "team_a", "team_b", "final_score", "scoreline", "role", "tier", "model_probability", "market_price", "raw_edge", "model_rank"]].to_csv(
            outdir / "cached_training_rows.csv", index=False
        )
        run_search(
            bets=bets,
            outdir=outdir,
            target_stake=TARGET_STAKE,
            min_order_size=MIN_ORDER_SIZE,
            rounding=ROUNDING,
            bankroll=BANKROLL,
            cap_mode=CAP_MODE,
            fold_count=FOLD_COUNT,
            random_trials=RANDOM_TRIALS,
            top_k=TOP_K,
            seed=SEED,
            hard_filter_args=HARD_FILTER_ARGS,
        )
        print(f"[{label}] wrote results to {outdir}")


if __name__ == "__main__":
    main()

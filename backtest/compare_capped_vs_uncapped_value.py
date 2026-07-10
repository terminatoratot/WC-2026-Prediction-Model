"""Isolate the effect of the new VALUE_1/VALUE_2/OUTLIER caps from the effect
of the newly-added matches, by re-running the OLD 7-dimension search (VALUE_1
fixed weight=1.0, uncapped; no v1_cap/v2_cap/outlier_cap) against the SAME
already-refreshed cached_training_rows.csv files the new 10-dim search just
used. Same data both times -- the only variable is whether the three new caps
exist at all.

"Uncapped" is reproduced by fixing v1_cap/v2_cap/outlier_cap to a constant
large enough it can never bind in a target_stake=5 backtest, rather than by
special-casing None -- so this reuses tier_weight_and_cap()/stake_match()
completely unmodified.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from tune_v46_4_tiers import BetRow, TierParams, clip, params_key, run_search

UNCAPPED = 999.0


def old_coarse_grid() -> list[TierParams]:
    values = {
        "v2": [0.50, 0.65, 0.80],
        "c1": [0.10, 0.20, 0.35],
        "c2": [0.05, 0.12, 0.22],
        "outlier": [0.25, 0.40, 0.55],
        "c1_cap": [0.20, 0.35, 0.50],
        "c2_cap": [0.10, 0.20, 0.35],
        "neg_mult": [0.35, 0.50, 0.65],
    }
    params = []
    for combo in itertools.product(
        values["v2"], values["c1"], values["c2"], values["outlier"],
        values["c1_cap"], values["c2_cap"], values["neg_mult"],
    ):
        params.append(TierParams(*combo, v1_cap=UNCAPPED, v2_cap=UNCAPPED, outlier_cap=UNCAPPED))
    return params


def old_sample_around(base: TierParams, rng) -> TierParams:
    return TierParams(
        v2=clip(rng.uniform(base.v2 - 0.10, base.v2 + 0.10), 0.40, 0.90),
        c1=clip(rng.uniform(base.c1 - 0.08, base.c1 + 0.08), 0.05, 0.45),
        c2=clip(rng.uniform(base.c2 - 0.06, base.c2 + 0.06), 0.00, 0.30),
        outlier=clip(rng.uniform(base.outlier - 0.12, base.outlier + 0.12), 0.10, 0.70),
        c1_cap=clip(rng.uniform(base.c1_cap - 0.10, base.c1_cap + 0.10), 0.10, 0.60),
        c2_cap=clip(rng.uniform(base.c2_cap - 0.08, base.c2_cap + 0.08), 0.05, 0.40),
        neg_mult=clip(rng.uniform(base.neg_mult - 0.12, base.neg_mult + 0.12), 0.20, 0.70),
        v1_cap=UNCAPPED,
        v2_cap=UNCAPPED,
        outlier_cap=UNCAPPED,
    )


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


def load_bets(cache_csv: Path) -> list[BetRow]:
    df = pd.read_csv(cache_csv)
    return [
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
        for _, r in df.iterrows()
    ]


def main() -> None:
    import tune_v46_4_tiers as tuner

    # monkeypatch: reuse run_search's internals but with the old 7-dim space
    tuner.coarse_grid = old_coarse_grid
    tuner.sample_around = old_sample_around

    for label, cache_csv, outdir in [
        ("combined", Path("outputs/v46_4_basev51_tier_optimization/cached_training_rows.csv"),
         Path("outputs/v46_4_basev51_tier_optimization_UNCAPPED_COMPARISON")),
        ("group-stage", Path("outputs/v46_4_basev51_tier_optimization_group_stage/cached_training_rows.csv"),
         Path("outputs/v46_4_basev51_tier_optimization_group_stage_UNCAPPED_COMPARISON")),
        ("knockout", Path("outputs/v46_4_basev51_tier_optimization_knockout/cached_training_rows.csv"),
         Path("outputs/v46_4_basev51_tier_optimization_knockout_UNCAPPED_COMPARISON")),
    ]:
        bets = load_bets(cache_csv)
        match_count = len({b.match_id for b in bets})
        print(f"[{label}] {match_count} matches, {len(bets)} candidate bets (uncapped/old-style search)")
        outdir.mkdir(parents=True, exist_ok=True)
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

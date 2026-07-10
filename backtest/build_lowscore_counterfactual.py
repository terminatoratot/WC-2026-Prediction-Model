"""Rebuild the 22 basev51 knockout backtest matches with the default
0-0/1-0/0-1 knockout exclusion turned OFF (--allow-low-score-knockout-buys
--ignore-scorelines ""), to directly measure how many matches the exclusion
rule actually helps vs hurts, instead of reasoning about it.

Writes counterfactual cards to a separate backtest_cards dir so the existing
baseline (exclusion on) is untouched.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from tune_v46_4_tiers import (
    clean_text,
    parse_bet_rows_from_buy_card,
    run_v46_for_match,
    sanitize_fotmob_facts,
)

KNOCKOUT_MATCH_IDS = [
    "4653705", "4653711", "4653703", "4653706", "4653712", "4653704", "4653713",
    "4653710", "4653709", "4653708", "4653707", "4653717", "4653716", "4653718",
    "4653843", "4653842", "4653846", "4653847", "4653844", "4653845", "4653848",
    "4653849",
]

FOTMOB_PATH = Path("data/fotmob_match_facts_clean.csv")
PRICE_HISTORY_ROOT = Path("data/polymarket_prematch_snapshots_from_fotmob")
OUTDIR = Path("outputs/v46_4_basev51_knockout_lowscore_counterfactual")
TARGET_STAKE = 5.0
MIN_ORDER_SIZE = 1.0
ROUNDING = 0.05


def main() -> None:
    fotmob = pd.read_csv(FOTMOB_PATH)
    fotmob["match_id"] = fotmob["match_id"].astype(str)
    sanitized_path = sanitize_fotmob_facts(FOTMOB_PATH, OUTDIR / "sanitized_fotmob_match_facts.csv")
    cache_cards_dir = OUTDIR / "backtest_cards"

    def process_one(match_number: int, match_id: str) -> dict:
        row = fotmob[fotmob["match_id"] == match_id].iloc[0].to_dict()
        card_dir = run_v46_for_match(
            script=Path("v46_4_basev51.py"),
            match_row=row,
            match_number=match_number,
            sanitized_fotmob_path=sanitized_path,
            price_history_root=PRICE_HISTORY_ROOT,
            cache_cards_dir=cache_cards_dir,
            extra_v46_args=["--allow-low-score-knockout-buys", "--ignore-scorelines", "", "--no-plots"],
            target_stake=TARGET_STAKE,
            min_order_size=MIN_ORDER_SIZE,
            rounding=ROUNDING,
            force=False,
        )
        if card_dir is None:
            print(f"[counterfactual] {match_id}: {row['home_team']} vs {row['away_team']} | FAILED")
            return {}

        bets = parse_bet_rows_from_buy_card(match_row=row, card_dir=card_dir)
        final_score = f"{int(row['home_score'])}-{int(row['away_score'])}"
        hit_bets = [b for b in bets if b.scoreline == final_score]
        result = {
            "match_id": match_id,
            "team_a": row["home_team"],
            "team_b": row["away_team"],
            "final_score": final_score,
            "counterfactual_selected": sorted({b.scoreline for b in bets}),
            "counterfactual_hit": bool(hit_bets),
            "counterfactual_hit_role": hit_bets[0].role if hit_bets else None,
        }
        print(f"[counterfactual] {match_id}: {row['home_team']} vs {row['away_team']} -> "
              f"final={final_score} hit={bool(hit_bets)} selected={sorted({b.scoreline for b in bets})}")
        return result

    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(process_one, n, mid): mid
            for n, mid in enumerate(KNOCKOUT_MATCH_IDS, start=1)
        }
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTDIR / "counterfactual_results.csv", index=False)
    print(f"\nWrote {OUTDIR / 'counterfactual_results.csv'}")


if __name__ == "__main__":
    main()

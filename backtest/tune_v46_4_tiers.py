#!/usr/bin/env python3
"""
tune_v46_4_tiers.py

Optimises the V46.4 tiered-balanced staking layer using cached historical
V46.4 cards.


cmd:
MPLCONFIGDIR=.matplotlib_cache LOKY_MAX_CPU_COUNT=8 \
.venv/bin/python tune_v46_4_tiers.py \
  --outdir outputs/v46_4_tier_optimization_final \
  --force-cache \
  --target-stake 5 \
  --rounding 0.05 \
  --cap-mode fraction \
  --extra-v46-arg=--no-plots

Design:
1. Read completed FotMob matches.
2. For each completed match, generate/cache one V46.4 historical buy card.
3. Extract selected scorelines, roles, model probabilities, prices, and final score.
4. Optimise only the staking layer:
   - VALUE_2 weight
   - COVER_1 weight
   - COVER_2 weight
   - OUTLIER weight
   - COVER_1 surplus cap
   - COVER_2 surplus cap
   - negative-edge cover cap multiplier
5. Run coarse grid search.
6. Run random search around the best grid results.
7. Save:
   - best_params.json
   - parameter_search_results.csv
   - fold_results.csv
   - cached_training_rows.csv
   - equity curve CSVs

This intentionally does not modify v46_4_optimisedbestbuys.py.
It treats V46.4 as the card generator and tunes the staking layer on top.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


# -----------------------------
# Basic file helpers
# -----------------------------

def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    for row in rows:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def sanitize_fotmob_facts(source: Path, dest: Path) -> Path:
    """Write a no-result FotMob facts file for prediction/card generation."""
    rows = read_csv(source)
    for row in rows:
        for key in ("home_score", "away_score", "status", "raw_text"):
            if key in row:
                row[key] = ""
    write_csv(dest, rows)
    return dest


def to_float(x: Any, default: float | None = None) -> float | None:
    if x is None:
        return default
    if isinstance(x, str) and x.strip().lower() in {"", "none", "nan", "null"}:
        return default
    try:
        value = float(x)
    except Exception:
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def to_int(x: Any, default: int | None = None) -> int | None:
    value = to_float(x)
    if value is None:
        return default
    return int(value)


def clean_text(x: Any) -> str:
    return str(x or "").strip()


def is_completed_match(row: dict[str, Any]) -> bool:
    return to_float(row.get("home_score")) is not None and to_float(row.get("away_score")) is not None


def final_score_from_fotmob_row(row: dict[str, Any]) -> str | None:
    home = to_int(row.get("home_score"))
    away = to_int(row.get("away_score"))
    if home is None or away is None:
        return None
    return f"{home}-{away}"


def is_knockout_round(row: dict[str, Any]) -> bool:
    text = clean_text(row.get("round")).lower()
    knockout_terms = [
        "round of",
        "knockout",
        "quarter",
        "semi",
        "final",
        "3rd",
        "third",
    ]
    return any(term in text for term in knockout_terms)


# -----------------------------
# Polymarket event slug helpers
# -----------------------------

def strip_exact_score_suffix(slug: str) -> str:
    # V46.4 needs the exact-score event slug for historical exact-score markets.
    # Do not strip '-exact-score'. The old base event slug returns raw markets but
    # zero exact-score markets.
    return clean_text(slug)


def find_polymarket_event_slug_in_obj(obj: Any) -> str | None:
    """Find a plausible Polymarket FIFWC event slug inside a nested JSON object."""
    if isinstance(obj, str):
        value = obj.strip()
        if value.startswith("fifwc-"):
            return strip_exact_score_suffix(value)
        return None

    if isinstance(obj, dict):
        preferred_keys = [
            "event_slug_found",
            "event_slug",
            "matched_slug",
            "slug",
        ]
        for key in preferred_keys:
            if key in obj:
                found = find_polymarket_event_slug_in_obj(obj[key])
                if found:
                    return found
        for value in obj.values():
            found = find_polymarket_event_slug_in_obj(value)
            if found:
                return found
        return None

    if isinstance(obj, list):
        for value in obj:
            found = find_polymarket_event_slug_in_obj(value)
            if found:
                return found
        return None

    return None


def read_polymarket_event_slug_for_match(price_history_root: Path, match_id: str) -> str | None:
    """Read the already-pulled Polymarket slug for a FotMob match snapshot."""
    match_dir = price_history_root / match_id
    if not match_dir.exists():
        return None

    for csv_name in ("prematch_prices.csv",):
        csv_path = match_dir / csv_name
        if csv_path.exists() and csv_path.stat().st_size > 0:
            for row in read_csv(csv_path):
                for key in ("event_slug", "event_slug_found", "matched_slug", "slug"):
                    found = find_polymarket_event_slug_in_obj(row.get(key))
                    if found:
                        return found

    for name in ("prematch_prices.json", "event.json", "pull_summary.json", "raw_markets.json"):
        path = match_dir / name
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        found = find_polymarket_event_slug_in_obj(obj)
        if found:
            return found

    all_prices_path = price_history_root / "all_prematch_prices.csv"
    if all_prices_path.exists() and all_prices_path.stat().st_size > 0:
        for row in read_csv(all_prices_path):
            if clean_text(row.get("match_id")) != match_id:
                continue
            for key in (
                "event_slug",
                "event_slug_found",
                "matched_slug",
                "slug",
                "market_slug",
                "condition_slug",
            ):
                found = find_polymarket_event_slug_in_obj(row.get(key))
                if found:
                    return found

    return None


def build_historical_polymarket_json(
    *,
    price_history_root: Path,
    match_id: str,
    outdir: Path,
) -> Path | None:
    """Convert cached prematch price rows into a V42/V46-compatible market JSON."""
    source = price_history_root / match_id / "prematch_prices.csv"
    rows = read_csv(source)
    if not rows:
        return None

    markets: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        yes_price = to_float(row.get("yes_price"))
        if yes_price is None or yes_price <= 0.0 or yes_price >= 1.0:
            continue

        no_price = max(0.0, min(1.0, 1.0 - yes_price))
        question = clean_text(row.get("market_question"))
        slug = clean_text(row.get("market_slug"))
        token_id = clean_text(row.get("yes_token_id")) or f"{match_id}-{idx}-yes"

        markets.append(
            {
                "id": f"{match_id}-{idx}",
                "slug": slug,
                "question": question,
                "title": question,
                "groupItemTitle": question.replace("Exact Score:", "").strip(),
                "sportsMarketType": "soccer_exact_score",
                "outcomes": json.dumps(["Yes", "No"]),
                "outcomePrices": json.dumps([round(yes_price, 6), round(no_price, 6)]),
                "clobTokenIds": json.dumps([token_id, f"{token_id}-no"]),
                "volume": 0.0,
                "volumeNum": 0.0,
                "liquidity": 0.0,
                "liquidityNum": 0.0,
            }
        )

    if not markets:
        return None

    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "historical_polymarket_markets.json"
    write_json(
        path,
        {
            "source": str(source),
            "match_id": match_id,
            "markets": markets,
        },
    )
    return path


# -----------------------------
# Data structures
# -----------------------------

@dataclass(frozen=True)
class TierParams:
    v2: float
    c1: float
    c2: float
    outlier: float
    c1_cap: float
    c2_cap: float
    neg_mult: float
    v1_cap: float
    v2_cap: float
    outlier_cap: float

    def as_output_dict(self) -> dict[str, float]:
        return {
            "tier_value_1_weight": 1.0,
            "tier_value_2_weight": self.v2,
            "tier_cover_1_weight": self.c1,
            "tier_cover_2_weight": self.c2,
            "tier_outlier_weight": self.outlier,
            "tier_cover_1_cap": self.c1_cap,
            "tier_cover_2_cap": self.c2_cap,
            "tier_negative_edge_cap_multiplier": self.neg_mult,
            "tier_value_1_cap": self.v1_cap,
            "tier_value_2_cap": self.v2_cap,
            "tier_outlier_cap": self.outlier_cap,
        }


@dataclass
class BetRow:
    match_id: str
    kickoff: str
    team_a: str
    team_b: str
    final_score: str
    scoreline: str
    role: str
    tier: str
    model_probability: float
    market_price: float
    raw_edge: float
    model_rank: int | None


@dataclass
class StakedBet:
    row: BetRow
    break_even_stake: float
    base_stake: float
    surplus_stake: float
    stake: float
    gross_payout_if_hit: float
    net_profit_if_hit: float


@dataclass
class MatchResult:
    match_id: str
    kickoff: str
    team_a: str
    team_b: str
    final_score: str
    selected_count: int
    total_stake: float
    hit: bool
    hit_scoreline: str | None
    hit_role: str | None
    profit: float
    value_win_profit: float
    outlier_win_profit: float
    cover_win_profit: float
    gross_positive_profit: float
    cover_leakage: float


# -----------------------------
# Cache building using V46.4
# -----------------------------

def find_buy_card(path: Path) -> Path | None:
    exact = path / "v46_4_buy_card.csv"
    if exact.exists():
        return exact

    candidates = sorted(path.glob("*buy*card*.csv"))
    if candidates:
        return candidates[0]

    return None


def find_selection_debug(path: Path) -> Path | None:
    exact = path / "v46_4_selection_debug.csv"
    if exact.exists():
        return exact

    candidates = sorted(path.glob("*selection*debug*.csv"))
    if candidates:
        return candidates[0]

    return None


def cached_selection_debug_is_usable(path: Path, *, min_rows: int = 2) -> bool:
    debug_csv = find_selection_debug(path)
    if debug_csv is None:
        return False

    usable = 0
    for row in read_csv(debug_csv):
        if not truthy_field(row.get("selected")):
            continue
        price = to_float(row.get("market_price"), to_float(row.get("price")))
        prob = to_float(row.get("model_probability"), to_float(row.get("fair")))
        if price is None or prob is None or price <= 0.0 or price >= 1.0 or prob <= 0.0:
            continue
        usable += 1
        if usable >= min_rows:
            return True

    return False


def run_v46_for_match(
    *,
    script: Path,
    match_row: dict[str, Any],
    match_number: int,
    sanitized_fotmob_path: Path,
    price_history_root: Path,
    cache_cards_dir: Path,
    extra_v46_args: list[str],
    target_stake: float,
    min_order_size: float,
    rounding: float,
    force: bool,
) -> Path | None:
    match_id = clean_text(match_row.get("match_id"))
    team_a = clean_text(match_row.get("home_team"))
    team_b = clean_text(match_row.get("away_team"))

    if not match_id or not team_a or not team_b:
        return None

    event_slug = read_polymarket_event_slug_for_match(price_history_root, match_id)

    if not event_slug:
        print(f"[cache] {match_id}: {team_a} vs {team_b} | slug=NOT_FOUND | skipped", file=sys.stderr)
        return None

    outdir = cache_cards_dir / match_id
    if cached_selection_debug_is_usable(outdir) and not force:
        return outdir

    polymarket_json = build_historical_polymarket_json(
        price_history_root=price_history_root,
        match_id=match_id,
        outdir=outdir,
    )
    if polymarket_json is None:
        print(f"[cache] {match_id}: {team_a} vs {team_b} | prices=NOT_FOUND | skipped", file=sys.stderr)
        return None

    cmd = [
        sys.executable,
        str(script),
        "--team-a",
        team_a,
        "--team-b",
        team_b,
        "--outdir",
        str(outdir),
        "--fotmob-match-facts",
        str(sanitized_fotmob_path),
        "--price-history-root",
        str(price_history_root),
        "--polymarket-json",
        str(polymarket_json),
        "--no-fetch-polymarket",
        "--stake-profile",
        "tiered-balanced",
        "--execution-target-stake",
        str(target_stake),
        "--min-order-size",
        str(min_order_size),
        "--stake-rounding",
        str(rounding),
        "--match-number",
        str(match_number),
        # The tuner must not inherit any pre-optimised V46.4 staking weights.
        # V46.4 is used here only to generate the selected card/roles,
        # including directional clean-sheet hedge logic. All staking weights
        # are re-applied and searched inside this tuner.
        "--tier-value-2-weight",
        "1.0",
        "--tier-cover-1-weight",
        "0.0",
        "--tier-cover-2-weight",
        "0.0",
        "--tier-outlier-weight",
        "1.0",
        "--tier-cover-1-cap",
        "0.0",
        "--tier-cover-2-cap",
        "0.0",
        "--tier-negative-edge-cap-multiplier",
        "0.0",
    ]

    if event_slug and "--polymarket-json" not in extra_v46_args:
        cmd.extend(["--polymarket-event-slug", event_slug])

    if is_knockout_round(match_row):
        cmd.append("--knockout")

    cmd.extend(extra_v46_args)

    print(f"[cache] {match_id}: {team_a} vs {team_b} | slug={event_slug}", file=sys.stderr)

    try:
        completed = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except Exception as exc:
        print(f"[cache] ERROR launching V46.4 for {match_id}: {exc}", file=sys.stderr)
        return None

    if completed.returncode != 0:
        print(f"[cache] V46.4 failed for {match_id}", file=sys.stderr)
        print(completed.stderr[-2500:], file=sys.stderr)
        return None

    if find_selection_debug(outdir) is None:
        print(f"[cache] no selection debug found for {match_id}", file=sys.stderr)
        return None

    return outdir


def infer_role(row: dict[str, Any]) -> str:
    role = clean_text(row.get("buy_hold_role"))
    if role:
        return role

    role = clean_text(row.get("role"))
    if role:
        return role

    section = clean_text(row.get("section")).upper()
    if "OUTLIER" in section:
        return "OUTLIER_VALUE"
    if "VALUE" in section:
        return "VALUE"
    if "COVER" in section:
        return "COVER"

    return "UNKNOWN"


def infer_tier(role_counts: dict[str, int], role: str) -> str:
    if role == "VALUE":
        role_counts["VALUE"] = role_counts.get("VALUE", 0) + 1
        return f"VALUE_{role_counts['VALUE']}"
    if role == "COVER":
        role_counts["COVER"] = role_counts.get("COVER", 0) + 1
        return f"COVER_{role_counts['COVER']}"
    if role == "OUTLIER_VALUE":
        return "OUTLIER"
    return role or "UNKNOWN"


def truthy_field(value: Any) -> bool:
    return clean_text(value).lower() in {"1", "true", "yes", "y"}


def parse_selected_candidate_rows(
    *,
    match_row: dict[str, Any],
    card_dir: Path,
) -> list[BetRow]:
    debug_csv = find_selection_debug(card_dir)
    if debug_csv is None:
        return []

    rows = read_csv(debug_csv)
    rows = [row for row in rows if truthy_field(row.get("selected"))]
    if not rows:
        return []

    rows.sort(
        key=lambda row: (
            to_int(row.get("selection_order"), 999999) or 999999,
            to_int(row.get("model_rank"), 999999) or 999999,
            clean_text(row.get("scoreline")),
        )
    )

    match_id = clean_text(match_row.get("match_id"))
    kickoff = clean_text(match_row.get("kickoff"))
    team_a = clean_text(match_row.get("home_team"))
    team_b = clean_text(match_row.get("away_team"))
    final_score = final_score_from_fotmob_row(match_row)

    if not final_score:
        return []

    parsed: list[BetRow] = []
    role_counts: dict[str, int] = {}

    for row in rows:
        scoreline = clean_text(row.get("scoreline"))
        if not scoreline or "-" not in scoreline:
            continue

        price = to_float(row.get("market_price"), to_float(row.get("price")))
        prob = to_float(row.get("model_probability"), to_float(row.get("fair")))
        if price is None or prob is None or price <= 0 or price >= 1 or prob <= 0:
            continue

        role = infer_role(row)
        if role not in {"VALUE", "COVER", "OUTLIER_VALUE"}:
            continue

        tier = clean_text(row.get("buy_hold_tier") or row.get("tier"))
        if not tier:
            tier = infer_tier(role_counts, role)

        raw_edge = to_float(row.get("raw_edge"), prob - price)
        model_rank = to_int(row.get("model_rank"))

        parsed.append(
            BetRow(
                match_id=match_id,
                kickoff=kickoff,
                team_a=team_a,
                team_b=team_b,
                final_score=final_score,
                scoreline=scoreline,
                role=role,
                tier=tier,
                model_probability=prob,
                market_price=price,
                raw_edge=float(raw_edge or 0.0),
                model_rank=model_rank,
            )
        )

    return parsed


def parse_bet_rows_from_buy_card(
    *,
    match_row: dict[str, Any],
    card_dir: Path,
) -> list[BetRow]:
    debug_rows = parse_selected_candidate_rows(match_row=match_row, card_dir=card_dir)
    if debug_rows:
        return debug_rows

    buy_card = find_buy_card(card_dir)
    if buy_card is None:
        return []

    rows = read_csv(buy_card)
    if not rows:
        return []

    match_id = clean_text(match_row.get("match_id"))
    kickoff = clean_text(match_row.get("kickoff"))
    team_a = clean_text(match_row.get("home_team"))
    team_b = clean_text(match_row.get("away_team"))
    final_score = final_score_from_fotmob_row(match_row)

    if not final_score:
        return []

    parsed: list[BetRow] = []
    role_counts: dict[str, int] = {}

    for row in rows:
        scoreline = clean_text(row.get("scoreline"))
        if not scoreline or "-" not in scoreline:
            continue

        price = to_float(row.get("market_price"))
        prob = to_float(row.get("model_probability"))
        if price is None or prob is None or price <= 0 or prob <= 0:
            continue

        role = infer_role(row)
        if role not in {"VALUE", "COVER", "OUTLIER_VALUE"}:
            continue

        tier = clean_text(row.get("buy_hold_tier"))
        if not tier:
            tier = infer_tier(role_counts, role)

        raw_edge = to_float(row.get("raw_edge"), prob - price)
        model_rank = to_int(row.get("model_rank"))

        parsed.append(
            BetRow(
                match_id=match_id,
                kickoff=kickoff,
                team_a=team_a,
                team_b=team_b,
                final_score=final_score,
                scoreline=scoreline,
                role=role,
                tier=tier,
                model_probability=prob,
                market_price=price,
                raw_edge=float(raw_edge or 0.0),
                model_rank=model_rank,
            )
        )

    return parsed


def build_or_load_training_cache(
    *,
    fotmob_path: Path,
    script: Path,
    price_history_root: Path,
    outdir: Path,
    force_cache: bool,
    max_matches: int | None,
    extra_v46_args: list[str],
    target_stake: float,
    min_order_size: float,
    rounding: float,
    workers: int = 1,
) -> list[BetRow]:
    cache_path = outdir / "cached_training_rows.csv"

    if cache_path.exists() and not force_cache:
        print(f"[cache] reading {cache_path}", file=sys.stderr)
        cached = [
            BetRow(
                match_id=clean_text(r["match_id"]),
                kickoff=clean_text(r["kickoff"]),
                team_a=clean_text(r["team_a"]),
                team_b=clean_text(r["team_b"]),
                final_score=clean_text(r["final_score"]),
                scoreline=clean_text(r["scoreline"]),
                role=clean_text(r["role"]),
                tier=clean_text(r["tier"]),
                model_probability=float(r["model_probability"]),
                market_price=float(r["market_price"]),
                raw_edge=float(r["raw_edge"]),
                model_rank=to_int(r.get("model_rank")),
            )
            for r in read_csv(cache_path)
        ]
        cached_match_count = len(set(b.match_id for b in cached))
        if cached_match_count >= 10:
            return cached
        print(
            f"[cache] existing cache has only {cached_match_count} matches; rebuilding",
            file=sys.stderr,
        )

    fotmob_rows = sorted(
        [r for r in read_csv(fotmob_path) if is_completed_match(r)],
        key=lambda r: (clean_text(r.get("kickoff")), clean_text(r.get("match_id"))),
    )

    if max_matches is not None:
        fotmob_rows = fotmob_rows[:max_matches]

    cache_cards_dir = outdir / "backtest_cards"
    sanitized_fotmob_path = sanitize_fotmob_facts(
        fotmob_path,
        outdir / "sanitized_fotmob_match_facts.csv",
    )
    all_bets: list[BetRow] = []

    def process_one(match_number: int, match_row: dict[str, Any]) -> list[BetRow]:
        card_dir = run_v46_for_match(
            script=script,
            match_row=match_row,
            match_number=match_number,
            sanitized_fotmob_path=sanitized_fotmob_path,
            price_history_root=price_history_root,
            cache_cards_dir=cache_cards_dir,
            extra_v46_args=extra_v46_args,
            target_stake=target_stake,
            min_order_size=min_order_size,
            rounding=rounding,
            force=force_cache,
        )
        if card_dir is None:
            return []

        bets = parse_bet_rows_from_buy_card(match_row=match_row, card_dir=card_dir)
        if len(bets) < 2:
            print(
                f"[cache] {clean_text(match_row.get('match_id'))}: "
                f"only {len(bets)} usable selected rows | skipped",
                file=sys.stderr,
            )
            return []

        return bets

    # match_number must reflect each row's fixed chronological position (it drives
    # --knockout via GROUP_STAGE_MATCH_COUNT), so it is assigned here before
    # dispatch, not derived from whichever worker happens to finish first.
    numbered_rows = list(enumerate(fotmob_rows, start=1))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(process_one, n, r) for n, r in numbered_rows]
            for future in as_completed(futures):
                all_bets.extend(future.result())
    else:
        for match_number, match_row in numbered_rows:
            all_bets.extend(process_one(match_number, match_row))

    all_bets.sort(key=lambda b: (b.kickoff, b.match_id, b.scoreline))

    write_csv(cache_path, [asdict(b) for b in all_bets])

    print(
        f"[cache] cached {len(all_bets)} bet rows across "
        f"{len(set(b.match_id for b in all_bets))} matches",
        file=sys.stderr,
    )

    return all_bets


# -----------------------------
# Tier staking allocator
# -----------------------------

def tier_weight_and_cap(
    bet: BetRow,
    params: TierParams,
    *,
    target_stake: float,
    cap_mode: str,
) -> tuple[float, float | None]:
    """
    Returns surplus weight and surplus cap.

    cap_mode:
    - absolute: cap values are currency units, matching the current V46.4 style.
    - fraction: cap values are fractions of target_stake.

    VALUE_1's cap is soft: stake_match() lets it absorb any surplus left over
    after every tier (including VALUE_1 itself) has saturated, so total
    staked still always sums to target_stake.
    """
    tier = bet.tier

    if tier == "VALUE_1":
        cap = params.v1_cap
        if cap_mode == "fraction":
            cap *= target_stake
        return 1.00, cap

    if tier == "VALUE_2":
        cap = params.v2_cap
        if cap_mode == "fraction":
            cap *= target_stake
        return params.v2, cap

    if tier == "OUTLIER":
        cap = params.outlier_cap
        if cap_mode == "fraction":
            cap *= target_stake
        if bet.raw_edge < 0:
            cap *= params.neg_mult
        return params.outlier, cap

    if tier == "COVER_1":
        cap = params.c1_cap
        if cap_mode == "fraction":
            cap *= target_stake
        if bet.raw_edge < 0:
            cap *= params.neg_mult
        return params.c1, cap

    if tier == "COVER_2":
        cap = params.c2_cap
        if cap_mode == "fraction":
            cap *= target_stake
        if bet.raw_edge < 0:
            cap *= params.neg_mult
        return params.c2, cap

    if bet.role == "COVER":
        cap = max(params.c2_cap * 0.5, 0.05)
        if cap_mode == "fraction":
            cap *= target_stake
        if bet.raw_edge < 0:
            cap *= params.neg_mult
        return max(params.c2 * 0.5, 0.01), cap

    return 0.0, 0.0


def round_to_increment(x: float, increment: float) -> float:
    if increment <= 0:
        return x
    return round(x / increment) * increment


def stake_match(
    bets: list[BetRow],
    params: TierParams,
    *,
    target_stake: float,
    min_order_size: float,
    rounding: float,
    cap_mode: str,
) -> list[StakedBet]:
    if not bets:
        return []

    working: list[dict[str, Any]] = []

    for bet in bets:
        break_even_stake = target_stake * bet.market_price
        base_stake = max(min_order_size, break_even_stake)
        weight, cap = tier_weight_and_cap(
            bet,
            params,
            target_stake=target_stake,
            cap_mode=cap_mode,
        )

        working.append(
            {
                "bet": bet,
                "break_even_stake": break_even_stake,
                "base_stake": base_stake,
                "weight": max(0.0, weight),
                "cap": cap,
                "surplus": 0.0,
            }
        )

    base_total = sum(x["base_stake"] for x in working)
    surplus_pool = max(0.0, target_stake - base_total)

    active = {i for i, x in enumerate(working) if x["weight"] > 0}

    while surplus_pool > 1e-12 and active:
        total_weight = sum(working[i]["weight"] for i in active)
        if total_weight <= 0:
            break

        spent = 0.0
        saturated: set[int] = set()

        for i in list(active):
            x = working[i]
            allocation = surplus_pool * x["weight"] / total_weight
            cap = x["cap"]

            if cap is not None:
                remaining_cap = max(0.0, float(cap) - x["surplus"])
                if allocation >= remaining_cap:
                    allocation = remaining_cap
                    saturated.add(i)

            x["surplus"] += allocation
            spent += allocation

        surplus_pool -= spent
        active -= saturated

        if spent <= 1e-12:
            break

    if surplus_pool > 1e-12:
        # Every tier (including VALUE_1's own cap) saturated with money still
        # left over. VALUE_1's cap is soft: it absorbs the remainder anyway so
        # total stake still sums to target_stake, same as the pre-cap behavior.
        value_1_idx = next((i for i, x in enumerate(working) if x["bet"].tier == "VALUE_1"), None)
        if value_1_idx is not None:
            working[value_1_idx]["surplus"] += surplus_pool
            surplus_pool = 0.0

    staked: list[StakedBet] = []

    for x in working:
        stake = x["base_stake"] + x["surplus"]
        stake = max(min_order_size, round_to_increment(stake, rounding))
        stake = round(stake, 2)

        bet = x["bet"]
        gross = stake / bet.market_price if bet.market_price > 0 else 0.0
        net_if_hit = gross - stake

        staked.append(
            StakedBet(
                row=bet,
                break_even_stake=x["break_even_stake"],
                base_stake=x["base_stake"],
                surplus_stake=max(0.0, stake - x["base_stake"]),
                stake=stake,
                gross_payout_if_hit=gross,
                net_profit_if_hit=net_if_hit,
            )
        )

    return staked


# -----------------------------
# Backtest metrics
# -----------------------------

def group_bets_by_match(bets: list[BetRow]) -> dict[str, list[BetRow]]:
    grouped: dict[str, list[BetRow]] = {}
    for bet in bets:
        grouped.setdefault(bet.match_id, []).append(bet)

    for match_id in grouped:
        grouped[match_id].sort(
            key=lambda b: (
                {"VALUE": 0, "COVER": 1, "OUTLIER_VALUE": 2}.get(b.role, 9),
                b.model_rank if b.model_rank is not None else 999,
                -b.model_probability,
            )
        )

    return grouped


def evaluate_match(
    bets: list[BetRow],
    params: TierParams,
    *,
    target_stake: float,
    min_order_size: float,
    rounding: float,
    cap_mode: str,
) -> MatchResult:
    staked = stake_match(
        bets,
        params,
        target_stake=target_stake,
        min_order_size=min_order_size,
        rounding=rounding,
        cap_mode=cap_mode,
    )

    first = bets[0]
    total_stake = sum(x.stake for x in staked)

    hit_bet = next((x for x in staked if x.row.scoreline == first.final_score), None)
    hit = hit_bet is not None

    if hit_bet is None:
        profit = -total_stake
        hit_scoreline = None
        hit_role = None
    else:
        profit = hit_bet.stake / hit_bet.row.market_price - total_stake
        hit_scoreline = hit_bet.row.scoreline
        hit_role = hit_bet.row.role

    value_win_profit = 0.0
    outlier_win_profit = 0.0
    cover_win_profit = 0.0
    gross_positive_profit = 0.0

    if hit_bet is not None:
        gross_win_profit = max(0.0, hit_bet.stake / hit_bet.row.market_price - hit_bet.stake)
        gross_positive_profit += gross_win_profit

        if hit_bet.row.role == "VALUE":
            value_win_profit += gross_win_profit
        elif hit_bet.row.role == "OUTLIER_VALUE":
            outlier_win_profit += gross_win_profit
        elif hit_bet.row.role == "COVER":
            cover_win_profit += gross_win_profit

    cover_leakage = 0.0
    for x in staked:
        if x.row.role == "COVER" and x.row.raw_edge < 0:
            needed_for_card_recovery = total_stake * x.row.market_price
            cover_leakage += max(0.0, x.stake - needed_for_card_recovery)

    return MatchResult(
        match_id=first.match_id,
        kickoff=first.kickoff,
        team_a=first.team_a,
        team_b=first.team_b,
        final_score=first.final_score,
        selected_count=len(staked),
        total_stake=total_stake,
        hit=hit,
        hit_scoreline=hit_scoreline,
        hit_role=hit_role,
        profit=profit,
        value_win_profit=value_win_profit,
        outlier_win_profit=outlier_win_profit,
        cover_win_profit=cover_win_profit,
        gross_positive_profit=gross_positive_profit,
        cover_leakage=cover_leakage,
    )


def max_drawdown(profits: list[float], *, bankroll: float) -> float:
    balance = bankroll
    peak = bankroll
    worst = 0.0

    for profit in profits:
        balance += profit
        peak = max(peak, balance)
        if peak > 0:
            worst = max(worst, (peak - balance) / peak)

    return worst


def utility_from_metrics(metrics: dict[str, float]) -> float:
    return (
        0.35 * metrics["roi"]
        + 0.25 * metrics["break_even_hit_rate"]
        + 0.20 * metrics["value_profit_share"]
        - 0.15 * metrics["max_drawdown"]
        - 0.05 * metrics["cover_leakage_ratio"]
    )


def evaluate_params_on_bets(
    bets: list[BetRow],
    params: TierParams,
    *,
    target_stake: float,
    min_order_size: float,
    rounding: float,
    bankroll: float,
    cap_mode: str,
) -> tuple[dict[str, float], list[MatchResult]]:
    grouped = group_bets_by_match(bets)

    results: list[MatchResult] = []
    for match_id in sorted(grouped, key=lambda mid: (grouped[mid][0].kickoff, mid)):
        results.append(
            evaluate_match(
                grouped[match_id],
                params,
                target_stake=target_stake,
                min_order_size=min_order_size,
                rounding=rounding,
                cap_mode=cap_mode,
            )
        )

    total_profit = sum(r.profit for r in results)
    total_stake = sum(r.total_stake for r in results)

    hit_results = [r for r in results if r.hit]
    break_even_hits = [r for r in hit_results if r.profit >= -1e-9]

    gross_positive_profit = sum(r.gross_positive_profit for r in results)
    value_outlier_win_profit = sum(r.value_win_profit + r.outlier_win_profit for r in results)

    cover_leakage = sum(r.cover_leakage for r in results)

    selected_counts = [r.selected_count for r in results]
    profits = [r.profit for r in results]

    roi = total_profit / total_stake if total_stake > 0 else -999.0
    behr = len(break_even_hits) / len(hit_results) if hit_results else 0.0
    value_profit_share = max(0.0, value_outlier_win_profit) / max(1.0, gross_positive_profit)
    dd = max_drawdown(profits, bankroll=bankroll)
    cover_leakage_ratio = cover_leakage / total_stake if total_stake > 0 else 999.0
    avg_selected = statistics.mean(selected_counts) if selected_counts else 0.0

    metrics = {
        "matches": float(len(results)),
        "total_stake": total_stake,
        "total_profit": total_profit,
        "roi": roi,
        "hit_rate": len(hit_results) / len(results) if results else 0.0,
        "break_even_hit_rate": behr,
        "value_profit_share": value_profit_share,
        "max_drawdown": dd,
        "cover_leakage": cover_leakage,
        "cover_leakage_ratio": cover_leakage_ratio,
        "avg_selected_bets": avg_selected,
        "utility": 0.0,
    }
    metrics["utility"] = utility_from_metrics(metrics)

    return metrics, results


def passes_hard_filters(
    metrics: dict[str, float],
    *,
    min_behr: float,
    min_value_share: float,
    max_drawdown_allowed: float,
    min_avg_bets: float,
    max_avg_bets: float,
    max_cover_leakage_ratio: float,
) -> bool:
    if metrics["break_even_hit_rate"] < min_behr:
        return False
    if metrics["value_profit_share"] < min_value_share:
        return False
    if metrics["max_drawdown"] > max_drawdown_allowed:
        return False
    if metrics["avg_selected_bets"] < min_avg_bets:
        return False
    if metrics["avg_selected_bets"] > max_avg_bets:
        return False
    if metrics["cover_leakage_ratio"] > max_cover_leakage_ratio:
        return False
    return True


# -----------------------------
# Fold logic
# -----------------------------

def split_into_chronological_folds(bets: list[BetRow], *, fold_count: int) -> list[list[str]]:
    grouped = group_bets_by_match(bets)
    match_ids = sorted(grouped, key=lambda mid: (grouped[mid][0].kickoff, mid))

    if fold_count <= 1:
        return [match_ids]

    folds: list[list[str]] = []
    n = len(match_ids)

    for i in range(fold_count):
        start = round(i * n / fold_count)
        end = round((i + 1) * n / fold_count)
        fold = match_ids[start:end]
        if fold:
            folds.append(fold)

    return folds


def filter_bets_to_match_ids(bets: list[BetRow], match_ids: set[str]) -> list[BetRow]:
    return [b for b in bets if b.match_id in match_ids]


def evaluate_stable_utility(
    bets: list[BetRow],
    params: TierParams,
    *,
    fold_count: int,
    target_stake: float,
    min_order_size: float,
    rounding: float,
    bankroll: float,
    cap_mode: str,
) -> tuple[float, list[dict[str, Any]]]:
    folds = split_into_chronological_folds(bets, fold_count=fold_count)

    fold_rows: list[dict[str, Any]] = []
    utilities: list[float] = []

    for idx, fold_match_ids in enumerate(folds, start=1):
        fold_bets = filter_bets_to_match_ids(bets, set(fold_match_ids))
        metrics, _results = evaluate_params_on_bets(
            fold_bets,
            params,
            target_stake=target_stake,
            min_order_size=min_order_size,
            rounding=rounding,
            bankroll=bankroll,
            cap_mode=cap_mode,
        )

        utilities.append(metrics["utility"])

        fold_rows.append(
            {
                "fold": idx,
                "match_count": int(metrics["matches"]),
                **params.as_output_dict(),
                **metrics,
            }
        )

    if not utilities:
        return -999.0, fold_rows

    mean_u = statistics.mean(utilities)
    std_u = statistics.pstdev(utilities) if len(utilities) > 1 else 0.0
    stable_u = mean_u - 0.5 * std_u

    return stable_u, fold_rows


# -----------------------------
# Search spaces
# -----------------------------

def coarse_grid() -> list[TierParams]:
    values = {
        "v2": [0.50, 0.65, 0.80],
        "c1": [0.10, 0.20, 0.35],
        "c2": [0.05, 0.12, 0.22],
        "outlier": [0.25, 0.40, 0.55],
        "c1_cap": [0.20, 0.35, 0.50],
        "c2_cap": [0.10, 0.20, 0.35],
        "neg_mult": [0.35, 0.50, 0.65],
        # VALUE_1's cap is soft (see stake_match): these are the point where
        # it starts ceding surplus to other still-active tiers, not a hard
        # ceiling -- it can still exceed this if everything else saturates.
        "v1_cap": [0.35, 0.55, 0.80],
        "v2_cap": [0.20, 0.35, 0.50],
        "outlier_cap": [0.10, 0.20, 0.35],
    }

    params: list[TierParams] = []
    for combo in itertools.product(
        values["v2"],
        values["c1"],
        values["c2"],
        values["outlier"],
        values["c1_cap"],
        values["c2_cap"],
        values["neg_mult"],
        values["v1_cap"],
        values["v2_cap"],
        values["outlier_cap"],
    ):
        params.append(TierParams(*combo))

    return params


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def sample_around(base: TierParams, rng: random.Random) -> TierParams:
    return TierParams(
        v2=clip(rng.uniform(base.v2 - 0.10, base.v2 + 0.10), 0.40, 0.90),
        c1=clip(rng.uniform(base.c1 - 0.08, base.c1 + 0.08), 0.05, 0.45),
        c2=clip(rng.uniform(base.c2 - 0.06, base.c2 + 0.06), 0.00, 0.30),
        outlier=clip(rng.uniform(base.outlier - 0.12, base.outlier + 0.12), 0.10, 0.70),
        c1_cap=clip(rng.uniform(base.c1_cap - 0.10, base.c1_cap + 0.10), 0.10, 0.60),
        c2_cap=clip(rng.uniform(base.c2_cap - 0.08, base.c2_cap + 0.08), 0.05, 0.40),
        neg_mult=clip(rng.uniform(base.neg_mult - 0.12, base.neg_mult + 0.12), 0.20, 0.70),
        v1_cap=clip(rng.uniform(base.v1_cap - 0.10, base.v1_cap + 0.10), 0.20, 1.00),
        v2_cap=clip(rng.uniform(base.v2_cap - 0.10, base.v2_cap + 0.10), 0.10, 0.70),
        outlier_cap=clip(rng.uniform(base.outlier_cap - 0.08, base.outlier_cap + 0.08), 0.05, 0.50),
    )


def params_key(params: TierParams) -> tuple[float, ...]:
    return (
        round(params.v2, 6),
        round(params.c1, 6),
        round(params.c2, 6),
        round(params.outlier, 6),
        round(params.c1_cap, 6),
        round(params.c2_cap, 6),
        round(params.neg_mult, 6),
        round(params.v1_cap, 6),
        round(params.v2_cap, 6),
        round(params.outlier_cap, 6),
    )


# -----------------------------
# Search runner
# -----------------------------

def evaluate_candidate(
    *,
    bets: list[BetRow],
    params: TierParams,
    phase: str,
    target_stake: float,
    min_order_size: float,
    rounding: float,
    bankroll: float,
    cap_mode: str,
    fold_count: int,
    hard_filter_args: dict[str, float],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[MatchResult]]:
    metrics, match_results = evaluate_params_on_bets(
        bets,
        params,
        target_stake=target_stake,
        min_order_size=min_order_size,
        rounding=rounding,
        bankroll=bankroll,
        cap_mode=cap_mode,
    )

    passed = passes_hard_filters(metrics, **hard_filter_args)

    stable_utility, fold_rows = evaluate_stable_utility(
        bets,
        params,
        fold_count=fold_count,
        target_stake=target_stake,
        min_order_size=min_order_size,
        rounding=rounding,
        bankroll=bankroll,
        cap_mode=cap_mode,
    )

    row = {
        "phase": phase,
        **params.as_output_dict(),
        **metrics,
        "stable_utility": stable_utility,
        "passed_hard_filters": passed,
    }

    return row, fold_rows, match_results


def run_search(
    *,
    bets: list[BetRow],
    outdir: Path,
    target_stake: float,
    min_order_size: float,
    rounding: float,
    bankroll: float,
    cap_mode: str,
    fold_count: int,
    random_trials: int,
    top_k: int,
    seed: int,
    hard_filter_args: dict[str, float],
) -> None:
    rng = random.Random(seed)

    results: list[dict[str, Any]] = []
    all_fold_rows: list[dict[str, Any]] = []

    print("[search] grid search", file=sys.stderr)

    grid_params = coarse_grid()

    for idx, params in enumerate(grid_params, start=1):
        row, fold_rows, _match_results = evaluate_candidate(
            bets=bets,
            params=params,
            phase="grid",
            target_stake=target_stake,
            min_order_size=min_order_size,
            rounding=rounding,
            bankroll=bankroll,
            cap_mode=cap_mode,
            fold_count=fold_count,
            hard_filter_args=hard_filter_args,
        )

        results.append(row)
        all_fold_rows.extend(fold_rows)

        if idx % 100 == 0:
            print(f"[search] grid {idx}/{len(grid_params)}", file=sys.stderr)

    grid_ranked = sorted(
        results,
        key=lambda r: (
            bool(r["passed_hard_filters"]),
            float(r["stable_utility"]),
            float(r["utility"]),
            float(r["roi"]),
        ),
        reverse=True,
    )

    bases: list[TierParams] = []
    for r in grid_ranked[:top_k]:
        bases.append(
            TierParams(
                v2=float(r["tier_value_2_weight"]),
                c1=float(r["tier_cover_1_weight"]),
                c2=float(r["tier_cover_2_weight"]),
                outlier=float(r["tier_outlier_weight"]),
                c1_cap=float(r["tier_cover_1_cap"]),
                c2_cap=float(r["tier_cover_2_cap"]),
                neg_mult=float(r["tier_negative_edge_cap_multiplier"]),
                v1_cap=float(r["tier_value_1_cap"]),
                v2_cap=float(r["tier_value_2_cap"]),
                outlier_cap=float(r["tier_outlier_cap"]),
            )
        )

    print(f"[search] random search around top {len(bases)} grid params", file=sys.stderr)

    seen = {params_key(p) for p in bases}
    for idx in range(1, random_trials + 1):
        base = rng.choice(bases)
        params = sample_around(base, rng)

        key = params_key(params)
        if key in seen:
            continue
        seen.add(key)

        row, fold_rows, _match_results = evaluate_candidate(
            bets=bets,
            params=params,
            phase="random",
            target_stake=target_stake,
            min_order_size=min_order_size,
            rounding=rounding,
            bankroll=bankroll,
            cap_mode=cap_mode,
            fold_count=fold_count,
            hard_filter_args=hard_filter_args,
        )

        results.append(row)
        all_fold_rows.extend(fold_rows)

        if idx % 250 == 0:
            print(f"[search] random {idx}/{random_trials}", file=sys.stderr)

    ranked = sorted(
        results,
        key=lambda r: (
            bool(r["passed_hard_filters"]),
            float(r["stable_utility"]),
            float(r["utility"]),
            float(r["roi"]),
        ),
        reverse=True,
    )

    write_csv(outdir / "parameter_search_results.csv", ranked)
    write_csv(outdir / "fold_results.csv", all_fold_rows)

    if not ranked:
        raise SystemExit("No parameter results produced.")

    best = ranked[0]

    best_params = TierParams(
        v2=float(best["tier_value_2_weight"]),
        c1=float(best["tier_cover_1_weight"]),
        c2=float(best["tier_cover_2_weight"]),
        outlier=float(best["tier_outlier_weight"]),
        c1_cap=float(best["tier_cover_1_cap"]),
        c2_cap=float(best["tier_cover_2_cap"]),
        neg_mult=float(best["tier_negative_edge_cap_multiplier"]),
        v1_cap=float(best["tier_value_1_cap"]),
        v2_cap=float(best["tier_value_2_cap"]),
        outlier_cap=float(best["tier_outlier_cap"]),
    )

    best_metrics, best_match_results = evaluate_params_on_bets(
        bets,
        best_params,
        target_stake=target_stake,
        min_order_size=min_order_size,
        rounding=rounding,
        bankroll=bankroll,
        cap_mode=cap_mode,
    )

    best_payload = {
        "params": best_params.as_output_dict(),
        "metrics": best_metrics,
        "stable_utility": best["stable_utility"],
        "passed_hard_filters": best["passed_hard_filters"],
        "cap_mode": cap_mode,
        "target_stake": target_stake,
        "min_order_size": min_order_size,
        "rounding": rounding,
        "bankroll": bankroll,
        "fold_count": fold_count,
    }

    write_json(outdir / "best_params.json", best_payload)

    equity_rows: list[dict[str, Any]] = []
    balance = bankroll
    peak = bankroll

    for idx, r in enumerate(best_match_results, start=1):
        balance += r.profit
        peak = max(peak, balance)
        dd = (peak - balance) / peak if peak > 0 else 0.0

        equity_rows.append(
            {
                "idx": idx,
                "match_id": r.match_id,
                "kickoff": r.kickoff,
                "team_a": r.team_a,
                "team_b": r.team_b,
                "final_score": r.final_score,
                "hit": r.hit,
                "hit_scoreline": r.hit_scoreline,
                "hit_role": r.hit_role,
                "selected_count": r.selected_count,
                "total_stake": r.total_stake,
                "profit": r.profit,
                "bankroll": balance,
                "peak": peak,
                "drawdown": dd,
            }
        )

    write_csv(outdir / "equity_curves" / "best_equity_curve.csv", equity_rows)

    print("\nBEST PARAMS")
    print(json.dumps(best_payload, indent=2, sort_keys=True))

    print("\nLIVE COMMAND FLAGS")
    print(
        " ".join(
            [
                f"--tier-value-2-weight {best_params.v2:.4f}",
                f"--tier-cover-1-weight {best_params.c1:.4f}",
                f"--tier-cover-2-weight {best_params.c2:.4f}",
                f"--tier-outlier-weight {best_params.outlier:.4f}",
                f"--tier-cover-1-cap {best_params.c1_cap:.4f}",
                f"--tier-cover-2-cap {best_params.c2_cap:.4f}",
                f"--tier-negative-edge-cap-multiplier {best_params.neg_mult:.4f}",
                f"--tier-value-1-cap {best_params.v1_cap:.4f}",
                f"--tier-value-2-cap {best_params.v2_cap:.4f}",
                f"--tier-outlier-cap {best_params.outlier_cap:.4f}",
            ]
        )
    )


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimise V46.4 tiered-balanced staking parameters using cached historical cards."
    )

    parser.add_argument("--v46-script", default="v46_4_optimisedbestbuys.py")
    parser.add_argument("--fotmob-match-facts", default="data/fotmob_match_facts_clean.csv")
    parser.add_argument("--price-history-root", default="data/polymarket_prematch_snapshots_from_fotmob")
    parser.add_argument("--outdir", default="outputs/v46_4_tier_optimization")

    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--max-matches", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Run this many V46.4 cache-build subprocess calls concurrently. Only affects the "
        "per-match cache-building loop, not the grid/random search (which is in-memory and "
        "already fast). 1 = sequential (default, unchanged behavior).",
    )

    parser.add_argument("--target-stake", type=float, default=5.0)
    parser.add_argument("--min-order-size", type=float, default=1.0)
    parser.add_argument("--rounding", type=float, default=0.01)
    parser.add_argument("--bankroll", type=float, default=100.0)

    parser.add_argument(
        "--cap-mode",
        choices=["absolute", "fraction"],
        default="absolute",
        help="absolute matches current V46.4 style; fraction interprets caps as fractions of target stake.",
    )

    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--random-trials", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=46)

    parser.add_argument("--min-behr", type=float, default=0.92)
    parser.add_argument("--min-value-share", type=float, default=0.65)
    parser.add_argument("--max-drawdown", type=float, default=0.40)
    parser.add_argument("--min-avg-bets", type=float, default=4.0)
    parser.add_argument("--max-avg-bets", type=float, default=5.0)
    parser.add_argument("--max-cover-leakage-ratio", type=float, default=0.15)

    parser.add_argument(
        "--extra-v46-arg",
        action="append",
        default=[],
        help="Extra single argument to pass to V46.4. Repeat this flag if needed.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    hard_filter_args = {
        "min_behr": args.min_behr,
        "min_value_share": args.min_value_share,
        "max_drawdown_allowed": args.max_drawdown,
        "min_avg_bets": args.min_avg_bets,
        "max_avg_bets": args.max_avg_bets,
        "max_cover_leakage_ratio": args.max_cover_leakage_ratio,
    }

    bets = build_or_load_training_cache(
        fotmob_path=Path(args.fotmob_match_facts),
        script=Path(args.v46_script),
        price_history_root=Path(args.price_history_root),
        outdir=outdir,
        force_cache=args.force_cache,
        max_matches=args.max_matches,
        extra_v46_args=list(args.extra_v46_arg or []),
        target_stake=args.target_stake,
        min_order_size=args.min_order_size,
        rounding=args.rounding,
        workers=args.workers,
    )

    match_count = len(set(b.match_id for b in bets))
    if match_count < 10:
        raise SystemExit(
            f"Only {match_count} cached matches found. Need more historical cards before tuning."
        )

    print(f"[data] bet rows: {len(bets)}", file=sys.stderr)
    print(f"[data] matches: {match_count}", file=sys.stderr)

    run_search(
        bets=bets,
        outdir=outdir,
        target_stake=args.target_stake,
        min_order_size=args.min_order_size,
        rounding=args.rounding,
        bankroll=args.bankroll,
        cap_mode=args.cap_mode,
        fold_count=args.fold_count,
        random_trials=args.random_trials,
        top_k=args.top_k,
        seed=args.seed,
        hard_filter_args=hard_filter_args,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

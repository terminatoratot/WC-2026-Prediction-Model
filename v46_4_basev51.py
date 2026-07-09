#!/usr/bin/env python3
"""V46.4-BaseV51: optimized best-buy card for exact-score markets, on a V51 prediction base.

This is a sibling of v46_4_optimisedbestbuys.py (and of v46_4_basev29.py,
which established this exact composition pattern) that keeps the exact same
buy-and-hold staking/Kelly/tiering objective, but swaps the hidden
scoreline-prediction layer from V42 (V36 FotMob current-form + V39/V39BD
coverage-outlier stack) to V51. V51 is v11's baseline model with v49's
bivariate-NegBin scoreline correction always on, plus two additive
(non-replacing) outlier tabs -- v39's coverage-outlier and v29's tail-risk
outlier -- layered on top of an untouched Top-3. It was validated via a full
expanding-window backtest (2010-2026) to preserve v11+v49's own top-3/result
accuracy while adding incremental coverage. So this file is "buy logic stays
the same, prediction source becomes V51 (v11+v49, plus additive v39/v29
outlier coverage)."

It reuses V42's Polymarket fetching/classification/edge/staking-confidence/
book-consistency pipeline as-is (imported, not reimplemented) -- only V42's
own outcome-prediction model construction is replaced. Critically, V51 does
NOT reuse or invoke V42's own outcome-prediction/accuracy stack (V36 FotMob
current-form model + V39/V39BD coverage-outlier blend) anywhere -- V51 is
built from v11 (+v49/v39/v29), never from v42's model layer. See
run_v51_for_v46() below for the swapped hidden layer.

Everything from "no Any Other Score by default; ignore 0-0, 0-1, 1-0" through
the final staking/Kelly/tiering/audit-output behavior is unchanged from
v46_4_optimisedbestbuys.py:

- no Any Other Score; ignore 0-0, 0-1, and 1-0 by default
- select a readable 4-5 exact-score card from top probability, value/Kelly support, and the base outlier
- classify selected scores as VALUE, COVER, or OUTLIER_VALUE
- fund every selected score at a break-even floor, then allocate surplus toward value/outlier value
- add audit outputs: role reasons, hit-outcome table, selection-debug CSV, and final validation checks

This file is a new, fully standalone, fully removable sibling: it does not
modify v46_4_optimisedbestbuys.py, v46_4_basev29.py, v42_fotmob_market_edge_model.py,
v51_combined_scoreline_model.py, v29_tail_risk_scoreline_model.py,
v39_coverage_outlier_model.py, or v11_wcq_results_model.py in any way. Deleting
this one file removes the entire V51-based card with no effect on any other model.



cd /Users/rajeevagrawal/Downloads/world_cup_advanced_model_package

MPLCONFIGDIR=.matplotlib_cache .venv/bin/python v46_4_basev51.py \
  --team-a "Brazil" \
  --team-b "Norway" \
  --knockout \
  --outdir outputs/outputs_v46_4_basev51_brazil_norway_live_polymarket_tiered_opt_7eur \
  --auto-polymarket \
  --polymarket-event-slug fifwc-bra-nor-2026-07-05 \
  --fetch-clob-orderbook \
  --stake-profile tiered-balanced \
  --execution-target-stake 7.0 \
  --stake-rounding 0.01 \
  --buy-hold-rank-limit 16 \
  --probability-cover-count 8 \
  --buy-hold-min-bets 5 \
  --buy-hold-max-bets 5 \
  --min-confidence 0.0 \
  --min-story-fit 0.0 \
  --tier-value-2-weight 0.8895 \
  --tier-cover-1-weight 0.0500 \
  --tier-cover-2-weight 0.0060 \
  --tier-outlier-weight 0.1416 \
  --tier-cover-1-cap 0.2415 \
  --tier-cover-2-cap 0.1834 \
  --tier-negative-edge-cap-multiplier 0.4070 \
  --ignore-scorelines 0-0,0-1,1-0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

# v11 (imported below, directly and transitively via v51) lives under versions/.
_VERSIONS_DIR = Path(__file__).resolve().parent / "versions"
if str(_VERSIONS_DIR) not in sys.path:
    sys.path.insert(0, str(_VERSIONS_DIR))

import v11_wcq_results_model as v11
import v42_fotmob_market_edge_model as v42
import v51_combined_scoreline_model as v51


def _to_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "none", "null"}:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def normalize_polymarket_event_slug(slug: str | None) -> str | None:
    """Normalize short match codes into Polymarket's full event slug format."""
    if slug is None:
        return None
    cleaned = slug.strip()
    if cleaned == "prt-hrv":
        return "fifwc-prt-hrv-2026-07-02"
    return cleaned


def run_v51_for_v46(args: argparse.Namespace, passthrough_args: list[str]) -> Path:
    """Build the V51 combined-scoreline prediction, run V42's Polymarket pipeline against it, return the dir.

    This is the hidden data/model layer for V46.4-BaseV51. The scoreline
    probability matrix comes entirely from V51's base model (v11's
    StrongWorldCupModel with v49's bivariate-NegBin scoreline correction
    always on) -- V51's own prediction is never touched. The Top-3/outlier
    candidate list fed downstream is V51's `v51_combined_top_scorelines`
    (the untouched Top-3 plus up to two additive outlier picks -- one from
    v39's coverage-outlier selector, one from v29's tail-risk selector --
    deduped), which is the wider coverage set worth pricing against the
    market. Turning that prediction into Polymarket-comparable exact-score
    edges, staking confidence, and book-consistency flags reuses V42's
    functions verbatim (imported from v42_fotmob_market_edge_model); V42's
    own outcome-prediction model (V36 FotMob current-form + V39/V39BD
    coverage-outlier stack) is not used anywhere in this function -- only
    V42's Polymarket fetching/classification/edge-computation machinery is
    imported and called.
    """
    del passthrough_args
    if args.input:
        return Path(args.input)

    if not args.team_a or not args.team_b:
        raise SystemExit("--team-a and --team-b are required unless --input points at an existing V51 output.")

    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    outdir = Path(args.outdir)
    output_dir = Path(args.v51_outdir) if args.v51_outdir else outdir / "_v51_source" / timestamp_slug()
    output_dir.mkdir(parents=True, exist_ok=True)

    model, _ = v51.build_from_zip(
        args.worldcupsai_zip or str(data_dir / "worldcupsai.zip"),
        train_csv=args.team_train or str(data_dir / "current_team_features_2026.csv"),
        test_csv=args.team_test,
        box_csv=args.box_data or str(data_dir / "FIFAallMatchBoxData.csv"),
        results_csv=args.results_data or str(data_dir / "results.csv"),
        former_names_csv=args.former_names or str(data_dir / "former_names.csv"),
        prediction_year=args.prediction_year or 2026,
        wc_prestige_weight=args.wc_prestige_weight,
        use_expanded_training_pool=True,
        use_volume_normalized_weighting=args.use_volume_normalized_weighting,
        score_matrix_r=args.score_matrix_r,
        fbref_world_cup_csv=args.fbref_world_cup_csv or str(data_dir / "fbref_world_cup_matches.csv"),
        fbref_international_csv=args.fbref_international_csv or str(data_dir / "fbref_international_matches.csv"),
    )
    prediction = model.predict(
        args.team_a,
        args.team_b,
        host_a=args.host_a,
        host_b=args.host_b,
        knockout=args.knockout,
    )

    # --- BEGIN OUTLIER-tier routing (opt-in override, on by default) --------------------
    # v51's v29 tail-risk / v39 coverage-outlier picks (when their gates fire) already
    # land in prediction["v51_combined_top_scorelines"] as an extra CORE_TOPn candidate,
    # but build_buy_hold_candidates()/select_buy_hold_candidates() need a scoreline
    # specifically tagged as the OUTLIER role (for the OUTLIER tier weight and
    # --require-outlier) -- that tag lives in summary["prediction_outlier"] below.
    # Prefer v29's genuine tail-risk pick (a deliberate blowout hedge) over v39's
    # coverage-outlier (a plain total-goals envelope fill) when both fired, since v29 is
    # closer to what "OUTLIER" is meant to mean here. --force-outlier-scoreline lets you
    # designate a specific scoreline as the OUTLIER pick even when neither gate fired for
    # this particular match (its probability is looked up from v51's own score matrix).
    v51_adjustments = prediction.get("v51_adjustments", {})
    outlier_pick = v51_adjustments.get("v29_outlier") or v51_adjustments.get("v39_outlier")
    if args.force_outlier_scoreline:
        forced_tuple = score_tuple(args.force_outlier_scoreline)
        if forced_tuple is None:
            raise SystemExit(f"--force-outlier-scoreline must look like '2-1', got: {args.force_outlier_scoreline!r}")
        forced_prob = next(
            (
                float(item["probability"])
                for item in prediction.get("scoreline_probabilities", [])
                if int(item["team_a_goals"]) == forced_tuple[0] and int(item["team_b_goals"]) == forced_tuple[1]
            ),
            0.0,
        )
        outlier_pick = {"team_a_goals": forced_tuple[0], "team_b_goals": forced_tuple[1], "probability": forced_prob, "source": "forced"}
    # --- END OUTLIER-tier routing --------------------------------------------------------

    # From here down this mirrors v42_fotmob_market_edge_model.main()'s Polymarket
    # fetch/classify/edge/staking-confidence pipeline exactly (functions imported,
    # not reimplemented) -- only the model-construction step above differs from V42.
    base_matrix = v42.normalize_matrix(v42.score_matrix_from_prediction(prediction))

    event_slug_arg = normalize_polymarket_event_slug(args.polymarket_event_slug)
    sports_url = args.polymarket_sports_url
    if not any(
        [args.polymarket_query, event_slug_arg, args.polymarket_sports_url, args.polymarket_json, args.no_fetch_polymarket]
    ):
        sports_url = v42.DEFAULT_POLYMARKET_SPORTS_GAMES_URL

    raw_markets, fetch_meta = v42.load_or_fetch_markets(
        team_a=args.team_a,
        team_b=args.team_b,
        query=args.polymarket_query,
        event_slug=event_slug_arg,
        sports_url=sports_url,
        json_path=args.polymarket_json,
        no_fetch=args.no_fetch_polymarket,
        limit=args.gamma_limit or v42.DEFAULT_GAMMA_LIMIT,
    )
    (output_dir / "polymarket_raw_markets.json").write_text(json.dumps(raw_markets, indent=2), encoding="utf-8")
    match_markets = v42.filter_match_markets(raw_markets, args.team_a, args.team_b)
    moneyline_markets = [
        row for row in (v42.classify_moneyline_market(market, args.team_a, args.team_b) for market in match_markets) if row
    ]
    btts_markets = [row for row in (v42.classify_btts_market(market) for market in match_markets) if row]
    spread_markets = [
        row for row in (v42.classify_spread_market(market, args.team_a, args.team_b) for market in match_markets) if row
    ]
    total_markets = [row for row in (v42.classify_total_market(market) for market in match_markets) if row]
    exact_markets = [
        row for row in (v42.classify_exact_score_market(market, args.team_a, args.team_b) for market in match_markets) if row
    ]
    total_ladder = v42.monotone_over_ladder(total_markets)
    market_distribution_rows = v42.market_total_distribution(total_ladder)

    fit = None
    market_total_reference_matrix = base_matrix
    total_comparison_rows: list[dict[str, Any]] = []
    if len(market_distribution_rows) >= 3:
        fit = v42.fit_total_tilt_beta(base_matrix, market_distribution_rows)
        market_total_reference_matrix = v42.tilt_matrix_by_beta(base_matrix, float(fit["beta"]))
        model_totals = v42.aggregate_total_buckets(base_matrix, market_distribution_rows)
        tilted_totals = v42.aggregate_total_buckets(market_total_reference_matrix, market_distribution_rows)
        for row in market_distribution_rows:
            label = str(row["total_bucket"])
            total_comparison_rows.append(
                {
                    **row,
                    "model_probability": model_totals.get(label, 0.0),
                    "tilted_model_probability": tilted_totals.get(label, 0.0),
                    "market_minus_model": float(row["market_probability"]) - model_totals.get(label, 0.0),
                    "market_minus_tilted": float(row["market_probability"]) - tilted_totals.get(label, 0.0),
                }
            )

    book_context = v42.build_book_context(moneyline_markets, btts_markets, spread_markets, market_distribution_rows)
    market_implied_matrix = None
    market_matrix_meta = None
    if market_distribution_rows and (moneyline_markets or btts_markets or spread_markets):
        market_implied_matrix, market_matrix_meta = v42.build_market_implied_score_matrix(
            market_total_reference_matrix,
            book_context,
        )

    min_edge = args.min_edge if args.min_edge is not None else v42.DEFAULT_MIN_EXACT_EDGE
    min_ev = args.min_ev if args.min_ev is not None else v42.DEFAULT_MIN_EV
    uncertainty_buffer = args.uncertainty_buffer if args.uncertainty_buffer is not None else v42.DEFAULT_UNCERTAINTY_BUFFER
    exact_edges = v42.edge_rows(
        base_matrix,
        base_matrix,
        exact_markets,
        min_edge=min_edge,
        min_ev=min_ev,
        uncertainty_buffer=uncertainty_buffer,
        market_implied_matrix=market_implied_matrix,
        market_matrix_meta=market_matrix_meta,
        market_total_reference_matrix=market_total_reference_matrix,
    )
    exact_edges = v42.add_book_consistency(exact_edges, book_context)
    exact_edges = v42.add_staking_confidence(exact_edges)
    event_slug = v42.current_event_slug(fetch_meta) or v42.infer_event_slug_from_exact_rows(exact_edges)
    exact_edges = v42.add_entry_price_columns(exact_edges, uncertainty_buffer)
    current_exact_edges = v42.current_exact_score_rows(exact_edges, event_slug)
    previous_price_rows, price_history_meta = v42.find_previous_price_rows(
        args.price_history_root,
        current_output_dir=output_dir,
        event_slug=event_slug,
    )
    exact_edges = v42.add_price_movement_columns(exact_edges, previous_price_rows)
    current_exact_edges = v42.current_exact_score_rows(exact_edges, event_slug)
    reference_rows = v42.load_reference_model_rows(
        args.reference_v43_output,
        team_a=args.team_a,
        team_b=args.team_b,
    )
    if reference_rows:
        exact_edges = v42.add_reference_model_agreement(exact_edges, reference_rows, uncertainty_buffer)
        current_exact_edges = v42.current_exact_score_rows(exact_edges, event_slug)
    if args.fetch_clob_orderbook:
        current_exact_edges = v42.add_clob_quotes(current_exact_edges, uncertainty_buffer)
    exact_score_baskets = v42.build_exact_score_baskets(
        current_exact_edges,
        team_a=args.team_a,
        team_b=args.team_b,
        uncertainty_buffer=uncertainty_buffer,
    )
    current_exact_edges_with_other = v42.current_exact_score_rows(exact_edges, event_slug, include_any_other=True)
    exact_score_decision_board = v42.build_decision_board_rows(current_exact_edges_with_other)
    v42.write_price_snapshot(
        output_dir / "price_snapshot.csv",
        current_exact_edges,
        event_slug=event_slug,
        team_a=args.team_a,
        team_b=args.team_b,
    )
    v42.write_decision_brief(
        output_dir / "decision_brief.md",
        team_a=args.team_a,
        team_b=args.team_b,
        event_slug=event_slug,
        decision_rows=exact_score_decision_board,
        basket_rows=exact_score_baskets,
        market_matrix_meta=market_matrix_meta,
        price_history_meta=price_history_meta,
    )

    fair_score_rows = [
        {
            "team_a_goals": key[0],
            "team_b_goals": key[1],
            "scoreline": f"{key[0]}-{key[1]}",
            "base_model_probability": base_matrix.get(key, 0.0),
            "model_only_fair_probability": base_matrix.get(key, 0.0),
            "scoreline_fair_probability": base_matrix.get(key, 0.0),
            "market_total_tilt_probability": base_matrix.get(key, 0.0),
            "polymarket_total_tilt_reference_probability": market_total_reference_matrix.get(key, 0.0),
            "polymarket_reference_probability_change": market_total_reference_matrix.get(key, 0.0) - base_matrix.get(key, 0.0),
        }
        for key in sorted(base_matrix, key=lambda item: base_matrix[item], reverse=True)
    ]

    v42.write_csv(output_dir / "polymarket_total_ladder.csv", total_ladder)
    v42.write_csv(output_dir / "polymarket_market_total_distribution.csv", market_distribution_rows)
    v42.write_csv(output_dir / "model_vs_market_total_distribution.csv", total_comparison_rows)
    v42.write_csv(output_dir / "polymarket_exact_score_edges.csv", exact_edges)
    v42.write_csv(output_dir / "exact_score_decision_board.csv", exact_score_decision_board)
    v42.write_csv(output_dir / "exact_score_baskets.csv", exact_score_baskets)
    v42.write_csv(output_dir / "model_fair_scoreline_probabilities.csv", fair_score_rows)
    (output_dir / "single_match_prediction.json").write_text(json.dumps(prediction, indent=2), encoding="utf-8")

    if not args.no_plots_v51_layer:
        v11.plot_prediction_outputs(prediction, output_dir)

    summary = {
        "version": "v46_4_basev51-hidden-layer",
        "base_model": "v51-combined-scoreline (v11+v49 bivariate-negbin base, additive v39/v29 outlier tabs)",
        "team_a": args.team_a,
        "team_b": args.team_b,
        # v51_combined_top_scorelines is the untouched Top-3 plus up to two additive
        # outlier picks (v39 coverage-outlier, v29 tail-risk), deduped -- the wider
        # coverage set worth pricing against the market, per v51's own docstring.
        # build_core_rows() below enumerates whatever list is here as CORE_TOPn rows,
        # so this can carry 3-5 entries without any other pipeline code changing.
        "prediction_top_3": prediction.get("v51_combined_top_scorelines", []),
        "prediction_outlier": outlier_pick,
        "result_probabilities": prediction.get("result_probabilities"),
        "v51_adjustments": prediction.get("v51_adjustments", {}),
        "polymarket_fetch": fetch_meta,
        "raw_market_count": len(raw_markets),
        "match_market_count": len(match_markets),
        "moneyline_market_count": len(moneyline_markets),
        "btts_market_count": len(btts_markets),
        "spread_market_count": len(spread_markets),
        "total_market_count": len(total_markets),
        "total_ladder_count": len(total_ladder),
        "exact_score_market_count": len(exact_markets),
        "market_total_distribution_available": bool(market_distribution_rows),
        "tilt_fit": fit,
        "market_implied_matrix": market_matrix_meta,
        "edge_filters": {
            "min_edge": min_edge,
            "min_ev": min_ev,
            "uncertainty_buffer": uncertainty_buffer,
        },
        "staking_confidence": {
            "policy": "confidence_shrunk_fractional_kelly",
            "applies_to": "joint_kelly_sizing_only",
            "components": [
                "total_goal_bucket_reliability",
                "market_liquidity",
                "market_implied_matrix_agreement",
                "base_vs_market_total_tilt_stability",
                "book_consistency",
            ],
        },
        "market_usage_policy": {
            "fair_probability": "model_only",
            "edge_formula": "fair_probability - posted_polymarket_price",
            "polymarket_changes_model": False,
            "market_total_ladder": "diagnostic_reference_only",
            "market_implied_matrix": "diagnostic_reference_only",
        },
        "positive_edge_count": sum(1 for row in exact_edges if row.get("passes_edge_filter")),
        "book_consistency_conflict_count": sum(1 for row in exact_edges if row.get("book_consistency_flags")),
        "book_ok_buy_count": sum(1 for row in exact_edges if row.get("book_adjusted_verdict") == "buy_book_ok"),
        "confidence_kelly_active_count": sum(
            1 for row in exact_edges if float(row.get("confidence_quarter_joint_kelly_fraction") or 0.0) > 0.0
        ),
        "decision_board": {
            "current_event_slug": event_slug,
            "rows": len(exact_score_decision_board),
            "buy_count": sum(1 for row in exact_score_decision_board if row.get("tier") == "BUY"),
            "watch_count": sum(1 for row in exact_score_decision_board if row.get("tier") == "WATCH"),
            "basket_count": len(exact_score_baskets),
            "basket_buy_count": sum(1 for row in exact_score_baskets if row.get("basket_verdict") == "buy_basket"),
            "reference_v43_output": args.reference_v43_output,
            "reference_v43_rows": len(reference_rows),
            "clob_orderbook_fetched": bool(args.fetch_clob_orderbook),
            "price_history": price_history_meta,
        },
        "top_edges": exact_edges[:10],
        "outputs": {
            "total_ladder_csv": str(output_dir / "polymarket_total_ladder.csv"),
            "market_total_distribution_csv": str(output_dir / "polymarket_market_total_distribution.csv"),
            "total_comparison_csv": str(output_dir / "model_vs_market_total_distribution.csv"),
            "exact_score_edges_csv": str(output_dir / "polymarket_exact_score_edges.csv"),
            "exact_score_decision_board_csv": str(output_dir / "exact_score_decision_board.csv"),
            "exact_score_baskets_csv": str(output_dir / "exact_score_baskets.csv"),
            "price_snapshot_csv": str(output_dir / "price_snapshot.csv"),
            "decision_brief_md": str(output_dir / "decision_brief.md"),
            "model_fair_scorelines_csv": str(output_dir / "model_fair_scoreline_probabilities.csv"),
            "single_match_prediction_json": str(output_dir / "single_match_prediction.json"),
        },
    }
    (output_dir / "model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.show_v51_output:
        print(json.dumps(summary, indent=2))
    return output_dir


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def validate_polymarket_inputs(
    summary: dict[str, Any],
    market_rows: list[dict[str, Any]],
    input_dir: Path,
    *,
    allow_empty: bool,
) -> None:
    if allow_empty:
        return

    fetch = summary.get("polymarket_fetch") or {}
    raw_count = int(summary.get("raw_market_count") or 0)
    exact_count = int(summary.get("exact_score_market_count") or 0)
    problems = []
    if raw_count <= 0:
        problems.append("Polymarket raw_market_count is 0")
    if exact_count <= 0 or not market_rows:
        problems.append("Polymarket exact-score markets are empty")
    if not problems:
        return

    details = [
        "V46.4 requires live Polymarket market data before writing a buy card.",
        f"Input dir: {input_dir}",
        f"Problems: {', '.join(problems)}",
        f"Fetch source: {fetch.get('source', '')}",
        f"Fetch reason: {fetch.get('reason', '')}",
        f"Fetch errors: {_compact_json(fetch.get('errors', []))}",
        f"Matched slug: {fetch.get('slug', '')}",
        f"Matched candidates: {_compact_json(fetch.get('matched_candidates', []))}",
        f"Related slug candidates: {_compact_json(fetch.get('related_slug_candidates', []))}",
        f"raw_market_count={raw_count}, exact_score_market_count={exact_count}, exact_score_rows={len(market_rows)}",
    ]
    raise RuntimeError(
        "\n".join(details)
        + "\nUse network access and a valid --polymarket-event-slug/--auto-polymarket source, "
        "or pass --allow-empty-polymarket only for offline/model-only debugging."
    )


def pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.{digits}f}%"


def cents(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.{digits}f}c"


def scoreline_from_goals(row: dict[str, Any]) -> str:
    a_goals = int(row["team_a_goals"])
    b_goals = int(row["team_b_goals"])
    return f"{a_goals}-{b_goals}"


def score_tuple(scoreline: str) -> tuple[int, int] | None:
    if "-" not in scoreline:
        return None
    left, right = scoreline.split("-", 1)
    try:
        return int(left), int(right)
    except ValueError:
        return None


def liquidity_confidence(liquidity: float | None) -> float:
    if liquidity is None or liquidity <= 0:
        return 0.35
    return max(0.35, min(1.0, (math.log10(float(liquidity) + 1.0) - 2.0) / 4.0))


def reliability_from_scoreline(scoreline: str, row: dict[str, Any] | None = None) -> float:
    if scoreline == "Any Other Score":
        return 0.35
    if row:
        value = _to_float(row.get("reliability_weight"))
        if value is not None:
            return max(0.10, min(1.0, value))
    parsed = score_tuple(scoreline)
    if parsed is None:
        return 0.50
    total = parsed[0] + parsed[1]
    margin = abs(parsed[0] - parsed[1])
    if total <= 3 and margin < 3:
        return 1.00
    if total <= 3:
        return 0.75
    if total == 4:
        return 0.50
    return 0.25


def story_fit(
    scoreline: str,
    *,
    favorite_side: str,
    knockout: bool,
) -> tuple[float, str]:
    if scoreline == "Any Other Score":
        return (1.25 if knockout else 1.10), "broad chaos/tail bucket"
    parsed = score_tuple(scoreline)
    if parsed is None:
        return 0.75, "unparsed scoreline"
    a_goals, b_goals = parsed
    total = a_goals + b_goals
    pieces: list[str] = []
    fit = 1.0

    if favorite_side == "team_a" and b_goals > a_goals:
        fit *= 1.25 if knockout else 1.10
        pieces.append("underdog upset tail")
    elif favorite_side == "team_b" and a_goals > b_goals:
        fit *= 1.25 if knockout else 1.10
        pieces.append("underdog upset tail")
    if a_goals == b_goals:
        fit *= 1.10 if knockout else 1.00
        pieces.append("draw/resistance path")
    if a_goals > 0 and b_goals > 0:
        fit *= 1.08
        pieces.append("clean-sheet break")
    if total >= 5:
        fit *= 1.15
        pieces.append("high-total chaos")
    elif total == 4:
        fit *= 1.06
        pieces.append("outlier total")

    if not pieces:
        pieces.append("plain score path")
    return min(1.75, fit), "; ".join(pieces)


def detect_favorite(summary: dict[str, Any]) -> str:
    probs = summary.get("result_probabilities") or {}
    team_a = _to_float(probs.get("team_a_win"), 0.0) or 0.0
    team_b = _to_float(probs.get("team_b_win"), 0.0) or 0.0
    return "team_a" if team_a >= team_b else "team_b"


def is_any_other_score(scoreline: str) -> bool:
    return scoreline.strip().lower() == "any other score"


def fair_probability_from_market_row(row: dict[str, Any] | None) -> float | None:
    if not row:
        return None
    return _to_float(
        row.get("scoreline_fair_probability"),
        _to_float(row.get("model_only_fair_probability"), _to_float(row.get("model_probability"))),
    )


def market_price_from_row(row: dict[str, Any] | None) -> float | None:
    if not row:
        return None
    return _to_float(row.get("raw_yes_price"), _to_float(row.get("yes_price"), _to_float(row.get("market_price"))))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def kelly_shrink_multiplier(
    row: dict[str, Any] | None,
    *,
    min_shrink: float,
    max_shrink: float,
) -> tuple[float, str]:
    """Bucket-specific Kelly shrink, applied after the joint buy-set solve.

    V42 already writes staking_confidence as a composite proxy for bucket depth,
    liquidity, market agreement, model stability, and book consistency. That is
    not a closed-form posterior-variance shrink, but it is the local uncertainty
    proxy available in the V42 output, so V46.4 applies it to the joint stake
    vector instead of applying a flat 1/4 multiplier per score.
    """
    if row:
        value = _to_float(row.get("staking_confidence"))
        if value is not None:
            return clamp(value, min_shrink, max_shrink), "staking_confidence"
        reliability = _to_float(row.get("reliability_weight"))
        if reliability is not None:
            return clamp(reliability, min_shrink, max_shrink), "reliability_weight"
    return clamp(0.50, min_shrink, max_shrink), "fallback"


def solve_capped_shrunk_joint_kelly(
    active_items: list[dict[str, Any]],
    *,
    tranche_budget: float,
) -> dict[str, Any]:
    """Cap the post-shrink joint-Kelly vector by re-solving the capped KKT system.

    The shrink layer is kept untouched. Let g_i be the already-shrunk stake.
    We reinterpret g_i as an unconstrained joint-Kelly solution for effective
    probabilities p_tilde_i. Then we solve the same mutually-exclusive capped
    Kelly problem and iteratively remove names whose capped KKT stake is <= 0.
    """
    active_items = [item for item in active_items if float(item.get("shrunk_fraction", 0.0)) > 0.0]
    shrunk_total = sum(float(item["shrunk_fraction"]) for item in active_items)
    if not active_items:
        return {
            "method": "none",
            "binding": False,
            "lambda": None,
            "effective_threshold": None,
            "survivors": [],
            "excluded": [],
            "fractions": {},
        }
    if tranche_budget <= 0 or shrunk_total <= tranche_budget:
        return {
            "method": "uncapped_shrunk_joint_kelly",
            "binding": False,
            "lambda": 1.0,
            "effective_threshold": 1.0 - shrunk_total,
            "survivors": [str(item["scoreline"]) for item in active_items],
            "excluded": [],
            "fractions": {str(item["scoreline"]): float(item["shrunk_fraction"]) for item in active_items},
        }

    c_total = sum(float(item["price"]) for item in active_items)
    # Effective threshold from the algebra: theta_tilde = 1 - G.
    theta_tilde = 1.0 - shrunk_total
    for item in active_items:
        item["effective_probability"] = float(item["shrunk_fraction"]) + float(item["price"]) * theta_tilde

    remaining = active_items[:]
    excluded: list[str] = []
    last_lambda: float | None = None
    last_raw: dict[str, float] = {}

    while remaining:
        p_sum = sum(float(item["effective_probability"]) for item in remaining)
        c_sum = sum(float(item["price"]) for item in remaining)
        denominator = tranche_budget * (1.0 - c_sum) + c_sum
        if denominator <= 0:
            break
        lam = p_sum / denominator
        last_lambda = lam
        raw_fractions = {
            str(item["scoreline"]): float(item["effective_probability"]) / lam
            - float(item["price"]) * (1.0 - tranche_budget)
            for item in remaining
        }
        last_raw = raw_fractions
        negative_scores = [score for score, value in raw_fractions.items() if value <= 1e-12]
        if not negative_scores:
            final_fractions = {score: max(0.0, value) for score, value in raw_fractions.items()}
            final_total = sum(final_fractions.values())
            if final_total > 0:
                scale = tranche_budget / final_total
                final_fractions = {score: value * scale for score, value in final_fractions.items()}
            return {
                "method": "effective_probability_capped_joint_kelly",
                "binding": True,
                "lambda": lam,
                "effective_threshold": theta_tilde,
                "survivors": sorted(final_fractions, key=final_fractions.get, reverse=True),
                "excluded": excluded,
                "fractions": final_fractions,
                "raw_capped_fractions_last_pass": raw_fractions,
            }
        excluded.extend(negative_scores)
        remaining = [item for item in remaining if str(item["scoreline"]) not in set(negative_scores)]

    return {
        "method": "effective_probability_capped_joint_kelly",
        "binding": True,
        "lambda": last_lambda,
        "effective_threshold": theta_tilde,
        "survivors": [],
        "excluded": excluded,
        "fractions": {},
        "raw_capped_fractions_last_pass": last_raw,
    }


def compute_joint_kelly_index(
    market_rows: list[dict[str, Any]],
    *,
    min_shrink: float,
    max_shrink: float,
    tranche_budget: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in market_rows:
        scoreline = str(row.get("scoreline") or "")
        fair = fair_probability_from_market_row(row)
        price = market_price_from_row(row)
        if not scoreline or fair is None or price is None or fair <= 0 or price <= 0 or price >= 1:
            continue
        candidates.append(
            {
                "scoreline": scoreline,
                "fair": float(fair),
                "price": float(price),
                "ratio": float(fair) / float(price),
                "row": row,
            }
        )
    candidates.sort(key=lambda item: item["ratio"], reverse=True)

    buy_set: list[dict[str, Any]] = []
    threshold = 1.0
    for candidate in candidates:
        trial = buy_set + [candidate]
        p_sum = sum(float(item["fair"]) for item in trial)
        c_sum = sum(float(item["price"]) for item in trial)
        if c_sum >= 1.0:
            break
        trial_threshold = (1.0 - p_sum) / (1.0 - c_sum)
        if candidate["ratio"] > trial_threshold:
            buy_set.append(candidate)
            threshold = trial_threshold
        else:
            break

    p_buy = sum(float(item["fair"]) for item in buy_set)
    c_buy = sum(float(item["price"]) for item in buy_set)
    threshold = (1.0 - p_buy) / (1.0 - c_buy) if buy_set and c_buy < 1.0 else 1.0
    active_scores = {str(item["scoreline"]) for item in buy_set}

    metrics: dict[str, dict[str, Any]] = {}
    active_cap_items: list[dict[str, Any]] = []
    shrunk_total_before_cap = 0.0
    for item in candidates:
        scoreline = str(item["scoreline"])
        fair = float(item["fair"])
        price = float(item["price"])
        full_fraction = max(0.0, fair - price * threshold) if scoreline in active_scores else 0.0
        shrink, shrink_source = kelly_shrink_multiplier(
            item["row"],
            min_shrink=min_shrink,
            max_shrink=max_shrink,
        )
        shrunk_fraction = full_fraction * shrink
        shrunk_total_before_cap += shrunk_fraction
        if scoreline in active_scores and shrunk_fraction > 0:
            active_cap_items.append(
                {
                    "scoreline": scoreline,
                    "fair": fair,
                    "price": price,
                    "full_fraction": full_fraction,
                    "shrunk_fraction": shrunk_fraction,
                }
            )
        metrics[scoreline] = {
            "joint_kelly_active": scoreline in active_scores,
            "kelly_ratio": fair / price,
            "joint_kelly_threshold": threshold,
            "joint_kelly_margin": fair / price - threshold,
            "joint_kelly_relative_margin": (fair / price / threshold - 1.0) if threshold > 0 else None,
            "joint_kelly_full_fraction": full_fraction,
            "kelly_shrink_multiplier": shrink,
            "kelly_shrink_source": shrink_source,
            "joint_kelly_fraction_before_cap": shrunk_fraction,
            "joint_kelly_buy_set_probability": p_buy,
            "joint_kelly_buy_set_cost": c_buy,
            "joint_kelly_cap_excluded": False,
            "joint_kelly_effective_probability": None,
            "joint_kelly_cap_lambda": None,
        }

    cap_result = solve_capped_shrunk_joint_kelly(active_cap_items, tranche_budget=tranche_budget)
    final_fractions = cap_result.get("fractions") or {}
    excluded_scores = set(str(score) for score in (cap_result.get("excluded") or []))
    effective_by_score = {str(item["scoreline"]): item.get("effective_probability") for item in active_cap_items}
    final_total = 0.0
    for scoreline, item_metrics in metrics.items():
        before_cap = float(item_metrics["joint_kelly_fraction_before_cap"] or 0.0)
        final_fraction = float(final_fractions.get(scoreline, 0.0))
        item_metrics["joint_kelly_fraction"] = final_fraction
        item_metrics["joint_kelly_cap_multiplier"] = (final_fraction / before_cap) if before_cap > 0 else None
        item_metrics["joint_kelly_cap_excluded"] = scoreline in excluded_scores or (
            bool(item_metrics["joint_kelly_active"]) and before_cap > 0 and final_fraction <= 0 and bool(cap_result.get("binding"))
        )
        item_metrics["joint_kelly_effective_probability"] = effective_by_score.get(scoreline)
        item_metrics["joint_kelly_cap_lambda"] = cap_result.get("lambda")
        final_total += final_fraction

    summary = {
        "candidate_count": len(candidates),
        "buy_set": [str(item["scoreline"]) for item in buy_set],
        "buy_set_probability": p_buy,
        "buy_set_cost": c_buy,
        "threshold": threshold,
        "full_fraction_sum": sum(float(metrics[str(item["scoreline"])] ["joint_kelly_full_fraction"]) for item in buy_set),
        "shrunk_fraction_sum_before_cap": shrunk_total_before_cap,
        "cap_method": cap_result.get("method"),
        "cap_binding": cap_result.get("binding"),
        "cap_lambda": cap_result.get("lambda"),
        "cap_effective_threshold": cap_result.get("effective_threshold"),
        "cap_survivors": cap_result.get("survivors"),
        "cap_excluded": cap_result.get("excluded"),
        "cap_multiplier": (final_total / shrunk_total_before_cap) if shrunk_total_before_cap > 0 else None,
        "final_fraction_sum": final_total,
        "tranche_budget": tranche_budget,
    }
    return metrics, summary

def decision_from_metrics(
    *,
    scoreline: str,
    fair: float,
    price: float | None,
    buffer: float,
    potential_band: float,
    min_material_kelly: float,
    kelly_metrics: dict[str, Any] | None,
    confidence: float,
    min_confidence: float,
    story_fit_value: float,
    min_story_fit: float,
    watch_threshold_margin: float,
    model_rank: int | None,
    max_buy_rank: int | None,
) -> dict[str, Any]:
    if price is None or price <= 0:
        return {
            "recommendation": "NO_MARKET",
            "decision": "NO",
            "joint_kelly_fraction": 0.0,
            "joint_kelly_full_fraction": 0.0,
            "kelly_shrink_multiplier": None,
            "kelly_shrink_source": None,
            "joint_kelly_fraction_before_cap": 0.0,
            "joint_kelly_cap_multiplier": None,
            "joint_kelly_cap_excluded": False,
            "joint_kelly_cap_lambda": None,
            "joint_kelly_effective_probability": None,
            "joint_kelly_active": False,
            "kelly_ratio": None,
            "joint_kelly_threshold": None,
            "joint_kelly_margin": None,
            "joint_kelly_relative_margin": None,
            "raw_edge": None,
            "edge_after_buffer": None,
            "value_status": "NO_MARKET",
            "passes_joint_buy_set": False,
            "passes_material_kelly": False,
            "passes_confidence": confidence >= min_confidence,
            "passes_story": story_fit_value >= min_story_fit,
            "passes_rank_gate": max_buy_rank is None or model_rank is None or model_rank <= max_buy_rank,
            "model_rank": model_rank,
            "manual_review": confidence < min_confidence or story_fit_value < min_story_fit,
            "gate_notes": "no listed exact-score price",
        }

    metrics = kelly_metrics or {}
    raw_edge = fair - price
    edge_after_buffer = raw_edge - buffer
    joint_kelly = float(metrics.get("joint_kelly_fraction") or 0.0)
    full_joint_kelly = float(metrics.get("joint_kelly_full_fraction") or 0.0)
    kelly_ratio = _to_float(metrics.get("kelly_ratio"))
    joint_threshold = _to_float(metrics.get("joint_kelly_threshold"), 1.0) or 1.0
    relative_margin = _to_float(metrics.get("joint_kelly_relative_margin"))
    active = bool(metrics.get("joint_kelly_active"))
    cap_excluded = bool(metrics.get("joint_kelly_cap_excluded"))
    positive_edge = fair > price
    near_fair = abs(fair - price) <= potential_band
    near_joint_threshold = relative_margin is not None and relative_margin >= -watch_threshold_margin
    passes_material_kelly = joint_kelly >= min_material_kelly
    passes_confidence = confidence >= min_confidence
    passes_story = story_fit_value >= min_story_fit
    passes_rank_gate = max_buy_rank is None or model_rank is None or model_rank <= max_buy_rank
    manual_review = not (passes_confidence and passes_story)

    if active and passes_material_kelly and passes_rank_gate and passes_story:
        recommendation = "BUY"
        decision = "BUY"
        value_status = "BUY_HOLD_MATERIAL_EXECUTABLE_RANKED"
    elif active and passes_material_kelly and not passes_rank_gate:
        recommendation = "RANK_EXCLUDED"
        decision = "WATCH"
        value_status = "BUY_HOLD_RANK_EXCLUDED"
    elif active and passes_material_kelly and not passes_story:
        recommendation = "STORY_EXCLUDED"
        decision = "WATCH"
        value_status = "BUY_HOLD_STORY_EXCLUDED"
    elif cap_excluded:
        recommendation = "CAP_EXCLUDED"
        decision = "CAP_EXCLUDED"
        value_status = "JOINT_KELLY_CAP_EXCLUDED"
    elif active and full_joint_kelly > 0:
        recommendation = "POTENTIAL_VALUE"
        decision = "POTENTIAL_VALUE"
        value_status = "JOINT_KELLY_TOO_SMALL_AFTER_CAP_OR_SHRINK"
    elif positive_edge and near_joint_threshold:
        recommendation = "WATCH"
        decision = "WATCH"
        value_status = "POSITIVE_EDGE_NEAR_JOINT_THRESHOLD"
    elif near_fair and price >= fair:
        recommendation = "NEAR_FAIR_WATCH"
        decision = "NEAR_FAIR_WATCH"
        value_status = "ABOVE_FAIR_WITHIN_BAND"
    elif positive_edge:
        recommendation = "WATCH"
        decision = "WATCH"
        value_status = "POSITIVE_EDGE_FAILED_GATES"
    elif price > fair + potential_band:
        recommendation = "NO"
        decision = "NO"
        value_status = "ABOVE_FAIR_BAND"
    else:
        recommendation = "NO"
        decision = "NO"
        value_status = "NO_EDGE"

    if value_status == "ABOVE_FAIR_BAND":
        gate_notes = f"price above fair + {cents(potential_band)}"
    elif value_status == "ABOVE_FAIR_WITHIN_BAND":
        gate_notes = f"price above fair but within {cents(potential_band)}"
    elif value_status == "JOINT_KELLY_CAP_EXCLUDED":
        gate_notes = "joint Kelly buy set, but growth-optimal tranche cap excludes it"
    elif value_status == "JOINT_KELLY_TOO_SMALL_AFTER_CAP_OR_SHRINK":
        gate_notes = f"joint Kelly buy set, but final stake below {pct(min_material_kelly, 2)}"
    elif value_status == "POSITIVE_EDGE_NEAR_JOINT_THRESHOLD":
        gate_notes = f"positive edge, near joint threshold {joint_threshold:.3f}"
    elif value_status == "BUY_HOLD_RANK_EXCLUDED":
        gate_notes = f"executable joint Kelly stake, but model rank {model_rank} is outside buy-hold rank limit {max_buy_rank}"
    elif value_status == "BUY_HOLD_STORY_EXCLUDED":
        gate_notes = "executable joint Kelly stake, but story-fit gate fails"
    elif recommendation == "BUY":
        gate_notes = "buy-hold eligible: executable final joint Kelly stake, rank gate passed, story gate passed"
    else:
        gate_notes = value_status.lower()
    if manual_review and recommendation == "BUY":
        gate_notes += "; manual review flag"

    return {
        "recommendation": recommendation,
        "decision": decision,
        "joint_kelly_fraction": joint_kelly,
        "joint_kelly_full_fraction": full_joint_kelly,
        "kelly_shrink_multiplier": metrics.get("kelly_shrink_multiplier"),
        "kelly_shrink_source": metrics.get("kelly_shrink_source"),
        "joint_kelly_fraction_before_cap": metrics.get("joint_kelly_fraction_before_cap"),
        "joint_kelly_cap_multiplier": metrics.get("joint_kelly_cap_multiplier"),
        "joint_kelly_cap_excluded": cap_excluded,
        "joint_kelly_cap_lambda": metrics.get("joint_kelly_cap_lambda"),
        "joint_kelly_effective_probability": metrics.get("joint_kelly_effective_probability"),
        "joint_kelly_active": active,
        "kelly_ratio": kelly_ratio,
        "joint_kelly_threshold": joint_threshold,
        "joint_kelly_margin": metrics.get("joint_kelly_margin"),
        "joint_kelly_relative_margin": relative_margin,
        "raw_edge": raw_edge,
        "edge_after_buffer": edge_after_buffer,
        "value_status": value_status,
        "passes_joint_buy_set": active,
        "passes_material_kelly": passes_material_kelly,
        "passes_confidence": passes_confidence,
        "passes_story": passes_story,
        "passes_rank_gate": passes_rank_gate,
        "model_rank": model_rank,
        "manual_review": manual_review,
        "gate_notes": gate_notes,
    }


def market_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        scoreline = str(row.get("scoreline") or "")
        if scoreline:
            result[scoreline] = row
    return result


def matrix_index(rows: list[dict[str, Any]]) -> dict[str, float]:
    result = {}
    for row in rows:
        scoreline = str(row.get("scoreline") or "")
        probability = _to_float(
            row.get("scoreline_fair_probability"),
            _to_float(row.get("model_only_fair_probability"), _to_float(row.get("base_model_probability"), 0.0)),
        )
        if scoreline:
            result[scoreline] = float(probability or 0.0)
    return result


def model_rank_index(matrix: dict[str, float]) -> dict[str, int]:
    ranked_scorelines = sorted(
        (
            (scoreline, fair)
            for scoreline, fair in matrix.items()
            if "-" in scoreline
            and scoreline.split("-", 1)[0].isdigit()
            and scoreline.split("-", 1)[1].isdigit()
            and fair > 0
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    return {scoreline: rank for rank, (scoreline, _fair) in enumerate(ranked_scorelines, start=1)}


def is_exact_scoreline(scoreline: str) -> bool:
    if "-" not in scoreline:
        return False
    left, right = scoreline.split("-", 1)
    return left.isdigit() and right.isdigit()


def is_any_other_score(scoreline: str) -> bool:
    return scoreline.strip().lower() == "any other score"


def parse_scoreline_set(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {piece.strip() for piece in raw.split(",") if piece.strip()}

LOW_SCORE_KNOCKOUT_EXCLUSIONS = {"0-0", "1-0", "0-1"}
GROUP_STAGE_MATCH_COUNT = 72

# --- BEGIN stage-conditional tier weights (opt-out via any explicit --tier-* flag) ----------
# Separately re-tuned via split_tier_weights_by_stage.py against
# outputs/v46_4_basev51_tier_optimization/cached_training_rows.csv (71 group-stage /
# 18 knockout matches, same underlying T-60 Polymarket backtest as the combined search).
# Knockout's config fails 3 of the tuner's own hard filters (small sample, chases ROI
# aggressively) -- treat it as directionally informative, not a fully vetted config.
STAGE_TIER_DEFAULTS = {
    "group_stage": {
        "tier_value_2_weight": 0.5530012020502271,
        "tier_cover_1_weight": 0.4236345129857588,
        "tier_cover_1_cap": 0.37058978787575936,
        "tier_cover_2_weight": 0.04680182765255726,
        "tier_cover_2_cap": 0.0671569729887917,
        "tier_negative_edge_cap_multiplier": 0.6133731848105837,
    },
    "knockout": {
        "tier_value_2_weight": 0.414972618465119,
        "tier_cover_1_weight": 0.05,
        "tier_cover_1_cap": 0.20237591282365225,
        "tier_cover_2_weight": 0.262916761375232,
        "tier_cover_2_cap": 0.27580769884099,
        "tier_negative_edge_cap_multiplier": 0.3759909888074068,
    },
}
# --- END stage-conditional tier weights ------------------------------------------------------


def infer_knockout_from_match_number(match_number: int | None) -> bool:
    return bool(match_number is not None and match_number > GROUP_STAGE_MATCH_COUNT)


def low_score_exclusion_set(
    *,
    knockout: bool,
    match_number: int | None,
    allow_low_scores: bool,
) -> set[str]:
    knockout_phase = bool(knockout) or infer_knockout_from_match_number(match_number)

    if knockout_phase and not allow_low_scores:
        return set(LOW_SCORE_KNOCKOUT_EXCLUSIONS)

    return set()


def clean_sheet_direction(scoreline: str) -> str | None:
    parsed = score_tuple(scoreline)
    if parsed is None:
        return None
    a_goals, b_goals = parsed
    if a_goals > 0 and b_goals == 0:
        return "team_a"
    if b_goals > 0 and a_goals == 0:
        return "team_b"
    return None


def is_directional_hedge_scoreline(scoreline: str) -> bool:
    parsed = score_tuple(scoreline)
    if parsed is None:
        return False
    a_goals, b_goals = parsed
    return a_goals == b_goals and a_goals >= 1


def _row_price(row: dict[str, Any]) -> float | None:
    return _to_float(row.get("market_price"), _to_float(row.get("raw_yes_price"), _to_float(row.get("yes_price"))))


def _row_fair(row: dict[str, Any]) -> float | None:
    return _to_float(row.get("model_probability"), _to_float(row.get("scoreline_fair_probability"), _to_float(row.get("model_only_fair_probability"))))


def model_market_agree(
    row: dict[str, Any],
    *,
    absolute_buffer: float,
    relative_buffer: float,
) -> bool:
    price = _row_price(row)
    fair = _row_fair(row)
    if price is None or fair is None or price <= 0 or fair <= 0:
        return False

    allowed_gap = max(float(absolute_buffer), price * float(relative_buffer))
    return abs(fair - price) <= allowed_gap


def directional_clean_sheet_strength(row: dict[str, Any]) -> tuple[float, float, float, float]:
    """Higher means more worth keeping as the one clean-sheet exposure."""
    return (
        float(row.get("expected_return") or -9.0),
        float(row.get("raw_edge") or -9.0),
        float(row.get("joint_kelly_fraction") or 0.0),
        float(row.get("model_probability") or 0.0),
    )


def summary_elo_gap(summary: dict[str, Any] | None) -> float | None:
    """Best-effort extraction because different versions write Elo keys differently."""
    if not summary:
        return None

    for key in [
        "elo_gap",
        "elo_difference",
        "rating_gap",
        "rating_difference",
        "team_elo_gap",
        "team_rating_gap",
    ]:
        value = _to_float(summary.get(key))
        if value is not None:
            return abs(value)

    for left_key, right_key in [
        ("team_a_elo", "team_b_elo"),
        ("team_a_rating", "team_b_rating"),
        ("elo_team_a", "elo_team_b"),
        ("rating_team_a", "rating_team_b"),
    ]:
        left = _to_float(summary.get(left_key))
        right = _to_float(summary.get(right_key))
        if left is not None and right is not None:
            return abs(left - right)

    return None


def apply_directional_risk_hedge(
    selected: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    summary: dict[str, Any] | None,
    price_tolerance: float,
    relative_price_tolerance: float,
    model_market_absolute_buffer: float,
    model_market_relative_buffer: float,
    big_elo_gap: float,
    max_replacements: int = 1,
) -> list[dict[str, Any]]:
    """
    Replace a weak/fairly-priced same-side clean-sheet stack member with a similarly priced draw hedge.

    Example:
    If selected contains 2-0 and 3-0, and 2-2 is similarly priced and model/market roughly agree,
    replace the weaker clean sheet with 2-2.

    Exception:
    If Elo gap is large, keep clean-sheet stacking because one-sided dominance is plausible.
    """
    if not selected or not candidates or max_replacements <= 0:
        return selected

    elo_gap = summary_elo_gap(summary)
    if elo_gap is not None and elo_gap >= big_elo_gap:
        return selected

    selected = [dict(row) for row in selected]
    selected_scores = {str(row.get("scoreline") or "") for row in selected}

    hedge_pool = [
        dict(row)
        for row in candidates
        if is_directional_hedge_scoreline(str(row.get("scoreline") or ""))
        and str(row.get("scoreline") or "") not in selected_scores
        and (_row_price(row) or 0.0) > 0
        and model_market_agree(
            row,
            absolute_buffer=model_market_absolute_buffer,
            relative_buffer=model_market_relative_buffer,
        )
    ]

    if not hedge_pool:
        return selected

    replacements_done = 0

    for side in ["team_a", "team_b"]:
        if replacements_done >= max_replacements:
            break

        clean_sheet_indexes = [
            idx
            for idx, row in enumerate(selected)
            if clean_sheet_direction(str(row.get("scoreline") or "")) == side
        ]

        if len(clean_sheet_indexes) < 2:
            continue

        # Keep the strongest clean-sheet exposure, replace one of the weaker/fairer ones.
        keep_idx = max(clean_sheet_indexes, key=lambda idx: directional_clean_sheet_strength(selected[idx]))
        replace_candidates = [idx for idx in clean_sheet_indexes if idx != keep_idx]

        replace_candidates = sorted(
            replace_candidates,
            key=lambda idx: (
                model_market_agree(
                    selected[idx],
                    absolute_buffer=model_market_absolute_buffer,
                    relative_buffer=model_market_relative_buffer,
                ),
                -directional_clean_sheet_strength(selected[idx])[0],
                -directional_clean_sheet_strength(selected[idx])[1],
                -directional_clean_sheet_strength(selected[idx])[2],
                -directional_clean_sheet_strength(selected[idx])[3],
            ),
            reverse=True,
        )

        for replace_idx in replace_candidates:
            target = selected[replace_idx]
            target_score = str(target.get("scoreline") or "")
            target_price = _row_price(target) or 0.0
            if target_price <= 0:
                continue

            max_price_gap = max(
                float(price_tolerance),
                target_price * float(relative_price_tolerance),
            )

            eligible: list[tuple[dict[str, Any], float, int, float]] = []

            for hedge in hedge_pool:
                hedge_score = str(hedge.get("scoreline") or "")
                if hedge_score in selected_scores:
                    continue

                hedge_price = _row_price(hedge) or 0.0
                hedge_fair = _row_fair(hedge) or 0.0
                if hedge_price <= 0 or hedge_fair <= 0:
                    continue

                price_gap = abs(hedge_price - target_price)
                if price_gap > max_price_gap:
                    continue

                parsed = score_tuple(hedge_score) or (0, 0)
                total_goals = parsed[0] + parsed[1]
                fair_price_gap = abs(hedge_fair - hedge_price)

                eligible.append((hedge, price_gap, total_goals, fair_price_gap))

            if not eligible:
                continue

            replacement_source, price_gap, total_goals, fair_price_gap = min(
                eligible,
                key=lambda item: (
                    item[1],             # closest market price to the clean sheet
                    item[3],             # best model-market agreement
                    abs(item[2] - 4),    # prefer 2-2 style hedge over 1-1/3-3 when comparable
                    -float(item[0].get("model_probability") or 0.0),
                    -float(item[0].get("raw_edge") or -9.0),
                ),
            )

            replacement = dict(replacement_source)
            replacement["buy_hold_role"] = target.get("buy_hold_role")
            replacement["selected_final"] = True
            replacement["selection_order"] = target.get("selection_order")
            replacement["selection_reason_final"] = f"directional_hedge_for_{target_score}"
            replacement["buy_hold_inclusion_reason"] = "directional_risk_hedge"
            replacement["role_reason"] = (
                f"{replacement.get('role_reason') or ''}+directional_risk_hedge_for_{target_score}"
            ).strip("+")
            replacement["selected_by_directional_hedge"] = True
            replacement["directional_hedge_replaced_scoreline"] = target_score
            replacement["directional_hedge_kept_clean_sheet"] = str(selected[keep_idx].get("scoreline") or "")
            replacement["directional_hedge_replaced_price"] = target_price
            replacement["directional_hedge_replacement_price"] = _row_price(replacement)
            replacement["directional_hedge_price_gap"] = price_gap
            replacement["directional_hedge_model_market_gap"] = fair_price_gap
            replacement["directional_hedge_side"] = side
            replacement["why"] = (
                f"directional-risk hedge: kept stronger clean sheet "
                f"{selected[keep_idx].get('scoreline')}, replaced {target_score} with "
                f"similarly priced draw hedge {replacement.get('scoreline')}; "
                f"price gap {cents(price_gap)}, model-market gap {cents(fair_price_gap)}; "
                f"{replacement.get('why') or ''}"
            )

            selected_scores.discard(target_score)
            selected_scores.add(str(replacement.get("scoreline") or ""))
            selected[replace_idx] = replacement
            replacements_done += 1
            break

    selected.sort(key=lambda row: int(row.get("selection_order") or 999))
    for idx, row in enumerate(selected, start=1):
        row["selection_order"] = idx

    return selected


def scoreline_from_summary_row(row: dict[str, Any] | None) -> str | None:
    if not row:
        return None
    if "scoreline" in row and row.get("scoreline"):
        return str(row.get("scoreline"))
    if "team_a_goals" in row and "team_b_goals" in row:
        try:
            return f"{int(row['team_a_goals'])}-{int(row['team_b_goals'])}"
        except (TypeError, ValueError):
            return None
    return None


def round_stakes_to_target(
    rows: list[dict[str, Any]],
    *,
    target_stake: float,
    min_order_size: float,
    rounding: float,
) -> list[float]:
    if not rows:
        return []
    if rounding <= 0:
        return [float(row["execution_stake_unrounded"]) for row in rows]

    stakes = []
    for row in rows:
        unrounded = float(row["execution_stake_unrounded"])
        cap = row.get("surplus_cap")
        if cap is not None:
            max_stake = float(row.get("base_stake") or 0.0) + float(cap)
            rounded = math.floor(min(unrounded, max_stake) / rounding) * rounding
        else:
            rounded = round(unrounded / rounding) * rounding
        stakes.append(max(min_order_size, rounded))
    # Adjust in discrete increments so the card total stays readable and near target.
    for _ in range(1000):
        diff = round(target_stake - sum(stakes), 10)
        if abs(diff) < rounding / 2:
            break
        if diff > 0:
            addable = []
            for i, stake in enumerate(stakes):
                cap = rows[i].get("surplus_cap")
                if cap is None:
                    addable.append(i)
                    continue
                max_stake = float(rows[i].get("base_stake") or 0.0) + float(cap)
                if stake + rounding <= max_stake + 1e-9:
                    addable.append(i)
            if not addable:
                addable = list(range(len(rows)))
            idx = max(
                addable,
                key=lambda i: (
                    float(rows[i].get("surplus_priority") or 0.0),
                    float(rows[i].get("raw_edge") or -9.0),
                    float(rows[i].get("model_probability") or 0.0),
                ),
            )
            stakes[idx] += rounding
            continue
        removable = [
            i for i, stake in enumerate(stakes)
            if stake - rounding >= max(min_order_size, float(rows[i].get("break_even_stake") or 0.0)) - 1e-9
        ]
        if not removable:
            removable = [i for i, stake in enumerate(stakes) if stake - rounding >= min_order_size - 1e-9]
        if not removable:
            break
        idx = min(
            removable,
            key=lambda i: (
                float(rows[i].get("surplus_priority") or 0.0),
                float(rows[i].get("raw_edge") or -9.0),
                float(rows[i].get("model_probability") or 0.0),
            ),
        )
        stakes[idx] -= rounding
    return [round(stake, 2) for stake in stakes]


def stake_profile_slug(profile: str) -> str:
    return profile.replace("-", "_")


def assign_buy_hold_tiers(rows: list[dict[str, Any]]) -> None:
    value_count = 0
    cover_count = 0
    for row in rows:
        role = str(row.get("buy_hold_role") or "")
        if role == "VALUE":
            value_count += 1
            tier = f"VALUE_{value_count}"
        elif role == "COVER":
            cover_count += 1
            tier = f"COVER_{cover_count}"
        elif role == "OUTLIER_VALUE":
            tier = "OUTLIER"
        else:
            tier = role or "UNTIERED"
        row["buy_hold_tier"] = tier


def tiered_balanced_surplus(
    row: dict[str, Any],
    *,
    fallback_cover_cap: float,
    tier_value_2_weight: float,
    tier_cover_1_weight: float,
    tier_cover_2_weight: float,
    tier_outlier_weight: float,
    tier_cover_1_cap: float,
    tier_cover_2_cap: float,
    tier_negative_edge_cap_multiplier: float,
) -> tuple[float, float | None]:
    tier = str(row.get("buy_hold_tier") or "")
    raw_edge = float(row.get("raw_edge") or 0.0)
    cover_1_cap = tier_cover_1_cap * (tier_negative_edge_cap_multiplier if raw_edge < 0 else 1.0)
    cover_2_cap = tier_cover_2_cap * (tier_negative_edge_cap_multiplier if raw_edge < 0 else 1.0)
    settings: dict[str, tuple[float, float | None]] = {
        "VALUE_1": (1.00, None),
        "VALUE_2": (tier_value_2_weight, None),
        "COVER_1": (tier_cover_1_weight, cover_1_cap),
        "COVER_2": (tier_cover_2_weight, cover_2_cap),
        "OUTLIER": (tier_outlier_weight, None if raw_edge >= 0 else fallback_cover_cap),
    }
    if tier.startswith("COVER_"):
        return settings.get(tier, (0.08, max(fallback_cover_cap, 0.10)))
    return settings.get(tier, (0.0, 0.0))


def allocate_breakeven_plus_value(
    selected: list[dict[str, Any]],
    *,
    target_stake: float,
    min_order_size: float,
    rounding: float,
    value_surplus_weight: float,
    outlier_surplus_weight: float,
    cover_surplus_weight: float,
    overpriced_cover_surplus_cap: float,
    tier_value_2_weight: float,
    tier_cover_1_weight: float,
    tier_cover_2_weight: float,
    tier_outlier_weight: float,
    tier_cover_1_cap: float,
    tier_cover_2_cap: float,
    tier_negative_edge_cap_multiplier: float,
    stake_profile: str,
) -> list[dict[str, Any]]:
    if not selected:
        return []
    assign_buy_hold_tiers(selected)
    for row in selected:
        price = float(row["market_price"])
        break_even_stake = target_stake * price
        base_stake = max(min_order_size, break_even_stake)
        row["break_even_stake"] = break_even_stake
        row["base_stake"] = base_stake
        row["stake_profile"] = stake_profile

    base_total = sum(float(row["base_stake"]) for row in selected)
    surplus = max(0.0, target_stake - base_total)

    for row in selected:
        role = str(row.get("buy_hold_role") or "")
        raw_edge = float(row.get("raw_edge") or 0.0)
        ev = float(row.get("expected_return") or 0.0)
        if stake_profile == "tiered-balanced":
            weight, cap = tiered_balanced_surplus(
                row,
                fallback_cover_cap=overpriced_cover_surplus_cap,
                tier_value_2_weight=tier_value_2_weight,
                tier_cover_1_weight=tier_cover_1_weight,
                tier_cover_2_weight=tier_cover_2_weight,
                tier_outlier_weight=tier_outlier_weight,
                tier_cover_1_cap=tier_cover_1_cap,
                tier_cover_2_cap=tier_cover_2_cap,
                tier_negative_edge_cap_multiplier=tier_negative_edge_cap_multiplier,
            )
            allocation_style = "tiered-balanced"
        elif role == "VALUE":
            weight = value_surplus_weight * (1.0 + max(ev, 0.0))
            cap = None
            allocation_style = "value-heavy"
        elif role == "OUTLIER_VALUE":
            weight = outlier_surplus_weight * (1.0 + max(ev, 0.0))
            cap = None if raw_edge >= 0 else overpriced_cover_surplus_cap
            allocation_style = "value-heavy"
        elif role == "COVER":
            weight = cover_surplus_weight
            cap = overpriced_cover_surplus_cap if raw_edge < 0 else None
            allocation_style = "value-heavy"
        else:
            weight = 0.0
            cap = 0.0
            allocation_style = "untiered"
        row["surplus_priority"] = weight
        row["surplus_cap"] = cap
        row["surplus_stake"] = 0.0
        row["allocation_style"] = allocation_style

    remaining = surplus
    active = {idx for idx, row in enumerate(selected) if float(row.get("surplus_priority") or 0.0) > 0}
    while remaining > 1e-9 and active:
        total_weight = sum(float(selected[idx]["surplus_priority"]) for idx in active)
        if total_weight <= 0:
            break
        spent = 0.0
        saturated: set[int] = set()
        for idx in list(active):
            row = selected[idx]
            allocation = remaining * float(row["surplus_priority"]) / total_weight
            cap = row.get("surplus_cap")
            current = float(row.get("surplus_stake") or 0.0)
            if cap is not None and current + allocation > float(cap):
                allocation = max(0.0, float(cap) - current)
                saturated.add(idx)
            row["surplus_stake"] = current + allocation
            spent += allocation
        remaining -= spent
        active -= saturated
        if spent <= 1e-9:
            break

    for row in selected:
        row["execution_stake_unrounded"] = float(row["base_stake"]) + float(row.get("surplus_stake") or 0.0)

    rounded_stakes = round_stakes_to_target(
        selected,
        target_stake=target_stake,
        min_order_size=min_order_size,
        rounding=rounding,
    )
    for row, stake in zip(selected, rounded_stakes):
        row["execution_stake"] = stake
        row["surplus_stake"] = max(0.0, stake - float(row["base_stake"]))
        price = float(row["market_price"])
        gross = stake / price if price > 0 else None
        row["hit_gross_payout"] = gross
        row["hit_net_profit"] = None if gross is None else gross - target_stake
        row["break_even_multiple"] = None if target_stake <= 0 else (gross / target_stake if gross is not None else None)
    return selected


def build_buy_hold_candidates(
    market_rows: list[dict[str, Any]],
    kelly_index: dict[str, dict[str, Any]],
    *,
    outlier_scoreline: str | None,
    favorite_side: str,
    knockout: bool,
    min_probability: float,
    min_confidence: float,
    min_story_fit: float,
    rank_by_score: dict[str, int],
    max_buy_rank: int | None,
    bankroll: float | None,
    min_order_size: float,
    probability_cover_count: int,
    ignored_scorelines: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Build auditable V46.4 buy-hold candidates without doing final staking."""
    ignored_scorelines = ignored_scorelines or set()

    base: list[dict[str, Any]] = []
    for row in market_rows:
        scoreline = str(row.get("scoreline") or "")
        if not is_exact_scoreline(scoreline) or is_any_other_score(scoreline):
            continue
        if scoreline in ignored_scorelines:
            continue
        price = _to_float(row.get("raw_yes_price"), _to_float(row.get("yes_price")))
        fair = _to_float(row.get("scoreline_fair_probability"), _to_float(row.get("model_only_fair_probability")))
        if price is None or price <= 0 or fair is None or fair <= 0:
            continue
        model_rank = rank_by_score.get(scoreline)
        if max_buy_rank is not None and (model_rank is None or model_rank > max_buy_rank):
            continue
        fit, story = story_fit(scoreline, favorite_side=favorite_side, knockout=knockout)
        reliability = reliability_from_scoreline(scoreline, row)
        prob_conf = max(0.30, min(1.0, fair / max(min_probability * 2.0, 1e-6)))
        liq_conf = liquidity_confidence(_to_float(row.get("liquidity")))
        confidence = 0.50 * reliability + 0.30 * prob_conf + 0.20 * liq_conf
        if confidence < min_confidence or fit < min_story_fit:
            continue

        metrics = kelly_index.get(scoreline) or {}
        joint_kelly = float(metrics.get("joint_kelly_fraction") or 0.0)
        active_kelly = bool(metrics.get("joint_kelly_active")) and not bool(metrics.get("joint_kelly_cap_excluded")) and joint_kelly > 0
        ev = fair / price - 1.0
        raw_edge = fair - price
        value_supported = raw_edge > 0 or (active_kelly and float(metrics.get("kelly_ratio") or 0.0) > 1.0)
        in_top_probability = model_rank is not None and model_rank <= (max_buy_rank or 10)
        is_outlier = bool(outlier_scoreline and scoreline == outlier_scoreline)
        if not (in_top_probability or value_supported or is_outlier):
            continue
        rank_score = 1.0 / max(float(model_rank or 999), 1.0)
        value_score = max(ev, 0.0) + 2.0 * max(raw_edge, 0.0) + max(joint_kelly, 0.0)
        coverage_score = fair + 0.01 * rank_score
        if is_outlier:
            role = "OUTLIER_VALUE"
            inclusion_reason = "model_outlier"
        elif value_supported:
            role = "VALUE"
            inclusion_reason = "value"
        elif model_rank is not None and model_rank <= probability_cover_count:
            role = "COVER"
            inclusion_reason = "probability_cover"
        else:
            role = "COVER"
            inclusion_reason = "ranked_cover"
        selected_by_top_rank = bool(in_top_probability)
        selected_by_value = bool(value_supported)
        selected_by_outlier = bool(is_outlier)
        selected_by_forced_cover = bool(role == "COVER" and model_rank is not None and model_rank <= probability_cover_count)
        role_reason_parts = []
        if selected_by_outlier:
            role_reason_parts.append("model_outlier")
        if selected_by_value:
            role_reason_parts.append("positive_edge_or_joint_kelly")
        if selected_by_forced_cover:
            role_reason_parts.append("forced_probability_cover")
        elif selected_by_top_rank:
            role_reason_parts.append("top_probability_rank")
        execution_weight = max(0.0001, value_score + coverage_score)
        base.append({
            "section": "BUY_HOLD",
            "scoreline": scoreline,
            "buy_hold_role": role,
            "model_probability": fair,
            "market_price": price,
            "payout_multiple": 1.0 / price,
            "expected_return": ev,
            "raw_edge": raw_edge,
            "edge_after_buffer": raw_edge,
            "max_entry_price": fair,
            "joint_kelly_fraction": joint_kelly,
            "joint_kelly_full_fraction": metrics.get("joint_kelly_full_fraction"),
            "kelly_shrink_multiplier": metrics.get("kelly_shrink_multiplier"),
            "kelly_shrink_source": metrics.get("kelly_shrink_source"),
            "joint_kelly_fraction_before_cap": metrics.get("joint_kelly_fraction_before_cap"),
            "joint_kelly_cap_multiplier": metrics.get("joint_kelly_cap_multiplier"),
            "joint_kelly_cap_excluded": bool(metrics.get("joint_kelly_cap_excluded")),
            "joint_kelly_cap_lambda": metrics.get("joint_kelly_cap_lambda"),
            "joint_kelly_effective_probability": metrics.get("joint_kelly_effective_probability"),
            "joint_kelly_active": active_kelly,
            "kelly_ratio": metrics.get("kelly_ratio"),
            "joint_kelly_threshold": metrics.get("joint_kelly_threshold"),
            "joint_kelly_margin": metrics.get("joint_kelly_margin"),
            "joint_kelly_relative_margin": metrics.get("joint_kelly_relative_margin"),
            "story_fit": fit,
            "confidence": confidence,
            "value_status": "BUY_HOLD_CANDIDATE",
            "recommendation": "CANDIDATE",
            "decision": "CANDIDATE",
            "passes_joint_buy_set": active_kelly,
            "passes_material_kelly": (joint_kelly * bankroll >= min_order_size) if bankroll and bankroll > 0 else joint_kelly > 0,
            "passes_confidence": True,
            "passes_story": True,
            "passes_rank_gate": True,
            "model_rank": model_rank,
            "manual_review": False,
            "selection_score": execution_weight,
            "execution_weight": execution_weight,
            "value_score": value_score,
            "coverage_score": coverage_score,
            "buy_hold_inclusion_reason": inclusion_reason,
            "role_reason": "+".join(role_reason_parts) if role_reason_parts else inclusion_reason,
            "selected_by_top_rank": selected_by_top_rank,
            "selected_by_value": selected_by_value,
            "selected_by_outlier": selected_by_outlier,
            "selected_by_forced_cover": selected_by_forced_cover,
            "selected_final": False,
            "selection_order": None,
            "selection_reason_final": "",
            "why": f"buy-hold candidate; {role}; {inclusion_reason}; {story}; model rank {model_rank}; model prob {pct(fair, 1)}; raw edge {cents(raw_edge)}; jK {pct(joint_kelly, 2)}",
        })

    return base


# --- BEGIN correlation-aware VALUE-pick diversification (opt-out, on by default) -----------
# The rest of this file's staking/tiering logic is deliberately unchanged from
# v46_4_optimisedbestbuys.py/v46_4_basev29.py (see module docstring). This block is a
# V51-specific addition: naive top-EV ranking can fill the VALUE tier with several
# scorelines that all die from the same underlying shock (e.g. 1-0/2-0/3-0 all require
# "away team fails to score" -- if the away team scores early, all three lose together).
# select_buy_hold_candidates() below picks its #1 VALUE pick by pure EV as the portfolio
# anchor, same as before, but for subsequent VALUE slots prefers the best-EV candidate
# from a *different* match-outcome path before falling back to a same-path candidate --
# so a high-probability anchor (e.g. 2-1) pairs with an uncorrelated hedge (e.g. an away
# blowout) rather than a third near-duplicate of the same scenario. This never overrides
# COVER/OUTLIER_VALUE selection or the staking amounts -- it only changes which scorelines
# get chosen for the VALUE slots. Disable with --disable-value-diversification to fall back
# to pure top-EV ranking (useful for backtesting whether diversification is load-bearing).
def scoreline_path_cluster(scoreline: str) -> str:
    """Tag a scoreline with the shared match-outcome 'shock' that determines whether it
    lives or dies -- used to detect when several VALUE candidates are really the same bet
    at different margins (1-0/2-0/3-0 are all "home wins, away fails to score")."""
    try:
        a_str, b_str = str(scoreline).split("-", 1)
        goals_a, goals_b = int(a_str), int(b_str)
    except (ValueError, AttributeError):
        return "other"
    if goals_a == goals_b:
        return "draw_clean_sheet" if goals_a == 0 else "draw_both_score"
    winner = "home" if goals_a > goals_b else "away"
    loser_goals = min(goals_a, goals_b)
    return f"{winner}_win_clean_sheet" if loser_goals == 0 else f"{winner}_win_concede"


def diversify_value_pool(value_pool: list[dict[str, Any]], target_n: int) -> list[dict[str, Any]]:
    """Greedy portfolio selection: keep value_pool's EV ranking for the anchor pick, then
    prefer candidates from an unused path cluster for each subsequent slot, falling back to
    the next-best-EV candidate (any cluster) only once every cluster has one pick already."""
    if target_n <= 0 or not value_pool:
        return []
    selected: list[dict[str, Any]] = [value_pool[0]]
    used_clusters = {scoreline_path_cluster(str(value_pool[0].get("scoreline") or ""))}
    remaining = list(value_pool[1:])
    while len(selected) < target_n and remaining:
        pick_idx = next(
            (i for i, row in enumerate(remaining) if scoreline_path_cluster(str(row.get("scoreline") or "")) not in used_clusters),
            None,
        )
        if pick_idx is None:
            break
        picked = remaining.pop(pick_idx)
        selected.append(picked)
        used_clusters.add(scoreline_path_cluster(str(picked.get("scoreline") or "")))
    for row in remaining:
        if len(selected) >= target_n:
            break
        selected.append(row)
    return selected
# --- END correlation-aware VALUE-pick diversification ---------------------------------------


def select_buy_hold_candidates(
    candidates: list[dict[str, Any]],
    *,
    outlier_scoreline: str | None,
    target_stake: float,
    min_order_size: float,
    max_bets: int,
    min_bets: int,
    min_value_bets: int,
    min_cover_bets: int,
    max_negative_edge_covers: int,
    require_outlier: bool,
    diversify_value_picks: bool = True,
) -> list[dict[str, Any]]:
    """Select the execution card before any stake allocation."""
    if target_stake <= 0 or max_bets <= 0 or not candidates:
        return []

    by_score = {str(row["scoreline"]): row for row in candidates}
    selected_scores: list[str] = []
    selection_reasons: dict[str, str] = {}

    def selected_count(role: str) -> int:
        return len([score for score in selected_scores if by_score[score].get("buy_hold_role") == role])

    def negative_cover_count() -> int:
        return len([
            score for score in selected_scores
            if by_score[score].get("buy_hold_role") == "COVER" and float(by_score[score].get("raw_edge") or 0.0) < 0
        ])

    def can_add(row: dict[str, Any]) -> bool:
        score = str(row.get("scoreline") or "")
        if not score or score in selected_scores or len(selected_scores) >= max_bets:
            return False
        role = str(row.get("buy_hold_role") or "")
        if role == "OUTLIER_VALUE" and selected_count("OUTLIER_VALUE") >= 1:
            return False
        if role == "COVER" and float(row.get("raw_edge") or 0.0) < 0 and negative_cover_count() >= max_negative_edge_covers:
            return False
        return True

    def add_row(row: dict[str, Any] | None, reason: str) -> None:
        if row is None or not can_add(row):
            return
        score = str(row["scoreline"])
        selected_scores.append(score)
        selection_reasons[score] = reason

    if require_outlier and outlier_scoreline:
        add_row(by_score.get(outlier_scoreline), "required_outlier")

    value_pool = sorted(
        [row for row in candidates if row.get("buy_hold_role") == "VALUE"],
        key=lambda r: (
            float(r.get("expected_return") or -9.0),
            float(r.get("raw_edge") or -9.0),
            float(r.get("model_probability") or 0.0),
        ),
        reverse=True,
    )
    value_target = min(min_value_bets, len(value_pool))
    value_ranked = diversify_value_pool(value_pool, value_target) if diversify_value_picks else value_pool
    anchor_score = str(value_pool[0].get("scoreline")) if value_pool else None
    for row in value_ranked:
        if selected_count("VALUE") >= value_target:
            break
        is_anchor = str(row.get("scoreline")) == anchor_score
        add_row(row, "required_value" if is_anchor else "required_value_diversified_hedge")

    cover_pool = sorted(
        [row for row in candidates if row.get("buy_hold_role") == "COVER"],
        key=lambda r: (int(r.get("model_rank") or 999), -float(r.get("model_probability") or 0.0)),
    )
    cover_target = min(min_cover_bets, len(cover_pool))
    for row in cover_pool:
        if selected_count("COVER") >= cover_target:
            break
        add_row(row, "required_cover")

    preferred_min = min(max_bets, max(min_bets, min_value_bets + min_cover_bets + (1 if require_outlier and outlier_scoreline else 0)))
    for row in cover_pool:
        if len(selected_scores) >= preferred_min:
            break
        add_row(row, "probability_cover_fill")

    for row in value_pool + cover_pool:
        if len(selected_scores) >= max_bets:
            break
        add_row(row, "best_available_fill")

    affordable_max = int(target_stake // min_order_size) if min_order_size > 0 else max_bets
    n_allowed = min(max_bets, affordable_max, len(selected_scores))
    selected_scores = selected_scores[:n_allowed]
    if not selected_scores:
        return []

    selected = [dict(by_score[score]) for score in selected_scores]
    while len(selected) > min_bets:
        base_total = sum(max(min_order_size, target_stake * float(row["market_price"])) for row in selected)
        if base_total <= target_stake:
            break
        drop_idx = min(
            range(len(selected)),
            key=lambda i: (
                selected[i].get("buy_hold_role") == "OUTLIER_VALUE",
                selected[i].get("buy_hold_role") == "VALUE",
                float(selected[i].get("model_probability") or 0.0),
            ),
        )
        selected.pop(drop_idx)

    selected_score_set = {str(row["scoreline"]) for row in selected}
    selected = [
        row for row in selected
        if str(row["scoreline"]) in selected_score_set
    ]

    role_order = {"VALUE": 0, "COVER": 1, "OUTLIER_VALUE": 2}
    selected.sort(
        key=lambda row: (
            role_order.get(str(row.get("buy_hold_role")), 9),
            int(row.get("model_rank") or 999) if row.get("buy_hold_role") == "COVER" else 999,
            -float(row.get("expected_return") or -9.0),
            -float(row.get("model_probability") or 0.0),
        )
    )
    for idx, row in enumerate(selected, start=1):
        score = str(row["scoreline"])
        row["selected_final"] = True
        row["selection_order"] = idx
        row["selection_reason_final"] = selection_reasons.get(score, "selected")
    return selected


def build_selection_debug_rows(
    candidates: list[dict[str, Any]],
    selected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected_by_score = {str(row.get("scoreline")): row for row in selected}
    rows: list[dict[str, Any]] = []
    for row in sorted(
        candidates,
        key=lambda r: (
            int(r.get("model_rank") or 999),
            -float(r.get("raw_edge") or -9.0),
            -float(r.get("model_probability") or 0.0),
        ),
    ):
        score = str(row.get("scoreline") or "")
        selected_row = selected_by_score.get(score)
        rows.append({
            "scoreline": score,
            "model_rank": row.get("model_rank"),
            "role": row.get("buy_hold_role"),
            "selected": bool(selected_row),
            "selection_order": (selected_row or {}).get("selection_order"),
            "selection_reason_final": (selected_row or {}).get("selection_reason_final"),
            "fair": row.get("model_probability"),
            "price": row.get("market_price"),
            "raw_edge": row.get("raw_edge"),
            "expected_return": row.get("expected_return"),
            "joint_kelly_fraction": row.get("joint_kelly_fraction"),
            "joint_kelly_active": row.get("joint_kelly_active"),
            "outlier": row.get("selected_by_outlier"),
            "selected_by_top_rank": row.get("selected_by_top_rank"),
            "selected_by_value": row.get("selected_by_value"),
            "selected_by_outlier": row.get("selected_by_outlier"),
            "selected_by_forced_cover": row.get("selected_by_forced_cover"),
            "role_reason": row.get("role_reason"),
            "confidence": row.get("confidence"),
            "story_fit": row.get("story_fit"),
            "selection_score": row.get("selection_score"),
        })
    return rows


def build_hit_outcome_rows(rows: list[dict[str, Any]], *, target_stake: float) -> list[dict[str, Any]]:
    outcome_rows: list[dict[str, Any]] = []
    for row in rows:
        stake = _to_float(row.get("execution_stake"), 0.0) or 0.0
        price = _to_float(row.get("market_price"), 0.0) or 0.0
        gross = stake / price if price > 0 else None
        outcome_rows.append({
            "scoreline": row.get("scoreline"),
            "role": row.get("buy_hold_role"),
            "tier": row.get("buy_hold_tier"),
            "stake_profile": row.get("stake_profile"),
            "stake": stake,
            "price": price,
            "model_probability": row.get("model_probability"),
            "raw_edge": row.get("raw_edge"),
            "break_even_stake": row.get("break_even_stake"),
            "base_stake": row.get("base_stake"),
            "surplus_stake": row.get("surplus_stake"),
            "gross_payout_if_hit": gross,
            "net_profit_if_hit": None if gross is None else gross - target_stake,
            "break_even_multiple": None if not gross or target_stake <= 0 else gross / target_stake,
        })
    return outcome_rows


def validate_buy_hold_card(
    rows: list[dict[str, Any]],
    *,
    candidates: list[dict[str, Any]],
    target_stake: float,
    min_order_size: float,
    rounding: float,
    overpriced_cover_surplus_cap: float,
    min_value_bets: int,
    min_cover_bets: int,
    max_negative_edge_covers: int,
    require_outlier: bool,
    outlier_scoreline: str | None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    tol = max(0.01, (rounding or 0.0) / 2.0 + 1e-9)
    total_stake = sum(float(row.get("execution_stake") or 0.0) for row in rows)
    selected_value_count = len([row for row in rows if row.get("buy_hold_role") == "VALUE"])
    selected_cover_count = len([row for row in rows if row.get("buy_hold_role") == "COVER"])
    selected_outlier_count = len([row for row in rows if row.get("buy_hold_role") == "OUTLIER_VALUE"])
    negative_cover_count = len([
        row for row in rows
        if row.get("buy_hold_role") == "COVER" and float(row.get("raw_edge") or 0.0) < 0
    ])
    available_value_count = len([row for row in candidates if row.get("buy_hold_role") == "VALUE"])
    available_cover_count = len([row for row in candidates if row.get("buy_hold_role") == "COVER"])

    if rows and abs(total_stake - target_stake) > tol:
        errors.append(f"execution stakes sum to {total_stake:.2f}, expected {target_stake:.2f}")
    for row in rows:
        scoreline = row.get("scoreline")
        stake = float(row.get("execution_stake") or 0.0)
        price = float(row.get("market_price") or 0.0)
        if stake + 1e-9 < min_order_size:
            errors.append(f"{scoreline} stake {stake:.2f} is below minimum order {min_order_size:.2f}")
        gross = stake / price if price > 0 else 0.0
        if gross + tol < target_stake:
            errors.append(f"{scoreline} would not break even if it hits: gross {gross:.2f} vs target {target_stake:.2f}")
        if row.get("buy_hold_role") == "COVER" and float(row.get("raw_edge") or 0.0) < 0:
            row_cap = row.get("surplus_cap")
            cap = overpriced_cover_surplus_cap if row_cap is None else float(row_cap)
            max_stake = float(row.get("base_stake") or 0.0) + cap
            if stake > max_stake + tol:
                errors.append(f"{scoreline} negative-edge COVER stake {stake:.2f} exceeds base+cap {max_stake:.2f}")

    if available_value_count >= min_value_bets and selected_value_count < min_value_bets:
        errors.append(f"selected {selected_value_count} VALUE bets, but {min_value_bets} were required and available")
    if available_cover_count >= min_cover_bets and selected_cover_count < min_cover_bets:
        errors.append(f"selected {selected_cover_count} COVER bets, but {min_cover_bets} were required and available")
    if require_outlier and outlier_scoreline and any(row.get("scoreline") == outlier_scoreline for row in candidates) and selected_outlier_count != 1:
        errors.append(f"required exactly one OUTLIER_VALUE selection for {outlier_scoreline}, selected {selected_outlier_count}")
    if selected_outlier_count > 1:
        errors.append(f"selected {selected_outlier_count} OUTLIER_VALUE bets; maximum is 1")
    if negative_cover_count > max_negative_edge_covers:
        errors.append(f"selected {negative_cover_count} negative-edge COVER bets; maximum is {max_negative_edge_covers}")
    if not rows:
        warnings.append("no executable buy-hold rows selected")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "total_stake": total_stake,
        "selected_value_count": selected_value_count,
        "selected_cover_count": selected_cover_count,
        "selected_outlier_count": selected_outlier_count,
        "negative_cover_count": negative_cover_count,
        "available_value_count": available_value_count,
        "available_cover_count": available_cover_count,
    }


def build_buy_hold_execution_rows(
    market_rows: list[dict[str, Any]],
    kelly_index: dict[str, dict[str, Any]],
    *,
    outlier_scoreline: str | None,
    favorite_side: str,
    knockout: bool,
    min_probability: float,
    min_confidence: float,
    min_story_fit: float,
    rank_by_score: dict[str, int],
    max_buy_rank: int | None,
    bankroll: float | None,
    target_stake: float | None,
    min_order_size: float,
    stake_rounding: float,
    max_bets: int,
    min_bets: int,
    probability_cover_count: int,
    value_surplus_weight: float,
    outlier_surplus_weight: float,
    cover_surplus_weight: float,
    overpriced_cover_surplus_cap: float,
    tier_value_2_weight: float,
    tier_cover_1_weight: float,
    tier_cover_2_weight: float,
    tier_outlier_weight: float,
    tier_cover_1_cap: float,
    tier_cover_2_cap: float,
    tier_negative_edge_cap_multiplier: float,
    min_value_bets: int,
    min_cover_bets: int,
    max_negative_edge_covers: int,
    require_outlier: bool,
    stake_profile: str,
    ignored_scorelines: set[str] | None = None,
    diversify_value_picks: bool = True,
    directional_risk_hedge: bool = True,
    directional_hedge_price_tolerance: float = 0.012,
    directional_hedge_relative_price_tolerance: float = 0.35,
    directional_hedge_model_market_absolute_buffer: float = 0.012,
    directional_hedge_model_market_relative_buffer: float = 0.35,
    directional_hedge_big_elo_gap: float = 250.0,
    directional_hedge_max_replacements: int = 1,
    summary_for_directional_hedge: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Build the V46.4 break-even-plus-value pre-game buy-and-hold card."""
    if target_stake is None or target_stake <= 0 or max_bets <= 0:
        return [], [], {"ok": False, "errors": [], "warnings": ["staking disabled"]}, []

    candidates = build_buy_hold_candidates(
        market_rows,
        kelly_index,
        outlier_scoreline=outlier_scoreline,
        favorite_side=favorite_side,
        knockout=knockout,
        min_probability=min_probability,
        min_confidence=min_confidence,
        min_story_fit=min_story_fit,
        rank_by_score=rank_by_score,
        max_buy_rank=max_buy_rank,
        bankroll=bankroll,
        min_order_size=min_order_size,
        probability_cover_count=probability_cover_count,
        ignored_scorelines=ignored_scorelines,
    )
    selected = select_buy_hold_candidates(
        candidates,
        outlier_scoreline=outlier_scoreline,
        target_stake=target_stake,
        min_order_size=min_order_size,
        max_bets=max_bets,
        min_bets=min_bets,
        min_value_bets=min_value_bets,
        min_cover_bets=min_cover_bets,
        max_negative_edge_covers=max_negative_edge_covers,
        require_outlier=require_outlier,
        diversify_value_picks=diversify_value_picks,
    )
    if directional_risk_hedge:
        selected = apply_directional_risk_hedge(
            selected,
            candidates,
            summary=summary_for_directional_hedge,
            price_tolerance=directional_hedge_price_tolerance,
            relative_price_tolerance=directional_hedge_relative_price_tolerance,
            model_market_absolute_buffer=directional_hedge_model_market_absolute_buffer,
            model_market_relative_buffer=directional_hedge_model_market_relative_buffer,
            big_elo_gap=directional_hedge_big_elo_gap,
            max_replacements=directional_hedge_max_replacements,
        )

    selected = allocate_breakeven_plus_value(
        selected,
        target_stake=target_stake,
        min_order_size=min_order_size,
        rounding=stake_rounding,
        value_surplus_weight=value_surplus_weight,
        outlier_surplus_weight=outlier_surplus_weight,
        cover_surplus_weight=cover_surplus_weight,
        overpriced_cover_surplus_cap=overpriced_cover_surplus_cap,
        tier_value_2_weight=tier_value_2_weight,
        tier_cover_1_weight=tier_cover_1_weight,
        tier_cover_2_weight=tier_cover_2_weight,
        tier_outlier_weight=tier_outlier_weight,
        tier_cover_1_cap=tier_cover_1_cap,
        tier_cover_2_cap=tier_cover_2_cap,
        tier_negative_edge_cap_multiplier=tier_negative_edge_cap_multiplier,
        stake_profile=stake_profile,
    )
    for idx, row in enumerate(selected, start=1):
        row["section"] = f"BUY_HOLD_{idx}"
        row["execution_fraction"] = None if not bankroll or bankroll <= 0 else row["execution_stake"] / bankroll
        row["execution_target_stake"] = target_stake
        row["recommendation"] = "BUY"
        row["decision"] = "BUY"
        row["value_status"] = "BREAKEVEN_PLUS_VALUE_SELECTED"
        row["passes_material_kelly"] = row["execution_stake"] >= min_order_size
        row["why"] = (
            f"breakeven-plus-value selected; profile {stake_profile}; tier {row.get('buy_hold_tier')}; role {row['buy_hold_role']}; "
            f"stake {row['execution_stake']:.2f}; base {row['base_stake']:.2f}; "
            f"surplus {row['surplus_stake']:.2f}; hit net {row['hit_net_profit']:.2f}; {row['why']}"
        )
    validation = validate_buy_hold_card(
        selected,
        candidates=candidates,
        target_stake=target_stake,
        min_order_size=min_order_size,
        rounding=stake_rounding,
        overpriced_cover_surplus_cap=overpriced_cover_surplus_cap,
        min_value_bets=min_value_bets,
        min_cover_bets=min_cover_bets,
        max_negative_edge_covers=max_negative_edge_covers,
        require_outlier=require_outlier,
        outlier_scoreline=outlier_scoreline,
    )
    debug_rows = build_selection_debug_rows(candidates, selected)
    hit_outcome_rows = build_hit_outcome_rows(selected, target_stake=target_stake)
    return selected, debug_rows, validation, hit_outcome_rows


def build_core_rows(
    summary: dict[str, Any],
    markets: dict[str, dict[str, Any]],
    matrix: dict[str, float],
    kelly_index: dict[str, dict[str, Any]],
    *,
    buffer: float,
    potential_band: float,
    min_material_kelly: float,
    min_confidence: float,
    min_story_fit: float,
    watch_threshold_margin: float,
    rank_by_score: dict[str, int],
    max_buy_rank: int | None,
) -> list[dict[str, Any]]:
    core_items: list[tuple[str, str, float | None]] = []
    for idx, row in enumerate(summary.get("prediction_top_3") or [], start=1):
        scoreline = scoreline_from_goals(row)
        core_items.append((f"CORE_TOP{idx}", scoreline, _to_float(row.get("probability"))))
    outlier = summary.get("prediction_outlier") or {}
    if outlier:
        outlier_scoreline = scoreline_from_goals(outlier)
        if outlier_scoreline not in {score for _section, score, _prob in core_items}:
            core_items.append(("OUTLIER", outlier_scoreline, _to_float(outlier.get("probability"))))

    rows = []
    for section, scoreline, summary_prob in core_items:
        market = markets.get(scoreline)
        fair = float(summary_prob if summary_prob is not None else matrix.get(scoreline, 0.0))
        price = _to_float((market or {}).get("raw_yes_price"), _to_float((market or {}).get("yes_price")))
        payout = None if not price or price <= 0 else 1.0 / price
        ev = None if not price or price <= 0 else fair / price - 1.0
        max_entry = max(0.0, fair - buffer)
        decision = decision_from_metrics(
            scoreline=scoreline,
            fair=fair,
            price=price,
            buffer=buffer,
            potential_band=potential_band,
            min_material_kelly=min_material_kelly,
            kelly_metrics=kelly_index.get(scoreline),
            confidence=1.0,
            min_confidence=min_confidence,
            story_fit_value=1.0,
            min_story_fit=min_story_fit,
            watch_threshold_margin=watch_threshold_margin,
            model_rank=rank_by_score.get(scoreline),
            max_buy_rank=max_buy_rank,
        )
        core_score = fair / max(price or fair or 1e-6, 1e-6)
        rows.append(
            {
                "section": section,
                "scoreline": scoreline,
                "model_probability": fair,
                "market_price": price,
                "payout_multiple": payout,
                "expected_return": ev,
                "raw_edge": decision["raw_edge"],
                "edge_after_buffer": decision["edge_after_buffer"],
                "max_entry_price": max_entry,
                "joint_kelly_fraction": decision["joint_kelly_fraction"],
                "joint_kelly_full_fraction": decision["joint_kelly_full_fraction"],
                "kelly_shrink_multiplier": decision["kelly_shrink_multiplier"],
                "kelly_shrink_source": decision["kelly_shrink_source"],
                "joint_kelly_fraction_before_cap": decision["joint_kelly_fraction_before_cap"],
                "joint_kelly_cap_multiplier": decision["joint_kelly_cap_multiplier"],
                "joint_kelly_cap_excluded": decision["joint_kelly_cap_excluded"],
                "joint_kelly_cap_lambda": decision["joint_kelly_cap_lambda"],
                "joint_kelly_effective_probability": decision["joint_kelly_effective_probability"],
                "joint_kelly_active": decision["joint_kelly_active"],
                "kelly_ratio": decision["kelly_ratio"],
                "joint_kelly_threshold": decision["joint_kelly_threshold"],
                "joint_kelly_margin": decision["joint_kelly_margin"],
                "joint_kelly_relative_margin": decision["joint_kelly_relative_margin"],
                "story_fit": 1.0,
                "confidence": 1.0,
                "value_status": decision["value_status"],
                "recommendation": decision["recommendation"],
                "decision": decision["decision"],
                "passes_joint_buy_set": decision["passes_joint_buy_set"],
                "passes_material_kelly": decision["passes_material_kelly"],
                "passes_confidence": decision["passes_confidence"],
                "passes_story": decision["passes_story"],
                "passes_rank_gate": decision["passes_rank_gate"],
                "model_rank": decision["model_rank"],
                "manual_review": decision["manual_review"],
                "selection_score": core_score,
                "why": f"core model score; {decision['gate_notes']}",
            }
        )
    return rows


def build_upside_rows(
    market_rows: list[dict[str, Any]],
    core_scorelines: set[str],
    kelly_index: dict[str, dict[str, Any]],
    *,
    favorite_side: str,
    knockout: bool,
    buffer: float,
    potential_band: float,
    min_probability: float,
    min_material_kelly: float,
    min_confidence: float,
    min_story_fit: float,
    watch_threshold_margin: float,
    rank_by_score: dict[str, int],
    max_buy_rank: int | None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in market_rows:
        scoreline = str(row.get("scoreline") or "")
        if not scoreline or scoreline in core_scorelines:
            continue
        price = _to_float(row.get("raw_yes_price"), _to_float(row.get("yes_price")))
        fair = _to_float(row.get("scoreline_fair_probability"), _to_float(row.get("model_only_fair_probability")))
        if price is None or price <= 0 or fair is None or fair <= 0:
            continue
        ev = fair / price - 1.0
        near_fair = abs(price - fair) <= potential_band
        if ev <= 0 and not near_fair:
            continue
        edge_after_buffer = fair - price - buffer
        fit, story = story_fit(scoreline, favorite_side=favorite_side, knockout=knockout)
        reliability = reliability_from_scoreline(scoreline, row)
        prob_conf = max(0.30, min(1.0, fair / max(min_probability * 2.0, 1e-6)))
        liq_conf = liquidity_confidence(_to_float(row.get("liquidity")))
        confidence = 0.50 * reliability + 0.30 * prob_conf + 0.20 * liq_conf
        decision = decision_from_metrics(
            scoreline=scoreline,
            fair=fair,
            price=price,
            buffer=buffer,
            potential_band=potential_band,
            min_material_kelly=min_material_kelly,
            kelly_metrics=kelly_index.get(scoreline),
            confidence=confidence,
            min_confidence=min_confidence,
            story_fit_value=fit,
            min_story_fit=min_story_fit,
            watch_threshold_margin=watch_threshold_margin,
            model_rank=rank_by_score.get(scoreline),
            max_buy_rank=max_buy_rank,
        )
        payout = 1.0 / price
        selection_score = float(decision["joint_kelly_fraction"] or 0.0) * payout * fit
        if decision["decision"] == "BUY":
            selection_score *= 1.0
        elif decision["decision"] in {"POTENTIAL_VALUE", "CAP_EXCLUDED", "NEAR_FAIR_WATCH"}:
            selection_score = max(selection_score, fair * ev * payout * fit * confidence * 0.35)
        elif decision["decision"] == "WATCH":
            selection_score = max(selection_score, fair * ev * payout * fit * confidence * 0.20)
        else:
            selection_score = fair * ev * payout * fit * confidence * 0.05
        candidates.append(
            {
                "section": "",
                "scoreline": scoreline,
                "model_probability": fair,
                "market_price": price,
                "payout_multiple": payout,
                "expected_return": ev,
                "raw_edge": decision["raw_edge"],
                "edge_after_buffer": edge_after_buffer,
                "max_entry_price": max(0.0, fair - buffer),
                "joint_kelly_fraction": decision["joint_kelly_fraction"],
                "joint_kelly_full_fraction": decision["joint_kelly_full_fraction"],
                "kelly_shrink_multiplier": decision["kelly_shrink_multiplier"],
                "kelly_shrink_source": decision["kelly_shrink_source"],
                "joint_kelly_fraction_before_cap": decision["joint_kelly_fraction_before_cap"],
                "joint_kelly_cap_multiplier": decision["joint_kelly_cap_multiplier"],
                "joint_kelly_cap_excluded": decision["joint_kelly_cap_excluded"],
                "joint_kelly_cap_lambda": decision["joint_kelly_cap_lambda"],
                "joint_kelly_effective_probability": decision["joint_kelly_effective_probability"],
                "joint_kelly_active": decision["joint_kelly_active"],
                "kelly_ratio": decision["kelly_ratio"],
                "joint_kelly_threshold": decision["joint_kelly_threshold"],
                "joint_kelly_margin": decision["joint_kelly_margin"],
                "joint_kelly_relative_margin": decision["joint_kelly_relative_margin"],
                "story_fit": fit,
                "confidence": confidence,
                "value_status": decision["value_status"],
                "recommendation": decision["recommendation"],
                "decision": decision["decision"],
                "passes_joint_buy_set": decision["passes_joint_buy_set"],
                "passes_material_kelly": decision["passes_material_kelly"],
                "passes_confidence": decision["passes_confidence"],
                "passes_story": decision["passes_story"],
                "passes_rank_gate": decision["passes_rank_gate"],
                "model_rank": decision["model_rank"],
                "manual_review": decision["manual_review"],
                "selection_score": selection_score,
                "why": f"{story}; EV {ev:.1%}; payout {payout:.1f}x; {decision['gate_notes']}",
            }
        )
    candidates.sort(
        key=lambda row: (
            row["decision"] == "BUY",
            row["decision"] == "POTENTIAL_VALUE",
            row["decision"] == "CAP_EXCLUDED",
            row["decision"] == "NEAR_FAIR_WATCH",
            row["decision"] == "WATCH",
            float(row["selection_score"]),
            float(row["expected_return"]),
        ),
        reverse=True,
    )
    selected = []
    for idx, row in enumerate(candidates[:2], start=1):
        row = dict(row)
        row["section"] = f"UPSIDE_EV_{idx}"
        selected.append(row)
    return selected


def build_watch_rows(
    market_rows: list[dict[str, Any]],
    selected_scorelines: set[str],
    kelly_index: dict[str, dict[str, Any]],
    *,
    favorite_side: str,
    knockout: bool,
    buffer: float,
    potential_band: float,
    min_probability: float,
    min_material_kelly: float,
    min_confidence: float,
    min_story_fit: float,
    watch_threshold_margin: float,
    rank_by_score: dict[str, int],
    max_buy_rank: int | None,
    limit: int,
) -> list[dict[str, Any]]:
    watches = []
    for row in market_rows:
        scoreline = str(row.get("scoreline") or "")
        if not scoreline or scoreline in selected_scorelines:
            continue
        price = _to_float(row.get("raw_yes_price"), _to_float(row.get("yes_price")))
        fair = _to_float(row.get("scoreline_fair_probability"), _to_float(row.get("model_only_fair_probability")))
        if price is None or price <= 0 or fair is None or fair <= 0:
            continue
        ev = fair / price - 1.0
        near_fair = abs(price - fair) <= potential_band
        if ev <= 0 and not near_fair:
            continue
        fit, story = story_fit(scoreline, favorite_side=favorite_side, knockout=knockout)
        reliability = reliability_from_scoreline(scoreline, row)
        prob_conf = max(0.30, min(1.0, fair / max(min_probability * 2.0, 1e-6)))
        liq_conf = liquidity_confidence(_to_float(row.get("liquidity")))
        confidence = 0.50 * reliability + 0.30 * prob_conf + 0.20 * liq_conf
        decision = decision_from_metrics(
            scoreline=scoreline,
            fair=fair,
            price=price,
            buffer=buffer,
            potential_band=potential_band,
            min_material_kelly=min_material_kelly,
            kelly_metrics=kelly_index.get(scoreline),
            confidence=confidence,
            min_confidence=min_confidence,
            story_fit_value=fit,
            min_story_fit=min_story_fit,
            watch_threshold_margin=watch_threshold_margin,
            model_rank=rank_by_score.get(scoreline),
            max_buy_rank=max_buy_rank,
        )
        if decision["decision"] == "NO":
            continue
        section = decision["decision"] if decision["decision"] in {"POTENTIAL_VALUE", "CAP_EXCLUDED", "NEAR_FAIR_WATCH"} else "WATCH"
        watches.append(
            {
                "section": section,
                "scoreline": scoreline,
                "model_probability": fair,
                "market_price": price,
                "payout_multiple": 1.0 / price,
                "expected_return": ev,
                "raw_edge": decision["raw_edge"],
                "edge_after_buffer": decision["edge_after_buffer"],
                "max_entry_price": max(0.0, fair - buffer),
                "joint_kelly_fraction": decision["joint_kelly_fraction"],
                "joint_kelly_full_fraction": decision["joint_kelly_full_fraction"],
                "kelly_shrink_multiplier": decision["kelly_shrink_multiplier"],
                "kelly_shrink_source": decision["kelly_shrink_source"],
                "joint_kelly_fraction_before_cap": decision["joint_kelly_fraction_before_cap"],
                "joint_kelly_cap_multiplier": decision["joint_kelly_cap_multiplier"],
                "joint_kelly_cap_excluded": decision["joint_kelly_cap_excluded"],
                "joint_kelly_cap_lambda": decision["joint_kelly_cap_lambda"],
                "joint_kelly_effective_probability": decision["joint_kelly_effective_probability"],
                "joint_kelly_active": decision["joint_kelly_active"],
                "kelly_ratio": decision["kelly_ratio"],
                "joint_kelly_threshold": decision["joint_kelly_threshold"],
                "joint_kelly_margin": decision["joint_kelly_margin"],
                "joint_kelly_relative_margin": decision["joint_kelly_relative_margin"],
                "story_fit": fit,
                "confidence": confidence,
                "value_status": decision["value_status"],
                "recommendation": decision["recommendation"],
                "decision": decision["decision"],
                "passes_joint_buy_set": decision["passes_joint_buy_set"],
                "passes_material_kelly": decision["passes_material_kelly"],
                "passes_confidence": decision["passes_confidence"],
                "passes_story": decision["passes_story"],
                "passes_rank_gate": decision["passes_rank_gate"],
                "model_rank": decision["model_rank"],
                "manual_review": decision["manual_review"],
                "selection_score": ev * fair / price,
                "why": f"{story}; {decision['gate_notes']}; not one of selected upside slots",
            }
        )
    watches.sort(key=lambda row: float(row["selection_score"]), reverse=True)
    return watches[:limit]


def write_card_plot(path: Path, rows: list[dict[str, Any]], team_a: str, team_b: str) -> None:
    display = list(rows)
    if not display:
        return
    labels = [f"{row['section']}\n{row['scoreline']}" for row in display]
    fair = [100.0 * float(row.get("model_probability") or 0.0) for row in display]
    price = [100.0 * float(row.get("market_price") or 0.0) for row in display]
    colors = []
    for row in display:
        section = str(row["section"])
        recommendation = str(row.get("recommendation") or "")
        if "POTENTIAL" in recommendation:
            colors.append("#A3E635")
        elif "CAP_EXCLUDED" in recommendation:
            colors.append("#FDBA74")
        elif "NEAR_FAIR" in recommendation:
            colors.append("#67E8F9")
        elif section.startswith("CORE"):
            colors.append("#4C78A8")
        elif section == "OUTLIER":
            colors.append("#54A24B")
        elif section.startswith("UPSIDE"):
            colors.append("#F58518")
        else:
            colors.append("#F2CF5B")

    fig, ax = plt.subplots(figsize=(13, max(5.5, 0.75 * len(display) + 2.0)))
    y = list(range(len(display)))
    ax.barh(y, fair, color=colors, alpha=0.86, label="Model fair")
    ax.scatter(price, y, color="#222222", marker="D", s=44, label="Market price")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Probability / price (%)")
    ax.set_title(f"V46.4 Optimized Best Buys: {team_a} vs {team_b}", fontweight="bold")
    ax.grid(axis="x", alpha=0.22)
    xmax = max(fair + price + [1.0]) * 1.32
    ax.set_xlim(0, xmax)
    for idx, row in enumerate(display):
        payout_text = "-" if row.get("payout_multiple") is None else f"{float(row['payout_multiple']):.1f}x"
        text = (
            f"{row['recommendation']} | price {cents(row.get('market_price'))} | "
            f"fair {pct(row.get('model_probability'))} | jK {pct(row.get('joint_kelly_fraction'), 2)} | "
            f"payout {payout_text}"
        )
        ax.text(max(fair[idx], price[idx]) + xmax * 0.018, idx, text, va="center", fontsize=8.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_hit_outcomes_plot(
    path: Path,
    rows: list[dict[str, Any]],
    team_a: str,
    team_b: str,
    *,
    target_stake: float | None,
) -> None:
    display = [row for row in rows if _to_float(row.get("stake"), 0.0)]
    if not display:
        return

    role_colors = {
        "VALUE": "#0F766E",
        "COVER": "#2563EB",
        "OUTLIER_VALUE": "#D97706",
    }
    role_soft = {
        "VALUE": "#CCFBF1",
        "COVER": "#DBEAFE",
        "OUTLIER_VALUE": "#FEF3C7",
    }
    labels = [str(row.get("scoreline") or "") for row in display]
    roles = [str(row.get("role") or "") for row in display]
    tiers = [str(row.get("tier") or row.get("role") or "") for row in display]
    stakes = [float(_to_float(row.get("stake"), 0.0) or 0.0) for row in display]
    prices = [float(_to_float(row.get("price"), 0.0) or 0.0) for row in display]
    fair = [float(_to_float(row.get("model_probability"), 0.0) or 0.0) for row in display]
    raw_edge = [float(_to_float(row.get("raw_edge"), 0.0) or 0.0) for row in display]
    gross = [float(_to_float(row.get("gross_payout_if_hit"), 0.0) or 0.0) for row in display]
    net = [float(_to_float(row.get("net_profit_if_hit"), 0.0) or 0.0) for row in display]
    base_stake = [float(_to_float(row.get("base_stake"), 0.0) or 0.0) for row in display]
    surplus = [float(_to_float(row.get("surplus_stake"), 0.0) or 0.0) for row in display]

    fig = plt.figure(figsize=(14, max(7.0, 0.90 * len(display) + 4.3)), facecolor="#F8FAFC")
    grid = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[4.2, 1.35], hspace=0.34)
    ax = fig.add_subplot(grid[0])
    ax_stake = fig.add_subplot(grid[1])
    ax.set_facecolor("#FFFFFF")
    ax_stake.set_facecolor("#FFFFFF")

    y = list(range(len(display)))
    colors = [role_colors.get(role, "#64748B") for role in roles]
    soft_colors = [role_soft.get(role, "#E2E8F0") for role in roles]
    ax.barh(y, net, color=soft_colors, edgecolor=colors, linewidth=2.4, height=0.64)
    ax.scatter(gross, y, color=colors, s=68, zorder=3, label="Gross payout if hit")
    ax.axvline(0, color="#0F172A", linewidth=1.2, alpha=0.85)
    if target_stake and target_stake > 0:
        ax.axvline(target_stake, color="#94A3B8", linewidth=1.2, linestyle="--", alpha=0.85)
        ax.text(target_stake, -0.72, "target stake", ha="center", va="bottom", fontsize=8.5, color="#475569")

    ax.set_yticks(y)
    ax.set_yticklabels([f"{label}  {tier.replace('_', ' ')}" for label, tier in zip(labels, tiers)], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Profit / payout in stake currency")
    subtitle = "Net profit if that exact score hits; dots show gross payout."
    if target_stake and target_stake > 0:
        subtitle += f" Total stake target {target_stake:.2f}."
    fig.text(0.12, 0.965, f"V46.4 Hit Outcomes: {team_a} vs {team_b}", ha="left", va="top", fontsize=18, fontweight="bold", color="#0F172A")
    fig.text(0.12, 0.932, subtitle, ha="left", va="top", fontsize=10.5, color="#475569")
    ax.grid(axis="x", color="#CBD5E1", alpha=0.55, linewidth=0.9)
    ax.tick_params(axis="x", labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)

    max_x = max(gross + net + [float(target_stake or 0.0), 1.0])
    left_x = min(0.0, min(net + [0.0]))
    ax.set_xlim(left_x - max_x * 0.035, max_x * 1.20)
    for idx, row in enumerate(display):
        annotation = (
            f"net {net[idx]:+.2f}   gross {gross[idx]:.2f}   "
            f"stake {stakes[idx]:.2f} @ {prices[idx] * 100:.1f}c"
        )
        ax.text(max(net[idx], gross[idx]) + max_x * 0.025, idx, annotation, va="center", fontsize=9.2, color="#0F172A")
        detail = f"fair {fair[idx] * 100:.1f}% | edge {raw_edge[idx] * 100:+.2f}c"
        ax.text(left_x + max_x * 0.010, idx + 0.28, detail, va="center", fontsize=8.2, color="#64748B")

    ax_stake.bar(labels, base_stake, color="#CBD5E1", edgecolor="#94A3B8", linewidth=1.0, label="Base / break-even floor")
    ax_stake.bar(labels, surplus, bottom=base_stake, color=colors, alpha=0.92, label="Surplus")
    ax_stake.set_ylabel("Stake")
    ax_stake.set_title("Stake construction", loc="left", fontsize=11, fontweight="bold", color="#0F172A")
    ax_stake.grid(axis="y", color="#CBD5E1", alpha=0.45)
    ax_stake.tick_params(axis="x", labelsize=10)
    ax_stake.tick_params(axis="y", labelsize=8)
    for spine in ax_stake.spines.values():
        spine.set_visible(False)
    for idx, stake in enumerate(stakes):
        ax_stake.text(idx, stake + max(stakes) * 0.035, f"{stake:.2f}", ha="center", va="bottom", fontsize=8.5, color="#0F172A")

    handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=role_colors["VALUE"], markersize=9, label="VALUE"),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=role_colors["COVER"], markersize=9, label="COVER"),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=role_colors["OUTLIER_VALUE"], markersize=9, label="OUTLIER VALUE"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#0F172A", markersize=7, label="Gross payout dot"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.018))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.875, bottom=0.125, hspace=0.38)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def write_grid_plot(
    path: Path,
    matrix: dict[str, float],
    markets: dict[str, dict[str, Any]],
    kelly_index: dict[str, dict[str, Any]],
    card_rows: list[dict[str, Any]],
    team_a: str,
    team_b: str,
    max_goals: int,
    buffer: float,
    potential_band: float,
    favorite_side: str,
    knockout: bool,
    min_probability: float,
    min_material_kelly: float,
    min_confidence: float,
    min_story_fit: float,
    watch_threshold_margin: float,
    show_potential_value: bool = False,
) -> None:
    marker_by_score = {}
    decision_by_score = {}
    execution_stake_by_score: dict[str, float] = {}
    execution_reason_by_score: dict[str, str] = {}
    for row in card_rows:
        section = str(row.get("section") or "")
        scoreline = str(row.get("scoreline") or "")
        if scoreline:
            decision_by_score[scoreline] = str(row.get("decision") or row.get("recommendation") or "")
            stake = _to_float(row.get("execution_stake"))
            if stake is not None:
                execution_stake_by_score[scoreline] = stake
                execution_reason_by_score[scoreline] = str(row.get("buy_hold_inclusion_reason") or row.get("value_status") or "")
        if section.startswith("CORE"):
            marker_by_score[scoreline] = f"#{section[-1]}"
        elif section == "OUTLIER":
            marker_by_score[scoreline] = "OUTLIER"
        elif section.startswith("UPSIDE"):
            marker_by_score[scoreline] = "UP"
        elif section.startswith("BUY_HOLD"):
            # card_rows here is the FINAL buy-hold execution list (section=BUY_HOLD_N
            # for every bought scoreline, VALUE/COVER/OUTLIER alike) -- the plain
            # section-name check above only fires for build_core_rows()' pre-staking
            # output, which write_grid_plot() never actually receives, so the
            # OUTLIER-tier pick was silently falling through to the generic "BH"
            # marker. Check the tier/role directly instead so it gets its own marker.
            role = str(row.get("buy_hold_role") or "")
            tier = str(row.get("buy_hold_tier") or "")
            if role == "OUTLIER_VALUE" or tier == "OUTLIER":
                marker_by_score[scoreline] = "OUTLIER"
            else:
                marker_by_score[scoreline] = "BH"

    ranked_scorelines = sorted(
        (
            (scoreline, fair)
            for scoreline, fair in matrix.items()
            if "-" in scoreline
            and scoreline.split("-", 1)[0].isdigit()
            and scoreline.split("-", 1)[1].isdigit()
            and fair > 0
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    rank_by_score = {scoreline: rank for rank, (scoreline, _) in enumerate(ranked_scorelines, start=1)}

    status_style = {
        "BUY": {"edge": "#166534", "face": "#22c55e", "text_face": "#bbf7d0", "label": "BUY"},
        "POTENTIAL_VALUE": {"edge": "#4d7c0f", "face": "#bef264", "text_face": "#ecfccb", "label": "PV"},
        "CAP_EXCLUDED": {"edge": "#7c2d12", "face": "#fdba74", "text_face": "#ffedd5", "label": "CAP"},
        "NEAR_FAIR_WATCH": {"edge": "#0e7490", "face": "#67e8f9", "text_face": "#cffafe", "label": "FAIR"},
        "WATCH": {"edge": "#a16207", "face": "#fde047", "text_face": "#fef9c3", "label": "WATCH"},
        "DONT": {"edge": "#b91c1c", "face": "#f87171", "text_face": "#fee2e2", "label": "NO"},
    }
    rank_color = {"#1": "#1f77b4", "#2": "#1f77b4", "#3": "#1f77b4", "OUTLIER": "#2ca02c", "UP": "#ff7f0e"}

    def cell_status(scoreline: str, fair: float, price: float | None) -> str:
        decision = decision_by_score.get(scoreline, "")
        if decision == "BUY":
            return "BUY"
        if decision == "POTENTIAL_VALUE":
            return "POTENTIAL_VALUE" if show_potential_value else "DONT"
        if decision == "CAP_EXCLUDED":
            return "CAP_EXCLUDED"
        if decision == "NEAR_FAIR_WATCH":
            return "NEAR_FAIR_WATCH"
        if decision == "WATCH":
            return "WATCH"
        if decision == "NO":
            return "DONT"
        if price is None or price <= 0 or fair <= 0:
            return "DONT"
        market = markets.get(scoreline) or {}
        fit, _story = story_fit(scoreline, favorite_side=favorite_side, knockout=knockout)
        reliability = reliability_from_scoreline(scoreline, market)
        prob_conf = max(0.30, min(1.0, fair / max(min_probability * 2.0, 1e-6)))
        liq_conf = liquidity_confidence(_to_float(market.get("liquidity")))
        confidence = 0.50 * reliability + 0.30 * prob_conf + 0.20 * liq_conf
        grid_decision = decision_from_metrics(
            scoreline=scoreline,
            fair=fair,
            price=price,
            buffer=buffer,
            potential_band=potential_band,
            min_material_kelly=min_material_kelly,
            kelly_metrics=kelly_index.get(scoreline),
            confidence=confidence,
            min_confidence=min_confidence,
            story_fit_value=fit,
            min_story_fit=min_story_fit,
            watch_threshold_margin=watch_threshold_margin,
            model_rank=rank_by_score.get(scoreline),
            max_buy_rank=10,
        )["decision"]
        if grid_decision == "BUY":
            return "BUY"
        if grid_decision == "POTENTIAL_VALUE":
            return "POTENTIAL_VALUE" if show_potential_value else "DONT"
        if grid_decision == "CAP_EXCLUDED":
            return "CAP_EXCLUDED"
        if grid_decision == "NEAR_FAIR_WATCH":
            return "NEAR_FAIR_WATCH"
        if grid_decision == "WATCH":
            return "WATCH"
        return "DONT"

    status_rank = {"DONT": 0, "WATCH": 1, "NEAR_FAIR_WATCH": 2, "CAP_EXCLUDED": 3, "POTENTIAL_VALUE": 4, "BUY": 5}
    action_grid = []
    for b_goals in range(max_goals + 1):
        action_row = []
        for a_goals in range(max_goals + 1):
            scoreline = f"{a_goals}-{b_goals}"
            fair = matrix.get(scoreline, 0.0)
            market = markets.get(scoreline)
            price = _to_float((market or {}).get("raw_yes_price"), _to_float((market or {}).get("yes_price")))
            action_row.append(status_rank[cell_status(scoreline, fair, price)])
        action_grid.append(action_row)

    fig, ax = plt.subplots(figsize=(13.5, 10.5))
    action_cmap = ListedColormap(
        [
            status_style["DONT"]["face"],
            status_style["WATCH"]["face"],
            status_style["NEAR_FAIR_WATCH"]["face"],
            status_style["CAP_EXCLUDED"]["face"],
            status_style["POTENTIAL_VALUE"]["face"],
            status_style["BUY"]["face"],
        ]
    )
    action_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], action_cmap.N)
    ax.imshow(action_grid, cmap=action_cmap, norm=action_norm, origin="lower", alpha=0.84)
    ax.set_xticks(range(max_goals + 1))
    ax.set_yticks(range(max_goals + 1))
    ax.set_xlabel(f"{team_a} goals")
    ax.set_ylabel(f"{team_b} goals")
    ax.set_title("V46.4 score grid: buy-hold stakes, value watches, and no-buys", fontweight="bold", fontsize=15)
    ax.set_xticks([tick - 0.5 for tick in range(max_goals + 2)], minor=True)
    ax.set_yticks([tick - 0.5 for tick in range(max_goals + 2)], minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)

    for b_goals in range(max_goals + 1):
        for a_goals in range(max_goals + 1):
            scoreline = f"{a_goals}-{b_goals}"
            fair = matrix.get(scoreline, 0.0)
            market = markets.get(scoreline)
            price = _to_float((market or {}).get("raw_yes_price"), _to_float((market or {}).get("yes_price")))
            marker = marker_by_score.get(scoreline, "")
            status = cell_status(scoreline, fair, price)
            style = status_style[status]
            if fair <= 0 and price is None and not marker:
                continue
            lw = 3.4 if marker else 1.35
            ax.add_patch(
                plt.Rectangle(
                    (a_goals - 0.5, b_goals - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor=style["edge"],
                    linewidth=lw,
                )
            )
            label = f"{fair*100:.1f}%"
            if price is not None:
                label += f"\n{price*100:.1f}c"
                if scoreline in execution_stake_by_score and status == "BUY":
                    label += f"\nstake {execution_stake_by_score[scoreline]:.2f}"
                    reason = execution_reason_by_score.get(scoreline, "")
                    if reason == "probability_cover":
                        label += "\nCOVER"
                    elif reason in {"kelly_value", "kelly_or_value"}:
                        label += "\nVALUE"
                    else:
                        label += "\nEXEC"
                else:
                    label += f"\njK {pct((kelly_index.get(scoreline) or {}).get('joint_kelly_fraction'), 2)}"
            rank_label = f"#{rank_by_score[scoreline]}" if scoreline in rank_by_score else ""
            if marker:
                label += f"\n{marker}"
                if marker != rank_label and rank_label:
                    label += f" {rank_label}"
            elif rank_label:
                label += f"\n{rank_label}"
            label += f"\n{style['label']}"
            ax.text(
                a_goals,
                b_goals,
                label,
                ha="center",
                va="center",
                fontsize=8.0 if marker else 7.1,
                color="#111111",
                fontweight="bold" if marker else "normal",
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "facecolor": style["text_face"],
                    "edgecolor": rank_color.get(marker, "none"),
                    "linewidth": 1.5 if marker else 0.0,
                    "alpha": 0.92,
                },
            )

    legend_handles = [
        plt.Line2D([0], [0], color="#15803d", lw=4, label="BUY: executable buy-hold stake"),
    ]
    if show_potential_value:
        legend_handles.append(plt.Line2D([0], [0], color="#65a30d", lw=4, label="PV: joint Kelly final stake too small"))
    legend_handles += [
        plt.Line2D([0], [0], color="#ea580c", lw=4, label="CAP: tranche-cap excluded"),
        plt.Line2D([0], [0], color="#0891b2", lw=4, label=f"FAIR: above fair but within {cents(potential_band)}"),
        plt.Line2D([0], [0], color="#ca8a04", lw=4, label="WATCH: positive EV but not entry"),
        plt.Line2D([0], [0], color="#dc2626", lw=4, label="NO: above fair/no market"),
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a V46.4-BaseV51 optimized best-buy card. By default this runs V51 "
            "(v11+v49 bivariate-NegBin base, plus additive v39/v29 outlier tabs) as the "
            "hidden prediction layer, feeds it through V42's Polymarket fetch/edge "
            "pipeline, and applies V46.4's unchanged buy-and-hold staking logic. Use "
            "--input to reuse an existing hidden-layer output."
        )
    )
    parser.add_argument("--input", help="Existing hidden-layer output directory with model_summary and Polymarket CSVs.")
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--v51-outdir",
        help="Optional directory for the hidden V51+Polymarket artifacts. Defaults to OUTDIR/_v51_source/<timestamp>.",
    )
    parser.add_argument(
        "--no-plots-v51-layer",
        dest="no_plots_v51_layer",
        action="store_true",
        help="Skip V51's own prediction plots in the hidden layer (V46.4's own plots are unaffected).",
    )
    parser.add_argument(
        "--show-v51-output",
        action="store_true",
        help="Print the hidden V51+Polymarket layer's JSON summary before the V46.4 summary.",
    )
    parser.add_argument(
        "--allow-empty-polymarket",
        action="store_true",
        help="Bypass the live Polymarket guardrail for offline/model-only debugging.",
    )

    parser.add_argument("--team-a")
    parser.add_argument("--team-b")
    parser.add_argument("--host-a", action="store_true")
    parser.add_argument("--host-b", action="store_true")
    parser.add_argument("--knockout", action="store_true")
    parser.add_argument(
        "--force-outlier-scoreline",
        default=None,
        help="Designate a specific scoreline (e.g. '3-1') as the OUTLIER-tier pick, "
        "overriding v51's own v29/v39 outlier gates (which may not fire for every "
        "match). Its probability is looked up from v51's own score matrix, not guessed.",
    )
    parser.add_argument(
    "--match-number",
    type=int,
    default=None,
    help="Tournament match number. Matches 1-72 are group-stage matches; matches 73+ are knockouts for automatic buy filters.",
    )
    parser.add_argument("--worldcupsai-zip")
    parser.add_argument("--team-train")
    parser.add_argument("--team-test")
    parser.add_argument("--box-data")
    parser.add_argument("--results-data")
    parser.add_argument(
        "--fotmob-match-facts",
        help="Unused by V51 (V51 has no FotMob current-form dependency, unlike V42's own "
        "accuracy stack). Accepted only for CLI compatibility with tune_v46_4_tiers.py, which "
        "passes this flag to whichever --v46-script it invokes.",
    )
    parser.add_argument("--former-names")
    parser.add_argument("--prediction-year", type=int)
    parser.add_argument(
        "--wc-prestige-weight",
        type=float,
        default=600.0,
        help="World Cup prestige weight fed to V51's base model. Default 600.0 (validated setting).",
    )
    parser.add_argument(
        "--use-volume-normalized-weighting",
        action="store_true",
        help="Use V51/v11's volume-normalized prestige-tier weighting instead of the flat prestige weight.",
    )
    parser.add_argument(
        "--score-matrix-r",
        type=float,
        default=25.0,
        help="Bivariate-NegBin score-matrix dispersion parameter (v49 calibrated). Default 25.0.",
    )
    parser.add_argument(
        "--fbref-world-cup-csv",
        default=None,
        help="FBref World Cup matches CSV for V51's expanded training pool. Defaults to data/fbref_world_cup_matches.csv.",
    )
    parser.add_argument(
        "--fbref-international-csv",
        default=None,
        help="FBref international matches CSV for V51's expanded training pool. Defaults to data/fbref_international_matches.csv.",
    )
    parser.add_argument("--polymarket-query")
    parser.add_argument("--polymarket-event-slug")
    parser.add_argument("--polymarket-sports-url")
    parser.add_argument("--auto-polymarket", action="store_true")
    parser.add_argument("--polymarket-json")
    parser.add_argument("--no-fetch-polymarket", action="store_true")
    parser.add_argument("--gamma-limit", type=int)
    parser.add_argument("--min-edge", type=float)
    parser.add_argument("--min-ev", type=float)
    parser.add_argument("--uncertainty-buffer", type=float)
    parser.add_argument("--reference-v43-output")
    parser.add_argument("--fetch-clob-orderbook", action="store_true")
    parser.add_argument("--no-clob-orderbook", action="store_true")
    parser.add_argument("--price-history-root")

    parser.add_argument(
        "--buffer",
        type=float,
        default=0.005,
        help="Entry buffer below fair price. Default 0.005 = 0.5c.",
    )
    parser.add_argument(
        "--disable-entry-buffer",
        action="store_true",
        help=(
            "ABLATION SWITCH (off by default): forces --buffer to 0, removing the entry-price "
            "margin below fair value entirely. Use this to test whether the buffer is load-bearing "
            "for backtest performance -- it does not change any other behavior."
        ),
    )
    parser.add_argument(
        "--potential-band",
        type=float,
        default=0.005,
        help="Potential-buy band around fair price. Default 0.005 = within 0.5c above or below fair.",
    )
    parser.add_argument("--min-upside-probability", type=float, default=0.006)
    parser.add_argument(
        "--probability-floor",
        type=float,
        default=0.01,
        help=(
            "Legacy/no-op in the joint Kelly rule. Kept for backward-compatible commands; "
            "the endogenous joint threshold now handles tiny fragile scores."
        ),
    )
    parser.add_argument(
        "--min-quarter-kelly",
        type=float,
        default=0.0025,
        help=(
            "Backward-compatible alias for minimum final shrunk joint-Kelly stake for BUY. "
            "Default 0.0025 = 0.25%% bankroll."
        ),
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=None,
        help="Execution bankroll used to convert minimum order size into a minimum final Kelly fraction.",
    )
    parser.add_argument(
        "--min-order-size",
        type=float,
        default=1.0,
        help="Minimum executable order size in bankroll currency. Default 1.0 = $1/€1.",
    )
    parser.add_argument(
        "--buy-hold-rank-limit",
        type=int,
        default=10,
        help="Only allow BUY labels for scorelines with model probability rank at or above this cutoff. Default 10.",
    )
    parser.add_argument(
        "--stake-mode",
        choices=["breakeven-plus-value"],
        default="breakeven-plus-value",
        help="Final V46.4 staking objective. Default: breakeven-plus-value.",
    )
    parser.add_argument(
        "--stake-profile",
        choices=["value-heavy", "tiered-balanced", "both"],
        default="value-heavy",
        help=(
            "Surplus allocation profile. value-heavy preserves current V46.4 behavior; "
            "tiered-balanced funds VALUE_1/VALUE_2/COVER_1/COVER_2/OUTLIER tiers; both writes both profiles."
        ),
    )
    parser.add_argument(
        "--execution-target-stake",
        type=float,
        default=9.0,
        help="Total buy-and-hold stake to allocate. Default 9.0.",
    )
    parser.add_argument("--stake-rounding", type=float, default=0.05, help="Round final stakes to this increment.")
    parser.add_argument("--value-surplus-weight", type=float, default=1.00)
    parser.add_argument("--outlier-surplus-weight", type=float, default=0.45)
    parser.add_argument("--cover-surplus-weight", type=float, default=0.075)
    # Defaults below are None here and resolved after parsing (see
    # STAGE_TIER_DEFAULTS / the args.knockout block below main()'s
    # parse_known_args() call) -- group-stage and knockout matches got
    # separately re-tuned from tune_v46_4_tiers.py's already-cached candidate
    # data (split_tier_weights_by_stage.py) and want different weights:
    # group stage favors COVER_1 heavily, knockout favors COVER_2 instead.
    # Passing any of these flags explicitly overrides the stage-based default.
    # VALUE_1 has no flag -- it's the fixed anchor weight (1.0) every other
    # tier's weight is relative to. OUTLIER weight is NOT stage-conditional:
    # it was never validated in any split (see module docstring) so there's
    # no basis to prefer one arbitrary number over another by stage.
    parser.add_argument("--tier-value-2-weight", type=float, default=None)
    parser.add_argument("--tier-cover-1-weight", type=float, default=None)
    parser.add_argument("--tier-cover-2-weight", type=float, default=None)
    parser.add_argument("--tier-outlier-weight", type=float, default=0.37103902262412886)
    parser.add_argument("--tier-cover-1-cap", type=float, default=None)
    parser.add_argument("--tier-cover-2-cap", type=float, default=None)
    parser.add_argument("--tier-negative-edge-cap-multiplier", type=float, default=None)
    parser.add_argument(
        "--overpriced-cover-surplus-cap",
        type=float,
        default=0.05,
        help="Maximum surplus added to each negative-edge COVER score. Default 0.05.",
    )
    parser.add_argument(
        "--buy-hold-max-bets",
        type=int,
        default=5,
        help="Maximum number of exact-score bets in the buy-hold execution card. Default 5.",
    )
    parser.add_argument(
        "--buy-hold-min-bets",
        type=int,
        default=4,
        help="Preferred minimum number of exact-score bets when enough candidates exist. Default 4.",
    )
    parser.add_argument(
    "--allow-low-score-knockout-buys",
    action="store_true",
    help="Allow 0-0, 1-0, and 0-1 in knockout buy-hold candidate selection. By default these are excluded only for knockout matches.",
    )
    parser.add_argument(
        "--probability-cover-count",
        type=int,
        default=5,
        help="Force the top N model-probability exact scorelines into the buy-hold card before adding value names. Default 5.",
    )
    parser.add_argument(
        "--min-value-bets",
        type=int,
        default=2,
        help="Minimum VALUE selections when available. Default 2.",
    )
    parser.add_argument(
        "--min-cover-bets",
        type=int,
        default=1,
        help="Minimum COVER selections when available. Default 1.",
    )
    parser.add_argument(
        "--max-negative-edge-covers",
        type=int,
        default=2,
        help="Maximum selected COVER scores with negative raw edge. Default 2.",
    )
    parser.add_argument(
        "--no-require-outlier",
        action="store_true",
        help="Do not force the model outlier into the selected execution card.",
    )
    parser.add_argument(
        "--dry-run-selection-debug",
        action="store_true",
        help="Write V46.4 candidate/selection debug CSV and exit before final staking/plots.",
    )
    parser.add_argument(
        "--ignore-scorelines",
        type=str,
        default="0-0,0-1,1-0",
        help="Comma-separated exact scorelines to exclude from the buy-hold execution card. Default excludes 0-0, 0-1, and 1-0.",
    )
    parser.add_argument(
        "--max-potential-value",
        type=int,
        default=2,
        help="Maximum number of PV rows to show when --show-potential-value is used. Default 2.",
    )
    parser.add_argument(
        "--exclude-any-other-score",
        action="store_true",
        default=True,
        help="Exclude the broad Any Other Score bucket. Default enabled for buy-hold execution.",
    )
    parser.add_argument(
        "--show-potential-value",
        action="store_true",
        help="Show PV rows/green PV grid cells. By default PVs are hidden to reduce confusion.",
    )
    parser.add_argument(
        "--min-kelly-shrink",
        type=float,
        default=0.10,
        help="Minimum bucket-specific shrink multiplier applied after the joint Kelly solve.",
    )
    parser.add_argument(
        "--max-kelly-shrink",
        type=float,
        default=1.00,
        help="Maximum bucket-specific shrink multiplier applied after the joint Kelly solve.",
    )
    parser.add_argument(
        "--kelly-tranche-budget",
        type=float,
        default=0.05,
        help="Cap on total final joint-Kelly stake for one match. Default 0.05 = 5%% bankroll.",
    )
    parser.add_argument(
        "--disable-kelly-tranche-cap",
        action="store_true",
        help=(
            "ABLATION SWITCH (off by default): forces --kelly-tranche-budget to 0, removing the "
            "growth-optimal tranche cap entirely so the shrunk joint-Kelly stakes are used uncapped. "
            "Use this to test whether the tranche cap is load-bearing for backtest performance -- "
            "it does not change any other behavior."
        ),
    )
    parser.add_argument(
        "--watch-threshold-margin",
        type=float,
        default=0.03,
        help="Relative distance below the joint Kelly threshold still shown as WATCH. Default 0.03 = within 3%%.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.50,
        help="Minimum confidence score for BUY.",
    )
    parser.add_argument(
        "--min-story-fit",
        type=float,
        default=0.75,
        help="Minimum story-fit score for BUY.",
    )
    parser.add_argument(
        "--disable-value-diversification",
        action="store_true",
        help="ABLATION SWITCH (off by default): disable the correlation-aware VALUE-pick "
        "portfolio selection (anchor + uncorrelated hedge instead of pure top-EV ranking), "
        "falling back to picking VALUE slots by raw EV alone. Use this to test whether "
        "diversification is load-bearing -- it does not change any other behavior.",
    )
    parser.add_argument(
        "--no-directional-risk-hedge",
        action="store_true",
        help="Disable the clean-sheet stack hedge that can swap one weak/fair clean-sheet score for a similarly priced draw score.",
    )
    parser.add_argument(
        "--directional-hedge-price-tolerance",
        type=float,
        default=0.012,
        help="Absolute max market-price gap for directional hedge replacement. Default 0.012 = 1.2c.",
    )
    parser.add_argument(
        "--directional-hedge-relative-price-tolerance",
        type=float,
        default=0.35,
        help="Relative max market-price gap for directional hedge replacement. Default 0.35.",
    )
    parser.add_argument(
        "--directional-hedge-model-market-absolute-buffer",
        type=float,
        default=0.012,
        help="Absolute model-market agreement buffer for the replacement draw hedge. Default 0.012 = 1.2c.",
    )
    parser.add_argument(
        "--directional-hedge-model-market-relative-buffer",
        type=float,
        default=0.35,
        help="Relative model-market agreement buffer for the replacement draw hedge. Default 0.35.",
    )
    parser.add_argument(
        "--directional-hedge-big-elo-gap",
        type=float,
        default=250.0,
        help="If Elo/rating gap is at least this large, do not replace clean-sheet stacks. Default 250.",
    )
    parser.add_argument(
        "--directional-hedge-max-replacements",
        type=int,
        default=1,
        help="Maximum number of clean-sheet scores to replace with draw hedges. Default 1.",
    )
    parser.add_argument("--watch-limit", type=int, default=8)
    parser.add_argument("--grid-max-goals", type=int, default=6)
    parser.add_argument("--no-plots", action="store_true")
    args, passthrough_args = parser.parse_known_args()

    if args.disable_entry_buffer:
        print(
            "[ablation] --disable-entry-buffer set: forcing --buffer to 0 "
            f"(was {args.buffer}). No entry-price margin below fair value is required.",
            file=sys.stderr,
        )
        args.buffer = 0.0
    if args.disable_kelly_tranche_cap:
        print(
            "[ablation] --disable-kelly-tranche-cap set: forcing --kelly-tranche-budget to 0 "
            f"(was {args.kelly_tranche_budget}). Shrunk joint-Kelly stakes are used uncapped.",
            file=sys.stderr,
        )
        args.kelly_tranche_budget = 0.0

    outdir = Path(args.outdir)
    if args.match_number is not None and args.match_number > GROUP_STAGE_MATCH_COUNT:
        args.knockout = True

    stage_defaults = STAGE_TIER_DEFAULTS["knockout" if args.knockout else "group_stage"]
    for flag_name, stage_value in stage_defaults.items():
        if getattr(args, flag_name) is None:
            setattr(args, flag_name, stage_value)

    input_dir = run_v51_for_v46(args, passthrough_args)
    outdir.mkdir(parents=True, exist_ok=True)
    plots_dir = outdir / "plots"

    summary = json.loads((input_dir / "model_summary.json").read_text(encoding="utf-8"))
    market_rows = read_csv(input_dir / "polymarket_exact_score_edges.csv")
    # Remove Any Other Score from all downstream portfolio and display logic.
    # This avoids letting the broad tail bucket consume Kelly mass when the user only wants exact-score bets.
    market_rows_for_validation = list(market_rows)
    validate_polymarket_inputs(
        summary,
        market_rows,
        input_dir,
        allow_empty=args.allow_empty_polymarket,
    )
    if args.exclude_any_other_score:
        market_rows = [row for row in market_rows if not is_any_other_score(str(row.get("scoreline") or ""))]
    matrix_rows = read_csv(input_dir / "model_fair_scoreline_probabilities.csv")
    markets = market_index(market_rows)
    matrix = matrix_index(matrix_rows)
    rank_by_score = model_rank_index(matrix)
    ignored_scorelines = parse_scoreline_set(args.ignore_scorelines)
    ignored_scorelines |= low_score_exclusion_set(
        knockout=bool(args.knockout),
        match_number=args.match_number,
        allow_low_scores=bool(args.allow_low_score_knockout_buys),
    )
    executable_min_joint_kelly = args.min_quarter_kelly
    if args.bankroll is not None and args.bankroll > 0 and args.min_order_size is not None and args.min_order_size > 0:
        executable_min_joint_kelly = max(args.min_quarter_kelly, args.min_order_size / args.bankroll)
    kelly_index, joint_kelly_summary = compute_joint_kelly_index(
        market_rows,
        min_shrink=args.min_kelly_shrink,
        max_shrink=args.max_kelly_shrink,
        tranche_budget=args.kelly_tranche_budget,
    )

    team_a = str(summary.get("team_a") or "Team A")
    team_b = str(summary.get("team_b") or "Team B")
    favorite_side = detect_favorite(summary)
    knockout = bool((summary.get("prediction_outlier") or {}).get("source")) or True
    outlier_scoreline = scoreline_from_summary_row(summary.get("prediction_outlier") or {})

    core_rows = build_core_rows(
        summary,
        markets,
        matrix,
        kelly_index,
        buffer=args.buffer,
        potential_band=args.potential_band,
        min_material_kelly=executable_min_joint_kelly,
        min_confidence=args.min_confidence,
        min_story_fit=args.min_story_fit,
        watch_threshold_margin=args.watch_threshold_margin,
        rank_by_score=rank_by_score,
        max_buy_rank=args.buy_hold_rank_limit,
    )
    core_scorelines = {str(row["scoreline"]) for row in core_rows}
    upside_rows = build_upside_rows(
        market_rows,
        core_scorelines,
        kelly_index,
        favorite_side=favorite_side,
        knockout=knockout,
        buffer=args.buffer,
        potential_band=args.potential_band,
        min_probability=args.min_upside_probability,
        min_material_kelly=executable_min_joint_kelly,
        min_confidence=args.min_confidence,
        min_story_fit=args.min_story_fit,
        watch_threshold_margin=args.watch_threshold_margin,
        rank_by_score=rank_by_score,
        max_buy_rank=args.buy_hold_rank_limit,
    )
    selected_scorelines = core_scorelines | {str(row["scoreline"]) for row in upside_rows}
    watch_rows = build_watch_rows(
        market_rows,
        selected_scorelines,
        kelly_index,
        favorite_side=favorite_side,
        knockout=knockout,
        buffer=args.buffer,
        potential_band=args.potential_band,
        min_probability=args.min_upside_probability,
        min_material_kelly=executable_min_joint_kelly,
        min_confidence=args.min_confidence,
        min_story_fit=args.min_story_fit,
        watch_threshold_margin=args.watch_threshold_margin,
        rank_by_score=rank_by_score,
        max_buy_rank=args.buy_hold_rank_limit,
        limit=args.watch_limit,
    )
    card_rows = core_rows + upside_rows + watch_rows

    selection_debug_rows: list[dict[str, Any]] = []
    hit_outcome_rows: list[dict[str, Any]] = []
    buy_hold_validation: dict[str, Any] = {}
    require_outlier = not args.no_require_outlier
    requested_stake_profiles = ["value-heavy", "tiered-balanced"] if args.stake_profile == "both" else [args.stake_profile]
    primary_stake_profile = requested_stake_profiles[0]
    stake_profile_results: dict[str, dict[str, Any]] = {}

    if args.dry_run_selection_debug:
        dry_candidates = build_buy_hold_candidates(
            market_rows,
            kelly_index,
            outlier_scoreline=outlier_scoreline,
            favorite_side=favorite_side,
            knockout=knockout,
            min_probability=args.min_upside_probability,
            min_confidence=args.min_confidence,
            min_story_fit=args.min_story_fit,
            rank_by_score=rank_by_score,
            max_buy_rank=args.buy_hold_rank_limit,
            bankroll=args.bankroll,
            min_order_size=args.min_order_size,
            probability_cover_count=args.probability_cover_count,
            ignored_scorelines=ignored_scorelines,
        )
        dry_selected = select_buy_hold_candidates(
            dry_candidates,
            outlier_scoreline=outlier_scoreline,
            target_stake=args.execution_target_stake or 0.0,
            min_order_size=args.min_order_size,
            max_bets=args.buy_hold_max_bets,
            min_bets=args.buy_hold_min_bets,
            min_value_bets=args.min_value_bets,
            min_cover_bets=args.min_cover_bets,
            max_negative_edge_covers=args.max_negative_edge_covers,
            require_outlier=require_outlier,
            diversify_value_picks=not args.disable_value_diversification,
        )
        selection_debug_rows = build_selection_debug_rows(dry_candidates, dry_selected)
        debug_path = outdir / "v46_4_selection_debug.csv"
        summary_path = outdir / "v46_4_basev51_summary.json"
        write_csv(debug_path, selection_debug_rows)
        dry_summary = {
            "method": "v46_4_basev51",
            "dry_run_selection_debug": True,
            "input": str(input_dir),
            "team_a": team_a,
            "team_b": team_b,
            "favorite_side": favorite_side,
            "outlier_scoreline": outlier_scoreline,
            "selection_constraints": {
                "buy_hold_max_bets": args.buy_hold_max_bets,
                "buy_hold_min_bets": args.buy_hold_min_bets,
                "min_value_bets": args.min_value_bets,
                "min_cover_bets": args.min_cover_bets,
                "max_negative_edge_covers": args.max_negative_edge_covers,
                "require_outlier": require_outlier,
            },
            "stake_profile": args.stake_profile,
            "candidate_count": len(dry_candidates),
            "selected_scorelines": [row.get("scoreline") for row in dry_selected],
            "outputs": {
                "selection_debug_csv": str(debug_path),
                "summary_json": str(summary_path),
            },
        }
        summary_path.write_text(json.dumps(dry_summary, indent=2), encoding="utf-8")
        print(json.dumps(dry_summary, indent=2))
        return

    execution_rows: list[dict[str, Any]] = []
    if args.stake_mode == "breakeven-plus-value" and args.execution_target_stake is not None:
        for profile in requested_stake_profiles:
            profile_execution_rows, profile_debug_rows, profile_validation, profile_hit_outcome_rows = build_buy_hold_execution_rows(
                market_rows,
                kelly_index,
                outlier_scoreline=outlier_scoreline,
                favorite_side=favorite_side,
                knockout=knockout,
                min_probability=args.min_upside_probability,
                min_confidence=args.min_confidence,
                min_story_fit=args.min_story_fit,
                rank_by_score=rank_by_score,
                max_buy_rank=args.buy_hold_rank_limit,
                bankroll=args.bankroll,
                target_stake=args.execution_target_stake,
                min_order_size=args.min_order_size,
                stake_rounding=args.stake_rounding,
                max_bets=args.buy_hold_max_bets,
                min_bets=args.buy_hold_min_bets,
                probability_cover_count=args.probability_cover_count,
                value_surplus_weight=args.value_surplus_weight,
                outlier_surplus_weight=args.outlier_surplus_weight,
                cover_surplus_weight=args.cover_surplus_weight,
                overpriced_cover_surplus_cap=args.overpriced_cover_surplus_cap,
                tier_value_2_weight=args.tier_value_2_weight,
                tier_cover_1_weight=args.tier_cover_1_weight,
                tier_cover_2_weight=args.tier_cover_2_weight,
                tier_outlier_weight=args.tier_outlier_weight,
                tier_cover_1_cap=args.tier_cover_1_cap,
                tier_cover_2_cap=args.tier_cover_2_cap,
                tier_negative_edge_cap_multiplier=args.tier_negative_edge_cap_multiplier,
                min_value_bets=args.min_value_bets,
                min_cover_bets=args.min_cover_bets,
                max_negative_edge_covers=args.max_negative_edge_covers,
                require_outlier=require_outlier,
                stake_profile=profile,
                ignored_scorelines=ignored_scorelines,
                diversify_value_picks=not args.disable_value_diversification,
                directional_risk_hedge=not args.no_directional_risk_hedge,
                directional_hedge_price_tolerance=args.directional_hedge_price_tolerance,
                directional_hedge_relative_price_tolerance=args.directional_hedge_relative_price_tolerance,
                directional_hedge_model_market_absolute_buffer=args.directional_hedge_model_market_absolute_buffer,
                directional_hedge_model_market_relative_buffer=args.directional_hedge_model_market_relative_buffer,
                directional_hedge_big_elo_gap=args.directional_hedge_big_elo_gap,
                directional_hedge_max_replacements=args.directional_hedge_max_replacements,
                summary_for_directional_hedge=summary,
            )
            stake_profile_results[profile] = {
                "execution_rows": profile_execution_rows,
                "selection_debug_rows": profile_debug_rows,
                "validation": profile_validation,
                "hit_outcome_rows": profile_hit_outcome_rows,
            }
        for profile, result in stake_profile_results.items():
            profile_validation = result.get("validation") or {}
            if profile_validation and not profile_validation.get("ok", False):
                slug = stake_profile_slug(profile)
                write_csv(outdir / f"v46_4_selection_debug_{slug}.csv", list(result.get("selection_debug_rows") or []))
                write_csv(outdir / f"v46_4_hit_outcomes_{slug}.csv", list(result.get("hit_outcome_rows") or []))
                errors = "; ".join(profile_validation.get("errors") or [])
                print(
                    "\n".join(
                        [
                            "",
                            "=" * 78,
                            f"NO BUY: {team_a} vs {team_b} ({profile})",
                            f"RuntimeError: V46.4 buy-hold validation failed for {profile}: {errors}",
                            "The staking layer can't build a valid card within its caps at these prices "
                            "-- the market is tripping balls relative to the model. Treat this match as a pass.",
                            "=" * 78,
                            "",
                        ]
                    ),
                    file=sys.stderr,
                )
                sys.exit(1)
        primary_result = stake_profile_results.get(primary_stake_profile) or {}
        execution_rows = list(primary_result.get("execution_rows") or [])
        selection_debug_rows = list(primary_result.get("selection_debug_rows") or [])
        buy_hold_validation = dict(primary_result.get("validation") or {})
        hit_outcome_rows = list(primary_result.get("hit_outcome_rows") or [])
        if execution_rows:
            # In execution mode, the card is the executable stake plan plus at most a small PV appendix.
            execution_scores = {str(exec_row.get("scoreline")) for exec_row in execution_rows}
            supplemental_pv_rows: list[dict[str, Any]] = []
            if args.show_potential_value and args.max_potential_value > 0:
                pv_pool = [
                    row for row in (core_rows + upside_rows + watch_rows)
                    if row.get("decision") == "POTENTIAL_VALUE"
                    and str(row.get("scoreline")) not in execution_scores
                    and str(row.get("scoreline")) not in ignored_scorelines
                ]
                pv_pool.sort(key=lambda row: (float(row.get("joint_kelly_fraction") or 0.0), float(row.get("model_probability") or 0.0)), reverse=True)
                supplemental_pv_rows = pv_pool[: args.max_potential_value]
            card_rows = execution_rows + supplemental_pv_rows
            core_rows = [row for row in core_rows if str(row.get("scoreline")) in execution_scores]
            upside_rows = []
            watch_rows = supplemental_pv_rows
    elif not args.show_potential_value:
        card_rows = [row for row in card_rows if row.get("decision") != "POTENTIAL_VALUE"]
    elif args.max_potential_value >= 0:
        pv_seen = 0
        limited_rows = []
        for row in card_rows:
            if row.get("decision") == "POTENTIAL_VALUE":
                if pv_seen >= args.max_potential_value:
                    continue
                pv_seen += 1
            limited_rows.append(row)
        card_rows = limited_rows

    selected_rows = execution_rows if execution_rows else core_rows + upside_rows
    potential_value_rows = [row for row in card_rows if row.get("decision") == "POTENTIAL_VALUE"] if args.show_potential_value else []
    cap_excluded_rows = [row for row in card_rows if row.get("decision") == "CAP_EXCLUDED"]
    near_fair_watch_rows = [row for row in card_rows if row.get("decision") == "NEAR_FAIR_WATCH"]
    buy_rows = [row for row in card_rows if row.get("decision") == "BUY"]
    watch_only_rows = [row for row in card_rows if row.get("decision") == "WATCH"]
    no_rows = [row for row in card_rows if row.get("decision") == "NO"]

    stake_profile_summaries: dict[str, dict[str, Any]] = {}
    for profile, result in stake_profile_results.items():
        slug = stake_profile_slug(profile)
        if args.stake_profile == "both":
            profile_outputs = {
                "buy_card_csv": str(outdir / f"v46_4_buy_card_{slug}.csv"),
                "selection_debug_csv": str(outdir / f"v46_4_selection_debug_{slug}.csv"),
                "hit_outcomes_csv": str(outdir / f"v46_4_hit_outcomes_{slug}.csv"),
                "buy_card_plot": str(plots_dir / f"v46_4_buy_card_{slug}.png"),
                "hit_outcomes_plot": str(plots_dir / f"v46_4_hit_outcomes_{slug}.png"),
            }
        else:
            profile_outputs = {
                "buy_card_csv": str(outdir / "v46_4_buy_card.csv"),
                "selection_debug_csv": str(outdir / "v46_4_selection_debug.csv"),
                "hit_outcomes_csv": str(outdir / "v46_4_hit_outcomes.csv"),
                "buy_card_plot": str(plots_dir / "v46_4_buy_card.png"),
                "hit_outcomes_plot": str(plots_dir / "v46_4_hit_outcomes.png"),
            }
        stake_profile_summaries[profile] = {
            "validation": result.get("validation") or {},
            "selected_card": result.get("execution_rows") or [],
            "hit_outcomes": result.get("hit_outcome_rows") or [],
            "outputs": profile_outputs,
        }

    summary_out = {
        "method": "v46_4_basev51",
        "input": str(input_dir),
        "team_a": team_a,
        "team_b": team_b,
        "favorite_side": favorite_side,
        "outlier_scoreline": outlier_scoreline,
        "stake_mode": args.stake_mode,
        "stake_profile": args.stake_profile,
        "primary_stake_profile": primary_stake_profile,
        "buffer": args.buffer,
        "entry_buffer_disabled": bool(args.disable_entry_buffer),
        "potential_band": args.potential_band,
        "min_upside_probability": args.min_upside_probability,
        "legacy_probability_floor_noop": args.probability_floor,
        "min_material_joint_kelly": args.min_quarter_kelly,
        "execution_bankroll": args.bankroll,
        "min_order_size": args.min_order_size,
        "stake_rounding": args.stake_rounding,
        "executable_min_joint_kelly": executable_min_joint_kelly,
        "buy_hold_rank_limit": args.buy_hold_rank_limit,
        "execution_target_stake": args.execution_target_stake,
        "value_surplus_weight": args.value_surplus_weight,
        "outlier_surplus_weight": args.outlier_surplus_weight,
        "cover_surplus_weight": args.cover_surplus_weight,
        "overpriced_cover_surplus_cap": args.overpriced_cover_surplus_cap,
        "buy_hold_max_bets": args.buy_hold_max_bets,
        "buy_hold_min_bets": args.buy_hold_min_bets,
        "probability_cover_count": args.probability_cover_count,
        "min_value_bets": args.min_value_bets,
        "min_cover_bets": args.min_cover_bets,
        "max_negative_edge_covers": args.max_negative_edge_covers,
        "require_outlier": require_outlier,
        "ignored_scorelines": sorted(ignored_scorelines),
        "exclude_any_other_score": args.exclude_any_other_score,
        "show_potential_value": args.show_potential_value,
        "max_potential_value": args.max_potential_value,
        "min_kelly_shrink": args.min_kelly_shrink,
        "max_kelly_shrink": args.max_kelly_shrink,
        "kelly_tranche_budget": args.kelly_tranche_budget,
        "kelly_tranche_cap_disabled": bool(args.disable_kelly_tranche_cap),
        "watch_threshold_margin": args.watch_threshold_margin,
        "joint_kelly": joint_kelly_summary,
        "buy_hold_validation": buy_hold_validation,
        "stake_profiles": stake_profile_summaries,
        "min_confidence": args.min_confidence,
        "min_story_fit": args.min_story_fit,
        "selected_card": selected_rows,
        "hit_outcomes": hit_outcome_rows,
        "buys": buy_rows,
        "potential_value": potential_value_rows,
        "cap_excluded": cap_excluded_rows,
        "near_fair_watch": near_fair_watch_rows,
        "watch": watch_only_rows,
        "no": no_rows,
        "outputs": {
            "buy_card_csv": str(outdir / "v46_4_buy_card.csv"),
            "selection_debug_csv": str(outdir / "v46_4_selection_debug.csv"),
            "hit_outcomes_csv": str(outdir / "v46_4_hit_outcomes.csv"),
            "summary_json": str(outdir / "v46_4_basev51_summary.json"),
            "buy_card_plot": str(plots_dir / "v46_4_buy_card.png"),
            "hit_outcomes_plot": str(plots_dir / "v46_4_hit_outcomes.png"),
            "grid_plot": str(plots_dir / "v46_4_score_grid.png"),
        },
    }

    write_csv(outdir / "v46_4_buy_card.csv", card_rows)
    write_csv(outdir / "v46_4_selection_debug.csv", selection_debug_rows)
    write_csv(outdir / "v46_4_hit_outcomes.csv", hit_outcome_rows)
    if args.stake_profile == "both":
        for profile, result in stake_profile_results.items():
            paths = (stake_profile_summaries.get(profile) or {}).get("outputs") or {}
            write_csv(Path(paths["buy_card_csv"]), list(result.get("execution_rows") or []))
            write_csv(Path(paths["selection_debug_csv"]), list(result.get("selection_debug_rows") or []))
            write_csv(Path(paths["hit_outcomes_csv"]), list(result.get("hit_outcome_rows") or []))
    (outdir / "v46_4_basev51_summary.json").write_text(json.dumps(summary_out, indent=2), encoding="utf-8")
    if not args.no_plots:
        write_card_plot(plots_dir / "v46_4_buy_card.png", card_rows, team_a, team_b)
        write_hit_outcomes_plot(
            plots_dir / "v46_4_hit_outcomes.png",
            hit_outcome_rows,
            team_a,
            team_b,
            target_stake=args.execution_target_stake,
        )
        write_grid_plot(
            plots_dir / "v46_4_score_grid.png",
            matrix,
            markets,
            kelly_index,
            card_rows,
            team_a,
            team_b,
            args.grid_max_goals,
            args.buffer,
            args.potential_band,
            favorite_side,
            knockout,
            args.min_upside_probability,
            executable_min_joint_kelly,
            args.min_confidence,
            args.min_story_fit,
            args.watch_threshold_margin,
            show_potential_value=args.show_potential_value,
        )
        if args.stake_profile == "both":
            for profile, result in stake_profile_results.items():
                paths = (stake_profile_summaries.get(profile) or {}).get("outputs") or {}
                profile_rows = list(result.get("execution_rows") or [])
                profile_hits = list(result.get("hit_outcome_rows") or [])
                write_card_plot(Path(paths["buy_card_plot"]), profile_rows, team_a, team_b)
                write_hit_outcomes_plot(
                    Path(paths["hit_outcomes_plot"]),
                    profile_hits,
                    team_a,
                    team_b,
                    target_stake=args.execution_target_stake,
                )
    print(json.dumps(summary_out, indent=2))


if __name__ == "__main__":
    main()

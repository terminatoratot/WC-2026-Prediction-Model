"""V51: v11 (+v49 bivariate-NegBin scoreline correction) plus TWO additive
outlier tabs on top of the untouched Top-3 -- one from v29's tail-risk
selector, one from v39's coverage-outlier selector.

This is a standalone, read-only-import combination -- it does not modify
v11_wcq_results_model.py, v29_tail_risk_scoreline_model.py, or
v39_coverage_outlier_model.py in any way, and can be deleted without
affecting any of them (same composition pattern as v50).

Why "additive, not replacing": v29's default behavior
(select_top_scorelines_with_tail_risk) REPLACES the 3rd Top-3 scoreline
with a blowout candidate when its gates fire. That's fine for its original
purpose (betting/staking risk coverage) but actively hurts top-3 accuracy,
which is the metric we care about most here (see conversation: v29 never
changes the WDL probabilities or score matrix, only trades away a
top-3 slot for tail coverage). So this file calls v29's selector purely to
harvest its candidate scoreline (via the same diagnostics dict
select_top_scorelines_with_tail_risk already returns), and appends it as a
4th "outlier" slot instead of overwriting slot 3. v39's existing
coverage-outlier logic (already used by v11's own top3_plus_outlier_accuracy
metric) is kept unchanged and tracked as a second, independent outlier tab,
so the backtest can show whether v29's tail-risk pick, v39's total-goals
envelope pick, or both together add the most incremental hit-rate on top of
a clean Top-3.

Uses the plain v39_coverage_outlier_model (not v39_withbetterdata) -- same
choice v11's own chronological_backtest already makes, since
v39_withbetterdata needs additional "better priors" inputs that have not
been sourced/validated for this backtest.

Metrics tracked per match: top3_hit (unchanged v11+v49 top-3), plus
top3_plus_v39_hit, top3_plus_v29_hit, and top3_plus_both_hit (union of
top-3, the v39 outlier, and the v29 outlier).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import market_edge

# The older model layers are bundled inside these three physical modules.
# Import through the bundle attributes so both Python and static analyzers
# (including Pylance in Codespaces) can resolve the dependency chain.
v11 = market_edge.feature_layers.core_engine.v11_wcq_results_model
v29 = market_edge.feature_layers.v29_tail_risk_scoreline_model
select_coverage_outlier = market_edge.v39_coverage_outlier_model.select_coverage_outlier

ScoreMatrix = Dict[Tuple[int, int], float]


def _v29_outlier(
    top_scorelines: List[Dict[str, Any]],
    score_probs: ScoreMatrix,
    result_probabilities: Dict[str, float],
    lambda_a: float,
    lambda_b: float,
) -> Tuple[Optional[Tuple[int, int]], Dict[str, Any]]:
    """Harvest v29's tail-risk candidate without letting it replace slot 3.

    Calls the same pure selector v29 itself uses, with the untouched Top-3
    as `current_top_scorelines`. When v29's gates fire, the candidate it
    *would* have swapped into slot 3 is read back out of the diagnostics
    dict (`candidate_scoreline`) instead of applying the swap.
    """
    top3 = top_scorelines[:3]
    _, diagnostics = v29.select_top_scorelines_with_tail_risk(
        score_probs,
        result_probabilities,
        lambda_a,
        lambda_b,
        current_top_scorelines=top3,
        top_n=3,
    )
    if not diagnostics.get("tail_risk_applied"):
        return None, diagnostics
    label = diagnostics["candidate_scoreline"]
    goals_a_str, goals_b_str = label.split("-")
    return (int(goals_a_str), int(goals_b_str)), diagnostics


def _combined_outlier_hits(
    top_scorelines: List[Dict[str, Any]],
    score_probs: ScoreMatrix,
    result_probabilities: Dict[str, float],
    lambda_a: float,
    lambda_b: float,
    actual_key: Tuple[int, int],
) -> Dict[str, Any]:
    top3 = top_scorelines[:3]
    top3_keys = {(int(s["team_a_goals"]), int(s["team_b_goals"])) for s in top3}
    top3_hit = int(actual_key in top3_keys)

    v39_outlier, _ = select_coverage_outlier(
        score_probs,
        top3,
        lambda_a,
        lambda_b,
        use_observed_game_state_priors=False,
    )
    v39_key = (
        (int(v39_outlier["team_a_goals"]), int(v39_outlier["team_b_goals"]))
        if v39_outlier is not None
        else None
    )

    v29_key, v29_diag = _v29_outlier(top_scorelines, score_probs, result_probabilities, lambda_a, lambda_b)

    plus_v39_keys = set(top3_keys)
    if v39_key is not None:
        plus_v39_keys.add(v39_key)
    plus_v29_keys = set(top3_keys)
    if v29_key is not None:
        plus_v29_keys.add(v29_key)
    plus_both_keys = set(top3_keys)
    if v39_key is not None:
        plus_both_keys.add(v39_key)
    if v29_key is not None:
        plus_both_keys.add(v29_key)

    return {
        "top3_hit": top3_hit,
        "top3_plus_v39_hit": int(actual_key in plus_v39_keys),
        "top3_plus_v29_hit": int(actual_key in plus_v29_keys),
        "top3_plus_both_hit": int(actual_key in plus_both_keys),
        "v39_outlier_triggered": int(v39_key is not None),
        "v29_outlier_triggered": int(v29_diag.get("tail_risk_applied", False)),
    }


def chronological_backtest_combined(
    zip_path: str,
    train_csv: Optional[str] = None,
    test_csv: Optional[str] = None,
    model_type: str = "ensemble",
    test_years: Optional[List[int]] = None,
    min_train_year: int = 1930,
    max_goals: int = 10,
    box_csv: Optional[str] = None,
    results_csv: Optional[str] = None,
    former_names_csv: Optional[str] = None,
    qualifier_blend_start_year: int = 2014,
    qualifier_full_weight_year: int = 2022,
    qualifier_minimum_influence: float = 0.0,
    recency_half_life_years: float = 16.0,
    recency_min_weight: float = 0.10,
    score_matrix_r: float = 25.0,
    use_expanded_training_pool: bool = True,
    wc_prestige_weight: float = 600.0,
    use_volume_normalized_weighting: bool = False,
    prestige_tier_target_shares: Dict[str, float] = v11.DEFAULT_PRESTIGE_TIER_TARGET_SHARES,
    fbref_world_cup_csv: Optional[str] = None,
    fbref_international_csv: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Same expanding-window methodology as v11.chronological_backtest, with
    the v49 bivariate-NegBin score matrix always on (that's the known-good
    baseline this file builds on top of), plus the v29/v39 additive outlier
    tabs layered on at evaluation time. Mirrors v11's loop structure
    (duplicated, not imported, since v11's loop doesn't expose per-match
    score matrices/top-scorelines in its return value -- only aggregate hit
    booleans) so v11 itself needs no changes.
    """
    loader = v11.WorldCupSAILoader(zip_path, Path(str(zip_path) + "_extracted"))
    all_matches = loader.load_matches()
    fbref_wc_matches = v11.load_fbref_world_cup_matches(fbref_world_cup_csv)
    if not fbref_wc_matches.empty:
        all_matches = pd.concat([all_matches, fbref_wc_matches], ignore_index=True, sort=False)
    all_matches["prestige_weight"] = wc_prestige_weight
    all_matches["prestige_tier"] = "world_cup"
    current = v11.load_current_team_features(train_csv, test_csv)
    box = v11.load_kaggle_box_data(box_csv)
    qualification_results = v11.load_world_cup_qualification_results(results_csv, former_names_csv)
    qualifier_source = qualification_results if not qualification_results.empty else box

    if use_expanded_training_pool:
        expanded_matches = v11.load_expanded_competition_matches(results_csv, former_names_csv)
        fbref_intl_matches = v11.load_fbref_international_matches(fbref_international_csv)
        train_pool = pd.concat([all_matches, expanded_matches, fbref_intl_matches], ignore_index=True, sort=False)
    else:
        train_pool = all_matches

    all_years = sorted(int(y) for y in all_matches["year"].dropna().unique())
    if test_years is None:
        test_years = [y for y in all_years if y >= 2010]
    else:
        test_years = [int(y) for y in test_years]

    pred_rows = []
    summary_rows = []

    for year in test_years:
        train_matches = train_pool[(train_pool["year"] < year) & (train_pool["year"] >= min_train_year)].copy()
        test_matches = all_matches[all_matches["year"] == year].copy()

        if len(train_matches) < 80 or len(test_matches) == 0:
            continue

        if use_volume_normalized_weighting:
            train_matches["prestige_weight"] = v11.assign_volume_normalized_weights(
                train_matches, prestige_tier_target_shares
            )

        historical_box = box[box["box_year"] < year].copy() if not box.empty else box
        if not qualification_results.empty:
            historical_qualifiers = qualifier_source[qualifier_source["box_year"] < year].copy()
        else:
            historical_qualifiers = historical_box
        historical_current = pd.DataFrame(columns=["team"])
        train_frame, features, events = v11.build_rolling_features(
            train_matches,
            historical_current,
            qualifier_box=historical_qualifiers,
            qualifier_fallback_box=historical_box,
            qualifier_blend_start_year=qualifier_blend_start_year,
            qualifier_full_weight_year=qualifier_full_weight_year,
            qualifier_minimum_influence=qualifier_minimum_influence,
        )
        model = (
            v11.StrongWorldCupModel(
                model_type=model_type,
                recency_half_life_years=recency_half_life_years,
                recency_min_weight=recency_min_weight,
                score_matrix_variant="bivariate_negbin_dc",
                score_matrix_r=score_matrix_r,
            )
            .fit(train_frame, features, events, historical_current)
            .set_box_data(historical_box)
            .set_qualifier_data(
                historical_qualifiers,
                fallback_box=historical_box,
                prediction_year=year,
                blend_start_year=qualifier_blend_start_year,
                full_weight_year=qualifier_full_weight_year,
                minimum_influence=qualifier_minimum_influence,
            )
        )

        for _, r in test_matches.iterrows():
            pred = model.predict(
                r.team_a,
                r.team_b,
                host_a=bool(r.host_a),
                host_b=bool(r.host_b),
                knockout=bool(r.is_knockout),
                max_goals=max_goals,
            )

            actual = v11.actual_result_label(r.goals_a, r.goals_b)
            actual_prob = pred["result_probabilities"][actual]
            predicted_result = max(pred["result_probabilities"], key=pred["result_probabilities"].get)

            score_probs = {
                (int(s["team_a_goals"]), int(s["team_b_goals"])): float(s["probability"])
                for s in pred["scoreline_probabilities"]
            }
            exact_prob = score_probs.get((int(r.goals_a), int(r.goals_b)), 0.0)
            actual_key = (int(r.goals_a), int(r.goals_b))

            hits = _combined_outlier_hits(
                pred["top_scorelines"],
                score_probs,
                pred["result_probabilities"],
                pred["lambda_a"],
                pred["lambda_b"],
                actual_key,
            )

            pred_rows.append(
                {
                    "test_year": year,
                    "match_id": r.match_id,
                    "date": r.date,
                    "stage": r.stage_name,
                    "team_a": r.team_a,
                    "team_b": r.team_b,
                    "actual_score": f"{int(r.goals_a)}-{int(r.goals_b)}",
                    "lambda_a": pred["lambda_a"],
                    "lambda_b": pred["lambda_b"],
                    "predicted_result": predicted_result,
                    "actual_result": actual,
                    "correct_result": int(predicted_result == actual),
                    "team_a_win_prob": pred["result_probabilities"]["team_a_win"],
                    "draw_prob": pred["result_probabilities"]["draw"],
                    "team_b_win_prob": pred["result_probabilities"]["team_b_win"],
                    "actual_result_probability": actual_prob,
                    "result_log_loss": v11.safe_log_loss(actual_prob),
                    "result_brier": v11.brier_score_3way(actual, pred["result_probabilities"]),
                    "goal_mae": (abs(pred["lambda_a"] - r.goals_a) + abs(pred["lambda_b"] - r.goals_b)) / 2.0,
                    "goal_diff_abs_error": abs((pred["lambda_a"] - pred["lambda_b"]) - (r.goals_a - r.goals_b)),
                    "exact_score_probability": exact_prob,
                    "exact_score_log_loss": v11.safe_log_loss(exact_prob),
                    **hits,
                }
            )

        year_df = pd.DataFrame([x for x in pred_rows if x["test_year"] == year])
        summary_rows.append(_summary_row(year, train_matches, test_matches, year_df, features, events, model_type,
                                          recency_half_life_years, recency_min_weight, qualifier_blend_start_year,
                                          qualifier_full_weight_year, qualifier_minimum_influence, score_matrix_r))

    pred_df = pd.DataFrame(pred_rows)
    summary_df = pd.DataFrame(summary_rows)

    if len(pred_df):
        overall = _summary_row("overall", pd.DataFrame(), pred_df, pred_df, [], [], model_type,
                                recency_half_life_years, recency_min_weight, qualifier_blend_start_year,
                                qualifier_full_weight_year, qualifier_minimum_influence, score_matrix_r,
                                is_overall=True)
        summary_df = pd.concat([summary_df, pd.DataFrame([overall])], ignore_index=True)

    return pred_df, summary_df


def _summary_row(
    year, train_matches, test_matches, year_df, features, events, model_type,
    recency_half_life_years, recency_min_weight, qualifier_blend_start_year,
    qualifier_full_weight_year, qualifier_minimum_influence, score_matrix_r,
    is_overall: bool = False,
) -> Dict[str, Any]:
    return {
        "test_year": year,
        "train_matches": np.nan if is_overall else int(len(train_matches)),
        "test_matches": int(len(test_matches)),
        "result_accuracy": float(year_df["correct_result"].mean()),
        "mean_result_log_loss": float(year_df["result_log_loss"].mean()),
        "mean_result_brier": float(year_df["result_brier"].mean()),
        "mean_goal_mae": float(year_df["goal_mae"].mean()),
        "mean_goal_diff_abs_error": float(year_df["goal_diff_abs_error"].mean()),
        "mean_exact_score_log_loss": float(year_df["exact_score_log_loss"].mean()),
        "mean_actual_result_probability": float(year_df["actual_result_probability"].mean()),
        "mean_exact_score_probability": float(year_df["exact_score_probability"].mean()),
        "top3_accuracy": float(year_df["top3_hit"].mean()),
        "top3_plus_v39_accuracy": float(year_df["top3_plus_v39_hit"].mean()),
        "top3_plus_v29_accuracy": float(year_df["top3_plus_v29_hit"].mean()),
        "top3_plus_both_accuracy": float(year_df["top3_plus_both_hit"].mean()),
        "v39_outlier_trigger_rate": float(year_df["v39_outlier_triggered"].mean()),
        "v29_outlier_trigger_rate": float(year_df["v29_outlier_triggered"].mean()),
        "features_used": np.nan if is_overall else len(features),
        "event_targets": "" if is_overall else ",".join(events),
        "model_type": model_type,
        "recency_half_life_years": recency_half_life_years,
        "recency_min_weight": recency_min_weight,
        "qualifier_influence": (
            np.nan
            if is_overall
            else v11.qualifier_influence_for_year(
                year,
                start_year=qualifier_blend_start_year,
                full_weight_year=qualifier_full_weight_year,
                minimum_influence=qualifier_minimum_influence,
            )
        ),
        "score_matrix_variant": "bivariate_negbin_dc",
        "score_matrix_r": score_matrix_r,
    }


# --- BEGIN live single-match prediction (train-once, not per-fold) ------------------------
# chronological_backtest_combined() above proves out the methodology across
# historical folds. For an actual live match prediction (used downstream by
# the v46_4 betting/staking layer), we need the equivalent of one more
# "fold" -- train on everything available today (expanded multi-competition
# pool, FBref 2022-2026 extension, wc_prestige_weight -- the exact settings
# that produced the validated backtest numbers), then predict a single
# team_a vs team_b matchup. v11's own build_from_zip() does NOT carry any
# of that (no v49 correction, no expanded pool, no FBref extension, no
# prestige weighting) -- it's the older/simpler path v29/v42 still use, so
# it's deliberately not reused here.
class V51CombinedScorelineModel:
    """Wraps a v11 StrongWorldCupModel (already fit with the v49 bivariate
    NegBin score matrix) and adds the two additive outlier tabs at predict
    time. Top-3 is left completely untouched -- v39/v29 outliers are
    reported as extra fields plus a `v51_combined_top_scorelines` list
    (top-3 + up to 2 outliers, deduped), for callers that want the widest
    reasonable scoreline coverage without disturbing top-3 accuracy.
    """

    def __init__(self, base_model: "v11.StrongWorldCupModel"):
        self.base_model = base_model
        self.training_data_summary = getattr(base_model, "training_data_summary", {})

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_model, name)

    def predict(self, *args, **kwargs) -> Dict[str, Any]:
        prediction = self.base_model.predict(*args, **kwargs)
        score_probs = {
            (int(s["team_a_goals"]), int(s["team_b_goals"])): float(s["probability"])
            for s in prediction["scoreline_probabilities"]
        }
        top3 = prediction["top_scorelines"][:3]

        v39_outlier, v39_diag = select_coverage_outlier(
            score_probs, top3, prediction["lambda_a"], prediction["lambda_b"], use_observed_game_state_priors=False
        )
        v29_key, v29_diag = _v29_outlier(
            prediction["top_scorelines"],
            score_probs,
            prediction["result_probabilities"],
            prediction["lambda_a"],
            prediction["lambda_b"],
        )
        v29_outlier = v29.score_item(v29_key, score_probs[v29_key]) if v29_key is not None else None

        combined = list(top3)
        seen = {(int(s["team_a_goals"]), int(s["team_b_goals"])) for s in top3}
        for extra in (v39_outlier, v29_outlier):
            if extra is None:
                continue
            key = (int(extra["team_a_goals"]), int(extra["team_b_goals"]))
            if key not in seen:
                combined.append(extra)
                seen.add(key)

        prediction["v51_combined_top_scorelines"] = combined
        prediction["v51_adjustments"] = {
            "base": "v11+v49_bivariate_negbin_dc",
            "top3_unchanged": True,
            "v39_outlier": v39_outlier,
            "v29_outlier": v29_outlier,
            "v39_diagnostics": v39_diag,
            "v29_diagnostics": v29_diag,
        }
        prediction["calibration_notes"] = {
            **prediction.get("calibration_notes", {}),
            "v51": prediction["v51_adjustments"],
        }
        return prediction


def build_from_zip(
    zip_path: str,
    train_csv: Optional[str] = None,
    test_csv: Optional[str] = None,
    model_type: str = "ensemble",
    box_csv: Optional[str] = None,
    results_csv: Optional[str] = None,
    former_names_csv: Optional[str] = None,
    prediction_year: int = 2026,
    qualifier_blend_start_year: int = 2014,
    qualifier_full_weight_year: int = 2022,
    qualifier_minimum_influence: float = 0.0,
    recency_half_life_years: float = 16.0,
    recency_min_weight: float = 0.10,
    score_matrix_r: float = 25.0,
    use_expanded_training_pool: bool = True,
    wc_prestige_weight: float = 600.0,
    use_volume_normalized_weighting: bool = False,
    prestige_tier_target_shares: Dict[str, float] = v11.DEFAULT_PRESTIGE_TIER_TARGET_SHARES,
    fbref_world_cup_csv: Optional[str] = None,
    fbref_international_csv: Optional[str] = None,
) -> Tuple[V51CombinedScorelineModel, Any]:
    """Train once on everything available (through fbref_world_cup_csv /
    fbref_international_csv if given), for a live prediction rather than a
    backtest fold. Mirrors chronological_backtest_combined's training-pool
    assembly with no test-year cutoff -- all matches count as training data.
    """
    loader = v11.WorldCupSAILoader(zip_path, Path(str(zip_path) + "_extracted"))
    all_matches = loader.load_matches()
    fbref_wc_matches = v11.load_fbref_world_cup_matches(fbref_world_cup_csv)
    if not fbref_wc_matches.empty:
        all_matches = pd.concat([all_matches, fbref_wc_matches], ignore_index=True, sort=False)
    all_matches["prestige_weight"] = wc_prestige_weight
    all_matches["prestige_tier"] = "world_cup"
    current = v11.load_current_team_features(train_csv, test_csv)
    box = v11.load_kaggle_box_data(box_csv)
    qualification_results = v11.load_world_cup_qualification_results(results_csv, former_names_csv)
    qualifier_source = qualification_results if not qualification_results.empty else box

    if use_expanded_training_pool:
        expanded_matches = v11.load_expanded_competition_matches(results_csv, former_names_csv)
        fbref_intl_matches = v11.load_fbref_international_matches(fbref_international_csv)
        train_matches = pd.concat([all_matches, expanded_matches, fbref_intl_matches], ignore_index=True, sort=False)
    else:
        train_matches = all_matches

    # build_rolling_features() walks rows in the order given, updating the
    # shared per-team elo/rolling-form dicts as it goes -- so this MUST be
    # chronological. Without this sort, the concat above leaves matches as
    # three separate date-ordered blocks (World Cup, then all of results.csv's
    # non-WC competitions, then fbref international) rather than one merged
    # timeline, so every World Cup match's elo/form features get computed as
    # of "end of the World Cup block", never seeing the (chronologically
    # earlier-and-later-interleaved) Euro/AFCON/qualifier/friendly results at
    # all -- e.g. a team's 2024-2026 qualifying run would never reach its
    # elo_diff for a 2026 World Cup match.
    train_matches = train_matches.sort_values("date").reset_index(drop=True)

    if use_volume_normalized_weighting:
        train_matches["prestige_weight"] = v11.assign_volume_normalized_weights(
            train_matches, prestige_tier_target_shares
        )

    frame, features, events = v11.build_rolling_features(
        train_matches,
        current,
        qualifier_box=qualifier_source,
        qualifier_fallback_box=box,
        qualifier_blend_start_year=qualifier_blend_start_year,
        qualifier_full_weight_year=qualifier_full_weight_year,
        qualifier_minimum_influence=qualifier_minimum_influence,
    )
    base_model = (
        v11.StrongWorldCupModel(
            model_type=model_type,
            recency_half_life_years=recency_half_life_years,
            recency_min_weight=recency_min_weight,
            score_matrix_variant="bivariate_negbin_dc",
            score_matrix_r=score_matrix_r,
        )
        .fit(frame, features, events, current)
        .set_box_data(box)
        .set_qualifier_data(
            qualifier_source,
            fallback_box=box,
            prediction_year=prediction_year,
            blend_start_year=qualifier_blend_start_year,
            full_weight_year=qualifier_full_weight_year,
            minimum_influence=qualifier_minimum_influence,
        )
    )
    model = V51CombinedScorelineModel(base_model)
    model.training_data_summary = {
        "v51_base": "v11+v49_bivariate_negbin_dc",
        "v51_outlier_sources": ["v39_coverage_outlier_model", "v29_tail_risk_scoreline_model (additive, non-replacing)"],
        "use_expanded_training_pool": use_expanded_training_pool,
        "wc_prestige_weight": wc_prestige_weight,
        "use_volume_normalized_weighting": use_volume_normalized_weighting,
        "score_matrix_r": score_matrix_r,
        "n_training_rows": int(len(train_matches)),
    }
    return model, DataBundle(matches=all_matches, training_frame=frame, box_frame=box, team_current=current)


class DataBundle:
    def __init__(self, matches, training_frame, box_frame, team_current):
        self.matches = matches
        self.training_frame = training_frame
        self.box_frame = box_frame
        self.team_current = team_current
# --- END live single-match prediction ------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="V51: v11+v49 baseline with additive v29 (tail-risk) and v39 (coverage-outlier) outlier tabs."
    )
    ap.add_argument("--worldcupsai-zip", default="data/worldcupsai.zip")
    ap.add_argument("--team-train", default="data/current_team_features_2026.csv")
    ap.add_argument("--team-test", default=None)
    ap.add_argument("--box-data", default="data/FIFAallMatchBoxData.csv")
    ap.add_argument("--results-data", default="data/results.csv")
    ap.add_argument("--former-names", default="data/former_names.csv")
    ap.add_argument("--qualifier-blend-start-year", type=int, default=2014)
    ap.add_argument("--qualifier-full-weight-year", type=int, default=2022)
    ap.add_argument("--qualifier-minimum-influence", type=float, default=0.0)
    ap.add_argument("--recency-half-life-years", type=float, default=16.0)
    ap.add_argument("--recency-min-weight", type=float, default=0.10)
    ap.add_argument("--test-years", nargs="*", type=int, default=None)
    ap.add_argument("--model", default="ensemble")
    ap.add_argument("--score-matrix-r", type=float, default=25.0)
    ap.add_argument("--wc-prestige-weight", type=float, default=600.0)
    ap.add_argument("--use-volume-normalized-weighting", action="store_true")
    ap.add_argument("--fbref-world-cup-csv", default=None)
    ap.add_argument("--fbref-international-csv", default=None)
    ap.add_argument("--outdir", default="outputs/v51_combined_backtest")
    args = ap.parse_args()

    out = v11.unique_output_dir(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    pred_df, summary_df = chronological_backtest_combined(
        zip_path=args.worldcupsai_zip,
        train_csv=args.team_train,
        test_csv=args.team_test,
        model_type=args.model,
        test_years=args.test_years,
        box_csv=args.box_data,
        results_csv=args.results_data,
        former_names_csv=args.former_names,
        qualifier_blend_start_year=args.qualifier_blend_start_year,
        qualifier_full_weight_year=args.qualifier_full_weight_year,
        qualifier_minimum_influence=args.qualifier_minimum_influence,
        recency_half_life_years=args.recency_half_life_years,
        recency_min_weight=args.recency_min_weight,
        score_matrix_r=args.score_matrix_r,
        wc_prestige_weight=args.wc_prestige_weight,
        use_volume_normalized_weighting=args.use_volume_normalized_weighting,
        fbref_world_cup_csv=args.fbref_world_cup_csv,
        fbref_international_csv=args.fbref_international_csv,
    )
    pred_df.to_csv(out / "backtest_predictions.csv", index=False)
    summary_df.to_csv(out / "backtest_summary.csv", index=False)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

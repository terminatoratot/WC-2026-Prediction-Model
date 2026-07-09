#!/usr/bin/env python3
"""V33: V32 top three plus a separately labeled outlier scoreline.

The first three scorelines are left exactly as the base model produced them.
V33 appends one "outlier ticket" for high-tail coverage when the current
tournament and match-specific player/xG signals justify it.

Run:
    MPLCONFIGDIR=.matplotlib_cache .venv/bin/python v33_outlier_slot_model.py --team-a "Switzerland" --team-b "Bosnia and Herzegovina" --outdir outputs/outputs_v33_switzerland_bosnia
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

import v11_wcq_results_model as v11
import v15_catboost_model as v15
import v26_top3_coverage_model as v26
import v28_current_worldcup_form_model as v28
import v29_tail_risk_scoreline_model as v29
import v30_player_role_form_model as v30
import v31_gated_role_selector_model as v31
import v32_third_slot_coverage_model as v32


ScoreMatrix = Dict[Tuple[int, int], float]

DEFAULT_OUTLIER_RELATIVE_FLOOR = 0.045
DEFAULT_OUTLIER_ABSOLUTE_FLOOR = 0.002
DEFAULT_OUTLIER_PROBABILITY_WEIGHT = 0.54
DEFAULT_OUTLIER_CHAOS_WEIGHT = 0.30
DEFAULT_OUTLIER_SIGNAL_WEIGHT = 0.34
DEFAULT_OUTLIER_MAX_GOALS = 7


@dataclass
class TournamentChaosProfile:
    matches: int = 0
    average_total_goals: float = 0.0
    high_total_rate: float = 0.0
    extreme_total_rate: float = 0.0
    blowout_rate: float = 0.0
    btts_high_total_rate: float = 0.0
    penalty_mention_rate: float = 0.0
    late_goal_match_rate: float = 0.0
    chaos_index: float = 0.0

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "matches": int(self.matches),
            "average_total_goals": float(self.average_total_goals),
            "high_total_rate": float(self.high_total_rate),
            "extreme_total_rate": float(self.extreme_total_rate),
            "blowout_rate": float(self.blowout_rate),
            "btts_high_total_rate": float(self.btts_high_total_rate),
            "penalty_mention_rate": float(self.penalty_mention_rate),
            "late_goal_match_rate": float(self.late_goal_match_rate),
            "chaos_index": float(self.chaos_index),
        }


def _score_key(item: Dict[str, Any]) -> Tuple[int, int]:
    return int(item["team_a_goals"]), int(item["team_b_goals"])


def _score_item(key: Tuple[int, int], probability: float) -> Dict[str, Any]:
    return {
        "team_a_goals": int(key[0]),
        "team_b_goals": int(key[1]),
        "probability": float(probability),
    }


def _score_label(key: Tuple[int, int]) -> str:
    return f"{int(key[0])}-{int(key[1])}"


def _score_outcome(key: Tuple[int, int]) -> str:
    if key[0] > key[1]:
        return "team_a_win"
    if key[1] > key[0]:
        return "team_b_win"
    return "draw"


def _favorite_score(
    side: str | None,
    favorite_goals: int,
    underdog_goals: int,
) -> Tuple[int, int]:
    if side == "team_b":
        return underdog_goals, favorite_goals
    return favorite_goals, underdog_goals


def _goal_events_after_75(raw_text: object) -> int:
    text = str(raw_text or "")
    count = 0
    for match in re.finditer(r"\((\d+)\s*-\s*(\d+)\)", text):
        window = text[max(0, match.start() - 80) : match.end() + 80]
        minute_matches = re.findall(r"(\d{1,3})(?:\s*[+]\s*\d{1,2})?\s*[’']", window)
        if not minute_matches:
            continue
        minute = max(int(value) for value in minute_matches)
        if minute >= 75:
            count += 1
    return count


def build_tournament_chaos_profile(
    match_facts_csv: str | Path | None,
) -> TournamentChaosProfile:
    if not match_facts_csv or not Path(match_facts_csv).exists():
        return TournamentChaosProfile()
    facts = pd.read_csv(match_facts_csv)
    if facts.empty or not {"home_score", "away_score"}.issubset(facts.columns):
        return TournamentChaosProfile()
    if "status" in facts:
        facts = facts[
            facts["status"].astype(str).str.contains(
                "full|ft|aet|pen",
                case=False,
                na=False,
            )
        ].copy()
    facts["home_score"] = pd.to_numeric(facts["home_score"], errors="coerce")
    facts["away_score"] = pd.to_numeric(facts["away_score"], errors="coerce")
    facts = facts.dropna(subset=["home_score", "away_score"]).copy()
    if facts.empty:
        return TournamentChaosProfile()

    total_goals = facts["home_score"] + facts["away_score"]
    margins = (facts["home_score"] - facts["away_score"]).abs()
    btts = (facts["home_score"] > 0) & (facts["away_score"] > 0)
    penalty_mentions = (
        facts.get("raw_text", pd.Series([""] * len(facts), index=facts.index))
        .astype(str)
        .str.contains("penalty|\\(pen\\)", case=False, regex=True, na=False)
    )
    late_goal_counts = (
        facts.get("raw_text", pd.Series([""] * len(facts), index=facts.index))
        .apply(_goal_events_after_75)
        .astype(int)
    )

    high_total_rate = float((total_goals >= 4).mean())
    extreme_total_rate = float((total_goals >= 5).mean())
    blowout_rate = float(((total_goals >= 4) & (margins >= 2)).mean())
    btts_high_total_rate = float(((total_goals >= 4) & btts).mean())
    penalty_mention_rate = float(penalty_mentions.mean())
    late_goal_match_rate = float((late_goal_counts > 0).mean())
    average_total_goals = float(total_goals.mean())
    historical_high_total_baseline = 0.24
    historical_extreme_baseline = 0.10
    historical_goals_baseline = 2.65
    chaos_index = (
        0.32 * max(0.0, high_total_rate - historical_high_total_baseline) / 0.26
        + 0.22 * max(0.0, extreme_total_rate - historical_extreme_baseline) / 0.18
        + 0.16 * max(0.0, average_total_goals - historical_goals_baseline) / 0.85
        + 0.14 * blowout_rate
        + 0.10 * btts_high_total_rate
        + 0.04 * penalty_mention_rate
        + 0.02 * late_goal_match_rate
    )
    return TournamentChaosProfile(
        matches=int(len(facts)),
        average_total_goals=average_total_goals,
        high_total_rate=high_total_rate,
        extreme_total_rate=extreme_total_rate,
        blowout_rate=blowout_rate,
        btts_high_total_rate=btts_high_total_rate,
        penalty_mention_rate=penalty_mention_rate,
        late_goal_match_rate=late_goal_match_rate,
        chaos_index=float(np.clip(chaos_index, 0.0, 1.0)),
    )


def _candidate_pool(
    score_matrix: ScoreMatrix,
    top_three: list[Tuple[int, int]],
    result_probabilities: Dict[str, float],
    max_goals: int,
) -> list[Tuple[int, int]]:
    side = v26.favorite_side(result_probabilities)
    pool: list[Tuple[int, int]] = []

    if side is not None:
        structured = [
            _favorite_score(side, 3, 0),
            _favorite_score(side, 3, 1),
            _favorite_score(side, 3, 2),
            _favorite_score(side, 4, 0),
            _favorite_score(side, 4, 1),
            _favorite_score(side, 4, 2),
            _favorite_score(side, 5, 1),
            _favorite_score(side, 5, 2),
        ]
        for key in structured:
            if key not in pool:
                pool.append(key)

    open_game = [(2, 2), (3, 2), (2, 3), (3, 3), (4, 2), (2, 4)]
    for key in open_game:
        if key not in pool:
            pool.append(key)

    ranked_tail = [
        key
        for key, _ in sorted(score_matrix.items(), key=lambda item: item[1], reverse=True)
        if sum(key) >= 4 or max(key) >= 4
    ][:16]
    for key in ranked_tail:
        if key not in pool:
            pool.append(key)

    return [
        key
        for key in pool
        if key in score_matrix
        and key not in set(top_three)
        and 0 <= key[0] <= max_goals
        and 0 <= key[1] <= max_goals
        and (sum(key) >= 4 or max(key) >= 4)
    ]


def _candidate_kind(key: Tuple[int, int], side: str | None) -> str:
    if key[0] == key[1]:
        return "high_draw_outlier"
    if side is not None:
        fav_goals, dog_goals, margin = v32._favorite_goal_view(key, side)
        if fav_goals >= 4 and margin >= 2:
            return "favorite_extreme_blowout"
        if fav_goals >= 3 and margin >= 2:
            return "favorite_blowout"
        if fav_goals >= 3 and dog_goals >= 2:
            return "open_favorite_win"
    if sum(key) >= 5:
        return "open_high_total"
    return "moderate_tail"


def _outlier_utility(
    key: Tuple[int, int],
    score_matrix: ScoreMatrix,
    top_three: list[Tuple[int, int]],
    result_probabilities: Dict[str, float],
    lambda_a: float,
    lambda_b: float,
    role_metrics: Dict[str, float],
    xg_metrics: Dict[str, float],
    chaos_profile: TournamentChaosProfile,
    probability_weight: float,
    chaos_weight: float,
    signal_weight: float,
) -> Dict[str, float]:
    side = v26.favorite_side(result_probabilities)
    top_probability = max(float(score_matrix.get(top_three[0], 0.0)), 1e-12)
    probability_ratio = float(score_matrix.get(key, 0.0)) / top_probability
    probability_component = float(probability_weight) * math.sqrt(
        max(probability_ratio, 0.0)
    )
    total_goals = sum(key)
    fav_goals, dog_goals, margin = v32._favorite_goal_view(key, side)
    predicted_outcome = max(result_probabilities, key=result_probabilities.get)

    total_lambda = float(lambda_a) + float(lambda_b)
    lambda_component = 0.10 * max(0.0, min(1.0, (total_lambda - 2.25) / 1.45))
    outcome_component = 0.08 if _score_outcome(key) == predicted_outcome else 0.0
    shape_component = 0.0
    if total_goals >= 4:
        shape_component += 0.10 + 0.045 * min(total_goals - 4, 2)
    if side is not None and fav_goals >= 3 and margin >= 2:
        shape_component += 0.10 + 0.035 * min(fav_goals - 3, 2)
    if dog_goals >= 1 and total_goals >= 4:
        shape_component += 0.045

    role_coverage = role_metrics.get("role_coverage", 0.0)
    xg_coverage = xg_metrics.get("xg_coverage", 0.0)
    signal_coverage = math.sqrt(max(role_coverage, 0.0) * max(xg_coverage, 0.0))
    role_tail = role_metrics.get("favorite_by_three_signal", 0.0)
    role_clean = role_metrics.get("clean_sheet_signal", 0.0)
    xg_tail = xg_metrics.get("favorite_by_three_xg_signal", 0.0)
    xg_clean = xg_metrics.get("clean_sheet_xg_signal", 0.0)
    underdog_score = xg_metrics.get("underdog_score_pressure", 0.0)
    favorite_pressure = xg_metrics.get("favorite_pressure", 0.0)
    tail_signal = signal_coverage * (
        0.50 * v32._clip_signal(role_tail) + 0.50 * v32._clip_signal(xg_tail)
    )
    both_score_signal = signal_coverage * v32._clip_signal(
        -0.42 * role_clean - 0.50 * xg_clean + 0.26 * underdog_score
    )
    attack_signal = signal_coverage * v32._clip_signal(
        0.65 * favorite_pressure + 0.35 * xg_tail
    )
    signal_component = 0.0
    if side is not None and fav_goals >= 3 and margin >= 2:
        signal_component += float(signal_weight) * 0.72 * tail_signal
    if dog_goals == 0 and side is not None:
        signal_component += float(signal_weight) * 0.42 * (
            0.50 * v32._clip_signal(role_clean) + 0.50 * v32._clip_signal(xg_clean)
        )
    if dog_goals >= 1:
        signal_component += float(signal_weight) * 0.52 * both_score_signal
    if total_goals >= 5:
        signal_component += float(signal_weight) * 0.25 * attack_signal

    chaos_component = float(chaos_weight) * float(chaos_profile.chaos_index)
    if total_goals >= 5:
        chaos_component *= 1.20
    elif total_goals == 4:
        chaos_component *= 0.92

    utility = (
        probability_component
        + chaos_component
        + signal_component
        + outcome_component
        + shape_component
        + lambda_component
    )
    return {
        "utility": float(utility),
        "probability_ratio": float(probability_ratio),
        "probability_component": float(probability_component),
        "chaos_component": float(chaos_component),
        "signal_component": float(signal_component),
        "outcome_component": float(outcome_component),
        "shape_component": float(shape_component),
        "lambda_component": float(lambda_component),
        "tail_signal": float(tail_signal),
        "both_score_signal": float(both_score_signal),
        "attack_signal": float(attack_signal),
    }


def select_outlier_scoreline(
    score_matrix: ScoreMatrix,
    result_probabilities: Dict[str, float],
    lambda_a: float,
    lambda_b: float,
    role_a: v30.PlayerRoleProfile,
    role_b: v30.PlayerRoleProfile,
    xg_a: v32.CurrentXGProfile,
    xg_b: v32.CurrentXGProfile,
    chaos_profile: TournamentChaosProfile,
    current_top_scorelines: list[Dict[str, Any]],
    relative_floor: float = DEFAULT_OUTLIER_RELATIVE_FLOOR,
    absolute_floor: float = DEFAULT_OUTLIER_ABSOLUTE_FLOOR,
    probability_weight: float = DEFAULT_OUTLIER_PROBABILITY_WEIGHT,
    chaos_weight: float = DEFAULT_OUTLIER_CHAOS_WEIGHT,
    signal_weight: float = DEFAULT_OUTLIER_SIGNAL_WEIGHT,
    max_goals: int = DEFAULT_OUTLIER_MAX_GOALS,
) -> tuple[Dict[str, Any] | None, Dict[str, Any]]:
    top_three = [_score_key(item) for item in current_top_scorelines[:3]]
    diagnostics: Dict[str, Any] = {
        "outlier_selector_enabled": True,
        "probability_matrix_changed": False,
        "top_three_changed": False,
        "relative_floor": float(relative_floor),
        "absolute_floor": float(absolute_floor),
        "probability_weight": float(probability_weight),
        "chaos_weight": float(chaos_weight),
        "signal_weight": float(signal_weight),
        "max_goals": int(max_goals),
        "base_top_3": [_score_label(key) for key in top_three],
        "tournament_chaos": chaos_profile.diagnostics(),
    }
    if len(top_three) < 3:
        diagnostics["skip_reason"] = "not_enough_top_scorelines"
        return None, diagnostics

    side = v26.favorite_side(result_probabilities)
    role_metrics = v31._favorite_role_metrics(side, role_a, role_b) if side else {}
    xg_metrics = v32._favorite_xg_metrics(side, xg_a, xg_b) if side else {}
    diagnostics.update(
        {
            "favorite_side": side,
            "role_metrics": role_metrics,
            "xg_metrics": xg_metrics,
        }
    )

    top_probability = max(float(score_matrix.get(top_three[0], 0.0)), 1e-12)
    floor = max(float(absolute_floor), top_probability * float(relative_floor))
    candidates = [
        key
        for key in _candidate_pool(score_matrix, top_three, result_probabilities, max_goals)
        if float(score_matrix.get(key, 0.0)) >= floor
    ]
    diagnostics["probability_floor"] = float(floor)
    diagnostics["candidate_count"] = int(len(candidates))
    if not candidates:
        diagnostics["skip_reason"] = "no_outlier_above_floor"
        return None, diagnostics

    utilities = {
        key: _outlier_utility(
            key,
            score_matrix,
            top_three,
            result_probabilities,
            lambda_a,
            lambda_b,
            role_metrics,
            xg_metrics,
            chaos_profile,
            probability_weight,
            chaos_weight,
            signal_weight,
        )
        for key in candidates
    }
    best = max(candidates, key=lambda key: utilities[key]["utility"])
    probability = float(score_matrix.get(best, 0.0))
    diagnostics.update(
        {
            "outlier_selected": True,
            "outlier_scoreline": _score_label(best),
            "outlier_probability": probability,
            "outlier_kind": _candidate_kind(best, side),
            "outlier_utility": float(utilities[best]["utility"]),
            "outlier_details": utilities[best],
            "top_candidates": [
                {
                    "scoreline": _score_label(key),
                    "probability": float(score_matrix.get(key, 0.0)),
                    "kind": _candidate_kind(key, side),
                    "utility": float(utilities[key]["utility"]),
                }
                for key in sorted(
                    candidates,
                    key=lambda key: utilities[key]["utility"],
                    reverse=True,
                )[:8]
            ],
        }
    )
    outlier = {
        **_score_item(best, probability),
        "rank_label": "outlier",
        "kind": diagnostics["outlier_kind"],
        "utility": float(utilities[best]["utility"]),
    }
    return outlier, diagnostics


class V33OutlierSlotModel:
    """Wrap V32 and append a separate high-tail scoreline ticket."""

    def __init__(
        self,
        base_model: v32.V32ThirdSlotCoverageModel,
        chaos_profile: TournamentChaosProfile,
        relative_floor: float = DEFAULT_OUTLIER_RELATIVE_FLOOR,
        absolute_floor: float = DEFAULT_OUTLIER_ABSOLUTE_FLOOR,
        probability_weight: float = DEFAULT_OUTLIER_PROBABILITY_WEIGHT,
        chaos_weight: float = DEFAULT_OUTLIER_CHAOS_WEIGHT,
        signal_weight: float = DEFAULT_OUTLIER_SIGNAL_WEIGHT,
        max_goals: int = DEFAULT_OUTLIER_MAX_GOALS,
    ):
        self.base_model = base_model
        self.chaos_profile = chaos_profile
        self.relative_floor = float(max(relative_floor, 0.0))
        self.absolute_floor = float(max(absolute_floor, 0.0))
        self.probability_weight = float(max(probability_weight, 0.0))
        self.chaos_weight = float(max(chaos_weight, 0.0))
        self.signal_weight = float(signal_weight)
        self.max_goals = int(max(max_goals, 4))
        self.training_data_summary = getattr(base_model, "training_data_summary", {})

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_model, name)

    def role_for_team(self, team: object) -> v30.PlayerRoleProfile:
        return self.base_model.role_for_team(team)

    def xg_for_team(self, team: object) -> v32.CurrentXGProfile:
        return self.base_model.xg_for_team(team)

    def predict(self, *args, **kwargs) -> Dict[str, Any]:
        team_a = kwargs.get("team_a", args[0] if args else "")
        team_b = kwargs.get("team_b", args[1] if len(args) > 1 else "")
        prediction = self.base_model.predict(*args, **kwargs)
        score_matrix = v29.score_matrix_from_prediction(prediction)
        outlier, diagnostics = select_outlier_scoreline(
            score_matrix,
            prediction["result_probabilities"],
            float(prediction["lambda_a"]),
            float(prediction["lambda_b"]),
            self.role_for_team(team_a),
            self.role_for_team(team_b),
            self.xg_for_team(team_a),
            self.xg_for_team(team_b),
            self.chaos_profile,
            prediction.get("top_scorelines", []),
            relative_floor=self.relative_floor,
            absolute_floor=self.absolute_floor,
            probability_weight=self.probability_weight,
            chaos_weight=self.chaos_weight,
            signal_weight=self.signal_weight,
            max_goals=self.max_goals,
        )
        prediction["outlier_scoreline"] = outlier
        prediction["top_scorelines_plus_outlier"] = [
            *prediction.get("top_scorelines", [])[:3],
            *([outlier] if outlier else []),
        ]
        prediction["v33_adjustments"] = {
            "base_model": "v32_third_slot_coverage",
            "scoreline_policy": "top_3_preserved_plus_labeled_outlier",
            "probability_matrix_changed": False,
            "scoreline_layer_affects_wdl": False,
            **diagnostics,
        }
        prediction["calibration_notes"] = {
            **prediction.get("calibration_notes", {}),
            "v33": prediction["v33_adjustments"],
            "outlier_policy": (
                "V33 preserves the base Top-3 exactly. It appends a fourth, "
                "labeled outlier ticket selected from high-total/high-margin "
                "scorelines using current tournament chaos, player role form, "
                "xG/xA/xGOT, and keeper/concession signals."
            ),
        }
        return prediction


def build_from_zip(
    zip_path,
    train_csv=None,
    test_csv=None,
    model_type="catboost",
    box_csv=None,
    results_csv=None,
    former_names_csv=None,
    prediction_year=2026,
    qualifier_blend_start_year=2014,
    qualifier_full_weight_year=2022,
    qualifier_minimum_influence=0.0,
    recency_half_life_years=16.0,
    recency_min_weight=0.10,
    player_ratings_csv=None,
    declared_squads_csv=None,
    fcratings_csv=None,
    results_as_of=v15.DEFAULT_RESULTS_AS_OF,
    observed_matches_csv=None,
    fotmob_leaders_csv=None,
    fotmob_player_stats_csv=None,
    fotmob_lineups_csv=None,
    fotmob_substitutions_csv=None,
    fotmob_keeper_stats_csv=None,
    fotmob_match_facts_csv=None,
    outlier_relative_floor=DEFAULT_OUTLIER_RELATIVE_FLOOR,
    outlier_absolute_floor=DEFAULT_OUTLIER_ABSOLUTE_FLOOR,
    outlier_probability_weight=DEFAULT_OUTLIER_PROBABILITY_WEIGHT,
    outlier_chaos_weight=DEFAULT_OUTLIER_CHAOS_WEIGHT,
    outlier_signal_weight=DEFAULT_OUTLIER_SIGNAL_WEIGHT,
    outlier_max_goals=DEFAULT_OUTLIER_MAX_GOALS,
    **kwargs,
):
    data_dir = Path(__file__).resolve().parent / "data"
    fotmob_match_facts_csv = fotmob_match_facts_csv or (
        data_dir / "fotmob_match_facts_clean.csv"
    )
    base_model, data = v32.build_from_zip(
        zip_path,
        train_csv=train_csv,
        test_csv=test_csv,
        model_type=model_type,
        box_csv=box_csv,
        results_csv=results_csv,
        former_names_csv=former_names_csv,
        prediction_year=prediction_year,
        qualifier_blend_start_year=qualifier_blend_start_year,
        qualifier_full_weight_year=qualifier_full_weight_year,
        qualifier_minimum_influence=qualifier_minimum_influence,
        recency_half_life_years=recency_half_life_years,
        recency_min_weight=recency_min_weight,
        player_ratings_csv=player_ratings_csv,
        declared_squads_csv=declared_squads_csv,
        fcratings_csv=fcratings_csv,
        results_as_of=results_as_of,
        observed_matches_csv=observed_matches_csv,
        fotmob_leaders_csv=fotmob_leaders_csv,
        fotmob_player_stats_csv=fotmob_player_stats_csv,
        fotmob_lineups_csv=fotmob_lineups_csv,
        fotmob_substitutions_csv=fotmob_substitutions_csv,
        fotmob_keeper_stats_csv=fotmob_keeper_stats_csv,
        **kwargs,
    )
    chaos_profile = build_tournament_chaos_profile(fotmob_match_facts_csv)
    model = V33OutlierSlotModel(
        base_model,
        chaos_profile,
        relative_floor=outlier_relative_floor,
        absolute_floor=outlier_absolute_floor,
        probability_weight=outlier_probability_weight,
        chaos_weight=outlier_chaos_weight,
        signal_weight=outlier_signal_weight,
        max_goals=outlier_max_goals,
    )
    model.training_data_summary = {
        **getattr(base_model, "training_data_summary", {}),
        "v33_scoreline_policy": "top_3_preserved_plus_labeled_outlier",
        "v33_probability_matrix_changed": False,
        "v33_fotmob_match_facts_csv": str(fotmob_match_facts_csv),
        "v33_tournament_chaos": chaos_profile.diagnostics(),
        "v33_outlier_relative_floor": model.relative_floor,
        "v33_outlier_absolute_floor": model.absolute_floor,
        "v33_outlier_probability_weight": model.probability_weight,
        "v33_outlier_chaos_weight": model.chaos_weight,
        "v33_outlier_signal_weight": model.signal_weight,
        "v33_outlier_max_goals": model.max_goals,
    }
    return model, data


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    parser = argparse.ArgumentParser(
        description="Run V33: V32 Top-3 preserved with one labeled outlier ticket."
    )
    parser.add_argument("--team-a", required=True)
    parser.add_argument("--team-b", required=True)
    parser.add_argument("--host-a", action="store_true")
    parser.add_argument("--host-b", action="store_true")
    parser.add_argument("--knockout", action="store_true")
    parser.add_argument("--outdir", default="outputs/outputs_v33_outlier_slot")
    parser.add_argument("--worldcupsai-zip", default=str(data_dir / "worldcupsai.zip"))
    parser.add_argument("--team-train", default=str(data_dir / "current_team_features_2026.csv"))
    parser.add_argument("--team-test")
    parser.add_argument("--box-data", default=str(data_dir / "FIFAallMatchBoxData.csv"))
    parser.add_argument("--results-data", default=str(data_dir / "results.csv"))
    parser.add_argument("--results-as-of", default=v15.DEFAULT_RESULTS_AS_OF)
    parser.add_argument("--former-names", default=str(data_dir / "former_names.csv"))
    parser.add_argument("--prediction-year", type=int, default=2026)
    parser.add_argument("--player-ratings", default=str(data_dir / "player_ratings_international.csv"))
    parser.add_argument("--declared-squads", default=str(data_dir / "world_cup_2026_declared_squads.csv"))
    parser.add_argument("--fcratings", default=str(data_dir / "fcratings_top50_worldcup2026.csv"))
    parser.add_argument("--observed-matches", default=str(data_dir / "wc2026_observed_matches_from_screenshots.csv"))
    parser.add_argument("--fotmob-leaders", default=str(data_dir / "fotmob_stat_leaders_clean.csv"))
    parser.add_argument("--fotmob-player-stats", default=str(data_dir / "fotmob_match_player_stats_clean.csv"))
    parser.add_argument("--fotmob-lineups", default=str(data_dir / "fotmob_match_lineups_clean.csv"))
    parser.add_argument("--fotmob-substitutions", default=str(data_dir / "fotmob_match_substitutions_clean.csv"))
    parser.add_argument("--fotmob-keeper-stats", default=str(data_dir / "fotmob_match_keeper_stats_clean.csv"))
    parser.add_argument("--fotmob-match-facts", default=str(data_dir / "fotmob_match_facts_clean.csv"))
    parser.add_argument("--blind-current-goals", action="store_true")
    parser.add_argument("--ignore-observed-goals", action="store_true")
    parser.add_argument("--ignore-fotmob-goal-stats", action="store_true")
    parser.add_argument("--disable-group-score-context", action="store_true")
    parser.add_argument("--outlier-relative-floor", type=float, default=DEFAULT_OUTLIER_RELATIVE_FLOOR)
    parser.add_argument("--outlier-absolute-floor", type=float, default=DEFAULT_OUTLIER_ABSOLUTE_FLOOR)
    parser.add_argument("--outlier-probability-weight", type=float, default=DEFAULT_OUTLIER_PROBABILITY_WEIGHT)
    parser.add_argument("--outlier-chaos-weight", type=float, default=DEFAULT_OUTLIER_CHAOS_WEIGHT)
    parser.add_argument("--outlier-signal-weight", type=float, default=DEFAULT_OUTLIER_SIGNAL_WEIGHT)
    parser.add_argument("--outlier-max-goals", type=int, default=DEFAULT_OUTLIER_MAX_GOALS)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    output_dir = v11.unique_output_dir(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, _ = build_from_zip(
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
        observed_matches_csv=args.observed_matches,
        fotmob_leaders_csv=args.fotmob_leaders,
        fotmob_player_stats_csv=args.fotmob_player_stats,
        fotmob_lineups_csv=args.fotmob_lineups,
        fotmob_substitutions_csv=args.fotmob_substitutions,
        fotmob_keeper_stats_csv=args.fotmob_keeper_stats,
        fotmob_match_facts_csv=args.fotmob_match_facts,
        include_observed_goals=not (
            args.blind_current_goals or args.ignore_observed_goals
        ),
        include_fotmob_goal_stats=not (
            args.blind_current_goals or args.ignore_fotmob_goal_stats
        ),
        include_group_score_context=not (
            args.blind_current_goals or args.disable_group_score_context
        ),
        outlier_relative_floor=args.outlier_relative_floor,
        outlier_absolute_floor=args.outlier_absolute_floor,
        outlier_probability_weight=args.outlier_probability_weight,
        outlier_chaos_weight=args.outlier_chaos_weight,
        outlier_signal_weight=args.outlier_signal_weight,
        outlier_max_goals=args.outlier_max_goals,
    )
    prediction = model.predict(
        args.team_a,
        args.team_b,
        host_a=args.host_a,
        host_b=args.host_b,
        knockout=args.knockout,
    )
    (output_dir / "single_match_prediction.json").write_text(
        json.dumps(prediction, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(prediction["top_scorelines"]).to_csv(
        output_dir / "scoreline_probabilities_top.csv",
        index=False,
    )
    pd.DataFrame(prediction["scoreline_probabilities"]).to_csv(
        output_dir / "scoreline_probabilities.csv",
        index=False,
    )
    pd.DataFrame(prediction["top_scorelines_plus_outlier"]).to_csv(
        output_dir / "scoreline_probabilities_top_plus_outlier.csv",
        index=False,
    )
    (output_dir / "model_summary.json").write_text(
        json.dumps(
            {
                "version": "v33-outlier-slot",
                "base_model": "v32-third-slot-coverage",
                "team_a": prediction["team_a"],
                "team_b": prediction["team_b"],
                "result_probabilities": prediction["result_probabilities"],
                "top_3_preserved": prediction["top_scorelines"][:3],
                "outlier_scoreline": prediction["outlier_scoreline"],
                "v29_adjustments": prediction.get("v29_adjustments", {}),
                "v32_adjustments": prediction.get("v32_adjustments", {}),
                "v33_adjustments": prediction["v33_adjustments"],
                "expanded_training_data": model.training_data_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not args.no_plots:
        v11.plot_prediction_outputs(prediction, output_dir)
        v28.plot_top3_scorelines(prediction, output_dir / "plots")
    print(
        json.dumps(
            {
                "result_probabilities": prediction["result_probabilities"],
                "predicted_result": prediction["predicted_result"],
                "lambda_a": prediction["lambda_a"],
                "lambda_b": prediction["lambda_b"],
                "top_3_preserved": prediction["top_scorelines"][:3],
                "outlier_scoreline": prediction["outlier_scoreline"],
                "v33_adjustments": {
                    key: value
                    for key, value in prediction["v33_adjustments"].items()
                    if key
                    not in {
                        "role_metrics",
                        "xg_metrics",
                        "top_candidates",
                    }
                },
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

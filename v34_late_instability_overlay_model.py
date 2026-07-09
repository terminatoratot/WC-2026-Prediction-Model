#!/usr/bin/env python3
"""V34: V33 top three plus a late-instability transition outlier.

V34 keeps the normal Top-3 exactly as produced by the base model. The fourth
scoreline is selected by simulating plausible 75+ minute score mutations from
normal source scorelines, using current tournament late-goal rates and
team-specific late goals for/against.

Run:
    MPLCONFIGDIR=.matplotlib_cache .venv/bin/python v34_late_instability_overlay_model.py --team-a "Switzerland" --team-b "Bosnia and Herzegovina" --outdir outputs/outputs_v34_switzerland_bosnia
"""

from __future__ import annotations

import argparse
import json
import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import v11_wcq_results_model as v11
import v15_catboost_model as v15
import v26_top3_coverage_model as v26
import v28_current_worldcup_form_model as v28
import v29_tail_risk_scoreline_model as v29
import v33_outlier_slot_model as v33


ScoreMatrix = Dict[Tuple[int, int], float]

DEFAULT_LATE_RELATIVE_FLOOR = 0.030
DEFAULT_LATE_ABSOLUTE_FLOOR = 0.0015
DEFAULT_LATE_MAX_GOALS = 7
DEFAULT_LATE_SOURCE_LIMIT = 10


@dataclass
class TeamLateProfile:
    team: str
    matches: int = 0
    late_for: int = 0
    late_against: int = 0
    post_second_break_for: int = 0
    post_second_break_against: int = 0
    stoppage_for: int = 0
    stoppage_against: int = 0

    @property
    def late_for_rate(self) -> float:
        return (self.late_for + 0.25) / (self.matches + 1.0)

    @property
    def late_against_rate(self) -> float:
        return (self.late_against + 0.25) / (self.matches + 1.0)

    @property
    def post_second_break_for_rate(self) -> float:
        return (self.post_second_break_for + 0.20) / (self.matches + 1.0)

    @property
    def post_second_break_against_rate(self) -> float:
        return (self.post_second_break_against + 0.20) / (self.matches + 1.0)

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "team": self.team,
            "matches": int(self.matches),
            "late_for": int(self.late_for),
            "late_against": int(self.late_against),
            "late_for_rate": float(self.late_for_rate),
            "late_against_rate": float(self.late_against_rate),
            "post_second_break_for": int(self.post_second_break_for),
            "post_second_break_against": int(self.post_second_break_against),
            "post_second_break_for_rate": float(self.post_second_break_for_rate),
            "post_second_break_against_rate": float(
                self.post_second_break_against_rate
            ),
            "stoppage_for": int(self.stoppage_for),
            "stoppage_against": int(self.stoppage_against),
        }


@dataclass
class LateTournamentProfile:
    event_goals: int = 0
    matches: int = 0
    late_goals: int = 0
    post_second_break_goals: int = 0
    stoppage_goals: int = 0
    late_goal_share: float = 0.0
    late_goals_per_match: float = 0.0
    post_second_break_share: float = 0.0
    stoppage_goal_share: float = 0.0
    late_multiplier: float = 1.0
    post_second_break_multiplier: float = 1.0
    stoppage_multiplier: float = 1.0
    instability_index: float = 0.0

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "event_goals": int(self.event_goals),
            "matches": int(self.matches),
            "late_goals": int(self.late_goals),
            "post_second_break_goals": int(self.post_second_break_goals),
            "stoppage_goals": int(self.stoppage_goals),
            "late_goal_share": float(self.late_goal_share),
            "late_goals_per_match": float(self.late_goals_per_match),
            "post_second_break_share": float(self.post_second_break_share),
            "stoppage_goal_share": float(self.stoppage_goal_share),
            "late_multiplier": float(self.late_multiplier),
            "post_second_break_multiplier": float(self.post_second_break_multiplier),
            "stoppage_multiplier": float(self.stoppage_multiplier),
            "instability_index": float(self.instability_index),
        }


def canon(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = "".join(ch for ch in text if ch.isalnum())
    aliases = {
        "bosnia": "bosniaandherzegovina",
        "bosniaherzegovina": "bosniaandherzegovina",
        "czechrepublic": "czechia",
        "turkey": "turkiye",
        "caboverde": "capeverde",
        "cabo": "capeverde",
        "democraticrepublicofthecongo": "drcongo",
        "congodr": "drcongo",
        "unitedstates": "usa",
    }
    return aliases.get(text, text)


def _score_key(item: Dict[str, Any]) -> Tuple[int, int]:
    return int(item["team_a_goals"]), int(item["team_b_goals"])


def _score_label(key: Tuple[int, int]) -> str:
    return f"{key[0]}-{key[1]}"


def _score_item(key: Tuple[int, int], probability: float) -> Dict[str, Any]:
    return {
        "team_a_goals": int(key[0]),
        "team_b_goals": int(key[1]),
        "probability": float(probability),
    }


def _score_outcome(key: Tuple[int, int]) -> str:
    if key[0] > key[1]:
        return "team_a_win"
    if key[1] > key[0]:
        return "team_b_win"
    return "draw"


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _empty_team_profile(team: object) -> TeamLateProfile:
    return TeamLateProfile(team=str(team or ""))


def build_late_profiles(
    goal_events_csv: str | Path | None,
) -> tuple[LateTournamentProfile, dict[str, TeamLateProfile]]:
    if not goal_events_csv or not Path(goal_events_csv).exists():
        return LateTournamentProfile(), {}
    events = pd.read_csv(goal_events_csv)
    if events.empty:
        return LateTournamentProfile(), {}
    events["elapsed_minute"] = pd.to_numeric(
        events.get("elapsed_minute", 0.0),
        errors="coerce",
    ).fillna(0.0)
    events["stoppage_minute"] = pd.to_numeric(
        events.get("stoppage_minute", 0.0),
        errors="coerce",
    ).fillna(0.0)
    late = events["elapsed_minute"] >= 75
    post_second_break = events["elapsed_minute"] >= 65
    stoppage = events["stoppage_minute"] > 0
    matches = int(events["match_id"].nunique())
    event_goals = int(len(events))
    late_goals = int(late.sum())
    post_second_break_goals = int(post_second_break.sum())
    stoppage_goals = int(stoppage.sum())
    late_share = late_goals / max(event_goals, 1)
    post_second_break_share = post_second_break_goals / max(event_goals, 1)
    stoppage_share = stoppage_goals / max(event_goals, 1)
    late_multiplier = float(np.clip(late_share / 0.22, 0.70, 1.85))
    post_second_break_multiplier = float(
        np.clip(post_second_break_share / 0.34, 0.70, 1.65)
    )
    stoppage_multiplier = float(np.clip(stoppage_share / 0.09, 0.70, 2.10))
    instability_index = _clip01(
        0.48 * max(0.0, late_multiplier - 1.0) / 0.85
        + 0.22 * max(0.0, post_second_break_multiplier - 1.0) / 0.65
        + 0.20 * max(0.0, stoppage_multiplier - 1.0) / 1.10
        + 0.10 * _clip01((late_goals / max(matches, 1) - 0.70) / 0.65)
    )
    tournament = LateTournamentProfile(
        event_goals=event_goals,
        matches=matches,
        late_goals=late_goals,
        post_second_break_goals=post_second_break_goals,
        stoppage_goals=stoppage_goals,
        late_goal_share=late_share,
        late_goals_per_match=late_goals / max(matches, 1),
        post_second_break_share=post_second_break_share,
        stoppage_goal_share=stoppage_share,
        late_multiplier=late_multiplier,
        post_second_break_multiplier=post_second_break_multiplier,
        stoppage_multiplier=stoppage_multiplier,
        instability_index=instability_index,
    )

    match_teams: dict[str, set[str]] = {}
    for row in events.to_dict(orient="records"):
        match_id = str(row.get("match_id", ""))
        for column in ["home_team", "away_team"]:
            team = str(row.get(column, "") or "")
            if team:
                match_teams.setdefault(canon(team), set()).add(match_id)

    profiles = {
        team_key: TeamLateProfile(team=team_key, matches=len(match_ids))
        for team_key, match_ids in match_teams.items()
    }
    for row in events.to_dict(orient="records"):
        scoring_key = canon(row.get("scoring_country", ""))
        opponent_key = canon(row.get("opponent_country", ""))
        elapsed = float(row.get("elapsed_minute", 0.0) or 0.0)
        stoppage_value = float(row.get("stoppage_minute", 0.0) or 0.0)
        if scoring_key not in profiles:
            profiles[scoring_key] = TeamLateProfile(team=scoring_key)
        if opponent_key not in profiles:
            profiles[opponent_key] = TeamLateProfile(team=opponent_key)
        if elapsed >= 75:
            profiles[scoring_key].late_for += 1
            profiles[opponent_key].late_against += 1
        if elapsed >= 65:
            profiles[scoring_key].post_second_break_for += 1
            profiles[opponent_key].post_second_break_against += 1
        if stoppage_value > 0:
            profiles[scoring_key].stoppage_for += 1
            profiles[opponent_key].stoppage_against += 1
    return tournament, profiles


def _favorite_side(result_probabilities: Dict[str, float]) -> str | None:
    return v26.favorite_side(result_probabilities)


def _team_add_signal(
    add_goals: int,
    team_profile: TeamLateProfile,
    opponent_profile: TeamLateProfile,
    tournament: LateTournamentProfile,
) -> float:
    if add_goals <= 0:
        return 0.0
    baseline = tournament.late_goals_per_match / 2.0 if tournament.matches else 0.40
    score_rate = 0.58 * team_profile.late_for_rate + 0.42 * opponent_profile.late_against_rate
    post_break_rate = (
        0.54 * team_profile.post_second_break_for_rate
        + 0.46 * opponent_profile.post_second_break_against_rate
    )
    signal = 0.68 * (score_rate - baseline) + 0.32 * (post_break_rate - baseline)
    return float(np.clip(signal, -0.35, 0.75)) * add_goals


def _mutation_candidates(
    source: Tuple[int, int],
    side: str | None,
    max_goals: int,
) -> list[tuple[Tuple[int, int], Tuple[int, int], str, float]]:
    if side == "team_b":
        favorite_add = (0, 1)
        underdog_add = (1, 0)
    else:
        favorite_add = (1, 0)
        underdog_add = (0, 1)

    base_mutations = [
        (favorite_add, "favorite_plus_one", 1.00),
        (underdog_add, "underdog_plus_one", 0.78),
        ((1, 1), "open_exchange_plus_one_each", 0.72),
        ((favorite_add[0] * 2, favorite_add[1] * 2), "favorite_plus_two", 0.55),
        ((underdog_add[0] * 2, underdog_add[1] * 2), "underdog_plus_two", 0.30),
        (
            (favorite_add[0] * 2 + underdog_add[0], favorite_add[1] * 2 + underdog_add[1]),
            "favorite_plus_two_underdog_plus_one",
            0.34,
        ),
        (
            (favorite_add[0] + underdog_add[0] * 2, favorite_add[1] + underdog_add[1] * 2),
            "underdog_plus_two_favorite_plus_one",
            0.20,
        ),
    ]
    candidates = []
    for add, kind, weight in base_mutations:
        final = (source[0] + add[0], source[1] + add[1])
        if 0 <= final[0] <= max_goals and 0 <= final[1] <= max_goals:
            candidates.append((final, add, kind, weight))
    return candidates


def _late_index(
    result_probabilities: Dict[str, float],
    lambda_a: float,
    lambda_b: float,
    top_three: list[Tuple[int, int]],
    tournament: LateTournamentProfile,
    team_a_profile: TeamLateProfile,
    team_b_profile: TeamLateProfile,
) -> Dict[str, float]:
    side = _favorite_side(result_probabilities)
    favorite_probability = (
        float(result_probabilities.get("team_a_win", 0.0))
        if side == "team_a"
        else float(result_probabilities.get("team_b_win", 0.0))
        if side == "team_b"
        else 0.0
    )
    lambda_gap = abs(float(lambda_a) - float(lambda_b))
    total_lambda = float(lambda_a) + float(lambda_b)
    top3_max_total = max(sum(key) for key in top_three) if top_three else 0
    top3_has_tail = any(sum(key) >= 4 or max(key) >= 4 for key in top_three)
    top3_low_total_undercoverage = (
        0.50 * float(top3_max_total <= 3)
        + 0.30 * float(not top3_has_tail)
        + 0.20 * _clip01((total_lambda - 2.35) / 0.95)
    )
    team_late_signal = 0.5 * (
        team_a_profile.late_for_rate
        + team_b_profile.late_for_rate
        + team_a_profile.late_against_rate
        + team_b_profile.late_against_rate
    )
    baseline_team_signal = tournament.late_goals_per_match / 2.0 if tournament.matches else 0.40
    team_component = _clip01((team_late_signal - baseline_team_signal + 0.25) / 0.75)
    total_component = _clip01((total_lambda - 2.20) / 1.45)
    favorite_component = _clip01(
        0.62 * ((favorite_probability - 0.50) / 0.30)
        + 0.38 * (lambda_gap / 1.35)
    )
    tournament_component = tournament.instability_index
    index = _clip01(
        0.42 * tournament_component
        + 0.18 * team_component
        + 0.17 * total_component
        + 0.13 * favorite_component
        + 0.10 * _clip01(top3_low_total_undercoverage)
    )
    return {
        "late_instability_index": float(index),
        "tournament_component": float(tournament_component),
        "team_component": float(team_component),
        "total_lambda_component": float(total_component),
        "favorite_component": float(favorite_component),
        "top3_low_total_undercoverage": float(_clip01(top3_low_total_undercoverage)),
        "favorite_side": side or "",
        "favorite_probability": float(favorite_probability),
        "lambda_gap": float(lambda_gap),
        "total_lambda": float(total_lambda),
        "top3_max_total": int(top3_max_total),
        "top3_has_tail": bool(top3_has_tail),
    }


def select_late_instability_outlier(
    score_matrix: ScoreMatrix,
    result_probabilities: Dict[str, float],
    lambda_a: float,
    lambda_b: float,
    top_scorelines: list[Dict[str, Any]],
    tournament: LateTournamentProfile,
    team_a_profile: TeamLateProfile,
    team_b_profile: TeamLateProfile,
    relative_floor: float = DEFAULT_LATE_RELATIVE_FLOOR,
    absolute_floor: float = DEFAULT_LATE_ABSOLUTE_FLOOR,
    source_limit: int = DEFAULT_LATE_SOURCE_LIMIT,
    max_goals: int = DEFAULT_LATE_MAX_GOALS,
) -> tuple[Dict[str, Any] | None, Dict[str, Any]]:
    ranked = [
        key
        for key, _ in sorted(score_matrix.items(), key=lambda item: item[1], reverse=True)
    ]
    top_three = [_score_key(item) for item in top_scorelines[:3]]
    diagnostics: Dict[str, Any] = {
        "selector": "late_instability_transition_overlay",
        "probability_matrix_changed": False,
        "top_three_changed": False,
        "base_top_3": [_score_label(key) for key in top_three],
        "relative_floor": float(relative_floor),
        "absolute_floor": float(absolute_floor),
        "source_limit": int(source_limit),
        "max_goals": int(max_goals),
        "tournament_late_profile": tournament.diagnostics(),
        "team_a_late_profile": team_a_profile.diagnostics(),
        "team_b_late_profile": team_b_profile.diagnostics(),
    }
    if len(top_three) < 3:
        diagnostics["skip_reason"] = "not_enough_top_scorelines"
        return None, diagnostics

    side = _favorite_side(result_probabilities)
    index = _late_index(
        result_probabilities,
        lambda_a,
        lambda_b,
        top_three,
        tournament,
        team_a_profile,
        team_b_profile,
    )
    diagnostics.update(index)
    top_probability = max(float(score_matrix.get(top_three[0], 0.0)), 1e-12)
    floor = max(float(absolute_floor), top_probability * float(relative_floor))
    sources: list[Tuple[int, int]] = []
    for key in [*top_three, *ranked[:source_limit]]:
        if key not in sources and key in score_matrix:
            sources.append(key)
    candidates: dict[Tuple[int, int], Dict[str, Any]] = {}
    for source in sources:
        source_probability = float(score_matrix.get(source, 0.0))
        for final, add, mutation_kind, mutation_weight in _mutation_candidates(
            source,
            side,
            max_goals,
        ):
            if final in set(top_three):
                continue
            final_probability = float(score_matrix.get(final, 0.0))
            if final_probability < floor:
                continue
            add_a, add_b = add
            team_signal = _team_add_signal(
                add_a,
                team_a_profile,
                team_b_profile,
                tournament,
            ) + _team_add_signal(
                add_b,
                team_b_profile,
                team_a_profile,
                tournament,
            )
            outcome_bonus = 0.06 if _score_outcome(final) == max(
                result_probabilities,
                key=result_probabilities.get,
            ) else 0.0
            total_bonus = 0.04 * max(0, sum(final) - 3)
            if sum(final) >= 5:
                total_bonus += 0.04 * tournament.stoppage_multiplier
            mutation_component = (
                mutation_weight
                * (0.68 + 0.70 * index["late_instability_index"])
                * tournament.late_multiplier
            )
            utility = (
                0.30 * math.sqrt(max(source_probability / top_probability, 0.0))
                + 0.24 * math.sqrt(max(final_probability / top_probability, 0.0))
                + 0.36 * mutation_component
                + 0.18 * team_signal
                + 0.10 * index["top3_low_total_undercoverage"]
                + outcome_bonus
                + total_bonus
            )
            current = candidates.get(final)
            candidate = {
                "scoreline": _score_label(final),
                "source_scoreline": _score_label(source),
                "added_goals": _score_label(add),
                "mutation_kind": mutation_kind,
                "probability": final_probability,
                "source_probability": source_probability,
                "utility": float(utility),
                "mutation_component": float(mutation_component),
                "team_signal": float(team_signal),
                "outcome_bonus": float(outcome_bonus),
                "total_bonus": float(total_bonus),
                "probability_ratio": float(final_probability / top_probability),
                "source_probability_ratio": float(source_probability / top_probability),
            }
            if current is None or candidate["utility"] > current["utility"]:
                candidates[final] = candidate

    diagnostics["candidate_count"] = len(candidates)
    diagnostics["probability_floor"] = float(floor)
    if not candidates:
        diagnostics["skip_reason"] = "no_late_transition_candidate"
        return None, diagnostics

    best_key, best = max(candidates.items(), key=lambda item: item[1]["utility"])
    diagnostics.update(
        {
            "outlier_selected": True,
            "outlier_scoreline": _score_label(best_key),
            "outlier_probability": float(best["probability"]),
            "outlier_utility": float(best["utility"]),
            "outlier_details": best,
            "top_candidates": sorted(
                candidates.values(),
                key=lambda item: item["utility"],
                reverse=True,
            )[:10],
        }
    )
    outlier = {
        **_score_item(best_key, float(best["probability"])),
        "rank_label": "late_instability_outlier",
        "kind": best["mutation_kind"],
        "source_scoreline": best["source_scoreline"],
        "added_goals": best["added_goals"],
        "utility": float(best["utility"]),
    }
    return outlier, diagnostics


class V34LateInstabilityOverlayModel:
    """Wrap V33 and replace the fourth outlier with a late-transition pick."""

    def __init__(
        self,
        base_model: v33.V33OutlierSlotModel,
        tournament_late_profile: LateTournamentProfile,
        team_late_profiles: dict[str, TeamLateProfile],
        relative_floor: float = DEFAULT_LATE_RELATIVE_FLOOR,
        absolute_floor: float = DEFAULT_LATE_ABSOLUTE_FLOOR,
        source_limit: int = DEFAULT_LATE_SOURCE_LIMIT,
        max_goals: int = DEFAULT_LATE_MAX_GOALS,
    ):
        self.base_model = base_model
        self.tournament_late_profile = tournament_late_profile
        self.team_late_profiles = team_late_profiles
        self.relative_floor = float(max(relative_floor, 0.0))
        self.absolute_floor = float(max(absolute_floor, 0.0))
        self.source_limit = int(max(source_limit, 3))
        self.max_goals = int(max(max_goals, 4))
        self.training_data_summary = getattr(base_model, "training_data_summary", {})

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_model, name)

    def late_profile_for_team(self, team: object) -> TeamLateProfile:
        team_key = canon(team)
        return self.team_late_profiles.get(team_key, _empty_team_profile(team_key))

    def predict(self, *args, **kwargs) -> Dict[str, Any]:
        team_a = kwargs.get("team_a", args[0] if args else "")
        team_b = kwargs.get("team_b", args[1] if len(args) > 1 else "")
        prediction = self.base_model.predict(*args, **kwargs)
        score_matrix = v29.score_matrix_from_prediction(prediction)
        team_a_profile = self.late_profile_for_team(team_a)
        team_b_profile = self.late_profile_for_team(team_b)
        outlier, diagnostics = select_late_instability_outlier(
            score_matrix,
            prediction["result_probabilities"],
            float(prediction["lambda_a"]),
            float(prediction["lambda_b"]),
            prediction.get("top_scorelines", []),
            self.tournament_late_profile,
            team_a_profile,
            team_b_profile,
            relative_floor=self.relative_floor,
            absolute_floor=self.absolute_floor,
            source_limit=self.source_limit,
            max_goals=self.max_goals,
        )
        prediction["v33_outlier_scoreline"] = prediction.get("outlier_scoreline")
        prediction["outlier_scoreline"] = outlier
        prediction["late_instability_outlier"] = outlier
        prediction["top_scorelines_plus_outlier"] = [
            *prediction.get("top_scorelines", [])[:3],
            *([outlier] if outlier else []),
        ]
        prediction["v34_adjustments"] = {
            "base_model": "v33_outlier_slot",
            "scoreline_policy": "top_3_preserved_plus_late_instability_outlier",
            "probability_matrix_changed": False,
            "scoreline_layer_affects_wdl": False,
            **diagnostics,
        }
        prediction["calibration_notes"] = {
            **prediction.get("calibration_notes", {}),
            "v34": prediction["v34_adjustments"],
            "late_instability_policy": (
                "V34 preserves the normal Top-3 exactly. The fourth scoreline "
                "is selected by simulating 75+ minute score mutations from "
                "base scoreline candidates using tournament late-goal rates, "
                "team late goals for/against, lambdas, favorite strength, and "
                "low-total Top-3 undercoverage."
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
    fotmob_goal_events_csv=None,
    late_relative_floor=DEFAULT_LATE_RELATIVE_FLOOR,
    late_absolute_floor=DEFAULT_LATE_ABSOLUTE_FLOOR,
    late_source_limit=DEFAULT_LATE_SOURCE_LIMIT,
    late_max_goals=DEFAULT_LATE_MAX_GOALS,
    **kwargs,
):
    data_dir = Path(__file__).resolve().parent / "data"
    fotmob_goal_events_csv = fotmob_goal_events_csv or (
        data_dir / "fotmob_match_goal_events_clean.csv"
    )
    base_model, data = v33.build_from_zip(
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
        fotmob_match_facts_csv=fotmob_match_facts_csv,
        **kwargs,
    )
    tournament_late, team_late = build_late_profiles(fotmob_goal_events_csv)
    model = V34LateInstabilityOverlayModel(
        base_model,
        tournament_late,
        team_late,
        relative_floor=late_relative_floor,
        absolute_floor=late_absolute_floor,
        source_limit=late_source_limit,
        max_goals=late_max_goals,
    )
    model.training_data_summary = {
        **getattr(base_model, "training_data_summary", {}),
        "v34_scoreline_policy": "top_3_preserved_plus_late_instability_outlier",
        "v34_probability_matrix_changed": False,
        "v34_fotmob_goal_events_csv": str(fotmob_goal_events_csv),
        "v34_tournament_late_profile": tournament_late.diagnostics(),
        "v34_team_late_profile_count": len(team_late),
        "v34_late_relative_floor": model.relative_floor,
        "v34_late_absolute_floor": model.absolute_floor,
        "v34_late_source_limit": model.source_limit,
        "v34_late_max_goals": model.max_goals,
    }
    return model, data


def plot_top3_plus_late_outlier(prediction: Dict[str, Any], outdir: Path) -> None:
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    rows = prediction.get("top_scorelines_plus_outlier", [])
    if not rows:
        return
    labels = []
    probabilities = []
    colors = []
    for idx, row in enumerate(rows):
        label = f"{int(row['team_a_goals'])}-{int(row['team_b_goals'])}"
        if row.get("rank_label") == "late_instability_outlier":
            label += "\nlate outlier"
            colors.append("#F58518")
        else:
            label += f"\n#{idx + 1}"
            colors.append("#4C78A8")
        labels.append(label)
        probabilities.append(float(row["probability"]) * 100.0)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.bar(labels, probabilities, color=colors)
    ax.set_ylabel("Probability (%)")
    ax.set_title("Top 3 plus Late-Instability Outlier")
    ax.spines[["top", "right"]].set_visible(False)
    for idx, value in enumerate(probabilities):
        ax.text(idx, value + 0.15, f"{value:.2f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(plot_dir / "top_3_plus_late_instability_outlier.png", dpi=180)
    plt.close(fig)


def _score_text(item: Dict[str, Any]) -> str:
    return f"{int(item['team_a_goals'])}-{int(item['team_b_goals'])}"


def _result_label(goals_a: int, goals_b: int) -> str:
    if goals_a > goals_b:
        return "team_a_win"
    if goals_b > goals_a:
        return "team_b_win"
    return "draw"


def _result_text(result: str, team_a: str, team_b: str) -> str:
    return {
        "team_a_win": team_a,
        "team_b_win": team_b,
        "draw": "Draw",
    }[result]


def evaluate_observed_matches(
    model: V34LateInstabilityOverlayModel,
    observed_matches_csv: str | Path,
    outdir: str | Path,
    limit: int = 0,
) -> Dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    observed = pd.read_csv(observed_matches_csv)
    if limit and limit > 0:
        observed = observed.head(limit)

    rows = []
    for idx, row in observed.iterrows():
        team_a = str(row["team_a"])
        team_b = str(row["team_b"])
        prediction = model.predict(
            team_a,
            team_b,
            host_a=team_a in {"Canada", "Mexico", "USA", "United States"},
            host_b=team_b in {"Canada", "Mexico", "USA", "United States"},
            knockout=str(row.get("stage", "")).strip().lower() != "group stage",
        )
        score_matrix = v29.score_matrix_from_prediction(prediction)
        actual_key = (int(row["goals_a"]), int(row["goals_b"]))
        actual = _score_label(actual_key)
        top3_items = prediction["top_scorelines"][:3]
        plus_items = prediction.get("top_scorelines_plus_outlier", top3_items)[:4]
        top3 = [_score_text(item) for item in top3_items]
        plus = [_score_text(item) for item in plus_items]
        outlier = prediction.get("late_instability_outlier") or {}
        actual_result = _result_label(*actual_key)
        result_probabilities = prediction["result_probabilities"]
        predicted_result = prediction.get(
            "predicted_result",
            max(result_probabilities, key=result_probabilities.get),
        )
        output_row = {
            "observed_order": int(idx) + 1,
            "match_id": row["match_id"],
            "team_a": team_a,
            "team_b": team_b,
            "actual_score": actual,
            "actual_result": actual_result,
            "actual_result_label": _result_text(actual_result, team_a, team_b),
            "predicted_result": predicted_result,
            "predicted_result_label": _result_text(predicted_result, team_a, team_b),
            "outcome_correct": predicted_result == actual_result,
            "actual_score_probability": float(score_matrix.get(actual_key, 0.0)),
            "team_a_win_probability": float(result_probabilities["team_a_win"]),
            "draw_probability": float(result_probabilities["draw"]),
            "team_b_win_probability": float(result_probabilities["team_b_win"]),
            "predicted_result_probability": float(result_probabilities[predicted_result]),
            "actual_result_probability": float(result_probabilities[actual_result]),
            "lambda_a": float(prediction["lambda_a"]),
            "lambda_b": float(prediction["lambda_b"]),
            "late_instability_index": float(
                prediction["v34_adjustments"].get("late_instability_index", 0.0)
            ),
            "late_outlier_scoreline": _score_text(outlier) if outlier else "",
            "late_outlier_probability": float(outlier.get("probability", 0.0) or 0.0),
            "late_outlier_kind": outlier.get("kind", ""),
            "late_outlier_source_scoreline": outlier.get("source_scoreline", ""),
            "late_outlier_added_goals": outlier.get("added_goals", ""),
            "v33_outlier_scoreline": (
                _score_text(prediction.get("v33_outlier_scoreline"))
                if prediction.get("v33_outlier_scoreline")
                else ""
            ),
        }
        for rank, item in enumerate(top3_items, start=1):
            scoreline = _score_text(item)
            output_row[f"top_{rank}_scoreline"] = scoreline
            output_row[f"top_{rank}_probability"] = float(item["probability"])
            output_row[f"actual_is_top_{rank}"] = actual == scoreline
        output_row["actual_in_top_3"] = actual in set(top3)
        for rank, item in enumerate(plus_items, start=1):
            scoreline = _score_text(item)
            output_row[f"plus_{rank}_scoreline"] = scoreline
            output_row[f"plus_{rank}_probability"] = float(item["probability"])
            output_row[f"actual_is_plus_{rank}"] = actual == scoreline
        output_row["actual_in_top_3_plus_outlier"] = actual in set(plus)
        output_row["outlier_gained_hit"] = (
            output_row["actual_in_top_3_plus_outlier"]
            and not output_row["actual_in_top_3"]
        )
        rows.append(output_row)

    eval_df = pd.DataFrame(rows)
    csv_path = outdir / "v34_late_instability_top3_plus_outlier_comparison_matches.csv"
    eval_df.to_csv(csv_path, index=False)
    summary = {
        "model": "v34_late_instability_overlay",
        "n_matches": int(len(eval_df)),
        "top_1_exact_score_hits": int(eval_df["actual_is_top_1"].sum()),
        "top_2_exact_score_hits": int(
            (eval_df["actual_is_top_1"] | eval_df["actual_is_top_2"]).sum()
        ),
        "top_3_exact_score_hits": int(eval_df["actual_in_top_3"].sum()),
        "top_3_exact_score_accuracy": float(eval_df["actual_in_top_3"].mean()),
        "top_3_plus_outlier_hits": int(
            eval_df["actual_in_top_3_plus_outlier"].sum()
        ),
        "top_3_plus_outlier_accuracy": float(
            eval_df["actual_in_top_3_plus_outlier"].mean()
        ),
        "outlier_gained_hits": int(eval_df["outlier_gained_hit"].sum()),
        "outcome_correct": int(eval_df["outcome_correct"].sum()),
        "mean_late_instability_index": float(eval_df["late_instability_index"].mean()),
        "csv": str(csv_path),
    }

    try:
        import compare_v11_top_scorelines as scoreline_chart

        top3_chart = eval_df.copy()
        top3_chart.attrs["model_label"] = "V34 normal Top-3"
        top3_chart.attrs["top_n"] = 3
        top3_chart.attrs["excluded_count"] = 0
        top3_chart.attrs["max_observed_goals_per_team"] = None
        top3_png = outdir / "v34_late_instability_top_three_scoreline_comparison_matches.png"
        scoreline_chart.draw_scoreline_chart(top3_chart, top3_png)

        plus_chart = eval_df.copy()
        for rank in range(1, 5):
            plus_chart[f"top_{rank}_scoreline"] = plus_chart[
                f"plus_{rank}_scoreline"
            ]
            plus_chart[f"top_{rank}_probability"] = plus_chart[
                f"plus_{rank}_probability"
            ]
            plus_chart[f"actual_is_top_{rank}"] = plus_chart[
                f"actual_is_plus_{rank}"
            ]
        plus_chart["actual_in_top_4"] = plus_chart[
            "actual_in_top_3_plus_outlier"
        ]
        plus_chart.attrs["model_label"] = "V34 Top-3 + late outlier"
        plus_chart.attrs["top_n"] = 4
        plus_chart.attrs["excluded_count"] = 0
        plus_chart.attrs["max_observed_goals_per_team"] = None
        plus_png = outdir / "v34_late_instability_top3_plus_outlier_comparison_matches.png"
        scoreline_chart.draw_scoreline_chart(plus_chart, plus_png)
        summary["top3_png"] = str(top3_png)
        summary["top3_plus_outlier_png"] = str(plus_png)
    except Exception as exc:
        summary["plot_error"] = str(exc)

    summary_path = outdir / "v34_late_instability_observed_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    parser = argparse.ArgumentParser(
        description="Run V34: top-3 preserved with late-instability outlier."
    )
    parser.add_argument("--team-a")
    parser.add_argument("--team-b")
    parser.add_argument("--host-a", action="store_true")
    parser.add_argument("--host-b", action="store_true")
    parser.add_argument("--knockout", action="store_true")
    parser.add_argument("--outdir", default="outputs/outputs_v34_late_instability")
    parser.add_argument("--eval-observed", action="store_true")
    parser.add_argument("--eval-outdir", default="observed_eval/observed_eval_v34_late_instability_top3")
    parser.add_argument("--eval-limit", type=int, default=0)
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
    parser.add_argument("--fotmob-goal-events", default=str(data_dir / "fotmob_match_goal_events_clean.csv"))
    parser.add_argument("--blind-current-goals", action="store_true")
    parser.add_argument("--ignore-observed-goals", action="store_true")
    parser.add_argument("--ignore-fotmob-goal-stats", action="store_true")
    parser.add_argument("--disable-group-score-context", action="store_true")
    parser.add_argument("--late-relative-floor", type=float, default=DEFAULT_LATE_RELATIVE_FLOOR)
    parser.add_argument("--late-absolute-floor", type=float, default=DEFAULT_LATE_ABSOLUTE_FLOOR)
    parser.add_argument("--late-source-limit", type=int, default=DEFAULT_LATE_SOURCE_LIMIT)
    parser.add_argument("--late-max-goals", type=int, default=DEFAULT_LATE_MAX_GOALS)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if not args.eval_observed and (not args.team_a or not args.team_b):
        parser.error("--team-a and --team-b are required unless --eval-observed is set")

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
        fotmob_goal_events_csv=args.fotmob_goal_events,
        include_observed_goals=not (
            args.blind_current_goals or args.ignore_observed_goals
        ),
        include_fotmob_goal_stats=not (
            args.blind_current_goals or args.ignore_fotmob_goal_stats
        ),
        include_group_score_context=not (
            args.blind_current_goals or args.disable_group_score_context
        ),
        late_relative_floor=args.late_relative_floor,
        late_absolute_floor=args.late_absolute_floor,
        late_source_limit=args.late_source_limit,
        late_max_goals=args.late_max_goals,
    )
    if args.eval_observed:
        summary = evaluate_observed_matches(
            model,
            args.observed_matches,
            args.eval_outdir,
            limit=args.eval_limit,
        )
        print(json.dumps(summary, indent=2))
        return

    output_dir = v11.unique_output_dir(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
        output_dir / "scoreline_probabilities_top_plus_late_outlier.csv",
        index=False,
    )
    (output_dir / "model_summary.json").write_text(
        json.dumps(
            {
                "version": "v34-late-instability-overlay",
                "base_model": "v33-outlier-slot",
                "team_a": prediction["team_a"],
                "team_b": prediction["team_b"],
                "result_probabilities": prediction["result_probabilities"],
                "top_3_preserved": prediction["top_scorelines"][:3],
                "late_instability_outlier": prediction["late_instability_outlier"],
                "v33_outlier_scoreline": prediction.get("v33_outlier_scoreline"),
                "v29_adjustments": prediction.get("v29_adjustments", {}),
                "v32_adjustments": prediction.get("v32_adjustments", {}),
                "v33_adjustments": prediction.get("v33_adjustments", {}),
                "v34_adjustments": prediction["v34_adjustments"],
                "expanded_training_data": model.training_data_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not args.no_plots:
        v11.plot_prediction_outputs(prediction, output_dir)
        v28.plot_top3_scorelines(prediction, output_dir / "plots")
        plot_top3_plus_late_outlier(prediction, output_dir)
    print(
        json.dumps(
            {
                "result_probabilities": prediction["result_probabilities"],
                "predicted_result": prediction["predicted_result"],
                "lambda_a": prediction["lambda_a"],
                "lambda_b": prediction["lambda_b"],
                "top_3_preserved": prediction["top_scorelines"][:3],
                "late_instability_outlier": prediction["late_instability_outlier"],
                "v34_adjustments": {
                    key: value
                    for key, value in prediction["v34_adjustments"].items()
                    if key not in {"top_candidates"}
                },
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""V35: top-3 preserved plus game-state late mutation outlier.

V35 improves V34's late outlier by learning a small, shrunk transition table:
score state around 75' -> final score mutation. The normal Top-3 remains
unchanged; only the separate fourth outlier is selected by this layer.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
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
import v34_late_instability_overlay_model as v34


ScoreMatrix = Dict[Tuple[int, int], float]

DEFAULT_STATE_RELATIVE_FLOOR = 0.025
DEFAULT_STATE_ABSOLUTE_FLOOR = 0.0012
DEFAULT_STATE_SOURCE_LIMIT = 12
DEFAULT_STATE_MAX_GOALS = 7
DEFAULT_STATE_PRIOR_STRENGTH = 4.0


@dataclass
class SubImpactProfile:
    team: str
    matches: int = 0
    late_sub_goals_for: int = 0
    late_sub_goals_against: int = 0

    @property
    def late_sub_for_rate(self) -> float:
        return (self.late_sub_goals_for + 0.20) / (self.matches + 1.0)

    @property
    def late_sub_against_rate(self) -> float:
        return (self.late_sub_goals_against + 0.20) / (self.matches + 1.0)

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "team": self.team,
            "matches": int(self.matches),
            "late_sub_goals_for": int(self.late_sub_goals_for),
            "late_sub_goals_against": int(self.late_sub_goals_against),
            "late_sub_for_rate": float(self.late_sub_for_rate),
            "late_sub_against_rate": float(self.late_sub_against_rate),
        }


@dataclass
class GameStateMutationTable:
    state_counts: dict[str, int]
    transition_counts: dict[str, dict[str, int]]
    global_transition_counts: dict[str, int]
    transition_examples: dict[str, list[dict[str, Any]]]
    prior_strength: float = DEFAULT_STATE_PRIOR_STRENGTH

    def transition_probability(self, state: str, mutation_kind: str) -> float:
        state_total = max(int(self.state_counts.get(state, 0)), 0)
        state_count = int(self.transition_counts.get(state, {}).get(mutation_kind, 0))
        global_total = max(sum(self.global_transition_counts.values()), 1)
        global_prob = self.global_transition_counts.get(mutation_kind, 0) / global_total
        return float(
            (state_count + self.prior_strength * global_prob)
            / (state_total + self.prior_strength)
        )

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "state_counts": self.state_counts,
            "transition_counts": self.transition_counts,
            "global_transition_counts": self.global_transition_counts,
            "prior_strength": float(self.prior_strength),
            "transition_examples": self.transition_examples,
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


def normalize_player(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", "", text)


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


def _leader(score: Tuple[int, int]) -> str:
    if score[0] > score[1]:
        return "team_a"
    if score[1] > score[0]:
        return "team_b"
    return "draw"


def state_category(score: Tuple[int, int]) -> str:
    diff = abs(score[0] - score[1])
    total = sum(score)
    if diff == 0:
        return "draw_low" if total <= 2 else "draw_high"
    if diff == 1:
        return "leader_by_1"
    return "leader_by_2plus"


def abstract_mutation_kind(source: Tuple[int, int], final: Tuple[int, int]) -> str:
    add = (max(final[0] - source[0], 0), max(final[1] - source[1], 0))
    if add == (0, 0):
        return "no_late_goal"
    leader = _leader(source)
    if leader == "draw":
        if add[0] > 0 and add[1] > 0:
            return "draw_both_score"
        if max(add) >= 2:
            return "draw_one_side_plus_two"
        return "draw_one_side_plus_one"

    leader_add = add[0] if leader == "team_a" else add[1]
    trailer_add = add[1] if leader == "team_a" else add[0]
    if leader_add > 0 and trailer_add > 0:
        if leader_add >= 2:
            return "leader_plus_two_trailer_plus_one"
        if trailer_add >= 2:
            return "trailer_plus_two_leader_plus_one"
        return "both_plus_one"
    if leader_add >= 2:
        return "leader_plus_two"
    if leader_add == 1:
        return "leader_plus_one"
    if trailer_add >= 2:
        return "trailer_plus_two"
    if trailer_add == 1:
        return "trailer_plus_one"
    return "other"


def score_at_minute(group: pd.DataFrame, minute: float) -> Tuple[int, int]:
    before = group[pd.to_numeric(group["elapsed_minute"], errors="coerce") <= minute]
    if before.empty:
        return (0, 0)
    row = before.sort_values("elapsed_minute").iloc[-1]
    return int(row["home_score_after"]), int(row["away_score_after"])


def build_game_state_mutation_table(
    goal_events_csv: str | Path | None,
    prior_strength: float = DEFAULT_STATE_PRIOR_STRENGTH,
) -> GameStateMutationTable:
    if not goal_events_csv or not Path(goal_events_csv).exists():
        return GameStateMutationTable({}, {}, {}, {}, prior_strength)
    events = pd.read_csv(goal_events_csv)
    if events.empty:
        return GameStateMutationTable({}, {}, {}, {}, prior_strength)
    events["elapsed_minute"] = pd.to_numeric(events["elapsed_minute"], errors="coerce").fillna(0.0)

    state_counts: Counter[str] = Counter()
    transition_counts: dict[str, Counter[str]] = defaultdict(Counter)
    global_counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for match_id, group in events.groupby("match_id"):
        group = group.sort_values("goal_index")
        first = group.iloc[0]
        source = score_at_minute(group, 75.0)
        final = (int(group["home_score_after"].iloc[-1]), int(group["away_score_after"].iloc[-1]))
        state = state_category(source)
        kind = abstract_mutation_kind(source, final)
        state_counts[state] += 1
        transition_counts[state][kind] += 1
        global_counts[kind] += 1
        if len(examples[state]) < 10:
            examples[state].append(
                {
                    "match_id": str(match_id),
                    "match_slug": str(first.get("match_slug", "")),
                    "home_team": str(first.get("home_team", "")),
                    "away_team": str(first.get("away_team", "")),
                    "score_at_75": _score_label(source),
                    "final_score": _score_label(final),
                    "mutation_kind": kind,
                }
            )
    return GameStateMutationTable(
        state_counts=dict(state_counts),
        transition_counts={key: dict(value) for key, value in transition_counts.items()},
        global_transition_counts=dict(global_counts),
        transition_examples=dict(examples),
        prior_strength=float(prior_strength),
    )


def build_sub_impact_profiles(
    goal_events_csv: str | Path | None,
    substitutions_csv: str | Path | None,
) -> dict[str, SubImpactProfile]:
    if (
        not goal_events_csv
        or not substitutions_csv
        or not Path(goal_events_csv).exists()
        or not Path(substitutions_csv).exists()
    ):
        return {}
    events = pd.read_csv(goal_events_csv)
    subs = pd.read_csv(substitutions_csv)
    if events.empty or subs.empty:
        return {}
    events["elapsed_minute"] = pd.to_numeric(events["elapsed_minute"], errors="coerce").fillna(0.0)
    subs["player_key"] = subs["player"].apply(normalize_player)
    sub_lookup = {
        (str(row["match_id"]), str(row["player_key"]))
        for row in subs.to_dict(orient="records")
        if str(row.get("player_key", ""))
    }
    match_teams: dict[str, set[str]] = defaultdict(set)
    for row in events.to_dict(orient="records"):
        match_id = str(row.get("match_id", ""))
        for column in ["home_team", "away_team"]:
            team_key = canon(row.get(column, ""))
            if team_key:
                match_teams[team_key].add(match_id)
    profiles = {
        team: SubImpactProfile(team=team, matches=len(matches))
        for team, matches in match_teams.items()
    }
    for row in events.to_dict(orient="records"):
        if float(row.get("elapsed_minute", 0.0) or 0.0) < 75:
            continue
        match_id = str(row.get("match_id", ""))
        scorer_key = normalize_player(row.get("scorer", ""))
        is_sub = (match_id, scorer_key) in sub_lookup
        if not is_sub:
            candidates = [
                player
                for mid, player in sub_lookup
                if mid == match_id
                and scorer_key
                and (player.endswith(scorer_key) or scorer_key.endswith(player))
            ]
            is_sub = len(candidates) == 1
        if not is_sub:
            continue
        scoring_key = canon(row.get("scoring_country", ""))
        opponent_key = canon(row.get("opponent_country", ""))
        profiles.setdefault(scoring_key, SubImpactProfile(team=scoring_key)).late_sub_goals_for += 1
        profiles.setdefault(opponent_key, SubImpactProfile(team=opponent_key)).late_sub_goals_against += 1
    return profiles


def _empty_sub_profile(team: object) -> SubImpactProfile:
    return SubImpactProfile(team=str(team or ""))


def _favorite_side(result_probabilities: Dict[str, float]) -> str | None:
    return v26.favorite_side(result_probabilities)


def _mutation_candidates_for_source(
    source: Tuple[int, int],
    side: str | None,
    max_goals: int,
) -> list[tuple[Tuple[int, int], Tuple[int, int], str]]:
    leader = _leader(source)
    if side == "team_b":
        favorite_add = (0, 1)
        underdog_add = (1, 0)
    else:
        favorite_add = (1, 0)
        underdog_add = (0, 1)

    if leader == "draw":
        raw = [
            (favorite_add, "draw_one_side_plus_one"),
            (underdog_add, "draw_one_side_plus_one"),
            ((1, 1), "draw_both_score"),
            ((favorite_add[0] * 2, favorite_add[1] * 2), "draw_one_side_plus_two"),
            ((underdog_add[0] * 2, underdog_add[1] * 2), "draw_one_side_plus_two"),
        ]
    else:
        leader_add = (1, 0) if leader == "team_a" else (0, 1)
        trailer_add = (0, 1) if leader == "team_a" else (1, 0)
        raw = [
            (leader_add, "leader_plus_one"),
            (trailer_add, "trailer_plus_one"),
            ((1, 1), "both_plus_one"),
            ((leader_add[0] * 2, leader_add[1] * 2), "leader_plus_two"),
            ((trailer_add[0] * 2, trailer_add[1] * 2), "trailer_plus_two"),
            (
                (leader_add[0] * 2 + trailer_add[0], leader_add[1] * 2 + trailer_add[1]),
                "leader_plus_two_trailer_plus_one",
            ),
            (
                (leader_add[0] + trailer_add[0] * 2, leader_add[1] + trailer_add[1] * 2),
                "trailer_plus_two_leader_plus_one",
            ),
        ]
    candidates = []
    seen = set()
    for add, kind in raw:
        final = (source[0] + add[0], source[1] + add[1])
        if final in seen:
            continue
        seen.add(final)
        if 0 <= final[0] <= max_goals and 0 <= final[1] <= max_goals:
            candidates.append((final, add, kind))
    return candidates


def _team_add_signal(
    add_goals: int,
    team_profile: v34.TeamLateProfile,
    opponent_profile: v34.TeamLateProfile,
    sub_profile: SubImpactProfile,
    opponent_sub_profile: SubImpactProfile,
    tournament: v34.LateTournamentProfile,
) -> float:
    if add_goals <= 0:
        return 0.0
    late_signal = v34._team_add_signal(add_goals, team_profile, opponent_profile, tournament)
    baseline = tournament.late_goals_per_match / 2.0 if tournament.matches else 0.40
    sub_rate = 0.55 * sub_profile.late_sub_for_rate + 0.45 * opponent_sub_profile.late_sub_against_rate
    sub_signal = float(np.clip((sub_rate - 0.30 * baseline) / 0.75, -0.20, 0.55))
    return float(late_signal + 0.28 * sub_signal * add_goals)


def _state_index(
    result_probabilities: Dict[str, float],
    lambda_a: float,
    lambda_b: float,
    top_three: list[Tuple[int, int]],
    tournament: v34.LateTournamentProfile,
    mutation_table: GameStateMutationTable,
) -> Dict[str, float]:
    base = v34._late_index(
        result_probabilities,
        lambda_a,
        lambda_b,
        top_three,
        tournament,
        v34.TeamLateProfile("dummy"),
        v34.TeamLateProfile("dummy"),
    )
    source_states = [state_category(key) for key in top_three]
    table_support = sum(mutation_table.state_counts.get(state, 0) for state in source_states)
    table_component = float(np.clip(table_support / 8.0, 0.0, 1.0))
    state_index = float(np.clip(0.78 * base["late_instability_index"] + 0.22 * table_component, 0.0, 1.0))
    return {
        **base,
        "game_state_table_support": int(table_support),
        "game_state_table_component": table_component,
        "game_state_late_index": state_index,
    }


def select_game_state_late_outlier(
    score_matrix: ScoreMatrix,
    result_probabilities: Dict[str, float],
    lambda_a: float,
    lambda_b: float,
    top_scorelines: list[Dict[str, Any]],
    tournament: v34.LateTournamentProfile,
    team_a_profile: v34.TeamLateProfile,
    team_b_profile: v34.TeamLateProfile,
    sub_a_profile: SubImpactProfile,
    sub_b_profile: SubImpactProfile,
    mutation_table: GameStateMutationTable,
    relative_floor: float = DEFAULT_STATE_RELATIVE_FLOOR,
    absolute_floor: float = DEFAULT_STATE_ABSOLUTE_FLOOR,
    source_limit: int = DEFAULT_STATE_SOURCE_LIMIT,
    max_goals: int = DEFAULT_STATE_MAX_GOALS,
) -> tuple[Dict[str, Any] | None, Dict[str, Any]]:
    ranked = [key for key, _ in sorted(score_matrix.items(), key=lambda item: item[1], reverse=True)]
    top_three = [_score_key(item) for item in top_scorelines[:3]]
    diagnostics: Dict[str, Any] = {
        "selector": "game_state_75_late_mutation_overlay",
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
        "team_a_sub_impact_profile": sub_a_profile.diagnostics(),
        "team_b_sub_impact_profile": sub_b_profile.diagnostics(),
        "mutation_table": mutation_table.diagnostics(),
    }
    if len(top_three) < 3:
        diagnostics["skip_reason"] = "not_enough_top_scorelines"
        return None, diagnostics
    side = _favorite_side(result_probabilities)
    index = _state_index(result_probabilities, lambda_a, lambda_b, top_three, tournament, mutation_table)
    diagnostics.update(index)
    top_probability = max(float(score_matrix.get(top_three[0], 0.0)), 1e-12)
    floor = max(float(absolute_floor), top_probability * float(relative_floor))
    top3_max_total = max(sum(key) for key in top_three)
    min_outlier_total = max(3, min(top3_max_total, 5)) if top3_max_total >= 3 else 2
    diagnostics["min_outlier_total"] = int(min_outlier_total)
    sources: list[Tuple[int, int]] = []
    for key in [*top_three, *ranked[:source_limit]]:
        if key not in sources and key in score_matrix:
            sources.append(key)
    candidates: dict[Tuple[int, int], Dict[str, Any]] = {}
    predicted_result = max(result_probabilities, key=result_probabilities.get)
    for source in sources:
        source_probability = float(score_matrix.get(source, 0.0))
        state = state_category(source)
        for final, add, kind in _mutation_candidates_for_source(source, side, max_goals):
            if final in set(top_three):
                continue
            if sum(final) < min_outlier_total:
                continue
            final_probability = float(score_matrix.get(final, 0.0))
            if final_probability < floor:
                continue
            add_a, add_b = add
            transition_probability = mutation_table.transition_probability(state, kind)
            team_signal = _team_add_signal(
                add_a,
                team_a_profile,
                team_b_profile,
                sub_a_profile,
                sub_b_profile,
                tournament,
            ) + _team_add_signal(
                add_b,
                team_b_profile,
                team_a_profile,
                sub_b_profile,
                sub_a_profile,
                tournament,
            )
            outcome_bonus = 0.07 if _score_outcome(final) == predicted_result else 0.0
            total_bonus = 0.035 * max(0, sum(final) - 3)
            if sum(final) >= 5:
                total_bonus += 0.03 * tournament.stoppage_multiplier
            mutation_component = (
                (0.50 + 1.85 * transition_probability)
                * (0.72 + 0.68 * index["game_state_late_index"])
                * tournament.late_multiplier
            )
            utility = (
                0.24 * math.sqrt(max(source_probability / top_probability, 0.0))
                + 0.22 * math.sqrt(max(final_probability / top_probability, 0.0))
                + 0.40 * mutation_component
                + 0.16 * team_signal
                + 0.08 * index["top3_low_total_undercoverage"]
                + outcome_bonus
                + total_bonus
            )
            candidate = {
                "scoreline": _score_label(final),
                "source_scoreline": _score_label(source),
                "source_state": state,
                "added_goals": _score_label(add),
                "mutation_kind": kind,
                "state_transition_probability": float(transition_probability),
                "probability": float(final_probability),
                "source_probability": float(source_probability),
                "utility": float(utility),
                "mutation_component": float(mutation_component),
                "team_signal": float(team_signal),
                "outcome_bonus": float(outcome_bonus),
                "total_bonus": float(total_bonus),
                "probability_ratio": float(final_probability / top_probability),
                "source_probability_ratio": float(source_probability / top_probability),
            }
            current = candidates.get(final)
            if current is None or candidate["utility"] > current["utility"]:
                candidates[final] = candidate
    diagnostics["candidate_count"] = len(candidates)
    diagnostics["probability_floor"] = float(floor)
    if not candidates:
        diagnostics["skip_reason"] = "no_game_state_candidate"
        return None, diagnostics
    best_key, best = max(candidates.items(), key=lambda item: item[1]["utility"])
    diagnostics.update(
        {
            "outlier_selected": True,
            "outlier_scoreline": _score_label(best_key),
            "outlier_probability": float(best["probability"]),
            "outlier_utility": float(best["utility"]),
            "outlier_details": best,
            "top_candidates": sorted(candidates.values(), key=lambda item: item["utility"], reverse=True)[:10],
        }
    )
    outlier = {
        **_score_item(best_key, float(best["probability"])),
        "rank_label": "game_state_late_outlier",
        "kind": best["mutation_kind"],
        "source_scoreline": best["source_scoreline"],
        "source_state": best["source_state"],
        "added_goals": best["added_goals"],
        "state_transition_probability": best["state_transition_probability"],
        "utility": float(best["utility"]),
    }
    return outlier, diagnostics


class V35GameStateLateMutationModel:
    """Wrap V34 and replace the fourth outlier with a 75'-state mutation pick."""

    def __init__(
        self,
        base_model: v34.V34LateInstabilityOverlayModel,
        mutation_table: GameStateMutationTable,
        sub_impact_profiles: dict[str, SubImpactProfile],
        relative_floor: float = DEFAULT_STATE_RELATIVE_FLOOR,
        absolute_floor: float = DEFAULT_STATE_ABSOLUTE_FLOOR,
        source_limit: int = DEFAULT_STATE_SOURCE_LIMIT,
        max_goals: int = DEFAULT_STATE_MAX_GOALS,
    ):
        self.base_model = base_model
        self.mutation_table = mutation_table
        self.sub_impact_profiles = sub_impact_profiles
        self.relative_floor = float(max(relative_floor, 0.0))
        self.absolute_floor = float(max(absolute_floor, 0.0))
        self.source_limit = int(max(source_limit, 3))
        self.max_goals = int(max(max_goals, 4))
        self.training_data_summary = getattr(base_model, "training_data_summary", {})

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_model, name)

    def late_profile_for_team(self, team: object) -> v34.TeamLateProfile:
        return self.base_model.late_profile_for_team(team)

    def sub_profile_for_team(self, team: object) -> SubImpactProfile:
        team_key = canon(team)
        return self.sub_impact_profiles.get(team_key, _empty_sub_profile(team_key))

    def predict(self, *args, **kwargs) -> Dict[str, Any]:
        team_a = kwargs.get("team_a", args[0] if args else "")
        team_b = kwargs.get("team_b", args[1] if len(args) > 1 else "")
        prediction = self.base_model.predict(*args, **kwargs)
        score_matrix = v29.score_matrix_from_prediction(prediction)
        team_a_profile = self.late_profile_for_team(team_a)
        team_b_profile = self.late_profile_for_team(team_b)
        sub_a = self.sub_profile_for_team(team_a)
        sub_b = self.sub_profile_for_team(team_b)
        outlier, diagnostics = select_game_state_late_outlier(
            score_matrix,
            prediction["result_probabilities"],
            float(prediction["lambda_a"]),
            float(prediction["lambda_b"]),
            prediction.get("top_scorelines", []),
            self.base_model.tournament_late_profile,
            team_a_profile,
            team_b_profile,
            sub_a,
            sub_b,
            self.mutation_table,
            relative_floor=self.relative_floor,
            absolute_floor=self.absolute_floor,
            source_limit=self.source_limit,
            max_goals=self.max_goals,
        )
        prediction["v34_outlier_scoreline"] = prediction.get("late_instability_outlier")
        prediction["outlier_scoreline"] = outlier
        prediction["late_instability_outlier"] = outlier
        prediction["game_state_late_outlier"] = outlier
        prediction["top_scorelines_plus_outlier"] = [
            *prediction.get("top_scorelines", [])[:3],
            *([outlier] if outlier else []),
        ]
        prediction["v35_adjustments"] = {
            "base_model": "v34_late_instability_overlay",
            "scoreline_policy": "top_3_preserved_plus_game_state_late_outlier",
            "probability_matrix_changed": False,
            "scoreline_layer_affects_wdl": False,
            **diagnostics,
        }
        prediction["calibration_notes"] = {
            **prediction.get("calibration_notes", {}),
            "v35": prediction["v35_adjustments"],
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
    state_relative_floor=DEFAULT_STATE_RELATIVE_FLOOR,
    state_absolute_floor=DEFAULT_STATE_ABSOLUTE_FLOOR,
    state_source_limit=DEFAULT_STATE_SOURCE_LIMIT,
    state_max_goals=DEFAULT_STATE_MAX_GOALS,
    state_prior_strength=DEFAULT_STATE_PRIOR_STRENGTH,
    **kwargs,
):
    data_dir = Path(__file__).resolve().parent / "data"
    fotmob_goal_events_csv = fotmob_goal_events_csv or (
        data_dir / "fotmob_match_goal_events_clean.csv"
    )
    fotmob_substitutions_csv = fotmob_substitutions_csv or (
        data_dir / "fotmob_match_substitutions_clean.csv"
    )
    base_model, data = v34.build_from_zip(
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
        fotmob_goal_events_csv=fotmob_goal_events_csv,
        **kwargs,
    )
    mutation_table = build_game_state_mutation_table(
        fotmob_goal_events_csv,
        prior_strength=state_prior_strength,
    )
    sub_profiles = build_sub_impact_profiles(fotmob_goal_events_csv, fotmob_substitutions_csv)
    model = V35GameStateLateMutationModel(
        base_model,
        mutation_table,
        sub_profiles,
        relative_floor=state_relative_floor,
        absolute_floor=state_absolute_floor,
        source_limit=state_source_limit,
        max_goals=state_max_goals,
    )
    model.training_data_summary = {
        **getattr(base_model, "training_data_summary", {}),
        "v35_scoreline_policy": "top_3_preserved_plus_game_state_late_outlier",
        "v35_probability_matrix_changed": False,
        "v35_fotmob_goal_events_csv": str(fotmob_goal_events_csv),
        "v35_fotmob_substitutions_csv": str(fotmob_substitutions_csv),
        "v35_mutation_table": mutation_table.diagnostics(),
        "v35_sub_impact_profile_count": len(sub_profiles),
        "v35_state_relative_floor": model.relative_floor,
        "v35_state_absolute_floor": model.absolute_floor,
        "v35_state_source_limit": model.source_limit,
        "v35_state_max_goals": model.max_goals,
    }
    return model, data


def plot_top3_plus_game_state_outlier(prediction: Dict[str, Any], outdir: Path) -> None:
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    rows = prediction.get("top_scorelines_plus_outlier", [])
    if not rows:
        return
    labels, probabilities, colors = [], [], []
    for idx, row in enumerate(rows):
        label = f"{int(row['team_a_goals'])}-{int(row['team_b_goals'])}"
        if row.get("rank_label") == "game_state_late_outlier":
            label += "\ngame-state outlier"
            colors.append("#F58518")
        else:
            label += f"\n#{idx + 1}"
            colors.append("#4C78A8")
        labels.append(label)
        probabilities.append(float(row["probability"]) * 100.0)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.bar(labels, probabilities, color=colors)
    ax.set_ylabel("Probability (%)")
    ax.set_title("Top 3 plus Game-State Late Outlier")
    ax.spines[["top", "right"]].set_visible(False)
    for idx, value in enumerate(probabilities):
        ax.text(idx, value + 0.15, f"{value:.2f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(plot_dir / "top_3_plus_game_state_late_outlier.png", dpi=180)
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
    return {"team_a_win": team_a, "team_b_win": team_b, "draw": "Draw"}[result]


def evaluate_observed_matches(
    model: V35GameStateLateMutationModel,
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
        team_a, team_b = str(row["team_a"]), str(row["team_b"])
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
        outlier = prediction.get("game_state_late_outlier") or {}
        result_probabilities = prediction["result_probabilities"]
        actual_result = _result_label(*actual_key)
        predicted_result = prediction.get("predicted_result", max(result_probabilities, key=result_probabilities.get))
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
            "game_state_late_index": float(prediction["v35_adjustments"].get("game_state_late_index", 0.0)),
            "late_outlier_scoreline": _score_text(outlier) if outlier else "",
            "late_outlier_probability": float(outlier.get("probability", 0.0) or 0.0),
            "late_outlier_kind": outlier.get("kind", ""),
            "late_outlier_source_scoreline": outlier.get("source_scoreline", ""),
            "late_outlier_source_state": outlier.get("source_state", ""),
            "late_outlier_added_goals": outlier.get("added_goals", ""),
            "late_outlier_transition_probability": float(outlier.get("state_transition_probability", 0.0) or 0.0),
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
        output_row["outlier_gained_hit"] = output_row["actual_in_top_3_plus_outlier"] and not output_row["actual_in_top_3"]
        rows.append(output_row)
    eval_df = pd.DataFrame(rows)
    csv_path = outdir / "v35_game_state_late_top3_plus_outlier_comparison_matches.csv"
    eval_df.to_csv(csv_path, index=False)
    summary = {
        "model": "v35_game_state_late_mutation",
        "n_matches": int(len(eval_df)),
        "top_1_exact_score_hits": int(eval_df["actual_is_top_1"].sum()),
        "top_2_exact_score_hits": int((eval_df["actual_is_top_1"] | eval_df["actual_is_top_2"]).sum()),
        "top_3_exact_score_hits": int(eval_df["actual_in_top_3"].sum()),
        "top_3_exact_score_accuracy": float(eval_df["actual_in_top_3"].mean()),
        "top_3_plus_outlier_hits": int(eval_df["actual_in_top_3_plus_outlier"].sum()),
        "top_3_plus_outlier_accuracy": float(eval_df["actual_in_top_3_plus_outlier"].mean()),
        "outlier_gained_hits": int(eval_df["outlier_gained_hit"].sum()),
        "outcome_correct": int(eval_df["outcome_correct"].sum()),
        "mean_game_state_late_index": float(eval_df["game_state_late_index"].mean()),
        "csv": str(csv_path),
    }
    try:
        import compare_v11_top_scorelines as scoreline_chart

        top3_chart = eval_df.copy()
        top3_chart.attrs["model_label"] = "V35 normal Top-3"
        top3_chart.attrs["top_n"] = 3
        top3_chart.attrs["excluded_count"] = 0
        top3_chart.attrs["max_observed_goals_per_team"] = None
        top3_png = outdir / "v35_game_state_late_top_three_scoreline_comparison_matches.png"
        scoreline_chart.draw_scoreline_chart(top3_chart, top3_png)

        plus_chart = eval_df.copy()
        for rank in range(1, 5):
            plus_chart[f"top_{rank}_scoreline"] = plus_chart[f"plus_{rank}_scoreline"]
            plus_chart[f"top_{rank}_probability"] = plus_chart[f"plus_{rank}_probability"]
            plus_chart[f"actual_is_top_{rank}"] = plus_chart[f"actual_is_plus_{rank}"]
        plus_chart["actual_in_top_4"] = plus_chart["actual_in_top_3_plus_outlier"]
        plus_chart.attrs["model_label"] = "V35 Top-3 + game-state outlier"
        plus_chart.attrs["top_n"] = 4
        plus_chart.attrs["excluded_count"] = 0
        plus_chart.attrs["max_observed_goals_per_team"] = None
        plus_png = outdir / "v35_game_state_late_top3_plus_outlier_comparison_matches.png"
        scoreline_chart.draw_scoreline_chart(plus_chart, plus_png)
        summary["top3_png"] = str(top3_png)
        summary["top3_plus_outlier_png"] = str(plus_png)
    except Exception as exc:
        summary["plot_error"] = str(exc)
    summary_path = outdir / "v35_game_state_late_observed_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    parser = argparse.ArgumentParser(description="Run V35: game-state late mutation outlier.")
    parser.add_argument("--team-a")
    parser.add_argument("--team-b")
    parser.add_argument("--host-a", action="store_true")
    parser.add_argument("--host-b", action="store_true")
    parser.add_argument("--knockout", action="store_true")
    parser.add_argument("--outdir", default="outputs/outputs_v35_game_state_late")
    parser.add_argument("--eval-observed", action="store_true")
    parser.add_argument("--eval-outdir", default="observed_eval/observed_eval_v35_game_state_late_top3")
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
    parser.add_argument("--state-relative-floor", type=float, default=DEFAULT_STATE_RELATIVE_FLOOR)
    parser.add_argument("--state-absolute-floor", type=float, default=DEFAULT_STATE_ABSOLUTE_FLOOR)
    parser.add_argument("--state-source-limit", type=int, default=DEFAULT_STATE_SOURCE_LIMIT)
    parser.add_argument("--state-max-goals", type=int, default=DEFAULT_STATE_MAX_GOALS)
    parser.add_argument("--state-prior-strength", type=float, default=DEFAULT_STATE_PRIOR_STRENGTH)
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
        include_observed_goals=not (args.blind_current_goals or args.ignore_observed_goals),
        include_fotmob_goal_stats=not (args.blind_current_goals or args.ignore_fotmob_goal_stats),
        include_group_score_context=not (args.blind_current_goals or args.disable_group_score_context),
        state_relative_floor=args.state_relative_floor,
        state_absolute_floor=args.state_absolute_floor,
        state_source_limit=args.state_source_limit,
        state_max_goals=args.state_max_goals,
        state_prior_strength=args.state_prior_strength,
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
    (output_dir / "single_match_prediction.json").write_text(json.dumps(prediction, indent=2), encoding="utf-8")
    pd.DataFrame(prediction["top_scorelines"]).to_csv(output_dir / "scoreline_probabilities_top.csv", index=False)
    pd.DataFrame(prediction["scoreline_probabilities"]).to_csv(output_dir / "scoreline_probabilities.csv", index=False)
    pd.DataFrame(prediction["top_scorelines_plus_outlier"]).to_csv(
        output_dir / "scoreline_probabilities_top_plus_game_state_outlier.csv",
        index=False,
    )
    (output_dir / "model_summary.json").write_text(
        json.dumps(
            {
                "version": "v35-game-state-late-mutation",
                "base_model": "v34-late-instability-overlay",
                "team_a": prediction["team_a"],
                "team_b": prediction["team_b"],
                "result_probabilities": prediction["result_probabilities"],
                "top_3_preserved": prediction["top_scorelines"][:3],
                "game_state_late_outlier": prediction["game_state_late_outlier"],
                "v34_outlier_scoreline": prediction.get("v34_outlier_scoreline"),
                "v35_adjustments": prediction["v35_adjustments"],
                "expanded_training_data": model.training_data_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not args.no_plots:
        v11.plot_prediction_outputs(prediction, output_dir)
        v28.plot_top3_scorelines(prediction, output_dir / "plots")
        plot_top3_plus_game_state_outlier(prediction, output_dir)
    print(
        json.dumps(
            {
                "result_probabilities": prediction["result_probabilities"],
                "predicted_result": prediction["predicted_result"],
                "lambda_a": prediction["lambda_a"],
                "lambda_b": prediction["lambda_b"],
                "top_3_preserved": prediction["top_scorelines"][:3],
                "game_state_late_outlier": prediction["game_state_late_outlier"],
                "v35_adjustments": {
                    key: value
                    for key, value in prediction["v35_adjustments"].items()
                    if key not in {"top_candidates", "mutation_table"}
                },
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

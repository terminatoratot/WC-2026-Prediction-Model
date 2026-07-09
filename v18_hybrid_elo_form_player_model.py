#!/usr/bin/env python3
"""V18: V15 CatBoost base with current FC Ratings squad-strength adjustment."""

from __future__ import annotations

import argparse
import json
import math
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd

import v11_wcq_results_model as v11
import v15_catboost_model as v15


canon_team = v11.canon_team

V18_TEAM_ALIASES = {
    "cabo verde": "Cape Verde",
    "cape verde islands": "Cape Verde",
    "curacao": "Curaçao",
    "cote d ivoire": "Côte d'Ivoire",
    "congo dr": "DR Congo",
    "dr congo": "DR Congo",
    "korea republic": "South Korea",
    "south korea": "South Korea",
    "turkiye": "Turkey",
    "usa": "United States",
}

ATTACK_POSITIONS = {"ST", "CF", "LW", "RW", "CAM", "LM", "RM"}
MIDFIELD_POSITIONS = {"CM", "CDM", "CAM", "LM", "RM"}
DEFENSE_POSITIONS = {"CB", "LB", "RB", "LWB", "RWB", "CDM", "GK"}

SQUAD_PROFILE_COMPONENTS = (
    "attack_raw",
    "defense_raw",
    "midfield_raw",
    "keeper_raw",
)

COMPONENT_POSITIONS = {
    "attack": ATTACK_POSITIONS,
    "defense": DEFENSE_POSITIONS,
    "midfield": MIDFIELD_POSITIONS,
    "keeper": {"GK"},
}

POSITIONAL_IMPUTATION_PRIOR_WEIGHT = 6.0
UNMATCHED_PLAYER_DISCOUNT = 1.5

DEFAULT_BETA_ATTACK = 0.11
DEFAULT_BETA_MIDFIELD = 0.035
DEFAULT_BETA_KEEPER = 0.02
DEFAULT_MAX_LOG_ADJUSTMENT = 0.16
DEFAULT_RESULT_BLEND_WEIGHT = 0.22
DEFAULT_PLAYER_RATINGS_AFFECT_WDL = False


def normalize_team(value: Any) -> str:
    key = v15.normalize_text(value)
    return canon_team(V18_TEAM_ALIASES.get(key, str(value).strip()))


def position_set(value: Any) -> set[str]:
    return {
        item.strip().upper()
        for item in str(value or "").split(",")
        if item.strip()
    }


def _tokens(value: str) -> list[str]:
    return [token for token in v15.normalize_text(value).split() if token]


def compact_name_variants(value: Any) -> set[str]:
    tokens = _tokens(str(value or ""))
    if not tokens:
        return set()
    variants = {" ".join(tokens)}
    if len(tokens) >= 2:
        variants.add(f"{tokens[0]} {tokens[-1]}")
        variants.add(f"{tokens[0][0]} {tokens[-1]}")
        variants.add(tokens[-1])
    return variants


def squad_name_variants(row: pd.Series) -> set[str]:
    variants: set[str] = set()
    for column in ("long_name", "short_name"):
        variants.update(compact_name_variants(row.get(column, "")))
    return variants - {""}


def name_match_score(squad_row: pd.Series, player_name: str) -> float:
    variants = squad_name_variants(squad_row)
    player_variants = compact_name_variants(player_name)
    if not variants or not player_variants:
        return 0.0
    if variants & player_variants:
        return 1.0

    long_tokens = set(_tokens(str(squad_row.get("long_name", ""))))
    player_ordered_tokens = _tokens(player_name)
    player_tokens = set(player_ordered_tokens)
    if len(player_ordered_tokens) >= 2 and player_tokens.issubset(long_tokens):
        return 0.96
    if (
        len(player_ordered_tokens) >= 2
        and player_ordered_tokens[-1:] == _tokens(str(squad_row.get("short_name", "")))[-1:]
    ):
        squad_short = _tokens(str(squad_row.get("short_name", "")))
        if squad_short and squad_short[0][:1] == player_ordered_tokens[0][:1]:
            return 0.94

    return max(
        SequenceMatcher(None, left, right).ratio()
        for left in variants
        for right in player_variants
    )


def load_current_squad_names(
    path: str | Path | None,
    fifa_version: int = 26,
) -> pd.DataFrame:
    if not path or not Path(path).exists():
        return pd.DataFrame()
    squads = pd.read_csv(path)
    required = {
        "fifa_version",
        "nationality_name",
        "short_name",
        "long_name",
        "player_positions",
    }
    missing = sorted(required - set(squads.columns))
    if missing:
        raise ValueError(f"Squad ratings file is missing columns: {missing}")
    squads = squads.loc[squads["fifa_version"].eq(fifa_version)].copy()
    squads["team"] = squads["nationality_name"].map(normalize_team)
    squads["positions"] = squads["player_positions"].map(position_set)
    squads["role"] = squads["player_positions"].map(v15.player_role)
    return squads.reset_index(drop=True)


def load_fcratings_players(path: str | Path | None) -> pd.DataFrame:
    if not path or not Path(path).exists():
        return pd.DataFrame()
    players = pd.read_csv(path)
    required = {
        "country",
        "player_name",
        "position",
        "ovr",
        "pac",
        "sho",
        "pas",
        "dri",
        "def",
        "phy",
    }
    missing = sorted(required - set(players.columns))
    if missing:
        raise ValueError(f"FC Ratings file is missing columns: {missing}")
    players = players.copy()
    players["team"] = players["country"].map(normalize_team)
    players["positions"] = players["position"].map(position_set)
    for column in ("ovr", "pac", "sho", "pas", "dri", "def", "phy"):
        players[column] = pd.to_numeric(players[column], errors="coerce")
    return players.dropna(subset=["team", "player_name", "ovr"]).reset_index(
        drop=True
    )


def attacking_value(row: pd.Series) -> float:
    return float(
        0.40 * row["sho"]
        + 0.25 * row["dri"]
        + 0.20 * row["pac"]
        + 0.15 * row["pas"]
    )


def defensive_value(row: pd.Series) -> float:
    return float(
        0.45 * row["def"]
        + 0.25 * row["phy"]
        + 0.15 * row["pac"]
        + 0.15 * row["pas"]
    )


def midfield_value(row: pd.Series) -> float:
    return float(
        0.35 * row["pas"]
        + 0.25 * row["dri"]
        + 0.20 * row["def"]
        + 0.20 * row["phy"]
    )


def _mean_top(values: Iterable[float], count: int) -> float:
    clean = sorted(
        [float(value) for value in values if pd.notna(value)],
        reverse=True,
    )
    if not clean:
        return float("nan")
    return float(np.mean(clean[:count]))


def component_value(row: pd.Series, component: str) -> float:
    if component == "attack":
        return attacking_value(row)
    if component == "defense":
        return defensive_value(row)
    if component == "midfield":
        return midfield_value(row)
    if component == "keeper":
        return float(row["ovr"])
    raise ValueError(f"Unknown rating component: {component}")


def component_values(
    frame: pd.DataFrame,
    component: str,
    positions_column: str = "positions",
) -> list[float]:
    values = []
    wanted_positions = COMPONENT_POSITIONS[component]
    for _, row in frame.iterrows():
        positions = row.get(positions_column, set())
        if not isinstance(positions, set):
            positions = position_set(positions)
        if positions & wanted_positions:
            values.append(component_value(row, component))
    return values


def median_or_nan(values: Iterable[float]) -> float:
    clean = [float(value) for value in values if pd.notna(value)]
    return float(np.median(clean)) if clean else float("nan")


def global_component_baselines(fcratings: pd.DataFrame) -> Dict[str, float]:
    return {
        component: median_or_nan(component_values(fcratings, component))
        for component in COMPONENT_POSITIONS
    }


def shrunken_component_baseline(
    team_values: list[float],
    global_value: float,
) -> float:
    if not team_values and pd.isna(global_value):
        return float("nan")
    if not team_values:
        return float(global_value)
    team_value = median_or_nan(team_values)
    if pd.isna(global_value):
        return float(team_value)
    support = len(team_values)
    team_weight = support / (support + POSITIONAL_IMPUTATION_PRIOR_WEIGHT)
    estimate = team_weight * team_value + (1.0 - team_weight) * global_value
    discount = UNMATCHED_PLAYER_DISCOUNT * (
        POSITIONAL_IMPUTATION_PRIOR_WEIGHT
        / (support + POSITIONAL_IMPUTATION_PRIOR_WEIGHT)
    )
    return float(estimate - discount)


def imputed_component_values_for_player(
    squad_row: pd.Series,
    team_component_values: Dict[str, list[float]],
    global_baselines: Dict[str, float],
) -> Dict[str, float]:
    positions = squad_row["positions"]
    imputed = {}
    for component, wanted_positions in COMPONENT_POSITIONS.items():
        if positions & wanted_positions:
            imputed[component] = shrunken_component_baseline(
                team_component_values.get(component, []),
                global_baselines.get(component, float("nan")),
            )
    return imputed


def best_fcratings_match(
    squad_row: pd.Series,
    candidates: pd.DataFrame,
    threshold: float = 0.84,
) -> tuple[pd.Series | None, float]:
    best_index: Any = None
    best_score = 0.0
    for index, candidate in candidates.iterrows():
        score = name_match_score(squad_row, str(candidate["player_name"]))
        if score > best_score:
            best_index = index
            best_score = score
    if best_index is not None and best_score >= threshold:
        return candidates.loc[best_index], float(best_score)
    return None, float(best_score)


def _aggregate_matched_players(
    players: pd.DataFrame,
    imputed_rows: list[Dict[str, Any]] | None = None,
) -> Dict[str, float]:
    imputed_rows = imputed_rows or []
    if players.empty and not imputed_rows:
        return {
            "attack_raw": float("nan"),
            "defense_raw": float("nan"),
            "midfield_raw": float("nan"),
            "keeper_raw": float("nan"),
        }

    attack_values = []
    defense_values = []
    midfield_values = []
    keeper_values = []
    for _, row in players.iterrows():
        positions = row["squad_positions"]
        if positions & ATTACK_POSITIONS:
            attack_values.append(attacking_value(row))
        if positions & DEFENSE_POSITIONS:
            defense_values.append(defensive_value(row))
        if positions & MIDFIELD_POSITIONS:
            midfield_values.append(midfield_value(row))
        if "GK" in positions:
            keeper_values.append(float(row["ovr"]))

    for row in imputed_rows:
        positions = row["positions"]
        values = row["component_values"]
        if positions & ATTACK_POSITIONS and pd.notna(values.get("attack")):
            attack_values.append(float(values["attack"]))
        if positions & DEFENSE_POSITIONS and pd.notna(values.get("defense")):
            defense_values.append(float(values["defense"]))
        if positions & MIDFIELD_POSITIONS and pd.notna(values.get("midfield")):
            midfield_values.append(float(values["midfield"]))
        if "GK" in positions and pd.notna(values.get("keeper")):
            keeper_values.append(float(values["keeper"]))

    return {
        "attack_raw": _mean_top(attack_values, 5),
        "defense_raw": _mean_top(defense_values, 6),
        "midfield_raw": _mean_top(midfield_values, 5),
        "keeper_raw": _mean_top(keeper_values, 1),
    }


def build_current_fcratings_squad_profiles(
    squad_names: pd.DataFrame,
    fcratings: pd.DataFrame,
    match_threshold: float = 0.84,
) -> Dict[str, Dict[str, Any]]:
    if squad_names.empty or fcratings.empty:
        return {}

    global_baselines = global_component_baselines(fcratings)
    profiles: Dict[str, Dict[str, Any]] = {}
    for team, squad in squad_names.groupby("team", sort=True):
        candidates = fcratings[fcratings["team"].eq(team)]
        team_component_values = {
            component: component_values(candidates, component)
            for component in COMPONENT_POSITIONS
        }
        matched_rows = []
        imputed_rows = []
        scores = []
        unmatched = []
        used_urls: set[str] = set()
        for _, squad_row in squad.iterrows():
            available = candidates
            if "player_url" in available.columns:
                available = available.loc[
                    ~available["player_url"].astype(str).isin(used_urls)
                ]
            matched, score = best_fcratings_match(
                squad_row,
                available,
                threshold=match_threshold,
            )
            if matched is None:
                unmatched.append(str(squad_row.get("long_name", "")))
                imputed_rows.append(
                    {
                        "squad_name": str(squad_row.get("long_name", "")),
                        "positions": squad_row["positions"],
                        "component_values": imputed_component_values_for_player(
                            squad_row,
                            team_component_values,
                            global_baselines,
                        ),
                    }
                )
                continue
            matched_copy = matched.copy()
            matched_copy["squad_positions"] = squad_row["positions"]
            matched_copy["squad_role"] = squad_row["role"]
            matched_copy["squad_short_name"] = squad_row.get("short_name", "")
            matched_copy["squad_long_name"] = squad_row.get("long_name", "")
            matched_copy["match_score"] = score
            if matched_copy.get("player_url"):
                used_urls.add(str(matched_copy.get("player_url")))
            scores.append(score)
            matched_rows.append(matched_copy)

        matched_frame = (
            pd.DataFrame(matched_rows)
            if matched_rows
            else pd.DataFrame(columns=list(fcratings.columns))
        )
        profile = _aggregate_matched_players(matched_frame, imputed_rows)
        match_coverage = float(len(matched_frame) / max(len(squad), 1))
        imputed_share = float(len(imputed_rows) / max(len(squad), 1))
        team_data_support = float(np.clip(len(candidates) / 20.0, 0.0, 1.0))
        rating_confidence = float(
            np.clip(match_coverage + 0.25 * imputed_share * team_data_support, 0.0, 1.0)
        )
        profile.update(
            {
                "team": team,
                "squad_size": int(len(squad)),
                "fcratings_country_rows": int(len(candidates)),
                "matched_players": int(len(matched_frame)),
                "imputed_players": int(len(imputed_rows)),
                "match_coverage": match_coverage,
                "rating_confidence": rating_confidence,
                "mean_name_match_score": float(np.mean(scores)) if scores else 0.0,
                "imputation_policy": (
                    "Unmatched squad players are assigned shrunken positional "
                    "FC Ratings estimates: team component median blended toward "
                    "the global component median, with a sparse-team discount. "
                    "Actual squad-file rating values are never used."
                ),
                "unmatched_players": unmatched[:12],
                "imputed_player_examples": [
                    {
                        "squad_name": row["squad_name"],
                        "positions": sorted(row["positions"]),
                        **{
                            f"imputed_{key}": float(value)
                            for key, value in row["component_values"].items()
                            if pd.notna(value)
                        },
                    }
                    for row in imputed_rows[:12]
                ],
                "matched_player_examples": [
                    {
                        "squad_name": str(row.get("squad_long_name", "")),
                        "fcratings_name": str(row.get("player_name", "")),
                        "position": str(row.get("position", "")),
                        "ovr": float(row.get("ovr", np.nan)),
                        "match_score": float(row.get("match_score", np.nan)),
                    }
                    for _, row in matched_frame.head(12).iterrows()
                ],
            }
        )
        profiles[team] = profile

    return normalize_squad_profile_components(profiles)


def normalize_squad_profile_components(
    profiles: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    if not profiles:
        return profiles
    for component in SQUAD_PROFILE_COMPONENTS:
        values = np.asarray(
            [
                float(profile.get(component, np.nan))
                for profile in profiles.values()
                if pd.notna(profile.get(component, np.nan))
            ],
            dtype=float,
        )
        mean = float(np.mean(values)) if len(values) else float("nan")
        std = float(np.std(values, ddof=0)) if len(values) else float("nan")
        if not np.isfinite(std) or std < 1e-9:
            std = 1.0
        for profile in profiles.values():
            raw_value = profile.get(component, np.nan)
            z_column = component.replace("_raw", "_z")
            if pd.notna(raw_value) and np.isfinite(mean):
                profile[z_column] = float((float(raw_value) - mean) / std)
            else:
                profile[z_column] = 0.0
    return profiles


def result_probs_from_score_matrix(
    score_matrix: Dict[Tuple[int, int], float],
) -> Dict[str, float]:
    return {
        "team_a_win": float(
            sum(prob for (a, b), prob in score_matrix.items() if a > b)
        ),
        "draw": float(
            sum(prob for (a, b), prob in score_matrix.items() if a == b)
        ),
        "team_b_win": float(
            sum(prob for (a, b), prob in score_matrix.items() if a < b)
        ),
    }


def blend_result_probabilities(
    base: Dict[str, float],
    adjusted: Dict[str, float],
    adjusted_weight: float,
) -> Dict[str, float]:
    result = {
        key: (1.0 - adjusted_weight) * float(base[key])
        + adjusted_weight * float(adjusted[key])
        for key in ("team_a_win", "draw", "team_b_win")
    }
    total = sum(result.values())
    return {key: float(value / total) for key, value in result.items()}


class V18HybridSquadModel:
    """Wrap V15 and apply current FC Ratings squad quality on log-goals."""

    def __init__(
        self,
        base_model: v15.V15CatBoostModel,
        squad_profiles: Dict[str, Dict[str, Any]],
        beta_attack: float = DEFAULT_BETA_ATTACK,
        beta_midfield: float = DEFAULT_BETA_MIDFIELD,
        beta_keeper: float = DEFAULT_BETA_KEEPER,
        max_log_adjustment: float = DEFAULT_MAX_LOG_ADJUSTMENT,
        result_blend_weight: float = DEFAULT_RESULT_BLEND_WEIGHT,
        player_ratings_affect_wdl: bool = DEFAULT_PLAYER_RATINGS_AFFECT_WDL,
    ):
        self.base_model = base_model
        self.squad_profiles = squad_profiles
        self.beta_attack = float(beta_attack)
        self.beta_midfield = float(beta_midfield)
        self.beta_keeper = float(beta_keeper)
        self.max_log_adjustment = float(max_log_adjustment)
        self.result_blend_weight = float(result_blend_weight)
        self.player_ratings_affect_wdl = bool(player_ratings_affect_wdl)
        self.training_data_summary = getattr(
            base_model,
            "training_data_summary",
            {},
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_model, name)

    def profile_for_team(self, team: str) -> Dict[str, Any]:
        return self.squad_profiles.get(normalize_team(team), {})

    @staticmethod
    def _coverage(profile: Dict[str, Any]) -> float:
        return float(
            np.clip(
                profile.get(
                    "rating_confidence",
                    profile.get("match_coverage", 0.0),
                ),
                0.0,
                1.0,
            )
        )

    def _log_adjustments(
        self,
        profile_a: Dict[str, Any],
        profile_b: Dict[str, Any],
    ) -> tuple[float, float, Dict[str, float]]:
        coverage_a = self._coverage(profile_a)
        coverage_b = self._coverage(profile_b)
        pair_coverage = math.sqrt(max(coverage_a * coverage_b, 0.0))

        attack_edge_a = float(profile_a.get("attack_z", 0.0)) - float(
            profile_b.get("defense_z", 0.0)
        )
        attack_edge_b = float(profile_b.get("attack_z", 0.0)) - float(
            profile_a.get("defense_z", 0.0)
        )
        midfield_edge_a = float(profile_a.get("midfield_z", 0.0)) - float(
            profile_b.get("midfield_z", 0.0)
        )
        keeper_edge_a = float(profile_a.get("keeper_z", 0.0)) - float(
            profile_b.get("keeper_z", 0.0)
        )

        raw_a = (
            self.beta_attack * attack_edge_a
            + self.beta_midfield * midfield_edge_a
            - self.beta_keeper * keeper_edge_a
        )
        raw_b = (
            self.beta_attack * attack_edge_b
            - self.beta_midfield * midfield_edge_a
            + self.beta_keeper * keeper_edge_a
        )
        log_a = float(
            np.clip(
                pair_coverage * raw_a,
                -self.max_log_adjustment,
                self.max_log_adjustment,
            )
        )
        log_b = float(
            np.clip(
                pair_coverage * raw_b,
                -self.max_log_adjustment,
                self.max_log_adjustment,
            )
        )
        details = {
            "coverage_a": coverage_a,
            "coverage_b": coverage_b,
            "raw_match_coverage_a": float(
                np.clip(profile_a.get("match_coverage", 0.0), 0.0, 1.0)
            ),
            "raw_match_coverage_b": float(
                np.clip(profile_b.get("match_coverage", 0.0), 0.0, 1.0)
            ),
            "pair_coverage_shrink": pair_coverage,
            "attack_edge_a": attack_edge_a,
            "attack_edge_b": attack_edge_b,
            "midfield_edge_a": midfield_edge_a,
            "keeper_edge_a": keeper_edge_a,
            "raw_log_adjustment_a": raw_a,
            "raw_log_adjustment_b": raw_b,
            "applied_log_adjustment_a": log_a,
            "applied_log_adjustment_b": log_b,
        }
        return log_a, log_b, details

    def predict(self, *args, **kwargs) -> Dict[str, Any]:
        team_a = kwargs.get("team_a", args[0] if args else "")
        team_b = kwargs.get("team_b", args[1] if len(args) > 1 else "")
        max_goals = int(kwargs.get("max_goals", args[5] if len(args) > 5 else 10))
        prediction = self.base_model.predict(*args, **kwargs)
        base_lambda_a = float(prediction["lambda_a"])
        base_lambda_b = float(prediction["lambda_b"])
        base_results = dict(prediction["result_probabilities"])

        profile_a = self.profile_for_team(str(team_a))
        profile_b = self.profile_for_team(str(team_b))
        log_a, log_b, adjustment_details = self._log_adjustments(
            profile_a,
            profile_b,
        )
        lambda_a = float(np.clip(base_lambda_a * math.exp(log_a), 0.15, 4.5))
        lambda_b = float(np.clip(base_lambda_b * math.exp(log_b), 0.15, 4.5))
        score_matrix = v11.poisson_score_matrix(lambda_a, lambda_b, max_goals)
        rho = prediction.get("calibration_notes", {}).get(
            "dixon_coles_rho",
            -0.08,
        )
        score_matrix = v11.apply_dixon_coles_adjustment(
            score_matrix,
            lambda_a,
            lambda_b,
            rho=rho,
        )
        adjusted_results = result_probs_from_score_matrix(score_matrix)
        if self.player_ratings_affect_wdl:
            result_probabilities = blend_result_probabilities(
                base_results,
                adjusted_results,
                self.result_blend_weight,
            )
        else:
            result_probabilities = base_results
        score_matrix = v11.reweight_score_matrix_to_results(
            score_matrix,
            result_probabilities,
        )

        prediction["lambda_a"] = lambda_a
        prediction["lambda_b"] = lambda_b
        prediction.update(v15.score_outputs(score_matrix, max_goals))
        prediction["result_probabilities"] = result_probabilities
        if self.player_ratings_affect_wdl:
            prediction["predicted_result"] = max(
                result_probabilities,
                key=result_probabilities.get,
            )
        prediction["v18_squad_profiles"] = {
            "team_a": profile_a,
            "team_b": profile_b,
        }
        prediction["v18_adjustments"] = {
            "base_model": "v15_catboost",
            "ratings_source": "fcratings_top50_worldcup2026",
            "squad_name_source": "player_ratings_international",
            "squad_file_rating_values_used": False,
            "player_ratings_affect_wdl": self.player_ratings_affect_wdl,
            "player_ratings_affect_scorelines": True,
            "result_blend_weight": self.result_blend_weight,
            "beta_attack": self.beta_attack,
            "beta_midfield": self.beta_midfield,
            "beta_keeper": self.beta_keeper,
            "max_log_adjustment": self.max_log_adjustment,
            "base_lambda_a": base_lambda_a,
            "base_lambda_b": base_lambda_b,
            "adjusted_poisson_result_probabilities": adjusted_results,
            "base_result_probabilities": base_results,
            **adjustment_details,
        }
        prediction["calibration_notes"] = {
            **prediction.get("calibration_notes", {}),
            "v18": prediction["v18_adjustments"],
            "hybrid_model_policy": (
                "V18 uses V15 CatBoost as the base model, then applies a "
                "tempered, confidence-shrunk FC Ratings squad-strength "
                "adjustment on the log-goal scale. The squad file supplies "
                "player names and positions only; numeric player ratings come "
                "from FC Ratings. Unmatched squad players are imputed from "
                "shrunken positional FC Ratings baselines. By default the "
                "squad adjustment changes expected goals and exact-score "
                "probabilities only, while W/D/L probabilities and the result "
                "decision stay with V15."
            ),
        }
        return prediction

    def update_after_match(
        self,
        team_a: str,
        team_b: str,
        goals_a: int,
        goals_b: int,
    ) -> Dict[str, float]:
        return self.base_model.update_after_match(
            team_a,
            team_b,
            goals_a,
            goals_b,
        )


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
    beta_attack=DEFAULT_BETA_ATTACK,
    beta_midfield=DEFAULT_BETA_MIDFIELD,
    beta_keeper=DEFAULT_BETA_KEEPER,
    max_log_adjustment=DEFAULT_MAX_LOG_ADJUSTMENT,
    result_blend_weight=DEFAULT_RESULT_BLEND_WEIGHT,
    player_ratings_affect_wdl=DEFAULT_PLAYER_RATINGS_AFFECT_WDL,
    match_threshold=0.84,
):
    data_dir = Path(__file__).resolve().parent / "data"
    player_ratings_csv = player_ratings_csv or (
        data_dir / "player_ratings_international.csv"
    )
    fcratings_csv = fcratings_csv or (data_dir / "fcratings_top50_worldcup2026.csv")
    base_model, data = v15.build_from_zip(
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
        results_as_of=results_as_of,
    )
    squad_names = load_current_squad_names(player_ratings_csv)
    fcratings = load_fcratings_players(fcratings_csv)
    squad_profiles = build_current_fcratings_squad_profiles(
        squad_names,
        fcratings,
        match_threshold=match_threshold,
    )
    model = V18HybridSquadModel(
        base_model,
        squad_profiles,
        beta_attack=beta_attack,
        beta_midfield=beta_midfield,
        beta_keeper=beta_keeper,
        max_log_adjustment=max_log_adjustment,
        result_blend_weight=result_blend_weight,
        player_ratings_affect_wdl=player_ratings_affect_wdl,
    )
    model.training_data_summary = {
        **getattr(base_model, "training_data_summary", {}),
        "v18_squad_profile_teams": len(squad_profiles),
        "v18_fcratings_rows": int(len(fcratings)),
        "v18_squad_name_rows": int(len(squad_names)),
    }
    return model, data


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    parser = argparse.ArgumentParser(
        description="Run V18: V15 CatBoost with FC Ratings squad adjustment."
    )
    parser.add_argument("--team-a", required=True)
    parser.add_argument("--team-b", required=True)
    parser.add_argument("--host-a", action="store_true")
    parser.add_argument("--host-b", action="store_true")
    parser.add_argument("--knockout", action="store_true")
    parser.add_argument(
        "--outdir",
        default="outputs/outputs_v18_hybrid_prediction",
    )
    parser.add_argument(
        "--worldcupsai-zip",
        default=str(data_dir / "worldcupsai.zip"),
    )
    parser.add_argument(
        "--team-train",
        default=str(data_dir / "current_team_features_2026.csv"),
    )
    parser.add_argument("--team-test")
    parser.add_argument(
        "--box-data",
        default=str(data_dir / "FIFAallMatchBoxData.csv"),
    )
    parser.add_argument(
        "--results-data",
        default=str(data_dir / "results.csv"),
    )
    parser.add_argument("--results-as-of", default=v15.DEFAULT_RESULTS_AS_OF)
    parser.add_argument(
        "--former-names",
        default=str(data_dir / "former_names.csv"),
    )
    parser.add_argument("--prediction-year", type=int, default=2026)
    parser.add_argument(
        "--player-ratings",
        default=str(data_dir / "player_ratings_international.csv"),
        help="Used for current squad player names/positions only in V18.",
    )
    parser.add_argument(
        "--declared-squads",
        default=str(data_dir / "world_cup_2026_declared_squads.csv"),
        help="Passed through to the V15 base model.",
    )
    parser.add_argument(
        "--fcratings",
        default=str(data_dir / "fcratings_top50_worldcup2026.csv"),
    )
    parser.add_argument("--beta-attack", type=float, default=DEFAULT_BETA_ATTACK)
    parser.add_argument(
        "--beta-midfield",
        type=float,
        default=DEFAULT_BETA_MIDFIELD,
    )
    parser.add_argument("--beta-keeper", type=float, default=DEFAULT_BETA_KEEPER)
    parser.add_argument(
        "--max-log-adjustment",
        type=float,
        default=DEFAULT_MAX_LOG_ADJUSTMENT,
    )
    parser.add_argument(
        "--result-blend-weight",
        type=float,
        default=DEFAULT_RESULT_BLEND_WEIGHT,
    )
    parser.add_argument(
        "--player-ratings-affect-wdl",
        action="store_true",
        default=DEFAULT_PLAYER_RATINGS_AFFECT_WDL,
        help=(
            "Let the FC Ratings squad layer blend W/D/L probabilities. By "
            "default, player ratings affect lambdas and exact scorelines only."
        ),
    )
    parser.add_argument("--match-threshold", type=float, default=0.84)
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
        beta_attack=args.beta_attack,
        beta_midfield=args.beta_midfield,
        beta_keeper=args.beta_keeper,
        max_log_adjustment=args.max_log_adjustment,
        result_blend_weight=args.result_blend_weight,
        player_ratings_affect_wdl=args.player_ratings_affect_wdl,
        match_threshold=args.match_threshold,
    )
    prediction = model.predict(
        args.team_a,
        args.team_b,
        host_a=args.host_a,
        host_b=args.host_b,
        knockout=args.knockout,
    )
    (output_dir / "single_match_prediction.json").write_text(
        json.dumps(prediction, indent=2)
    )
    (output_dir / "v18_squad_profiles.json").write_text(
        json.dumps(prediction["v18_squad_profiles"], indent=2)
    )
    pd.DataFrame(prediction["top_scorelines"]).to_csv(
        output_dir / "scoreline_probabilities_top.csv",
        index=False,
    )
    pd.DataFrame(prediction["scoreline_probabilities"]).to_csv(
        output_dir / "scoreline_probabilities.csv",
        index=False,
    )
    (output_dir / "model_summary.json").write_text(
        json.dumps(
            {
                "version": "v18-hybrid-elo-form-fcratings",
                "base_model": "v15_catboost",
                "team_a": prediction["team_a"],
                "team_b": prediction["team_b"],
                "player_ratings_source": args.fcratings,
                "squad_name_source": args.player_ratings,
                "squad_file_rating_values_used": False,
                "expanded_training_data": model.training_data_summary,
                "v18_adjustments": prediction["v18_adjustments"],
                "coverage": {
                    "team_a": prediction["v18_adjustments"]["coverage_a"],
                    "team_b": prediction["v18_adjustments"]["coverage_b"],
                },
            },
            indent=2,
        )
    )
    if not args.no_plots:
        v11.plot_prediction_outputs(prediction, output_dir)
    print(
        json.dumps(
            {
                "result_probabilities": prediction["result_probabilities"],
                "predicted_result": prediction["predicted_result"],
                "lambda_a": prediction["lambda_a"],
                "lambda_b": prediction["lambda_b"],
                "top_scorelines": prediction["top_scorelines"][:5],
                "v18_adjustments": prediction["v18_adjustments"],
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

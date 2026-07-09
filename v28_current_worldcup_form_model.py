#!/usr/bin/env python3
"""V28: V20 plus current World Cup form, FotMob signals, calibration, and Top-3 selection.

Run:
    .venv/bin/python v28_current_worldcup_form_model.py --team-a "Argentina" --team-b "France" --no-plots
    .venv/bin/python v28_current_worldcup_form_model.py --team-a "Argentina" --team-b "France" --blind-current-goals --no-plots
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

import v11_wcq_results_model as v11
import v15_catboost_model as v15
import v20_scoreline_ensemble_model as v20
import v26_top3_coverage_model as v26
import v27_total_goals_calibrated_model as v27


ScoreMatrix = Dict[Tuple[int, int], float]

DEFAULT_CURRENT_WDL_BLEND = 0.25
DEFAULT_CURRENT_SCORELINE_BLEND = 0.65
DEFAULT_BETA_ATTACK_EDGE = 0.10
DEFAULT_BETA_TEMPO = 0.035
DEFAULT_BETA_GROUP_PRESSURE = 0.025
DEFAULT_MAX_LOG_ADJUSTMENT = 0.22
DEFAULT_TOTAL_CALIBRATION_BLEND = 0.25
DEFAULT_TAIL_RELATIVE_FLOOR = 0.45

POSITIVE_ATTACK_STATS = {
    "Top scorer",
    "Goals per 90",
    "Goals per match",
    "Goals + Assists",
    "Expected goals (xG)",
    "xG per 90",
    "Expected goals on target (xGOT)",
    "Shots per 90",
    "Shots per match",
    "Shots on target per 90",
    "Shots on target per match",
    "Big chances",
    "Touches in opposition box",
    "Set piece goals",
    "Penalties awarded",
}
GOAL_DERIVED_FOTMOB_STATS = {
    "Top scorer",
    "Goals per 90",
    "Goals per match",
    "Goals + Assists",
    "Set piece goals",
    "Clean sheets",
    "Goals conceded per 90",
    "Goals conceded per match",
    "Set piece goals conceded",
}
POSITIVE_CREATION_STATS = {
    "Assists",
    "Expected assists (xA)",
    "xA per 90",
    "xG + xA per 90",
    "Chances created",
    "Big chances created",
    "Accurate crosses per match",
    "Accurate long balls per 90",
    "Accurate long balls per match",
    "Accurate passes per 90",
    "Accurate passes per match",
    "Successful dribbles per 90",
    "Average possession",
}
POSITIVE_DEFENSE_STATS = {
    "Clean sheets",
    "Defensive contributions per 90",
    "Tackles per 90",
    "Tackles per match",
    "Interceptions per 90",
    "Interceptions per match",
    "Clearances per 90",
    "Clearances per match",
    "Blocks per 90",
    "Recoveries per 90",
    "Possession won final 3rd per 90",
    "Possession won final 3rd per match",
}
POSITIVE_KEEPER_STATS = {
    "Save percentage",
    "Goals prevented",
    "Saves per 90",
    "Saves per match",
}
NEGATIVE_DEFENSE_STATS = {
    "Goals conceded per 90",
    "Goals conceded per match",
    "xG conceded",
    "Set piece goals conceded",
    "Penalties conceded",
}
NEGATIVE_ATTACK_STATS = {
    "Big chances missed",
}
NEGATIVE_DISCIPLINE_STATS = {
    "Yellow cards",
    "Red cards",
    "Fouls committed per 90",
    "Fouls per match",
}

TEAM_ALIASES = {
    "usa": "United States",
    "united states": "United States",
    "united states of america": "United States",
    "south korea": "Korea Republic",
    "korea republic": "Korea Republic",
    "czech republic": "Czechia",
    "turkiye": "Turkey",
    "türkiye": "Turkey",
    "curacao": "Curaçao",
    "cape verde islands": "Cape Verde",
    "cape verde": "Cape Verde",
}


@dataclass
class CurrentTeamForm:
    team: str
    matches: float = 0.0
    observed_attack: float = 0.0
    observed_defense: float = 0.0
    observed_tempo: float = 0.0
    observed_discipline_risk: float = 0.0
    fotmob_attack: float = 0.0
    fotmob_defense: float = 0.0
    fotmob_tempo: float = 0.0
    fotmob_discipline_risk: float = 0.0
    fotmob_rows: int = 0
    group_pressure: float = 0.0
    group_points: float = 0.0
    group_goal_difference: float = 0.0
    group_matches: float = 0.0

    @property
    def attack(self) -> float:
        return float(0.70 * self.observed_attack + 0.30 * self.fotmob_attack)

    @property
    def defense(self) -> float:
        return float(0.70 * self.observed_defense + 0.30 * self.fotmob_defense)

    @property
    def tempo(self) -> float:
        return float(0.75 * self.observed_tempo + 0.25 * self.fotmob_tempo)

    @property
    def discipline_risk(self) -> float:
        return float(
            0.70 * self.observed_discipline_risk
            + 0.30 * self.fotmob_discipline_risk
        )

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "team": self.team,
            "matches": float(self.matches),
            "attack": self.attack,
            "defense": self.defense,
            "tempo": self.tempo,
            "discipline_risk": self.discipline_risk,
            "observed_attack": self.observed_attack,
            "observed_defense": self.observed_defense,
            "observed_tempo": self.observed_tempo,
            "observed_discipline_risk": self.observed_discipline_risk,
            "fotmob_attack": self.fotmob_attack,
            "fotmob_defense": self.fotmob_defense,
            "fotmob_tempo": self.fotmob_tempo,
            "fotmob_discipline_risk": self.fotmob_discipline_risk,
            "fotmob_rows": int(self.fotmob_rows),
            "group_pressure": self.group_pressure,
            "group_points": self.group_points,
            "group_goal_difference": self.group_goal_difference,
            "group_matches": self.group_matches,
        }


def canon_team(name: object) -> str:
    if pd.isna(name):
        return ""
    text = str(name).strip()
    key = text.lower().replace("&", "and")
    return TEAM_ALIASES.get(key, v11.canon_team(text))


def parse_number(value: object) -> float:
    if pd.isna(value):
        return 0.0
    text = str(value).strip().replace(",", "")
    if not text:
        return 0.0
    is_percent = text.endswith("%")
    text = text[:-1] if is_percent else text
    try:
        number = float(text)
    except ValueError:
        return 0.0
    return number / 100.0 if is_percent else number


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    std = float(values.std(ddof=0))
    if std <= 1e-12:
        return pd.Series(0.0, index=series.index)
    return ((values - float(values.mean())) / std).clip(-2.5, 2.5)


def shrink_by_matches(values: pd.Series, matches: pd.Series, prior_matches: float) -> pd.Series:
    weights = np.sqrt(
        pd.to_numeric(matches, errors="coerce").fillna(0.0)
        / (
            pd.to_numeric(matches, errors="coerce").fillna(0.0)
            + float(prior_matches)
        ).clip(lower=1e-9)
    )
    return values * weights


def normalized_stat_contribution(frame: pd.DataFrame, stat: str) -> pd.Series:
    stat_frame = frame.loc[frame["stat"].eq(stat)].copy()
    if stat_frame.empty:
        return pd.Series(dtype=float)
    values = pd.to_numeric(stat_frame["numeric_value"], errors="coerce").fillna(0.0)
    max_value = max(float(values.abs().max()), 1e-9)
    rank = pd.to_numeric(stat_frame.get("rank", 6), errors="coerce").fillna(6.0)
    rank_weight = ((7.0 - rank.clip(lower=1.0, upper=6.0)) / 6.0).clip(0.20, 1.0)
    return (values / max_value) * rank_weight


def build_observed_form(
    path: str | Path | None,
    include_goals: bool = True,
) -> pd.DataFrame:
    if not path or not Path(path).exists():
        return pd.DataFrame(columns=["team"])

    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame(columns=["team"])

    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        for side, opp_side in (("a", "b"), ("b", "a")):
            team = canon_team(getattr(row, f"team_{side}"))
            if not team:
                continue
            rows.append(
                {
                    "team": team,
                    "group": getattr(row, "group", ""),
                    "goals_for": parse_number(getattr(row, f"goals_{side}")),
                    "goals_against": parse_number(getattr(row, f"goals_{opp_side}")),
                    "shots_for": parse_number(getattr(row, f"shots_{side}", 0)),
                    "shots_against": parse_number(getattr(row, f"shots_{opp_side}", 0)),
                    "sot_for": parse_number(getattr(row, f"shots_on_target_{side}", 0)),
                    "sot_against": parse_number(
                        getattr(row, f"shots_on_target_{opp_side}", 0)
                    ),
                    "corners_for": parse_number(getattr(row, f"corners_{side}", 0)),
                    "corners_against": parse_number(
                        getattr(row, f"corners_{opp_side}", 0)
                    ),
                    "possession": parse_number(
                        getattr(row, f"possession_{side}_pct", 0)
                    ),
                    "pass_accuracy": parse_number(
                        getattr(row, f"pass_accuracy_{side}_pct", 0)
                    ),
                    "fouls": parse_number(getattr(row, f"fouls_{side}", 0)),
                    "yellow_cards": parse_number(
                        getattr(row, f"yellow_cards_{side}", 0)
                    ),
                    "red_cards": parse_number(getattr(row, f"red_cards_{side}", 0)),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["team"])

    team_rows = pd.DataFrame(rows)
    grouped = (
        team_rows.groupby("team", as_index=False)
        .agg(
            matches=("team", "size"),
            goals_for=("goals_for", "mean"),
            goals_against=("goals_against", "mean"),
            shots_for=("shots_for", "mean"),
            shots_against=("shots_against", "mean"),
            sot_for=("sot_for", "mean"),
            sot_against=("sot_against", "mean"),
            corners_for=("corners_for", "mean"),
            corners_against=("corners_against", "mean"),
            possession=("possession", "mean"),
            pass_accuracy=("pass_accuracy", "mean"),
            fouls=("fouls", "mean"),
            yellow_cards=("yellow_cards", "mean"),
            red_cards=("red_cards", "mean"),
        )
        .copy()
    )

    if include_goals:
        grouped["attack_raw"] = (
            0.38 * zscore(grouped["goals_for"])
            + 0.28 * zscore(grouped["sot_for"])
            + 0.20 * zscore(grouped["shots_for"])
            + 0.14 * zscore(grouped["corners_for"])
        )
        grouped["defense_raw"] = -(
            0.40 * zscore(grouped["goals_against"])
            + 0.30 * zscore(grouped["sot_against"])
            + 0.20 * zscore(grouped["shots_against"])
            + 0.10 * zscore(grouped["corners_against"])
        )
    else:
        grouped["attack_raw"] = (
            0.45 * zscore(grouped["sot_for"])
            + 0.35 * zscore(grouped["shots_for"])
            + 0.20 * zscore(grouped["corners_for"])
        )
        grouped["defense_raw"] = -(
            0.45 * zscore(grouped["sot_against"])
            + 0.35 * zscore(grouped["shots_against"])
            + 0.20 * zscore(grouped["corners_against"])
        )
    grouped["tempo_raw"] = (
        0.45 * zscore(grouped["shots_for"] + grouped["shots_against"])
        + 0.25 * zscore(grouped["corners_for"] + grouped["corners_against"])
        + 0.20 * zscore(grouped["possession"])
        + 0.10 * zscore(grouped["pass_accuracy"])
    )
    grouped["discipline_raw"] = (
        0.40 * zscore(grouped["fouls"])
        + 0.30 * zscore(grouped["yellow_cards"])
        + 0.30 * zscore(grouped["red_cards"])
    )

    grouped["observed_attack"] = shrink_by_matches(
        grouped["attack_raw"], grouped["matches"], prior_matches=2.0
    )
    grouped["observed_defense"] = shrink_by_matches(
        grouped["defense_raw"], grouped["matches"], prior_matches=2.0
    )
    grouped["observed_tempo"] = shrink_by_matches(
        grouped["tempo_raw"], grouped["matches"], prior_matches=2.0
    )
    grouped["observed_discipline_risk"] = shrink_by_matches(
        grouped["discipline_raw"], grouped["matches"], prior_matches=2.0
    )
    return grouped[
        [
            "team",
            "matches",
            "observed_attack",
            "observed_defense",
            "observed_tempo",
            "observed_discipline_risk",
        ]
    ]


def build_group_context(
    path: str | Path | None,
    include_score_context: bool = True,
) -> pd.DataFrame:
    if not include_score_context:
        return pd.DataFrame(columns=["team"])
    if not path or not Path(path).exists():
        return pd.DataFrame(columns=["team"])
    frame = pd.read_csv(path)
    if frame.empty or "group" not in frame:
        return pd.DataFrame(columns=["team"])

    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        goals_a = int(parse_number(getattr(row, "goals_a", 0)))
        goals_b = int(parse_number(getattr(row, "goals_b", 0)))
        if goals_a > goals_b:
            points_a, points_b = 3, 0
        elif goals_a == goals_b:
            points_a, points_b = 1, 1
        else:
            points_a, points_b = 0, 3
        rows.extend(
            [
                {
                    "team": canon_team(getattr(row, "team_a")),
                    "group": getattr(row, "group", ""),
                    "points": points_a,
                    "goal_difference": goals_a - goals_b,
                    "goals_for": goals_a,
                },
                {
                    "team": canon_team(getattr(row, "team_b")),
                    "group": getattr(row, "group", ""),
                    "points": points_b,
                    "goal_difference": goals_b - goals_a,
                    "goals_for": goals_b,
                },
            ]
        )
    if not rows:
        return pd.DataFrame(columns=["team"])
    standings = (
        pd.DataFrame(rows)
        .groupby("team", as_index=False)
        .agg(
            group=("group", "last"),
            group_matches=("team", "size"),
            group_points=("points", "sum"),
            group_goal_difference=("goal_difference", "sum"),
            group_goals_for=("goals_for", "sum"),
        )
    )

    def pressure(row: pd.Series) -> float:
        matches = float(row["group_matches"])
        points = float(row["group_points"])
        goal_difference = float(row["group_goal_difference"])
        if matches >= 3:
            return 0.0
        if points <= 1:
            return 0.55
        if points <= 3 and goal_difference <= 0:
            return 0.35
        if points >= 6:
            return -0.25
        return 0.10

    standings["group_pressure"] = standings.apply(pressure, axis=1)
    return standings[
        [
            "team",
            "group_matches",
            "group_points",
            "group_goal_difference",
            "group_pressure",
        ]
    ]


def fotmob_stat_bucket(stat: str) -> tuple[str | None, float]:
    if stat in POSITIVE_ATTACK_STATS:
        return "attack", 1.0
    if stat in POSITIVE_CREATION_STATS:
        return "attack", 0.65
    if stat in POSITIVE_DEFENSE_STATS:
        return "defense", 0.75
    if stat in POSITIVE_KEEPER_STATS:
        return "defense", 0.85
    if stat in NEGATIVE_DEFENSE_STATS:
        return "defense", -0.85
    if stat in NEGATIVE_ATTACK_STATS:
        return "attack", -0.55
    if stat in NEGATIVE_DISCIPLINE_STATS:
        return "discipline", 1.0
    if stat == "FotMob rating":
        return "attack", 0.35
    return None, 0.0


def build_fotmob_form(
    path: str | Path | None,
    include_goal_stats: bool = True,
) -> pd.DataFrame:
    if not path or not Path(path).exists():
        return pd.DataFrame(columns=["team"])
    frame = pd.read_csv(path)
    required = {"stat", "country_or_team", "value"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=["team"])

    clean = frame.copy()
    clean["team"] = clean["country_or_team"].map(canon_team)
    clean = clean[clean["team"].ne("")]
    if clean.empty:
        return pd.DataFrame(columns=["team"])
    clean["numeric_value"] = clean["value"].map(parse_number)

    records: list[dict[str, Any]] = []
    for stat in sorted(clean["stat"].dropna().unique()):
        if not include_goal_stats and str(stat) in GOAL_DERIVED_FOTMOB_STATS:
            continue
        bucket, sign = fotmob_stat_bucket(str(stat))
        if bucket is None:
            continue
        contributions = normalized_stat_contribution(clean, str(stat))
        if contributions.empty:
            continue
        stat_frame = clean.loc[contributions.index]
        for idx, contribution in contributions.items():
            records.append(
                {
                    "team": stat_frame.loc[idx, "team"],
                    "bucket": bucket,
                    "contribution": float(contribution) * float(sign),
                }
            )

    if not records:
        return pd.DataFrame(columns=["team"])
    contrib = pd.DataFrame(records)
    pivot = contrib.pivot_table(
        index="team",
        columns="bucket",
        values="contribution",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()
    for column in ("attack", "defense", "discipline"):
        if column not in pivot:
            pivot[column] = 0.0

    counts = contrib.groupby("team").size().rename("fotmob_rows").reset_index()
    pivot = pivot.merge(counts, on="team", how="left")
    coverage_weight = np.sqrt(
        pivot["fotmob_rows"].clip(lower=0) / (pivot["fotmob_rows"].clip(lower=0) + 4.0)
    )
    pivot["fotmob_attack"] = zscore(pivot["attack"]) * coverage_weight
    pivot["fotmob_defense"] = zscore(pivot["defense"]) * coverage_weight
    pivot["fotmob_tempo"] = zscore(pivot["attack"].abs() + pivot["defense"].abs()) * coverage_weight
    pivot["fotmob_discipline_risk"] = zscore(pivot["discipline"]) * coverage_weight
    return pivot[
        [
            "team",
            "fotmob_attack",
            "fotmob_defense",
            "fotmob_tempo",
            "fotmob_discipline_risk",
            "fotmob_rows",
        ]
    ]


def merge_current_forms(
    observed: pd.DataFrame,
    fotmob: pd.DataFrame,
    group_context: pd.DataFrame,
) -> dict[str, CurrentTeamForm]:
    teams = set()
    for frame in (observed, fotmob, group_context):
        if "team" in frame:
            teams.update(str(team) for team in frame["team"].dropna() if str(team))

    observed_map = observed.set_index("team").to_dict("index") if not observed.empty else {}
    fotmob_map = fotmob.set_index("team").to_dict("index") if not fotmob.empty else {}
    group_map = (
        group_context.set_index("team").to_dict("index")
        if not group_context.empty
        else {}
    )
    forms: dict[str, CurrentTeamForm] = {}
    for team in sorted(teams):
        obs = observed_map.get(team, {})
        fot = fotmob_map.get(team, {})
        group = group_map.get(team, {})
        forms[team] = CurrentTeamForm(
            team=team,
            matches=float(obs.get("matches", 0.0) or 0.0),
            observed_attack=float(obs.get("observed_attack", 0.0) or 0.0),
            observed_defense=float(obs.get("observed_defense", 0.0) or 0.0),
            observed_tempo=float(obs.get("observed_tempo", 0.0) or 0.0),
            observed_discipline_risk=float(
                obs.get("observed_discipline_risk", 0.0) or 0.0
            ),
            fotmob_attack=float(fot.get("fotmob_attack", 0.0) or 0.0),
            fotmob_defense=float(fot.get("fotmob_defense", 0.0) or 0.0),
            fotmob_tempo=float(fot.get("fotmob_tempo", 0.0) or 0.0),
            fotmob_discipline_risk=float(
                fot.get("fotmob_discipline_risk", 0.0) or 0.0
            ),
            fotmob_rows=int(fot.get("fotmob_rows", 0) or 0),
            group_pressure=float(group.get("group_pressure", 0.0) or 0.0),
            group_points=float(group.get("group_points", 0.0) or 0.0),
            group_goal_difference=float(
                group.get("group_goal_difference", 0.0) or 0.0
            ),
            group_matches=float(group.get("group_matches", 0.0) or 0.0),
        )
    return forms


def score_matrix_from_prediction(prediction: Dict[str, Any]) -> ScoreMatrix:
    return {
        (int(item["team_a_goals"]), int(item["team_b_goals"])): float(
            item["probability"]
        )
        for item in prediction["scoreline_probabilities"]
    }


def blend_result_probabilities(
    base: Dict[str, float],
    adjusted: Dict[str, float],
    adjusted_weight: float,
) -> Dict[str, float]:
    weight = float(np.clip(adjusted_weight, 0.0, 1.0))
    blended = {
        label: (1.0 - weight) * float(base.get(label, 0.0))
        + weight * float(adjusted.get(label, 0.0))
        for label in ("team_a_win", "draw", "team_b_win")
    }
    total = max(float(sum(blended.values())), 1e-12)
    return {label: float(value) / total for label, value in blended.items()}


def expected_goals(score_matrix: ScoreMatrix) -> tuple[float, float]:
    return (
        float(sum(goals_a * prob for (goals_a, _), prob in score_matrix.items())),
        float(sum(goals_b * prob for (_, goals_b), prob in score_matrix.items())),
    )


def plot_top3_scorelines(prediction: Dict[str, Any], outdir: Path) -> Path:
    v11._require_matplotlib()
    outdir.mkdir(parents=True, exist_ok=True)
    top = prediction["top_scorelines"][:3][::-1]
    labels = [
        f"{row['team_a_goals']}-{row['team_b_goals']}"
        for row in top
    ]
    values = [float(row["probability"]) for row in top]
    colors = ["#7c3aed", "#2563eb", "#059669"][-len(labels):]
    fig, ax = v11.plt.subplots(figsize=(7.5, 3.8))
    bars = ax.barh(
        labels,
        values,
        color=colors,
        alpha=0.88,
        edgecolor="black",
        linewidth=1.2,
    )
    ax.set_xlabel("Probability", fontsize=11)
    ax.set_title(
        f"Top 3 Exact Scores: {prediction['team_a']} vs {prediction['team_b']}",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlim(0, max(values) * 1.22 if values else 1.0)
    for bar, value in zip(bars, values):
        ax.text(
            value + max(values) * 0.025,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1%}",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
    return v11._save_plot(fig, outdir / "top_3_scorelines.png")


class V28CurrentWorldCupFormModel:
    """Wrap V20 with current tournament attack-vs-defense adjustments."""

    def __init__(
        self,
        base_model: v20.V20ScorelineEnsembleModel,
        current_forms: dict[str, CurrentTeamForm],
        calibration_model: v27.TotalGoalsCalibrationModel,
        current_wdl_blend: float = DEFAULT_CURRENT_WDL_BLEND,
        current_scoreline_blend: float = DEFAULT_CURRENT_SCORELINE_BLEND,
        beta_attack_edge: float = DEFAULT_BETA_ATTACK_EDGE,
        beta_tempo: float = DEFAULT_BETA_TEMPO,
        beta_group_pressure: float = DEFAULT_BETA_GROUP_PRESSURE,
        max_log_adjustment: float = DEFAULT_MAX_LOG_ADJUSTMENT,
        total_calibration_blend: float = DEFAULT_TOTAL_CALIBRATION_BLEND,
        tail_relative_floor: float = DEFAULT_TAIL_RELATIVE_FLOOR,
        favorite_win_gate: float = v26.DEFAULT_FAVORITE_WIN_GATE,
        total_lambda_gate: float = v26.DEFAULT_TOTAL_LAMBDA_GATE,
        favorite_lambda_gate: float = v26.DEFAULT_FAVORITE_LAMBDA_GATE,
        draw_ceiling: float = v26.DEFAULT_DRAW_CEILING,
    ):
        self.base_model = base_model
        self.current_forms = current_forms
        self.calibration_model = calibration_model
        self.current_wdl_blend = float(np.clip(current_wdl_blend, 0.0, 1.0))
        self.current_scoreline_blend = float(np.clip(current_scoreline_blend, 0.0, 1.0))
        self.beta_attack_edge = float(beta_attack_edge)
        self.beta_tempo = float(beta_tempo)
        self.beta_group_pressure = float(beta_group_pressure)
        self.max_log_adjustment = float(max(max_log_adjustment, 0.0))
        self.total_calibration_blend = float(np.clip(total_calibration_blend, 0.0, 1.0))
        self.tail_relative_floor = float(max(tail_relative_floor, 0.0))
        self.favorite_win_gate = float(np.clip(favorite_win_gate, 0.0, 1.0))
        self.total_lambda_gate = float(max(total_lambda_gate, 0.0))
        self.favorite_lambda_gate = float(max(favorite_lambda_gate, 0.0))
        self.draw_ceiling = float(np.clip(draw_ceiling, 0.0, 1.0))
        self.training_data_summary = getattr(base_model, "training_data_summary", {})

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_model, name)

    def form_for_team(self, team: object) -> CurrentTeamForm:
        canonical = canon_team(team)
        return self.current_forms.get(canonical, CurrentTeamForm(team=canonical))

    def _current_adjusted_matrix(
        self,
        prediction: Dict[str, Any],
        form_a: CurrentTeamForm,
        form_b: CurrentTeamForm,
        knockout: bool,
        max_goals: int,
    ) -> tuple[ScoreMatrix, Dict[str, Any]]:
        base_lambda_a = float(prediction["lambda_a"])
        base_lambda_b = float(prediction["lambda_b"])

        attack_edge_a = (
            form_a.attack
            - form_b.defense
            + 0.15 * form_b.discipline_risk
            - 0.05 * form_a.discipline_risk
        )
        attack_edge_b = (
            form_b.attack
            - form_a.defense
            + 0.15 * form_a.discipline_risk
            - 0.05 * form_b.discipline_risk
        )
        tempo_edge = 0.5 * (form_a.tempo + form_b.tempo)
        pressure_a = 0.20 if knockout else form_a.group_pressure
        pressure_b = 0.20 if knockout else form_b.group_pressure

        log_a = (
            self.beta_attack_edge * attack_edge_a
            + self.beta_tempo * tempo_edge
            + self.beta_group_pressure * pressure_a
        )
        log_b = (
            self.beta_attack_edge * attack_edge_b
            + self.beta_tempo * tempo_edge
            + self.beta_group_pressure * pressure_b
        )
        log_a = float(np.clip(log_a, -self.max_log_adjustment, self.max_log_adjustment))
        log_b = float(np.clip(log_b, -self.max_log_adjustment, self.max_log_adjustment))

        lambda_a = float(np.clip(base_lambda_a * math.exp(log_a), 0.15, 5.25))
        lambda_b = float(np.clip(base_lambda_b * math.exp(log_b), 0.15, 5.25))
        score_matrix = v11.poisson_score_matrix(lambda_a, lambda_b, max_goals)
        rho = prediction.get("calibration_notes", {}).get("dixon_coles_rho", -0.08)
        score_matrix = v11.apply_dixon_coles_adjustment(
            score_matrix,
            lambda_a,
            lambda_b,
            rho=rho,
        )
        return score_matrix, {
            "base_lambda_a": base_lambda_a,
            "base_lambda_b": base_lambda_b,
            "current_lambda_a": lambda_a,
            "current_lambda_b": lambda_b,
            "log_adjustment_a": log_a,
            "log_adjustment_b": log_b,
            "attack_edge_a": float(attack_edge_a),
            "attack_edge_b": float(attack_edge_b),
            "tempo_edge": float(tempo_edge),
            "group_or_knockout_pressure_a": float(pressure_a),
            "group_or_knockout_pressure_b": float(pressure_b),
        }

    def predict(self, *args, **kwargs) -> Dict[str, Any]:
        team_a = kwargs.get("team_a", args[0] if args else "")
        team_b = kwargs.get("team_b", args[1] if len(args) > 1 else "")
        max_goals = int(kwargs.get("max_goals", args[5] if len(args) > 5 else 10))
        knockout = bool(kwargs.get("knockout", False))

        prediction = self.base_model.predict(*args, **kwargs)
        base_matrix = score_matrix_from_prediction(prediction)
        base_result_probabilities = dict(prediction["result_probabilities"])
        form_a = self.form_for_team(team_a)
        form_b = self.form_for_team(team_b)

        current_matrix, current_diagnostics = self._current_adjusted_matrix(
            prediction,
            form_a,
            form_b,
            knockout=knockout,
            max_goals=max_goals,
        )
        current_result_probabilities = v11.result_probs(current_matrix)
        final_result_probabilities = blend_result_probabilities(
            base_result_probabilities,
            current_result_probabilities,
            self.current_wdl_blend,
        )

        score_matrix = v20.blend_score_matrices(
            base_matrix,
            current_matrix,
            adjusted_weight=self.current_scoreline_blend,
        )
        score_matrix = v11.reweight_score_matrix_to_results(
            score_matrix,
            final_result_probabilities,
        )

        calibrated_matrix, calibration_diagnostics = self.calibration_model.apply(
            score_matrix,
            final_result_probabilities,
            float(current_diagnostics["current_lambda_a"])
            + float(current_diagnostics["current_lambda_b"]),
            knockout=knockout,
        )
        score_matrix = v20.blend_score_matrices(
            score_matrix,
            calibrated_matrix,
            adjusted_weight=self.total_calibration_blend,
        )
        score_matrix = v11.reweight_score_matrix_to_results(
            score_matrix,
            final_result_probabilities,
        )

        lambda_a, lambda_b = expected_goals(score_matrix)
        prediction["lambda_a"] = lambda_a
        prediction["lambda_b"] = lambda_b
        prediction["result_probabilities"] = final_result_probabilities
        prediction["predicted_result"] = max(
            final_result_probabilities,
            key=final_result_probabilities.get,
        )
        prediction.update(v15.score_outputs(score_matrix, max_goals))
        top_scorelines, coverage_diagnostics = v26.select_top_scorelines_with_coverage(
            score_matrix,
            final_result_probabilities,
            lambda_a,
            lambda_b,
            top_n=15,
            tail_relative_floor=self.tail_relative_floor,
            favorite_win_gate=self.favorite_win_gate,
            total_lambda_gate=self.total_lambda_gate,
            favorite_lambda_gate=self.favorite_lambda_gate,
            draw_ceiling=self.draw_ceiling,
        )
        prediction["top_scorelines"] = top_scorelines
        prediction["v28_adjustments"] = {
            "base_model": "v20_scoreline_ensemble",
            "scoreline_policy": (
                "current_worldcup_form_plus_total_calibration_plus_top3_selection"
            ),
            "current_layer_affects_wdl": True,
            "current_wdl_blend": self.current_wdl_blend,
            "current_scoreline_blend": self.current_scoreline_blend,
            "total_calibration_blend": self.total_calibration_blend,
            "beta_attack_edge": self.beta_attack_edge,
            "beta_tempo": self.beta_tempo,
            "beta_group_pressure": self.beta_group_pressure,
            "max_log_adjustment": self.max_log_adjustment,
            "team_a_form": form_a.diagnostics(),
            "team_b_form": form_b.diagnostics(),
            "base_result_probabilities": base_result_probabilities,
            "current_result_probabilities": current_result_probabilities,
            **current_diagnostics,
            "total_calibration": calibration_diagnostics,
            "top3_coverage": coverage_diagnostics,
        }
        prediction["calibration_notes"] = {
            **prediction.get("calibration_notes", {}),
            "v28": prediction["v28_adjustments"],
            "current_worldcup_policy": (
                "V28 starts from V20, builds current attack-vs-defense edges "
                "from observed 2026 box-score form and FotMob leaderboards, "
                "allows a light W/D/L blend, applies total-goals calibration, "
                "then uses the V26 Top-3 coverage selector."
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
    include_observed_goals=True,
    include_fotmob_goal_stats=True,
    include_group_score_context=True,
    total_calibration_strength=v27.DEFAULT_TOTAL_CALIBRATION_STRENGTH,
    total_multiplier_clip_low=v27.DEFAULT_MULTIPLIER_CLIP_LOW,
    total_multiplier_clip_high=v27.DEFAULT_MULTIPLIER_CLIP_HIGH,
    total_smoothing=v27.DEFAULT_TOTAL_SMOOTHING,
    min_bin_support=v27.DEFAULT_MIN_BIN_SUPPORT,
    max_train_matches=v27.DEFAULT_MAX_TRAIN_MATCHES,
    current_wdl_blend=DEFAULT_CURRENT_WDL_BLEND,
    current_scoreline_blend=DEFAULT_CURRENT_SCORELINE_BLEND,
    beta_attack_edge=DEFAULT_BETA_ATTACK_EDGE,
    beta_tempo=DEFAULT_BETA_TEMPO,
    beta_group_pressure=DEFAULT_BETA_GROUP_PRESSURE,
    max_log_adjustment=DEFAULT_MAX_LOG_ADJUSTMENT,
    total_calibration_blend=DEFAULT_TOTAL_CALIBRATION_BLEND,
    tail_relative_floor=DEFAULT_TAIL_RELATIVE_FLOOR,
    favorite_win_gate=v26.DEFAULT_FAVORITE_WIN_GATE,
    total_lambda_gate=v26.DEFAULT_TOTAL_LAMBDA_GATE,
    favorite_lambda_gate=v26.DEFAULT_FAVORITE_LAMBDA_GATE,
    draw_ceiling=v26.DEFAULT_DRAW_CEILING,
):
    data_dir = Path(__file__).resolve().parent / "data"
    observed_matches_csv = observed_matches_csv or (
        data_dir / "wc2026_observed_matches_from_screenshots.csv"
    )
    fotmob_leaders_csv = fotmob_leaders_csv or (
        data_dir / "fotmob_stat_leaders_clean.csv"
    )

    base_model, data = v20.build_from_zip(
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
    )
    observed_form = build_observed_form(
        observed_matches_csv,
        include_goals=include_observed_goals,
    )
    fotmob_form = build_fotmob_form(
        fotmob_leaders_csv,
        include_goal_stats=include_fotmob_goal_stats,
    )
    group_context = build_group_context(
        observed_matches_csv,
        include_score_context=include_group_score_context,
    )
    current_forms = merge_current_forms(observed_form, fotmob_form, group_context)
    calibration_model = v27.fit_total_goals_calibration(
        base_model,
        max_goals=10,
        recency_half_life_years=recency_half_life_years,
        recency_min_weight=recency_min_weight,
        strength=total_calibration_strength,
        clip_low=total_multiplier_clip_low,
        clip_high=total_multiplier_clip_high,
        smoothing=total_smoothing,
        min_bin_support=min_bin_support,
        max_train_matches=max_train_matches,
    )
    model = V28CurrentWorldCupFormModel(
        base_model,
        current_forms,
        calibration_model,
        current_wdl_blend=current_wdl_blend,
        current_scoreline_blend=current_scoreline_blend,
        beta_attack_edge=beta_attack_edge,
        beta_tempo=beta_tempo,
        beta_group_pressure=beta_group_pressure,
        max_log_adjustment=max_log_adjustment,
        total_calibration_blend=total_calibration_blend,
        tail_relative_floor=tail_relative_floor,
        favorite_win_gate=favorite_win_gate,
        total_lambda_gate=total_lambda_gate,
        favorite_lambda_gate=favorite_lambda_gate,
        draw_ceiling=draw_ceiling,
    )
    model.training_data_summary = {
        **getattr(base_model, "training_data_summary", {}),
        "v28_current_form_teams": len(current_forms),
        "v28_observed_form_rows": int(len(observed_form)),
        "v28_fotmob_form_rows": int(len(fotmob_form)),
        "v28_group_context_rows": int(len(group_context)),
        "v28_observed_matches_csv": str(observed_matches_csv),
        "v28_fotmob_leaders_csv": str(fotmob_leaders_csv),
        "v28_include_observed_goals": bool(include_observed_goals),
        "v28_include_fotmob_goal_stats": bool(include_fotmob_goal_stats),
        "v28_include_group_score_context": bool(include_group_score_context),
        "v28_current_wdl_blend": model.current_wdl_blend,
        "v28_current_scoreline_blend": model.current_scoreline_blend,
        "v28_total_calibration_blend": model.total_calibration_blend,
        "v28_tail_relative_floor": model.tail_relative_floor,
        "v28_total_calibration_diagnostics": calibration_model.diagnostics(),
    }
    return model, data


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    parser = argparse.ArgumentParser(
        description="Run V28: current World Cup form adjusted scoreline model."
    )
    parser.add_argument("--team-a", required=True)
    parser.add_argument("--team-b", required=True)
    parser.add_argument("--host-a", action="store_true")
    parser.add_argument("--host-b", action="store_true")
    parser.add_argument("--knockout", action="store_true")
    parser.add_argument("--outdir", default="outputs/outputs_v28_current_worldcup_form")
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
    parser.add_argument(
        "--blind-current-goals",
        action="store_true",
        help=(
            "Ablation mode: hide current observed goals, goal-derived group "
            "context, and FotMob goal/clean-sheet leaderboards from v28."
        ),
    )
    parser.add_argument("--ignore-observed-goals", action="store_true")
    parser.add_argument("--ignore-fotmob-goal-stats", action="store_true")
    parser.add_argument("--disable-group-score-context", action="store_true")
    parser.add_argument("--current-wdl-blend", type=float, default=DEFAULT_CURRENT_WDL_BLEND)
    parser.add_argument("--current-scoreline-blend", type=float, default=DEFAULT_CURRENT_SCORELINE_BLEND)
    parser.add_argument("--beta-attack-edge", type=float, default=DEFAULT_BETA_ATTACK_EDGE)
    parser.add_argument("--beta-tempo", type=float, default=DEFAULT_BETA_TEMPO)
    parser.add_argument("--beta-group-pressure", type=float, default=DEFAULT_BETA_GROUP_PRESSURE)
    parser.add_argument("--max-log-adjustment", type=float, default=DEFAULT_MAX_LOG_ADJUSTMENT)
    parser.add_argument("--total-calibration-blend", type=float, default=DEFAULT_TOTAL_CALIBRATION_BLEND)
    parser.add_argument("--total-calibration-strength", type=float, default=v27.DEFAULT_TOTAL_CALIBRATION_STRENGTH)
    parser.add_argument("--total-multiplier-clip-low", type=float, default=v27.DEFAULT_MULTIPLIER_CLIP_LOW)
    parser.add_argument("--total-multiplier-clip-high", type=float, default=v27.DEFAULT_MULTIPLIER_CLIP_HIGH)
    parser.add_argument("--total-smoothing", type=float, default=v27.DEFAULT_TOTAL_SMOOTHING)
    parser.add_argument("--min-bin-support", type=float, default=v27.DEFAULT_MIN_BIN_SUPPORT)
    parser.add_argument("--tail-relative-floor", type=float, default=DEFAULT_TAIL_RELATIVE_FLOOR)
    parser.add_argument("--favorite-win-gate", type=float, default=v26.DEFAULT_FAVORITE_WIN_GATE)
    parser.add_argument("--total-lambda-gate", type=float, default=v26.DEFAULT_TOTAL_LAMBDA_GATE)
    parser.add_argument("--favorite-lambda-gate", type=float, default=v26.DEFAULT_FAVORITE_LAMBDA_GATE)
    parser.add_argument("--draw-ceiling", type=float, default=v26.DEFAULT_DRAW_CEILING)
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
        include_observed_goals=not (
            args.blind_current_goals or args.ignore_observed_goals
        ),
        include_fotmob_goal_stats=not (
            args.blind_current_goals or args.ignore_fotmob_goal_stats
        ),
        include_group_score_context=not (
            args.blind_current_goals or args.disable_group_score_context
        ),
        current_wdl_blend=args.current_wdl_blend,
        current_scoreline_blend=args.current_scoreline_blend,
        beta_attack_edge=args.beta_attack_edge,
        beta_tempo=args.beta_tempo,
        beta_group_pressure=args.beta_group_pressure,
        max_log_adjustment=args.max_log_adjustment,
        total_calibration_blend=args.total_calibration_blend,
        total_calibration_strength=args.total_calibration_strength,
        total_multiplier_clip_low=args.total_multiplier_clip_low,
        total_multiplier_clip_high=args.total_multiplier_clip_high,
        total_smoothing=args.total_smoothing,
        min_bin_support=args.min_bin_support,
        tail_relative_floor=args.tail_relative_floor,
        favorite_win_gate=args.favorite_win_gate,
        total_lambda_gate=args.total_lambda_gate,
        favorite_lambda_gate=args.favorite_lambda_gate,
        draw_ceiling=args.draw_ceiling,
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
    (output_dir / "model_summary.json").write_text(
        json.dumps(
            {
                "version": "v28-current-worldcup-form",
                "base_model": "v20-scoreline-ensemble",
                "team_a": prediction["team_a"],
                "team_b": prediction["team_b"],
                "result_probabilities": prediction["result_probabilities"],
                "v28_adjustments": prediction["v28_adjustments"],
                "expanded_training_data": model.training_data_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not args.no_plots:
        v11.plot_prediction_outputs(prediction, output_dir)
        plot_top3_scorelines(prediction, output_dir / "plots")
    print(
        json.dumps(
            {
                "result_probabilities": prediction["result_probabilities"],
                "predicted_result": prediction["predicted_result"],
                "lambda_a": prediction["lambda_a"],
                "lambda_b": prediction["lambda_b"],
                "top_scorelines": prediction["top_scorelines"][:5],
                "v28_adjustments": {
                    "current_wdl_blend": prediction["v28_adjustments"][
                        "current_wdl_blend"
                    ],
                    "current_scoreline_blend": prediction["v28_adjustments"][
                        "current_scoreline_blend"
                    ],
                    "total_calibration_blend": prediction["v28_adjustments"][
                        "total_calibration_blend"
                    ],
                    "log_adjustment_a": prediction["v28_adjustments"][
                        "log_adjustment_a"
                    ],
                    "log_adjustment_b": prediction["v28_adjustments"][
                        "log_adjustment_b"
                    ],
                    "top3_coverage": prediction["v28_adjustments"]["top3_coverage"],
                },
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

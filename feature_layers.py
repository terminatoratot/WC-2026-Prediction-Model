"""Namespace-isolated module bundle.

Each embedded source block below is one original file's content, byte-for-byte
unchanged. `_load_submodule` executes it into its own `types.ModuleType` and
registers it in `sys.modules` under its original filename-derived name, so
every `import vNN_x as vNN` statement elsewhere in this project keeps working
exactly as it did when these were separate files.
"""
from __future__ import annotations

import sys
import types


def _load_submodule(name: str, source: str, filename: str) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    mod.__file__ = filename
    sys.modules[name] = mod
    exec(compile(source, filename, "exec"), mod.__dict__)
    return mod

import core_engine  # noqa: F401  (loads its own submodules into sys.modules)


# ======================================================================
# v28_current_worldcup_form_model.py  (bundled as an isolated sub-module)
# ======================================================================
_V28_CURRENT_WORLDCUP_FORM_MODEL_SOURCE = r'''
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
'''
v28_current_worldcup_form_model = _load_submodule("v28_current_worldcup_form_model", _V28_CURRENT_WORLDCUP_FORM_MODEL_SOURCE, "feature_layers.py:v28_current_worldcup_form_model")

# ======================================================================
# v29_tail_risk_scoreline_model.py  (bundled as an isolated sub-module)
# ======================================================================
_V29_TAIL_RISK_SCORELINE_MODEL_SOURCE = r'''
#!/usr/bin/env python3
"""V29: V28 plus a conservative blowout/tail-risk Top-3 selector.

Full feature run:
    MPLCONFIGDIR=.matplotlib_cache .venv/bin/python v29_tail_risk_scoreline_model.py --team-a "Argentina" --team-b "France" --knockout --outdir outputs/outputs_v29_argentina_france --observed-matches data/wc2026_observed_matches_from_screenshots.csv --fotmob-leaders data/fotmob_stat_leaders_clean.csv --tail-favorite-win-gate 0.66 --tail-extreme-favorite-win-gate 0.78 --tail-draw-ceiling 0.27 --tail-favorite-lambda-gate 1.75 --tail-extreme-lambda-gate 2.40 --tail-lambda-gap-gate 0.75 --tail-total-lambda-gate 2.45 --tail-selector-relative-floor 0.12 --tail-selector-absolute-floor 0.008 --tail-max-winner-goals 7

All current completed matches:
    .venv/bin/python -c "from pathlib import Path; import v36_fotmob_current_form_model as v36; v36.completed_fotmob_facts_to_observed(Path('data/fotmob_match_facts_clean.csv'), Path('data/fotmob_completed_matches_observed_schema_current.csv'))"
    MPLCONFIGDIR=.matplotlib_cache LOKY_MAX_CPU_COUNT=8 .venv/bin/python eval_v29_v36_completed_worldcup.py --models v29_tail_risk --observed data/fotmob_completed_matches_observed_schema_current.csv --outdir observed_eval/observed_eval_v29_all_current_matches_no_score_leak
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
VERSIONS_DIR = PROJECT_DIR / "versions"
if str(VERSIONS_DIR) not in sys.path:
    sys.path.insert(0, str(VERSIONS_DIR))

import v11_wcq_results_model as v11
import v15_catboost_model as v15
import v26_top3_coverage_model as v26
import v27_total_goals_calibrated_model as v27
import v28_current_worldcup_form_model as v28


ScoreMatrix = Dict[Tuple[int, int], float]

DEFAULT_TAIL_FAVORITE_WIN_GATE = 0.66
DEFAULT_TAIL_EXTREME_FAVORITE_WIN_GATE = 0.78
DEFAULT_TAIL_DRAW_CEILING = 0.27
DEFAULT_TAIL_FAVORITE_LAMBDA_GATE = 1.75
DEFAULT_TAIL_EXTREME_LAMBDA_GATE = 2.40
DEFAULT_TAIL_LAMBDA_GAP_GATE = 0.75
DEFAULT_TAIL_TOTAL_LAMBDA_GATE = 2.45
DEFAULT_TAIL_RELATIVE_FLOOR = 0.12
DEFAULT_TAIL_ABSOLUTE_FLOOR = 0.008
DEFAULT_TAIL_MAX_WINNER_GOALS = 7


def score_matrix_from_prediction(prediction: Dict[str, Any]) -> ScoreMatrix:
    return {
        (int(item["team_a_goals"]), int(item["team_b_goals"])): float(
            item["probability"]
        )
        for item in prediction["scoreline_probabilities"]
    }


def score_item(key: Tuple[int, int], probability: float) -> Dict[str, Any]:
    return {
        "team_a_goals": int(key[0]),
        "team_b_goals": int(key[1]),
        "probability": float(probability),
    }


def is_favorite_tail_score(key: Tuple[int, int], side: str) -> bool:
    goals_a, goals_b = key
    winner_goals = goals_a if side == "team_a" else goals_b
    loser_goals = goals_b if side == "team_a" else goals_a
    margin = winner_goals - loser_goals
    return winner_goals >= 3 and margin >= 2 and loser_goals <= 2


def is_existing_tail_coverage(top_three: list[Tuple[int, int]], side: str) -> bool:
    return any(is_favorite_tail_score(key, side) and max(key) <= 3 for key in top_three)


def tail_candidates(
    score_matrix: ScoreMatrix,
    side: str,
    max_winner_goals: int,
) -> list[Tuple[int, int]]:
    candidates = [
        key
        for key in score_matrix
        if is_favorite_tail_score(key, side)
        and (key[0] if side == "team_a" else key[1]) <= max_winner_goals
    ]
    return sorted(candidates, key=lambda key: score_matrix.get(key, 0.0), reverse=True)


def tail_utility(
    key: Tuple[int, int],
    probability: float,
    side: str,
    extreme: bool,
) -> float:
    goals_a, goals_b = key
    winner_goals = goals_a if side == "team_a" else goals_b
    loser_goals = goals_b if side == "team_a" else goals_a
    if not extreme:
        return float(probability)
    tail_bonus = np.exp(0.75 * max(winner_goals - 3, 0) + 0.75 * loser_goals)
    return float(probability) * float(tail_bonus)


def select_top_scorelines_with_tail_risk(
    score_matrix: ScoreMatrix,
    result_probabilities: Dict[str, float],
    lambda_a: float,
    lambda_b: float,
    current_top_scorelines: list[Dict[str, Any]] | None = None,
    top_n: int = 15,
    favorite_win_gate: float = DEFAULT_TAIL_FAVORITE_WIN_GATE,
    extreme_favorite_win_gate: float = DEFAULT_TAIL_EXTREME_FAVORITE_WIN_GATE,
    draw_ceiling: float = DEFAULT_TAIL_DRAW_CEILING,
    favorite_lambda_gate: float = DEFAULT_TAIL_FAVORITE_LAMBDA_GATE,
    extreme_lambda_gate: float = DEFAULT_TAIL_EXTREME_LAMBDA_GATE,
    lambda_gap_gate: float = DEFAULT_TAIL_LAMBDA_GAP_GATE,
    total_lambda_gate: float = DEFAULT_TAIL_TOTAL_LAMBDA_GATE,
    relative_floor: float = DEFAULT_TAIL_RELATIVE_FLOOR,
    absolute_floor: float = DEFAULT_TAIL_ABSOLUTE_FLOOR,
    max_winner_goals: int = DEFAULT_TAIL_MAX_WINNER_GOALS,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    ranked_keys = [
        key
        for key, _ in sorted(
            score_matrix.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]
    current_keys = (
        [
            (int(item["team_a_goals"]), int(item["team_b_goals"]))
            for item in current_top_scorelines
        ]
        if current_top_scorelines
        else ranked_keys
    )
    selected = current_keys[:top_n]
    diagnostics: Dict[str, Any] = {
        "tail_risk_selector_enabled": True,
        "tail_risk_applied": False,
        "favorite_win_gate": float(favorite_win_gate),
        "extreme_favorite_win_gate": float(extreme_favorite_win_gate),
        "draw_ceiling": float(draw_ceiling),
        "favorite_lambda_gate": float(favorite_lambda_gate),
        "extreme_lambda_gate": float(extreme_lambda_gate),
        "lambda_gap_gate": float(lambda_gap_gate),
        "total_lambda_gate": float(total_lambda_gate),
        "relative_floor": float(relative_floor),
        "absolute_floor": float(absolute_floor),
        "max_winner_goals": int(max_winner_goals),
    }
    if len(selected) < 3:
        return [score_item(key, score_matrix[key]) for key in selected], diagnostics

    side = v26.favorite_side(result_probabilities)
    favorite_probability = (
        float(result_probabilities.get("team_a_win", 0.0))
        if side == "team_a"
        else float(result_probabilities.get("team_b_win", 0.0))
        if side == "team_b"
        else 0.0
    )
    favorite_lambda = float(lambda_a) if side == "team_a" else float(lambda_b)
    underdog_lambda = float(lambda_b) if side == "team_a" else float(lambda_a)
    lambda_gap = favorite_lambda - underdog_lambda
    total_lambda = float(lambda_a) + float(lambda_b)
    top_probability = max(float(score_matrix.get(selected[0], 0.0)), 1e-12)
    probability_floor = max(
        float(absolute_floor),
        top_probability * float(relative_floor),
    )
    extreme = (
        favorite_probability >= float(extreme_favorite_win_gate)
        and favorite_lambda >= float(extreme_lambda_gate)
    )
    diagnostics.update(
        {
            "favorite_side": side,
            "favorite_probability": favorite_probability,
            "favorite_lambda": favorite_lambda,
            "underdog_lambda": underdog_lambda,
            "lambda_gap": float(lambda_gap),
            "total_lambda": total_lambda,
            "probability_floor": probability_floor,
            "extreme_tail_mode": bool(extreme),
        }
    )

    qualifies = (
        side is not None
        and favorite_probability >= float(favorite_win_gate)
        and float(result_probabilities.get("draw", 0.0)) <= float(draw_ceiling)
        and favorite_lambda >= float(favorite_lambda_gate)
        and lambda_gap >= float(lambda_gap_gate)
        and total_lambda >= float(total_lambda_gate)
    )
    if not qualifies:
        diagnostics["skip_reason"] = "gates_not_met"
        return [score_item(key, score_matrix[key]) for key in selected], diagnostics

    top_three = selected[:3]
    if not extreme and is_existing_tail_coverage(top_three, side):
        diagnostics["skip_reason"] = "tail_already_covered"
        return [score_item(key, score_matrix[key]) for key in selected], diagnostics

    selected_set = set(top_three)
    available = [
        key
        for key in tail_candidates(score_matrix, side, max_winner_goals)
        if key not in selected_set
    ]
    if not available:
        diagnostics["skip_reason"] = "no_candidate"
        return [score_item(key, score_matrix[key]) for key in selected], diagnostics

    candidate = max(
        available,
        key=lambda key: tail_utility(
            key,
            score_matrix.get(key, 0.0),
            side,
            extreme=extreme,
        ),
    )
    candidate_probability = float(score_matrix.get(candidate, 0.0))
    diagnostics["candidate_scoreline"] = f"{candidate[0]}-{candidate[1]}"
    diagnostics["candidate_probability"] = candidate_probability
    diagnostics["candidate_utility"] = tail_utility(
        candidate,
        candidate_probability,
        side,
        extreme=extreme,
    )
    if candidate_probability < probability_floor:
        diagnostics["skip_reason"] = "candidate_below_floor"
        return [score_item(key, score_matrix[key]) for key in selected], diagnostics

    top_three[-1] = candidate
    rebuilt: list[Tuple[int, int]] = []
    for key in [*top_three, *selected, *ranked_keys]:
        if key not in rebuilt and key in score_matrix:
            rebuilt.append(key)
        if len(rebuilt) >= top_n:
            break
    diagnostics["tail_risk_applied"] = True
    diagnostics["replaced_third_scoreline"] = f"{selected[2][0]}-{selected[2][1]}"
    return [score_item(key, score_matrix[key]) for key in rebuilt], diagnostics


class V29TailRiskScorelineModel:
    """Wrap V28 and add one conservative blowout candidate to the Top-3."""

    def __init__(
        self,
        base_model: v28.V28CurrentWorldCupFormModel,
        favorite_win_gate: float = DEFAULT_TAIL_FAVORITE_WIN_GATE,
        extreme_favorite_win_gate: float = DEFAULT_TAIL_EXTREME_FAVORITE_WIN_GATE,
        draw_ceiling: float = DEFAULT_TAIL_DRAW_CEILING,
        favorite_lambda_gate: float = DEFAULT_TAIL_FAVORITE_LAMBDA_GATE,
        extreme_lambda_gate: float = DEFAULT_TAIL_EXTREME_LAMBDA_GATE,
        lambda_gap_gate: float = DEFAULT_TAIL_LAMBDA_GAP_GATE,
        total_lambda_gate: float = DEFAULT_TAIL_TOTAL_LAMBDA_GATE,
        relative_floor: float = DEFAULT_TAIL_RELATIVE_FLOOR,
        absolute_floor: float = DEFAULT_TAIL_ABSOLUTE_FLOOR,
        max_winner_goals: int = DEFAULT_TAIL_MAX_WINNER_GOALS,
    ):
        self.base_model = base_model
        self.favorite_win_gate = float(np.clip(favorite_win_gate, 0.0, 1.0))
        self.extreme_favorite_win_gate = float(
            np.clip(extreme_favorite_win_gate, 0.0, 1.0)
        )
        self.draw_ceiling = float(np.clip(draw_ceiling, 0.0, 1.0))
        self.favorite_lambda_gate = float(max(favorite_lambda_gate, 0.0))
        self.extreme_lambda_gate = float(max(extreme_lambda_gate, 0.0))
        self.lambda_gap_gate = float(max(lambda_gap_gate, 0.0))
        self.total_lambda_gate = float(max(total_lambda_gate, 0.0))
        self.relative_floor = float(max(relative_floor, 0.0))
        self.absolute_floor = float(max(absolute_floor, 0.0))
        self.max_winner_goals = int(max(max_winner_goals, 3))
        self.training_data_summary = getattr(base_model, "training_data_summary", {})

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_model, name)

    def predict(self, *args, **kwargs) -> Dict[str, Any]:
        prediction = self.base_model.predict(*args, **kwargs)
        score_matrix = score_matrix_from_prediction(prediction)
        top_scorelines, diagnostics = select_top_scorelines_with_tail_risk(
            score_matrix,
            prediction["result_probabilities"],
            float(prediction["lambda_a"]),
            float(prediction["lambda_b"]),
            current_top_scorelines=prediction.get("top_scorelines", []),
            top_n=15,
            favorite_win_gate=self.favorite_win_gate,
            extreme_favorite_win_gate=self.extreme_favorite_win_gate,
            draw_ceiling=self.draw_ceiling,
            favorite_lambda_gate=self.favorite_lambda_gate,
            extreme_lambda_gate=self.extreme_lambda_gate,
            lambda_gap_gate=self.lambda_gap_gate,
            total_lambda_gate=self.total_lambda_gate,
            relative_floor=self.relative_floor,
            absolute_floor=self.absolute_floor,
            max_winner_goals=self.max_winner_goals,
        )
        prediction["top_scorelines"] = top_scorelines
        prediction["v29_adjustments"] = {
            "base_model": "v28_current_worldcup_form",
            "scoreline_policy": "tail_risk_top3_selector_only",
            "probability_matrix_changed": False,
            "scoreline_layer_affects_wdl": False,
            **diagnostics,
        }
        prediction["calibration_notes"] = {
            **prediction.get("calibration_notes", {}),
            "v29": prediction["v29_adjustments"],
            "tail_risk_policy": (
                "V29 leaves V28 probabilities unchanged and only replaces the "
                "third displayed Top-3 scoreline with one blowout candidate "
                "when favorite, lambda, draw, and probability-floor gates pass."
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
    current_wdl_blend=v28.DEFAULT_CURRENT_WDL_BLEND,
    current_scoreline_blend=v28.DEFAULT_CURRENT_SCORELINE_BLEND,
    beta_attack_edge=v28.DEFAULT_BETA_ATTACK_EDGE,
    beta_tempo=v28.DEFAULT_BETA_TEMPO,
    beta_group_pressure=v28.DEFAULT_BETA_GROUP_PRESSURE,
    max_log_adjustment=v28.DEFAULT_MAX_LOG_ADJUSTMENT,
    total_calibration_blend=v28.DEFAULT_TOTAL_CALIBRATION_BLEND,
    tail_relative_floor=v28.DEFAULT_TAIL_RELATIVE_FLOOR,
    favorite_win_gate=v26.DEFAULT_FAVORITE_WIN_GATE,
    total_lambda_gate=v26.DEFAULT_TOTAL_LAMBDA_GATE,
    favorite_lambda_gate=v26.DEFAULT_FAVORITE_LAMBDA_GATE,
    draw_ceiling=v26.DEFAULT_DRAW_CEILING,
    tail_favorite_win_gate=DEFAULT_TAIL_FAVORITE_WIN_GATE,
    tail_extreme_favorite_win_gate=DEFAULT_TAIL_EXTREME_FAVORITE_WIN_GATE,
    tail_draw_ceiling=DEFAULT_TAIL_DRAW_CEILING,
    tail_favorite_lambda_gate=DEFAULT_TAIL_FAVORITE_LAMBDA_GATE,
    tail_extreme_lambda_gate=DEFAULT_TAIL_EXTREME_LAMBDA_GATE,
    tail_lambda_gap_gate=DEFAULT_TAIL_LAMBDA_GAP_GATE,
    tail_total_lambda_gate=DEFAULT_TAIL_TOTAL_LAMBDA_GATE,
    tail_selector_relative_floor=DEFAULT_TAIL_RELATIVE_FLOOR,
    tail_selector_absolute_floor=DEFAULT_TAIL_ABSOLUTE_FLOOR,
    tail_max_winner_goals=DEFAULT_TAIL_MAX_WINNER_GOALS,
):
    base_model, data = v28.build_from_zip(
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
        include_observed_goals=include_observed_goals,
        include_fotmob_goal_stats=include_fotmob_goal_stats,
        include_group_score_context=include_group_score_context,
        total_calibration_strength=total_calibration_strength,
        total_multiplier_clip_low=total_multiplier_clip_low,
        total_multiplier_clip_high=total_multiplier_clip_high,
        total_smoothing=total_smoothing,
        min_bin_support=min_bin_support,
        max_train_matches=max_train_matches,
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
    model = V29TailRiskScorelineModel(
        base_model,
        favorite_win_gate=tail_favorite_win_gate,
        extreme_favorite_win_gate=tail_extreme_favorite_win_gate,
        draw_ceiling=tail_draw_ceiling,
        favorite_lambda_gate=tail_favorite_lambda_gate,
        extreme_lambda_gate=tail_extreme_lambda_gate,
        lambda_gap_gate=tail_lambda_gap_gate,
        total_lambda_gate=tail_total_lambda_gate,
        relative_floor=tail_selector_relative_floor,
        absolute_floor=tail_selector_absolute_floor,
        max_winner_goals=tail_max_winner_goals,
    )
    model.training_data_summary = {
        **getattr(base_model, "training_data_summary", {}),
        "v29_scoreline_policy": "tail_risk_top3_selector_only",
        "v29_probability_matrix_changed": False,
        "v29_tail_favorite_win_gate": model.favorite_win_gate,
        "v29_tail_extreme_favorite_win_gate": model.extreme_favorite_win_gate,
        "v29_tail_draw_ceiling": model.draw_ceiling,
        "v29_tail_favorite_lambda_gate": model.favorite_lambda_gate,
        "v29_tail_extreme_lambda_gate": model.extreme_lambda_gate,
        "v29_tail_lambda_gap_gate": model.lambda_gap_gate,
        "v29_tail_total_lambda_gate": model.total_lambda_gate,
        "v29_tail_selector_relative_floor": model.relative_floor,
        "v29_tail_selector_absolute_floor": model.absolute_floor,
        "v29_tail_max_winner_goals": model.max_winner_goals,
    }
    return model, data


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    parser = argparse.ArgumentParser(
        description="Run V29: V28 with conservative tail-risk Top-3 selection."
    )
    parser.add_argument("--team-a", required=True)
    parser.add_argument("--team-b", required=True)
    parser.add_argument("--host-a", action="store_true")
    parser.add_argument("--host-b", action="store_true")
    parser.add_argument("--knockout", action="store_true")
    parser.add_argument("--outdir", default="outputs/outputs_v29_tail_risk")
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
    parser.add_argument("--blind-current-goals", action="store_true")
    parser.add_argument("--ignore-observed-goals", action="store_true")
    parser.add_argument("--ignore-fotmob-goal-stats", action="store_true")
    parser.add_argument("--disable-group-score-context", action="store_true")
    parser.add_argument("--tail-favorite-win-gate", type=float, default=DEFAULT_TAIL_FAVORITE_WIN_GATE)
    parser.add_argument("--tail-extreme-favorite-win-gate", type=float, default=DEFAULT_TAIL_EXTREME_FAVORITE_WIN_GATE)
    parser.add_argument("--tail-draw-ceiling", type=float, default=DEFAULT_TAIL_DRAW_CEILING)
    parser.add_argument("--tail-favorite-lambda-gate", type=float, default=DEFAULT_TAIL_FAVORITE_LAMBDA_GATE)
    parser.add_argument("--tail-extreme-lambda-gate", type=float, default=DEFAULT_TAIL_EXTREME_LAMBDA_GATE)
    parser.add_argument("--tail-lambda-gap-gate", type=float, default=DEFAULT_TAIL_LAMBDA_GAP_GATE)
    parser.add_argument("--tail-total-lambda-gate", type=float, default=DEFAULT_TAIL_TOTAL_LAMBDA_GATE)
    parser.add_argument("--tail-selector-relative-floor", type=float, default=DEFAULT_TAIL_RELATIVE_FLOOR)
    parser.add_argument("--tail-selector-absolute-floor", type=float, default=DEFAULT_TAIL_ABSOLUTE_FLOOR)
    parser.add_argument("--tail-max-winner-goals", type=int, default=DEFAULT_TAIL_MAX_WINNER_GOALS)
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
        tail_favorite_win_gate=args.tail_favorite_win_gate,
        tail_extreme_favorite_win_gate=args.tail_extreme_favorite_win_gate,
        tail_draw_ceiling=args.tail_draw_ceiling,
        tail_favorite_lambda_gate=args.tail_favorite_lambda_gate,
        tail_extreme_lambda_gate=args.tail_extreme_lambda_gate,
        tail_lambda_gap_gate=args.tail_lambda_gap_gate,
        tail_total_lambda_gate=args.tail_total_lambda_gate,
        tail_selector_relative_floor=args.tail_selector_relative_floor,
        tail_selector_absolute_floor=args.tail_selector_absolute_floor,
        tail_max_winner_goals=args.tail_max_winner_goals,
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
                "version": "v29-tail-risk-scoreline",
                "base_model": "v28-current-worldcup-form",
                "team_a": prediction["team_a"],
                "team_b": prediction["team_b"],
                "result_probabilities": prediction["result_probabilities"],
                "v29_adjustments": prediction["v29_adjustments"],
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
                "top_scorelines": prediction["top_scorelines"][:5],
                "v29_adjustments": prediction["v29_adjustments"],
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
'''
v29_tail_risk_scoreline_model = _load_submodule("v29_tail_risk_scoreline_model", _V29_TAIL_RISK_SCORELINE_MODEL_SOURCE, "feature_layers.py:v29_tail_risk_scoreline_model")

# ======================================================================
# v30_player_role_form_model.py  (bundled as an isolated sub-module)
# ======================================================================
_V30_PLAYER_ROLE_FORM_MODEL_SOURCE = r'''
#!/usr/bin/env python3
"""V30: V29 plus conservative player-role form and matchup adjustments.

Run:
    .venv/bin/python v30_player_role_form_model.py --team-a "Switzerland" --team-b "Bosnia and Herzegovina"
    .venv/bin/python v30_player_role_form_model.py --team-a "Switzerland" --team-b "Bosnia and Herzegovina" --no-plots
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
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
import v28_current_worldcup_form_model as v28
import v29_tail_risk_scoreline_model as v29


ScoreMatrix = Dict[Tuple[int, int], float]

DEFAULT_ROLE_FORM_WDL_BLEND = 0.14
DEFAULT_ROLE_FORM_SCORELINE_BLEND = 0.30
DEFAULT_BETA_ROLE_ATTACK_EDGE = 0.055
DEFAULT_BETA_SET_PIECE_EDGE = 0.020
DEFAULT_BETA_KEEPER_FRAGILITY = 0.025
DEFAULT_BETA_DISCIPLINE_EDGE = 0.015
DEFAULT_MAX_ROLE_LOG_ADJUSTMENT = 0.10

ATTACKER_STATS = {
    "Top scorer": 0.65,
    "Goals per 90": 0.65,
    "Goals per match": 0.65,
    "Expected goals (xG)": 1.00,
    "xG per 90": 1.00,
    "Expected goals on target (xGOT)": 1.00,
    "Shots per 90": 0.55,
    "Shots per match": 0.55,
    "Shots on target per 90": 0.75,
    "Shots on target per match": 0.75,
    "Big chances": 0.80,
    "Touches in opposition box": 0.55,
    "Penalties awarded": 0.25,
    "Big chances missed": -0.35,
}
CREATOR_STATS = {
    "Assists": 0.45,
    "Expected assists (xA)": 1.00,
    "xA per 90": 1.00,
    "xG + xA per 90": 0.80,
    "Chances created": 0.80,
    "Big chances created": 0.95,
    "Accurate crosses per match": 0.40,
    "Accurate long balls per 90": 0.35,
    "Accurate long balls per match": 0.35,
    "Accurate passes per 90": 0.25,
    "Accurate passes per match": 0.25,
    "Successful dribbles per 90": 0.45,
}
DEFENDER_STATS = {
    "Clean sheets": 0.45,
    "Defensive contributions per 90": 0.85,
    "Tackles per 90": 0.60,
    "Tackles per match": 0.60,
    "Interceptions per 90": 0.65,
    "Interceptions per match": 0.65,
    "Clearances per 90": 0.50,
    "Clearances per match": 0.50,
    "Blocks per 90": 0.55,
    "Recoveries per 90": 0.45,
    "Possession won final 3rd per 90": 0.40,
    "Possession won final 3rd per match": 0.40,
}
KEEPER_STATS = {
    "Save percentage": 0.70,
    "Goals prevented": 1.00,
    "Saves per 90": 0.45,
    "Saves per match": 0.45,
}
FRAGILITY_STATS = {
    "xG conceded": 1.00,
    "Goals conceded per 90": 0.55,
    "Goals conceded per match": 0.55,
    "Penalties conceded": 0.35,
}
SET_PIECE_FOR_STATS = {
    "Set piece goals": 0.80,
    "Corners": 0.45,
    "Accurate crosses per match": 0.30,
}
SET_PIECE_AGAINST_STATS = {
    "Set piece goals conceded": 0.80,
}
DISCIPLINE_STATS = {
    "Yellow cards": 0.40,
    "Red cards": 0.90,
    "Fouls committed per 90": 0.45,
    "Fouls per match": 0.45,
}
MINUTES_STATS = {"Minutes played", "FotMob rating"}


@dataclass
class PlayerRoleProfile:
    team: str
    attacker: float = 0.0
    creator: float = 0.0
    defender: float = 0.0
    keeper: float = 0.0
    defensive_fragility: float = 0.0
    set_piece_for: float = 0.0
    set_piece_against: float = 0.0
    discipline_risk: float = 0.0
    finishing_delta: float = 0.0
    coverage: float = 0.0
    rows: int = 0
    minutes_rows: int = 0

    @property
    def attack_unit(self) -> float:
        return float(0.60 * self.attacker + 0.40 * self.creator)

    @property
    def defense_unit(self) -> float:
        return float(0.65 * self.defender + 0.35 * self.keeper)

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "team": self.team,
            "attacker": self.attacker,
            "creator": self.creator,
            "defender": self.defender,
            "keeper": self.keeper,
            "defensive_fragility": self.defensive_fragility,
            "set_piece_for": self.set_piece_for,
            "set_piece_against": self.set_piece_against,
            "discipline_risk": self.discipline_risk,
            "finishing_delta": self.finishing_delta,
            "attack_unit": self.attack_unit,
            "defense_unit": self.defense_unit,
            "coverage": self.coverage,
            "rows": int(self.rows),
            "minutes_rows": int(self.minutes_rows),
        }


def parse_number(value: object) -> float:
    return v28.parse_number(value)


def zscore(series: pd.Series) -> pd.Series:
    return v28.zscore(series)


def normalized_stat_contribution(frame: pd.DataFrame, stat: str) -> pd.Series:
    return v28.normalized_stat_contribution(frame, stat)


def stat_bucket(stat: str) -> tuple[str | None, float]:
    for bucket_name, mapping in (
        ("attacker", ATTACKER_STATS),
        ("creator", CREATOR_STATS),
        ("defender", DEFENDER_STATS),
        ("keeper", KEEPER_STATS),
        ("defensive_fragility", FRAGILITY_STATS),
        ("set_piece_for", SET_PIECE_FOR_STATS),
        ("set_piece_against", SET_PIECE_AGAINST_STATS),
        ("discipline_risk", DISCIPLINE_STATS),
    ):
        if stat in mapping:
            return bucket_name, float(mapping[stat])
    return None, 0.0


def normalize_player_name(name: object) -> str:
    text = str(name).lower().strip()
    text = "".join(
        char
        for char in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(char)
    )
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def player_match_score(player_name: object, roster_name: object) -> float:
    player = normalize_player_name(player_name)
    roster = normalize_player_name(roster_name)
    if not player or not roster:
        return 0.0
    player_tokens = player.split()
    roster_tokens = roster.split()
    if player == roster:
        return 100.0
    if roster in player or player in roster:
        return 90.0
    if player_tokens[-1] == roster_tokens[-1]:
        return 80.0
    overlap = len(set(player_tokens) & set(roster_tokens))
    return 50.0 + 10.0 * overlap if overlap else 0.0


def attach_roster_context(
    player_stats: pd.DataFrame,
    lineups: pd.DataFrame,
    substitutions: pd.DataFrame,
    min_score: float = 70.0,
) -> pd.DataFrame:
    roster_columns = [
        "match_id",
        "team",
        "player",
        "position",
        "squad_role",
        "minute_on",
        "minute_off",
    ]
    roster_parts = []
    for frame in (lineups, substitutions):
        if not frame.empty and set(roster_columns).issubset(frame.columns):
            roster_parts.append(frame[roster_columns])
    if not roster_parts:
        enriched = player_stats.copy()
        enriched["team"] = ""
        enriched["position"] = ""
        enriched["roster_match_score"] = 0.0
        return enriched

    roster = pd.concat(roster_parts, ignore_index=True).drop_duplicates()
    rows = []
    for match_id, group in player_stats.groupby("match_id", dropna=False):
        choices = roster[roster["match_id"].eq(match_id)]
        for _, row in group.iterrows():
            best = None
            best_score = 0.0
            for _, candidate in choices.iterrows():
                score = player_match_score(row["player"], candidate["player"])
                if score > best_score:
                    best = candidate
                    best_score = score
            record = row.to_dict()
            if best is not None and best_score >= min_score:
                record["team"] = best["team"]
                record["position"] = best["position"]
                record["squad_role"] = best.get("squad_role", "")
                record["lineup_minute_on"] = best.get("minute_on", np.nan)
                record["lineup_minute_off"] = best.get("minute_off", np.nan)
                record["roster_player"] = best["player"]
                record["roster_match_score"] = float(best_score)
            else:
                record["team"] = ""
                record["position"] = ""
                record["squad_role"] = ""
                record["lineup_minute_on"] = np.nan
                record["lineup_minute_off"] = np.nan
                record["roster_player"] = ""
                record["roster_match_score"] = float(best_score)
            rows.append(record)
    return pd.DataFrame(rows)


def aggregate_player_match_stats(player_stats: pd.DataFrame) -> pd.DataFrame:
    if player_stats.empty:
        return player_stats
    numeric_columns = [
        column
        for column in player_stats.columns
        if column
        not in {
            "match_id",
            "match_slug",
            "match_url",
            "title",
            "home_team",
            "away_team",
            "kickoff",
            "round",
            "venue",
            "referee",
            "status",
            "category",
            "player",
        }
        and pd.api.types.is_numeric_dtype(player_stats[column])
    ]
    aggregations = {column: "max" for column in numeric_columns}
    for column in (
        "match_slug",
        "match_url",
        "title",
        "home_team",
        "away_team",
        "kickoff",
        "round",
        "status",
    ):
        if column in player_stats:
            aggregations[column] = "first"
    return (
        player_stats.groupby(["match_id", "player"], as_index=False)
        .agg(aggregations)
        .copy()
    )


def build_match_player_role_profiles(
    player_stats_csv: str | Path | None,
    lineups_csv: str | Path | None,
    substitutions_csv: str | Path | None,
    keeper_stats_csv: str | Path | None,
) -> dict[str, PlayerRoleProfile]:
    paths = [player_stats_csv, lineups_csv, substitutions_csv]
    if not all(path and Path(path).exists() for path in paths):
        return {}
    player_stats = pd.read_csv(player_stats_csv)
    lineups = pd.read_csv(lineups_csv)
    substitutions = pd.read_csv(substitutions_csv)
    if player_stats.empty or lineups.empty:
        return {}

    player_stats = aggregate_player_match_stats(player_stats)
    enriched = attach_roster_context(player_stats, lineups, substitutions)
    enriched = enriched[enriched["team"].astype(str).ne("")]
    if enriched.empty:
        return {}

    numeric = [
        "minutes_played",
        "fotmob_rating",
        "goals",
        "assists",
        "xg",
        "xa",
        "xgot",
        "total_shots",
        "shots_on_target",
        "touches_in_opposition_box",
        "successful_dribbles",
        "big_chances_missed",
        "chances_created",
        "passes_into_final_third",
        "accurate_crosses",
        "accurate_long_balls",
        "defensive_contributions",
        "tackles",
        "interceptions",
        "blocks",
        "recoveries",
        "clearances",
        "fouls_committed",
    ]
    for column in numeric:
        if column not in enriched:
            enriched[column] = 0.0
        enriched[column] = pd.to_numeric(enriched[column], errors="coerce").fillna(0.0)

    minutes = enriched["minutes_played"].where(enriched["minutes_played"] > 0, 90.0)
    minute_weight = np.sqrt(np.clip(minutes / 90.0, 0.05, 1.25))
    position = enriched["position"].astype(str).str.upper()
    attacker_position = position.isin(["FWD", "FW", "ST", "CF"])
    creator_position = position.isin(["AM", "MID", "MF", "CM", "DM", "W"])
    defender_position = position.isin(["DEF", "DF", "CB", "LB", "RB", "WB"])
    keeper_position = position.isin(["GK", "KEEPER"])

    enriched["attacker_raw"] = minute_weight * (
        (1.00 * enriched["xg"] + 0.80 * enriched["xgot"])
        + 0.22 * enriched["shots_on_target"]
        + 0.12 * enriched["total_shots"]
        + 0.08 * enriched["touches_in_opposition_box"]
        + 0.05 * (enriched["fotmob_rating"] - 6.5).clip(lower=-1.5)
        - 0.25 * enriched["big_chances_missed"]
    ) * np.where(attacker_position | creator_position, 1.0, 0.40)
    enriched["creator_raw"] = minute_weight * (
        1.00 * enriched["xa"]
        + 0.28 * enriched["chances_created"]
        + 0.08 * enriched["passes_into_final_third"]
        + 0.10 * enriched["accurate_crosses"]
        + 0.08 * enriched["successful_dribbles"]
        + 0.04 * (enriched["fotmob_rating"] - 6.5).clip(lower=-1.5)
    ) * np.where(attacker_position | creator_position, 1.0, 0.45)
    enriched["defender_raw"] = minute_weight * (
        0.20 * enriched["defensive_contributions"]
        + 0.22 * enriched["tackles"]
        + 0.28 * enriched["interceptions"]
        + 0.28 * enriched["blocks"]
        + 0.08 * enriched["recoveries"]
        + 0.12 * enriched["clearances"]
        + 0.04 * (enriched["fotmob_rating"] - 6.5).clip(lower=-1.5)
    ) * np.where(defender_position | creator_position, 1.0, 0.35)
    enriched["set_piece_for_raw"] = minute_weight * (
        0.16 * enriched["accurate_crosses"]
        + 0.10 * enriched["accurate_long_balls"]
        + 0.30 * enriched["xa"]
    )
    enriched["discipline_raw"] = minute_weight * (0.20 * enriched["fouls_committed"])

    team = (
        enriched.groupby("team", as_index=False)
        .agg(
            rows=("player", "size"),
            matches=("match_id", "nunique"),
            attacker=("attacker_raw", "sum"),
            creator=("creator_raw", "sum"),
            defender=("defender_raw", "sum"),
            set_piece_for=("set_piece_for_raw", "sum"),
            discipline_risk=("discipline_raw", "sum"),
            minutes_rows=("minutes_played", lambda s: int((s > 0).sum())),
        )
        .copy()
    )

    for column in [
        "attacker",
        "creator",
        "defender",
        "set_piece_for",
        "discipline_risk",
    ]:
        team[column] = zscore(team[column])
    team["keeper"] = 0.0
    team["defensive_fragility"] = 0.0
    team["set_piece_against"] = 0.0

    if keeper_stats_csv and Path(keeper_stats_csv).exists():
        keeper = pd.read_csv(keeper_stats_csv)
        if not keeper.empty and {"match_id", "player"}.issubset(keeper.columns):
            keeper_agg = aggregate_player_match_stats(keeper)
            keeper_enriched = attach_roster_context(keeper_agg, lineups, substitutions)
            keeper_enriched = keeper_enriched[keeper_enriched["team"].astype(str).ne("")]
            for column in [
                "saves",
                "goals_conceded",
                "xgot_faced",
                "goals_prevented",
                "acted_as_sweeper",
                "high_claim",
            ]:
                if column not in keeper_enriched:
                    keeper_enriched[column] = 0.0
                keeper_enriched[column] = pd.to_numeric(
                    keeper_enriched[column], errors="coerce"
                ).fillna(0.0)
            if not keeper_enriched.empty:
                keeper_team = (
                    keeper_enriched.groupby("team", as_index=False)
                    .agg(
                        keeper=(
                            "goals_prevented",
                            "sum",
                        ),
                        xgot_faced=("xgot_faced", "sum"),
                        goals_conceded=("goals_conceded", "sum"),
                        saves=("saves", "sum"),
                    )
                    .copy()
                )
                keeper_team["keeper"] = zscore(
                    keeper_team["keeper"] + 0.08 * keeper_team["saves"]
                )
                keeper_team["defensive_fragility"] = zscore(
                    keeper_team["xgot_faced"]
                    + 0.45 * keeper_team["goals_conceded"]
                    - 0.35 * keeper_team["saves"]
                )
                team = team.merge(
                    keeper_team[["team", "keeper", "defensive_fragility"]],
                    on="team",
                    how="left",
                    suffixes=("", "_keeper"),
                )
                team["keeper"] = team["keeper_keeper"].fillna(team["keeper"])
                team["defensive_fragility"] = team[
                    "defensive_fragility_keeper"
                ].fillna(team["defensive_fragility"])
                team = team.drop(
                    columns=["keeper_keeper", "defensive_fragility_keeper"]
                )

    coverage = np.sqrt(team["rows"] / (team["rows"] + 8.0))
    coverage *= np.sqrt(team["matches"] / (team["matches"] + 2.0))
    team["coverage"] = np.clip(coverage, 0.0, 1.0)
    team["finishing_delta"] = zscore(team["attacker"] - 0.65 * team["creator"])

    profiles: dict[str, PlayerRoleProfile] = {}
    for row in team.to_dict(orient="records"):
        coverage_value = float(row.get("coverage", 0.0) or 0.0)
        profiles[str(row["team"])] = PlayerRoleProfile(
            team=str(row["team"]),
            attacker=float(row.get("attacker", 0.0)) * coverage_value,
            creator=float(row.get("creator", 0.0)) * coverage_value,
            defender=float(row.get("defender", 0.0)) * coverage_value,
            keeper=float(row.get("keeper", 0.0)) * coverage_value,
            defensive_fragility=float(row.get("defensive_fragility", 0.0))
            * coverage_value,
            set_piece_for=float(row.get("set_piece_for", 0.0)) * coverage_value,
            set_piece_against=float(row.get("set_piece_against", 0.0))
            * coverage_value,
            discipline_risk=float(row.get("discipline_risk", 0.0))
            * coverage_value,
            finishing_delta=float(row.get("finishing_delta", 0.0)) * coverage_value,
            coverage=coverage_value,
            rows=int(row.get("rows", 0) or 0),
            minutes_rows=int(row.get("minutes_rows", 0) or 0),
        )
    return profiles


def merge_role_profiles(
    leaderboard_profiles: dict[str, PlayerRoleProfile],
    match_profiles: dict[str, PlayerRoleProfile],
) -> dict[str, PlayerRoleProfile]:
    teams = set(leaderboard_profiles) | set(match_profiles)
    merged: dict[str, PlayerRoleProfile] = {}
    fields = [
        "attacker",
        "creator",
        "defender",
        "keeper",
        "defensive_fragility",
        "set_piece_for",
        "set_piece_against",
        "discipline_risk",
        "finishing_delta",
    ]
    for team in teams:
        leader = leaderboard_profiles.get(team, PlayerRoleProfile(team=team))
        match = match_profiles.get(team, PlayerRoleProfile(team=team))
        match_weight = 0.70 * min(max(match.coverage, 0.0), 1.0)
        values = {
            field: (1.0 - match_weight) * getattr(leader, field)
            + match_weight * getattr(match, field)
            for field in fields
        }
        merged[team] = PlayerRoleProfile(
            team=team,
            **values,
            coverage=max(leader.coverage, match.coverage),
            rows=leader.rows + match.rows,
            minutes_rows=leader.minutes_rows + match.minutes_rows,
        )
    return merged


def build_player_role_profiles(path: str | Path | None) -> dict[str, PlayerRoleProfile]:
    if not path or not Path(path).exists():
        return {}
    frame = pd.read_csv(path)
    required = {"stat", "country_or_team", "value"}
    if frame.empty or not required.issubset(frame.columns):
        return {}

    clean = frame.copy()
    clean["team"] = clean["country_or_team"].map(v28.canon_team)
    clean = clean[clean["team"].ne("")]
    if clean.empty:
        return {}
    clean["numeric_value"] = clean["value"].map(parse_number)

    records: list[dict[str, Any]] = []
    for stat in sorted(clean["stat"].dropna().unique()):
        bucket, weight = stat_bucket(str(stat))
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
                    "contribution": float(contribution) * weight,
                }
            )

    if not records:
        return {}
    contrib = pd.DataFrame(records)
    pivot = contrib.pivot_table(
        index="team",
        columns="bucket",
        values="contribution",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()
    buckets = [
        "attacker",
        "creator",
        "defender",
        "keeper",
        "defensive_fragility",
        "set_piece_for",
        "set_piece_against",
        "discipline_risk",
    ]
    for bucket in buckets:
        if bucket not in pivot:
            pivot[bucket] = 0.0
        pivot[bucket] = zscore(pivot[bucket])

    counts = contrib.groupby("team").size().rename("rows").reset_index()
    minutes = (
        clean[clean["stat"].isin(MINUTES_STATS)]
        .groupby("team")
        .size()
        .rename("minutes_rows")
        .reset_index()
    )
    pivot = pivot.merge(counts, on="team", how="left")
    pivot = pivot.merge(minutes, on="team", how="left")
    pivot["minutes_rows"] = pivot["minutes_rows"].fillna(0)
    base_coverage = np.sqrt(pivot["rows"] / (pivot["rows"] + 5.0))
    minutes_coverage = 1.0 + 0.08 * np.minimum(pivot["minutes_rows"], 3.0)
    pivot["coverage"] = np.clip(base_coverage * minutes_coverage, 0.0, 1.0)

    if {"attacker", "defensive_fragility"}.issubset(pivot.columns):
        pivot["finishing_delta"] = zscore(pivot["attacker"] - 0.45 * pivot["creator"])
    else:
        pivot["finishing_delta"] = 0.0

    profiles: dict[str, PlayerRoleProfile] = {}
    for row in pivot.to_dict(orient="records"):
        coverage = float(row.get("coverage", 0.0) or 0.0)
        profiles[str(row["team"])] = PlayerRoleProfile(
            team=str(row["team"]),
            attacker=float(row.get("attacker", 0.0)) * coverage,
            creator=float(row.get("creator", 0.0)) * coverage,
            defender=float(row.get("defender", 0.0)) * coverage,
            keeper=float(row.get("keeper", 0.0)) * coverage,
            defensive_fragility=float(row.get("defensive_fragility", 0.0)) * coverage,
            set_piece_for=float(row.get("set_piece_for", 0.0)) * coverage,
            set_piece_against=float(row.get("set_piece_against", 0.0)) * coverage,
            discipline_risk=float(row.get("discipline_risk", 0.0)) * coverage,
            finishing_delta=float(row.get("finishing_delta", 0.0)) * coverage,
            coverage=coverage,
            rows=int(row.get("rows", 0) or 0),
            minutes_rows=int(row.get("minutes_rows", 0) or 0),
        )
    return profiles


def score_matrix_from_prediction(prediction: Dict[str, Any]) -> ScoreMatrix:
    return v29.score_matrix_from_prediction(prediction)


def blend_result_probabilities(
    base: Dict[str, float],
    adjusted: Dict[str, float],
    adjusted_weight: float,
) -> Dict[str, float]:
    return v28.blend_result_probabilities(base, adjusted, adjusted_weight)


def expected_goals(score_matrix: ScoreMatrix) -> tuple[float, float]:
    return v28.expected_goals(score_matrix)


class V30PlayerRoleFormModel:
    """Wrap V29 with small player-role matchup adjustments."""

    def __init__(
        self,
        base_model: v29.V29TailRiskScorelineModel,
        role_profiles: dict[str, PlayerRoleProfile],
        role_form_wdl_blend: float = DEFAULT_ROLE_FORM_WDL_BLEND,
        role_form_scoreline_blend: float = DEFAULT_ROLE_FORM_SCORELINE_BLEND,
        beta_role_attack_edge: float = DEFAULT_BETA_ROLE_ATTACK_EDGE,
        beta_set_piece_edge: float = DEFAULT_BETA_SET_PIECE_EDGE,
        beta_keeper_fragility: float = DEFAULT_BETA_KEEPER_FRAGILITY,
        beta_discipline_edge: float = DEFAULT_BETA_DISCIPLINE_EDGE,
        max_role_log_adjustment: float = DEFAULT_MAX_ROLE_LOG_ADJUSTMENT,
    ):
        self.base_model = base_model
        self.role_profiles = role_profiles
        self.role_form_wdl_blend = float(np.clip(role_form_wdl_blend, 0.0, 1.0))
        self.role_form_scoreline_blend = float(
            np.clip(role_form_scoreline_blend, 0.0, 1.0)
        )
        self.beta_role_attack_edge = float(beta_role_attack_edge)
        self.beta_set_piece_edge = float(beta_set_piece_edge)
        self.beta_keeper_fragility = float(beta_keeper_fragility)
        self.beta_discipline_edge = float(beta_discipline_edge)
        self.max_role_log_adjustment = float(max(max_role_log_adjustment, 0.0))
        self.training_data_summary = getattr(base_model, "training_data_summary", {})

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_model, name)

    def role_for_team(self, team: object) -> PlayerRoleProfile:
        canonical = v28.canon_team(team)
        return self.role_profiles.get(canonical, PlayerRoleProfile(team=canonical))

    def _role_adjusted_matrix(
        self,
        prediction: Dict[str, Any],
        role_a: PlayerRoleProfile,
        role_b: PlayerRoleProfile,
        max_goals: int,
    ) -> tuple[ScoreMatrix, Dict[str, Any]]:
        base_lambda_a = float(prediction["lambda_a"])
        base_lambda_b = float(prediction["lambda_b"])
        attack_edge_a = role_a.attack_unit - role_b.defense_unit
        attack_edge_b = role_b.attack_unit - role_a.defense_unit
        set_piece_edge_a = role_a.set_piece_for + role_b.set_piece_against
        set_piece_edge_b = role_b.set_piece_for + role_a.set_piece_against
        fragility_edge_a = role_b.defensive_fragility - 0.35 * role_b.keeper
        fragility_edge_b = role_a.defensive_fragility - 0.35 * role_a.keeper
        discipline_edge_a = role_b.discipline_risk - 0.25 * role_a.discipline_risk
        discipline_edge_b = role_a.discipline_risk - 0.25 * role_b.discipline_risk

        log_a = (
            self.beta_role_attack_edge * attack_edge_a
            + self.beta_set_piece_edge * set_piece_edge_a
            + self.beta_keeper_fragility * fragility_edge_a
            + self.beta_discipline_edge * discipline_edge_a
        )
        log_b = (
            self.beta_role_attack_edge * attack_edge_b
            + self.beta_set_piece_edge * set_piece_edge_b
            + self.beta_keeper_fragility * fragility_edge_b
            + self.beta_discipline_edge * discipline_edge_b
        )
        log_a = float(
            np.clip(log_a, -self.max_role_log_adjustment, self.max_role_log_adjustment)
        )
        log_b = float(
            np.clip(log_b, -self.max_role_log_adjustment, self.max_role_log_adjustment)
        )
        lambda_a = float(np.clip(base_lambda_a * math.exp(log_a), 0.15, 5.5))
        lambda_b = float(np.clip(base_lambda_b * math.exp(log_b), 0.15, 5.5))
        matrix = v11.poisson_score_matrix(lambda_a, lambda_b, max_goals)
        rho = prediction.get("calibration_notes", {}).get("dixon_coles_rho", -0.08)
        matrix = v11.apply_dixon_coles_adjustment(matrix, lambda_a, lambda_b, rho=rho)
        return matrix, {
            "role_base_lambda_a": base_lambda_a,
            "role_base_lambda_b": base_lambda_b,
            "role_lambda_a": lambda_a,
            "role_lambda_b": lambda_b,
            "role_log_adjustment_a": log_a,
            "role_log_adjustment_b": log_b,
            "role_attack_edge_a": float(attack_edge_a),
            "role_attack_edge_b": float(attack_edge_b),
            "set_piece_edge_a": float(set_piece_edge_a),
            "set_piece_edge_b": float(set_piece_edge_b),
            "fragility_edge_a": float(fragility_edge_a),
            "fragility_edge_b": float(fragility_edge_b),
            "discipline_edge_a": float(discipline_edge_a),
            "discipline_edge_b": float(discipline_edge_b),
        }

    def predict(self, *args, **kwargs) -> Dict[str, Any]:
        team_a = kwargs.get("team_a", args[0] if args else "")
        team_b = kwargs.get("team_b", args[1] if len(args) > 1 else "")
        max_goals = int(kwargs.get("max_goals", args[5] if len(args) > 5 else 10))
        prediction = self.base_model.predict(*args, **kwargs)
        base_matrix = score_matrix_from_prediction(prediction)
        base_result_probabilities = dict(prediction["result_probabilities"])
        role_a = self.role_for_team(team_a)
        role_b = self.role_for_team(team_b)

        role_matrix, role_diagnostics = self._role_adjusted_matrix(
            prediction,
            role_a,
            role_b,
            max_goals=max_goals,
        )
        role_result_probabilities = v11.result_probs(role_matrix)
        final_result_probabilities = blend_result_probabilities(
            base_result_probabilities,
            role_result_probabilities,
            self.role_form_wdl_blend,
        )
        score_matrix = v20.blend_score_matrices(
            base_matrix,
            role_matrix,
            adjusted_weight=self.role_form_scoreline_blend,
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
        top_scorelines, tail_diagnostics = v29.select_top_scorelines_with_tail_risk(
            score_matrix,
            final_result_probabilities,
            lambda_a,
            lambda_b,
            current_top_scorelines=prediction.get("top_scorelines", []),
            top_n=15,
            favorite_win_gate=self.base_model.favorite_win_gate,
            extreme_favorite_win_gate=self.base_model.extreme_favorite_win_gate,
            draw_ceiling=self.base_model.draw_ceiling,
            favorite_lambda_gate=self.base_model.favorite_lambda_gate,
            extreme_lambda_gate=self.base_model.extreme_lambda_gate,
            lambda_gap_gate=self.base_model.lambda_gap_gate,
            total_lambda_gate=self.base_model.total_lambda_gate,
            relative_floor=self.base_model.relative_floor,
            absolute_floor=self.base_model.absolute_floor,
            max_winner_goals=self.base_model.max_winner_goals,
        )
        prediction["top_scorelines"] = top_scorelines
        prediction["v30_adjustments"] = {
            "base_model": "v29_tail_risk_scoreline",
            "scoreline_policy": "player_role_form_plus_tail_risk",
            "role_layer_affects_wdl": True,
            "role_form_wdl_blend": self.role_form_wdl_blend,
            "role_form_scoreline_blend": self.role_form_scoreline_blend,
            "beta_role_attack_edge": self.beta_role_attack_edge,
            "beta_set_piece_edge": self.beta_set_piece_edge,
            "beta_keeper_fragility": self.beta_keeper_fragility,
            "beta_discipline_edge": self.beta_discipline_edge,
            "max_role_log_adjustment": self.max_role_log_adjustment,
            "team_a_role_profile": role_a.diagnostics(),
            "team_b_role_profile": role_b.diagnostics(),
            "base_result_probabilities": base_result_probabilities,
            "role_result_probabilities": role_result_probabilities,
            **role_diagnostics,
            "tail_risk": tail_diagnostics,
        }
        prediction["calibration_notes"] = {
            **prediction.get("calibration_notes", {}),
            "v30": prediction["v30_adjustments"],
            "player_role_policy": (
                "V30 starts from V29, adds small coverage-shrunk player-role "
                "matchup adjustments from FotMob match-level player, lineup, "
                "keeper, and leaderboard data, then re-applies the V29 "
                "tail-risk Top-3 selector."
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
    include_observed_goals=True,
    include_fotmob_goal_stats=True,
    include_group_score_context=True,
    role_form_wdl_blend=DEFAULT_ROLE_FORM_WDL_BLEND,
    role_form_scoreline_blend=DEFAULT_ROLE_FORM_SCORELINE_BLEND,
    beta_role_attack_edge=DEFAULT_BETA_ROLE_ATTACK_EDGE,
    beta_set_piece_edge=DEFAULT_BETA_SET_PIECE_EDGE,
    beta_keeper_fragility=DEFAULT_BETA_KEEPER_FRAGILITY,
    beta_discipline_edge=DEFAULT_BETA_DISCIPLINE_EDGE,
    max_role_log_adjustment=DEFAULT_MAX_ROLE_LOG_ADJUSTMENT,
):
    data_dir = Path(__file__).resolve().parent / "data"
    fotmob_leaders_csv = fotmob_leaders_csv or (
        data_dir / "fotmob_stat_leaders_clean.csv"
    )
    fotmob_player_stats_csv = fotmob_player_stats_csv or (
        data_dir / "fotmob_match_player_stats_clean.csv"
    )
    fotmob_lineups_csv = fotmob_lineups_csv or (
        data_dir / "fotmob_match_lineups_clean.csv"
    )
    fotmob_substitutions_csv = fotmob_substitutions_csv or (
        data_dir / "fotmob_match_substitutions_clean.csv"
    )
    fotmob_keeper_stats_csv = fotmob_keeper_stats_csv or (
        data_dir / "fotmob_match_keeper_stats_clean.csv"
    )
    base_model, data = v29.build_from_zip(
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
        include_observed_goals=include_observed_goals,
        include_fotmob_goal_stats=include_fotmob_goal_stats,
        include_group_score_context=include_group_score_context,
    )
    leaderboard_profiles = build_player_role_profiles(fotmob_leaders_csv)
    match_profiles = build_match_player_role_profiles(
        fotmob_player_stats_csv,
        fotmob_lineups_csv,
        fotmob_substitutions_csv,
        fotmob_keeper_stats_csv,
    )
    role_profiles = merge_role_profiles(leaderboard_profiles, match_profiles)
    model = V30PlayerRoleFormModel(
        base_model,
        role_profiles,
        role_form_wdl_blend=role_form_wdl_blend,
        role_form_scoreline_blend=role_form_scoreline_blend,
        beta_role_attack_edge=beta_role_attack_edge,
        beta_set_piece_edge=beta_set_piece_edge,
        beta_keeper_fragility=beta_keeper_fragility,
        beta_discipline_edge=beta_discipline_edge,
        max_role_log_adjustment=max_role_log_adjustment,
    )
    model.training_data_summary = {
        **getattr(base_model, "training_data_summary", {}),
        "v30_role_profile_teams": len(role_profiles),
        "v30_leaderboard_role_profile_teams": len(leaderboard_profiles),
        "v30_match_role_profile_teams": len(match_profiles),
        "v30_fotmob_leaders_csv": str(fotmob_leaders_csv),
        "v30_fotmob_player_stats_csv": str(fotmob_player_stats_csv),
        "v30_fotmob_lineups_csv": str(fotmob_lineups_csv),
        "v30_fotmob_substitutions_csv": str(fotmob_substitutions_csv),
        "v30_fotmob_keeper_stats_csv": str(fotmob_keeper_stats_csv),
        "v30_role_form_wdl_blend": model.role_form_wdl_blend,
        "v30_role_form_scoreline_blend": model.role_form_scoreline_blend,
        "v30_beta_role_attack_edge": model.beta_role_attack_edge,
        "v30_beta_set_piece_edge": model.beta_set_piece_edge,
        "v30_beta_keeper_fragility": model.beta_keeper_fragility,
        "v30_beta_discipline_edge": model.beta_discipline_edge,
        "v30_max_role_log_adjustment": model.max_role_log_adjustment,
    }
    return model, data


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    parser = argparse.ArgumentParser(
        description="Run V30: V29 with player-role form adjustments."
    )
    parser.add_argument("--team-a", required=True)
    parser.add_argument("--team-b", required=True)
    parser.add_argument("--host-a", action="store_true")
    parser.add_argument("--host-b", action="store_true")
    parser.add_argument("--knockout", action="store_true")
    parser.add_argument("--outdir", default="outputs/outputs_v30_player_role_form")
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
    parser.add_argument("--blind-current-goals", action="store_true")
    parser.add_argument("--ignore-observed-goals", action="store_true")
    parser.add_argument("--ignore-fotmob-goal-stats", action="store_true")
    parser.add_argument("--disable-group-score-context", action="store_true")
    parser.add_argument("--role-form-wdl-blend", type=float, default=DEFAULT_ROLE_FORM_WDL_BLEND)
    parser.add_argument("--role-form-scoreline-blend", type=float, default=DEFAULT_ROLE_FORM_SCORELINE_BLEND)
    parser.add_argument("--beta-role-attack-edge", type=float, default=DEFAULT_BETA_ROLE_ATTACK_EDGE)
    parser.add_argument("--beta-set-piece-edge", type=float, default=DEFAULT_BETA_SET_PIECE_EDGE)
    parser.add_argument("--beta-keeper-fragility", type=float, default=DEFAULT_BETA_KEEPER_FRAGILITY)
    parser.add_argument("--beta-discipline-edge", type=float, default=DEFAULT_BETA_DISCIPLINE_EDGE)
    parser.add_argument("--max-role-log-adjustment", type=float, default=DEFAULT_MAX_ROLE_LOG_ADJUSTMENT)
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
        include_observed_goals=not (
            args.blind_current_goals or args.ignore_observed_goals
        ),
        include_fotmob_goal_stats=not (
            args.blind_current_goals or args.ignore_fotmob_goal_stats
        ),
        include_group_score_context=not (
            args.blind_current_goals or args.disable_group_score_context
        ),
        role_form_wdl_blend=args.role_form_wdl_blend,
        role_form_scoreline_blend=args.role_form_scoreline_blend,
        beta_role_attack_edge=args.beta_role_attack_edge,
        beta_set_piece_edge=args.beta_set_piece_edge,
        beta_keeper_fragility=args.beta_keeper_fragility,
        beta_discipline_edge=args.beta_discipline_edge,
        max_role_log_adjustment=args.max_role_log_adjustment,
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
                "version": "v30-player-role-form",
                "base_model": "v29-tail-risk-scoreline",
                "team_a": prediction["team_a"],
                "team_b": prediction["team_b"],
                "result_probabilities": prediction["result_probabilities"],
                "v30_adjustments": prediction["v30_adjustments"],
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
                "top_scorelines": prediction["top_scorelines"][:5],
                "v30_adjustments": {
                    "role_log_adjustment_a": prediction["v30_adjustments"][
                        "role_log_adjustment_a"
                    ],
                    "role_log_adjustment_b": prediction["v30_adjustments"][
                        "role_log_adjustment_b"
                    ],
                    "tail_risk": prediction["v30_adjustments"]["tail_risk"],
                },
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
'''
v30_player_role_form_model = _load_submodule("v30_player_role_form_model", _V30_PLAYER_ROLE_FORM_MODEL_SOURCE, "feature_layers.py:v30_player_role_form_model")

# ======================================================================
# v31_gated_role_selector_model.py  (bundled as an isolated sub-module)
# ======================================================================
_V31_GATED_ROLE_SELECTOR_MODEL_SOURCE = r'''
#!/usr/bin/env python3
"""V31: V29 base selector plus gated V30 role-data Top-3 overrides.

Run:
    MPLCONFIGDIR=.matplotlib_cache .venv/bin/python v31_gated_role_selector_model.py --team-a "Switzerland" --team-b "Bosnia and Herzegovina" --outdir outputs/outputs_v31_switzerland_bosnia
    MPLCONFIGDIR=.matplotlib_cache .venv/bin/python v31_gated_role_selector_model.py --team-a "Argentina" --team-b "France" --no-plots
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

import v11_wcq_results_model as v11
import v15_catboost_model as v15
import v26_top3_coverage_model as v26
import v27_total_goals_calibrated_model as v27
import v28_current_worldcup_form_model as v28
import v29_tail_risk_scoreline_model as v29
import v30_player_role_form_model as v30


ScoreMatrix = Dict[Tuple[int, int], float]

DEFAULT_ROLE_COVERAGE_GATE = 0.50
DEFAULT_ROLE_SIGNAL_GATE = 0.72
DEFAULT_ROLE_FAVORITE_WIN_GATE = 0.58
DEFAULT_ROLE_DRAW_CEILING = 0.31
DEFAULT_ROLE_RELATIVE_FLOOR = 0.42
DEFAULT_ROLE_ABSOLUTE_FLOOR = 0.035
DEFAULT_ROLE_MAX_REPLACEMENT_INDEX = 2


def score_matrix_from_prediction(prediction: Dict[str, Any]) -> ScoreMatrix:
    return v29.score_matrix_from_prediction(prediction)


def score_item(key: Tuple[int, int], probability: float) -> Dict[str, Any]:
    return v29.score_item(key, probability)


def _score_key(item: Dict[str, Any]) -> Tuple[int, int]:
    return int(item["team_a_goals"]), int(item["team_b_goals"])


def _favorite_role_metrics(
    side: str,
    role_a: v30.PlayerRoleProfile,
    role_b: v30.PlayerRoleProfile,
) -> Dict[str, float]:
    favorite = role_a if side == "team_a" else role_b
    underdog = role_b if side == "team_a" else role_a
    coverage = math.sqrt(
        max(float(favorite.coverage), 0.0) * max(float(underdog.coverage), 0.0)
    )
    attack_mismatch = favorite.attack_unit - underdog.defense_unit
    weak_underdog_defense = (
        underdog.defensive_fragility
        - underdog.defense_unit
        - 0.20 * underdog.keeper
    )
    clean_sheet_edge = (
        favorite.defense_unit
        - underdog.attack_unit
        - 0.25 * underdog.set_piece_for
        - 0.20 * underdog.finishing_delta
    )
    creator_attacker_mismatch = (
        0.55 * favorite.attacker
        + 0.45 * favorite.creator
        - 0.70 * underdog.defender
        - 0.20 * underdog.keeper
        + 0.15 * underdog.defensive_fragility
    )
    favorite_by_three_signal = (
        0.45 * attack_mismatch
        + 0.30 * weak_underdog_defense
        + 0.25 * creator_attacker_mismatch
    )
    clean_sheet_signal = 0.65 * clean_sheet_edge + 0.35 * weak_underdog_defense
    high_goal_clean_sheet_signal = (
        0.55 * favorite_by_three_signal + 0.45 * clean_sheet_signal
    )
    return {
        "role_coverage": float(coverage),
        "attack_mismatch": float(attack_mismatch),
        "weak_underdog_defense": float(weak_underdog_defense),
        "clean_sheet_edge": float(clean_sheet_edge),
        "creator_attacker_mismatch": float(creator_attacker_mismatch),
        "favorite_by_three_signal": float(favorite_by_three_signal),
        "clean_sheet_signal": float(clean_sheet_signal),
        "high_goal_clean_sheet_signal": float(high_goal_clean_sheet_signal),
    }


def _favorite_score(
    side: str,
    favorite_goals: int,
    underdog_goals: int,
) -> Tuple[int, int]:
    if side == "team_a":
        return favorite_goals, underdog_goals
    return underdog_goals, favorite_goals


def _available_candidate(
    score_matrix: ScoreMatrix,
    current_top: list[Tuple[int, int]],
    candidates: list[Tuple[int, int]],
    relative_floor: float,
    absolute_floor: float,
) -> Tuple[int, int] | None:
    current_set = set(current_top)
    top_probability = max(
        [float(score_matrix.get(key, 0.0)) for key in current_top[:3]] or [0.0]
    )
    floor = max(float(absolute_floor), top_probability * float(relative_floor))
    available = [
        key
        for key in candidates
        if key in score_matrix
        and key not in current_set
        and float(score_matrix.get(key, 0.0)) >= floor
    ]
    if not available:
        return None
    return max(available, key=lambda key: score_matrix.get(key, 0.0))


def _replace_third_scoreline(
    selected: list[Tuple[int, int]],
    ranked_keys: list[Tuple[int, int]],
    candidate: Tuple[int, int],
    score_matrix: ScoreMatrix,
    top_n: int,
) -> list[Dict[str, Any]]:
    selected_top = list(selected[:3])
    selected_top[DEFAULT_ROLE_MAX_REPLACEMENT_INDEX] = candidate
    rebuilt: list[Tuple[int, int]] = []
    for key in [*selected_top, *selected, *ranked_keys]:
        if key not in rebuilt and key in score_matrix:
            rebuilt.append(key)
        if len(rebuilt) >= top_n:
            break
    return [score_item(key, score_matrix[key]) for key in rebuilt]


def select_top_scorelines_with_gated_roles(
    score_matrix: ScoreMatrix,
    result_probabilities: Dict[str, float],
    role_a: v30.PlayerRoleProfile,
    role_b: v30.PlayerRoleProfile,
    current_top_scorelines: list[Dict[str, Any]] | None = None,
    top_n: int = 15,
    role_coverage_gate: float = DEFAULT_ROLE_COVERAGE_GATE,
    role_signal_gate: float = DEFAULT_ROLE_SIGNAL_GATE,
    favorite_win_gate: float = DEFAULT_ROLE_FAVORITE_WIN_GATE,
    draw_ceiling: float = DEFAULT_ROLE_DRAW_CEILING,
    relative_floor: float = DEFAULT_ROLE_RELATIVE_FLOOR,
    absolute_floor: float = DEFAULT_ROLE_ABSOLUTE_FLOOR,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    ranked_keys = [
        key
        for key, _ in sorted(
            score_matrix.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]
    selected = (
        [_score_key(item) for item in current_top_scorelines]
        if current_top_scorelines
        else ranked_keys[:top_n]
    )
    diagnostics: Dict[str, Any] = {
        "role_selector_enabled": True,
        "role_selector_applied": False,
        "role_coverage_gate": float(role_coverage_gate),
        "role_signal_gate": float(role_signal_gate),
        "favorite_win_gate": float(favorite_win_gate),
        "draw_ceiling": float(draw_ceiling),
        "relative_floor": float(relative_floor),
        "absolute_floor": float(absolute_floor),
    }
    if len(selected) < 3:
        return [score_item(key, score_matrix[key]) for key in selected], diagnostics

    side = v26.favorite_side(result_probabilities)
    favorite_probability = (
        float(result_probabilities.get("team_a_win", 0.0))
        if side == "team_a"
        else float(result_probabilities.get("team_b_win", 0.0))
        if side == "team_b"
        else 0.0
    )
    diagnostics.update(
        {
            "favorite_side": side,
            "favorite_probability": favorite_probability,
            "draw_probability": float(result_probabilities.get("draw", 0.0)),
        }
    )
    if (
        side is None
        or favorite_probability < float(favorite_win_gate)
        or float(result_probabilities.get("draw", 0.0)) > float(draw_ceiling)
    ):
        diagnostics["skip_reason"] = "result_gates_not_met"
        return [score_item(key, score_matrix[key]) for key in selected], diagnostics

    metrics = _favorite_role_metrics(side, role_a, role_b)
    diagnostics.update(metrics)
    if metrics["role_coverage"] < float(role_coverage_gate):
        diagnostics["skip_reason"] = "coverage_gate_not_met"
        return [score_item(key, score_matrix[key]) for key in selected], diagnostics

    candidates: list[tuple[str, float, list[Tuple[int, int]]]] = []
    if metrics["favorite_by_three_signal"] >= float(role_signal_gate):
        candidates.append(
            (
                "favorite_by_three",
                metrics["favorite_by_three_signal"],
                [
                    _favorite_score(side, 3, 0),
                    _favorite_score(side, 3, 1),
                    _favorite_score(side, 4, 1),
                    _favorite_score(side, 4, 0),
                ],
            )
        )
    if metrics["clean_sheet_signal"] >= float(role_signal_gate):
        candidates.append(
            (
                "clean_sheet_protection",
                metrics["clean_sheet_signal"],
                [
                    _favorite_score(side, 2, 0),
                    _favorite_score(side, 1, 0),
                    _favorite_score(side, 3, 0),
                ],
            )
        )
    if metrics["high_goal_clean_sheet_signal"] >= float(role_signal_gate):
        candidates.append(
            (
                "creator_attacker_clean_sheet_mismatch",
                metrics["high_goal_clean_sheet_signal"],
                [
                    _favorite_score(side, 3, 0),
                    _favorite_score(side, 2, 0),
                    _favorite_score(side, 4, 0),
                ],
            )
        )

    if not candidates:
        diagnostics["skip_reason"] = "role_signal_gate_not_met"
        return [score_item(key, score_matrix[key]) for key in selected], diagnostics

    candidates = sorted(candidates, key=lambda item: item[1], reverse=True)
    for reason, signal, candidate_list in candidates:
        candidate = _available_candidate(
            score_matrix,
            selected[:3],
            candidate_list,
            relative_floor=relative_floor,
            absolute_floor=absolute_floor,
        )
        if candidate is None:
            continue
        diagnostics.update(
            {
                "role_selector_applied": True,
                "role_selector_reason": reason,
                "role_selector_signal": float(signal),
                "candidate_scoreline": f"{candidate[0]}-{candidate[1]}",
                "candidate_probability": float(score_matrix.get(candidate, 0.0)),
                "replaced_third_scoreline": f"{selected[2][0]}-{selected[2][1]}",
            }
        )
        return (
            _replace_third_scoreline(selected, ranked_keys, candidate, score_matrix, top_n),
            diagnostics,
        )

    diagnostics["skip_reason"] = "no_candidate_above_probability_floor"
    return [score_item(key, score_matrix[key]) for key in selected], diagnostics


class V31GatedRoleSelectorModel:
    """Wrap V29 and let V30 role data make only gated Top-3 substitutions."""

    def __init__(
        self,
        base_model: v29.V29TailRiskScorelineModel,
        role_profiles: dict[str, v30.PlayerRoleProfile],
        role_coverage_gate: float = DEFAULT_ROLE_COVERAGE_GATE,
        role_signal_gate: float = DEFAULT_ROLE_SIGNAL_GATE,
        favorite_win_gate: float = DEFAULT_ROLE_FAVORITE_WIN_GATE,
        draw_ceiling: float = DEFAULT_ROLE_DRAW_CEILING,
        relative_floor: float = DEFAULT_ROLE_RELATIVE_FLOOR,
        absolute_floor: float = DEFAULT_ROLE_ABSOLUTE_FLOOR,
    ):
        self.base_model = base_model
        self.role_profiles = role_profiles
        self.role_coverage_gate = float(np.clip(role_coverage_gate, 0.0, 1.0))
        self.role_signal_gate = float(role_signal_gate)
        self.favorite_win_gate = float(np.clip(favorite_win_gate, 0.0, 1.0))
        self.draw_ceiling = float(np.clip(draw_ceiling, 0.0, 1.0))
        self.relative_floor = float(max(relative_floor, 0.0))
        self.absolute_floor = float(max(absolute_floor, 0.0))
        self.training_data_summary = getattr(base_model, "training_data_summary", {})

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_model, name)

    def role_for_team(self, team: object) -> v30.PlayerRoleProfile:
        canonical = v28.canon_team(team)
        return self.role_profiles.get(canonical, v30.PlayerRoleProfile(team=canonical))

    def predict(self, *args, **kwargs) -> Dict[str, Any]:
        team_a = kwargs.get("team_a", args[0] if args else "")
        team_b = kwargs.get("team_b", args[1] if len(args) > 1 else "")
        prediction = self.base_model.predict(*args, **kwargs)
        score_matrix = score_matrix_from_prediction(prediction)
        role_a = self.role_for_team(team_a)
        role_b = self.role_for_team(team_b)
        top_scorelines, role_diagnostics = select_top_scorelines_with_gated_roles(
            score_matrix,
            prediction["result_probabilities"],
            role_a,
            role_b,
            current_top_scorelines=prediction.get("top_scorelines", []),
            top_n=15,
            role_coverage_gate=self.role_coverage_gate,
            role_signal_gate=self.role_signal_gate,
            favorite_win_gate=self.favorite_win_gate,
            draw_ceiling=self.draw_ceiling,
            relative_floor=self.relative_floor,
            absolute_floor=self.absolute_floor,
        )
        prediction["top_scorelines"] = top_scorelines
        prediction["v31_adjustments"] = {
            "base_model": "v29_tail_risk_scoreline",
            "scoreline_policy": "gated_role_top3_selector_only",
            "probability_matrix_changed": False,
            "scoreline_layer_affects_wdl": False,
            "team_a_role_profile": role_a.diagnostics(),
            "team_b_role_profile": role_b.diagnostics(),
            **role_diagnostics,
        }
        prediction["calibration_notes"] = {
            **prediction.get("calibration_notes", {}),
            "v31": prediction["v31_adjustments"],
            "gated_role_policy": (
                "V31 leaves V29 probabilities and W/D/L unchanged. V30 role "
                "data can only replace the third displayed Top-3 scoreline "
                "when coverage, result, role-signal, and probability-floor "
                "gates all pass."
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
    include_observed_goals=True,
    include_fotmob_goal_stats=True,
    include_group_score_context=True,
    role_coverage_gate=DEFAULT_ROLE_COVERAGE_GATE,
    role_signal_gate=DEFAULT_ROLE_SIGNAL_GATE,
    role_favorite_win_gate=DEFAULT_ROLE_FAVORITE_WIN_GATE,
    role_draw_ceiling=DEFAULT_ROLE_DRAW_CEILING,
    role_relative_floor=DEFAULT_ROLE_RELATIVE_FLOOR,
    role_absolute_floor=DEFAULT_ROLE_ABSOLUTE_FLOOR,
    total_calibration_strength=v27.DEFAULT_TOTAL_CALIBRATION_STRENGTH,
    total_multiplier_clip_low=v27.DEFAULT_MULTIPLIER_CLIP_LOW,
    total_multiplier_clip_high=v27.DEFAULT_MULTIPLIER_CLIP_HIGH,
    total_smoothing=v27.DEFAULT_TOTAL_SMOOTHING,
    min_bin_support=v27.DEFAULT_MIN_BIN_SUPPORT,
    max_train_matches=v27.DEFAULT_MAX_TRAIN_MATCHES,
    current_wdl_blend=v28.DEFAULT_CURRENT_WDL_BLEND,
    current_scoreline_blend=v28.DEFAULT_CURRENT_SCORELINE_BLEND,
    beta_attack_edge=v28.DEFAULT_BETA_ATTACK_EDGE,
    beta_tempo=v28.DEFAULT_BETA_TEMPO,
    beta_group_pressure=v28.DEFAULT_BETA_GROUP_PRESSURE,
    max_log_adjustment=v28.DEFAULT_MAX_LOG_ADJUSTMENT,
    total_calibration_blend=v28.DEFAULT_TOTAL_CALIBRATION_BLEND,
    tail_relative_floor=v28.DEFAULT_TAIL_RELATIVE_FLOOR,
    favorite_win_gate=v26.DEFAULT_FAVORITE_WIN_GATE,
    total_lambda_gate=v26.DEFAULT_TOTAL_LAMBDA_GATE,
    favorite_lambda_gate=v26.DEFAULT_FAVORITE_LAMBDA_GATE,
    draw_ceiling=v26.DEFAULT_DRAW_CEILING,
    tail_favorite_win_gate=v29.DEFAULT_TAIL_FAVORITE_WIN_GATE,
    tail_extreme_favorite_win_gate=v29.DEFAULT_TAIL_EXTREME_FAVORITE_WIN_GATE,
    tail_draw_ceiling=v29.DEFAULT_TAIL_DRAW_CEILING,
    tail_favorite_lambda_gate=v29.DEFAULT_TAIL_FAVORITE_LAMBDA_GATE,
    tail_extreme_lambda_gate=v29.DEFAULT_TAIL_EXTREME_LAMBDA_GATE,
    tail_lambda_gap_gate=v29.DEFAULT_TAIL_LAMBDA_GAP_GATE,
    tail_total_lambda_gate=v29.DEFAULT_TAIL_TOTAL_LAMBDA_GATE,
    tail_selector_relative_floor=v29.DEFAULT_TAIL_RELATIVE_FLOOR,
    tail_selector_absolute_floor=v29.DEFAULT_TAIL_ABSOLUTE_FLOOR,
    tail_max_winner_goals=v29.DEFAULT_TAIL_MAX_WINNER_GOALS,
):
    data_dir = Path(__file__).resolve().parent / "data"
    fotmob_leaders_csv = fotmob_leaders_csv or (
        data_dir / "fotmob_stat_leaders_clean.csv"
    )
    fotmob_player_stats_csv = fotmob_player_stats_csv or (
        data_dir / "fotmob_match_player_stats_clean.csv"
    )
    fotmob_lineups_csv = fotmob_lineups_csv or (
        data_dir / "fotmob_match_lineups_clean.csv"
    )
    fotmob_substitutions_csv = fotmob_substitutions_csv or (
        data_dir / "fotmob_match_substitutions_clean.csv"
    )
    fotmob_keeper_stats_csv = fotmob_keeper_stats_csv or (
        data_dir / "fotmob_match_keeper_stats_clean.csv"
    )
    base_model, data = v29.build_from_zip(
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
        include_observed_goals=include_observed_goals,
        include_fotmob_goal_stats=include_fotmob_goal_stats,
        include_group_score_context=include_group_score_context,
        total_calibration_strength=total_calibration_strength,
        total_multiplier_clip_low=total_multiplier_clip_low,
        total_multiplier_clip_high=total_multiplier_clip_high,
        total_smoothing=total_smoothing,
        min_bin_support=min_bin_support,
        max_train_matches=max_train_matches,
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
        tail_favorite_win_gate=tail_favorite_win_gate,
        tail_extreme_favorite_win_gate=tail_extreme_favorite_win_gate,
        tail_draw_ceiling=tail_draw_ceiling,
        tail_favorite_lambda_gate=tail_favorite_lambda_gate,
        tail_extreme_lambda_gate=tail_extreme_lambda_gate,
        tail_lambda_gap_gate=tail_lambda_gap_gate,
        tail_total_lambda_gate=tail_total_lambda_gate,
        tail_selector_relative_floor=tail_selector_relative_floor,
        tail_selector_absolute_floor=tail_selector_absolute_floor,
        tail_max_winner_goals=tail_max_winner_goals,
    )
    leaderboard_profiles = v30.build_player_role_profiles(fotmob_leaders_csv)
    match_profiles = v30.build_match_player_role_profiles(
        fotmob_player_stats_csv,
        fotmob_lineups_csv,
        fotmob_substitutions_csv,
        fotmob_keeper_stats_csv,
    )
    role_profiles = v30.merge_role_profiles(leaderboard_profiles, match_profiles)
    model = V31GatedRoleSelectorModel(
        base_model,
        role_profiles,
        role_coverage_gate=role_coverage_gate,
        role_signal_gate=role_signal_gate,
        favorite_win_gate=role_favorite_win_gate,
        draw_ceiling=role_draw_ceiling,
        relative_floor=role_relative_floor,
        absolute_floor=role_absolute_floor,
    )
    model.training_data_summary = {
        **getattr(base_model, "training_data_summary", {}),
        "v31_scoreline_policy": "gated_role_top3_selector_only",
        "v31_probability_matrix_changed": False,
        "v31_role_profile_teams": len(role_profiles),
        "v31_leaderboard_role_profile_teams": len(leaderboard_profiles),
        "v31_match_role_profile_teams": len(match_profiles),
        "v31_fotmob_leaders_csv": str(fotmob_leaders_csv),
        "v31_fotmob_player_stats_csv": str(fotmob_player_stats_csv),
        "v31_fotmob_lineups_csv": str(fotmob_lineups_csv),
        "v31_fotmob_substitutions_csv": str(fotmob_substitutions_csv),
        "v31_fotmob_keeper_stats_csv": str(fotmob_keeper_stats_csv),
        "v31_role_coverage_gate": model.role_coverage_gate,
        "v31_role_signal_gate": model.role_signal_gate,
        "v31_role_favorite_win_gate": model.favorite_win_gate,
        "v31_role_draw_ceiling": model.draw_ceiling,
        "v31_role_relative_floor": model.relative_floor,
        "v31_role_absolute_floor": model.absolute_floor,
    }
    return model, data


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    parser = argparse.ArgumentParser(
        description="Run V31: V29 with gated V30 role-data Top-3 overrides."
    )
    parser.add_argument("--team-a", required=True)
    parser.add_argument("--team-b", required=True)
    parser.add_argument("--host-a", action="store_true")
    parser.add_argument("--host-b", action="store_true")
    parser.add_argument("--knockout", action="store_true")
    parser.add_argument("--outdir", default="outputs/outputs_v31_gated_role_selector")
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
    parser.add_argument("--blind-current-goals", action="store_true")
    parser.add_argument("--ignore-observed-goals", action="store_true")
    parser.add_argument("--ignore-fotmob-goal-stats", action="store_true")
    parser.add_argument("--disable-group-score-context", action="store_true")
    parser.add_argument("--role-coverage-gate", type=float, default=DEFAULT_ROLE_COVERAGE_GATE)
    parser.add_argument("--role-signal-gate", type=float, default=DEFAULT_ROLE_SIGNAL_GATE)
    parser.add_argument("--role-favorite-win-gate", type=float, default=DEFAULT_ROLE_FAVORITE_WIN_GATE)
    parser.add_argument("--role-draw-ceiling", type=float, default=DEFAULT_ROLE_DRAW_CEILING)
    parser.add_argument("--role-relative-floor", type=float, default=DEFAULT_ROLE_RELATIVE_FLOOR)
    parser.add_argument("--role-absolute-floor", type=float, default=DEFAULT_ROLE_ABSOLUTE_FLOOR)
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
        include_observed_goals=not (
            args.blind_current_goals or args.ignore_observed_goals
        ),
        include_fotmob_goal_stats=not (
            args.blind_current_goals or args.ignore_fotmob_goal_stats
        ),
        include_group_score_context=not (
            args.blind_current_goals or args.disable_group_score_context
        ),
        role_coverage_gate=args.role_coverage_gate,
        role_signal_gate=args.role_signal_gate,
        role_favorite_win_gate=args.role_favorite_win_gate,
        role_draw_ceiling=args.role_draw_ceiling,
        role_relative_floor=args.role_relative_floor,
        role_absolute_floor=args.role_absolute_floor,
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
                "version": "v31-gated-role-selector",
                "base_model": "v29-tail-risk-scoreline",
                "team_a": prediction["team_a"],
                "team_b": prediction["team_b"],
                "result_probabilities": prediction["result_probabilities"],
                "v29_adjustments": prediction.get("v29_adjustments", {}),
                "v31_adjustments": prediction["v31_adjustments"],
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
                "top_scorelines": prediction["top_scorelines"][:5],
                "v29_adjustments": prediction.get("v29_adjustments", {}),
                "v31_adjustments": {
                    key: value
                    for key, value in prediction["v31_adjustments"].items()
                    if key
                    not in {
                        "team_a_role_profile",
                        "team_b_role_profile",
                    }
                },
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
'''
v31_gated_role_selector_model = _load_submodule("v31_gated_role_selector_model", _V31_GATED_ROLE_SELECTOR_MODEL_SOURCE, "feature_layers.py:v31_gated_role_selector_model")

# ======================================================================
# v32_third_slot_coverage_model.py  (bundled as an isolated sub-module)
# ======================================================================
_V32_THIRD_SLOT_COVERAGE_MODEL_SOURCE = r'''
#!/usr/bin/env python3
"""V32: V29 base with a coverage-oriented third-slot selector.

Run:
    MPLCONFIGDIR=.matplotlib_cache .venv/bin/python v32_third_slot_coverage_model.py --team-a "Switzerland" --team-b "Bosnia and Herzegovina" --outdir outputs/outputs_v32_switzerland_bosnia
    MPLCONFIGDIR=.matplotlib_cache .venv/bin/python v32_third_slot_coverage_model.py --eval-observed --eval-outdir observed_eval/observed_eval_v32_third_slot_top3
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd

import v11_wcq_results_model as v11
import v15_catboost_model as v15
import v26_top3_coverage_model as v26
import v27_total_goals_calibrated_model as v27
import v28_current_worldcup_form_model as v28
import v29_tail_risk_scoreline_model as v29
import v30_player_role_form_model as v30
import v31_gated_role_selector_model as v31


ScoreMatrix = Dict[Tuple[int, int], float]

DEFAULT_THIRD_SLOT_RELATIVE_FLOOR = 0.34
DEFAULT_THIRD_SLOT_ABSOLUTE_FLOOR = 0.026
DEFAULT_THIRD_SLOT_MIN_UTILITY_GAIN = 0.09
DEFAULT_THIRD_SLOT_PROBABILITY_WEIGHT = 0.84
DEFAULT_THIRD_SLOT_SIGNAL_WEIGHT = 0.28
DEFAULT_THIRD_SLOT_INCUMBENT_BONUS = 0.035
DEFAULT_THIRD_SLOT_TAIL_INCUMBENT_BONUS = 0.16
DEFAULT_THIRD_SLOT_EXTREME_TAIL_INCUMBENT_BONUS = 0.42
DEFAULT_THIRD_SLOT_RANKED_CANDIDATES = 9


@dataclass
class CurrentXGProfile:
    team: str
    attack_pressure: float = 0.0
    creative_pressure: float = 0.0
    finishing_pressure: float = 0.0
    defensive_leakiness: float = 0.0
    keeper_form: float = 0.0
    coverage: float = 0.0
    matches: int = 0
    rows: int = 0

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "team": self.team,
            "attack_pressure": self.attack_pressure,
            "creative_pressure": self.creative_pressure,
            "finishing_pressure": self.finishing_pressure,
            "defensive_leakiness": self.defensive_leakiness,
            "keeper_form": self.keeper_form,
            "coverage": self.coverage,
            "matches": int(self.matches),
            "rows": int(self.rows),
        }


def score_matrix_from_prediction(prediction: Dict[str, Any]) -> ScoreMatrix:
    return v29.score_matrix_from_prediction(prediction)


def score_item(key: Tuple[int, int], probability: float) -> Dict[str, Any]:
    return v29.score_item(key, probability)


def _score_key(item: Dict[str, Any]) -> Tuple[int, int]:
    return int(item["team_a_goals"]), int(item["team_b_goals"])


def _clip_signal(value: float, low: float = -1.0, high: float = 1.75) -> float:
    return float(np.clip(value, low, high))


def build_current_xg_profiles(
    player_stats_csv: str | Path | None,
    lineups_csv: str | Path | None,
    substitutions_csv: str | Path | None,
    keeper_stats_csv: str | Path | None,
) -> dict[str, CurrentXGProfile]:
    paths = [player_stats_csv, lineups_csv, substitutions_csv]
    if not all(path and Path(path).exists() for path in paths):
        return {}
    player_stats = pd.read_csv(player_stats_csv)
    lineups = pd.read_csv(lineups_csv)
    substitutions = pd.read_csv(substitutions_csv)
    if player_stats.empty or lineups.empty:
        return {}

    player_stats = v30.aggregate_player_match_stats(player_stats)
    enriched = v30.attach_roster_context(player_stats, lineups, substitutions)
    enriched = enriched[enriched["team"].astype(str).ne("")]
    if enriched.empty:
        return {}

    numeric = [
        "xg",
        "xa",
        "xgot",
        "total_shots",
        "shots_on_target",
        "touches_in_opposition_box",
        "chances_created",
    ]
    for column in numeric:
        if column not in enriched:
            enriched[column] = 0.0
        enriched[column] = pd.to_numeric(enriched[column], errors="coerce").fillna(0.0)

    team = (
        enriched.groupby("team", as_index=False)
        .agg(
            rows=("player", "size"),
            matches=("match_id", "nunique"),
            xg=("xg", "sum"),
            xa=("xa", "sum"),
            xgot=("xgot", "sum"),
            total_shots=("total_shots", "sum"),
            shots_on_target=("shots_on_target", "sum"),
            box_touches=("touches_in_opposition_box", "sum"),
            chances_created=("chances_created", "sum"),
        )
        .copy()
    )
    match_denominator = team["matches"].replace(0, 1)
    for column in [
        "xg",
        "xa",
        "xgot",
        "total_shots",
        "shots_on_target",
        "box_touches",
        "chances_created",
    ]:
        team[f"{column}_pm"] = team[column] / match_denominator

    team["attack_pressure"] = v28.zscore(
        team["xg_pm"]
        + 0.45 * team["xgot_pm"]
        + 0.10 * team["shots_on_target_pm"]
        + 0.025 * team["box_touches_pm"]
    )
    team["creative_pressure"] = v28.zscore(
        team["xa_pm"] + 0.16 * team["chances_created_pm"]
    )
    team["finishing_pressure"] = v28.zscore(team["xgot_pm"] - 0.65 * team["xg_pm"])
    team["defensive_leakiness"] = 0.0
    team["keeper_form"] = 0.0

    if keeper_stats_csv and Path(keeper_stats_csv).exists():
        keeper = pd.read_csv(keeper_stats_csv)
        if not keeper.empty and {"match_id", "player"}.issubset(keeper.columns):
            keeper_agg = v30.aggregate_player_match_stats(keeper)
            keeper_enriched = v30.attach_roster_context(
                keeper_agg,
                lineups,
                substitutions,
            )
            keeper_enriched = keeper_enriched[
                keeper_enriched["team"].astype(str).ne("")
            ]
            for column in [
                "saves",
                "goals_conceded",
                "xgot_faced",
                "goals_prevented",
            ]:
                if column not in keeper_enriched:
                    keeper_enriched[column] = 0.0
                keeper_enriched[column] = pd.to_numeric(
                    keeper_enriched[column],
                    errors="coerce",
                ).fillna(0.0)
            if not keeper_enriched.empty:
                keeper_team = (
                    keeper_enriched.groupby("team", as_index=False)
                    .agg(
                        keeper_matches=("match_id", "nunique"),
                        saves=("saves", "sum"),
                        goals_conceded=("goals_conceded", "sum"),
                        xgot_faced=("xgot_faced", "sum"),
                        goals_prevented=("goals_prevented", "sum"),
                    )
                    .copy()
                )
                keeper_denominator = keeper_team["keeper_matches"].replace(0, 1)
                keeper_team["defensive_leakiness"] = v28.zscore(
                    keeper_team["xgot_faced"] / keeper_denominator
                    + 0.35 * keeper_team["goals_conceded"] / keeper_denominator
                    - 0.15 * keeper_team["saves"] / keeper_denominator
                    - 0.25
                    * keeper_team["goals_prevented"]
                    / keeper_denominator
                )
                keeper_team["keeper_form"] = v28.zscore(
                    keeper_team["goals_prevented"] / keeper_denominator
                    + 0.10 * keeper_team["saves"] / keeper_denominator
                )
                team = team.merge(
                    keeper_team[["team", "defensive_leakiness", "keeper_form"]],
                    on="team",
                    how="left",
                    suffixes=("", "_keeper"),
                )
                team["defensive_leakiness"] = team[
                    "defensive_leakiness_keeper"
                ].fillna(team["defensive_leakiness"])
                team["keeper_form"] = team["keeper_form_keeper"].fillna(
                    team["keeper_form"]
                )
                team = team.drop(
                    columns=["defensive_leakiness_keeper", "keeper_form_keeper"]
                )

    coverage = np.sqrt(team["rows"] / (team["rows"] + 8.0))
    coverage *= np.sqrt(team["matches"] / (team["matches"] + 2.0))
    team["coverage"] = np.clip(coverage, 0.0, 1.0)

    profiles: dict[str, CurrentXGProfile] = {}
    for row in team.to_dict(orient="records"):
        coverage_value = float(row.get("coverage", 0.0) or 0.0)
        profiles[str(row["team"])] = CurrentXGProfile(
            team=str(row["team"]),
            attack_pressure=float(row.get("attack_pressure", 0.0)) * coverage_value,
            creative_pressure=float(row.get("creative_pressure", 0.0))
            * coverage_value,
            finishing_pressure=float(row.get("finishing_pressure", 0.0))
            * coverage_value,
            defensive_leakiness=float(row.get("defensive_leakiness", 0.0))
            * coverage_value,
            keeper_form=float(row.get("keeper_form", 0.0)) * coverage_value,
            coverage=coverage_value,
            matches=int(row.get("matches", 0) or 0),
            rows=int(row.get("rows", 0) or 0),
        )
    return profiles


def _favorite_xg_metrics(
    side: str,
    xg_a: CurrentXGProfile,
    xg_b: CurrentXGProfile,
) -> Dict[str, float]:
    favorite = xg_a if side == "team_a" else xg_b
    underdog = xg_b if side == "team_a" else xg_a
    coverage = math.sqrt(
        max(float(favorite.coverage), 0.0) * max(float(underdog.coverage), 0.0)
    )
    favorite_pressure = (
        0.62 * favorite.attack_pressure
        + 0.38 * favorite.creative_pressure
        + 0.35 * underdog.defensive_leakiness
    )
    underdog_score_pressure = (
        0.70 * underdog.attack_pressure
        + 0.30 * underdog.creative_pressure
        + 0.35 * favorite.defensive_leakiness
        - 0.25 * favorite.keeper_form
    )
    clean_sheet_xg_signal = -underdog_score_pressure
    favorite_by_three_xg_signal = (
        0.70 * favorite_pressure + 0.30 * clean_sheet_xg_signal
    )
    return {
        "xg_coverage": float(coverage),
        "favorite_pressure": float(favorite_pressure),
        "underdog_score_pressure": float(underdog_score_pressure),
        "clean_sheet_xg_signal": float(clean_sheet_xg_signal),
        "favorite_by_three_xg_signal": float(favorite_by_three_xg_signal),
    }


def _favorite_score(side: str, favorite_goals: int, underdog_goals: int) -> Tuple[int, int]:
    return v31._favorite_score(side, favorite_goals, underdog_goals)


def _score_outcome(key: Tuple[int, int]) -> str:
    if key[0] > key[1]:
        return "team_a_win"
    if key[1] > key[0]:
        return "team_b_win"
    return "draw"


def _candidate_pool(
    ranked_keys: list[Tuple[int, int]],
    selected: list[Tuple[int, int]],
    result_probabilities: Dict[str, float],
    ranked_limit: int,
) -> list[Tuple[int, int]]:
    pool: list[Tuple[int, int]] = []
    for key in [*selected[:3], *ranked_keys[:ranked_limit]]:
        if key not in pool:
            pool.append(key)

    side = v26.favorite_side(result_probabilities)
    if side is not None:
        structured = [
            _favorite_score(side, 1, 0),
            _favorite_score(side, 2, 0),
            _favorite_score(side, 2, 1),
            _favorite_score(side, 3, 0),
            _favorite_score(side, 3, 1),
            _favorite_score(side, 4, 0),
            _favorite_score(side, 4, 1),
        ]
        for key in structured:
            if key not in pool:
                pool.append(key)

    if max(result_probabilities.values()) < 0.52 or result_probabilities.get(
        "draw",
        0.0,
    ) >= 0.24:
        for key in [(0, 0), (1, 1), (2, 2), (1, 0), (0, 1), (2, 1), (1, 2)]:
            if key not in pool:
                pool.append(key)
    return pool


def _favorite_goal_view(
    key: Tuple[int, int],
    side: str | None,
) -> tuple[int, int, int]:
    if side == "team_a":
        favorite_goals, underdog_goals = key
    elif side == "team_b":
        underdog_goals, favorite_goals = key
    else:
        favorite_goals, underdog_goals = max(key), min(key)
    return favorite_goals, underdog_goals, favorite_goals - underdog_goals


def candidate_coverage_utility(
    key: Tuple[int, int],
    score_matrix: ScoreMatrix,
    selected: list[Tuple[int, int]],
    result_probabilities: Dict[str, float],
    role_metrics: Dict[str, float],
    xg_metrics: Dict[str, float],
    probability_weight: float = DEFAULT_THIRD_SLOT_PROBABILITY_WEIGHT,
    signal_weight: float = DEFAULT_THIRD_SLOT_SIGNAL_WEIGHT,
    incumbent_bonus: float = DEFAULT_THIRD_SLOT_INCUMBENT_BONUS,
    tail_incumbent_bonus: float = DEFAULT_THIRD_SLOT_TAIL_INCUMBENT_BONUS,
    extreme_tail_incumbent_bonus: float = DEFAULT_THIRD_SLOT_EXTREME_TAIL_INCUMBENT_BONUS,
) -> Dict[str, float]:
    side = v26.favorite_side(result_probabilities)
    top_probability = max(float(score_matrix.get(selected[0], 0.0)), 1e-12)
    probability_ratio = float(score_matrix.get(key, 0.0)) / top_probability
    probability_component = float(probability_weight) * math.sqrt(
        max(probability_ratio, 0.0)
    )
    favorite_goals, underdog_goals, margin = _favorite_goal_view(key, side)
    predicted_outcome = max(result_probabilities, key=result_probabilities.get)
    outcome_component = 0.06 if _score_outcome(key) == predicted_outcome else 0.0
    normal_total_component = 0.035 if 2 <= sum(key) <= 4 else 0.0
    if key == selected[2]:
        outcome_component += float(incumbent_bonus)
        if side is not None and v29.is_favorite_tail_score(key, side):
            outcome_component += float(tail_incumbent_bonus)
            if max(key) >= 5:
                outcome_component += float(extreme_tail_incumbent_bonus)

    role_tail = role_metrics.get("favorite_by_three_signal", 0.0)
    role_clean = role_metrics.get("clean_sheet_signal", 0.0)
    xg_tail = xg_metrics.get("favorite_by_three_xg_signal", 0.0)
    xg_clean = xg_metrics.get("clean_sheet_xg_signal", 0.0)
    role_coverage = role_metrics.get("role_coverage", 0.0)
    xg_coverage = xg_metrics.get("xg_coverage", 0.0)
    signal_coverage = math.sqrt(max(role_coverage, 0.0) * max(xg_coverage, 0.0))
    tail_signal = signal_coverage * (
        0.55 * _clip_signal(role_tail) + 0.45 * _clip_signal(xg_tail)
    )
    clean_signal = signal_coverage * (
        0.48 * _clip_signal(role_clean) + 0.52 * _clip_signal(xg_clean)
    )
    both_score_signal = signal_coverage * _clip_signal(
        -0.45 * role_clean
        - 0.55 * xg_clean
        + 0.18 * xg_metrics.get("underdog_score_pressure", 0.0)
    )

    signal_component = 0.0
    if side is not None and margin > 0:
        if favorite_goals >= 3 and margin >= 2:
            signal_component += float(signal_weight) * tail_signal
        if underdog_goals == 0:
            signal_component += float(signal_weight) * 0.82 * clean_signal
        elif underdog_goals == 1:
            signal_component += float(signal_weight) * 0.46 * both_score_signal
        if favorite_goals == 2 and underdog_goals == 0:
            signal_component += float(signal_weight) * 0.24 * clean_signal

    utility = (
        probability_component
        + outcome_component
        + normal_total_component
        + signal_component
    )
    return {
        "utility": float(utility),
        "probability_ratio": float(probability_ratio),
        "probability_component": float(probability_component),
        "signal_component": float(signal_component),
        "outcome_component": float(outcome_component),
        "normal_total_component": float(normal_total_component),
        "tail_signal": float(tail_signal),
        "clean_signal": float(clean_signal),
        "both_score_signal": float(both_score_signal),
    }


def _rebuild_top_scorelines(
    selected: list[Tuple[int, int]],
    ranked_keys: list[Tuple[int, int]],
    candidate: Tuple[int, int],
    score_matrix: ScoreMatrix,
    top_n: int,
) -> list[Dict[str, Any]]:
    top_three = [selected[0], selected[1], candidate]
    rebuilt: list[Tuple[int, int]] = []
    for key in [*top_three, *selected, *ranked_keys]:
        if key not in rebuilt and key in score_matrix:
            rebuilt.append(key)
        if len(rebuilt) >= top_n:
            break
    return [score_item(key, score_matrix[key]) for key in rebuilt]


def select_top_scorelines_with_third_slot_coverage(
    score_matrix: ScoreMatrix,
    result_probabilities: Dict[str, float],
    role_a: v30.PlayerRoleProfile,
    role_b: v30.PlayerRoleProfile,
    xg_a: CurrentXGProfile,
    xg_b: CurrentXGProfile,
    current_top_scorelines: list[Dict[str, Any]] | None = None,
    top_n: int = 15,
    relative_floor: float = DEFAULT_THIRD_SLOT_RELATIVE_FLOOR,
    absolute_floor: float = DEFAULT_THIRD_SLOT_ABSOLUTE_FLOOR,
    min_utility_gain: float = DEFAULT_THIRD_SLOT_MIN_UTILITY_GAIN,
    probability_weight: float = DEFAULT_THIRD_SLOT_PROBABILITY_WEIGHT,
    signal_weight: float = DEFAULT_THIRD_SLOT_SIGNAL_WEIGHT,
    incumbent_bonus: float = DEFAULT_THIRD_SLOT_INCUMBENT_BONUS,
    tail_incumbent_bonus: float = DEFAULT_THIRD_SLOT_TAIL_INCUMBENT_BONUS,
    extreme_tail_incumbent_bonus: float = DEFAULT_THIRD_SLOT_EXTREME_TAIL_INCUMBENT_BONUS,
    ranked_candidates: int = DEFAULT_THIRD_SLOT_RANKED_CANDIDATES,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    ranked_keys = [
        key
        for key, _ in sorted(
            score_matrix.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]
    selected = (
        [_score_key(item) for item in current_top_scorelines]
        if current_top_scorelines
        else ranked_keys[:top_n]
    )
    diagnostics: Dict[str, Any] = {
        "third_slot_selector_enabled": True,
        "third_slot_changed": False,
        "relative_floor": float(relative_floor),
        "absolute_floor": float(absolute_floor),
        "min_utility_gain": float(min_utility_gain),
        "probability_weight": float(probability_weight),
        "signal_weight": float(signal_weight),
        "incumbent_bonus": float(incumbent_bonus),
        "tail_incumbent_bonus": float(tail_incumbent_bonus),
        "extreme_tail_incumbent_bonus": float(extreme_tail_incumbent_bonus),
        "ranked_candidates": int(ranked_candidates),
    }
    if len(selected) < 3:
        return [score_item(key, score_matrix[key]) for key in selected], diagnostics

    side = v26.favorite_side(result_probabilities)
    role_metrics = (
        v31._favorite_role_metrics(side, role_a, role_b) if side else {}
    )
    xg_metrics = _favorite_xg_metrics(side, xg_a, xg_b) if side else {}
    diagnostics.update(
        {
            "favorite_side": side,
            "base_top_3": [f"{key[0]}-{key[1]}" for key in selected[:3]],
            "role_metrics": role_metrics,
            "xg_metrics": xg_metrics,
        }
    )
    pool = [
        key
        for key in _candidate_pool(
            ranked_keys,
            selected,
            result_probabilities,
            int(ranked_candidates),
        )
        if key in score_matrix and key not in set(selected[:2])
    ]
    top_probability = max(float(score_matrix.get(selected[0], 0.0)), 1e-12)
    floor = max(float(absolute_floor), top_probability * float(relative_floor))
    pool = [key for key in pool if float(score_matrix.get(key, 0.0)) >= floor]
    if selected[2] not in pool and selected[2] in score_matrix:
        pool.append(selected[2])
    if not pool:
        diagnostics["skip_reason"] = "no_candidate_above_floor"
        return [score_item(key, score_matrix[key]) for key in selected], diagnostics

    utilities = {
        key: candidate_coverage_utility(
            key,
            score_matrix,
            selected,
            result_probabilities,
            role_metrics,
            xg_metrics,
            probability_weight=probability_weight,
            signal_weight=signal_weight,
            incumbent_bonus=incumbent_bonus,
            tail_incumbent_bonus=tail_incumbent_bonus,
            extreme_tail_incumbent_bonus=extreme_tail_incumbent_bonus,
        )
        for key in pool
    }
    incumbent = selected[2]
    best = max(pool, key=lambda key: utilities[key]["utility"])
    incumbent_utility = utilities.get(
        incumbent,
        candidate_coverage_utility(
            incumbent,
            score_matrix,
            selected,
            result_probabilities,
            role_metrics,
            xg_metrics,
            probability_weight=probability_weight,
            signal_weight=signal_weight,
            incumbent_bonus=incumbent_bonus,
            tail_incumbent_bonus=tail_incumbent_bonus,
            extreme_tail_incumbent_bonus=extreme_tail_incumbent_bonus,
        ),
    )["utility"]
    best_utility = utilities[best]["utility"]
    diagnostics.update(
        {
            "candidate_count": len(pool),
            "incumbent_third_scoreline": f"{incumbent[0]}-{incumbent[1]}",
            "incumbent_utility": float(incumbent_utility),
            "best_candidate_scoreline": f"{best[0]}-{best[1]}",
            "best_candidate_probability": float(score_matrix.get(best, 0.0)),
            "best_candidate_utility": float(best_utility),
            "best_candidate_details": utilities[best],
        }
    )
    if best == incumbent:
        diagnostics["skip_reason"] = "incumbent_best"
        return [score_item(key, score_matrix[key]) for key in selected], diagnostics
    if best_utility < incumbent_utility + float(min_utility_gain):
        diagnostics["skip_reason"] = "utility_gain_too_small"
        return [score_item(key, score_matrix[key]) for key in selected], diagnostics

    diagnostics["third_slot_changed"] = True
    diagnostics["replaced_third_scoreline"] = f"{incumbent[0]}-{incumbent[1]}"
    diagnostics["new_third_scoreline"] = f"{best[0]}-{best[1]}"
    return _rebuild_top_scorelines(selected, ranked_keys, best, score_matrix, top_n), diagnostics


class V32ThirdSlotCoverageModel:
    """Wrap V29 and choose the third Top-3 slot by coverage utility."""

    def __init__(
        self,
        base_model: v29.V29TailRiskScorelineModel,
        role_profiles: dict[str, v30.PlayerRoleProfile],
        xg_profiles: dict[str, CurrentXGProfile],
        relative_floor: float = DEFAULT_THIRD_SLOT_RELATIVE_FLOOR,
        absolute_floor: float = DEFAULT_THIRD_SLOT_ABSOLUTE_FLOOR,
        min_utility_gain: float = DEFAULT_THIRD_SLOT_MIN_UTILITY_GAIN,
        probability_weight: float = DEFAULT_THIRD_SLOT_PROBABILITY_WEIGHT,
        signal_weight: float = DEFAULT_THIRD_SLOT_SIGNAL_WEIGHT,
        incumbent_bonus: float = DEFAULT_THIRD_SLOT_INCUMBENT_BONUS,
        tail_incumbent_bonus: float = DEFAULT_THIRD_SLOT_TAIL_INCUMBENT_BONUS,
        extreme_tail_incumbent_bonus: float = DEFAULT_THIRD_SLOT_EXTREME_TAIL_INCUMBENT_BONUS,
        ranked_candidates: int = DEFAULT_THIRD_SLOT_RANKED_CANDIDATES,
    ):
        self.base_model = base_model
        self.role_profiles = role_profiles
        self.xg_profiles = xg_profiles
        self.relative_floor = float(max(relative_floor, 0.0))
        self.absolute_floor = float(max(absolute_floor, 0.0))
        self.min_utility_gain = float(max(min_utility_gain, 0.0))
        self.probability_weight = float(max(probability_weight, 0.0))
        self.signal_weight = float(signal_weight)
        self.incumbent_bonus = float(max(incumbent_bonus, 0.0))
        self.tail_incumbent_bonus = float(max(tail_incumbent_bonus, 0.0))
        self.extreme_tail_incumbent_bonus = float(
            max(extreme_tail_incumbent_bonus, 0.0)
        )
        self.ranked_candidates = int(max(ranked_candidates, 3))
        self.training_data_summary = getattr(base_model, "training_data_summary", {})

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_model, name)

    def role_for_team(self, team: object) -> v30.PlayerRoleProfile:
        canonical = v28.canon_team(team)
        return self.role_profiles.get(canonical, v30.PlayerRoleProfile(team=canonical))

    def xg_for_team(self, team: object) -> CurrentXGProfile:
        canonical = v28.canon_team(team)
        return self.xg_profiles.get(canonical, CurrentXGProfile(team=canonical))

    def predict(self, *args, **kwargs) -> Dict[str, Any]:
        team_a = kwargs.get("team_a", args[0] if args else "")
        team_b = kwargs.get("team_b", args[1] if len(args) > 1 else "")
        prediction = self.base_model.predict(*args, **kwargs)
        score_matrix = score_matrix_from_prediction(prediction)
        role_a = self.role_for_team(team_a)
        role_b = self.role_for_team(team_b)
        xg_a = self.xg_for_team(team_a)
        xg_b = self.xg_for_team(team_b)
        top_scorelines, diagnostics = select_top_scorelines_with_third_slot_coverage(
            score_matrix,
            prediction["result_probabilities"],
            role_a,
            role_b,
            xg_a,
            xg_b,
            current_top_scorelines=prediction.get("top_scorelines", []),
            top_n=15,
            relative_floor=self.relative_floor,
            absolute_floor=self.absolute_floor,
            min_utility_gain=self.min_utility_gain,
            probability_weight=self.probability_weight,
            signal_weight=self.signal_weight,
            incumbent_bonus=self.incumbent_bonus,
            tail_incumbent_bonus=self.tail_incumbent_bonus,
            extreme_tail_incumbent_bonus=self.extreme_tail_incumbent_bonus,
            ranked_candidates=self.ranked_candidates,
        )
        prediction["top_scorelines"] = top_scorelines
        prediction["v32_adjustments"] = {
            "base_model": "v29_tail_risk_scoreline",
            "scoreline_policy": "third_slot_coverage_selector_only",
            "probability_matrix_changed": False,
            "scoreline_layer_affects_wdl": False,
            "team_a_role_profile": role_a.diagnostics(),
            "team_b_role_profile": role_b.diagnostics(),
            "team_a_xg_profile": xg_a.diagnostics(),
            "team_b_xg_profile": xg_b.diagnostics(),
            **diagnostics,
        }
        prediction["calibration_notes"] = {
            **prediction.get("calibration_notes", {}),
            "v32": prediction["v32_adjustments"],
            "third_slot_policy": (
                "V32 leaves V29 probabilities and W/D/L unchanged. It keeps "
                "the first two displayed scorelines and ranks only the third "
                "slot using probability plus player-role, xG/xA, xGOT, and "
                "keeper/concession coverage signals."
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
    include_observed_goals=True,
    include_fotmob_goal_stats=True,
    include_group_score_context=True,
    third_slot_relative_floor=DEFAULT_THIRD_SLOT_RELATIVE_FLOOR,
    third_slot_absolute_floor=DEFAULT_THIRD_SLOT_ABSOLUTE_FLOOR,
    third_slot_min_utility_gain=DEFAULT_THIRD_SLOT_MIN_UTILITY_GAIN,
    third_slot_probability_weight=DEFAULT_THIRD_SLOT_PROBABILITY_WEIGHT,
    third_slot_signal_weight=DEFAULT_THIRD_SLOT_SIGNAL_WEIGHT,
    third_slot_incumbent_bonus=DEFAULT_THIRD_SLOT_INCUMBENT_BONUS,
    third_slot_tail_incumbent_bonus=DEFAULT_THIRD_SLOT_TAIL_INCUMBENT_BONUS,
    third_slot_extreme_tail_incumbent_bonus=DEFAULT_THIRD_SLOT_EXTREME_TAIL_INCUMBENT_BONUS,
    third_slot_ranked_candidates=DEFAULT_THIRD_SLOT_RANKED_CANDIDATES,
    total_calibration_strength=v27.DEFAULT_TOTAL_CALIBRATION_STRENGTH,
    total_multiplier_clip_low=v27.DEFAULT_MULTIPLIER_CLIP_LOW,
    total_multiplier_clip_high=v27.DEFAULT_MULTIPLIER_CLIP_HIGH,
    total_smoothing=v27.DEFAULT_TOTAL_SMOOTHING,
    min_bin_support=v27.DEFAULT_MIN_BIN_SUPPORT,
    max_train_matches=v27.DEFAULT_MAX_TRAIN_MATCHES,
    current_wdl_blend=v28.DEFAULT_CURRENT_WDL_BLEND,
    current_scoreline_blend=v28.DEFAULT_CURRENT_SCORELINE_BLEND,
    beta_attack_edge=v28.DEFAULT_BETA_ATTACK_EDGE,
    beta_tempo=v28.DEFAULT_BETA_TEMPO,
    beta_group_pressure=v28.DEFAULT_BETA_GROUP_PRESSURE,
    max_log_adjustment=v28.DEFAULT_MAX_LOG_ADJUSTMENT,
    total_calibration_blend=v28.DEFAULT_TOTAL_CALIBRATION_BLEND,
    tail_relative_floor=v28.DEFAULT_TAIL_RELATIVE_FLOOR,
    favorite_win_gate=v26.DEFAULT_FAVORITE_WIN_GATE,
    total_lambda_gate=v26.DEFAULT_TOTAL_LAMBDA_GATE,
    favorite_lambda_gate=v26.DEFAULT_FAVORITE_LAMBDA_GATE,
    draw_ceiling=v26.DEFAULT_DRAW_CEILING,
    tail_favorite_win_gate=v29.DEFAULT_TAIL_FAVORITE_WIN_GATE,
    tail_extreme_favorite_win_gate=v29.DEFAULT_TAIL_EXTREME_FAVORITE_WIN_GATE,
    tail_draw_ceiling=v29.DEFAULT_TAIL_DRAW_CEILING,
    tail_favorite_lambda_gate=v29.DEFAULT_TAIL_FAVORITE_LAMBDA_GATE,
    tail_extreme_lambda_gate=v29.DEFAULT_TAIL_EXTREME_LAMBDA_GATE,
    tail_lambda_gap_gate=v29.DEFAULT_TAIL_LAMBDA_GAP_GATE,
    tail_total_lambda_gate=v29.DEFAULT_TAIL_TOTAL_LAMBDA_GATE,
    tail_selector_relative_floor=v29.DEFAULT_TAIL_RELATIVE_FLOOR,
    tail_selector_absolute_floor=v29.DEFAULT_TAIL_ABSOLUTE_FLOOR,
    tail_max_winner_goals=v29.DEFAULT_TAIL_MAX_WINNER_GOALS,
):
    data_dir = Path(__file__).resolve().parent / "data"
    fotmob_leaders_csv = fotmob_leaders_csv or (
        data_dir / "fotmob_stat_leaders_clean.csv"
    )
    fotmob_player_stats_csv = fotmob_player_stats_csv or (
        data_dir / "fotmob_match_player_stats_clean.csv"
    )
    fotmob_lineups_csv = fotmob_lineups_csv or (
        data_dir / "fotmob_match_lineups_clean.csv"
    )
    fotmob_substitutions_csv = fotmob_substitutions_csv or (
        data_dir / "fotmob_match_substitutions_clean.csv"
    )
    fotmob_keeper_stats_csv = fotmob_keeper_stats_csv or (
        data_dir / "fotmob_match_keeper_stats_clean.csv"
    )
    base_model, data = v29.build_from_zip(
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
        include_observed_goals=include_observed_goals,
        include_fotmob_goal_stats=include_fotmob_goal_stats,
        include_group_score_context=include_group_score_context,
        total_calibration_strength=total_calibration_strength,
        total_multiplier_clip_low=total_multiplier_clip_low,
        total_multiplier_clip_high=total_multiplier_clip_high,
        total_smoothing=total_smoothing,
        min_bin_support=min_bin_support,
        max_train_matches=max_train_matches,
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
        tail_favorite_win_gate=tail_favorite_win_gate,
        tail_extreme_favorite_win_gate=tail_extreme_favorite_win_gate,
        tail_draw_ceiling=tail_draw_ceiling,
        tail_favorite_lambda_gate=tail_favorite_lambda_gate,
        tail_extreme_lambda_gate=tail_extreme_lambda_gate,
        tail_lambda_gap_gate=tail_lambda_gap_gate,
        tail_total_lambda_gate=tail_total_lambda_gate,
        tail_selector_relative_floor=tail_selector_relative_floor,
        tail_selector_absolute_floor=tail_selector_absolute_floor,
        tail_max_winner_goals=tail_max_winner_goals,
    )
    leaderboard_profiles = v30.build_player_role_profiles(fotmob_leaders_csv)
    match_profiles = v30.build_match_player_role_profiles(
        fotmob_player_stats_csv,
        fotmob_lineups_csv,
        fotmob_substitutions_csv,
        fotmob_keeper_stats_csv,
    )
    role_profiles = v30.merge_role_profiles(leaderboard_profiles, match_profiles)
    xg_profiles = build_current_xg_profiles(
        fotmob_player_stats_csv,
        fotmob_lineups_csv,
        fotmob_substitutions_csv,
        fotmob_keeper_stats_csv,
    )
    model = V32ThirdSlotCoverageModel(
        base_model,
        role_profiles,
        xg_profiles,
        relative_floor=third_slot_relative_floor,
        absolute_floor=third_slot_absolute_floor,
        min_utility_gain=third_slot_min_utility_gain,
        probability_weight=third_slot_probability_weight,
        signal_weight=third_slot_signal_weight,
        incumbent_bonus=third_slot_incumbent_bonus,
        tail_incumbent_bonus=third_slot_tail_incumbent_bonus,
        extreme_tail_incumbent_bonus=third_slot_extreme_tail_incumbent_bonus,
        ranked_candidates=third_slot_ranked_candidates,
    )
    model.training_data_summary = {
        **getattr(base_model, "training_data_summary", {}),
        "v32_scoreline_policy": "third_slot_coverage_selector_only",
        "v32_probability_matrix_changed": False,
        "v32_role_profile_teams": len(role_profiles),
        "v32_match_role_profile_teams": len(match_profiles),
        "v32_xg_profile_teams": len(xg_profiles),
        "v32_fotmob_leaders_csv": str(fotmob_leaders_csv),
        "v32_fotmob_player_stats_csv": str(fotmob_player_stats_csv),
        "v32_fotmob_lineups_csv": str(fotmob_lineups_csv),
        "v32_fotmob_substitutions_csv": str(fotmob_substitutions_csv),
        "v32_fotmob_keeper_stats_csv": str(fotmob_keeper_stats_csv),
        "v32_third_slot_relative_floor": model.relative_floor,
        "v32_third_slot_absolute_floor": model.absolute_floor,
        "v32_third_slot_min_utility_gain": model.min_utility_gain,
        "v32_third_slot_probability_weight": model.probability_weight,
        "v32_third_slot_signal_weight": model.signal_weight,
        "v32_third_slot_incumbent_bonus": model.incumbent_bonus,
        "v32_third_slot_tail_incumbent_bonus": model.tail_incumbent_bonus,
        "v32_third_slot_extreme_tail_incumbent_bonus": model.extreme_tail_incumbent_bonus,
        "v32_third_slot_ranked_candidates": model.ranked_candidates,
    }
    return model, data


def evaluate_observed_matches(
    model: V32ThirdSlotCoverageModel,
    observed_matches_csv: str | Path,
    outdir: str | Path,
    limit: int = 20,
) -> Dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    observed = pd.read_csv(observed_matches_csv).head(limit)
    rows = []
    for idx, row in observed.iterrows():
        prediction = model.predict(row["team_a"], row["team_b"])
        score_matrix = score_matrix_from_prediction(prediction)
        actual = f"{int(row['goals_a'])}-{int(row['goals_b'])}"
        actual_key = (int(row["goals_a"]), int(row["goals_b"]))
        top = [
            f"{int(item['team_a_goals'])}-{int(item['team_b_goals'])}"
            for item in prediction["top_scorelines"][:3]
        ]
        probs = [
            float(item["probability"])
            for item in prediction["top_scorelines"][:3]
        ]
        base_top = prediction["v32_adjustments"].get("base_top_3", top)
        base_hit = actual in base_top
        final_hit = actual in top
        actual_result = (
            "team_a_win"
            if row["goals_a"] > row["goals_b"]
            else "team_b_win"
            if row["goals_b"] > row["goals_a"]
            else "draw"
        )
        result_probabilities = prediction["result_probabilities"]
        predicted_result = prediction["predicted_result"]
        result_label = {
            "team_a_win": row["team_a"],
            "team_b_win": row["team_b"],
            "draw": "Draw",
        }
        changed = bool(prediction["v32_adjustments"].get("third_slot_changed", False))
        rows.append(
            {
                "observed_order": idx + 1,
                "match_id": row["match_id"],
                "team_a": row["team_a"],
                "team_b": row["team_b"],
                "actual_score": actual,
                "base_top_1_scoreline": base_top[0],
                "base_top_2_scoreline": base_top[1],
                "base_top_3_scoreline": base_top[2],
                "top_1_scoreline": top[0],
                "top_1_probability": probs[0],
                "top_2_scoreline": top[1],
                "top_2_probability": probs[1],
                "top_3_scoreline": top[2],
                "top_3_probability": probs[2],
                "actual_is_top_1": actual == top[0],
                "actual_is_top_2": actual == top[1],
                "actual_is_top_3": actual == top[2],
                "actual_in_top_3": final_hit,
                "actual_score_probability": float(score_matrix.get(actual_key, 0.0)),
                "base_actual_in_top_3": base_hit,
                "third_slot_changed": changed,
                "change_gained_hit": changed and final_hit and not base_hit,
                "change_lost_hit": changed and base_hit and not final_hit,
                "change_kept_hit": changed and final_hit and base_hit,
                "change_kept_miss": changed and not final_hit and not base_hit,
                "replaced_third_scoreline": prediction["v32_adjustments"].get(
                    "replaced_third_scoreline",
                    "",
                ),
                "new_third_scoreline": prediction["v32_adjustments"].get(
                    "new_third_scoreline",
                    "",
                ),
                "best_candidate_utility": prediction["v32_adjustments"].get(
                    "best_candidate_utility",
                    "",
                ),
                "incumbent_utility": prediction["v32_adjustments"].get(
                    "incumbent_utility",
                    "",
                ),
                "actual_result": actual_result,
                "actual_result_label": result_label[actual_result],
                "predicted_result": predicted_result,
                "predicted_result_label": result_label[predicted_result],
                "outcome_correct": predicted_result == actual_result,
                "predicted_result_probability": float(
                    result_probabilities[predicted_result]
                ),
                "actual_result_probability": float(result_probabilities[actual_result]),
                "team_a_win_probability": float(result_probabilities["team_a_win"]),
                "draw_probability": float(result_probabilities["draw"]),
                "team_b_win_probability": float(result_probabilities["team_b_win"]),
                "v29_tail_applied": bool(
                    prediction.get("v29_adjustments", {}).get(
                        "tail_risk_applied",
                        False,
                    )
                ),
            }
        )
    eval_df = pd.DataFrame(rows)
    csv_path = outdir / "v32_third_slot_top_three_scoreline_comparison_20_matches.csv"
    eval_df.to_csv(csv_path, index=False)
    normal = eval_df[
        eval_df["actual_score"].str.split("-").apply(
            lambda parts: int(parts[0]) <= 3 and int(parts[1]) <= 3
        )
    ]
    summary = {
        "model": "v32_third_slot_coverage",
        "n_matches": int(len(eval_df)),
        "top_1_exact_score_hits": int(eval_df["actual_is_top_1"].sum()),
        "top_2_exact_score_hits": int(
            (eval_df["actual_is_top_1"] | eval_df["actual_is_top_2"]).sum()
        ),
        "top_3_exact_score_hits": int(eval_df["actual_in_top_3"].sum()),
        "top_3_exact_score_accuracy": float(eval_df["actual_in_top_3"].mean()),
        "outcome_correct": int(eval_df["outcome_correct"].sum()),
        "normal_range_matches": int(len(normal)),
        "normal_range_top_3_hits": int(normal["actual_in_top_3"].sum()),
        "normal_range_top_3_accuracy": float(normal["actual_in_top_3"].mean()),
        "tail_risk_applied_matches": int(eval_df["v29_tail_applied"].sum()),
        "third_slot_changed_matches": int(eval_df["third_slot_changed"].sum()),
        "changed_matches_gained_hits": int(eval_df["change_gained_hit"].sum()),
        "changed_matches_lost_hits": int(eval_df["change_lost_hit"].sum()),
        "changed_matches_kept_hits": int(eval_df["change_kept_hit"].sum()),
        "changed_matches_kept_misses": int(eval_df["change_kept_miss"].sum()),
        "csv": str(csv_path),
    }
    summary_path = outdir / "v32_third_slot_top3_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    try:
        import compare_v11_top_scorelines as scoreline_chart

        png_path = outdir / "v32_third_slot_top_three_scoreline_comparison_20_matches.png"
        chart_df = eval_df.copy()
        chart_df.attrs["model_label"] = "V32 third-slot coverage"
        chart_df.attrs["top_n"] = 3
        chart_df.attrs["excluded_count"] = 0
        chart_df.attrs["max_observed_goals_per_team"] = None
        scoreline_chart.draw_scoreline_chart(chart_df, png_path)
        summary["png"] = str(png_path)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except Exception as exc:
        summary["plot_error"] = str(exc)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    parser = argparse.ArgumentParser(
        description="Run V32: V29 with a role/xG third-slot selector."
    )
    parser.add_argument("--team-a")
    parser.add_argument("--team-b")
    parser.add_argument("--host-a", action="store_true")
    parser.add_argument("--host-b", action="store_true")
    parser.add_argument("--knockout", action="store_true")
    parser.add_argument("--outdir", default="outputs/outputs_v32_third_slot")
    parser.add_argument("--eval-observed", action="store_true")
    parser.add_argument("--eval-outdir", default="observed_eval/observed_eval_v32_third_slot_top3")
    parser.add_argument("--eval-limit", type=int, default=20)
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
    parser.add_argument("--blind-current-goals", action="store_true")
    parser.add_argument("--ignore-observed-goals", action="store_true")
    parser.add_argument("--ignore-fotmob-goal-stats", action="store_true")
    parser.add_argument("--disable-group-score-context", action="store_true")
    parser.add_argument("--third-slot-relative-floor", type=float, default=DEFAULT_THIRD_SLOT_RELATIVE_FLOOR)
    parser.add_argument("--third-slot-absolute-floor", type=float, default=DEFAULT_THIRD_SLOT_ABSOLUTE_FLOOR)
    parser.add_argument("--third-slot-min-utility-gain", type=float, default=DEFAULT_THIRD_SLOT_MIN_UTILITY_GAIN)
    parser.add_argument("--third-slot-probability-weight", type=float, default=DEFAULT_THIRD_SLOT_PROBABILITY_WEIGHT)
    parser.add_argument("--third-slot-signal-weight", type=float, default=DEFAULT_THIRD_SLOT_SIGNAL_WEIGHT)
    parser.add_argument("--third-slot-incumbent-bonus", type=float, default=DEFAULT_THIRD_SLOT_INCUMBENT_BONUS)
    parser.add_argument("--third-slot-tail-incumbent-bonus", type=float, default=DEFAULT_THIRD_SLOT_TAIL_INCUMBENT_BONUS)
    parser.add_argument("--third-slot-extreme-tail-incumbent-bonus", type=float, default=DEFAULT_THIRD_SLOT_EXTREME_TAIL_INCUMBENT_BONUS)
    parser.add_argument("--third-slot-ranked-candidates", type=int, default=DEFAULT_THIRD_SLOT_RANKED_CANDIDATES)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

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
        include_observed_goals=not (
            args.blind_current_goals or args.ignore_observed_goals
        ),
        include_fotmob_goal_stats=not (
            args.blind_current_goals or args.ignore_fotmob_goal_stats
        ),
        include_group_score_context=not (
            args.blind_current_goals or args.disable_group_score_context
        ),
        third_slot_relative_floor=args.third_slot_relative_floor,
        third_slot_absolute_floor=args.third_slot_absolute_floor,
        third_slot_min_utility_gain=args.third_slot_min_utility_gain,
        third_slot_probability_weight=args.third_slot_probability_weight,
        third_slot_signal_weight=args.third_slot_signal_weight,
        third_slot_incumbent_bonus=args.third_slot_incumbent_bonus,
        third_slot_tail_incumbent_bonus=args.third_slot_tail_incumbent_bonus,
        third_slot_extreme_tail_incumbent_bonus=args.third_slot_extreme_tail_incumbent_bonus,
        third_slot_ranked_candidates=args.third_slot_ranked_candidates,
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
    if not args.team_a or not args.team_b:
        parser.error("--team-a and --team-b are required unless --eval-observed is set")

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
    (output_dir / "model_summary.json").write_text(
        json.dumps(
            {
                "version": "v32-third-slot-coverage",
                "base_model": "v29-tail-risk-scoreline",
                "team_a": prediction["team_a"],
                "team_b": prediction["team_b"],
                "result_probabilities": prediction["result_probabilities"],
                "v29_adjustments": prediction.get("v29_adjustments", {}),
                "v32_adjustments": prediction["v32_adjustments"],
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
                "top_scorelines": prediction["top_scorelines"][:5],
                "v29_adjustments": prediction.get("v29_adjustments", {}),
                "v32_adjustments": {
                    key: value
                    for key, value in prediction["v32_adjustments"].items()
                    if key
                    not in {
                        "team_a_role_profile",
                        "team_b_role_profile",
                        "team_a_xg_profile",
                        "team_b_xg_profile",
                    }
                },
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
'''
v32_third_slot_coverage_model = _load_submodule("v32_third_slot_coverage_model", _V32_THIRD_SLOT_COVERAGE_MODEL_SOURCE, "feature_layers.py:v32_third_slot_coverage_model")

# ======================================================================
# v33_outlier_slot_model.py  (bundled as an isolated sub-module)
# ======================================================================
_V33_OUTLIER_SLOT_MODEL_SOURCE = r'''
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
'''
v33_outlier_slot_model = _load_submodule("v33_outlier_slot_model", _V33_OUTLIER_SLOT_MODEL_SOURCE, "feature_layers.py:v33_outlier_slot_model")

# ======================================================================
# v34_late_instability_overlay_model.py  (bundled as an isolated sub-module)
# ======================================================================
_V34_LATE_INSTABILITY_OVERLAY_MODEL_SOURCE = r'''
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
'''
v34_late_instability_overlay_model = _load_submodule("v34_late_instability_overlay_model", _V34_LATE_INSTABILITY_OVERLAY_MODEL_SOURCE, "feature_layers.py:v34_late_instability_overlay_model")

# ======================================================================
# v35_game_state_late_mutation_model.py  (bundled as an isolated sub-module)
# ======================================================================
_V35_GAME_STATE_LATE_MUTATION_MODEL_SOURCE = r'''
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
'''
v35_game_state_late_mutation_model = _load_submodule("v35_game_state_late_mutation_model", _V35_GAME_STATE_LATE_MUTATION_MODEL_SOURCE, "feature_layers.py:v35_game_state_late_mutation_model")

# ======================================================================
# v36_fotmob_current_form_model.py  (bundled as an isolated sub-module)
# ======================================================================
_V36_FOTMOB_CURRENT_FORM_MODEL_SOURCE = r'''
#!/usr/bin/env python3
"""V36: V35 plus a reusable shrunk FotMob xG/player/keeper form layer.

Full feature run:
    MPLCONFIGDIR=.matplotlib_cache .venv/bin/python v36_fotmob_current_form_model.py --team-a "Argentina" --team-b "France" --knockout --outdir outputs/outputs_v36_argentina_france --fotmob-leaders data/fotmob_full_stat_tables_clean.csv --fotmob-player-stats data/fotmob_match_player_stats_clean.csv --fotmob-lineups data/fotmob_match_lineups_clean.csv --fotmob-substitutions data/fotmob_match_substitutions_clean.csv --fotmob-keeper-stats data/fotmob_match_keeper_stats_clean.csv --fotmob-match-facts data/fotmob_match_facts_clean.csv --fotmob-goal-events data/fotmob_match_goal_events_clean.csv --profile-prior-matches 3.0 --fotmob-wdl-blend 0.00 --fotmob-scoreline-blend 0.15 --max-log-adjustment 0.12

The important operational bit: this model reads the current `data/fotmob_*_clean.csv`
files at build time. When scraper.py adds new completed matches, rerunning V36
automatically folds the new information into the form profiles without code edits.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
VERSIONS_DIR = PROJECT_DIR / "versions"
if str(VERSIONS_DIR) not in sys.path:
    sys.path.insert(0, str(VERSIONS_DIR))

import v11_wcq_results_model as v11
import v15_catboost_model as v15
import v20_scoreline_ensemble_model as v20
import v28_current_worldcup_form_model as v28
import v29_tail_risk_scoreline_model as v29
import v30_player_role_form_model as v30
import v35_game_state_late_mutation_model as v35


ScoreMatrix = Dict[Tuple[int, int], float]

DEFAULT_FOTMOB_WDL_BLEND = 0.00
DEFAULT_FOTMOB_SCORELINE_BLEND = 0.15
DEFAULT_BETA_XG_EDGE = 0.060
DEFAULT_BETA_XGOT_EDGE = 0.040
DEFAULT_BETA_SHOT_EDGE = 0.020
DEFAULT_BETA_KEEPER_EDGE = 0.030
DEFAULT_BETA_TEMPO = 0.018
DEFAULT_MAX_LOG_ADJUSTMENT = 0.12
DEFAULT_PROFILE_PRIOR_MATCHES = 3.0


@dataclass
class FotmobTeamFormProfile:
    team: str
    matches: int = 0
    rows: int = 0
    xg_for: float = 0.0
    xg_against: float = 0.0
    xgot_for: float = 0.0
    xgot_against: float = 0.0
    shots_for: float = 0.0
    shots_against: float = 0.0
    sot_for: float = 0.0
    sot_against: float = 0.0
    box_touches_for: float = 0.0
    box_touches_against: float = 0.0
    keeper_goals_prevented: float = 0.0
    keeper_xgot_faced: float = 0.0
    lineup_rating: float = 0.0
    attack_signal: float = 0.0
    defense_signal: float = 0.0
    xgot_signal: float = 0.0
    shot_signal: float = 0.0
    tempo_signal: float = 0.0
    keeper_signal: float = 0.0
    coverage: float = 0.0

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "team": self.team,
            "matches": int(self.matches),
            "rows": int(self.rows),
            "xg_for": float(self.xg_for),
            "xg_against": float(self.xg_against),
            "xgot_for": float(self.xgot_for),
            "xgot_against": float(self.xgot_against),
            "shots_for": float(self.shots_for),
            "shots_against": float(self.shots_against),
            "sot_for": float(self.sot_for),
            "sot_against": float(self.sot_against),
            "box_touches_for": float(self.box_touches_for),
            "box_touches_against": float(self.box_touches_against),
            "keeper_goals_prevented": float(self.keeper_goals_prevented),
            "keeper_xgot_faced": float(self.keeper_xgot_faced),
            "lineup_rating": float(self.lineup_rating),
            "attack_signal": float(self.attack_signal),
            "defense_signal": float(self.defense_signal),
            "xgot_signal": float(self.xgot_signal),
            "shot_signal": float(self.shot_signal),
            "tempo_signal": float(self.tempo_signal),
            "keeper_signal": float(self.keeper_signal),
            "coverage": float(self.coverage),
        }


def _to_number(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    std = float(values.std(ddof=0))
    if std <= 1e-12:
        return pd.Series(0.0, index=series.index)
    return ((values - float(values.mean())) / std).clip(-2.5, 2.5)


def completed_fotmob_facts_to_observed(
    match_facts_csv: str | Path | None,
    output_csv: str | Path,
) -> Path | None:
    if not match_facts_csv or not Path(match_facts_csv).exists():
        return None
    facts = pd.read_csv(match_facts_csv)
    required = {"match_id", "home_team", "away_team", "home_score", "away_score", "status"}
    if facts.empty or not required.issubset(facts.columns):
        return None
    completed = facts.loc[
        facts["status"].astype(str).str.lower().str.contains("full", na=False)
    ].copy()
    if completed.empty:
        return None
    completed["kickoff_sort"] = pd.to_datetime(
        completed.get("kickoff", ""),
        errors="coerce",
        utc=True,
    )
    completed = completed.sort_values(["kickoff_sort", "match_id"], na_position="last")
    observed = pd.DataFrame(
        {
            "match_id": completed["match_id"].astype(str),
            "date_label": completed.get("kickoff", "").astype(str),
            "stage": "Group Stage",
            "group": "",
            "team_a": completed["home_team"].astype(str),
            "team_b": completed["away_team"].astype(str),
            "goals_a": _to_number(completed["home_score"]).astype(int),
            "goals_b": _to_number(completed["away_score"]).astype(int),
            "source": "fotmob_match_facts_clean",
        }
    )
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    observed.to_csv(output_path, index=False)
    return output_path


def _team_match_from_player_stats(
    player_stats_csv: str | Path | None,
    lineups_csv: str | Path | None,
    substitutions_csv: str | Path | None,
) -> pd.DataFrame:
    paths = [player_stats_csv, lineups_csv, substitutions_csv]
    if not all(path and Path(path).exists() for path in paths):
        return pd.DataFrame()
    player_stats = pd.read_csv(player_stats_csv)
    lineups = pd.read_csv(lineups_csv)
    substitutions = pd.read_csv(substitutions_csv)
    if player_stats.empty or lineups.empty:
        return pd.DataFrame()
    player_stats = v30.aggregate_player_match_stats(player_stats)
    enriched = v30.attach_roster_context(player_stats, lineups, substitutions)
    enriched = enriched[enriched["team"].astype(str).ne("")]
    if enriched.empty:
        return pd.DataFrame()

    numeric = [
        "xg",
        "xgot",
        "total_shots",
        "shots_on_target",
        "touches_in_opposition_box",
        "fotmob_rating",
    ]
    for column in numeric:
        if column not in enriched:
            enriched[column] = 0.0
        enriched[column] = _to_number(enriched[column])

    team_match = (
        enriched.groupby(["match_id", "team"], as_index=False)
        .agg(
            home_team=("home_team", "first"),
            away_team=("away_team", "first"),
            rows=("player", "size"),
            xg_for=("xg", "sum"),
            xgot_for=("xgot", "sum"),
            shots_for=("total_shots", "sum"),
            sot_for=("shots_on_target", "sum"),
            box_touches_for=("touches_in_opposition_box", "sum"),
            lineup_rating=("fotmob_rating", "mean"),
        )
        .copy()
    )
    opponents = []
    for row in team_match.to_dict(orient="records"):
        home = str(row.get("home_team", ""))
        away = str(row.get("away_team", ""))
        team = str(row.get("team", ""))
        opponents.append(away if team == home else home if team == away else "")
    team_match["opponent"] = opponents
    against = team_match[
        [
            "match_id",
            "team",
            "xg_for",
            "xgot_for",
            "shots_for",
            "sot_for",
            "box_touches_for",
        ]
    ].rename(
        columns={
            "team": "opponent",
            "xg_for": "xg_against",
            "xgot_for": "xgot_against",
            "shots_for": "shots_against",
            "sot_for": "sot_against",
            "box_touches_for": "box_touches_against",
        }
    )
    return team_match.merge(against, on=["match_id", "opponent"], how="left")


def _keeper_team_match(
    keeper_stats_csv: str | Path | None,
    lineups_csv: str | Path | None,
    substitutions_csv: str | Path | None,
) -> pd.DataFrame:
    paths = [keeper_stats_csv, lineups_csv, substitutions_csv]
    if not all(path and Path(path).exists() for path in paths):
        return pd.DataFrame()
    keeper = pd.read_csv(keeper_stats_csv)
    lineups = pd.read_csv(lineups_csv)
    substitutions = pd.read_csv(substitutions_csv)
    if keeper.empty or lineups.empty:
        return pd.DataFrame()
    keeper = v30.aggregate_player_match_stats(keeper)
    enriched = v30.attach_roster_context(keeper, lineups, substitutions)
    enriched = enriched[enriched["team"].astype(str).ne("")]
    if enriched.empty:
        return pd.DataFrame()
    for column in ["goals_prevented", "xgot_faced"]:
        if column not in enriched:
            enriched[column] = 0.0
        enriched[column] = _to_number(enriched[column])
    return (
        enriched.groupby(["match_id", "team"], as_index=False)
        .agg(
            keeper_goals_prevented=("goals_prevented", "sum"),
            keeper_xgot_faced=("xgot_faced", "sum"),
        )
        .copy()
    )


def build_fotmob_team_form_profiles(
    player_stats_csv: str | Path | None,
    lineups_csv: str | Path | None,
    substitutions_csv: str | Path | None,
    keeper_stats_csv: str | Path | None,
    prior_matches: float = DEFAULT_PROFILE_PRIOR_MATCHES,
    exclude_match_ids: set[str] | None = None,
) -> dict[str, FotmobTeamFormProfile]:
    team_match = _team_match_from_player_stats(
        player_stats_csv,
        lineups_csv,
        substitutions_csv,
    )
    if team_match.empty:
        return {}
    if exclude_match_ids:
        excluded = {str(match_id) for match_id in exclude_match_ids}
        team_match = team_match[~team_match["match_id"].astype(str).isin(excluded)].copy()
        if team_match.empty:
            return {}
    keeper = _keeper_team_match(keeper_stats_csv, lineups_csv, substitutions_csv)
    if not keeper.empty:
        if exclude_match_ids:
            excluded = {str(match_id) for match_id in exclude_match_ids}
            keeper = keeper[~keeper["match_id"].astype(str).isin(excluded)].copy()
        team_match = team_match.merge(keeper, on=["match_id", "team"], how="left")
    for column in [
        "xg_against",
        "xgot_against",
        "shots_against",
        "sot_against",
        "box_touches_against",
        "keeper_goals_prevented",
        "keeper_xgot_faced",
    ]:
        if column not in team_match:
            team_match[column] = 0.0
        team_match[column] = _to_number(team_match[column])

    team = (
        team_match.groupby("team", as_index=False)
        .agg(
            matches=("match_id", "nunique"),
            rows=("rows", "sum"),
            xg_for=("xg_for", "mean"),
            xg_against=("xg_against", "mean"),
            xgot_for=("xgot_for", "mean"),
            xgot_against=("xgot_against", "mean"),
            shots_for=("shots_for", "mean"),
            shots_against=("shots_against", "mean"),
            sot_for=("sot_for", "mean"),
            sot_against=("sot_against", "mean"),
            box_touches_for=("box_touches_for", "mean"),
            box_touches_against=("box_touches_against", "mean"),
            keeper_goals_prevented=("keeper_goals_prevented", "mean"),
            keeper_xgot_faced=("keeper_xgot_faced", "mean"),
            lineup_rating=("lineup_rating", "mean"),
        )
        .copy()
    )
    if team.empty:
        return {}

    team["attack_signal_raw"] = (
        0.48 * _zscore(team["xg_for"])
        + 0.22 * _zscore(team["sot_for"])
        + 0.18 * _zscore(team["box_touches_for"])
        + 0.12 * _zscore(team["lineup_rating"])
    )
    team["defense_signal_raw"] = -(
        0.52 * _zscore(team["xg_against"])
        + 0.22 * _zscore(team["sot_against"])
        + 0.16 * _zscore(team["box_touches_against"])
        + 0.10 * _zscore(team["keeper_xgot_faced"])
    )
    team["xgot_signal_raw"] = (
        0.56 * _zscore(team["xgot_for"])
        - 0.44 * _zscore(team["xgot_against"])
    )
    team["shot_signal_raw"] = (
        0.50 * _zscore(team["shots_for"])
        + 0.25 * _zscore(team["sot_for"])
        - 0.25 * _zscore(team["shots_against"])
    )
    team["tempo_signal_raw"] = _zscore(
        team["shots_for"]
        + team["shots_against"]
        + 0.10 * (team["box_touches_for"] + team["box_touches_against"])
    )
    team["keeper_signal_raw"] = (
        0.72 * _zscore(team["keeper_goals_prevented"])
        - 0.28 * _zscore(team["xgot_against"])
    )
    coverage = np.sqrt(
        _to_number(team["matches"]) / (_to_number(team["matches"]) + float(prior_matches))
    )
    team["coverage"] = np.clip(coverage, 0.0, 1.0)

    profiles: dict[str, FotmobTeamFormProfile] = {}
    for row in team.to_dict(orient="records"):
        coverage_value = float(row.get("coverage", 0.0) or 0.0)
        team_key = v28.canon_team(row.get("team", ""))
        profiles[team_key] = FotmobTeamFormProfile(
            team=team_key,
            matches=int(row.get("matches", 0) or 0),
            rows=int(row.get("rows", 0) or 0),
            xg_for=float(row.get("xg_for", 0.0) or 0.0),
            xg_against=float(row.get("xg_against", 0.0) or 0.0),
            xgot_for=float(row.get("xgot_for", 0.0) or 0.0),
            xgot_against=float(row.get("xgot_against", 0.0) or 0.0),
            shots_for=float(row.get("shots_for", 0.0) or 0.0),
            shots_against=float(row.get("shots_against", 0.0) or 0.0),
            sot_for=float(row.get("sot_for", 0.0) or 0.0),
            sot_against=float(row.get("sot_against", 0.0) or 0.0),
            box_touches_for=float(row.get("box_touches_for", 0.0) or 0.0),
            box_touches_against=float(row.get("box_touches_against", 0.0) or 0.0),
            keeper_goals_prevented=float(row.get("keeper_goals_prevented", 0.0) or 0.0),
            keeper_xgot_faced=float(row.get("keeper_xgot_faced", 0.0) or 0.0),
            lineup_rating=float(row.get("lineup_rating", 0.0) or 0.0),
            attack_signal=float(row.get("attack_signal_raw", 0.0) or 0.0) * coverage_value,
            defense_signal=float(row.get("defense_signal_raw", 0.0) or 0.0) * coverage_value,
            xgot_signal=float(row.get("xgot_signal_raw", 0.0) or 0.0) * coverage_value,
            shot_signal=float(row.get("shot_signal_raw", 0.0) or 0.0) * coverage_value,
            tempo_signal=float(row.get("tempo_signal_raw", 0.0) or 0.0) * coverage_value,
            keeper_signal=float(row.get("keeper_signal_raw", 0.0) or 0.0) * coverage_value,
            coverage=coverage_value,
        )
    return profiles


class V36FotmobCurrentFormModel:
    """Wrap V35 with a shrunk FotMob current-form lambda adjustment."""

    def __init__(
        self,
        base_model: v35.V35GameStateLateMutationModel,
        team_form_profiles: dict[str, FotmobTeamFormProfile],
        fotmob_wdl_blend: float = DEFAULT_FOTMOB_WDL_BLEND,
        fotmob_scoreline_blend: float = DEFAULT_FOTMOB_SCORELINE_BLEND,
        beta_xg_edge: float = DEFAULT_BETA_XG_EDGE,
        beta_xgot_edge: float = DEFAULT_BETA_XGOT_EDGE,
        beta_shot_edge: float = DEFAULT_BETA_SHOT_EDGE,
        beta_keeper_edge: float = DEFAULT_BETA_KEEPER_EDGE,
        beta_tempo: float = DEFAULT_BETA_TEMPO,
        max_log_adjustment: float = DEFAULT_MAX_LOG_ADJUSTMENT,
    ):
        self.base_model = base_model
        self.team_form_profiles = team_form_profiles
        self.fotmob_wdl_blend = float(np.clip(fotmob_wdl_blend, 0.0, 1.0))
        self.fotmob_scoreline_blend = float(np.clip(fotmob_scoreline_blend, 0.0, 1.0))
        self.beta_xg_edge = float(beta_xg_edge)
        self.beta_xgot_edge = float(beta_xgot_edge)
        self.beta_shot_edge = float(beta_shot_edge)
        self.beta_keeper_edge = float(beta_keeper_edge)
        self.beta_tempo = float(beta_tempo)
        self.max_log_adjustment = float(max(max_log_adjustment, 0.0))
        self.training_data_summary = getattr(base_model, "training_data_summary", {})

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_model, name)

    def form_for_team(self, team: object) -> FotmobTeamFormProfile:
        team_key = v28.canon_team(team)
        return self.team_form_profiles.get(team_key, FotmobTeamFormProfile(team=team_key))

    def _form_adjusted_matrix(
        self,
        prediction: Dict[str, Any],
        form_a: FotmobTeamFormProfile,
        form_b: FotmobTeamFormProfile,
        max_goals: int,
    ) -> tuple[ScoreMatrix, Dict[str, Any]]:
        base_lambda_a = float(prediction["lambda_a"])
        base_lambda_b = float(prediction["lambda_b"])
        xg_edge_a = form_a.attack_signal - form_b.defense_signal
        xg_edge_b = form_b.attack_signal - form_a.defense_signal
        xgot_edge_a = form_a.xgot_signal - form_b.keeper_signal
        xgot_edge_b = form_b.xgot_signal - form_a.keeper_signal
        shot_edge_a = form_a.shot_signal - 0.35 * form_b.shot_signal
        shot_edge_b = form_b.shot_signal - 0.35 * form_a.shot_signal
        keeper_edge_a = -form_b.keeper_signal
        keeper_edge_b = -form_a.keeper_signal
        tempo = 0.5 * (form_a.tempo_signal + form_b.tempo_signal)

        log_a = (
            self.beta_xg_edge * xg_edge_a
            + self.beta_xgot_edge * xgot_edge_a
            + self.beta_shot_edge * shot_edge_a
            + self.beta_keeper_edge * keeper_edge_a
            + self.beta_tempo * tempo
        )
        log_b = (
            self.beta_xg_edge * xg_edge_b
            + self.beta_xgot_edge * xgot_edge_b
            + self.beta_shot_edge * shot_edge_b
            + self.beta_keeper_edge * keeper_edge_b
            + self.beta_tempo * tempo
        )
        log_a = float(np.clip(log_a, -self.max_log_adjustment, self.max_log_adjustment))
        log_b = float(np.clip(log_b, -self.max_log_adjustment, self.max_log_adjustment))
        lambda_a = float(np.clip(base_lambda_a * math.exp(log_a), 0.15, 5.8))
        lambda_b = float(np.clip(base_lambda_b * math.exp(log_b), 0.15, 5.8))
        matrix = v11.poisson_score_matrix(lambda_a, lambda_b, max_goals)
        rho = prediction.get("calibration_notes", {}).get("dixon_coles_rho", -0.08)
        matrix = v11.apply_dixon_coles_adjustment(matrix, lambda_a, lambda_b, rho=rho)
        return matrix, {
            "base_lambda_a": base_lambda_a,
            "base_lambda_b": base_lambda_b,
            "fotmob_lambda_a": lambda_a,
            "fotmob_lambda_b": lambda_b,
            "fotmob_log_adjustment_a": log_a,
            "fotmob_log_adjustment_b": log_b,
            "xg_edge_a": float(xg_edge_a),
            "xg_edge_b": float(xg_edge_b),
            "xgot_edge_a": float(xgot_edge_a),
            "xgot_edge_b": float(xgot_edge_b),
            "shot_edge_a": float(shot_edge_a),
            "shot_edge_b": float(shot_edge_b),
            "keeper_edge_a": float(keeper_edge_a),
            "keeper_edge_b": float(keeper_edge_b),
            "tempo_edge": float(tempo),
        }

    def predict(self, *args, **kwargs) -> Dict[str, Any]:
        team_a = kwargs.get("team_a", args[0] if args else "")
        team_b = kwargs.get("team_b", args[1] if len(args) > 1 else "")
        max_goals = int(kwargs.get("max_goals", args[5] if len(args) > 5 else 10))

        prediction = self.base_model.predict(*args, **kwargs)
        base_matrix = v29.score_matrix_from_prediction(prediction)
        base_results = dict(prediction["result_probabilities"])
        form_a = self.form_for_team(team_a)
        form_b = self.form_for_team(team_b)
        adjusted_matrix, diagnostics = self._form_adjusted_matrix(
            prediction,
            form_a,
            form_b,
            max_goals=max_goals,
        )
        adjusted_results = v11.result_probs(adjusted_matrix)
        final_results = v28.blend_result_probabilities(
            base_results,
            adjusted_results,
            self.fotmob_wdl_blend,
        )
        score_matrix = v20.blend_score_matrices(
            base_matrix,
            adjusted_matrix,
            adjusted_weight=self.fotmob_scoreline_blend,
        )
        score_matrix = v11.reweight_score_matrix_to_results(score_matrix, final_results)
        lambda_a, lambda_b = v28.expected_goals(score_matrix)

        prediction["lambda_a"] = lambda_a
        prediction["lambda_b"] = lambda_b
        prediction["result_probabilities"] = final_results
        prediction["predicted_result"] = max(final_results, key=final_results.get)
        prediction.update(v15.score_outputs(score_matrix, max_goals))
        top_scorelines, tail_diagnostics = v29.select_top_scorelines_with_tail_risk(
            score_matrix,
            final_results,
            lambda_a,
            lambda_b,
            current_top_scorelines=prediction.get("top_scorelines", []),
            top_n=15,
        )
        prediction["top_scorelines"] = top_scorelines

        outlier, outlier_diagnostics = v35.select_game_state_late_outlier(
            score_matrix,
            final_results,
            lambda_a,
            lambda_b,
            prediction.get("top_scorelines", []),
            self.base_model.tournament_late_profile,
            self.base_model.late_profile_for_team(team_a),
            self.base_model.late_profile_for_team(team_b),
            self.base_model.sub_profile_for_team(team_a),
            self.base_model.sub_profile_for_team(team_b),
            self.base_model.mutation_table,
            relative_floor=self.base_model.relative_floor,
            absolute_floor=self.base_model.absolute_floor,
            source_limit=self.base_model.source_limit,
            max_goals=self.base_model.max_goals,
        )
        prediction["game_state_late_outlier"] = outlier
        prediction["late_instability_outlier"] = outlier
        prediction["outlier_scoreline"] = outlier
        prediction["top_scorelines_plus_outlier"] = [
            *prediction.get("top_scorelines", [])[:3],
            *([outlier] if outlier else []),
        ]
        prediction["v36_adjustments"] = {
            "base_model": "v35_game_state_late_mutation",
            "scoreline_policy": "shrunk_fotmob_xg_player_keeper_form_plus_v35_outlier",
            "fotmob_form_affects_wdl": self.fotmob_wdl_blend > 0,
            "fotmob_wdl_blend": self.fotmob_wdl_blend,
            "fotmob_scoreline_blend": self.fotmob_scoreline_blend,
            "max_log_adjustment": self.max_log_adjustment,
            "team_a_fotmob_form": form_a.diagnostics(),
            "team_b_fotmob_form": form_b.diagnostics(),
            "base_result_probabilities": base_results,
            "fotmob_adjusted_result_probabilities": adjusted_results,
            "tail_risk_selector": tail_diagnostics,
            "game_state_outlier": outlier_diagnostics,
            **diagnostics,
        }
        prediction["calibration_notes"] = {
            **prediction.get("calibration_notes", {}),
            "v36": prediction["v36_adjustments"],
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
    profile_prior_matches=DEFAULT_PROFILE_PRIOR_MATCHES,
    fotmob_wdl_blend=DEFAULT_FOTMOB_WDL_BLEND,
    fotmob_scoreline_blend=DEFAULT_FOTMOB_SCORELINE_BLEND,
    beta_xg_edge=DEFAULT_BETA_XG_EDGE,
    beta_xgot_edge=DEFAULT_BETA_XGOT_EDGE,
    beta_shot_edge=DEFAULT_BETA_SHOT_EDGE,
    beta_keeper_edge=DEFAULT_BETA_KEEPER_EDGE,
    beta_tempo=DEFAULT_BETA_TEMPO,
    max_log_adjustment=DEFAULT_MAX_LOG_ADJUSTMENT,
    **kwargs,
):
    data_dir = Path(__file__).resolve().parent / "data"
    fotmob_player_stats_csv = fotmob_player_stats_csv or (
        data_dir / "fotmob_match_player_stats_clean.csv"
    )
    fotmob_lineups_csv = fotmob_lineups_csv or (
        data_dir / "fotmob_match_lineups_clean.csv"
    )
    fotmob_substitutions_csv = fotmob_substitutions_csv or (
        data_dir / "fotmob_match_substitutions_clean.csv"
    )
    fotmob_keeper_stats_csv = fotmob_keeper_stats_csv or (
        data_dir / "fotmob_match_keeper_stats_clean.csv"
    )
    fotmob_match_facts_csv = fotmob_match_facts_csv or (
        data_dir / "fotmob_match_facts_clean.csv"
    )
    fotmob_goal_events_csv = fotmob_goal_events_csv or (
        data_dir / "fotmob_match_goal_events_clean.csv"
    )
    if observed_matches_csv is None:
        generated_observed = completed_fotmob_facts_to_observed(
            fotmob_match_facts_csv,
            data_dir / "fotmob_completed_matches_observed_schema.csv",
        )
        observed_matches_csv = str(generated_observed) if generated_observed else None

    base_model, data = v35.build_from_zip(
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
    profiles = build_fotmob_team_form_profiles(
        fotmob_player_stats_csv,
        fotmob_lineups_csv,
        fotmob_substitutions_csv,
        fotmob_keeper_stats_csv,
        prior_matches=profile_prior_matches,
    )
    model = V36FotmobCurrentFormModel(
        base_model,
        profiles,
        fotmob_wdl_blend=fotmob_wdl_blend,
        fotmob_scoreline_blend=fotmob_scoreline_blend,
        beta_xg_edge=beta_xg_edge,
        beta_xgot_edge=beta_xgot_edge,
        beta_shot_edge=beta_shot_edge,
        beta_keeper_edge=beta_keeper_edge,
        beta_tempo=beta_tempo,
        max_log_adjustment=max_log_adjustment,
    )
    model.training_data_summary = {
        **getattr(base_model, "training_data_summary", {}),
        "v36_scoreline_policy": "shrunk_fotmob_xg_player_keeper_form_plus_v35_outlier",
        "v36_profile_teams": len(profiles),
        "v36_observed_matches_csv": str(observed_matches_csv),
        "v36_fotmob_player_stats_csv": str(fotmob_player_stats_csv),
        "v36_fotmob_lineups_csv": str(fotmob_lineups_csv),
        "v36_fotmob_substitutions_csv": str(fotmob_substitutions_csv),
        "v36_fotmob_keeper_stats_csv": str(fotmob_keeper_stats_csv),
        "v36_fotmob_match_facts_csv": str(fotmob_match_facts_csv),
        "v36_fotmob_goal_events_csv": str(fotmob_goal_events_csv),
        "v36_profile_prior_matches": float(profile_prior_matches),
        "v36_fotmob_wdl_blend": model.fotmob_wdl_blend,
        "v36_fotmob_scoreline_blend": model.fotmob_scoreline_blend,
        "v36_max_log_adjustment": model.max_log_adjustment,
    }
    return model, data


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    parser = argparse.ArgumentParser(
        description="Run V36: shrunk FotMob current-form layer plus V35 outlier."
    )
    parser.add_argument("--team-a", required=True)
    parser.add_argument("--team-b", required=True)
    parser.add_argument("--host-a", action="store_true")
    parser.add_argument("--host-b", action="store_true")
    parser.add_argument("--knockout", action="store_true")
    parser.add_argument("--outdir", default="outputs/outputs_v36_fotmob_current_form")
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
    parser.add_argument("--observed-matches")
    parser.add_argument("--fotmob-leaders", default=str(data_dir / "fotmob_full_stat_tables_clean.csv"))
    parser.add_argument("--fotmob-player-stats", default=str(data_dir / "fotmob_match_player_stats_clean.csv"))
    parser.add_argument("--fotmob-lineups", default=str(data_dir / "fotmob_match_lineups_clean.csv"))
    parser.add_argument("--fotmob-substitutions", default=str(data_dir / "fotmob_match_substitutions_clean.csv"))
    parser.add_argument("--fotmob-keeper-stats", default=str(data_dir / "fotmob_match_keeper_stats_clean.csv"))
    parser.add_argument("--fotmob-match-facts", default=str(data_dir / "fotmob_match_facts_clean.csv"))
    parser.add_argument("--fotmob-goal-events", default=str(data_dir / "fotmob_match_goal_events_clean.csv"))
    parser.add_argument("--profile-prior-matches", type=float, default=DEFAULT_PROFILE_PRIOR_MATCHES)
    parser.add_argument("--fotmob-wdl-blend", type=float, default=DEFAULT_FOTMOB_WDL_BLEND)
    parser.add_argument("--fotmob-scoreline-blend", type=float, default=DEFAULT_FOTMOB_SCORELINE_BLEND)
    parser.add_argument("--max-log-adjustment", type=float, default=DEFAULT_MAX_LOG_ADJUSTMENT)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

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
        profile_prior_matches=args.profile_prior_matches,
        fotmob_wdl_blend=args.fotmob_wdl_blend,
        fotmob_scoreline_blend=args.fotmob_scoreline_blend,
        max_log_adjustment=args.max_log_adjustment,
    )
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
        output_dir / "scoreline_probabilities_top_plus_game_state_outlier.csv",
        index=False,
    )
    (output_dir / "model_summary.json").write_text(
        json.dumps(
            {
                "version": "v36-fotmob-current-form",
                "base_model": "v35-game-state-late-mutation",
                "team_a": prediction["team_a"],
                "team_b": prediction["team_b"],
                "result_probabilities": prediction["result_probabilities"],
                "top_3": prediction["top_scorelines"][:3],
                "game_state_late_outlier": prediction["game_state_late_outlier"],
                "v36_adjustments": prediction["v36_adjustments"],
                "expanded_training_data": model.training_data_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not args.no_plots:
        v11.plot_prediction_outputs(prediction, output_dir)
        v28.plot_top3_scorelines(prediction, output_dir / "plots")
        v35.plot_top3_plus_game_state_outlier(prediction, output_dir)
    print(
        json.dumps(
            {
                "result_probabilities": prediction["result_probabilities"],
                "predicted_result": prediction["predicted_result"],
                "lambda_a": prediction["lambda_a"],
                "lambda_b": prediction["lambda_b"],
                "top_3": prediction["top_scorelines"][:3],
                "game_state_late_outlier": prediction["game_state_late_outlier"],
                "v36_adjustments": {
                    key: value
                    for key, value in prediction["v36_adjustments"].items()
                    if key
                    not in {
                        "game_state_outlier",
                        "tail_risk_selector",
                    }
                },
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
'''
v36_fotmob_current_form_model = _load_submodule("v36_fotmob_current_form_model", _V36_FOTMOB_CURRENT_FORM_MODEL_SOURCE, "feature_layers.py:v36_fotmob_current_form_model")

# ======================================================================
# v38_total_lambda_calibrated_model.py  (bundled as an isolated sub-module)
# ======================================================================
_V38_TOTAL_LAMBDA_CALIBRATED_MODEL_SOURCE = r'''
#!/usr/bin/env python3
"""V38: V35 plus one shrunk global total-lambda correction.

This is intentionally small: one estimated multiplier, no hand-built extra
thresholds. It targets the under-bracketing diagnosis by lifting or lowering
the total-goals scale while preserving the base model's W/D/L probabilities.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

import v11_wcq_results_model as v11
import v15_catboost_model as v15
import v20_scoreline_ensemble_model as v20
import v28_current_worldcup_form_model as v28
import v29_tail_risk_scoreline_model as v29
import v35_game_state_late_mutation_model as v35
import v36_fotmob_current_form_model as v36


ScoreMatrix = Dict[Tuple[int, int], float]

DEFAULT_BOOTSTRAP_SAMPLES = 2500
DEFAULT_SHRINKAGE_PRIOR_SD = 0.08
DEFAULT_RANDOM_SEED = 20260623


@dataclass
class TotalLambdaCalibration:
    raw_multiplier: float = 1.0
    shrunk_multiplier: float = 1.0
    shrinkage_weight: float = 0.0
    bootstrap_mean: float = 1.0
    bootstrap_std: float = 0.0
    bootstrap_ci_low: float = 1.0
    bootstrap_ci_high: float = 1.0
    mean_actual_total: float = 0.0
    mean_predicted_lambda_sum: float = 0.0
    n_matches: int = 0
    prior_sd: float = DEFAULT_SHRINKAGE_PRIOR_SD

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "raw_multiplier": float(self.raw_multiplier),
            "shrunk_multiplier": float(self.shrunk_multiplier),
            "shrinkage_weight": float(self.shrinkage_weight),
            "bootstrap_mean": float(self.bootstrap_mean),
            "bootstrap_std": float(self.bootstrap_std),
            "bootstrap_ci_low": float(self.bootstrap_ci_low),
            "bootstrap_ci_high": float(self.bootstrap_ci_high),
            "mean_actual_total": float(self.mean_actual_total),
            "mean_predicted_lambda_sum": float(self.mean_predicted_lambda_sum),
            "n_matches": int(self.n_matches),
            "prior_sd": float(self.prior_sd),
        }


def result_label(goals_a: int, goals_b: int) -> str:
    if goals_a > goals_b:
        return "team_a_win"
    if goals_b > goals_a:
        return "team_b_win"
    return "draw"


def estimate_total_lambda_calibration(
    base_model: Any,
    observed_matches_csv: str | Path | None,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    shrinkage_prior_sd: float = DEFAULT_SHRINKAGE_PRIOR_SD,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> TotalLambdaCalibration:
    if not observed_matches_csv or not Path(observed_matches_csv).exists():
        return TotalLambdaCalibration(prior_sd=shrinkage_prior_sd)
    observed = pd.read_csv(observed_matches_csv)
    required = {"team_a", "team_b", "goals_a", "goals_b"}
    if observed.empty or not required.issubset(observed.columns):
        return TotalLambdaCalibration(prior_sd=shrinkage_prior_sd)

    rows = []
    for row in observed.to_dict(orient="records"):
        team_a = str(row["team_a"])
        team_b = str(row["team_b"])
        prediction = base_model.predict(
            team_a,
            team_b,
            host_a=team_a in {"Canada", "Mexico", "USA", "United States"},
            host_b=team_b in {"Canada", "Mexico", "USA", "United States"},
            knockout=str(row.get("stage", "")).strip().lower() != "group stage",
        )
        lambda_sum = float(prediction.get("lambda_a", 0.0)) + float(
            prediction.get("lambda_b", 0.0)
        )
        actual_total = int(row["goals_a"]) + int(row["goals_b"])
        if lambda_sum > 1e-9:
            rows.append({"actual_total": actual_total, "lambda_sum": lambda_sum})
    if not rows:
        return TotalLambdaCalibration(prior_sd=shrinkage_prior_sd)

    frame = pd.DataFrame(rows)
    actual = frame["actual_total"].to_numpy(dtype=float)
    predicted = frame["lambda_sum"].to_numpy(dtype=float)
    raw_multiplier = float(actual.sum() / max(predicted.sum(), 1e-9))

    rng = np.random.default_rng(int(random_seed))
    ratios = []
    n = len(frame)
    for _ in range(max(int(bootstrap_samples), 1)):
        idx = rng.integers(0, n, size=n)
        denom = float(predicted[idx].sum())
        ratios.append(float(actual[idx].sum() / max(denom, 1e-9)))
    boot = np.asarray(ratios, dtype=float)
    boot_std = float(np.std(boot, ddof=1)) if len(boot) > 1 else 0.0
    ci_low, ci_high = np.quantile(boot, [0.025, 0.975])

    prior_sd = max(float(shrinkage_prior_sd), 1e-9)
    shrinkage_weight = float((prior_sd * prior_sd) / (prior_sd * prior_sd + boot_std * boot_std))
    shrunk_multiplier = float(1.0 + shrinkage_weight * (raw_multiplier - 1.0))
    return TotalLambdaCalibration(
        raw_multiplier=raw_multiplier,
        shrunk_multiplier=shrunk_multiplier,
        shrinkage_weight=shrinkage_weight,
        bootstrap_mean=float(np.mean(boot)),
        bootstrap_std=boot_std,
        bootstrap_ci_low=float(ci_low),
        bootstrap_ci_high=float(ci_high),
        mean_actual_total=float(np.mean(actual)),
        mean_predicted_lambda_sum=float(np.mean(predicted)),
        n_matches=int(n),
        prior_sd=prior_sd,
    )


class V38TotalLambdaCalibratedModel:
    """Wrap V35 and apply a single shrunk total-goals multiplier."""

    def __init__(
        self,
        base_model: v35.V35GameStateLateMutationModel,
        calibration: TotalLambdaCalibration,
    ):
        self.base_model = base_model
        self.calibration = calibration
        self.training_data_summary = getattr(base_model, "training_data_summary", {})

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_model, name)

    @property
    def total_lambda_multiplier(self) -> float:
        return float(self.calibration.shrunk_multiplier)

    def _adjusted_score_matrix(
        self,
        prediction: Dict[str, Any],
        max_goals: int,
    ) -> tuple[ScoreMatrix, Dict[str, Any]]:
        base_lambda_a = float(prediction["lambda_a"])
        base_lambda_b = float(prediction["lambda_b"])
        multiplier = self.total_lambda_multiplier
        lambda_a = float(np.clip(base_lambda_a * multiplier, 0.05, 7.5))
        lambda_b = float(np.clip(base_lambda_b * multiplier, 0.05, 7.5))
        matrix = v11.poisson_score_matrix(lambda_a, lambda_b, max_goals)
        rho = prediction.get("calibration_notes", {}).get("dixon_coles_rho", -0.08)
        matrix = v11.apply_dixon_coles_adjustment(matrix, lambda_a, lambda_b, rho=rho)
        result_probabilities = dict(prediction["result_probabilities"])
        matrix = v11.reweight_score_matrix_to_results(matrix, result_probabilities)
        adjusted_lambda_a, adjusted_lambda_b = v28.expected_goals(matrix)
        return matrix, {
            "base_lambda_a": base_lambda_a,
            "base_lambda_b": base_lambda_b,
            "raw_scaled_lambda_a": lambda_a,
            "raw_scaled_lambda_b": lambda_b,
            "adjusted_lambda_a": adjusted_lambda_a,
            "adjusted_lambda_b": adjusted_lambda_b,
            "base_lambda_sum": base_lambda_a + base_lambda_b,
            "adjusted_lambda_sum": adjusted_lambda_a + adjusted_lambda_b,
            "total_lambda_multiplier": multiplier,
            "wdl_preserved": True,
        }

    def predict(self, *args, **kwargs) -> Dict[str, Any]:
        team_a = kwargs.get("team_a", args[0] if args else "")
        team_b = kwargs.get("team_b", args[1] if len(args) > 1 else "")
        max_goals = int(kwargs.get("max_goals", args[5] if len(args) > 5 else 10))
        prediction = self.base_model.predict(*args, **kwargs)
        result_probabilities = dict(prediction["result_probabilities"])
        matrix, diagnostics = self._adjusted_score_matrix(prediction, max_goals)
        lambda_a, lambda_b = v28.expected_goals(matrix)

        prediction["lambda_a"] = lambda_a
        prediction["lambda_b"] = lambda_b
        prediction["result_probabilities"] = result_probabilities
        prediction["predicted_result"] = max(result_probabilities, key=result_probabilities.get)
        prediction.update(v15.score_outputs(matrix, max_goals))
        top_scorelines, tail_diagnostics = v29.select_top_scorelines_with_tail_risk(
            matrix,
            result_probabilities,
            lambda_a,
            lambda_b,
            current_top_scorelines=prediction.get("top_scorelines", []),
            top_n=15,
        )
        prediction["top_scorelines"] = top_scorelines

        outlier, outlier_diagnostics = v35.select_game_state_late_outlier(
            matrix,
            result_probabilities,
            lambda_a,
            lambda_b,
            prediction.get("top_scorelines", []),
            self.base_model.tournament_late_profile,
            self.base_model.late_profile_for_team(team_a),
            self.base_model.late_profile_for_team(team_b),
            self.base_model.sub_profile_for_team(team_a),
            self.base_model.sub_profile_for_team(team_b),
            self.base_model.mutation_table,
            relative_floor=self.base_model.relative_floor,
            absolute_floor=self.base_model.absolute_floor,
            source_limit=self.base_model.source_limit,
            max_goals=self.base_model.max_goals,
        )
        prediction["game_state_late_outlier"] = outlier
        prediction["late_instability_outlier"] = outlier
        prediction["outlier_scoreline"] = outlier
        prediction["top_scorelines_plus_outlier"] = [
            *prediction.get("top_scorelines", [])[:3],
            *([outlier] if outlier else []),
        ]
        prediction["v38_adjustments"] = {
            "base_model": "v35_game_state_late_mutation",
            "scoreline_policy": "single_shrunk_total_lambda_multiplier",
            "calibration": self.calibration.diagnostics(),
            "score_matrix_changed": True,
            "scoreline_layer_affects_wdl": False,
            "tail_risk_selector": tail_diagnostics,
            "game_state_outlier": outlier_diagnostics,
            **diagnostics,
        }
        prediction["calibration_notes"] = {
            **prediction.get("calibration_notes", {}),
            "v38": prediction["v38_adjustments"],
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
    bootstrap_samples=DEFAULT_BOOTSTRAP_SAMPLES,
    shrinkage_prior_sd=DEFAULT_SHRINKAGE_PRIOR_SD,
    random_seed=DEFAULT_RANDOM_SEED,
    **kwargs,
):
    data_dir = Path(__file__).resolve().parent / "data"
    fotmob_match_facts_csv = fotmob_match_facts_csv or (
        data_dir / "fotmob_match_facts_clean.csv"
    )
    if observed_matches_csv is None:
        generated_observed = v36.completed_fotmob_facts_to_observed(
            fotmob_match_facts_csv,
            data_dir / "fotmob_completed_matches_observed_schema.csv",
        )
        observed_matches_csv = str(generated_observed) if generated_observed else None

    base_model, data = v35.build_from_zip(
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
    calibration = estimate_total_lambda_calibration(
        base_model,
        observed_matches_csv,
        bootstrap_samples=bootstrap_samples,
        shrinkage_prior_sd=shrinkage_prior_sd,
        random_seed=random_seed,
    )
    model = V38TotalLambdaCalibratedModel(base_model, calibration)
    model.training_data_summary = {
        **getattr(base_model, "training_data_summary", {}),
        "v38_scoreline_policy": "single_shrunk_total_lambda_multiplier",
        "v38_observed_matches_csv": str(observed_matches_csv),
        "v38_total_lambda_calibration": calibration.diagnostics(),
    }
    return model, data


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    parser = argparse.ArgumentParser(description="Run V38: total-lambda calibrated V35.")
    parser.add_argument("--team-a", required=True)
    parser.add_argument("--team-b", required=True)
    parser.add_argument("--host-a", action="store_true")
    parser.add_argument("--host-b", action="store_true")
    parser.add_argument("--knockout", action="store_true")
    parser.add_argument("--outdir", default="outputs/outputs_v38_total_lambda_calibrated")
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
    parser.add_argument("--observed-matches")
    parser.add_argument("--fotmob-leaders", default=str(data_dir / "fotmob_full_stat_tables_clean.csv"))
    parser.add_argument("--fotmob-player-stats", default=str(data_dir / "fotmob_match_player_stats_clean.csv"))
    parser.add_argument("--fotmob-lineups", default=str(data_dir / "fotmob_match_lineups_clean.csv"))
    parser.add_argument("--fotmob-substitutions", default=str(data_dir / "fotmob_match_substitutions_clean.csv"))
    parser.add_argument("--fotmob-keeper-stats", default=str(data_dir / "fotmob_match_keeper_stats_clean.csv"))
    parser.add_argument("--fotmob-match-facts", default=str(data_dir / "fotmob_match_facts_clean.csv"))
    parser.add_argument("--fotmob-goal-events", default=str(data_dir / "fotmob_match_goal_events_clean.csv"))
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--shrinkage-prior-sd", type=float, default=DEFAULT_SHRINKAGE_PRIOR_SD)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

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
        bootstrap_samples=args.bootstrap_samples,
        shrinkage_prior_sd=args.shrinkage_prior_sd,
    )
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
    pd.DataFrame(prediction["top_scorelines"]).to_csv(output_dir / "scoreline_probabilities_top.csv", index=False)
    pd.DataFrame(prediction["scoreline_probabilities"]).to_csv(output_dir / "scoreline_probabilities.csv", index=False)
    pd.DataFrame(prediction["top_scorelines_plus_outlier"]).to_csv(
        output_dir / "scoreline_probabilities_top_plus_game_state_outlier.csv",
        index=False,
    )
    (output_dir / "model_summary.json").write_text(
        json.dumps(
            {
                "version": "v38-total-lambda-calibrated",
                "base_model": "v35-game-state-late-mutation",
                "team_a": prediction["team_a"],
                "team_b": prediction["team_b"],
                "result_probabilities": prediction["result_probabilities"],
                "top_3": prediction["top_scorelines"][:3],
                "game_state_late_outlier": prediction["game_state_late_outlier"],
                "v38_adjustments": prediction["v38_adjustments"],
                "expanded_training_data": model.training_data_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not args.no_plots:
        v11.plot_prediction_outputs(prediction, output_dir)
        v28.plot_top3_scorelines(prediction, output_dir / "plots")
        v35.plot_top3_plus_game_state_outlier(prediction, output_dir)
    print(
        json.dumps(
            {
                "result_probabilities": prediction["result_probabilities"],
                "predicted_result": prediction["predicted_result"],
                "lambda_a": prediction["lambda_a"],
                "lambda_b": prediction["lambda_b"],
                "top_3": prediction["top_scorelines"][:3],
                "game_state_late_outlier": prediction["game_state_late_outlier"],
                "v38_calibration": prediction["v38_adjustments"]["calibration"],
                "v38_multiplier": prediction["v38_adjustments"]["total_lambda_multiplier"],
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
'''
v38_total_lambda_calibrated_model = _load_submodule("v38_total_lambda_calibrated_model", _V38_TOTAL_LAMBDA_CALIBRATED_MODEL_SOURCE, "feature_layers.py:v38_total_lambda_calibrated_model")

# ======================================================================
# compare_v11_top_scorelines.py  (bundled as an isolated sub-module)
# ======================================================================
_COMPARE_V11_TOP_SCORELINES_SOURCE = r'''
#!/usr/bin/env python3
"""Plot actual scores against a model's leading scorelines."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


HOSTS_2026 = {"Canada", "Mexico", "USA", "United States"}
OBSERVED_ALIASES = {"Türkiye": "Turkey"}
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"


def load_model_module(model_file: str):
    model_path = Path(model_file)
    spec = importlib.util.spec_from_file_location("wc_model_scoreline_chart", model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import model file: {model_file}")
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(model_path)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def score_text(item: dict) -> str:
    return f"{int(item['team_a_goals'])}-{int(item['team_b_goals'])}"


def rank_word(rank: int) -> str:
    words = {
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
    }
    return words.get(rank, str(rank))


def ordinal_text(rank: int) -> str:
    words = {
        1: "Most likely",
        2: "Second most likely",
        3: "Third most likely",
        4: "Fourth most likely",
        5: "Fifth most likely",
    }
    return words.get(rank, f"Rank {rank}")


def result_label(goals_a: int, goals_b: int) -> str:
    if goals_a > goals_b:
        return "team_a_win"
    if goals_a == goals_b:
        return "draw"
    return "team_b_win"


def result_text(result: str, team_a: str, team_b: str) -> str:
    return {
        "team_a_win": team_a,
        "draw": "Draw",
        "team_b_win": team_b,
    }[result]


def build_comparison(args: argparse.Namespace) -> pd.DataFrame:
    observed = pd.read_csv(args.observed)
    required = {"match_id", "team_a", "team_b", "goals_a", "goals_b"}
    missing = sorted(required - set(observed.columns))
    if missing:
        raise ValueError(f"Observed file is missing columns: {missing}")

    excluded_count = 0
    if args.max_observed_goals_per_team is not None:
        outlier = (
            observed["goals_a"].gt(args.max_observed_goals_per_team)
            | observed["goals_b"].gt(args.max_observed_goals_per_team)
        )
        excluded_count = int(outlier.sum())
        observed = observed.loc[~outlier]
    observed = observed.head(args.matches).copy()

    wc = load_model_module(args.model_file)
    kwargs = {
        "train_csv": args.team_train,
        "test_csv": args.team_test,
        "model_type": args.model,
        "box_csv": args.box_data,
        "results_csv": args.results_data,
        "former_names_csv": args.former_names,
        "prediction_year": args.prediction_year,
    }
    supported = inspect.signature(wc.build_from_zip).parameters
    model, _ = wc.build_from_zip(
        args.worldcupsai_zip,
        **{key: value for key, value in kwargs.items() if key in supported},
    )

    rows = []
    for observed_order, row in observed.iterrows():
        display_a = str(row["team_a"])
        display_b = str(row["team_b"])
        team_a = OBSERVED_ALIASES.get(display_a, display_a)
        team_b = OBSERVED_ALIASES.get(display_b, display_b)
        prediction = model.predict(
            team_a,
            team_b,
            host_a=team_a in HOSTS_2026,
            host_b=team_b in HOSTS_2026,
            knockout=str(row.get("stage", "")).strip().lower() != "group stage",
        )
        top_scorelines = prediction["top_scorelines"][: args.top_n]
        if len(top_scorelines) < args.top_n:
            raise ValueError(
                f"Prediction for {team_a} vs {team_b} returned only "
                f"{len(top_scorelines)} scorelines, fewer than --top-n {args.top_n}"
            )
        goals_a = int(row["goals_a"])
        goals_b = int(row["goals_b"])
        actual = f"{goals_a}-{goals_b}"
        top_texts = [score_text(item) for item in top_scorelines]
        actual_result = result_label(goals_a, goals_b)
        result_probabilities = prediction["result_probabilities"]
        predicted_result = prediction.get(
            "predicted_result",
            max(result_probabilities, key=result_probabilities.get),
        )
        output_row = {
            "observed_order": int(observed_order) + 1,
            "match_id": row["match_id"],
            "team_a": display_a,
            "team_b": display_b,
            "actual_score": actual,
        }
        for rank, (scoreline, item) in enumerate(
            zip(top_texts, top_scorelines),
            start=1,
        ):
            output_row[f"top_{rank}_scoreline"] = scoreline
            output_row[f"top_{rank}_probability"] = float(item["probability"])
        for rank, scoreline in enumerate(top_texts, start=1):
            output_row[f"actual_is_top_{rank}"] = actual == scoreline
        output_row[f"actual_in_top_{args.top_n}"] = actual in set(top_texts)
        output_row.update(
            {
                "actual_score_probability": next(
                    (
                        float(item["probability"])
                        for item in prediction["scoreline_probabilities"]
                        if score_text(item) == actual
                    ),
                    0.0,
                ),
                "actual_result": actual_result,
                "actual_result_label": result_text(
                    actual_result,
                    display_a,
                    display_b,
                ),
                "predicted_result": predicted_result,
                "predicted_result_label": result_text(
                    predicted_result,
                    display_a,
                    display_b,
                ),
                "outcome_correct": predicted_result == actual_result,
                "predicted_result_probability": float(
                    result_probabilities[predicted_result]
                ),
                "actual_result_probability": float(
                    result_probabilities[actual_result]
                ),
                "team_a_win_probability": result_probabilities["team_a_win"],
                "draw_probability": result_probabilities["draw"],
                "team_b_win_probability": result_probabilities["team_b_win"],
            }
        )
        rows.append(output_row)
        update_after_match = getattr(model, "update_after_match", None)
        if callable(update_after_match) and not args.no_live_updates:
            update_after_match(team_a, team_b, goals_a, goals_b)

    comparison = pd.DataFrame(rows)
    comparison.attrs["excluded_count"] = excluded_count
    comparison.attrs["max_observed_goals_per_team"] = (
        args.max_observed_goals_per_team
    )
    comparison.attrs["model_label"] = args.model_label
    comparison.attrs["top_n"] = args.top_n
    comparison.attrs["live_updates"] = not args.no_live_updates
    return comparison


def draw_scoreline_chart(comparison: pd.DataFrame, output_path: Path) -> None:
    """Draw a clean, compact scoreline comparison."""
    count = len(comparison)
    top_n = int(comparison.attrs.get("top_n", 2))
    score_xs = [7.15 + 3.4 * index for index in range(top_n)]
    coverage_x = score_xs[-1] + 3.7
    outcome_x = coverage_x + 2.45
    x_limit = outcome_x + 1.85
    fig, ax = plt.subplots(figsize=(x_limit, max(7.2, count * 0.82 + 2.2)))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_xlim(0, x_limit)
    ax.set_ylim(-0.8, count + 0.75)
    ax.axis("off")

    navy = "#14213d"
    muted = "#64748b"
    blue = "#2563eb"
    pale_blue = "#dbeafe"
    green = "#15803d"
    pale_green = "#dcfce7"
    amber = "#b45309"
    pale_amber = "#fef3c7"
    red = "#b91c1c"
    pale_red = "#fee2e2"
    purple = "#7c3aed"
    divider = "#e2e8f0"
    model_label = comparison.attrs.get("model_label", "Model")
    top_label = f"top-{rank_word(top_n)}"
    coverage_column = f"actual_in_top_{top_n}"

    fig.text(
        0.06,
        0.94,
        f"Actual scores vs {top_label} forecasts: {model_label}",
        fontsize=20,
        fontweight="bold",
        color=navy,
        va="top",
    )
    hits = int(comparison[coverage_column].sum())
    outcome_hits = int(comparison["outcome_correct"].sum())
    excluded_count = int(comparison.attrs.get("excluded_count", 0))
    max_observed_goals = comparison.attrs.get("max_observed_goals_per_team")
    sample_text = f"{count} observed matches"
    if max_observed_goals is not None:
        sample_text += (
            f" after excluding {excluded_count} with a team scoring "
            f"more than {max_observed_goals}"
        )
    fig.text(
        0.06,
        0.895,
        (
            f"{sample_text}     "
            f"{top_label.title()} exact-score coverage: {hits}/{count} "
            f"({hits / count:.0%})"
            f"     Correct outcomes: {outcome_hits}/{count} "
            f"({outcome_hits / count:.0%})"
        ),
        fontsize=11.5,
        color=muted,
        va="top",
    )

    headers = [(0.25, "Match"), (5.25, "Actual")]
    headers.extend((x, ordinal_text(rank)) for rank, x in enumerate(score_xs, start=1))
    headers.extend([(coverage_x - 0.3, "Coverage"), (outcome_x - 0.2, "Outcome")])
    y_header = count + 0.15
    for x, label in headers:
        ax.text(
            x,
            y_header,
            label,
            fontsize=10,
            fontweight="bold",
            color=muted,
            va="center",
        )
    ax.plot(
        [0.2, x_limit - 0.25],
        [count - 0.22, count - 0.22],
        color=divider,
        linewidth=1,
    )

    for display_index, row in comparison.reset_index(drop=True).iterrows():
        y = count - 0.8 - display_index
        if display_index % 2 == 1:
            ax.axhspan(y - 0.39, y + 0.39, color="#f8fafc", zorder=0)
        ax.plot(
            [0.2, x_limit - 0.25],
            [y - 0.41, y - 0.41],
            color=divider,
            linewidth=0.75,
        )

        ax.text(
            0.25,
            y + 0.10,
            f"{row.team_a} vs {row.team_b}",
            fontsize=11.4,
            fontweight="bold",
            color=navy,
            va="center",
        )
        ax.text(
            0.25,
            y - 0.18,
            f"Match {int(row.observed_order)}",
            fontsize=8.7,
            color=muted,
            va="center",
        )

        actual_rank = next(
            (
                rank
                for rank in range(1, top_n + 1)
                if bool(row[f"actual_is_top_{rank}"])
            ),
            None,
        )
        if actual_rank == 1:
            status_text, status_color, status_bg = "Top 1", green, pale_green
        elif actual_rank == 2:
            status_text, status_color, status_bg = "Top 2", amber, pale_amber
        elif actual_rank is not None:
            status_text, status_color, status_bg = f"Top {actual_rank}", blue, pale_blue
        else:
            status_text, status_color, status_bg = "Outside", red, pale_red

        ax.text(
            5.55,
            y,
            row.actual_score,
            fontsize=15,
            fontweight="bold",
            color=navy,
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.32",
                "facecolor": status_bg,
                "edgecolor": "none",
            },
        )

        bar_colors = [blue, "#60a5fa", purple, "#14b8a6", "#f97316"]
        for rank, x in enumerate(score_xs, start=1):
            score = row[f"top_{rank}_scoreline"]
            probability = float(row[f"top_{rank}_probability"])
            bar_color = bar_colors[(rank - 1) % len(bar_colors)]
            ax.text(
                x,
                y + 0.10,
                score,
                fontsize=13.5,
                fontweight="bold",
                color=navy,
                va="center",
            )
            track_left = x + 0.78
            track_width = 1.75
            ax.barh(y + 0.08, track_width, height=0.12, left=track_left, color=pale_blue)
            ax.barh(
                y + 0.08,
                track_width * min(probability / 0.13, 1.0),
                height=0.12,
                left=track_left,
                color=bar_color,
            )
            ax.text(
                x,
                y - 0.22,
                f"{probability:.1%}",
                fontsize=9.3,
                color=muted,
                va="center",
            )

        ax.text(
            coverage_x,
            y + 0.07,
            status_text,
            fontsize=10,
            fontweight="bold",
            color=status_color,
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": status_bg,
                "edgecolor": "none",
            },
        )
        ax.text(
            coverage_x,
            y - 0.23,
            f"Actual: {row.actual_score_probability:.1%}",
            fontsize=8.6,
            color=muted,
            ha="center",
            va="center",
        )

        if row.outcome_correct:
            outcome_text, outcome_color, outcome_bg = "Correct", green, pale_green
        else:
            outcome_text, outcome_color, outcome_bg = "Wrong", red, pale_red
        ax.text(
            outcome_x,
            y + 0.07,
            outcome_text,
            fontsize=10,
            fontweight="bold",
            color=outcome_color,
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": outcome_bg,
                "edgecolor": "none",
            },
        )
        ax.text(
            outcome_x,
            y - 0.23,
            (
                f"Pred: {row.predicted_result_label} "
                f"({row.predicted_result_probability:.0%})"
            ),
            fontsize=8.4,
            color=muted,
            ha="center",
            va="center",
        )

    ax.text(
        0.25,
        -0.62,
        (
            "Coverage indicates whether the actual exact score landed in "
            f"the displayed {top_label} exact scorelines or outside them. "
            "Outcome uses the model's W/D/L decision."
        ),
        fontsize=9.2,
        color=muted,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(top=0.83, bottom=0.08, left=0.055, right=0.98)
    fig.savefig(output_path, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-file", default=str(PROJECT_DIR / "v11_wcq_results_model.py")
    )
    parser.add_argument(
        "--model-label",
        default="V11",
        help="Model name displayed in the chart title (default: V11).",
    )
    parser.add_argument(
        "--worldcupsai-zip", default=str(DATA_DIR / "worldcupsai.zip")
    )
    parser.add_argument(
        "--team-train",
        default=str(DATA_DIR / "current_team_features_2026.csv"),
    )
    parser.add_argument("--team-test")
    parser.add_argument(
        "--box-data", default=str(DATA_DIR / "FIFAallMatchBoxData.csv")
    )
    parser.add_argument("--results-data", default=str(DATA_DIR / "results.csv"))
    parser.add_argument(
        "--former-names", default=str(DATA_DIR / "former_names.csv")
    )
    parser.add_argument(
        "--observed",
        default=str(DATA_DIR / "wc2026_observed_matches_from_screenshots.csv"),
    )
    parser.add_argument("--prediction-year", type=int, default=2026)
    parser.add_argument("--model", default="ensemble")
    parser.add_argument(
        "--matches",
        type=int,
        default=7,
        help="Use the first N observed rows (default: 7).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=2,
        help="Number of leading exact scorelines to compare (default: 2).",
    )
    parser.add_argument(
        "--max-observed-goals-per-team",
        type=int,
        help=(
            "Exclude matches where either team scored more than this number "
            "of goals."
        ),
    )
    parser.add_argument(
        "--output-dir", default="observed_eval/observed_eval_v11_with_current"
    )
    parser.add_argument(
        "--output-prefix",
        default="v11",
        help="Prefix used for generated CSV and PNG filenames (default: v11).",
    )
    parser.add_argument(
        "--no-live-updates",
        action="store_true",
        help=(
            "Do not update live model state with observed test-set results "
            "between matches. Use this for strict fixed-test evaluation."
        ),
    )
    args = parser.parse_args()

    if args.matches < 1:
        raise ValueError("--matches must be at least 1")
    if args.top_n < 1:
        raise ValueError("--top-n must be at least 1")
    if (
        args.max_observed_goals_per_team is not None
        and args.max_observed_goals_per_team < 0
    ):
        raise ValueError("--max-observed-goals-per-team cannot be negative")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison = build_comparison(args)
    suffix = f"{len(comparison)}_matches"
    if args.max_observed_goals_per_team is not None:
        suffix += f"_max_{args.max_observed_goals_per_team}_goals"
    top_filename = f"top_{rank_word(args.top_n)}"
    csv_path = output_dir / (
        f"{args.output_prefix}_{top_filename}_scoreline_comparison_{suffix}.csv"
    )
    plot_path = output_dir / (
        f"{args.output_prefix}_{top_filename}_scoreline_comparison_{suffix}.png"
    )
    comparison.to_csv(csv_path, index=False)
    draw_scoreline_chart(comparison, plot_path)

    print(comparison.to_string(index=False))
    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {plot_path}")


if __name__ == "__main__":
    main()
'''
compare_v11_top_scorelines = _load_submodule("compare_v11_top_scorelines", _COMPARE_V11_TOP_SCORELINES_SOURCE, "feature_layers.py:compare_v11_top_scorelines")

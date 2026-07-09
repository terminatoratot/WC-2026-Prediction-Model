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

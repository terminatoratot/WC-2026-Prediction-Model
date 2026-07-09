#!/usr/bin/env python3
"""V39: V38 plus one top-3 total-envelope coverage outlier rule.

This deliberately keeps the normal Top-3 untouched. The only new behavior is
the fourth/outlier slot: when the model's expected total goals sit above the
displayed Top-3 ceiling, V39 uses the highest-probability scoreline from the
next higher total band as the outlier. If the existing V35/V38 game-state
outlier already covers that band, V39 leaves it alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
VERSIONS_DIR = PROJECT_DIR / "versions"
if str(VERSIONS_DIR) not in sys.path:
    sys.path.insert(0, str(VERSIONS_DIR))

import v11_wcq_results_model as v11
import v15_catboost_model as v15
import v28_current_worldcup_form_model as v28
import v35_game_state_late_mutation_model as v35
import v38_total_lambda_calibrated_model as v38


ScoreMatrix = Dict[Tuple[int, int], float]

DEFAULT_COVERAGE_MARGIN = 0.0
DEFAULT_GAME_STATE_ANALYSIS_DIR = PROJECT_DIR / "analysis" / "match_data_avenues_latest"
OBSERVED_TRANSITION_CHECKPOINT_WEIGHTS = {
    "halftime": 0.20,
    "second_hydration_proxy": 0.55,
    "stoppage_start": 0.25,
}


def _score_key(item: Dict[str, Any]) -> Tuple[int, int]:
    return int(item["team_a_goals"]), int(item["team_b_goals"])


def _score_label(key: Tuple[int, int]) -> str:
    return f"{key[0]}-{key[1]}"


def _parse_score_label(value: object) -> Tuple[int, int] | None:
    try:
        left, right = str(value).split("-", 1)
        return int(left), int(right)
    except (TypeError, ValueError):
        return None


def _mirror_score(key: Tuple[int, int]) -> Tuple[int, int]:
    return int(key[1]), int(key[0])


def _leader(key: Tuple[int, int]) -> str:
    if key[0] > key[1]:
        return "team_a"
    if key[1] > key[0]:
        return "team_b"
    return "draw"


def _score_item(key: Tuple[int, int], probability: float, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "team_a_goals": int(key[0]),
        "team_b_goals": int(key[1]),
        "probability": float(probability),
        "rank_label": "coverage_total_outlier",
        "kind": "top3_total_envelope_coverage",
        **diagnostics,
    }


class ObservedGameStatePriors:
    """Observed checkpoint priors from analysis/match_data_avenues_latest."""

    def __init__(self, analysis_dir: str | Path = DEFAULT_GAME_STATE_ANALYSIS_DIR):
        self.analysis_dir = Path(analysis_dir)
        self.transition_rows: dict[str, dict[Tuple[int, int], dict[Tuple[int, int], dict[str, float]]]] = {}
        self.source_support: dict[str, dict[Tuple[int, int], int]] = {}
        self.survival_rows: dict[str, dict[int, dict[str, float]]] = {}
        self.state_goal_rows: dict[str, dict[str, float]] = {}
        self.team_late_rows: dict[str, dict[str, float]] = {}
        self.loaded_files: list[str] = []
        self._load()

    @property
    def available(self) -> bool:
        return bool(self.transition_rows or self.survival_rows or self.state_goal_rows or self.team_late_rows)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "analysis_dir": str(self.analysis_dir),
            "available": self.available,
            "loaded_files": self.loaded_files,
            "transition_checkpoints": sorted(self.transition_rows),
            "survival_checkpoints": sorted(self.survival_rows),
            "team_profile_count": len(self.team_late_rows),
        }

    def _load(self) -> None:
        for checkpoint in OBSERVED_TRANSITION_CHECKPOINT_WEIGHTS:
            path = self.analysis_dir / f"exact_score_transition_map_{checkpoint}.csv"
            if path.exists():
                self._load_transition_map(path, checkpoint)
        survival_path = self.analysis_dir / "total_goal_survival_summary.csv"
        if survival_path.exists():
            self._load_survival_summary(survival_path)
        state_path = self.analysis_dir / "score_state_goal_summary.csv"
        if state_path.exists():
            self._load_score_state_summary(state_path)
        team_path = self.analysis_dir / "team_late_and_post65_profiles.csv"
        if team_path.exists():
            self._load_team_profiles(team_path)

    def _load_transition_map(self, path: Path, checkpoint: str) -> None:
        frame = pd.read_csv(path)
        rows: dict[Tuple[int, int], dict[Tuple[int, int], dict[str, float]]] = {}
        support: dict[Tuple[int, int], int] = {}
        for row in frame.to_dict(orient="records"):
            source = _parse_score_label(row.get("score_at_checkpoint"))
            final = _parse_score_label(row.get("final_score"))
            if source is None or final is None:
                continue
            matches = int(float(row.get("matches", 0) or 0))
            rows.setdefault(source, {})[final] = {
                "probability": float(row.get("probability", 0.0) or 0.0),
                "matches": float(matches),
            }
            support[source] = support.get(source, 0) + matches
        self.transition_rows[checkpoint] = rows
        self.source_support[checkpoint] = support
        self.loaded_files.append(str(path))

    def _load_survival_summary(self, path: Path) -> None:
        frame = pd.read_csv(path)
        for row in frame.to_dict(orient="records"):
            checkpoint = str(row.get("checkpoint", ""))
            total = int(float(row.get("total_at_checkpoint", 0) or 0))
            self.survival_rows.setdefault(checkpoint, {})[total] = {
                "matches": float(row.get("matches", 0.0) or 0.0),
                "p_1plus_more": float(row.get("p_1plus_more", 0.0) or 0.0),
                "p_2plus_more": float(row.get("p_2plus_more", 0.0) or 0.0),
                "p_3plus_more": float(row.get("p_3plus_more", 0.0) or 0.0),
            }
        self.loaded_files.append(str(path))

    def _load_score_state_summary(self, path: Path) -> None:
        frame = pd.read_csv(path)
        for row in frame.to_dict(orient="records"):
            state = str(row.get("scoring_team_state_before", ""))
            goals = float(row.get("goals", 0.0) or 0.0)
            self.state_goal_rows[state] = {
                "share": float(row.get("share", 0.0) or 0.0),
                "late_share": float(row.get("late_75_plus", 0.0) or 0.0) / max(goals, 1.0),
            }
        self.loaded_files.append(str(path))

    def _load_team_profiles(self, path: Path) -> None:
        frame = pd.read_csv(path)
        for row in frame.to_dict(orient="records"):
            team = v35.canon(row.get("team"))
            if not team:
                continue
            self.team_late_rows[team] = {
                "late_for_75_plus": float(row.get("late_for_75_plus", 0.0) or 0.0),
                "post65_for": float(row.get("post65_for", 0.0) or 0.0),
                "post65_against": float(row.get("post65_against", 0.0) or 0.0),
            }
        self.loaded_files.append(str(path))

    def transition_probability(self, checkpoint: str, source: Tuple[int, int], final: Tuple[int, int]) -> tuple[float, int]:
        values = []
        for src, dst in [(source, final), (_mirror_score(source), _mirror_score(final))]:
            row = self.transition_rows.get(checkpoint, {}).get(src, {}).get(dst)
            if row:
                values.append((float(row["probability"]), int(row["matches"])))
        if not values:
            return 0.0, 0
        return max(values, key=lambda item: (item[0], item[1]))

    def transition_score(self, source: Tuple[int, int], final: Tuple[int, int]) -> tuple[float, dict[str, Any]]:
        numer = 0.0
        denom = 0.0
        best_probability = 0.0
        best_checkpoint = ""
        total_match_support = 0
        for checkpoint, weight in OBSERVED_TRANSITION_CHECKPOINT_WEIGHTS.items():
            probability, matches = self.transition_probability(checkpoint, source, final)
            support = max(
                self.source_support.get(checkpoint, {}).get(source, 0),
                self.source_support.get(checkpoint, {}).get(_mirror_score(source), 0),
            )
            support_weight = min(1.0, support / 8.0) if support else 0.0
            weighted = weight * support_weight
            numer += weighted * probability
            denom += weighted
            total_match_support += matches
            if probability > best_probability:
                best_probability = probability
                best_checkpoint = checkpoint
        score = numer / denom if denom > 0 else 0.0
        return score, {
            "transition_score": float(score),
            "best_transition_probability": float(best_probability),
            "best_transition_checkpoint": best_checkpoint,
            "transition_match_support": int(total_match_support),
        }

    def survival_score(self, source: Tuple[int, int], final: Tuple[int, int]) -> tuple[float, dict[str, Any]]:
        add_goals = sum(final) - sum(source)
        if add_goals <= 0:
            return 0.0, {"survival_score": 0.0, "survival_added_goals": int(add_goals)}
        key = "p_1plus_more" if add_goals == 1 else "p_2plus_more" if add_goals == 2 else "p_3plus_more"
        numer = 0.0
        denom = 0.0
        total_support = 0.0
        for checkpoint, weight in OBSERVED_TRANSITION_CHECKPOINT_WEIGHTS.items():
            row = self.survival_rows.get(checkpoint, {}).get(sum(source))
            if not row:
                continue
            support_weight = min(1.0, float(row.get("matches", 0.0)) / 10.0)
            weighted = weight * support_weight
            numer += weighted * float(row.get(key, 0.0))
            denom += weighted
            total_support += float(row.get("matches", 0.0))
        score = numer / denom if denom > 0 else 0.0
        return score, {
            "survival_score": float(score),
            "survival_added_goals": int(add_goals),
            "survival_support": float(total_support),
        }

    def state_goal_score(self, source: Tuple[int, int], final: Tuple[int, int]) -> tuple[float, dict[str, Any]]:
        add_a = max(final[0] - source[0], 0)
        add_b = max(final[1] - source[1], 0)
        if add_a == 0 and add_b == 0:
            return 0.0, {"state_goal_score": 0.0, "scoring_state": "none"}
        leader = _leader(source)
        if leader == "draw":
            state = "drawing"
        elif (leader == "team_a" and add_a > 0) or (leader == "team_b" and add_b > 0):
            state = "leading"
        else:
            state = "trailing"
        row = self.state_goal_rows.get(state, {})
        score = 0.65 * float(row.get("share", 0.0)) + 0.35 * float(row.get("late_share", 0.0))
        if add_a > 0 and add_b > 0:
            score *= 1.10
        return score, {"state_goal_score": float(score), "scoring_state": state}

    def team_late_score(
        self,
        source: Tuple[int, int],
        final: Tuple[int, int],
        team_a: object = "",
        team_b: object = "",
    ) -> tuple[float, dict[str, Any]]:
        add_a = max(final[0] - source[0], 0)
        add_b = max(final[1] - source[1], 0)
        team_a_profile = self.team_late_rows.get(v35.canon(team_a), {})
        team_b_profile = self.team_late_rows.get(v35.canon(team_b), {})

        def side_score(add: int, own: dict[str, float], opp: dict[str, float]) -> float:
            if add <= 0 or not own:
                return 0.0
            raw = (
                0.55 * float(own.get("post65_for", 0.0))
                + 0.30 * float(own.get("late_for_75_plus", 0.0))
                + 0.15 * float(opp.get("post65_against", 0.0))
            )
            return min(1.0, raw / 4.0) * min(1.0, add / 2.0)

        score = max(side_score(add_a, team_a_profile, team_b_profile), side_score(add_b, team_b_profile, team_a_profile))
        return score, {
            "team_late_score": float(score),
            "team_a_observed_profile_found": bool(team_a_profile),
            "team_b_observed_profile_found": bool(team_b_profile),
        }


_DEFAULT_OBSERVED_PRIORS: ObservedGameStatePriors | None = None


def default_observed_game_state_priors() -> ObservedGameStatePriors:
    global _DEFAULT_OBSERVED_PRIORS
    if _DEFAULT_OBSERVED_PRIORS is None:
        _DEFAULT_OBSERVED_PRIORS = ObservedGameStatePriors(DEFAULT_GAME_STATE_ANALYSIS_DIR)
    return _DEFAULT_OBSERVED_PRIORS


def score_matrix_from_prediction(prediction: Dict[str, Any]) -> ScoreMatrix:
    return {
        (int(item["team_a_goals"]), int(item["team_b_goals"])): float(item["probability"])
        for item in prediction.get("scoreline_probabilities", [])
    }


def _observed_candidate_utility(
    key: Tuple[int, int],
    score_matrix: ScoreMatrix,
    top_three: list[Tuple[int, int]],
    observed_priors: ObservedGameStatePriors | None,
    team_a: object = "",
    team_b: object = "",
) -> dict[str, Any]:
    probability = float(score_matrix.get(key, 0.0))
    if not observed_priors or not observed_priors.available:
        return {
            "scoreline": _score_label(key),
            "raw_probability": probability,
            "observed_utility": probability,
            "observed_lift": 0.0,
            "observed_source_scoreline": "",
            "observed_priors_available": False,
        }

    best: dict[str, Any] | None = None
    for source in top_three:
        if sum(key) <= sum(source):
            continue
        transition_score, transition_diag = observed_priors.transition_score(source, key)
        survival_score, survival_diag = observed_priors.survival_score(source, key)
        state_score, state_diag = observed_priors.state_goal_score(source, key)
        team_score, team_diag = observed_priors.team_late_score(source, key, team_a, team_b)
        observed_lift = (
            1.75 * transition_score
            + 0.80 * survival_score
            + 0.45 * state_score
            + 0.35 * team_score
        )
        utility = probability * (1.0 + observed_lift)
        candidate = {
            "scoreline": _score_label(key),
            "raw_probability": probability,
            "observed_utility": float(utility),
            "observed_lift": float(observed_lift),
            "observed_source_scoreline": _score_label(source),
            "observed_priors_available": True,
            **transition_diag,
            **survival_diag,
            **state_diag,
            **team_diag,
        }
        if best is None or candidate["observed_utility"] > best["observed_utility"]:
            best = candidate
    if best is None:
        return {
            "scoreline": _score_label(key),
            "raw_probability": probability,
            "observed_utility": probability,
            "observed_lift": 0.0,
            "observed_source_scoreline": "",
            "observed_priors_available": observed_priors.available,
        }
    return best


def select_coverage_outlier(
    score_matrix: ScoreMatrix,
    top_scorelines: list[Dict[str, Any]],
    lambda_a: float,
    lambda_b: float,
    existing_outlier: Dict[str, Any] | None = None,
    margin: float = DEFAULT_COVERAGE_MARGIN,
    observed_priors: ObservedGameStatePriors | None = None,
    use_observed_game_state_priors: bool = True,
    team_a: object = "",
    team_b: object = "",
) -> tuple[Dict[str, Any] | None, Dict[str, Any]]:
    top_three = [_score_key(item) for item in top_scorelines[:3]]
    lambda_sum = float(lambda_a) + float(lambda_b)
    if use_observed_game_state_priors and observed_priors is None:
        observed_priors = default_observed_game_state_priors()
    diagnostics: Dict[str, Any] = {
        "selector": "top3_total_envelope_coverage_outlier",
        "top_three_changed": False,
        "probability_matrix_changed": False,
        "coverage_margin": float(margin),
        "lambda_sum": lambda_sum,
        "base_top_3": [_score_label(key) for key in top_three],
        "observed_game_state_priors": (
            observed_priors.diagnostics()
            if use_observed_game_state_priors and observed_priors is not None
            else {"available": False, "disabled": True}
        ),
    }
    if len(top_three) < 3:
        diagnostics["skip_reason"] = "not_enough_top_scorelines"
        return existing_outlier, diagnostics

    top3_max_total = max(sum(key) for key in top_three)
    top3_min_total = min(sum(key) for key in top_three)
    trigger_level = top3_max_total + float(margin)
    min_candidate_total = top3_max_total + 1
    diagnostics.update(
        {
            "top3_min_total": int(top3_min_total),
            "top3_max_total": int(top3_max_total),
            "trigger_level": float(trigger_level),
            "min_candidate_total": int(min_candidate_total),
            "coverage_triggered": bool(lambda_sum > trigger_level),
        }
    )
    if lambda_sum <= trigger_level:
        diagnostics["skip_reason"] = "lambda_sum_not_above_top3_ceiling"
        return existing_outlier, diagnostics

    existing_key = _score_key(existing_outlier) if existing_outlier else None
    if existing_key is not None and sum(existing_key) >= min_candidate_total:
        diagnostics.update(
            {
                "existing_outlier_already_covers_higher_total": True,
                "existing_outlier_scoreline": _score_label(existing_key),
                "existing_outlier_total": int(sum(existing_key)),
            }
        )
        if not use_observed_game_state_priors:
            diagnostics["skip_reason"] = "existing_outlier_already_covers_higher_total"
            return existing_outlier, diagnostics

    top_set = set(top_three)
    candidates = [
        key
        for key in score_matrix
        if key not in top_set and sum(key) >= min_candidate_total
    ]
    diagnostics["candidate_count"] = int(len(candidates))
    if not candidates:
        diagnostics["skip_reason"] = "no_higher_total_candidate"
        return existing_outlier, diagnostics

    candidate_details = {
        key: _observed_candidate_utility(
            key,
            score_matrix,
            top_three,
            observed_priors if use_observed_game_state_priors else None,
            team_a=team_a,
            team_b=team_b,
        )
        for key in candidates
    }
    best_key = max(candidates, key=lambda key: candidate_details[key]["observed_utility"])
    top_candidate_details = sorted(
        candidate_details.values(),
        key=lambda row: (float(row["observed_utility"]), float(row["raw_probability"])),
        reverse=True,
    )[:10]
    best_probability = float(score_matrix.get(best_key, 0.0))
    best_details = candidate_details[best_key]
    selected_diagnostics = {
        "source": "v39_coverage_outlier",
        "coverage_replaced_existing_outlier": bool(existing_outlier),
        "candidate_scoreline": _score_label(best_key),
        "candidate_total": int(sum(best_key)),
        "candidate_probability": best_probability,
        "candidate_observed_utility": float(best_details["observed_utility"]),
        "candidate_observed_lift": float(best_details["observed_lift"]),
        "candidate_observed_source_scoreline": best_details["observed_source_scoreline"],
    }
    diagnostics.update(
        {
            **selected_diagnostics,
            "outlier_selected": True,
            "outlier_scoreline": _score_label(best_key),
            "outlier_probability": best_probability,
            "top_observed_game_state_candidates": top_candidate_details,
        }
    )
    return _score_item(best_key, best_probability, selected_diagnostics), diagnostics


class V39CoverageOutlierModel:
    """Wrap V38 and replace only the fourth outlier when total coverage is thin."""

    def __init__(
        self,
        base_model: v38.V38TotalLambdaCalibratedModel,
        coverage_margin: float = DEFAULT_COVERAGE_MARGIN,
        observed_game_state_priors: ObservedGameStatePriors | None = None,
        use_observed_game_state_priors: bool = True,
    ):
        self.base_model = base_model
        self.coverage_margin = float(max(coverage_margin, 0.0))
        self.use_observed_game_state_priors = bool(use_observed_game_state_priors)
        self.observed_game_state_priors = (
            observed_game_state_priors
            if observed_game_state_priors is not None
            else default_observed_game_state_priors()
            if self.use_observed_game_state_priors
            else None
        )
        self.training_data_summary = getattr(base_model, "training_data_summary", {})

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_model, name)

    def predict(self, *args, **kwargs) -> Dict[str, Any]:
        team_a = kwargs.get("team_a", args[0] if args else "")
        team_b = kwargs.get("team_b", args[1] if len(args) > 1 else "")
        prediction = self.base_model.predict(*args, **kwargs)
        score_matrix = score_matrix_from_prediction(prediction)
        existing_outlier = prediction.get("game_state_late_outlier")
        coverage_outlier, diagnostics = select_coverage_outlier(
            score_matrix,
            prediction.get("top_scorelines", []),
            float(prediction["lambda_a"]),
            float(prediction["lambda_b"]),
            existing_outlier=existing_outlier,
            margin=self.coverage_margin,
            observed_priors=self.observed_game_state_priors,
            use_observed_game_state_priors=self.use_observed_game_state_priors,
            team_a=team_a,
            team_b=team_b,
        )
        prediction["v39_original_game_state_late_outlier"] = existing_outlier
        prediction["coverage_total_outlier"] = coverage_outlier
        prediction["game_state_late_outlier"] = coverage_outlier
        prediction["late_instability_outlier"] = coverage_outlier
        prediction["outlier_scoreline"] = coverage_outlier
        prediction["top_scorelines_plus_outlier"] = [
            *prediction.get("top_scorelines", [])[:3],
            *([coverage_outlier] if coverage_outlier else []),
        ]
        prediction["v39_adjustments"] = {
            "base_model": "v38_total_lambda_calibrated",
            "scoreline_policy": "top_3_preserved_plus_observed_game_state_coverage_outlier",
            "scoreline_layer_affects_wdl": False,
            **diagnostics,
        }
        prediction["calibration_notes"] = {
            **prediction.get("calibration_notes", {}),
            "v39": prediction["v39_adjustments"],
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
    bootstrap_samples=v38.DEFAULT_BOOTSTRAP_SAMPLES,
    shrinkage_prior_sd=v38.DEFAULT_SHRINKAGE_PRIOR_SD,
    random_seed=v38.DEFAULT_RANDOM_SEED,
    coverage_margin=DEFAULT_COVERAGE_MARGIN,
    game_state_analysis_dir=DEFAULT_GAME_STATE_ANALYSIS_DIR,
    use_observed_game_state_priors=True,
    **kwargs,
):
    base_model, data = v38.build_from_zip(
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
        bootstrap_samples=bootstrap_samples,
        shrinkage_prior_sd=shrinkage_prior_sd,
        random_seed=random_seed,
        **kwargs,
    )
    observed_priors = (
        ObservedGameStatePriors(game_state_analysis_dir)
        if use_observed_game_state_priors
        else None
    )
    model = V39CoverageOutlierModel(
        base_model,
        coverage_margin=coverage_margin,
        observed_game_state_priors=observed_priors,
        use_observed_game_state_priors=use_observed_game_state_priors,
    )
    model.training_data_summary = {
        **getattr(base_model, "training_data_summary", {}),
        "v39_scoreline_policy": "top_3_preserved_plus_observed_game_state_coverage_outlier",
        "v39_probability_matrix_changed": False,
        "v39_coverage_margin": float(model.coverage_margin),
        "v39_observed_game_state_priors": (
            observed_priors.diagnostics() if observed_priors else {"available": False, "disabled": True}
        ),
    }
    return model, data


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    parser = argparse.ArgumentParser(description="Run V39: V38 plus one total-coverage outlier.")
    parser.add_argument("--team-a", required=True)
    parser.add_argument("--team-b", required=True)
    parser.add_argument("--host-a", action="store_true")
    parser.add_argument("--host-b", action="store_true")
    parser.add_argument("--knockout", action="store_true")
    parser.add_argument("--outdir", default="outputs/outputs_v39_coverage_outlier")
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
    parser.add_argument("--bootstrap-samples", type=int, default=v38.DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--shrinkage-prior-sd", type=float, default=v38.DEFAULT_SHRINKAGE_PRIOR_SD)
    parser.add_argument("--coverage-margin", type=float, default=DEFAULT_COVERAGE_MARGIN)
    parser.add_argument("--game-state-analysis-dir", default=str(DEFAULT_GAME_STATE_ANALYSIS_DIR))
    parser.add_argument("--disable-observed-game-state-priors", action="store_true")
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
        coverage_margin=args.coverage_margin,
        game_state_analysis_dir=args.game_state_analysis_dir,
        use_observed_game_state_priors=not args.disable_observed_game_state_priors,
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
        output_dir / "scoreline_probabilities_top_plus_coverage_outlier.csv",
        index=False,
    )
    (output_dir / "model_summary.json").write_text(
        json.dumps(
            {
                "version": "v39-coverage-outlier",
                "base_model": "v38-total-lambda-calibrated",
                "team_a": prediction["team_a"],
                "team_b": prediction["team_b"],
                "result_probabilities": prediction["result_probabilities"],
                "top_3": prediction["top_scorelines"][:3],
                "coverage_total_outlier": prediction["coverage_total_outlier"],
                "v39_adjustments": prediction["v39_adjustments"],
                "v38_adjustments": prediction.get("v38_adjustments", {}),
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
                "coverage_total_outlier": prediction["coverage_total_outlier"],
                "v39_adjustments": prediction["v39_adjustments"],
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

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

import feature_layers  # noqa: F401  (loads its own submodules into sys.modules)


# ======================================================================
# v39_coverage_outlier_model.py  (bundled as an isolated sub-module)
# ======================================================================
_V39_COVERAGE_OUTLIER_MODEL_SOURCE = r'''
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
'''
v39_coverage_outlier_model = _load_submodule("v39_coverage_outlier_model", _V39_COVERAGE_OUTLIER_MODEL_SOURCE, "market_edge.py:v39_coverage_outlier_model")

# ======================================================================
# v39_withbetterdata.py  (bundled as an isolated sub-module)
# ======================================================================
_V39_WITHBETTERDATA_SOURCE = r'''
#!/usr/bin/env python3
"""v39 analysis layer using the combined FotMob + Database feature set.

Full feature run:
    MPLCONFIGDIR=.matplotlib_cache .venv/bin/python v39_withbetterdata.py --team-a "Argentina" --team-b "Cape Verde" --knockout --outdir outputs/outputs_v39_argentina_france --fotmob-leaders data/fotmob_full_stat_tables_clean.csv --fotmob-player-stats data/fotmob_match_player_stats_clean.csv --fotmob-lineups data/fotmob_match_lineups_clean.csv --fotmob-substitutions data/fotmob_match_substitutions_clean.csv --fotmob-keeper-stats data/fotmob_match_keeper_stats_clean.csv --fotmob-match-facts data/fotmob_match_facts_clean.csv --fotmob-goal-events data/fotmob_match_goal_events_clean.csv --betterdata-profile-csv analysis/v39_withbetterdata_latest/v39_withbetterdata_team_profiles.csv --betterdata-scoreline-blend 0.00 --betterdata-wdl-blend 0.25 --betterdata-max-log-adjustment 0.12 --coverage-margin 0.0

Analysis refresh only:
    MPLCONFIGDIR=.matplotlib_cache .venv/bin/python v39_withbetterdata.py --analysis-only --refresh-analysis

This script assumes `versions/build_combined_fotmob_database_features.py` has
already built `analysis/combined_fotmob_database_latest`. It turns that joined
data into analysis frames and plots for:

- upgraded match-level features
- live mutation / pressure diagnostics
- set-piece pressure
- discipline instability
- market calibration
- half-time adjustment
- richer team profiles and style clusters
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
import v39_coverage_outlier_model as v39


ROOT = PROJECT_DIR
COMBINED = ROOT / "analysis" / "combined_fotmob_database_latest"
OUTDIR = ROOT / "analysis" / "v39_withbetterdata_latest"

DEFAULT_BETTERDATA_SCORELINE_BLEND = 0.00
DEFAULT_BETTERDATA_WDL_BLEND = 0.25
DEFAULT_BETTERDATA_MAX_LOG_ADJUSTMENT = 0.12
PLOTDIR = OUTDIR / "plots"

COLORS = {
    "ink": "#17212b",
    "muted": "#667085",
    "grid": "#e6e8ee",
    "blue": "#276ef1",
    "cyan": "#18a0b5",
    "green": "#1f9d55",
    "yellow": "#f2b84b",
    "red": "#d64545",
    "purple": "#7a5af8",
    "gray": "#98a2b3",
}


def read_combined(name: str) -> pd.DataFrame:
    path = COMBINED / name
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: MPLCONFIGDIR=.matplotlib_cache .venv/bin/python "
            "versions/build_combined_fotmob_database_features.py"
        )
    return pd.read_csv(path)


def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    std = s.std(ddof=0)
    if pd.isna(std) or std == 0:
        return s * 0
    return (s - s.mean()) / std


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return pd.to_numeric(num, errors="coerce") / pd.to_numeric(den, errors="coerce").replace(0, np.nan)


def setup(title: str, subtitle: str, figsize=(11, 7)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_title(title, loc="left", fontsize=18, fontweight="bold", color=COLORS["ink"], pad=18)
    ax.text(
        0,
        1.02,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        color=COLORS["muted"],
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(COLORS["grid"])
    ax.tick_params(colors=COLORS["muted"], labelsize=10)
    ax.grid(True, color=COLORS["grid"], linewidth=0.8, zorder=0)
    return fig, ax


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(PLOTDIR / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_betterdata_features(team_match: pd.DataFrame) -> pd.DataFrame:
    df = team_match.copy()
    df["match_label"] = df["team"] + " vs " + df["opponent"]
    df["post75_scored"] = (df["fotmob_live_future_goals_after_cutoff_75"].fillna(0) > 0).astype(int)
    df["post75_multi_scored"] = (df["fotmob_live_future_goals_after_cutoff_75"].fillna(0) >= 2).astype(int)

    df["corners_per_xg"] = safe_div(df["db_corners_for_ft"], df["db_xg_for_ft"])
    df["corners_per_touch_box"] = safe_div(df["db_corners_for_ft"], df["touches_box_for"])
    df["corner_pressure_minus_opponent"] = (
        zscore(df["db_corner_diff_ft"]) + 0.5 * zscore(df["db_corners_for_2h"]) + 0.35 * zscore(df["corners_per_touch_box"])
    )
    df["set_piece_pressure_index"] = (
        zscore(df["db_corners_for_ft"])
        + 0.65 * zscore(df["db_corner_diff_ft"])
        + 0.45 * zscore(df["db_corners_for_2h"])
        + 0.30 * zscore(df["corners_per_xg"].replace([np.inf, -np.inf], np.nan).fillna(0))
    )

    df["discipline_instability_index"] = (
        zscore(df["db_total_yellow_cards_ft"])
        + 0.65 * zscore(df["db_yellow_cards_for_ft"] + df["db_yellow_cards_against_ft"])
        + 0.45 * zscore(df["db_yellow_cards_for_2h"] + df["db_yellow_cards_against_2h"])
        + 0.25 * zscore(df["db_total_corners_ft"])
    )

    df["team_2h_attack_swing"] = (
        zscore(df["db_xg_for_2h"] - df["db_xg_for_1h"])
        + 0.55 * zscore(df["db_shots_for_2h"] - df["db_shots_for_1h"])
        + 0.55 * zscore(df["db_sot_for_2h"] - df["db_sot_for_1h"])
        + 0.35 * zscore(df["db_corners_for_2h"] - df["db_corners_for_1h"])
    )
    df["opponent_2h_attack_swing"] = (
        zscore(df["db_xg_against_2h"] - df["db_xg_against_1h"])
        + 0.55 * zscore(df["db_shots_against_2h"] - df["db_shots_against_1h"])
        + 0.55 * zscore(df["db_sot_against_2h"] - df["db_sot_against_1h"])
        + 0.35 * zscore(df["db_corners_against_2h"] - df["db_corners_against_1h"])
    )
    df["second_half_net_surge_index"] = df["team_2h_attack_swing"] - df["opponent_2h_attack_swing"]

    df["live_pressure_index_75_descriptive"] = (
        zscore(df["db_xg_for_2h"])
        + 0.65 * zscore(df["db_corners_for_2h"])
        + 0.55 * zscore(df["db_sot_for_2h"])
        + 0.35 * zscore(df["db_possession_for_2h"] - df["db_possession_against_2h"])
        + 0.20 * zscore(df["fotmob_live_subs_used_75"].fillna(0))
        - 0.20 * zscore(df["db_yellow_cards_for_2h"])
    )

    df["halftime_live_safe_pressure_index"] = (
        zscore(df["db_xg_for_1h"])
        + 0.55 * zscore(df["db_corners_for_1h"])
        + 0.50 * zscore(df["db_sot_for_1h"])
        + 0.30 * zscore(df["db_possession_for_1h"] - df["db_possession_against_1h"])
        - 0.15 * zscore(df["db_yellow_cards_for_1h"])
    )

    df["market_xg_edge_residual"] = df["db_xg_diff_ft"] - (df["db_team_win_prob"] - df["db_opponent_win_prob"]) * 3.0
    df["over25_xg_residual"] = df["db_total_xg_ft"] - (1.2 + 3.0 * df["db_implied_over25_prob"])
    df["btts_xg_support"] = np.minimum(df["db_xg_for_ft"], df["db_xg_against_ft"])
    df["btts_xg_residual"] = df["btts_xg_support"] - df["db_implied_btts_yes_prob"]

    df["pressure_tier_75"] = pd.qcut(
        df["live_pressure_index_75_descriptive"].rank(method="first"),
        q=3,
        labels=["low pressure", "mid pressure", "high pressure"],
    )
    df["halftime_pressure_tier"] = pd.qcut(
        df["halftime_live_safe_pressure_index"].rank(method="first"),
        q=3,
        labels=["low HT pressure", "mid HT pressure", "high HT pressure"],
    )
    return df


def build_match_feature_upgrade(match_features: pd.DataFrame) -> pd.DataFrame:
    df = match_features.copy()
    df["match"] = df["database_home_team"] + " vs " + df["database_away_team"]
    df["actual_total_goals"] = df["fotmob_total_goals"]
    df["actual_btts"] = ((df["fotmob_home_goals_for"] > 0) & (df["fotmob_away_goals_for"] > 0)).astype(int)
    df["actual_over25"] = (df["actual_total_goals"] > 2.5).astype(int)
    df["both_team_xg_min"] = df[["db_xg_homeXGFT", "db_xg_awayXGFT"]].min(axis=1)
    df["market_total_expectation_proxy"] = 1.2 + 3.0 * df["db_implied_over25_prob"]
    df["total_xg_market_residual"] = df["db_total_xg_ft"] - df["market_total_expectation_proxy"]
    df["match_instability_index"] = zscore(df["db_total_yellow_cards_ft"]) + 0.35 * zscore(df["db_total_corners_ft"])
    df["match_attack_volume_index"] = zscore(df["db_total_xg_ft"]) + 0.45 * zscore(df["db_total_shots_ft"]) + 0.40 * zscore(df["db_total_sot_ft"])
    return df


def summarize_live_mutation(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["fotmob_live_score_state_75", "pressure_tier_75"]
    out = (
        df.groupby(group_cols, observed=True)
        .agg(
            team_states=("match_id", "count"),
            p_score_after_75=("post75_scored", "mean"),
            p_multi_score_after_75=("post75_multi_scored", "mean"),
            avg_goals_after_75=("fotmob_live_future_goals_after_cutoff_75", "mean"),
            avg_2h_xg=("db_xg_for_2h", "mean"),
            avg_2h_corners=("db_corners_for_2h", "mean"),
            avg_2h_sot=("db_sot_for_2h", "mean"),
            avg_subs_by_75=("fotmob_live_subs_used_75", "mean"),
        )
        .reset_index()
        .sort_values(["p_score_after_75", "team_states"], ascending=[False, False])
    )
    return out


def summarize_set_pieces(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("team")
        .agg(
            matches=("match_id", "count"),
            corners_for=("db_corners_for_ft", "sum"),
            corners_against=("db_corners_against_ft", "sum"),
            corners_2h=("db_corners_for_2h", "sum"),
            touches_box_for=("touches_box_for", "sum"),
            xg_for=("db_xg_for_ft", "sum"),
            goals_for=("goals_for", "sum"),
            avg_set_piece_pressure=("set_piece_pressure_index", "mean"),
            avg_corner_pressure_minus_opponent=("corner_pressure_minus_opponent", "mean"),
        )
        .reset_index()
        .assign(
            corners_per_match=lambda d: d["corners_for"] / d["matches"],
            corners_per_xg=lambda d: d["corners_for"] / d["xg_for"].replace(0, np.nan),
            corners_per_touch_box=lambda d: d["corners_for"] / d["touches_box_for"].replace(0, np.nan),
            corner_diff=lambda d: d["corners_for"] - d["corners_against"],
            corner_2h_share=lambda d: d["corners_2h"] / d["corners_for"].replace(0, np.nan),
        )
        .sort_values("avg_set_piece_pressure", ascending=False)
    )


def summarize_discipline(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("team")
        .agg(
            matches=("match_id", "count"),
            yellow_for=("db_yellow_cards_for_ft", "sum"),
            yellow_against=("db_yellow_cards_against_ft", "sum"),
            yellow_2h_for=("db_yellow_cards_for_2h", "sum"),
            total_match_yellows_avg=("db_total_yellow_cards_ft", "mean"),
            avg_instability=("discipline_instability_index", "mean"),
            late_goals_for=("fotmob_live_future_goals_after_cutoff_75", "sum"),
        )
        .reset_index()
        .assign(
            yellow_for_per_match=lambda d: d["yellow_for"] / d["matches"],
            yellow_against_per_match=lambda d: d["yellow_against"] / d["matches"],
            yellow_2h_share=lambda d: d["yellow_2h_for"] / d["yellow_for"].replace(0, np.nan),
        )
        .sort_values("avg_instability", ascending=False)
    )


def summarize_market(match_df: pd.DataFrame) -> pd.DataFrame:
    out = match_df[
        [
            "match_id",
            "database_id",
            "match",
            "db_implied_favorite_prob",
            "db_market_favorite_team",
            "db_implied_over25_prob",
            "db_implied_btts_yes_prob",
            "db_total_xg_ft",
            "both_team_xg_min",
            "actual_total_goals",
            "actual_over25",
            "actual_btts",
            "total_xg_market_residual",
            "match_attack_volume_index",
            "match_instability_index",
        ]
    ].copy()
    out["over25_market_error"] = out["actual_over25"] - out["db_implied_over25_prob"]
    out["btts_market_error"] = out["actual_btts"] - out["db_implied_btts_yes_prob"]
    return out.sort_values("total_xg_market_residual", ascending=False)


def summarize_halftime(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("team")
        .agg(
            matches=("match_id", "count"),
            xg_1h=("db_xg_for_1h", "sum"),
            xg_2h=("db_xg_for_2h", "sum"),
            shots_1h=("db_shots_for_1h", "sum"),
            shots_2h=("db_shots_for_2h", "sum"),
            sot_1h=("db_sot_for_1h", "sum"),
            sot_2h=("db_sot_for_2h", "sum"),
            corners_1h=("db_corners_for_1h", "sum"),
            corners_2h=("db_corners_for_2h", "sum"),
            avg_second_half_net_surge=("second_half_net_surge_index", "mean"),
            post75_goals=("fotmob_live_future_goals_after_cutoff_75", "sum"),
        )
        .reset_index()
    )
    out["xg_2h_minus_1h"] = out["xg_2h"] - out["xg_1h"]
    out["shots_2h_minus_1h"] = out["shots_2h"] - out["shots_1h"]
    out["corners_2h_minus_1h"] = out["corners_2h"] - out["corners_1h"]
    return out.sort_values("avg_second_half_net_surge", ascending=False)


def build_better_team_profiles(team_df: pd.DataFrame, team_tournament: pd.DataFrame) -> pd.DataFrame:
    set_piece = summarize_set_pieces(team_df)
    discipline = summarize_discipline(team_df)
    halftime = summarize_halftime(team_df)
    base = team_tournament.copy()
    profile = (
        base.merge(set_piece[["team", "avg_set_piece_pressure", "corners_per_match", "corner_diff", "corner_2h_share"]], on="team", how="left")
        .merge(discipline[["team", "avg_instability", "yellow_for_per_match", "yellow_2h_share"]], on="team", how="left")
        .merge(halftime[["team", "avg_second_half_net_surge", "xg_2h_minus_1h", "post75_goals"]], on="team", how="left")
    )
    profile["betterdata_attack_index"] = (
        zscore(profile["db_xg_diff"])
        + 0.55 * zscore(profile["db_sot_rate"])
        + 0.45 * zscore(profile["avg_set_piece_pressure"])
        + 0.35 * zscore(profile["avg_second_half_net_surge"])
    )
    profile["betterdata_risk_index"] = (
        zscore(-profile["goal_diff"])
        + 0.50 * zscore(profile["avg_instability"])
        + 0.40 * zscore(-profile["db_corner_diff"])
        + 0.35 * zscore(profile["db_yellow_cards_against_ft"])
    )
    profile["market_expectation_gap"] = profile["fotmob_xg_diff"] - (profile["avg_db_team_win_prob"] - 0.5) * profile["matches"] * 3.0
    profile["style_cluster"] = np.select(
        [
            (profile["betterdata_attack_index"] > 1.0) & (profile["avg_second_half_net_surge"] > 0),
            (profile["avg_set_piece_pressure"] > 0.8) & (profile["corner_diff"] > 0),
            (profile["betterdata_risk_index"] > 1.0),
            (profile["avg_instability"] > 0.9),
            (profile["avg_second_half_net_surge"] < -0.9),
        ],
        [
            "front-foot pressure",
            "set-piece pressure",
            "fragile profile",
            "unstable/physical",
            "second-half fade",
        ],
        default="balanced",
    )
    return profile.sort_values("betterdata_attack_index", ascending=False)


def plot_match_upgrade(match_df: pd.DataFrame) -> None:
    fig, ax = setup(
        "Upgraded match-level feature map",
        "Total xG, shots, corners, cards, and market expectation now live on one match row",
        figsize=(11, 7.2),
    )
    sc = ax.scatter(
        match_df["db_total_xg_ft"],
        match_df["db_total_corners_ft"],
        s=70 + match_df["db_total_sot_ft"] * 18,
        c=match_df["db_implied_over25_prob"],
        cmap="viridis",
        edgecolor="white",
        linewidth=1.1,
        alpha=0.9,
        zorder=3,
    )
    ax.set_xlabel("Database total xG", color=COLORS["muted"])
    ax.set_ylabel("Total corners", color=COLORS["muted"])
    label_rows = pd.concat([match_df.nlargest(5, "db_total_xg_ft"), match_df.nlargest(5, "db_total_corners_ft")]).drop_duplicates("match_id")
    for _, r in label_rows.iterrows():
        ax.text(r["db_total_xg_ft"] + 0.04, r["db_total_corners_ft"] + 0.12, r["match"], fontsize=8.2, color=COLORS["ink"])
    cbar = fig.colorbar(sc, ax=ax, shrink=0.78, pad=0.02)
    cbar.set_label("Market implied over 2.5", color=COLORS["muted"])
    save(fig, "01_upgraded_match_feature_map.png")


def plot_live_mutation(summary: pd.DataFrame) -> None:
    state_order = ["trailing by 2+", "trailing by 1", "drawing", "leading by 1", "leading by 2+"]
    tier_order = ["low pressure", "mid pressure", "high pressure"]
    data = summary.set_index(["fotmob_live_score_state_75", "pressure_tier_75"])
    fig, ax = setup(
        "Live mutation pressure diagnostic",
        "Post-75 scoring rate by score state and 2H pressure tier; descriptive because 2H stats are not minute-stamped",
        figsize=(10.5, 7.2),
    )
    for i, state in enumerate(state_order):
        for j, tier in enumerate(tier_order):
            if (state, tier) not in data.index:
                continue
            row = data.loc[(state, tier)]
            p = row["p_score_after_75"]
            n = row["team_states"]
            ax.scatter(j, i, s=90 + 42 * n, c=[p], cmap="YlGnBu", vmin=0, vmax=0.9, edgecolor="white", linewidth=1.2)
            ax.text(j, i, f"{p:.0%}\nn={int(n)}", ha="center", va="center", fontsize=8.5, color=COLORS["ink"])
    ax.set_xticks(range(len(tier_order)), tier_order)
    ax.set_yticks(range(len(state_order)), state_order)
    ax.set_xlabel("Pressure tier", color=COLORS["muted"])
    ax.set_ylabel("Score state at 75'", color=COLORS["muted"])
    sm = plt.cm.ScalarMappable(cmap="YlGnBu", norm=plt.Normalize(0, 0.9))
    cbar = fig.colorbar(sm, ax=ax, shrink=0.78, pad=0.02)
    cbar.set_label("P(score after 75')", color=COLORS["muted"])
    save(fig, "02_live_mutation_pressure_diagnostic.png")


def plot_set_piece(set_piece: pd.DataFrame) -> None:
    data = set_piece.head(16).sort_values("avg_set_piece_pressure")
    fig, ax = setup(
        "Set-piece pressure model",
        "Corners, corner differential, 2H corner volume, and corners per xG blended into one index",
        figsize=(11, 7.5),
    )
    y = np.arange(len(data))
    ax.barh(y, data["avg_set_piece_pressure"], color=COLORS["blue"], alpha=0.88, zorder=3)
    ax.set_yticks(y, data["team"])
    ax.set_xlabel("Average set-piece pressure index", color=COLORS["muted"])
    for i, (_, r) in enumerate(data.iterrows()):
        ax.text(r["avg_set_piece_pressure"] + 0.04, i, f"{r['corners_per_match']:.1f} corners/m", va="center", fontsize=8.8, color=COLORS["muted"])
    save(fig, "03_set_piece_pressure_model.png")


def plot_discipline(discipline: pd.DataFrame) -> None:
    data = discipline.sort_values("avg_instability", ascending=False).head(18)
    fig, ax = setup(
        "Discipline / instability model",
        "Yellow-card burden plus total match card and corner activity",
        figsize=(11, 7.2),
    )
    sc = ax.scatter(
        data["yellow_for_per_match"],
        data["total_match_yellows_avg"],
        s=90 + data["late_goals_for"] * 22,
        c=data["avg_instability"],
        cmap="YlOrRd",
        edgecolor="white",
        linewidth=1.1,
        alpha=0.9,
        zorder=3,
    )
    ax.set_xlabel("Team yellow cards per match", color=COLORS["muted"])
    ax.set_ylabel("Total yellow cards in team's matches", color=COLORS["muted"])
    for _, r in data.head(10).iterrows():
        ax.text(r["yellow_for_per_match"] + 0.025, r["total_match_yellows_avg"] + 0.025, r["team"], fontsize=8.5, color=COLORS["ink"])
    cbar = fig.colorbar(sc, ax=ax, shrink=0.78, pad=0.02)
    cbar.set_label("Instability index", color=COLORS["muted"])
    save(fig, "04_discipline_instability_model.png")


def plot_market(market: pd.DataFrame) -> None:
    fig, ax = setup(
        "Market calibration: totals",
        "Positive residual means actual xG ran hotter than the market total proxy",
        figsize=(11, 7.2),
    )
    sc = ax.scatter(
        market["db_implied_over25_prob"],
        market["db_total_xg_ft"],
        s=80 + market["db_implied_btts_yes_prob"] * 130,
        c=market["total_xg_market_residual"],
        cmap="RdYlGn",
        vmin=-2.0,
        vmax=2.0,
        edgecolor="white",
        linewidth=1.1,
        alpha=0.9,
        zorder=3,
    )
    ax.set_xlabel("Market implied over 2.5 probability", color=COLORS["muted"])
    ax.set_ylabel("Actual total xG", color=COLORS["muted"])
    ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    for _, r in pd.concat([market.head(5), market.tail(5)]).drop_duplicates("match_id").iterrows():
        ax.text(r["db_implied_over25_prob"] + 0.004, r["db_total_xg_ft"] + 0.04, r["match"], fontsize=8.0, color=COLORS["ink"])
    cbar = fig.colorbar(sc, ax=ax, shrink=0.78, pad=0.02)
    cbar.set_label("xG minus market proxy", color=COLORS["muted"])
    save(fig, "05_market_calibration_totals.png")


def plot_halftime(halftime: pd.DataFrame) -> None:
    data = pd.concat([halftime.head(8), halftime.tail(8)]).drop_duplicates("team").sort_values("avg_second_half_net_surge")
    fig, ax = setup(
        "Half-time adjustment logic",
        "Teams with the biggest second-half attack surge or fade",
        figsize=(11, 8),
    )
    y = np.arange(len(data))
    colors = np.where(data["avg_second_half_net_surge"] >= 0, COLORS["green"], COLORS["red"])
    ax.barh(y, data["avg_second_half_net_surge"], color=colors, alpha=0.9, zorder=3)
    ax.axvline(0, color=COLORS["grid"], linewidth=1.2)
    ax.set_yticks(y, data["team"])
    ax.set_xlabel("Average second-half net surge index", color=COLORS["muted"])
    for i, (_, r) in enumerate(data.iterrows()):
        ax.text(r["avg_second_half_net_surge"] + (0.05 if r["avg_second_half_net_surge"] >= 0 else -0.05), i, f"xG swing {r['xg_2h_minus_1h']:+.2f}", ha="left" if r["avg_second_half_net_surge"] >= 0 else "right", va="center", fontsize=8.5, color=COLORS["muted"])
    save(fig, "06_halftime_adjustment_surge_fade.png")


def plot_profiles(profile: pd.DataFrame) -> None:
    fig, ax = setup(
        "Better-data team profiles",
        "Attack index blends xG, SOT rate, set pieces, and 2H surge; risk index blends goals, cards, and pressure conceded",
        figsize=(11, 7.5),
    )
    palette = {
        "front-foot pressure": COLORS["green"],
        "set-piece pressure": COLORS["blue"],
        "fragile profile": COLORS["red"],
        "unstable/physical": COLORS["yellow"],
        "second-half fade": COLORS["purple"],
        "balanced": COLORS["gray"],
    }
    for cluster, grp in profile.groupby("style_cluster"):
        ax.scatter(
            grp["betterdata_attack_index"],
            grp["betterdata_risk_index"],
            s=90 + grp["matches"] * 20,
            color=palette.get(cluster, COLORS["gray"]),
            edgecolor="white",
            linewidth=1.1,
            alpha=0.9,
            label=cluster,
            zorder=3,
        )
    labels = pd.concat(
        [
            profile.nlargest(6, "betterdata_attack_index"),
            profile.nlargest(5, "betterdata_risk_index"),
            profile.nsmallest(5, "betterdata_attack_index"),
        ]
    ).drop_duplicates("team")
    for _, r in labels.iterrows():
        ax.text(r["betterdata_attack_index"] + 0.04, r["betterdata_risk_index"] + 0.04, r["team"], fontsize=8.5, color=COLORS["ink"])
    ax.axhline(0, color=COLORS["grid"], linewidth=1.1)
    ax.axvline(0, color=COLORS["grid"], linewidth=1.1)
    ax.set_xlabel("Better-data attack index", color=COLORS["muted"])
    ax.set_ylabel("Better-data risk index", color=COLORS["muted"])
    ax.legend(frameon=False, fontsize=8.5, loc="best")
    save(fig, "07_better_team_profiles.png")


def canon_team(name: object) -> str:
    return v35.canon(name)


def blend_result_probabilities(base: dict[str, float], adjusted: dict[str, float], weight: float) -> dict[str, float]:
    w = float(np.clip(weight, 0.0, 1.0))
    labels = ("team_a_win", "draw", "team_b_win")
    blended = {label: (1.0 - w) * float(base.get(label, 0.0)) + w * float(adjusted.get(label, 0.0)) for label in labels}
    total = sum(blended.values())
    if total > 0:
        blended = {k: v / total for k, v in blended.items()}
    return blended


def blend_score_matrices(
    base: dict[tuple[int, int], float],
    adjusted: dict[tuple[int, int], float],
    weight: float,
) -> dict[tuple[int, int], float]:
    w = float(np.clip(weight, 0.0, 1.0))
    keys = set(base) | set(adjusted)
    matrix = {key: (1.0 - w) * float(base.get(key, 0.0)) + w * float(adjusted.get(key, 0.0)) for key in keys}
    total = sum(matrix.values())
    if total > 0:
        matrix = {key: value / total for key, value in matrix.items()}
    return matrix


def score_matrix_expected_goals(matrix: dict[tuple[int, int], float]) -> tuple[float, float]:
    return v28.expected_goals(matrix)


class BetterDataPriors:
    """Pre-match-safe priors built from v39_withbetterdata team profiles."""

    def __init__(self, profile_csv: str | Path = OUTDIR / "v39_withbetterdata_team_profiles.csv"):
        self.profile_csv = Path(profile_csv)
        self.rows: dict[str, dict[str, float | str]] = {}
        self.loaded = False
        self._load()

    def _load(self) -> None:
        if not self.profile_csv.exists():
            return
        frame = pd.read_csv(self.profile_csv)
        for row in frame.to_dict(orient="records"):
            team = canon_team(row.get("team"))
            if not team:
                continue
            self.rows[team] = row
        self.loaded = True

    def profile_for_team(self, team: object) -> dict[str, float | str]:
        return self.rows.get(canon_team(team), {})

    def diagnostics(self) -> dict[str, Any]:
        return {
            "profile_csv": str(self.profile_csv),
            "loaded": bool(self.loaded),
            "team_count": len(self.rows),
        }

    @staticmethod
    def _num(row: dict[str, Any], key: str, default: float = 0.0) -> float:
        value = row.get(key, default)
        try:
            if pd.isna(value):
                return float(default)
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def lambda_log_adjustments(
        self,
        team_a: object,
        team_b: object,
        max_abs_log_adjustment: float = 0.12,
    ) -> tuple[float, float, dict[str, Any]]:
        a = self.profile_for_team(team_a)
        b = self.profile_for_team(team_b)
        if not a or not b:
            return 0.0, 0.0, {
                "available": False,
                "team_a_profile_found": bool(a),
                "team_b_profile_found": bool(b),
            }

        matches_a = self._num(a, "matches", 0.0)
        matches_b = self._num(b, "matches", 0.0)
        shrink = min(1.0, max(0.0, min(matches_a, matches_b) / 4.0))

        def side_raw(own: dict[str, Any], opp: dict[str, Any]) -> float:
            own_attack = self._num(own, "betterdata_attack_index")
            opp_risk = self._num(opp, "betterdata_risk_index")
            own_set_piece = self._num(own, "avg_set_piece_pressure")
            opp_instability = self._num(opp, "avg_instability")
            own_surge = self._num(own, "avg_second_half_net_surge")
            own_market = self._num(own, "avg_db_team_win_prob", 0.5)
            opp_market = self._num(opp, "avg_db_team_win_prob", 0.5)
            return (
                0.030 * own_attack
                + 0.018 * opp_risk
                + 0.015 * own_set_piece
                + 0.012 * opp_instability
                + 0.010 * own_surge
                + 0.025 * (own_market - opp_market)
            )

        raw_a = side_raw(a, b) * shrink
        raw_b = side_raw(b, a) * shrink
        log_a = float(np.clip(raw_a, -max_abs_log_adjustment, max_abs_log_adjustment))
        log_b = float(np.clip(raw_b, -max_abs_log_adjustment, max_abs_log_adjustment))
        return log_a, log_b, {
            "available": True,
            "team_a_profile_found": True,
            "team_b_profile_found": True,
            "team_a_profile": {
                "team": a.get("team"),
                "matches": matches_a,
                "betterdata_attack_index": self._num(a, "betterdata_attack_index"),
                "betterdata_risk_index": self._num(a, "betterdata_risk_index"),
                "avg_set_piece_pressure": self._num(a, "avg_set_piece_pressure"),
                "avg_instability": self._num(a, "avg_instability"),
                "avg_second_half_net_surge": self._num(a, "avg_second_half_net_surge"),
                "avg_db_team_win_prob": self._num(a, "avg_db_team_win_prob", 0.5),
                "style_cluster": a.get("style_cluster", ""),
            },
            "team_b_profile": {
                "team": b.get("team"),
                "matches": matches_b,
                "betterdata_attack_index": self._num(b, "betterdata_attack_index"),
                "betterdata_risk_index": self._num(b, "betterdata_risk_index"),
                "avg_set_piece_pressure": self._num(b, "avg_set_piece_pressure"),
                "avg_instability": self._num(b, "avg_instability"),
                "avg_second_half_net_surge": self._num(b, "avg_second_half_net_surge"),
                "avg_db_team_win_prob": self._num(b, "avg_db_team_win_prob", 0.5),
                "style_cluster": b.get("style_cluster", ""),
            },
            "support_shrinkage": float(shrink),
            "raw_log_adjustment_a": float(raw_a),
            "raw_log_adjustment_b": float(raw_b),
            "log_adjustment_a": log_a,
            "log_adjustment_b": log_b,
            "max_abs_log_adjustment": float(max_abs_log_adjustment),
        }

    def outlier_lift(
        self,
        source: tuple[int, int],
        final: tuple[int, int],
        team_a: object,
        team_b: object,
    ) -> tuple[float, dict[str, Any]]:
        a = self.profile_for_team(team_a)
        b = self.profile_for_team(team_b)
        if not a or not b:
            return 0.0, {"betterdata_lift": 0.0, "betterdata_profiles_available": False}

        add_a = max(final[0] - source[0], 0)
        add_b = max(final[1] - source[1], 0)
        if add_a == 0 and add_b == 0:
            return 0.0, {"betterdata_lift": 0.0, "betterdata_profiles_available": True}

        def side(add: int, own: dict[str, Any], opp: dict[str, Any]) -> float:
            if add <= 0:
                return 0.0
            raw = (
                0.10 * np.clip(self._num(own, "avg_set_piece_pressure") / 3.0, -0.5, 1.2)
                + 0.08 * np.clip(self._num(opp, "avg_instability") / 3.0, -0.5, 1.2)
                + 0.07 * np.clip(self._num(own, "avg_second_half_net_surge") / 3.0, -0.7, 1.2)
                + 0.05 * np.clip(self._num(own, "betterdata_attack_index") / 3.0, -0.7, 1.3)
                + 0.05 * np.clip(self._num(own, "avg_match_over25_prob", 0.5) - 0.5, -0.25, 0.35)
            )
            return float(raw * min(1.4, 0.85 + 0.25 * add))

        lift_a = side(add_a, a, b)
        lift_b = side(add_b, b, a)
        both_score = 0.06 * np.clip(
            (self._num(a, "avg_instability") + self._num(b, "avg_instability")) / 5.0,
            -0.5,
            1.0,
        ) if add_a > 0 and add_b > 0 else 0.0
        total_bonus = 0.04 if sum(final) >= 5 else 0.0
        lift = float(np.clip(lift_a + lift_b + both_score + total_bonus, -0.20, 0.55))
        return lift, {
            "betterdata_lift": lift,
            "betterdata_profiles_available": True,
            "betterdata_lift_a": float(lift_a),
            "betterdata_lift_b": float(lift_b),
            "betterdata_both_score_bonus": float(both_score),
            "betterdata_high_total_bonus": float(total_bonus),
        }


def select_betterdata_coverage_outlier(
    score_matrix: dict[tuple[int, int], float],
    top_scorelines: list[dict[str, Any]],
    lambda_a: float,
    lambda_b: float,
    existing_outlier: dict[str, Any] | None,
    coverage_margin: float,
    observed_priors: Any,
    use_observed_game_state_priors: bool,
    better_priors: BetterDataPriors,
    team_a: object,
    team_b: object,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    top_three = [v39._score_key(item) for item in top_scorelines[:3]]
    lambda_sum = float(lambda_a) + float(lambda_b)
    diagnostics: dict[str, Any] = {
        "selector": "v39_betterdata_coverage_outlier",
        "probability_matrix_changed": False,
        "top_three_changed": False,
        "lambda_sum": lambda_sum,
        "coverage_margin": float(coverage_margin),
        "base_top_3": [v39._score_label(key) for key in top_three],
        "betterdata_priors": better_priors.diagnostics(),
    }
    if len(top_three) < 3:
        diagnostics["skip_reason"] = "not_enough_top_scorelines"
        return existing_outlier, diagnostics

    top3_max_total = max(sum(key) for key in top_three)
    min_candidate_total = top3_max_total + 1
    diagnostics.update(
        {
            "top3_max_total": int(top3_max_total),
            "min_candidate_total": int(min_candidate_total),
            "coverage_triggered": bool(lambda_sum > top3_max_total + float(coverage_margin)),
        }
    )
    if lambda_sum <= top3_max_total + float(coverage_margin):
        diagnostics["skip_reason"] = "lambda_sum_not_above_top3_ceiling"
        return existing_outlier, diagnostics

    top_set = set(top_three)
    candidates = [key for key in score_matrix if key not in top_set and sum(key) >= min_candidate_total]
    diagnostics["candidate_count"] = int(len(candidates))
    if not candidates:
        diagnostics["skip_reason"] = "no_higher_total_candidate"
        return existing_outlier, diagnostics

    details = {}
    for key in candidates:
        best_source_detail: dict[str, Any] | None = None
        for source in top_three:
            if sum(key) <= sum(source):
                continue
            observed = v39._observed_candidate_utility(
                key,
                score_matrix,
                top_three,
                observed_priors if use_observed_game_state_priors else None,
                team_a=team_a,
                team_b=team_b,
            )
            better_lift, better_diag = better_priors.outlier_lift(source, key, team_a, team_b)
            raw_probability = float(score_matrix.get(key, 0.0))
            observed_lift = float(observed.get("observed_lift", 0.0))
            utility = raw_probability * (1.0 + observed_lift + better_lift)
            candidate = {
                **observed,
                **better_diag,
                "scoreline": v39._score_label(key),
                "raw_probability": raw_probability,
                "observed_lift": observed_lift,
                "combined_lift": float(observed_lift + better_lift),
                "betterdata_utility": float(utility),
                "betterdata_source_scoreline": v39._score_label(source),
            }
            if best_source_detail is None or candidate["betterdata_utility"] > best_source_detail["betterdata_utility"]:
                best_source_detail = candidate
        if best_source_detail is not None:
            details[key] = best_source_detail

    if not details:
        diagnostics["skip_reason"] = "no_candidate_with_source"
        return existing_outlier, diagnostics

    best_key = max(details, key=lambda key: details[key]["betterdata_utility"])
    best = details[best_key]
    top_details = sorted(
        details.values(),
        key=lambda row: (float(row["betterdata_utility"]), float(row["raw_probability"])),
        reverse=True,
    )[:10]
    selected = {
        "source": "v39_withbetterdata",
        "coverage_replaced_existing_outlier": bool(existing_outlier),
        "candidate_scoreline": v39._score_label(best_key),
        "candidate_total": int(sum(best_key)),
        "candidate_probability": float(score_matrix.get(best_key, 0.0)),
        "candidate_observed_utility": float(best.get("observed_utility", 0.0)),
        "candidate_betterdata_utility": float(best["betterdata_utility"]),
        "candidate_observed_lift": float(best.get("observed_lift", 0.0)),
        "candidate_betterdata_lift": float(best.get("betterdata_lift", 0.0)),
        "candidate_combined_lift": float(best.get("combined_lift", 0.0)),
        "candidate_observed_source_scoreline": best.get("observed_source_scoreline", ""),
        "candidate_betterdata_source_scoreline": best.get("betterdata_source_scoreline", ""),
    }
    diagnostics.update(
        {
            **selected,
            "outlier_selected": True,
            "outlier_scoreline": v39._score_label(best_key),
            "outlier_probability": float(score_matrix.get(best_key, 0.0)),
            "top_betterdata_outlier_candidates": top_details,
        }
    )
    return v39._score_item(best_key, float(score_matrix.get(best_key, 0.0)), selected), diagnostics


class V39BetterDataModel:
    """Wrap V39 with small pre-match better-data lambda and outlier adjustments."""

    def __init__(
        self,
        base_model: v39.V39CoverageOutlierModel,
        better_priors: BetterDataPriors | None = None,
        betterdata_scoreline_blend: float = DEFAULT_BETTERDATA_SCORELINE_BLEND,
        betterdata_wdl_blend: float = DEFAULT_BETTERDATA_WDL_BLEND,
        max_abs_log_adjustment: float = DEFAULT_BETTERDATA_MAX_LOG_ADJUSTMENT,
    ):
        self.base_model = base_model
        self.better_priors = better_priors or BetterDataPriors()
        self.betterdata_scoreline_blend = float(np.clip(betterdata_scoreline_blend, 0.0, 1.0))
        self.betterdata_wdl_blend = float(np.clip(betterdata_wdl_blend, 0.0, 1.0))
        self.max_abs_log_adjustment = float(max(max_abs_log_adjustment, 0.0))
        self.training_data_summary = {
            **getattr(base_model, "training_data_summary", {}),
            "v39_withbetterdata": {
                "betterdata_priors": self.better_priors.diagnostics(),
                "betterdata_scoreline_blend": self.betterdata_scoreline_blend,
                "betterdata_wdl_blend": self.betterdata_wdl_blend,
                "max_abs_log_adjustment": self.max_abs_log_adjustment,
            },
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_model, name)

    def predict(self, *args, **kwargs) -> dict[str, Any]:
        team_a = kwargs.get("team_a", args[0] if args else "")
        team_b = kwargs.get("team_b", args[1] if len(args) > 1 else "")
        max_goals = int(kwargs.get("max_goals", args[5] if len(args) > 5 else 10))
        base_prediction = self.base_model.predict(*args, **kwargs)
        base_matrix = v39.score_matrix_from_prediction(base_prediction)
        base_lambda_a = float(base_prediction["lambda_a"])
        base_lambda_b = float(base_prediction["lambda_b"])

        log_a, log_b, prior_diag = self.better_priors.lambda_log_adjustments(
            team_a,
            team_b,
            max_abs_log_adjustment=self.max_abs_log_adjustment,
        )
        adjusted_lambda_a = float(np.clip(base_lambda_a * math.exp(log_a), 0.05, 7.5))
        adjusted_lambda_b = float(np.clip(base_lambda_b * math.exp(log_b), 0.05, 7.5))
        adjusted_matrix = v11.poisson_score_matrix(adjusted_lambda_a, adjusted_lambda_b, max_goals)
        rho = base_prediction.get("calibration_notes", {}).get("dixon_coles_rho", -0.08)
        adjusted_matrix = v11.apply_dixon_coles_adjustment(
            adjusted_matrix,
            adjusted_lambda_a,
            adjusted_lambda_b,
            rho=rho,
        )
        adjusted_result_probabilities = v11.result_probs(adjusted_matrix)
        result_probabilities = blend_result_probabilities(
            dict(base_prediction["result_probabilities"]),
            adjusted_result_probabilities,
            self.betterdata_wdl_blend,
        )
        score_matrix = blend_score_matrices(
            base_matrix,
            adjusted_matrix,
            self.betterdata_scoreline_blend,
        )
        score_matrix = v11.reweight_score_matrix_to_results(score_matrix, result_probabilities)
        lambda_a, lambda_b = score_matrix_expected_goals(score_matrix)

        prediction = dict(base_prediction)
        prediction["v39_base_prediction_before_betterdata"] = {
            "lambda_a": base_lambda_a,
            "lambda_b": base_lambda_b,
            "result_probabilities": dict(base_prediction["result_probabilities"]),
            "top_scorelines": base_prediction.get("top_scorelines", [])[:5],
            "coverage_total_outlier": base_prediction.get("coverage_total_outlier"),
        }
        prediction["lambda_a"] = lambda_a
        prediction["lambda_b"] = lambda_b
        prediction["result_probabilities"] = result_probabilities
        prediction["predicted_result"] = max(result_probabilities, key=result_probabilities.get)
        prediction.update(v15.score_outputs(score_matrix, max_goals))

        outlier, outlier_diag = select_betterdata_coverage_outlier(
            score_matrix,
            prediction.get("top_scorelines", []),
            lambda_a,
            lambda_b,
            existing_outlier=base_prediction.get("coverage_total_outlier"),
            coverage_margin=getattr(self.base_model, "coverage_margin", v39.DEFAULT_COVERAGE_MARGIN),
            observed_priors=getattr(self.base_model, "observed_game_state_priors", None),
            use_observed_game_state_priors=getattr(self.base_model, "use_observed_game_state_priors", True),
            better_priors=self.better_priors,
            team_a=team_a,
            team_b=team_b,
        )
        prediction["coverage_total_outlier"] = outlier
        prediction["game_state_late_outlier"] = outlier
        prediction["late_instability_outlier"] = outlier
        prediction["outlier_scoreline"] = outlier
        prediction["top_scorelines_plus_outlier"] = [
            *prediction.get("top_scorelines", [])[:3],
            *([outlier] if outlier else []),
        ]
        better_diag = {
            "base_model": "v39_coverage_outlier",
            "scoreline_policy": "v39_plus_betterdata_pre_match_lambda_and_outlier_overlay",
            "score_matrix_changed": True,
            "scoreline_layer_affects_wdl": bool(self.betterdata_wdl_blend > 0),
            "betterdata_scoreline_blend": self.betterdata_scoreline_blend,
            "betterdata_wdl_blend": self.betterdata_wdl_blend,
            "max_abs_log_adjustment": self.max_abs_log_adjustment,
            "base_lambda_a": base_lambda_a,
            "base_lambda_b": base_lambda_b,
            "adjusted_candidate_lambda_a": adjusted_lambda_a,
            "adjusted_candidate_lambda_b": adjusted_lambda_b,
            "final_lambda_a": float(lambda_a),
            "final_lambda_b": float(lambda_b),
            "betterdata_priors": prior_diag,
            "adjusted_result_probabilities": adjusted_result_probabilities,
            "outlier_selector": outlier_diag,
        }
        prediction["v39_withbetterdata_adjustments"] = better_diag
        prediction["calibration_notes"] = {
            **prediction.get("calibration_notes", {}),
            "v39_withbetterdata": better_diag,
        }
        return prediction


def build_betterdata_model_from_zip(
    zip_path: str | Path,
    betterdata_profile_csv: str | Path = OUTDIR / "v39_withbetterdata_team_profiles.csv",
    betterdata_scoreline_blend: float = DEFAULT_BETTERDATA_SCORELINE_BLEND,
    betterdata_wdl_blend: float = DEFAULT_BETTERDATA_WDL_BLEND,
    betterdata_max_log_adjustment: float = DEFAULT_BETTERDATA_MAX_LOG_ADJUSTMENT,
    **kwargs,
) -> tuple[V39BetterDataModel, Any]:
    base_model, data = v39.build_from_zip(zip_path, **kwargs)
    model = V39BetterDataModel(
        base_model,
        better_priors=BetterDataPriors(betterdata_profile_csv),
        betterdata_scoreline_blend=betterdata_scoreline_blend,
        betterdata_wdl_blend=betterdata_wdl_blend,
        max_abs_log_adjustment=betterdata_max_log_adjustment,
    )
    return model, data


def write_readme(summary: dict[str, object]) -> None:
    lines = [
        "# v39 with better data",
        "",
        "This is the v39 analysis layer that uses the combined FotMob + Database feature tables.",
        "",
        "## Outputs",
        "",
        "- `v39_withbetterdata_team_match_model_frame.csv`: one row per team-match with derived pressure, discipline, market, and halftime features.",
        "- `v39_withbetterdata_match_feature_upgrade.csv`: one row per match with upgraded match-level features.",
        "- `v39_withbetterdata_live_mutation_pressure.csv`: score-state-by-pressure post-75 scoring diagnostics.",
        "- `v39_withbetterdata_set_piece_pressure.csv`: team set-piece pressure profile.",
        "- `v39_withbetterdata_discipline_instability.csv`: team instability/card profile.",
        "- `v39_withbetterdata_market_calibration.csv`: market-vs-xG calibration frame.",
        "- `v39_withbetterdata_halftime_adjustment.csv`: second-half surge/fade profile.",
        "- `v39_withbetterdata_team_profiles.csv`: rebuilt team profiles with better-data clusters.",
        "",
        "## Plots",
        "",
        "Plots live under `plots/` and are numbered in the same order as the analysis blocks.",
        "",
        "## Caveat",
        "",
        "The 1H fields are live-safe at half-time. The 2H fields are descriptive/backtest features unless the source later provides minute-stamped corners/xG/cards; using full 2H values for a 75-minute decision would leak information from minutes 76-90.",
        "",
        "## Current run",
        "",
        f"- Matches: {summary['matches']}",
        f"- Team-match rows: {summary['team_match_rows']}",
        f"- Teams: {summary['teams']}",
        f"- Combined source matched rows: {summary['matched_database_rows']} / {summary['database_rows']}",
    ]
    (OUTDIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis() -> dict[str, object]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PLOTDIR.mkdir(parents=True, exist_ok=True)

    team_match = read_combined("team_match_features.csv")
    match_features = read_combined("match_features.csv")
    team_tournament = read_combined("team_tournament_features.csv")
    diagnostics = json.loads((COMBINED / "combined_feature_diagnostics.json").read_text(encoding="utf-8"))

    model_frame = add_betterdata_features(team_match)
    match_upgrade = build_match_feature_upgrade(match_features)
    live_mutation = summarize_live_mutation(model_frame)
    set_piece = summarize_set_pieces(model_frame)
    discipline = summarize_discipline(model_frame)
    market = summarize_market(match_upgrade)
    halftime = summarize_halftime(model_frame)
    profiles = build_better_team_profiles(model_frame, team_tournament)

    model_frame.to_csv(OUTDIR / "v39_withbetterdata_team_match_model_frame.csv", index=False)
    match_upgrade.to_csv(OUTDIR / "v39_withbetterdata_match_feature_upgrade.csv", index=False)
    live_mutation.to_csv(OUTDIR / "v39_withbetterdata_live_mutation_pressure.csv", index=False)
    set_piece.to_csv(OUTDIR / "v39_withbetterdata_set_piece_pressure.csv", index=False)
    discipline.to_csv(OUTDIR / "v39_withbetterdata_discipline_instability.csv", index=False)
    market.to_csv(OUTDIR / "v39_withbetterdata_market_calibration.csv", index=False)
    halftime.to_csv(OUTDIR / "v39_withbetterdata_halftime_adjustment.csv", index=False)
    profiles.to_csv(OUTDIR / "v39_withbetterdata_team_profiles.csv", index=False)

    plot_match_upgrade(match_upgrade)
    plot_live_mutation(live_mutation)
    plot_set_piece(set_piece)
    plot_discipline(discipline)
    plot_market(market)
    plot_halftime(halftime)
    plot_profiles(profiles)

    summary = {
        "matches": int(match_features["match_id"].nunique()),
        "team_match_rows": int(len(model_frame)),
        "teams": int(model_frame["team"].nunique()),
        "database_rows": diagnostics.get("database_rows"),
        "matched_database_rows": diagnostics.get("matched_database_rows"),
        "output_dir": str(OUTDIR.relative_to(ROOT)),
    }
    (OUTDIR / "v39_withbetterdata_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_readme(summary)
    return summary


def write_prediction_outputs(prediction: dict[str, Any], model: V39BetterDataModel, output_dir: Path, no_plots: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
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
        output_dir / "scoreline_probabilities_top_plus_betterdata_outlier.csv",
        index=False,
    )
    (output_dir / "model_summary.json").write_text(
        json.dumps(
            {
                "version": "v39-withbetterdata",
                "base_model": "v39-coverage-outlier",
                "team_a": prediction["team_a"],
                "team_b": prediction["team_b"],
                "result_probabilities": prediction["result_probabilities"],
                "top_3": prediction["top_scorelines"][:3],
                "coverage_total_outlier": prediction["coverage_total_outlier"],
                "v39_withbetterdata_adjustments": prediction["v39_withbetterdata_adjustments"],
                "v39_adjustments": prediction.get("v39_adjustments", {}),
                "v38_adjustments": prediction.get("v38_adjustments", {}),
                "expanded_training_data": model.training_data_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not no_plots:
        v11.plot_prediction_outputs(prediction, output_dir)
        v28.plot_top3_scorelines(prediction, output_dir / "plots")
        v35.plot_top3_plus_game_state_outlier(prediction, output_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    data_dir = ROOT / "data"
    parser = argparse.ArgumentParser(
        description="Run v39 with better FotMob + Database data, or refresh the analysis tables."
    )
    parser.add_argument("--team-a")
    parser.add_argument("--team-b")
    parser.add_argument("--host-a", action="store_true")
    parser.add_argument("--host-b", action="store_true")
    parser.add_argument("--knockout", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--refresh-analysis", action="store_true")
    parser.add_argument("--outdir", default="outputs/outputs_v39_withbetterdata")
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
    parser.add_argument("--coverage-margin", type=float, default=v39.DEFAULT_COVERAGE_MARGIN)
    parser.add_argument("--game-state-analysis-dir", default=str(v39.DEFAULT_GAME_STATE_ANALYSIS_DIR))
    parser.add_argument("--disable-observed-game-state-priors", action="store_true")
    parser.add_argument("--betterdata-profile-csv", default=str(OUTDIR / "v39_withbetterdata_team_profiles.csv"))
    parser.add_argument("--betterdata-scoreline-blend", type=float, default=DEFAULT_BETTERDATA_SCORELINE_BLEND)
    parser.add_argument("--betterdata-wdl-blend", type=float, default=DEFAULT_BETTERDATA_WDL_BLEND)
    parser.add_argument("--betterdata-max-log-adjustment", type=float, default=DEFAULT_BETTERDATA_MAX_LOG_ADJUSTMENT)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def run_prediction(args: argparse.Namespace) -> dict[str, Any]:
    profile_path = Path(args.betterdata_profile_csv)
    if args.refresh_analysis or not profile_path.exists():
        run_analysis()

    model, _ = build_betterdata_model_from_zip(
        args.worldcupsai_zip,
        betterdata_profile_csv=args.betterdata_profile_csv,
        betterdata_scoreline_blend=args.betterdata_scoreline_blend,
        betterdata_wdl_blend=args.betterdata_wdl_blend,
        betterdata_max_log_adjustment=args.betterdata_max_log_adjustment,
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
    prediction = model.predict(
        args.team_a,
        args.team_b,
        host_a=args.host_a,
        host_b=args.host_b,
        knockout=args.knockout,
    )
    write_prediction_outputs(prediction, model, output_dir, no_plots=args.no_plots)
    return {
        "result_probabilities": prediction["result_probabilities"],
        "predicted_result": prediction["predicted_result"],
        "lambda_a": prediction["lambda_a"],
        "lambda_b": prediction["lambda_b"],
        "top_3": prediction["top_scorelines"][:3],
        "coverage_total_outlier": prediction["coverage_total_outlier"],
        "v39_withbetterdata_adjustments": prediction["v39_withbetterdata_adjustments"],
        "output_dir": str(output_dir),
    }


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.analysis_only or (not args.team_a and not args.team_b):
        summary = run_analysis()
        print(json.dumps(summary, indent=2))
        return
    if not args.team_a or not args.team_b:
        parser.error("--team-a and --team-b must be provided together for prediction mode")
    result = run_prediction(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
'''
v39_withbetterdata = _load_submodule("v39_withbetterdata", _V39_WITHBETTERDATA_SOURCE, "market_edge.py:v39_withbetterdata")

# ======================================================================
# v42_fotmob_market_edge_model.py  (bundled as an isolated sub-module)
# ======================================================================
_V42_FOTMOB_MARKET_EDGE_MODEL_SOURCE = r'''
#!/usr/bin/env python3
"""V42: FotMob-aware exact-score fair-price detector.

Full feature run:
    MPLCONFIGDIR=.matplotlib_cache .venv/bin/python v42_fotmob_market_edge_model.py --team-a "Argentina" --team-b "France" --knockout --outdir outputs/outputs_v42_argentina_france --fotmob-leaders data/fotmob_full_stat_tables_clean.csv --fotmob-player-stats data/fotmob_match_player_stats_clean.csv --fotmob-lineups data/fotmob_match_lineups_clean.csv --fotmob-substitutions data/fotmob_match_substitutions_clean.csv --fotmob-keeper-stats data/fotmob_match_keeper_stats_clean.csv --fotmob-match-facts data/fotmob_match_facts_clean.csv --fotmob-goal-events data/fotmob_match_goal_events_clean.csv --fotmob-wdl-blend 0.20 --fotmob-scoreline-blend 0.15 --fotmob-max-log-adjustment 0.12 --betterdata-profile-csv analysis/v39_withbetterdata_latest/v39_withbetterdata_team_profiles.csv --betterdata-scoreline-blend 0.00 --betterdata-wdl-blend 0.25 --betterdata-max-log-adjustment 0.12 --auto-polymarket --fetch-clob-orderbook --min-edge 0.01 --min-ev 0.08 --uncertainty-buffer 0.01 --price-history-root outputs --plot-set decision

Offline/no Polymarket fetch:
    MPLCONFIGDIR=.matplotlib_cache .venv/bin/python v42_fotmob_market_edge_model.py --team-a "Argentina" --team-b "France" --outdir outputs/outputs_v42_argentina_france_offline --no-fetch-polymarket

This keeps V41's market-pricing machinery, but fixes the base-model stack:
V36's FotMob/player/xG lambda adjustment is used first, then V39's one-slot
coverage outlier is applied on top. Polymarket is only a quote/diagnostic
layer: it never changes the model fair probabilities.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import ssl
import subprocess
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
VERSIONS_DIR = PROJECT_DIR / "versions"
if str(VERSIONS_DIR) not in sys.path:
    sys.path.insert(0, str(VERSIONS_DIR))

import v11_wcq_results_model as v11
import v15_catboost_model as v15
import v28_current_worldcup_form_model as v28
import v35_game_state_late_mutation_model as v35
import v36_fotmob_current_form_model as v36
import v39_coverage_outlier_model as v39
import v39_withbetterdata as v39bd

try:
    import certifi
except Exception:  # pragma: no cover - optional local certificate helper
    certifi = None


DATA_DIR = PROJECT_DIR / "data"
GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
GAMMA_EVENT_SLUG_URL = "https://gamma-api.polymarket.com/events/slug/{slug}"
CLOB_BOOK_URL = "https://clob.polymarket.com/book"
DEFAULT_POLYMARKET_SPORTS_GAMES_URL = "https://polymarket.com/sports/world-cup/games"
DEFAULT_WORLDCUP_TAG_ID = "102232"
DEFAULT_GAMMA_LIMIT = 500
DEFAULT_TAG_PAGE_LIMIT = 100
DEFAULT_TAG_MAX_PAGES = 15
DEFAULT_RELATED_EVENT_SUFFIXES = (
    "more-markets",
    "halftime-result",
    "player-props",
)
DEFAULT_MIN_EXACT_EDGE = 0.01
DEFAULT_MIN_EV = 0.08
DEFAULT_UNCERTAINTY_BUFFER = 0.01
DEFAULT_COVERAGE_MARGIN = 0.0
ScoreKey = Tuple[int, int]
ScoreMatrix = Dict[ScoreKey, float]


class V42FotmobCoverageModel:
    """Wrap V36 and apply V39's total-envelope outlier selector."""

    def __init__(self, base_model: Any, coverage_margin: float = DEFAULT_COVERAGE_MARGIN):
        self.base_model = base_model
        self.coverage_margin = float(max(0.0, coverage_margin))
        self.training_data_summary = getattr(base_model, "training_data_summary", {})

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_model, name)

    def predict(self, *args, **kwargs) -> dict[str, Any]:
        team_a = kwargs.get("team_a", args[0] if args else "")
        team_b = kwargs.get("team_b", args[1] if len(args) > 1 else "")
        prediction = self.base_model.predict(*args, **kwargs)
        score_matrix = v39.score_matrix_from_prediction(prediction)
        existing_outlier = prediction.get("game_state_late_outlier")
        coverage_outlier, diagnostics = v39.select_coverage_outlier(
            score_matrix,
            prediction.get("top_scorelines", []),
            float(prediction["lambda_a"]),
            float(prediction["lambda_b"]),
            existing_outlier=existing_outlier,
            margin=self.coverage_margin,
            team_a=team_a,
            team_b=team_b,
        )
        prediction["v42_original_game_state_late_outlier"] = existing_outlier
        prediction["coverage_total_outlier"] = coverage_outlier
        prediction["game_state_late_outlier"] = coverage_outlier
        prediction["late_instability_outlier"] = coverage_outlier
        prediction["outlier_scoreline"] = coverage_outlier
        prediction["top_scorelines_plus_outlier"] = [
            *prediction.get("top_scorelines", [])[:3],
            *([coverage_outlier] if coverage_outlier else []),
        ]
        prediction["v42_adjustments"] = {
            "base_model": "v36_fotmob_current_form",
            "scoreline_policy": "v36_fotmob_lambda_base_plus_v39_observed_game_state_coverage_outlier",
            "scoreline_layer_affects_wdl": False,
            **diagnostics,
        }
        prediction["calibration_notes"] = {
            **prediction.get("calibration_notes", {}),
            "v42": prediction["v42_adjustments"],
        }
        return prediction

TEAM_ABBREVIATIONS = {
    "england": "eng",
    "ghana": "gha",
    "france": "fra",
    "iraq": "irq",
    "belgium": "bel",
    "iran": "irn",
    "austria": "aut",
    "argentina": "arg",
    "canada": "can",
    "qatar": "qat",
    "germany": "ger",
    "ivory coast": "civ",
    "cote divoire": "civ",
    "côte d ivoire": "civ",
    "netherlands": ("ned", "nld"),
    "sweden": "swe",
    "uzbekistan": "uzb",
    "colombia": "col",
    "czechia": "cze",
    "south africa": "rsa",
    "portugal": ("por", "prt"),
    "dr congo": "cod",
    "croatia": ("cro", "hrv"),
    "panama": "pan",
    "norway": "nor",
    "mexico": "mex",
    "south korea": "kor",
    "usa": "usa",
    "united states": "usa",
    "australia": "aus",
    "morocco": "mar",
    "scotland": "sco",
    "haiti": "hai",
    "brazil": "bra",
    "turkiye": "tur",
    "turkey": "tur",
    "paraguay": "par",
    "japan": "jpn",
    "tunisia": "tun",
    "senegal": "sen",
    "algeria": "alg",
    "jordan": "jor",
    "new zealand": "nzl",
    "switzerland": "che",
    "bosnia and herzegovina": "bih",
    "bosnia": "bih",
}


def _json_loads_maybe(value: Any, fallback: Any = None) -> Any:
    if value is None:
        return fallback
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return fallback
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            quoted_items = re.findall(r'"([^"]*)"', text)
            if quoted_items:
                return quoted_items
            if text.startswith("[") and text.endswith("]"):
                numeric_items = re.findall(r"-?\d+(?:\.\d+)?", text)
                if numeric_items:
                    return numeric_items
            return fallback
    return fallback


def _to_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _norm_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = value.lower()
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _team_tokens(team: str) -> list[str]:
    aliases = {
        "usa": ["usa", "united states", "us"],
        "united states": ["united states", "usa", "us"],
        "dr congo": ["dr congo", "democratic republic of congo", "drc", "congo dr"],
        "cote divoire": ["cote divoire", "cote d ivoire", "ivory coast", "côte d ivoire"],
        "cote d ivoire": ["cote d ivoire", "cote divoire", "ivory coast", "côte d ivoire"],
        "ivory coast": ["ivory coast", "cote divoire", "cote d ivoire", "côte d ivoire"],
        "czechia": ["czechia", "czech republic"],
        "turkiye": ["turkiye", "turkey", "türkiye"],
        "curacao": ["curacao", "curaçao"],
        "iran": ["iran", "ir iran"],
        "cape verde": ["cape verde", "cabo verde"],
        "cabo verde": ["cabo verde", "cape verde"],
        "south korea": ["south korea", "korea republic"],
        "bosnia": ["bosnia", "bosnia herzegovina", "bosnia and herzegovina"],
        "bosnia herzegovina": ["bosnia herzegovina", "bosnia and herzegovina", "bosnia"],
        "bosnia and herzegovina": ["bosnia and herzegovina", "bosnia herzegovina", "bosnia"],
        "switzerland": ["switzerland", "swiss"],
    }
    key = _norm_text(team)
    tokens = [_norm_text(item) for item in aliases.get(key, [team])]
    for abbreviation in _team_abbreviations(key):
        tokens.append(_norm_text(abbreviation))
    return list(dict.fromkeys(token for token in tokens if token))


def _team_abbreviations(team_key: str) -> list[str]:
    value = TEAM_ABBREVIATIONS.get(team_key)
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value if item]


def _contains_team(text: str, team: str) -> bool:
    norm = _norm_text(text)
    return any(token in norm for token in _team_tokens(team))


def _question(market: dict[str, Any]) -> str:
    return " ".join(
        str(market.get(key, "") or "")
        for key in ("question", "title", "slug", "groupItemTitle", "description")
    )


def fetch_gamma_markets(query: str, limit: int = DEFAULT_GAMMA_LIMIT) -> list[dict[str, Any]]:
    params = {
        "limit": str(limit),
        "active": "true",
        "closed": "false",
        "search": query,
    }
    url = f"{GAMMA_MARKETS_URL}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": "world-cup-v42-edge-model/1.0"})
    context = ssl.create_default_context(cafile=certifi.where()) if certifi else None
    with urllib.request.urlopen(request, timeout=20, context=context) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if isinstance(payload, dict):
        for key in ("data", "markets", "results"):
            if isinstance(payload.get(key), list):
                return payload[key]
        return []
    if isinstance(payload, list):
        return payload
    return []


def fetch_gamma_markets_with_params(params: dict[str, Any]) -> list[dict[str, Any]]:
    clean_params = {
        key: str(value).lower() if isinstance(value, bool) else str(value)
        for key, value in params.items()
        if value is not None
    }
    url = f"{GAMMA_MARKETS_URL}?{urllib.parse.urlencode(clean_params)}"
    request = urllib.request.Request(url, headers={"User-Agent": "world-cup-v42-edge-model/1.0"})
    context = ssl.create_default_context(cafile=certifi.where()) if certifi else None
    with urllib.request.urlopen(request, timeout=25, context=context) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if isinstance(payload, dict):
        for key in ("data", "markets", "results"):
            if isinstance(payload.get(key), list):
                return payload[key]
        return []
    if isinstance(payload, list):
        return payload
    return []


def slug_from_event_input(value: str) -> str:
    text = value.strip()
    if not text:
        return text
    if text.startswith("http://") or text.startswith("https://"):
        parsed = urllib.parse.urlparse(text)
        parts = [part for part in parsed.path.split("/") if part]
        if not parts:
            return text
        # Polymarket event URLs usually look like /event/{slug}; if a plain
        # page URL is passed, the last path component is still the slug.
        return parts[-1]
    return text


def fetch_gamma_event_by_slug(slug_or_url: str) -> dict[str, Any]:
    slug = slug_from_event_input(slug_or_url)
    url = GAMMA_EVENT_SLUG_URL.format(slug=urllib.parse.quote(slug, safe=""))
    request = urllib.request.Request(url, headers={"User-Agent": "world-cup-v42-edge-model/1.0"})
    context = ssl.create_default_context(cafile=certifi.where()) if certifi else None
    with urllib.request.urlopen(request, timeout=25, context=context) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload if isinstance(payload, dict) else {}


def fetch_text_url(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    context = ssl.create_default_context(cafile=certifi.where()) if certifi else None
    with urllib.request.urlopen(request, timeout=25, context=context) as response:
        return response.read().decode("utf-8", "ignore")


def sports_slug_from_url(url: str, team_a: str, team_b: str) -> tuple[str | None, dict[str, Any]]:
    parsed = urllib.parse.urlparse(url.strip())
    parts = [part for part in parsed.path.split("/") if part]
    meta: dict[str, Any] = {"sports_url": url, "path_parts": parts}
    if "sports" not in parts:
        return None, {**meta, "reason": "not_a_sports_url"}

    if parts and parts[-1] != "games":
        slug = parts[-1]
        return slug, {**meta, "source": "specific_game_url", "slug": slug}

    html = fetch_text_url(url)
    a_key = _norm_text(team_a)
    b_key = _norm_text(team_b)
    a_abbrs = _team_abbreviations(a_key)
    b_abbrs = _team_abbreviations(b_key)
    candidates = sorted(
        set(re.findall(r"/sports/world-cup/([a-z0-9-]+)", html, flags=re.IGNORECASE))
    )
    related_candidates = [
        slug
        for slug in candidates
        if (
            any(re.search(rf"(?:^|-){re.escape(abbr)}(?:-|$)", slug.lower()) for abbr in a_abbrs)
            and any(re.search(rf"(?:^|-){re.escape(abbr)}(?:-|$)", slug.lower()) for abbr in b_abbrs)
        )
        or (
            _norm_text(team_a).replace(" ", "-") in slug.lower()
            and _norm_text(team_b).replace(" ", "-") in slug.lower()
        )
    ]
    meta.update(
        {
            "source": "sports_games_page",
            "candidate_count": len(candidates),
            "team_a_abbreviation": a_abbrs[0] if a_abbrs else None,
            "team_b_abbreviation": b_abbrs[0] if b_abbrs else None,
            "team_a_abbreviations": a_abbrs,
            "team_b_abbreviations": b_abbrs,
            "related_slug_candidates": related_candidates[:80],
        }
    )
    scored = []
    for slug in candidates:
        norm_slug = slug.lower()
        score = 0
        if any(re.search(rf"(?:^|-){re.escape(abbr)}(?:-|$)", norm_slug) for abbr in a_abbrs):
            score += 2
        if any(re.search(rf"(?:^|-){re.escape(abbr)}(?:-|$)", norm_slug) for abbr in b_abbrs):
            score += 2
        if _norm_text(team_a).replace(" ", "-") in norm_slug:
            score += 1
        if _norm_text(team_b).replace(" ", "-") in norm_slug:
            score += 1
        if score:
            scored.append({"slug": slug, "score": score})
    scored.sort(key=lambda row: (row["score"], row["slug"]), reverse=True)
    meta["matched_candidates"] = scored[:10]
    if scored and scored[0]["score"] >= 4:
        return scored[0]["slug"], {**meta, "slug": scored[0]["slug"]}
    return None, {**meta, "reason": "no_confident_team_slug_match"}


def related_event_slug_candidates(base_slug: str, page_candidates: list[str] | None = None) -> list[str]:
    base = slug_from_event_input(base_slug)
    candidates = [base]
    for suffix in DEFAULT_RELATED_EVENT_SUFFIXES:
        candidates.append(f"{base}-{suffix}")
    for item in page_candidates or []:
        if item not in candidates:
            candidates.append(item)
    return candidates


def event_markets_for_related_slugs(
    base_slug: str,
    *,
    team_a: str,
    team_b: str,
    page_candidates: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    markets_by_id: dict[str, dict[str, Any]] = {}
    fetches: list[dict[str, Any]] = []
    for slug in related_event_slug_candidates(base_slug, page_candidates):
        try:
            event = fetch_gamma_event_by_slug(slug)
        except Exception as exc:
            fetches.append({"slug": slug, "error": str(exc), "market_count": 0})
            continue
        event_markets, tag_fetches = event_markets_with_tag_props(
            event,
            team_a=team_a,
            team_b=team_b,
        )
        for market in event_markets:
            key = str(market.get("id") or market.get("conditionId") or market.get("slug") or len(markets_by_id))
            markets_by_id[key] = market
        fetches.append(
            {
                "slug": slug,
                "event_title": event.get("title"),
                "event_id": event.get("id"),
                "market_count": len(event_markets),
                "tag_fetches": tag_fetches,
            }
        )
        time.sleep(0.05)
    return list(markets_by_id.values()), fetches


def event_markets_with_tag_props(
    event: dict[str, Any],
    *,
    team_a: str,
    team_b: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    markets_by_id: dict[str, dict[str, Any]] = {}
    for market in list(event.get("markets") or []):
        key = str(market.get("id") or market.get("conditionId") or market.get("slug") or len(markets_by_id))
        markets_by_id[key] = market

    event_tags = list(event.get("tags") or [])
    priority_tags = [
        tag
        for tag in event_tags
        if str(tag.get("slug", "")).lower() in {"fifa-world-cup", "world-cup"}
        or "world cup" in str(tag.get("label", "")).lower()
    ]
    if not priority_tags:
        priority_tags = [
            tag
            for tag in event_tags
            if str(tag.get("slug", "")).lower() in {"games", "soccer", "sports"}
        ][:1]
    if not priority_tags:
        priority_tags = [
            {
                "id": DEFAULT_WORLDCUP_TAG_ID,
                "slug": "fifa-world-cup",
                "label": "FIFA World Cup",
            }
        ]

    tag_fetches = []
    for tag in priority_tags[:2]:
        tag_id = str(tag.get("id") or "")
        if not tag_id:
            continue
        tag_markets, tag_meta = fetch_tag_markets_for_match(
            tag_id=tag_id,
            team_a=team_a,
            team_b=team_b,
        )
        tag_meta["tag_slug"] = tag.get("slug")
        tag_meta["tag_label"] = tag.get("label")
        tag_fetches.append(tag_meta)
        for market in tag_markets:
            key = str(market.get("id") or market.get("conditionId") or market.get("slug") or len(markets_by_id))
            markets_by_id[key] = market
    return list(markets_by_id.values()), tag_fetches


def load_or_fetch_markets(
    *,
    team_a: str,
    team_b: str,
    query: str | None,
    event_slug: str | None,
    sports_url: str | None,
    json_path: str | None,
    no_fetch: bool,
    limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if json_path:
        path = Path(json_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        markets = payload if isinstance(payload, list) else payload.get("markets", [])
        return list(markets), {"source": "json", "path": str(path)}
    if event_slug:
        try:
            markets, related_fetches = event_markets_for_related_slugs(
                event_slug,
                team_a=team_a,
                team_b=team_b,
            )
            main_fetch = next((row for row in related_fetches if row.get("slug") == slug_from_event_input(event_slug)), {})
            return markets, {
                "source": "gamma_event_slug",
                "event_slug": slug_from_event_input(event_slug),
                "event_title": main_fetch.get("event_title"),
                "event_id": main_fetch.get("event_id"),
                "related_event_fetches": related_fetches,
                "raw_count": len(markets),
            }
        except Exception as exc:
            return [], {
                "source": "gamma_event_slug",
                "event_slug": slug_from_event_input(event_slug),
                "errors": [{"error": str(exc)}],
                "raw_count": 0,
            }
    if sports_url:
        try:
            slug, slug_meta = sports_slug_from_url(sports_url, team_a, team_b)
            if not slug:
                return [], {
                    "source": "polymarket_sports_url",
                    **slug_meta,
                    "raw_count": 0,
                }
            markets, related_fetches = event_markets_for_related_slugs(
                slug,
                team_a=team_a,
                team_b=team_b,
                page_candidates=slug_meta.get("related_slug_candidates") or [],
            )
            main_fetch = next((row for row in related_fetches if row.get("slug") == slug), {})
            return markets, {
                "source": "polymarket_sports_url",
                **slug_meta,
                "event_title": main_fetch.get("event_title"),
                "event_id": main_fetch.get("event_id"),
                "related_event_fetches": related_fetches,
                "raw_count": len(markets),
            }
        except Exception as exc:
            return [], {
                "source": "polymarket_sports_url",
                "sports_url": sports_url,
                "errors": [{"error": str(exc)}],
                "raw_count": 0,
            }
    if no_fetch:
        return [], {"source": "disabled", "reason": "--no-fetch-polymarket"}

    queries = []
    base = query or f"{team_a} {team_b}"
    queries.append(base)
    queries.append(f"{team_b} {team_a}")
    queries.append(f"{team_a} vs {team_b}")
    queries.append(f"{team_b} vs {team_a}")
    queries.append(f"{team_a} {team_b} world cup")
    queries.append(f"{team_b} {team_a} world cup")
    queries.append(f"{team_a} {team_b} soccer")
    queries.append(f"{team_b} {team_a} soccer")

    all_markets: dict[str, dict[str, Any]] = {}
    errors = []
    for item in dict.fromkeys(queries):
        try:
            for market in fetch_gamma_markets(item, limit=limit):
                key = str(market.get("id") or market.get("conditionId") or market.get("slug") or len(all_markets))
                all_markets[key] = market
            time.sleep(0.15)
        except Exception as exc:  # network/API failures are surfaced in summary.
            errors.append({"query": item, "error": str(exc)})
    return list(all_markets.values()), {
        "source": "gamma_api",
        "queries": queries,
        "errors": errors,
        "raw_count": len(all_markets),
    }


def filter_match_markets(markets: Iterable[dict[str, Any]], team_a: str, team_b: str) -> list[dict[str, Any]]:
    filtered = []
    for market in markets:
        text = _question(market)
        if _contains_team(text, team_a) and _contains_team(text, team_b):
            filtered.append(market)
    return filtered


def fetch_tag_markets_for_match(
    *,
    tag_id: str,
    team_a: str,
    team_b: str,
    max_pages: int = DEFAULT_TAG_MAX_PAGES,
    page_limit: int = DEFAULT_TAG_PAGE_LIMIT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    matched: dict[str, dict[str, Any]] = {}
    pages = []
    for page in range(max_pages):
        offset = page * page_limit
        markets = fetch_gamma_markets_with_params(
            {
                "tag_id": tag_id,
                "limit": page_limit,
                "offset": offset,
                "active": True,
                "closed": False,
            }
        )
        page_matches = filter_match_markets(markets, team_a, team_b)
        for market in page_matches:
            key = str(market.get("id") or market.get("conditionId") or market.get("slug") or len(matched))
            matched[key] = market
        pages.append({"offset": offset, "count": len(markets), "matched": len(page_matches)})
        if not markets:
            break
        time.sleep(0.05)
    return list(matched.values()), {
        "tag_id": tag_id,
        "pages": pages,
        "matched_count": len(matched),
    }


def binary_prices(market: dict[str, Any]) -> dict[str, Any]:
    outcomes = _json_loads_maybe(market.get("outcomes"), [])
    prices = _json_loads_maybe(market.get("outcomePrices"), [])
    token_ids = _json_loads_maybe(market.get("clobTokenIds"), [])
    if not isinstance(prices, list):
        prices = re.findall(r"-?\d+(?:\.\d+)?", str(market.get("outcomePrices") or ""))
    if not isinstance(outcomes, list):
        outcomes = []
    if not outcomes and isinstance(prices, list) and len(prices) == 2:
        sports_type = str(market.get("sportsMarketType") or "").lower()
        text = _norm_text(_question(market))
        outcomes = ["Over", "Under"] if sports_type == "totals" or "o u" in text else ["Yes", "No"]
    if not isinstance(token_ids, list):
        token_ids = []
    out: dict[str, Any] = {
        "outcomes": outcomes or [],
        "prices": prices or [],
        "token_ids": token_ids or [],
        "yes_price": None,
        "no_price": None,
        "devig_yes": None,
    }
    if not isinstance(outcomes, list) or not isinstance(prices, list):
        return out
    pairs = []
    for idx, outcome in enumerate(outcomes):
        price = _to_float(prices[idx] if idx < len(prices) else None)
        token_id = token_ids[idx] if isinstance(token_ids, list) and idx < len(token_ids) else None
        label = str(outcome).strip().strip('"').strip("'").lower()
        pairs.append((label, price, token_id))
    for label, price, token_id in pairs:
        if label in {"yes", "y", "over", "home", "team a"}:
            out["yes_price"] = price
            out["yes_token_id"] = token_id
        elif label in {"no", "n", "under", "away", "team b"}:
            out["no_price"] = price
            out["no_token_id"] = token_id
    if out["yes_price"] is None and len(pairs) >= 1:
        out["yes_price"] = pairs[0][1]
        out["yes_token_id"] = pairs[0][2]
    if out["no_price"] is None and len(pairs) >= 2:
        out["no_price"] = pairs[1][1]
        out["no_token_id"] = pairs[1][2]
    yes = _to_float(out["yes_price"])
    no = _to_float(out["no_price"])
    if yes is not None and no is not None and yes + no > 0:
        out["devig_yes"] = yes / (yes + no)
    elif yes is not None:
        out["devig_yes"] = yes
    return out


def classify_moneyline_market(market: dict[str, Any], team_a: str, team_b: str) -> dict[str, Any] | None:
    text = " ".join(
        str(market.get(key, "") or "")
        for key in ("question", "title", "groupItemTitle")
    )
    norm = _norm_text(text)
    sports_type = str(market.get("sportsMarketType") or "").lower()
    if sports_type and sports_type != "moneyline":
        return None
    if not sports_type and "win" not in norm and "draw" not in norm:
        return None

    result = None
    if "draw" in norm:
        result = "draw"
    elif _contains_team(text, team_a) and "win" in norm:
        result = "team_a_win"
    elif _contains_team(text, team_b) and "win" in norm:
        result = "team_b_win"
    if result is None:
        return None

    prices = binary_prices(market)
    devig_yes = _to_float(prices.get("devig_yes"))
    if devig_yes is None:
        return None
    return {
        "market_id": market.get("id"),
        "slug": market.get("slug"),
        "question": market.get("question") or market.get("title"),
        "result": result,
        "market_probability": max(0.0, min(1.0, float(devig_yes))),
        "yes_price": prices.get("yes_price"),
        "no_price": prices.get("no_price"),
        "volume": _to_float(market.get("volumeNum"), _to_float(market.get("volume"), 0.0)),
        "liquidity": _to_float(market.get("liquidityNum"), _to_float(market.get("liquidity"), 0.0)),
    }


def classify_btts_market(market: dict[str, Any]) -> dict[str, Any] | None:
    text = _question(market)
    norm = _norm_text(text)
    sports_type = str(market.get("sportsMarketType") or "").lower()
    if sports_type and sports_type != "both_teams_to_score":
        return None
    if sports_type != "both_teams_to_score" and "both teams to score" not in norm:
        return None

    prices = binary_prices(market)
    devig_yes = _to_float(prices.get("devig_yes"))
    if devig_yes is None:
        return None
    p_yes = max(0.0, min(1.0, float(devig_yes)))
    return {
        "market_id": market.get("id"),
        "slug": market.get("slug"),
        "question": market.get("question") or market.get("title"),
        "p_btts_yes": p_yes,
        "p_btts_no": 1.0 - p_yes,
        "yes_price": prices.get("yes_price"),
        "no_price": prices.get("no_price"),
        "volume": _to_float(market.get("volumeNum"), _to_float(market.get("volume"), 0.0)),
        "liquidity": _to_float(market.get("liquidityNum"), _to_float(market.get("liquidity"), 0.0)),
    }


def classify_spread_market(market: dict[str, Any], team_a: str, team_b: str) -> dict[str, Any] | None:
    text = " ".join(
        str(market.get(key, "") or "")
        for key in ("question", "title", "groupItemTitle")
    )
    norm = _norm_text(text)
    sports_type = str(market.get("sportsMarketType") or "").lower()
    if sports_type != "spreads" and "spread" not in norm:
        return None
    line = _to_float(market.get("line"))
    if line is None:
        match = re.search(r"\((-?\d+(?:\.\d+)?)\)", text)
        if match:
            line = float(match.group(1))
    if line is None:
        return None

    team = None
    if _contains_team(text, team_a):
        team = "team_a"
    elif _contains_team(text, team_b):
        team = "team_b"
    if team is None:
        return None

    prices = binary_prices(market)
    devig_yes = _to_float(prices.get("devig_yes"))
    if devig_yes is None:
        return None
    return {
        "market_id": market.get("id"),
        "slug": market.get("slug"),
        "question": market.get("question") or market.get("title"),
        "spread_team": team,
        "line": float(line),
        "p_cover": max(0.0, min(1.0, float(devig_yes))),
        "p_not_cover": max(0.0, min(1.0, 1.0 - float(devig_yes))),
        "yes_price": prices.get("yes_price"),
        "no_price": prices.get("no_price"),
        "volume": _to_float(market.get("volumeNum"), _to_float(market.get("volume"), 0.0)),
        "liquidity": _to_float(market.get("liquidityNum"), _to_float(market.get("liquidity"), 0.0)),
    }


def classify_total_market(market: dict[str, Any]) -> dict[str, Any] | None:
    text = _question(market)
    norm = _norm_text(text)
    sports_type = str(market.get("sportsMarketType") or "").lower()
    if sports_type and sports_type != "totals":
        return None
    if (
        sports_type != "totals"
        and "goal" not in norm
        and "total" not in norm
        and "over" not in norm
        and "under" not in norm
        and "o u" not in norm
    ):
        return None
    match = re.search(r"\b(over|under)\s+(\d+(?:\.\d+)?)\b", norm)
    side_in_question = None
    threshold = None
    line_value = _to_float(market.get("line"))
    if sports_type == "totals" and line_value is not None:
        side_in_question = "over"
        threshold = float(line_value)
    elif match:
        side_in_question = match.group(1)
        threshold = float(match.group(2))
    else:
        match = re.search(r"\bo\s*u\s+(\d+(?:\.\d+)?)\b", norm)
        if match:
            threshold = float(match.group(1))
            side_in_question = "over"
        match = re.search(r"\b(\d+(?:\.\d+)?)\s+(?:goals?)\b", norm)
        if threshold is None and match and ("over" in norm or "under" in norm):
            threshold = float(match.group(1))
            side_in_question = "over" if "over" in norm else "under"
    if threshold is None or abs((threshold % 1.0) - 0.5) > 1e-6:
        return None

    prices = binary_prices(market)
    devig_yes = _to_float(prices.get("devig_yes"))
    if devig_yes is None:
        return None
    p_over = devig_yes if side_in_question != "under" else 1.0 - devig_yes
    return {
        "market_id": market.get("id"),
        "slug": market.get("slug"),
        "question": market.get("question") or market.get("title"),
        "threshold": float(threshold),
        "side_in_question": side_in_question,
        "p_over": max(0.0, min(1.0, float(p_over))),
        "yes_price": prices.get("yes_price"),
        "no_price": prices.get("no_price"),
        "devig_yes": devig_yes,
        "volume": _to_float(market.get("volumeNum"), _to_float(market.get("volume"), 0.0)),
        "liquidity": _to_float(market.get("liquidityNum"), _to_float(market.get("liquidity"), 0.0)),
    }


def classify_exact_score_market(market: dict[str, Any], team_a: str, team_b: str) -> dict[str, Any] | None:
    primary_text = " ".join(
        str(market.get(key, "") or "")
        for key in ("question", "title", "slug", "groupItemTitle")
    )
    text = _question(market)
    primary_norm = _norm_text(primary_text)
    norm = _norm_text(text)
    sports_type = str(market.get("sportsMarketType") or "").lower()
    is_exact_market = sports_type == "soccer_exact_score" or "exact" in norm or bool(re.search(r"\b\d+\s+\d+\b", norm))
    if not is_exact_market:
        return None
    prices = binary_prices(market)
    devig_yes = _to_float(prices.get("devig_yes"))
    if devig_yes is None:
        return None
    if "any other" in primary_norm:
        return {
            "market_id": market.get("id"),
            "slug": market.get("slug"),
            "question": market.get("question") or market.get("title"),
            "team_a_goals": None,
            "team_b_goals": None,
            "scoreline": "Any Other Score",
            "is_any_other_score": True,
            "yes_price": prices.get("yes_price"),
            "no_price": prices.get("no_price"),
            "yes_token_id": prices.get("yes_token_id"),
            "no_token_id": prices.get("no_token_id"),
            "market_probability": devig_yes,
            "volume": _to_float(market.get("volumeNum"), _to_float(market.get("volume"), 0.0)),
            "liquidity": _to_float(market.get("liquidityNum"), _to_float(market.get("liquidity"), 0.0)),
        }

    aliases_a = [re.escape(alias) for alias in _team_tokens(team_a)]
    aliases_b = [re.escape(alias) for alias in _team_tokens(team_b)]
    patterns = []
    for alias_a in aliases_a:
        for alias_b in aliases_b:
            patterns.extend(
                [
                    (rf"{alias_a}\s+(\d+)\s+(\d+)\s+{alias_b}", False),
                    (rf"{alias_a}\s+(\d+)\s*[-–]\s*(\d+)\s+{alias_b}", False),
                    (rf"{alias_b}\s+(\d+)\s+(\d+)\s+{alias_a}", True),
                    (rf"{alias_b}\s+(\d+)\s*[-–]\s*(\d+)\s+{alias_a}", True),
                ]
            )
    for pattern, reverse in patterns:
        match = re.search(pattern, norm)
        if not match:
            continue
        first, second = int(match.group(1)), int(match.group(2))
        a_goals, b_goals = (second, first) if reverse else (first, second)
        return {
            "market_id": market.get("id"),
            "slug": market.get("slug"),
            "question": market.get("question") or market.get("title"),
            "team_a_goals": a_goals,
            "team_b_goals": b_goals,
            "scoreline": f"{a_goals}-{b_goals}",
            "yes_price": prices.get("yes_price"),
            "no_price": prices.get("no_price"),
            "yes_token_id": prices.get("yes_token_id"),
            "no_token_id": prices.get("no_token_id"),
            "market_probability": devig_yes,
            "volume": _to_float(market.get("volumeNum"), _to_float(market.get("volume"), 0.0)),
            "liquidity": _to_float(market.get("liquidityNum"), _to_float(market.get("liquidity"), 0.0)),
        }
    return None


def monotone_over_ladder(total_markets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_threshold: dict[float, dict[str, Any]] = {}
    for row in total_markets:
        threshold = float(row["threshold"])
        prev = best_by_threshold.get(threshold)
        if prev is None or float(row.get("liquidity") or 0.0) > float(prev.get("liquidity") or 0.0):
            best_by_threshold[threshold] = row.copy()
    rows = [best_by_threshold[key] for key in sorted(best_by_threshold)]
    prev = 1.0
    for row in rows:
        p = max(0.0, min(prev, float(row["p_over"])))
        row["p_over_monotone"] = p
        prev = p
    return rows


def market_total_distribution(total_ladder: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not total_ladder:
        return []
    over = {float(row["threshold"]): float(row["p_over_monotone"]) for row in total_ladder}
    thresholds = sorted(over)
    distribution = []
    prev_over = 1.0
    for threshold in thresholds:
        goals = int(round(threshold - 0.5))
        mass = max(0.0, prev_over - over[threshold])
        distribution.append(
            {
                "total_bucket": str(goals),
                "bucket_min_total": goals,
                "bucket_max_total": goals,
                "market_probability": mass,
            }
        )
        prev_over = over[threshold]
    tail_min = int(round(thresholds[-1] + 0.5))
    distribution.append(
        {
            "total_bucket": f"{tail_min}+",
            "bucket_min_total": tail_min,
            "bucket_max_total": None,
            "market_probability": max(0.0, prev_over),
        }
    )
    total = sum(float(row["market_probability"]) for row in distribution)
    if total > 0:
        for row in distribution:
            row["market_probability"] = float(row["market_probability"]) / total
    return distribution


def score_matrix_from_prediction(prediction: dict[str, Any]) -> ScoreMatrix:
    return {
        (int(row["team_a_goals"]), int(row["team_b_goals"])): float(row["probability"])
        for row in prediction.get("scoreline_probabilities", [])
    }


def normalize_matrix(matrix: ScoreMatrix) -> ScoreMatrix:
    total = sum(max(0.0, value) for value in matrix.values())
    if total <= 0:
        return matrix.copy()
    return {key: max(0.0, value) / total for key, value in matrix.items()}


def tilt_matrix_by_beta(matrix: ScoreMatrix, beta: float) -> ScoreMatrix:
    tilted = {key: prob * math.exp(float(beta) * (key[0] + key[1])) for key, prob in matrix.items()}
    return normalize_matrix(tilted)


def aggregate_total_buckets(matrix: ScoreMatrix, market_distribution: list[dict[str, Any]]) -> dict[str, float]:
    bucket_probs = {}
    for bucket in market_distribution:
        label = str(bucket["total_bucket"])
        min_total = int(bucket["bucket_min_total"])
        max_total = bucket.get("bucket_max_total")
        if max_total is None or pd.isna(max_total):
            prob = sum(value for key, value in matrix.items() if sum(key) >= min_total)
        else:
            prob = sum(value for key, value in matrix.items() if min_total <= sum(key) <= int(max_total))
        bucket_probs[label] = float(prob)
    total = sum(bucket_probs.values())
    if total > 0:
        bucket_probs = {key: value / total for key, value in bucket_probs.items()}
    return bucket_probs


def force_total_distribution(matrix: ScoreMatrix, market_distribution: list[dict[str, Any]]) -> ScoreMatrix:
    if not market_distribution:
        return normalize_matrix(matrix)
    out = {key: 0.0 for key in matrix}
    for bucket in market_distribution:
        target = float(bucket["market_probability"])
        min_total = int(bucket["bucket_min_total"])
        max_total = bucket.get("bucket_max_total")
        if max_total is None or pd.isna(max_total):
            keys = [key for key in matrix if sum(key) >= min_total]
        else:
            keys = [key for key in matrix if min_total <= sum(key) <= int(max_total)]
        current = sum(float(matrix.get(key, 0.0)) for key in keys)
        if current <= 0:
            if keys:
                share = target / len(keys)
                for key in keys:
                    out[key] += share
            continue
        scale = target / current
        for key in keys:
            out[key] += float(matrix.get(key, 0.0)) * scale
    return normalize_matrix(out)


def kl_divergence(target: dict[str, float], candidate: dict[str, float]) -> float:
    eps = 1e-12
    loss = 0.0
    for label, p in target.items():
        q = candidate.get(label, 0.0)
        if p > 0:
            loss += p * math.log(max(eps, p) / max(eps, q))
    return float(loss)


def matrix_kl_divergence(reference: ScoreMatrix, candidate: ScoreMatrix) -> float:
    eps = 1e-12
    loss = 0.0
    for key, p in reference.items():
        q = candidate.get(key, 0.0)
        if p > 0:
            loss += float(p) * math.log(max(eps, float(p)) / max(eps, float(q)))
    return float(loss)


def fit_total_tilt_beta(
    base_matrix: ScoreMatrix,
    market_distribution_rows: list[dict[str, Any]],
    beta_min: float = -0.60,
    beta_max: float = 0.80,
) -> dict[str, Any]:
    target = {
        str(row["total_bucket"]): float(row["market_probability"])
        for row in market_distribution_rows
    }
    best = {"beta": 0.0, "kl": float("inf")}
    # Coarse-to-fine grid search: one scalar, deterministic, no scipy needed.
    ranges = [(beta_min, beta_max, 141), (None, None, 101), (None, None, 101)]
    center = 0.0
    width = beta_max - beta_min
    for idx, (lo, hi, count) in enumerate(ranges):
        if idx == 0:
            start, end = float(lo), float(hi)
        else:
            width *= 0.18
            start, end = center - width / 2.0, center + width / 2.0
        for step in range(count):
            beta = start + (end - start) * step / max(1, count - 1)
            tilted = tilt_matrix_by_beta(base_matrix, beta)
            candidate = aggregate_total_buckets(tilted, market_distribution_rows)
            loss = kl_divergence(target, candidate)
            if loss < best["kl"]:
                best = {"beta": float(beta), "kl": float(loss), "candidate": candidate}
                center = float(beta)
    return best


def scoreline_reliability(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("is_any_other_score") or row.get("team_a_goals") is None or row.get("team_b_goals") is None:
        return {
            "total_goals": None,
            "goal_margin": None,
            "selection_bucket": "any_other_bucket",
            "reliability_weight": 0.35,
        }
    a_goals = int(row["team_a_goals"])
    b_goals = int(row["team_b_goals"])
    total_goals = a_goals + b_goals
    margin = abs(a_goals - b_goals)
    if total_goals <= 3 and margin < 3:
        bucket = "validated_total_le_3"
        weight = 1.0
    elif total_goals <= 3:
        bucket = "low_total_lopsided"
        weight = 0.75
    elif total_goals == 4:
        bucket = "high_total_4_discounted"
        weight = 0.50
    else:
        bucket = "tail_total_5_plus_heavily_discounted"
        weight = 0.25
    return {
        "total_goals": total_goals,
        "goal_margin": margin,
        "selection_bucket": bucket,
        "reliability_weight": weight,
    }


def kelly_fraction_for_binary(fair: float, price: float) -> float | None:
    if price <= 0.0 or price >= 1.0:
        return None
    return max(0.0, (fair - price) / (1.0 - price))


def _apply_joint_kelly_for_probability(
    rows: list[dict[str, Any]],
    probability_key: str,
    verdict_key: str,
    output_prefix: str,
    active_verdicts: set[str],
) -> list[dict[str, Any]]:
    active_flag = f"{output_prefix}_active"
    joint_key = f"{output_prefix}_fraction"
    half_key = f"half_{output_prefix}_fraction"
    quarter_key = f"quarter_{output_prefix}_fraction"
    for row in rows:
        row[active_flag] = False
        row[joint_key] = 0.0
        row[half_key] = 0.0
        row[quarter_key] = 0.0

    active = [
        row
        for row in rows
        if row.get(verdict_key) in active_verdicts
        and (_to_float(row.get("raw_yes_price"), 0.0) or 0.0) > 0.0
        and (_to_float(row.get("raw_yes_price"), 1.0) or 1.0) < 1.0
        and (_to_float(row.get(probability_key), 0.0) or 0.0)
        > (_to_float(row.get("raw_yes_price"), 0.0) or 0.0)
    ]

    while active:
        s_prob = sum(float(row[probability_key]) for row in active)
        q_price = sum(float(row.get("raw_yes_price") or row.get("market_implied_probability") or 0.0) for row in active)
        if q_price >= 1.0:
            break
        ratio = (1.0 - s_prob) / (1.0 - q_price)
        stakes = []
        for row in active:
            price = float(row.get("raw_yes_price") or row.get("market_implied_probability") or 0.0)
            stake = float(row[probability_key]) - price * ratio
            stakes.append((row, stake))
        keep = [(row, stake) for row, stake in stakes if stake > 0.0]
        if len(keep) == len(active):
            for row, stake in keep:
                row[active_flag] = True
                row[joint_key] = stake
                row[half_key] = 0.5 * stake
                row[quarter_key] = 0.25 * stake
            break
        active = [row for row, _stake in keep]
    return rows


def apply_joint_kelly(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        price = _to_float(row.get("raw_yes_price"), _to_float(row.get("market_implied_probability"), 0.0)) or 0.0
        adjusted_edge = _to_float(row.get("risk_adjusted_edge"), 0.0) or 0.0
        row["risk_adjusted_probability"] = max(0.0, min(1.0, price + adjusted_edge))
    return _apply_joint_kelly_for_probability(
        rows,
        probability_key="risk_adjusted_probability",
        verdict_key="buy_verdict",
        output_prefix="joint_kelly",
        active_verdicts={"buy"},
    )


def edge_rows(
    base_matrix: ScoreMatrix,
    fair_matrix: ScoreMatrix,
    exact_markets: list[dict[str, Any]],
    min_edge: float,
    min_ev: float,
    uncertainty_buffer: float,
    market_implied_matrix: ScoreMatrix | None = None,
    market_matrix_meta: dict[str, Any] | None = None,
    market_total_reference_matrix: ScoreMatrix | None = None,
) -> list[dict[str, Any]]:
    rows = []
    model_rank_by_key = {
        key: rank
        for rank, (key, _prob) in enumerate(
            sorted(base_matrix.items(), key=lambda item: item[1], reverse=True),
            start=1,
        )
    }
    listed_score_keys = {
        (int(market["team_a_goals"]), int(market["team_b_goals"]))
        for market in exact_markets
        if not market.get("is_any_other_score")
        and market.get("team_a_goals") is not None
        and market.get("team_b_goals") is not None
    }
    for market in exact_markets:
        market_prob = float(market["market_probability"])
        raw_yes = _to_float(market.get("yes_price"), market_prob) or market_prob
        if market.get("is_any_other_score"):
            model_rank = None
            fair = sum(prob for key, prob in fair_matrix.items() if key not in listed_score_keys)
            base = sum(prob for key, prob in base_matrix.items() if key not in listed_score_keys)
            market_total_reference_prob = (
                sum(prob for key, prob in market_total_reference_matrix.items() if key not in listed_score_keys)
                if market_total_reference_matrix
                else None
            )
            market_matrix_prob = (
                sum(prob for key, prob in market_implied_matrix.items() if key not in listed_score_keys)
                if market_implied_matrix
                else None
            )
        else:
            key = (int(market["team_a_goals"]), int(market["team_b_goals"]))
            model_rank = model_rank_by_key.get(key)
            fair = float(fair_matrix.get(key, 0.0))
            base = float(base_matrix.get(key, 0.0))
            market_total_reference_prob = (
                float(market_total_reference_matrix.get(key, 0.0))
                if market_total_reference_matrix
                else None
            )
            market_matrix_prob = float(market_implied_matrix.get(key, 0.0)) if market_implied_matrix else None
        edge = fair - market_prob
        executable_edge = fair - raw_yes - float(uncertainty_buffer)
        ev = fair / raw_yes - 1.0 if raw_yes > 0 else None
        reliability = scoreline_reliability(market)
        kelly = kelly_fraction_for_binary(fair, raw_yes)
        reliability_weight = float(reliability["reliability_weight"])
        adjusted_edge = max(0.0, edge) * reliability_weight
        adjusted_kelly = None if kelly is None else kelly * reliability_weight
        buy_verdict = "skip"
        if edge > 0:
            if adjusted_edge >= 0.015 and executable_edge > 0:
                buy_verdict = "buy"
            elif adjusted_edge >= 0.0075:
                buy_verdict = "watch"
            elif reliability_weight < 0.75:
                buy_verdict = "discounted_tail_watch"
        row = {
            **market,
            **reliability,
            "model_scoreline_rank": model_rank,
            "base_model_probability": base,
            "model_only_fair_probability": fair,
            "scoreline_fair_probability": fair,
            # Legacy column name kept for old plotting/portfolio helpers. In V42
            # this is model-only fair, not a Polymarket-tilted probability.
            "market_total_tilt_probability": fair,
            "polymarket_total_tilt_reference_probability": market_total_reference_prob,
            "market_implied_score_probability": market_matrix_prob,
            "our_vs_market_matrix_edge": None if market_matrix_prob is None else fair - float(market_matrix_prob),
            "posted_vs_market_matrix_edge": None if market_matrix_prob is None else market_prob - float(market_matrix_prob),
            "market_matrix_confidence": None if not market_matrix_meta else market_matrix_meta.get("market_matrix_confidence"),
            "market_implied_probability": market_prob,
            "raw_yes_price": raw_yes,
            "edge_vs_devig_market": edge,
            "edge_vs_raw_yes_after_buffer": executable_edge,
            "expected_return_vs_raw_yes": ev,
            "kelly_fraction": kelly,
            "risk_adjusted_edge": adjusted_edge,
            "risk_adjusted_kelly_fraction": adjusted_kelly,
            "buy_verdict": buy_verdict,
            "passes_edge_filter": bool(edge >= min_edge and ev is not None and ev >= min_ev and executable_edge > 0),
        }
        rows.append(row)
    rows = apply_joint_kelly(rows)
    rows.sort(
        key=lambda row: (
            bool(row["passes_edge_filter"]),
            float(row.get("risk_adjusted_edge") or 0.0),
            float(row["edge_vs_devig_market"]),
            float(row.get("market_total_tilt_probability") or 0.0),
        ),
        reverse=True,
    )
    return rows


def build_book_context(
    moneyline_markets: list[dict[str, Any]],
    btts_markets: list[dict[str, Any]],
    spread_markets: list[dict[str, Any]],
    market_distribution_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    moneyline_probs: dict[str, float] = {}
    best_moneyline: dict[str, dict[str, Any]] = {}
    for row in moneyline_markets:
        result = str(row.get("result") or "")
        if not result:
            continue
        prev = best_moneyline.get(result)
        if prev is None or float(row.get("liquidity") or 0.0) > float(prev.get("liquidity") or 0.0):
            best_moneyline[result] = row
    raw_sum = sum(float(row.get("market_probability") or 0.0) for row in best_moneyline.values())
    if raw_sum > 0 and {"team_a_win", "draw", "team_b_win"}.issubset(best_moneyline):
        for result, row in best_moneyline.items():
            moneyline_probs[result] = float(row.get("market_probability") or 0.0) / raw_sum

    btts = None
    if btts_markets:
        btts = max(btts_markets, key=lambda row: float(row.get("liquidity") or 0.0))

    return {
        "moneyline_probs": moneyline_probs,
        "btts": btts,
        "spreads": spread_markets,
        "total_distribution": market_distribution_rows,
    }


def score_event_keys(matrix: ScoreMatrix, event_name: str) -> list[ScoreKey]:
    if event_name == "team_a_win":
        return [key for key in matrix if key[0] > key[1]]
    if event_name == "draw":
        return [key for key in matrix if key[0] == key[1]]
    if event_name == "team_b_win":
        return [key for key in matrix if key[0] < key[1]]
    if event_name == "btts_yes":
        return [key for key in matrix if key[0] > 0 and key[1] > 0]
    if event_name == "btts_no":
        return [key for key in matrix if key[0] == 0 or key[1] == 0]
    return []


def event_probability(matrix: ScoreMatrix, keys: list[ScoreKey]) -> float:
    return sum(float(matrix.get(key, 0.0)) for key in keys)


def apply_binary_event_constraint(
    matrix: ScoreMatrix,
    event_keys: list[ScoreKey],
    target: float,
    damping: float = 0.70,
) -> ScoreMatrix:
    if not event_keys:
        return matrix
    target = max(1e-6, min(1.0 - 1e-6, float(target)))
    event_set = set(event_keys)
    current = max(1e-9, min(1.0 - 1e-9, event_probability(matrix, event_keys)))
    event_scale = (target / current) ** damping
    other_scale = ((1.0 - target) / (1.0 - current)) ** damping
    adjusted = {
        key: float(value) * (event_scale if key in event_set else other_scale)
        for key, value in matrix.items()
    }
    return normalize_matrix(adjusted)


def spread_event_keys(matrix: ScoreMatrix, row: dict[str, Any]) -> list[ScoreKey]:
    team = str(row.get("spread_team") or "")
    line = _to_float(row.get("line"))
    if line is None:
        return []
    if team == "team_a":
        return [key for key in matrix if key[0] + float(line) > key[1]]
    if team == "team_b":
        return [key for key in matrix if key[1] + float(line) > key[0]]
    return []


def build_market_implied_score_matrix(
    prior_matrix: ScoreMatrix,
    context: dict[str, Any],
    iterations: int = 160,
) -> tuple[ScoreMatrix, dict[str, Any]]:
    total_distribution = context.get("total_distribution", []) or []
    matrix = force_total_distribution(prior_matrix, total_distribution)
    starting_matrix = dict(matrix)
    constraints: list[dict[str, Any]] = []

    for result, target in (context.get("moneyline_probs", {}) or {}).items():
        keys = score_event_keys(matrix, result)
        if keys:
            constraints.append({"kind": result, "target": float(target), "keys": keys})

    btts = context.get("btts")
    if btts:
        keys = score_event_keys(matrix, "btts_yes")
        constraints.append({"kind": "btts_yes", "target": float(btts["p_btts_yes"]), "keys": keys})

    for spread in context.get("spreads", []) or []:
        keys = spread_event_keys(matrix, spread)
        if keys:
            constraints.append(
                {
                    "kind": f"spread_{spread.get('spread_team')}_{spread.get('line')}",
                    "target": float(spread["p_cover"]),
                    "keys": keys,
                }
            )

    for _ in range(iterations):
        for constraint in constraints:
            matrix = apply_binary_event_constraint(matrix, constraint["keys"], constraint["target"])
        # Keep the totals ladder as the anchor. The other markets shape the split.
        matrix = force_total_distribution(matrix, total_distribution)

    residuals = []
    for constraint in constraints:
        fitted = event_probability(matrix, constraint["keys"])
        residuals.append(
            {
                "kind": constraint["kind"],
                "target": constraint["target"],
                "fitted": fitted,
                "abs_error": abs(fitted - constraint["target"]),
            }
        )
    mean_abs_error = sum(row["abs_error"] for row in residuals) / len(residuals) if residuals else None
    confidence = None if mean_abs_error is None else max(0.0, min(1.0, 1.0 - mean_abs_error / 0.08))
    meta = {
        "method": "minimum_cross_entropy_ipf",
        "prior": "market_total_tilted_model_matrix",
        "description": "Iterative proportional fitting: match totals first, then moneyline/BTTS/spreads while staying KL-close to the prior.",
        "constraint_count": len(constraints),
        "mean_abs_constraint_error": mean_abs_error,
        "market_matrix_confidence": confidence,
        "kl_from_total_tilt_prior": matrix_kl_divergence(starting_matrix, matrix),
        "constraint_residuals": residuals,
    }
    return matrix, meta


def total_market_cap_for_score(
    score_total: int,
    market_distribution_rows: list[dict[str, Any]],
) -> float | None:
    for row in market_distribution_rows:
        min_total = int(row["bucket_min_total"])
        max_total = row.get("bucket_max_total")
        if max_total is None or pd.isna(max_total):
            if score_total >= min_total:
                return float(row["market_probability"])
        elif min_total <= score_total <= int(max_total):
            return float(row["market_probability"])
    return None


def spread_event_probability(row: dict[str, Any], a_goals: int, b_goals: int) -> float | None:
    team = str(row.get("spread_team") or "")
    line = _to_float(row.get("line"))
    if line is None:
        return None
    if team == "team_a":
        covers = (a_goals + float(line)) > b_goals
    elif team == "team_b":
        covers = (b_goals + float(line)) > a_goals
    else:
        return None
    return float(row.get("p_cover") if covers else row.get("p_not_cover") or 0.0)


def add_book_consistency(
    rows: list[dict[str, Any]],
    context: dict[str, Any],
    tolerance: float = 0.003,
) -> list[dict[str, Any]]:
    moneyline_probs = context.get("moneyline_probs", {}) or {}
    btts = context.get("btts")
    spreads = context.get("spreads", []) or []
    total_distribution = context.get("total_distribution", []) or []

    for row in rows:
        raw_price = _to_float(row.get("raw_yes_price"), _to_float(row.get("market_implied_probability"), 0.0)) or 0.0
        caps: list[tuple[str, float]] = []
        if row.get("is_any_other_score") or row.get("team_a_goals") is None or row.get("team_b_goals") is None:
            row["book_consistency_cap"] = None
            row["book_consistency_cap_source"] = ""
            row["book_consistency_margin"] = None
            row["book_consistency_flags"] = ""
            row["book_consistency_ok"] = True
            row["book_adjusted_buy_score"] = float(row.get("risk_adjusted_edge") or 0.0)
            row["book_adjusted_verdict"] = row.get("buy_verdict")
            continue

        a_goals = int(row["team_a_goals"])
        b_goals = int(row["team_b_goals"])
        if a_goals > b_goals and "team_a_win" in moneyline_probs:
            caps.append(("moneyline_team_a_win", float(moneyline_probs["team_a_win"])))
        elif a_goals == b_goals and "draw" in moneyline_probs:
            caps.append(("moneyline_draw", float(moneyline_probs["draw"])))
        elif a_goals < b_goals and "team_b_win" in moneyline_probs:
            caps.append(("moneyline_team_b_win", float(moneyline_probs["team_b_win"])))

        total_cap = total_market_cap_for_score(a_goals + b_goals, total_distribution)
        if total_cap is not None:
            caps.append((f"total_{a_goals + b_goals}", float(total_cap)))

        if btts:
            btts_cap = float(btts["p_btts_yes"] if a_goals > 0 and b_goals > 0 else btts["p_btts_no"])
            caps.append(("btts_yes" if a_goals > 0 and b_goals > 0 else "btts_no", btts_cap))

        for spread in spreads:
            spread_prob = spread_event_probability(spread, a_goals, b_goals)
            if spread_prob is not None:
                label = f"spread_{spread.get('spread_team')}_{spread.get('line')}"
                caps.append((label, float(spread_prob)))

        if caps:
            cap_source, cap = min(caps, key=lambda item: item[1])
            flags = [name for name, value in caps if raw_price > value + tolerance]
            margin = cap - raw_price
            ok = not flags
        else:
            cap_source, cap, flags, margin, ok = "", None, [], None, True

        risk_edge = float(row.get("risk_adjusted_edge") or 0.0)
        penalty = 1.0 if ok else 0.25
        row["book_consistency_cap"] = cap
        row["book_consistency_cap_source"] = cap_source
        row["book_consistency_margin"] = margin
        row["book_consistency_flags"] = ";".join(flags)
        row["book_consistency_ok"] = bool(ok)
        row["book_adjusted_buy_score"] = risk_edge * penalty
        if not ok:
            row["book_adjusted_verdict"] = "avoid_book_conflict" if row.get("buy_verdict") == "buy" else "book_conflict"
        elif row.get("buy_verdict") == "buy":
            row["book_adjusted_verdict"] = "buy_book_ok"
        elif "watch" in str(row.get("buy_verdict", "")):
            row["book_adjusted_verdict"] = "watch_book_ok"
        else:
            row["book_adjusted_verdict"] = row.get("buy_verdict")
    return rows


def _bounded_confidence(value: float, floor: float = 0.15) -> float:
    return float(max(floor, min(1.0, value)))


def _liquidity_confidence(liquidity: float | None) -> float:
    if liquidity is None or liquidity <= 0:
        return 0.35
    # A smooth, simple scale: thin exact-score books are not treated like deep books.
    # Roughly: 1k -> 0.50, 10k -> 0.67, 100k -> 0.83, 1m+ -> 1.00.
    return _bounded_confidence((math.log10(float(liquidity) + 1.0) - 2.0) / 4.0, floor=0.35)


def _relative_agreement_confidence(a: float | None, b: float | None, floor: float = 0.25) -> float:
    if a is None or b is None:
        return 0.75
    denom = max(abs(float(a)), abs(float(b)), 0.01)
    disagreement = abs(float(a) - float(b)) / denom
    return _bounded_confidence(1.0 - 0.65 * disagreement, floor=floor)


def add_staking_confidence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Shrink Kelly by a transparent model-confidence estimate.

    This is deliberately a staking layer only. It does not rewrite fair prices,
    edges, verdicts, or scoreline rankings.
    """
    for row in rows:
        bucket_conf = float(row.get("reliability_weight") or 0.0)
        liquidity_conf = _liquidity_confidence(_to_float(row.get("liquidity")))
        market_matrix_conf = _relative_agreement_confidence(
            _to_float(row.get("scoreline_fair_probability")),
            _to_float(row.get("market_implied_score_probability")),
        )
        model_stability_conf = _relative_agreement_confidence(
            _to_float(row.get("scoreline_fair_probability")),
            _to_float(row.get("base_model_probability")),
            floor=0.35,
        )
        book_conf = 1.0 if row.get("book_consistency_ok", True) else 0.25

        staking_confidence = _bounded_confidence(
            bucket_conf
            * (0.55 + 0.45 * liquidity_conf)
            * (0.55 + 0.45 * market_matrix_conf)
            * (0.70 + 0.30 * model_stability_conf)
            * book_conf,
            floor=0.10,
        )
        row["staking_confidence"] = staking_confidence
        row["staking_confidence_bucket"] = bucket_conf
        row["staking_confidence_liquidity"] = liquidity_conf
        row["staking_confidence_market_matrix_agreement"] = market_matrix_conf
        row["staking_confidence_model_stability"] = model_stability_conf
        row["staking_confidence_book"] = book_conf

        for source, target in [
            ("joint_kelly_fraction", "confidence_joint_kelly_fraction"),
            ("half_joint_kelly_fraction", "confidence_half_joint_kelly_fraction"),
            ("quarter_joint_kelly_fraction", "confidence_quarter_joint_kelly_fraction"),
            ("risk_adjusted_kelly_fraction", "confidence_risk_adjusted_kelly_fraction"),
        ]:
            row[target] = float(row.get(source) or 0.0) * staking_confidence
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def unique_artifact_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    for idx in range(2, 1000):
        candidate = parent / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find a free artifact path for {path}")


def current_event_slug(fetch_meta: dict[str, Any]) -> str:
    return str(fetch_meta.get("event_slug") or fetch_meta.get("slug") or "").strip()


def infer_event_slug_from_exact_rows(rows: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        slug = str(row.get("slug") or "")
        if "-exact-score" not in slug:
            continue
        base = slug.split("-exact-score", 1)[0]
        if base:
            counts[base] = counts.get(base, 0) + 1
    if not counts:
        return ""
    return max(counts.items(), key=lambda item: item[1])[0]


def row_matches_event_slug(row: dict[str, Any], event_slug: str) -> bool:
    if not event_slug:
        return True
    slug = str(row.get("slug") or "")
    return slug == event_slug or slug.startswith(f"{event_slug}-")


def current_exact_score_rows(
    rows: list[dict[str, Any]],
    event_slug: str,
    *,
    include_any_other: bool = False,
) -> list[dict[str, Any]]:
    def usable(row: dict[str, Any]) -> bool:
        if row.get("is_any_other_score") and not include_any_other:
            return False
        if not row.get("is_any_other_score") and (
            row.get("team_a_goals") is None or row.get("team_b_goals") is None
        ):
            return False
        return True

    filtered = [
        row
        for row in rows
        if usable(row)
        and row_matches_event_slug(row, event_slug)
        and (not event_slug or "exact-score" in str(row.get("slug") or ""))
    ]
    if filtered:
        return filtered
    return [row for row in rows if usable(row)]


def add_entry_price_columns(rows: list[dict[str, Any]], uncertainty_buffer: float) -> list[dict[str, Any]]:
    for row in rows:
        fair = _to_float(row.get("model_only_fair_probability"), _to_float(row.get("scoreline_fair_probability"), 0.0)) or 0.0
        risk_adjusted = _to_float(row.get("risk_adjusted_probability"), fair) or fair
        price = _to_float(row.get("raw_yes_price"), _to_float(row.get("market_implied_probability"), 0.0)) or 0.0
        max_entry = max(0.0, fair - float(uncertainty_buffer))
        risk_max_entry = max(0.0, risk_adjusted - float(uncertainty_buffer))
        row["max_entry_price"] = max_entry
        row["max_entry_price_cents"] = max_entry * 100.0
        row["entry_room_vs_raw_yes"] = max_entry - price
        row["risk_adjusted_max_entry_price"] = risk_max_entry
        row["risk_adjusted_max_entry_price_cents"] = risk_max_entry * 100.0
        row["risk_adjusted_entry_room_vs_raw_yes"] = risk_max_entry - price
    return rows


def fetch_clob_orderbook(token_id: Any) -> dict[str, Any]:
    params = urllib.parse.urlencode({"token_id": str(token_id)})
    request = urllib.request.Request(
        f"{CLOB_BOOK_URL}?{params}",
        headers={"User-Agent": "world-cup-v42-edge-model/1.0"},
    )
    context = ssl.create_default_context(cafile=certifi.where()) if certifi else None
    with urllib.request.urlopen(request, timeout=15, context=context) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload if isinstance(payload, dict) else {}


def best_orderbook_quote(levels: Any, *, side: str) -> tuple[float | None, float | None]:
    if not isinstance(levels, list) or not levels:
        return None, None
    parsed: list[tuple[float, float]] = []
    for level in levels:
        if not isinstance(level, dict):
            continue
        price = _to_float(level.get("price"))
        size = _to_float(level.get("size"))
        if price is not None:
            parsed.append((float(price), float(size or 0.0)))
    if not parsed:
        return None, None
    price, size = (min(parsed, key=lambda item: item[0]) if side == "ask" else max(parsed, key=lambda item: item[0]))
    return price, size


def add_clob_quotes(rows: list[dict[str, Any]], uncertainty_buffer: float) -> list[dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    for row in rows:
        token_id = row.get("yes_token_id")
        if not token_id:
            row["clob_error"] = "missing_yes_token_id"
            continue
        token_key = str(token_id)
        try:
            if token_key not in cache:
                cache[token_key] = fetch_clob_orderbook(token_key)
                time.sleep(0.03)
            book = cache[token_key]
            best_ask, best_ask_size = best_orderbook_quote(book.get("asks"), side="ask")
            best_bid, best_bid_size = best_orderbook_quote(book.get("bids"), side="bid")
            fair = _to_float(row.get("model_only_fair_probability"), 0.0) or 0.0
            max_entry = max(0.0, fair - float(uncertainty_buffer))
            row["clob_yes_best_ask"] = best_ask
            row["clob_yes_best_ask_size"] = best_ask_size
            row["clob_yes_best_bid"] = best_bid
            row["clob_yes_best_bid_size"] = best_bid_size
            row["clob_ask_edge_after_buffer"] = None if best_ask is None else fair - float(best_ask) - float(uncertainty_buffer)
            row["clob_expected_return_vs_ask"] = None if not best_ask else fair / float(best_ask) - 1.0
            row["clob_ask_room_to_max_entry"] = None if best_ask is None else max_entry - float(best_ask)
            row["clob_error"] = ""
        except Exception as exc:
            row["clob_error"] = str(exc)
    return rows


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_reference_model_rows(
    reference_path: str | None,
    *,
    team_a: str = "",
    team_b: str = "",
) -> dict[str, dict[str, str]]:
    if not reference_path:
        return {}
    path = Path(reference_path)
    if path.is_dir():
        summary_path = path / "model_summary.json"
        if summary_path.exists() and (team_a or team_b):
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                ref_a = str(summary.get("team_a") or "")
                ref_b = str(summary.get("team_b") or "")
                if _norm_text(ref_a) != _norm_text(team_a) or _norm_text(ref_b) != _norm_text(team_b):
                    return {}
            except Exception:
                pass
        for name in ("polymarket_exact_score_edges.csv", "model_fair_scoreline_probabilities.csv"):
            candidate = path / name
            rows = read_csv_rows(candidate)
            if rows:
                break
        else:
            rows = []
    else:
        rows = read_csv_rows(path)
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        scoreline = str(row.get("scoreline") or "")
        if scoreline and scoreline != "Any Other Score":
            out[scoreline] = row
    return out


def add_reference_model_agreement(
    rows: list[dict[str, Any]],
    reference_rows: dict[str, dict[str, str]],
    uncertainty_buffer: float,
    *,
    label: str = "v43",
) -> list[dict[str, Any]]:
    for row in rows:
        scoreline = str(row.get("scoreline") or "")
        reference = reference_rows.get(scoreline)
        price = _to_float(row.get("raw_yes_price"), 0.0) or 0.0
        our_edge_after_buffer = _to_float(row.get("edge_vs_raw_yes_after_buffer"), 0.0) or 0.0
        if not reference:
            row[f"{label}_fair_probability"] = None
            row[f"{label}_edge_vs_raw_yes_after_buffer"] = None
            row[f"{label}_agreement_tag"] = f"missing_{label}"
            continue
        ref_fair = (
            _to_float(reference.get("model_only_fair_probability"))
            or _to_float(reference.get("scoreline_fair_probability"))
            or _to_float(reference.get("market_total_tilt_probability"))
            or 0.0
        )
        ref_edge_after_buffer = float(ref_fair) - price - float(uncertainty_buffer)
        our_fair = _to_float(row.get("model_only_fair_probability"), 0.0) or 0.0
        row[f"{label}_fair_probability"] = ref_fair
        row[f"{label}_edge_vs_raw_yes_after_buffer"] = ref_edge_after_buffer
        row[f"{label}_fair_probability_diff"] = our_fair - float(ref_fair)
        if our_edge_after_buffer > 0 and ref_edge_after_buffer > 0:
            tag = "both_buy_zone"
        elif our_edge_after_buffer > 0 and ref_edge_after_buffer > -0.005:
            tag = f"v42_buy_{label}_neutral"
        elif our_edge_after_buffer > 0:
            tag = f"v42_only_{label}_disagrees"
        elif ref_edge_after_buffer > 0:
            tag = f"{label}_only"
        else:
            tag = "both_skip"
        row[f"{label}_agreement_tag"] = tag
    return rows


def build_exact_score_baskets(
    rows: list[dict[str, Any]],
    *,
    team_a: str,
    team_b: str,
    uncertainty_buffer: float,
) -> list[dict[str, Any]]:
    listed = [
        row
        for row in rows
        if not row.get("is_any_other_score")
        and row.get("team_a_goals") is not None
        and row.get("team_b_goals") is not None
    ]

    def emit(name: str, description: str, predicate: Any) -> dict[str, Any] | None:
        members = [
            row
            for row in listed
            if predicate(int(row["team_a_goals"]), int(row["team_b_goals"]))
        ]
        if not members:
            return None
        price_sum = sum(_to_float(row.get("raw_yes_price"), 0.0) or 0.0 for row in members)
        fair_sum = sum(_to_float(row.get("model_only_fair_probability"), 0.0) or 0.0 for row in members)
        risk_fair_sum = sum(_to_float(row.get("risk_adjusted_probability"), row.get("model_only_fair_probability")) or 0.0 for row in members)
        market_matrix_values = [_to_float(row.get("market_implied_score_probability")) for row in members]
        market_matrix_sum = None if any(value is None for value in market_matrix_values) else sum(float(value or 0.0) for value in market_matrix_values)
        ask_values = [_to_float(row.get("clob_yes_best_ask")) for row in members]
        clob_ask_count = sum(1 for value in ask_values if value is not None)
        clob_ask_sum = None if clob_ask_count != len(members) else sum(float(value or 0.0) for value in ask_values)
        edge_after_buffer = fair_sum - price_sum - float(uncertainty_buffer)
        risk_edge_after_buffer = risk_fair_sum - price_sum - float(uncertainty_buffer)
        if edge_after_buffer > 0.015:
            verdict = "buy_basket"
        elif edge_after_buffer > 0:
            verdict = "watch_basket"
        else:
            verdict = "skip_basket"
        return {
            "basket": name,
            "description": description,
            "constituent_count": len(members),
            "constituents": " + ".join(str(row.get("scoreline") or "") for row in members),
            "posted_price_sum": price_sum,
            "model_fair_probability": fair_sum,
            "risk_adjusted_probability_sum": risk_fair_sum,
            "market_implied_matrix_probability_sum": market_matrix_sum,
            "edge_vs_posted_sum": fair_sum - price_sum,
            "edge_after_buffer": edge_after_buffer,
            "expected_return_vs_posted_sum": fair_sum / price_sum - 1.0 if price_sum > 0 else None,
            "max_entry_price_sum": max(0.0, fair_sum - float(uncertainty_buffer)),
            "entry_room_vs_posted_sum": fair_sum - float(uncertainty_buffer) - price_sum,
            "risk_adjusted_max_entry_price_sum": max(0.0, risk_fair_sum - float(uncertainty_buffer)),
            "risk_adjusted_entry_room_vs_posted_sum": risk_edge_after_buffer,
            "clob_best_ask_sum": clob_ask_sum,
            "clob_best_ask_coverage": f"{clob_ask_count}/{len(members)}",
            "clob_ask_edge_after_buffer": None if clob_ask_sum is None else fair_sum - clob_ask_sum - float(uncertainty_buffer),
            "basket_verdict": verdict,
        }

    specs = [
        (
            f"{team_a} clean-sheet win",
            f"{team_a} wins to nil using listed exact-score markets",
            lambda a, b: a > b and b == 0,
        ),
        (
            f"{team_b} clean-sheet win",
            f"{team_b} wins to nil using listed exact-score markets",
            lambda a, b: b > a and a == 0,
        ),
        (f"{team_a} listed regulation win", f"{team_a} wins across listed exact scores", lambda a, b: a > b),
        (f"{team_b} listed regulation win", f"{team_b} wins across listed exact scores", lambda a, b: b > a),
        ("Listed draw", "Draw exact scores that Polymarket listed", lambda a, b: a == b),
        (f"{team_a} win by 2+", f"{team_a} covers a multi-goal win among listed exact scores", lambda a, b: a - b >= 2),
        ("Listed total 0-2", "All listed exact scores with two or fewer goals", lambda a, b: a + b <= 2),
        ("Listed total 0-3", "All listed exact scores with three or fewer goals", lambda a, b: a + b <= 3),
    ]
    baskets = [row for row in (emit(*spec) for spec in specs) if row]
    baskets.sort(
        key=lambda row: (
            str(row.get("basket_verdict")) == "buy_basket",
            float(row.get("edge_after_buffer") or 0.0),
            float(row.get("model_fair_probability") or 0.0),
        ),
        reverse=True,
    )
    return baskets


def decision_tier(row: dict[str, Any]) -> str:
    verdict = str(row.get("book_adjusted_verdict") or row.get("buy_verdict") or "")
    q_kelly = _to_float(row.get("confidence_quarter_joint_kelly_fraction"), 0.0) or 0.0
    edge_after_buffer = _to_float(row.get("edge_vs_raw_yes_after_buffer"), 0.0) or 0.0
    risk_edge = _to_float(row.get("risk_adjusted_edge"), 0.0) or 0.0
    if verdict in {"buy", "buy_book_ok"} and q_kelly > 0 and edge_after_buffer > 0:
        return "BUY"
    if "watch" in verdict or (edge_after_buffer > 0 and risk_edge > 0):
        return "WATCH"
    return "PASS"


def decision_reason(row: dict[str, Any]) -> str:
    pieces: list[str] = []
    verdict = str(row.get("book_adjusted_verdict") or row.get("buy_verdict") or "")
    if verdict == "buy_book_ok":
        pieces.append("passes edge, EV, and book checks")
    elif verdict == "buy":
        pieces.append("passes edge and EV checks")
    elif "watch" in verdict:
        pieces.append("positive but not portfolio-sized")
    else:
        pieces.append("below buy threshold")
    if (_to_float(row.get("entry_room_vs_raw_yes"), 0.0) or 0.0) <= 0:
        pieces.append("above max entry")
    agreement = clean_label(row.get("v43_agreement_tag"))
    if agreement and not agreement.startswith("missing"):
        pieces.append(agreement)
    movement = clean_label(row.get("movement_tag"))
    if movement:
        pieces.append(movement)
    return "; ".join(pieces)


def action_call(row: dict[str, Any]) -> str:
    tier = decision_tier(row)
    max_entry = _to_float(row.get("max_entry_price"), 0.0) or 0.0
    price = _to_float(row.get("raw_yes_price"), 0.0) or 0.0
    ask = _to_float(row.get("clob_yes_best_ask"))
    ask_edge = _to_float(row.get("clob_ask_edge_after_buffer"))
    q_kelly = _to_float(row.get("confidence_quarter_joint_kelly_fraction"), 0.0) or 0.0
    if tier == "BUY":
        if ask is not None:
            if ask <= max_entry and (ask_edge is None or ask_edge > 0):
                return f"BUY ask <= {max_entry * 100:.2f}c"
            return f"WAIT ask > {max_entry * 100:.2f}c"
        return f"BUY <= {max_entry * 100:.2f}c"
    if max_entry <= 0.0:
        return "NO, fair < buffer"
    if ask is not None and ask <= max_entry and ask_edge is not None and ask_edge > 0:
        return f"SMALL/VERIFY ask {ask * 100:.2f}c"
    if price > max_entry:
        return f"NO, need <= {max_entry * 100:.2f}c"
    if q_kelly <= 0:
        return "NO SIZE"
    return "WATCH ONLY"


def previous_call_from_row(previous: dict[str, str] | None) -> str:
    if not previous:
        return ""
    previous_tier = str(previous.get("decision_tier") or previous.get("tier") or "")
    previous_action = str(previous.get("recommended_action") or previous.get("action") or "")
    previous_verdict = str(previous.get("book_adjusted_verdict") or previous.get("buy_verdict") or "")
    if previous_tier:
        return previous_tier
    if previous_action:
        return previous_action.split()[0]
    if previous_verdict == "buy_book_ok":
        return "BUY"
    if "watch" in previous_verdict:
        return "WATCH"
    if previous_verdict:
        return "PASS"
    return ""


def still_good_status(row: dict[str, Any]) -> str:
    previous_call = str(row.get("previous_call") or "")
    current_call = decision_tier(row)
    if not previous_call:
        return "new snapshot"
    if previous_call == "BUY" and current_call == "BUY":
        return "still good"
    if previous_call == "BUY" and current_call != "BUY":
        return "lost buy"
    if previous_call != "BUY" and current_call == "BUY":
        return "new buy"
    if previous_call == current_call:
        return "same"
    return f"{previous_call.lower()} -> {current_call.lower()}"


def pct_text(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:.{digits}f}%"


def entry_cap_text(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    if float(value) <= 0.0:
        return "none"
    return pct_text(value, digits)


def pp_text(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:+.{digits}f}pp"


def clean_label(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value)
    return "" if text.lower() in {"nan", "none"} else text


def build_decision_board_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    board: list[dict[str, Any]] = []
    for row in rows:
        tier = decision_tier(row)
        recommendation = action_call(row)
        row["decision_tier"] = tier
        row["recommended_action"] = recommendation
        row["still_good_status"] = still_good_status(row)
        board.append(
            {
                "tier": tier,
                "scoreline": row.get("scoreline"),
                "price": _to_float(row.get("raw_yes_price")),
                "fair": _to_float(row.get("model_only_fair_probability")),
                "max_entry_price": _to_float(row.get("max_entry_price")),
                "edge_after_buffer": _to_float(row.get("edge_vs_raw_yes_after_buffer")),
                "expected_return": _to_float(row.get("expected_return_vs_raw_yes")),
                "confidence": _to_float(row.get("staking_confidence")),
                "quarter_kelly": _to_float(row.get("confidence_quarter_joint_kelly_fraction")),
                "kelly_source": "confidence_joint_kelly",
                "market_matrix": _to_float(row.get("market_implied_score_probability")),
                "vs_market_matrix": _to_float(row.get("our_vs_market_matrix_edge")),
                "v43_fair": _to_float(row.get("v43_fair_probability")),
                "v43_tag": row.get("v43_agreement_tag"),
                "previous_price": _to_float(row.get("previous_raw_yes_price")),
                "price_change": _to_float(row.get("raw_yes_price_change")),
                "edge_change": _to_float(row.get("edge_after_buffer_change")),
                "previous_call": row.get("previous_call"),
                "still_good": row.get("still_good_status"),
                "clob_best_ask": _to_float(row.get("clob_yes_best_ask")),
                "clob_ask_edge_after_buffer": _to_float(row.get("clob_ask_edge_after_buffer")),
                "verdict": row.get("book_adjusted_verdict") or row.get("buy_verdict"),
                "recommendation": recommendation,
                "reason": decision_reason(row),
            }
        )
    tier_rank = {"BUY": 0, "WATCH": 1, "PASS": 2}
    board.sort(
        key=lambda row: (
            tier_rank.get(str(row.get("tier")), 9),
            -float(row.get("quarter_kelly") or 0.0),
            -float(row.get("edge_after_buffer") or -99.0),
            -float(row.get("fair") or 0.0),
        )
    )
    return board


def write_price_snapshot(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    event_slug: str,
    team_a: str,
    team_b: str,
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    snapshot_rows = []
    for row in rows:
        snapshot_rows.append(
            {
                "snapshot_utc": timestamp,
                "event_slug": event_slug,
                "team_a": team_a,
                "team_b": team_b,
                "slug": row.get("slug"),
                "scoreline": row.get("scoreline"),
                "raw_yes_price": row.get("raw_yes_price"),
                "market_implied_probability": row.get("market_implied_probability"),
                "model_only_fair_probability": row.get("model_only_fair_probability"),
                "edge_vs_raw_yes_after_buffer": row.get("edge_vs_raw_yes_after_buffer"),
                "decision_tier": row.get("decision_tier") or decision_tier(row),
                "recommended_action": row.get("recommended_action") or action_call(row),
                "buy_verdict": row.get("buy_verdict"),
                "book_adjusted_verdict": row.get("book_adjusted_verdict"),
            }
        )
    write_csv(path, snapshot_rows)


def find_previous_price_rows(
    history_root: str | None,
    *,
    current_output_dir: Path,
    event_slug: str,
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    if not history_root:
        return {}, {"source": "disabled"}
    root = Path(history_root)
    if not root.exists():
        return {}, {"source": "missing_history_root", "history_root": str(root)}
    candidates: list[tuple[float, Path, list[dict[str, str]]]] = []
    for path in root.rglob("price_snapshot.csv"):
        if current_output_dir in path.parents:
            continue
        rows = [
            row
            for row in read_csv_rows(path)
            if (not event_slug or row.get("event_slug") == event_slug or str(row.get("slug") or "").startswith(f"{event_slug}-"))
        ]
        if rows:
            candidates.append((path.stat().st_mtime, path, rows))
    if not candidates:
        for path in root.rglob("polymarket_exact_score_edges.csv"):
            if current_output_dir in path.parents:
                continue
            rows = [
                row
                for row in read_csv_rows(path)
                if row.get("scoreline") != "Any Other Score"
                and (not event_slug or str(row.get("slug") or "").startswith(f"{event_slug}-"))
            ]
            if rows:
                candidates.append((path.stat().st_mtime, path, rows))
    if not candidates:
        return {}, {"source": "none_found", "history_root": str(root)}
    _mtime, path, rows = max(candidates, key=lambda item: item[0])
    mapped = {str(row.get("scoreline") or ""): row for row in rows if row.get("scoreline")}
    return mapped, {"source": str(path), "row_count": len(mapped)}


def add_price_movement_columns(
    rows: list[dict[str, Any]],
    previous_rows: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    for row in rows:
        previous = previous_rows.get(str(row.get("scoreline") or ""))
        if not previous:
            row["previous_raw_yes_price"] = None
            row["raw_yes_price_change"] = None
            row["previous_edge_after_buffer"] = None
            row["edge_after_buffer_change"] = None
            row["previous_call"] = ""
            row["movement_tag"] = ""
            continue
        current_price = _to_float(row.get("raw_yes_price"), 0.0) or 0.0
        previous_price = _to_float(previous.get("raw_yes_price"), _to_float(previous.get("yes_price"), 0.0)) or 0.0
        current_edge = _to_float(row.get("edge_vs_raw_yes_after_buffer"), 0.0) or 0.0
        previous_edge = _to_float(previous.get("edge_vs_raw_yes_after_buffer"), 0.0) or 0.0
        price_change = current_price - previous_price
        edge_change = current_edge - previous_edge
        if edge_change > 0.0025:
            movement_tag = "edge expanding"
        elif edge_change < -0.0025:
            movement_tag = "edge compressing"
        elif abs(price_change) <= 0.001:
            movement_tag = "flat/stale quote"
        else:
            movement_tag = "small move"
        row["previous_raw_yes_price"] = previous_price
        row["raw_yes_price_change"] = price_change
        row["previous_edge_after_buffer"] = previous_edge
        row["edge_after_buffer_change"] = edge_change
        row["previous_call"] = previous_call_from_row(previous)
        row["movement_tag"] = movement_tag
    return rows


def write_decision_brief(
    path: Path,
    *,
    team_a: str,
    team_b: str,
    event_slug: str,
    decision_rows: list[dict[str, Any]],
    basket_rows: list[dict[str, Any]],
    market_matrix_meta: dict[str, Any] | None,
    price_history_meta: dict[str, Any],
) -> None:
    buys = [row for row in decision_rows if row.get("tier") == "BUY"]
    watches = [row for row in decision_rows if row.get("tier") == "WATCH"]
    basket_buys = [row for row in basket_rows if row.get("basket_verdict") == "buy_basket"]
    lines = [
        f"# {team_a} vs {team_b} Exact-Score Decision Brief",
        "",
        f"- Source: `{event_slug or 'unknown'}`",
        f"- Market matrix confidence: {pct_text(_to_float((market_matrix_meta or {}).get('market_matrix_confidence')), 1)}",
        f"- Price history source: `{price_history_meta.get('source', 'none')}`",
        "",
        "## Buy",
    ]
    if buys:
        for row in buys:
            lines.append(
                f"- **{row['scoreline']}** at {pct_text(row.get('price'), 2)}; fair {pct_text(row.get('fair'), 2)}; "
                f"max entry {entry_cap_text(row.get('max_entry_price'), 2)}; edge {pp_text(row.get('edge_after_buffer'), 2)}; "
                f"1/4 Kelly {pct_text(row.get('quarter_kelly'), 3)}; {row.get('reason')}"
            )
    else:
        lines.append("- No portfolio-sized exact-score buys.")
    lines.extend(["", "## Watch"])
    if watches:
        for row in watches[:8]:
            lines.append(
                f"- **{row['scoreline']}** at {pct_text(row.get('price'), 2)}; fair {pct_text(row.get('fair'), 2)}; "
                f"max entry {entry_cap_text(row.get('max_entry_price'), 2)}; edge {pp_text(row.get('edge_after_buffer'), 2)}; {row.get('reason')}"
            )
    else:
        lines.append("- None.")
    lines.extend(["", "## Basket Pricing"])
    if basket_buys:
        for row in basket_buys[:6]:
            lines.append(
                f"- **{row['basket']}**: posted sum {pct_text(_to_float(row.get('posted_price_sum')), 2)}; "
                f"fair {pct_text(_to_float(row.get('model_fair_probability')), 2)}; "
                f"max entry {pct_text(_to_float(row.get('max_entry_price_sum')), 2)}; "
                f"edge {pp_text(_to_float(row.get('edge_after_buffer')), 2)}; "
                f"legs `{row.get('constituents')}`"
            )
    else:
        for row in basket_rows[:5]:
            lines.append(
                f"- {row['basket']}: posted {pct_text(_to_float(row.get('posted_price_sum')), 2)}, "
                f"fair {pct_text(_to_float(row.get('model_fair_probability')), 2)}, "
                f"edge {pp_text(_to_float(row.get('edge_after_buffer')), 2)}"
            )
    lines.extend(
        [
            "",
            "## Files",
            "- `exact_score_decision_board.csv`: BUY/WATCH/PASS table with max entry and movement columns.",
            "- `exact_score_baskets.csv`: summed mutually exclusive exact-score constructions.",
            "- `price_snapshot.csv` and `price_movement.csv`: refresh-to-refresh price tracking.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_decision_board_plot(
    path: Path,
    decision_rows: list[dict[str, Any]],
    basket_rows: list[dict[str, Any]],
    *,
    team_a: str,
    team_b: str,
    limit: int = 12,
) -> None:
    rows = decision_rows[:limit]
    if not rows and not basket_rows:
        return
    fig_h = max(6.2, 0.44 * len(rows) + 2.4 + 0.22 * min(len(basket_rows), 5))
    fig, ax = plt.subplots(figsize=(17.5, fig_h))
    ax.axis("off")
    ax.set_title(f"{team_a} vs {team_b}: exact-score refresh board", fontweight="bold", pad=14)
    columns = [
        ("Now", 0.01),
        ("Was", 0.08),
        ("Score", 0.15),
        ("Price", 0.23),
        ("Δ", 0.30),
        ("Max", 0.37),
        ("Ask", 0.44),
        ("Edge", 0.51),
        ("1/4K", 0.59),
        ("Status", 0.67),
        ("Recommendation", 0.78),
    ]
    y = 0.94
    for name, x in columns:
        ax.text(x, y, name, transform=ax.transAxes, ha="left", va="center", fontsize=9, fontweight="bold", color="#333333")
    y -= 0.035
    ax.hlines(y, 0.0, 1.0, transform=ax.transAxes, color="#DDDDDD", linewidth=0.8)
    y -= 0.035
    colors = {"BUY": "#147D56", "WATCH": "#9A5A00", "PASS": "#555555"}
    for row in rows:
        tier = str(row.get("tier") or "")
        color = colors.get(tier, "#555555")
        values = [
            tier,
            str(row.get("previous_call") or "-"),
            str(row.get("scoreline") or ""),
            pct_text(_to_float(row.get("price")), 2),
            pp_text(_to_float(row.get("price_change")), 2),
            entry_cap_text(_to_float(row.get("max_entry_price")), 2),
            pct_text(_to_float(row.get("clob_best_ask")), 2),
            pp_text(_to_float(row.get("edge_after_buffer")), 2),
            pct_text(_to_float(row.get("quarter_kelly")), 2),
            str(row.get("still_good") or "")[:16],
            str(row.get("recommendation") or "")[:30],
        ]
        for (_name, x), value in zip(columns, values):
            ax.text(x, y, value, transform=ax.transAxes, ha="left", va="center", fontsize=8.2, color=color)
        y -= 0.04
        if y < 0.20:
            break
    if basket_rows and y > 0.12:
        y -= 0.025
        ax.text(0.01, y, "Baskets", transform=ax.transAxes, ha="left", va="center", fontsize=10, fontweight="bold", color="#333333")
        y -= 0.04
        for row in basket_rows[:5]:
            text = (
                f"{row.get('basket')}: price {pct_text(_to_float(row.get('posted_price_sum')), 2)} | "
                f"fair {pct_text(_to_float(row.get('model_fair_probability')), 2)} | "
                f"edge {pp_text(_to_float(row.get('edge_after_buffer')), 2)} | {row.get('constituents')}"
            )
            color = "#147D56" if row.get("basket_verdict") == "buy_basket" else "#555555"
            ax.text(0.01, y, text[:150], transform=ax.transAxes, ha="left", va="center", fontsize=8.2, color=color)
            y -= 0.035
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_action_delta_plot(
    path: Path,
    decision_rows: list[dict[str, Any]],
    *,
    team_a: str,
    team_b: str,
    limit: int = 10,
) -> None:
    if not decision_rows:
        return

    def row_rank(row: dict[str, Any]) -> tuple[int, float, float, float]:
        tier_rank = {"BUY": 0, "WATCH": 1, "PASS": 2}.get(str(row.get("tier") or ""), 9)
        edge = _to_float(row.get("edge_after_buffer"), 0.0) or 0.0
        edge_change = _to_float(row.get("edge_change"), 0.0) or 0.0
        price_change = _to_float(row.get("price_change"), 0.0) or 0.0
        return (tier_rank, -edge, -edge_change, price_change)

    rows = sorted(decision_rows, key=row_rank)[:limit]
    labels = [str(row.get("scoreline") or "") for row in rows]
    y_positions = list(range(len(rows)))

    def display_pp(value: Any, cap: float = 75.0) -> float:
        raw = (_to_float(value, 0.0) or 0.0) * 100.0
        if not math.isfinite(raw):
            return 0.0
        return max(-cap, min(cap, raw))

    edge = [display_pp(row.get("edge_after_buffer")) for row in rows]
    edge_delta = [display_pp(row.get("edge_change")) for row in rows]
    price_delta = [display_pp(row.get("price_change")) for row in rows]

    fig_h = max(5.8, 0.62 * len(rows) + 2.0)
    fig, (edge_ax, delta_ax) = plt.subplots(
        ncols=2,
        figsize=(16.5, fig_h),
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )
    colors = {"BUY": "#0F9F6E", "WATCH": "#F59E0B", "PASS": "#9CA3AF"}
    bar_colors = [colors.get(str(row.get("tier") or ""), "#9CA3AF") for row in rows]

    edge_ax.barh(y_positions, edge, color=bar_colors, height=0.55)
    edge_ax.axvline(0, color="#111827", linewidth=0.9)
    edge_ax.set_yticks(y_positions)
    edge_ax.set_yticklabels(labels)
    edge_ax.invert_yaxis()
    edge_ax.set_xlabel("Current edge after buffer (pp)")
    edge_ax.set_title("Action edge", fontweight="bold")
    edge_ax.grid(axis="x", alpha=0.22)
    edge_span = max([abs(value) for value in edge] + [5.0])
    edge_ax.set_xlim(-edge_span * 1.22, edge_span * 1.75)

    for idx, row in enumerate(rows):
        tier = str(row.get("tier") or "")
        price = _to_float(row.get("price"), 0.0) or 0.0
        max_entry = _to_float(row.get("max_entry_price"), 0.0)
        qk = _to_float(row.get("quarter_kelly"), 0.0) or 0.0
        rec = str(row.get("recommendation") or "")
        detail = (
            f"{tier} | price {price * 100:.2f}c | max {entry_cap_text(max_entry, 2)} | "
            f"1/4K {qk * 100:.3f}% | {rec}"
        )
        x_text = edge[idx] + (0.45 if edge[idx] >= 0 else -0.45)
        ha = "left" if edge[idx] >= 0 else "right"
        edge_ax.text(x_text, idx, detail[:96], va="center", ha=ha, fontsize=8.1, color="#111827")

    delta_ax.axvline(0, color="#111827", linewidth=0.9)
    delta_ax.scatter(price_delta, y_positions, marker="o", s=58, color="#2563EB", label="Price delta")
    delta_ax.scatter(edge_delta, [y + 0.16 for y in y_positions], marker="D", s=50, color="#DC2626", label="Edge delta")
    delta_ax.set_yticks(y_positions)
    delta_ax.set_yticklabels([])
    delta_ax.invert_yaxis()
    delta_ax.set_xlabel("Change since previous refresh (pp)")
    delta_ax.set_title("Delta tape", fontweight="bold")
    delta_ax.grid(axis="x", alpha=0.22)
    delta_ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    delta_span = max([abs(value) for value in price_delta + edge_delta] + [2.0])
    delta_ax.set_xlim(-delta_span * 1.25, delta_span * 1.65)

    for idx, row in enumerate(rows):
        movement = str(row.get("still_good") or "")
        delta_ax.text(
            max(price_delta[idx], edge_delta[idx], 0.0) + 0.35,
            idx,
            movement[:22] or "new snapshot",
            va="center",
            ha="left",
            fontsize=8.0,
            color="#374151",
        )

    fig.suptitle(f"{team_a} vs {team_b}: what changed / what to buy", fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_basket_value_plot(
    path: Path,
    basket_rows: list[dict[str, Any]],
    *,
    team_a: str,
    team_b: str,
    limit: int = 8,
) -> None:
    rows = basket_rows[:limit]
    if not rows:
        return
    labels = [str(row.get("basket") or "") for row in rows]
    y_positions = list(range(len(rows)))
    posted = [float(row.get("posted_price_sum") or 0.0) * 100.0 for row in rows]
    fair = [float(row.get("model_fair_probability") or 0.0) * 100.0 for row in rows]
    ask = [
        (float(row.get("clob_best_ask_sum") or 0.0) * 100.0) if row.get("clob_best_ask_sum") not in (None, "") else None
        for row in rows
    ]

    fig_h = max(5.2, 0.55 * len(rows) + 1.8)
    fig, ax = plt.subplots(figsize=(13.5, fig_h))
    bar_h = 0.24
    ax.barh([y - bar_h for y in y_positions], fair, height=bar_h, color="#147D56", label="Model fair")
    ax.barh(y_positions, posted, height=bar_h, color="#6B7280", label="Posted sum")
    if any(value is not None for value in ask):
        ax.barh(
            [y + bar_h for y in y_positions],
            [value or 0.0 for value in ask],
            height=bar_h,
            color="#2563EB",
            label="CLOB ask sum",
        )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Probability / price (%)")
    ax.set_title(f"{team_a} vs {team_b}: basket value", fontweight="bold")
    ax.grid(axis="x", alpha=0.22)
    x_max = max(fair + posted + [value or 0.0 for value in ask] + [1.0])
    ax.set_xlim(0, x_max * 1.35)
    for idx, row in enumerate(rows):
        edge = _to_float(row.get("edge_after_buffer"), 0.0) or 0.0
        action = "BUY" if row.get("basket_verdict") == "buy_basket" else "SKIP"
        detail = f"{action} | edge {edge * 100:+.1f}pp | {row.get('constituents')}"
        color = "#147D56" if action == "BUY" else "#555555"
        ax.text(max(fair[idx], posted[idx], ask[idx] or 0.0) + x_max * 0.025, idx, detail[:88], va="center", fontsize=8.0, color=color)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_exact_score_value_map_plot(
    path: Path,
    decision_rows: list[dict[str, Any]],
    *,
    team_a: str,
    team_b: str,
) -> None:
    rows = [
        row
        for row in decision_rows
        if re.match(r"^\d+-\d+$", str(row.get("scoreline") or ""))
    ]
    if not rows:
        return

    parsed: list[tuple[int, int, dict[str, Any]]] = []
    for row in rows:
        a_text, b_text = str(row.get("scoreline")).split("-", 1)
        parsed.append((int(a_text), int(b_text), row))
    max_a = max(a for a, _b, _row in parsed)
    max_b = max(b for _a, b, _row in parsed)
    row_by_score = {(a, b): row for a, b, row in parsed}

    fig_w = max(11.5, 2.25 * (max_b + 1) + 4.0)
    fig_h = max(8.0, 1.75 * (max_a + 1) + 2.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(-0.5, max_b + 0.5)
    ax.set_ylim(-0.5, max_a + 0.5)
    ax.invert_yaxis()
    ax.set_xticks(range(max_b + 1))
    ax.set_yticks(range(max_a + 1))
    ax.set_xlabel(f"{team_b} goals")
    ax.set_ylabel(f"{team_a} goals")
    ax.set_title(f"{team_a} vs {team_b}: exact-score value map", fontweight="bold", pad=14)
    ax.set_aspect("equal")
    ax.grid(color="#E5E7EB", linewidth=1.0)

    edge_values = [abs(float(row.get("edge_after_buffer") or 0.0)) for _a, _b, row in parsed]
    max_edge = max(edge_values + [0.01])
    colors = {
        "BUY": "#0F9F6E",
        "WATCH": "#F59E0B",
        "PASS": "#D1D5DB",
    }
    text_colors = {
        "BUY": "#064E3B",
        "WATCH": "#7C2D12",
        "PASS": "#374151",
    }

    for a in range(max_a + 1):
        for b in range(max_b + 1):
            row = row_by_score.get((a, b))
            if row is None:
                ax.add_patch(plt.Rectangle((b - 0.48, a - 0.48), 0.96, 0.96, facecolor="#F9FAFB", edgecolor="#E5E7EB"))
                ax.text(b, a, f"{a}-{b}\nnot listed", ha="center", va="center", fontsize=8.0, color="#9CA3AF")
                continue

            tier = str(row.get("tier") or "PASS")
            edge = float(row.get("edge_after_buffer") or 0.0)
            alpha = 0.25 + 0.65 * min(1.0, abs(edge) / max_edge)
            face = colors.get(tier, "#D1D5DB")
            edge_color = "#065F46" if tier == "BUY" else "#B45309" if tier == "WATCH" else "#9CA3AF"
            linewidth = 2.6 if tier == "BUY" else 1.8 if tier == "WATCH" else 1.0
            ax.add_patch(
                plt.Rectangle(
                    (b - 0.48, a - 0.48),
                    0.96,
                    0.96,
                    facecolor=face,
                    edgecolor=edge_color,
                    linewidth=linewidth,
                    alpha=alpha,
                )
            )
            ask = _to_float(row.get("clob_best_ask"))
            ask_text = "-" if ask is None else f"{ask * 100:.1f}c"
            price = _to_float(row.get("price"), 0.0) or 0.0
            max_entry = _to_float(row.get("max_entry_price"), 0.0) or 0.0
            status = str(row.get("still_good") or "")
            action = str(row.get("recommendation") or "")
            if action.startswith("BUY"):
                action_label = "BUY"
            elif action.startswith("SMALL"):
                action_label = "SMALL"
            elif action.startswith("WAIT"):
                action_label = "WAIT"
            else:
                action_label = "NO"
            cell_text = (
                f"{a}-{b}  {tier}\n"
                f"p {price * 100:.1f}c | ask {ask_text}\n"
                f"max {max_entry * 100:.1f}c | edge {edge * 100:+.1f}pp\n"
                f"{action_label}  {status[:12]}"
            )
            ax.text(
                b,
                a,
                cell_text,
                ha="center",
                va="center",
                fontsize=7.7,
                color=text_colors.get(tier, "#374151"),
                fontweight="bold" if tier == "BUY" else "normal",
                linespacing=1.25,
            )

    legend_items = [
        ("BUY", "#0F9F6E"),
        ("WATCH", "#F59E0B"),
        ("PASS", "#D1D5DB"),
        ("unlisted", "#F9FAFB"),
    ]
    for idx, (label, color) in enumerate(legend_items):
        x = 0.02 + idx * 0.15
        ax.add_patch(
            plt.Rectangle(
                (x, -0.13),
                0.035,
                0.035,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor="#9CA3AF",
                clip_on=False,
            )
        )
        ax.text(x + 0.045, -0.112, label, transform=ax.transAxes, va="center", fontsize=9, color="#333333")

    ax.text(
        0.02,
        -0.19,
        "Each cell shows listed exact score, current price, executable ask, max entry, edge after buffer, and current action.",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=9,
        color="#555555",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_total_svg(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    width, height = 920, 440
    ml, mt, mr, mb = 70, 45, 30, 65
    chart_w = width - ml - mr
    chart_h = height - mt - mb
    max_v = max([float(row.get("market_probability", 0.0)) for row in rows] + [0.01])
    max_v = max(max_v, max([float(row.get("model_probability", 0.0)) for row in rows] + [0.01]))
    max_v = min(1.0, max(0.25, max_v * 1.25))
    group_w = chart_w / max(1, len(rows))
    bar_w = group_w / 3.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="{width/2:.0f}" y="28" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{title}</text>',
    ]
    for tick in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        if tick > max_v + 1e-9:
            continue
        y = mt + chart_h - (tick / max_v) * chart_h
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{width-mr}" y2="{y:.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="{ml-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{tick:.1f}</text>')
    for idx, row in enumerate(rows):
        gx = ml + idx * group_w
        label = str(row["total_bucket"])
        parts.append(f'<text x="{gx+group_w/2:.1f}" y="{height-35}" text-anchor="middle" font-family="Arial" font-size="12">{label}</text>')
        for offset, key, color in [
            (0.45, "model_probability", "#4c78a8"),
            (1.35, "market_probability", "#f58518"),
            (2.25, "tilted_model_probability", "#54a24b"),
        ]:
            value = float(row.get(key, 0.0))
            bh = (value / max_v) * chart_h
            x = gx + offset * bar_w
            y = mt + chart_h - bh
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w*0.8:.1f}" height="{bh:.1f}" fill="{color}"/>')
    legend = [("Model fair", "#4c78a8"), ("Polymarket", "#f58518"), ("Market-total reference", "#54a24b")]
    for idx, (name, color) in enumerate(legend):
        x = ml + idx * 180
        parts.append(f'<rect x="{x}" y="{height-18}" width="13" height="13" fill="{color}"/>')
        parts.append(f'<text x="{x+19}" y="{height-7}" font-family="Arial" font-size="12">{name}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_edge_svg(path: Path, rows: list[dict[str, Any]], title: str, limit: int = 12) -> None:
    rows = rows[:limit]
    width, height = 1000, max(360, 80 + 28 * len(rows))
    ml, mt, mr, mb = 190, 45, 30, 40
    chart_w = width - ml - mr
    max_abs = max([abs(float(row.get("edge_vs_devig_market", 0.0))) for row in rows] + [0.02])
    max_abs = max(0.05, max_abs * 1.2)
    zero_x = ml + chart_w / 2.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="{width/2:.0f}" y="28" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{title}</text>',
        f'<line x1="{zero_x:.1f}" y1="{mt}" x2="{zero_x:.1f}" y2="{height-mb}" stroke="#333"/>',
    ]
    for idx, row in enumerate(rows):
        y = mt + 24 + idx * 28
        edge = float(row.get("edge_vs_devig_market", 0.0))
        half_w = (abs(edge) / max_abs) * (chart_w / 2.0)
        x = zero_x if edge >= 0 else zero_x - half_w
        color = "#2ca25f" if edge >= 0 else "#de2d26"
        label = str(row.get("scoreline", ""))
        ev = row.get("expected_return_vs_raw_yes")
        ev_text = "" if ev is None else f" EV {float(ev)*100:.0f}%"
        parts.append(f'<text x="{ml-12}" y="{y+5}" text-anchor="end" font-family="Arial" font-size="12">{label}</text>')
        parts.append(f'<rect x="{x:.1f}" y="{y-9}" width="{half_w:.1f}" height="18" fill="{color}"/>')
        parts.append(
            f'<text x="{(x + half_w + 6) if edge >= 0 else (x - 6):.1f}" y="{y+5}" '
            f'text-anchor="{"start" if edge >= 0 else "end"}" font-family="Arial" font-size="11">'
            f'{edge*100:+.1f}%{ev_text}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_top_scorelines_market_svg(
    path: Path,
    *,
    base_matrix: ScoreMatrix,
    fair_matrix: ScoreMatrix,
    exact_markets: list[dict[str, Any]],
    team_a: str,
    team_b: str,
    limit: int = 16,
) -> None:
    market_by_score = {
        str(row["scoreline"]): row
        for row in exact_markets
        if not row.get("is_any_other_score")
    }
    listed_score_keys = {
        (int(row["team_a_goals"]), int(row["team_b_goals"]))
        for row in exact_markets
        if not row.get("is_any_other_score")
        and row.get("team_a_goals") is not None
        and row.get("team_b_goals") is not None
    }
    any_other_market = next((row for row in exact_markets if row.get("is_any_other_score")), None)
    rows = []
    for key, base_prob in sorted(base_matrix.items(), key=lambda item: item[1], reverse=True):
        scoreline = f"{key[0]}-{key[1]}"
        market = market_by_score.get(scoreline)
        if not market:
            continue
        market_price = _to_float(market.get("yes_price")) if market else None
        market_prob = _to_float(market.get("market_probability")) if market else None
        fair = float(fair_matrix.get(key, 0.0))
        rows.append(
            {
                "scoreline": scoreline,
                "base": float(base_prob),
                "fair": fair,
                "market": market_prob,
                "raw": market_price,
                "edge": None if market_prob is None else fair - market_prob,
            }
        )
        if len(rows) >= limit:
            break
    if any_other_market:
        any_base = sum(prob for key, prob in base_matrix.items() if key not in listed_score_keys)
        any_fair = sum(prob for key, prob in fair_matrix.items() if key not in listed_score_keys)
        any_market = _to_float(any_other_market.get("market_probability"))
        any_raw = _to_float(any_other_market.get("yes_price"))
        rows.append(
            {
                "scoreline": "Any Other Score",
                "base": any_base,
                "fair": any_fair,
                "market": any_market,
                "raw": any_raw,
                "edge": None if any_market is None else any_fair - any_market,
            }
        )

    if not rows:
        return
    labels = [row["scoreline"] for row in rows]
    y_positions = list(range(len(rows)))
    bar_h = 0.24
    fig_h = max(6.0, 0.46 * len(rows) + 1.8)
    fig, ax = plt.subplots(figsize=(12.5, fig_h))
    base_values = [row["base"] * 100 for row in rows]
    fair_values = [row["fair"] * 100 for row in rows]
    market_values = [(row["market"] or 0.0) * 100 for row in rows]
    ax.barh([y - bar_h for y in y_positions], base_values, height=bar_h, color="#4C78A8", label="Base model")
    ax.barh(y_positions, fair_values, height=bar_h, color="#54A24B", label="Model fair")
    ax.barh([y + bar_h for y in y_positions], market_values, height=bar_h, color="#F58518", label="Polymarket Yes")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Probability / price (%)")
    ax.set_title(f"{team_a} vs {team_b}: scoreline probabilities vs Polymarket prices", fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=3, frameon=False)
    x_max = max(base_values + fair_values + market_values + [1.0])
    ax.set_xlim(0, x_max * 1.18)
    for idx, row in enumerate(rows):
        market = row["market"]
        if market is None:
            text = "not listed"
            color = "#666666"
        else:
            edge = float(row["edge"] or 0.0)
            text = f"edge {edge*100:+.1f}pp"
            color = "#1B9E77" if edge > 0 else "#D62728"
        row_max = max(row["base"], row["fair"], row["market"] or 0.0) * 100
        ax.text(row_max + x_max * 0.018, idx, text, va="center", ha="left", fontsize=9, color=color, fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_buy_candidate_plot(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    team_a: str,
    team_b: str,
    limit: int = 14,
) -> None:
    candidates = [
        row
        for row in rows
        if _to_float(row.get("edge_vs_devig_market"), 0.0) is not None
        and float(row.get("edge_vs_devig_market") or 0.0) > 0
    ]
    candidates.sort(
        key=lambda row: (
            float(row.get("risk_adjusted_edge") or 0.0),
            float(row.get("edge_vs_devig_market") or 0.0),
        ),
        reverse=True,
    )
    candidates = candidates[:limit]
    if not candidates:
        return
    bucket_short = {
        "validated_total_le_3": "valid T<=3",
        "low_total_lopsided": "lopsided",
        "high_total_4_discounted": "T4 disc",
        "tail_total_5_plus_heavily_discounted": "T5+ disc",
        "any_other_bucket": "any other",
    }

    labels = [
        f"#{int(row['model_scoreline_rank'])} {row.get('scoreline', '')}"
        if row.get("model_scoreline_rank") not in (None, "")
        else str(row.get("scoreline", ""))
        for row in candidates
    ]
    y_positions = list(range(len(candidates)))
    raw_edges = [float(row.get("edge_vs_devig_market") or 0.0) * 100.0 for row in candidates]
    adjusted_edges = [float(row.get("risk_adjusted_edge") or 0.0) * 100.0 for row in candidates]

    fig_h = max(5.8, 0.48 * len(candidates) + 1.8)
    fig, (ax, detail_ax) = plt.subplots(
        ncols=2,
        figsize=(14.8, fig_h),
        gridspec_kw={"width_ratios": [1.0, 1.38], "wspace": 0.04},
    )
    bar_h = 0.34
    ax.barh([y - bar_h / 2 for y in y_positions], raw_edges, height=bar_h, color="#4C78A8", label="Raw edge")
    ax.barh([y + bar_h / 2 for y in y_positions], adjusted_edges, height=bar_h, color="#F58518", label="Risk-adjusted edge")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Edge over Polymarket price (percentage points)")
    ax.set_title(f"{team_a} vs {team_b}: buy candidates after tail-risk discount", fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    x_max = max(raw_edges + adjusted_edges + [1.0])
    ax.set_xlim(0, x_max * 1.12)

    detail_ax.set_ylim(ax.get_ylim())
    detail_ax.set_xlim(0, 1)
    detail_ax.axis("off")
    column_specs = [
        ("Fair", 0.02, "left"),
        ("Price", 0.15, "left"),
        ("Adj Edge", 0.28, "left"),
        ("Conf", 0.43, "left"),
        ("1/4 Kelly", 0.55, "left"),
        ("Mkt Ref", 0.70, "left"),
        ("Tag", 0.84, "left"),
    ]
    header_y = -0.72
    for label, xpos, align in column_specs:
        detail_ax.text(xpos, header_y, label, ha=align, va="center", fontsize=8.4, fontweight="bold", color="#333333")
    detail_ax.hlines(-0.45, 0.0, 1.0, color="#DDDDDD", linewidth=0.8)

    for idx, row in enumerate(candidates):
        adjusted = float(row.get("risk_adjusted_edge") or 0.0)
        price = float(row.get("raw_yes_price") or row.get("market_implied_probability") or 0.0)
        fair = float(row.get("scoreline_fair_probability") or row.get("market_total_tilt_probability") or 0.0)
        confidence = float(row.get("staking_confidence") or 0.0)
        confidence_quarter = row.get("confidence_quarter_joint_kelly_fraction")
        market_ref = row.get("market_implied_score_probability")
        bucket = bucket_short.get(str(row.get("selection_bucket", "")), str(row.get("selection_bucket", "")))
        verdict = str(row.get("buy_verdict", ""))
        color = "#1B9E77" if verdict == "buy" else "#B36B00" if "watch" in verdict else "#555555"
        values = [
            f"{fair*100:.1f}%",
            f"{price*100:.1f}c",
            f"{adjusted*100:.1f}pp",
            f"{confidence*100:.0f}%",
            f"{float(confidence_quarter or 0.0)*100:.1f}%",
            "--" if market_ref is None else f"{float(market_ref)*100:.1f}%",
            f"{bucket} / {verdict}",
        ]
        for (_label, xpos, align), value in zip(column_specs, values):
            detail_ax.text(xpos, idx, value, ha=align, va="center", fontsize=8.2, color=color)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_book_consistency_plot(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    team_a: str,
    team_b: str,
    limit: int = 14,
) -> None:
    candidates = [
        row
        for row in rows
        if not row.get("is_any_other_score")
        and _to_float(row.get("edge_vs_devig_market"), 0.0) is not None
        and float(row.get("edge_vs_devig_market") or 0.0) > 0
        and row.get("book_consistency_cap") is not None
    ]
    candidates.sort(
        key=lambda row: (
            float(row.get("book_adjusted_buy_score") or 0.0),
            float(row.get("risk_adjusted_edge") or 0.0),
            float(row.get("edge_vs_devig_market") or 0.0),
        ),
        reverse=True,
    )
    candidates = candidates[:limit]
    if not candidates:
        return

    labels = [str(row.get("scoreline", "")) for row in candidates]
    y_positions = list(range(len(candidates)))
    prices = [float(row.get("raw_yes_price") or row.get("market_implied_probability") or 0.0) * 100 for row in candidates]
    caps = [float(row.get("book_consistency_cap") or 0.0) * 100 for row in candidates]
    fair = [float(row.get("risk_adjusted_probability") or row.get("market_total_tilt_probability") or 0.0) * 100 for row in candidates]

    fig_h = max(5.8, 0.50 * len(candidates) + 1.8)
    fig, ax = plt.subplots(figsize=(12.8, fig_h))
    bar_h = 0.23
    ax.barh([y - bar_h for y in y_positions], prices, height=bar_h, color="#F58518", label="Exact price")
    ax.barh(y_positions, caps, height=bar_h, color="#9ECAE1", label="Tightest book cap")
    ax.barh([y + bar_h for y in y_positions], fair, height=bar_h, color="#54A24B", label="Risk-adjusted fair")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Probability / price (%)")
    ax.set_title(f"{team_a} vs {team_b}: exact-score book consistency", fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    x_max = max(prices + caps + fair + [1.0])
    ax.set_xlim(0, x_max * 1.48)

    for idx, row in enumerate(candidates):
        cap_source = str(row.get("book_consistency_cap_source", ""))
        margin = float(row.get("book_consistency_margin") or 0.0)
        verdict = str(row.get("book_adjusted_verdict", ""))
        flags = str(row.get("book_consistency_flags", ""))
        if flags:
            detail = f"conflict: {flags}"
            color = "#D62728"
        else:
            detail = f"{verdict} | slack {margin*100:+.1f}pp | cap {cap_source}"
            color = "#1B9E77" if "buy" in verdict else "#555555"
        ax.text(max(prices[idx], caps[idx], fair[idx]) + x_max * 0.025, idx, detail, va="center", ha="left", fontsize=8.3, color=color)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_market_implied_matrix_plot(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    team_a: str,
    team_b: str,
    limit: int = 14,
) -> None:
    candidates = [
        row
        for row in rows
        if row.get("market_implied_score_probability") is not None
        and not row.get("is_any_other_score")
    ]
    candidates.sort(
        key=lambda row: (
            float(row.get("risk_adjusted_edge") or 0.0),
            float(row.get("edge_vs_devig_market") or 0.0),
            float(row.get("market_total_tilt_probability") or 0.0),
        ),
        reverse=True,
    )
    candidates = candidates[:limit]
    if not candidates:
        return

    labels = [str(row.get("scoreline", "")) for row in candidates]
    y_positions = list(range(len(candidates)))
    our_probs = [float(row.get("market_total_tilt_probability") or 0.0) * 100.0 for row in candidates]
    market_matrix = [float(row.get("market_implied_score_probability") or 0.0) * 100.0 for row in candidates]
    exact_prices = [float(row.get("raw_yes_price") or row.get("market_implied_probability") or 0.0) * 100.0 for row in candidates]

    fig_h = max(6.0, 0.50 * len(candidates) + 1.8)
    fig, ax = plt.subplots(figsize=(13.0, fig_h))
    bar_h = 0.23
    ax.barh([y - bar_h for y in y_positions], our_probs, height=bar_h, color="#54A24B", label="Our fair")
    ax.barh(y_positions, market_matrix, height=bar_h, color="#4C78A8", label="Market-implied matrix")
    ax.barh([y + bar_h for y in y_positions], exact_prices, height=bar_h, color="#F58518", label="Exact-score price")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Probability / price (%)")
    ax.set_title(f"{team_a} vs {team_b}: full market-implied score matrix", fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    x_max = max(our_probs + market_matrix + exact_prices + [1.0])
    ax.set_xlim(0, x_max * 1.45)

    for idx, row in enumerate(candidates):
        our_edge = row.get("our_vs_market_matrix_edge")
        posted_edge = row.get("posted_vs_market_matrix_edge")
        confidence = row.get("market_matrix_confidence")
        text = (
            f"our-mkt {float(our_edge or 0.0)*100:+.1f}pp | "
            f"price-mkt {float(posted_edge or 0.0)*100:+.1f}pp | "
            f"conf {float(confidence or 0.0)*100:.0f}%"
        )
        color = "#1B9E77" if float(row.get("risk_adjusted_edge") or 0.0) > 0 else "#555555"
        ax.text(max(our_probs[idx], market_matrix[idx], exact_prices[idx]) + x_max * 0.025, idx, text, va="center", ha="left", fontsize=8.2, color=color)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_merged_decision_plot(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    team_a: str,
    team_b: str,
    limit: int = 14,
) -> None:
    candidates = [
        row
        for row in rows
        if _to_float(row.get("edge_vs_devig_market"), 0.0) is not None
        and not row.get("is_any_other_score")
    ]
    candidates.sort(
        key=lambda row: (
            float(row.get("risk_adjusted_edge") or 0.0),
            float(row.get("edge_vs_devig_market") or 0.0),
            float(row.get("scoreline_fair_probability") or 0.0),
        ),
        reverse=True,
    )
    candidates = candidates[:limit]
    if not candidates:
        return


    bucket_short = {
        "validated_total_le_3": "T<=3",
        "low_total_lopsided": "lop",
        "high_total_4_discounted": "T4",
        "tail_total_5_plus_heavily_discounted": "T5+",
        "any_other_bucket": "other",
    }
    labels = [
        f"#{int(row['model_scoreline_rank'])} {row.get('scoreline', '')}"
        if row.get("model_scoreline_rank") not in (None, "")
        else str(row.get("scoreline", ""))
        for row in candidates
    ]
    y_positions = list(range(len(candidates)))
    fair = [float(row.get("scoreline_fair_probability") or 0.0) * 100.0 for row in candidates]
    market_matrix = [float(row.get("market_implied_score_probability") or 0.0) * 100.0 for row in candidates]
    exact_price = [float(row.get("raw_yes_price") or row.get("market_implied_probability") or 0.0) * 100.0 for row in candidates]
    adjusted_edge = [float(row.get("risk_adjusted_edge") or 0.0) * 100.0 for row in candidates]

    fig_h = max(6.2, 0.52 * len(candidates) + 1.8)
    fig, (prob_ax, edge_ax, detail_ax) = plt.subplots(
        ncols=3,
        figsize=(17.0, fig_h),
        gridspec_kw={"width_ratios": [1.35, 0.78, 1.18], "wspace": 0.08},
    )
    bar_h = 0.22
    prob_ax.barh([y - bar_h for y in y_positions], fair, height=bar_h, color="#54A24B", label="Our fair")
    prob_ax.barh(y_positions, market_matrix, height=bar_h, color="#4C78A8", label="Market matrix")
    prob_ax.barh([y + bar_h for y in y_positions], exact_price, height=bar_h, color="#F58518", label="Exact price")
    prob_ax.set_yticks(y_positions)
    prob_ax.set_yticklabels(labels)
    prob_ax.invert_yaxis()
    prob_ax.set_xlabel("Probability / price (%)")
    prob_ax.set_title("Fair vs Market", fontweight="bold")
    prob_ax.grid(axis="x", alpha=0.22)
    prob_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False, fontsize=8)
    prob_xmax = max(fair + market_matrix + exact_price + [1.0])
    prob_ax.set_xlim(0, prob_xmax * 1.12)

    edge_colors = ["#1B9E77" if value > 0 else "#D62728" for value in adjusted_edge]
    edge_ax.barh(y_positions, adjusted_edge, height=0.42, color=edge_colors)
    edge_ax.set_yticks(y_positions)
    edge_ax.set_yticklabels([])
    edge_ax.invert_yaxis()
    edge_ax.axvline(0, color="#333333", linewidth=0.9)
    edge_ax.set_xlabel("Adj edge pp")
    edge_ax.set_title("Buy Edge", fontweight="bold")
    edge_ax.grid(axis="x", alpha=0.22)
    edge_abs = max([abs(v) for v in adjusted_edge] + [1.0])
    edge_ax.set_xlim(-edge_abs * 0.18, edge_abs * 1.18)

    detail_ax.set_ylim(prob_ax.get_ylim())
    detail_ax.set_xlim(0, 1)
    detail_ax.axis("off")
    columns = [
        ("Conf", 0.02),
        ("1/4K", 0.16),
        ("Our-Mkt", 0.30),
        ("Book", 0.47),
        ("Tag", 0.66),
    ]
    header_y = -0.72
    for name, xpos in columns:
        detail_ax.text(xpos, header_y, name, va="center", ha="left", fontsize=8.4, fontweight="bold", color="#333333")
    detail_ax.hlines(-0.45, 0.0, 1.0, color="#DDDDDD", linewidth=0.8)

    for idx, row in enumerate(candidates):
        confidence = float(row.get("staking_confidence") or 0.0)
        q_kelly = float(row.get("confidence_quarter_joint_kelly_fraction") or 0.0)
        our_mkt = row.get("our_vs_market_matrix_edge")
        book = str(row.get("book_adjusted_verdict") or row.get("buy_verdict") or "")
        bucket = bucket_short.get(str(row.get("selection_bucket", "")), str(row.get("selection_bucket", "")))
        verdict = str(row.get("buy_verdict", ""))
        color = "#1B9E77" if verdict == "buy" else "#B36B00" if "watch" in verdict else "#555555"
        values = [
            f"{confidence*100:.0f}%",
            f"{q_kelly*100:.1f}%",
            "--" if our_mkt is None else f"{float(our_mkt)*100:+.1f}pp",
            book.replace("_", " "),
            f"{bucket} / {verdict}",
        ]
        for (_name, xpos), value in zip(columns, values):
            detail_ax.text(xpos, idx, value, va="center", ha="left", fontsize=8.0, color=color)

    fig.suptitle(f"{team_a} vs {team_b}: merged exact-score decision view", fontweight="bold", y=0.985)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


POLL_VALUE_FLAGS = {
    "--poll-interval-minutes",
    "--poll-runs",
    "--poll-output-root",
}
POLL_BOOLEAN_FLAGS = {
    "--poll-rebuild-seed",
    "--poll-stop-on-error",
}
POLL_CHILD_VALUE_FLAGS = {
    "--outdir",
    "--price-history-root",
}
POLL_CHILD_BOOLEAN_FLAGS = {
    "--refresh-polymarket-only",
    "--reuse-existing-prediction",
}


def slug_safe(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def format_remaining(seconds: float) -> str:
    if seconds >= 60.0:
        return f"{seconds / 60.0:.1f} minutes"
    return f"{max(0.0, seconds):.0f} seconds"


def sleep_with_countdown(seconds: float, *, update_seconds: float = 60.0) -> None:
    if seconds <= 0:
        print("Next v42 refresh starting now.", flush=True)
        return

    deadline = time.monotonic() + seconds
    next_refresh_wall = datetime.fromtimestamp(time.time() + seconds).astimezone().strftime("%H:%M:%S %Z")
    print(f"Next v42 refresh at {next_refresh_wall} ({format_remaining(seconds)} from now).", flush=True)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(update_seconds, remaining))
        remaining = deadline - time.monotonic()
        if remaining > 0:
            print(f"Next v42 refresh in {format_remaining(remaining)}.", flush=True)
    print("Next v42 refresh starting now.", flush=True)


def poll_child_argv(raw_argv: list[str]) -> list[str]:
    child: list[str] = []
    skip_next = False
    for item in raw_argv:
        if skip_next:
            skip_next = False
            continue
        item_key = item.split("=", 1)[0]
        if item_key in POLL_VALUE_FLAGS or item_key in POLL_CHILD_VALUE_FLAGS:
            if "=" not in item:
                skip_next = True
            continue
        if item_key in POLL_BOOLEAN_FLAGS or item_key in POLL_CHILD_BOOLEAN_FLAGS:
            continue
        child.append(item)
    return child


def has_any_market_source(args: argparse.Namespace) -> bool:
    return bool(
        args.polymarket_query
        or args.polymarket_event_slug
        or args.polymarket_sports_url
        or args.polymarket_json
        or args.auto_polymarket
        or args.no_fetch_polymarket
    )


def run_v42_child(child_args: list[str], *, outdir: Path, output_root: Path, refresh_only: bool) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        *child_args,
        "--outdir",
        str(outdir),
        "--price-history-root",
        str(output_root),
    ]
    if refresh_only:
        cmd.append("--refresh-polymarket-only")
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(PROJECT_DIR), check=True)


def run_poll_mode(args: argparse.Namespace, raw_argv: list[str]) -> None:
    if args.poll_interval_minutes is None:
        return
    if args.poll_interval_minutes <= 0:
        raise ValueError("--poll-interval-minutes must be positive")

    child_args = poll_child_argv(raw_argv)
    if not has_any_market_source(args):
        child_args.append("--auto-polymarket")

    match_name = f"{slug_safe(args.team_a)}_{slug_safe(args.team_b)}"
    source_slug = slug_safe(args.polymarket_event_slug or args.polymarket_query or "auto_polymarket")
    output_root = (
        Path(args.poll_output_root).expanduser().resolve()
        if args.poll_output_root
        else PROJECT_DIR / "outputs" / f"v42_poll_{match_name}_{source_slug}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    seed_dir = output_root / "_seed"
    seed_prediction = seed_dir / "single_match_prediction.json"
    supplied_seed_prediction = Path(args.outdir).expanduser().resolve() / "single_match_prediction.json"
    if (
        not args.poll_rebuild_seed
        and not seed_prediction.exists()
        and supplied_seed_prediction.exists()
        and supplied_seed_prediction != seed_prediction
    ):
        seed_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(supplied_seed_prediction, seed_prediction)
        print(f"Seeded poll mode from existing prediction: {supplied_seed_prediction}", flush=True)
    if args.poll_rebuild_seed or not seed_prediction.exists():
        run_v42_child(child_args, outdir=seed_dir, output_root=output_root, refresh_only=False)
    else:
        print(f"Reusing seed prediction: {seed_prediction}", flush=True)

    runs = int(args.poll_runs or 0)
    interval_seconds = float(args.poll_interval_minutes) * 60.0
    print(
        f"Polling v42 every {args.poll_interval_minutes:g} minutes for {args.team_a} vs {args.team_b}. "
        f"Output root: {output_root}",
        flush=True,
    )

    run_index = 0
    while runs <= 0 or run_index < runs:
        run_index += 1
        run_dir = output_root / f"run_{timestamp_slug()}"
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(seed_prediction, run_dir / "single_match_prediction.json")
        started = time.monotonic()
        try:
            run_v42_child(child_args, outdir=run_dir, output_root=output_root, refresh_only=True)
            print(f"Completed v42 poll refresh {run_index}: {run_dir}", flush=True)
        except Exception as exc:
            print(f"v42 poll refresh {run_index} failed: {exc}", file=sys.stderr, flush=True)
            if args.poll_stop_on_error:
                raise
        if runs > 0 and run_index >= runs:
            break
        sleep_with_countdown(max(0.0, interval_seconds - (time.monotonic() - started)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V42 FotMob-aware Polymarket exact-score edge detector.")
    parser.add_argument("--team-a", required=True)
    parser.add_argument("--team-b", required=True)
    parser.add_argument("--host-a", action="store_true")
    parser.add_argument("--host-b", action="store_true")
    parser.add_argument("--knockout", action="store_true")
    parser.add_argument("--outdir", default="outputs/outputs_v42_fotmob_market_edge")
    parser.add_argument("--worldcupsai-zip", default=str(DATA_DIR / "worldcupsai.zip"))
    parser.add_argument("--team-train", default=str(DATA_DIR / "current_team_features_2026.csv"))
    parser.add_argument("--team-test")
    parser.add_argument("--box-data", default=str(DATA_DIR / "FIFAallMatchBoxData.csv"))
    parser.add_argument("--results-data", default=str(DATA_DIR / "results.csv"))
    parser.add_argument("--results-as-of", default=v15.DEFAULT_RESULTS_AS_OF)
    parser.add_argument("--former-names", default=str(DATA_DIR / "former_names.csv"))
    parser.add_argument("--prediction-year", type=int, default=2026)
    parser.add_argument("--player-ratings", default=str(DATA_DIR / "player_ratings_international.csv"))
    parser.add_argument("--declared-squads", default=str(DATA_DIR / "world_cup_2026_declared_squads.csv"))
    parser.add_argument("--fcratings", default=str(DATA_DIR / "fcratings_top50_worldcup2026.csv"))
    parser.add_argument("--observed-matches")
    parser.add_argument("--fotmob-leaders", default=str(DATA_DIR / "fotmob_full_stat_tables_clean.csv"))
    parser.add_argument("--fotmob-player-stats", default=str(DATA_DIR / "fotmob_match_player_stats_clean.csv"))
    parser.add_argument("--fotmob-lineups", default=str(DATA_DIR / "fotmob_match_lineups_clean.csv"))
    parser.add_argument("--fotmob-substitutions", default=str(DATA_DIR / "fotmob_match_substitutions_clean.csv"))
    parser.add_argument("--fotmob-keeper-stats", default=str(DATA_DIR / "fotmob_match_keeper_stats_clean.csv"))
    parser.add_argument("--fotmob-match-facts", default=str(DATA_DIR / "fotmob_match_facts_clean.csv"))
    parser.add_argument("--fotmob-goal-events", default=str(DATA_DIR / "fotmob_match_goal_events_clean.csv"))
    parser.add_argument(
        "--fotmob-wdl-blend",
        type=float,
        default=0.20,
        help="Blend V36 FotMob current-form WDL probabilities into the base result probabilities.",
    )
    parser.add_argument("--fotmob-scoreline-blend", type=float, default=v36.DEFAULT_FOTMOB_SCORELINE_BLEND)
    parser.add_argument("--fotmob-max-log-adjustment", type=float, default=v36.DEFAULT_MAX_LOG_ADJUSTMENT)
    parser.add_argument(
        "--betterdata-profile-csv",
        default=str(PROJECT_DIR / "analysis" / "v39_withbetterdata_latest" / "v39_withbetterdata_team_profiles.csv"),
        help="Team-profile priors from v39_withbetterdata.",
    )
    parser.add_argument("--betterdata-scoreline-blend", type=float, default=v39bd.DEFAULT_BETTERDATA_SCORELINE_BLEND)
    parser.add_argument("--betterdata-wdl-blend", type=float, default=v39bd.DEFAULT_BETTERDATA_WDL_BLEND)
    parser.add_argument("--betterdata-max-log-adjustment", type=float, default=v39bd.DEFAULT_BETTERDATA_MAX_LOG_ADJUSTMENT)
    parser.add_argument(
        "--disable-betterdata",
        action="store_true",
        help="Fall back to the previous v42 model stack without v39_withbetterdata priors.",
    )
    parser.add_argument("--polymarket-query")
    parser.add_argument(
        "--polymarket-event-slug",
        help="Polymarket event slug or /event/{slug} URL, e.g. the World Cup page.",
    )
    parser.add_argument(
        "--polymarket-sports-url",
        help="Polymarket sports game URL or /sports/world-cup/games listing URL.",
    )
    parser.add_argument(
        "--auto-polymarket",
        action="store_true",
        help="Use the Polymarket World Cup games page automatically when no market source is supplied.",
    )
    parser.add_argument("--polymarket-json")
    parser.add_argument("--no-fetch-polymarket", action="store_true")
    parser.add_argument(
        "--reuse-existing-prediction",
        action="store_true",
        help="If OUTDIR already has single_match_prediction.json, reuse it and only refresh market outputs.",
    )
    parser.add_argument(
        "--refresh-polymarket-only",
        action="store_true",
        help="Require OUTDIR/single_match_prediction.json and refresh Polymarket odds/plots without rebuilding the model.",
    )
    parser.add_argument("--gamma-limit", type=int, default=DEFAULT_GAMMA_LIMIT)
    parser.add_argument("--min-edge", type=float, default=DEFAULT_MIN_EXACT_EDGE)
    parser.add_argument("--min-ev", type=float, default=DEFAULT_MIN_EV)
    parser.add_argument("--uncertainty-buffer", type=float, default=DEFAULT_UNCERTAINTY_BUFFER)
    parser.add_argument(
        "--reference-v43-output",
        help="Optional v43 output directory or CSV. Adds scoreline-by-scoreline v42/v43 agreement columns.",
    )
    parser.add_argument(
        "--fetch-clob-orderbook",
        action="store_true",
        help="Fetch CLOB best bid/ask for exact-score YES token IDs and add executable ask diagnostics.",
    )
    parser.add_argument(
        "--no-clob-orderbook",
        action="store_false",
        dest="fetch_clob_orderbook",
        help="Disable CLOB orderbook fetching; default unless --fetch-clob-orderbook is supplied.",
    )
    parser.add_argument(
        "--price-history-root",
        default="outputs",
        help="Root to scan for previous price_snapshot.csv files. Use empty string to disable.",
    )
    parser.add_argument(
        "--plot-set",
        choices=["all", "decision"],
        default="all",
        help="Use 'decision' to write only the compact betting/refresh plots.",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--poll-interval-minutes",
        type=float,
        help="Run v42 repeatedly on this interval. Omit for a normal one-shot run.",
    )
    parser.add_argument(
        "--poll-runs",
        type=int,
        default=0,
        help="Number of timed refreshes to run. Use 0 to keep polling until interrupted.",
    )
    parser.add_argument(
        "--poll-output-root",
        help="Root directory for _seed and timestamped poll refresh folders.",
    )
    parser.add_argument(
        "--poll-rebuild-seed",
        action="store_true",
        help="Rebuild the seed model prediction before starting timed refreshes.",
    )
    parser.add_argument(
        "--poll-stop-on-error",
        action="store_true",
        help="Stop polling on the first failed refresh instead of logging and continuing.",
    )
    args = parser.parse_args()

    if (
        args.auto_polymarket
        and not args.polymarket_sports_url
        and not args.polymarket_event_slug
        and not args.polymarket_query
        and not args.polymarket_json
        and not args.no_fetch_polymarket
    ):
        args.polymarket_sports_url = DEFAULT_POLYMARKET_SPORTS_GAMES_URL

    if args.poll_interval_minutes is not None:
        run_poll_mode(args, sys.argv[1:])
        return

    requested_output_dir = Path(args.outdir)
    existing_prediction_path = requested_output_dir / "single_match_prediction.json"
    should_reuse_prediction = args.refresh_polymarket_only or (
        args.reuse_existing_prediction and existing_prediction_path.exists()
    )

    if should_reuse_prediction:
        output_dir = requested_output_dir
        if not existing_prediction_path.exists():
            raise FileNotFoundError(
                f"{existing_prediction_path} is required for --refresh-polymarket-only"
            )
        prediction = json.loads(existing_prediction_path.read_text(encoding="utf-8"))
    else:
        base_model, _ = v36.build_from_zip(
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
            fotmob_wdl_blend=args.fotmob_wdl_blend,
            fotmob_scoreline_blend=args.fotmob_scoreline_blend,
            max_log_adjustment=args.fotmob_max_log_adjustment,
        )
        if args.disable_betterdata:
            model = V42FotmobCoverageModel(base_model, coverage_margin=DEFAULT_COVERAGE_MARGIN)
        else:
            betterdata_profile = Path(args.betterdata_profile_csv)
            if not betterdata_profile.exists():
                v39bd.run_analysis()
            v39_base = v39.V39CoverageOutlierModel(
                base_model,
                coverage_margin=DEFAULT_COVERAGE_MARGIN,
                observed_game_state_priors=v39.default_observed_game_state_priors(),
                use_observed_game_state_priors=True,
            )
            model = v39bd.V39BetterDataModel(
                v39_base,
                better_priors=v39bd.BetterDataPriors(betterdata_profile),
                betterdata_scoreline_blend=args.betterdata_scoreline_blend,
                betterdata_wdl_blend=args.betterdata_wdl_blend,
                max_abs_log_adjustment=args.betterdata_max_log_adjustment,
            )
        output_dir = v11.unique_output_dir(args.outdir)
        prediction = model.predict(
            args.team_a,
            args.team_b,
            host_a=args.host_a,
            host_b=args.host_b,
            knockout=args.knockout,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    base_matrix = normalize_matrix(score_matrix_from_prediction(prediction))

    raw_markets, fetch_meta = load_or_fetch_markets(
        team_a=args.team_a,
        team_b=args.team_b,
        query=args.polymarket_query,
        event_slug=args.polymarket_event_slug,
        sports_url=args.polymarket_sports_url,
        json_path=args.polymarket_json,
        no_fetch=args.no_fetch_polymarket,
        limit=args.gamma_limit,
    )
    (output_dir / "polymarket_raw_markets.json").write_text(
        json.dumps(raw_markets, indent=2),
        encoding="utf-8",
    )
    match_markets = filter_match_markets(raw_markets, args.team_a, args.team_b)
    moneyline_markets = [
        row
        for row in (classify_moneyline_market(market, args.team_a, args.team_b) for market in match_markets)
        if row
    ]
    btts_markets = [row for row in (classify_btts_market(market) for market in match_markets) if row]
    spread_markets = [
        row
        for row in (classify_spread_market(market, args.team_a, args.team_b) for market in match_markets)
        if row
    ]
    total_markets = [row for row in (classify_total_market(market) for market in match_markets) if row]
    exact_markets = [
        row
        for row in (classify_exact_score_market(market, args.team_a, args.team_b) for market in match_markets)
        if row
    ]
    total_ladder = monotone_over_ladder(total_markets)
    market_distribution_rows = market_total_distribution(total_ladder)

    fit = None
    market_total_reference_matrix = base_matrix
    total_comparison_rows: list[dict[str, Any]] = []
    if len(market_distribution_rows) >= 3:
        fit = fit_total_tilt_beta(base_matrix, market_distribution_rows)
        market_total_reference_matrix = tilt_matrix_by_beta(base_matrix, float(fit["beta"]))
        model_totals = aggregate_total_buckets(base_matrix, market_distribution_rows)
        tilted_totals = aggregate_total_buckets(market_total_reference_matrix, market_distribution_rows)
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

    book_context = build_book_context(moneyline_markets, btts_markets, spread_markets, market_distribution_rows)
    market_implied_matrix: ScoreMatrix | None = None
    market_matrix_meta: dict[str, Any] | None = None
    if market_distribution_rows and (
        moneyline_markets or btts_markets or spread_markets
    ):
        market_implied_matrix, market_matrix_meta = build_market_implied_score_matrix(
            market_total_reference_matrix,
            book_context,
        )
    exact_edges = edge_rows(
        base_matrix,
        base_matrix,
        exact_markets,
        min_edge=args.min_edge,
        min_ev=args.min_ev,
        uncertainty_buffer=args.uncertainty_buffer,
        market_implied_matrix=market_implied_matrix,
        market_matrix_meta=market_matrix_meta,
        market_total_reference_matrix=market_total_reference_matrix,
    )
    exact_edges = add_book_consistency(exact_edges, book_context)
    exact_edges = add_staking_confidence(exact_edges)
    event_slug = current_event_slug(fetch_meta) or infer_event_slug_from_exact_rows(exact_edges)
    exact_edges = add_entry_price_columns(exact_edges, args.uncertainty_buffer)
    current_exact_edges = current_exact_score_rows(exact_edges, event_slug)
    previous_price_rows, price_history_meta = find_previous_price_rows(
        args.price_history_root,
        current_output_dir=output_dir,
        event_slug=event_slug,
    )
    exact_edges = add_price_movement_columns(exact_edges, previous_price_rows)
    current_exact_edges = current_exact_score_rows(exact_edges, event_slug)
    reference_rows = load_reference_model_rows(
        args.reference_v43_output,
        team_a=args.team_a,
        team_b=args.team_b,
    )
    if reference_rows:
        exact_edges = add_reference_model_agreement(exact_edges, reference_rows, args.uncertainty_buffer)
        current_exact_edges = current_exact_score_rows(exact_edges, event_slug)
    if args.fetch_clob_orderbook:
        current_exact_edges = add_clob_quotes(current_exact_edges, args.uncertainty_buffer)
    exact_score_baskets = build_exact_score_baskets(
        current_exact_edges,
        team_a=args.team_a,
        team_b=args.team_b,
        uncertainty_buffer=args.uncertainty_buffer,
    )
    current_exact_edges_with_other = current_exact_score_rows(exact_edges, event_slug, include_any_other=True)
    exact_score_decision_board = build_decision_board_rows(current_exact_edges_with_other)
    price_movement_rows = [
        {
            "scoreline": row.get("scoreline"),
            "slug": row.get("slug"),
            "previous_raw_yes_price": row.get("previous_raw_yes_price"),
            "current_raw_yes_price": row.get("raw_yes_price"),
            "raw_yes_price_change": row.get("raw_yes_price_change"),
            "previous_edge_after_buffer": row.get("previous_edge_after_buffer"),
            "current_edge_after_buffer": row.get("edge_vs_raw_yes_after_buffer"),
            "edge_after_buffer_change": row.get("edge_after_buffer_change"),
            "previous_call": row.get("previous_call"),
            "current_call": row.get("decision_tier"),
            "still_good": row.get("still_good_status"),
            "recommended_action": row.get("recommended_action"),
            "movement_tag": row.get("movement_tag"),
        }
        for row in current_exact_edges
    ]
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
    market_matrix_rows = [
        {
            "team_a_goals": key[0],
            "team_b_goals": key[1],
            "scoreline": f"{key[0]}-{key[1]}",
            "market_implied_score_probability": market_implied_matrix.get(key, 0.0),
            "our_model_only_fair_probability": base_matrix.get(key, 0.0),
            "polymarket_total_tilt_reference_probability": market_total_reference_matrix.get(key, 0.0),
            "base_model_probability": base_matrix.get(key, 0.0),
            "market_minus_our_probability": market_implied_matrix.get(key, 0.0) - base_matrix.get(key, 0.0),
        }
        for key in sorted(market_implied_matrix or {}, key=lambda item: (market_implied_matrix or {}).get(item, 0.0), reverse=True)
    ]

    write_csv(output_dir / "polymarket_total_ladder.csv", total_ladder)
    write_csv(output_dir / "polymarket_market_total_distribution.csv", market_distribution_rows)
    write_csv(output_dir / "model_vs_market_total_distribution.csv", total_comparison_rows)
    write_csv(output_dir / "polymarket_exact_score_edges.csv", exact_edges)
    write_csv(output_dir / "exact_score_decision_board.csv", exact_score_decision_board)
    write_csv(output_dir / "exact_score_baskets.csv", exact_score_baskets)
    write_csv(output_dir / "price_movement.csv", price_movement_rows)
    write_price_snapshot(
        output_dir / "price_snapshot.csv",
        current_exact_edges,
        event_slug=event_slug,
        team_a=args.team_a,
        team_b=args.team_b,
    )
    write_decision_brief(
        output_dir / "decision_brief.md",
        team_a=args.team_a,
        team_b=args.team_b,
        event_slug=event_slug,
        decision_rows=exact_score_decision_board,
        basket_rows=exact_score_baskets,
        market_matrix_meta=market_matrix_meta,
        price_history_meta=price_history_meta,
    )
    write_csv(output_dir / "model_fair_scoreline_probabilities.csv", fair_score_rows)
    write_csv(output_dir / "market_total_tilt_scoreline_probabilities.csv", fair_score_rows)
    write_csv(output_dir / "market_implied_scoreline_matrix.csv", market_matrix_rows)
    write_csv(
        output_dir / "polymarket_markets_classified.csv",
        [
            {
                "kind": "total",
                **row,
            }
            for row in total_markets
        ]
        + [
            {
                "kind": "moneyline",
                **row,
            }
            for row in moneyline_markets
        ]
        + [
            {
                "kind": "both_teams_to_score",
                **row,
            }
            for row in btts_markets
        ]
        + [
            {
                "kind": "spread",
                **row,
            }
            for row in spread_markets
        ]
        + [
            {
                "kind": "exact_score",
                **row,
            }
            for row in exact_markets
        ],
    )
    (output_dir / "single_match_prediction.json").write_text(json.dumps(prediction, indent=2), encoding="utf-8")

    book_consistency_plot_path = plots_dir / "polymarket_book_consistency.png"
    market_implied_matrix_plot_path = plots_dir / "market_implied_scoreline_matrix.png"
    merged_decision_plot_path = plots_dir / "polymarket_merged_decision_view.png"
    decision_board_plot_path = plots_dir / "exact_score_decision_board.png"
    action_delta_plot_path = plots_dir / "exact_score_action_delta.png"
    basket_value_plot_path = plots_dir / "exact_score_basket_value.png"
    exact_score_value_map_plot_path = plots_dir / "exact_score_value_map.png"
    if not args.no_plots:
        if args.plot_set == "all":
            v11.plot_prediction_outputs(prediction, output_dir)
            v28.plot_top3_scorelines(prediction, plots_dir)
            v35.plot_top3_plus_game_state_outlier(prediction, output_dir)
            if total_comparison_rows:
                write_total_svg(
                    plots_dir / "market_vs_model_total_distribution.svg",
                    total_comparison_rows,
                    f"{args.team_a} vs {args.team_b}: total goals",
                )
        if exact_edges:
            if args.plot_set == "all":
                write_edge_svg(
                    plots_dir / "polymarket_exact_score_edges.svg",
                    exact_edges,
                    f"{args.team_a} vs {args.team_b}: exact-score edge",
                )
                write_buy_candidate_plot(
                    plots_dir / "polymarket_buy_candidates_adjusted.png",
                    exact_edges,
                    team_a=args.team_a,
                    team_b=args.team_b,
                )
                book_consistency_plot_path = unique_artifact_path(book_consistency_plot_path)
                write_book_consistency_plot(
                    book_consistency_plot_path,
                    exact_edges,
                    team_a=args.team_a,
                    team_b=args.team_b,
                )
                write_merged_decision_plot(
                    merged_decision_plot_path,
                    exact_edges,
                    team_a=args.team_a,
                    team_b=args.team_b,
                )
            market_implied_matrix_plot_path = unique_artifact_path(market_implied_matrix_plot_path)
            write_market_implied_matrix_plot(
                market_implied_matrix_plot_path,
                exact_edges,
                team_a=args.team_a,
                team_b=args.team_b,
            )
            write_decision_board_plot(
                decision_board_plot_path,
                exact_score_decision_board,
                exact_score_baskets,
                team_a=args.team_a,
                team_b=args.team_b,
            )
            write_action_delta_plot(
                action_delta_plot_path,
                exact_score_decision_board,
                team_a=args.team_a,
                team_b=args.team_b,
            )
            write_basket_value_plot(
                basket_value_plot_path,
                exact_score_baskets,
                team_a=args.team_a,
                team_b=args.team_b,
            )
            write_exact_score_value_map_plot(
                exact_score_value_map_plot_path,
                exact_score_decision_board,
                team_a=args.team_a,
                team_b=args.team_b,
            )
        if exact_markets and args.plot_set == "all":
            write_top_scorelines_market_svg(
                plots_dir / "top_scorelines_with_polymarket_prices.png",
                base_matrix=base_matrix,
                fair_matrix=base_matrix,
                exact_markets=exact_markets,
                team_a=args.team_a,
                team_b=args.team_b,
            )

    summary = {
        "version": "v42-fotmob-market-edge",
        "base_model": (
            "v36-fotmob-current-form-plus-v39-withbetterdata"
            if prediction.get("v39_withbetterdata_adjustments")
            else "v36-fotmob-current-form-plus-v39-coverage-outlier"
        ),
        "team_a": args.team_a,
        "team_b": args.team_b,
        "prediction_top_3": prediction.get("top_scorelines", [])[:3],
        "prediction_outlier": prediction.get("coverage_total_outlier"),
        "result_probabilities": prediction.get("result_probabilities"),
        "betterdata_enabled": bool(prediction.get("v39_withbetterdata_adjustments")),
        "betterdata_profile_csv": args.betterdata_profile_csv,
        "v39_withbetterdata_adjustments": prediction.get("v39_withbetterdata_adjustments", {}),
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
            "min_edge": args.min_edge,
            "min_ev": args.min_ev,
            "uncertainty_buffer": args.uncertainty_buffer,
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
            "price_movement_csv": str(output_dir / "price_movement.csv"),
            "decision_brief_md": str(output_dir / "decision_brief.md"),
            "model_fair_scorelines_csv": str(output_dir / "model_fair_scoreline_probabilities.csv"),
            "fair_scorelines_csv": str(output_dir / "market_total_tilt_scoreline_probabilities.csv"),
            "market_implied_matrix_csv": str(output_dir / "market_implied_scoreline_matrix.csv"),
            "top_scorelines_market_plot": str(plots_dir / "top_scorelines_with_polymarket_prices.png"),
            "buy_candidates_adjusted_plot": str(plots_dir / "polymarket_buy_candidates_adjusted.png"),
            "book_consistency_plot": str(book_consistency_plot_path),
            "market_implied_matrix_plot": str(market_implied_matrix_plot_path),
            "merged_decision_plot": str(merged_decision_plot_path),
            "decision_board_plot": str(decision_board_plot_path),
            "action_delta_plot": str(action_delta_plot_path),
            "basket_value_plot": str(basket_value_plot_path),
            "exact_score_value_map_plot": str(exact_score_value_map_plot_path),
        },
    }
    (output_dir / "model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
'''
v42_fotmob_market_edge_model = _load_submodule("v42_fotmob_market_edge_model", _V42_FOTMOB_MARKET_EDGE_MODEL_SOURCE, "market_edge.py:v42_fotmob_market_edge_model")

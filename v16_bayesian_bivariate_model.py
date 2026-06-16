#!/usr/bin/env python3
"""V16: PyMC hierarchy, forward calibration, and bivariate scorelines."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable

_DEFAULT_PYTENSOR_CACHE = (
    Path(__file__).resolve().parent / "data" / "v16_pytensor_cache"
)
os.environ.setdefault(
    "PYTENSOR_FLAGS",
    f"base_compiledir={_DEFAULT_PYTENSOR_CACHE}",
)

import numpy as np
import pandas as pd
import arviz as az
import pymc as pm
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln

import v11_wcq_results_model as v11
import v15_catboost_model as v15


canon_team = v11.canon_team

DEFAULT_BAYES_GOAL_BLEND_GRID = tuple(
    step / 100.0 for step in range(0, 71, 5)
)
DEFAULT_GROUP_DRAW_SCALE_GRID = (1.0, 1.05, 1.10, 1.15, 1.20, 1.25)
SCORE_CALIBRATION_SCHEMA = "v16.2-top2-blend70-draw-scale-v1"

HIERARCHY_FEATURES = (
    "bayes_attack_a",
    "bayes_defense_a",
    "bayes_attack_b",
    "bayes_defense_b",
    "bayes_attack_diff",
    "bayes_defense_diff",
    "bayes_log_goal_a",
    "bayes_log_goal_b",
    "bayes_log_goal_diff",
)


def require_pymc() -> None:
    if pm is None:
        raise RuntimeError(
            "V16 requires PyMC. Install it with "
            "`.venv/bin/pip install pymc arviz`."
        )


def normalize_matrix(
    matrix: Dict[tuple[int, int], float],
) -> Dict[tuple[int, int], float]:
    clean = {
        key: max(float(value), 0.0)
        for key, value in matrix.items()
        if math.isfinite(float(value))
    }
    total = float(sum(clean.values()))
    if total <= 0:
        raise ValueError("Score matrix has no positive finite probability")
    normalized = {key: value / total for key, value in clean.items()}
    if abs(sum(normalized.values()) - 1.0) >= 1e-10:
        raise AssertionError("Score matrix failed normalization")
    return normalized


def bivariate_poisson_score_matrix(
    mean_a: float,
    mean_b: float,
    covariance: float,
    max_goals: int = 10,
) -> Dict[tuple[int, int], float]:
    """Karlis-Ntzoufras bivariate Poisson preserving marginal means."""
    covariance = float(
        np.clip(covariance, 0.0, min(mean_a, mean_b) - 1e-6)
    )
    lambda_1 = max(float(mean_a) - covariance, 1e-6)
    lambda_2 = max(float(mean_b) - covariance, 1e-6)
    lambda_3 = max(covariance, 0.0)
    matrix: Dict[tuple[int, int], float] = {}
    common_exp = -(lambda_1 + lambda_2 + lambda_3)
    for goals_a in range(max_goals + 1):
        for goals_b in range(max_goals + 1):
            log_terms = []
            for shared in range(min(goals_a, goals_b) + 1):
                log_term = (
                    (goals_a - shared) * math.log(lambda_1)
                    - gammaln(goals_a - shared + 1)
                    + (goals_b - shared) * math.log(lambda_2)
                    - gammaln(goals_b - shared + 1)
                )
                if shared:
                    if lambda_3 <= 0:
                        continue
                    log_term += (
                        shared * math.log(lambda_3)
                        - gammaln(shared + 1)
                    )
                log_terms.append(log_term)
            maximum = max(log_terms)
            probability = math.exp(common_exp + maximum) * sum(
                math.exp(term - maximum) for term in log_terms
            )
            matrix[(goals_a, goals_b)] = probability
    return normalize_matrix(matrix)


def apply_zero_inflation(
    matrix: Dict[tuple[int, int], float],
    probability: float,
) -> Dict[tuple[int, int], float]:
    probability = float(np.clip(probability, 0.0, 0.30))
    adjusted = {
        key: (1.0 - probability) * value
        for key, value in matrix.items()
    }
    adjusted[(0, 0)] = adjusted.get((0, 0), 0.0) + probability
    return normalize_matrix(adjusted)


def apply_draw_score_scaling(
    matrix: Dict[tuple[int, int], float],
    scale: float,
) -> Dict[tuple[int, int], float]:
    scale = max(float(scale), 0.0)
    adjusted = {
        key: value * scale if key[0] == key[1] else value
        for key, value in matrix.items()
    }
    return normalize_matrix(adjusted)


def temperature_scale(
    probabilities: Dict[str, float],
    temperature: float,
) -> Dict[str, float]:
    labels = ("team_a_win", "draw", "team_b_win")
    logits = np.log(
        np.clip(
            [probabilities[label] for label in labels],
            1e-12,
            1.0,
        )
    )
    scaled = np.exp(logits / max(float(temperature), 1e-6))
    scaled = scaled / scaled.sum()
    return dict(zip(labels, scaled.astype(float)))


def _hierarchy_cache_path(
    cache_dir: Path,
    cutoff: pd.Timestamp,
    lookback_years: int,
    draws: int,
    tune: int,
    chains: int,
    recency_half_life_years: float,
) -> Path:
    stamp = cutoff.strftime("%Y%m%d")
    half_life = str(float(recency_half_life_years)).replace(".", "p")
    return cache_dir / (
        f"hierarchy_v3_weighted_{stamp}_lb{lookback_years}_d{draws}_"
        f"t{tune}_c{chains}_rh{half_life}.json"
    )


def fit_pymc_hierarchy(
    results: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    cache_dir: str | Path,
    lookback_years: int = 12,
    draws: int = 800,
    tune: int = 800,
    chains: int = 4,
    target_accept: float = 0.92,
    seed: int = 16,
    recency_half_life_years: float = 16.0,
) -> Dict[str, Any]:
    """Fit and cache a genuine NUTS hierarchical attack/defence model."""
    require_pymc()
    cutoff = pd.Timestamp(cutoff)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _hierarchy_cache_path(
        cache_dir,
        cutoff,
        lookback_years,
        draws,
        tune,
        chains,
        recency_half_life_years,
    )
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    start = cutoff - pd.DateOffset(years=lookback_years)
    sample = results[
        (results["date"] < cutoff) & (results["date"] >= start)
    ].copy()
    if len(sample) < 100:
        raise ValueError(
            f"Only {len(sample)} matches available before {cutoff.date()}"
        )

    teams = sorted(set(sample["team_a"]) | set(sample["team_b"]))
    team_index = {team: index for index, team in enumerate(teams)}
    home_index = sample["team_a"].map(team_index).to_numpy(dtype=int)
    away_index = sample["team_b"].map(team_index).to_numpy(dtype=int)
    goals_a = sample["goals_a"].to_numpy(dtype=int)
    goals_b = sample["goals_b"].to_numpy(dtype=int)
    non_neutral = (~sample["neutral"].astype(bool)).to_numpy(dtype=float)
    ages = (
        (cutoff - sample["date"]).dt.days.to_numpy(dtype=float) / 365.25
    )
    recency_weights = np.power(
        0.5,
        ages / max(float(recency_half_life_years), 1e-6),
    )
    prestige_weights = pd.to_numeric(
        sample["prestige_weight"],
        errors="coerce",
    ).fillna(0.25).to_numpy(dtype=float)
    match_weights = np.clip(
        prestige_weights * recency_weights,
        0.01,
        None,
    )
    match_weights = match_weights / match_weights.mean()
    coordinates = {
        "team": teams,
        "match": np.arange(len(sample)),
    }

    with pm.Model(coords=coordinates) as model:
        home_team = pm.Data("home_team", home_index, dims="match")
        away_team = pm.Data("away_team", away_index, dims="match")
        venue = pm.Data("non_neutral", non_neutral, dims="match")
        likelihood_weight = pm.Data(
            "likelihood_weight",
            match_weights,
            dims="match",
        )
        sigma_attack = pm.HalfNormal("sigma_attack", sigma=0.5)
        sigma_defense = pm.HalfNormal("sigma_defense", sigma=0.5)
        attack_offset = pm.Normal(
            "attack_offset",
            mu=0.0,
            sigma=1.0,
            dims="team",
        )
        defense_offset = pm.Normal(
            "defense_offset",
            mu=0.0,
            sigma=1.0,
            dims="team",
        )
        attack_raw = sigma_attack * attack_offset
        defense_raw = sigma_defense * defense_offset
        attack = pm.Deterministic(
            "attack",
            attack_raw - pm.math.mean(attack_raw),
            dims="team",
        )
        defense = pm.Deterministic(
            "defense",
            defense_raw - pm.math.mean(defense_raw),
            dims="team",
        )
        mu = pm.Normal("mu", mu=math.log(1.25), sigma=0.5)
        home_advantage = pm.Normal(
            "home_advantage",
            mu=0.10,
            sigma=0.15,
        )
        rate_a = pm.math.exp(
            mu
            + attack[home_team]
            - defense[away_team]
            + home_advantage * venue
        )
        rate_b = pm.math.exp(
            mu + attack[away_team] - defense[home_team]
        )
        pm.Potential(
            "weighted_goals_a",
            pm.math.sum(
                likelihood_weight
                * pm.logp(pm.Poisson.dist(mu=rate_a), goals_a)
            ),
        )
        pm.Potential(
            "weighted_goals_b",
            pm.math.sum(
                likelihood_weight
                * pm.logp(pm.Poisson.dist(mu=rate_b), goals_b)
            ),
        )
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=min(chains, 4),
            target_accept=target_accept,
            random_seed=seed,
            progressbar=True,
            return_inferencedata=True,
        )

    posterior = trace.posterior
    attack_mean = posterior["attack"].mean(("chain", "draw")).to_numpy()
    defense_mean = posterior["defense"].mean(("chain", "draw")).to_numpy()
    diagnostics = az.summary(
        trace,
        var_names=[
            "mu",
            "home_advantage",
            "sigma_attack",
            "sigma_defense",
            "attack",
            "defense",
        ],
        kind="diagnostics",
    )
    profile = {
        "cutoff": str(cutoff.date()),
        "lookback_years": int(lookback_years),
        "recency_half_life_years": float(recency_half_life_years),
        "matches": int(len(sample)),
        "teams": int(len(teams)),
        "draws": int(draws),
        "tune": int(tune),
        "chains": int(chains),
        "mu": float(posterior["mu"].mean()),
        "home_advantage": float(posterior["home_advantage"].mean()),
        "sigma_attack": float(posterior["sigma_attack"].mean()),
        "sigma_defense": float(posterior["sigma_defense"].mean()),
        "likelihood_weights": {
            "normalized_min": float(match_weights.min()),
            "normalized_max": float(match_weights.max()),
            "normalized_mean": float(match_weights.mean()),
            "prestige_min": float(prestige_weights.min()),
            "prestige_max": float(prestige_weights.max()),
        },
        "diagnostics": {
            "max_rhat": float(diagnostics["r_hat"].max()),
            "min_ess_bulk": float(diagnostics["ess_bulk"].min()),
            "min_ess_tail": float(diagnostics["ess_tail"].min()),
            "divergences": int(
                trace.sample_stats["diverging"].sum().to_numpy()
            ),
        },
        "team_profiles": {
            team: {
                "attack": float(attack_mean[index]),
                "defense": float(defense_mean[index]),
            }
            for index, team in enumerate(teams)
        },
    }
    cache_path.write_text(json.dumps(profile, indent=2))
    return profile


def build_hierarchy_snapshots(
    results: pd.DataFrame,
    snapshot_dates: Iterable[pd.Timestamp],
    **fit_kwargs,
) -> list[tuple[pd.Timestamp, Dict[str, Any]]]:
    snapshots = []
    for snapshot_date in sorted(set(pd.Timestamp(date) for date in snapshot_dates)):
        snapshots.append(
            (
                snapshot_date,
                fit_pymc_hierarchy(
                    results,
                    snapshot_date,
                    **fit_kwargs,
                ),
            )
        )
    return snapshots


def hierarchy_pair_features(
    profile: Dict[str, Any] | None,
    team_a: str,
    team_b: str,
    host_a: bool = False,
    host_b: bool = False,
) -> Dict[str, float]:
    if not profile:
        return {column: 0.0 for column in HIERARCHY_FEATURES}
    teams = profile["team_profiles"]
    a = teams.get(v11.canon_team(team_a), {"attack": 0.0, "defense": 0.0})
    b = teams.get(v11.canon_team(team_b), {"attack": 0.0, "defense": 0.0})
    log_a = (
        float(profile["mu"])
        + float(a["attack"])
        - float(b["defense"])
        + float(profile["home_advantage"]) * float(host_a)
    )
    log_b = (
        float(profile["mu"])
        + float(b["attack"])
        - float(a["defense"])
        + float(profile["home_advantage"]) * float(host_b)
    )
    return {
        "bayes_attack_a": float(a["attack"]),
        "bayes_defense_a": float(a["defense"]),
        "bayes_attack_b": float(b["attack"]),
        "bayes_defense_b": float(b["defense"]),
        "bayes_attack_diff": float(a["attack"] - b["attack"]),
        "bayes_defense_diff": float(a["defense"] - b["defense"]),
        "bayes_log_goal_a": log_a,
        "bayes_log_goal_b": log_b,
        "bayes_log_goal_diff": log_a - log_b,
    }


def add_hierarchy_features(
    frame: pd.DataFrame,
    snapshots: list[tuple[pd.Timestamp, Dict[str, Any]]],
    feature_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    enriched = frame.copy()
    for column in HIERARCHY_FEATURES:
        enriched[column] = 0.0
    for index, row in enriched.iterrows():
        match_date = pd.Timestamp(row["date"])
        available = [
            profile
            for snapshot_date, profile in snapshots
            if snapshot_date <= match_date
        ]
        profile = available[-1] if available else None
        values = hierarchy_pair_features(
            profile,
            row["team_a"],
            row["team_b"],
            bool(row.get("host_a", False)),
            bool(row.get("host_b", False)),
        )
        for column, value in values.items():
            enriched.at[index, column] = value
    return enriched, [*feature_columns, *HIERARCHY_FEATURES]


def _weighted_predictions(models, X: pd.DataFrame) -> np.ndarray:
    predictions = []
    weights = []
    for _, model, weight in models:
        prediction = np.asarray(model.predict(X), dtype=float)
        if np.isfinite(prediction).all():
            predictions.append(prediction)
            weights.append(float(weight))
    normalized = np.asarray(weights, dtype=float)
    normalized = normalized / normalized.sum()
    return np.average(np.vstack(predictions), axis=0, weights=normalized)


def predict_rows_for_calibration(
    model: v15.V15CatBoostWorldCupModel,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    X = frame[model.feature_cols]
    lambda_a = np.clip(
        _weighted_predictions(model.goal_a_models, X),
        0.15,
        4.5,
    )
    lambda_b = np.clip(
        _weighted_predictions(model.goal_b_models, X),
        0.15,
        4.5,
    )
    diff = _weighted_predictions(model.goal_diff_models, X)
    rows = []
    for position, (_, source) in enumerate(frame.iterrows()):
        blended_a, blended_b = model._apply_goal_difference_blend(
            float(lambda_a[position]),
            float(lambda_b[position]),
            float(diff[position]),
        )
        v15_lambda_a = float(np.clip(blended_a, 0.15, 4.5))
        v15_lambda_b = float(np.clip(blended_b, 0.15, 4.5))
        bayes_lambda_a = v15_lambda_a
        bayes_lambda_b = v15_lambda_b
        if {
            "bayes_log_goal_a",
            "bayes_log_goal_b",
        }.issubset(frame.columns):
            bayes_lambda_a = math.exp(
                float(source["bayes_log_goal_a"])
            )
            bayes_lambda_b = math.exp(
                float(source["bayes_log_goal_b"])
            )
        features = X.iloc[[position]]
        matrix = v11.apply_dixon_coles_adjustment(
            v11.poisson_score_matrix(v15_lambda_a, v15_lambda_b),
            v15_lambda_a,
            v15_lambda_b,
            rho=model.dixon_coles_rho,
        )
        result = v11.result_probs(matrix)
        classifier = model._weighted_classification_prediction(
            model.result_models,
            features,
        )
        result = {
            key: 0.86 * result[key] + 0.14 * classifier[key]
            for key in result
        }
        result = temperature_scale(result, model.temperature)
        draw_signal = model._predict_draw_probability(features)
        draw_probability = float(
            np.clip(
                model.draw_model_weight * draw_signal
                + (1.0 - model.draw_model_weight) * result["draw"],
                0.05,
                0.55,
            )
        )
        non_draw = max(
            result["team_a_win"] + result["team_b_win"],
            1e-12,
        )
        result = {
            "team_a_win": (1.0 - draw_probability)
            * result["team_a_win"]
            / non_draw,
            "draw": draw_probability,
            "team_b_win": (1.0 - draw_probability)
            * result["team_b_win"]
            / non_draw,
        }
        rows.append(
            {
                "date": source["date"],
                "goals_a": int(source["goals_a"]),
                "goals_b": int(source["goals_b"]),
                "is_knockout": bool(source.get("is_knockout", False)),
                "v15_lambda_a": v15_lambda_a,
                "v15_lambda_b": v15_lambda_b,
                "bayes_lambda_a": bayes_lambda_a,
                "bayes_lambda_b": bayes_lambda_b,
                **result,
            }
        )
    return pd.DataFrame(rows)


def forward_chain_predictions(
    frame: pd.DataFrame,
    features: list[str],
    fold_years: Iterable[int] = (2006, 2010, 2014, 2018, 2022),
) -> pd.DataFrame:
    predictions = []
    years = pd.to_datetime(frame["date"]).dt.year
    for year in fold_years:
        train = frame[years < year].copy()
        test = frame[
            (years == year) & frame["training_source"].eq("world_cup")
        ].copy()
        if len(train) < 500 or test.empty:
            continue
        fold_model = v15.V15CatBoostWorldCupModel().fit(
            train,
            features,
            [],
            pd.DataFrame(columns=["team"]),
        )
        fold_predictions = predict_rows_for_calibration(
            fold_model,
            test,
        )
        fold_predictions["test_year"] = int(year)
        predictions.append(fold_predictions)
    if not predictions:
        raise ValueError("No forward calibration folds were available")
    return pd.concat(predictions, ignore_index=True)


def fit_temperature(oof: pd.DataFrame) -> float:
    labels = np.where(
        oof["goals_a"] > oof["goals_b"],
        0,
        np.where(oof["goals_a"] == oof["goals_b"], 1, 2),
    )
    probabilities = np.clip(
        oof[["team_a_win", "draw", "team_b_win"]].to_numpy(),
        1e-12,
        1.0,
    )

    def objective(temperature: float) -> float:
        logits = np.log(probabilities) / max(float(temperature), 1e-6)
        logits -= logits.max(axis=1, keepdims=True)
        scaled = np.exp(logits)
        scaled /= scaled.sum(axis=1, keepdims=True)
        return float(
            -np.log(scaled[np.arange(len(labels)), labels]).mean()
        )

    result = minimize_scalar(
        objective,
        bounds=(0.60, 2.50),
        method="bounded",
    )
    return float(result.x)


def blend_lambdas(
    row: pd.Series,
    bayes_goal_weight: float,
) -> tuple[float, float]:
    return (
        (1.0 - bayes_goal_weight) * float(row["v15_lambda_a"])
        + bayes_goal_weight * float(row["bayes_lambda_a"]),
        (1.0 - bayes_goal_weight) * float(row["v15_lambda_b"])
        + bayes_goal_weight * float(row["bayes_lambda_b"]),
    )


def apply_result_mass_correction(
    matrix: Dict[tuple[int, int], float],
    target: Dict[str, float],
    strength: float,
) -> Dict[tuple[int, int], float]:
    """Blend an unforced matrix with its fully W/D/L-reweighted version."""
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 1e-12:
        return normalize_matrix(matrix)
    forced = v11.reweight_score_matrix_to_results(matrix, target)
    blended = {
        key: (1.0 - strength) * matrix[key] + strength * forced[key]
        for key in matrix
    }
    return normalize_matrix(blended)


def score_parameter_loss(
    subset: pd.DataFrame,
    temperature: float,
    bayes_goal_weight: float,
    covariance: float,
    zero_inflation: float,
    draw_score_scale: float,
    result_reweight_strength: float,
) -> float:
    losses = []
    for _, row in subset.iterrows():
        matrix = score_matrix_for_row(
            row,
            temperature,
            bayes_goal_weight,
            covariance,
            zero_inflation,
            draw_score_scale,
            result_reweight_strength,
        )
        probability = matrix.get(
            (int(row["goals_a"]), int(row["goals_b"])),
            1e-12,
        )
        losses.append(-math.log(max(probability, 1e-12)))
    return float(np.mean(losses))


def score_matrix_for_row(
    row: pd.Series,
    temperature: float,
    bayes_goal_weight: float,
    covariance: float,
    zero_inflation: float,
    draw_score_scale: float,
    result_reweight_strength: float,
) -> Dict[tuple[int, int], float]:
    lambda_a, lambda_b = blend_lambdas(row, bayes_goal_weight)
    matrix = bivariate_poisson_score_matrix(
        lambda_a,
        lambda_b,
        covariance,
    )
    matrix = apply_zero_inflation(matrix, zero_inflation)
    matrix = apply_draw_score_scaling(matrix, draw_score_scale)
    target = temperature_scale(
        {
            "team_a_win": row["team_a_win"],
            "draw": row["draw"],
            "team_b_win": row["team_b_win"],
        },
        temperature,
    )
    return apply_result_mass_correction(
        matrix,
        target,
        result_reweight_strength,
    )


def score_parameter_metrics(
    subset: pd.DataFrame,
    temperature: float,
    bayes_goal_weight: float,
    covariance: float,
    zero_inflation: float,
    draw_score_scale: float,
    result_reweight_strength: float,
) -> Dict[str, float]:
    losses = []
    top2_hits = 0
    for _, row in subset.iterrows():
        actual = (int(row["goals_a"]), int(row["goals_b"]))
        matrix = score_matrix_for_row(
            row,
            temperature,
            bayes_goal_weight,
            covariance,
            zero_inflation,
            draw_score_scale,
            result_reweight_strength,
        )
        probability = matrix.get(actual, 1e-12)
        losses.append(-math.log(max(probability, 1e-12)))
        top_two = [
            score
            for score, _ in sorted(
                matrix.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:2]
        ]
        top2_hits += int(actual in top_two)
    return {
        "exact_score_log_loss": float(np.mean(losses)),
        "exact_score_top2_hits": int(top2_hits),
        "exact_score_top2_rate": float(top2_hits / len(subset))
        if len(subset)
        else 0.0,
    }


def fit_stage_score_parameters(
    oof: pd.DataFrame,
    temperature: float,
    bayes_goal_weight: float,
    group_draw_scale_grid: Iterable[
        float
    ] = DEFAULT_GROUP_DRAW_SCALE_GRID,
) -> Dict[str, Dict[str, float]]:
    group_draw_scale_grid = tuple(
        float(scale) for scale in group_draw_scale_grid
    )
    parameters = {}
    for stage, subset in (
        ("group", oof[~oof["is_knockout"]]),
        ("knockout", oof[oof["is_knockout"]]),
    ):
        if subset.empty:
            parameters[stage] = {
                "covariance": 0.10,
                "zero_inflation": 0.0,
                "draw_score_scale": 1.0,
                "result_reweight_strength": 0.0,
            }
            continue

        candidates = []
        def objective(values: np.ndarray) -> float:
            return score_parameter_loss(
                subset,
                temperature,
                bayes_goal_weight,
                covariance=float(values[0]),
                zero_inflation=float(values[1]),
                draw_score_scale=1.0,
                result_reweight_strength=float(values[2]),
            )

        for reweight_start in (0.0, 0.25, 1.0):
            result = minimize(
                objective,
                x0=np.array([0.08, 0.00, reweight_start]),
                bounds=((0.0, 0.45), (0.0, 0.08), (0.0, 1.0)),
                method="L-BFGS-B",
            )
            candidates.append((result, 1.0))
        result, _ = min(
            candidates,
            key=lambda candidate: candidate[0].fun,
        )
        draw_scale_losses = []
        draw_scales = (
            group_draw_scale_grid if stage == "group" else (1.0,)
        )
        for draw_score_scale in draw_scales:
            metrics = score_parameter_metrics(
                subset,
                temperature,
                bayes_goal_weight,
                covariance=float(result.x[0]),
                zero_inflation=float(result.x[1]),
                draw_score_scale=float(draw_score_scale),
                result_reweight_strength=float(result.x[2]),
            )
            draw_scale_losses.append(
                {
                    "draw_score_scale": float(draw_score_scale),
                    **metrics,
                }
            )
        selected_draw = max(
            draw_scale_losses,
            key=lambda row: (
                row["exact_score_top2_rate"],
                -row["exact_score_log_loss"],
            ),
        )
        parameters[stage] = {
            "covariance": float(result.x[0]),
            "zero_inflation": float(result.x[1]),
            "draw_score_scale": float(selected_draw["draw_score_scale"]),
            "result_reweight_strength": float(result.x[2]),
            "matches": int(len(subset)),
            "exact_score_log_loss": float(
                selected_draw["exact_score_log_loss"]
            ),
            "exact_score_top2_hits": int(
                selected_draw["exact_score_top2_hits"]
            ),
            "exact_score_top2_rate": float(
                selected_draw["exact_score_top2_rate"]
            ),
            "draw_scale_grid_results": draw_scale_losses,
        }
    return parameters


def tune_score_layer(
    oof: pd.DataFrame,
    temperature: float,
    blend_grid: Iterable[float] = DEFAULT_BAYES_GOAL_BLEND_GRID,
    group_draw_scale_grid: Iterable[
        float
    ] = DEFAULT_GROUP_DRAW_SCALE_GRID,
) -> Dict[str, Any]:
    blend_grid = tuple(float(weight) for weight in blend_grid)
    group_draw_scale_grid = tuple(
        float(scale) for scale in group_draw_scale_grid
    )
    rows = []
    fitted_parameters = {}
    for bayes_goal_weight in blend_grid:
        stage_parameters = fit_stage_score_parameters(
            oof,
            temperature,
            float(bayes_goal_weight),
            group_draw_scale_grid,
        )
        weighted_exact_loss = sum(
            stage_parameters[stage]["matches"]
            * stage_parameters[stage]["exact_score_log_loss"]
            for stage in ("group", "knockout")
        ) / len(oof)
        top2_hits = sum(
            stage_parameters[stage]["exact_score_top2_hits"]
            for stage in ("group", "knockout")
        )
        top2_rate = float(top2_hits / len(oof))
        goal_errors = []
        for _, row in oof.iterrows():
            lambda_a, lambda_b = blend_lambdas(
                row,
                float(bayes_goal_weight),
            )
            goal_errors.extend(
                [
                    lambda_a - float(row["goals_a"]),
                    lambda_b - float(row["goals_b"]),
                ]
            )
        goal_rmse = float(np.sqrt(np.mean(np.square(goal_errors))))
        rows.append(
            {
                "bayes_goal_weight": float(bayes_goal_weight),
                "goal_rmse": goal_rmse,
                "exact_score_log_loss": float(weighted_exact_loss),
                "exact_score_top2_hits": int(top2_hits),
                "exact_score_top2_rate": top2_rate,
                "group_draw_score_scale": float(
                    stage_parameters["group"]["draw_score_scale"]
                ),
                "group_exact_score_log_loss": float(
                    stage_parameters["group"]["exact_score_log_loss"]
                ),
                "group_exact_score_top2_rate": float(
                    stage_parameters["group"]["exact_score_top2_rate"]
                ),
                "knockout_exact_score_log_loss": float(
                    stage_parameters["knockout"][
                        "exact_score_log_loss"
                    ]
                ),
                "knockout_exact_score_top2_rate": float(
                    stage_parameters["knockout"][
                        "exact_score_top2_rate"
                    ]
                ),
            }
        )
        fitted_parameters[float(bayes_goal_weight)] = stage_parameters

    selected = max(
        rows,
        key=lambda row: (
            row["exact_score_top2_rate"],
            -row["exact_score_log_loss"],
        ),
    )
    best_goal = min(rows, key=lambda row: row["goal_rmse"])
    best_log_loss = min(rows, key=lambda row: row["exact_score_log_loss"])
    selected_weight = float(selected["bayes_goal_weight"])
    return {
        "selected_bayes_goal_weight": selected_weight,
        "score_selection_objective": "exact_score_top2_rate",
        "best_goal_rmse_weight": float(
            best_goal["bayes_goal_weight"]
        ),
        "best_exact_score_log_loss_weight": float(
            best_log_loss["bayes_goal_weight"]
        ),
        "blend_grid": [
            float(weight) for weight in blend_grid
        ],
        "group_draw_scale_grid": [
            float(scale) for scale in group_draw_scale_grid
        ],
        "stage_parameters": fitted_parameters[selected_weight],
        "grid_results": rows,
    }


def calibration_table(
    oof: pd.DataFrame,
    temperature: float,
    bins: int = 10,
) -> list[Dict[str, Any]]:
    rows = []
    for _, row in oof.iterrows():
        scaled = temperature_scale(
            {
                "team_a_win": row["team_a_win"],
                "draw": row["draw"],
                "team_b_win": row["team_b_win"],
            },
            temperature,
        )
        actual = (
            "team_a_win"
            if row["goals_a"] > row["goals_b"]
            else "draw"
            if row["goals_a"] == row["goals_b"]
            else "team_b_win"
        )
        for label, probability in scaled.items():
            rows.append(
                {
                    "label": label,
                    "probability": probability,
                    "observed": float(actual == label),
                }
            )
    long = pd.DataFrame(rows)
    long["bin"] = pd.cut(
        long["probability"],
        bins=np.linspace(0.0, 1.0, bins + 1),
        include_lowest=True,
    )
    table = (
        long.groupby(["label", "bin"], observed=True)
        .agg(
            predictions=("observed", "size"),
            mean_probability=("probability", "mean"),
            observed_rate=("observed", "mean"),
        )
        .reset_index()
    )
    table["bin"] = table["bin"].astype(str)
    return table.to_dict(orient="records")


class V16CatBoostWorldCupModel(v15.V15CatBoostWorldCupModel):
    """Inject current PyMC posterior means into V15's live feature row."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_hierarchy_profile: Dict[str, Any] | None = None

    def set_current_hierarchy_profile(self, profile: Dict[str, Any]):
        self.current_hierarchy_profile = profile
        return self

    def make_features(
        self,
        team_a,
        team_b,
        host_a=False,
        host_b=False,
        knockout=False,
    ):
        features = super().make_features(
            team_a,
            team_b,
            host_a,
            host_b,
            knockout,
        )
        values = hierarchy_pair_features(
            self.current_hierarchy_profile,
            team_a,
            team_b,
            host_a,
            host_b,
        )
        for column, value in values.items():
            if column in features:
                features.loc[:, column] = value
        return features


class V16BayesianBivariateModel(v15.V15CatBoostModel):
    def __init__(
        self,
        player_model: v15.V15CatBoostWorldCupModel,
        outcome_model: v15.V15CatBoostWorldCupModel,
        hierarchy_profile: Dict[str, Any],
        temperature: float,
        stage_parameters: Dict[str, Dict[str, float]],
        bayes_goal_weight: float = 0.20,
    ):
        super().__init__(player_model, outcome_model)
        self.hierarchy_profile = hierarchy_profile
        self.v16_temperature = float(temperature)
        self.stage_parameters = stage_parameters
        self.bayes_goal_weight = float(bayes_goal_weight)

    def predict(self, *args, **kwargs) -> Dict[str, Any]:
        team_a = kwargs.get("team_a", args[0] if args else "")
        team_b = kwargs.get("team_b", args[1] if len(args) > 1 else "")
        host_a = kwargs.get("host_a", args[2] if len(args) > 2 else False)
        host_b = kwargs.get("host_b", args[3] if len(args) > 3 else False)
        knockout = kwargs.get(
            "knockout",
            args[4] if len(args) > 4 else False,
        )
        max_goals = kwargs.get(
            "max_goals",
            args[5] if len(args) > 5 else 10,
        )
        prediction = super().predict(*args, **kwargs)
        hierarchy = hierarchy_pair_features(
            self.hierarchy_profile,
            team_a,
            team_b,
            host_a,
            host_b,
        )
        bayes_lambda_a = float(math.exp(hierarchy["bayes_log_goal_a"]))
        bayes_lambda_b = float(math.exp(hierarchy["bayes_log_goal_b"]))
        lambda_a = (
            (1.0 - self.bayes_goal_weight) * prediction["lambda_a"]
            + self.bayes_goal_weight * bayes_lambda_a
        )
        lambda_b = (
            (1.0 - self.bayes_goal_weight) * prediction["lambda_b"]
            + self.bayes_goal_weight * bayes_lambda_b
        )
        results = temperature_scale(
            prediction["result_probabilities"],
            self.v16_temperature,
        )
        stage = "knockout" if knockout else "group"
        stage_parameters = self.stage_parameters[stage]
        matrix = bivariate_poisson_score_matrix(
            lambda_a,
            lambda_b,
            stage_parameters["covariance"],
            max_goals,
        )
        matrix = apply_zero_inflation(
            matrix,
            stage_parameters["zero_inflation"],
        )
        matrix = apply_draw_score_scaling(
            matrix,
            stage_parameters.get("draw_score_scale", 1.0),
        )
        matrix = apply_result_mass_correction(
            matrix,
            results,
            stage_parameters["result_reweight_strength"],
        )
        matrix_sum = float(sum(matrix.values()))
        if abs(matrix_sum - 1.0) >= 1e-6:
            raise AssertionError(
                f"V16 score matrix sums to {matrix_sum:.12f}"
            )

        prediction["lambda_a"] = float(lambda_a)
        prediction["lambda_b"] = float(lambda_b)
        prediction["score_matrix_result_probabilities"] = v11.result_probs(
            matrix
        )
        prediction["result_probabilities"] = results
        prediction["predicted_result"] = max(
            prediction["result_probabilities"],
            key=prediction["result_probabilities"].get,
        )
        prediction.update(v15.score_outputs(matrix, max_goals))
        prediction["hierarchical_team_strength"] = {
            "team_a": {
                "attack": hierarchy["bayes_attack_a"],
                "defense": hierarchy["bayes_defense_a"],
                "expected_goals": bayes_lambda_a,
            },
            "team_b": {
                "attack": hierarchy["bayes_attack_b"],
                "defense": hierarchy["bayes_defense_b"],
                "expected_goals": bayes_lambda_b,
            },
        }
        prediction["v16_adjustments"] = {
            "wdl_temperature": self.v16_temperature,
            "bayesian_goal_weight": self.bayes_goal_weight,
            "bivariate_covariance": stage_parameters["covariance"],
            "zero_inflation": stage_parameters["zero_inflation"],
            "draw_score_scale": stage_parameters.get(
                "draw_score_scale",
                1.0,
            ),
            "result_reweight_strength": stage_parameters[
                "result_reweight_strength"
            ],
            "stage_parameter_set": stage,
            "matrix_sum": matrix_sum,
            "hierarchy_cutoff": self.hierarchy_profile["cutoff"],
            "hierarchy_matches": self.hierarchy_profile["matches"],
            "hierarchy_teams": self.hierarchy_profile["teams"],
            "hierarchy_diagnostics": self.hierarchy_profile.get(
                "diagnostics",
                {},
            ),
            "pymc_sampler": "NUTS",
        }
        prediction["calibration_notes"]["v16"] = prediction[
            "v16_adjustments"
        ]
        prediction["calibration_notes"]["hybrid_model_policy"] = (
            "V16 retains V15 CatBoost, player, continental, and live-Elo "
            "layers; blends genuine PyMC hierarchical expected goals; then "
            "uses forward-calibrated W/D/L probabilities and a stage-specific "
            "bivariate Poisson score matrix whose W/D/L mass correction is "
            "itself selected by forward validation."
        )
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
    results_as_of="2026-06-10",
    pymc_draws=800,
    pymc_tune=800,
    pymc_chains=4,
    pymc_lookback_years=12,
    pymc_recency_half_life_years=16.0,
    pymc_cache_dir=None,
    calibration_cache=None,
    bayes_goal_weight=None,
):
    require_pymc()
    data_dir = Path(__file__).resolve().parent / "data"
    cache_dir = Path(
        pymc_cache_dir or data_dir / "v16_pymc_cache"
    )
    loader = v11.WorldCupSAILoader(
        zip_path,
        Path(str(zip_path) + "_extracted"),
    )
    matches = loader.load_matches()
    current = v11.load_current_team_features(train_csv, test_csv)
    box = v11.load_kaggle_box_data(box_csv)
    qualification_results = v11.load_world_cup_qualification_results(
        results_csv,
        former_names_csv,
    )
    qualifier_source = (
        qualification_results if not qualification_results.empty else box
    )
    frame, _, events = v11.build_rolling_features(
        matches,
        pd.DataFrame(columns=["team"]),
        qualifier_box=qualifier_source,
        qualifier_fallback_box=box,
        qualifier_blend_start_year=qualifier_blend_start_year,
        qualifier_full_weight_year=qualifier_full_weight_year,
        qualifier_minimum_influence=qualifier_minimum_influence,
    )
    international_results = v15.load_international_results(
        results_csv,
        former_names_csv=former_names_csv,
        as_of=results_as_of,
    )
    timeline, international_state = v15.build_international_timeline(
        international_results
    )
    expanded_frame, expanded_features, expansion_summary = (
        v15.build_expanded_training_frame(frame, timeline)
    )

    historical_years = [
        year
        for year in (2002, 2006, 2010, 2014, 2018, 2022, 2026)
        if year <= prediction_year
    ]
    historical_dates = [pd.Timestamp(year=year, month=1, day=1) for year in historical_years]
    live_cutoff = pd.Timestamp(results_as_of) + pd.Timedelta(days=1)
    snapshot_dates = [*historical_dates, live_cutoff]
    hierarchy_fit_kwargs = {
        "cache_dir": cache_dir,
        "lookback_years": pymc_lookback_years,
        "draws": pymc_draws,
        "tune": pymc_tune,
        "chains": pymc_chains,
        "recency_half_life_years": pymc_recency_half_life_years,
    }
    snapshots = build_hierarchy_snapshots(
        international_results,
        snapshot_dates,
        **hierarchy_fit_kwargs,
    )
    hierarchy_frame, hierarchy_features = add_hierarchy_features(
        expanded_frame,
        snapshots,
        expanded_features,
    )
    live_hierarchy = snapshots[-1][1]

    player_ratings_csv = player_ratings_csv or (
        data_dir / "player_ratings_international.csv"
    )
    declared_squads_csv = declared_squads_csv or (
        data_dir / "world_cup_2026_declared_squads.csv"
    )
    player_ratings = v15.load_player_ratings(player_ratings_csv)
    declared_squads = v15.load_declared_squads(declared_squads_csv)
    current_squad_profiles = v15.build_current_squad_profiles(
        declared_squads,
        player_ratings,
    )
    player_frame, player_features = v15.add_historical_player_features(
        hierarchy_frame,
        player_ratings,
        hierarchy_features,
    )

    calibration_path = Path(
        calibration_cache
        or cache_dir / "v16_2_forward_calibration.json"
    )
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text())
        cached_grid = calibration.get("blend_grid")
        cached_draw_grid = calibration.get("group_draw_scale_grid")
        expected_grid = list(DEFAULT_BAYES_GOAL_BLEND_GRID)
        expected_draw_grid = list(DEFAULT_GROUP_DRAW_SCALE_GRID)
        if (
            calibration.get("score_calibration_schema")
            != SCORE_CALIBRATION_SCHEMA
            or "selected_bayes_goal_weight" not in calibration
            or cached_grid != expected_grid
            or cached_draw_grid != expected_draw_grid
        ):
            calibration = {}
    else:
        calibration = {}
    if not calibration:
        oof = forward_chain_predictions(
            hierarchy_frame,
            hierarchy_features,
        )
        temperature = fit_temperature(oof)
        score_tuning = tune_score_layer(oof, temperature)
        calibration = {
            "score_calibration_schema": SCORE_CALIBRATION_SCHEMA,
            "temperature": temperature,
            **score_tuning,
            "fold_years": sorted(
                int(year) for year in oof["test_year"].unique()
            ),
            "matches": int(len(oof)),
            "calibration_table": calibration_table(oof, temperature),
        }
        calibration_path.parent.mkdir(parents=True, exist_ok=True)
        calibration_path.write_text(json.dumps(calibration, indent=2))

    selected_bayes_goal_weight = (
        float(bayes_goal_weight)
        if bayes_goal_weight is not None
        else float(calibration["selected_bayes_goal_weight"])
    )

    outcome_model = (
        V16CatBoostWorldCupModel(
            recency_half_life_years=recency_half_life_years,
            recency_min_weight=recency_min_weight,
        )
        .fit(hierarchy_frame, hierarchy_features, [], current)
        .set_box_data(box)
        .set_qualifier_data(
            qualifier_source,
            fallback_box=box,
            prediction_year=prediction_year,
            blend_start_year=qualifier_blend_start_year,
            full_weight_year=qualifier_full_weight_year,
            minimum_influence=qualifier_minimum_influence,
        )
        .set_current_international_state(
            international_state,
            results_as_of,
        )
        .set_current_hierarchy_profile(live_hierarchy)
    )
    player_model = (
        V16CatBoostWorldCupModel(
            recency_half_life_years=recency_half_life_years,
            recency_min_weight=recency_min_weight,
        )
        .fit(player_frame, player_features, events, current)
        .set_box_data(box)
        .set_qualifier_data(
            qualifier_source,
            fallback_box=box,
            prediction_year=prediction_year,
            blend_start_year=qualifier_blend_start_year,
            full_weight_year=qualifier_full_weight_year,
            minimum_influence=qualifier_minimum_influence,
        )
        .set_current_squad_profiles(current_squad_profiles)
        .set_current_international_state(
            international_state,
            results_as_of,
        )
        .set_current_hierarchy_profile(live_hierarchy)
    )
    model = V16BayesianBivariateModel(
        player_model,
        outcome_model,
        live_hierarchy,
        calibration["temperature"],
        calibration["stage_parameters"],
        bayes_goal_weight=selected_bayes_goal_weight,
    )
    model.training_data_summary = {
        **expansion_summary,
        "results_as_of": str(pd.Timestamp(results_as_of).date()),
        "pymc_hierarchy": {
            "snapshots": [str(date.date()) for date, _ in snapshots],
            "draws": pymc_draws,
            "tune": pymc_tune,
            "chains": pymc_chains,
            "lookback_years": pymc_lookback_years,
            "recency_half_life_years": (
                pymc_recency_half_life_years
            ),
            "cache_dir": str(cache_dir),
            "snapshot_diagnostics": {
                str(date.date()): profile.get("diagnostics", {})
                for date, profile in snapshots
            },
        },
        "forward_calibration": calibration,
    }
    data = v11.DataBundle(
        matches=matches,
        team_current=current,
        training_frame=player_frame,
        event_columns=events,
        box_frame=box,
    )
    return model, data


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    parser = argparse.ArgumentParser(
        description=(
            "Run V16 with PyMC hierarchy, forward calibration, and "
            "bivariate Poisson exact scores."
        )
    )
    parser.add_argument("--team-a", required=True)
    parser.add_argument("--team-b", required=True)
    parser.add_argument("--host-a", action="store_true")
    parser.add_argument("--host-b", action="store_true")
    parser.add_argument("--knockout", action="store_true")
    parser.add_argument("--outdir", default="outputs/outputs_v16_prediction")
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
    parser.add_argument(
        "--former-names",
        default=str(data_dir / "former_names.csv"),
    )
    parser.add_argument(
        "--player-ratings",
        default=str(data_dir / "player_ratings_international.csv"),
    )
    parser.add_argument(
        "--declared-squads",
        default=str(data_dir / "world_cup_2026_declared_squads.csv"),
    )
    parser.add_argument("--results-as-of", default="2026-06-10")
    parser.add_argument("--pymc-draws", type=int, default=800)
    parser.add_argument("--pymc-tune", type=int, default=800)
    parser.add_argument("--pymc-chains", type=int, default=4)
    parser.add_argument("--pymc-lookback-years", type=int, default=12)
    parser.add_argument(
        "--pymc-recency-half-life-years",
        type=float,
        default=16.0,
    )
    parser.add_argument("--pymc-cache-dir")
    parser.add_argument("--calibration-cache")
    parser.add_argument(
        "--bayes-goal-weight",
        type=float,
        help=(
            "Override the forward-CV-selected Bayesian goal blend. "
            "By default V16.1 uses the held-out optimum."
        ),
    )
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
        player_ratings_csv=args.player_ratings,
        declared_squads_csv=args.declared_squads,
        results_as_of=args.results_as_of,
        pymc_draws=args.pymc_draws,
        pymc_tune=args.pymc_tune,
        pymc_chains=args.pymc_chains,
        pymc_lookback_years=args.pymc_lookback_years,
        pymc_recency_half_life_years=(
            args.pymc_recency_half_life_years
        ),
        pymc_cache_dir=args.pymc_cache_dir,
        calibration_cache=args.calibration_cache,
        bayes_goal_weight=args.bayes_goal_weight,
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
                "version": "v16.2-pymc-bivariate-top2-tuned",
                "team_a": prediction["team_a"],
                "team_b": prediction["team_b"],
                "training_data": model.training_data_summary,
                "v16_adjustments": prediction["v16_adjustments"],
            },
            indent=2,
        )
    )
    if not args.no_plots:
        v11.plot_prediction_outputs(prediction, output_dir)
    print(
        json.dumps(
            {
                "result_probabilities": prediction[
                    "result_probabilities"
                ],
                "predicted_result": prediction["predicted_result"],
                "lambda_a": prediction["lambda_a"],
                "lambda_b": prediction["lambda_b"],
                "top_scorelines": prediction["top_scorelines"][:5],
                "v16_adjustments": prediction["v16_adjustments"],
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

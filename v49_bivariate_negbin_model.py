"""V49: true bivariate Negative Binomial scoreline model.

Standalone module. Does not import or modify v11/v39/v42/v43/v46 -- it can be
dropped into any of those pipelines later by calling `score_matrix_from_prediction`
or `build_score_matrix`, but for now it only reads/writes its own outputs.

Why this exists
----------------
The pipeline that currently produces buy cards (v11 -> v39 -> v42 -> v46_4) builds
scorelines from *independent* Poisson marginals and then patches only the
(0,0)/(1,0)/(0,1)/(1,1) corner with a Dixon-Coles tau term. Every other cell --
every 2-3, 3-1, 4-2, etc. -- is exactly what independent Poisson says it is, with
zero overdispersion and zero cross-team correlation. v43 already swaps in
independent Negative Binomial marginals (fixing each team's own overdispersion),
but the two teams are still combined as pa[i] * pb[j]: no correlation, so a
"both teams get dragged into a chaotic, high-event match together" scenario still
isn't modeled anywhere in the tails.

This module fixes both gaps at once with a shared-frailty construction:

    Z ~ Gamma(shape=r, scale=1/r)      # E[Z] = 1, Var[Z] = 1/r ("match volatility")
    A | Z ~ Poisson(lambda_a * Z)
    B | Z ~ Poisson(lambda_b * Z)      # conditionally independent given Z

Both teams' goal counts share the same draw of Z, so a volatile match (large Z)
pushes both teams' scoring up together, and a shut-down match (small Z) pushes
both down together. Marginalizing Z out analytically (it's a standard
Poisson-Gamma mixture) gives a closed-form bivariate Negative Binomial --
sometimes called a "negative trinomial" -- with no numerical integration needed:

    P(A=i, B=j) = [Gamma(i+j+r) / (i! j! Gamma(r))] * theta^r * p_a^i * p_b^j

    where theta = r / (lambda_a + lambda_b + r)
          p_a   = lambda_a / (lambda_a + lambda_b + r)
          p_b   = lambda_b / (lambda_a + lambda_b + r)

This recovers NB marginals (Var(A) = lambda_a + lambda_a^2 / r, matching v43's
per-team overdispersion form when r == alpha) *and* a genuine positive
covariance Cov(A, B) = lambda_a * lambda_b / r across the whole grid, not just
a hand-patched 2x2 corner. As r -> inf, Var[Z] -> 0 and this collapses back to
plain independent Poisson; smaller r means more shared volatility and fatter,
more correlated tails in both directions (high-combined-score blowouts and
low-scoring mutual shutdowns).

Dixon-Coles is layered on top, unchanged in form, because it corrects a
different, well-documented effect (tactical behavior at low scores, e.g. sides
playing for a 1-1) that shared-frailty overdispersion doesn't capture.

Calibration of `r`
-------------------
DEFAULT_R = 25.0 is fit from data, not guessed -- see calibrate_v49_dispersion.py
and its output at outputs/v49_dispersion_calibration/calibration_summary.json.
That script fits a standard multiplicative attack/defense Poisson model
(attack_i * defense_j * home_advantage) to data/results.csv (2010-present,
~15k matches) to get a match-specific lambda_home/lambda_away, then estimates
`r` from the *residuals* against those lambdas -- naively plugging raw
(mean, variance) of goals scored into implied_r_from_moments would conflate
between-team strength heterogeneity with genuine overdispersion and imply a
much smaller (fatter-tailed) r than the data supports.

On the World Cup finals subset specifically (n=260, the actual target
domain -- qualifiers/friendlies include lopsided mismatches that behave
differently): r_marginal_pooled=27.4 and r_correlation=26.6 agree closely;
r_marginal_away_only=9.1 is noisier (n=130) and close to the old guessed
default of 8.0, which is probably a coincidence rather than validation, since
home-goals overdispersion comes out essentially infinite (Poisson-like, no
excess variance once attack/defense is controlled for) in both the finals
subset and the broader 2010-present sample across all competitions. r=25
follows the two numbers that agree with each other.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

ScoreKey = tuple[int, int]
ScoreMatrix = dict[ScoreKey, float]

DEFAULT_R = 25.0
DEFAULT_DC_RHO = -0.08


def normalize_matrix(matrix: ScoreMatrix) -> ScoreMatrix:
    total = sum(matrix.values())
    if total <= 0:
        return matrix
    return {k: v / total for k, v in matrix.items()}


def _log_pmf_shift(i: int, j: int, r: float) -> float:
    return (
        math.lgamma(i + j + r)
        - math.lgamma(i + 1)
        - math.lgamma(j + 1)
        - math.lgamma(r)
    )


def bivariate_negbin_matrix(
    lambda_a: float,
    lambda_b: float,
    *,
    r: float = DEFAULT_R,
    max_goals: int = 10,
) -> ScoreMatrix:
    """Shared-gamma-frailty bivariate Negative Binomial scoreline grid.

    `r` is the shared dispersion shape: r -> inf recovers independent Poisson;
    small r produces fat, positively-correlated tails in both directions.
    """
    lambda_a = max(float(lambda_a), 1e-9)
    lambda_b = max(float(lambda_b), 1e-9)
    r = max(float(r), 1e-6)
    denom = lambda_a + lambda_b + r
    log_theta = math.log(r) - math.log(denom)
    log_pa = math.log(lambda_a) - math.log(denom)
    log_pb = math.log(lambda_b) - math.log(denom)

    matrix: ScoreMatrix = {}
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            log_p = _log_pmf_shift(i, j, r) + r * log_theta + i * log_pa + j * log_pb
            matrix[(i, j)] = math.exp(log_p)
    return normalize_matrix(matrix)


def apply_dixon_coles_adjustment(
    score_probs: ScoreMatrix,
    lambda_a: float,
    lambda_b: float,
    rho: float = DEFAULT_DC_RHO,
) -> ScoreMatrix:
    """Low-score tactical correction, layered on top of the bivariate NB grid."""
    adjusted = dict(score_probs)
    for (i, j), p in score_probs.items():
        if (i, j) == (0, 0):
            tau = 1.0 - lambda_a * lambda_b * rho
        elif (i, j) == (0, 1):
            tau = 1.0 + lambda_a * rho
        elif (i, j) == (1, 0):
            tau = 1.0 + lambda_b * rho
        elif (i, j) == (1, 1):
            tau = 1.0 - rho
        else:
            continue
        adjusted[(i, j)] = p * max(tau, 1e-6)
    return normalize_matrix(adjusted)


def build_score_matrix(
    lambda_a: float,
    lambda_b: float,
    *,
    r: float = DEFAULT_R,
    dc_rho: float | None = DEFAULT_DC_RHO,
    max_goals: int = 10,
) -> ScoreMatrix:
    matrix = bivariate_negbin_matrix(lambda_a, lambda_b, r=r, max_goals=max_goals)
    if dc_rho is not None:
        matrix = apply_dixon_coles_adjustment(matrix, lambda_a, lambda_b, rho=dc_rho)
    return matrix


def score_matrix_from_prediction(
    prediction: dict[str, Any],
    *,
    r: float = DEFAULT_R,
    dc_rho: float | None = None,
    max_goals: int = 10,
) -> ScoreMatrix:
    """Convenience entry point matching the `score_matrix_from_prediction` shape
    used by v11/v39/v42/v43, so this module is a drop-in replacement if wired in
    later -- reads `lambda_a`/`lambda_b`/`calibration_notes.dixon_coles_rho` off
    an existing prediction dict without needing to import those modules.
    """
    lambda_a = float(prediction["lambda_a"])
    lambda_b = float(prediction["lambda_b"])
    rho = (
        float(dc_rho)
        if dc_rho is not None
        else float(prediction.get("calibration_notes", {}).get("dixon_coles_rho", DEFAULT_DC_RHO))
    )
    return build_score_matrix(lambda_a, lambda_b, r=r, dc_rho=rho, max_goals=max_goals)


def result_probs(score_probs: ScoreMatrix) -> dict[str, float]:
    a = sum(p for (i, j), p in score_probs.items() if i > j)
    d = sum(p for (i, j), p in score_probs.items() if i == j)
    b = sum(p for (i, j), p in score_probs.items() if i < j)
    return {"team_a_win": a, "draw": d, "team_b_win": b}


def matrix_moments(score_probs: ScoreMatrix) -> dict[str, float]:
    mean_a = sum(i * p for (i, j), p in score_probs.items())
    mean_b = sum(j * p for (i, j), p in score_probs.items())
    var_a = sum(((i - mean_a) ** 2) * p for (i, j), p in score_probs.items())
    var_b = sum(((j - mean_b) ** 2) * p for (i, j), p in score_probs.items())
    cov_ab = sum((i - mean_a) * (j - mean_b) * p for (i, j), p in score_probs.items())
    corr_ab = cov_ab / math.sqrt(var_a * var_b) if var_a > 0 and var_b > 0 else 0.0
    return {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "var_a": var_a,
        "var_b": var_b,
        "cov_ab": cov_ab,
        "corr_ab": corr_ab,
    }


def implied_r_from_moments(mean: float, variance: float) -> float:
    """Method-of-moments estimate of r from an observed (mean, variance) pair,
    for calibrating against real scoring data: Var = mean + mean^2 / r."""
    excess = variance - mean
    if excess <= 0:
        return float("inf")
    return (mean * mean) / excess


def score_matrix_to_rows(matrix: ScoreMatrix) -> list[dict[str, Any]]:
    return [
        {"team_a_goals": i, "team_b_goals": j, "probability": p}
        for (i, j), p in sorted(matrix.items(), key=lambda kv: -kv[1])
    ]


def compare_to_independent_poisson(
    lambda_a: float, lambda_b: float, *, r: float, max_goals: int = 10
) -> list[dict[str, Any]]:
    """Diagnostic: which scorelines gain/lose the most probability mass versus
    plain independent Poisson at the same lambdas."""

    def poisson_pmf(k: int, lam: float) -> float:
        return math.exp(-lam) * lam**k / math.factorial(k)

    poisson_matrix = normalize_matrix(
        {
            (i, j): poisson_pmf(i, lambda_a) * poisson_pmf(j, lambda_b)
            for i in range(max_goals + 1)
            for j in range(max_goals + 1)
        }
    )
    nb_matrix = bivariate_negbin_matrix(lambda_a, lambda_b, r=r, max_goals=max_goals)
    rows = []
    for key in poisson_matrix:
        p_poisson = poisson_matrix[key]
        p_nb = nb_matrix[key]
        rows.append(
            {
                "scoreline": f"{key[0]}-{key[1]}",
                "poisson_probability": p_poisson,
                "bivariate_negbin_probability": p_nb,
                "absolute_lift": p_nb - p_poisson,
                "relative_lift": (p_nb / p_poisson - 1.0) if p_poisson > 0 else float("inf"),
            }
        )
    rows.sort(key=lambda row: -row["absolute_lift"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "V49: standalone true bivariate Negative Binomial scoreline model "
            "(shared-gamma-frailty overdispersion + correlation) for comparing "
            "against the independent-Poisson+Dixon-Coles pipeline used elsewhere."
        )
    )
    parser.add_argument("--lambda-a", type=float, required=True, help="Team A expected goals")
    parser.add_argument("--lambda-b", type=float, required=True, help="Team B expected goals")
    parser.add_argument(
        "--r",
        type=float,
        default=DEFAULT_R,
        help="Shared dispersion shape: lower = fatter, more correlated tails (default 25.0, calibrated -- see calibrate_v49_dispersion.py)",
    )
    parser.add_argument(
        "--dc-rho",
        type=float,
        default=DEFAULT_DC_RHO,
        help="Dixon-Coles low-score rho applied on top; pass 0 to disable",
    )
    parser.add_argument("--max-goals", type=int, default=10)
    parser.add_argument("--top", type=int, default=15, help="How many scorelines to print in the comparison table")
    parser.add_argument("--outdir", default=None, help="Optional directory to write score_matrix.json + comparison CSV")
    args = parser.parse_args()

    matrix = build_score_matrix(
        args.lambda_a, args.lambda_b, r=args.r, dc_rho=args.dc_rho, max_goals=args.max_goals
    )
    moments = matrix_moments(matrix)
    results = result_probs(matrix)
    comparison = compare_to_independent_poisson(args.lambda_a, args.lambda_b, r=args.r, max_goals=args.max_goals)

    print(f"lambda_a={args.lambda_a}, lambda_b={args.lambda_b}, r={args.r}, dc_rho={args.dc_rho}")
    print(f"result_probabilities: {results}")
    print(f"moments: {moments}")
    print(f"\nTop {args.top} scorelines gaining the most probability vs independent Poisson:")
    for row in comparison[: args.top]:
        print(
            f"  {row['scoreline']:>5}  poisson={row['poisson_probability']:.4f}  "
            f"bivar_negbin={row['bivariate_negbin_probability']:.4f}  "
            f"lift={row['absolute_lift']:+.4f} ({row['relative_lift']:+.1%})"
        )

    if args.outdir:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "score_matrix.json").write_text(json.dumps(score_matrix_to_rows(matrix), indent=2))
        with (outdir / "poisson_vs_bivariate_negbin_comparison.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(comparison[0].keys()))
            writer.writeheader()
            writer.writerows(comparison)
        print(f"\nWrote {outdir / 'score_matrix.json'} and comparison CSV to {outdir}.")


if __name__ == "__main__":
    main()

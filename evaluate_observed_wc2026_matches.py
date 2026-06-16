#!/usr/bin/env python3
"""Evaluate model v11 against observed 2026 World Cup matches.

The evaluator produces per-match predictions, proper scoring rules, goal-error
metrics, calibration and confusion tables, subgroup diagnostics, box-event
errors, bootstrap confidence intervals, plots, and a Markdown report.

Full v11 evaluation:
    python evaluate_observed_wc2026_matches.py \
      --model-file v11_wcq_results_model.py \
      --outdir observed_eval/observed_eval_v11_comprehensive

Regenerate reports from saved predictions without fitting v11:
    python evaluate_observed_wc2026_matches.py \
      --predictions-input observed_eval/observed_eval_v11_comprehensive/observed_match_predictions.csv \
      --outdir observed_eval/observed_eval_v11_report
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def load_model_module(model_file: str):
    import sys
    model_path = Path(model_file)
    spec = importlib.util.spec_from_file_location("wc_model", model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import model file: {model_file}")
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(model_path)
    sys.modules["wc_model"] = module
    spec.loader.exec_module(module)
    return module


def result_label(goals_a: int, goals_b: int) -> str:
    if goals_a > goals_b:
        return "team_a_win"
    if goals_a == goals_b:
        return "draw"
    return "team_b_win"


def safe_prob(p: float, eps: float = 1e-12) -> float:
    return max(float(p), eps)


def normalize_observed_team(name: str) -> str:
    aliases = {
        "Türkiye": "Turkey",
    }
    return aliases.get(str(name).strip(), str(name).strip())


RESULT_LABELS = ("team_a_win", "draw", "team_b_win")
PROBABILITY_COLUMNS = {
    "team_a_win": "team_a_win_prob",
    "draw": "draw_prob",
    "team_b_win": "team_b_win_prob",
}
CORE_METRICS = (
    "result_accuracy",
    "mean_result_log_loss",
    "mean_result_brier",
    "mean_result_rps",
    "mean_goal_mae",
    "mean_goal_rmse",
    "mean_goal_difference_abs_error",
    "mean_exact_score_log_loss",
)


def validate_observed_data(observed: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the observed-match input before model fitting."""
    required = {"team_a", "team_b", "goals_a", "goals_b"}
    missing = sorted(required - set(observed.columns))
    if missing:
        raise ValueError(f"Observed data is missing required columns: {missing}")
    if observed.empty:
        raise ValueError("Observed data contains no matches")

    clean = observed.copy()
    for column in ("team_a", "team_b"):
        clean[column] = clean[column].astype("string").str.strip()
        if clean[column].isna().any() or clean[column].eq("").any():
            raise ValueError(f"Observed data has blank values in {column}")

    for column in ("goals_a", "goals_b"):
        numeric = pd.to_numeric(clean[column], errors="coerce")
        invalid = numeric.isna() | (numeric < 0) | (numeric % 1 != 0)
        if invalid.any():
            rows = clean.index[invalid].tolist()
            raise ValueError(
                f"Observed data has invalid non-negative integer values in "
                f"{column} at rows {rows}"
            )
        clean[column] = numeric.astype(int)

    if "match_id" not in clean:
        clean["match_id"] = [
            f"observed_{i + 1}_{a}_{b}"
            for i, (a, b) in enumerate(zip(clean["team_a"], clean["team_b"]))
        ]
    duplicate_ids = clean["match_id"].astype(str).duplicated(keep=False)
    if duplicate_ids.any():
        duplicates = sorted(clean.loc[duplicate_ids, "match_id"].astype(str).unique())
        raise ValueError(f"Observed data has duplicate match_id values: {duplicates}")
    return clean


def validate_prediction_frame(eval_df: pd.DataFrame) -> pd.DataFrame:
    """Validate a previously generated prediction CSV."""
    required = {
        "match_id",
        "team_a",
        "team_b",
        "actual_score",
        "actual_result",
        "predicted_result",
        "correct_result",
        "pred_lambda_a",
        "pred_lambda_b",
        "goal_mae",
        "goal_difference_abs_error",
        "actual_result_probability",
        "result_log_loss",
        "result_brier",
        "exact_score_probability",
        "exact_score_log_loss",
        *PROBABILITY_COLUMNS.values(),
    }
    missing = sorted(required - set(eval_df.columns))
    if missing:
        raise ValueError(f"Prediction data is missing required columns: {missing}")
    clean = eval_df.copy()
    probabilities = clean[list(PROBABILITY_COLUMNS.values())].apply(
        pd.to_numeric, errors="coerce"
    )
    if probabilities.isna().any().any():
        raise ValueError("Prediction data contains non-numeric result probabilities")
    if ((probabilities < 0) | (probabilities > 1)).any().any():
        raise ValueError("Prediction probabilities must be between zero and one")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Prediction result probabilities do not sum to one")
    clean[list(PROBABILITY_COLUMNS.values())] = probabilities
    return clean


def ranked_probability_score(actual: str, probabilities: dict[str, float]) -> float:
    """Three-category ranked probability score using A-win, draw, B-win order."""
    forecast = np.array([probabilities[label] for label in RESULT_LABELS], dtype=float)
    observed = np.array([float(label == actual) for label in RESULT_LABELS], dtype=float)
    return float(np.sum((np.cumsum(forecast)[:-1] - np.cumsum(observed)[:-1]) ** 2) / 2)


def prediction_rank(actual: str, probabilities: dict[str, float]) -> int:
    ordered = sorted(RESULT_LABELS, key=lambda label: probabilities[label], reverse=True)
    return ordered.index(actual) + 1


def metric_values(eval_df: pd.DataFrame) -> dict[str, float]:
    goal_errors = np.concatenate(
        [
            pd.to_numeric(eval_df["pred_lambda_a"], errors="coerce").to_numpy()
            - eval_df["actual_score"].str.split("-", expand=True)[0].astype(float).to_numpy(),
            pd.to_numeric(eval_df["pred_lambda_b"], errors="coerce").to_numpy()
            - eval_df["actual_score"].str.split("-", expand=True)[1].astype(float).to_numpy(),
        ]
    )
    return {
        "result_accuracy": float(eval_df["correct_result"].mean()),
        "mean_result_log_loss": float(eval_df["result_log_loss"].mean()),
        "mean_result_brier": float(eval_df["result_brier"].mean()),
        "mean_result_rps": float(eval_df["result_rps"].mean()),
        "mean_goal_mae": float(eval_df["goal_mae"].mean()),
        "mean_goal_rmse": float(np.sqrt(np.mean(np.square(goal_errors)))),
        "mean_goal_difference_abs_error": float(
            eval_df["goal_difference_abs_error"].mean()
        ),
        "mean_exact_score_log_loss": float(eval_df["exact_score_log_loss"].mean()),
        "mean_actual_result_probability": float(
            eval_df["actual_result_probability"].mean()
        ),
        "mean_exact_score_probability": float(
            eval_df["exact_score_probability"].mean()
        ),
        "exact_score_accuracy": float(
            (eval_df["top_scoreline"] == eval_df["actual_score"]).mean()
        )
        if "top_scoreline" in eval_df
        else float("nan"),
        "top_2_result_accuracy": float((eval_df["actual_result_rank"] <= 2).mean()),
    }


def bootstrap_confidence_intervals(
    eval_df: pd.DataFrame,
    samples: int,
    confidence: float,
    seed: int,
) -> dict[str, dict[str, float]]:
    if samples <= 0:
        return {}
    if not 0 < confidence < 1:
        raise ValueError("--confidence must be between zero and one")

    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {metric: [] for metric in CORE_METRICS}
    for _ in range(samples):
        indices = rng.integers(0, len(eval_df), size=len(eval_df))
        metrics = metric_values(eval_df.iloc[indices].reset_index(drop=True))
        for metric in CORE_METRICS:
            values[metric].append(metrics[metric])

    alpha = (1.0 - confidence) / 2.0
    return {
        metric: {
            "lower": float(np.quantile(metric_values_, alpha)),
            "upper": float(np.quantile(metric_values_, 1.0 - alpha)),
            "confidence": confidence,
        }
        for metric, metric_values_ in values.items()
    }


def build_calibration_table(eval_df: pd.DataFrame, bins: int) -> pd.DataFrame:
    """Reliability table for each forecast outcome, using all match/outcome pairs."""
    rows = []
    for label, probability_column in PROBABILITY_COLUMNS.items():
        for probability, actual in zip(
            eval_df[probability_column],
            (eval_df["actual_result"] == label).astype(float),
        ):
            rows.append(
                {
                    "outcome": label,
                    "predicted_probability": float(probability),
                    "observed": float(actual),
                }
            )
    long = pd.DataFrame(rows)
    edges = np.linspace(0.0, 1.0, bins + 1)
    long["bin"] = pd.cut(
        long["predicted_probability"],
        bins=edges,
        include_lowest=True,
        duplicates="drop",
    )
    table = (
        long.groupby(["outcome", "bin"], observed=False)
        .agg(
            n=("observed", "size"),
            mean_predicted_probability=("predicted_probability", "mean"),
            observed_frequency=("observed", "mean"),
        )
        .reset_index()
    )
    table = table[table["n"] > 0].copy()
    table["calibration_error"] = (
        table["mean_predicted_probability"] - table["observed_frequency"]
    ).abs()
    return table


def build_subgroup_table(eval_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    subgroup_specs = [
        ("overall", pd.Series("all", index=eval_df.index)),
        ("actual_result", eval_df["actual_result"]),
    ]
    for optional in ("stage", "group"):
        if optional in eval_df and eval_df[optional].notna().any():
            subgroup_specs.append((optional, eval_df[optional].fillna("unknown")))

    for dimension, values in subgroup_specs:
        grouped = eval_df.assign(_subgroup=values).groupby("_subgroup", dropna=False)
        for subgroup, frame in grouped:
            metrics = metric_values(frame)
            frames.append(
                {
                    "dimension": dimension,
                    "subgroup": subgroup,
                    "n_matches": len(frame),
                    "result_accuracy": metrics["result_accuracy"],
                    "mean_result_log_loss": metrics["mean_result_log_loss"],
                    "mean_result_brier": metrics["mean_result_brier"],
                    "mean_result_rps": metrics["mean_result_rps"],
                    "mean_goal_mae": metrics["mean_goal_mae"],
                }
            )
    return pd.DataFrame(frames)


def build_event_metrics(eval_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    event_names = [
        "shots",
        "shots_on_target",
        "possession",
        "fouls",
        "yellow_cards",
        "red_cards",
    ]
    for event_name in event_names:
        predicted: list[float] = []
        actual: list[float] = []
        for side in ("a", "b"):
            pred_col = f"pred_box_{event_name}_{side}"
            actual_col = f"actual_box_{event_name}_{side}"
            if pred_col in eval_df and actual_col in eval_df:
                predicted.extend(pd.to_numeric(eval_df[pred_col], errors="coerce"))
                actual.extend(pd.to_numeric(eval_df[actual_col], errors="coerce"))
        if not predicted:
            continue
        pred_values = np.asarray(predicted, dtype=float)
        actual_values = np.asarray(actual, dtype=float)
        valid = np.isfinite(pred_values) & np.isfinite(actual_values)
        pred_values = pred_values[valid]
        actual_values = actual_values[valid]
        if len(actual_values) == 0:
            continue
        errors = pred_values - actual_values
        correlation = (
            float(np.corrcoef(pred_values, actual_values)[0, 1])
            if len(actual_values) > 1
            and np.std(pred_values) > 0
            and np.std(actual_values) > 0
            else float("nan")
        )
        rows.append(
            {
                "event": event_name,
                "n_team_observations": int(len(actual_values)),
                "mae": float(np.mean(np.abs(errors))),
                "rmse": float(np.sqrt(np.mean(np.square(errors)))),
                "bias": float(np.mean(errors)),
                "correlation": correlation,
            }
        )
    return pd.DataFrame(rows)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a small dataframe as Markdown without the optional tabulate package."""
    if frame.empty:
        return "_No data available._"

    def render(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.3f}"
        return str(value).replace("|", "\\|")

    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def json_safe(value: Any) -> Any:
    """Convert numpy values and non-finite floats to strict JSON values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def write_markdown_report(
    summary: dict[str, Any],
    subgroup_df: pd.DataFrame,
    event_df: pd.DataFrame,
    outdir: Path,
) -> Path:
    ci = summary.get("bootstrap_confidence_intervals", {})

    def format_metric(name: str, value: float, percent: bool = False) -> str:
        rendered = f"{value:.1%}" if percent else f"{value:.3f}"
        if name in ci:
            lower = ci[name]["lower"]
            upper = ci[name]["upper"]
            interval = (
                f"{lower:.1%} to {upper:.1%}"
                if percent
                else f"{lower:.3f} to {upper:.3f}"
            )
            rendered += f" ({summary['bootstrap_confidence_level']:.0%} CI {interval})"
        return rendered

    model_name = Path(str(summary.get("model_file", "World Cup model"))).stem
    lines = [
        f"# {model_name} observed World Cup evaluation",
        "",
        f"- Matches: {summary['n_matches']}",
        f"- Result accuracy: {format_metric('result_accuracy', summary['result_accuracy'], True)}",
        f"- Result log loss: {format_metric('mean_result_log_loss', summary['mean_result_log_loss'])}",
        f"- Three-way Brier score: {format_metric('mean_result_brier', summary['mean_result_brier'])}",
        f"- Ranked probability score: {format_metric('mean_result_rps', summary['mean_result_rps'])}",
        f"- Goal MAE: {format_metric('mean_goal_mae', summary['mean_goal_mae'])}",
        f"- Goal RMSE: {format_metric('mean_goal_rmse', summary['mean_goal_rmse'])}",
        f"- Exact-score accuracy: {summary['exact_score_accuracy']:.1%}",
        f"- Log-loss skill vs uniform forecast: {summary['log_loss_skill_vs_uniform']:.1%}",
        "",
        "Bootstrap intervals measure sampling uncertainty across this observed match set. "
        "With a small number of matches, they should be expected to be wide.",
        "",
        "## Subgroups",
        "",
        dataframe_to_markdown(subgroup_df),
    ]
    if not event_df.empty:
        lines.extend(["", "## Box events", "", dataframe_to_markdown(event_df)])
    path = outdir / "observed_match_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def plot_evaluation_outputs(
    eval_df: pd.DataFrame,
    summary: dict,
    outdir: Path,
) -> Path:
    """Create a compact diagnostic dashboard for the observed matches."""
    if plt is None:
        raise ImportError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        )
    if eval_df.empty:
        raise ValueError("Cannot plot an empty evaluation dataframe")

    outdir.mkdir(parents=True, exist_ok=True)
    labels = [
        f"{team_a}\nvs {team_b}"
        for team_a, team_b in zip(eval_df["team_a"], eval_df["team_b"])
    ]
    x = np.arange(len(eval_df))
    actual_goals = eval_df["actual_score"].str.split("-", expand=True).astype(float)

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    ax = axes[0, 0]
    probability_columns = [
        ("team_a_win_prob", "Team A win", "#2e86de"),
        ("draw_prob", "Draw", "#f5b041"),
        ("team_b_win_prob", "Team B win", "#e74c3c"),
    ]
    bottom = np.zeros(len(eval_df))
    for column, legend_label, color in probability_columns:
        values = eval_df[column].to_numpy()
        ax.bar(x, values, bottom=bottom, label=legend_label, color=color)
        bottom += values
    ax.set_title("Predicted Result Probabilities")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")

    ax = axes[0, 1]
    width = 0.2
    ax.bar(x - 1.5 * width, eval_df["pred_lambda_a"], width, label="Predicted A", color="#5dade2")
    ax.bar(x - 0.5 * width, actual_goals[0], width, label="Actual A", color="#1b4f72")
    ax.bar(x + 0.5 * width, eval_df["pred_lambda_b"], width, label="Predicted B", color="#f1948a")
    ax.bar(x + 1.5 * width, actual_goals[1], width, label="Actual B", color="#922b21")
    ax.set_title("Predicted vs Actual Goals")
    ax.set_ylabel("Goals")
    ax.legend()

    ax = axes[1, 0]
    ax.bar(
        x,
        eval_df["result_brier"],
        color=np.where(eval_df["correct_result"].astype(bool), "#58d68d", "#ec7063"),
    )
    ax.axhline(
        summary["mean_result_brier"],
        color="#34495e",
        linestyle="--",
        label="Mean Brier score",
    )
    ax.set_title("Three-Way Result Brier Score (Lower Is Better)")
    ax.set_ylabel("Brier score")
    ax.legend()

    ax = axes[1, 1]
    ax.bar(
        x - width,
        eval_df["result_log_loss"],
        2 * width,
        label="Result log loss",
        color="#7dcea0",
    )
    ax.bar(
        x + width,
        eval_df["exact_score_log_loss"],
        2 * width,
        label="Exact-score log loss",
        color="#af7ac5",
    )
    ax.set_title("Loss by Match (Lower Is Better)")
    ax.set_ylabel("Log loss")
    ax.legend()

    ax = axes[2, 0]
    colors = np.where(eval_df["correct_result"].astype(bool), "#27ae60", "#c0392b")
    confidence = eval_df["actual_result_probability"].to_numpy()
    ax.bar(x, confidence, color=colors)
    ax.axhline(
        summary["mean_actual_result_probability"],
        color="#34495e",
        linestyle="--",
        label="Mean confidence",
    )
    ax.set_title("Probability Assigned to Actual Result")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.legend()
    for index, value in enumerate(confidence):
        ax.text(index, value + 0.025, f"{value:.0%}", ha="center", fontsize=9)

    ax = axes[2, 1]
    actual_difference = eval_df["actual_goal_difference"].to_numpy()
    predicted_difference = eval_df["pred_goal_difference"].to_numpy()
    limits = [
        min(actual_difference.min(), predicted_difference.min()) - 0.5,
        max(actual_difference.max(), predicted_difference.max()) + 0.5,
    ]
    ax.scatter(actual_difference, predicted_difference, s=80, color="#2e86de")
    ax.plot(limits, limits, linestyle="--", color="#34495e", label="Perfect prediction")
    for index, label in enumerate(labels):
        ax.annotate(
            label.replace("\n", " "),
            (actual_difference[index], predicted_difference[index]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
        )
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_title("Predicted vs Actual Goal Difference")
    ax.set_xlabel("Actual goal difference")
    ax.set_ylabel("Predicted goal difference")
    ax.legend()

    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.25)
    for ax in axes.flat[:-1]:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)

    fig.suptitle(
        "Observed WC2026 Match Evaluation "
        f"| Accuracy: {summary['result_accuracy']:.1%} "
        f"| Log loss: {summary['mean_result_log_loss']:.3f} "
        f"| Matches: {summary['n_matches']}",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plot_path = outdir / "observed_match_evaluation.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def plot_box_event_outputs(eval_df: pd.DataFrame, outdir: Path) -> Path | None:
    event_specs = [
        ("shots", "Shots"),
        ("shots_on_target", "Shots on target"),
        ("possession", "Possession (%)"),
        ("fouls", "Fouls"),
        ("yellow_cards", "Yellow cards"),
        ("red_cards", "Red cards"),
    ]
    available = [
        (event_name, label)
        for event_name, label in event_specs
        if f"pred_box_{event_name}_a" in eval_df
    ]
    if not available:
        return None

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (event_name, label) in zip(axes.flat, available):
        predicted = pd.concat(
            [
                pd.to_numeric(eval_df[f"pred_box_{event_name}_a"], errors="coerce"),
                pd.to_numeric(eval_df[f"pred_box_{event_name}_b"], errors="coerce"),
            ],
            ignore_index=True,
        )
        actual = pd.concat(
            [
                pd.to_numeric(eval_df[f"actual_box_{event_name}_a"], errors="coerce"),
                pd.to_numeric(eval_df[f"actual_box_{event_name}_b"], errors="coerce"),
            ],
            ignore_index=True,
        )
        valid = predicted.notna() & actual.notna()
        predicted = predicted[valid].to_numpy(dtype=float)
        actual = actual[valid].to_numpy(dtype=float)
        if len(actual) == 0:
            ax.set_axis_off()
            continue

        low = min(float(actual.min()), float(predicted.min()))
        high = max(float(actual.max()), float(predicted.max()))
        margin = max((high - low) * 0.08, 0.2)
        limits = [low - margin, high + margin]
        mae = float(np.mean(np.abs(predicted - actual)))

        ax.scatter(actual, predicted, color="#2e86de", alpha=0.8)
        ax.plot(limits, limits, linestyle="--", color="#34495e")
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_title(f"{label} | MAE {mae:.2f}")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.grid(alpha=0.25)

    for ax in axes.flat[len(available):]:
        ax.set_axis_off()

    fig.suptitle(
        "Observed WC2026 Box-Event Predictions",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plot_path = outdir / "observed_box_event_evaluation.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Generate and comprehensively evaluate World Cup model predictions "
            "against observed match results."
        )
    )
    ap.add_argument(
        "--model-file",
        default=str(PROJECT_DIR / "v11_wcq_results_model.py"),
        help="Path to the model file to evaluate (default: v11_wcq_results_model.py)",
    )
    ap.add_argument(
        "--worldcupsai-zip",
        default=str(DATA_DIR / "worldcupsai.zip"),
        help="WorldCupSAI archive (default: data/worldcupsai.zip).",
    )
    ap.add_argument(
        "--team-train",
        default=str(DATA_DIR / "current_team_features_2026.csv"),
    )
    ap.add_argument("--team-test", default=None)
    ap.add_argument(
        "--observed",
        default=str(DATA_DIR / "wc2026_observed_matches_from_screenshots.csv"),
    )
    ap.add_argument(
        "--box-data", default=str(DATA_DIR / "FIFAallMatchBoxData.csv")
    )
    ap.add_argument("--results-data", default=str(DATA_DIR / "results.csv"))
    ap.add_argument(
        "--former-names", default=str(DATA_DIR / "former_names.csv")
    )
    ap.add_argument("--model", default="ensemble")
    ap.add_argument("--outdir", default="observed_eval/observed_eval_outputs")
    ap.add_argument("--max-goals", type=int, default=10)
    ap.add_argument(
        "--predictions-input",
        help=(
            "Evaluate an existing observed_match_predictions.csv without fitting "
            "the model again."
        ),
    )
    ap.add_argument(
        "--bootstrap-samples",
        type=int,
        default=5000,
        help="Match-level bootstrap samples for confidence intervals (0 disables).",
    )
    ap.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Bootstrap confidence level (default: 0.95).",
    )
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument(
        "--calibration-bins",
        type=int,
        default=5,
        help="Number of reliability bins.",
    )
    ap.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not generate the evaluation dashboard PNG",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.calibration_bins < 2:
        raise ValueError("--calibration-bins must be at least 2")

    if args.predictions_input:
        eval_df = validate_prediction_frame(pd.read_csv(args.predictions_input))
    else:
        wc = load_model_module(args.model_file)
        build_kwargs = {
            "train_csv": args.team_train,
            "test_csv": args.team_test,
            "model_type": args.model,
            "box_csv": args.box_data,
            "results_csv": args.results_data,
            "former_names_csv": args.former_names,
        }
        supported = inspect.signature(wc.build_from_zip).parameters
        model, _ = wc.build_from_zip(
            args.worldcupsai_zip,
            **{key: value for key, value in build_kwargs.items() if key in supported},
        )

        observed = validate_observed_data(pd.read_csv(args.observed))
        rows = []

        for _, r in observed.iterrows():
            observed_team_a = str(r["team_a"])
            observed_team_b = str(r["team_b"])
            team_a = normalize_observed_team(observed_team_a)
            team_b = normalize_observed_team(observed_team_b)

            # Hosts of the 2026 tournament: Canada, Mexico, and United States.
            host_a = team_a in {"USA", "United States", "Mexico", "Canada"}
            host_b = team_b in {"USA", "United States", "Mexico", "Canada"}

            pred = model.predict(
                team_a=team_a,
                team_b=team_b,
                host_a=host_a,
                host_b=host_b,
                knockout=str(r.get("stage", "")).strip().lower() != "group stage",
                max_goals=args.max_goals,
            )

            ga = int(r["goals_a"])
            gb = int(r["goals_b"])
            actual_result = result_label(ga, gb)

            score_probs = {
                (int(item["team_a_goals"]), int(item["team_b_goals"])): float(
                    item["probability"]
                )
                for item in pred["scoreline_probabilities"]
            }
            exact_score_prob = safe_prob(score_probs.get((ga, gb), 0.0))
            result_probs = {
                label: float(pred["result_probabilities"][label])
                for label in RESULT_LABELS
            }
            actual_result_prob = safe_prob(result_probs[actual_result])
            result_brier = sum(
                (result_probs[label] - float(label == actual_result)) ** 2
                for label in RESULT_LABELS
            )
            predicted_result = pred.get(
                "predicted_result",
                max(result_probs, key=result_probs.get),
            )

            row = {
                "match_id": r["match_id"],
                "stage": r.get("stage", ""),
                "group": r.get("group", ""),
                "team_a": observed_team_a,
                "team_b": observed_team_b,
                "model_team_a": pred["team_a"],
                "model_team_b": pred["team_b"],
                "actual_score": f"{ga}-{gb}",
                "pred_lambda_a": pred["lambda_a"],
                "pred_lambda_b": pred["lambda_b"],
                "pred_goal_difference": pred["lambda_a"] - pred["lambda_b"],
                "actual_goal_difference": ga - gb,
                "goal_mae": (
                    abs(pred["lambda_a"] - ga) + abs(pred["lambda_b"] - gb)
                )
                / 2.0,
                "goal_difference_abs_error": abs(
                    (pred["lambda_a"] - pred["lambda_b"]) - (ga - gb)
                ),
                "actual_result": actual_result,
                "predicted_result": predicted_result,
                "correct_result": int(predicted_result == actual_result),
                "actual_result_rank": prediction_rank(actual_result, result_probs),
                "actual_result_probability": actual_result_prob,
                "result_log_loss": -math.log(actual_result_prob),
                "result_brier": result_brier,
                "result_rps": ranked_probability_score(actual_result, result_probs),
                "exact_score_probability": exact_score_prob,
                "exact_score_log_loss": -math.log(exact_score_prob),
                "team_a_win_prob": result_probs["team_a_win"],
                "draw_prob": result_probs["draw"],
                "team_b_win_prob": result_probs["team_b_win"],
                "top_scoreline": (
                    f"{pred['top_scorelines'][0]['team_a_goals']}-"
                    f"{pred['top_scorelines'][0]['team_b_goals']}"
                ),
                "top_scoreline_probability": pred["top_scorelines"][0]["probability"],
                "host_a": host_a,
                "host_b": host_b,
            }
            for key, value in pred.get("v13_adjustments", {}).items():
                row[f"model_adjustment_{key}"] = value

            events = pred.get("event_predictions", {})
            for event in (
                "yellow_cards",
                "red_cards",
                "penalties",
                "penalty_goals",
                "own_goals",
                "substitutions",
            ):
                if event in events:
                    key_a = "expected_" + wc.canon_team(team_a)
                    key_b = "expected_" + wc.canon_team(team_b)
                    if key_a in events[event]:
                        row[f"pred_{event}_a"] = events[event][key_a]
                    if key_b in events[event]:
                        row[f"pred_{event}_b"] = events[event][key_b]

            box_event_columns = {
                "shots": "shots",
                "shots_on_target": "shots_on_target",
                "possession": "possession_a_pct",
                "fouls": "fouls",
                "yellow_cards": "yellow_cards",
                "red_cards": "red_cards",
            }
            for event_name, observed_prefix in box_event_columns.items():
                prediction = events.get(f"box_{event_name}", {})
                key_a = "expected_" + wc.canon_team(team_a)
                key_b = "expected_" + wc.canon_team(team_b)
                actual_a_column = (
                    observed_prefix
                    if observed_prefix.endswith("_a_pct")
                    else f"{observed_prefix}_a"
                )
                actual_b_column = (
                    "possession_b_pct"
                    if observed_prefix.endswith("_a_pct")
                    else f"{observed_prefix}_b"
                )
                if key_a in prediction and key_b in prediction:
                    row[f"pred_box_{event_name}_a"] = prediction[key_a]
                    row[f"pred_box_{event_name}_b"] = prediction[key_b]
                    row[f"actual_box_{event_name}_a"] = r.get(actual_a_column, np.nan)
                    row[f"actual_box_{event_name}_b"] = r.get(actual_b_column, np.nan)

            rows.append(row)
            update_after_match = getattr(model, "update_after_match", None)
            if callable(update_after_match):
                update_details = update_after_match(team_a, team_b, ga, gb)
                if isinstance(update_details, dict):
                    for key, value in update_details.items():
                        row[f"post_match_{key}"] = value

        eval_df = validate_prediction_frame(pd.DataFrame(rows))

    if "result_rps" not in eval_df:
        eval_df["result_rps"] = [
            ranked_probability_score(
                row.actual_result,
                {
                    "team_a_win": row.team_a_win_prob,
                    "draw": row.draw_prob,
                    "team_b_win": row.team_b_win_prob,
                },
            )
            for row in eval_df.itertuples()
        ]
    if "actual_result_rank" not in eval_df:
        eval_df["actual_result_rank"] = [
            prediction_rank(
                row.actual_result,
                {
                    "team_a_win": row.team_a_win_prob,
                    "draw": row.draw_prob,
                    "team_b_win": row.team_b_win_prob,
                },
            )
            for row in eval_df.itertuples()
        ]

    summary = {
        "n_matches": int(len(eval_df)),
        **metric_values(eval_df),
        "model_type": args.model,
        "model_file": args.model_file,
        "uniform_baseline_log_loss": math.log(3.0),
        "uniform_baseline_brier": 2.0 / 3.0,
    }
    summary["log_loss_skill_vs_uniform"] = float(
        1.0 - summary["mean_result_log_loss"] / summary["uniform_baseline_log_loss"]
    )
    summary["brier_skill_vs_uniform"] = float(
        1.0 - summary["mean_result_brier"] / summary["uniform_baseline_brier"]
    )
    summary["bootstrap_samples"] = args.bootstrap_samples
    summary["bootstrap_confidence_level"] = args.confidence
    summary["bootstrap_confidence_intervals"] = bootstrap_confidence_intervals(
        eval_df,
        samples=args.bootstrap_samples,
        confidence=args.confidence,
        seed=args.seed,
    )

    event_df = build_event_metrics(eval_df)
    summary["box_event_metrics"] = {
        row["event"]: {
            key: value
            for key, value in row.items()
            if key not in {"event"}
        }
        for row in event_df.to_dict(orient="records")
    }
    # Retain the old summary key for downstream consumers.
    summary["box_event_mae"] = {
        row["event"]: row["mae"] for row in event_df.to_dict(orient="records")
    }

    calibration_df = build_calibration_table(eval_df, args.calibration_bins)
    subgroup_df = build_subgroup_table(eval_df)
    confusion_df = pd.crosstab(
        eval_df["actual_result"],
        eval_df["predicted_result"],
        rownames=["actual_result"],
        colnames=["predicted_result"],
        dropna=False,
    ).reindex(index=RESULT_LABELS, columns=RESULT_LABELS, fill_value=0)

    eval_df.to_csv(outdir / "observed_match_predictions.csv", index=False)
    calibration_df.to_csv(outdir / "observed_match_calibration.csv", index=False)
    subgroup_df.to_csv(outdir / "observed_match_subgroups.csv", index=False)
    confusion_df.to_csv(outdir / "observed_match_confusion_matrix.csv")
    event_df.to_csv(outdir / "observed_box_event_metrics.csv", index=False)
    with open(outdir / "observed_match_summary.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, indent=2, allow_nan=False)
    report_path = write_markdown_report(summary, subgroup_df, event_df, outdir)

    plot_paths = []
    if not args.no_plots:
        plot_paths.append(plot_evaluation_outputs(eval_df, summary, outdir))
        event_plot_path = plot_box_event_outputs(eval_df, outdir)
        if event_plot_path is not None:
            plot_paths.append(event_plot_path)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote: {outdir / 'observed_match_predictions.csv'}")
    print(f"Wrote: {outdir / 'observed_match_summary.json'}")
    print(f"Wrote: {outdir / 'observed_match_calibration.csv'}")
    print(f"Wrote: {outdir / 'observed_match_subgroups.csv'}")
    print(f"Wrote: {outdir / 'observed_match_confusion_matrix.csv'}")
    print(f"Wrote: {outdir / 'observed_box_event_metrics.csv'}")
    print(f"Wrote: {report_path}")
    for plot_path in plot_paths:
        print(f"Wrote: {plot_path}")


if __name__ == "__main__":
    main()

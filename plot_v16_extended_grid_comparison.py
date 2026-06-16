#!/usr/bin/env python3
"""Plot V16 blend-CV results and observed old/new comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def top_two_metrics(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path)
    return {
        "Exact top 1": float(frame["actual_is_top_1"].mean()),
        "Exact top 2": float(frame["actual_in_top_2"].mean()),
    }


def find_top_two_csv(directory: Path) -> Path:
    matches = sorted(
        directory.glob("*top_two_scoreline_comparison_12_matches.csv")
    )
    if not matches:
        raise FileNotFoundError(
            f"No top-two scoreline comparison CSV found in {directory}"
        )
    return matches[-1]


def plot_blend_curve(calibration: dict, output: Path) -> None:
    frame = pd.DataFrame(calibration["grid_results"])
    weights = 100.0 * frame["bayes_goal_weight"].to_numpy()

    fig, left = plt.subplots(figsize=(10, 6))
    right = left.twinx()
    line_top2 = left.plot(
        weights,
        100.0 * frame["exact_score_top2_rate"],
        marker="o",
        linewidth=2.5,
        color="#2463a6",
        label="Exact-score top-2 rate",
    )
    line_loss = right.plot(
        weights,
        frame["exact_score_log_loss"],
        marker="s",
        linewidth=2.5,
        color="#d97924",
        label="Exact-score log loss",
    )
    draw_scale = frame["group_draw_score_scale"].to_numpy()
    for weight, top2, scale in zip(
        weights,
        100.0 * frame["exact_score_top2_rate"],
        draw_scale,
    ):
        left.annotate(
            f"δ={scale:.2f}",
            (weight, top2),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color="#2463a6",
        )
    selected = 100.0 * calibration["selected_bayes_goal_weight"]
    left.axvline(
        selected,
        color="#2b8a3e",
        linestyle="--",
        linewidth=2,
        label=f"Selected blend: {selected:.0f}%",
    )
    left.axvline(
        50,
        color="#777777",
        linestyle=":",
        linewidth=2,
        label="Previous V16: 50%",
    )
    left.set_title("V16 PyMC Blend: 320-Match Forward CV")
    left.set_xlabel("PyMC goal blend")
    left.set_ylabel("Exact-score top-2 rate (%)")
    right.set_ylabel("Exact-score log loss (lower is better)")
    left.set_xticks(weights)
    left.set_xticklabels([f"{weight:.0f}%" for weight in weights])
    left.grid(axis="y", alpha=0.25)
    lines = line_top2 + line_loss + left.get_lines()[1:]
    left.legend(
        lines,
        [line.get_label() for line in lines],
        loc="upper right",
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_observed_comparison(
    old_summary: dict,
    new_summary: dict,
    old_top_two: dict[str, float],
    new_top_two: dict[str, float],
    selected_blend: float,
    selected_draw_scale: float,
    output: Path,
) -> pd.DataFrame:
    labels = [
        "Previous V16 (50%, δ=1.00)",
        (
            f"Current V16 ({selected_blend:.0f}%, "
            f"δ={selected_draw_scale:.2f})"
        ),
    ]
    rows = []
    for label, summary, top_two in (
        (labels[0], old_summary, old_top_two),
        (labels[1], new_summary, new_top_two),
    ):
        rows.append(
            {
                "version": label,
                "Result accuracy": summary["result_accuracy"],
                "Exact top 1": top_two["Exact top 1"],
                "Exact top 2": top_two["Exact top 2"],
                "Result log loss": summary["mean_result_log_loss"],
                "Goal MAE": summary["mean_goal_mae"],
                "Goal RMSE": summary["mean_goal_rmse"],
                "Exact-score log loss": summary[
                    "mean_exact_score_log_loss"
                ],
            }
        )
    comparison = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ["#7b8794", "#2463a6"]
    x = np.arange(3)
    width = 0.36
    accuracy_metrics = ["Result accuracy", "Exact top 1", "Exact top 2"]
    for index, label in enumerate(labels):
        values = (
            comparison.loc[comparison["version"] == label, accuracy_metrics]
            .iloc[0]
            .to_numpy(dtype=float)
        )
        bars = axes[0].bar(
            x + (index - 0.5) * width,
            100.0 * values,
            width,
            label=label,
            color=colors[index],
        )
        axes[0].bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    axes[0].set_title("Observed 2026 Accuracy and Coverage (12 Matches)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(accuracy_metrics)
    axes[0].set_ylabel("Matches covered")
    axes[0].set_ylim(0, 65)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    quality_metrics = [
        "Result log loss",
        "Goal MAE",
        "Goal RMSE",
        "Exact-score log loss",
    ]
    x = np.arange(len(quality_metrics))
    for index, label in enumerate(labels):
        values = (
            comparison.loc[comparison["version"] == label, quality_metrics]
            .iloc[0]
            .to_numpy(dtype=float)
        )
        bars = axes[1].bar(
            x + (index - 0.5) * width,
            values,
            width,
            label=label,
            color=colors[index],
        )
        axes[1].bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    axes[1].set_title("Observed Error Metrics (Lower Is Better)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        ["Result\nlog loss", "Goal\nMAE", "Goal\nRMSE", "Exact-score\nlog loss"]
    )
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    fig.suptitle("Previous vs Current V16", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibration",
        default="data/v16_pymc_cache/v16_2_forward_calibration.json",
    )
    parser.add_argument(
        "--old-dir",
        default="observed_eval/observed_eval_v16_extended_grid_2026",
    )
    parser.add_argument(
        "--new-dir",
        default="observed_eval/observed_eval_v16_2_2026",
    )
    args = parser.parse_args()

    calibration = load_json(Path(args.calibration))
    old_dir = Path(args.old_dir)
    new_dir = Path(args.new_dir)
    old_summary = load_json(old_dir / "observed_match_summary.json")
    new_summary = load_json(new_dir / "observed_match_summary.json")
    old_top_two = top_two_metrics(find_top_two_csv(old_dir))
    new_top_two = top_two_metrics(find_top_two_csv(new_dir))
    selected_blend = 100.0 * calibration["selected_bayes_goal_weight"]
    selected_draw_scale = calibration["stage_parameters"]["group"][
        "draw_score_scale"
    ]

    plot_blend_curve(
        calibration,
        new_dir / "v16_pymc_blend_forward_cv.png",
    )
    comparison = plot_observed_comparison(
        old_summary,
        new_summary,
        old_top_two,
        new_top_two,
        selected_blend,
        selected_draw_scale,
        new_dir / "v16_previous_vs_current_observed_comparison.png",
    )
    comparison.to_csv(
        new_dir / "v16_previous_vs_current_observed_comparison.csv",
        index=False,
    )


if __name__ == "__main__":
    main()

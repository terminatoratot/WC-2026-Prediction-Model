"""Plots for the leakage-free V51 (v11+v49+v39+v29, no Polymarket) walk-forward
evaluation on all completed 2026 World Cup matches (backtest/eval_v51_knockout_walkforward.py).

Reads outputs/v51_walkforward/walkforward_results.csv, writes PNGs into
outputs/v51_walkforward/plots/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"
SERIES_1 = "#2a78d6"
SERIES_2 = "#1baf7a"
SERIES_3 = "#c77d1f"

OUTDIR = Path("outputs/v51_walkforward")
PLOTDIR = OUTDIR / "plots"


def _style_axes(ax, y_grid=True) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    if y_grid:
        ax.yaxis.grid(True, color=GRIDLINE, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)


def plot_accuracy_summary(df: pd.DataFrame) -> None:
    metric_cols = ["directional_hit", "top3_hit", "top3_outlier_hit"]
    metric_labels = ["Directional", "Top-3", "Top-3 + outlier"]
    groups = [("Overall", df), ("Group stage", df[~df["is_knockout"]]), ("Knockout", df[df["is_knockout"]])]

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)

    x = np.arange(len(metric_cols))
    width = 0.25
    colors = [SERIES_1, SERIES_2, SERIES_3]

    for i, (label, gdf) in enumerate(groups):
        values = [gdf[c].mean() * 100 for c in metric_cols]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width=width * 0.9, color=colors[i], label=f"{label} (n={len(gdf)})", zorder=3)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=7.5, color=INK_PRIMARY)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Accuracy (%)", color=INK_SECONDARY, fontsize=9.5)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY)
    ax.set_title("V51 Walk-Forward Accuracy — All Completed 2026 Matches (no Polymarket, leakage-free)",
                 color=INK_PRIMARY, fontsize=12, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(PLOTDIR / "v51_accuracy_summary.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_match_detail(df: pd.DataFrame) -> None:
    labels = [f"{'[KO] ' if r.is_knockout else ''}{r.team_a} {r.final_score} {r.team_b}" for r in df.itertuples()]
    metrics = ["directional_hit", "top3_hit", "top3_outlier_hit"]
    metric_labels = ["Directional", "Top-3", "Top-3+Outlier"]

    fig_height = 0.20 * len(df) + 1.6
    fig, ax = plt.subplots(figsize=(9.0, fig_height), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)

    n = len(df)
    for row_idx, r in enumerate(df.itertuples()):
        y = n - 1 - row_idx
        for col_idx, m in enumerate(metrics):
            hit = getattr(r, m)
            color = STATUS_GOOD if hit else STATUS_CRITICAL
            ax.scatter(col_idx, y, s=90, color=color, zorder=3, edgecolors=SURFACE, linewidths=0.8)

    ax.set_yticks(range(n))
    ax.set_yticklabels(labels[::-1], color=INK_SECONDARY, fontsize=6.5)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metric_labels, color=INK_SECONDARY, fontsize=9.5)
    ax.set_xlim(-0.6, len(metrics) - 0.4)
    ax.set_ylim(-0.6, n - 0.4)
    ax.tick_params(length=0)

    ax.set_title("V51 Walk-Forward Per-Match Detail (all completed matches, [KO]=knockout)",
                 color=INK_PRIMARY, fontsize=11.5, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(PLOTDIR / "v51_match_detail.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_result_confusion(df: pd.DataFrame, suffix: str = "") -> None:
    labels = ["team_a_win", "draw", "team_b_win"]
    label_display = {"team_a_win": "A wins", "draw": "Draw", "team_b_win": "B wins"}
    matrix = pd.crosstab(df["actual_result"], df["predicted_result"]).reindex(
        index=labels, columns=labels, fill_value=0
    )

    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    im = ax.imshow(matrix.values, cmap="Greens", vmin=0, vmax=max(matrix.values.max(), 1))
    for i in range(3):
        for j in range(3):
            v = matrix.values[i, j]
            text_color = "white" if v > matrix.values.max() * 0.6 else INK_PRIMARY
            ax.text(j, i, str(v), ha="center", va="center", fontsize=13, color=text_color, fontweight="bold")

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels([label_display[l] for l in labels], color=INK_SECONDARY, fontsize=9.5)
    ax.set_yticklabels([label_display[l] for l in labels], color=INK_SECONDARY, fontsize=9.5)
    ax.set_xlabel("Predicted", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Actual", color=INK_SECONDARY, fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    title = f"V51 Directional Confusion Matrix{' — ' + suffix if suffix else ''}"
    ax.set_title(title, color=INK_PRIMARY, fontsize=12, loc="left", pad=12)
    fig.tight_layout()
    fname = f"v51_result_confusion{'_' + suffix.lower().replace(' ', '_') if suffix else ''}.png"
    fig.savefig(PLOTDIR / fname, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(OUTDIR / "walkforward_results.csv")
    PLOTDIR.mkdir(parents=True, exist_ok=True)
    plot_accuracy_summary(df)
    plot_match_detail(df)
    plot_result_confusion(df, "Overall")
    plot_result_confusion(df[~df["is_knockout"]], "Group Stage")
    plot_result_confusion(df[df["is_knockout"]], "Knockout")
    print(f"Wrote plots to {PLOTDIR}")


if __name__ == "__main__":
    main()

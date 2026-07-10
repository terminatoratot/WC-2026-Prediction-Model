"""Plots for the leakage-free V51 (v11+v49+v39+v29, no Polymarket) walk-forward
evaluation on the 22 basev51 knockout matches (backtest/eval_v51_knockout_walkforward.py).

Reads outputs/v51_knockout_walkforward/walkforward_results.csv, writes PNGs
into outputs/v51_knockout_walkforward/plots/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

OUTDIR = Path("outputs/v51_knockout_walkforward")
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
    metrics = {
        "Directional\n(win/draw/loss)": df["directional_hit"].mean(),
        "Top-3\nexact score": df["top3_hit"].mean(),
        "Top-3 + outlier\nexact score": df["top3_outlier_hit"].mean(),
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)

    labels = list(metrics.keys())
    values = [v * 100 for v in metrics.values()]
    colors = [SERIES_1, SERIES_2, SERIES_3]
    bars = ax.bar(labels, values, color=colors, width=0.55, zorder=3)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=11, color=INK_PRIMARY, fontweight="bold")

    ax.set_ylim(0, max(values) * 1.25)
    ax.set_ylabel("Accuracy (%)", color=INK_SECONDARY, fontsize=9.5)
    n = len(df)
    ax.set_title(f"V51 Walk-Forward Accuracy — {n} Knockout Matches (no Polymarket, leakage-free)",
                 color=INK_PRIMARY, fontsize=12, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(PLOTDIR / "v51_accuracy_summary.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_match_detail(df: pd.DataFrame) -> None:
    labels = [f"{r.team_a} {r.final_score} {r.team_b}" for r in df.itertuples()]
    metrics = ["directional_hit", "top3_hit", "top3_outlier_hit"]
    metric_labels = ["Directional", "Top-3", "Top-3+Outlier"]

    fig_height = 0.34 * len(df) + 1.6
    fig, ax = plt.subplots(figsize=(8.5, fig_height), dpi=150)
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
            ax.scatter(col_idx, y, s=260, color=color, zorder=3, edgecolors=SURFACE, linewidths=1.5)
            ax.text(col_idx, y, "Y" if hit else "N", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold", zorder=4)

    ax.set_yticks(range(n))
    ax.set_yticklabels(labels[::-1], color=INK_SECONDARY, fontsize=8.5)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metric_labels, color=INK_SECONDARY, fontsize=9.5)
    ax.set_xlim(-0.6, len(metrics) - 0.4)
    ax.set_ylim(-0.6, n - 0.4)
    ax.tick_params(length=0)

    directional = df["directional_hit"].mean() * 100
    top3 = df["top3_hit"].mean() * 100
    top3o = df["top3_outlier_hit"].mean() * 100
    ax.set_title(
        f"V51 Walk-Forward Per-Match Detail (Dir {directional:.0f}% / Top-3 {top3:.0f}% / +Outlier {top3o:.0f}%)",
        color=INK_PRIMARY, fontsize=11.5, loc="left", pad=12,
    )
    fig.tight_layout()
    fig.savefig(PLOTDIR / "v51_match_detail.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_result_confusion(df: pd.DataFrame) -> None:
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
    ax.set_title("V51 Directional Confusion Matrix", color=INK_PRIMARY, fontsize=12, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(PLOTDIR / "v51_result_confusion.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(OUTDIR / "walkforward_results.csv")
    PLOTDIR.mkdir(parents=True, exist_ok=True)
    plot_accuracy_summary(df)
    plot_match_detail(df)
    plot_result_confusion(df)
    print(f"Wrote plots to {PLOTDIR}")


if __name__ == "__main__":
    main()

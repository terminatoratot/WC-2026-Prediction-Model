"""Per-match performance detail for a v46_4_basev51 tier-optimization knockout
backtest: one labeled horizontal bar per real match (teams + final score),
chronological, colored by hit/miss, with the profit for each labeled directly.

This is a labeled companion to match_profit_bars.png (which only plots bare
match index, not team names) -- built to actually name each result, since the
match-level story (which specific games hit/missed and why) is the point.

Reads outdir/equity_curves/best_equity_curve.csv, writes
outdir/plots/knockout_match_performance_detail.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Same palette as plot_v49_scorematrix_backtest.py / the rest of this repo's
# tuning plots -- reused unchanged for visual consistency.
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)


def plot_match_performance(outdir: Path, plot_dir: Path) -> None:
    df = pd.read_csv(outdir / "equity_curves" / "best_equity_curve.csv")
    df = df.sort_values("idx", ascending=True).reset_index(drop=True)

    labels = [
        f"{r.team_a} {r.final_score} {r.team_b}"
        for r in df.itertuples()
    ]
    colors = [STATUS_GOOD if hit else STATUS_CRITICAL for hit in df["hit"]]

    fig_height = 0.34 * len(df) + 1.4
    fig, ax = plt.subplots(figsize=(9.5, fig_height), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)

    y = range(len(df))[::-1]  # chronological top-to-bottom
    bars = ax.barh(list(y), df["profit"], color=colors, height=0.62, zorder=3)

    ax.axvline(0, color=BASELINE, linewidth=1.0, zorder=2)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, color=INK_SECONDARY, fontsize=8.5)
    ax.set_xlabel("Profit (\\$, target stake \\$5/match)", color=INK_SECONDARY, fontsize=9.5)
    ax.set_ylim(-0.6, len(df) - 0.4)

    max_abs = max(df["profit"].abs().max(), 1.0)
    for bar, profit in zip(bars, df["profit"]):
        offset = max_abs * 0.02
        x = bar.get_width() + offset if profit >= 0 else bar.get_width() - offset
        ha = "left" if profit >= 0 else "right"
        ax.text(
            x, bar.get_y() + bar.get_height() / 2, f"{profit:+.1f}",
            va="center", ha=ha, fontsize=7.5, color=INK_MUTED,
        )

    hits = int(df["hit"].sum())
    total = len(df)
    ax.set_title(
        f"Knockout Backtest: Match-by-Match Performance ({hits}/{total} hit, uncapped VALUE tiers)",
        color=INK_PRIMARY, fontsize=12, loc="left", pad=12,
    )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=STATUS_GOOD, label="Hit"),
        plt.Rectangle((0, 0), 1, 1, color=STATUS_CRITICAL, label="Miss"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)

    fig.tight_layout()
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / "knockout_match_performance_detail.png"
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Labeled per-match performance detail for a knockout tier backtest.")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    outdir = Path(args.outdir)
    plot_match_performance(outdir, outdir / "plots")


if __name__ == "__main__":
    main()

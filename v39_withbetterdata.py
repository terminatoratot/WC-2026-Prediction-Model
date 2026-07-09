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

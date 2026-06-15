#!/usr/bin/env python3
"""Build per-team player-aggregate features for the v14 model.

Reads the compact national-team player extract
(`data/player_ratings_international.csv`, produced from the FIFA dataset) and emits
two feature tables with an identical schema:

- squad_player_features.csv  -- aggregated over the whole national squad (Variant A)
- lineup_player_features.csv -- aggregated over the FIFA-designated starting XI,
                                falling back to top-11-by-overall (Variant B)

Both are keyed by (team, fifa_version). The model joins them on the canonical team
name and the FIFA edition appropriate to a match's date.

The aggregation functions are also importable so the model can build features for a
single team on the fly.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

# --- position -> line classification ---------------------------------------
# FIFA position codes (used in both `player_positions` and `nation_position`).
_DEF = {"CB", "RCB", "LCB", "RB", "LB", "RWB", "LWB", "SW"}
_MID = {"CDM", "RDM", "LDM", "CM", "RCM", "LCM", "CAM", "RAM", "LAM",
        "RM", "LM", "CF"}  # CF leans creative; attackers handled below by ST/W
_ATT = {"ST", "RS", "LS", "RW", "LW", "RF", "LF", "CF"}
_GK = {"GK"}
# Bench markers that are NOT part of the starting XI.
_BENCH = {"SUB", "RES", ""}

# Output feature columns (the aggregate schema shared by both variants).
FEATURE_COLS = [
    "gk", "def", "mid", "att", "top11_overall", "squad_overall",
    "star_power", "depth", "value_log", "avg_age_top11", "n_players",
]


def _primary_position(player_positions: object) -> str:
    """First listed position, upper-cased (e.g. 'ST, CF' -> 'ST')."""
    if pd.isna(player_positions):
        return ""
    return str(player_positions).split(",")[0].strip().upper()


def _line_of(pos: str) -> str:
    if pos in _GK:
        return "GK"
    if pos in _DEF:
        return "DEF"
    if pos in _ATT:
        return "ATT"
    if pos in _MID:
        return "MID"
    return "MID"  # safe default for unknown codes


def _line_mean(df: pd.DataFrame, line: str, top_n: int, agg: str = "mean") -> float:
    vals = df.loc[df["line"] == line, "overall"].sort_values(ascending=False)
    if vals.empty:
        # Fall back to overall squad mean so a missing line doesn't zero out.
        vals = df["overall"].sort_values(ascending=False)
    vals = vals.head(top_n)
    if vals.empty:
        return np.nan
    return float(vals.max() if agg == "max" else vals.mean())


def aggregate_team(players: pd.DataFrame) -> dict:
    """Aggregate a set of a single team's players into the feature schema."""
    df = players.copy()
    df["overall"] = pd.to_numeric(df["overall"], errors="coerce")
    df = df[df["overall"].notna()]
    if df.empty:
        return {c: np.nan for c in FEATURE_COLS}

    if "line" not in df.columns:
        df["line"] = df["player_positions"].map(_primary_position).map(_line_of)
    df["value_eur"] = pd.to_numeric(df.get("value_eur"), errors="coerce")
    df["age"] = pd.to_numeric(df.get("age"), errors="coerce")

    ordered = df.sort_values("overall", ascending=False)
    top11 = ordered.head(11)
    top23 = ordered.head(23)

    return {
        "gk": _line_mean(df, "GK", 1, agg="max"),
        "def": _line_mean(df, "DEF", 4),
        "mid": _line_mean(df, "MID", 4),
        "att": _line_mean(df, "ATT", 3),
        "top11_overall": float(top11["overall"].mean()),
        "squad_overall": float(top23["overall"].mean()),
        "star_power": float(ordered["overall"].head(3).mean()),
        "depth": float(ordered["overall"].iloc[11:23].mean())
        if len(ordered) > 11 else float(top11["overall"].mean()),
        "value_log": float(np.log1p(np.nansum(top23["value_eur"].clip(lower=0)))),
        "avg_age_top11": float(top11["age"].mean()),
        "n_players": int(len(df)),
    }


def _starting_xi(team_df: pd.DataFrame) -> pd.DataFrame:
    """FIFA-designated XI: players with a real (non-bench) nation_position.

    Falls back to the 11 highest-rated players if fewer than 11 starters are
    flagged (common for older editions / smaller nations).
    """
    pos = team_df["nation_position"].fillna("").astype(str).str.upper().str.strip()
    starters = team_df[~pos.isin(_BENCH)]
    if len(starters) >= 11:
        return starters
    return team_df.sort_values("overall", ascending=False).head(11)


def build_tables(extract: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (squad_features, lineup_features), keyed by (team, fifa_version)."""
    df = extract.copy()
    df["overall"] = pd.to_numeric(df["overall"], errors="coerce")
    df["line"] = df["player_positions"].map(_primary_position).map(_line_of)
    df = df.rename(columns={"nationality_name": "team"})

    squad_rows, lineup_rows = [], []
    for (team, version), grp in df.groupby(["team", "fifa_version"], sort=False):
        squad_rows.append({"team": team, "fifa_version": version, **aggregate_team(grp)})
        xi = _starting_xi(grp)
        lineup_rows.append({"team": team, "fifa_version": version, **aggregate_team(xi)})

    squad = pd.DataFrame(squad_rows)
    lineup = pd.DataFrame(lineup_rows)
    return squad, lineup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DATA_DIR / "player_ratings_international.csv"))
    parser.add_argument("--squad-out", default=str(DATA_DIR / "squad_player_features.csv"))
    parser.add_argument("--lineup-out", default=str(DATA_DIR / "lineup_player_features.csv"))
    args = parser.parse_args()

    extract = pd.read_csv(args.input)
    squad, lineup = build_tables(extract)
    squad.to_csv(args.squad_out, index=False)
    lineup.to_csv(args.lineup_out, index=False)
    print(f"squad features:  {squad.shape} -> {args.squad_out}")
    print(f"lineup features: {lineup.shape} -> {args.lineup_out}")
    print("teams:", squad["team"].nunique(), "| editions:", sorted(squad["fifa_version"].unique()))
    print(squad.sort_values("top11_overall", ascending=False).head(8).to_string(index=False))


if __name__ == "__main__":
    main()

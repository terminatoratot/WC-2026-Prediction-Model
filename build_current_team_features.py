#!/usr/bin/env python3
"""Build current national-team strength features from local data.

Combines:
- latest rank and rating from football_rankings.csv
- international results from the four years ending at --as-of
- men's World Cup appearances and titles before --as-of

The output is directly consumable by v11_wcq_results_model.py via --team-train.
Squad market value is intentionally omitted because no local source is available.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import zipfile

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"


TEAM_ALIASES = {
    "usa": "United States",
    "united states of america": "United States",
    "iran": "IR Iran",
    "south korea": "Korea Republic",
    "north korea": "Korea DPR",
    "ivory coast": "Côte d'Ivoire",
    "cote d'ivoire": "Côte d'Ivoire",
    "côte d’ivoire": "Côte d'Ivoire",
    "czech republic": "Czechia",
    "türkiye": "Turkey",
    "curacao": "Curaçao",
    "west germany": "Germany",
}


def canonical_team(name: object) -> str:
    if pd.isna(name):
        return ""
    text = str(name).strip()
    key = text.lower().replace("&", "and")
    return TEAM_ALIASES.get(key, text)


def load_former_name_map(path: str | Path | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not path or not Path(path).exists():
        return mapping
    frame = pd.read_csv(path)
    if not {"current", "former"}.issubset(frame.columns):
        raise ValueError("former-names data must contain current and former columns")
    for row in frame.itertuples(index=False):
        mapping[canonical_team(row.former)] = canonical_team(row.current)
    # FIFA treats West Germany's record as part of Germany's history.
    mapping["West Germany"] = "Germany"
    return mapping


def normalize_team(name: object, former_names: dict[str, str]) -> str:
    team = canonical_team(name)
    return former_names.get(team, team)


def load_rankings(path: str | Path, former_names: dict[str, str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"rank", "team", "rating"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Rankings data is missing required columns: {missing}")

    rankings = frame[list(required)].copy()
    rankings["team"] = rankings["team"].map(
        lambda value: normalize_team(value, former_names)
    )
    rankings["fifa_rank_pre_tournament"] = pd.to_numeric(
        rankings["rank"], errors="coerce"
    )
    rankings["fifa_points_pre_tournament"] = pd.to_numeric(
        rankings["rating"], errors="coerce"
    )
    if rankings[
        ["fifa_rank_pre_tournament", "fifa_points_pre_tournament"]
    ].isna().any().any():
        raise ValueError("Rankings contain non-numeric rank or rating values")
    if rankings["team"].duplicated().any():
        duplicates = sorted(rankings.loc[rankings["team"].duplicated(False), "team"])
        raise ValueError(f"Rankings contain duplicate normalized teams: {duplicates}")
    return rankings[
        ["team", "fifa_rank_pre_tournament", "fifa_points_pre_tournament"]
    ]


def build_recent_form(
    path: str | Path,
    as_of: pd.Timestamp,
    former_names: dict[str, str],
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"date", "home_team", "away_team", "home_score", "away_score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Results data is missing required columns: {missing}")

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["home_score"] = pd.to_numeric(frame["home_score"], errors="coerce")
    frame["away_score"] = pd.to_numeric(frame["away_score"], errors="coerce")
    start = as_of - pd.DateOffset(years=4)
    frame = frame[
        frame["date"].between(start, as_of, inclusive="both")
        & frame["home_score"].notna()
        & frame["away_score"].notna()
    ].copy()

    frame["home_team"] = frame["home_team"].map(
        lambda value: normalize_team(value, former_names)
    )
    frame["away_team"] = frame["away_team"].map(
        lambda value: normalize_team(value, former_names)
    )

    home = pd.DataFrame(
        {
            "team": frame["home_team"],
            "goals_for": frame["home_score"],
            "goals_against": frame["away_score"],
            "win": (frame["home_score"] > frame["away_score"]).astype(int),
            "draw": (frame["home_score"] == frame["away_score"]).astype(int),
            "loss": (frame["home_score"] < frame["away_score"]).astype(int),
        }
    )
    away = pd.DataFrame(
        {
            "team": frame["away_team"],
            "goals_for": frame["away_score"],
            "goals_against": frame["home_score"],
            "win": (frame["away_score"] > frame["home_score"]).astype(int),
            "draw": (frame["away_score"] == frame["home_score"]).astype(int),
            "loss": (frame["away_score"] < frame["home_score"]).astype(int),
        }
    )
    team_matches = pd.concat([home, away], ignore_index=True)
    return (
        team_matches.groupby("team", as_index=False)
        .agg(
            matches_last_4y=("team", "size"),
            goals_scored_last_4y=("goals_for", "sum"),
            goals_received_last_4y=("goals_against", "sum"),
            wins_last_4y=("win", "sum"),
            draws_last_4y=("draw", "sum"),
            losses_last_4y=("loss", "sum"),
        )
    )


def build_world_cup_history(
    worldcupsai_zip: str | Path,
    as_of: pd.Timestamp,
    former_names: dict[str, str],
) -> pd.DataFrame:
    archive_path = Path(worldcupsai_zip)
    members = {
        "tournaments": "curated/tournaments_curated.csv",
        "qualified": "curated/qualified_teams_curated.csv",
        "standings": "curated/tournament_standings_curated.csv",
    }
    with zipfile.ZipFile(archive_path) as archive:
        missing = [member for member in members.values() if member not in archive.namelist()]
        if missing:
            raise ValueError(
                f"WorldCupSAI archive is missing required members: {missing}"
            )
        with archive.open(members["tournaments"]) as source:
            tournaments = pd.read_csv(source)
        with archive.open(members["qualified"]) as source:
            qualified = pd.read_csv(source)
        with archive.open(members["standings"]) as source:
            standings = pd.read_csv(source)
    required_tournaments = {"tournament_id", "tournament_name", "men", "year"}
    required_qualified = {"tournament_id", "team_name"}
    required_standings = {"tournament_id", "position", "team_name"}
    if not required_tournaments.issubset(tournaments.columns):
        raise ValueError("Tournament data does not have the expected schema")
    if not required_qualified.issubset(qualified.columns):
        raise ValueError("Qualified-team data does not have the expected schema")
    if not required_standings.issubset(standings.columns):
        raise ValueError("Tournament standings do not have the expected schema")

    men = tournaments[
        tournaments["men"].astype(str).str.upper().eq("TRUE")
        & tournaments["tournament_name"].astype(str).str.contains(
            "FIFA Men's World Cup", case=False, na=False
        )
        & (pd.to_numeric(tournaments["year"], errors="coerce") < as_of.year)
    ][["tournament_id", "year"]]

    appearances = qualified.merge(men, on="tournament_id", how="inner")
    appearances["team"] = appearances["team_name"].map(
        lambda value: normalize_team(value, former_names)
    )
    appearances = appearances.groupby("team", as_index=False).agg(
        world_cup_participations_before=("tournament_id", "nunique")
    )

    champions = standings.merge(men, on="tournament_id", how="inner")
    champions["team"] = champions["team_name"].map(
        lambda value: normalize_team(value, former_names)
    )
    champions["position"] = pd.to_numeric(champions["position"], errors="coerce")
    champions = (
        champions[champions["position"] == 1]
        .groupby("team", as_index=False)
        .agg(world_cup_titles_before=("tournament_id", "nunique"))
    )
    return appearances.merge(champions, on="team", how="outer")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rankings", default=str(DATA_DIR / "football_rankings.csv")
    )
    parser.add_argument("--results", default=str(DATA_DIR / "results.csv"))
    parser.add_argument(
        "--former-names", default=str(DATA_DIR / "former_names.csv")
    )
    parser.add_argument(
        "--worldcupsai-zip",
        default=str(DATA_DIR / "worldcupsai.zip"),
    )
    parser.add_argument(
        "--as-of",
        default=date.today().isoformat(),
        help="Inclusive cutoff date for recent results (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--output", default=str(DATA_DIR / "current_team_features_2026.csv")
    )
    args = parser.parse_args()

    as_of = pd.Timestamp(args.as_of).normalize()
    former_names = load_former_name_map(args.former_names)
    rankings = load_rankings(args.rankings, former_names)
    recent = build_recent_form(args.results, as_of, former_names)
    history = build_world_cup_history(
        args.worldcupsai_zip,
        as_of,
        former_names,
    )

    output = rankings.merge(recent, on="team", how="left").merge(
        history, on="team", how="left"
    )
    count_columns = [
        "matches_last_4y",
        "goals_scored_last_4y",
        "goals_received_last_4y",
        "wins_last_4y",
        "draws_last_4y",
        "losses_last_4y",
        "world_cup_participations_before",
        "world_cup_titles_before",
    ]
    output[count_columns] = output[count_columns].fillna(0).astype(int)
    output["data_as_of"] = as_of.date().isoformat()
    output["ranking_source"] = Path(args.rankings).name
    output["recent_form_source"] = Path(args.results).name
    output["world_cup_history_source"] = Path(args.worldcupsai_zip).name
    output = output.sort_values("fifa_rank_pre_tournament").reset_index(drop=True)
    output.to_csv(args.output, index=False)

    print(
        f"Wrote {len(output)} teams to {args.output}; "
        f"recent form window {as_of - pd.DateOffset(years=4):%Y-%m-%d} "
        f"through {as_of:%Y-%m-%d}."
    )


if __name__ == "__main__":
    main()

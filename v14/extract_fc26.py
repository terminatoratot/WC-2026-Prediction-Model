#!/usr/bin/env python3
"""Download EA Sports FC 26 (FIFA 26) player ratings and add them to the v14
international player extract as **edition 26**.

Source: Kaggle dataset `rovnez/fc-26-fifa-26-player-data` (~18k players scraped
from SoFIFA, snapshot 2025-09-19). Its schema is a superset of the existing
`data/player_ratings_international.csv` extract, so this only selects/renames the
11 columns the rest of the v14 pipeline expects and tags them `fifa_version=26`.

National-team squads for edition 26 are built two ways and concatenated:

  * **Licensed teams** (EA fills `nation_position`): the real call-up squad,
    including the designated starting XI -> used directly. ~28 nations.
  * **All other nations**: a proxy squad = the top-N players of that
    `nationality_name` by `overall`, with `nation_position` left blank. The lineup
    variant then falls back to top-11-by-overall (see build_player_features
    `_starting_xi`). This is what gives most WC2026 teams coverage that EA's
    sparse national-team licensing does not.

The historical editions 15-23 already in the extract are preserved (the old file
is backed up first), so the point-in-time Euro/Copa evaluation is unaffected;
edition 26 only becomes the active snapshot for matches from 2025-09 onward
(i.e. the 2026 World Cup).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

KAGGLE_DATASET = "rovnez/fc-26-fifa-26-player-data"
FC26_EDITION = 26
FC26_SNAPSHOT_DATE = "2025-09-19"  # the dataset's fifa_update_date

# The 11-column schema shared with data/player_ratings_international.csv.
SCHEMA_COLS = [
    "fifa_version", "fifa_update_date", "nationality_name", "short_name",
    "long_name", "player_positions", "nation_position", "overall", "potential",
    "value_eur", "age",
]

FALLBACK_SQUAD_SIZE = 26  # proxy squad size for non-licensed nations


def download_fc26() -> Path:
    """Download the FC 26 dataset via kagglehub and return the players CSV path."""
    import kagglehub

    # kagglehub reads the new KGAT token from KAGGLE_API_TOKEN.
    token_file = Path.home() / ".kaggle" / "access_token"
    if "KAGGLE_API_TOKEN" not in os.environ and token_file.exists():
        os.environ["KAGGLE_API_TOKEN"] = token_file.read_text().strip()

    root = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    csvs = sorted(root.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found in downloaded dataset at {root}")
    # The dataset ships a single FC26_YYYYMMDD.csv; take the newest if several.
    return csvs[-1]


def build_edition26(fc26_csv: Path) -> pd.DataFrame:
    """Return an edition-26 frame in the SCHEMA_COLS layout."""
    df = pd.read_csv(fc26_csv, low_memory=False)
    df = df[df["overall"].notna()].copy()
    df["fifa_version"] = FC26_EDITION
    df["fifa_update_date"] = FC26_SNAPSHOT_DATE

    licensed_mask = df["nation_position"].notna()
    licensed = df[licensed_mask].copy()
    licensed_nations = set(licensed["nationality_name"].unique())
    print(f"  licensed national teams (real squads): {len(licensed_nations)} "
          f"({len(licensed)} players)")

    # Proxy squads for every other nation: top-N by overall, no nation_position.
    others = df[~licensed_mask & ~df["nationality_name"].isin(licensed_nations)].copy()
    others["nation_position"] = pd.NA
    proxy = (
        others.sort_values("overall", ascending=False)
        .groupby("nationality_name", sort=False)
        .head(FALLBACK_SQUAD_SIZE)
    )
    print(f"  proxy national teams (top-{FALLBACK_SQUAD_SIZE} by overall): "
          f"{proxy['nationality_name'].nunique()} ({len(proxy)} players)")

    edition = pd.concat([licensed, proxy], ignore_index=True)
    for col in SCHEMA_COLS:
        if col not in edition.columns:
            edition[col] = pd.NA
    return edition[SCHEMA_COLS]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fc26-csv", default=None,
                    help="Path to the FC26 players CSV (default: download from Kaggle)")
    ap.add_argument("--extract", default=str(DATA_DIR / "player_ratings_international.csv"),
                    help="The international extract to update in place")
    ap.add_argument("--backup", default=str(DATA_DIR / "player_ratings_international_fifa15-23.csv"))
    args = ap.parse_args()

    fc26_csv = Path(args.fc26_csv) if args.fc26_csv else download_fc26()
    print(f">>> FC 26 source CSV: {fc26_csv}")

    edition26 = build_edition26(fc26_csv)
    print(f">>> edition 26 rows: {len(edition26)}  "
          f"nations: {edition26['nationality_name'].nunique()}")

    extract_path = Path(args.extract)
    existing = pd.read_csv(extract_path)
    existing = existing[existing["fifa_version"] != FC26_EDITION]  # idempotent re-run

    backup_path = Path(args.backup)
    if not backup_path.exists():
        existing.to_csv(backup_path, index=False)
        print(f">>> backed up editions 15-23 to {backup_path}")

    combined = pd.concat([existing, edition26], ignore_index=True)
    combined.to_csv(extract_path, index=False)
    print(f">>> wrote {extract_path}: {len(combined)} rows, "
          f"editions {sorted(combined['fifa_version'].unique())}")


if __name__ == "__main__":
    main()

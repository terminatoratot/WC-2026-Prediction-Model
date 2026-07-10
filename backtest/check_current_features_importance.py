"""Two things, using the EXACT same training-pool assembly build_from_zip()
uses (v51_combined_scoreline_model.py:431) -- base WC matches + fbref 2026 +
expanded_matches (all international results.csv) + fbref_international, i.e.
the full ~51k-row pool, not just the ~1060 World-Cup-only rows:

1. Correct the 22-knockout-match Elo table from earlier: self.latest_elo (what
   the live model actually uses for elo_a/elo_b/elo_diff/current_strength_diff)
   is built by StrongWorldCupModel._cache_latest_team_stats() by walking the
   FULL expanded frame, not the WC-only one build_lowscore_counterfactual.py's
   sibling script used -- so the earlier table's numbers don't match what the
   live model actually computes. This recomputes them correctly.

2. Fits a representative RandomForestRegressor on goal_diff (same
   hyperparameters as StrongWorldCupModel._diff_regressor's rf equivalent
   isn't directly it -- this uses the same n_estimators/min_samples_leaf as
   _named_regressors()'s "rf" entry, on the same weighted training setup
   fit() uses) and reports feature_importances_ for elo_diff/abs_elo_diff vs
   the cur_diff_fifa_* columns, to answer whether current_team_features_2026
   .csv (built by build_current_team_features.py, sourced from
   football_rankings.csv) is actually influencing predictions or just present
   and ignored.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import v11_wcq_results_model as v11

KNOCKOUT_MATCHES = [
    ("South Africa", "Canada", "0-1", False), ("Brazil", "Japan", "2-1", True),
    ("Germany", "Paraguay", "1-1", True), ("Netherlands", "Morocco", "1-1", True),
    ("Ivory Coast", "Norway", "1-2", True), ("France", "Sweden", "3-0", True),
    ("Mexico", "Ecuador", "2-0", True), ("Belgium", "Senegal", "3-2", False),
    ("USA", "Bosnia and Herzegovina", "2-0", True), ("Spain", "Austria", "3-0", True),
    ("Portugal", "Croatia", "2-1", True), ("Switzerland", "Algeria", "2-0", True),
    ("Australia", "Egypt", "1-1", True), ("Colombia", "Ghana", "1-0", False),
    ("Canada", "Morocco", "0-3", True), ("Paraguay", "France", "0-1", False),
    ("Brazil", "Norway", "1-2", False), ("Mexico", "England", "2-3", False),
    ("Portugal", "Spain", "0-1", False), ("USA", "Belgium", "1-4", False),
    ("Argentina", "Egypt", "3-2", False), ("Switzerland", "Colombia", "0-0", False),
]


def main() -> None:
    data_dir = Path("data")
    zip_path = "data/worldcupsai.zip"

    loader = v11.WorldCupSAILoader(zip_path, Path(zip_path + "_extracted"))
    all_matches = loader.load_matches()
    fbref_wc = v11.load_fbref_world_cup_matches(str(data_dir / "fbref_world_cup_matches.csv"))
    all_matches = pd.concat([all_matches, fbref_wc], ignore_index=True, sort=False)
    all_matches["prestige_weight"] = 600.0
    all_matches["prestige_tier"] = "world_cup"

    current = v11.load_current_team_features(str(data_dir / "current_team_features_2026.csv"), None)
    box = v11.load_kaggle_box_data(str(data_dir / "FIFAallMatchBoxData.csv"))
    qual = v11.load_world_cup_qualification_results(str(data_dir / "results.csv"), str(data_dir / "former_names.csv"))
    qualifier_source = qual if not qual.empty else box

    expanded = v11.load_expanded_competition_matches(str(data_dir / "results.csv"), str(data_dir / "former_names.csv"))
    fbref_intl = v11.load_fbref_international_matches(str(data_dir / "fbref_international_matches.csv"))
    train_matches = pd.concat([all_matches, expanded, fbref_intl], ignore_index=True, sort=False)

    frame, features, events = v11.build_rolling_features(
        train_matches, current, qualifier_box=qualifier_source, qualifier_fallback_box=box,
    )
    print(f"[setup] {len(frame)} training rows, {len(features)} feature columns\n")

    # --- Part 1: corrected Elo table for the 22 knockout matches ---
    frame_sorted = frame.sort_values("date")
    latest_elo_asof: dict[tuple[str, str], tuple[float, float]] = {}
    fbref_wc["date"] = pd.to_datetime(fbref_wc["date"])
    target_keys = set()
    for team_a, team_b, _, _ in KNOCKOUT_MATCHES:
        target_keys.add(frozenset([v11.canon_team(team_a), v11.canon_team(team_b)]))

    # Walk frame chronologically; each row's OWN elo_a/elo_b is the pre-match
    # value already (build_rolling_features records it before updating), so
    # just pull those rows directly for our 22 matches instead of re-deriving.
    rows = []
    for team_a, team_b, final, hit in KNOCKOUT_MATCHES:
        ca, cb = v11.canon_team(team_a), v11.canon_team(team_b)
        match = frame[(frame["team_a"] == ca) & (frame["team_b"] == cb) & (frame["date"] >= "2026-01-01")]
        swapped = False
        if match.empty:
            match = frame[(frame["team_a"] == cb) & (frame["team_b"] == ca) & (frame["date"] >= "2026-01-01")]
            swapped = True
        if match.empty:
            rows.append({"team_a": team_a, "team_b": team_b, "final_score": final, "hit": hit,
                         "elo_a": None, "elo_b": None, "elo_diff": None})
            continue
        r = match.iloc[0]
        if swapped:
            elo_a, elo_b = r["elo_b"], r["elo_a"]
        else:
            elo_a, elo_b = r["elo_a"], r["elo_b"]
        rows.append({"team_a": team_a, "team_b": team_b, "final_score": final, "hit": hit,
                     "elo_a": round(float(elo_a), 1), "elo_b": round(float(elo_b), 1),
                     "elo_diff": round(float(elo_a - elo_b), 1)})

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print("=== CORRECTED Elo table (full expanded-pool Elo, matches self.latest_elo) ===")
    print(out.to_string(index=False))
    out.to_csv("outputs/knockout_elo_diffs_corrected.csv", index=False)
    print()

    # --- Part 2: feature importance ---
    X = frame[features]
    y = frame["goal_diff"]
    sample_weight = v11.combine_training_weights(frame, v11.build_year_recency_weights(frame, 16.0, 0.10))

    print("[fit] RandomForestRegressor(n_estimators=300, min_samples_leaf=3) on goal_diff, "
          f"{len(X)} rows x {len(features)} features ...")
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=7, n_jobs=-1)
    rf.fit(X.fillna(X.median(numeric_only=True)), y, sample_weight=sample_weight.to_numpy())

    importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    print("\n=== Top 20 feature importances (goal_diff RandomForest) ===")
    print(importances.head(20).to_string())

    print("\n=== Elo vs current_team_features columns specifically ===")
    watch = [c for c in features if c.startswith("cur_diff_") or c in ("elo_diff", "abs_elo_diff", "elo_prob_a")]
    print(importances[watch].sort_values(ascending=False).to_string())
    print(f"\nRank of elo_diff: {list(importances.index).index('elo_diff') + 1} of {len(importances)}")
    print(f"Rank of cur_diff_fifa_points_pre_tournament: "
          f"{list(importances.index).index('cur_diff_fifa_points_pre_tournament') + 1} of {len(importances)}")
    print(f"Rank of cur_diff_fifa_rank_pre_tournament: "
          f"{list(importances.index).index('cur_diff_fifa_rank_pre_tournament') + 1} of {len(importances)}")


if __name__ == "__main__":
    main()

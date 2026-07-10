"""Walk-forward Elo difference for each of the 22 basev51 knockout backtest
matches, using v11_wcq_results_model.py's own elo_expected/elo_margin_multiplier
update rule (the same K=24 * margin-multiplier logic build_rolling_features()
runs internally) applied over the same historical dataset (worldcupsai.zip +
fbref_world_cup_matches.csv for 2026), sorted chronologically so each match's
elo is the pre-match rating -- not the full build_rolling_features() call,
since that also does qualifier-profile merging that needs data we don't have
here and isn't needed for elo alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import v11_wcq_results_model as v11

KNOCKOUT_MATCHES = [
    ("South Africa", "Canada", "0-1", False),
    ("Brazil", "Japan", "2-1", True),
    ("Germany", "Paraguay", "1-1", True),
    ("Netherlands", "Morocco", "1-1", True),
    ("Ivory Coast", "Norway", "1-2", True),
    ("France", "Sweden", "3-0", True),
    ("Mexico", "Ecuador", "2-0", True),
    ("Belgium", "Senegal", "3-2", False),
    ("USA", "Bosnia and Herzegovina", "2-0", True),
    ("Spain", "Austria", "3-0", True),
    ("Portugal", "Croatia", "2-1", True),
    ("Switzerland", "Algeria", "2-0", True),
    ("Australia", "Egypt", "1-1", True),
    ("Colombia", "Ghana", "1-0", False),
    ("Canada", "Morocco", "0-3", True),
    ("Paraguay", "France", "0-1", False),
    ("Brazil", "Norway", "1-2", False),
    ("Mexico", "England", "2-3", False),
    ("Portugal", "Spain", "0-1", False),
    ("USA", "Belgium", "1-4", False),
    ("Argentina", "Egypt", "3-2", False),
    ("Switzerland", "Colombia", "0-0", False),
]


def main() -> None:
    zip_path = "data/worldcupsai.zip"
    loader = v11.WorldCupSAILoader(zip_path, Path(zip_path + "_extracted"))
    base = loader.load_matches()
    fbref = v11.load_fbref_world_cup_matches("data/fbref_world_cup_matches.csv")

    all_matches = pd.concat([base, fbref], ignore_index=True, sort=False)
    all_matches["date"] = pd.to_datetime(all_matches["date"])
    all_matches = all_matches.sort_values("date").reset_index(drop=True)

    elo: dict[str, float] = {}
    pre_match_elo: dict[tuple[str, str, str], tuple[float, float]] = {}

    for r in all_matches.itertuples():
        a, b = r.team_a, r.team_b
        elo_a = float(elo.get(a, 1500.0))
        elo_b = float(elo.get(b, 1500.0))
        pre_match_elo[(a, b, str(r.date.date()))] = (elo_a, elo_b)

        elo_prob_a = v11.elo_expected(elo_a, elo_b)
        actual_a = 1.0 if r.goals_a > r.goals_b else 0.5 if r.goals_a == r.goals_b else 0.0
        k = 24.0 * v11.elo_margin_multiplier(r.goals_a - r.goals_b)
        elo[a] = elo_a + k * (actual_a - elo_prob_a)
        elo[b] = elo_b + k * ((1.0 - actual_a) - (1.0 - elo_prob_a))

    fbref_2026 = fbref.copy()
    fbref_2026["date"] = pd.to_datetime(fbref_2026["date"])

    rows = []
    for team_a, team_b, final, hit in KNOCKOUT_MATCHES:
        ca, cb = v11.canon_team(team_a), v11.canon_team(team_b)
        match = fbref_2026[(fbref_2026["team_a"] == ca) & (fbref_2026["team_b"] == cb)]
        swapped = False
        if match.empty:
            match = fbref_2026[(fbref_2026["team_a"] == cb) & (fbref_2026["team_b"] == ca)]
            swapped = True
        if match.empty:
            rows.append({"team_a": team_a, "team_b": team_b, "final_score": final, "hit": hit,
                         "elo_a": None, "elo_b": None, "elo_diff": None, "abs_elo_diff": None})
            continue

        m = match.iloc[0]
        key = (m["team_a"], m["team_b"], str(m["date"].date()))
        if key not in pre_match_elo:
            rows.append({"team_a": team_a, "team_b": team_b, "final_score": final, "hit": hit,
                         "elo_a": None, "elo_b": None, "elo_diff": None, "abs_elo_diff": None})
            continue

        elo_x, elo_y = pre_match_elo[key]
        if swapped:
            elo_a, elo_b = elo_y, elo_x
        else:
            elo_a, elo_b = elo_x, elo_y

        rows.append({
            "team_a": team_a, "team_b": team_b, "final_score": final, "hit": hit,
            "elo_a": round(elo_a, 1), "elo_b": round(elo_b, 1),
            "elo_diff": round(elo_a - elo_b, 1), "abs_elo_diff": round(abs(elo_a - elo_b), 1),
        })

    out = pd.DataFrame(rows)
    out.to_csv("outputs/knockout_elo_diffs.csv", index=False)
    pd.set_option("display.width", 160)
    print(out.to_string(index=False))
    print("\nWrote outputs/knockout_elo_diffs.csv")


if __name__ == "__main__":
    main()

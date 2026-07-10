"""Walk-forward, leakage-free evaluation of V51 (v11+v49+v39+v29, no
Polymarket layer) on the 22 basev51 knockout backtest matches, using the
now-fixed chronologically-sorted Elo/rolling-form pipeline.

Leakage control: for each target match, fbref_world_cup_matches.csv is
filtered to only rows strictly BEFORE that match's own kickoff (so no other
2026 result -- earlier OR later in the tournament -- leaks into that match's
training data beyond what would genuinely be known at that point in time).
current_team_features_2026.csv (FIFA rankings) is a static pre-tournament
snapshot (2026-06-10) so it never needs filtering. worldcupsai.zip (<=2022),
results.csv/fbref_international_matches.csv (both end 2026-06-10, pre-
tournament) are already safe as-is.

No Polymarket call anywhere -- this only exercises v51.build_from_zip() +
.predict(), never v42's market pipeline.

Metrics: directional accuracy (predicted most-likely result vs actual
win/draw/loss), top-3 accuracy (actual scoreline in the untouched v11+v49
top-3), top-3+outlier accuracy (actual scoreline in v51_combined_top_scorelines,
i.e. top-3 plus the v39/v29 additive outlier picks).
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import v51_combined_scoreline_model as v51
import v11_wcq_results_model as v11

KNOCKOUT_MATCHES = [
    ("South Africa", "Canada", "0-1", "2026-06-28T19:00:00.000Z"),
    ("Brazil", "Japan", "2-1", "2026-06-29T17:00:00.000Z"),
    ("Germany", "Paraguay", "1-1", "2026-06-29T20:30:00.000Z"),
    ("Netherlands", "Morocco", "1-1", "2026-06-30T01:00:00.000Z"),
    ("Ivory Coast", "Norway", "1-2", "2026-06-30T17:00:00.000Z"),
    ("France", "Sweden", "3-0", "2026-06-30T21:00:00.000Z"),
    ("Mexico", "Ecuador", "2-0", "2026-07-01T02:00:00.000Z"),
    ("Belgium", "Senegal", "3-2", "2026-07-01T20:00:00.000Z"),
    ("USA", "Bosnia and Herzegovina", "2-0", "2026-07-02T00:00:00.000Z"),
    ("Spain", "Austria", "3-0", "2026-07-02T19:00:00.000Z"),
    ("Portugal", "Croatia", "2-1", "2026-07-02T23:00:00.000Z"),
    ("Switzerland", "Algeria", "2-0", "2026-07-03T03:00:00.000Z"),
    ("Australia", "Egypt", "1-1", "2026-07-03T18:00:00.000Z"),
    ("Colombia", "Ghana", "1-0", "2026-07-04T01:30:00.000Z"),
    ("Canada", "Morocco", "0-3", "2026-07-04T17:00:00.000Z"),
    ("Paraguay", "France", "0-1", "2026-07-04T21:00:00.000Z"),
    ("Brazil", "Norway", "1-2", "2026-07-05T20:00:00.000Z"),
    ("Mexico", "England", "2-3", "2026-07-06T01:00:00.000Z"),
    ("Portugal", "Spain", "0-1", "2026-07-06T19:00:00.000Z"),
    ("USA", "Belgium", "1-4", "2026-07-07T00:00:00.000Z"),
    ("Argentina", "Egypt", "3-2", "2026-07-07T16:00:00.000Z"),
    ("Switzerland", "Colombia", "0-0", "2026-07-07T20:00:00.000Z"),
]

TMP_DIR = Path("outputs/v51_knockout_walkforward")
TMP_DIR.mkdir(parents=True, exist_ok=True)


def result_label(goals_a: int, goals_b: int) -> str:
    if goals_a > goals_b:
        return "team_a_win"
    if goals_b > goals_a:
        return "team_b_win"
    return "draw"


def evaluate_one(team_a: str, team_b: str, final_score: str, kickoff: str) -> dict:
    ga, gb = (int(x) for x in final_score.split("-"))
    actual_result = result_label(ga, gb)

    fbref_full = pd.read_csv("data/fbref_world_cup_matches.csv")
    fbref_full["_date"] = pd.to_datetime(fbref_full["date"])
    cutoff = pd.Timestamp(kickoff).tz_convert(None).normalize()
    fbref_safe = fbref_full[fbref_full["_date"] < cutoff].drop(columns=["_date"])

    safe_csv = TMP_DIR / f"fbref_wc_before_{team_a}_{team_b}.csv".replace(" ", "_")
    fbref_safe.to_csv(safe_csv, index=False)

    t0 = time.time()
    model, _ = v51.build_from_zip(
        "data/worldcupsai.zip",
        train_csv="data/current_team_features_2026.csv",
        box_csv="data/FIFAallMatchBoxData.csv",
        results_csv="data/results.csv",
        former_names_csv="data/former_names.csv",
        fbref_world_cup_csv=str(safe_csv),
        fbref_international_csv="data/fbref_international_matches.csv",
    )
    pred = model.predict(team_a, team_b, knockout=True)
    elapsed = time.time() - t0

    top3 = [(int(s["team_a_goals"]), int(s["team_b_goals"])) for s in pred["top_scorelines"][:3]]
    combined = [(int(s["team_a_goals"]), int(s["team_b_goals"])) for s in pred["v51_combined_top_scorelines"]]

    rp = pred["result_probabilities"]
    predicted_result = max(rp, key=rp.get)

    row = {
        "team_a": team_a, "team_b": team_b, "final_score": final_score,
        "kickoff": kickoff, "n_wc_rows_used": len(fbref_safe),
        "actual_result": actual_result, "predicted_result": predicted_result,
        "directional_hit": predicted_result == actual_result,
        "team_a_win_prob": round(rp["team_a_win"], 4), "draw_prob": round(rp["draw"], 4),
        "team_b_win_prob": round(rp["team_b_win"], 4),
        "top3": top3, "top3_hit": (ga, gb) in top3,
        "combined": combined, "top3_outlier_hit": (ga, gb) in combined,
        "fit_seconds": round(elapsed, 1),
    }
    print(f"[{team_a} vs {team_b}] {final_score} | dir={'Y' if row['directional_hit'] else 'N'} "
          f"top3={'Y' if row['top3_hit'] else 'N'} top3+outlier={'Y' if row['top3_outlier_hit'] else 'N'} "
          f"| {len(fbref_safe)} prior-WC rows | {elapsed:.0f}s")
    return row


def main() -> None:
    results = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(evaluate_one, ta, tb, fs, ko): (ta, tb)
            for ta, tb, fs, ko in KNOCKOUT_MATCHES
        }
        for future in as_completed(futures):
            results.append(future.result())

    out = pd.DataFrame(results)
    # restore chronological order for readability
    order = {(ta, tb): i for i, (ta, tb, _, _) in enumerate(KNOCKOUT_MATCHES)}
    out["_order"] = out.apply(lambda r: order[(r["team_a"], r["team_b"])], axis=1)
    out = out.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    out.to_csv(TMP_DIR / "walkforward_results.csv", index=False)

    n = len(out)
    directional_acc = out["directional_hit"].mean()
    top3_acc = out["top3_hit"].mean()
    top3_outlier_acc = out["top3_outlier_hit"].mean()

    summary = {
        "n_matches": n,
        "directional_accuracy": round(directional_acc, 4),
        "top3_accuracy": round(top3_acc, 4),
        "top3_plus_outlier_accuracy": round(top3_outlier_acc, 4),
    }
    pd.Series(summary).to_json(TMP_DIR / "summary.json", indent=2)
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

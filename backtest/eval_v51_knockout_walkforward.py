"""Walk-forward, leakage-free evaluation of V51 (v11+v49+v39+v29, no
Polymarket layer) on ALL completed 2026 World Cup matches (group stage +
knockout), using the now-fixed chronologically-sorted Elo/rolling-form
pipeline.

Match list is loaded dynamically from data/fotmob_match_facts_clean.csv
(currently 97 completed matches) rather than hardcoded, so this stays
current as more matches complete. Group-stage vs knockout is inferred from
chronological position (first GROUP_STAGE_MATCH_COUNT=72 matches by kickoff
= group stage, same convention used elsewhere in this codebase via
infer_knockout_from_match_number) -- this determines the `knockout` input
FEATURE passed to .predict(), which matters (it's read by make_features()),
not just a label.

Leakage control: for each target match, fbref_world_cup_matches.csv is
filtered to only rows strictly BEFORE that match's own kickoff (so no other
2026 result -- earlier OR later in the tournament -- leaks into that match's
training data beyond what would genuinely be known at that point in time).
current_team_features_2026.csv (FIFA rankings) is a static pre-tournament
snapshot (2026-06-10) so it never needs filtering. worldcupsai.zip (<=2022),
results.csv/fbref_international_matches.csv (both end 2026-06-10, pre-
tournament) are already safe as-is. Note fbref_world_cup_matches.csv may lag
fotmob_match_facts_clean.csv by a match or two -- if a target match isn't in
fbref yet, its own result simply isn't available to leak either way.

No Polymarket call anywhere -- this only exercises v51.build_from_zip() +
.predict(), never v42's market pipeline.

Metrics: directional accuracy (predicted most-likely result vs actual
win/draw/loss), top-3 accuracy (actual scoreline in the untouched v11+v49
top-3), top-3+outlier accuracy (actual scoreline in v51_combined_top_scorelines,
i.e. top-3 plus the v39/v29 additive outlier picks). Reported overall and
split by group-stage vs knockout.

Tuning parallelism: each fit takes ~129s using all cores (n_jobs=-1,
unconstrained) on a 10-core machine -- that's the only number actually
measured, don't assume single-threaded scales linearly from it. On a bigger
box, raise WORKERS below; don't force n_jobs=1 via LOKY_MAX_CPU_COUNT, that
made things slower, not faster, when tried locally.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import v51_combined_scoreline_model as v51

GROUP_STAGE_MATCH_COUNT = 72
WORKERS = max(1, int(os.environ.get("V51_WORKERS", "4")))

TMP_DIR = Path("outputs/v51_walkforward")
TMP_DIR.mkdir(parents=True, exist_ok=True)


def result_label(goals_a: int, goals_b: int) -> str:
    if goals_a > goals_b:
        return "team_a_win"
    if goals_b > goals_a:
        return "team_b_win"
    return "draw"


def load_all_matches() -> list[dict]:
    df = pd.read_csv("data/fotmob_match_facts_clean.csv")
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df = df.sort_values("kickoff").reset_index(drop=True)
    matches = []
    for idx, r in df.iterrows():
        matches.append({
            "team_a": r["home_team"],
            "team_b": r["away_team"],
            "final_score": f"{int(r['home_score'])}-{int(r['away_score'])}",
            "kickoff": r["kickoff"],
            "is_knockout": idx >= GROUP_STAGE_MATCH_COUNT,
        })
    return matches


def evaluate_one(team_a: str, team_b: str, final_score: str, kickoff: str, is_knockout: bool) -> dict:
    ga, gb = (int(x) for x in final_score.split("-"))
    actual_result = result_label(ga, gb)

    fbref_full = pd.read_csv("data/fbref_world_cup_matches.csv")
    fbref_full["_date"] = pd.to_datetime(fbref_full["date"])
    cutoff = pd.Timestamp(kickoff).tz_convert(None).normalize()
    fbref_safe = fbref_full[fbref_full["_date"] < cutoff].drop(columns=["_date"])

    safe_csv = TMP_DIR / f"fbref_wc_before_{team_a}_{team_b}_{kickoff}.csv".replace(" ", "_").replace(":", "-")
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
    pred = model.predict(team_a, team_b, knockout=is_knockout)
    elapsed = time.time() - t0

    top3 = [(int(s["team_a_goals"]), int(s["team_b_goals"])) for s in pred["top_scorelines"][:3]]
    combined = [(int(s["team_a_goals"]), int(s["team_b_goals"])) for s in pred["v51_combined_top_scorelines"]]

    rp = pred["result_probabilities"]
    predicted_result = max(rp, key=rp.get)

    row = {
        "team_a": team_a, "team_b": team_b, "final_score": final_score,
        "kickoff": kickoff, "is_knockout": is_knockout, "n_wc_rows_used": len(fbref_safe),
        "actual_result": actual_result, "predicted_result": predicted_result,
        "directional_hit": predicted_result == actual_result,
        "team_a_win_prob": round(rp["team_a_win"], 4), "draw_prob": round(rp["draw"], 4),
        "team_b_win_prob": round(rp["team_b_win"], 4),
        "top3": top3, "top3_hit": (ga, gb) in top3,
        "combined": combined, "top3_outlier_hit": (ga, gb) in combined,
        "fit_seconds": round(elapsed, 1),
    }
    stage = "KO" if is_knockout else "GRP"
    print(f"[{stage}] [{team_a} vs {team_b}] {final_score} | dir={'Y' if row['directional_hit'] else 'N'} "
          f"top3={'Y' if row['top3_hit'] else 'N'} top3+outlier={'Y' if row['top3_outlier_hit'] else 'N'} "
          f"| {len(fbref_safe)} prior-WC rows | {elapsed:.0f}s")
    return row


def print_summary(label: str, df: pd.DataFrame) -> dict:
    n = len(df)
    summary = {
        "n_matches": n,
        "directional_accuracy": round(df["directional_hit"].mean(), 4) if n else None,
        "top3_accuracy": round(df["top3_hit"].mean(), 4) if n else None,
        "top3_plus_outlier_accuracy": round(df["top3_outlier_hit"].mean(), 4) if n else None,
    }
    print(f"\n=== {label} (n={n}) ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    return summary


def main() -> None:
    matches = load_all_matches()
    print(f"[setup] {len(matches)} completed matches "
          f"({sum(not m['is_knockout'] for m in matches)} group-stage, "
          f"{sum(m['is_knockout'] for m in matches)} knockout)")

    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {
            pool.submit(evaluate_one, m["team_a"], m["team_b"], m["final_score"], m["kickoff"], m["is_knockout"]): i
            for i, m in enumerate(matches)
        }
        for future in as_completed(futures):
            results.append(future.result())

    out = pd.DataFrame(results)
    out["kickoff"] = pd.to_datetime(out["kickoff"])
    out = out.sort_values("kickoff").reset_index(drop=True)
    out.to_csv(TMP_DIR / "walkforward_results.csv", index=False)

    overall = print_summary("OVERALL", out)
    group = print_summary("GROUP STAGE", out[~out["is_knockout"]])
    knockout = print_summary("KNOCKOUT", out[out["is_knockout"]])

    pd.Series({"overall": overall, "group_stage": group, "knockout": knockout}).to_json(
        TMP_DIR / "summary.json", indent=2
    )


if __name__ == "__main__":
    main()

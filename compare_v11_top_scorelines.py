#!/usr/bin/env python3
"""Plot actual scores against a model's leading scorelines."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


HOSTS_2026 = {"Canada", "Mexico", "USA", "United States"}
OBSERVED_ALIASES = {"Türkiye": "Turkey"}
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"


def load_model_module(model_file: str):
    model_path = Path(model_file)
    spec = importlib.util.spec_from_file_location("wc_model_scoreline_chart", model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import model file: {model_file}")
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(model_path)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def score_text(item: dict) -> str:
    return f"{int(item['team_a_goals'])}-{int(item['team_b_goals'])}"


def rank_word(rank: int) -> str:
    words = {
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
    }
    return words.get(rank, str(rank))


def ordinal_text(rank: int) -> str:
    words = {
        1: "Most likely",
        2: "Second most likely",
        3: "Third most likely",
        4: "Fourth most likely",
        5: "Fifth most likely",
    }
    return words.get(rank, f"Rank {rank}")


def result_label(goals_a: int, goals_b: int) -> str:
    if goals_a > goals_b:
        return "team_a_win"
    if goals_a == goals_b:
        return "draw"
    return "team_b_win"


def result_text(result: str, team_a: str, team_b: str) -> str:
    return {
        "team_a_win": team_a,
        "draw": "Draw",
        "team_b_win": team_b,
    }[result]


def build_comparison(args: argparse.Namespace) -> pd.DataFrame:
    observed = pd.read_csv(args.observed)
    required = {"match_id", "team_a", "team_b", "goals_a", "goals_b"}
    missing = sorted(required - set(observed.columns))
    if missing:
        raise ValueError(f"Observed file is missing columns: {missing}")

    excluded_count = 0
    if args.max_observed_goals_per_team is not None:
        outlier = (
            observed["goals_a"].gt(args.max_observed_goals_per_team)
            | observed["goals_b"].gt(args.max_observed_goals_per_team)
        )
        excluded_count = int(outlier.sum())
        observed = observed.loc[~outlier]
    observed = observed.head(args.matches).copy()

    wc = load_model_module(args.model_file)
    kwargs = {
        "train_csv": args.team_train,
        "test_csv": args.team_test,
        "model_type": args.model,
        "box_csv": args.box_data,
        "results_csv": args.results_data,
        "former_names_csv": args.former_names,
        "prediction_year": args.prediction_year,
    }
    supported = inspect.signature(wc.build_from_zip).parameters
    model, _ = wc.build_from_zip(
        args.worldcupsai_zip,
        **{key: value for key, value in kwargs.items() if key in supported},
    )

    rows = []
    for observed_order, row in observed.iterrows():
        display_a = str(row["team_a"])
        display_b = str(row["team_b"])
        team_a = OBSERVED_ALIASES.get(display_a, display_a)
        team_b = OBSERVED_ALIASES.get(display_b, display_b)
        prediction = model.predict(
            team_a,
            team_b,
            host_a=team_a in HOSTS_2026,
            host_b=team_b in HOSTS_2026,
            knockout=str(row.get("stage", "")).strip().lower() != "group stage",
        )
        top_scorelines = prediction["top_scorelines"][: args.top_n]
        if len(top_scorelines) < args.top_n:
            raise ValueError(
                f"Prediction for {team_a} vs {team_b} returned only "
                f"{len(top_scorelines)} scorelines, fewer than --top-n {args.top_n}"
            )
        goals_a = int(row["goals_a"])
        goals_b = int(row["goals_b"])
        actual = f"{goals_a}-{goals_b}"
        top_texts = [score_text(item) for item in top_scorelines]
        actual_result = result_label(goals_a, goals_b)
        result_probabilities = prediction["result_probabilities"]
        predicted_result = prediction.get(
            "predicted_result",
            max(result_probabilities, key=result_probabilities.get),
        )
        output_row = {
            "observed_order": int(observed_order) + 1,
            "match_id": row["match_id"],
            "team_a": display_a,
            "team_b": display_b,
            "actual_score": actual,
        }
        for rank, (scoreline, item) in enumerate(
            zip(top_texts, top_scorelines),
            start=1,
        ):
            output_row[f"top_{rank}_scoreline"] = scoreline
            output_row[f"top_{rank}_probability"] = float(item["probability"])
        for rank, scoreline in enumerate(top_texts, start=1):
            output_row[f"actual_is_top_{rank}"] = actual == scoreline
        output_row[f"actual_in_top_{args.top_n}"] = actual in set(top_texts)
        output_row.update(
            {
                "actual_score_probability": next(
                    (
                        float(item["probability"])
                        for item in prediction["scoreline_probabilities"]
                        if score_text(item) == actual
                    ),
                    0.0,
                ),
                "actual_result": actual_result,
                "actual_result_label": result_text(
                    actual_result,
                    display_a,
                    display_b,
                ),
                "predicted_result": predicted_result,
                "predicted_result_label": result_text(
                    predicted_result,
                    display_a,
                    display_b,
                ),
                "outcome_correct": predicted_result == actual_result,
                "predicted_result_probability": float(
                    result_probabilities[predicted_result]
                ),
                "actual_result_probability": float(
                    result_probabilities[actual_result]
                ),
                "team_a_win_probability": result_probabilities["team_a_win"],
                "draw_probability": result_probabilities["draw"],
                "team_b_win_probability": result_probabilities["team_b_win"],
            }
        )
        rows.append(output_row)
        update_after_match = getattr(model, "update_after_match", None)
        if callable(update_after_match):
            update_after_match(team_a, team_b, goals_a, goals_b)

    comparison = pd.DataFrame(rows)
    comparison.attrs["excluded_count"] = excluded_count
    comparison.attrs["max_observed_goals_per_team"] = (
        args.max_observed_goals_per_team
    )
    comparison.attrs["model_label"] = args.model_label
    comparison.attrs["top_n"] = args.top_n
    return comparison


def draw_scoreline_chart(comparison: pd.DataFrame, output_path: Path) -> None:
    """Draw a clean, compact scoreline comparison."""
    count = len(comparison)
    top_n = int(comparison.attrs.get("top_n", 2))
    score_xs = [7.15 + 3.4 * index for index in range(top_n)]
    coverage_x = score_xs[-1] + 3.7
    outcome_x = coverage_x + 2.45
    x_limit = outcome_x + 1.85
    fig, ax = plt.subplots(figsize=(x_limit, max(7.2, count * 0.82 + 2.2)))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_xlim(0, x_limit)
    ax.set_ylim(-0.8, count + 0.75)
    ax.axis("off")

    navy = "#14213d"
    muted = "#64748b"
    blue = "#2563eb"
    pale_blue = "#dbeafe"
    green = "#15803d"
    pale_green = "#dcfce7"
    amber = "#b45309"
    pale_amber = "#fef3c7"
    red = "#b91c1c"
    pale_red = "#fee2e2"
    purple = "#7c3aed"
    divider = "#e2e8f0"
    model_label = comparison.attrs.get("model_label", "Model")
    top_label = f"top-{rank_word(top_n)}"
    coverage_column = f"actual_in_top_{top_n}"

    fig.text(
        0.06,
        0.94,
        f"Actual scores vs {top_label} forecasts: {model_label}",
        fontsize=20,
        fontweight="bold",
        color=navy,
        va="top",
    )
    hits = int(comparison[coverage_column].sum())
    outcome_hits = int(comparison["outcome_correct"].sum())
    excluded_count = int(comparison.attrs.get("excluded_count", 0))
    max_observed_goals = comparison.attrs.get("max_observed_goals_per_team")
    sample_text = f"{count} observed matches"
    if max_observed_goals is not None:
        sample_text += (
            f" after excluding {excluded_count} with a team scoring "
            f"more than {max_observed_goals}"
        )
    fig.text(
        0.06,
        0.895,
        (
            f"{sample_text}     "
            f"{top_label.title()} exact-score coverage: {hits}/{count} "
            f"({hits / count:.0%})"
            f"     Correct outcomes: {outcome_hits}/{count} "
            f"({outcome_hits / count:.0%})"
        ),
        fontsize=11.5,
        color=muted,
        va="top",
    )

    headers = [(0.25, "Match"), (5.25, "Actual")]
    headers.extend((x, ordinal_text(rank)) for rank, x in enumerate(score_xs, start=1))
    headers.extend([(coverage_x - 0.3, "Coverage"), (outcome_x - 0.2, "Outcome")])
    y_header = count + 0.15
    for x, label in headers:
        ax.text(
            x,
            y_header,
            label,
            fontsize=10,
            fontweight="bold",
            color=muted,
            va="center",
        )
    ax.plot(
        [0.2, x_limit - 0.25],
        [count - 0.22, count - 0.22],
        color=divider,
        linewidth=1,
    )

    for display_index, row in comparison.reset_index(drop=True).iterrows():
        y = count - 0.8 - display_index
        if display_index % 2 == 1:
            ax.axhspan(y - 0.39, y + 0.39, color="#f8fafc", zorder=0)
        ax.plot(
            [0.2, x_limit - 0.25],
            [y - 0.41, y - 0.41],
            color=divider,
            linewidth=0.75,
        )

        ax.text(
            0.25,
            y + 0.10,
            f"{row.team_a} vs {row.team_b}",
            fontsize=11.4,
            fontweight="bold",
            color=navy,
            va="center",
        )
        ax.text(
            0.25,
            y - 0.18,
            f"Match {int(row.observed_order)}",
            fontsize=8.7,
            color=muted,
            va="center",
        )

        actual_rank = next(
            (
                rank
                for rank in range(1, top_n + 1)
                if bool(row[f"actual_is_top_{rank}"])
            ),
            None,
        )
        if actual_rank == 1:
            status_text, status_color, status_bg = "Top 1", green, pale_green
        elif actual_rank == 2:
            status_text, status_color, status_bg = "Top 2", amber, pale_amber
        elif actual_rank is not None:
            status_text, status_color, status_bg = f"Top {actual_rank}", blue, pale_blue
        else:
            status_text, status_color, status_bg = "Outside", red, pale_red

        ax.text(
            5.55,
            y,
            row.actual_score,
            fontsize=15,
            fontweight="bold",
            color=navy,
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.32",
                "facecolor": status_bg,
                "edgecolor": "none",
            },
        )

        bar_colors = [blue, "#60a5fa", purple, "#14b8a6", "#f97316"]
        for rank, x in enumerate(score_xs, start=1):
            score = row[f"top_{rank}_scoreline"]
            probability = float(row[f"top_{rank}_probability"])
            bar_color = bar_colors[(rank - 1) % len(bar_colors)]
            ax.text(
                x,
                y + 0.10,
                score,
                fontsize=13.5,
                fontweight="bold",
                color=navy,
                va="center",
            )
            track_left = x + 0.78
            track_width = 1.75
            ax.barh(y + 0.08, track_width, height=0.12, left=track_left, color=pale_blue)
            ax.barh(
                y + 0.08,
                track_width * min(probability / 0.13, 1.0),
                height=0.12,
                left=track_left,
                color=bar_color,
            )
            ax.text(
                x,
                y - 0.22,
                f"{probability:.1%}",
                fontsize=9.3,
                color=muted,
                va="center",
            )

        ax.text(
            coverage_x,
            y + 0.07,
            status_text,
            fontsize=10,
            fontweight="bold",
            color=status_color,
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": status_bg,
                "edgecolor": "none",
            },
        )
        ax.text(
            coverage_x,
            y - 0.23,
            f"Actual: {row.actual_score_probability:.1%}",
            fontsize=8.6,
            color=muted,
            ha="center",
            va="center",
        )

        if row.outcome_correct:
            outcome_text, outcome_color, outcome_bg = "Correct", green, pale_green
        else:
            outcome_text, outcome_color, outcome_bg = "Wrong", red, pale_red
        ax.text(
            outcome_x,
            y + 0.07,
            outcome_text,
            fontsize=10,
            fontweight="bold",
            color=outcome_color,
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": outcome_bg,
                "edgecolor": "none",
            },
        )
        ax.text(
            outcome_x,
            y - 0.23,
            (
                f"Pred: {row.predicted_result_label} "
                f"({row.predicted_result_probability:.0%})"
            ),
            fontsize=8.4,
            color=muted,
            ha="center",
            va="center",
        )

    ax.text(
        0.25,
        -0.62,
        (
            "Coverage indicates whether the actual exact score landed in "
            f"the displayed {top_label} exact scorelines or outside them. "
            "Outcome uses the model's W/D/L decision."
        ),
        fontsize=9.2,
        color=muted,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(top=0.83, bottom=0.08, left=0.055, right=0.98)
    fig.savefig(output_path, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-file", default=str(PROJECT_DIR / "v11_wcq_results_model.py")
    )
    parser.add_argument(
        "--model-label",
        default="V11",
        help="Model name displayed in the chart title (default: V11).",
    )
    parser.add_argument(
        "--worldcupsai-zip", default=str(DATA_DIR / "worldcupsai.zip")
    )
    parser.add_argument(
        "--team-train",
        default=str(DATA_DIR / "current_team_features_2026.csv"),
    )
    parser.add_argument("--team-test")
    parser.add_argument(
        "--box-data", default=str(DATA_DIR / "FIFAallMatchBoxData.csv")
    )
    parser.add_argument("--results-data", default=str(DATA_DIR / "results.csv"))
    parser.add_argument(
        "--former-names", default=str(DATA_DIR / "former_names.csv")
    )
    parser.add_argument(
        "--observed",
        default=str(DATA_DIR / "wc2026_observed_matches_from_screenshots.csv"),
    )
    parser.add_argument("--prediction-year", type=int, default=2026)
    parser.add_argument("--model", default="ensemble")
    parser.add_argument(
        "--matches",
        type=int,
        default=7,
        help="Use the first N observed rows (default: 7).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=2,
        help="Number of leading exact scorelines to compare (default: 2).",
    )
    parser.add_argument(
        "--max-observed-goals-per-team",
        type=int,
        help=(
            "Exclude matches where either team scored more than this number "
            "of goals."
        ),
    )
    parser.add_argument(
        "--output-dir", default="observed_eval/observed_eval_v11_with_current"
    )
    parser.add_argument(
        "--output-prefix",
        default="v11",
        help="Prefix used for generated CSV and PNG filenames (default: v11).",
    )
    args = parser.parse_args()

    if args.matches < 1:
        raise ValueError("--matches must be at least 1")
    if args.top_n < 1:
        raise ValueError("--top-n must be at least 1")
    if (
        args.max_observed_goals_per_team is not None
        and args.max_observed_goals_per_team < 0
    ):
        raise ValueError("--max-observed-goals-per-team cannot be negative")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison = build_comparison(args)
    suffix = f"{len(comparison)}_matches"
    if args.max_observed_goals_per_team is not None:
        suffix += f"_max_{args.max_observed_goals_per_team}_goals"
    top_filename = f"top_{rank_word(args.top_n)}"
    csv_path = output_dir / (
        f"{args.output_prefix}_{top_filename}_scoreline_comparison_{suffix}.csv"
    )
    plot_path = output_dir / (
        f"{args.output_prefix}_{top_filename}_scoreline_comparison_{suffix}.png"
    )
    comparison.to_csv(csv_path, index=False)
    draw_scoreline_chart(comparison, plot_path)

    print(comparison.to_string(index=False))
    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {plot_path}")


if __name__ == "__main__":
    main()

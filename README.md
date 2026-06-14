# WC 2026 Prediction Model

A Python model for estimating World Cup match results, scorelines, goals, and
selected match statistics.

The model trains on men's World Cup history and combines it with current team
rankings, recent international results, World Cup qualification form, and
available box-score data. It can predict a single match or run a chronological
backtest over previous tournaments.

## What it produces

For a single match, the model writes:

- win, draw, and loss probabilities
- expected goals for both teams
- the most likely scorelines
- total-goal and over/under probabilities
- expected match statistics such as shots, possession, fouls, and cards
- CSV and JSON output files
- charts for the main predictions

Example output folders for Germany vs Curaçao and Netherlands vs Japan are
included in the repository.

## Setup

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r strong_ml_requirements.txt
```

LightGBM, XGBoost, and CatBoost are optional. The default ensemble works without
them.

## Predict a match

```bash
python v11_wcq_results_model.py \
  --team-a Germany \
  --team-b Curaçao \
  --model ensemble \
  --outdir outputs_germany_curacao
```

Use `--host-a` or `--host-b` when one team has home advantage. Add `--knockout`
for a knockout-stage match.

The script uses the files in `data/` by default, so no extra data arguments are
needed for the included dataset.

## Run a backtest

```bash
python v11_wcq_results_model.py \
  --backtest \
  --test-years 2014 2018 2022 \
  --model ensemble \
  --outdir outputs_backtest
```

To compare several model types:

```bash
python v11_wcq_results_model.py \
  --backtest \
  --compare-models \
  --comparison-models poisson rf ensemble \
  --test-years 2014 2018 2022 \
  --outdir outputs_model_comparison
```

Backtests use an expanding time window: each tournament is predicted using only
earlier tournament data.

## Rebuild current team features

```bash
python build_current_team_features.py \
  --as-of 2026-06-14 \
  --output data/current_team_features_2026.csv
```

This combines the latest available ranking for each team with recent form and
World Cup history. Change `--as-of` when rebuilding the file with newer source
data.

## Evaluate observed matches

```bash
python evaluate_observed_wc2026_matches.py \
  --model-file v11_wcq_results_model.py \
  --outdir observed_eval_outputs
```

The evaluation script creates match-level predictions, scoring metrics,
calibration tables, confidence intervals, plots, and a Markdown report.

## Main files

- `v11_wcq_results_model.py` - model training, prediction, and backtesting
- `build_current_team_features.py` - builds the current team feature table
- `evaluate_observed_wc2026_matches.py` - evaluates saved or newly generated predictions
- `compare_v11_top_scorelines.py` - compares leading scoreline predictions
- `data/` - historical matches, rankings, qualification results, and team features
- `outputs_*` - saved predictions, reports, and charts

## Notes

Football predictions are uncertain. A high win probability does not mean a
result is guaranteed, and exact-score probabilities are especially sensitive to
small changes in expected goals. The output is best read as a range of possible
outcomes rather than a fixed forecast.


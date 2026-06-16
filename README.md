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

LightGBM and XGBoost are optional. V13 works without them; V15 requires
CatBoost, which is included in `strong_ml_requirements.txt`.

## Predict a match

```bash
python v11_wcq_results_model.py \
  --team-a Germany \
  --team-b Curaçao \
  --model ensemble \
  --outdir outputs/outputs_germany_curacao
```

Use `--host-a` or `--host-b` when one team has home advantage. Add `--knockout`
for a knockout-stage match.

To use V13 for W/D/L with V11 exact scores:

```bash
python v13_live_signal_model.py \
  --team-a Belgium \
  --team-b Egypt \
  --outdir outputs/outputs_v13_belgium_egypt
```

To run the CatBoost-enhanced V15 ensemble:

```bash
python v15_catboost_model.py \
  --team-a Belgium \
  --team-b Egypt \
  --outdir outputs/outputs_v15_belgium_egypt
```

V15 also loads `data/player_ratings_international.csv` and
`data/world_cup_2026_declared_squads.csv` by default. Historical rating
snapshots are attached only when their update date precedes the match. The
declared 2026 squad profiles affect expected goals and exact scores through a
conservative 25% blend; they do not change the W/D/L head. Override the files
with `--player-ratings` and `--declared-squads`.

V15 expands training with EURO, Copa América, AFCON, AFC Asian Cup, Gold Cup,
and Oceania Nations Cup matches from `data/results.csv`. It rebuilds Elo over
all internationals in date order, adds rolling continental form and recency
features, and applies tournament prestige as a training sample weight. Live
state defaults to results through June 10, 2026 to avoid leaking the observed
2026 evaluation matches. Change that explicit cutoff with
`--results-as-of YYYY-MM-DD`. V15 chooses its final W/D/L label from the
highest class probability instead of V13's separate draw threshold.

`results.csv` does not identify rounds or stages, so V15 does not currently use
continental stage-reached or reigning-champion features.

To run V17, which keeps the V15 CatBoost architecture but trains on every
pre-cutoff non-World-Cup international result with stronger recency weighting:

```bash
python v17_recency_all_matches_model.py \
  --team-a Belgium \
  --team-b Egypt \
  --outdir outputs/outputs_v17_belgium_egypt
```

V17's default training weights use a 6-year half-life with a 3% old-match
floor, so recent team form matters much more than 20-year-old team history. Its
backtest mode is deliberately World Cup-only on the test side: each fold trains
on prior World Cups plus all earlier internationals such as EURO, Copa América,
AFCON, qualifiers, friendlies, and other results from `data/results.csv`, then
tests only the held-out World Cup.

```bash
python v17_recency_all_matches_model.py \
  --backtest \
  --test-years 2014 2018 2022 \
  --outdir outputs/outputs_v17_world_cup_backtest
```

To run V16 with the full PyMC hierarchy and bivariate score model:

```bash
python v16_bayesian_bivariate_model.py \
  --team-a Belgium \
  --team-b Egypt \
  --outdir outputs/outputs_v16_belgium_egypt
```

V16 keeps every V15 layer and adds:

- PyMC NUTS attack/defence posteriors with partial pooling
- leakage-free posterior snapshots at historical tournament cutoffs
- posterior features in the CatBoost ensemble and a conservative Bayesian xG blend
- World Cup forward-chain calibration folds for 2006 through 2022
- learned W/D/L temperature scaling and calibration tables
- stage-specific bivariate Poisson covariance and fitted 0-0 inflation
- strict score-matrix normalization assertions

V16.2 tunes the score-construction layer only on the 320 expanding-window
World Cup predictions. It grid-searches the PyMC goal blend from 0% to 70%
in five-point increments, including the unchanged V15 goal baseline,
fits group/knockout covariance and zero inflation, and tests a group-stage
draw-score multiplier from 1.00 to 1.25 in 0.05 increments. Knockout draw
scaling stays at 1.00. The blend and draw-scaling grids are selected by
maximizing exact-score top-two coverage on those 320 held-out predictions,
with exact-score log loss retained as a tie-breaker and diagnostic. Forward
exact-score log loss still fits the continuous covariance, zero-inflation, and
W/D/L mass-correction parameters before the discrete top-two grid choice. The
calibrated W/D/L probabilities remain available separately even when the exact
score matrix is left unforced. PyMC itself uses the same tournament-prestige
ladder as V15 plus chronological recency decay in its weighted likelihood.

PyMC posterior snapshots are cached under `data/v16_pymc_cache`, so later runs
reuse identical fits. The default is 800 tuning and 800 retained draws across
four chains per snapshot, using a non-centered hierarchy. Each cache entry
records R-hat, effective sample size, and divergence diagnostics. Adjust the
sampling with `--pymc-tune`, `--pymc-draws`, and `--pymc-chains`.

The script uses the files in `data/` by default, so no extra data arguments are
needed for the included dataset.

## Run a backtest

```bash
python v11_wcq_results_model.py \
  --backtest \
  --test-years 2014 2018 2022 \
  --model ensemble \
  --outdir outputs/outputs_backtest
```

To compare several model types:

```bash
python v11_wcq_results_model.py \
  --backtest \
  --compare-models \
  --comparison-models poisson rf ensemble \
  --test-years 2014 2018 2022 \
  --outdir outputs/outputs_model_comparison
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
  --model-file v13_live_signal_model.py \
  --outdir observed_eval/observed_eval_v13_hybrid
```

The evaluation script creates match-level predictions, scoring metrics,
calibration tables, confidence intervals, plots, and a Markdown report.
`v13_live_signal_model.py` uses V13 for W/D/L probabilities and the result
decision. Exact scores and score-derived markets use V11's expected goals with
its calibrated Poisson/Dixon-Coles score policy, which retained better observed
top-two score coverage than the experimental unreweighted score matrix.

## Main files

- `v11_wcq_results_model.py` - model training, prediction, and backtesting
- `v13_live_signal_model.py` - V13 W/D/L with V11 Poisson/Dixon-Coles scores
- `v15_catboost_model.py` - CatBoost and player-profile enhanced predictions
- `v17_recency_all_matches_model.py` - V15-based all-international training with World Cup-only forward tests
- `v16_bayesian_bivariate_model.py` - PyMC hierarchy and bivariate score model
- `build_current_team_features.py` - builds the current team feature table
- `evaluate_observed_wc2026_matches.py` - evaluates saved or newly generated predictions
- `compare_v11_top_scorelines.py` - compares leading scoreline predictions
- `data/` - historical matches, rankings, qualification results, and team features
- `outputs/outputs_*` - saved predictions, reports, and charts
- `observed_eval/observed_eval_*` - observed-match evaluations and plots

## Notes

Football predictions are uncertain. A high win probability does not mean a
result is guaranteed, and exact-score probabilities are especially sensitive to
small changes in expected goals. The output is best read as a range of possible
outcomes rather than a fixed forecast.

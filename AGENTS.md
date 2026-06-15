# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r strong_ml_requirements.txt
```

Requires Python 3.10+. Core deps: numpy, pandas, scipy, scikit-learn, matplotlib.
LightGBM / XGBoost / CatBoost are optional — each is imported in a `try/except`
and the `ensemble` model silently drops any that are missing, so the default
ensemble runs without them. There is no separate `--model lightgbm` value unless
the library is installed.

## Common commands

There is no test suite, linter, or build step. The scripts are run directly.

```bash
# Predict a single match (writes JSON/CSV/charts into --outdir)
python v11_wcq_results_model.py --team-a Germany --team-b Curaçao \
  --model ensemble --outdir outputs_germany_curacao
# add --host-a/--host-b for home advantage, --knockout for knockout stage,
# --no-plots to skip chart generation

# Chronological expanding-window backtest over chosen World Cup years
python v11_wcq_results_model.py --backtest --test-years 2014 2018 2022 \
  --model ensemble --outdir outputs_backtest

# Compare several model types in one backtest run
python v11_wcq_results_model.py --backtest --compare-models \
  --comparison-models poisson rf ensemble --test-years 2014 2018 2022 \
  --outdir outputs_model_comparison

# Rebuild the current-team feature table (regenerates data/current_team_features_2026.csv)
python build_current_team_features.py --as-of 2026-06-14 \
  --output data/current_team_features_2026.csv

# Evaluate stored/observed WC2026 predictions -> metrics, calibration, report
python evaluate_observed_wc2026_matches.py \
  --model-file v11_wcq_results_model.py --outdir observed_eval_outputs

# Compare leading scorelines across runs
python compare_v11_top_scorelines.py
```

All data-file CLI args default to files under `data/`, so the commands above need
no extra data arguments for the bundled dataset. `--outdir` is auto-suffixed
(`_2`, `_3`, …) by `unique_output_dir` if the directory already exists rather
than overwriting.

## Architecture

The project is a small set of standalone scripts, not a package — there are no
internal imports between them. `v11_wcq_results_model.py` is the engine
(~2700 lines); the other three scripts load it as a module via
`importlib.util.spec_from_file_location` (see `load_model_module`) and reuse its
classes/functions, which is why they take a `--model-file` argument.

### The model engine (`v11_wcq_results_model.py`)

End-to-end flow lives in `build_from_zip()` → `StrongWorldCupModel` → `predict()`:

1. **Data loading.** `WorldCupSAILoader` extracts `data/worldcupsai.zip` and
   `load_matches()` normalizes it into a match table. Training uses **men's World
   Cup final-tournament matches only** (`MEN_WORLD_CUP_FINAL_YEARS`). Other
   sources layered in: `load_kaggle_box_data` (box-score event stats),
   `load_world_cup_qualification_results` (qualifier form from `results.csv`), and
   `load_current_team_features` (the rebuilt 2026 ranking/form table).

2. **Feature engineering.** `build_rolling_features()` builds chronological,
   leak-free rolling features (form, goals, Elo via `elo_expected`) plus
   qualifier features added by `add_qualifier_features`. Critically, current 2026
   team strength is **NOT** attached to historical training rows (would leak the
   future); it is kept on the fitted model only and applied at predict time as an
   explicit correction (`_apply_current_strength_correction`). See the comment at
   `build_from_zip` ~line 1778.

3. **Fitting** (`StrongWorldCupModel.fit`). Trains parallel models for: goals-for
   each side, goal difference, 3-way result classifier, a separate **draw model**
   (`_fit_draw_model`, balanced binary classifier with chronological holdout
   calibration), and one regressor pair per box-score event target. With
   `--model ensemble` each target is a weighted blend of several named learners
   (`_named_regressors` / `_named_classifiers`); other `--model` values use a
   single learner. Training rows are weighted by **exponential year-recency**
   (`build_year_recency_weights`, `--recency-half-life-years`).

4. **Prediction** (`predict`). This is a multi-stage pipeline, not a single model
   output — order matters:
   - regressors produce raw λ_a, λ_b and a goal-difference prediction
   - current-strength correction, then goal-difference blend, then clipping
   - `poisson_score_matrix` → **Dixon-Coles** low-score adjustment
     (`apply_dixon_coles_adjustment`) → result probabilities
   - light blend with the 3-way classifier, then `temperature_smooth_result_probs`
   - the dedicated draw model overrides the draw probability (two-stage:
     P(draw) then conditional win split), and `reweight_score_matrix_to_results`
     forces the scoreline matrix to agree with the final result probabilities
   - scorelines, spreads, totals, over/under, expected events, and box-score
     event expectations are all derived from that final matrix.

   The returned dict carries `calibration_notes` documenting every knob used.
   Exact scorelines are treated as **derived output, not a fit objective**.

5. **Backtesting.** `chronological_backtest` predicts each test year using only
   earlier tournaments (expanding window); `run_model_comparison_backtest` loops
   that over several model types. Metrics emphasize result log-loss and Brier
   score over exact-score accuracy.

### Supporting scripts

- `build_current_team_features.py` — merges latest FIFA `football_rankings.csv` +
  recent-form window from `results.csv` (controlled by `--as-of`) + World Cup
  history from the zip into `data/current_team_features_2026.csv`. Rerun with a
  new `--as-of` to refresh strength inputs.
- `evaluate_observed_wc2026_matches.py` — loads the model module, predicts the
  observed matches in `data/wc2026_observed_matches_from_screenshots.csv`, and
  emits scoring metrics, bootstrap confidence intervals, calibration/subgroup
  tables, plots, and a Markdown report.
- `compare_v11_top_scorelines.py` — builds a side-by-side comparison chart of
  top scoreline predictions across matches/runs.

### Team-name normalization

Country names are inconsistent across sources. Normalization is layered:
`TEAM_ALIASES` + `canon_team()` in the model, `data/former_names.csv` mapping in
`build_current_team_features.py` (`load_former_name_map`/`normalize_team`), and
per-script `normalize_*` helpers. When adding a team or data source, expect to
update the alias/former-name mappings rather than the calling code.

### Outputs

`outputs_*/` and `observed_eval_*/` directories are committed example runs
(predictions, reports, charts). Treat them as artifacts — new runs go in fresh
`--outdir` folders.

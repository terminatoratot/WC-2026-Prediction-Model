# WC 2026 Prediction Model

A Python model for predicting World Cup match results, scorelines, goals, and
Polymarket exact-score betting edges. It trains on men's World Cup history,
current team ratings, recent international results, FotMob current-form
data, and World Cup qualification form.

This repository is a curated slice of a larger research project: it contains
the current model and everything it actually imports at runtime, not the
full history of superseded iterations that were tried along the way.

## How the model works

The prediction stack is built in layers, each one correcting a specific gap
in the layer below it rather than replacing it wholesale. From the ground up:

**1. Base engine (`v11`).** Trains on men's World Cup history only (pulled
from a curated match archive plus `data/results.csv`), and predicts each
match as goal rates (lambdas) for both teams via an ensemble of models —
gradient boosting, random forest, and Poisson regression, blended together —
rather than a single estimator. Notable design choices baked into it:

- a **two-stage result model**: a binary draw-vs-not classifier first, then a
  conditional winner probability, rather than one 3-way softmax (draws are a
  structurally different outcome, not just a third bucket)
- **chronological Elo features** and **year-based exponential recency
  weighting**, so a result from 2022 counts for more than one from 2002
- a **Dixon-Coles correction** to fix the low-score bias that plain
  independent-Poisson scorelines have (0-0, 1-0, 0-1, 1-1 are systematically
  under/over-predicted by naive Poisson otherwise)
- **temperature-smoothed result probabilities** and an explicit
  **current-strength correction**, so a team's long-run historical rate gets
  pulled toward its current form rather than trusted blindly
- exact scorelines are treated as *derived* from the goal-rate model, not
  fit directly as the primary objective
- a Kaggle box-score layer adds predicted match stats (shots, shots on
  target, possession, fouls, saves, cards) alongside the scoreline
- evaluated via **chronological expanding-window backtesting** (train on
  the past, test on the next tournament, never the reverse)

**2. Scoreline correction (`v49`).** Plain independent-Poisson scorelines
assume the two teams' goal counts are uncorrelated and that each team's own
goal count has variance equal to its mean — both are false in practice (some
matches are just more "chaotic" for both sides at once). `v49` replaces the
scoreline matrix with a **bivariate Negative Binomial via shared frailty**:
both teams' goal counts are driven by a shared random "match volatility" draw
`Z ~ Gamma(r, 1/r)`, conditionally Poisson given `Z`. This induces
correlation between the teams and fat tails on total goals without touching
the underlying W/D/L probabilities.

**3. Additive outlier layers (`v29`, `v39_coverage_outlier`).** The model
displays a Top-3 most-likely-scorelines list, which by construction misses
any match that goes to an unusual score. Rather than let a tail event
silently overwrite one of the Top-3 slots, `v29` (tail-risk: gated by
favorite-win/lambda thresholds, catches blowouts) and `v39`'s coverage
outlier (catches matches whose expected total goals sit above the
displayed Top-3's ceiling) each contribute a **4th, additive "outlier" slot**
on top of an untouched Top-3 — the normal prediction never gets overwritten
to buy tail coverage.

**4. Current-form layers (`v28`, `v35`, `v36`, `v38`).** Pure history isn't
enough once a tournament is underway — `v28` folds in observed World Cup
2026 form and FotMob signals; `v35` learns a small, shrunk transition table
from 75th-minute game state to final score for a separate late-mutation
outlier slot; `v36` adds a reusable FotMob xG/player/keeper form layer that
automatically picks up new completed matches as they're scraped, with no
code changes; `v38` applies one shrunk global multiplier to correct a
total-goals under-bracketing bias while leaving W/D/L untouched.

**5. Market layer (`market_edge.py`, i.e. `v42`).** Fetches live Polymarket
odds for exact-score and moneyline markets, classifies them, and compares
the model's fair probabilities against market-implied ones to find priced
edges. Market data is a read-only diagnostic here — it is never blended
into the model's own probabilities.

**6. Combination (`v51`).** Takes `v11` + `v49`'s corrected scorelines and
adds *both* the `v29` and `v39` outlier tabs independently (so a backtest can
show which one, if either, is actually adding hit-rate on top of a clean
Top-3) — this is the actual prediction source used by the current pipeline.

**7. Buy-card generation (`v46_4_basev51`, current entry point).** Runs `v51`
for the scoreline distribution, feeds it through `v42`'s Polymarket pipeline
to find priced edges, then applies Kelly-based staking, tiering, and a
break-even funding floor to produce a concrete "buy card": which exact
scores to back and how much to stake on each, plus audit outputs (selection
reasoning, hit-outcome tracking, a validation pass).

An independent audit of the base models (`BASE_MODEL_AUDIT.md`) found that
none of them — including the market layer — has demonstrated a proper-scoring
edge over Polymarket itself; see that file for the full breakdown and honest
caveats. Read that before assuming any of this is a source of positive
expected value on its own.

## Model lineage & why this is 5 files, not ~25

`v46_4_basev51.py`'s own imports transitively pull in 23 other named
iterations (v11, v13, v15, v18, v20, v23, v24, v26-v42, v49) — this was
verified by parsing every `import` statement in the chain and then actually
executing it, not guessed. Rather than ship two dozen `vNN_*.py` files, the
ones that are never run directly — only ever imported by something else — are
grouped into three files by what they do, not by version number:

```
v46_4_basev51.py     "buy card": staking / Kelly / tiering, Polymarket edges
├── v51_combined_scoreline_model.py   prediction source (see "How it works" above)
└── market_edge.py   Polymarket fetch/classify/edge pipeline
      -> feature_layers.py  current-form/tail-risk/game-state layers (v28-v38)
           -> core_engine.py   base engine + its precursor lineage (v11, v13-v27, v49)
```

- **`core_engine.py`** — the base engine (v11, v49) and its own precursor
  lineage (v13, v15, v18, v20, v23, v24, v26, v27)
- **`feature_layers.py`** — current-form / tail-risk / game-state layers
  (v28-v36, v38)
- **`market_edge.py`** — the Polymarket fetch/classify/edge pipeline
  (v39_coverage_outlier, v39_withbetterdata, v42)
- **`v51_combined_scoreline_model.py`** and **`v46_4_basev51.py`** stay as
  ordinary, independently runnable top-level files

Each original file's source is embedded byte-for-byte unchanged inside the
three grouped files and executed into its own isolated namespace, registered
in `sys.modules` under its original name — the same mechanism Python's own
`import` uses internally, just triggered manually so several "modules" can
live in one physical file. Nothing was renamed, rewritten, or merged at the
symbol level, so every internal `import vNN_x as vNN` statement anywhere in
the codebase works completely unchanged. This was verified by diffing actual
predictions and running the full test suite against the original 25-file
layout before switching over — see git history for the pre-bundling commit
if you want the individual files.

The evaluation harness that validates all of this
(`backtest/audit_base_models.py` and its supporting walk-forward scripts) is
included too — see [`BASE_MODEL_AUDIT.md`](BASE_MODEL_AUDIT.md) for a full
write-up of how the base models compare against each other and the market,
including known gaps.

## Setup

Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

LightGBM and XGBoost are optional alternate regressors inside `v11`; the
model runs fine without them.

## Predict a match (core engine)

```bash
python core_engine.py \
  --team-a Germany \
  --team-b Curaçao \
  --model ensemble \
  --outdir outputs/outputs_germany_curacao
```

Use `--host-a` / `--host-b` for home advantage, `--knockout` for a knockout
match. (This runs `v11`'s CLI — `core_engine.py` dispatches to it when run
directly, exactly as `python v11_wcq_results_model.py ...` did before the
files were bundled.)

## Run the combined scoreline model (v51)

`v51` is a train/test backtesting script (it doesn't take `--team-a`/`--team-b`
directly — for a single match, use `core_engine.py` above or
`v46_4_basev51.py` below). It reads `data/worldcupsai.zip`,
`data/current_team_features_2026.csv`, `data/FIFAallMatchBoxData.csv`,
`data/results.csv`, and `data/former_names.csv` by default — all included in
`data/`.

```bash
python v51_combined_scoreline_model.py --test-years 2022 --outdir outputs/outputs_v51_2022
```

## Generate a buy card (v46.4, current entry point)

By default this fetches live Polymarket odds for the matchup, so it only
works for matches Polymarket is actively quoting. For an offline run against
the model layer only:

```bash
python v46_4_basev51.py --team-a Germany --team-b Curaçao \
  --allow-empty-polymarket \
  --outdir outputs/outputs_v46_4_germany_curacao
```

This script has a large surface area (Polymarket live-market fetching,
staking/Kelly tuning, tiering weights) — run `python v46_4_basev51.py --help`
for the full option set.

## Evaluate against completed matches

```bash
python backtest/audit_base_models.py --outdir outputs/base_model_audit
```

Cross-model evaluation (v11, v15, v29, v36, v39_withbetterdata, v42) on a
common, leak-controlled, walk-forward footing — proper scoring rules
(log-loss, Brier, RPS), a Polymarket market benchmark, and bootstrap CIs. See
`BASE_MODEL_AUDIT.md` for the results and what they mean.

## Tests

```bash
python -m unittest discover -s tests
```

## Data

`data/` holds only the curated inputs the current model chain reads —
World Cup match history, current team features, FotMob box-score exports,
FBref match data, and FIFA rankings. Raw scrape intermediates and one-off
snapshot dumps used to build these files are not included.

## Directory layout

```
core_engine.py       Base engine (v11, v49) + its precursor lineage
feature_layers.py     Current-form / tail-risk / game-state layers (v28-v38)
market_edge.py        Polymarket fetch/classify/edge pipeline (v39/v42)
v51_combined_scoreline_model.py   Prediction source: v11+v49, +v29/v39 outliers
v46_4_basev51.py       Buy-card generator (current entry point)
data/                 Curated input data the model chain reads
backtest/              Cross-model evaluation harness
tests/                 Unit tests (unittest discover -s tests)
```

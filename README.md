# WC 2026 Prediction Model

A Python model for predicting World Cup match results, scorelines, goals, and
Polymarket exact-score betting edges. It trains on men's World Cup history,
current team ratings, recent international results, FotMob current-form
data, and World Cup qualification form.

This repository is a curated slice of a larger research project: it contains
the current model and everything it actually imports at runtime, not the
full history of superseded iterations that were tried along the way.

## Model lineage

The current entry point is **`v46_4_basev51.py`** — an exact-score buy-card
generator that layers Polymarket edge detection and Kelly staking on top of
a scoreline model. It composes three layers:

```
v46_4_basev51.py            "buy card": staking / Kelly / tiering, Polymarket edges
├── v11_wcq_results_model.py    core Poisson / Dixon-Coles engine, trained on World Cup history
├── v51_combined_scoreline_model.py   v11 + v49's bivariate-NegBin correction,
│                                      with v39 (coverage) and v29 (tail-risk) outlier layers
└── v42_fotmob_market_edge_model.py   Polymarket fetch/classify/edge pipeline (reused as-is)
```

`v51_combined_scoreline_model.py` is the prediction source used for accuracy
(v11 + v49, plus additive v39/v29 outlier coverage); `v42` is reused only for
its Polymarket plumbing, not its own prediction stack. Both trace further
back through `v15`, `v20`, `v26`–`v36`, `v39_withbetterdata`, and a handful of
other versioned modules — all included here since the top-level script
imports them directly.

Everything in this repo is either the current model, a module it depends on
to run, or the evaluation harness used to validate it
(`backtest/audit_base_models.py` and its supporting walk-forward scripts —
see [`BASE_MODEL_AUDIT.md`](BASE_MODEL_AUDIT.md) for a full write-up of how
the base models compare against each other and the market, including known
gaps).

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
python v11_wcq_results_model.py \
  --team-a Germany \
  --team-b Curaçao \
  --model ensemble \
  --outdir outputs/outputs_germany_curacao
```

Use `--host-a` / `--host-b` for home advantage, `--knockout` for a knockout
match.

## Run the combined scoreline model (v51)

`v51` is a train/test backtesting script (it doesn't take `--team-a`/`--team-b`
directly — for a single match, use `v11` above or `v46_4_basev51` below).
It reads `data/worldcupsai.zip`, `data/current_team_features_2026.csv`,
`data/FIFAallMatchBoxData.csv`, `data/results.csv`, and `data/former_names.csv`
by default — all included in `data/`.

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
(log-loss, Brier, RPS), a Polymarket market benchmark, and bootstrap CIs.

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
v*.py            Model chain (see "Model lineage" above)
data/            Curated input data the model chain reads
backtest/        Cross-model evaluation harness
tests/           Unit tests (unittest discover -s tests)
```

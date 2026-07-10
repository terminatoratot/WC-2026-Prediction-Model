# WC 2026 Prediction Model

A Python model for predicting World Cup match results, scorelines, goals, and
Polymarket exact-score betting edges. It trains on men's World Cup history,
current team ratings, recent international results, FotMob current-form
data, and World Cup qualification form.

This repository is a curated slice of a larger research project: it contains
the current model and everything it actually imports at runtime, not the
full history of superseded iterations that were tried along the way.

## How the model works

The prediction stack is built in layers, each one correcting a specific,
named gap in the layer below it rather than replacing it wholesale. From the
ground up:

### 1. Base engine (`v11`) — goal rates, not a single classifier

Trains on men's World Cup history only (a curated match archive plus
`data/results.csv`) and reduces each match to two numbers: a goal rate
("lambda") for team A and one for team B. Everything else — win/draw/loss,
scorelines, totals — is derived from those two numbers, not fit separately.
A few specific mechanisms worth calling out because they're easy to miss
just skimming the file list:

- **The lambdas themselves come from an ensemble**, not one model: gradient
  boosting, random forest, and Poisson regression each produce a lambda
  estimate, and the ensemble blends them (`--model ensemble` vs. picking one
  directly, e.g. `--model poisson`).
- **Win/draw/loss is two-stage, not a single 3-way softmax.** A balanced
  logistic regression first answers "is this a draw?" (features: Elo
  difference, host status, group-vs-knockout stage, same-confederation flag),
  and only then is a winner decided conditional on "not a draw." Draws are
  treated as a structurally different outcome to predict, not just a third
  bucket of one softmax.
- **Current-strength correction is a multiplicative Elo-diff shrink**, applied
  *after* the raw ensemble lambda: `factor_a = exp(k * elo_diff)`,
  `factor_b = exp(-k * elo_diff)`, with the Elo difference clipped to
  ±3 before exponentiating. This nudges a team's long-run historical rate
  toward its current Elo standing without letting one wildly out-of-form
  Elo reading blow up the prediction.
- **Dixon-Coles correction fixes the low-score bias** that independent
  Poisson marginals have on their own. Independent Poisson systematically
  mis-prices the four cells where the two teams' goals interact most
  (0-0, 1-0, 0-1, 1-1); Dixon-Coles reweights exactly those four cells by a
  `tau` factor that's a function of both lambdas and a single correlation
  parameter (`rho = -0.08` here), then renormalizes the matrix.
- **Temperature smoothing** (factor `1.08`) pulls the final W/D/L
  probabilities slightly toward uniform, a conservative hedge against the
  ensemble being overconfident on any single match.
- Exact scorelines are *derived* from the (corrected) lambda pair via a
  score matrix, not fit as their own direct target — the primary objective
  the model is evaluated on is result log-loss/Brier/RPS, not top-1
  scoreline accuracy.
- A Kaggle box-score layer separately predicts match stats (shots, shots on
  target, possession, fouls, saves, cards) alongside the scoreline.
- Evaluated via **chronological expanding-window backtesting**: train on
  everything up to tournament N, test on tournament N+1, slide forward.
  Never trained and tested on the same window.

### 2. Scoreline correction (`v49`) — correlated, fat-tailed goals

Independent Poisson marginals carry two false assumptions: that the two
teams' goal counts are uncorrelated, and that each team's own goal count has
variance exactly equal to its mean. Neither holds — some matches are simply
more "chaotic" for both sides at once (end-to-end, error-prone, end up
5-4 rather than 2-1), and that correlation is invisible to independent
Poisson no matter how you tune the two lambdas individually.

`v49` replaces the scoreline matrix with a **bivariate Negative Binomial via
shared frailty**: draw one shared "match volatility" factor
`Z ~ Gamma(shape=r, scale=1/r)` (so `E[Z]=1`), then treat each team's goals as
`Poisson(lambda * Z)` *conditional on the same draw of Z*. Because both
teams see the same `Z`, a high-volatility draw pushes both team's goal counts
up together — inducing correlation and fatter tails on total goals — while a
low-volatility draw does the opposite. Integrating `Z` out analytically is
what turns each team's *marginal* distribution into a Negative Binomial. The
W/D/L probabilities from `v11` are left untouched; only the shape of the
scoreline matrix around them changes.

### 3. Additive outlier layers (`v29`, `v39` coverage) — never overwrite the Top-3

The model surfaces a Top-3 most-likely-scorelines list, which by
construction can't cover every match — some games land on a score outside
whatever three cells were highest-probability pre-match. The naive fix
(let a tail-risk rule *replace* the 3rd slot when it fires) trades away
real Top-3 accuracy to buy tail coverage. Both of these layers instead add a
**4th, additive "outlier" slot** and leave the Top-3 itself alone:

- **`v29` (tail-risk)** fires only when the model already expects a
  lopsided result: the favorite's win probability and its lambda both have
  to clear gates at the same time (e.g. win probability ≥ 0.66 *and*
  lambda ≥ 1.75 for the standard gate; ≥ 0.78 and ≥ 2.40 for an "extreme
  favorite" gate), plus a cap on the draw probability and a minimum gap
  between the two teams' lambdas. When both conditions hold, it proposes a
  bigger-margin blowout scoreline as the 4th slot instead of trusting the
  Top-3 to have already covered that possibility.
- **`v39`'s coverage outlier** is triggered by a different signal: total
  expected goals (`lambda_a + lambda_b`) exceeding the highest total already
  represented in the Top-3, plus a margin. When that happens, it picks the
  single highest-probability scoreline from the next total-goals band up
  (e.g. if the Top-3 tops out at a 3-goal match, it looks in the 4-goal band)
  as the outlier slot — a targeted fix for the specific case where the
  Top-3 systematically under-represents high-scoring games.

### 4. Current-form layers (`v28`, `v35`, `v36`, `v38`) — history isn't enough mid-tournament

Pure pre-tournament history stops being sufficient once matches start being
played — a team's actual 2026 form needs to feed back in without waiting for
the next model version:

- **`v28`** folds in observed World Cup 2026 results and FotMob signals as
  they become available.
- **`v35`** learns a small, deliberately shrunk transition table from
  75th-minute game state to final score, used only for a separate
  late-game-mutation outlier slot — it doesn't touch the base Top-3.
- **`v36`** adds a reusable FotMob xG/player/keeper form layer that reads
  `data/fotmob_*_clean.csv` directly at build time, so re-running the model
  after new matches are scraped automatically folds the new signal in with
  no code changes.
- **`v38`** is intentionally the smallest layer: one shrunk global
  multiplier that corrects a measured total-goals under-bracketing bias
  (the model was found to systematically predict too few goals), while
  leaving the W/D/L probabilities exactly as `v35` produced them.

### 5. Market layer (`market_edge.py`, built from `v42`) — read-only comparison

Fetches live Polymarket odds for exact-score and moneyline markets,
classifies them, and computes an edge for each priced outcome: model fair
probability minus the market-implied probability (de-vigged), further
reduced by an "executable edge" that accounts for actual order-book prices
rather than posted mid prices. A row only clears the `buy` bar when the
edge exceeds a minimum threshold (`≥ 0.015` by default) *and* the trade is
actually fillable at that edge, not just theoretically priced that way;
smaller edges get downgraded to a `watch` verdict instead. Market data only
flows one direction here — it's compared against the model's own
probabilities, never blended into them.

### 6. Combination (`v51`) — the actual prediction source

Takes `v11`'s W/D/L and `v49`'s corrected scoreline matrix, then adds *both*
the `v29` and `v39` outlier tabs independently, as two separate columns
rather than picking one. That lets a backtest directly answer "does the
tail-risk outlier add hits, does the coverage outlier add hits, does having
both add more than either alone" — instead of baking in an assumption about
which one is better. This combined output is what actually feeds the current
pipeline.

### 7. Buy-card generation (`v46_4_basev51`, current entry point)

Runs `v51` for the scoreline distribution, feeds it through `v42`'s
Polymarket pipeline to find priced edges, then converts those edges into an
actual staking plan:

- Each selected score is first funded to a **break-even floor** (the stake
  size at which a hit exactly covers the total outlay across the card), then
  any remaining budget ("surplus") is allocated across scores weighted by
  three separate priorities — value edge, outlier upside, and downside
  cover — each with its own tunable weight (`--value-surplus-weight`,
  `--outlier-surplus-weight`, `--cover-surplus-weight`).
- **Kelly sizing is shrunk, not applied raw.** Each row's theoretical Kelly
  fraction is multiplied by a shrink factor derived from `v42`'s own
  `staking_confidence` score (a composite of bucket depth, liquidity, market
  agreement, model stability, and book consistency) rather than a flat
  fraction — a score `v42` is less confident in gets a smaller bet, not just
  a smaller theoretical edge.
- Scores get classified as `VALUE`, `COVER`, or `OUTLIER_VALUE` and tiered
  accordingly, with stakes rounded to a fixed increment for a clean, postable
  card.
- Ships with audit outputs alongside the card itself: per-score selection
  reasoning, a hit-outcome tracking table, and a validation pass, so a
  finished card can be checked rather than just trusted.

## Model lineage

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

The walk-forward evaluation scripts used to validate the model chain
(`backtest/`) are included too.

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
python backtest/eval_v29_v36_completed_worldcup.py --outdir outputs/eval_v29_v36
```

Walk-forward evaluation against completed matches, with leak controls (no
model sees a match's own result before predicting it). `eval_v42_completed_worldcup.py`
and `eval_v36_v39_walkforward_no_leak.py` cover the later layers the same way.

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

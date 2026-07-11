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
state defaults to `--results-as-of latest`, which resolves to the latest
completed non-2026-World-Cup match in `data/results.csv` and excludes current
2026 World Cup finals rows to avoid leaking observed tournament matches. Change
that explicit cutoff with `--results-as-of YYYY-MM-DD`. V15 chooses its final
W/D/L label from the highest class probability instead of V13's separate draw
threshold.

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

## Current research path: V51 prediction and V46.4 execution

The actively maintained exact-score path is deliberately split in two:

1. `v51_combined_scoreline_model.py` estimates a scoreline distribution. It
   does not see market prices and does not size bets.
2. `v46_4_basev51.py` converts that distribution and a live exact-score order
   book into an auditable, mutually exclusive buy-and-hold card. It does not
   change the V51 probabilities.

That separation matters. A better probability model is not automatically a
better trading rule, and an attractive card backtest is not evidence that the
probability model itself is calibrated. Keeping the layers separate makes both
claims testable.

### Model construction

V51 starts from the V11 expanding-history model. Pre-match features are built
strictly in date order: Elo and rolling form are recorded before each match,
then updated with that match only after its feature row exists. The model uses
historical World Cup results, pre-cutoff non-World-Cup internationals, and a
pre-tournament team-feature snapshot. It predicts expected goals and a
scoreline matrix; no market price is an input to V51.

The score matrix is V49's shared-volatility bivariate Negative Binomial with a
Dixon-Coles correction at the low-score corner. In compact notation:

```text
Z ~ Gamma(shape=r, scale=1/r)
Home goals | Z ~ Poisson(lambda_home * Z)
Away goals | Z ~ Poisson(lambda_away * Z)
```

Integrating out `Z` gives a joint distribution with
`Cov(home goals, away goals) = lambda_home * lambda_away / r`. Unlike two
independent Poisson variables, it can represent matches where both teams are
simultaneously pulled into a high-event or low-event state. The default
`r = 25` is a calibrated dispersion parameter: it was estimated from residual
variation after an attack/defence Poisson fit to 2010+ international results,
then checked against World Cup-finals estimates. It is not a subjective tail
multiplier.

V51 preserves the base top three scorelines exactly. It may append, never
replace, up to two diagnostic coverage candidates:

- V39 adds the highest-probability scoreline above the top-three total-goal
  envelope when expected total goals exceed that envelope.
- V29 adds a favourite-tail scoreline only when its explicit win-probability,
  draw, expected-goal, and probability-floor gates are met.

These are additional candidate scorelines, not probability reweighting and not
extra confidence in the base forecast.

### Weight register and provenance

The repository has several different things called “weights.” They answer
different questions and should not be conflated.

| Layer | Parameter or rule | Default / live value | Provenance and purpose |
| --- | --- | --- | --- |
| Elo feature | `K = 24 * 1.25 * log(1 + goal_margin)` for margins above one goal | Code-defined | Standard Elo-style sequential rating update; a pre-match feature, not a bet-size weight. |
| Training recency | `max(0.5^(age_years / 16), 0.10)` | 16-year half-life; 10% floor | Applies more influence to recent history without dropping older World Cups. All sample weights are mean-normalized after construction. |
| Training prestige | Friendly / other / qualifier / continental finals | 20 / 30 / 40 / 50 | Uses the World Football Elo K-factor hierarchy as an event-importance prior. |
| World Cup training prestige | `wc_prestige_weight` | 600 | A validated V51 setting. The expanded pool contains roughly 47 non-World-Cup rows per World Cup row, so the World Cup weight is intentionally higher than the 60-point event-importance scale; it prevents the target competition from being numerically drowned out. |
| Qualification features | blend schedule | starts 2014; full from 2022 | Gradual feature inclusion, not a label or target weight. It avoids pretending older qualification data has the same coverage as modern data. |
| Score distribution | shared-volatility dispersion `r` | 25 | Residual calibration described above; controls joint over-dispersion, not the mean goal forecast. |
| Execution allocation | V46.4 tier weights and caps | stage-specific; below | Fitted only on historical, price-stamped card outcomes. They allocate a fixed card budget; they do not alter V51 probabilities. |

The optional volume-normalized prestige scheme is off by default. If enabled,
it sets each tournament tier's aggregate training contribution to fixed target
shares (World Cup 35%, continental finals 25%, qualifiers/Nations League 20%,
other 12%, friendlies 8%) rather than using flat per-match prestige. It is an
explicit ablation, not silently active in production.

### V46.4 card construction and the five stake buckets

Exact-score outcomes are mutually exclusive, so V46.4 treats a card as one
portfolio rather than a set of independent Kelly bets. Candidate rows must
pass price, probability, confidence, story-fit, and rank checks. They are then
assigned one of five allocation buckets:

| Bucket | Meaning | Relative surplus weight: group / knockout | Surplus cap: group / knockout |
| --- | --- | ---: | ---: |
| `VALUE_1` | Primary positive-edge or supported joint-Kelly anchor | 1.000 / 1.000 | 0.2575 / 0.2688 |
| `VALUE_2` | Second value allocation | 0.6369 / 0.5165 | 0.2674 / 0.2488 |
| `COVER_1` | Higher-probability portfolio coverage | 0.3402 / 0.0500 | 0.1917 / 0.1904 |
| `COVER_2` | Secondary coverage | 0.0000 / 0.2534 | 0.0685 / 0.2753 |
| `OUTLIER` | V51-designated V29/V39 additive scoreline | 0.3158 / 0.4710 | 0.1468 / 0.0833 |

Caps are fractions of the target card stake. If a cover has negative raw edge,
its `COVER_1`, `COVER_2`, or `OUTLIER` cap is additionally multiplied by
0.2474 in group stage or 0.4013 in knockout. This is a concentration control:
coverage can reduce card fragility, but it cannot consume the card merely
because it is a high-probability outcome. `VALUE_1` has a soft cap so any
unallocated remainder is still assigned and the completed card sums to the
target stake.

The first value selection is highest expected value. Additional value rows
prefer distinct score-path clusters (for example, a clean-sheet home win and a
both-teams-score home win) before falling back to another version of the same
scenario. This is a simple correlation-control rule, not a claim of a fully
estimated covariance matrix.

### Where the live tier weights came from

The stage-specific values above were produced by
`backtest/tune_v46_4_tiers.py` followed by
`backtest/split_tier_weights_by_stage.py`. The process is reproducible:

1. Build one historical V46.4 card per completed match from a score-sanitized
   FotMob snapshot and a prematch Polymarket exact-score price snapshot. Final
   score fields are blanked before card generation and used only to settle the
   simulated card.
2. Cache candidate scorelines, roles, model probabilities, prices, and
   realised score. The current stage split contains 75 group-stage and 22
   knockout matches.
3. Search ten allocation parameters: four relative weights, two cover caps,
   the negative-edge cap multiplier, and three value/outlier caps. The search
   runs a coarse grid followed by 1,000 seeded local random trials around the
   strongest grid settings.
4. Reject parameter sets that fail hard risk constraints: break-even hit rate
   at least 0.92, value/outlier share of positive profit at least 0.65, maximum
   drawdown at most 0.40 on a 100-unit reference bankroll, 4–5 selected bets
   per card, and cover leakage no greater than 15% of stake.
5. Rank surviving sets by a stability-adjusted utility over five chronological
   folds: mean fold utility minus half a standard deviation. Utility combines
   ROI, break-even hit rate, value-profit share, drawdown, and cover leakage.

This is a parameter-selection exercise on a small historical sample, not an
independent proof of expected return. In particular, the 22-match knockout
sample is below the tuner’s own comfort threshold and should be treated as
directional. A reviewer should inspect `cached_training_rows.csv`,
`parameter_search_results.csv`, `fold_results.csv`, and the equity curve in
the corresponding `outputs/v46_4_basev51_tier_optimization*` folders before
relying on any historical ROI figure.

### Leakage controls and evaluation protocol

`backtest/eval_v51_knockout_walkforward.py` is the main prediction evaluation
for completed 2026 World Cup matches. For every target kickoff it rebuilds V51
using only information knowable before that kickoff:

- `worldcupsai.zip` supplies historical World Cups through 2022 only.
- Current team features are a static 2026-06-10 pre-tournament snapshot.
- Expanded `results.csv` training excludes `tournament == FIFA World Cup`, so
  completed 2026 finals do not enter the non-World-Cup pool.
- FBref World Cup rows are filtered to prior, completed matches only. The
  target matchup is explicitly removed across adjacent calendar dates, and
  same-evening matches are excluded unless they ended at least 3.5 hours before
  the target kickoff.
- Team aliases are canonicalized and the evaluator asserts that a target row
  cannot slip through under a spelling variant.

The evaluator reports directional result accuracy, untouched base top-three
exact-score accuracy, and top-three-plus-additive-outlier accuracy. It makes no
Polymarket calls. Market and card performance must be measured separately with
historical prematch prices; neither is used to score V51 prediction accuracy.

Run the import check before the expensive evaluation:

```bash
python -m unittest tests.test_v51_imports
```

Then run the leakage-controlled walk-forward evaluation locally:

```bash
caffeinate -i env V51_WORKERS=2 MPLCONFIGDIR=.matplotlib_cache \
  .venv/bin/python backtest/eval_v51_knockout_walkforward.py
```

On a machine with more cores, increase `V51_WORKERS` only after checking memory
and CPU saturation. Each model fit is already internally parallel, so setting
workers equal to logical CPU count can make the evaluation slower or unstable.

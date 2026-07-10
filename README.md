# World Cup V46.4 / V51 Model

V46.4 is a pre-match exact-score decision pipeline. It:

1. trains the V51 football model;
2. produces a probability distribution over exact scores;
3. reads live Polymarket prices;
4. compares model probability with executable market price;
5. selects a small mutually exclusive scoreline portfolio; and
6. allocates and validates stakes.

The public entry point is `v46_4_basev51.py`. Market prices are used for
valuation and staking only. They are not blended into V51's model probability.

## Repository layout

```text
v46_4_basev51.py
├── v51_combined_scoreline_model.py
│   ├── core_engine.py
│   ├── feature_layers.py
│   └── market_edge.py
├── market_edge.py
└── data/
```

| File | Responsibility |
|---|---|
| `v46_4_basev51.py` | candidate filtering, portfolio Kelly, card selection, stake allocation, caps, validation, CLI |
| `v51_combined_scoreline_model.py` | V11/V49 prediction plus additive V29 and V39 coverage scorelines |
| `core_engine.py` | bundled V11, V13-V27, and V49 base-model modules |
| `feature_layers.py` | bundled V28-V38 current-form, tail-risk, and game-state modules |
| `market_edge.py` | bundled V39/V42 market discovery, pricing, consistency, and confidence modules |
| `data/` | curated model inputs required by the pipeline |

The bundled files register their embedded modules in `sys.modules` under the
original names. Imports such as `v11_wcq_results_model` and
`v42_fotmob_market_edge_model` therefore work without shipping dozens of
separate historical files. Each embedded module has its own namespace.

## End-to-end data flow

```text
historical results + rankings + qualifiers + current FotMob data
                              |
                              v
                chronological Elo/form features
                              |
                              v
        goal-rate ensemble -> lambda_a, lambda_b
                              |
                              v
       bivariate Negative Binomial + Dixon-Coles grid
                              |
                              v
       result calibration + V29/V39 additive coverage
                              |
                              v
                 V51 model-only probabilities
                              |
            +-----------------+-----------------+
            |                                   |
            v                                   v
  Polymarket exact-score prices       moneyline/totals/BTTS/spreads
            |                                   |
            +-----------------+-----------------+
                              v
       edge, executable edge, confidence, joint Kelly
                              |
                              v
       VALUE/COVER/OUTLIER selection and stake card
```

## 1. Chronological strength and form

### Elo state

Every team starts at 1500. Before match `t`, the expected score for team A is

```math
E_A = \frac{1}{1 + 10^{-(R_A-R_B)/400}}.
```

The observed score is `S_A = 1` for a win, `0.5` for a draw, and `0` for a
loss. Ratings update after the pre-match feature row is created:

```math
R_A' = R_A + K(S_A-E_A),
```

```math
R_B' = R_B + K((1-S_A)-(1-E_A)).
```

The base K-factor is 24. A win by more than one goal applies

```math
m(d)=1.25\log(1+|d|), \qquad K=24m(d),
```

while a one-goal margin or draw uses `m(d)=1`.

### Ordering fix

The live V51 builder concatenates World Cup, other international, and FBref
rows, then explicitly runs

```python
train_matches = train_matches.sort_values("date").reset_index(drop=True)
```

before constructing rolling features. This is necessary because Elo and form
are stateful. Without the global sort, the state would follow source-block
order instead of match-date order.

### Rolling features

For each team, the model uses the latest 12 prior matches to calculate goals
for, goals against, goal difference, win rate, and draw rate. A match's final
score is appended to team history only after its feature row has been emitted.
The feature set also includes stage, host status, confederation relationship,
qualifier profiles, current team ratings, and available match-event fields.

## 2. Goal-rate model

The base engine estimates expected goals `lambda_a` and `lambda_b`. In ensemble
mode, the core regressors are:

| Regressor | Default ensemble weight |
|---|---:|
| Random forest | 0.25 |
| Histogram gradient boosting | 0.20 |
| Poisson regression | 0.15 |
| LightGBM, when installed | 0.15 |
| XGBoost, when installed | 0.15 |
| CatBoost, when installed | 0.20 |

Available weights are normalized when optional libraries are absent. Separate
ensembles predict A goals, B goals, and goal difference.

Training rows receive exponential recency weights with a 16-year half-life and
a raw floor of 0.10:

```math
w_t^{recency}=\max\left(0.5^{age_t/16},0.10\right).
```

This is multiplied by tournament-prestige weight and normalized to mean one
before fitting.

### Current-strength correction

Let `Delta` be the difference in the current-strength feature, clipped to
`[-3,3]`. The raw goal rates are adjusted by

```math
\lambda_A' = \lambda_A e^{0.10\Delta},
\qquad
\lambda_B' = \lambda_B e^{-0.10\Delta}.
```

### Goal-difference blend

The independently predicted goal difference is blended at weight 0.30 with
the difference implied by the two goal-rate regressions. The total expected
goals is held fixed while the two rates are redistributed to match the blended
difference. Final goal rates are clipped to `[0.15, 4.5]`.

## 3. Exact-score distribution: V49

V49 replaces independent Poisson score counts with a shared-volatility
bivariate Negative Binomial construction:

```math
Z \sim \operatorname{Gamma}(r,1/r),
```

```math
A\mid Z \sim \operatorname{Poisson}(\lambda_A Z),
\qquad
B\mid Z \sim \operatorname{Poisson}(\lambda_B Z).
```

After integrating out `Z`, the score probability is

```math
P(A=i,B=j)=
\frac{\Gamma(i+j+r)}{i!j!\Gamma(r)}
\left(\frac{r}{\lambda_A+\lambda_B+r}\right)^r
\left(\frac{\lambda_A}{\lambda_A+\lambda_B+r}\right)^i
\left(\frac{\lambda_B}{\lambda_A+\lambda_B+r}\right)^j.
```

The default dispersion is `r=25`. This produces

```math
\operatorname{Var}(A)=\lambda_A+\frac{\lambda_A^2}{r},
\qquad
\operatorname{Cov}(A,B)=\frac{\lambda_A\lambda_B}{r}.
```

Smaller `r` means fatter, more correlated tails. As `r` approaches infinity,
the construction approaches independent Poisson counts.

### Dixon-Coles low-score correction

V49 then multiplies the four low-score cells by `tau`, using `rho=-0.08`:

```math
\tau(0,0)=1-\lambda_A\lambda_B\rho,
```

```math
\tau(0,1)=1+\lambda_A\rho,
\qquad
\tau(1,0)=1+\lambda_B\rho,
```

```math
\tau(1,1)=1-\rho.
```

The matrix is normalized after applying the correction.

## 4. Result probabilities

Win, draw, and loss mass is first summed from the exact-score grid. In ensemble
mode, that distribution receives a 14% classification-model blend. A
temperature of 1.08 flattens the result probabilities:

```math
\tilde p_k = \frac{p_k^{1/1.08}}{\sum_j p_j^{1/1.08}}.
```

A dedicated draw model then contributes 75% of the final draw estimate; the
score-grid draw estimate contributes 25%. The final draw probability is clipped
to `[0.05,0.55]`. Remaining mass is split between the two win sides in their
previous proportion. The exact-score grid is reweighted to match these final
result totals.

## 5. V51 coverage composition

V51 leaves the highest-probability Top 3 unchanged. It may append two distinct
coverage candidates:

### V29 tail-risk candidate

V29 considers a larger favorite score only when its directional gates fire.
Defaults include:

- favorite win probability at least 0.66;
- extreme-favorite win probability at least 0.78;
- draw probability no greater than 0.27;
- favorite goal rate at least 1.75, or 2.40 for the extreme gate;
- goal-rate gap at least 0.75;
- total goal rate at least 2.45;
- candidate probability at least 0.008 and at least 12% of the reference
  probability; and
- no more than seven goals for the winning side.

V51 harvests the candidate that V29 would use but appends it instead of letting
V29 replace the third-ranked score.

### V39 total-envelope candidate

V39 compares `lambda_a + lambda_b` with the maximum total goals represented in
the Top 3. If expected total goals exceed that displayed ceiling, it selects the
highest-probability score from the next total-goal band. This candidate is also
additive.

The resulting list contains the untouched Top 3 plus zero, one, or two unique
coverage scores.

## 6. Market layer: V42

### Exact-score prices

For a binary Yes/No market with posted prices `c_yes` and `c_no`, the de-vigged
Yes probability is

```math
q_{yes}=\frac{c_{yes}}{c_{yes}+c_{no}}.
```

For model fair probability `p`, V42 reports:

```math
\text{edge}=p-q_{yes},
```

```math
\text{executable edge}=p-c_{yes}-u,
```

```math
\text{expected return}=\frac{p}{c_{yes}}-1,
```

where `u` is the uncertainty buffer. A positive model edge is insufficient if
the executable edge at the posted or order-book price is non-positive.

The binary Kelly fraction shown for a single outcome is

```math
f^*=\max\left(0,\frac{p-c}{1-c}\right).
```

V46 does not size the final card by treating these single-outcome fractions as
independent bets.

### Cross-market checks

Moneyline, totals, both-teams-to-score, and spread markets are converted into
constraints on the score grid. Iterative proportional fitting creates a
diagnostic market-implied matrix while staying close to the totals-anchored
prior. This matrix is used for agreement and consistency diagnostics, not to
replace the model's fair probabilities.

Each exact score receives a staking-confidence multiplier based on:

- scoreline reliability bucket;
- market liquidity;
- agreement with the market-implied matrix;
- stability relative to the base model matrix; and
- consistency with moneyline, total, BTTS, and spread prices.

## 7. V46.4 portfolio selection

Exact-score claims are mutually exclusive: at most one scoreline wins. V46.4
therefore solves them as one portfolio rather than as independent Kelly bets.

### Joint Kelly buy set

For a trial set of scores, define

```math
P=\sum_i p_i, \qquad C=\sum_i c_i,
```

and the miss-state threshold

```math
\theta=\frac{1-P}{1-C}.
```

Candidates are ordered by `p_i/c_i` and admitted while

```math
\frac{p_i}{c_i}>\theta.
```

The unconstrained mutually exclusive Kelly fraction for an admitted score is

```math
f_i=\max(0,p_i-c_i\theta).
```

Each `f_i` is multiplied by the score's staking-confidence shrink factor. If
the shrunk vector exceeds the per-match Kelly tranche budget, V46.4 re-solves a
capped KKT system and removes any score whose constrained fraction becomes
non-positive. The default tranche cap is 5% of bankroll when expressed as a
fractional Kelly budget.

### Candidate roles

Candidates passing rank, confidence, story-fit, and price checks receive one of
three roles:

| Role | Meaning |
|---|---|
| `VALUE` | positive raw edge or supported by the active joint-Kelly set |
| `COVER` | high-probability portfolio coverage without sufficient value support |
| `OUTLIER_VALUE` | V51's designated additive outlier score |

`Any Other Score` is excluded by default. Low-score exclusions and directional
hedges can be stage-dependent. Multiple VALUE scores are diversified across
outcome-path clusters where possible, so the card does not fill every value
slot with variants of the same underlying match state.

### Break-even floor and surplus

Let `T` be the target card outlay and `c_i` the entry price. The stake required
for score `i` to return the whole card outlay if it wins is

```math
b_i=T c_i.
```

V46.4 starts each selected score at

```math
s_i^{base}=\max(s_{min},b_i),
```

where `s_min` is the minimum executable order size. Remaining budget is

```math
S=\max\left(0,T-\sum_i s_i^{base}\right).
```

Surplus is distributed by role or stage-specific tier weights. Negative-edge
covers have surplus caps. Stakes are then rounded to the configured increment
while preserving the target total when a valid card exists.

For stake `s_i` at price `c_i`:

```math
\text{gross payout if hit}=\frac{s_i}{c_i},
```

```math
\text{net card profit if hit}=\frac{s_i}{c_i}-T.
```

The validation pass checks total stake, break-even coverage, minimum orders,
negative-edge cover counts, tier caps, and required role counts. A validation
failure is a no-buy condition rather than a silently malformed card.

## 8. Leakage controls

- Rolling Elo and form features are created before the current match result is
  added to team state.
- The live V51 training pool is globally sorted by date before stateful feature
  generation.
- Historical evaluation uses expanding time windows: test-tournament outcomes
  are not training inputs for that fold.
- Market prices affect valuation and staking diagnostics, not model fair
  probability.

## Data inputs

The repository includes the curated files needed by V46.4/V51:

| Input | Purpose |
|---|---|
| `worldcupsai.zip` | historical World Cup archive |
| `results.csv` | broader international and qualification results |
| `current_team_features_2026.csv` | current team-level features |
| `FIFAallMatchBoxData.csv` | historical World Cup event/box data |
| `fcratings_top50_worldcup2026.csv` | current player-component ratings |
| `player_ratings_international.csv` | current player and squad-name inputs |
| `world_cup_2026_declared_squads.csv` | declared tournament squads |
| `fbref_*.csv` | World Cup and international match additions |
| `fotmob_*.csv` | current tournament facts, players, lineups, keepers, substitutions, and events |
| `former_names.csv` | canonical team-name mapping |

Generated outputs, caches, and raw scrape intermediates are not committed.

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/terminatoratot/WC-2026-Prediction-Model.git
cd WC-2026-Prediction-Model
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the complete dependency chain:

```bash
python -c "import v46_4_basev51; print('V46.4/V51 ready')"
```

## Running the model

### Automatic Polymarket discovery

```bash
MPLCONFIGDIR=.matplotlib_cache .venv/bin/python v46_4_basev51.py \
  --team-a Germany \
  --team-b Paraguay \
  --knockout \
  --auto-polymarket \
  --outdir outputs/germany_paraguay
```

### Explicit Polymarket event and executable order book

```bash
MPLCONFIGDIR=.matplotlib_cache .venv/bin/python v46_4_basev51.py \
  --team-a Germany \
  --team-b Paraguay \
  --knockout \
  --polymarket-event-slug fifwc-ger-par-2026-06-29 \
  --fetch-clob-orderbook \
  --outdir outputs/germany_paraguay
```

### Offline model smoke run

This builds the model and writes diagnostics without requiring a live market:

```bash
MPLCONFIGDIR=.matplotlib_cache .venv/bin/python v46_4_basev51.py \
  --team-a Germany \
  --team-b Paraguay \
  --knockout \
  --allow-empty-polymarket \
  --outdir outputs/germany_paraguay_offline
```

All CLI controls:

```bash
.venv/bin/python v46_4_basev51.py --help
```

## Outputs

A normal run writes files under the requested `--outdir`, including:

| Output | Contents |
|---|---|
| `v46_4_buy_card.csv` | final selected scores, roles, prices, edges, and stakes |
| `v46_4_selection_debug.csv` | candidate-by-candidate inclusion and rejection diagnostics |
| `v46_4_hit_outcomes.csv` | payout and net-card result if each selected score wins |
| `v46_4_basev51_summary.json` | complete machine-readable run summary |
| `plots/v46_4_buy_card.png` | card visualization |
| `plots/v46_4_hit_outcomes.png` | per-hit payoff visualization |
| `plots/v46_4_score_grid.png` | score probability and market-decision grid |
| `_v51_source/` | underlying V51 prediction artifacts used by the card |

## Interpretation

- `model_probability` is the V51 fair probability.
- `market_price` is the entry cost per unit payout.
- `raw_edge = model_probability - market_price` for the executable card layer.
- `joint_kelly_fraction` is the final confidence-shrunk, tranche-capped
  mutually exclusive Kelly fraction.
- `execution_stake` is the rounded currency stake after card-level allocation.
- `COVER` does not mean positive expected value. It is controlled portfolio
  coverage and can be capped or rejected.
- An empty executable card means the current prices did not support a valid
  purchase under the selected constraints.

Exact-score probabilities are low and sensitive to lineup, injury, market,
and data changes. The outputs are estimates and staking diagnostics, not a
guarantee of profit.

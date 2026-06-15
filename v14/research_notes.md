# Paper notes: arXiv 2403.07669

Bunker, R., Yeung, C., Fujii, K. (2024), *Machine Learning for Soccer Match Result Prediction*
(book chapter / survey, arXiv:2403.07669). Read in full (41 pages). Section/page references below
are to the chapter's internal numbering (Sec. 1.1 – 1.6) and the extracted PDF pages.

## Summary

- The survey's headline finding (Abstract; Sec. 1.6 Conclusions): **gradient-boosted tree models —
  especially CatBoost — applied to soccer-specific *ratings* such as pi-ratings are currently the
  best-performing approach on datasets that contain only goals as the match feature.** Best published
  result on the benchmark is **CatBoost + pi-ratings: RPS 0.1925, accuracy 0.5582** (Razali et al.,
  Table 1.2).
- Ratings (Elo, pi-ratings, Berrar, GAP) are discussed as *features* feeding ML models, not as
  standalone predictors — using ratings as model features beats using the rating directly to pick the
  higher-rated team (Sec. 1.4.1).
- The dominant benchmark dataset is the **Open International Soccer Database** (2017 Soccer Prediction
  Challenge): 216,743 matches, 52 leagues, 35 countries, 2000–2017 — **but it contains only goals**
  (no player data, no betting odds, no match stats). Sec. 1.2.2.
- For **player-level data**, the survey repeatedly points to the **European Soccer Database (Kaggle)**
  and **FIFA video-game ratings (FIFA Index / SoFIFA)**, plus event-data approaches (VAEP) and
  **plus-minus** player ratings. Key empirical result (Arntzen & Hvattum [4]): **combining player +
  team ratings beats either alone.** Sec. 1.4.3 / 1.4.4.
- Evaluation: the field favours **RPS** (ordinal-aware), with **log loss / Ignorance score** argued by
  some to be preferable (local, strictly proper), and **Brier**. Accuracy is still common but
  discouraged for 3-way. **Chronological splits are mandatory** — random CV leaks the future. Sec. 1.5.
- The paper gives a **usable pi-ratings formula** (Sec. 1.4.1.2) and usable Elo, Berrar, GAP, streak,
  and form formulas. The pi-ratings update equations are reproduced below with one caveat: the PDF text
  extraction garbles some subscripts, so the *structure* of the formula is reliable but the exact
  symbol pairing should be cross-checked against Constantinou & Fenton [30] before coding.

## Datasets (with relevance to player data)

From Table 1.1 (Sec. 1.2.1) plus discussion. Marked **[PLAYER]** where the dataset carries
player-level ratings/attributes, **[TEAM]** for team ratings, **[ODDS]**, **[EVENT]**, **[XG]**.

| Dataset | Contents | Player data? | Source / URL | Coverage |
|---|---|---|---|---|
| **European Soccer Database (Kaggle)** | 25,000 matches, **10,000 players** in 11 European leagues. **Player and team attributes from the EA Sports FIFA video game.** Also lineups & formations (x,y), betting odds from up to 10 bookmakers, and for >10,000 matches: events (goal types, possession, corners, crosses, fouls, cards). | **[PLAYER][TEAM][ODDS][EVENT]** YES — richest single public source | kaggle.com/datasets/hugomathien/soccer | 2008–2016 |
| **FIFA Index** | Player and team ratings from the EA Sports FIFA video game | **[PLAYER][TEAM]** YES | fifaindex.com | current/seasonal |
| **soccerdata (Python scraper)** | Scrapes Club Elo, ESPN, FBref, FiveThirtyEight, Football-Data.co.uk, **SoFIFA**, WhoScored | **[PLAYER]** via SoFIFA + FBref stats | github.com/probberechts/soccerdata | varies |
| **Wyscout event data (Pappalardo et al.)** | Spatiotemporal **event** data, 1,941 matches | **[EVENT]** player actions — supports VAEP / aggregated player ratings | doi.org/10.6084/m9.figshare.c.4415000.v5 | top-5 EU leagues, **EURO 2016, 2018 World Cup** |
| **StatsBomb open data** | Event log data + lineup/match metadata | **[EVENT]** player-action level | github.com/statsbomb/open-data | varies (incl. internationals) |
| **Open International Soccer DB** | Match results only (Sea, Lge, Date, HT, AT, HS, AS, GD, WDL); 216,743 matches | NO — goals only; the 2017 Challenge benchmark | osf.io/kqcye/ | 19/03/2000–21/03/2017, 52 lg / 35 countries |
| **World Football Elo Ratings** | Current **Elo of national teams** | NO (team-level, **international**) | eloratings.net | current |
| **Football Database** | Current Elo of club teams | NO (team-level) | footballdatabase.com | current |
| **engsoccerdata (R)** | English + other EU + US MLS + South African league results | NO (goals/results) | github.com/jalapic/engsoccerdata | historical |
| **understat.com** | **xG** (computation opaque) | **[XG]** team/shot-level | understat.com | top leagues |
| **football-data.co.uk** | Betting odds (multiple providers) + some match stats | **[ODDS]** | football-data.co.uk | many leagues/seasons |
| Betting websites | Odds, sometimes extra features | **[ODDS]** | football-data.co.uk, betfair-datascientists.github.io, oddsportal.com | varies |

Other player/team value sources named in prose (Sec. 1.4.3–1.4.5): **transfermarkt-style club market
values / transfer budgets** ([109]); **Football Manager** player ratings (argued to be "arguably
better" than FIFA, Sec. 1.4.3); **UEFA five-year club coefficients** ([109], Sec. 1.4.4).

**Most relevant for THIS project (international, World Cup).** Most listed sets are club-league. For a
*national-team* model the directly usable player sources are: **SoFIFA / FIFA Index** (FIFA national-team
squads have ratings), **StatsBomb open data** and **Wyscout** (cover EURO 2016 + 2018 WC, so
international event data exists), and **transfermarkt squad/player market values** (the project already
uses *aggregate* squad value — see cross-check).

## Methodology & rating systems (incl. pi-ratings math)

### Model families compared (Sec. 1.3)
- **Statistical / goals models**: independent & bivariate Poisson, dependent Poisson, negative
  binomial, Weibull, generalized extreme value; ordered logistic regression. **Dixon–Coles [37]**
  extends Maher's Poisson [81] for incomplete data and temporal variation. Bivariate / Double Poisson
  remain strong baselines ([78], [61]).
- **ML / probabilistic-graphical**: Logistic Regression, ANN, Bayesian Networks, Decision Trees,
  k-NN, Naïve Bayes, **Random Forest**, SVM; fuzzy/genetic methods.
- **Ensembles** (Sec. 1.3.4): boosting (**XGBoost**, **CatBoost**, also LightGBM) vs bagging
  (**Random Forest**). On goals-only data, boosting + ratings wins; **on datasets with match stats,
  Random Forest is competitive with and sometimes beats boosting** ([5],[107],[1],[43]). CatBoost
  highlighted for handling categoricals (ordered target encoding, leak-safe) and **well-calibrated
  win/draw/loss probabilities**.
- **Deep learning** (Sec. 1.3.5): LSTM/RNN (sequence nature), TabNet (tabular DL — **TabNet + pi-ratings
  RPS 0.1956**), CNN-on-tabular-as-image. Verdict: few studies, no clear win over boosting; CatBoost
  beat a CNN in [82].

### Feature engineering (Sec. 1.4)
- **Rating features** (best signal — see below).
- **Recency / rolling aggregation**: average features over the last N matches. No fixed N — Buursma
  found 20 best, **Berrar used the last 9** ("recency features": attacking strength, defensive
  strength, home advantage, opposition strength). Simple averaging ignores recency → **exponential
  time weighting** ([37],[78],[60]) is recommended instead.
- **Match stats** (in-play, only known after the match → must be aggregated over history): shots
  on/off target, corners, possession, passes, tackles, fouls, cards, etc.
- **xG** as a *less rare* alternative target to goals (Sec. 1.4.2 / 1.5.1).
- **Player stats** (Sec. 1.4.3): FIFA video-game player ratings; **plus-minus** ratings; **VAEP**
  (Valuing Actions by Estimating Probabilities, from SPADL event data) aggregated to player then team
  level; player "hot hand"/hot-shoe form.
- **Team stats** (Sec. 1.4.4): team ratings from FIFA/Football Manager; UEFA coefficients; aggregating
  player ratings into role-based (FW/MF/DF/GK) team indicators (Carpita et al. — 33 FIFA player
  features → 7 indicators → role indicators); inter-player **chemistry** (Bransen & Van Haaren, VAEP-based);
  passing-network indicators.
- **External features** (Sec. 1.4.5, known pre-match, no aggregation needed): home advantage, venue,
  travel, **player availability/injuries**, average squad age, **number of international players**,
  **market values / transfer budgets**, weather, referee, social-media sentiment.
- **Feature selection** (Sec. 1.4.6): filter / wrapper / embedded; RF embedded importance;
  CFS/ReliefF; sequential forward selection.

### Rating systems — math

**Elo** (Sec. 1.4.1.1). Start 1500. E-step expected score
`E_a = 1 / (1 + 10^(-(R_a - R_b)/400))`. U-step `R_a ← R_a + K·(S_a - E_a)` with S ∈ {1, 0.5, 0}.
K typically 20–32. **Goal-based K (Hvattum & Arntzen [65]): `K = K0·(1 + δ)^λ`** where δ = absolute
goal margin. Home advantage (Ryall & Bedford [104]): add a term h_{a,b} inside the exponent:
`E_a = 1 / (1 + 10^(-((R_a - R_b) + h_{a,b})/400))`. Bradley–Terry is the theoretical foundation of Elo:
`P(a beats b) = v_a / (v_a + v_b)`, and Elo relates to BT strength via `R_a = 400·log10(v_a)`, i.e.
`v_a = 10^(R_a/400)`.

**pi-ratings** (Sec. 1.4.1.2, Constantinou & Fenton [30]) — the survey's recommended rating, paired
with CatBoost gives SOTA. Each team has **separate home and away ratings**; overall rating is their
mean: `R_team = (R_home + R_away) / 2`. Each team starts at 0; a rating is the team's strength relative
to the *average* team. For a match, predict a goal difference from the home team's home rating and the
away team's away rating, compare to actual, and update both ratings of both teams with **separate
learning rates**.

Expected goal difference vs an average opponent at ground g (home or away):
`ŷ_g = (10^(|R|/c)) - 1`, with **c = 3** (paper uses b=10 / c=3 notation; the constant base is 10 and
the divisor/scale is 3). Predicted matchup diff `ŷ = ŷ_home(home team) - ŷ_away(away team)`; if a
rating is negative the expected outcome is negated.

Error between expected and actual goal diff: `e = |GD_actual - GD_predicted|`.

**Damping function** (the key idea — limits the impact of blowouts):
`ψ(e) = c · log10(1 + e)`, with **c = 3**.

Update (schematically; learning rates λ for own-ground update, γ for cross-ground propagation):
```
R_home_team(home) ← R_home_team(home) + λ · ψ(e) · sign(error)
R_home_team(away) ← R_home_team(away) + γ · (ΔR_home_team(home))
```
and analogously for the away team's away/home ratings. **Caveat:** the PDF extraction scrambles the
exact subscript/learning-rate pairing in these four update lines — treat the `R = (R_h + R_a)/2`
aggregation, the `10^(|R|/3) - 1` expected-diff, the `e = |actual - expected|` error, and the
`ψ = 3·log10(1+e)` damping as reliable; verify the precise update wiring against Constantinou & Fenton
[30] before implementing. Constantinou's Challenge variant [28] additionally weighted the
**result** (W/D/L) more than the goal margin and added a **team form** term; outcome probabilities are
produced via an **ordered logit** over the rating features.

**Berrar ratings** (Sec. 1.4.1.3): logistic functions of offensive/defensive ratings predict expected
goals for each side: `ĝ_h = α_h / (1 + exp(-a_h·(o_h - d_a) - b_h))` (and symmetric for away), then
update o/d ratings toward the goal residuals. XGBoost + Berrar ≈ pi-ratings performance.

**GAP ratings** (Sec. 1.4.1.4, Wheatcroft [118]): pi-ratings extended to predict **non-rare match
statistics** (shots, corners) rather than goals — four ratings per team (home/away × attack/defence),
updated toward (actual stat − expected), parameters fit by least squares; ignore first/last 6 games of
a season. Motivation: goals are too rare to be informative; shots/corners are better.

**Plus-minus player ratings** ([64], Sec. 1.4.3): rate a player by how the team performs (e.g. goals)
with vs without them on the pitch — usable to inject *player* signal into a team model.

**VAEP** ([36], Sec. 1.4.2/1.4.3): value each on-ball action via Δ P(score)/P(concede) from SPADL
event data; aggregate per player → player rating → team rating.

## Evaluation metrics

(Sec. 1.5.3). Lower is better for all the proper scores; 0 = perfect, 1 = wrong (RPS/Brier bounded).

- **Accuracy** = correct / total. Common but discouraged for 3-way (ignores probabilities & ordering).
- **Brier score (BS)** — multi-class `(1/n)Σ_i Σ_c (p̂_ic - y_ic)^2`; strictly proper but **not**
  ordinal-aware; deemed inappropriate for the Challenge for that reason.
- **Ranked Probability Score (RPS)** — the competition metric. `RPS = (1/(r-1)) Σ_{i=1}^{r-1}
  (Σ_{j=1}^{i}(p_j - o_j))^2` over the *cumulative* distributions, r = #outcomes (3). **Strictly
  proper AND sensitive to distance** (treats home-win as "closer" to draw than to away-win). Averaged
  over matches. Benchmark numbers in Table 1.2 are all RPS.
- **Ignorance score / Log loss (IGN)** — `-(1/n)Σ (y·log2 p + (1-y)·log2(1-p))`. Strictly proper and
  the **only local** rule discussed. Wheatcroft [119] argues IGN/log-loss is actually *more*
  appropriate than RPS (locality; distance-sensitivity disputed).
- **RMSE** — for numeric targets (goals / goal margin).
- Scoring-rule properties (Sec. 1.5.3.6): **proper / strictly proper**, **local**, **sensitive to
  distance**. Of the set, Brier+RPS+IGN are strictly proper; only IGN is local; only RPS is
  distance-sensitive. Use a *strictly proper* rule so the model is incentivised to report true probs.

**Validation best practice** (Sec. 1.5.4, flagged "Important"): **preserve temporal order** — never
use random k-fold CV, because future matches would leak into predicting past ones. Suspiciously high
accuracy (80–90% on a 3-way problem where 45–55% is normal) is a red flag for this leak. Use
expanding-window / time-series CV; older seasons matter less (roster/strength drift) unless
player-level features capture squad changes. Target can be 3-class W/D/L, goal margin (numeric), or
per-team goal counts; draws are the hard class (some studies drop/merge them — discouraged as it
inflates apparent accuracy).

## Cross-check vs existing v11/v13 code

| Paper recommendation | Already done? | Where / how |
|---|---|---|
| Poisson goals model | **Yes** | `poisson_score_matrix()` v11:84 |
| **Dixon–Coles** low-score correction | **Yes** | `apply_dixon_coles_adjustment()` v11:99 |
| **Elo ratings** (expected-score logistic) | **Yes** | `elo_expected()` v11:805; built in `build_rolling_features` v11:863-865; `elo_diff`/`elo_prob_a` features in `make_features` v11:1502-1505 |
| Elo with home advantage | **Partial** | No additive home term in Elo itself; home modelled separately via `host_a/host_b/host_diff` features (v11:1497-1500) and host args |
| Goal-based / dynamic K-factor | **Partial** | v13 has fixed `live_elo_k=24` (v13:25); no goal-margin-scaled K (`K0(1+δ)^λ`) |
| **Gradient-boosted trees** (XGBoost/LightGBM/CatBoost) | **Yes (optional)** | `_named_regressors`/`_named_classifiers` v11:1104,1166; optional imports in try/except (CLAUDE.md); CatBoost is among them |
| Ensemble blending of learners | **Yes** | weighted blend per target, `--model ensemble` |
| **Random Forest** baseline / comparison | **Yes** | `--comparison-models ... rf ...`; `run_model_comparison_backtest` |
| Recency / rolling aggregation of form | **Yes** | `build_rolling_features` rolling gf/ga/gd/win/draw over last 12 (v11:870) |
| **Exponential time weighting** of training data | **Yes** | `build_year_recency_weights()` v11:158; `--recency-half-life-years` |
| **Chronological / expanding-window validation** | **Yes** | `chronological_backtest`; CLAUDE.md "expanding window"; no random CV |
| Proper scoring: **log loss** | **Yes** | `safe_log_loss()` v11:1825, reported per backtest |
| Proper scoring: **Brier** | **Yes** | `brier_score_3way()` v11:1812; over/under Brier v11:2704 |
| Proper scoring: **RPS** (ordinal, the field's primary metric) | **No** | Not computed anywhere — only log-loss + Brier |
| Negative-binomial / overdispersed goals | **Yes (v13)** | `_negative_binomial_pmf` / `_overdispersed_score_matrix` v13:28,46 |
| Draw-handling / calibration | **Yes** | dedicated `_fit_draw_model` v11:1285; v13 draw redistribution v13:69 + `draw_decision_threshold` |
| Aggregate **team strength prior** (FIFA rank, **squad market value**, WC history, recent form) | **Yes (team-level)** | `build_current_strength_table()` v11:989 uses `fifa_points/rank_pre_tournament`, `squad_total_market_value_eur`, goals/wins/losses last 4y, WC titles/participations; applied at predict time as multiplicative λ correction `_apply_current_strength_correction` v11:1570 |
| Leak-free handling of future-strength info | **Yes** | current strength kept off historical rows, applied only at predict time (CLAUDE.md; v11:1602) |
| **pi-ratings** (home/away split, goal-diff update with log damping) | **No** | Project uses Elo, not pi-ratings |
| Berrar / GAP ratings | **No** | — |
| **Player-level data** (FIFA/SoFIFA player ratings, plus-minus, VAEP, lineups, injuries) | **No** | Only aggregate squad market value exists; no per-player features, no lineups, no availability/injury inputs |
| xG as target/feature | **No** | Goals only |
| Betting odds as feature/baseline | **No** | Not ingested |
| Interpretability (SHAP / RF importance) | **No** | Not surfaced in outputs |

Net: the project **already follows most of the survey's "do this" list** — Elo, gradient boosting +
RF comparison, Dixon–Coles, exponential recency weighting, leak-free expanding-window backtests, proper
scoring (log loss + Brier), draw-specific calibration, and an aggregate FIFA/market-value/history
strength prior. **The clear gaps where v14 can add value: (1) player-level data, (2) pi-ratings,
(3) RPS as a reported metric, (4) optionally xG / betting-odds features.**

## Recommendations for v14 (adding player data)

Ordered by expected payoff vs effort, anchored to specific paper findings.

1. **Add aggregated player-rating team features (highest ROI).** The strongest player-data evidence in
   the survey is Arntzen & Hvattum [4]: **player ratings + team ratings combined beats either alone**,
   and FIFA video-game player ratings are repeatedly used ([25],[35],[92],[124],[24]). Concretely, for
   each 2026 squad pull **SoFIFA / FIFA Index** national-team player overalls, aggregate into
   role-based indicators (GK/DEF/MID/FWD means + top-XI mean + best-N strength) per the Carpita et al.
   recipe, and feed the *difference* between teams as new columns. This slots directly into the existing
   `build_current_strength_table()` pattern (v11:989) — it is already a z-scored multi-column prior, so
   add player-aggregate columns there and/or as `make_features` inputs. Keep the leak discipline:
   attach as a *current* prior applied at predict time, not on historical training rows.

2. **Implement pi-ratings as an additional rating feature alongside Elo.** Survey SOTA is
   **CatBoost + pi-ratings**. Add a `pi_rating_home/away` builder mirroring `build_rolling_features`'s
   Elo loop (v11:861-865): maintain per-team home/away ratings, update with the log-damped goal-diff
   rule (`ŷ=10^(|R|/3)-1`, `e=|GD−ŷ|`, `ψ=3·log10(1+e)`). Expose `pi_diff` as a feature next to
   `elo_diff`. Verify exact update wiring against Constantinou & Fenton [30] (PDF subscripts are
   garbled). This is purely additive — it does not disturb the Poisson/DC pipeline.

3. **Report RPS in backtests and observed-match eval.** Cheap and aligns the project with the field's
   primary metric. Add an `rps_3way(actual, probs)` next to `brier_score_3way` (v11:1812) using the
   cumulative-distribution formula, and include `mean_result_rps` in the backtest / comparison summary
   dicts (v11:1972, 2002, 2064) and in `evaluate_observed_wc2026_matches.py`.

4. **Add player-availability / external features.** The survey lists injuries, key-player availability,
   number of international players, and squad age as promising external features that are *known
   pre-match* (no aggregation needed, Sec. 1.4.5) and were specifically flagged by Constantinou [28]
   and Berrar et al. [8] as the next improvement. For internationals these are realistic to source
   manually per tournament and would feed `make_features` directly.

5. **Optional / longer-term.** (a) Replace/augment the goals target with **xG** (less rare → more
   informative, Sec. 1.4.2) if an international xG source is found. (b) **VAEP / plus-minus** player
   ratings from **StatsBomb open data / Wyscout** (which cover EURO 2016 + 2018 WC) for a
   performance-derived alternative to FIFA ratings. (c) **Betting odds** as a benchmark column (hard to
   beat — Baboota & Kaur, [100]) to sanity-check calibration, not necessarily as a feature.
   (d) **SHAP / RF feature importance** in outputs for interpretability.

**Concrete datasets worth downloading:** SoFIFA / FIFA Index (national-team player ratings — primary
for player data), `soccerdata` Python scraper (wraps SoFIFA + FBref + Club Elo), StatsBomb open-data
and Wyscout (international event data incl. World Cup, for VAEP), European Soccer Database on Kaggle
(reference for the FIFA-attribute + lineup + odds schema), and transfermarkt for per-player market
values (the project already has *squad* totals).

**Does the paper give a usable pi-ratings formula?** Yes — Sec. 1.4.1.2 gives the home/away split,
the `R = (R_h + R_a)/2` aggregation, the expected-goal-difference `10^(|R|/3) - 1`, the error
`e = |actual − expected|`, and the **log damping `ψ(e) = 3·log10(1+e)`**, which are enough to implement.
The only soft spot is the exact pairing of the four update lines' subscripts and learning rates, which
the PDF extraction garbled and which should be confirmed against the original Constantinou & Fenton [30].

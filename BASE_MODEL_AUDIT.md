# Base Model Audit: v11, v15, v29, v36, v39_betterdata, v42

**Date:** 2026-07-06
**Scope:** Methodology, data hygiene, and evaluation hygiene across the base scoreline models, evaluated on a common denominator for the first time.
**Harness:** `audit_base_models.py` (new) — reused the leak-control patterns from `eval_v29_v36_walkforward_no_leak.py` / `eval_36_v39_walkforward_no_leak.py`, extended with proper scoring rules, a Polymarket market benchmark, and bootstrap CIs. Sanity-checked against three previously published numbers (v29, v36, v39_betterdata top-3 accuracy) — reproduced within 1 hit of 82, validating the harness.

Full outputs: `outputs/base_model_audit/` (`model_summary.csv`, `per_match_metrics.csv`, `pairwise_deltas.csv`, `pairwise_deltas_vs_market_covered_only.csv`, `plots/`).

---

## Headline results (82 completed matches, walk-forward, no leak)

| Model | Exact log-loss | Result log-loss | Brier | RPS | Outcome acc. | Top-3 acc. |
|---|---|---|---|---|---|---|
| v11 | 2.937 | 0.896 | 0.527 | 0.172 | 59.8% | 29.3% |
| v15 | 2.862 | 0.836 | 0.492 | 0.155 | 64.6% | 40.2% |
| **v29** | **2.847** | 0.829 | 0.488 | 0.153 | 64.6% | **41.5%** |
| v36 | 2.853 | 0.831 | 0.489 | 0.153 | 64.6% | 37.8% |
| v39_betterdata | 2.856 | 0.831 | 0.488 | 0.152 | 64.6% | 41.5% |
| v42_production | 2.849 | **0.827** | **0.486** | **0.151** | **65.9%** | 39.0% |
| market (all 81) | 3.743¹ | 0.802 | 0.468 | 0.143 | 71.6% | 34.6%¹ |
| market (66/81 quote-covered) | 2.500 | — | — | — | — | — |

¹ The market's raw exact-score log-loss and top-3 accuracy are measured against only the 16 highest-probability scorelines Polymarket quotes per match; 15/81 matches had an actual score outside that list, and those get a harsh floor penalty. See Finding 1 — the covered-subset number (2.500) is the fair comparison.

Plots: `01_metric_comparison.png`, `02_pairwise_deltas.png`, `03_calibration.png`, `04_goals_clean_sheets.png`, `05_layer_ablation.png`.

---

## Findings

### 1. [Evaluation hygiene] The market beats every model on every proper-scoring metric — and no model beats it on a fair exact-score comparison either

On result-level metrics (log-loss, Brier, RPS, outcome accuracy) the market is unambiguously best, by a decisive margin. On exact-score log-loss, the naive number looks like models "beat" the market (2.85 vs 3.74) — but that's an artifact: Polymarket only quotes ~16 scorelines per match, and any actual result outside that list gets a floor penalty (`-ln(1e-4) ≈ 9.2`) that doesn't apply to the models (which always have a full probability matrix). Restricting to the 66/81 matches the market actually quotes, its exact-score log-loss is **2.500** — better than every model's 82-match average of ~2.85. A bootstrap on this fair subset shows all six models' deltas vs. market are positive (market better) but not statistically significant at n=66 (see `02_pairwise_deltas.png`, bottom panel).

**Read:** there is no evidence in this sample that any base model has an edge over the market, at either the result or exact-score level. This matches and reinforces the staking-layer conclusion reached earlier (v46.4's backtested ROI was likely a pricing/ranking artifact, not real edge).

### 2. [Methodology] The FotMob/betterdata/market layers added after v29 show no measurable improvement over v29 — on some metrics, a regression

The layer-ablation waterfall (`05_layer_ablation.png`) is the most actionable finding:

- v11 → v15 → v29: real, monotonic improvement (log-loss 2.937 → 2.862 → 2.847; top-3 29.3% → 40.2% → 41.5%).
- v29 → v36 (+FotMob current-form): **both metrics get worse** (log-loss 2.847 → 2.853; top-3 41.5% → 37.8%).
- v36 → v39_betterdata (+Database joins): recovers back to v29's level, does not exceed it (top-3 41.5%, tied with v29; log-loss 2.856, slightly worse than v29).
- v39_betterdata → v42_production (+coverage-outlier, +market layer): log-loss improves marginally (2.849) but top-3 drops again (39.0%).

**v29 is the best or tied-best model on this sample**, despite being the simplest of the post-v20 lineage (a rules-based tail-risk selector on top of v28, no FotMob/Database/CatBoost inputs at all). Every layer added since has cost engineering and data-maintenance effort (see Finding 4) without a demonstrated accuracy payoff on a fair test.

This also **resolves a concern raised in initial exploration**: the only previously published v42 evaluation (n=42, no walk-forward, no betterdata) showed v42 as the worst-performing model. Under this properly-constructed walk-forward + full-stack evaluation, v42_production is mid-pack — comparable to v29/v39_betterdata, and actually best on result-level metrics. The earlier concern was a measurement artifact of an incomplete old eval, not a real deficiency.

### 3. [Data hygiene] `v39_withbetterdata`'s feature build is reading stale Database CSVs — 14 completed matches missing

`versions/build_combined_fotmob_database_features.py` hardcodes `data/Database - *.csv` (no suffix), last modified **Jun 30**. Newer refreshes (`data/Database - *-CS-2.csv`, **Jul 5** — 5 days later) sit right next to them and are never referenced anywhere in the codebase. The old file has 76 rows; the new one has **90** — 14 additional completed-match rows. `analysis/v39_withbetterdata_latest/` (consumed by both `v39_withbetterdata` and `v42_production`) is timestamped Jun 30 17:19, confirming it was built from the stale files.

**Impact:** every v39_betterdata/v42_production prediction in this audit (and in production) is missing roughly the tournament's most recent week of match data from its "better data" layer — likely masking some real signal, though given Finding 2 the ceiling on that signal looks low anyway.

### 4. [Methodology gap] v11 has no protection against 2026 World Cup score leakage; v15+ does

`v15_catboost_model.py`'s `load_international_results()` explicitly filters out `tournament == "FIFA World Cup" & year == 2026` before resolving `results_as_of="latest"` — verified live: `resolved_as_of` = `2026-06-10` (the day before kickoff), zero 2026 WC rows in the loaded training set. This guard is inherited correctly by every downstream model (v29, v36, v39, v42 all route through it).

**v11's own `build_from_zip()` has no such filter** — it takes a raw `results_csv` path with no date-based exclusion. `data/results.csv` contains 72 WC2026 finals rows including final scores. Left as-is, v11 (run standalone, e.g. via its own CLI) can train on the actual answer key for any already-completed 2026 WC match. This audit worked around it by feeding v11 a copy of `results.csv` truncated to pre-2026-06-11 (see `pre_tournament_results_csv()` in `audit_base_models.py`), but the underlying script has no such safeguard for a normal user.

### 5. [Evaluation hygiene] The metric regime break (v11–v27 vs v28+) meant nobody has watched calibration/log-loss for 15+ versions

Confirmed from exploration: v11–v27 were evaluated with proper scoring (log-loss, Brier, RPS) on samples of 8–20 matches; from v28 onward, every evaluation switched to "top-3 coverage" only (does the actual score appear in the top 3 displayed?) with no proper-scoring companion. Top-3 coverage says nothing about whether the *probabilities* feeding the staking layer's edge calculations are calibrated — which is the only thing that matters for betting. This audit is the first proper-scoring evaluation of v29 through v42, on any sample size.

### 6. [Minor, resolved during audit] Answer-key vs. staking-cache match-count mismatch is benign

The audit's answer key (`fotmob_completed_matches_observed_schema_current.csv`, 82 matches) and the earlier v46.4 staking-layer cache (83 matches) differ by 5 match IDs. Traced to two benign causes, not a bug: (a) 3 matches decided by extra-time/penalties (Germany–Paraguay, Netherlands–Morocco, Belgium–Senegal) are correctly excluded from the 90-minute-exact-score answer key but were present in the older staking cache; (b) one match (Switzerland–Algeria) completed after the staking cache was frozen; (c) one match (USA–Bosnia, `4653709`) fails inside `v46_4/v46_5_optimizedbuys.py` regardless of answer key (already known from the staking-layer work).

### 7. [Weak/inconclusive] Draw probabilities look miscalibrated across every model and the market — but the test is underpowered and possibly confounded

Per-class reliability curves (`03_calibration.png`) show win probabilities (both team A and team B) are reasonably well calibrated with a mild *underconfidence* bias (empirical win rate slightly exceeds predicted, for every model including the market). Draw probabilities show the opposite pattern in this sample — but the draw bucket has few matches (quintile buckets of ~16 each, draws are a minority class), so the curve is noisy. It may also be **partially a sample-construction artifact**: knockout matches that are drawn after 90 minutes proceed to extra time/penalties and are excluded from the 90-minute answer key entirely (see Finding 6), which mechanically removes real draws from the evaluated sample without removing them from what models were built to predict. Not enough data here to distinguish genuine miscalibration from this exclusion effect.

### 8. [Weak/inconclusive] All models mildly under-predict total goals and over-predict clean sheets

Every model predicts a mean total of 2.72–2.80 goals vs. an actual 2.93 (a 0.13–0.21 goal gap), and 51–52% clean-sheet probability vs. an actual 47.6% rate. Directionally consistent across every model (suggesting a shared structural tendency, e.g. shrinkage toward historical/pre-tournament base rates), but the gaps are small relative to what 82 matches can resolve with confidence.

---

## Fix shortlist (prioritized, with effort estimates)

| # | Fix | Effort | Why |
|---|---|---|---|
| 1 | Point `build_combined_fotmob_database_features.py` at the `-CS-2.csv` refreshes (or auto-select the newest file matching the pattern) and rebuild `analysis/v39_withbetterdata_latest/` | **S** | Concrete, quantified staleness (14 matches); mechanical fix |
| 2 | Add the same `results_as_of` / current-WC exclusion filter to `v11_wcq_results_model.py`'s `build_from_zip()` that v15 already has | **S** | v11 currently has zero protection against training on the answer key; one function to port |
| 3 | Re-evaluate whether v36→v42's added complexity (FotMob form blend, Database joins, coverage-outlier, market layer) is worth keeping as the *scoreline* base model, given v29 matches or beats all of them here | **M** | Real engineering/maintenance cost (Finding 3 exists *because* this stack has more moving parts) without demonstrated accuracy gain on this sample; needs a larger out-of-sample test (more WC2026 matches as they complete) before a final call, but the prior should shift toward "v29 is good enough" |
| 4 | Adopt `audit_base_models.py`'s proper-scoring metrics (log-loss, Brier, RPS) as a standing part of the eval suite for any future model version, not just top-3 coverage | **M** | Restores 15+ versions' worth of missing calibration visibility; harness already built and validated |
| 5 | Document the answer-key vs. staking-cache match-count discrepancy (Finding 6) somewhere near the staking-layer scripts | **S** | Purely to save the next person from re-investigating a non-bug |
| 6 | Note in a comment/doc: v46.x's actual production recipe (v36 base → V39CoverageOutlier → V39BetterData) should be the one benchmarked going forward, not a bare `v42_fotmob_market_edge_model.py` run without betterdata — the old n=42 eval undersold it | **S** | Prevents re-litigating Finding 2's "resolved concern" with the wrong recipe again |

### Not worth fixing right now

- **Draw calibration (Finding 7)** — plausibly confounded by knockout-draw exclusion, and n is too small to separate signal from noise. Would need the pre-ET 90-minute scores for the 3 excluded matches (likely recoverable from FotMob) re-added to the answer key before this is worth another look.
- **Total-goals / clean-sheet bias (Finding 8)** — consistent direction but small magnitude; not actionable without a much larger sample.
- **The market's superior performance (Finding 1)** — this isn't a "fixable" gap in the models; it's the ceiling. Consistent with the staking-layer conclusion: the realistic expectation for this whole system is that it matches the market, not beats it, and the betting layer's job is capturing the rare/temporary mispricings, not a persistent edge in the base model.

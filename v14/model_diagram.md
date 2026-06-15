# v14 — player-data match model: design, diagram & math

This is the logical + mathematical overview for v14. It extends the v11 engine
(via the v13 live-signal wrapper) with **individual player data**, following the
recommendation of arXiv 2403.07669 (see `research_notes.md`): combine player-level
ratings with the existing team-level signals (the paper reports player + team
ratings beat either alone).

## 0. Where v14 sits

```
 v11 engine  ──wrapped by──►  v13 (live Elo, overdispersion, draw recal)
                                      ──wrapped by──►  v14 (player data)
```
v14 adds player information through **two channels** and otherwise leaves the
proven pipeline (Poisson → Dixon–Coles → draw model) untouched.

## 1. Data → player features

Source: `data/player_ratings_international.csv` (compact extract of the FIFA
dataset; one pre-edition snapshot per `fifa_version`, national-squad players only,
identified by non-null `nation_position`). Point-in-time: each match uses the FIFA
edition whose snapshot pre-dates kickoff.

For a team `t` in edition/year `y`, from its players' `overall` (o), `value_eur`
(v), `age`, and `nation_position` (→ line ∈ {GK, DEF, MID, ATT}):

```
gk_t   = max overall among GKs
def_t  = mean of top-4 DEF overalls
mid_t  = mean of top-4 MID overalls
att_t  = mean of top-3 ATT overalls
top11_t      = mean overall of the 11 highest-rated squad players
squad_t      = mean overall of the top-23
star_t       = mean of top-3 overalls (squad-wide)
depth_t      = mean overall of ranks 12–23
value_log_t  = log1p( Σ value_eur over top-23 )
age_t        = mean age of the top-11
```

Two variants share this exact schema, differing only in **which players** feed the
aggregation:
- **Variant A (squad):** all squad members of nation `t` in edition `y`.
- **Variant B (lineup):** only the FIFA-designated XI (`nation_position` in the
  standard 11 starting slots); falls back to top-11-by-overall if the XI is
  incomplete.

## 2. Player features → match feature vector

Football scoring is a **line mismatch** problem: a team scores by pitting its
attack against the opponent's defence. For teams `a` (vs) `b` we form deltas:

```
att_vs_def_a = att_a − def_b        # pressure A puts on B  → raises λ_a
att_vs_def_b = att_b − def_a        # pressure B puts on A  → raises λ_b
overall_diff = top11_a − top11_b
gk_diff      = gk_a − gk_b
value_diff   = value_log_a − value_log_b
star_diff    = star_a − star_b
depth_diff   = depth_a − depth_b
age_diff     = age_a − age_b
```
These columns are appended to the existing feature vector built by
`make_features()` / `build_rolling_features()`.

## 3. Channel 1 — retrained ML features

The player deltas join `self.feature_cols` and the training matrix. The existing
λ-regressors, goal-difference regressor and result classifier are **retrained** so
they learn player effects directly. Coverage caveat: only WC editions with FIFA
ratings (≈2014/2018/2022) get real player values; earlier editions get neutral
(zero-delta) imputation, and the model's exponential **recency weighting** already
favours the data-rich recent tournaments.

## 4. Channel 2 — predict-time correction (no retrain needed)

Reuses the existing `build_current_strength_table()` + `_apply_current_strength_
correction()` seam. The player aggregates (`top11`, `att/def/mid`, `value_log`)
are z-scored into the composite strength `S_t`, and the per-team difference shifts
the expected goals multiplicatively — identical math to the current model:

```
Δ      = S_a − S_b
λ_a ← λ_a · exp(+k · Δ)
λ_b ← λ_b · exp(−k · Δ)         # k is the existing current_strength_k = 0.10
```

This channel works even where Channel 1 has no historical player data, because it
only needs the two teams' current ratings.

## 5. Downstream (unchanged)

```
corrected λ_a, λ_b
   → Poisson score matrix
   → Dixon–Coles low-score adjustment
   → result probabilities (win/draw/loss)
   → dedicated draw model override
   → v13: live-Elo nudge, negative-binomial overdispersion, draw recalibration
   → final result probs + scoreline matrix + xG + derived markets
```

## 6. End-to-end diagram

```
 data/player_ratings_international.csv ──┐
 (FIFA snapshot, national squads)        │  build_player_features.py
                                         ▼
        squad-aggregate table (A)  ┐   lineup/XI table (B) ┐
                                   └───────────┬───────────┘
                                               │  per (team, edition) features
                                               ▼
            ┌──────── v14_player_model(variant=A|B) ─────────┐
 team feats │ feature vector = team signals ⊕ player deltas   │
 (Elo,form, │     │                                           │
  ranking)  │     ├── Channel 1: retrained λ / GD / result    │
            │     └── Channel 2: exp(±k·Δstrength) on λ        │
            │                 │                                │
            │     λ_a, λ_b ────▼ Poisson → Dixon–Coles         │
            │        → result probs → draw model → v13 layer   │
            └─────────────────────┬──────────────────────────┘
                                  ▼
            result probs + scoreline matrix + expected goals
                                  ▼
   evaluate_headtohead.py : Euro + Copa América, out-of-sample
   metrics: adjusted confidence ↑, individual goal difference ↓,
            accuracy, RPS, log-loss   →   A vs B vs v13 baseline
```

## 7. Evaluation math (the new test method)

For each test match with actual result and score `(g_a, g_b)`:

- **Adjusted confidence** (↑ better): rank the three outcomes {A-win, draw, B-win}
  by predicted probability; let `r` be the rank of the *actual* outcome
  (`r=1` if it was the model's top pick). Then
  `adj_conf = P(actual outcome) / r`.
- **Individual goal difference** (↓ better): with the model's most-likely scoreline
  `(p_a, p_b) = argmax` of the score matrix,
  `igd = |p_a − g_a| + |p_b − g_b|`  (e.g. predict 1–1, actual 0–2 → 2).
- Context: accuracy, ranked probability score (RPS), log-loss.

Reported as a table comparing **Variant A**, **Variant B**, and the **v13 baseline**
(no player data), naming the winner on each metric.

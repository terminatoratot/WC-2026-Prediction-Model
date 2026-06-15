# v14 — match prediction with individual player data

v14 extends the v13 live-signal model (which itself wraps the v11 engine) with
**individual player data** from the FIFA/SoFIFA national-team ratings, following
the recommendation of arXiv 2403.07669 (see `research_notes.md`). Player ratings
are turned into per-team line strengths (GK / DEF / MID / ATT, top-11, squad value)
and used to shift expected goals **asymmetrically** — a team's attack is pitted
against the opponent's defence.

See `model_diagram.md` for the full logic + math, and `research_notes.md` for the
paper findings.

## Do I need to train it first?

**No separate training step.** One call builds everything:

```python
import v14_player_model as v14
model, _ = v14.build_from_zip(str(v14.DATA_DIR / "worldcupsai.zip"), variant="lineup")
```

This (a) trains the base v11 ensemble + v13 layer on World Cup history, and (b)
fits the player-impact layer (a PoissonRegressor) on ~1,257 international matches,
all in that one call. Each prediction then retrains nothing.

The only prerequisites are the data files, **which already exist** in `../data/`:
`player_ratings_international.csv`, `squad_player_features.csv`,
`lineup_player_features.csv`. You only need to rebuild them if the underlying FIFA
extract changes:

```bash
python v14/build_player_features.py    # regenerates squad_/lineup_player_features.csv
```

### Refreshing the player ratings (EA Sports FC source)

`player_ratings_international.csv` holds national-squad snapshots per FIFA/EA-FC
edition. Editions 15–23 come from the FIFA-23 Kaggle extract; **edition 26** comes
from the **EA Sports FC 26** dataset (`rovnez/fc-26-fifa-26-player-data`, SoFIFA
snapshot 2025-09-19) and is the active snapshot for the 2026 World Cup. To re-pull
FC 26 (or a newer edition) and rebuild everything:

```bash
python v14/extract_fc26.py            # downloads via kagglehub, adds edition 26
python v14/build_player_features.py   # rebuilds the feature tables
```

`extract_fc26.py` keeps the ~28 EA-licensed national teams with their real call-up
squad + designated XI, and for every other nation builds a proxy squad from the
top-26 players of that nationality by `overall` (so smaller WC2026 sides are still
covered). The pre-FC26 extract is backed up to
`data/player_ratings_international_fifa15-23.csv`.

## Predict a new match

```bash
# Recommended variant is "lineup" (see results below)
python v14/v14_player_model.py --team-a Spain --team-b "Cabo Verde" --variant lineup

# Optional: --match-date YYYY-MM-DD picks the point-in-time FIFA edition
#           (default = latest player data, FC 26). --knockout for a knockout tie.
#           --auto-host sets home advantage for the 2026 hosts (Canada/Mexico/USA);
#           or set it explicitly with --host-a / --host-b. --outdir / --no-plots.
python v14/v14_player_model.py --team-a Brazil --team-b Argentina \
  --variant lineup --match-date 2021-07-10
```

Variants: `lineup` (FIFA starting XI), `squad` (whole national squad), `none`
(plain v13 baseline, no player data).

This writes the **same output suite as v11** into `--outdir` (default
`outputs_v14_<a>_<b>`, auto-suffixed so runs don't overwrite):
`single_match_prediction.json`, `scoreline_probabilities.csv`,
`scoreline_probabilities_top.csv`, and a `plots/` folder with
`result_probabilities`, `top_scorelines`, `scoreline_heatmap`, `goal_spread`,
`total_goals`, `over_under`, and `event_expectations` PNGs. The console also prints
win/draw/loss probabilities, expected goals, the top scoreline, and a
`player_adjustment` block with the per-team log-multipliers applied (and
`covered: False` when a team has no player data — see coverage note).

## Evaluate the model (head-to-head A/B test)

Out-of-sample on **UEFA Euro + Copa América** matches (not in the WC training set),
using only pre-kickoff data. Compares baseline vs squad vs lineup with your two
metrics plus accuracy / RPS / log-loss:

```bash
python v14/evaluate_headtohead.py
# writes v14/headtohead_eval/{summary.csv, per_match_predictions.csv, winners.json}
```

Metrics:
- **Adjusted confidence** (↑): `P(actual outcome) / rank-of-actual-outcome`.
- **Individual goal difference** (↓): `|pred_a − act_a| + |pred_b − act_b|` on the
  most-likely scoreline (e.g. predict 1–1, actual 0–2 → 2).
- Context: accuracy, ranked probability score (RPS), log-loss.

### Result (187 covered Euro/Copa matches)

| model | adj. confidence ↑ | indiv. goal diff ↓ | accuracy ↑ | RPS ↓ | log-loss ↓ |
|---|---|---|---|---|---|
| baseline (v13) | 0.3347 | **1.791** | 0.4920 | 0.2088 | 1.0419 |
| squad | 0.3540 | 1.840 | 0.4866 | 0.2048 | 1.0280 |
| **lineup** | **0.3545** | 1.807 | 0.4920 | **0.2037** | **1.0241** |

**The `lineup` variant is recommended:** it wins adjusted confidence, RPS, and
log-loss, and ties accuracy. Baseline is marginally better on individual goal
difference (exact scoreline), so player data helps the *probabilistic* quality
most. Both player variants beat the baseline on RPS and log-loss.

## WC2026 group stage (real matches played so far)

Out-of-sample on the WC2026 group games in
`../data/wc2026_observed_matches_from_screenshots.csv`, predicted with the latest
edition (FC 26):

```bash
python v14/evaluate_wc2026.py
# writes v14/wc2026_eval/{summary.csv, per_match_predictions.csv, winners.json}
```

### Result (12 matches, **12/12 player-covered** with FC 26)

| model | adj. confidence ↑ | indiv. goal diff ↓ | accuracy ↑ | RPS ↓ | log-loss ↓ |
|---|---|---|---|---|---|
| baseline (v13) | 0.2997 | 2.167 | 0.417 | 0.1958 | 1.0306 |
| **lineup** | **0.3376** | **1.750** | **0.500** | **0.1920** | **1.0026** |

With the FC 26 data the **lineup variant wins every metric** on the 2026 games.
(With the old FIFA-23 extract only 1/12 matches were covered, so the player layer
was effectively dormant for 2026 — see git history.)

## Coverage limitation

EA Sports FC licenses ~**28 national teams** with full squads (`nation_position`)
per edition; v14 supplements these with **proxy squads** (top-26 by `overall` per
nationality) so most international sides get a player adjustment. Player
adjustments only apply when **both** teams are covered at the relevant edition;
otherwise v14 falls back to plain v13 (`player_adjustment.covered` is `False`).
Historical editions span 15–23 (2014–2023); **edition 26 (FC 26, 2025-09)** covers
the 2026 World Cup. Euro 2024 / Copa América 2024 fall in the gap between the FIFA
23 and FC 26 snapshots and remain uncovered. See `../data/PLAYER_DATA_RELEVANCE.md`.

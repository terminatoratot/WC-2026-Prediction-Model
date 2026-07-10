# World Cup V46.4 / V51 Model

This repository contains one runnable model: `v46_4_basev51.py`.

It predicts World Cup scorelines with V51, compares those probabilities with
Polymarket exact-score prices, and produces a V46.4 buy card with stake and
selection diagnostics.

## Setup

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/terminatoratot/WC-2026-Prediction-Model.git
cd WC-2026-Prediction-Model
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Verify that the complete model chain loads:

```bash
python -c "import v46_4_basev51; print('V46.4/V51 ready')"
```

## Run

Example live Polymarket run:

```bash
MPLCONFIGDIR=.matplotlib_cache .venv/bin/python v46_4_basev51.py \
  --team-a Germany \
  --team-b Paraguay \
  --knockout \
  --auto-polymarket \
  --outdir outputs/germany_paraguay
```

Use `--polymarket-event-slug SLUG` when automatic event discovery does not
find the correct market. Add `--fetch-clob-orderbook` when executable order
book quotes are required. Run this for every available option:

```bash
.venv/bin/python v46_4_basev51.py --help
```

## Files

```text
v46_4_basev51.py                V46.4 card selection, staking, and CLI
v51_combined_scoreline_model.py V51 prediction composition
market_edge.py                  V39-V42 market and coverage modules
feature_layers.py               V28-V38 current-form and scoreline modules
core_engine.py                  V11-V27 and V49 model modules
data/                           Curated runtime inputs
requirements.txt                Python dependencies
```

The older version modules are intentionally bundled inside
`core_engine.py`, `feature_layers.py`, and `market_edge.py`. They register the
original module names at import time, so V46.4 can call the full version chain
without shipping dozens of separate `vNN_*.py` files.

Generated files are written under `outputs/` and are not committed.

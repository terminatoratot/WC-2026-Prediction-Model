# Run V51 in GitHub Codespaces

Create or rebuild a Codespace on the `v13` branch. The container installs the
Python dependencies automatically from `requirements.txt`.

Verify the model chain before starting the expensive evaluation:

```bash
python -m unittest tests.test_v51_imports
```

Run the leakage-safe walk-forward evaluation:

```bash
python backtest/eval_v51_knockout_walkforward.py
```

Results are written to `outputs/v51_walkforward/`. Parallelism is controlled
by `WORKERS` near the top of `backtest/eval_v51_knockout_walkforward.py`.

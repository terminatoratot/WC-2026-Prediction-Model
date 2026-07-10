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

## Check and use the available cores

Check what the Codespace actually exposes before running:

```bash
nproc
lscpu | grep -E 'CPU\(s\)|Core|Thread'
free -h
```

If `nproc` reports 32, start with `WORKERS = 6` or `WORKERS = 8` in
`backtest/eval_v51_knockout_walkforward.py`. Do **not** set it to 32: every
model fit already uses multiple CPU threads internally, so 32 simultaneous
fits will oversubscribe the machine and can run slower or exhaust memory.

Use `htop` in a second terminal while the first few matches run. Increase from
6 to 8 only if CPU use is well below 100% and memory has plenty of headroom;
drop to 4 if the Codespace starts swapping or jobs are killed. Do not set
`LOKY_MAX_CPU_COUNT=1`, because that forces the model fits to be single-core.

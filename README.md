# Does Walk-Forward Validation Predict Out-of-Sample Performance?

A reproducible, controlled study of walk-forward optimization (WFO), the **Walk-Forward Efficiency
Ratio** (WFER = sum OOS PnL / sum IS PnL) with its popular 0.8/0.5/0.3 thresholds, anchored vs
rolling windows, window-length design, naive k-fold CV and CPCV-lite — all evaluated against a
simulated strategy family with a **known, time-varying true Sharpe surface** (stationary / drifting
optimum / abrupt break / no-edge null), so the true forward Sharpe of every deployed parameter is
computable exactly.

This grew out of a [marketmaker.cc](https://marketmaker.cc) blog post
("Walk-Forward Optimization: The Only Honest Strategy Test"); the experiments here test that post's
quantitative claims honestly — including the ones that do not survive.

## Reproduce everything

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_all.py            # full run -> results/results.json + records*.csv
python -m wfo_experiments.figures    # -> paper/figures/*.pdf (4 vector figures)
```

Everything is deterministic given the seeds in `scripts/run_all.py` (no wall-clock or unseeded
randomness). `--quick` runs a small batch for a smoke check.

## Layout

```
wfo_experiments/
  model.py        # DGP: known time-varying Sharpe surface (4 regimes) + correlated returns
  procedures.py   # single split, anchored/rolling WFO, naive k-fold, CPCV-lite; WFER bookkeeping
  simulate.py     # one experiment end-to-end, Monte-Carlo batches, window-design sweep
  analysis.py     # Q1 WFER diagnostic quality, Q2 anchored-vs-rolling, Q3 windows, Q4 ranking
  figures.py      # the paper's 4 figures
scripts/run_all.py
tests/            # pytest sanity checks (python -m pytest -q)
results/          # results.json + per-experiment CSVs (generated)
paper/figures/    # vector PDFs (generated)
```

## Tests

```bash
python -m pytest -q   # DGP recovery, regime paths, fold bookkeeping, explicit leakage test,
                      # WFER arithmetic, determinism, directional sanity
```

## License

Code: MIT. Paper text and figures (when written): CC BY 4.0.

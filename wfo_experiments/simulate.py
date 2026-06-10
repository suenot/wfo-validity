"""One experiment = one simulated validation problem, end to end.

We draw a ground-truth time-varying Sharpe surface (regime sampled and recorded), simulate one
history of returns, run all five validation procedures on the SAME history, and record — together
with every config field — each procedure's validation diagnostics (WFER raw/normalized, stitched
OOS Sharpe, OOS t-stat, IS->OOS rank correlation, ...) and its ground-truth outcome: the true
expected Sharpe of the deployed theta over the future horizon, computed from the known surface.

Batches of these records are the raw material for every number in the paper.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .model import (
    REGIMES,
    ExperimentConfig,
    forward_sharpe_by_theta,
    sample_experiment_config,
    simulate_history,
    with_windows,
)
from .procedures import PROCEDURES, ProcName, run_procedure

# per-procedure scalar diagnostics copied into the flat record
_PROC_FIELDS = (
    "wfer_raw", "wfer_norm", "oos_sharpe", "oos_tstat", "rankcorr", "frac_pos", "theta_std",
    "n_folds", "n_is", "n_oos", "n_is_unique", "is_mean", "is_sd",
    "min_fold_is_z", "max_abs_fold_wfer",
)


def run_experiment(
    cfg: ExperimentConfig,
    rng: np.random.Generator,
    *,
    procedures: tuple[ProcName, ...] = PROCEDURES,
) -> dict:
    """Simulate one history, run the requested procedures, return one flat record."""
    returns, surface = simulate_history(cfg, rng)
    fwd = forward_sharpe_by_theta(surface, cfg)  # true forward Sharpe per theta
    fwd_oracle = float(np.max(fwd))

    record: dict = {
        **{f"cfg_{k}": v for k, v in asdict(cfg).items()},
        "fwd_oracle": fwd_oracle,
        "fwd_mean_theta": float(np.mean(fwd)),  # forward Sharpe of a uniformly random selection
    }
    for name in procedures:
        res = run_procedure(name, returns, cfg, rng)
        th = res["theta_final"]
        record[f"{name}_theta_final"] = th
        record[f"{name}_fwd_sharpe"] = float(fwd[th])
        record[f"{name}_fwd_regret"] = float(fwd_oracle - fwd[th])
        for k in _PROC_FIELDS:
            record[f"{name}_{k}"] = res[k]
    return record


def run_batch(
    n_experiments: int,
    *,
    seed: int = 0,
    regimes: tuple[str, ...] = REGIMES,
    procedures: tuple[ProcName, ...] = PROCEDURES,
    progress_every: int = 0,
    **sampler_kwargs,
) -> list[dict]:
    """Run ``n_experiments`` with independently-seeded RNGs; regimes are cycled for equal counts."""
    child_seeds = np.random.SeedSequence(seed).spawn(n_experiments)
    records = []
    for i, cs in enumerate(child_seeds):
        rng = np.random.default_rng(cs)
        cfg = sample_experiment_config(rng, regime=regimes[i % len(regimes)], **sampler_kwargs)
        records.append(run_experiment(cfg, rng, procedures=procedures))
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  {i + 1}/{n_experiments}", flush=True)
    return records


def window_sweep(
    n_per_cell: int,
    *,
    seed: int = 0,
    t_hist: int = 756,
    train_lens: tuple[int, ...] = (63, 126, 189, 252, 378, 504),
    test_div: int = 3,
    min_test_len: int = 21,
    regimes: tuple[str, ...] = REGIMES,
    progress: bool = False,
) -> list[dict]:
    """Window-design sweep (rolling WFO only): force ``train_len`` on a grid, ``test_len = train/3``.

    Holds the history length fixed so cells differ only in window design; the number of folds falls
    from ~33 (train=63) to 1 (train=504), tracing the selection-quality vs estimation-noise tradeoff.
    """
    records = []
    for j, train_len in enumerate(train_lens):
        test_len = max(min_test_len, train_len // test_div)
        child_seeds = np.random.SeedSequence(seed).spawn(len(train_lens))[j].spawn(n_per_cell)
        for i, cs in enumerate(child_seeds):
            rng = np.random.default_rng(cs)
            cfg = sample_experiment_config(rng, regime=regimes[i % len(regimes)],
                                           t_hist_choices=(t_hist,))
            cfg = with_windows(cfg, train_len=train_len, test_len=test_len)
            records.append(run_experiment(cfg, rng, procedures=("rolling",)))
        if progress:
            print(f"  train_len={train_len} done ({n_per_cell} runs)", flush=True)
    return records

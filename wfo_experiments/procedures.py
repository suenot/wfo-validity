"""Validation procedures: single split, anchored/rolling walk-forward, naive k-fold, CPCV-lite.

Each procedure turns a simulated history ``returns (t_hist, n_theta)`` into

* a list of folds ``(train_idx, test_idx)`` — selection happens ONLY on ``train_idx``;
* per-fold IS/OOS PnL of the selected theta and the stitched OOS return series;
* validation diagnostics: the blog post's Walk-Forward Efficiency Ratio in its literal raw form
  (``sum OOS PnL / sum IS PnL``) and a per-period normalized form, the stitched OOS Sharpe, an OOS
  t-statistic, the mean IS->OOS per-theta rank correlation, the fraction of positive-OOS folds and
  the dispersion of selected thetas across folds;
* a deployment selection ``theta_final`` — the configuration the procedure's logic would put into
  production after validation (judged later against the known forward truth).

Deployment rules (documented design decisions):

* ``split``    deploys the argmax of the train portion (the test stays untouched, as the recipe demands);
* ``anchored`` deploys the argmax over the full history (its last expanding window, extended to the end);
* ``rolling``  deploys the argmax over the most recent ``train_len`` periods;
* ``kfold``    deploys the textbook CV grid selection: argmax of the mean per-fold test Sharpe;
* ``cpcv``     deploys the argmax over the full history (final fit on all data, per de Prado).

Anchored and rolling share IDENTICAL test segments (test start times are ``train_len + j*test_len``),
so their comparison is paired: same data, same OOS periods, different training windows. The rolling
step equals ``test_len`` so OOS segments never overlap.

CPCV-lite is a documented simplification of combinatorial purged cross-validation: contiguous groups,
``C(n_groups, k_test)`` train/test combinations, and an embargo of ``cfg.embargo`` periods dropped
from the train set at each test-group boundary. In this DGP returns are serially independent given
the surface, so purging cannot remove genuine leakage (there is none); it is included to mirror the
procedure as practiced.
"""

from __future__ import annotations

from itertools import combinations
from typing import Literal

import numpy as np
from scipy import stats

from .model import ExperimentConfig, sharpe

ProcName = Literal["split", "anchored", "rolling", "kfold", "cpcv"]
PROCEDURES: tuple[ProcName, ...] = ("split", "anchored", "rolling", "kfold", "cpcv")

Fold = tuple[np.ndarray, np.ndarray]  # (train_idx, test_idx), both 1-D int arrays


# --------------------------------------------------------------------------- #
# fold construction
# --------------------------------------------------------------------------- #
def _wfo_test_starts(t: int, cfg: ExperimentConfig) -> list[int]:
    """Aligned OOS segment start times shared by anchored and rolling WFO."""
    starts, s = [], cfg.train_len
    while s + cfg.test_len <= t:
        starts.append(s)
        s += cfg.test_len
    return starts


def build_folds(name: ProcName, t: int, cfg: ExperimentConfig, rng: np.random.Generator) -> list[Fold]:
    """Train/test index pairs for one procedure. ``rng`` is consumed only by ``kfold`` (the shuffle)."""
    idx = np.arange(t)
    if name == "split":
        n_test = int(round(cfg.split_test_frac * t))
        return [(idx[: t - n_test], idx[t - n_test :])]

    if name in ("anchored", "rolling"):
        folds: list[Fold] = []
        for s in _wfo_test_starts(t, cfg):
            train = idx[:s] if name == "anchored" else idx[s - cfg.train_len : s]
            folds.append((train, idx[s : s + cfg.test_len]))
        return folds

    if name == "kfold":  # naive, shuffled — the deliberately time-blind baseline
        perm = rng.permutation(t)
        parts = np.array_split(perm, cfg.k_folds)
        return [(np.sort(np.concatenate(parts[:i] + parts[i + 1 :])), np.sort(parts[i]))
                for i in range(cfg.k_folds)]

    if name == "cpcv":
        groups = np.array_split(idx, cfg.cpcv_groups)
        bounds = [(int(g[0]), int(g[-1]) + 1) for g in groups]
        folds = []
        for test_ids in combinations(range(cfg.cpcv_groups), cfg.cpcv_k_test):
            test = np.sort(np.concatenate([groups[i] for i in test_ids]))
            train_mask = np.ones(t, dtype=bool)
            train_mask[test] = False
            for i in test_ids:  # embargo around each test group
                a, b = bounds[i]
                train_mask[max(a - cfg.embargo, 0) : a] = False
                train_mask[b : min(b + cfg.embargo, t)] = False
            folds.append((idx[train_mask], test))
        return folds

    raise ValueError(f"unknown procedure {name!r}")


def select_theta(returns: np.ndarray, train_idx: np.ndarray, periods_per_year: int) -> int:
    """Selection rule used everywhere: argmax of the train-window annualized Sharpe."""
    return int(np.nanargmax(sharpe(returns[train_idx], periods_per_year)))


def deployment_train_idx(name: ProcName, t: int, cfg: ExperimentConfig) -> np.ndarray | None:
    """Indices the procedure re-optimizes on for deployment; ``None`` => kfold's CV-mean rule."""
    if name == "split":
        n_test = int(round(cfg.split_test_frac * t))
        return np.arange(t - n_test)
    if name == "rolling":
        return np.arange(t - cfg.train_len, t)
    if name in ("anchored", "cpcv"):
        return np.arange(t)
    return None  # kfold


# --------------------------------------------------------------------------- #
# WFER and the procedure run
# --------------------------------------------------------------------------- #
def wfer_raw(is_pnls: np.ndarray, oos_pnls: np.ndarray) -> float:
    """The blog post's literal WFER: sum of OOS PnL over sum of IS PnL (nan if IS PnL is exactly 0)."""
    denom = float(np.sum(is_pnls))
    return float(np.sum(oos_pnls) / denom) if denom != 0.0 else float("nan")


def wfer_normalized(is_pnls: np.ndarray, oos_pnls: np.ndarray, n_is: int, n_oos: int) -> float:
    """Per-period WFER: (OOS PnL per period) / (IS PnL per period).

    The raw ratio compares PnL accumulated over windows of different lengths (IS windows are 3-4x the
    OOS windows here), so even a perfectly stationary edge scores far below 1. The normalized form is
    the only reading under which the post's 0.8/0.5/0.3 thresholds can be meaningful.
    """
    denom = float(np.sum(is_pnls)) / max(n_is, 1)
    return float((np.sum(oos_pnls) / max(n_oos, 1)) / denom) if denom != 0.0 else float("nan")


def _fold_rank_corr(returns: np.ndarray, fold: Fold, ppy: int) -> float:
    """Spearman correlation across theta between train-window and test-window Sharpe."""
    sh_tr = sharpe(returns[fold[0]], ppy)
    sh_te = sharpe(returns[fold[1]], ppy)
    ok = np.isfinite(sh_tr) & np.isfinite(sh_te)
    if ok.sum() < 3 or np.std(sh_tr[ok]) == 0 or np.std(sh_te[ok]) == 0:
        return float("nan")
    return float(stats.spearmanr(sh_tr[ok], sh_te[ok]).statistic)


def run_procedure(name: ProcName, returns: np.ndarray, cfg: ExperimentConfig,
                  rng: np.random.Generator) -> dict:
    """Run one validation procedure end to end on a simulated history; see module docstring."""
    t = returns.shape[0]
    ppy = cfg.periods_per_year
    folds = build_folds(name, t, cfg, rng)

    fold_thetas, is_pnls, oos_pnls, rank_corrs, fold_is_z = [], [], [], [], []
    oos_parts, is_parts = [], []
    for train_idx, test_idx in folds:
        th = select_theta(returns, train_idx, ppy)
        fold_thetas.append(th)
        tr_ret = returns[train_idx, th]
        is_pnls.append(float(tr_ret.sum()))
        oos_pnls.append(float(returns[test_idx, th].sum()))
        oos_parts.append(returns[test_idx, th])
        is_parts.append(tr_ret)
        rank_corrs.append(_fold_rank_corr(returns, (train_idx, test_idx), ppy))
        # z-score of this fold's IS PnL against zero: |sum| / (sd * sqrt(n)); the blog reads WFER
        # per window, and the per-window ratio is ill-conditioned exactly when this z is small.
        sd = float(tr_ret.std(ddof=1))
        fold_is_z.append(abs(is_pnls[-1]) / (sd * np.sqrt(tr_ret.size)) if sd > 0 else float("nan"))

    is_arr, oos_arr = np.asarray(is_pnls), np.asarray(oos_pnls)
    oos_stitched = np.concatenate(oos_parts)
    is_stitched = np.concatenate(is_parts)
    n_is, n_oos = is_stitched.size, oos_stitched.size

    oos_sd = float(oos_stitched.std(ddof=1)) if n_oos > 1 else float("nan")
    oos_sharpe = float(oos_stitched.mean() / oos_sd * np.sqrt(ppy)) if oos_sd and oos_sd > 0 else float("nan")
    oos_tstat = float(oos_stitched.mean() / oos_sd * np.sqrt(n_oos)) if oos_sd and oos_sd > 0 else float("nan")

    # deployment selection
    dep_idx = deployment_train_idx(name, t, cfg)
    if dep_idx is None:  # kfold: argmax of mean per-fold test Sharpe across all theta
        per_fold = np.vstack([sharpe(returns[te], ppy) for _, te in folds])
        theta_final = int(np.nanargmax(np.nanmean(per_fold, axis=0)))
    else:
        theta_final = select_theta(returns, dep_idx, ppy)

    finite_rc = [r for r in rank_corrs if np.isfinite(r)]
    n_is_unique = int(np.unique(np.concatenate([tr for tr, _ in folds])).size)
    theta_vals = np.asarray(fold_thetas, dtype=float) / max(cfg.n_theta - 1, 1)  # in [0, 1] units
    with np.errstate(divide="ignore", invalid="ignore"):
        fold_wfers = np.where(is_arr != 0.0, oos_arr / is_arr, np.nan)

    return {
        "name": name,
        "folds": folds,
        "fold_thetas": fold_thetas,
        "is_pnls": is_arr,
        "oos_pnls": oos_arr,
        "oos_stitched": oos_stitched,
        "theta_final": theta_final,
        # diagnostics ------------------------------------------------------
        "wfer_raw": wfer_raw(is_arr, oos_arr),
        "wfer_norm": wfer_normalized(is_arr, oos_arr, n_is, n_oos),
        "oos_sharpe": oos_sharpe,
        "oos_tstat": oos_tstat,
        "rankcorr": float(np.mean(finite_rc)) if finite_rc else float("nan"),
        "frac_pos": float(np.mean(oos_arr > 0)),
        "theta_std": float(theta_vals.std()) if theta_vals.size > 1 else 0.0,
        # bookkeeping for the ill-conditioning analysis ----------------------
        "n_folds": len(folds),
        "n_is": int(n_is),
        "n_oos": int(n_oos),
        "n_is_unique": n_is_unique,
        "is_mean": float(is_stitched.mean()),
        "is_sd": float(is_stitched.std(ddof=1)) if n_is > 1 else float("nan"),
        "min_fold_is_z": float(np.nanmin(fold_is_z)) if fold_is_z else float("nan"),
        "max_abs_fold_wfer": float(np.nanmax(np.abs(fold_wfers))) if np.isfinite(fold_wfers).any() else float("nan"),
    }

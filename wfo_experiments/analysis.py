"""Turn batches of experiment records into the paper's quantitative results.

Four questions, all judged against the known forward truth and stratified by regime:

  1. Is WFER a useful diagnostic? Spearman/AUC of WFER (raw and per-period normalized) vs the true
     forward Sharpe and the "deployed a loser" label, compared against simpler alternatives
     (stitched OOS Sharpe, OOS t-stat, IS->OOS rank correlation, fold win-rate, theta stability).
     Plus: what the blog's 0.8/0.5/0.3 threshold bins actually contain, and how unstable the ratio
     is when the IS PnL denominator is near zero.
  2. Anchored vs rolling: which deploys the better theta under each regime (paired on shared data)?
  3. Window design: forward Sharpe and diagnostic quality across the train-length sweep.
  4. Procedure ranking: forward Sharpe and validation-estimate bias for all five procedures.

No cherry-picking: every table reports per-regime AND aggregate values, including negatives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from .model import REGIMES
from .procedures import PROCEDURES

# diagnostic -> (+1 if larger means BETTER expected forward performance, else -1)
DIAGNOSTIC_ORIENTATION = {
    "wfer_norm": +1,
    "wfer_raw": +1,
    "oos_sharpe": +1,
    "oos_tstat": +1,
    "rankcorr": +1,
    "frac_pos": +1,
    "theta_std": -1,  # unstable selections across folds = blog's instability warning sign
}

# the blog post's WFER interpretation table (">0.8 excellent, 0.5-0.8 ok, 0.3-0.5 borderline,
# <0.3 overfit, <0 unprofitable"); bins are applied to experiments with positive IS PnL, the
# implicit assumption of the table, and the IS<=0 remainder is reported as its own row.
BLOG_WFER_EDGES = (-np.inf, 0.0, 0.3, 0.5, 0.8, np.inf)
BLOG_WFER_LABELS = ("<0", "0-0.3", "0.3-0.5", "0.5-0.8", ">=0.8")


def to_frame(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return float("nan"), float("nan")
    r = stats.spearmanr(x[ok], y[ok])
    return float(r.statistic), float(r.pvalue)


def _auc(score: np.ndarray, label: np.ndarray) -> float:
    ok = np.isfinite(score) & np.isfinite(label)
    y = label[ok].astype(int)
    if not (0 < y.sum() < y.size):
        return float("nan")
    return float(roc_auc_score(y, score[ok]))


# --------------------------------------------------------------------------- #
# Q1a: diagnostic quality
# --------------------------------------------------------------------------- #
def diagnostic_quality(df: pd.DataFrame, proc: str = "rolling") -> list[dict]:
    """Per diagnostic: Spearman vs true forward Sharpe (overall / per regime / excluding null) and
    AUC for detecting "deployed a loser" (forward Sharpe < 0)."""
    fwd = df[f"{proc}_fwd_sharpe"].to_numpy(dtype=float)
    loser = (fwd < 0.0).astype(float)
    rows = []
    for diag, orient in DIAGNOSTIC_ORIENTATION.items():
        val = orient * df[f"{proc}_{diag}"].to_numpy(dtype=float)
        rho, p = _spearman(val, fwd)
        row = {"procedure": proc, "diagnostic": diag, "rho_fwd": rho, "p_fwd": p,
               "auc_loser": _auc(-val, loser)}
        for reg in REGIMES:
            m = (df["cfg_regime"] == reg).to_numpy()
            row[f"rho_fwd_{reg}"] = _spearman(val[m], fwd[m])[0]
        m = (df["cfg_regime"] != "null").to_numpy()
        row["rho_fwd_ex_null"] = _spearman(val[m], fwd[m])[0]
        row["auc_loser_ex_null"] = _auc(-val[m], loser[m])
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Q1b: the blog's WFER threshold bins
# --------------------------------------------------------------------------- #
def wfer_calibration(df: pd.DataFrame, proc: str = "rolling", col: str = "wfer_norm") -> list[dict]:
    """What each blog bin actually contains: n, regime mix, mean/median forward Sharpe, loser rate."""
    wfer = df[f"{proc}_{col}"].to_numpy(dtype=float)
    fwd = df[f"{proc}_fwd_sharpe"].to_numpy(dtype=float)
    is_pos = df[f"{proc}_is_mean"].to_numpy(dtype=float) > 0
    null = (df["cfg_regime"] == "null").to_numpy()

    def _row(name: str, m: np.ndarray) -> dict:
        return {
            "bin": name, "n": int(m.sum()),
            "mean_fwd_sharpe": float(np.nanmean(fwd[m])) if m.any() else float("nan"),
            "median_fwd_sharpe": float(np.nanmedian(fwd[m])) if m.any() else float("nan"),
            "frac_loser": float(np.mean(fwd[m] < 0)) if m.any() else float("nan"),
            "frac_null_regime": float(np.mean(null[m])) if m.any() else float("nan"),
        }

    rows = [_row("IS<=0 (ratio ill-defined)", ~is_pos | ~np.isfinite(wfer))]
    binned = is_pos & np.isfinite(wfer)
    for lo, hi, name in zip(BLOG_WFER_EDGES[:-1], BLOG_WFER_EDGES[1:], BLOG_WFER_LABELS):
        rows.append(_row(name, binned & (wfer >= lo) & (wfer < hi)))
    return rows


# --------------------------------------------------------------------------- #
# Q1c: ill-conditioning of the WFER ratio
# --------------------------------------------------------------------------- #
def ill_conditioning(df: pd.DataFrame, proc: str = "rolling",
                     k_list: tuple[float, ...] = (0.5, 1.0, 2.0)) -> dict:
    """How often is the IS-PnL denominator statistically indistinguishable from zero, and what does
    that do to WFER?

    Two levels, because the blog reads WFER both aggregated and per window:
    * aggregate: flag experiments with ``|mean IS pnl| < k * se``, ``se = is_sd/sqrt(n_is_unique)``
      (IS windows overlap across folds, so this se is approximate; disclosed);
    * per window: flag experiments where the WEAKEST fold has ``|IS pnl| < k * sd * sqrt(n)``
      (``min_fold_is_z < k``) and report what happens to the largest per-window ratio there.
    """
    is_mean = df[f"{proc}_is_mean"].to_numpy(dtype=float)
    se = df[f"{proc}_is_sd"].to_numpy(dtype=float) / np.sqrt(df[f"{proc}_n_is_unique"].to_numpy(dtype=float))
    wfer = df[f"{proc}_wfer_norm"].to_numpy(dtype=float)
    fwd = df[f"{proc}_fwd_sharpe"].to_numpy(dtype=float)
    regime = df["cfg_regime"].to_numpy()
    min_z = df[f"{proc}_min_fold_is_z"].to_numpy(dtype=float)
    max_fold = df[f"{proc}_max_abs_fold_wfer"].to_numpy(dtype=float)

    out: dict = {"procedure": proc, "by_k": []}
    for k in k_list:
        flag = np.abs(is_mean) < k * se
        fin = np.isfinite(wfer)
        wflag = min_z < k  # at least one near-zero-IS window
        wfin = np.isfinite(max_fold)
        entry = {
            "k": k,
            # aggregate-ratio conditioning ----------------------------------
            "frac_flagged": float(np.mean(flag)),
            "frac_flagged_by_regime": {r: float(np.mean(flag[regime == r])) for r in REGIMES},
            "wfer_iqr_flagged": float(stats.iqr(wfer[flag & fin])) if (flag & fin).any() else float("nan"),
            "wfer_iqr_unflagged": float(stats.iqr(wfer[~flag & fin])) if (~flag & fin).any() else float("nan"),
            "rho_fwd_flagged": _spearman(wfer[flag], fwd[flag])[0],
            "rho_fwd_unflagged": _spearman(wfer[~flag], fwd[~flag])[0],
            "frac_abs_wfer_gt_3_flagged": float(np.mean(np.abs(wfer[flag & fin]) > 3.0)) if (flag & fin).any() else float("nan"),
            "frac_abs_wfer_gt_3_unflagged": float(np.mean(np.abs(wfer[~flag & fin]) > 3.0)) if (~flag & fin).any() else float("nan"),
            # per-window-ratio conditioning ----------------------------------
            "frac_runs_with_weak_is_window": float(np.mean(wflag)),
            "frac_weak_window_by_regime": {r: float(np.mean(wflag[regime == r])) for r in REGIMES},
            "median_max_abs_fold_wfer_weak": float(np.nanmedian(max_fold[wflag & wfin])) if (wflag & wfin).any() else float("nan"),
            "median_max_abs_fold_wfer_strong": float(np.nanmedian(max_fold[~wflag & wfin])) if (~wflag & wfin).any() else float("nan"),
            "frac_max_fold_wfer_gt_5_weak": float(np.nanmean(max_fold[wflag & wfin] > 5.0)) if (wflag & wfin).any() else float("nan"),
            "frac_max_fold_wfer_gt_5_strong": float(np.nanmean(max_fold[~wflag & wfin] > 5.0)) if (~wflag & wfin).any() else float("nan"),
        }
        out["by_k"].append(entry)
    return out


# --------------------------------------------------------------------------- #
# Q2: anchored vs rolling (paired on the same simulated history)
# --------------------------------------------------------------------------- #
def anchored_vs_rolling(df: pd.DataFrame) -> list[dict]:
    diff = (df["rolling_fwd_sharpe"] - df["anchored_fwd_sharpe"]).to_numpy(dtype=float)
    rows = []
    for scope in ("overall", *REGIMES):
        m = np.ones(len(df), dtype=bool) if scope == "overall" else (df["cfg_regime"] == scope).to_numpy()
        d = diff[m]
        nz = d[d != 0.0]
        try:
            p = float(stats.wilcoxon(nz).pvalue) if nz.size >= 10 else float("nan")
        except ValueError:
            p = float("nan")
        rows.append({
            "scope": scope, "n": int(m.sum()),
            "mean_fwd_anchored": float(df.loc[m, "anchored_fwd_sharpe"].mean()),
            "mean_fwd_rolling": float(df.loc[m, "rolling_fwd_sharpe"].mean()),
            "mean_diff_rolling_minus_anchored": float(d.mean()),
            "sem_diff": float(d.std(ddof=1) / np.sqrt(max(d.size, 2))),
            "win_rate_rolling": float(np.mean(d > 0)),
            "tie_rate": float(np.mean(d == 0)),
            "wilcoxon_p": p,
        })
    return rows


# --------------------------------------------------------------------------- #
# Q4: full procedure ranking + estimate bias
# --------------------------------------------------------------------------- #
def procedure_ranking(df: pd.DataFrame) -> list[dict]:
    """Per procedure: mean true forward Sharpe (overall and per regime), the bias of its own
    validation estimate (stitched OOS Sharpe minus forward truth), and how well that estimate ranks
    outcomes across experiments."""
    rows = []
    for proc in PROCEDURES:
        fwd = df[f"{proc}_fwd_sharpe"].to_numpy(dtype=float)
        est = df[f"{proc}_oos_sharpe"].to_numpy(dtype=float)
        row = {
            "procedure": proc,
            "mean_fwd_sharpe": float(np.nanmean(fwd)),
            "mean_fwd_regret": float(df[f"{proc}_fwd_regret"].mean()),
            "mean_estimate_bias": float(np.nanmean(est - fwd)),
            "rho_estimate_vs_fwd": _spearman(est, fwd)[0],
        }
        for reg in REGIMES:
            m = (df["cfg_regime"] == reg).to_numpy()
            row[f"mean_fwd_{reg}"] = float(np.nanmean(fwd[m]))
            row[f"bias_{reg}"] = float(np.nanmean(est[m] - fwd[m]))
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Q3: window-design sweep summary
# --------------------------------------------------------------------------- #
def summarize_sweep(df: pd.DataFrame) -> list[dict]:
    """Per train-length cell: forward Sharpe by regime + diagnostic quality of rolling WFER/OOS Sharpe."""
    rows = []
    for train_len, g in df.groupby("cfg_train_len"):
        fwd = g["rolling_fwd_sharpe"].to_numpy(dtype=float)
        wfer = g["rolling_wfer_norm"].to_numpy(dtype=float)
        est = g["rolling_oos_sharpe"].to_numpy(dtype=float)
        row = {
            "train_len": int(train_len),
            "test_len": int(g["cfg_test_len"].iloc[0]),
            "n": int(len(g)),
            "mean_n_folds": float(g["rolling_n_folds"].mean()),
            "mean_fwd_sharpe": float(np.nanmean(fwd)),
            "rho_wfer_fwd": _spearman(wfer, fwd)[0],
            "rho_oos_sharpe_fwd": _spearman(est, fwd)[0],
            "rmse_estimate": float(np.sqrt(np.nanmean((est - fwd) ** 2))),
            "mean_estimate_bias": float(np.nanmean(est - fwd)),
        }
        for reg in REGIMES:
            m = (g["cfg_regime"] == reg).to_numpy()
            row[f"mean_fwd_{reg}"] = float(np.nanmean(fwd[m]))
            row[f"sem_fwd_{reg}"] = float(np.nanstd(fwd[m], ddof=1) / np.sqrt(max(m.sum(), 2)))
        rows.append(row)
    return sorted(rows, key=lambda r: r["train_len"])


# --------------------------------------------------------------------------- #
# one call -> everything the paper reports
# --------------------------------------------------------------------------- #
def summarize(df: pd.DataFrame) -> dict:
    return {
        "n_experiments": int(len(df)),
        "n_by_regime": {r: int((df["cfg_regime"] == r).sum()) for r in REGIMES},
        "diagnostic_quality_rolling": diagnostic_quality(df, "rolling"),
        "diagnostic_quality_anchored": diagnostic_quality(df, "anchored"),
        "wfer_bins_normalized": wfer_calibration(df, "rolling", "wfer_norm"),
        "wfer_bins_raw": wfer_calibration(df, "rolling", "wfer_raw"),
        "ill_conditioning_rolling": ill_conditioning(df, "rolling"),
        "anchored_vs_rolling": anchored_vs_rolling(df),
        "procedure_ranking": procedure_ranking(df),
    }

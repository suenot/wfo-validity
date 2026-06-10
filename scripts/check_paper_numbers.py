#!/usr/bin/env python
"""Verify every numeric claim quoted in paper/main.tex against results/results.json.

Each check pairs a number as printed in the paper (at its printed precision) with the value
computed from results.json; the assertion is that the paper value equals the correctly rounded
result (|actual - quoted| <= 0.5 * 10^-decimals, plus float epsilon). P-values quoted in
"m x 10^-e" form are checked at one significant figure. Exits non-zero on any mismatch.

    python scripts/check_paper_numbers.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "results" / "results.json").read_text())
M = R["main"]

dq = {r["diagnostic"]: r for r in M["diagnostic_quality_rolling"]}
dqa = {r["diagnostic"]: r for r in M["diagnostic_quality_anchored"]}
binsN = {r["bin"]: r for r in M["wfer_bins_normalized"]}
binsR = {r["bin"]: r for r in M["wfer_bins_raw"]}
ill = {e["k"]: e for e in M["ill_conditioning_rolling"]["by_k"]}
avr = {r["scope"]: r for r in M["anchored_vs_rolling"]}
proc = {r["procedure"]: r for r in M["procedure_ranking"]}
sweep = {r["train_len"]: r for r in R["window_sweep"]}

checks: list[tuple[str, float, float, int]] = []


def c(label: str, actual: float, quoted: float, dec: int) -> None:
    checks.append((label, float(actual), float(quoted), dec))


pchecks: list[tuple[str, float, int, int]] = []


def p(label: str, actual: float, mantissa: int, exponent: int) -> None:
    """Quoted as 'mantissa x 10^-exponent' at one significant figure."""
    pchecks.append((label, float(actual), mantissa, exponent))


# --------------------------------------------------------------------------- #
# experiment counts (abstract / setup)
# --------------------------------------------------------------------------- #
c("n_main", M["n_experiments"], 8000, 0)
for reg in ("stationary", "drift", "break", "null"):
    c(f"n_{reg}", M["n_by_regime"][reg], 2000, 0)
c("n_sweep_total", sum(s["n"] for s in sweep.values()), 9000, 0)
c("n_per_sweep_cell", R["meta"]["n_per_sweep_cell"], 1500, 0)
c("n_total_17000", M["n_experiments"] + sum(s["n"] for s in sweep.values()), 17000, 0)
c("seed_main", R["meta"]["seeds"]["main"], 101, 0)
c("seed_sweep", R["meta"]["seeds"]["sweep"], 303, 0)

# --------------------------------------------------------------------------- #
# Table 1: diagnostic quality (rolling), 3 dp
# --------------------------------------------------------------------------- #
T1 = {
    "wfer_norm":  (0.451, 0.387, 0.365, 0.260,  0.022, 0.338, 0.743),
    "wfer_raw":   (0.447, 0.381, 0.360, 0.257,  0.021, 0.333, 0.742),
    "oos_sharpe": (0.529, 0.469, 0.435, 0.320,  0.017, 0.408, 0.779),
    "oos_tstat":  (0.532, 0.467, 0.436, 0.327,  0.017, 0.410, 0.781),
    "rankcorr":   (0.516, 0.508, 0.442, 0.318,  0.036, 0.424, 0.752),
    "frac_pos":   (0.434, 0.386, 0.335, 0.246, -0.012, 0.323, 0.733),
    "theta_std":  (0.408, 0.371, 0.279, 0.142, -0.004, 0.266, 0.733),
}
for diag, (ov, st, dr, br, nu, ex, auc) in T1.items():
    d = dq[diag]
    c(f"T1 {diag} rho", d["rho_fwd"], ov, 3)
    c(f"T1 {diag} rho stationary", d["rho_fwd_stationary"], st, 3)
    c(f"T1 {diag} rho drift", d["rho_fwd_drift"], dr, 3)
    c(f"T1 {diag} rho break", d["rho_fwd_break"], br, 3)
    c(f"T1 {diag} rho null", d["rho_fwd_null"], nu, 3)
    c(f"T1 {diag} rho ex-null", d["rho_fwd_ex_null"], ex, 3)
    c(f"T1 {diag} AUC", d["auc_loser"], auc, 3)

# 6.1 prose
c("6.1 wfer AUC ex-null", dq["wfer_norm"]["auc_loser_ex_null"], 0.685, 3)
c("6.1 null col max |rho|",
  max(abs(dq[d]["rho_fwd_null"]) for d in T1), 0.036, 3)
c("6.1 anchored wfer rho", dqa["wfer_norm"]["rho_fwd"], 0.417, 3)
c("6.1 anchored tstat rho", dqa["oos_tstat"]["rho_fwd"], 0.527, 3)
c("6.1 anchored rankcorr rho", dqa["rankcorr"]["rho_fwd"], 0.542, 3)

# abstract (2 dp)
c("abs wfer rho", dq["wfer_norm"]["rho_fwd"], 0.45, 2)
c("abs wfer AUC", dq["wfer_norm"]["auc_loser"], 0.74, 2)
c("abs tstat AUC", dq["oos_tstat"]["auc_loser"], 0.78, 2)
c("abs oos_sharpe rho", dq["oos_sharpe"]["rho_fwd"], 0.53, 2)

# --------------------------------------------------------------------------- #
# Table 2: folk threshold bins
# --------------------------------------------------------------------------- #
T2N = {  # bin: (n, mean fwd SR, loser %, null %)
    "<0":      (1947, 0.246, 56.5, 51.7),
    "0-0.3":   (2324, 0.588, 28.4, 24.1),
    "0.3-0.5": (1728, 0.830, 16.0, 13.4),
    "0.5-0.8": (1543, 0.994, 11.4,  9.6),
    ">=0.8":   ( 454, 0.980, 11.9, 11.0),
}
T2R = {
    "<0":      (1947, 0.246, 56.5, 51.7),
    "0-0.3":   (5887, 0.786, 19.3, 16.4),
    "0.3-0.5": ( 156, 0.956, 15.4, 14.1),
    "0.5-0.8": (   4, 0.029, 75.0, 75.0),
    ">=0.8":   (   2, 0.212, 50.0, 50.0),
}
for tag, table, bins in (("norm", T2N, binsN), ("raw", T2R, binsR)):
    for b, (n, sr, lo, nu) in table.items():
        c(f"T2 {tag} {b} n", bins[b]["n"], n, 0)
        c(f"T2 {tag} {b} mean fwd", bins[b]["mean_fwd_sharpe"], sr, 3)
        c(f"T2 {tag} {b} loser%", 100 * bins[b]["frac_loser"], lo, 1)
        c(f"T2 {tag} {b} null%", 100 * bins[b]["frac_null_regime"], nu, 1)
c("T2 IS<=0 excluded n", binsN["IS<=0 (ratio ill-defined)"]["n"], 4, 0)
c("T2 IS<=0 all null", binsN["IS<=0 (ratio ill-defined)"]["frac_null_regime"], 1.0, 3)
c("T2 IS<=0 all losers", binsN["IS<=0 (ratio ill-defined)"]["frac_loser"], 1.0, 3)

# 6.2 prose
c("6.2 abs top-bin 0.98", binsN[">=0.8"]["mean_fwd_sharpe"], 0.98, 2)
c("6.2 abs 0.5-0.8 0.99", binsN["0.5-0.8"]["mean_fwd_sharpe"], 0.99, 2)
c("6.2 abs overfit-zone 0.59", binsN["0-0.3"]["mean_fwd_sharpe"], 0.59, 2)
c("6.2 abs borderline 0.83", binsN["0.3-0.5"]["mean_fwd_sharpe"], 0.83, 2)
c("6.2 below-0.5 discarded n", binsN["0-0.3"]["n"] + binsN["0.3-0.5"]["n"], 4052, 0)
c("6.2 raw >=0.5 count = 6", binsR["0.5-0.8"]["n"] + binsR[">=0.8"]["n"], 6, 0)
c("6.2 raw 0-0.3 share of positive-IS runs",
  100 * binsR["0-0.3"]["n"] / (8000 - binsN["IS<=0 (ratio ill-defined)"]["n"]), 74, 0)
six_mean = (binsR["0.5-0.8"]["n"] * binsR["0.5-0.8"]["mean_fwd_sharpe"]
            + binsR[">=0.8"]["n"] * binsR[">=0.8"]["mean_fwd_sharpe"]) / 6
c("6.2 six raw->0.5 runs mean fwd +0.09", six_mean, 0.09, 2)
six_losers = (binsR["0.5-0.8"]["n"] * binsR["0.5-0.8"]["frac_loser"]
              + binsR[">=0.8"]["n"] * binsR[">=0.8"]["frac_loser"])
c("6.2 six raw->0.5 runs: 4 losers", six_losers, 4, 0)

# --------------------------------------------------------------------------- #
# 6.3 ill-conditioning (k = 1 unless stated)
# --------------------------------------------------------------------------- #
k1, k2 = ill[1.0], ill[2.0]
c("6.3 weak-window runs %", 100 * k1["frac_runs_with_weak_is_window"], 21.4, 1)
c("6.3 weak stationary %", 100 * k1["frac_weak_window_by_regime"]["stationary"], 12.3, 1)
c("6.3 weak drift %", 100 * k1["frac_weak_window_by_regime"]["drift"], 12.7, 1)
c("6.3 weak break %", 100 * k1["frac_weak_window_by_regime"]["break"], 13.5, 1)
c("6.3 weak null %", 100 * k1["frac_weak_window_by_regime"]["null"], 47.3, 1)
c("6.3 median max|fold wfer| weak", k1["median_max_abs_fold_wfer_weak"], 0.88, 2)
c("6.3 median max|fold wfer| strong", k1["median_max_abs_fold_wfer_strong"], 0.43, 2)
c("6.3 P(max>5) weak %", 100 * k1["frac_max_fold_wfer_gt_5_weak"], 5.4, 1)
c("6.3 P(max>5) strong %", 100 * k1["frac_max_fold_wfer_gt_5_strong"], 0.0, 1)
c("6.3 weak-window runs % (k=2)", 100 * k2["frac_runs_with_weak_is_window"], 69.0, 1)
c("6.3 weak null % (k=2)", 100 * k2["frac_weak_window_by_regime"]["null"], 96, 0)
c("6.3 aggregate flagged %", 100 * k1["frac_flagged"], 0.46, 2)
c("6.3 aggregate flagged null %", 100 * k1["frac_flagged_by_regime"]["null"], 1.6, 1)
c("6.3 WFER IQR flagged", k1["wfer_iqr_flagged"], 2.44, 2)
c("6.3 WFER IQR clean", k1["wfer_iqr_unflagged"], 0.49, 2)
c("6.3 rho fwd flagged", k1["rho_fwd_flagged"], 0.08, 2)
c("6.3 rho fwd clean", k1["rho_fwd_unflagged"], 0.45, 2)
c("6.3 |WFER|>3 flagged %", 100 * k1["frac_abs_wfer_gt_3_flagged"], 32, 0)
c("6.3 |WFER|>3 clean %", 100 * k1["frac_abs_wfer_gt_3_unflagged"], 0.01, 2)
c("abs weak-window 21%", 100 * k1["frac_runs_with_weak_is_window"], 21, 0)
c("abs weak null 47%", 100 * k1["frac_weak_window_by_regime"]["null"], 47, 0)

# --------------------------------------------------------------------------- #
# 6.4 anchored vs rolling (paired)
# --------------------------------------------------------------------------- #
c("6.4 stationary anchored", avr["stationary"]["mean_fwd_anchored"], 1.062, 3)
c("6.4 stationary rolling", avr["stationary"]["mean_fwd_rolling"], 0.962, 3)
c("6.4 stationary diff -0.100",
  avr["stationary"]["mean_diff_rolling_minus_anchored"], -0.100, 3)
c("6.4 stationary sem", avr["stationary"]["sem_diff"], 0.007, 3)
c("6.4 drift diff +0.079", avr["drift"]["mean_diff_rolling_minus_anchored"], 0.079, 3)
c("6.4 drift sem", avr["drift"]["sem_diff"], 0.009, 3)
c("6.4 drift win %", 100 * avr["drift"]["win_rate_rolling"], 44, 0)
c("6.4 drift lose %",
  100 * (1 - avr["drift"]["win_rate_rolling"] - avr["drift"]["tie_rate"]), 30, 0)
c("6.4 break diff +0.154", avr["break"]["mean_diff_rolling_minus_anchored"], 0.154, 3)
c("6.4 break sem", avr["break"]["sem_diff"], 0.011, 3)
c("6.4 break win %", 100 * avr["break"]["win_rate_rolling"], 46, 0)
c("6.4 break lose %",
  100 * (1 - avr["break"]["win_rate_rolling"] - avr["break"]["tie_rate"]), 28, 0)
c("6.4 overall diff +0.034", avr["overall"]["mean_diff_rolling_minus_anchored"], 0.034, 3)
c("6.4 overall sem", avr["overall"]["sem_diff"], 0.004, 3)
c("6.4 null exact tie diff", avr["null"]["mean_diff_rolling_minus_anchored"], 0.0, 6)
c("6.4 null tie rate", avr["null"]["tie_rate"], 1.0, 6)
p("6.4 stationary wilcoxon", avr["stationary"]["wilcoxon_p"], 2, 39)
p("6.4 drift wilcoxon", avr["drift"]["wilcoxon_p"], 8, 18)
p("6.4 break wilcoxon", avr["break"]["wilcoxon_p"], 5, 40)
p("6.4 overall wilcoxon", avr["overall"]["wilcoxon_p"], 6, 12)
c("abs anchored stationary edge 0.10",
  -avr["stationary"]["mean_diff_rolling_minus_anchored"], 0.10, 2)
c("abs rolling drift edge 0.08", avr["drift"]["mean_diff_rolling_minus_anchored"], 0.08, 2)
c("abs rolling break edge 0.15", avr["break"]["mean_diff_rolling_minus_anchored"], 0.15, 2)

# --------------------------------------------------------------------------- #
# 6.5 window-design sweep
# --------------------------------------------------------------------------- #
fwd_quotes = {63: 0.567, 126: 0.640, 189: 0.651, 252: 0.655, 378: 0.651, 504: 0.662}
for L, q in fwd_quotes.items():
    c(f"6.5 fwd L={L}", sweep[L]["mean_fwd_sharpe"], q, 3)
rho_w = {63: 0.467, 189: 0.516, 252: 0.456, 378: 0.388, 504: 0.225}
for L, q in rho_w.items():
    c(f"6.5 rho_wfer L={L}", sweep[L]["rho_wfer_fwd"], q, 3)
c("6.5 rho_wfer peak at L=189",
  max(s["rho_wfer_fwd"] for s in sweep.values()), sweep[189]["rho_wfer_fwd"], 9)
c("6.5 rho_oos L=63", sweep[63]["rho_oos_sharpe_fwd"], 0.496, 3)
c("6.5 rho_oos L=189", sweep[189]["rho_oos_sharpe_fwd"], 0.575, 3)
c("6.5 rho_oos L=504", sweep[504]["rho_oos_sharpe_fwd"], 0.325, 3)
c("6.5 folds at 63", sweep[63]["mean_n_folds"], 33, 0)
c("6.5 folds at 189", sweep[189]["mean_n_folds"], 9, 0)
c("6.5 folds at 504", sweep[504]["mean_n_folds"], 1, 0)
c("6.5 stationary L=63", sweep[63]["mean_fwd_stationary"], 0.815, 3)
c("6.5 stationary L=504", sweep[504]["mean_fwd_stationary"], 1.026, 3)
c("6.5 break L=189", sweep[189]["mean_fwd_break"], 0.855, 3)
c("6.5 break L=504", sweep[504]["mean_fwd_break"], 0.784, 3)
c("6.5 drift min (L>=126)",
  min(sweep[L]["mean_fwd_drift"] for L in (126, 189, 252, 378, 504)), 0.84, 2)
c("6.5 drift max (L>=126)",
  max(sweep[L]["mean_fwd_drift"] for L in (126, 189, 252, 378, 504)), 0.89, 2)
c("6.5 rmse min (63/126)", min(sweep[63]["rmse_estimate"], sweep[126]["rmse_estimate"]), 0.72, 2)
c("6.5 rmse max (63/126)", max(sweep[63]["rmse_estimate"], sweep[126]["rmse_estimate"]), 0.73, 2)
c("6.5 rmse L=504", sweep[504]["rmse_estimate"], 1.33, 2)
assert max(abs(s["mean_estimate_bias"]) for s in sweep.values()) < 0.02, \
    "6.5 claim: |estimate bias| < 0.02 in every sweep cell"
c("abs sweep rho peak 0.52", sweep[189]["rho_wfer_fwd"], 0.52, 2)
c("abs sweep rho 1-fold 0.23", sweep[504]["rho_wfer_fwd"], 0.23, 2)

# --------------------------------------------------------------------------- #
# Table 3 + 6.6 prose: procedure ranking
# --------------------------------------------------------------------------- #
T3 = {  # overall, stationary, drift, break | bias, bias_drift, bias_break, rho
    "split":    (0.558, 1.030, 0.702, 0.551, 0.073, 0.212,  0.103, 0.477),
    "anchored": (0.624, 1.062, 0.790, 0.692, 0.048, 0.149,  0.071, 0.524),
    "rolling":  (0.657, 0.962, 0.869, 0.846, 0.013, 0.060, -0.034, 0.529),
    "kfold":    (0.623, 1.066, 0.788, 0.688, 0.083, 0.191,  0.147, 0.541),
    "cpcv":     (0.624, 1.062, 0.790, 0.692, 0.049, 0.155,  0.079, 0.636),
}
for name, (ov, st, dr, br, b, bd, bb, rho) in T3.items():
    r = proc[name]
    c(f"T3 {name} fwd", r["mean_fwd_sharpe"], ov, 3)
    c(f"T3 {name} fwd stationary", r["mean_fwd_stationary"], st, 3)
    c(f"T3 {name} fwd drift", r["mean_fwd_drift"], dr, 3)
    c(f"T3 {name} fwd break", r["mean_fwd_break"], br, 3)
    c(f"T3 {name} bias", r["mean_estimate_bias"], b, 3)
    c(f"T3 {name} bias drift", r["bias_drift"], bd, 3)
    c(f"T3 {name} bias break", r["bias_break"], bb, 3)
    c(f"T3 {name} rho(est,fwd)", r["rho_estimate_vs_fwd"], rho, 3)
    c(f"T3 {name} fwd null -0.049", r["mean_fwd_null"], -0.049, 3)
c("6.6 anchored == cpcv deployment",
  proc["anchored"]["mean_fwd_sharpe"] - proc["cpcv"]["mean_fwd_sharpe"], 0.0, 9)
c("6.6 rolling regret", proc["rolling"]["mean_fwd_regret"], 0.226, 3)
c("6.6 split regret", proc["split"]["mean_fwd_regret"], 0.325, 3)
c("6.6 kfold bias stationary -0.006", proc["kfold"]["bias_stationary"], -0.006, 3)
c("6.6 rolling bias stationary +0.002", proc["rolling"]["bias_stationary"], 0.002, 3)
assert min(p_["bias_drift"] for p_ in proc.values()) > 0, \
    "6.6 claim: under drift every procedure's estimate bias is positive"
others = [proc[n]["rho_estimate_vs_fwd"] for n in ("anchored", "rolling", "kfold")]
c("6.6 rho others min 0.524", min(others), 0.524, 3)
c("6.6 rho others max 0.541", max(others), 0.541, 3)
c("abs rolling fwd 0.66", proc["rolling"]["mean_fwd_sharpe"], 0.66, 2)
c("abs anchored fwd 0.62", proc["anchored"]["mean_fwd_sharpe"], 0.62, 2)
c("abs kfold fwd 0.62", proc["kfold"]["mean_fwd_sharpe"], 0.62, 2)
c("abs cpcv fwd 0.62", proc["cpcv"]["mean_fwd_sharpe"], 0.62, 2)
c("abs split fwd 0.56", proc["split"]["mean_fwd_sharpe"], 0.56, 2)
c("abs kfold bias 0.08", proc["kfold"]["mean_estimate_bias"], 0.08, 2)
c("abs cpcv rho 0.64", proc["cpcv"]["rho_estimate_vs_fwd"], 0.64, 2)
c("disc kfold drift bias 0.19", proc["kfold"]["bias_drift"], 0.19, 2)
c("disc kfold break bias 0.15", proc["kfold"]["bias_break"], 0.15, 2)

# --------------------------------------------------------------------------- #
# run the checks
# --------------------------------------------------------------------------- #
fails = 0
for label, actual, quoted, dec in checks:
    tol = 0.5 * 10 ** (-dec) + 1e-12
    ok = math.isfinite(actual) and abs(actual - quoted) <= tol
    status = "PASS" if ok else "FAIL"
    if not ok:
        fails += 1
        print(f"  {status}  {label}: paper={quoted!r} actual={actual!r} (dec={dec})")
for label, actual, mant, expo in pchecks:
    e = math.floor(math.log10(actual))
    m = round(actual / 10 ** e)
    if m == 10:  # 9.5e-40 -> 1e-39
        m, e = 1, e + 1
    ok = (m == mant) and (e == -expo)
    if not ok:
        fails += 1
        print(f"  FAIL  {label}: paper={mant}e-{expo} actual={actual:.2e}")

n = len(checks) + len(pchecks)
print(f"{n - fails}/{n} checks passed "
      f"({len(checks)} numeric + {len(pchecks)} p-value, + 2 inline assertions)")
if fails:
    sys.exit(1)
print("OK: every number quoted in the paper matches results/results.json within rounding.")

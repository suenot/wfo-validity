"""Reproduce every number and figure input in the paper.

    python scripts/run_all.py            # full run -> results/results.json + record CSVs
    python scripts/run_all.py --quick    # small batch for a smoke check

Deterministic given the fixed seeds below. No wall-clock / unseeded randomness leaks into results
(timestamps are not used anywhere). Run from the project root.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wfo_experiments import __version__
from wfo_experiments import analysis as A
from wfo_experiments.simulate import run_batch, window_sweep

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

SEED_MAIN, SEED_SWEEP = 101, 303


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    n_main = 400 if args.quick else 8000
    n_cell = 80 if args.quick else 1500
    RESULTS.mkdir(exist_ok=True)

    print(f"[1/3] main batch (n={n_main}, regimes cycled, 5 procedures) ...", flush=True)
    recs = run_batch(n_main, seed=SEED_MAIN, progress_every=n_main // 4)
    df = A.to_frame(recs)
    df.to_csv(RESULTS / "records.csv", index=False)

    print(f"[2/3] window-design sweep ({n_cell} runs/cell, rolling only) ...", flush=True)
    sweep_recs = window_sweep(n_cell, seed=SEED_SWEEP, progress=True)
    df_sweep = A.to_frame(sweep_recs)
    df_sweep.to_csv(RESULTS / "records_sweep.csv", index=False)

    print("[3/3] summaries ...", flush=True)
    summary = A.summarize(df)
    results = {
        "meta": {
            "package_version": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "n_main": n_main,
            "n_per_sweep_cell": n_cell,
            "seeds": {"main": SEED_MAIN, "sweep": SEED_SWEEP},
            "notes": "Deterministic; reproduce with python scripts/run_all.py",
        },
        "main": summary,
        "window_sweep": A.summarize_sweep(df_sweep),
    }
    (RESULTS / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"\nWrote {RESULTS / 'results.json'}, records.csv, records_sweep.csv.")

    # ---- headline numbers to stdout ---------------------------------------
    print("\n--- HEADLINE NUMBERS ---")
    print(f"experiments: {summary['n_experiments']}  by regime: {summary['n_by_regime']}")

    print("\nQ1 diagnostic quality (rolling WFO) — Spearman vs true forward Sharpe / AUC(loser):")
    for r in summary["diagnostic_quality_rolling"]:
        print(f"  {r['diagnostic']:11} rho {r['rho_fwd']:+.3f} (ex-null {r['rho_fwd_ex_null']:+.3f})  "
              f"AUC {r['auc_loser']:.3f}  by regime "
              + " ".join(f"{reg[:4]} {r[f'rho_fwd_{reg}']:+.2f}" for reg in
                         ("stationary", "drift", "break", "null")))

    print("\nQ1 blog WFER bins (normalized | raw) -> mean forward Sharpe (frac losers):")
    for bn, br in zip(summary["wfer_bins_normalized"], summary["wfer_bins_raw"]):
        print(f"  {bn['bin']:26} n={bn['n']:5d} {bn['mean_fwd_sharpe']:+.3f} ({bn['frac_loser']:.0%})"
              f"   | n={br['n']:5d} {br['mean_fwd_sharpe']:+.3f} ({br['frac_loser']:.0%})")

    ill = summary["ill_conditioning_rolling"]["by_k"][1]  # k = 1.0
    print(f"\nQ1 ill-conditioning: aggregate |IS mean| < 1 se in {ill['frac_flagged']:.1%} of runs "
          f"(null {ill['frac_flagged_by_regime']['null']:.1%}); "
          f">=1 weak-IS window in {ill['frac_runs_with_weak_is_window']:.1%} of runs "
          f"(null {ill['frac_weak_window_by_regime']['null']:.1%}); "
          f"median max |per-window WFER| {ill['median_max_abs_fold_wfer_weak']:.1f} (weak) vs "
          f"{ill['median_max_abs_fold_wfer_strong']:.1f} (strong); "
          f"P(max>5) {ill['frac_max_fold_wfer_gt_5_weak']:.0%} vs {ill['frac_max_fold_wfer_gt_5_strong']:.0%}")

    print("\nQ2 anchored vs rolling (true forward Sharpe, paired):")
    for r in summary["anchored_vs_rolling"]:
        print(f"  {r['scope']:10} anchored {r['mean_fwd_anchored']:+.3f}  rolling {r['mean_fwd_rolling']:+.3f}  "
              f"diff {r['mean_diff_rolling_minus_anchored']:+.3f} (sem {r['sem_diff']:.3f}, "
              f"p {r['wilcoxon_p']:.1e})")

    print("\nQ4 procedure ranking (mean true forward Sharpe; estimate bias = OOS Sharpe - truth):")
    for r in summary["procedure_ranking"]:
        print(f"  {r['procedure']:9} fwd {r['mean_fwd_sharpe']:+.3f}  regret {r['mean_fwd_regret']:.3f}  "
              f"bias {r['mean_estimate_bias']:+.3f}  rho(est,fwd) {r['rho_estimate_vs_fwd']:+.3f}  "
              + " ".join(f"{reg[:4]} {r[f'mean_fwd_{reg}']:+.2f}" for reg in
                         ("stationary", "drift", "break", "null")))

    print("\nQ3 window sweep (rolling, history=756): train_len -> mean fwd / rho(WFER,fwd) / folds")
    for r in results["window_sweep"]:
        print(f"  L={r['train_len']:3d} (test {r['test_len']:3d}, {r['mean_n_folds']:.0f} folds)  "
              f"fwd {r['mean_fwd_sharpe']:+.3f}  rho_wfer {r['rho_wfer_fwd']:+.3f}  "
              f"rho_oos {r['rho_oos_sharpe_fwd']:+.3f}  "
              f"drift {r['mean_fwd_drift']:+.3f}  stat {r['mean_fwd_stationary']:+.3f}")


if __name__ == "__main__":
    main()

"""Generate the paper's figures (vector PDF) from the saved results.

    python -m wfo_experiments.figures      # writes paper/figures/*.pdf

Reads results/records.csv, results/records_sweep.csv and results/results.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import BLOG_WFER_LABELS
from .model import REGIMES, canonical_configs, optimal_center_path, true_sharpe_surface
from .procedures import build_folds

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGDIR = ROOT / "paper" / "figures"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9,
    "axes.labelsize": 9, "figure.dpi": 120, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})
REGIME_COLORS = {"stationary": "#1f3b73", "drift": "#e0a458", "break": "#c0392b", "null": "#9aa6b2"}
C_ANCH, C_ROLL = "#1f3b73", "#c0392b"


# --------------------------------------------------------------------------- #
# fig 1: setup — the time-varying surfaces + procedure windows
# --------------------------------------------------------------------------- #
def fig_setup(path: Path) -> None:
    cfgs = canonical_configs()
    fig = plt.figure(figsize=(11, 4.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.35, 1.0], hspace=0.55, wspace=0.3)

    for j, (name, cfg) in enumerate(cfgs.items()):
        ax = fig.add_subplot(gs[0, j])
        surf = true_sharpe_surface(cfg)
        im = ax.imshow(surf.T, aspect="auto", origin="lower", cmap="RdBu_r",
                       extent=(0, surf.shape[0], 0, 1), vmin=-1.3, vmax=1.3)
        ax.plot(np.arange(surf.shape[0]), optimal_center_path(cfg), color="k", lw=1.2, ls="--")
        ax.axvline(cfg.t_hist, color="k", lw=1.0)
        ax.text(cfg.t_hist + 15, 0.04, "forward", fontsize=7, rotation=90, va="bottom")
        ax.set_title(name)
        ax.set_xlabel("time $t$ (periods)")
        if j == 0:
            ax.set_ylabel(r"parameter $\theta$")
    fig.colorbar(im, ax=fig.axes, fraction=0.012, pad=0.01, label="true Sharpe")

    # bottom row: the procedure windows on one history
    cfg = cfgs["stationary"]
    rng = np.random.default_rng(0)
    for j, proc in enumerate(("anchored", "rolling", "kfold", "cpcv")):
        ax = fig.add_subplot(gs[1, j])
        folds = build_folds(proc, cfg.t_hist, cfg, rng)
        show = folds if len(folds) <= 8 else folds[:8]
        for fi, (tr, te) in enumerate(show):
            for seg_start, seg_stop in _contiguous(tr):
                ax.broken_barh([(seg_start, seg_stop - seg_start)], (fi - 0.35, 0.7),
                               color="#aebfdd", lw=0)
            for seg_start, seg_stop in _contiguous(te):
                ax.broken_barh([(seg_start, seg_stop - seg_start)], (fi - 0.35, 0.7),
                               color="#c0392b", lw=0)
        ax.set_ylim(-0.8, len(show) - 0.2)
        ax.set_xlim(0, cfg.t_hist)
        ax.invert_yaxis()
        ax.set_title(f"{proc}" + (" (first 8 of 15)" if proc == "cpcv" else ""), fontsize=8)
        ax.set_xlabel("time $t$")
        if j == 0:
            ax.set_ylabel("fold")
        ax.set_yticks([])
    fig.text(0.13, 0.475, "train", color="#7d96c4", fontsize=8)
    fig.text(0.18, 0.475, "test", color="#c0392b", fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def _contiguous(idx: np.ndarray) -> list[tuple[int, int]]:
    """Split a sorted index array into [start, stop) runs for plotting."""
    if idx.size == 0:
        return []
    brk = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[brk + 1]]
    stops = np.r_[idx[brk] + 1, idx[-1] + 1]
    return list(zip(starts.tolist(), stops.tolist()))


# --------------------------------------------------------------------------- #
# fig 2: WFER vs forward truth + calibration of the blog bins
# --------------------------------------------------------------------------- #
def fig_wfer(path: Path, df: pd.DataFrame, summary: dict) -> None:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.6, 3.3))
    wfer = df["rolling_wfer_norm"].clip(-1.5, 2.5)
    fwd = df["rolling_fwd_sharpe"]
    for reg in REGIMES:
        m = df["cfg_regime"] == reg
        axL.scatter(wfer[m], fwd[m], s=5, alpha=0.35, color=REGIME_COLORS[reg], label=reg, lw=0)
    for x in (0.3, 0.5, 0.8):
        axL.axvline(x, color="k", lw=0.6, ls=":")
    axL.axhline(0, color="k", lw=0.6)
    rho = next(r for r in summary["diagnostic_quality_rolling"] if r["diagnostic"] == "wfer_norm")["rho_fwd"]
    axL.set_xlabel("normalized WFER (clipped to [-1.5, 2.5])")
    axL.set_ylabel("true forward Sharpe")
    axL.set_title(f"WFER vs forward truth (Spearman {rho:.2f})")
    axL.legend(fontsize=6.5, markerscale=2.0, framealpha=0.9)

    # calibration of the blog bins (normalized vs raw WFER), skipping the IS<=0 row
    norm_bins = summary["wfer_bins_normalized"][1:]
    raw_bins = summary["wfer_bins_raw"][1:]
    x = np.arange(len(BLOG_WFER_LABELS))
    w = 0.38
    axR.bar(x - w / 2, [b["mean_fwd_sharpe"] for b in norm_bins], w,
            color="#1f3b73", label="normalized WFER")
    axR.bar(x + w / 2, [b["mean_fwd_sharpe"] for b in raw_bins], w,
            color="#9aa6b2", label="raw WFER (literal PnL-sum formula)")
    for xi, b in zip(x, norm_bins):
        if b["n"] > 0:
            axR.text(xi - w / 2, 0.02, f"{b['frac_loser']:.0%}", ha="center", fontsize=6, rotation=90)
    for xi, b in zip(x, raw_bins):
        if b["n"] > 0:
            axR.text(xi + w / 2, 0.02, f"{b['frac_loser']:.0%}", ha="center", fontsize=6, rotation=90)
    axR.axhline(0, color="k", lw=0.8)
    axR.set_xticks(x, BLOG_WFER_LABELS)
    axR.set_xlabel("folk WFER bin")
    axR.set_ylabel("mean true forward Sharpe")
    axR.set_title("What the 0.3/0.5/0.8 bins contain\n(text: fraction deployed losers)")
    axR.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# fig 3: anchored vs rolling by regime
# --------------------------------------------------------------------------- #
def fig_anchored_vs_rolling(path: Path, df: pd.DataFrame) -> None:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.4, 3.2))
    x = np.arange(len(REGIMES))
    w = 0.38
    means_a, sems_a, means_r, sems_r = [], [], [], []
    for reg in REGIMES:
        g = df[df["cfg_regime"] == reg]
        means_a.append(g["anchored_fwd_sharpe"].mean())
        sems_a.append(g["anchored_fwd_sharpe"].sem())
        means_r.append(g["rolling_fwd_sharpe"].mean())
        sems_r.append(g["rolling_fwd_sharpe"].sem())
    axL.bar(x - w / 2, means_a, w, yerr=sems_a, capsize=2, color=C_ANCH, label="anchored")
    axL.bar(x + w / 2, means_r, w, yerr=sems_r, capsize=2, color=C_ROLL, label="rolling")
    axL.axhline(0, color="k", lw=0.8)
    axL.set_xticks(x, REGIMES)
    axL.set_ylabel("mean true forward Sharpe")
    axL.set_title("Deployed forward Sharpe by regime")
    axL.legend(fontsize=7)

    diffs = [df.loc[df["cfg_regime"] == reg, "rolling_fwd_sharpe"]
             - df.loc[df["cfg_regime"] == reg, "anchored_fwd_sharpe"] for reg in REGIMES]
    parts = axR.violinplot(diffs, positions=x, widths=0.7, showmeans=True, showextrema=False)
    for body, reg in zip(parts["bodies"], REGIMES):
        body.set_facecolor(REGIME_COLORS[reg])
        body.set_alpha(0.6)
    axR.axhline(0, color="k", lw=0.8)
    axR.set_xticks(x, REGIMES)
    axR.set_ylabel(r"forward Sharpe: rolling $-$ anchored")
    axR.set_title("Paired difference (same simulated history)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# fig 4: window-design tradeoff
# --------------------------------------------------------------------------- #
def fig_window_design(path: Path, sweep: list[dict]) -> None:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.6, 3.2))
    lens = [r["train_len"] for r in sweep]
    for reg in REGIMES:
        axL.errorbar(lens, [r[f"mean_fwd_{reg}"] for r in sweep],
                     yerr=[r[f"sem_fwd_{reg}"] for r in sweep],
                     marker="o", ms=3, capsize=2, color=REGIME_COLORS[reg], label=reg)
    axL.axhline(0, color="k", lw=0.8)
    axL.set_xlabel("rolling train length (periods; test = train/3, history = 756)")
    axL.set_ylabel("mean true forward Sharpe")
    axL.set_title("Selection quality vs window length")
    axL.legend(fontsize=7)

    axR.plot(lens, [r["rho_wfer_fwd"] for r in sweep], "o-", color="#1f3b73",
             label=r"Spearman(WFER, forward)")
    axR.plot(lens, [r["rho_oos_sharpe_fwd"] for r in sweep], "s--", color="#c0392b",
             label=r"Spearman(OOS Sharpe, forward)")
    axR.set_xlabel("rolling train length (periods)")
    axR.set_ylabel("rank correlation with forward truth")
    ax2 = axR.twinx()
    ax2.spines.right.set_visible(True)
    ax2.plot(lens, [r["mean_n_folds"] for r in sweep], ":", color="#9aa6b2", label="# folds")
    ax2.set_ylabel("mean number of folds", color="#9aa6b2")
    ax2.tick_params(axis="y", colors="#9aa6b2")
    axR.set_title("Diagnostic quality vs window length")
    h1, l1 = axR.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axR.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    summary = json.loads((RESULTS / "results.json").read_text())
    # keep_default_na=False so the regime label "null" survives the round trip; numeric NaNs are
    # written by to_csv as empty strings, which na_values=[""] still maps back to NaN.
    df = pd.read_csv(RESULTS / "records.csv", keep_default_na=False, na_values=[""])
    fig_setup(FIGDIR / "fig_setup.pdf")
    fig_wfer(FIGDIR / "fig_wfer.pdf", df, summary["main"])
    fig_anchored_vs_rolling(FIGDIR / "fig_anchored_vs_rolling.pdf", df)
    fig_window_design(FIGDIR / "fig_window_design.pdf", summary["window_sweep"])
    print(f"wrote 4 figures to {FIGDIR}")


if __name__ == "__main__":
    main()

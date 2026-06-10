"""Fast sanity tests for the DGP, the validation procedures and the WFER bookkeeping.

Run: python -m pytest -q   (from the project root)
"""

from __future__ import annotations

from math import comb

import numpy as np
import pytest

from wfo_experiments import (
    PROCEDURES,
    build_folds,
    canonical_configs,
    forward_sharpe_by_theta,
    optimal_center_path,
    run_experiment,
    sample_experiment_config,
    select_theta,
    sharpe,
    simulate_returns,
    true_sharpe_surface,
    wfer_normalized,
    wfer_raw,
)


# --------------------------------------------------------------------------- #
# DGP
# --------------------------------------------------------------------------- #
def test_dgp_recovers_configured_sharpe():
    """With a huge stationary sample, the estimated Sharpe matches the configured surface."""
    cfg = canonical_configs()["stationary"]
    row = true_sharpe_surface(cfg)[:1]
    r = simulate_returns(np.repeat(row, 200_000, axis=0), cfg, np.random.default_rng(0))
    assert np.nanmax(np.abs(sharpe(r, cfg.periods_per_year) - row[0])) < 0.1


def test_factor_share_sets_cross_correlation():
    cfg = canonical_configs()["stationary"]
    r = simulate_returns(true_sharpe_surface(cfg)[:1].repeat(50_000, axis=0),
                         cfg, np.random.default_rng(1))
    corr = np.corrcoef(r.T)
    off = corr[np.triu_indices_from(corr, k=1)]
    assert abs(np.mean(off) - cfg.factor_share) < 0.05


def test_null_regime_has_no_edge_anywhere():
    """Null surfaces are non-positive everywhere, so every procedure deploys a non-winner."""
    rng = np.random.default_rng(2)
    for _ in range(10):
        cfg = sample_experiment_config(rng, regime="null")
        surf = true_sharpe_surface(cfg)
        assert surf.max() <= 0.0
        rec = run_experiment(cfg, rng)
        for proc in PROCEDURES:
            assert rec[f"{proc}_fwd_sharpe"] <= 0.0


def test_drift_and_break_center_paths():
    cfgs = canonical_configs()
    drift = optimal_center_path(cfgs["drift"])
    assert drift[0] == pytest.approx(cfgs["drift"].c0)
    assert drift[-1] == pytest.approx(cfgs["drift"].c1)
    assert np.all(np.diff(drift) > 0)
    brk_cfg = cfgs["break"]
    brk = optimal_center_path(brk_cfg)
    t_break = int(round(brk_cfg.t_break_frac * brk_cfg.t_hist))
    assert np.all(brk[:t_break] == brk_cfg.c0) and np.all(brk[t_break:] == brk_cfg.c1)


def test_forward_truth_matches_surface_mean():
    cfg = canonical_configs()["drift"]
    surf = true_sharpe_surface(cfg)
    fwd = forward_sharpe_by_theta(surf, cfg)
    assert fwd.shape == (cfg.n_theta,)
    assert np.allclose(fwd, surf[cfg.t_hist:].mean(axis=0))


def test_run_experiment_is_deterministic():
    cfg = canonical_configs()["break"]
    a = run_experiment(cfg, np.random.default_rng(123))
    b = run_experiment(cfg, np.random.default_rng(123))
    assert a == b


# --------------------------------------------------------------------------- #
# window bookkeeping
# --------------------------------------------------------------------------- #
def test_rolling_window_bookkeeping():
    cfg = canonical_configs()["stationary"]
    folds = build_folds("rolling", cfg.t_hist, cfg, np.random.default_rng(0))
    assert len(folds) == (cfg.t_hist - cfg.train_len) // cfg.test_len
    prev_end = None
    for tr, te in folds:
        assert tr.size == cfg.train_len and te.size == cfg.test_len
        assert tr[-1] + 1 == te[0]                      # test starts where train ends
        assert te[-1] < cfg.t_hist
        if prev_end is not None:
            assert te[0] == prev_end                    # OOS segments tile without overlap
        prev_end = te[-1] + 1


def test_anchored_window_expands_from_zero():
    cfg = canonical_configs()["stationary"]
    rolling = build_folds("rolling", cfg.t_hist, cfg, np.random.default_rng(0))
    anchored = build_folds("anchored", cfg.t_hist, cfg, np.random.default_rng(0))
    sizes = [tr.size for tr, _ in anchored]
    assert all(tr[0] == 0 and tr[-1] + 1 == te[0] for tr, te in anchored)
    assert sizes == sorted(sizes) and sizes[0] == cfg.train_len
    # anchored and rolling share identical test segments (paired comparison)
    assert all(np.array_equal(ta, tr) for (_, ta), (_, tr) in zip(anchored, rolling))


def test_no_train_test_overlap_any_procedure():
    cfg = canonical_configs()["stationary"]
    rng = np.random.default_rng(7)
    for proc in PROCEDURES:
        for tr, te in build_folds(proc, cfg.t_hist, cfg, rng):
            assert np.intersect1d(tr, te).size == 0
    kfolds = build_folds("kfold", cfg.t_hist, cfg, rng)
    covered = np.sort(np.concatenate([te for _, te in kfolds]))
    assert np.array_equal(covered, np.arange(cfg.t_hist))  # k-fold tests partition the history


def test_cpcv_embargo_and_combinations():
    cfg = canonical_configs()["stationary"]
    folds = build_folds("cpcv", cfg.t_hist, cfg, np.random.default_rng(0))
    assert len(folds) == comb(cfg.cpcv_groups, cfg.cpcv_k_test)
    for tr, te in folds:
        # no train index within `embargo` periods of any test index
        for boundary in (te[0], te[-1]):
            assert np.all(np.abs(tr - boundary) > 0)
        dists = np.min(np.abs(tr[:, None] - te[None, ::max(te.size // 8, 1)]), axis=1)
        assert dists.min() > 0
        a, b = te[0], te[-1] + 1
        assert not np.any((tr >= a - cfg.embargo) & (tr < a))
        assert not np.any((tr >= b) & (tr < b + cfg.embargo))


def test_selection_ignores_test_data():
    """Explicit leakage test: massively perturbing a fold's TEST rows must not change its selection."""
    cfg = canonical_configs()["drift"]
    rng = np.random.default_rng(11)
    returns = simulate_returns(true_sharpe_surface(cfg)[: cfg.t_hist], cfg, rng)
    for proc in PROCEDURES:
        for tr, te in build_folds(proc, cfg.t_hist, cfg, np.random.default_rng(5)):
            base = select_theta(returns, tr, cfg.periods_per_year)
            perturbed = returns.copy()
            perturbed[te, :] = 0.0
            perturbed[te, (base + 7) % cfg.n_theta] = 100.0   # make a wrong theta look amazing OOS
            assert select_theta(perturbed, tr, cfg.periods_per_year) == base


# --------------------------------------------------------------------------- #
# WFER computation
# --------------------------------------------------------------------------- #
def test_wfer_raw_and_normalized():
    is_pnls, oos_pnls = np.array([2.0, 2.0]), np.array([1.0, 1.0])
    assert wfer_raw(is_pnls, oos_pnls) == pytest.approx(0.5)
    # IS windows 100 periods each, OOS 50 each: per-period PnL identical -> normalized WFER = 1
    assert wfer_normalized(is_pnls, oos_pnls, n_is=200, n_oos=100) == pytest.approx(1.0)
    assert np.isnan(wfer_raw(np.array([0.0]), np.array([1.0])))


def test_record_wfer_consistent_with_fold_pnls():
    cfg = canonical_configs()["stationary"]
    rng = np.random.default_rng(3)
    returns = simulate_returns(true_sharpe_surface(cfg)[: cfg.t_hist], cfg, rng)
    from wfo_experiments.procedures import run_procedure

    res = run_procedure("rolling", returns, cfg, rng)
    assert res["wfer_raw"] == pytest.approx(res["oos_pnls"].sum() / res["is_pnls"].sum())
    # recompute each fold's PnL straight from the fold indices
    for (tr, te), th, isp, oosp in zip(res["folds"], res["fold_thetas"],
                                       res["is_pnls"], res["oos_pnls"]):
        assert isp == pytest.approx(returns[tr, th].sum())
        assert oosp == pytest.approx(returns[te, th].sum())


# --------------------------------------------------------------------------- #
# directional sanity (small Monte-Carlo, fixed seeds)
# --------------------------------------------------------------------------- #
def _mean_over_seeds(cfg, field: str, n: int = 40) -> float:
    seeds = np.random.SeedSequence(99).spawn(n)
    return float(np.mean([run_experiment(cfg, np.random.default_rng(s),
                                         procedures=("anchored", "rolling"))[field]
                          for s in seeds]))


def test_rolling_beats_anchored_under_strong_drift():
    cfg = canonical_configs()["drift"]
    diff = (_mean_over_seeds(cfg, "rolling_fwd_sharpe")
            - _mean_over_seeds(cfg, "anchored_fwd_sharpe"))
    assert diff > 0.05


def test_anchored_not_worse_under_stationarity():
    cfg = canonical_configs()["stationary"]
    diff = (_mean_over_seeds(cfg, "anchored_fwd_sharpe")
            - _mean_over_seeds(cfg, "rolling_fwd_sharpe"))
    assert diff > -0.02  # more data, same optimum: anchored should not lose


def test_records_have_all_procedure_fields():
    rng = np.random.default_rng(4)
    cfg = sample_experiment_config(rng, regime="drift")
    rec = run_experiment(cfg, rng)
    for proc in PROCEDURES:
        for f in ("fwd_sharpe", "fwd_regret", "wfer_raw", "wfer_norm", "oos_sharpe",
                  "oos_tstat", "rankcorr", "frac_pos", "theta_std", "n_folds"):
            assert f"{proc}_{f}" in rec
        assert rec[f"{proc}_n_folds"] >= 1
        assert np.isfinite(rec[f"{proc}_fwd_sharpe"])

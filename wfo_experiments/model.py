"""Generative model: a 1-D strategy family with a KNOWN, time-varying Sharpe surface.

We model walk-forward validation as selecting, from a grid of candidate configurations indexed by
``theta`` in ``[0, 1]``, the one that maximizes an in-sample Sharpe estimate, then deploying it over a
FUTURE horizon. The population (true) annualized Sharpe surface ``SR_true(theta, t)`` is what we
control, so for every simulated experiment the true forward Sharpe of any selection is computable
exactly — that is the ground truth every validation procedure is judged against.

The surface is a Gaussian bump in ``theta`` whose center ``c(t)`` moves through time according to one
of four regimes:

* ``stationary`` — ``c(t) = c0`` for the whole timeline (history + forward horizon);
* ``drift``      — ``c(t)`` moves linearly from ``c0`` to ``c1`` across the whole timeline;
* ``break``      — ``c(t) = c0`` until a (sampled) break time inside the history, then jumps to ``c1``;
* ``null``       — no edge anywhere: amplitude 0 and a non-positive base level.

Returns carry a shared market factor (cross-strategy correlation ``factor_share``), as in the sibling
plateau project, so nearby strategies co-move — the structure walk-forward folds actually face.

Every constant that affects results is either a sampled config field (recorded per experiment) or a
disclosed keyword default of :func:`sample_experiment_config` — there are no hidden constants.
Everything is deterministic given a seeded ``numpy`` Generator.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

PERIODS_PER_YEAR = 252  # daily bars -> annualization factor sqrt(252)

Regime = Literal["stationary", "drift", "break", "null"]
REGIMES: tuple[Regime, ...] = ("stationary", "drift", "break", "null")


@dataclass(frozen=True)
class ExperimentConfig:
    """Ground truth + window design for one simulated validation problem (all fields recorded)."""

    # --- true surface -----------------------------------------------------
    regime: str = "stationary"
    n_theta: int = 31            # grid points on [0, 1]
    t_hist: int = 756            # history length available to the validation procedures
    t_fwd: int = 252             # forward horizon over which ground truth is evaluated
    amp: float = 1.0             # height of the Gaussian bump (annualized Sharpe units); 0 for null
    width: float = 0.20          # Gaussian sigma of the bump (in theta units)
    base: float = 0.0            # Sharpe floor away from the bump (<= 0 for null)
    c0: float = 0.5              # bump center at t = 0
    c1: float = 0.5              # bump center at the end (drift) / after the break (break)
    t_break_frac: float = 0.5    # break time as a fraction of t_hist (break regime only)
    factor_share: float = 0.3    # fraction of return variance from the shared factor, in [0, 1)
    sr_cap: float = 4.0          # clip true annualized Sharpe to a sane ceiling
    periods_per_year: int = PERIODS_PER_YEAR

    # --- validation-window design ------------------------------------------
    train_len: int = 252         # rolling train window; also the first anchored train window
    test_len: int = 63           # OOS test window per fold (folds are aligned across anchored/rolling)
    split_test_frac: float = 0.3  # single train/test split: fraction of history held out at the end
    k_folds: int = 5             # naive shuffled k-fold CV
    cpcv_groups: int = 6         # CPCV-lite: contiguous groups
    cpcv_k_test: int = 2         # CPCV-lite: test groups per combination
    embargo: int = 10            # CPCV-lite: periods dropped from train at each test-group boundary

    label: str = "custom"        # bookkeeping tag


# --------------------------------------------------------------------------- #
# true surface
# --------------------------------------------------------------------------- #
def theta_grid(cfg: ExperimentConfig) -> np.ndarray:
    """Shared parameter grid, shape ``(n_theta,)`` on [0, 1]."""
    return np.linspace(0.0, 1.0, cfg.n_theta)


def optimal_center_path(cfg: ExperimentConfig) -> np.ndarray:
    """True bump center ``c(t)`` for every period of history + forward horizon, shape ``(t_total,)``."""
    t_total = cfg.t_hist + cfg.t_fwd
    if cfg.regime == "drift":
        u = np.arange(t_total) / max(t_total - 1, 1)
        return cfg.c0 + (cfg.c1 - cfg.c0) * u
    if cfg.regime == "break":
        t_break = int(round(cfg.t_break_frac * cfg.t_hist))
        path = np.full(t_total, cfg.c0)
        path[t_break:] = cfg.c1
        return path
    return np.full(t_total, cfg.c0)  # stationary and null


def true_sharpe_surface(cfg: ExperimentConfig) -> np.ndarray:
    """Population annualized Sharpe at every (t, theta), shape ``(t_hist + t_fwd, n_theta)``."""
    theta = theta_grid(cfg)[None, :]
    centers = optimal_center_path(cfg)[:, None]
    sr = cfg.base + cfg.amp * np.exp(-((theta - centers) ** 2) / (2.0 * cfg.width**2))
    return np.clip(sr, -cfg.sr_cap, cfg.sr_cap)


def forward_sharpe_by_theta(surface: np.ndarray, cfg: ExperimentConfig) -> np.ndarray:
    """True expected annualized Sharpe of deploying each theta over the forward horizon, shape (n_theta,).

    Per-period returns have mean ``SR_true(theta, t) / sqrt(P)`` and unit variance, so the deployed
    Sharpe over the forward window is the time-average of the surface there (the O(SR^2/P) variance
    contribution from a moving mean is negligible at these Sharpe levels).
    """
    return surface[cfg.t_hist :].mean(axis=0)


# --------------------------------------------------------------------------- #
# return simulation
# --------------------------------------------------------------------------- #
def simulate_returns(sr_rows: np.ndarray, cfg: ExperimentConfig, rng: np.random.Generator) -> np.ndarray:
    """Simulate per-period returns for a block of surface rows, shape ``(t, n_theta)``.

    Each column has unit per-period variance; a shared factor injects cross-strategy correlation
    ``factor_share``. Per-period mean is ``sr_rows / sqrt(periods_per_year)`` so the population
    annualized Sharpe at time t equals ``sr_rows[t]``.
    """
    t, n = sr_rows.shape
    c = float(cfg.factor_share)
    per_period_mean = sr_rows / np.sqrt(cfg.periods_per_year)
    factor = rng.standard_normal((t, 1))
    idio = rng.standard_normal((t, n))
    shocks = np.sqrt(c) * factor + np.sqrt(1.0 - c) * idio
    return per_period_mean + shocks


def simulate_history(cfg: ExperimentConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(returns, surface)``: simulated history ``(t_hist, n_theta)`` + the full true surface."""
    surface = true_sharpe_surface(cfg)
    returns = simulate_returns(surface[: cfg.t_hist], cfg, rng)
    return returns, surface


def sharpe(returns: np.ndarray, periods_per_year: int = PERIODS_PER_YEAR, axis: int = 0) -> np.ndarray:
    """Annualized Sharpe ratio per column (ddof=1), guarding zero variance."""
    mu = returns.mean(axis=axis)
    sd = returns.std(axis=axis, ddof=1)
    sd = np.where(sd <= 1e-12, np.nan, sd)
    return mu / sd * np.sqrt(periods_per_year)


# --------------------------------------------------------------------------- #
# random config sampler for Monte-Carlo batches
# --------------------------------------------------------------------------- #
def sample_experiment_config(
    rng: np.random.Generator,
    *,
    regime: Regime | Literal["random"] = "random",
    n_theta_choices: tuple[int, ...] = (25, 31, 41),
    t_hist_choices: tuple[int, ...] = (504, 756, 1008),
    t_fwd: int = 252,
    amp_range: tuple[float, float] = (0.4, 2.0),
    width_range: tuple[float, float] = (0.10, 0.35),
    base_range: tuple[float, float] = (-0.2, 0.2),
    null_base_range: tuple[float, float] = (-0.10, 0.0),
    c0_range: tuple[float, float] = (0.25, 0.75),
    shift_range: tuple[float, float] = (0.2, 0.5),
    break_frac_range: tuple[float, float] = (0.3, 0.9),
    factor_share_range: tuple[float, float] = (0.05, 0.5),
    train_frac_choices: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5),
    test_frac_of_train_choices: tuple[float, ...] = (0.25, 1.0 / 3.0),
    split_test_frac_choices: tuple[float, ...] = (0.25, 0.30, 0.35),
    min_test_len: int = 21,
    k_folds: int = 5,
    cpcv_groups: int = 6,
    cpcv_k_test: int = 2,
    embargo: int = 10,
    sr_cap: float = 4.0,
) -> ExperimentConfig:
    """Draw one fully-specified experiment; every sampled value lands in a recorded config field.

    Window design follows the blog post's own rules of thumb: train is a disclosed fraction of the
    history, test is 25-33% of train, and the rolling step equals the test length so OOS segments
    never overlap (train windows then overlap by 67-75%, near the post's "~50% overlap" advice).
    """
    reg: str = str(rng.choice(REGIMES)) if regime == "random" else regime
    t_hist = int(rng.choice(t_hist_choices))
    c0 = float(rng.uniform(*c0_range))

    if reg == "null":
        amp, base, c1 = 0.0, float(rng.uniform(*null_base_range)), c0
    else:
        amp = float(rng.uniform(*amp_range))
        base = float(rng.uniform(*base_range))
        if reg == "stationary":
            c1 = c0
        else:  # drift or break: optimum moves by a sampled shift in a random direction
            shift = float(rng.uniform(*shift_range)) * float(rng.choice([-1.0, 1.0]))
            c1 = float(np.clip(c0 + shift, 0.05, 0.95))

    train_len = int(round(float(rng.choice(train_frac_choices)) * t_hist))
    test_len = max(min_test_len, int(round(float(rng.choice(test_frac_of_train_choices)) * train_len)))

    return ExperimentConfig(
        regime=reg,
        n_theta=int(rng.choice(n_theta_choices)),
        t_hist=t_hist,
        t_fwd=t_fwd,
        amp=amp,
        width=float(rng.uniform(*width_range)),
        base=base,
        c0=c0,
        c1=c1,
        t_break_frac=float(rng.uniform(*break_frac_range)),
        factor_share=float(rng.uniform(*factor_share_range)),
        sr_cap=sr_cap,
        train_len=train_len,
        test_len=test_len,
        split_test_frac=float(rng.choice(split_test_frac_choices)),
        k_folds=k_folds,
        cpcv_groups=cpcv_groups,
        cpcv_k_test=cpcv_k_test,
        embargo=embargo,
        label=reg,
    )


def canonical_configs() -> dict[str, ExperimentConfig]:
    """Illustrative fixed configs (one per regime) used by the setup figure and sanity tests."""
    common = dict(n_theta=41, t_hist=756, t_fwd=252, width=0.18, factor_share=0.3,
                  train_len=252, test_len=63)
    return {
        "stationary": ExperimentConfig(regime="stationary", amp=1.2, base=0.0, c0=0.4, c1=0.4,
                                       label="stationary", **common),
        "drift": ExperimentConfig(regime="drift", amp=1.2, base=0.0, c0=0.2, c1=0.8,
                                  label="drift", **common),
        "break": ExperimentConfig(regime="break", amp=1.2, base=0.0, c0=0.25, c1=0.75,
                                  t_break_frac=0.6, label="break", **common),
        "null": ExperimentConfig(regime="null", amp=0.0, base=-0.05, c0=0.5, c1=0.5,
                                 label="null", **common),
    }


def with_windows(cfg: ExperimentConfig, *, train_len: int, test_len: int,
                 t_hist: int | None = None) -> ExperimentConfig:
    """Copy of ``cfg`` with a forced window design (used by the window-design sweep)."""
    kw: dict = {"train_len": int(train_len), "test_len": int(test_len)}
    if t_hist is not None:
        kw["t_hist"] = int(t_hist)
    return replace(cfg, **kw)

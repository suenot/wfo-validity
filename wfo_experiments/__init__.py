"""Controlled validation of walk-forward optimization and the walk-forward efficiency ratio."""

from .model import (
    REGIMES,
    ExperimentConfig,
    canonical_configs,
    forward_sharpe_by_theta,
    optimal_center_path,
    sample_experiment_config,
    sharpe,
    simulate_history,
    simulate_returns,
    theta_grid,
    true_sharpe_surface,
    with_windows,
)
from .procedures import (
    PROCEDURES,
    build_folds,
    deployment_train_idx,
    run_procedure,
    select_theta,
    wfer_normalized,
    wfer_raw,
)
from .simulate import run_batch, run_experiment, window_sweep

__all__ = [
    "REGIMES",
    "PROCEDURES",
    "ExperimentConfig",
    "canonical_configs",
    "sample_experiment_config",
    "theta_grid",
    "optimal_center_path",
    "true_sharpe_surface",
    "forward_sharpe_by_theta",
    "simulate_returns",
    "simulate_history",
    "sharpe",
    "with_windows",
    "build_folds",
    "deployment_train_idx",
    "select_theta",
    "run_procedure",
    "wfer_raw",
    "wfer_normalized",
    "run_experiment",
    "run_batch",
    "window_sweep",
]
__version__ = "0.1.0"

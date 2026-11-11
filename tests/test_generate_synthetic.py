"""Unit tests for the synthetic GBM trajectory generator
(data/generate_synthetic.py)."""

from __future__ import annotations

import numpy as np
import pytest

from data.generate_synthetic import simulate_gbm


def test_simulate_gbm_shapes_and_positivity():
    t_grid, x_path = simulate_gbm(x0=8.0, t_horizon=4.0, dt=1e-3, seed=42)
    assert t_grid.shape == x_path.shape
    assert t_grid[0] == 0.0
    assert np.isclose(t_grid[-1], 4.0)
    assert np.all(x_path > 0.0)  # geometric Brownian motion never hits 0
    assert x_path[0] == pytest.approx(8.0)


def test_simulate_gbm_is_reproducible_with_same_seed():
    _, path_a = simulate_gbm(8.0, 2.0, 1e-3, seed=7)
    _, path_b = simulate_gbm(8.0, 2.0, 1e-3, seed=7)
    assert np.array_equal(path_a, path_b)


def test_simulate_gbm_different_seeds_differ():
    _, path_a = simulate_gbm(8.0, 2.0, 1e-3, seed=1)
    _, path_b = simulate_gbm(8.0, 2.0, 1e-3, seed=2)
    assert not np.array_equal(path_a, path_b)


def test_simulate_gbm_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        simulate_gbm(-1.0, 1.0, 0.1, seed=0)
    with pytest.raises(ValueError):
        simulate_gbm(1.0, 0.0, 0.1, seed=0)
    with pytest.raises(ValueError):
        simulate_gbm(1.0, 1.0, 2.0, seed=0)


def test_simulate_gbm_log_drift_matches_closed_form():
    # E[log(X_t/X_0)] = (DRIFT_COEF - 0.5*VOL_COEF^2) * t = -t for Sec. 8
    # parameters; check this over many independent trajectories.
    log_returns = []
    for seed in range(200):
        _, x_path = simulate_gbm(1.0, 1.0, 1e-2, seed=seed)
        log_returns.append(np.log(x_path[-1]))
    mean_log_return = np.mean(log_returns)
    assert mean_log_return == pytest.approx(-1.0, abs=0.15)

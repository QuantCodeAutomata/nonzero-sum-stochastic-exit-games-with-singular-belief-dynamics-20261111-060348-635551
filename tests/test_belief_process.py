"""Unit tests for the belief-process ODE (Eq. 5.1) and its limit
construction (src/belief_process.py), checked on a small synthetic
trajectory so the suite runs quickly."""

from __future__ import annotations

import numpy as np

from src.belief_process import ThetaUniform, belief_process_limit, solve_Z_delta
from src.model import build_model, solve_delta_family


def _small_model_and_path():
    model = build_model(0.5, 5.0, n_theta=40)
    t_grid = np.linspace(0.0, 1.0, 101)
    x_path = 8.0 * np.exp(-t_grid)  # deterministic decaying path, no randomness needed
    return model, t_grid, x_path


def test_theta_uniform_inverse_cdf_roundtrip():
    dist = ThetaUniform(0.5, 5.0)
    z = np.linspace(0.0, 1.0, 11)
    theta = dist.Y(z)
    assert np.allclose(dist.Z(theta), z)
    assert theta[0] == 0.5 and theta[-1] == 5.0


def test_Z_delta_stays_in_unit_interval():
    model, t_grid, x_path = _small_model_and_path()
    z_path = solve_Z_delta(t_grid, x_path, delta=0.1, theta_dist=model.theta_dist,
                            boundaries=model.boundaries, M_hat_fn=model.M_hat)
    assert z_path[0] == 1.0
    assert np.all(z_path >= 0.0) and np.all(z_path <= 1.0)


def test_Z_delta_is_nonincreasing():
    # dZ/dt = -Z*lambda^delta with lambda^delta >= 0 (Eq. 5.1): the
    # approximating belief process can only decrease.
    model, t_grid, x_path = _small_model_and_path()
    z_path = solve_Z_delta(t_grid, x_path, delta=0.1, theta_dist=model.theta_dist,
                            boundaries=model.boundaries, M_hat_fn=model.M_hat)
    assert np.all(np.diff(z_path) <= 1e-9)


def test_Z_delta_monotone_in_delta():
    # Lemma 5.9: Z_t^delta is non-decreasing in delta (a larger delta damps
    # the rate lambda^delta less aggressively... i.e. caps the denominator
    # less, giving a *smaller* rate and hence a *larger* Z_t^delta).
    model, t_grid, x_path = _small_model_and_path()
    deltas = (1.0, 0.1, 0.01)
    paths = solve_delta_family(t_grid, x_path, deltas, model)
    z_1 = paths[1.0]
    z_01 = paths[0.1]
    z_001 = paths[0.01]
    # pointwise ordering Z^{1.0} >= Z^{0.1} >= Z^{0.01}
    assert np.all(z_1 >= z_01 - 1e-6)
    assert np.all(z_01 >= z_001 - 1e-6)


def test_belief_process_limit_is_pointwise_infimum():
    paths = [np.array([1.0, 0.8, 0.5]), np.array([1.0, 0.6, 0.4]), np.array([1.0, 0.9, 0.3])]
    z_limit = belief_process_limit(paths)
    assert np.array_equal(z_limit, np.array([1.0, 0.6, 0.3]))


def test_no_lookahead_in_rhs_evaluation():
    # The ODE right-hand side at time t depends on x_path interpolated only
    # at t itself (via the shared interp1d), not on any future value of the
    # trajectory: truncating the path after t must not change Z up to t.
    model, t_grid, x_path = _small_model_and_path()
    z_full = solve_Z_delta(t_grid, x_path, delta=0.1, theta_dist=model.theta_dist,
                            boundaries=model.boundaries, M_hat_fn=model.M_hat)
    half = len(t_grid) // 2
    z_half = solve_Z_delta(t_grid[: half + 1], x_path[: half + 1], delta=0.1,
                            theta_dist=model.theta_dist, boundaries=model.boundaries,
                            M_hat_fn=model.M_hat)
    assert np.allclose(z_full[: half + 1], z_half, atol=1e-6)

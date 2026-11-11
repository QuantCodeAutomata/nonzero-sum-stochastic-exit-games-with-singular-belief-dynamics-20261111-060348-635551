"""Unit tests for the resolvents D_hat, M_hat (src/resolvents.py).

Cross-checks the closed-form Green's function formula (Eq. 2.5) against an
independent Monte Carlo estimate of the same expectation, and checks the
boundedness and monotonicity properties stated in Remark 3.2 and Lemma 3.1 of
the paper.
"""

from __future__ import annotations

import numpy as np

from src.config import M_PREMIUM, R
from src.resolvents import D_profit, M_profit, resolvent_at_point


def _monte_carlo_resolvent(x0: float, profit_fn, n_paths: int, t_max: float,
                            dt: float, seed: int) -> float:
    """E_x0[int_0^t_max e^{-r t} f(X_t) dt] estimated by simulating n_paths
    independent geometric Brownian motion paths with the Section 8
    parameters (drift -0.5, volatility 1) and a trapezoidal time integral."""
    rng = np.random.default_rng(seed)
    n_steps = int(round(t_max / dt))
    t_grid = np.linspace(0.0, t_max, n_steps + 1)
    increments = rng.normal(0.0, np.sqrt(dt), size=(n_paths, n_steps))
    brownian = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(increments, axis=1)], axis=1)
    paths = x0 * np.exp(-t_grid[None, :] + brownian)
    discounted = np.exp(-R * t_grid)[None, :] * profit_fn(paths)
    integrals = np.trapz(discounted, t_grid, axis=1)
    return float(np.mean(integrals))


def test_D_hat_matches_monte_carlo():
    x0 = 2.0
    analytic = resolvent_at_point(x0, D_profit)
    mc = _monte_carlo_resolvent(x0, D_profit, n_paths=20000, t_max=60.0, dt=0.01, seed=1)
    assert abs(analytic - mc) < 0.05 * max(analytic, 1e-6) + 0.02


def test_M_hat_matches_monte_carlo():
    x0 = 2.0
    analytic = resolvent_at_point(x0, M_profit)
    mc = _monte_carlo_resolvent(x0, M_profit, n_paths=20000, t_max=60.0, dt=0.01, seed=2)
    assert abs(analytic - mc) < 0.05 * max(analytic, 1e-6) + 0.02


def test_M_hat_equals_D_hat_plus_premium_over_r():
    # Linearity of the resolvent: M = D + M_PREMIUM implies
    # M_hat(x) = D_hat(x) + M_PREMIUM / r exactly.
    for x in (0.05, 0.5, 5.0, 50.0, 500.0):
        d_hat = resolvent_at_point(x, D_profit)
        m_hat = resolvent_at_point(x, M_profit)
        assert np.isclose(m_hat, d_hat + M_PREMIUM / R, rtol=1e-8)


def test_resolvent_is_nonnegative_and_bounded():
    x = np.geomspace(1e-4, 1e4, 30)
    values = np.array([resolvent_at_point(xi, D_profit) for xi in x])
    assert np.all(values >= 0.0)
    assert np.all(values < 50.0)  # D is bounded by sqrt(X_M), so D_hat <= sqrt(X_M)/r


def test_resolvent_vanishes_as_x_to_zero():
    # Remark 3.2 / boundedness of D_hat combined with D(0) = 0: D_hat(x) -> 0
    # as x -> 0 for this profit function.
    small = resolvent_at_point(1e-6, D_profit)
    larger = resolvent_at_point(1e-2, D_profit)
    assert small < larger
    assert small < 0.05

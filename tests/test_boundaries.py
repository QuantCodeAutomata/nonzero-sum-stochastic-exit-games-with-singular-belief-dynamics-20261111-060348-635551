"""Unit tests for the optimal stopping boundaries alpha(theta), beta(theta)
(src/boundaries.py), checked against the qualitative properties stated in
the paper (Assumption 2.3, Proposition 2.4, Remark 3.2) rather than against
figure-read pixel coordinates."""

from __future__ import annotations

import numpy as np

from src.boundaries import find_boundary, tabulate_boundaries
from src.resolvents import D_profit, M_profit, build_resolvent_interpolator


def test_boundary_is_nonnegative():
    D_hat = build_resolvent_interpolator(D_profit)
    for theta in (0.5, 1.0, 3.0, 5.0):
        boundary, _ = find_boundary(theta, D_hat)
        assert boundary >= 0.0


def test_alpha_is_nondecreasing_in_theta():
    # Larger exit value theta makes exit more attractive, hence raises the
    # exit boundary alpha(theta) (monotone comparative statics, Section 3).
    D_hat = build_resolvent_interpolator(D_profit)
    theta_grid = np.linspace(0.5, 5.0, 12)
    boundaries, _ = tabulate_boundaries(theta_grid, D_hat)
    assert np.all(np.diff(boundaries) >= -1e-6)


def test_beta_is_zero_for_small_theta_upper_bound():
    # Section 8 / reduced model (Figure 2): M(x) = D(x) + 2 >= 2 for all x, so
    # for theta <= 2 we have r*theta <= 2 <= M(x) everywhere, which forces
    # beta(theta) = 0 for every theta in [0.5, 2] (condition (ii) of
    # Assumption 2.3 combined with Proposition 2.4's bound beta <= c_M).
    M_hat = build_resolvent_interpolator(M_profit)
    theta_grid = np.linspace(0.5, 2.0, 8)
    boundaries, _ = tabulate_boundaries(theta_grid, M_hat)
    assert np.allclose(boundaries, 0.0, atol=1e-2)


def test_beta_at_theta_l_is_zero_for_full_model():
    # Same bound applies pointwise: beta(0.5) = 0 because M(x) >= 2 > r*0.5
    # everywhere, independent of theta_U (Figure 1 model, theta ~ U[0.5, 5]).
    M_hat = build_resolvent_interpolator(M_profit)
    boundary, _ = find_boundary(0.5, M_hat)
    assert boundary < 1e-2


def test_alpha_positive_for_large_theta():
    D_hat = build_resolvent_interpolator(D_profit)
    boundary, _ = find_boundary(5.0, D_hat)
    assert boundary > 0.1

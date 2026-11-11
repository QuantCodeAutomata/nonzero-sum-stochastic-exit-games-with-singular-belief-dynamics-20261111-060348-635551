"""Unit tests for the fundamental solutions phi, psi and the scale/speed
densities (src/fundamental_solutions.py)."""

from __future__ import annotations

import numpy as np

from src.config import SQRT3
from src.fundamental_solutions import phi, psi, scale_density, speed_density, wronskian


def test_phi_is_decreasing():
    x = np.geomspace(1e-3, 1e3, 50)
    values = phi(x)
    assert np.all(np.diff(values) < 0), "phi must be strictly decreasing"


def test_psi_is_increasing():
    x = np.geomspace(1e-3, 1e3, 50)
    values = psi(x)
    assert np.all(np.diff(values) > 0), "psi must be strictly increasing"


def test_phi_diverges_at_zero_psi_vanishes_at_zero():
    assert phi(1e-8) > phi(1e-4) > phi(1.0)
    assert psi(1e-8) < psi(1e-4) < psi(1.0)


def test_indicial_roots_match_sqrt3():
    # (L - r)u = 0 for u = x^p gives p^2 - 2p - 2 = 0 for DRIFT_COEF=-0.5, R=1,
    # whose roots are p = 1 +/- sqrt(3) (Section 8, paper's methodology step 1).
    p_minus, p_plus = 1.0 - SQRT3, 1.0 + SQRT3
    assert abs(p_minus**2 - 2 * p_minus - 2) < 1e-9
    assert abs(p_plus**2 - 2 * p_plus - 2) < 1e-9


def test_wronskian_value():
    assert np.isclose(wronskian(), -2.0 * SQRT3)


def test_wronskian_matches_direct_differentiation():
    x = np.array([0.3, 1.0, 2.5, 17.0])
    h = 1e-6
    dphi = (phi(x + h) - phi(x - h)) / (2 * h)
    dpsi = (psi(x + h) - psi(x - h)) / (2 * h)
    w_numeric = (dphi * psi(x) - dpsi * phi(x)) / scale_density(x)
    assert np.allclose(w_numeric, wronskian(), rtol=1e-4)


def test_speed_density_positive_and_scaling():
    x = np.geomspace(1e-2, 1e2, 20)
    m_prime = speed_density(x)
    assert np.all(m_prime > 0)
    assert np.allclose(m_prime, 2.0 / x**3)


def test_phi_psi_require_positive_domain():
    import pytest

    with pytest.raises(ValueError):
        phi(0.0)
    with pytest.raises(ValueError):
        psi(-1.0)

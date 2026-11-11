"""Fundamental solutions of (L - r)u = 0 for the Section 8 geometric Brownian
motion, and the associated scale and speed densities.

The generator is L = (1/2) sigma(x)^2 u'' + mu(x) u' with mu(x) = DRIFT_COEF*x
and sigma(x) = VOL_COEF*x. Substituting u(x) = x^p gives the indicial equation
(1/2) p(p-1) + DRIFT_COEF * p - R = 0, whose roots for the Section 8
parameters (DRIFT_COEF=-0.5, R=1) are p = 1 +/- sqrt(3) (paper_pipeline
methodology, Step 1). phi is the decreasing solution (p = 1 - sqrt(3)) and psi
the increasing solution (p = 1 + sqrt(3)).
"""

from __future__ import annotations

import numpy as np

from src.config import PHI_EXPONENT, PSI_EXPONENT


def phi(x: np.ndarray | float) -> np.ndarray | float:
    """Decreasing fundamental solution phi(x) = x^(1 - sqrt(3))."""
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0):
        raise ValueError("phi is defined for x > 0 only")
    return np.power(x, PHI_EXPONENT)


def psi(x: np.ndarray | float) -> np.ndarray | float:
    """Increasing fundamental solution psi(x) = x^(1 + sqrt(3))."""
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0):
        raise ValueError("psi is defined for x > 0 only")
    return np.power(x, PSI_EXPONENT)


def scale_density(x: np.ndarray | float) -> np.ndarray | float:
    """Scale density s'(x) = x for this geometric Brownian motion.

    Derived from s'(x) = exp(-int 2*mu(x)/sigma(x)^2 dx) with
    mu(x) = -0.5*x, sigma(x) = x, which integrates to s'(x) = x.
    """
    return np.asarray(x, dtype=float)


def speed_density(x: np.ndarray | float) -> np.ndarray | float:
    """Speed density m'(x) = 2 / (sigma(x)^2 * s'(x)) = 2 / x^3.

    The factor 2 corresponds to the generator convention
    L = (1/2) sigma(x)^2 u'' + mu(x) u' used throughout the paper, consistent
    with the Wronskian value computed in `wronskian`.
    """
    x = np.asarray(x, dtype=float)
    return 2.0 / x**3


def wronskian() -> float:
    """Wronskian w = (phi' psi - psi' phi) / s'(x), constant in x.

    For phi(x) = x^(1-sqrt(3)), psi(x) = x^(1+sqrt(3)):
        phi'(x) psi(x) - psi'(x) phi(x) = -2*sqrt(3)*x,
    and dividing by s'(x) = x gives w = -2*sqrt(3), independent of x
    (verified symbolically; see tests/test_fundamental_solutions.py).
    """
    from src.config import SQRT3

    return -2.0 * SQRT3

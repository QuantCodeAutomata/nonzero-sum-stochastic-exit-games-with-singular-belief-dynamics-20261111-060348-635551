"""Profit flow functions D, M and their resolvents D-hat, M-hat.

The resolvent of a bounded profit function f under the Section 8 geometric
Brownian motion is (Eq. 2.5)

    f_hat(x) = E_x[ int_0^inf e^{-r t} f(X_t) dt ].

Because both boundaries of I = (0, infinity) are natural (singular) for this
diffusion, f_hat is given by the Green's function formula of a regular
one-dimensional diffusion (Borodin & Salminen, Handbook of Brownian Motion,
Ch. II.10, cited by the paper as reference [4]):

    f_hat(x) = (1 / |w|) * [ phi(x) * int_0^x psi(y) f(y) m'(y) dy
                             + psi(x) * int_x^inf phi(y) f(y) m'(y) dy ],

pairing the *increasing* solution psi with the inner integral (over (0, x])
and the *decreasing* solution phi with the outer integral (over [x, infinity)).
This pairing (rather than the opposite one) is the one for which both
integrals converge for the Section 8 data: near y = 0 the integrand of the
inner integral behaves like y^(sqrt(3) - 3/2), which is integrable since
sqrt(3) - 3/2 > -1, while near y = infinity the integrand of the outer
integral behaves like y^(-2 - sqrt(3)), integrable since -2 - sqrt(3) < -1.
With the opposite pairing neither integral converges. The resulting f_hat is
bounded on (0, infinity), consistent with Remark 3.2 and Lemma 3.1 of the
paper, which is verified numerically in tests/test_resolvents.py both by
checking boundedness and by comparison with Monte Carlo simulation of the
defining expectation (Eq. 2.5).
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.interpolate import PchipInterpolator

from src.config import (
    M_PREMIUM,
    QUAD_LIMIT,
    RESOLVENT_N_GRID,
    RESOLVENT_X_MAX,
    RESOLVENT_X_MIN,
    SQRT3,
    X_M,
)
from src.fundamental_solutions import phi, psi, speed_density, wronskian


def D_profit(x: np.ndarray | float) -> np.ndarray | float:
    """Duopoly profit flow D(x) = min(sqrt(x), sqrt(X_M)) (Sec. 8)."""
    x = np.asarray(x, dtype=float)
    return np.minimum(np.sqrt(x), np.sqrt(X_M))


def M_profit(x: np.ndarray | float) -> np.ndarray | float:
    """Monopoly profit flow M(x) = D(x) + M_PREMIUM (Sec. 8)."""
    return D_profit(x) + M_PREMIUM


def _inner_integrand(y: float, profit_fn) -> float:
    return float(psi(y) * profit_fn(y) * speed_density(y))


def _outer_integrand(y: float, profit_fn) -> float:
    return float(phi(y) * profit_fn(y) * speed_density(y))


# For y >= X_M both profit flows are the constant C = sqrt(X_M) (+ M_PREMIUM
# for M), so psi(y)*m'(y) = 2*y^(sqrt(3)-2) and phi(y)*m'(y) = 2*y^(-sqrt(3)-2)
# have elementary antiderivatives there. Using these closed forms instead of
# numerical quadrature over the constant tail avoids the poor conditioning of
# scipy.integrate.quad on a slowly-decaying semi-infinite integrand.
_INNER_TAIL_EXPONENT = SQRT3 - 1.0  # antiderivative exponent of 2*y^(sqrt(3)-2)
_OUTER_TAIL_EXPONENT = -SQRT3 - 1.0  # antiderivative exponent of 2*y^(-sqrt(3)-2)


def _inner_tail_antiderivative(y: float, const: float) -> float:
    return 2.0 * const * y**_INNER_TAIL_EXPONENT / _INNER_TAIL_EXPONENT


def _outer_tail_antiderivative(y: float, const: float) -> float:
    return 2.0 * const * y**_OUTER_TAIL_EXPONENT / _OUTER_TAIL_EXPONENT


def resolvent_at_point(x: float, profit_fn) -> float:
    """Evaluate the Green's function resolvent of `profit_fn` at a single x > 0."""
    if x <= 0:
        raise ValueError("resolvent_at_point requires x > 0")
    w = abs(wronskian())
    const = float(profit_fn(np.array([X_M]))[0])  # profit flow for y >= X_M

    inner_below = min(x, X_M)
    with warnings.catch_warnings():
        # The two quad calls below integrate a smooth, bounded integrand over
        # a finite range; the occasional roundoff warning reflects quad's
        # internal extrapolation only and does not affect the result, which
        # is cross-checked against Monte Carlo in tests/test_resolvents.py.
        warnings.simplefilter("ignore", category=IntegrationWarning)
        inner, _ = quad(_inner_integrand, 0.0, inner_below, args=(profit_fn,), limit=QUAD_LIMIT)
        if x < X_M:
            outer, _ = quad(_outer_integrand, x, X_M, args=(profit_fn,), limit=QUAD_LIMIT)
        else:
            outer = 0.0
    if x > X_M:
        inner += _inner_tail_antiderivative(x, const) - _inner_tail_antiderivative(X_M, const)
    # Tail from max(x, X_M) to infinity, in closed form: lim_{y->inf} y^(-sqrt3-1) = 0.
    outer += 0.0 - _outer_tail_antiderivative(max(x, X_M), const)

    return (phi(x) * inner + psi(x) * outer) / w


def build_resolvent_interpolator(
    profit_fn,
    x_min: float = RESOLVENT_X_MIN,
    x_max: float = RESOLVENT_X_MAX,
    n_grid: int = RESOLVENT_N_GRID,
) -> PchipInterpolator:
    """Tabulate the resolvent of `profit_fn` on a log-spaced grid and build a
    monotone cubic (PCHIP) interpolator for fast repeated evaluation.

    Evaluation beyond [x_min, x_max] is clamped to the boundary grid value,
    which is consistent with the resolvent's asymptotic behaviour: f_hat(x)
    -> 0 as x -> 0 and f_hat(x) -> a finite constant as x -> infinity.
    """
    grid = np.geomspace(x_min, x_max, n_grid)
    values = np.array([resolvent_at_point(x, profit_fn) for x in grid])
    interpolator = PchipInterpolator(grid, values, extrapolate=False)
    x_lo, x_hi = grid[0], grid[-1]
    v_lo, v_hi = values[0], values[-1]

    def evaluate(x: np.ndarray | float) -> np.ndarray | float:
        x_arr = np.asarray(x, dtype=float)
        x_clamped = np.clip(x_arr, x_lo, x_hi)
        out = interpolator(x_clamped)
        out = np.where(x_arr < x_lo, v_lo * (x_arr / x_lo), out)
        out = np.where(x_arr > x_hi, v_hi, out)
        if np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0):
            return float(out)
        return out

    evaluate.grid = grid
    evaluate.values = values
    return evaluate

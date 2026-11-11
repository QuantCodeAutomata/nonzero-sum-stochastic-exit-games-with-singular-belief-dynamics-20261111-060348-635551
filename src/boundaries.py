"""Optimal stopping boundaries alpha(theta), beta(theta) (Assumption 2.3).

For theta in [theta_L, theta_U], alpha(theta) = argmax_x a_theta(x) and
beta(theta) = argmax_x b_theta(x), with

    a_theta(x) = (theta - D_hat(x)) / phi(x),      b_theta(x) = (theta - M_hat(x)) / phi(x).

Remark 3.2 of the paper shows that a_theta(x) -> 0 and b_theta(x) -> 0 as
x -> x_L = 0 (because phi(x) -> infinity while D_hat(x), M_hat(x) -> 0 there).
The maximiser is found by a coarse log-spaced grid search followed by a local
refinement with `scipy.optimize.minimize_scalar`. If the coarse grid maximum
occurs at the smallest grid point (i.e. the function is still increasing
towards x = 0), the boundary is reported as 0, matching the boundary value
a_theta(0) = b_theta(0) = 0 given in Remark 3.2.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from src.config import BOUNDARY_N_COARSE, BOUNDARY_X_MAX, BOUNDARY_X_MIN
from src.fundamental_solutions import phi


def a_or_b_theta(x: np.ndarray | float, theta: float, hat_fn) -> np.ndarray | float:
    """Generic objective (theta - f_hat(x)) / phi(x) for f in {D, M}."""
    return (theta - hat_fn(x)) / phi(x)


def find_boundary(
    theta: float,
    hat_fn,
    x_min: float = BOUNDARY_X_MIN,
    x_max: float = BOUNDARY_X_MAX,
    n_coarse: int = BOUNDARY_N_COARSE,
) -> tuple[float, float]:
    """Return (boundary, objective value at the boundary) for a single theta."""
    grid = np.geomspace(x_min, x_max, n_coarse)
    objective = a_or_b_theta(grid, theta, hat_fn)
    i_max = int(np.argmax(objective))

    if i_max == 0:
        # The objective is still increasing as x decreases towards the grid's
        # smallest point: the true supremum is the boundary limit, which is 0
        # (Remark 3.2).
        return 0.0, 0.0

    lo = grid[max(i_max - 1, 0)]
    hi = grid[min(i_max + 1, n_coarse - 1)]
    neg_objective = lambda x: -a_or_b_theta(x, theta, hat_fn)
    result = minimize_scalar(neg_objective, bounds=(lo, hi), method="bounded",
                              options={"xatol": 1e-9})
    x_star = float(result.x)
    val_star = -float(result.fun)
    if val_star < objective[i_max]:
        # Numerical refinement failed to improve on the coarse grid optimum;
        # fall back to the coarse grid value.
        x_star, val_star = float(grid[i_max]), float(objective[i_max])
    return x_star, val_star


def tabulate_boundaries(
    theta_grid: np.ndarray, hat_fn
) -> tuple[np.ndarray, np.ndarray]:
    """Tabulate (boundary(theta), objective(boundary(theta))) on `theta_grid`."""
    boundaries = np.empty_like(theta_grid, dtype=float)
    values = np.empty_like(theta_grid, dtype=float)
    for i, theta in enumerate(theta_grid):
        boundaries[i], values[i] = find_boundary(float(theta), hat_fn)
    return boundaries, values

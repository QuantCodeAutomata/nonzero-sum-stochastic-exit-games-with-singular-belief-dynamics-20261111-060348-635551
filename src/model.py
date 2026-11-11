"""Shared model-assembly pipeline used by every experiment (Section 8).

`build_model` ties together the fundamental solutions, resolvents and
optimal-stopping boundaries for a given support [theta_l, theta_u] of the
uniformly distributed exit value theta, producing the object that
exp/exp1.py, exp/exp2.py and exp/exp3.py all consume. There is exactly one
implementation of each shared component: resolvents.py, boundaries.py and
belief_process.py are never duplicated or re-implemented in exp/.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.belief_process import BoundaryInterpolators, ThetaUniform
from src.boundaries import tabulate_boundaries
from src.config import THETA_N_GRID
from src.resolvents import D_profit, M_profit, build_resolvent_interpolator


@dataclass
class ModelComponents:
    """All model objects needed to simulate the belief process for one
    exit-value support [theta_l, theta_u]."""

    theta_dist: ThetaUniform
    D_hat: Callable[[np.ndarray], np.ndarray]
    M_hat: Callable[[np.ndarray], np.ndarray]
    theta_grid: np.ndarray
    alpha_grid: np.ndarray
    a_at_alpha_grid: np.ndarray
    beta_grid: np.ndarray
    b_at_beta_grid: np.ndarray
    boundaries: BoundaryInterpolators

    def alpha(self, theta: np.ndarray | float) -> np.ndarray | float:
        """Duopoly exit boundary alpha(theta)."""
        return self.boundaries.alpha(theta)

    def beta(self, theta: np.ndarray | float) -> np.ndarray | float:
        """Monopoly exit boundary beta(theta)."""
        return self.boundaries.beta(theta)


def build_model(
    theta_l: float, theta_u: float, n_theta: int = THETA_N_GRID
) -> ModelComponents:
    """Build the resolvents and tabulate alpha(theta), beta(theta) on
    [theta_l, theta_u] (Sections 3-4 of the paper).

    The resolvents D_hat, M_hat depend only on the profit flows D, M and the
    common geometric Brownian motion (Sec. 8), not on theta_l, theta_u, so
    they are identical across Figures 1-3 up to numerical interpolation noise.
    """
    if not (0.0 < theta_l < theta_u):
        raise ValueError("require 0 < theta_l < theta_u")

    theta_dist = ThetaUniform(theta_l, theta_u)
    D_hat = build_resolvent_interpolator(D_profit)
    M_hat = build_resolvent_interpolator(M_profit)

    theta_grid = np.linspace(theta_l, theta_u, n_theta)
    alpha_grid, a_at_alpha_grid = tabulate_boundaries(theta_grid, D_hat)
    beta_grid, b_at_beta_grid = tabulate_boundaries(theta_grid, M_hat)

    boundaries = BoundaryInterpolators(
        theta_grid, alpha_grid, a_at_alpha_grid, beta_grid, b_at_beta_grid
    )
    return ModelComponents(
        theta_dist=theta_dist,
        D_hat=D_hat,
        M_hat=M_hat,
        theta_grid=theta_grid,
        alpha_grid=alpha_grid,
        a_at_alpha_grid=a_at_alpha_grid,
        beta_grid=beta_grid,
        b_at_beta_grid=b_at_beta_grid,
        boundaries=boundaries,
    )


def solve_delta_family(
    t_grid: np.ndarray,
    x_path: np.ndarray,
    delta_grid: tuple[float, ...],
    model: ModelComponents,
) -> dict[float, np.ndarray]:
    """Solve the approximating ODE (Eq. 5.1) for every delta in `delta_grid`
    along the single path `x_path`, returning {delta: Z_t^delta path}."""
    from src.belief_process import solve_Z_delta

    paths: dict[float, np.ndarray] = {}
    for delta in delta_grid:
        method = "Radau" if delta <= 0.1 else "RK45"
        paths[delta] = solve_Z_delta(
            t_grid, x_path, delta, model.theta_dist, model.boundaries, model.M_hat,
            method=method,
        )
    return paths

"""Synthetic data generation: simulated trajectories of the underlying
geometric Brownian motion (X_t).

No external market data is used anywhere in this replication: Section 8 of
the paper specifies the underlying process as

    dX_t = DRIFT_COEF * X_t dt + VOL_COEF * X_t dW_t,  DRIFT_COEF = -0.5, VOL_COEF = 1,

simulated here with its closed-form solution X_t = X_0 * exp(-t + W_t),
consistent with d(log X_t) = (DRIFT_COEF - 0.5 * VOL_COEF^2) dt + VOL_COEF dW_t.
"""

from __future__ import annotations

import numpy as np

from src.config import DRIFT_COEF, VOL_COEF


def simulate_gbm(
    x0: float, t_horizon: float, dt: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a single geometric Brownian motion trajectory.

    Parameters
    ----------
    x0 : initial value X_0 > 0.
    t_horizon : simulation horizon T.
    dt : time step of the fine grid used for the Euler-equivalent exact solution.
    seed : seed of the random generator, fixing a single common trajectory.

    Returns
    -------
    t_grid : array of shape (n_steps + 1,), time points from 0 to T.
    x_path : array of shape (n_steps + 1,), X_t at each time point.
    """
    if x0 <= 0:
        raise ValueError("x0 must be strictly positive")
    if t_horizon <= 0:
        raise ValueError("t_horizon must be strictly positive")
    if dt <= 0 or dt > t_horizon:
        raise ValueError("dt must be in (0, t_horizon]")

    n_steps = int(round(t_horizon / dt))
    t_grid = np.linspace(0.0, t_horizon, n_steps + 1)
    rng = np.random.default_rng(seed)
    increments = rng.normal(loc=0.0, scale=np.sqrt(dt), size=n_steps)
    brownian = np.concatenate(([0.0], np.cumsum(increments)))
    log_drift = DRIFT_COEF - 0.5 * VOL_COEF**2  # = -1 for Section 8 parameters
    x_path = x0 * np.exp(log_drift * t_grid + VOL_COEF * brownian)
    return t_grid, x_path

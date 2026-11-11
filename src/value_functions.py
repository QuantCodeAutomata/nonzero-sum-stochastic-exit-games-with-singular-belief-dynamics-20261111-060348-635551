"""Value functions d(x; theta) and m(x; theta) for the duopoly and monopoly
optimal stopping problems (Eq. 3.2 and its analogue for m).

    d(x; theta) = a_theta(alpha(theta)) * phi(x) + D_hat(x),  x > alpha(theta)
                = theta,                                     x <= alpha(theta)

and symmetrically for m with beta(theta), M_hat in place of alpha(theta), D_hat.
"""

from __future__ import annotations

import numpy as np

from src.fundamental_solutions import phi


def value_function(
    x: np.ndarray | float,
    theta: float,
    boundary: float,
    objective_at_boundary: float,
    hat_fn,
) -> np.ndarray | float:
    """Generic value function (Eq. 3.2) for a single theta."""
    x_arr = np.asarray(x, dtype=float)
    scalar_input = x_arr.ndim == 0
    x_arr = np.atleast_1d(x_arr)
    out = np.where(
        x_arr > boundary,
        objective_at_boundary * phi(np.clip(x_arr, 1e-300, None)) + hat_fn(x_arr),
        theta,
    )
    return float(out[0]) if scalar_input else out

"""Plotting utilities for the (X_t, Z_t) trajectory figures (Figures 1 and 2)
and the belief-process convergence table/plot (Figure 3)."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt


def plot_state_space_trajectory(
    x_path: np.ndarray,
    z_path: np.ndarray,
    alpha_of_theta,
    beta_of_theta,
    theta_dist,
    x_max: float,
    out_path: str,
    title: str,
    has_jump_region: bool = True,
) -> None:
    """Plot the (X_t, Z_t) trajectory with the jump / decay / constant regions.

    `alpha_of_theta` and `beta_of_theta` are callables theta -> boundary value.
    The boundary curves are drawn as x = alpha(Y(z)) and x = beta(Y(z)) for
    z in [0, 1], matching the state-space picture described in Section 4 of
    the paper (discussion preceding Figure 1).
    """
    z_grid = np.linspace(0.0, 1.0, 400)
    theta_grid = theta_dist.Y(z_grid)
    alpha_curve = np.clip(alpha_of_theta(theta_grid), 0.0, x_max)
    beta_curve = np.clip(beta_of_theta(theta_grid), 0.0, x_max) if has_jump_region else np.zeros_like(z_grid)

    fig, ax = plt.subplots(figsize=(7, 6))

    if has_jump_region:
        ax.fill_betweenx(z_grid, 0.0, beta_curve, color="0.85", label="Jump region")
    ax.fill_betweenx(z_grid, beta_curve, alpha_curve, color="0.6", label="Continuous decay region")
    # Constant (white) region is the unfilled background above alpha_curve.

    ax.plot(alpha_curve, z_grid, "k--", linewidth=1.2, label=r"$\alpha(Y(z))$")
    if has_jump_region:
        ax.plot(beta_curve, z_grid, "k:", linewidth=1.2, label=r"$\beta(Y(z))$")

    ax.plot(x_path, z_path, "b-", linewidth=1.8, label=r"$(X_t, Z_t)$")
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_delta_convergence(
    t_grid: np.ndarray,
    z_delta_paths: dict,
    z_limit: np.ndarray,
    out_path: str,
    title: str,
) -> None:
    """Plot Z_t^delta for each delta together with the pointwise limit."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for delta, path in z_delta_paths.items():
        ax.plot(t_grid, path, label=rf"$\delta={delta}$", linewidth=1.2)
    ax.plot(t_grid, z_limit, "k--", linewidth=2.0, label=r"limit $Z_t$")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$Z_t^\delta$")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

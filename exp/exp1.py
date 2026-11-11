"""Experiment 1: numerical illustration of the full model with monopoly exit
(Figure 1 of Cao & Palczewski, "Nonzero-sum stochastic exit games with
singular belief dynamics").

Exit values theta are uniform on [0.5, 5], large enough that the monopolist
optimally exits for some types (Section 8). This produces market collapse
(both firms exit in finite time) and clustering (firms of different types
exit simultaneously). The experiment tabulates the optimal stopping
boundaries alpha(theta), beta(theta), constructs the belief process (Z_t) as
the limit of the approximations (Z_t^delta), and plots the trajectory
(X_t, Z_t) with the jump / continuous-decay / constant regions.

Run as: python -m exp.exp1
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data.generate_synthetic import simulate_gbm
from src.belief_process import belief_process_limit
from src.config import (
    DELTA_GRID_FIGS_1_2,
    DT_SIMULATION,
    FIG1_REFERENCE,
    RANDOM_SEED,
    T_HORIZON,
    THETA_L,
    THETA_U_FIG1,
    X0_TRAJECTORY,
)
from src.model import build_model, solve_delta_family
from src.plotting import plot_delta_convergence, plot_state_space_trajectory

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run() -> dict:
    """Run Experiment 1 end to end and save all artifacts to results/."""
    RESULTS_DIR.mkdir(exist_ok=True)

    model = build_model(THETA_L, THETA_U_FIG1)

    t_grid, x_path = simulate_gbm(X0_TRAJECTORY, T_HORIZON, DT_SIMULATION, RANDOM_SEED)

    z_delta_paths = solve_delta_family(t_grid, x_path, DELTA_GRID_FIGS_1_2, model)
    z_limit = belief_process_limit(list(z_delta_paths.values()))

    beta_0_5 = float(model.beta(0.5))
    alpha_0_5 = float(model.alpha(0.5))
    alpha_5 = float(model.alpha(5.0))
    theta_at_z_0_75 = float(model.theta_dist.Y(0.75))
    beta_at_z_0_75 = float(model.beta(theta_at_z_0_75))

    jump_region = x_path[0] < beta_0_5 or x_path[0] < model.beta(model.theta_dist.Y(z_limit[0]))
    z_jump_detected = bool(z_limit[0] < 1.0 - 1e-6)

    plot_state_space_trajectory(
        x_path, z_limit, model.alpha, model.beta, model.theta_dist,
        x_max=float(max(10.0, x_path.max() * 1.05)),
        out_path=str(RESULTS_DIR / "exp1_fig1_trajectory.png"),
        title="Figure 1: full model with monopoly exit, theta ~ U[0.5, 5]",
        has_jump_region=True,
    )
    plot_delta_convergence(
        t_grid, z_delta_paths, z_limit,
        out_path=str(RESULTS_DIR / "exp1_delta_convergence.png"),
        title="Convergence of Z_t^delta to the limit (Figure 1 model)",
    )

    collapse_time = None
    collapsed = np.where(z_limit <= 1e-6)[0]
    if collapsed.size > 0:
        collapse_time = float(t_grid[collapsed[0]])

    metrics = {
        "experiment": "exp_1",
        "theta_l": THETA_L,
        "theta_u": THETA_U_FIG1,
        "x0_trajectory": X0_TRAJECTORY,
        "t_horizon": T_HORIZON,
        "dt_simulation": DT_SIMULATION,
        "random_seed": RANDOM_SEED,
        "computed": {
            "beta_0_5": beta_0_5,
            "alpha_0_5": alpha_0_5,
            "alpha_5": alpha_5,
            "beta_at_theta_0_75_quantile": beta_at_z_0_75,
        },
        "reference_figure_1": FIG1_REFERENCE,
        "z0": float(z_limit[0]),
        "z_jump_at_t0": z_jump_detected,
        "market_collapse_time": collapse_time,
        "z_final": float(z_limit[-1]),
    }
    with open(RESULTS_DIR / "exp1_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Experiment 1 complete.")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    run()

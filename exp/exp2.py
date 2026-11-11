"""Experiment 2: numerical illustration of the reduced model without monopoly
exit (Figure 2 of Cao & Palczewski, "Nonzero-sum stochastic exit games with
singular belief dynamics").

Exit values theta are uniform on [0.5, 2], small enough that the monopolist
never exits the market: beta(theta) = 0 for every theta in this range
(Section 8). This reduces the problem to the setting of Kwon & Palczewski
[23]. The same underlying trajectory of (X_t) as in Experiment 1 is used, so
that the two figures can be compared directly (Section 8).

Run as: python -m exp.exp2
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
    FIG2_REFERENCE,
    RANDOM_SEED,
    T_HORIZON,
    THETA_L,
    THETA_U_FIG2,
    X0_TRAJECTORY,
)
from src.model import build_model, solve_delta_family
from src.plotting import plot_delta_convergence, plot_state_space_trajectory

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run() -> dict:
    """Run Experiment 2 end to end and save all artifacts to results/."""
    RESULTS_DIR.mkdir(exist_ok=True)

    model = build_model(THETA_L, THETA_U_FIG2)

    # Same random seed and horizon as Experiment 1 (Section 8): "the
    # trajectory of the process (X_t) is the same as in Figure 1."
    t_grid, x_path = simulate_gbm(X0_TRAJECTORY, T_HORIZON, DT_SIMULATION, RANDOM_SEED)

    beta_theta_grid = model.beta(model.theta_grid)
    beta_is_zero = bool(np.allclose(beta_theta_grid, 0.0, atol=1e-3))

    z_delta_paths = solve_delta_family(t_grid, x_path, DELTA_GRID_FIGS_1_2, model)
    z_limit = belief_process_limit(list(z_delta_paths.values()))

    alpha_0_5 = float(model.alpha(0.5))
    alpha_2 = float(model.alpha(2.0))
    theta_at_z_0_55 = float(model.theta_dist.Y(0.55))
    theta_at_z_0_85 = float(model.theta_dist.Y(0.85))
    theta_at_z_0_98 = float(model.theta_dist.Y(0.98))
    alpha_at_z_0_55 = float(model.alpha(theta_at_z_0_55))
    alpha_at_z_0_85 = float(model.alpha(theta_at_z_0_85))
    alpha_at_z_0_98 = float(model.alpha(theta_at_z_0_98))

    plot_state_space_trajectory(
        x_path, z_limit, model.alpha, model.beta, model.theta_dist,
        x_max=float(max(10.0, x_path.max() * 1.05)),
        out_path=str(RESULTS_DIR / "exp2_fig2_trajectory.png"),
        title="Figure 2: reduced model without monopoly exit, theta ~ U[0.5, 2]",
        has_jump_region=False,
    )
    plot_delta_convergence(
        t_grid, z_delta_paths, z_limit,
        out_path=str(RESULTS_DIR / "exp2_delta_convergence.png"),
        title="Convergence of Z_t^delta to the limit (Figure 2 model)",
    )

    collapse_time = None
    collapsed = np.where(z_limit <= 1e-6)[0]
    if collapsed.size > 0:
        collapse_time = float(t_grid[collapsed[0]])

    metrics = {
        "experiment": "exp_2",
        "theta_l": THETA_L,
        "theta_u": THETA_U_FIG2,
        "x0_trajectory": X0_TRAJECTORY,
        "t_horizon": T_HORIZON,
        "dt_simulation": DT_SIMULATION,
        "random_seed": RANDOM_SEED,
        "beta_identically_zero": beta_is_zero,
        "max_abs_beta": float(np.max(np.abs(beta_theta_grid))),
        "computed": {
            "alpha_0_5": alpha_0_5,
            "alpha_at_z_0_55": alpha_at_z_0_55,
            "alpha_at_z_0_85": alpha_at_z_0_85,
            "alpha_at_z_0_98": alpha_at_z_0_98,
            "alpha_2": alpha_2,
        },
        "reference_figure_2": FIG2_REFERENCE,
        "z0": float(z_limit[0]),
        "market_collapse_time": collapse_time,
        "z_final": float(z_limit[-1]),
    }
    with open(RESULTS_DIR / "exp2_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Experiment 2 complete.")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    run()

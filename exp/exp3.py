"""Experiment 3: convergence of the belief-process approximations (Z_t^delta)
for decreasing delta (Figure 3 of Cao & Palczewski, "Nonzero-sum stochastic
exit games with singular belief dynamics").

Uses the same model (theta ~ U[0.5, 5]) and the same underlying trajectory of
(X_t) as Experiment 1. Tabulates Z_t^delta at the time points t = 0, 0.5, ...,
4.0 for delta in {10, 5, 2, 1, 0.5, 0.1, 0.05, 0.02} (Section 8) and checks
that the family is monotone in delta and converges to the limit process,
which exhibits a jump at t = 0 when X_0 lies in the jump region.

Run as: python -m exp.exp3
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data.generate_synthetic import simulate_gbm
from src.belief_process import belief_process_limit
from src.config import (
    DELTA_GRID_FIG3,
    DT_SIMULATION,
    FIG3_TIME_POINTS,
    RANDOM_SEED,
    T_HORIZON,
    THETA_L,
    THETA_U_FIG1,
    X0_TRAJECTORY,
)
from src.model import build_model, solve_delta_family
from src.plotting import plot_delta_convergence

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run() -> dict:
    """Run Experiment 3 end to end and save all artifacts to results/."""
    RESULTS_DIR.mkdir(exist_ok=True)

    model = build_model(THETA_L, THETA_U_FIG1)
    t_grid, x_path = simulate_gbm(X0_TRAJECTORY, T_HORIZON, DT_SIMULATION, RANDOM_SEED)

    z_delta_paths = solve_delta_family(t_grid, x_path, DELTA_GRID_FIG3, model)
    z_limit = belief_process_limit(list(z_delta_paths.values()))

    time_idx = [int(round(t / DT_SIMULATION)) for t in FIG3_TIME_POINTS]
    table = {}
    for delta in DELTA_GRID_FIG3:
        path = z_delta_paths[delta]
        table[str(delta)] = [float(path[i]) for i in time_idx]
    table["limit"] = [float(z_limit[i]) for i in time_idx]

    # Monotonicity check: for each fixed t, Z_t^delta is non-decreasing in
    # delta (Lemma 5.9: larger delta gives a slower decrease of the
    # approximating process).
    deltas_sorted = sorted(DELTA_GRID_FIG3)
    values_sorted = np.array([table[str(d)] for d in deltas_sorted])  # (n_delta, n_t)
    diffs = np.diff(values_sorted, axis=0)
    monotone_violations = int(np.sum(diffs < -1e-6))
    max_violation = float(np.min(diffs)) if diffs.size else 0.0

    jump_at_t0 = bool(z_limit[0] < 1.0 - 1e-6)

    plot_delta_convergence(
        t_grid, z_delta_paths, z_limit,
        out_path=str(RESULTS_DIR / "exp3_delta_convergence.png"),
        title="Figure 3: convergence of Z_t^delta as delta -> 0 (theta ~ U[0.5, 5])",
    )

    metrics = {
        "experiment": "exp_3",
        "theta_l": THETA_L,
        "theta_u": THETA_U_FIG1,
        "x0_trajectory": X0_TRAJECTORY,
        "t_horizon": T_HORIZON,
        "dt_simulation": DT_SIMULATION,
        "random_seed": RANDOM_SEED,
        "delta_grid": list(DELTA_GRID_FIG3),
        "time_points": list(FIG3_TIME_POINTS),
        "table": table,
        "monotone_violations": monotone_violations,
        "max_monotonicity_violation": max_violation,
        "jump_at_t0": jump_at_t0,
        "z0": float(z_limit[0]),
    }
    with open(RESULTS_DIR / "exp3_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Experiment 3 complete.")
    print(json.dumps({k: v for k, v in metrics.items() if k != "table"}, indent=2))
    print("Table (delta -> Z_t^delta at reference time points):")
    print(json.dumps(table, indent=2))
    return metrics


if __name__ == "__main__":
    run()

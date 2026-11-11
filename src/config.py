"""Model and numerical parameters for the replication.

All parameters that define the numerical illustration are taken from Section 8
("Numerical illustration") of Cao & Palczewski, "Nonzero-sum stochastic exit
games with singular belief dynamics". Parameters needed only for the numerical
scheme (grids, tolerances, simulation horizon) are not specified in the paper
and are chosen conservatively; they are marked as such below.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Underlying diffusion: dX_t = DRIFT_COEF * X_t dt + VOL_COEF * X_t dW_t
# Section 8: dX_t = -0.5 X_t dt + X_t dW_t, state space I = (0, infinity).
# ---------------------------------------------------------------------------
DRIFT_COEF = -0.5  # drift coefficient mu(x) = DRIFT_COEF * x (Sec. 8)
VOL_COEF = 1.0  # diffusion coefficient sigma(x) = VOL_COEF * x (Sec. 8)
R = 1.0  # discount rate r (Sec. 8)

# Fundamental solutions phi (decreasing) and psi (increasing) of (L-r)u=0 for
# this geometric Brownian motion are power functions x^p with
# p = 1 +/- sqrt(1 - 2*DRIFT_COEF/VOL_COEF^2 ... ) which for the Sec. 8
# parameters reduces to p = 1 +/- sqrt(3) (paper_pipeline methodology, Step 1).
SQRT3 = np.sqrt(3.0)
PHI_EXPONENT = 1.0 - SQRT3  # decreasing solution exponent, approx -0.732
PSI_EXPONENT = 1.0 + SQRT3  # increasing solution exponent, approx 2.732

# ---------------------------------------------------------------------------
# Profit flow functions (Sec. 8):
# D(x) = min(sqrt(x), sqrt(X_M)); M(x) = D(x) + M_PREMIUM.
# ---------------------------------------------------------------------------
X_M = 1000.0  # duopoly profit cap threshold (Sec. 8); verified numerically by
# the authors to be large enough for Assumption 2.3 to hold
M_PREMIUM = 2.0  # monopoly premium over duopoly profit, M(x) = D(x) + 2 (Sec. 8)

# ---------------------------------------------------------------------------
# Distribution of exit values theta ~ U[THETA_L, THETA_U]
# ---------------------------------------------------------------------------
THETA_L = 0.5  # lower bound of exit-value support, common to Figures 1-3 (Sec. 8)
THETA_U_FIG1 = 5.0  # upper bound for Figure 1 / Figure 3: monopoly exit allowed (Sec. 8)
THETA_U_FIG2 = 2.0  # upper bound for Figure 2: monopoly never exits (Sec. 8)

# ---------------------------------------------------------------------------
# Resolvent computation (Green's function quadrature) -- numerical scheme,
# not specified in the paper beyond the definition (2.5).
# ---------------------------------------------------------------------------
RESOLVENT_X_MIN = 1e-3  # smallest grid point for resolvent interpolation
RESOLVENT_X_MAX = 2.0e3  # largest grid point for resolvent interpolation
RESOLVENT_N_GRID = 400  # number of log-spaced grid points
QUAD_LIMIT = 200  # scipy.integrate.quad subdivision limit

# ---------------------------------------------------------------------------
# Optimal stopping boundaries alpha(theta), beta(theta): coarse-grid + refine
# search for the global maximiser of a_theta, b_theta (Assumption 2.3) --
# numerical scheme, not specified in the paper.
# ---------------------------------------------------------------------------
BOUNDARY_X_MIN = 1e-6  # smallest x considered in the boundary search
BOUNDARY_X_MAX = 50.0  # largest x considered in the boundary search; the
# reference values alpha(5) ~ 0.85 and beta(Y(0.75)) ~ 1.00 in Sec. 8 lie
# well within this range
BOUNDARY_N_COARSE = 2000  # number of log-spaced points in the coarse search
THETA_N_GRID = 300  # number of theta grid points on which alpha, beta are
# tabulated and then linearly interpolated

# ---------------------------------------------------------------------------
# GBM trajectory simulation -- the exact initial value, horizon and time step
# used by the authors for Figures 1-3 are not specified in the paper; chosen
# here (uncertainty recorded in results/RESULTS.md) to reproduce the paper's
# qualitative phenomena (visiting both the jump and continuation regions) and
# to match the Figure 3 time grid t in {0, 0.5, ..., 4.0}.
# ---------------------------------------------------------------------------
X0_TRAJECTORY = 8.0  # initial value of X_t, inferred from the x-axis range of Figure 1
T_HORIZON = 4.0  # simulation horizon, matches the Figure 3 table range
DT_SIMULATION = 1e-3  # time step of the exact GBM solution X_t = X0*exp(-t+W_t)
RANDOM_SEED = 20260615  # fixes the single common trajectory used in Figures 1-3

# ---------------------------------------------------------------------------
# Belief-process approximation Z_t^delta (Sec. 5, Eq. 5.1)
# ---------------------------------------------------------------------------
DELTA_GRID_FIGS_1_2 = (0.1, 0.05, 0.02, 0.01, 0.005, 0.001)  # delta values used
# to approximate the limit process Z_t in Figures 1 and 2; not individually
# specified in the paper beyond "decreasing delta values" (Sec. 8 discussion)
DELTA_GRID_FIG3 = (10.0, 5.0, 2.0, 1.0, 0.5, 0.1, 0.05, 0.02)  # delta values
# tabulated in the Figure 3 table (Sec. 8 / References, Figure 3 caption)
FIG3_TIME_POINTS = tuple(np.round(np.arange(0.0, 4.0 + 1e-9, 0.5), 2))  # time
# points of the Figure 3 table

ODE_RTOL = 1e-8  # relative tolerance for scipy.integrate.solve_ivp
ODE_ATOL = 1e-10  # absolute tolerance for scipy.integrate.solve_ivp

# ---------------------------------------------------------------------------
# Reference values read off Figure 1 / Figure 2 (approximate, from the plotted
# curves; the paper gives an exact table only for Figure 3) used for
# validation in results/RESULTS.md.
# ---------------------------------------------------------------------------
FIG1_REFERENCE = {
    "beta_at_theta_0_5": 0.38,
    "beta_at_z_0_75": 1.00,  # beta(Y(0.75)) with Y(z)=0.5+4.5z i.e. theta=3.875
    "alpha_at_theta_0_5": 0.00,
    "alpha_at_theta_5": 0.85,
}
FIG2_REFERENCE = {
    "alpha_at_theta_0_5": 0.0,
    "alpha_at_z_0_55": 1.0,  # alpha(Y(0.55)) with Y(z)=0.5+1.5z i.e. theta=1.325
    "alpha_at_z_0_85": 2.0,  # theta = 1.775
    "alpha_at_z_0_98": 3.0,  # theta = 1.97
    "alpha_at_theta_2": 4.0,
}

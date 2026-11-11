# Nonzero-Sum Stochastic Exit Games with Singular Belief Dynamics

Replication of selected numerical illustrations from:

> Cao, Y. and Palczewski, J., "Nonzero-sum stochastic exit games with singular
> belief dynamics".

This repository reproduces the Section 8 numerical illustrations: the
symmetric Nash equilibrium of a duopoly exit game under incomplete
information about the opponent's exit value, the belief process Z_t that a
player maintains about the opponent's type, and the convergence of the
family of approximating processes Z_t^delta used to construct Z_t as a
well-posed limit.

Three experiments are implemented:

* **exp_1** -- Full model with monopoly exit (Figure 1): theta ~ U[0.5, 5].
  Both firms may exit; the belief process can jump at t = 0 and the game can
  end in finite time (market collapse).
* **exp_2** -- Reduced model without monopoly exit (Figure 2): theta ~
  U[0.5, 2]. The monopolist never exits (beta(theta) = 0 for every theta in
  the support), so the belief process decreases at a bounded rate and the
  jump region is empty.
* **exp_3** -- Convergence of Z_t^delta to Z_t (Figure 3): tabulates the
  approximating processes for delta in {10, 5, 2, 1, 0.5, 0.1, 0.05, 0.02} at
  t in {0, 0.5, ..., 4.0} on a single trajectory, and checks monotonicity in
  delta.

No external market data is used: the underlying process X_t is a geometric
Brownian motion simulated in closed form, and the exit-value distribution is
the uniform law specified in Section 8. `data/generate_synthetic.py` holds
this simulation code.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Tests

```bash
python -m pytest
```

## Run the experiments

```bash
python -m exp.exp1
python -m exp.exp2
python -m exp.exp3
```

Each command overwrites its artifacts (plots, `*_metrics.json`) directly in
`results/`. `reproduce.sh` runs the tests and all three experiments in
sequence.

## Repository layout

```
src/    fundamental solutions, resolvents, stopping boundaries, value
        functions, belief-process ODE, plotting (shared by all experiments)
exp/    one script per experiment, calling only the src/ API
data/   data/generate_synthetic.py: GBM trajectory simulation
tests/  pytest unit tests for src/ and data/
results/ generated tables, plots, metrics, and RESULTS.md
```

See `results/RESULTS.md` for the comparison against the paper, deviations,
and unresolved points. See `AGENTS.md` for conventions when extending this
repository.

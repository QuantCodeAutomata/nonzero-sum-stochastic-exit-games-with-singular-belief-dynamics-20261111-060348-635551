# AGENTS.md

Guide for coding agents and contributors working in this repository.

## What this implements

A replication of the Section 8 numerical illustration of Cao & Palczewski,
"Nonzero-sum stochastic exit games with singular belief dynamics": the
symmetric Nash equilibrium of a duopoly exit game under incomplete
information, the optimal stopping boundaries alpha(theta) and beta(theta),
the belief process Z_t, and its approximating family Z_t^delta.

## Layout

* `src/` -- flat modules shared by every experiment: fundamental solutions
  and resolvents (`fundamental_solutions.py`, `resolvents.py`), stopping
  boundaries and value functions (`boundaries.py`, `value_functions.py`),
  the belief-process ODE (`belief_process.py`), plotting (`plotting.py`),
  model assembly (`model.py`), and all parameters (`config.py`).
* `exp/` -- one script per experiment (`exp1.py`, `exp2.py`, `exp3.py`),
  calling only the `src/` API.
* `data/` -- `generate_synthetic.py` simulates the geometric Brownian motion
  trajectories used by every experiment. No market data is used by this
  replication.
* `tests/` -- pytest unit tests for `src/` and `data/`.
* `results/` -- generated tables, plots and metrics, plus `RESULTS.md`.

## Commands

```bash
pip install -r requirements.txt
python -m pytest
python -m exp.exp1
python -m exp.exp2
python -m exp.exp3
```

`reproduce.sh` runs all of the above in sequence.

## Conventions

* All experiment parameters live in `src/config.py` as module-level
  UPPER_CASE variables, each with a source comment (paper section or "not
  specified in the paper"). No parameters are hardcoded elsewhere.
* Data preparation code lives only in `data/`; `src/` and `exp/` call it,
  never reimplement it.
* Run experiments as `python -m exp.<name>` and tests as `python -m pytest`
  from the repository root; no `sys.path` edits.
* Rerunning an experiment command overwrites its files in `results/`; the
  numbers and verdicts in `results/RESULTS.md` are checked against the new
  files and updated by hand after each rerun.
* No market or API keys are used by this replication.

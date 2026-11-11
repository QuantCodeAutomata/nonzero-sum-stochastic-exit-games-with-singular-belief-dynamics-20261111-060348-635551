#!/usr/bin/env bash
# Runs the test suite and every experiment of the replication, in order.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

python -m pytest -q
python -m exp.exp1
python -m exp.exp2
python -m exp.exp3

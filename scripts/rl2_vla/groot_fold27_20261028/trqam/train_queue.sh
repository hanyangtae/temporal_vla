#!/usr/bin/env bash
# Harness-guarded TRQAM fold scheduler (central GR00T harness: ledger, per-machine caps, srv coordinator, release on exit).
# env: GPUS=<g>[,<g>...]  HARNESS_MACHINE=kanu|srv48|srv50  HARNESS_SESSION=<owner>  [HARNESS_COORDINATOR_SSH for srv]
#      FOLD27_PY=<python with jax/flax/torch/omegaconf> (default .venvs/rl2_iid/bin/python)
# Bounded launch (releases the GPU when the bundle ends):  ... train_queue.sh --sources S --folds <fold_id> [--folds ...]
# usage: GPUS=5 HARNESS_MACHINE=kanu HARNESS_SESSION=fold27-trqam bash train_queue.sh --sources <sources.json> [--folds ...]
set -euo pipefail
cd "$(dirname "$0")/../../../.."
source scripts/utils/groot_launch_guard.sh
SERVES_PER_GPU=${SERVES_PER_GPU:-1} groot_launch_guard eval "$0" "$@"
PY=${FOLD27_PY:-.venvs/rl2_iid/bin/python}
exec "$PY" scripts/rl2_vla/groot_fold27_20261028/trqam/train_queue.py --python "$PY" "$@"

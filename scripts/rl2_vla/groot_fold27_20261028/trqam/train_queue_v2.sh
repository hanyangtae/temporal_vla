#!/usr/bin/env bash
# Harness-guarded TRQAM fold scheduler v2 (full-state resume trainer; new out root). Same guard as train_queue.sh.
# env: GPUS=<g>[,<g>...] HARNESS_MACHINE=kanu|srv48|srv50 HARNESS_SESSION=<owner> [FOLD27_PY=<jax python>]
# usage: GPUS=5,6,7 HARNESS_MACHINE=kanu HARNESS_SESSION=fold27-trqam-v2 bash train_queue_v2.sh --sources S --folds F [--reuse-ae-from <v1 root>]
set -euo pipefail
cd "$(dirname "$0")/../../../.."
source scripts/utils/groot_launch_guard.sh
SERVES_PER_GPU=${SERVES_PER_GPU:-1} groot_launch_guard eval "$0" "$@"
PY=${FOLD27_PY:-.venvs/rl2_iid/bin/python}
exec "$PY" scripts/rl2_vla/groot_fold27_20261028/trqam/train_queue_v2.py --python "$PY" "$@"

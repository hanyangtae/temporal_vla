#!/usr/bin/env bash
# Durable chain for one fold bundle (3 arms): CPU-only wait for inputs, THEN the harness-guarded training queue.
# 1) wait_ready.py: no GPU held; waits (bounded) until the fold's latents are COMPLETE(200) + every --require file exists
# 2) wait_launch.py -> train_queue.sh: reserves GPUS via the groot harness, waits only on "[harness] BLOCKED"
# env: GPUS=<g> (default 5) HARNESS_MACHINE=kanu HARNESS_SESSION=fold27-trqam  WAIT_H (default 24)
# usage: first_fold_chain.sh <fold_id> [--require <gate file> ...]
set -euo pipefail
cd "$(dirname "$0")/../../../.."
FOLD=${1:?fold_id}; shift
S=scripts/rl2_vla/groot_fold27_20261028/trqam
SRC=outputs/analysis/groot_fold27_20261028/trqam_sources_kanu.json
PY=.venvs/rl2_iid/bin/python
"$PY" "$S/wait_ready.py" --sources "$SRC" --folds "$FOLD" --max-wait-h "${WAIT_H:-24}" "$@"
exec "$PY" "$S/wait_launch.py" --max-wait-h "${WAIT_H:-24}" --state outputs/rl2_vla/fold27_trqam_20261102/queue/wait -- \
  env GPUS="${GPUS:-5}" HARNESS_MACHINE="${HARNESS_MACHINE:-kanu}" HARNESS_SESSION="${HARNESS_SESSION:-fold27-trqam}" \
  bash "$S/train_queue.sh" --sources "$SRC" --folds "$FOLD"

#!/usr/bin/env bash
# fold27 production eval queue for ONE collection host (kanu / srv48 / srv50). Durable loop, resumable:
#   HARNESS_MACHINE=<host> HARNESS_SESSION=<owner> GPUS=<g[,g]> [SERVES_PER_GPU=n] [FOLLOW=1] [POLL_S=600] \
#     bash scripts/rl2_vla/groot_fold27_20261028/eval/runtime/eval_queue.sh [--retry-failed]
# Each round: runnable jobs? (READY folds of this host lacking RUN_COMPLETE) -> wait until the harness ledger admits
# GPUS x SERVES_PER_GPU (never forces / never releases others) -> harness run -> eval_pipeline.py queue (exits when idle,
# GPUs released). FOLLOW=1 keeps polling for newly READY folds until all of this host's folds are complete.
# Exit: 0 nothing left, 4 a (fold, arm) failed (markers under <root>/queue/<host>/failed; rerun with --retry-failed).
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)
SELF="$REPO_ROOT/scripts/rl2_vla/groot_fold27_20261028/eval/runtime/eval_queue.sh"
RT="$REPO_ROOT/scripts/rl2_vla/groot_fold27_20261028/eval/runtime"
cd "$REPO_ROOT"
PY=python3; [ -x .venvs/rl2_iid/bin/python ] && PY=.venvs/rl2_iid/bin/python   # kanu docker helpers need numpy
: "${HARNESS_MACHINE:?}" "${HARNESS_SESSION:?}" "${GPUS:?}"
case "$HARNESS_MACHINE" in kanu) SERVES_PER_GPU=${SERVES_PER_GPU:-2} ;; *) SERVES_PER_GPU=${SERVES_PER_GPU:-6} ;; esac
export SERVES_PER_GPU
if [ "${1:-}" = --inner ]; then          # running under `groot_harness.py run` (HARNESS_RECEIPT set)
  shift
  source scripts/utils/groot_launch_guard.sh
  groot_launch_guard eval "$SELF" --inner "$@"
  exec "$PY" -u "$RT/eval_pipeline.py" queue "$@"
fi
while true; do
  set +e; "$PY" "$RT/eval_pipeline.py" pending --host "$HARNESS_MACHINE" "$@"; rc=$?; set -e
  if [ $rc = 3 ]; then
    [ "${FOLLOW:-0}" = 1 ] || { echo "[eval_queue] nothing runnable on $HARNESS_MACHINE"; exit 0; }
    "$PY" "$RT/eval_pipeline.py" status > /dev/null
    left=$("$PY" -c "import json,sys;s=json.load(open('$REPO_ROOT/outputs/eval/robocasa/groot_n15/fold27_mil_qam_20261102/PIPELINE_STATE.json'))['hosts']['$HARNESS_MACHINE'];print(sum(v!='verified' and v!='prepare_failed' for v in s.values()))")
    [ "$left" = 0 ] && { echo "[eval_queue] all $HARNESS_MACHINE folds verified/failed"; exit 0; }
    sleep "${POLL_S:-600}"; continue
  fi
  [ $rc = 0 ] || exit $rc
  "$PY" "$RT/eval_pipeline.py" wait-resources --host "$HARNESS_MACHINE" --gpus "$GPUS" --serves "$SERVES_PER_GPU" --session "$HARNESS_SESSION"
  set +e
  ( source scripts/utils/groot_launch_guard.sh; groot_launch_guard eval "$SELF" --inner "$@" ); rc=$?
  set -e
  case $rc in
    0) ;;                                      # idle -> next round picks up newly READY folds
    4) echo "[eval_queue] job failure; markers in queue/$HARNESS_MACHINE/failed"; exit 4 ;;
    *) echo "[eval_queue] harness/launch rc=$rc (race or refusal); waiting and retrying"; sleep "${POLL_S:-600}" ;;
  esac
done

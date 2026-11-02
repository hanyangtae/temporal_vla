#!/usr/bin/env bash
# Per-GPU eval follower (new; eval_queue.sh unchanged and must NOT run at the same time on the same host/pool).
# One instance per pool GPU, e.g. GPU=5 / GPU=6 / GPU=7 with POOL=5,6,7. Each round:
#   may-start (read-only: unclaimed READY job exists AND eval under its fair share vs a waiting trainer) ->
#   wait-resources for this ONE GPU (central harness, never forces) -> groot_harness run (receipt for this GPU only) ->
#   eval_gpu_queue.py run (claims shared with the other GPUs; exits when idle or when yielding to the trainer, receipt released).
#   env: HARNESS_MACHINE HARNESS_SESSION GPU [POOL=5,6,7] [SERVES_PER_GPU] [FOLLOW=1] [POLL_S=60]
#        [PIPE_ARGS="--trqam-root ..."] [ENV_CONTAINER_PREFIX=fold27-iso-env-gpu (srv)] [EXPECT_IMAGE=sha256:... (srv)]
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)
SELF="$REPO_ROOT/scripts/rl2_vla/groot_fold27_20261028/eval/runtime/eval_gpu_queue.sh"
RT="$REPO_ROOT/scripts/rl2_vla/groot_fold27_20261028/eval/runtime"
cd "$REPO_ROOT"
PY=python3; [ -x .venvs/rl2_iid/bin/python ] && PY=.venvs/rl2_iid/bin/python
PY=${EVAL_GPU_PY:-$PY}                     # tests only
: "${HARNESS_MACHINE:?}" "${HARNESS_SESSION:?}" "${GPU:?one pool GPU}"
POOL=${POOL:-5,6,7}
case ",$POOL," in *",$GPU,"*) ;; *) echo "[eval_gpu_queue] GPU $GPU not in POOL $POOL" >&2; exit 2 ;; esac
case "$HARNESS_MACHINE" in kanu) SERVES_PER_GPU=${SERVES_PER_GPU:-2} ;; *) SERVES_PER_GPU=${SERVES_PER_GPU:-6} ;; esac
export SERVES_PER_GPU GPUS=$GPU
read -r -a PA <<< "${PIPE_ARGS:-}"
RUN_ARGS=(--min-jobs "${MIN_JOBS:-1}")
[ -n "${ENV_CONTAINER_PREFIX:-}" ] && RUN_ARGS+=(--env-container-prefix "$ENV_CONTAINER_PREFIX")
[ -n "${EXPECT_IMAGE:-}" ] && RUN_ARGS+=(--expect-image "$EXPECT_IMAGE")
if [ "${1:-}" = --inner ]; then          # under `groot_harness.py run` (HARNESS_RECEIPT set, GPUS=$GPU)
  source scripts/utils/groot_launch_guard.sh
  groot_launch_guard eval "$SELF" --inner
  exec "$PY" -u "$RT/eval_gpu_queue.py" "${PA[@]}" --pool "$POOL" run "${RUN_ARGS[@]}"
fi
while true; do
  set +e; "$PY" "$RT/eval_gpu_queue.py" "${PA[@]}" --pool "$POOL" may-start --host "$HARNESS_MACHINE" --gpu "$GPU" --serves "$SERVES_PER_GPU"; rc=$?; set -e
  if [ $rc = 3 ]; then
    [ "${FOLLOW:-0}" = 1 ] || { echo "[eval_gpu_queue] GPU$GPU: nothing to take now"; exit 0; }
    sleep "${POLL_S:-60}"; continue
  fi
  [ $rc = 0 ] || exit $rc
  "$PY" "$RT/eval_pipeline.py" "${PA[@]}" wait-resources --host "$HARNESS_MACHINE" --gpus "$GPU" --serves "$SERVES_PER_GPU" --session "$HARNESS_SESSION"
  set +e; "$PY" "$RT/eval_gpu_queue.py" "${PA[@]}" --pool "$POOL" may-start --host "$HARNESS_MACHINE" --gpu "$GPU" --serves "$SERVES_PER_GPU" >/dev/null; rc=$?; set -e
  [ $rc = 0 ] || continue                    # state changed while waiting (trainer got priority / jobs taken): no reservation made
  set +e
  ( source scripts/utils/groot_launch_guard.sh; groot_launch_guard eval "$SELF" --inner ); rc=$?
  set -e
  case $rc in
    0) ;;
    4) echo "[eval_gpu_queue] GPU$GPU job failure; markers in queue/$HARNESS_MACHINE/failed"; exit 4 ;;
    *) echo "[eval_gpu_queue] GPU$GPU harness/launch rc=$rc (race or refusal); retrying after poll"; sleep "${POLL_S:-60}" ;;
  esac
done

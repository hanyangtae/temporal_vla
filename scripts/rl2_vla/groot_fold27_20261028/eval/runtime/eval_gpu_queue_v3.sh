#!/usr/bin/env bash
# v3: identical to eval_gpu_queue_v2.sh (unchanged) except the resource wait uses wait_resources_shared.py, which honours
# an explicit one-time sharing contract HARNESS_SHARING_AUTH (scripts/utils/gpu_sharing_authorization.py). Without that
# variable behaviour equals v2. The harness (groot_harness.py run) re-verifies the same contract at admission, right after
# admission and before release; the variable is inherited by the harness and the consumer through the environment.
#   env: HARNESS_MACHINE HARNESS_SESSION GPU [POOL] [SERVES_PER_GPU] [FOLLOW=1] [POLL_S=60] [PIPE_ARGS] [ENV_CONTAINER_PREFIX]
#        [EXPECT_IMAGE] [HARNESS_SHARING_AUTH=<contract json>]
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)
SELF="$REPO_ROOT/scripts/rl2_vla/groot_fold27_20261028/eval/runtime/eval_gpu_queue_v3.sh"
RT="$REPO_ROOT/scripts/rl2_vla/groot_fold27_20261028/eval/runtime"
cd "$REPO_ROOT"
PY=python3; [ -x .venvs/rl2_iid/bin/python ] && PY=.venvs/rl2_iid/bin/python
PY=${EVAL_GPU_PY:-$PY}                     # tests only
: "${HARNESS_MACHINE:?}" "${HARNESS_SESSION:?}" "${GPU:?one pool GPU}"
POOL=${POOL:-5,6,7}
case ",$POOL," in *",$GPU,"*) ;; *) echo "[eval_gpu_queue] GPU $GPU not in POOL $POOL" >&2; exit 2 ;; esac
case "$HARNESS_MACHINE" in kanu) SERVES_PER_GPU=${SERVES_PER_GPU:-2} ;; *) SERVES_PER_GPU=${SERVES_PER_GPU:-6} ;; esac
export SERVES_PER_GPU GPUS=$GPU
[ -z "${HARNESS_SHARING_AUTH:-}" ] || export HARNESS_SHARING_AUTH
read -r -a PA <<< "${PIPE_ARGS:-}"
RUN_ARGS=(--min-jobs "${MIN_JOBS:-1}")
[ -n "${ENV_CONTAINER_PREFIX:-}" ] && RUN_ARGS+=(--env-container-prefix "$ENV_CONTAINER_PREFIX")
[ -n "${EXPECT_IMAGE:-}" ] && RUN_ARGS+=(--expect-image "$EXPECT_IMAGE")
if [ "${1:-}" = --inner ]; then          # under `groot_harness.py run` (HARNESS_RECEIPT set, GPUS=$GPU)
  source scripts/utils/groot_launch_guard.sh
  groot_launch_guard eval "$SELF" --inner
  exec "$PY" -u "$RT/eval_gpu_queue_v2.py" "${PA[@]}" --pool "$POOL" run "${RUN_ARGS[@]}"
fi
while true; do
  set +e; "$PY" "$RT/eval_gpu_queue_v2.py" "${PA[@]}" --pool "$POOL" may-start --host "$HARNESS_MACHINE" --gpu "$GPU" --serves "$SERVES_PER_GPU"; rc=$?; set -e
  if [ $rc = 3 ]; then
    [ "${FOLLOW:-0}" = 1 ] || { echo "[eval_gpu_queue] GPU$GPU: nothing to take now"; exit 0; }
    sleep "${POLL_S:-60}"; continue
  fi
  [ $rc = 0 ] || exit $rc
  "$PY" "$RT/wait_resources_shared.py" --host "$HARNESS_MACHINE" --gpus "$GPU" --serves "$SERVES_PER_GPU" --session "$HARNESS_SESSION"
  set +e; "$PY" "$RT/eval_gpu_queue_v2.py" "${PA[@]}" --pool "$POOL" may-start --host "$HARNESS_MACHINE" --gpu "$GPU" --serves "$SERVES_PER_GPU" >/dev/null; rc=$?; set -e
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

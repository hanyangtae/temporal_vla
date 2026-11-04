#!/usr/bin/env bash
# v4 = eval_gpu_queue_v3.sh (unchanged, frozen) + guardian_v4 handoff. Two differences only:
#  (a) outer loop: no reservation unless `handoff_gate.py ready` (live, fresh, non-HOLD guardian_v4 for this GPU).
#      Without FOLLOW=1 a not-ready guardian ends the loop (rc 75); with FOLLOW=1 it waits POLL_S.
#  (b) --inner (under `groot_harness.py run`): after the harness verify and BEFORE exec of the consumer, i.e. before
#      any GPU work, `handoff_gate.py request` must be acked by that guardian for this receipt/pid/start; else exit 75
#      (harness then releases the receipt; no consumer ran). exec keeps the pid, so the adopted pid IS the consumer.
# rc 241 (= harness child killed by SIGTERM, sys.exit(-15)) and other errors still retry after POLL_S as in v3, but
# every retry passes (a) and (b) again, so no consumer can start unguarded.
#  (c) --inner also starts consumer_watchdog.py (detached, outside the consumer tree) and names it in the request:
#      if the adopting guardian process is lost, the watchdog stops only this consumer with the v1 procedure.
#   env: v3 env + GUARDIAN_STATE_DIR=<guardian_v4 --state-dir> GUARDIAN_CONFIG=<guardian_v4 config> (required)
#        [HANDOFF_TIMEOUT_S=120]
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)
SELF="$REPO_ROOT/scripts/rl2_vla/groot_fold27_20261028/eval/guardian_handoff_20261104/eval_gpu_queue_v4.sh"
HD="$REPO_ROOT/scripts/rl2_vla/groot_fold27_20261028/eval/guardian_handoff_20261104"
RT="$REPO_ROOT/scripts/rl2_vla/groot_fold27_20261028/eval/runtime"
cd "$REPO_ROOT"
PY=python3; [ -x .venvs/rl2_iid/bin/python ] && PY=.venvs/rl2_iid/bin/python
PY=${EVAL_GPU_PY:-$PY}                     # tests only
: "${HARNESS_MACHINE:?}" "${HARNESS_SESSION:?}" "${GPU:?one pool GPU}" "${GUARDIAN_STATE_DIR:?guardian_v4 state dir required (fail closed)}"
POOL=${POOL:-5,6,7}
case ",$POOL," in *",$GPU,"*) ;; *) echo "[eval_gpu_queue] GPU $GPU not in POOL $POOL" >&2; exit 2 ;; esac
case "$HARNESS_MACHINE" in kanu) SERVES_PER_GPU=${SERVES_PER_GPU:-2} ;; *) SERVES_PER_GPU=${SERVES_PER_GPU:-6} ;; esac
export SERVES_PER_GPU GPUS=$GPU GUARDIAN_STATE_DIR
[ -z "${HARNESS_SHARING_AUTH:-}" ] || export HARNESS_SHARING_AUTH
read -r -a PA <<< "${PIPE_ARGS:-}"
RUN_ARGS=(--min-jobs "${MIN_JOBS:-1}")
[ -n "${ENV_CONTAINER_PREFIX:-}" ] && RUN_ARGS+=(--env-container-prefix "$ENV_CONTAINER_PREFIX")
[ -n "${EXPECT_IMAGE:-}" ] && RUN_ARGS+=(--expect-image "$EXPECT_IMAGE")
GATE=(--state-dir "$GUARDIAN_STATE_DIR" --host "$HARNESS_MACHINE" --gpu "$GPU" --session "$HARNESS_SESSION")
if [ "${1:-}" = --inner ]; then          # under `groot_harness.py run` (HARNESS_RECEIPT set, GPUS=$GPU)
  source scripts/utils/groot_launch_guard.sh
  groot_launch_guard eval "$SELF" --inner
  : "${GUARDIAN_CONFIG:?guardian_v4 config required (watchdog)}"
  mkdir -p "$GUARDIAN_STATE_DIR/handoff"
  NONCE=$(tr -d '-' < /proc/sys/kernel/random/uuid)
  WDP="$GUARDIAN_STATE_DIR/handoff/watchdog.$$.$NONCE.pid"
  # watchdog detached OUTSIDE the consumer tree (subshell exits -> re-parented), started before the request
  ( setsid "$PY" "$HD/consumer_watchdog.py" --config "$GUARDIAN_CONFIG" --state-dir "$GUARDIAN_STATE_DIR" \
      --consumer-pid $$ --receipt "${HARNESS_RECEIPT:?}" --nonce "$NONCE" \
      </dev/null >>"$GUARDIAN_STATE_DIR/handoff/watchdog.$$.$NONCE.log" 2>&1 & echo $! > "$WDP" )
  "$PY" "$HD/handoff_gate.py" request "${GATE[@]}" --receipt "$HARNESS_RECEIPT" --pid $$ --nonce "$NONCE" \
      --watchdog-pid "$(cat "$WDP")" --timeout "${HANDOFF_TIMEOUT_S:-120}" || exit 75
  exec "$PY" -u "$RT/eval_gpu_queue_v2.py" "${PA[@]}" --pool "$POOL" run "${RUN_ARGS[@]}"
fi
guardian_ready() {
  "$PY" "$HD/handoff_gate.py" ready "${GATE[@]}" >/dev/null && return 0
  [ "${FOLLOW:-0}" = 1 ] || { echo "[eval_gpu_queue] GPU$GPU: guardian_v4 not ready; no launch" >&2; exit 75; }
  return 1
}
while true; do
  guardian_ready || { sleep "${POLL_S:-60}"; continue; }
  set +e; "$PY" "$RT/eval_gpu_queue_v2.py" "${PA[@]}" --pool "$POOL" may-start --host "$HARNESS_MACHINE" --gpu "$GPU" --serves "$SERVES_PER_GPU"; rc=$?; set -e
  if [ $rc = 3 ]; then
    [ "${FOLLOW:-0}" = 1 ] || { echo "[eval_gpu_queue] GPU$GPU: nothing to take now"; exit 0; }
    sleep "${POLL_S:-60}"; continue
  fi
  [ $rc = 0 ] || exit $rc
  "$PY" "$RT/wait_resources_shared.py" --host "$HARNESS_MACHINE" --gpus "$GPU" --serves "$SERVES_PER_GPU" --session "$HARNESS_SESSION"
  set +e; "$PY" "$RT/eval_gpu_queue_v2.py" "${PA[@]}" --pool "$POOL" may-start --host "$HARNESS_MACHINE" --gpu "$GPU" --serves "$SERVES_PER_GPU" >/dev/null; rc=$?; set -e
  [ $rc = 0 ] || continue                    # state changed while waiting (trainer got priority / jobs taken): no reservation made
  guardian_ready || { sleep "${POLL_S:-60}"; continue; }
  set +e
  ( source scripts/utils/groot_launch_guard.sh; groot_launch_guard eval "$SELF" --inner ); rc=$?
  set -e
  case $rc in
    0) ;;
    4) echo "[eval_gpu_queue] GPU$GPU job failure; markers in queue/$HARNESS_MACHINE/failed"; exit 4 ;;
    *) echo "[eval_gpu_queue] GPU$GPU harness/launch rc=$rc (race, refusal, guardian stop=241 or handoff=75); retrying after poll"; sleep "${POLL_S:-60}" ;;
  esac
done

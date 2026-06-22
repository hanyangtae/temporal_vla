#!/usr/bin/env bash
# Run ONE steered eval cell: start steered lerobot serve -> wait ready ->
# collect 20 steered rollouts (resumable) -> kill that serve.
# Mitigation: small serve window, ports 8410-8414, --skip-existing.
set -euo pipefail

CELL_ID="$1"
CONCEPTOR_NPZ="$2"
MANIFEST="$3"
GPU="$4"
PORT="$5"

OUT_BASE="outputs/eval/robocasa/groot_n15/coast4_steer_eval/${CELL_ID}/steered"
RAW_DIR="${OUT_BASE}/raw_rollouts"
EP_META_DIR="${OUT_BASE}/ep_meta"
LOG="/tmp/coast4_steer_${CELL_ID}.log"

echo "[cell ${CELL_ID}] === starting steered serve: gpu=${GPU} port=${PORT} ==="

# Start steered serve detached, survives SIGTERM to this shell (setsid).
docker exec -d -e CUDA_VISIBLE_DEVICES="${GPU}" lerobot bash -lc \
  "cd /temporal_vla && setsid nohup python scripts/serve/lerobot.py \
     --profile configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml \
     --host '*' --port ${PORT} --device cuda --collect --capture-vl \
     --steering-npz ${CONCEPTOR_NPZ} --steering-pathway dit \
     --steering-alpha 0.1 --steering-beta 0.1 --steering-key C_steer \
     > ${LOG} 2>&1 < /dev/null &"

# Wait for /health ready + steering registration in log.
echo "[cell ${CELL_ID}] waiting for serve /health ready ..."
ready=0
for i in $(seq 1 120); do
  status=$(docker exec lerobot bash -lc \
    "curl -s -m 3 http://127.0.0.1:${PORT}/health 2>/dev/null | python -c 'import sys,json; print(json.load(sys.stdin).get(\"status\",\"\"))' 2>/dev/null" || true)
  if [ "${status}" = "ok" ] || [ "${status}" = "ready" ] || [ "${status}" = "healthy" ]; then
    ready=1; break
  fi
  # Detect early crash
  if ! docker exec lerobot bash -lc "pgrep -f 'lerobot.py.*--port ${PORT}' >/dev/null"; then
    if [ "${i}" -gt 3 ]; then
      echo "[cell ${CELL_ID}] SERVE PROCESS GONE during startup (i=${i}). tail log:"
      docker exec lerobot bash -lc "tail -30 ${LOG}" || true
      exit 11
    fi
  fi
  sleep 5
done

if [ "${ready}" -ne 1 ]; then
  echo "[cell ${CELL_ID}] serve NOT ready after timeout. tail log:"
  docker exec lerobot bash -lc "tail -40 ${LOG}" || true
  exit 12
fi

# Confirm load did not fail with a steering error. The module logger has no
# handler so the "Conceptor steering registered" line is not emitted; instead we
# verify load_model completed without a steering/traceback error (a bad npz/key
# raises inside load_model and prevents /health from ever returning "ok").
if docker exec lerobot bash -lc "grep -qE 'load_model FAILED|Conceptor steering|steering_hooks|build_steering_matrix|KeyError' ${LOG} && grep -qE 'Traceback|FAILED' ${LOG}"; then
  echo "[cell ${CELL_ID}] ERROR: steering registration appears to have failed:"
  docker exec lerobot bash -lc "tail -50 ${LOG}" || true
  docker exec lerobot pkill -f "lerobot.py.*--port ${PORT}" || true
  exit 13
fi
echo "[cell ${CELL_ID}] serve healthy, no steering load error -> steering registered (alpha0.1 C_steer, dit, beta0.1)."

echo "[cell ${CELL_ID}] === collecting 20 steered rollouts (resumable) ==="
set +e
docker exec -e MUJOCO_GL=egl robocasa bash -lc \
  "cd /temporal_vla && python scripts/safe/groot_n15/robocasa/collect/collect_instruction_fixed_http_features.py \
     --selected-seeds ${MANIFEST} \
     --cell-id ${CELL_ID} \
     --output-dir ${RAW_DIR} \
     --ep-meta-dir ${EP_META_DIR} \
     --vla-server http://127.0.0.1:${PORT} \
     --max-episodes-per-cell 20 \
     --n-action-steps 16 \
     --max-episode-steps 700 \
     --no-overlay --skip-existing --wait-ready"
collect_rc=$?
set -e
echo "[cell ${CELL_ID}] collect rc=${collect_rc}"

echo "[cell ${CELL_ID}] === killing steered serve on port ${PORT} ==="
docker exec lerobot pkill -f "lerobot.py.*--port ${PORT}" || true
sleep 2

# Tally results.
npkl=$(find "${RAW_DIR}" -name "task*--ep*--succ*.pkl" 2>/dev/null | wc -l | tr -d ' ')
nsucc=$(find "${RAW_DIR}" -name "task*--ep*--succ1.pkl" 2>/dev/null | wc -l | tr -d ' ')
echo "[cell ${CELL_ID}] RESULT pkls=${npkl} succ=${nsucc}"
exit "${collect_rc}"

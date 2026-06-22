#!/bin/bash
# COAST-faithful GR00T N1.5 RoboCasa natural feature collection (lerobot HTTP path).
#
# COAST A.7.2/A.8.4: 7 atomic-seen tasks, natural RoboCasa resets (no ep_meta fixing),
# DiT residual at transformer_blocks[i] i in {0..15}, action-token mean, denoise K=4 kept.
# Capture is [L=16, K=4, D=1536] per env-step (Phase 0 safe_hooks change) + VL [2048].
#
# Per-GPU lerobot HTTP serve (--collect) in the `lerobot` container; robocasa worker
# (http_feature_collect.py natural mode = NO --cell-id/--canonical-instruction) in the
# `robocasa` container. Instruction labels live in each pkl's task_description; the
# classify step (classify_instructions.py) groups them post-hoc.
#
# Run on host. Output: outputs/eval/robocasa/groot_n15/<RUN_ID>/raw_rollouts/<task>/*.pkl
set -u

REPO_HOST="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
REPO=/temporal_vla
cd "${REPO_HOST}"

GPUS=(${GPUS:-0 1 2 3})
EPISODES_PER_TASK=${EPISODES_PER_TASK:-30}
SEED_START=${SEED_START:-100000}
BASE_PORT=${BASE_PORT:-8400}
PROFILE=${PROFILE:-${REPO}/configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml}
RUN_ID=${RUN_ID:-coast_faithful_7task_30ep}
OUT_ROOT=${REPO}/outputs/eval/robocasa/groot_n15/${RUN_ID}/raw_rollouts
OUT_ROOT_HOST=outputs/eval/robocasa/groot_n15/${RUN_ID}/raw_rollouts
EP_META_ROOT=${REPO}/outputs/eval/robocasa/groot_n15/${RUN_ID}/ep_meta
EP_META_ROOT_HOST=outputs/eval/robocasa/groot_n15/${RUN_ID}/ep_meta
LOG=/tmp/coast_faithful_collect
# COAST: all 16 DiT layers (transformer_blocks 0..15); quota selects L5/L11.
CAPTURE_LAYERS=${CAPTURE_LAYERS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
LEROBOT_CONTAINER=${LEROBOT_CONTAINER:-lerobot}
ROBOCASA_CONTAINER=${ROBOCASA_CONTAINER:-robocasa}
N_ACTION_STEPS=${N_ACTION_STEPS:-16}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-}   # empty = http_feature_collect default (native horizon)
INFERENCE_SEED=${INFERENCE_SEED:-}         # empty = natural stochastic policy (no fixed denoise seed)
NPROC=${#GPUS[@]}

# COAST A.8.4 7 atomic-seen tasks.
TASKS=(CloseFridge CoffeeSetupMug OpenDrawer OpenStandMixerHead PickPlaceCounterToCabinet
       PickPlaceCounterToStove TurnOnElectricKettle)
if [[ -n "${TASKS_OVERRIDE:-}" ]]; then
  # shellcheck disable=SC2206
  TASKS=( ${TASKS_OVERRIDE} )
fi

PYPATH="${REPO}/src/policies/Isaac-GR00T:${REPO}/src/benchmarks/robocasa:${REPO}/src/benchmarks/robosuite:${REPO}"

# cleanup: kill only the serves/workers this script started (port-scoped; protects others).
cleanup() {
  for idx in "${!GPUS[@]}"; do
    local port=$((BASE_PORT + idx))
    docker exec "${LEROBOT_CONTAINER}" bash -lc "pkill -f 'lerobot.py.*--port ${port}'" 2>/dev/null
    docker exec "${ROBOCASA_CONTAINER}" bash -lc "pkill -f 'vla-server http://127.0.0.1:${port}'" 2>/dev/null
  done
}
trap cleanup EXIT INT TERM

mkdir -p "${OUT_ROOT_HOST}" "${LOG}"
# serve/worker write logs inside the containers (container-local /tmp); create there too.
docker exec "${LEROBOT_CONTAINER}" bash -lc "mkdir -p ${LOG}" 2>/dev/null
docker exec "${ROBOCASA_CONTAINER}" bash -lc "mkdir -p ${LOG}" 2>/dev/null
echo "[setup] GPUs=${GPUS[*]} eps/task=${EPISODES_PER_TASK} tasks=${#TASKS[@]} layers='${CAPTURE_LAYERS}'"
echo "[setup] out=${OUT_ROOT_HOST}"

# ── 1. GPU당 lerobot HTTP serve (collect mode, VL + 16 DiT layers) ──
for idx in "${!GPUS[@]}"; do
  gpu=${GPUS[$idx]}; port=$((BASE_PORT + idx))
  docker exec -d -e CUDA_VISIBLE_DEVICES=${gpu} -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${LEROBOT_CONTAINER}" bash -lc \
    "cd ${REPO} && python scripts/serve/lerobot.py --profile ${PROFILE} --host '*' --port ${port} \
       --device cuda --collect --capture-vl --groot-dit-capture-layers ${CAPTURE_LAYERS} \
       > ${LOG}/serve_${port}.log 2>&1"
  echo "[serve] gpu=${gpu} port=${port} launched"
done

# ── 2. /health ready 대기 (host networking → localhost curl) ──
for idx in "${!GPUS[@]}"; do
  port=$((BASE_PORT + idx)); ok=0
  for i in $(seq 1 90); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then ok=1; break; fi
    if docker exec "${LEROBOT_CONTAINER}" bash -lc "grep -qiE 'Traceback|Error|Exception' ${LOG}/serve_${port}.log 2>/dev/null"; then
      echo "[serve] port ${port} ERROR"; docker exec "${LEROBOT_CONTAINER}" bash -lc "tail -25 ${LOG}/serve_${port}.log"; exit 1; fi
    sleep 10
  done
  [ "${ok}" = 1 ] && echo "[serve] port ${port} ready" || { echo "[serve] port ${port} TIMEOUT"; docker exec "${LEROBOT_CONTAINER}" bash -lc "tail -25 ${LOG}/serve_${port}.log"; exit 1; }
done

# ── 3. worker: GPU별 task round-robin, task당 1 호출(--n-episodes N) ──
worker() {
  local server_idx=$1
  local gpu=${GPUS[$server_idx]}; local port=$((BASE_PORT + server_idx))
  for t in "${!TASKS[@]}"; do
    [ $((t % NPROC)) -ne "${server_idx}" ] && continue
    local task="${TASKS[$t]}"; local env="robocasa_panda_omron/${task}_PandaOmron_Env"
    mkdir -p "${EP_META_ROOT_HOST}/${task}"
    local max_arg=""
    [[ -n "${MAX_EPISODE_STEPS}" ]] && max_arg="--max-episode-steps ${MAX_EPISODE_STEPS}"
    local inf_arg=""
    [[ -n "${INFERENCE_SEED}" ]] && inf_arg="--inference-seed ${INFERENCE_SEED}"
    echo "[worker gpu=${gpu}] task${t} ${task} start (n=${EPISODES_PER_TASK})"
    docker exec -e MUJOCO_GL=egl -e PYTHONPATH="${PYPATH}" "${ROBOCASA_CONTAINER}" \
      python ${REPO}/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py \
        --vla-server "http://127.0.0.1:${port}" --task "${task}" --env-name "${env}" \
        --output-dir "${OUT_ROOT}" --task-id ${t} \
        --episode-start-idx 0 --n-episodes ${EPISODES_PER_TASK} \
        --seed ${SEED_START} ${inf_arg} \
        --n-action-steps ${N_ACTION_STEPS} ${max_arg} \
        --ep-meta-dir "${EP_META_ROOT}/${task}" --wait-ready \
        >> ${LOG}/collect_port${port}.log 2>&1
    echo "[worker gpu=${gpu}] task${t} ${task} done"
  done
}
for idx in "${!GPUS[@]}"; do worker "${idx}" & done
wait

echo "[done] -> ${OUT_ROOT_HOST}"
find "${OUT_ROOT_HOST}" -name "*.pkl" | wc -l | xargs echo "pkl count:"
echo "[next] classify: python scripts/safe/groot_n15/robocasa/collect/classify_instructions.py ${OUT_ROOT_HOST} --out-root <classified>"

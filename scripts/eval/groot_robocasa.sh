#!/bin/bash
# GR00T N1.6 + RoboCasa 평가
#
# 실행 순서:
#   1. groot 컨테이너에서 서버 시작:
#      docker exec -it groot bash /temporal_vla/scripts/eval/groot_robocasa.sh server
#
#   2. robocasa 컨테이너에서 클라이언트 실행:
#      docker exec -it robocasa bash /temporal_vla/scripts/eval/groot_robocasa.sh client [ENV_NAME] [N_EPISODES]
#
# 최초 실행 시 (한 번만):
#   docker exec -it groot bash /temporal_vla/scripts/eval/groot_robocasa.sh setup-server
#   docker exec -it robocasa bash /temporal_vla/scripts/eval/groot_robocasa.sh setup-client

set -e
cd /temporal_vla

MODE=${1:-help}
MODEL_PATH=${MODEL_PATH:-"/temporal_vla/checkpoints/nvidia/GR00T-N1.6-3B"}
PORT=${PORT:-5556}
POLICY_CLIENT_HOST=${POLICY_CLIENT_HOST:-166.104.35.98}
EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR:-"/temporal_vla/outputs/eval/robocasa/groot"}
EVAL_RUN_ID=${EVAL_RUN_ID:-"$(date +%Y%m%d_%H%M%S)"}
export PORT POLICY_CLIENT_HOST EVAL_OUTPUT_DIR EVAL_RUN_ID
export HF_MODULES_CACHE=${HF_MODULES_CACHE:-/tmp/hf_modules}
export MUJOCO_GL=${MUJOCO_GL:-egl}

GR00T_SOURCE_DIR="/temporal_vla/src/policies/Isaac-GR00T"
ROBOCASA_SOURCE_DIR="/temporal_vla/src/benchmarks/robocasa"
ROBOSUITE_SOURCE_DIR="/temporal_vla/src/benchmarks/robosuite"
export PYTHONPATH="${GR00T_SOURCE_DIR}:${ROBOCASA_SOURCE_DIR}:${ROBOSUITE_SOURCE_DIR}:${PYTHONPATH:-}"

gr00t_imports_from_source() {
    python - "${GR00T_SOURCE_DIR}" <<'PY'
import sys
from pathlib import Path

try:
    import gr00t
except Exception:
    sys.exit(1)

source_dir = Path(sys.argv[1]).resolve()
module_path = Path(gr00t.__file__).resolve()
sys.exit(0 if source_dir in module_path.parents else 1)
PY
}

ensure_gr00t_editable_install() {
    if gr00t_imports_from_source; then
        echo "Isaac-GR00T package already importable from ${GR00T_SOURCE_DIR}; skipping install."
    else
        echo "Installing Isaac-GR00T package..."
        python -m pip install -e "${GR00T_SOURCE_DIR}" --no-deps
    fi
}

robocasa_kitchen_assets_ready() {
    PYTHONPATH="${ROBOCASA_SOURCE_DIR}:${ROBOSUITE_SOURCE_DIR}:${PYTHONPATH:-}" python - <<'PY'
import os
import sys

try:
    import robocasa
except Exception:
    sys.exit(1)

asset_root = os.path.join(robocasa.__path__[0], "models", "assets")
required_dirs = [
    os.path.join(asset_root, "textures"),
    os.path.join(asset_root, "generative_textures"),
    os.path.join(asset_root, "fixtures"),
    os.path.join(asset_root, "objects", "objaverse"),
    os.path.join(asset_root, "objects", "aigen_objs"),
    os.path.join(asset_root, "objects", "lightwheel"),
]

for path in required_dirs:
    if not os.path.isdir(path) or not os.listdir(path):
        sys.exit(1)

sys.exit(0)
PY
}

ensure_robocasa_kitchen_assets() {
    if robocasa_kitchen_assets_ready; then
        echo "RoboCasa kitchen assets already present; skipping download."
    else
        echo "Downloading RoboCasa kitchen assets..."
        printf 'y\n' | PYTHONPATH="${ROBOCASA_SOURCE_DIR}:${ROBOSUITE_SOURCE_DIR}:${PYTHONPATH:-}" python "${ROBOCASA_SOURCE_DIR}/robocasa/scripts/download_kitchen_assets.py"
    fi
}

case "$MODE" in

  # ── 최초 설치 (groot 컨테이너) ──
  setup-server)
    ensure_gr00t_editable_install
    echo "Done. Run: bash $0 server"
    ;;

  # ── 최초 설치 (robocasa 컨테이너) ──
  setup-client)
    ensure_gr00t_editable_install
    ensure_robocasa_kitchen_assets
    echo "Done. Run: bash $0 client"
    ;;

  # ── 서버 시작 (groot 컨테이너) ──
  server)
    echo "============================================"
    echo "Starting GR00T N1.6 server"
    echo "  Model: ${MODEL_PATH}"
    echo "  Port:  ${PORT}"
    echo "============================================"
    python /temporal_vla/src/policies/Isaac-GR00T/gr00t/eval/run_gr00t_server.py \
        --model-path "${MODEL_PATH}" \
        --embodiment-tag ROBOCASA_PANDA_OMRON \
        --use-sim-policy-wrapper \
        --port "${PORT}"
    ;;

  # ── 클라이언트 평가 (robocasa 컨테이너) ──
  client)
    ENV_NAME=${2:-"robocasa_panda_omron/PnPCounterToCab_PandaOmron_Env"}
    N_EPISODES=${3:-10}
    N_ENVS=${4:-5}
    N_ACTION_STEPS=${5:-8}
    MAX_STEPS=${6:-720}
    TASK_SHORT="${ENV_NAME#*/}"
    TASK_SHORT="${TASK_SHORT%_PandaOmron_Env}"
    if [ -n "${VIDEO_DIR:-}" ]; then
        CLIENT_VIDEO_DIR="${VIDEO_DIR}"
    else
        CLIENT_VIDEO_DIR="${EVAL_OUTPUT_DIR}/${EVAL_RUN_ID}/videos/${TASK_SHORT}"
    fi
    mkdir -p "${CLIENT_VIDEO_DIR}"

    echo "============================================"
    echo "Running RoboCasa evaluation"
    echo "  Env:          ${ENV_NAME}"
    echo "  Server:       ${POLICY_CLIENT_HOST}:${PORT}"
    echo "  Video dir:    ${CLIENT_VIDEO_DIR}"
    echo "  Episodes:     ${N_EPISODES}"
    echo "  Parallel:     ${N_ENVS}"
    echo "  Action steps: ${N_ACTION_STEPS}"
    echo "  Max steps:    ${MAX_STEPS}"
    echo "============================================"

    export MUJOCO_GL=egl

    python /temporal_vla/scripts/eval/groot_robocasa_zmq_eval.py \
        --policy-client-host "${POLICY_CLIENT_HOST}" \
        --policy-client-port "${PORT}" \
        --env-name "${ENV_NAME}" \
        --n-episodes "${N_EPISODES}" \
        --n-envs "${N_ENVS}" \
        --n-action-steps "${N_ACTION_STEPS}" \
        --max-episode-steps "${MAX_STEPS}" \
        --video-dir "${CLIENT_VIDEO_DIR}"
    ;;

  # ── 전체 벤치마크 (실패 많은 task 위주, ZMQ 클라이언트) ──
  client-batch)
    N_EPISODES=${2:-20}
    TASKS=(
        "robocasa_panda_omron/PnPCounterToMicrowave_PandaOmron_Env"
        "robocasa_panda_omron/PnPMicrowaveToCounter_PandaOmron_Env"
        "robocasa_panda_omron/CoffeeSetupMug_PandaOmron_Env"
        "robocasa_panda_omron/TurnOffStove_PandaOmron_Env"
        "robocasa_panda_omron/OpenDoubleDoor_PandaOmron_Env"
        "robocasa_panda_omron/PnPCounterToCab_PandaOmron_Env"
        "robocasa_panda_omron/PnPCabToCounter_PandaOmron_Env"
        "robocasa_panda_omron/PnPCounterToSink_PandaOmron_Env"
        "robocasa_panda_omron/PnPSinkToCounter_PandaOmron_Env"
        "robocasa_panda_omron/PnPCounterToStove_PandaOmron_Env"
        "robocasa_panda_omron/PnPStoveToCounter_PandaOmron_Env"
    )

    for TASK in "${TASKS[@]}"; do
        TASK_SHORT=$(echo "$TASK" | sed 's|robocasa_panda_omron/||; s|_PandaOmron_Env||')
        echo ""
        echo "============================================"
        echo "Task: ${TASK_SHORT} (${N_EPISODES} episodes)"
        echo "============================================"
        bash "$0" client "$TASK" "$N_EPISODES" 5 8 720
    done
    ;;

  # ── target atomic-seen 15-task scope 평가 (ZMQ 클라이언트) ──
  client-target15)
    N_EPISODES=${2:-50}
    N_ENVS=${3:-4}
    N_ACTION_STEPS=${4:-8}
    MAX_STEPS=${5:-720}
    TASKS=(
        "robocasa_panda_omron/CloseFridge_PandaOmron_Env"
        "robocasa_panda_omron/CloseToasterOvenDoor_PandaOmron_Env"
        "robocasa_panda_omron/CoffeeSetupMug_PandaOmron_Env"
        "robocasa_panda_omron/OpenCabinet_PandaOmron_Env"
        "robocasa_panda_omron/OpenDrawer_PandaOmron_Env"
        "robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env"
        "robocasa_panda_omron/PickPlaceCounterToStove_PandaOmron_Env"
        "robocasa_panda_omron/PickPlaceDrawerToCounter_PandaOmron_Env"
        "robocasa_panda_omron/PickPlaceSinkToCounter_PandaOmron_Env"
        "robocasa_panda_omron/PickPlaceToasterToCounter_PandaOmron_Env"
        "robocasa_panda_omron/SlideDishwasherRack_PandaOmron_Env"
        "robocasa_panda_omron/TurnOffStove_PandaOmron_Env"
        "robocasa_panda_omron/TurnOnElectricKettle_PandaOmron_Env"
        "robocasa_panda_omron/TurnOnMicrowave_PandaOmron_Env"
        "robocasa_panda_omron/TurnOnSinkFaucet_PandaOmron_Env"
    )

    echo "============================================"
    echo "Running RoboCasa target atomic-seen 15-task evaluation"
    echo "  Server:       ${POLICY_CLIENT_HOST}:${PORT}"
    echo "  Output:       ${EVAL_OUTPUT_DIR}/${EVAL_RUN_ID}"
    echo "  Episodes:     ${N_EPISODES}"
    echo "  Parallel:     ${N_ENVS}"
    echo "  Action steps: ${N_ACTION_STEPS}"
    echo "  Max steps:    ${MAX_STEPS}"
    echo "============================================"

    for TASK in "${TASKS[@]}"; do
        TASK_SHORT=$(echo "$TASK" | sed 's|robocasa_panda_omron/||; s|_PandaOmron_Env||')
        echo ""
        echo "============================================"
        echo "Task: ${TASK_SHORT} (${N_EPISODES} episodes)"
        echo "============================================"
        bash "$0" client "$TASK" "$N_EPISODES" "$N_ENVS" "$N_ACTION_STEPS" "$MAX_STEPS"
    done
    ;;

  # ── legacy pretrain 10-task scope 평가 (ZMQ 클라이언트) ──
  client-train10)
    N_EPISODES=${2:-50}
    N_ENVS=${3:-4}
    N_ACTION_STEPS=${4:-8}
    MAX_STEPS=${5:-720}
    TASKS=(
        "robocasa_panda_omron/OpenDrawer_PandaOmron_Env"
        "robocasa_panda_omron/CloseDrawer_PandaOmron_Env"
        "robocasa_panda_omron/OpenCabinet_PandaOmron_Env"
        "robocasa_panda_omron/CloseCabinet_PandaOmron_Env"
        "robocasa_panda_omron/OpenFridge_PandaOmron_Env"
        "robocasa_panda_omron/CloseFridge_PandaOmron_Env"
        "robocasa_panda_omron/OpenMicrowave_PandaOmron_Env"
        "robocasa_panda_omron/CloseMicrowave_PandaOmron_Env"
        "robocasa_panda_omron/PickPlaceCounterToStove_PandaOmron_Env"
        "robocasa_panda_omron/PickPlaceCounterToSink_PandaOmron_Env"
    )

    echo "============================================"
    echo "Running RoboCasa legacy pretrain 10-task evaluation"
    echo "  Server:       ${POLICY_CLIENT_HOST}:${PORT}"
    echo "  Output:       ${EVAL_OUTPUT_DIR}/${EVAL_RUN_ID}"
    echo "  Episodes:     ${N_EPISODES}"
    echo "  Parallel:     ${N_ENVS}"
    echo "  Action steps: ${N_ACTION_STEPS}"
    echo "  Max steps:    ${MAX_STEPS}"
    echo "============================================"

    for TASK in "${TASKS[@]}"; do
        TASK_SHORT=$(echo "$TASK" | sed 's|robocasa_panda_omron/||; s|_PandaOmron_Env||')
        echo ""
        echo "============================================"
        echo "Task: ${TASK_SHORT} (${N_EPISODES} episodes)"
        echo "============================================"
        bash "$0" client "$TASK" "$N_EPISODES" "$N_ENVS" "$N_ACTION_STEPS" "$MAX_STEPS"
    done
    ;;

  # ── 로컬 단일 태스크 평가 (groot 컨테이너 단독, 서버 불필요) ──
  local)
    ENV_NAME=${2:-"robocasa_panda_omron/PnPCounterToCab_PandaOmron_Env"}
    N_EPISODES=${3:-10}
    N_ENVS=${4:-5}
    N_ACTION_STEPS=${5:-8}
    MAX_STEPS=${6:-720}

    echo "============================================"
    echo "Running local evaluation (no ZMQ server)"
    echo "  Model:        ${MODEL_PATH}"
    echo "  Env:          ${ENV_NAME}"
    echo "  Episodes:     ${N_EPISODES}"
    echo "  Parallel:     ${N_ENVS}"
    echo "  Action steps: ${N_ACTION_STEPS}"
    echo "  Max steps:    ${MAX_STEPS}"
    echo "============================================"

    python /temporal_vla/src/policies/Isaac-GR00T/gr00t/eval/rollout_policy.py \
        --model_path "${MODEL_PATH}" \
        --env_name "${ENV_NAME}" \
        --n_episodes "${N_EPISODES}" \
        --n_envs "${N_ENVS}" \
        --n_action_steps "${N_ACTION_STEPS}" \
        --max_episode_steps "${MAX_STEPS}"
    ;;

  # ── 로컬 배치 평가 (groot 컨테이너 단독, 서버 불필요) ──
  # 사용법: $0 local-batch [N_EPISODES] [N_ENVS]
  # 결과/영상: outputs/robocasa_eval/groot/ 에 저장. 완료된 태스크는 skip.
  local-batch)
    N_EPISODES=${2:-10}
    N_ENVS=${3:-5}
    MAX_STEPS=${4:-1080}
    RUN_ID=$(date +%y%m%d%H%M)
    OUT_BASE="/temporal_vla/outputs/robocasa_eval/groot/${RUN_ID}"
    VIDEO_BASE="${OUT_BASE}/videos"
    RESULT_BASE="${OUT_BASE}/results"
    mkdir -p "${VIDEO_BASE}" "${RESULT_BASE}"

    TASKS=(
        "robocasa_panda_omron/PnPCounterToMicrowave_PandaOmron_Env"
        "robocasa_panda_omron/PnPMicrowaveToCounter_PandaOmron_Env"
        "robocasa_panda_omron/CoffeeSetupMug_PandaOmron_Env"
        "robocasa_panda_omron/TurnOffStove_PandaOmron_Env"
        "robocasa_panda_omron/OpenDoubleDoor_PandaOmron_Env"
        "robocasa_panda_omron/OpenDrawer_PandaOmron_Env"
        "robocasa_panda_omron/PnPCounterToCab_PandaOmron_Env"
        "robocasa_panda_omron/PnPCabToCounter_PandaOmron_Env"
        "robocasa_panda_omron/PnPCounterToSink_PandaOmron_Env"
        "robocasa_panda_omron/PnPSinkToCounter_PandaOmron_Env"
        "robocasa_panda_omron/PnPCounterToStove_PandaOmron_Env"
        "robocasa_panda_omron/PnPStoveToCounter_PandaOmron_Env"
    )

    SUMMARY_FILE="${RESULT_BASE}/batch_summary_$(date +%Y%m%d_%H%M%S).log"
    {
      echo "GR00T Local Batch Evaluation - $(date)"
      echo "Model: ${MODEL_PATH}"
      echo "Run ID: ${RUN_ID}"
      echo "Episodes: ${N_EPISODES}, Parallel envs: ${N_ENVS}, Max steps: ${MAX_STEPS}"
      echo "========================================"
    } | tee "${SUMMARY_FILE}"

    for TASK in "${TASKS[@]}"; do
        TASK_SHORT=$(echo "$TASK" | sed 's|robocasa_panda_omron/||; s|_PandaOmron_Env||')
        TASK_VIDEO_DIR="${VIDEO_BASE}/${TASK_SHORT}"

        # 이미 영상이 충분히 있으면 skip
        EXISTING=$(ls -1 "${TASK_VIDEO_DIR}"/*.mp4 2>/dev/null | wc -l)
        if [ "${EXISTING}" -ge "${N_EPISODES}" ]; then
            echo "[SKIP] ${TASK_SHORT}: already has ${EXISTING} videos" | tee -a "${SUMMARY_FILE}"
            continue
        fi

        echo "" | tee -a "${SUMMARY_FILE}"
        echo "============================================" | tee -a "${SUMMARY_FILE}"
        echo "Task: ${TASK_SHORT} (${N_EPISODES} ep, ${N_ENVS} envs)" | tee -a "${SUMMARY_FILE}"
        echo "============================================" | tee -a "${SUMMARY_FILE}"

        # 실행 전 /tmp 영상 디렉토리 목록 기록
        BEFORE=$(ls -d /tmp/sim_eval_videos_* 2>/dev/null | sort || true)

        # rollout 실행
        TASK_LOG="${RESULT_BASE}/${TASK_SHORT}.log"
        bash "$0" local "$TASK" "$N_EPISODES" "$N_ENVS" 8 "${MAX_STEPS}" 2>&1 | tee "${TASK_LOG}"

        # 결과 추출
        grep -E "success rate|results:" "${TASK_LOG}" >> "${SUMMARY_FILE}" 2>/dev/null || true

        # 새로 생긴 영상 디렉토리를 찾아서 이동
        AFTER=$(ls -d /tmp/sim_eval_videos_* 2>/dev/null | sort || true)
        NEW_DIR=$(comm -13 <(echo "$BEFORE") <(echo "$AFTER") | head -1)
        if [ -n "${NEW_DIR}" ] && [ -d "${NEW_DIR}" ]; then
            mkdir -p "${TASK_VIDEO_DIR}"
            mv "${NEW_DIR}"/*.mp4 "${TASK_VIDEO_DIR}/" 2>/dev/null || true
            rm -rf "${NEW_DIR}"
            N_MOVED=$(ls -1 "${TASK_VIDEO_DIR}"/*.mp4 2>/dev/null | wc -l)
            echo "  Videos: ${TASK_VIDEO_DIR}/ (${N_MOVED} files)" | tee -a "${SUMMARY_FILE}"
        fi
    done

    echo "" | tee -a "${SUMMARY_FILE}"
    echo "========================================" | tee -a "${SUMMARY_FILE}"
    echo "BATCH COMPLETE - Summary:" | tee -a "${SUMMARY_FILE}"
    grep "success rate" "${SUMMARY_FILE}" 2>/dev/null | tee -a /dev/null || true
    echo "Results: ${RESULT_BASE}/" | tee -a "${SUMMARY_FILE}"
    echo "Videos:  ${VIDEO_BASE}/" | tee -a "${SUMMARY_FILE}"
    ;;

  *)
    echo "Usage:"
    echo "  $0 setup-server   — groot 컨테이너 초기 설치"
    echo "  $0 setup-client   — robocasa 컨테이너 초기 설치"
    echo "  $0 server         — groot 서버 시작 (ZMQ)"
    echo "  $0 client [env] [n_ep] [n_envs] [n_act] [max_steps]  — ZMQ 클라이언트"
    echo "  $0 client-batch [n_ep]  — ZMQ 일괄 평가"
    echo "  $0 client-target15 [n_ep] [n_envs] [n_act] [max_steps] — target atomic-seen 15-task ZMQ 평가"
    echo "  $0 client-train10 [n_ep] [n_envs] [n_act] [max_steps] — legacy pretrain 10-task ZMQ 평가"
    echo "  $0 local [env] [n_ep] [n_envs] [n_act] [max_steps]   — 로컬 단독 실행"
    echo "  $0 local-batch [n_ep] [n_envs] — 로컬 일괄 평가 (완료 skip, 영상 자동 이동)"
    ;;

esac

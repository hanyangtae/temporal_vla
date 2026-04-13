#!/bin/bash
# GR00T trajectory 수집 + Loop 패턴 분석 배치 스크립트
#
# groot 컨테이너에서 실행:
#   docker exec -e HF_MODULES_CACHE=/tmp/hf_modules -e MUJOCO_GL=egl groot \
#       bash /temporal_vla/scripts/collect_and_analyze_groot.sh [N_EPISODES]

set -e
cd /temporal_vla

N_EPISODES=${1:-20}
MODEL_PATH=${MODEL_PATH:-"/temporal_vla/checkpoints/nvidia/GR00T-N1.6-3B"}
SAVE_DIR="/temporal_vla/outputs/trajectories"
ANALYSIS_DIR="/temporal_vla/outputs/analysis"

export MUJOCO_GL=${MUJOCO_GL:-egl}
export HF_MODULES_CACHE=${HF_MODULES_CACHE:-/tmp/hf_modules}

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

echo "============================================"
echo "GR00T Trajectory Collection + Analysis"
echo "  Model:    ${MODEL_PATH}"
echo "  Episodes: ${N_EPISODES} per task"
echo "  Tasks:    ${#TASKS[@]}"
echo "  Save:     ${SAVE_DIR}"
echo "============================================"

# 1. Trajectory 수집
for TASK in "${TASKS[@]}"; do
    TASK_SHORT=$(echo "$TASK" | sed 's|robocasa_panda_omron/||; s|_PandaOmron_Env||')
    echo ""
    echo "============================================"
    echo "[COLLECT] ${TASK_SHORT} (${N_EPISODES} episodes)"
    echo "============================================"
    python /temporal_vla/scripts/collect_groot_trajectories.py \
        --model_path "${MODEL_PATH}" \
        --env_name "${TASK}" \
        --n_episodes "${N_EPISODES}" \
        --save_dir "${SAVE_DIR}"
done

# 2. Loop 패턴 분석
echo ""
echo "============================================"
echo "[ANALYZE] Running loop pattern analysis"
echo "============================================"
python /temporal_vla/scripts/analyze_loop_patterns.py \
    --traj_dir "${SAVE_DIR}" \
    --output_dir "${ANALYSIS_DIR}"

echo ""
echo "============================================"
echo "Done! Results in ${ANALYSIS_DIR}"
echo "============================================"

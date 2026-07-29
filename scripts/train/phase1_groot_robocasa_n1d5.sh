#!/bin/bash
# Phase 1 (GR00T **N1.5** × RoboCasa) ProgressPredictor 학습.
#
# N1.6 의 ``phase1_groot_robocasa.sh`` 와 별개 — N1.5 의 Eagle 로 추출된 cache 사용.
# 다른 점:
#   input_dim = 2048  (N1.5 LLM hidden_size = N1.6 와 동일)
#   proj_dim  = 1536  (N1.5 DiT KV embedding dim; Phase 2 wrapper 에 직접 concat)
#   data_root = data/robocasa/v1.0/target/atomic   (15 task, episode<200)
#   cache_root = data/robocasa_eagle_pre_llm_target_n1d5
#
# 실행:
#   docker exec lerobot bash /temporal_vla/scripts/train/phase1_groot_robocasa_n1d5.sh
#
# 옵션 env var override:
#   INNER_MODEL_TYPE=linear_preLN
#   SAVE_SUFFIX=mytag
#   N_EPOCHS=20
#   MAX_EPISODES_PER_TASK=200

set -e
cd /temporal_vla

INNER_MODEL_TYPE="${INNER_MODEL_TYPE:-linear}"
SAVE_SUFFIX="${SAVE_SUFFIX:-target15_n1d5}"
N_EPOCHS="${N_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
MAX_WINDOWS_PER_EPISODE="${MAX_WINDOWS_PER_EPISODE:-24}"
MAX_EPISODES_PER_TASK="${MAX_EPISODES_PER_TASK:-200}"

DEFAULT_TASKS=(
  CloseFridge CloseToasterOvenDoor CoffeeSetupMug OpenCabinet OpenDrawer
  PickPlaceCounterToCabinet PickPlaceCounterToStove PickPlaceDrawerToCounter
  PickPlaceSinkToCounter PickPlaceToasterToCounter SlideDishwasherRack
  TurnOffStove TurnOnElectricKettle TurnOnMicrowave TurnOnSinkFaucet
)
if [ -n "${TASKS:-}" ]; then
  # shellcheck disable=SC2206
  TASKS_ARR=( ${TASKS} )
else
  TASKS_ARR=( "${DEFAULT_TASKS[@]}" )
fi
WANDB_RUN_NAME="${WANDB_RUN_NAME:-phase1_groot_robocasa_n1d5_${#TASKS_ARR[@]}tasks_${INNER_MODEL_TYPE}}"

cmd=(
    python scripts/train/phase1_groot_robocasa.py
    --data_root  data/robocasa/v1.0/target/atomic
    --cache_root data/robocasa_eagle_pre_llm_target_n1d5
    --tasks      "${TASKS_ARR[@]}"
    --window_size 8
    --max_windows_per_episode "${MAX_WINDOWS_PER_EPISODE}"
    --train_frac 0.9
    --input_dim 2048
    --proj_dim  1536
    --inner_model_type "${INNER_MODEL_TYPE}"
    --head_hidden_dim 128
    --eta_base 0.1
    --n_epochs "${N_EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --lr "${LR}"
    --lambda_self 0.5
    --val_steps 50
    --log_interval 50
    --num_workers 0
    --device cuda
    --save_dir outputs/train/phase1_groot_robocasa
    --wandb_project temporal-vla
    --wandb_run_name "${WANDB_RUN_NAME}"
    --max_episodes_per_task "${MAX_EPISODES_PER_TASK}"
    --save_suffix "${SAVE_SUFFIX}_${INNER_MODEL_TYPE}"
)

echo "=========================================="
echo "Phase 1 training (N1.5, RoboCasa target ${#TASKS_ARR[@]} task)"
echo "  inner_model_type:        ${INNER_MODEL_TYPE}"
echo "  max_episodes_per_task:   ${MAX_EPISODES_PER_TASK}"
echo "  max_windows_per_episode: ${MAX_WINDOWS_PER_EPISODE}"
echo "  n_epochs:                ${N_EPOCHS}"
echo "  batch_size:              ${BATCH_SIZE}"
echo "  wandb_run_name:          ${WANDB_RUN_NAME}"
echo "  save_suffix:             ${SAVE_SUFFIX}_${INNER_MODEL_TYPE}"
echo "=========================================="
exec "${cmd[@]}"

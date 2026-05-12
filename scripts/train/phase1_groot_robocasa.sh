#!/bin/bash
# Phase 1 (GR00T × RoboCasa) ProgressPredictor 학습.
#
# Stage 0 (Eagle pre-LLM 캐시) 가 완료된 뒤 실행. Stage 2 에서 frozen 으로 사용
# 되므로 input_dim/proj_dim=2048 (= Eagle Qwen3-1.7B hidden = DiT KV dim) 고정
# — wrapper 에서 별도 projection 없이 DiT cross-attn KV 에 직접 concat 하기 위해.
#
# 실행:
#   docker compose exec lerobot bash -lc \
#       'bash /temporal_vla/scripts/train/phase1_groot_robocasa.sh'
#
# 옵션 env var override:
#   INNER_MODEL_TYPE=linear_preLN bash phase1_groot_robocasa.sh
#   SAVE_SUFFIX=mytag             # outputs/.../<date>_<suffix>/
#   WANDB_RUN_NAME=phase1_xx
#   N_EPOCHS=20

set -e
cd /temporal_vla

INNER_MODEL_TYPE="${INNER_MODEL_TYPE:-linear}"
SAVE_SUFFIX="${SAVE_SUFFIX:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-phase1_groot_robocasa_30tasks_${INNER_MODEL_TYPE}}"
N_EPOCHS="${N_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
MAX_WINDOWS_PER_EPISODE="${MAX_WINDOWS_PER_EPISODE:-24}"

cmd=(
    python scripts/train/phase1_groot_robocasa.py
    --data_root  data/robocasa/v1.0/pretrain/atomic
    --cache_root data/robocasa_eagle_pre_llm
    --tasks
        OpenDrawer CloseDrawer OpenCabinet CloseCabinet
        OpenFridge CloseFridge OpenMicrowave CloseMicrowave
        PickPlaceCounterToStove PickPlaceCounterToSink
        OpenOven OpenDishwasher OpenFridgeDrawer OpenToasterOvenDoor
        OpenBlenderLid OpenElectricKettleLid OpenStandMixerHead
        PickPlaceCounterToCabinet PickPlaceCabinetToCounter PickPlaceCounterToDrawer
        PickPlaceCounterToMicrowave PickPlaceCounterToOven
        PickPlaceSinkToCounter PickPlaceStoveToCounter
        TurnOnSinkFaucet TurnOnStove TurnOnMicrowave
        TurnOnBlender TurnOnElectricKettle TurnOnToaster
    --window_size 8
    --max_windows_per_episode "${MAX_WINDOWS_PER_EPISODE}"
    --train_frac 0.9
    --input_dim 2048
    --proj_dim  2048
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
)

if [ -n "${SAVE_SUFFIX}" ]; then
    cmd+=(--save_suffix "${SAVE_SUFFIX}")
fi

echo "=========================================="
echo "Phase 1 training (RoboCasa 30 task)"
echo "  inner_model_type:        ${INNER_MODEL_TYPE}"
echo "  max_windows_per_episode: ${MAX_WINDOWS_PER_EPISODE}"
echo "  n_epochs:                ${N_EPOCHS}"
echo "  batch_size:              ${BATCH_SIZE}"
echo "  wandb_run_name:          ${WANDB_RUN_NAME}"
echo "  save_suffix:             ${SAVE_SUFFIX:-<none>}"
echo "=========================================="
exec "${cmd[@]}"

# 매 epoch 끝마다 epoch_NN.pt (plain state_dict) 저장. 5ep / 10ep 비교는 epoch_05.pt
# 와 epoch_10.pt 를 Phase 2 wrapper.load_predictor_state(...) 에 각각 넣어서.

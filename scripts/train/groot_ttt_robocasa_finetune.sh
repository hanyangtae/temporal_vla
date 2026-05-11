#!/usr/bin/env bash
# GR00T N1.6 + TTT fine-tuning on RoboCasa 10-task merged LeRobot v2.1 dataset.
#
# 베이스: scripts/train/groot_robocasa_finetune.sh (do-dong-park).
# 차이: launch_finetune_ttt.py (TTT mirror entry) 사용 + Phase 1 predictor 경로 주입.
#
# Full TTT fine-tune:
#   docker compose exec groot bash /temporal_vla/scripts/train/groot_ttt_robocasa_finetune.sh
#
# Baseline (TTT 없이) — 같은 entry 로 baseline 도 가능. TTT_PREDICTOR_PATH 비우기:
#   TTT_PREDICTOR_PATH= bash .../groot_ttt_robocasa_finetune.sh

set -euo pipefail

cd /temporal_vla/src/policies/Isaac-GR00T

NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/temporal_vla/checkpoints/nvidia/GR00T-N1.6-3B}"
DATASET_PATH="${DATASET_PATH:-/temporal_vla/data/datasets/robocasa_10tasks_lerobot_v21}"
MODALITY_CONFIG_PATH="${MODALITY_CONFIG_PATH:-/temporal_vla/configs/policies/groot_robocasa_panda_omron_config.py}"
OUTPUT_DIR="${OUTPUT_DIR:-/temporal_vla/outputs/groot_ttt_robocasa_10tasks}"

# Phase 1 체크포인트. 비우면 baseline GR00T finetune (TTT 비활성).
TTT_PREDICTOR_PATH="${TTT_PREDICTOR_PATH:-/temporal_vla/outputs/train/phase1_groot_robocasa/latest/phase1_final.pt}"
TTT_UPDATE_IN_TRAIN="${TTT_UPDATE_IN_TRAIN:-False}"

MAX_STEPS="${MAX_STEPS:-20000}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
SHARD_SIZE="${SHARD_SIZE:-1024}"
NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH:-100000}"
EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-0.1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
USE_WANDB="${USE_WANDB:-0}"
TUNE_PROJECTOR="${TUNE_PROJECTOR:-1}"
TUNE_DIFFUSION_MODEL="${TUNE_DIFFUSION_MODEL:-1}"

cmd=(
    /temporal_vla/scripts/train/launch_finetune_ttt.py
    --base_model_path "${BASE_MODEL_PATH}"
    --dataset_path "${DATASET_PATH}"
    --embodiment_tag ROBOCASA_PANDA_OMRON
    --modality_config_path "${MODALITY_CONFIG_PATH}"
    --num_gpus "${NUM_GPUS}"
    --output_dir "${OUTPUT_DIR}"
    --save_steps "${SAVE_STEPS}"
    --save_total_limit "${SAVE_TOTAL_LIMIT}"
    --max_steps "${MAX_STEPS}"
    --warmup_ratio "${WARMUP_RATIO}"
    --weight_decay "${WEIGHT_DECAY}"
    --learning_rate "${LEARNING_RATE}"
    --global_batch_size "${GLOBAL_BATCH_SIZE}"
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
    --shard_size "${SHARD_SIZE}"
    --num_shards_per_epoch "${NUM_SHARDS_PER_EPOCH}"
    --episode_sampling_rate "${EPISODE_SAMPLING_RATE}"
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
)

if [ "${TUNE_PROJECTOR}" = "1" ]; then
    cmd+=(--tune-projector)
else
    cmd+=(--no-tune-projector)
fi

if [ "${TUNE_DIFFUSION_MODEL}" = "1" ]; then
    cmd+=(--tune-diffusion-model)
else
    cmd+=(--no-tune-diffusion-model)
fi

if [ "${USE_WANDB}" = "1" ]; then
    cmd+=(--use_wandb)
fi

# TTT 옵션
if [ -n "${TTT_PREDICTOR_PATH}" ]; then
    cmd+=(--ttt_predictor_path "${TTT_PREDICTOR_PATH}")
fi
if [ "${TTT_UPDATE_IN_TRAIN}" = "True" ] || [ "${TTT_UPDATE_IN_TRAIN}" = "1" ]; then
    cmd+=(--ttt_update_in_train)
fi

echo "=========================================="
echo "GR00T + TTT RoboCasa 10-task fine-tune"
echo "  Dataset:      ${DATASET_PATH}"
echo "  Base model:   ${BASE_MODEL_PATH}"
echo "  Output:       ${OUTPUT_DIR}"
echo "  Steps:        ${MAX_STEPS}"
echo "  Batch size:   ${GLOBAL_BATCH_SIZE}"
echo "  GPUs:         ${NUM_GPUS}"
echo "  TTT predictor: ${TTT_PREDICTOR_PATH:-<disabled>}"
echo "  TTT update:   ${TTT_UPDATE_IN_TRAIN}"
echo "=========================================="

if [ "${NUM_GPUS}" = "1" ]; then
    exec python "${cmd[@]}"
fi

exec torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" "${cmd[@]}"

#!/bin/bash
# X-VLA LoRA Fine-tuning on RoboCasa (xvla 컨테이너 내에서 실행)
#
# 사용법:
#   docker compose --profile xvla run --rm xvla bash /temporal_vla/scripts/train/xvla.sh
#
# 또는 직접 컨테이너 안에서:
#   bash /temporal_vla/scripts/train/xvla.sh

set -e

# ── 설정 ─────────────────────────────────────────
DATASET_PATH="${DATASET_PATH:-/temporal_vla/data/datasets_v3/v1.0/pretrain}"
MODEL_PATH="${MODEL_PATH:-lerobot/xvla-base}"
OUTPUT_DIR="${OUTPUT_DIR:-/temporal_vla/outputs/xvla_robocasa}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-1e-4}"
STEPS="${STEPS:-50000}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
# ──────────────────────────────────────────────────

echo "============================================"
echo "  X-VLA LoRA Fine-tuning on RoboCasa"
echo "============================================"
echo "Model:      ${MODEL_PATH}"
echo "Dataset:    ${DATASET_PATH}"
echo "Output:     ${OUTPUT_DIR}"
echo "Batch size: ${BATCH_SIZE}"
echo "LR:         ${LR}"
echo "Steps:      ${STEPS}"
echo "============================================"

lerobot-train \
    --dataset.repo_id=local \
    --dataset.root="${DATASET_PATH}" \
    --policy.path="${MODEL_PATH}" \
    --policy.dtype=bfloat16 \
    --batch_size="${BATCH_SIZE}" \
    --lr="${LR}" \
    --steps="${STEPS}" \
    --save_freq="${SAVE_FREQ}" \
    --output_dir="${OUTPUT_DIR}" \
    --gradient_checkpointing=true \
    --wandb.disable=true

echo "Training complete. Checkpoint saved to ${OUTPUT_DIR}"

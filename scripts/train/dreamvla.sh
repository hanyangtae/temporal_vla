#!/bin/bash
# DreamVLA Fine-tuning on RoboCasa (dreamvla 컨테이너 내에서 실행)
#
# 사전 준비:
#   git submodule update --init src/policies/dreamvla
#
# 사용법:
#   docker compose --profile dreamvla run --rm dreamvla bash /temporal_vla/scripts/train/dreamvla.sh

set -e

DREAMVLA_DIR="/temporal_vla/src/policies/dreamvla"
DATA_DIR="${DATA_DIR:-/temporal_vla/data/datasets_dreamvla}"
OUTPUT_DIR="${OUTPUT_DIR:-/temporal_vla/outputs/dreamvla_robocasa}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-1e-4}"
EPOCHS="${EPOCHS:-50}"

if [ ! -d "${DREAMVLA_DIR}" ]; then
    echo "Error: DreamVLA repo not found at ${DREAMVLA_DIR}"
    echo "Run: git submodule update --init src/policies/dreamvla"
    exit 1
fi

echo "============================================"
echo "  DreamVLA Fine-tuning on RoboCasa"
echo "============================================"
echo "Data:       ${DATA_DIR}"
echo "Output:     ${OUTPUT_DIR}"
echo "Batch size: ${BATCH_SIZE}"
echo "LR:         ${LR}"
echo "Epochs:     ${EPOCHS}"
echo "============================================"

cd "${DREAMVLA_DIR}"

python train.py \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --epochs "${EPOCHS}" \
    --gradient_checkpointing

echo "Training complete. Checkpoint saved to ${OUTPUT_DIR}"

#!/bin/bash
# UP-VLA Fine-tuning (upvla 컨테이너 내에서 실행)
#
# 사전 준비:
#   1. HuggingFace에서 backbone 다운로드:
#      - showlab/show-o-w-clip-vit-512x512 → /temporal_vla/upvla/showlab/show-o-w-clip-vit-512x512
#      - microsoft/phi-1_5              → /temporal_vla/upvla/showlab/phi-1_5
#   2. Calvin 또는 사용자 데이터 전처리 (preprocess_data/ 참조)
#
# 사용법:
#   docker compose run --rm upvla bash /temporal_vla/scripts/train_upvla.sh
#
# 환경변수로 설정 변경 가능:
#   STAGE=pred|action  학습 단계 (기본: action)
#   DATA_DIR           전처리된 데이터 경로
#   MMU_DATA_DIR       LLaVA MMU 데이터 경로 (없으면 코트레이닝 스킵)
#   OUTPUT_DIR         체크포인트 저장 경로
#   GPUS               사용할 GPU (기본: 0,1,2,3)
#   NUM_GPUS           GPU 수 (기본: 4)
#   PORT               accelerate main_process_port (기본: 8888)

set -e

UPVLA_DIR="/temporal_vla/upvla"
STAGE="${STAGE:-action}"
DATA_DIR="${DATA_DIR:-/temporal_vla/data/datasets_upvla}"
MMU_DATA_DIR="${MMU_DATA_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/temporal_vla/outputs/upvla_$(date +%Y%m%d_%H%M%S)}"
GPUS="${GPUS:-0,1,2,3}"
NUM_GPUS="${NUM_GPUS:-4}"
PORT="${PORT:-8888}"

if [ ! -d "${UPVLA_DIR}" ]; then
    echo "Error: UP-VLA repo not found at ${UPVLA_DIR}"
    exit 1
fi

case "${STAGE}" in
    pred)
        CONFIG="${UPVLA_DIR}/config/upvla_pred_tuning.yaml"
        ;;
    action)
        CONFIG="${UPVLA_DIR}/config/upvla_action_tuning.yaml"
        ;;
    *)
        echo "Error: STAGE must be 'pred' or 'action', got '${STAGE}'"
        exit 1
        ;;
esac

# GPU 수에 맞는 accelerate config 선택
ACCEL_CONFIG="${UPVLA_DIR}/accelerate_configs/4_gpus_deepspeed_zero2.yaml"
if [ ! -f "${ACCEL_CONFIG}" ]; then
    echo "Error: accelerate config not found at ${ACCEL_CONFIG}"
    exit 1
fi

echo "============================================"
echo "  UP-VLA Training"
echo "============================================"
echo "Stage:      ${STAGE}"
echo "Config:     ${CONFIG}"
echo "Data:       ${DATA_DIR}"
echo "Output:     ${OUTPUT_DIR}"
echo "GPUs:       ${GPUS} (${NUM_GPUS} GPUs)"
echo "============================================"

mkdir -p "${OUTPUT_DIR}"
cd "${UPVLA_DIR}"

CUDA_VISIBLE_DEVICES="${GPUS}" accelerate launch \
    --config_file "${ACCEL_CONFIG}" \
    --num_processes "${NUM_GPUS}" \
    --main_process_port "${PORT}" \
    train_upvla.py \
    config="${CONFIG}" \
    dataset.params.train_pre_shards_path_or_url="${DATA_DIR}" \
    experiment.output_dir="${OUTPUT_DIR}" \
    ${MMU_DATA_DIR:+dataset.params.train_mmu_shards_path_or_url="${MMU_DATA_DIR}"}

echo "Training complete. Checkpoint saved to ${OUTPUT_DIR}"

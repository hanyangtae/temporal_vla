#!/usr/bin/env bash
# GR00T N1.6 baseline fine-tuning on RoboCasa 10-task atomic dataset (per-task mixture).
#
# 변경 이력: 이전엔 merged single LeRobot v2.1 dataset 을 upstream launch_finetune.py 로
# 학습했으나, per-task mixture 로 전환하면서 mirror entry launch_finetune_ttt.py 의 ":"-split
# 을 사용한다 (upstream 의 --dataset_path 가 단일 str 라 multi-path 미지원). TTT 학습은
# groot_ttt_robocasa_finetune.sh 별도. 이 wrapper 는 baseline 전용으로 TTT 인자 없음.
#
# Full baseline fine-tune:
#   bash scripts/train/groot_robocasa_finetune.sh
#   REPO_ROOT=/home/dongkyu/temporal_vla bash scripts/train/groot_robocasa_finetune.sh
#   docker compose exec groot bash /temporal_vla/scripts/train/groot_robocasa_finetune.sh
#
# Short syntax check run:
#   MAX_STEPS=2 SAVE_STEPS=2 GLOBAL_BATCH_SIZE=1 bash scripts/train/groot_robocasa_finetune.sh

set -euo pipefail

expand_home() {
    local path="$1"
    printf '%s\n' "${path/#\~/$HOME}"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(expand_home "${REPO_ROOT:-${DEFAULT_REPO_ROOT}}")"
ISAAC_GR00T_DIR="$(expand_home "${ISAAC_GR00T_DIR:-${REPO_ROOT}/src/policies/Isaac-GR00T}")"

cd "${ISAAC_GR00T_DIR}"

if [ -x ".venv/bin/python" ]; then
    PYTHON_CMD=(".venv/bin/python")
    TORCHRUN_CMD=(".venv/bin/torchrun")
elif command -v uv >/dev/null 2>&1; then
    PYTHON_CMD=(uv run python)
    TORCHRUN_CMD=(uv run torchrun)
else
    echo "ERROR: Isaac-GR00T environment not found at $(pwd)/.venv/bin/python, and uv is unavailable" >&2
    echo "Run: cd ${ISAAC_GR00T_DIR} && uv sync" >&2
    exit 1
fi

NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"

LOCAL_BASE_MODEL_PATH="${REPO_ROOT}/checkpoints/nvidia/GR00T-N1.6-3B"
if [ -z "${BASE_MODEL_PATH:-}" ]; then
    if [ -f "${LOCAL_BASE_MODEL_PATH}/config.json" ]; then
        BASE_MODEL_PATH="${LOCAL_BASE_MODEL_PATH}"
    else
        BASE_MODEL_PATH="nvidia/GR00T-N1.6-3B"
    fi
fi

# Per-task mixture: 10 atomic task 경로를 ":" 로 join. launch_finetune_ttt 가 split.
ATOMIC_ROOT="${ATOMIC_ROOT:-${REPO_ROOT}/data/robocasa/v1.0/pretrain/atomic}"
DATE_TAG="${DATE_TAG:-20250819}"
DATASET_PATH="${DATASET_PATH:-\
${ATOMIC_ROOT}/OpenDrawer/${DATE_TAG}/lerobot:\
${ATOMIC_ROOT}/CloseDrawer/${DATE_TAG}/lerobot:\
${ATOMIC_ROOT}/OpenCabinet/${DATE_TAG}/lerobot:\
${ATOMIC_ROOT}/CloseCabinet/${DATE_TAG}/lerobot:\
${ATOMIC_ROOT}/OpenFridge/${DATE_TAG}/lerobot:\
${ATOMIC_ROOT}/CloseFridge/${DATE_TAG}/lerobot:\
${ATOMIC_ROOT}/OpenMicrowave/${DATE_TAG}/lerobot:\
${ATOMIC_ROOT}/CloseMicrowave/${DATE_TAG}/lerobot:\
${ATOMIC_ROOT}/PickPlaceCounterToStove/${DATE_TAG}/lerobot:\
${ATOMIC_ROOT}/PickPlaceCounterToSink/${DATE_TAG}/lerobot}"

MODALITY_CONFIG_PATH="${MODALITY_CONFIG_PATH:-${REPO_ROOT}/configs/policies/groot_robocasa_panda_omron_config.py}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/groot_robocasa_baseline_10tasks}"
ENTRY_SCRIPT="${ENTRY_SCRIPT:-${REPO_ROOT}/scripts/train/launch_finetune_ttt.py}"

BASE_MODEL_PATH="$(expand_home "${BASE_MODEL_PATH}")"
MODALITY_CONFIG_PATH="$(expand_home "${MODALITY_CONFIG_PATH}")"
OUTPUT_DIR="$(expand_home "${OUTPUT_DIR}")"
ENTRY_SCRIPT="$(expand_home "${ENTRY_SCRIPT}")"

# ckpt 1개 ≈ 22GB. SAVE_STEPS=5000 × LIMIT=4 = 88GB 디스크 사용 가정.
# 이전에 1k 주기 × limit=20 (=440GB) 으로 디스크 풀로 학습 사망한 사례 있음.
MAX_STEPS="${MAX_STEPS:-20000}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
SHARD_SIZE="${SHARD_SIZE:-1024}"
NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH:-100000}"
EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-0.1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
USE_WANDB="${USE_WANDB:-1}"
TUNE_PROJECTOR="${TUNE_PROJECTOR:-1}"
TUNE_DIFFUSION_MODEL="${TUNE_DIFFUSION_MODEL:-1}"

cmd=(
    "${ENTRY_SCRIPT}"
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

n_tasks=$(echo "${DATASET_PATH}" | awk -F':' '{print NF}')
echo "============================================"
echo "GR00T RoboCasa baseline fine-tune (per-task mixture)"
echo "  Repo root:    ${REPO_ROOT}"
echo "  GR00T dir:    ${ISAAC_GR00T_DIR}"
echo "  Entry:        ${ENTRY_SCRIPT}"
echo "  Datasets:     ${n_tasks} tasks"
echo "  Base model:   ${BASE_MODEL_PATH}"
echo "  Output:       ${OUTPUT_DIR}"
echo "  Steps:        ${MAX_STEPS}   (save every ${SAVE_STEPS}, keep ${SAVE_TOTAL_LIMIT})"
echo "  Batch size:   ${GLOBAL_BATCH_SIZE}"
echo "  GPUs:         ${NUM_GPUS}"
echo "  Tune proj:    ${TUNE_PROJECTOR}"
echo "  Tune DiT:     ${TUNE_DIFFUSION_MODEL}"
echo "  Optim:        adamw_torch (from upstream launch_finetune.py)"
echo "  VLLN:         upstream default"
echo "  Python:       ${PYTHON_CMD[*]}"
echo "============================================"

if [ "${NUM_GPUS}" = "1" ]; then
    exec "${PYTHON_CMD[@]}" "${cmd[@]}"
fi

exec "${TORCHRUN_CMD[@]}" --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" "${cmd[@]}"

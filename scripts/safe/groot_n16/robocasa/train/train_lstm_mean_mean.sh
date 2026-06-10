#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../run_config.sh"

SAFE_REPO="${SAFE_REPO:-/home/dongkyu/pkt_ws/SAFE}"
CONDA_ENV="${CONDA_ENV:-vla-safe}"

OUT_ROOT="${OUT_ROOT:-${ROBOCASA_SAFE_OUT_ROOT}}"
RUN_ROOT="${RUN_ROOT:-${ROBOCASA_SAFE_RUN_ROOT}}"
LOG_ROOT="${LOG_ROOT:-${RUN_ROOT}/experiments/mean_mean_baseline/train_logs}"
WANDB_DIR="${WANDB_DIR:-${RUN_ROOT}/experiments/mean_mean_baseline/wandb}"
WANDB_MODE="${WANDB_MODE:-online}"
DATA_PATH="${DATA_PATH:-${RUN_ROOT}/split}"
SEED="${SEED:-0}"

if [ ! -d "${DATA_PATH}" ]; then
  echo "ERROR: DATA_PATH 디렉토리가 없음: ${DATA_PATH}" >&2
  echo "       run env(예: seen18_env.sh)를 source 하거나 DATA_PATH 를 직접 지정하세요." >&2
  exit 1
fi

cd "${SAFE_REPO}"

env \
  PYTHONPATH="${SAFE_REPO}${PYTHONPATH:+:${PYTHONPATH}}" \
  WANDB_MODE="${WANDB_MODE}" \
  conda run -n "${CONDA_ENV}" python -m failure_prob.train \
    dataset=groot_n16 \
    model=lstm \
    dataset.data_path="${DATA_PATH}" \
    dataset.subset_name="${ROBOCASA_SAFE_SUBSET_NAME}" \
    dataset.diff_idx_rel=mean \
    dataset.horizon_idx_rel=mean \
    model.batch_size=64 \
    model.lr=3e-4 \
    model.lambda_reg=1e-2 \
    model.n_epochs=1000 \
    train.seed="${SEED}" \
    train.roc_every=25 \
    train.eval_save_ckpt=true \
    train.eval_save_logs=true \
    train.eval_save_timing_plots=false \
    train.logs_save_root="${LOG_ROOT}" \
    train.wandb_dir="${WANDB_DIR}" \
    train.wandb_group_name=groot_n16_safe_lstm_seen4_unseen2 \
    train.exp_suffix="seed${SEED}_mean_mean"

#!/usr/bin/env bash
set -euo pipefail

SAFE_REPO="${SAFE_REPO:-/home/dongkyu/pdk_ws/SAFE}"
CONDA_ENV="${CONDA_ENV:-vla-safe}"

OUT_ROOT="${OUT_ROOT:-/home/dongkyu/pdk_ws/temporal_vla/outputs/eval/robocasa/groot_n16}"
RUN_ROOT="${RUN_ROOT:-${OUT_ROOT}/safe_seen4_unseen2_100ep}"
LOG_ROOT="${LOG_ROOT:-${RUN_ROOT}/experiments/mean_mean_baseline/train_logs}"
WANDB_DIR="${WANDB_DIR:-${RUN_ROOT}/experiments/mean_mean_baseline/wandb}"
WANDB_MODE="${WANDB_MODE:-online}"
SEED="${SEED:-0}"

cd "${SAFE_REPO}"

env \
  PYTHONPATH="${SAFE_REPO}${PYTHONPATH:+:${PYTHONPATH}}" \
  WANDB_MODE="${WANDB_MODE}" \
  conda run -n "${CONDA_ENV}" python -m failure_prob.train \
    dataset=groot_n16 \
    model=lstm \
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

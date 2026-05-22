#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../run_config.sh"

SAFE_REPO="${SAFE_REPO:-/home/dongkyu/pdk_ws/SAFE}"
CONDA_ENV="${CONDA_ENV:-vla-safe}"

OUT_ROOT="${OUT_ROOT:-${ROBOCASA_SAFE_OUT_ROOT}}"
RUN_ROOT="${RUN_ROOT:-${ROBOCASA_SAFE_RUN_ROOT}}"
LOG_ROOT="${LOG_ROOT:-${RUN_ROOT}/experiments/aggregation_ablation/train_logs}"
WANDB_DIR="${WANDB_DIR:-${RUN_ROOT}/experiments/aggregation_ablation/wandb}"
HYDRA_ROOT="${HYDRA_ROOT:-${RUN_ROOT}/experiments/aggregation_ablation/hydra}"
WANDB_MODE="${WANDB_MODE:-online}"

LR="${LR:-3e-4}"
LAMBDA_REG="${LAMBDA_REG:-1e-2}"
N_EPOCHS="${N_EPOCHS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
ROC_EVERY="${ROC_EVERY:-25}"

HORIZON_VALUES=(${HORIZON_VALUES:-0.0 1.0 mean concat-2})
DIFF_VALUES=(${DIFF_VALUES:-0.0 1.0 mean concat-2})
SEEDS=(${SEEDS:-0 1 2})

mkdir -p "${LOG_ROOT}" "${WANDB_DIR}" "${HYDRA_ROOT}"

cd "${SAFE_REPO}"

for horizon_idx_rel in "${HORIZON_VALUES[@]}"; do
  for diff_idx_rel in "${DIFF_VALUES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      suffix="agg_h${horizon_idx_rel}_d${diff_idx_rel}_seed${seed}"
      suffix="${suffix//./p}"
      suffix="${suffix//-/_}"

      run_log_root="${LOG_ROOT}/groot_n16-${ROBOCASA_SAFE_SUBSET_NAME}-lstm-${suffix}"
      if find "${run_log_root}" -path "*/model_final.ckpt" -type f -print -quit 2>/dev/null | grep -q .; then
        echo "Skipping existing ${suffix}"
        continue
      fi

      echo "Running ${suffix}: horizon_idx_rel=${horizon_idx_rel}, diff_idx_rel=${diff_idx_rel}, seed=${seed}"
      env \
        PYTHONDONTWRITEBYTECODE=1 \
        MPLCONFIGDIR=/tmp/matplotlib \
        PYTHONPATH="${SAFE_REPO}${PYTHONPATH:+:${PYTHONPATH}}" \
        WANDB_MODE="${WANDB_MODE}" \
        conda run -n "${CONDA_ENV}" python -m failure_prob.train \
          dataset=groot_n16 \
          model=lstm \
          dataset.horizon_idx_rel="${horizon_idx_rel}" \
          dataset.diff_idx_rel="${diff_idx_rel}" \
          model.batch_size="${BATCH_SIZE}" \
          model.lr="${LR}" \
          model.lambda_reg="${LAMBDA_REG}" \
          model.n_epochs="${N_EPOCHS}" \
          train.seed="${seed}" \
          train.roc_every="${ROC_EVERY}" \
          train.eval_save_ckpt=true \
          train.eval_save_logs=true \
          train.eval_save_timing_plots=false \
          train.logs_save_root="${LOG_ROOT}" \
          train.wandb_dir="${WANDB_DIR}" \
          train.wandb_group_name=groot_n16_safe_lstm_aggregation_ablation \
          train.exp_suffix="${suffix}" \
          hydra.run.dir="${HYDRA_ROOT}/${suffix}"
    done
  done
done

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../run_config.sh"

SAFE_REPO="${SAFE_REPO:-/home/dongkyu/pkt_ws/SAFE}"
CONDA_ENV="${CONDA_ENV:-vla-safe}"

OUT_ROOT="${OUT_ROOT:-${ROBOCASA_SAFE_OUT_ROOT}}"
RUN_ROOT="${RUN_ROOT:-${ROBOCASA_SAFE_RUN_ROOT}}"
LOG_ROOT="${LOG_ROOT:-${RUN_ROOT}/experiments/${ROBOCASA_SAFE_HPARAM_SWEEP_ID}/train_logs}"
WANDB_DIR="${WANDB_DIR:-${RUN_ROOT}/experiments/${ROBOCASA_SAFE_HPARAM_SWEEP_ID}/wandb}"
HYDRA_ROOT="${HYDRA_ROOT:-${RUN_ROOT}/experiments/${ROBOCASA_SAFE_HPARAM_SWEEP_ID}/hydra}"
WANDB_MODE="${WANDB_MODE:-online}"
DATA_PATH="${DATA_PATH:-${RUN_ROOT}/split}"

HORIZON_IDX_REL="${HORIZON_IDX_REL:-${ROBOCASA_SAFE_FINAL_HORIZON_IDX_REL}}"
DIFF_IDX_REL="${DIFF_IDX_REL:-${ROBOCASA_SAFE_FINAL_DIFF_IDX_REL}}"
N_EPOCHS="${N_EPOCHS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
ROC_EVERY="${ROC_EVERY:-25}"

LR_VALUES=(${LR_VALUES:-1e-4 3e-4 1e-3})
LAMBDA_REG_VALUES=(${LAMBDA_REG_VALUES:-1e-2 1e-1 1})
SEEDS=(${SEEDS:-0 1 2})

mkdir -p "${LOG_ROOT}" "${WANDB_DIR}" "${HYDRA_ROOT}"

if [ ! -d "${DATA_PATH}" ]; then
  echo "ERROR: DATA_PATH 디렉토리가 없음: ${DATA_PATH}" >&2
  echo "       run env(예: seen18_env.sh)를 source 하거나 DATA_PATH 를 직접 지정하세요." >&2
  exit 1
fi

cd "${SAFE_REPO}"

for lr in "${LR_VALUES[@]}"; do
  for lambda_reg in "${LAMBDA_REG_VALUES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      suffix="agg_h${HORIZON_IDX_REL}_d${DIFF_IDX_REL}_lr${lr}_reg${lambda_reg}_seed${seed}"
      suffix="${suffix//./p}"
      suffix="${suffix//-/_}"

      run_log_root="${LOG_ROOT}/groot_n16-${ROBOCASA_SAFE_SUBSET_NAME}-lstm-${suffix}"
      if find "${run_log_root}" -path "*/model_final.ckpt" -type f -print -quit 2>/dev/null | grep -q .; then
        echo "Skipping existing ${suffix}"
        continue
      fi

      echo "Running ${suffix}: horizon_idx_rel=${HORIZON_IDX_REL}, diff_idx_rel=${DIFF_IDX_REL}, lr=${lr}, lambda_reg=${lambda_reg}, seed=${seed}"
      env \
        PYTHONDONTWRITEBYTECODE=1 \
        MPLCONFIGDIR=/tmp/matplotlib \
        PYTHONPATH="${SAFE_REPO}${PYTHONPATH:+:${PYTHONPATH}}" \
        WANDB_MODE="${WANDB_MODE}" \
        conda run -n "${CONDA_ENV}" python -m failure_prob.train \
          dataset=groot_n16 \
          model=lstm \
          dataset.data_path="${DATA_PATH}" \
          dataset.subset_name="${ROBOCASA_SAFE_SUBSET_NAME}" \
          dataset.horizon_idx_rel="${HORIZON_IDX_REL}" \
          dataset.diff_idx_rel="${DIFF_IDX_REL}" \
          model.batch_size="${BATCH_SIZE}" \
          model.lr="${lr}" \
          model.lambda_reg="${lambda_reg}" \
          model.n_epochs="${N_EPOCHS}" \
          train.seed="${seed}" \
          train.roc_every="${ROC_EVERY}" \
          train.eval_save_ckpt=true \
          train.eval_save_logs=true \
          train.eval_save_timing_plots=false \
          train.logs_save_root="${LOG_ROOT}" \
          train.wandb_dir="${WANDB_DIR}" \
          train.wandb_group_name="groot_n16_safe_lstm_hparam_sweep_${ROBOCASA_SAFE_FINAL_AGGREGATION_SLUG}" \
          train.exp_suffix="${suffix}" \
          hydra.run.dir="${HYDRA_ROOT}/${suffix}"
    done
  done
done

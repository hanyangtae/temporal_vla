#!/usr/bin/env bash
# Sweep SAFE detector hyperparameters on the GR00T N1.5 5-task RoboCasa rollout set.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/dongkyu/pdk_ws/temporal_vla}"
SAFE_ROOT="${SAFE_ROOT:-/home/dongkyu/pdk_ws/SAFE}"
RUN_ID="${RUN_ID:-seen5_trainval75_cp15_test10_subset100}"

MODELS="${MODELS:-lstm}"
LRS="${LRS:-1e-4,3e-4,1e-3}"
LAMBDA_REGS="${LAMBDA_REGS:-1e-3,1e-2,1e-1,1}"
SEEDS="${SEEDS:-0,1,2}"
EPOCHS="${EPOCHS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
N_LAYERS="${N_LAYERS:-1}"
FEATURE_POOL="${FEATURE_POOL:-masked_mean}"
CUMSUM="${CUMSUM:-false}"
NORMALIZE_HIDDEN_STATES="${NORMALIZE_HIDDEN_STATES:-false}"
SEEN_TRAIN_RATIO="${SEEN_TRAIN_RATIO:-0.8}"
UNSEEN_TASK_RATIO="${UNSEEN_TASK_RATIO:-0.0}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/outputs/eval/robocasa/groot_n15/split_seen5_trainval75_cp15_test10_subset100/trainval}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/eval/robocasa/groot_n15/safe_lstm_detector_sweep_${RUN_ID}_orig_grid}"
HYDRA_DIR="${HYDRA_DIR:-${OUT_ROOT}/hydra}"
WANDB_DIR="${WANDB_DIR:-${OUT_ROOT}/wandb}"
LOGS_ROOT="${LOGS_ROOT:-${OUT_ROOT}/logs}"
WANDB_GROUP="${WANDB_GROUP:-groot_n15_safe_tune_${RUN_ID}}"

mkdir -p "${HYDRA_DIR}" "${WANDB_DIR}" "${LOGS_ROOT}"

echo "SAFE_ROOT=${SAFE_ROOT}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "MODELS=${MODELS}, LRS=${LRS}, LAMBDA_REGS=${LAMBDA_REGS}, SEEDS=${SEEDS}"
echo "FEATURE_POOL=${FEATURE_POOL}, CUMSUM=${CUMSUM}, EPOCHS=${EPOCHS}"
echo "SEEN_TRAIN_RATIO=${SEEN_TRAIN_RATIO}, UNSEEN_TASK_RATIO=${UNSEEN_TASK_RATIO}"

MPLCONFIGDIR=/tmp/mpl \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="${SAFE_ROOT}" \
WANDB_MODE="${WANDB_MODE:-online}" \
python "${SAFE_ROOT}/failure_prob/train.py" \
    --multirun \
    "dataset=groot_n15" \
    "model=${MODELS}" \
    "dataset.data_path=${DATA_ROOT}" \
    "dataset.feature_pool=${FEATURE_POOL}" \
    "dataset.load_to_cuda=false" \
    "dataset.normalize_hidden_states=${NORMALIZE_HIDDEN_STATES}" \
    "dataset.seen_train_ratio=${SEEN_TRAIN_RATIO}" \
    "dataset.unseen_task_ratio=${UNSEEN_TASK_RATIO}" \
    "model.n_epochs=${EPOCHS}" \
    "model.batch_size=${BATCH_SIZE}" \
    "model.hidden_dim=${HIDDEN_DIM}" \
    "model.n_layers=${N_LAYERS}" \
    "model.cumsum=${CUMSUM}" \
    "model.lr=${LRS}" \
    "model.lambda_reg=${LAMBDA_REGS}" \
    "train.seed=${SEEDS}" \
    "train.wandb_dir=${WANDB_DIR}" \
    "train.logs_save_root=${LOGS_ROOT}" \
    "train.wandb_group_name=${WANDB_GROUP}" \
    "train.exp_suffix=groot_n15_safe_tune_cumsum_${CUMSUM}_seed_\${train.seed}" \
    "train.eval_save_logs=true" \
    "train.eval_save_ckpt=true" \
    "train.roc_every=9999" \
    "hydra.sweep.dir=${HYDRA_DIR}"

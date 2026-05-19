#!/usr/bin/env bash
# Train SAFE detectors on the GR00T N1.5 5-task RoboCasa rollout set.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/dongkyu/pdk_ws/temporal_vla}"
SAFE_ROOT="${SAFE_ROOT:-/home/dongkyu/pdk_ws/SAFE}"
RUN_ID="${RUN_ID:-seen5_trainval75_cp15_test10_subset100}"
MODEL="${MODEL:-lstm}"
EPOCHS="${EPOCHS:-1000}"
SEEDS="${SEEDS:-0-1-2}"
FEATURE_POOL="${FEATURE_POOL:-masked_mean}"
CUMSUM="${CUMSUM:-false}"
NORMALIZE_HIDDEN_STATES="${NORMALIZE_HIDDEN_STATES:-false}"
SEEN_TRAIN_RATIO="${SEEN_TRAIN_RATIO:-0.8}"
UNSEEN_TASK_RATIO="${UNSEEN_TASK_RATIO:-0.0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
N_LAYERS="${N_LAYERS:-1}"
LR="${LR:-}"
LAMBDA_REG="${LAMBDA_REG:-}"
DROPOUT="${DROPOUT:-0.0}"
ROC_EVERY="${ROC_EVERY:-9999}"
WANDB_GROUP="${WANDB_GROUP:-groot_n15_safe_train_${RUN_ID}}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/outputs/eval/robocasa/groot_n15/split_seen5_trainval75_cp15_test10_subset100/trainval}"
if [[ -n "${OUT_ROOT:-}" ]]; then
    OUT_ROOT_BASE="${OUT_ROOT_BASE:-${OUT_ROOT}}"
else
    OUT_ROOT_BASE="${OUT_ROOT_BASE:-${REPO_ROOT}/outputs/eval/robocasa/groot_n15/safe_lstm_detector_${RUN_ID}}"
fi

seed_spec="${SEEDS//,/ }"
seed_spec="${seed_spec//-/ }"
read -r -a SEED_LIST <<< "${seed_spec}"
if [[ "${#SEED_LIST[@]}" -eq 0 ]]; then
    echo "ERROR: no seeds parsed from SEEDS=${SEEDS}" >&2
    exit 1
fi

COMMON_OVERRIDES=(
    "dataset=groot_n15"
    "model=${MODEL}"
    "dataset.data_path=${DATA_ROOT}"
    "dataset.feature_pool=${FEATURE_POOL}"
    "dataset.load_to_cuda=false"
    "dataset.normalize_hidden_states=${NORMALIZE_HIDDEN_STATES}"
    "dataset.seen_train_ratio=${SEEN_TRAIN_RATIO}"
    "dataset.unseen_task_ratio=${UNSEEN_TASK_RATIO}"
    "model.n_epochs=${EPOCHS}"
    "model.batch_size=${BATCH_SIZE}"
    "model.hidden_dim=${HIDDEN_DIM}"
    "model.n_layers=${N_LAYERS}"
    "model.cumsum=${CUMSUM}"
    "train.roc_every=${ROC_EVERY}"
    "train.eval_save_logs=true"
    "train.eval_save_ckpt=true"
)

if [[ "${MODEL}" == "indep" ]]; then
    LR="${LR:-0.001}"
    LAMBDA_REG="${LAMBDA_REG:-0.001}"
    COMMON_OVERRIDES+=(
        "model.lr=${LR}"
        "model.lambda_reg=${LAMBDA_REG}"
    )
elif [[ "${MODEL}" == "lstm" ]]; then
    LR="${LR:-0.0003}"
    LAMBDA_REG="${LAMBDA_REG:-1.0}"
    COMMON_OVERRIDES+=(
        "model.lr=${LR}"
        "model.lambda_reg=${LAMBDA_REG}"
        "model.dropout=${DROPOUT}"
    )
else
    if [[ -n "${LR}" ]]; then
        COMMON_OVERRIDES+=("model.lr=${LR}")
    fi
    if [[ -n "${LAMBDA_REG}" ]]; then
        COMMON_OVERRIDES+=("model.lambda_reg=${LAMBDA_REG}")
    fi
fi

echo "SAFE_ROOT=${SAFE_ROOT}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "OUT_ROOT_BASE=${OUT_ROOT_BASE}"
echo "MODEL=${MODEL}, EPOCHS=${EPOCHS}, SEEDS=${SEEDS}, LR=${LR}, LAMBDA_REG=${LAMBDA_REG}"
echo "FEATURE_POOL=${FEATURE_POOL}, CUMSUM=${CUMSUM}, SEEN_TRAIN_RATIO=${SEEN_TRAIN_RATIO}, UNSEEN_TASK_RATIO=${UNSEEN_TASK_RATIO}"
echo "NORMALIZE_HIDDEN_STATES=${NORMALIZE_HIDDEN_STATES}"
echo "ROC_EVERY=${ROC_EVERY}, WANDB_GROUP=${WANDB_GROUP}"

for seed in "${SEED_LIST[@]}"; do
    if ! [[ "${seed}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: seed must be a non-negative integer, got: ${seed}" >&2
        exit 1
    fi

    RUN_OUT_ROOT="${OUT_ROOT_BASE}_seed${seed}"
    HYDRA_DIR="${RUN_OUT_ROOT}/hydra_${MODEL}"
    WANDB_DIR="${RUN_OUT_ROOT}/wandb"
    LOGS_ROOT="${RUN_OUT_ROOT}/logs"
    mkdir -p "${HYDRA_DIR}" "${WANDB_DIR}" "${LOGS_ROOT}"

    echo
    echo "== SAFE train seed ${seed} =="
    echo "RUN_OUT_ROOT=${RUN_OUT_ROOT}"

    SEED_OVERRIDES=(
        "${COMMON_OVERRIDES[@]}"
        "train.seed=${seed}"
        "train.wandb_dir=${WANDB_DIR}"
        "train.wandb_group_name=${WANDB_GROUP}"
        "train.logs_save_root=${LOGS_ROOT}"
        "train.exp_suffix=groot_n15_${RUN_ID}_${MODEL}_seed${seed}"
        "hydra.run.dir=${HYDRA_DIR}"
    )

    MPLCONFIGDIR=/tmp/mpl \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${SAFE_ROOT}" \
    WANDB_MODE="${WANDB_MODE:-online}" \
    python "${SAFE_ROOT}/failure_prob/train.py" "${SEED_OVERRIDES[@]}"
done

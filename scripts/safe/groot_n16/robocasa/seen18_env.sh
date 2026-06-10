# Source this before running the SAFE detector recipes on the seen18 run.
#
#   source scripts/safe/groot_n16/robocasa/seen18_env.sh
#   bash scripts/safe/groot_n16/robocasa/train/train_lstm_aggregation_ablation.sh
#
# It overrides the ROBOCASA_SAFE_* identity that both run_config.sh (shell) and
# run_config.py (analysis scripts) read, plus the SAFE repo / conda env / wandb
# knobs the train scripts use. After the aggregation ablation picks the best
# aggregation (Step 3), set ROBOCASA_SAFE_FINAL_HORIZON_IDX_REL / _DIFF_IDX_REL
# / _AGGREGATION_SLUG below before running the hparam sweep (Step 4).

_SEEN18_ENV_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)"
_SEEN18_REPO_ROOT="$(cd -- "${_SEEN18_ENV_DIR}/../../../.." && pwd)"

# Run identity (read by run_config.sh defaults and run_config.py env lookups).
export ROBOCASA_SAFE_RUN_ID="safe_seen18_4unseen_100ep"
export ROBOCASA_SAFE_EXPERIMENT_ID="seen18_4unseen_100ep"
export ROBOCASA_SAFE_SUBSET_NAME="robocasa_seen18_4unseen_100ep"

# SAFE source + runtime env (train scripts' broken defaults are pdk_ws/vla-safe).
export SAFE_REPO="$(cd -- "${_SEEN18_REPO_ROOT}/.." && pwd)/SAFE"
export CONDA_ENV="vla-safe"
export WANDB_MODE="offline"

# Materialized seen/unseen split (train/val_seen/val_unseen pkl symlink tree).
export DATA_PATH="${_SEEN18_REPO_ROOT}/outputs/eval/robocasa/groot_n16/safe_split_seen18_4unseen_100ep"

# --- Best aggregation picked by Step 3 (val_seen bal-acc): horizon=mean, diff=1.0 ---
export ROBOCASA_SAFE_FINAL_HORIZON_IDX_REL="mean"
export ROBOCASA_SAFE_FINAL_DIFF_IDX_REL="1.0"
export ROBOCASA_SAFE_FINAL_AGGREGATION_SLUG="hmean_d1"

echo "[seen18_env] RUN_ID=${ROBOCASA_SAFE_RUN_ID} SAFE_REPO=${SAFE_REPO} CONDA_ENV=${CONDA_ENV}"
echo "[seen18_env] DATA_PATH=${DATA_PATH}"
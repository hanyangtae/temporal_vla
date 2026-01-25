#!/bin/bash

# PC1 (Evaluation Machine) Script
# Usage: ./scripts/run_evaluation.sh [optional: task_name]

# Default arguments
CHECKPOINT_PATH="models/xvla-libero"
TASK_NAME=${1:-"open_the_middle_drawer_of_the_cabinet"}
EVAL_EPISODES=5
BATCH_SIZE=1
SEED=142

echo "Starting evaluation for task: $TASK_NAME"
echo "Using checkpoint: $CHECKPOINT_PATH"

# Ensure the checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint directory not found at $CHECKPOINT_PATH"
    echo "Please ensure you have transferred the checkpoint from the server."
    exit 1
fi

# Run evaluation script
# Note: Use --policy.path instead of --policy to properly load local checkpoints with overrides
python3 scripts/eval_xvla_libero.py \
    --policy.path="$CHECKPOINT_PATH" \
    --env.type=libero \
    --env.task=libero_goal \
    --env.control_mode=absolute \
    --eval.batch_size=$BATCH_SIZE \
    --eval.n_episodes=$EVAL_EPISODES \
    --env.episode_length=800 \
    --seed=$SEED \
    --eval.use_async_envs=false \
    --target_task_name="$TASK_NAME"

echo "Evaluation complete. Results saved to outputs/eval_results/"

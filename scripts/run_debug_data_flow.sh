#!/bin/bash

# Debug script to inspect data flow between LIBERO env and X-VLA policy
# Usage: ./scripts/run_debug_data_flow.sh [optional: task_name]

# Default arguments
CHECKPOINT_PATH="models/xvla-libero"
TASK_NAME=${1:-"open_the_middle_drawer_of_the_cabinet"}
SEED=142

echo "=========================================="
echo "🔍 Debugging data flow for task: $TASK_NAME"
echo "Using checkpoint: $CHECKPOINT_PATH"
echo "=========================================="

# Ensure the checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint directory not found at $CHECKPOINT_PATH"
    echo "Please ensure you have transferred the checkpoint from the server."
    exit 1
fi

# Run debug script
python3 scripts/debug_data_flow.py \
    --policy.path="$CHECKPOINT_PATH" \
    --env.type=libero \
    --env.task=libero_goal \
    --env.control_mode=absolute \
    --env.episode_length=800 \
    --seed=$SEED \
    --target_task_name="$TASK_NAME"

echo ""
echo "=========================================="
echo "✅ Debug complete!"
echo "=========================================="

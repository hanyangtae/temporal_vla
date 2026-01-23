#!/bin/bash

# 사용법: ./scripts/run_xvla_tuning.sh [mode]
# mode: head_only | soft_prompt | full

MODE=$1
DATASET_PATH="data/lerobot_libero_goal_time"
POLICY_PATH="models/xvla-libero"
OUTPUT_BASE="outputs/finetuned"
TIME_KEY="time_to_go_task"
DATASET_REPO_ID="lerobot_libero_goal_time"
DATASET_ROOT="/workspace/vla_tset/data/lerobot_libero_goal_time"

# 공통 파라미터
COMMON_ARGS="--policy.path=$POLICY_PATH \
  --dataset.repo_id=$DATASET_REPO_ID \
  --dataset.root=$DATASET_ROOT \
  --env.type=libero \
  --env.task=libero_goal \
  --policy.chunk_size=32 \
  --policy.n_action_steps=32 \
  --policy.time_feature_key=$TIME_KEY \
  --steps=20000 \
  --save_freq=5000 \
  --eval_freq=5000 \
  --eval.batch_size=1 \
  --eval.n_episodes=5 \
  --policy.push_to_hub=false" 

if [ "$MODE" == "head_only" ]; then
    echo "🚀 [1/3] Running Head Only Tuning..."
    echo "   - Backbone: Frozen"
    echo "   - Head: Trainable"
    
    python3 lerobot/src/lerobot/scripts/lerobot_train.py \
        $COMMON_ARGS \
        --output_dir="$OUTPUT_BASE/xvla_head_only" \
        --batch_size=8 \
        --policy.freeze_vision_encoder=true \
        --policy.freeze_language_encoder=true \
        --policy.train_policy_transformer=false \
        --policy.train_soft_prompts=false

elif [ "$MODE" == "soft_prompt" ]; then
    echo "🚀 [2/3] Running Soft Prompt & Policy Tuning..."
    echo "   - VLM: Frozen"
    echo "   - Transformer & Soft Prompts: Trainable"

    python3 lerobot/src/lerobot/scripts/lerobot_train.py \
        $COMMON_ARGS \
        --output_dir="$OUTPUT_BASE/xvla_soft_prompt" \
        --batch_size=8 \
        --policy.freeze_vision_encoder=true \
        --policy.freeze_language_encoder=true \
        --policy.train_policy_transformer=true \
        --policy.train_soft_prompts=true

elif [ "$MODE" == "full" ]; then
    echo "🚀 [3/3] Running Full Finetuning..."
    echo "   - All Parameters: Trainable"
    echo "   - Batch Size Reduced to 4 (Memory Safety)"

    python3 lerobot/src/lerobot/scripts/lerobot_train.py \
        $COMMON_ARGS \
        --output_dir="$OUTPUT_BASE/xvla_full" \
        --batch_size=4 \
        --policy.freeze_vision_encoder=false \
        --policy.freeze_language_encoder=false \
        --policy.train_policy_transformer=true \
        --policy.train_soft_prompts=true

else
    echo "Usage: $0 {head_only|soft_prompt|full}"
    echo "Example: $0 soft_prompt"
    exit 1
fi

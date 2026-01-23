#!/bin/bash
set -e

# 평가 설정
N_EPISODES=1
OUTPUT_DIR="outputs/eval_results"

# Task 목록 (ID 1개 + OOD 2개)
TASKS=(
    "open_the_middle_drawer_of_the_cabinet" 
    "push_the_plate_to_the_front_of_the_stove" 
    "put_the_wine_bottle_on_top_of_the_cabinet"
)

mkdir -p $OUTPUT_DIR

echo "=================================================="
echo "🚀 Starting Comprehensive Evaluation (8 Scenarios)"
echo "   Tasks: ${TASKS[@]}"
echo "   Episodes per task: $N_EPISODES"
echo "=================================================="

for TASK in "${TASKS[@]}"; do
    echo ""
    echo "##################################################"
    echo "📌 Evaluating Task: $TASK"
    echo "##################################################"
    
    mkdir -p "$OUTPUT_DIR/$TASK"

    # 1. Baseline
    if [ ! -f "$OUTPUT_DIR/$TASK/eval_results_baseline.json" ]; then
        echo "🧪 [$TASK] Evaluating Baseline..."
        python3 scripts/eval_subtask_based.py \
            --task_name $TASK \
            --policy.path lerobot/xvla-libero \
            --output_dir $OUTPUT_DIR \
            --subtask.baseline=true \
            --eval.n_episodes $N_EPISODES \
            --save_video=true \
            > "$OUTPUT_DIR/$TASK/eval_baseline.log" 2>&1
        echo "✅ Baseline Completed."
    else
        echo "⏩ Baseline Already Evaluated. Skipping..."
    fi

done

echo "🎉 All evaluations finished!"

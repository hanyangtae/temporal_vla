#!/bin/bash
set -e

# 평가 설정
N_EPISODES=20
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
            --task $TASK \
            --model_path lerobot/xvla-libero \
            --n_episodes $N_EPISODES \
            --output_dir $OUTPUT_DIR \
            --baseline \
            > "$OUTPUT_DIR/$TASK/eval_baseline.log" 2>&1
        echo "✅ Baseline Completed."
    else
        echo "⏩ Baseline Already Evaluated. Skipping..."
    fi

    # 2. Exp 1: Freeze + Subtask
    if [ ! -f "$OUTPUT_DIR/$TASK/eval_results_time_prediction_exp1.json" ]; then
        echo "🧪 [$TASK] Evaluating Exp 1 (Freeze + Subtask)..."
        python3 scripts/eval_subtask_based.py \
            --task $TASK \
            --model_path outputs/exp1_freeze_subtask/final \
            --n_episodes $N_EPISODES \
            --output_dir $OUTPUT_DIR \
            --switch_mode time_prediction \
            > "$OUTPUT_DIR/$TASK/eval_exp1.log" 2>&1
        mv "$OUTPUT_DIR/$TASK/eval_results_time_prediction.json" "$OUTPUT_DIR/$TASK/eval_results_time_prediction_exp1.json" || true
        echo "✅ Exp 1 Completed."
    else
        echo "⏩ Exp 1 Already Evaluated. Skipping..."
    fi

    # 3. Exp 2: Freeze + Task
    if [ ! -f "$OUTPUT_DIR/$TASK/eval_results_task_level_time_exp2.json" ]; then
        echo "🧪 [$TASK] Evaluating Exp 2 (Freeze + Task)..."
        python3 scripts/eval_subtask_based.py \
            --task $TASK \
            --model_path outputs/exp2_freeze_task/final \
            --n_episodes $N_EPISODES \
            --output_dir $OUTPUT_DIR \
            --task_level_time \
            > "$OUTPUT_DIR/$TASK/eval_exp2.log" 2>&1
        mv "$OUTPUT_DIR/$TASK/eval_results_task_level_time.json" "$OUTPUT_DIR/$TASK/eval_results_task_level_time_exp2.json" || true
        echo "✅ Exp 2 Completed."
    else
        echo "⏩ Exp 2 Already Evaluated. Skipping..."
    fi

    # 4. Exp 3: PEFT + Subtask
    if [ ! -f "$OUTPUT_DIR/$TASK/eval_results_time_prediction_exp3.json" ]; then
        echo "🧪 [$TASK] Evaluating Exp 3 (PEFT + Subtask)..."
        python3 scripts/eval_subtask_based.py \
            --task $TASK \
            --model_path outputs/exp3_peft_subtask/final \
            --n_episodes $N_EPISODES \
            --output_dir $OUTPUT_DIR \
            --switch_mode time_prediction \
            > "$OUTPUT_DIR/$TASK/eval_exp3.log" 2>&1
        mv "$OUTPUT_DIR/$TASK/eval_results_time_prediction.json" "$OUTPUT_DIR/$TASK/eval_results_time_prediction_exp3.json" || true
        echo "✅ Exp 3 Completed."
    else
        echo "⏩ Exp 3 Already Evaluated. Skipping..."
    fi

    # 5. Exp 4: PEFT + Task
    if [ ! -f "$OUTPUT_DIR/$TASK/eval_results_task_level_time_exp4.json" ]; then
        echo "🧪 [$TASK] Evaluating Exp 4 (PEFT + Task)..."
        python3 scripts/eval_subtask_based.py \
            --task $TASK \
            --model_path outputs/exp4_peft_task/final \
            --n_episodes $N_EPISODES \
            --output_dir $OUTPUT_DIR \
            --task_level_time \
            > "$OUTPUT_DIR/$TASK/eval_exp4.log" 2>&1
        mv "$OUTPUT_DIR/$TASK/eval_results_task_level_time.json" "$OUTPUT_DIR/$TASK/eval_results_task_level_time_exp4.json" || true
        echo "✅ Exp 4 Completed."
    else
        echo "⏩ Exp 4 Already Evaluated. Skipping..."
    fi

    # 6. Exp 5: Seq + Subtask
    if [ ! -f "$OUTPUT_DIR/$TASK/eval_results_time_prediction_exp5.json" ]; then
        echo "🧪 [$TASK] Evaluating Exp 5 (Seq + Subtask)..."
        python3 scripts/eval_subtask_based.py \
            --task $TASK \
            --model_path outputs/exp5_seq_subtask/final \
            --n_episodes $N_EPISODES \
            --output_dir $OUTPUT_DIR \
            --switch_mode time_prediction \
            > "$OUTPUT_DIR/$TASK/eval_exp5.log" 2>&1
        mv "$OUTPUT_DIR/$TASK/eval_results_time_prediction.json" "$OUTPUT_DIR/$TASK/eval_results_time_prediction_exp5.json" || true
        echo "✅ Exp 5 Completed."
    else
        echo "⏩ Exp 5 Already Evaluated. Skipping..."
    fi

    # 7. Exp 6: Seq + Task
    if [ ! -f "$OUTPUT_DIR/$TASK/eval_results_task_level_time_exp6.json" ]; then
        echo "🧪 [$TASK] Evaluating Exp 6 (Seq + Task)..."
        python3 scripts/eval_subtask_based.py \
            --task $TASK \
            --model_path outputs/exp6_seq_task/final \
            --n_episodes $N_EPISODES \
            --output_dir $OUTPUT_DIR \
            --task_level_time \
            > "$OUTPUT_DIR/$TASK/eval_exp6.log" 2>&1
        mv "$OUTPUT_DIR/$TASK/eval_results_task_level_time.json" "$OUTPUT_DIR/$TASK/eval_results_task_level_time_exp6.json" || true
        echo "✅ Exp 6 Completed."
    else
        echo "⏩ Exp 6 Already Evaluated. Skipping..."
    fi

    # 8. Exp 7: Full + Subtask
    if [ ! -f "$OUTPUT_DIR/$TASK/eval_results_time_prediction_exp7.json" ]; then
        echo "🧪 [$TASK] Evaluating Exp 7 (Full + Subtask)..."
        python3 scripts/eval_subtask_based.py \
            --task $TASK \
            --model_path outputs/exp7_full_subtask/final \
            --n_episodes $N_EPISODES \
            --output_dir $OUTPUT_DIR \
            --switch_mode time_prediction \
            > "$OUTPUT_DIR/$TASK/eval_exp7.log" 2>&1
        mv "$OUTPUT_DIR/$TASK/eval_results_time_prediction.json" "$OUTPUT_DIR/$TASK/eval_results_time_prediction_exp7.json" || true
        echo "✅ Exp 7 Completed."
    else
        echo "⏩ Exp 7 Already Evaluated. Skipping..."
    fi

    # 9. Exp 8: Full + Task
    if [ ! -f "$OUTPUT_DIR/$TASK/eval_results_task_level_time_exp8.json" ]; then
        echo "🧪 [$TASK] Evaluating Exp 8 (Full + Task)..."
        python3 scripts/eval_subtask_based.py \
            --task $TASK \
            --model_path outputs/exp8_full_task/final \
            --n_episodes $N_EPISODES \
            --output_dir $OUTPUT_DIR \
            --task_level_time \
            > "$OUTPUT_DIR/$TASK/eval_exp8.log" 2>&1
        mv "$OUTPUT_DIR/$TASK/eval_results_task_level_time.json" "$OUTPUT_DIR/$TASK/eval_results_task_level_time_exp8.json" || true
        echo "✅ Exp 8 Completed."
    else
        echo "⏩ Exp 8 Already Evaluated. Skipping..."
    fi
done

echo "🎉 All evaluations finished!"

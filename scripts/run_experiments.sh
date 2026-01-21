#!/bin/bash
set -e  # 에러 발생 시 즉시 중단

# 로그 디렉토리 생성
mkdir -p outputs/logs

# 공통 설정 (A100 기준 - 메모리 최적화)
DATASET_PATH="data/datasets/lerobot_libero_goal"

# LR=8e-5 (Effective Batch 512 기준)
LR=8e-5 
# Epoch 50
NUM_EPOCHS=50
# Save every 10 epochs
SAVE_EVERY=10
# Workers: 32
WORKERS=32

# ==============================================================================
# 1. Backbone Freeze (Head Only) + Subtask Unit
# ==============================================================================
BATCH_SIZE=512
GRAD_ACC=1
# Effective Batch Size: 512 * 1 = 512

if [ ! -f "outputs/exp1_freeze_subtask/final/config.json" ]; then
    echo "🚀 [1/8] Running Exp 1: Head Only, Subtask Unit"
    python3 scripts/finetune_xvla_time.py \
        --dataset_path $DATASET_PATH \
        --output_dir outputs/exp1_freeze_subtask \
        --target_time_key time_to_go_subtask \
        --freeze_backbone \
        --num_epochs $NUM_EPOCHS \
        --save_every $SAVE_EVERY \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation $GRAD_ACC \
        --learning_rate $LR \
        --num_workers $WORKERS \
        2>&1 | tee outputs/logs/exp1.log
else
    echo "✅ Exp 1 Already Completed. Skipping..."
fi

# ==============================================================================
# 2. Backbone Freeze (Head Only) + Task Unit
# ==============================================================================
BATCH_SIZE=512
GRAD_ACC=1
# Effective Batch Size: 512 * 1 = 512

if [ ! -f "outputs/exp2_freeze_task/final/config.json" ]; then
    echo "🚀 [2/8] Running Exp 2: Head Only, Task Unit"
    python3 scripts/finetune_xvla_time.py \
        --dataset_path $DATASET_PATH \
        --output_dir outputs/exp2_freeze_task \
        --target_time_key time_to_go_full \
        --freeze_backbone \
        --num_epochs $NUM_EPOCHS \
        --save_every $SAVE_EVERY \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation $GRAD_ACC \
        --learning_rate $LR \
        --num_workers $WORKERS \
        2>&1 | tee outputs/logs/exp2.log
else
    echo "✅ Exp 2 Already Completed. Skipping..."
fi

# ==============================================================================
# 3. PEFT (VLM Freeze) + Subtask Unit
# ==============================================================================
# Batch 128 -> OOM 발생 -> Batch 64로 감소
BATCH_SIZE=64
GRAD_ACC=8
# Effective Batch Size: 64 * 8 = 512

if [ ! -f "outputs/exp3_peft_subtask/final/config.json" ]; then
    echo "🚀 [3/8] Running Exp 3: PEFT (Policy+Prompt), Subtask Unit"
    python3 scripts/finetune_xvla_time.py \
        --dataset_path $DATASET_PATH \
        --output_dir outputs/exp3_peft_subtask \
        --target_time_key time_to_go_subtask \
        --use_peft \
        --num_epochs $NUM_EPOCHS \
        --save_every $SAVE_EVERY \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation $GRAD_ACC \
        --learning_rate $LR \
        --num_workers $WORKERS \
        2>&1 | tee outputs/logs/exp3.log
else
    echo "✅ Exp 3 Already Completed. Skipping..."
fi

# ==============================================================================
# 4. PEFT (VLM Freeze) + Task Unit
# ==============================================================================
BATCH_SIZE=64
GRAD_ACC=8
# Effective Batch Size: 64 * 8 = 512

if [ ! -f "outputs/exp4_peft_task/final/config.json" ]; then
    echo "🚀 [4/8] Running Exp 4: PEFT (Policy+Prompt), Task Unit"
    python3 scripts/finetune_xvla_time.py \
        --dataset_path $DATASET_PATH \
        --output_dir outputs/exp4_peft_task \
        --target_time_key time_to_go_full \
        --use_peft \
        --num_epochs $NUM_EPOCHS \
        --save_every $SAVE_EVERY \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation $GRAD_ACC \
        --learning_rate $LR \
        --num_workers $WORKERS \
        2>&1 | tee outputs/logs/exp4.log
else
    echo "✅ Exp 4 Already Completed. Skipping..."
fi

# ==============================================================================
# 5. Sequential (Head Only -> PEFT) + Subtask Unit
# ==============================================================================
# Stage 1: Head Only (20 Epochs) - Batch 512
# Stage 2: PEFT (30 Epochs) - Batch 64
if [ ! -f "outputs/exp5_seq_subtask/final/config.json" ]; then
    echo "🚀 [5/8] Running Exp 5: Seq (Head->PEFT), Subtask Unit"
    
    # Stage 1: Head Only Training (Epoch 1-20)
    if [ ! -d "outputs/exp5_seq_subtask/stage1" ]; then
        echo "   🔸 Stage 1: Head Only Training (Epoch 1-20)..."
        BATCH_SIZE=512
        GRAD_ACC=1
        
        python3 scripts/finetune_xvla_time.py \
            --dataset_path $DATASET_PATH \
            --output_dir outputs/exp5_seq_subtask/stage1 \
            --target_time_key time_to_go_subtask \
            --freeze_backbone \
            --num_epochs 20 \
            --save_every 10 \
            --batch_size $BATCH_SIZE \
            --gradient_accumulation $GRAD_ACC \
            --learning_rate $LR \
            --num_workers $WORKERS \
            2>&1 | tee outputs/logs/exp5_stage1.log
    else
        echo "   ✅ Stage 1 Already Completed. Skipping..."
    fi

    # Stage 2: PEFT Training
    echo "   🔸 Stage 2: PEFT Training (Epoch 21-50)..."
    BATCH_SIZE=64
    GRAD_ACC=8
    # LR Decay (1/5)
    
    python3 scripts/finetune_xvla_time.py \
        --dataset_path $DATASET_PATH \
        --output_dir outputs/exp5_seq_subtask \
        --resume_from_checkpoint outputs/exp5_seq_subtask/stage1/final \
        --target_time_key time_to_go_subtask \
        --use_peft \
        --num_epochs 50 \
        --save_every $SAVE_EVERY \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation $GRAD_ACC \
        --learning_rate 2e-5 \
        --num_workers $WORKERS \
        2>&1 | tee outputs/logs/exp5_stage2.log
else
    echo "✅ Exp 5 Already Completed. Skipping..."
fi

# ==============================================================================
# 6. Sequential (Head Only -> PEFT) + Task Unit
# ==============================================================================
# Stage 1: Head Only (20 Epochs) - Batch 512
# Stage 2: PEFT (30 Epochs) - Batch 64
if [ ! -f "outputs/exp6_seq_task/final/config.json" ]; then
    echo "🚀 [6/8] Running Exp 6: Seq (Head->PEFT), Task Unit"
    
    # Stage 1: Head Only Initialization
    if [ ! -d "outputs/exp6_seq_task/stage1" ]; then
        echo "   🔸 Stage 1: Head Only Training (Epoch 1-20)..."
        BATCH_SIZE=512
        GRAD_ACC=1
        
        python3 scripts/finetune_xvla_time.py \
            --dataset_path $DATASET_PATH \
            --output_dir outputs/exp6_seq_task/stage1 \
            --target_time_key time_to_go_full \
            --freeze_backbone \
            --num_epochs 20 \
            --save_every 10 \
            --batch_size $BATCH_SIZE \
            --gradient_accumulation $GRAD_ACC \
            --learning_rate $LR \
            --num_workers $WORKERS \
            2>&1 | tee outputs/logs/exp6_stage1.log
    else
        echo "   ✅ Stage 1 Already Completed. Skipping..."
    fi

    # Stage 2: PEFT Training
    echo "   🔸 Stage 2: PEFT Training (Epoch 21-50)..."
    BATCH_SIZE=64
    GRAD_ACC=8
    
    python3 scripts/finetune_xvla_time.py \
        --dataset_path $DATASET_PATH \
        --output_dir outputs/exp6_seq_task \
        --resume_from_checkpoint outputs/exp6_seq_task/stage1/final \
        --target_time_key time_to_go_full \
        --use_peft \
        --num_epochs 50 \
        --save_every $SAVE_EVERY \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation $GRAD_ACC \
        --learning_rate 2e-5 \
        --num_workers $WORKERS \
        2>&1 | tee outputs/logs/exp6_stage2.log
else
    echo "✅ Exp 6 Already Completed. Skipping..."
fi

# ==============================================================================
# 7. Full Finetune + Subtask Unit
# ==============================================================================
# Full Finetune은 메모리를 매우 많이 사용하므로 Batch를 대폭 줄이고 Grad Acc를 늘림
BATCH_SIZE=32
GRAD_ACC=16
# Effective Batch Size: 32 * 16 = 512

if [ ! -f "outputs/exp7_full_subtask/final/config.json" ]; then
    echo "🚀 [7/8] Running Exp 7: Full Finetune, Subtask Unit"
    python3 scripts/finetune_xvla_time.py \
        --dataset_path $DATASET_PATH \
        --output_dir outputs/exp7_full_subtask \
        --target_time_key time_to_go_subtask \
        --num_epochs $NUM_EPOCHS \
        --save_every $SAVE_EVERY \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation $GRAD_ACC \
        --learning_rate 2e-5 \
        --num_workers $WORKERS \
        2>&1 | tee outputs/logs/exp7.log
else
    echo "✅ Exp 7 Already Completed. Skipping..."
fi

# ==============================================================================
# 8. Full Finetune + Task Unit
# ==============================================================================
BATCH_SIZE=32
GRAD_ACC=16
# Effective Batch Size: 32 * 16 = 512

if [ ! -f "outputs/exp8_full_task/final/config.json" ]; then
    echo "🚀 [8/8] Running Exp 8: Full Finetune, Task Unit"
    python3 scripts/finetune_xvla_time.py \
        --dataset_path $DATASET_PATH \
        --output_dir outputs/exp8_full_task \
        --target_time_key time_to_go_full \
        --num_epochs $NUM_EPOCHS \
        --save_every $SAVE_EVERY \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation $GRAD_ACC \
        --learning_rate 2e-5 \
        --num_workers $WORKERS \
        2>&1 | tee outputs/logs/exp8.log
else
    echo "✅ Exp 8 Already Completed. Skipping..."
fi

echo "🎉 All 8 experiments finished successfully!"

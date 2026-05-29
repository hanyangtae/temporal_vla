#!/bin/bash
# [DEPRECATED] Phase 1 Window 모드 학습 + 평가 파이프라인
# phase1_step50k 재현 시도: window 기반, task 필터 없음, 50k steps
# 결과: Spearman이 episodic(full traj) 대비 낮아 deprecated
set -e

cd "$(dirname "$0")/.."

# ── 설정 (wandb 원본 config 기반) ──
DATASET_MODE="window"
INNER_MODEL="mlp"
TOTAL_STEPS=50000
BATCH_SIZE=32
TRAIN_EPISODES=2986
VAL_EPISODES=287
VAL_INTERVAL=1000
SAVE_INTERVAL=10000
SAVE_SUFFIX="window_${INNER_MODEL}"

DATE_STR=$(date +%Y%m%d_%H%M)
EVAL_DIR="outputs/eval_results/phase1_window_${DATE_STR}"

echo "=========================================="
echo " Phase 1 Window Training (step50k 재현)"
echo " steps: ${TOTAL_STEPS}, no_task_filter"
echo "=========================================="

# ── 1. 학습 ──
python scripts/train/phase1_predictor.py \
    --dataset_mode ${DATASET_MODE} \
    --no_task_filter \
    --inner_model_type ${INNER_MODEL} \
    --total_steps ${TOTAL_STEPS} \
    --batch_size ${BATCH_SIZE} \
    --train_episodes ${TRAIN_EPISODES} \
    --val_episodes ${VAL_EPISODES} \
    --val_interval ${VAL_INTERVAL} \
    --save_interval ${SAVE_INTERVAL} \
    --save_dir /cache/checkpoints/phase1 \
    --save_suffix ${SAVE_SUFFIX} \
    --wandb_run_name "phase1_window_nofilter_${INNER_MODEL}_${TOTAL_STEPS}"

LATEST_CKPT_DIR=$(ls -dt /cache/checkpoints/phase1/*_${SAVE_SUFFIX} 2>/dev/null | head -1)
FINAL_CKPT="${LATEST_CKPT_DIR}/phase1_final.pt"

if [ ! -f "${FINAL_CKPT}" ]; then
    echo "ERROR: ${FINAL_CKPT} not found"
    exit 1
fi
echo "Checkpoint: ${FINAL_CKPT}"

# ── 2. 평가 (n_samples=100, n_videos=5) ──
python scripts/eval/phase1_predictor.py \
    --ckpt "${FINAL_CKPT}" \
    --inner_model_type ${INNER_MODEL} \
    --train_episodes ${TRAIN_EPISODES} \
    --val_episodes ${VAL_EPISODES} \
    --n_samples 100 \
    --save_dir "${EVAL_DIR}" \
    --save_video \
    --n_videos 5 \
    --save_json

echo ""
echo "=========================================="
echo " Done!"
echo " Checkpoint: ${FINAL_CKPT}"
echo " Results:    ${EVAL_DIR}/metrics.txt"
echo "=========================================="
cat "${EVAL_DIR}/metrics.txt"

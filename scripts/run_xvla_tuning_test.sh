#!/bin/bash

# =============================================================================
# X-VLA Fine-tuning Script for LIBERO (Optimized for A100 80GB)
# =============================================================================
# 사용법: ./scripts/run_xvla_tuning.sh [mode]
# mode: head_only | soft_prompt | full
#
# 이 스크립트는 Docker 컨테이너 내부에서 실행됩니다.
# =============================================================================

set -e  # Exit on error

MODE=$1

# =============================================================================
# 경로 설정 (Docker 컨테이너 내부 절대 경로)
# =============================================================================
WORKSPACE="/workspace/vla_tset"
POLICY_PATH="${WORKSPACE}/models/xvla-libero"
OUTPUT_BASE="${WORKSPACE}/outputs/finetuned/test"
# NOTE: 데이터셋은 10D action 형식으로 변환된 버전 사용
# 변환 스크립트: scripts/convert_libero_to_lerobot.py
# 시간 정보 추가: scripts/add_time_features.py
DATASET_REPO_ID="lerobot_libero_goal_10d_time"
DATASET_ROOT="${WORKSPACE}/data/lerobot_libero_goal_10d_time"
TIME_KEY="time_to_go_task"

# =============================================================================
# 하이퍼파라미터 설정 (A100 80GB 최적화)
# =============================================================================
# Linear Scaling Rule: batch_size 2배 → LR 2배
# 기본 LR: 1e-4, batch 8
# A100 80GB: 메모리 여유가 많으므로 배치 크게 증가 가능

# Data loading 최적화
NUM_WORKERS=16               # A100 서버는 CPU 코어 많음 → 병렬화 증가

# Batch sizes (GPU 메모리 ~60-70GB 사용 목표)
BATCH_HEAD_ONLY=128          # Frozen backbone → 대폭 증가 가능
BATCH_SOFT_PROMPT=64         # Frozen VLM → 크게 증가 가능
BATCH_FULL=32                # 전체 학습 → 적당히 증가

# Learning rates (Linear Scaling Rule 적용: batch 8배 → LR 8배)
LR_HEAD_ONLY=1.6e-3          # batch 128 / 16 = 8x → LR 8x (2e-4 * 8)
LR_SOFT_PROMPT=8e-4          # batch 64 / 16 = 4x → LR 4x (2e-4 * 4)
LR_FULL=4e-4                 # batch 32 / 8 = 4x → LR 4x (1e-4 * 4)

# =============================================================================
# Evaluation 설정
# =============================================================================
EVAL_FREQ=1              # 2500 steps마다 eval (0이면 비활성화)
EVAL_BATCH_SIZE=1       # eval 시 병렬 환경 수 (메모리 고려)
EVAL_N_EPISODES=1       # eval 시 에피소드 수

# =============================================================================
# 공통 파라미터
# =============================================================================
# NOTE: action_mode="ee6d_10d"를 사용하여 10D EE6D 데이터에 적합한 loss 계산
# - Position: MSE * 500 (높은 가중치로 정확한 위치 학습)
# - Rotation: MSE * 10 (적당한 가중치로 방향 학습)
# - Gripper: BCE * 1 (이진 분류로 그리퍼 열림/닫힘 학습)
# 모델 출력: 20D → trim → 10D → XVLARotation6DToAxisAngleProcessorStep → 7D (환경 입력)
COMMON_ARGS="--policy.path=${POLICY_PATH} \
  --dataset.repo_id=${DATASET_REPO_ID} \
  --dataset.root=${DATASET_ROOT} \
  --dataset.video_backend=pyav \
  --num_workers=${NUM_WORKERS} \
  --env.type=libero \
  --env.task=libero_goal \
  --env.control_mode=absolute \
  --env.episode_length=800 \
  --policy.chunk_size=32 \
  --policy.n_action_steps=32 \
  --policy.action_mode=auto \
  --policy.time_feature_key=${TIME_KEY} \
  --steps=1 \
  --save_freq=1 \
  --eval_freq=${EVAL_FREQ} \
  --eval.batch_size=${EVAL_BATCH_SIZE} \
  --eval.n_episodes=${EVAL_N_EPISODES} \
  --log_freq=10 \
  --policy.push_to_hub=false" 

# =============================================================================
# 실행 모드별 설정
# =============================================================================

if [ "$MODE" == "head_only" ]; then
    echo "=============================================="
    echo "🚀 [1/3] Running Head Only Tuning (Optimized)"
    echo "   - Vision Encoder: Frozen"
    echo "   - Language Encoder: Frozen"
    echo "   - Policy Transformer: Frozen"
    echo "   - Soft Prompts: Frozen"
    echo "   - Action/Time Heads: Trainable"
    echo "   - Batch Size: ${BATCH_HEAD_ONLY}"
    echo "   - Learning Rate: ${LR_HEAD_ONLY}"
    echo "   - Num Workers: ${NUM_WORKERS}"
    echo "=============================================="
    
    python3 ${WORKSPACE}/scripts/train_xvla_libero.py \
        ${COMMON_ARGS} \
        --output_dir="${OUTPUT_BASE}/xvla_head_only" \
        --batch_size=${BATCH_HEAD_ONLY} \
        --policy.optimizer_lr=${LR_HEAD_ONLY} \
        --policy.freeze_vision_encoder=true \
        --policy.freeze_language_encoder=true \
        --policy.train_policy_transformer=false \
        --policy.train_soft_prompts=false

elif [ "$MODE" == "soft_prompt" ]; then
    echo "=============================================="
    echo "🚀 [2/3] Running Soft Prompt & Policy Tuning (Optimized)"
    echo "   - Vision Encoder: Frozen"
    echo "   - Language Encoder: Frozen"  
    echo "   - Policy Transformer: Trainable"
    echo "   - Soft Prompts: Trainable"
    echo "   - Batch Size: ${BATCH_SOFT_PROMPT}"
    echo "   - Learning Rate: ${LR_SOFT_PROMPT}"
    echo "   - Num Workers: ${NUM_WORKERS}"
    echo "=============================================="

    python3 ${WORKSPACE}/scripts/train_xvla_libero.py \
        ${COMMON_ARGS} \
        --output_dir="${OUTPUT_BASE}/xvla_soft_prompt" \
        --batch_size=${BATCH_SOFT_PROMPT} \
        --policy.optimizer_lr=${LR_SOFT_PROMPT} \
        --policy.freeze_vision_encoder=true \
        --policy.freeze_language_encoder=true \
        --policy.train_policy_transformer=true \
        --policy.train_soft_prompts=true

elif [ "$MODE" == "full" ]; then
    echo "=============================================="
    echo "🚀 [3/3] Running Full Finetuning (Optimized)"
    echo "   - All Parameters: Trainable"
    echo "   - Batch Size: ${BATCH_FULL}"
    echo "   - Learning Rate: ${LR_FULL}"
    echo "   - Num Workers: ${NUM_WORKERS}"
    echo "=============================================="

    python3 ${WORKSPACE}/scripts/train_xvla_libero.py \
        ${COMMON_ARGS} \
        --output_dir="${OUTPUT_BASE}/xvla_full" \
        --batch_size=${BATCH_FULL} \
        --policy.optimizer_lr=${LR_FULL} \
        --policy.freeze_vision_encoder=false \
        --policy.freeze_language_encoder=false \
        --policy.train_policy_transformer=true \
        --policy.train_soft_prompts=true

else
    echo "=============================================="
    echo "Usage: $0 {head_only|soft_prompt|full}"
    echo ""
    echo "Modes:"
    echo "  head_only   - Train only action/time heads (fastest)"
    echo "                Batch: ${BATCH_HEAD_ONLY}, LR: ${LR_HEAD_ONLY}"
    echo "  soft_prompt - Train policy transformer + soft prompts"
    echo "                Batch: ${BATCH_SOFT_PROMPT}, LR: ${LR_SOFT_PROMPT}"
    echo "  full        - Train all parameters (slowest)"
    echo "                Batch: ${BATCH_FULL}, LR: ${LR_FULL}"
    echo ""
    echo "Example: $0 soft_prompt"
    echo "=============================================="
    exit 1
fi

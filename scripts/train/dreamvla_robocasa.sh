#!/bin/bash
# DreamVLA finetuning on RoboCasa (PickPlaceSinkToCounter)
# pretrain + human 데이터로 학습, target 데이터는 eval 전용
#
# Prerequisites:
#   - Calvin pretrained checkpoint
#   - RoboCasa LeRobot dataset (pretrain + human, v3.0)
#   - Extracted SAM features + CoTracker trajectory labels
#   - DreamVLA auxiliary checkpoints (co-tracker, sam, dinov2)

set -euo pipefail

# ---------- paths (adjust to your environment) ----------
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ROBOCASA_DATASET="${ROBOCASA_DATASET:-${REPO_ROOT}/src/benchmarks/robocasa/datasets/v1.0/pretrain/atomic/PickPlaceSinkToCounter/20250819/lerobot}"
SAVE_CHECKPOINT_PATH="${REPO_ROOT}/checkpoints/dreamvla_robocasa"
VIT_CHECKPOINT_PATH="${REPO_ROOT}/checkpoints/dreamvla/mae_pretrain_vit_base.pth"

# Auxiliary feature paths (extracted by scripts/extract/extract_sam_robocasa.py / extract_cotrack_robocasa.py)
SAM_FEATURE_PATH="${SAM_FEATURE_PATH:-${REPO_ROOT}/src/benchmarks/robocasa/datasets/features/sam}"
TRACK_LABEL_PATH="${TRACK_LABEL_PATH:-${REPO_ROOT}/src/benchmarks/robocasa/datasets/features/tracks}"

# DreamVLA pretrained checkpoint (Calvin pretrained)
PRETRAINED_CKPT="${PRETRAINED_CKPT:-${REPO_ROOT}/checkpoints/dreamvla/dreamvla_dynamic_depth_semantic-001.pth}"

# ---------- distributed ----------
NODE=1
NPROC=${NPROC:-2}
MASTER_PORT=${MASTER_PORT:-10216}

# ---------- training ----------
torchrun \
    --nnodes=${NODE} \
    --nproc_per_node=${NPROC} \
    --master_port=${MASTER_PORT} \
    "${REPO_ROOT}/scripts/train/dreamvla_robocasa.py" \
    --robocasa_dataset "${ROBOCASA_DATASET}" \
    --vit_checkpoint_path "${VIT_CHECKPOINT_PATH}" \
    --save_checkpoint_path "${SAVE_CHECKPOINT_PATH}" \
    --finetune_type "calvin" \
    --gripper_width \
    --traj_cons \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --gradient_accumulation_steps 1 \
    --bf16_module "vision_encoder" \
    --workers 12 \
    --lr_scheduler cosine \
    --num_epochs 50 \
    --seed 42 \
    --batch_size 12 \
    --precision fp32 \
    --learning_rate 3.5e-4 \
    --weight_decay 1e-4 \
    --warmup_epochs 2 \
    --num_resampler_query 16 \
    --num_obs_token_per_image 9 \
    --run_name "finetune_dreamvla_robocasa_pickplace" \
    --save_checkpoint \
    --save_checkpoint_seq 5 \
    --start_save_checkpoint 0 \
    --transformer_layers 24 \
    --hidden_dim 1024 \
    --transformer_heads 16 \
    --phase "finetune" \
    --action_pred_steps 3 \
    --sequence_length 10 \
    --future_steps 3 \
    --window_size 13 \
    --multi_step_action 3 \
    --loss_action \
    --obs_pred \
    --loss_image \
    --use_dit_head \
    --sam_feat_pred \
    --loss_sam_feat \
    --load_sam_features \
    --sam_features_path "${SAM_FEATURE_PATH}" \
    --trajectory_pred \
    --loss_trajectory \
    --load_track_labels \
    --track_label_path "${TRACK_LABEL_PATH}" \
    --track_label_patch_size 8 \
    --flow_as_mask \
    --attn_implementation "sdpa" \
    --report_to_wandb \
    ${PRETRAINED_CKPT:+--finetune_from_pretrained_ckpt "${PRETRAINED_CKPT}"} \
    ${PRETRAINED_CKPT:+--reset_obs_token} \
    ${PRETRAINED_CKPT:+--reset_action_decoder} \
    "$@"

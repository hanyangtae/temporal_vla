#!/bin/bash
# Phase 1: MLP vs Linear 비교 실험
# - 둘 다 10 epoch 학습
# - epoch 4~10에 대해 evaluate
#
# 실행:
#   docker exec -it lerobot bash /temporal_vla/scripts/train/eval_phase1_mlp_linear.sh

set -e
cd /temporal_vla

EMBED_CACHE="/cache/datasets/bridge_v2_pick_place_clip_embeddings.pt"

# ============================================
# 0. Pick-and-place 임베딩 캐시 생성 (없으면)
# ============================================
if [ ! -f "$EMBED_CACHE" ]; then
    echo "============================================"
    echo "Building CLIP embedding cache for pick-and-place episodes"
    echo "============================================"
    python scripts/analysis/build_clip_cache.py \
        --save_path "$EMBED_CACHE" \
        --task_keywords put place pick \
        --max_episodes 5000 \
        --device cuda
fi

TRAIN_COMMON="
    --data_root      /cache/datasets/bridge_v2_lerobot
    --repo_id        FedorX8/bridge_v2_lerobot
    --embed_cache_path $EMBED_CACHE
    --task_keywords  put place pick
    --train_episodes 2986
    --val_episodes   287
    --split_seed     42
    --input_dim      1024
    --proj_dim       64
    --head_hidden_dim 128
    --eta_base       0.1
    --n_epochs       10
    --batch_size     32
    --lr             1e-4
    --lambda_self    0.5
    --log_interval   10
    --num_workers    0
    --device         cuda
    --save_dir       /cache/checkpoints/phase1
"

EVAL_COMMON="
    --data_root      /cache/datasets/bridge_v2_lerobot
    --repo_id        FedorX8/bridge_v2_lerobot
    --embed_cache    $EMBED_CACHE
    --task_keywords  put place pick
    --train_episodes 2986
    --val_episodes   287
    --split_seed     42
    --n_samples      100
    --device         cuda
    --input_dim      1024
    --proj_dim       64
    --head_hidden_dim 128
    --eta_base       0.1
    --n_examples     5
    --save_video
    --n_videos       10
    --fps            10
"

EVAL_EPOCHS="004 005 006 007 008 009 010"

# ============================================
# 1. MLP 학습 (10 epochs)
# ============================================
echo "============================================"
echo "Training MLP (10 epochs)"
echo "============================================"
python scripts/train/phase1_predictor.py \
    --inner_model_type mlp \
    --save_suffix mlp \
    --wandb_run_name phase1_vita_full_traj_mlp \
    $TRAIN_COMMON

# 학습 완료 후 가장 최근 생성된 mlp 폴더 찾기
CKPT_MLP=$(ls -td /cache/checkpoints/phase1/*_mlp | head -1)
echo "MLP checkpoint dir: ${CKPT_MLP}"

# ============================================
# 2. Linear 학습 (10 epochs)
# ============================================
echo "============================================"
echo "Training Linear (10 epochs)"
echo "============================================"
python scripts/train/phase1_predictor.py \
    --inner_model_type linear \
    --save_suffix linear \
    --wandb_run_name phase1_vita_full_traj_linear \
    $TRAIN_COMMON

CKPT_LINEAR=$(ls -td /cache/checkpoints/phase1/*_linear | head -1)
echo "Linear checkpoint dir: ${CKPT_LINEAR}"

# ============================================
# 3. Evaluate MLP (epoch 4~10)
# ============================================
for EPOCH in $EVAL_EPOCHS; do
    echo "============================================"
    echo "Evaluating MLP epoch ${EPOCH}"
    echo "============================================"
    python scripts/eval/phase1_predictor.py \
        --ckpt "${CKPT_MLP}/epoch_${EPOCH}.pt" \
        --inner_model_type mlp \
        --save_dir "outputs/eval_results/phase1/mlp_ep${EPOCH}" \
        $EVAL_COMMON
done

# ============================================
# 4. Evaluate Linear (epoch 4~10)
# ============================================
for EPOCH in $EVAL_EPOCHS; do
    echo "============================================"
    echo "Evaluating Linear epoch ${EPOCH}"
    echo "============================================"
    python scripts/eval/phase1_predictor.py \
        --ckpt "${CKPT_LINEAR}/epoch_${EPOCH}.pt" \
        --inner_model_type linear \
        --save_dir "outputs/eval_results/phase1/linear_ep${EPOCH}" \
        $EVAL_COMMON
done

echo ""
echo "========== All training & evaluation complete =========="
echo "MLP checkpoints:    ${CKPT_MLP}"
echo "Linear checkpoints: ${CKPT_LINEAR}"
echo "Eval results:       outputs/eval_results/phase1/"

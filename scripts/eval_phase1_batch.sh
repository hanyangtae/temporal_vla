#!/bin/bash
# Phase 1 일괄 평가: linear(0403_0457) vs mlp(0403_0801) × epoch 5/15/30/50
# 영상: split당 10개, n_samples=100
#
# 실행:
#   docker exec -it lerobot bash /temporal_vla/scripts/eval_phase1_batch.sh

set -e
cd /temporal_vla

# CKPT_LINEAR="checkpoints/phase1/20260403_0457"
CKPT_MLP="checkpoints/phase1/20260403_0910"

EPOCHS="005 015 030 050"

COMMON_ARGS="
    --data_root     data/bridge_v2_lerobot
    --repo_id       FedorX8/bridge_v2_lerobot
    --embed_cache   data/bridge_v2_lerobot_clip_embeddings.pt
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

for EPOCH in $EPOCHS; do
    # echo "============================================"
    # echo "Evaluating linear epoch ${EPOCH}"
    # echo "============================================"
    # python scripts/eval_phase1_predictor.py \
    #     --ckpt "${CKPT_LINEAR}/epoch_${EPOCH}.pt" \
    #     --inner_model_type linear \
    #     --save_dir "outputs/eval_results/phase1/linear_ep${EPOCH}" \
    #     $COMMON_ARGS

    echo "============================================"
    echo "Evaluating mlp epoch ${EPOCH}"
    echo "============================================"
    python scripts/eval_phase1_predictor.py \
        --ckpt "${CKPT_MLP}/epoch_${EPOCH}.pt" \
        --inner_model_type mlp \
        --save_dir "outputs/eval_results/phase1/2mlp_ep${EPOCH}" \
        $COMMON_ARGS
done

echo ""
echo "========== All evaluations complete =========="
echo "Results saved under outputs/eval_results/phase1/"

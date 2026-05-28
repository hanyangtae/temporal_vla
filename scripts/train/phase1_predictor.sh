#!/bin/bash
# Phase 1 ProgressPredictor 학습 — VITA D.1 세팅 (full trajectory)
#
# train: 2,986 에피소드 / val: 287 에피소드 (BridgeData V2)
# n_epochs: 5  (≈ 94 steps/epoch × 5 ≈ 470 total steps)
# 에피소드 전체(≤120프레임) 순차 TTT → train/inference mismatch 없음
#
# 실행:
#   docker exec -it lerobot bash /temporal_vla/scripts/train/phase1_predictor.sh

set -e
cd /temporal_vla

python scripts/train/phase1_predictor.py \
    --data_root     /cache/datasets/bridge_v2_lerobot \
    --repo_id       FedorX8/bridge_v2_lerobot \
    --image_key     observation.images.primary \
    --window_size   8 \
    --max_windows_per_episode 8 \
    --train_episodes 2986 \
    --val_episodes   287 \
    --embed_cache_path /cache/datasets/bridge_v2_lerobot_clip_embeddings.pt \
    \
    --input_dim     1024 \
    --proj_dim      64 \
    --inner_model_type mlp \
    --head_hidden_dim  128 \
    --eta_base      0.1 \
    \
    --n_epochs      50 \
    --batch_size    32 \
    --lr            1e-4 \
    --lambda_self   0.5 \
    --val_steps     50 \
    --log_interval  10 \
    --num_workers   0 \
    --embed_device  cuda \
    \
    --device        cuda \
    --save_dir      /cache/checkpoints/phase1 \
    --wandb_project temporal-vla \
    --wandb_run_name phase1_vita_full_traj_w8_k8_5ep

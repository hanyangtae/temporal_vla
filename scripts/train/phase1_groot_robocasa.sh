#!/bin/bash
# Phase 1 (GR00T × RoboCasa) ProgressPredictor 학습.
#
# Stage 0 (Eagle pre-LLM 캐시) 가 완료된 뒤 실행. Stage 2 에서 frozen 으로 사용
# 되므로 input_dim/proj_dim=2048 (= Eagle Qwen3-1.7B hidden = DiT KV dim) 고정
# — wrapper 에서 별도 projection 없이 DiT cross-attn KV 에 직접 concat 하기 위해.
#
# 실행:
#   docker compose exec lerobot bash -lc \
#       'bash /temporal_vla/scripts/train/phase1_groot_robocasa.sh'

set -e
cd /temporal_vla

python scripts/train/phase1_groot_robocasa.py \
    --data_root  data/robocasa/v1.0/pretrain/atomic \
    --cache_root data/robocasa_eagle_pre_llm \
    --tasks OpenDrawer CloseDrawer OpenCabinet CloseCabinet \
            OpenFridge CloseFridge OpenMicrowave CloseMicrowave \
            PickPlaceCounterToStove PickPlaceCounterToSink \
    --window_size 8 \
    --max_windows_per_episode 8 \
    --train_frac 0.9 \
    \
    --input_dim 2048 \
    --proj_dim  2048 \
    --inner_model_type linear \
    --head_hidden_dim 128 \
    --eta_base 0.1 \
    \
    --n_epochs 5 \
    --batch_size 32 \
    --lr 1e-4 \
    --lambda_self 0.5 \
    --val_steps 50 \
    --log_interval 50 \
    --num_workers 0 \
    \
    --device cuda \
    --save_dir outputs/train/phase1_groot_robocasa \
    --wandb_project temporal-vla \
    --wandb_run_name phase1_groot_robocasa_10tasks

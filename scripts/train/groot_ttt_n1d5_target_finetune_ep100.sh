#!/usr/bin/env bash
# GR00T **N1.5** + TTT Phase 2 finetune on RoboCasa target/atomic 15 task,
# **episode_index < 100** subset.
#
# TTT 모듈 (Phase 1 predictor) 은 episode<200 으로 학습된 ckpt 그대로 사용.
# Phase 2 (VLA finetune) 만 episode<100 data 로 학습.
#
# 기존 sh (``groot_ttt_n1d5_target_finetune.sh``, ep<200) 와 분리:
# - output_dir 분리 (``outputs/groot_ttt_n1d5_target15_ep100``)
# - MAX_EPISODES_PER_TASK=100
#
# 실행:
#   docker exec \
#       -e TTT_PREDICTOR_PATH=... -e TTT_INNER_MODEL=... \
#       groot_extract bash /temporal_vla/scripts/train/groot_ttt_n1d5_target_finetune_ep100.sh
#
# 옵션 env var:
#   TTT_PREDICTOR_PATH    Phase 1 결과 ckpt
#   TTT_INNER_MODEL       linear 또는 linear_preLN
#   OUTPUT_DIR            default outputs/groot_ttt_n1d5_target15_ep100
#   MAX_STEPS=20000
#   SAVE_STEPS=5000
#   MAX_EPISODES_PER_TASK=100

set -euo pipefail
cd /temporal_vla

BASE_MODEL_PATH="${BASE_MODEL_PATH:-/temporal_vla/checkpoints/nvidia/GR00T-N1.5-3B}"
TTT_PREDICTOR_PATH="${TTT_PREDICTOR_PATH:?TTT_PREDICTOR_PATH must be set (Phase 1 ckpt)}"
TTT_EAGLE_CACHE_ROOT="${TTT_EAGLE_CACHE_ROOT:-/temporal_vla/data/robocasa_eagle_pre_llm_target_n1d5}"
TTT_INNER_MODEL="${TTT_INNER_MODEL:-linear}"
TTT_UPDATE_IN_TRAIN="${TTT_UPDATE_IN_TRAIN:-True}"

OUTPUT_DIR="${OUTPUT_DIR:-/temporal_vla/outputs/groot_ttt_n1d5_target15_ep100}"
# Defaults — GPU2 의 vanilla N1.5 finetune (gr00t_n15_finetune_subset.py) 와 일치.
#   batch_size 64, max_steps 5895 (~1 epoch on 15 tasks × 100 ep, batch 64),
#   save_steps 3000, num_workers 12. lr/wd/warmup 은 공식 N1.5 default.
MAX_STEPS="${MAX_STEPS:-5895}"
SAVE_STEPS="${SAVE_STEPS:-3000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-12}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
MAX_EPISODES_PER_TASK="${MAX_EPISODES_PER_TASK:-100}"

# target/atomic 15 task — Phase 1 학습한 task list 와 동일
TARGET_ROOT="/temporal_vla/data/robocasa/v1.0/target/atomic"
DATASETS=(
  "${TARGET_ROOT}/CloseFridge/20250816/lerobot"
  "${TARGET_ROOT}/CloseToasterOvenDoor/20250818/lerobot"
  "${TARGET_ROOT}/CoffeeSetupMug/20250813/lerobot"
  "${TARGET_ROOT}/OpenCabinet/20250813/lerobot"
  "${TARGET_ROOT}/OpenDrawer/20250816/lerobot"
  "${TARGET_ROOT}/PickPlaceCounterToCabinet/20250811/lerobot"
  "${TARGET_ROOT}/PickPlaceCounterToStove/20250818/lerobot"
  "${TARGET_ROOT}/PickPlaceDrawerToCounter/20250820/lerobot"
  "${TARGET_ROOT}/PickPlaceSinkToCounter/20250813/lerobot"
  "${TARGET_ROOT}/PickPlaceToasterToCounter/20250817/lerobot"
  "${TARGET_ROOT}/SlideDishwasherRack/20250820/lerobot"
  "${TARGET_ROOT}/TurnOffStove/20250812/lerobot"
  "${TARGET_ROOT}/TurnOnElectricKettle/20250817/lerobot"
  "${TARGET_ROOT}/TurnOnMicrowave/20250813/lerobot"
  "${TARGET_ROOT}/TurnOnSinkFaucet/20250812/lerobot"
)

echo "=========================================="
echo "GR00T N1.5 + TTT finetune (target/atomic 15 task, **episode<${MAX_EPISODES_PER_TASK}**)"
echo "  base_model:            $BASE_MODEL_PATH"
echo "  ttt_predictor_path:    $TTT_PREDICTOR_PATH"
echo "  ttt_eagle_cache_root:  $TTT_EAGLE_CACHE_ROOT"
echo "  ttt_inner_model:       $TTT_INNER_MODEL"
echo "  output_dir:            $OUTPUT_DIR"
echo "  steps:                 $MAX_STEPS  (save every $SAVE_STEPS, keep $SAVE_TOTAL_LIMIT)"
echo "  batch_size:            $BATCH_SIZE"
echo "  max_eps_per_task:      $MAX_EPISODES_PER_TASK"
echo "=========================================="

exec python scripts/train/launch_finetune_ttt_n1d5.py \
    --base-model-path "$BASE_MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --dataset-path "${DATASETS[@]}" \
    --ttt-predictor-path "$TTT_PREDICTOR_PATH" \
    --ttt-eagle-cache-root "$TTT_EAGLE_CACHE_ROOT" \
    --ttt-predictor-input-dim 2048 \
    --ttt-predictor-proj-dim 1536 \
    --ttt-predictor-inner-model "$TTT_INNER_MODEL" \
    --ttt-update-in-train \
    --batch-size "$BATCH_SIZE" \
    --max-steps "$MAX_STEPS" \
    --save-steps "$SAVE_STEPS" \
    --save-total-limit "$SAVE_TOTAL_LIMIT" \
    --learning-rate "$LEARNING_RATE" \
    --weight-decay "$WEIGHT_DECAY" \
    --warmup-ratio "$WARMUP_RATIO" \
    --dataloader-num-workers "$DATALOADER_NUM_WORKERS" \
    --max-episodes-per-task "$MAX_EPISODES_PER_TASK"

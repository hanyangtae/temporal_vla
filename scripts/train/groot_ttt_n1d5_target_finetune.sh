#!/usr/bin/env bash
# GR00T **N1.5** + TTT fine-tuning on RoboCasa target/atomic 15 task (episode<200).
#
# 별도 launcher (``launch_finetune_ttt_n1d5.py``) 호출 — N1.5 codebase + TTT 통합.
# N1.6 의 ``groot_ttt_robocasa_finetune.sh`` 와 무관.
#
# 실행:
#   docker exec groot_extract bash /temporal_vla/scripts/train/groot_ttt_n1d5_target_finetune.sh
#   (또는 GPU 1 컨테이너 어디든)
#
# 옵션 env var:
#   TTT_PREDICTOR_PATH    Phase 1 결과 ckpt (linear / linear_preLN 둘 중 우세한 거)
#   TTT_INNER_MODEL       linear 또는 linear_preLN
#   OUTPUT_DIR
#   MAX_STEPS=20000
#   SAVE_STEPS=5000

set -euo pipefail
cd /temporal_vla

BASE_MODEL_PATH="${BASE_MODEL_PATH:-/temporal_vla/checkpoints/nvidia/GR00T-N1.5-3B}"
TTT_PREDICTOR_PATH="${TTT_PREDICTOR_PATH:?TTT_PREDICTOR_PATH must be set (Phase 1 ckpt)}"
TTT_EAGLE_CACHE_ROOT="${TTT_EAGLE_CACHE_ROOT:-/temporal_vla/data/robocasa_eagle_pre_llm_target_n1d5}"
TTT_INNER_MODEL="${TTT_INNER_MODEL:-linear}"
TTT_UPDATE_IN_TRAIN="${TTT_UPDATE_IN_TRAIN:-True}"

OUTPUT_DIR="${OUTPUT_DIR:-/temporal_vla/outputs/groot_ttt_n1d5_target15}"
MAX_STEPS="${MAX_STEPS:-20000}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-4}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"

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
echo "GR00T N1.5 + TTT finetune (target/atomic 15 task)"
echo "  base_model:            $BASE_MODEL_PATH"
echo "  ttt_predictor_path:    $TTT_PREDICTOR_PATH"
echo "  ttt_eagle_cache_root:  $TTT_EAGLE_CACHE_ROOT"
echo "  ttt_inner_model:       $TTT_INNER_MODEL"
echo "  output_dir:            $OUTPUT_DIR"
echo "  steps:                 $MAX_STEPS  (save every $SAVE_STEPS, keep $SAVE_TOTAL_LIMIT)"
echo "  batch_size:            $BATCH_SIZE"
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
    --max-episodes-per-task 200

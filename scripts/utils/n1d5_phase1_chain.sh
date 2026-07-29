#!/usr/bin/env bash
# N1.5 Phase 1 자동 chain — 추출 끝 대기 → Phase 1 두 case 학습 → eval.
# 터미널 꺼져도 nohup 으로 detach 시켜두면 host 의 daemon 으로 계속 진행.
#
# Phase 2 (finetune) 는 case 선택 (linear vs linear_preLN) 이 필요하므로 chain 에 포함 X.
# eval 결과 보고 사용자 결정 후 별도 시작.
#
# 사용 (터미널 끄기 전):
#   nohup bash /temporal_vla/scripts/utils/n1d5_phase1_chain.sh \
#     > /home/junhyeong/pkt_ws/temporal_vla/outputs/n1d5_chain.log 2>&1 &
#   echo "CHAIN_PID=$!"
#
# 다시 터미널 켜면 진행 확인:
#   tail -f outputs/n1d5_chain.log
#   ls outputs/train/phase1_groot_robocasa/*_target15_n1d5_*  # ckpt dir 보임
#   ls outputs/eval/phase1_n1d5/                              # eval 결과
set -u

CACHE_DIR=/home/junhyeong/pkt_ws/temporal_vla/data/robocasa_eagle_pre_llm_target_n1d5
TIMEOUT_S=$((8 * 3600))   # 추출 8h 안에 끝 가정

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
log()   { echo "[$(stamp)] $*"; }

# ─── 1. 추출 끝 wait ───────────────────────────────────────────────
log "Step 1: waiting for extraction (15 task caches)"
START=$(date +%s)
while :; do
  N=$(ls "$CACHE_DIR"/*/embeddings.pt 2>/dev/null | wc -l)
  ELAPSED=$(($(date +%s) - START))
  log "  $N/15 caches  elapsed=${ELAPSED}s"
  if [ "$N" -ge 15 ]; then
    log "  Extraction complete."
    break
  fi
  if [ "$ELAPSED" -gt "$TIMEOUT_S" ]; then
    log "  TIMEOUT — extraction not done after ${TIMEOUT_S}s. Aborting."
    exit 1
  fi
  sleep 300
done

# GPU 2 의 groot_extract_g2 컨테이너 정리 (auto-rm 라 process 끝나면 자동 제거되지만 확실히)
log "Step 1b: cleanup groot_extract_g2 (GPU 2)"
docker stop groot_extract_g2 2>&1 | head -1 || true
sleep 5

# ─── 2. Phase 1 학습 case A (linear) ──────────────────────────────
log "Step 2a: Phase 1 training — linear"
INNER_MODEL_TYPE=linear SAVE_SUFFIX=target15_n1d5 \
  docker exec lerobot bash /temporal_vla/scripts/train/phase1_groot_robocasa_n1d5.sh \
  || { log "  Phase 1 (linear) FAILED"; exit 2; }
log "  Phase 1 (linear) done"

# ─── 3. Phase 1 학습 case B (linear_preLN) ────────────────────────
log "Step 2b: Phase 1 training — linear_preLN"
INNER_MODEL_TYPE=linear_preLN SAVE_SUFFIX=target15_n1d5 \
  docker exec lerobot bash /temporal_vla/scripts/train/phase1_groot_robocasa_n1d5.sh \
  || { log "  Phase 1 (linear_preLN) FAILED"; exit 3; }
log "  Phase 1 (linear_preLN) done"

# ─── 4. Phase 1 eval ──────────────────────────────────────────────
log "Step 3: Phase 1 eval (compare two ckpts)"
LINEAR_DIR=$(ls -td /home/junhyeong/pkt_ws/temporal_vla/outputs/train/phase1_groot_robocasa/*_target15_n1d5_linear 2>/dev/null | head -1)
PRELN_DIR=$(ls -td /home/junhyeong/pkt_ws/temporal_vla/outputs/train/phase1_groot_robocasa/*_target15_n1d5_linear_preLN 2>/dev/null | head -1)
log "  LINEAR_DIR=$LINEAR_DIR"
log "  PRELN_DIR=$PRELN_DIR"

LINEAR_CKPT="$LINEAR_DIR/epoch_08.pt"
PRELN_CKPT="$PRELN_DIR/epoch_08.pt"
if [ ! -f "$LINEAR_CKPT" ] || [ ! -f "$PRELN_CKPT" ]; then
  log "  ckpt missing (linear=$LINEAR_CKPT preln=$PRELN_CKPT)"
  exit 4
fi

EVAL_DIR=/home/junhyeong/pkt_ws/temporal_vla/outputs/eval/phase1_n1d5
docker exec lerobot bash -lc "cd /temporal_vla && python scripts/eval/phase1_predictor.py \
  --ckpts ${LINEAR_CKPT/#\/home\/junhyeong\/pkt_ws\/temporal_vla\//} ${PRELN_CKPT/#\/home\/junhyeong\/pkt_ws\/temporal_vla\//} \
  --ckpt_labels linear_n1d5 linear_preLN_n1d5 \
  --inner_model_types linear linear_preLN \
  --data_root data/robocasa/v1.0/target/atomic \
  --cache_root data/robocasa_eagle_pre_llm_target_n1d5 \
  --tasks CloseFridge CloseToasterOvenDoor CoffeeSetupMug OpenCabinet OpenDrawer \
          PickPlaceCounterToCabinet PickPlaceCounterToStove PickPlaceDrawerToCounter \
          PickPlaceSinkToCounter PickPlaceToasterToCounter SlideDishwasherRack \
          TurnOffStove TurnOnElectricKettle TurnOnMicrowave TurnOnSinkFaucet \
  --input_dim 2048 --proj_dim 1536 \
  --train_batch_size 32 --n_samples 50 \
  --save_dir ${EVAL_DIR/#\/home\/junhyeong\/pkt_ws\/temporal_vla\//}" \
  || { log "  eval FAILED"; exit 5; }
log "  Phase 1 eval done. Results: $EVAL_DIR/metrics.txt"

# ─── 5. Phase 2 결정용 hint 출력 ───────────────────────────────────
log "Step 4: Phase 1 metrics summary"
cat "$EVAL_DIR/metrics.txt" 2>/dev/null || true

# ─── 5. Phase 2 finetune (default linear_preLN — 사용자 미개입 시) ──────────
# 사용자가 다시 터미널 켜서 다른 case 원하면 chain process kill + 직접 시작:
#   pkill -f n1d5_phase1_chain.sh                                # chain 끔
#   pkill -f launch_finetune_ttt_n1d5                            # 진행 중 Phase 2 끔
#   TTT_PREDICTOR_PATH=<linear ckpt> TTT_INNER_MODEL=linear \\
#     docker exec groot_extract bash .../groot_ttt_n1d5_target_finetune.sh
log "Step 5: Phase 2 finetune (default: linear_preLN)"
log "  predictor=$PRELN_CKPT"
TTT_PREDICTOR_PATH="$PRELN_CKPT" TTT_INNER_MODEL=linear_preLN \
  docker exec groot_extract bash /temporal_vla/scripts/train/groot_ttt_n1d5_target_finetune.sh \
  || { log "  Phase 2 FAILED"; exit 6; }

log "=========================================="
log "CHAIN ALL DONE. Output:"
log "  Phase 1 linear ckpt:       $LINEAR_CKPT"
log "  Phase 1 linear_preLN ckpt: $PRELN_CKPT"
log "  Phase 1 eval:              $EVAL_DIR/metrics.txt"
log "  Phase 2 finetune output:   /home/junhyeong/pkt_ws/temporal_vla/outputs/groot_ttt_n1d5_target15"
log "=========================================="

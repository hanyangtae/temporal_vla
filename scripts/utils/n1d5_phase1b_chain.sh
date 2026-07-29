#!/usr/bin/env bash
# N1.5 Phase 1 후속 chain — linear_preLN 학습 → eval → Phase 2 finetune.
# 기존 chain (n1d5_phase1_chain.sh) 이 docker exec env 미통과 버그로 linear_preLN
# 안 돌고 죽어서, 이걸로 이어 진행.
#
# 사용:
#   nohup bash /temporal_vla/scripts/utils/n1d5_phase1b_chain.sh \
#     > /home/junhyeong/pkt_ws/temporal_vla/outputs/n1d5_chain.log 2>&1 &
#   disown
set -u

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
log()   { echo "[$(stamp)] $*"; }

LINEAR_DIR=/home/junhyeong/pkt_ws/temporal_vla/outputs/train/phase1_groot_robocasa/20260512_2240_target15_n1d5_linear
LINEAR_CKPT="$LINEAR_DIR/epoch_08.pt"

# ─── 1. Phase 1 linear_preLN 학습 (env var 를 `docker exec -e` 로 명시) ──
log "Step 2b: Phase 1 training — linear_preLN"
docker exec \
    -e INNER_MODEL_TYPE=linear_preLN \
    -e SAVE_SUFFIX=target15_n1d5 \
    lerobot bash /temporal_vla/scripts/train/phase1_groot_robocasa_n1d5.sh \
    || { log "  Phase 1 (linear_preLN) FAILED"; exit 1; }
log "  Phase 1 (linear_preLN) done"

# ─── 2. preln ckpt 디렉토리 찾기 ─────────────────────────────────
PRELN_DIR=$(ls -td /home/junhyeong/pkt_ws/temporal_vla/outputs/train/phase1_groot_robocasa/*_target15_n1d5_linear_preLN 2>/dev/null | head -1)
log "PRELN_DIR=$PRELN_DIR"
PRELN_CKPT="$PRELN_DIR/epoch_08.pt"
if [ ! -f "$PRELN_CKPT" ] || [ ! -f "$LINEAR_CKPT" ]; then
  log "  ckpt missing (linear=$LINEAR_CKPT preln=$PRELN_CKPT)"
  exit 2
fi

# ─── 3. Phase 1 eval ──────────────────────────────────────────────
log "Step 3: Phase 1 eval (compare two ckpts)"
EVAL_DIR=outputs/eval/phase1_n1d5
LINEAR_REL=${LINEAR_CKPT#/home/junhyeong/pkt_ws/temporal_vla/}
PRELN_REL=${PRELN_CKPT#/home/junhyeong/pkt_ws/temporal_vla/}
docker exec lerobot bash -lc "cd /temporal_vla && python scripts/eval/phase1_predictor.py \
  --ckpts $LINEAR_REL $PRELN_REL \
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
  --save_dir $EVAL_DIR" \
  || { log "  eval FAILED"; exit 3; }
log "  Phase 1 eval done — see /home/junhyeong/pkt_ws/temporal_vla/$EVAL_DIR/metrics.txt"
cat /home/junhyeong/pkt_ws/temporal_vla/$EVAL_DIR/metrics.txt 2>/dev/null || true

# ─── 4. Phase 2 자동 진행 차단 ────────────────────────────────────
# 사용자가 Phase 1 eval 결과 본 후 ckpt 직접 결정하기로 함.
# Phase 2 (episode<100 학습) 는 별도 sh 로 detach 실행:
#   scripts/train/groot_ttt_n1d5_target_finetune_ep100.sh
# 자동 진행 부분 비활성화. eval 결과 확인 후 수동 launch.
log "=========================================="
log "CHAIN-B STOPPED AFTER PHASE 1 EVAL (Phase 2 disabled)"
log "  Phase 1 linear ckpt:       $LINEAR_CKPT"
log "  Phase 1 linear_preLN ckpt: $PRELN_CKPT"
log "  Phase 1 eval:              $EVAL_DIR/metrics.txt"
log "  → 사용자: metrics.txt 확인 후 Phase 2 (ep<100) 수동 실행"
log "=========================================="
exit 0

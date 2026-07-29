#!/usr/bin/env bash
# Phase 2 (N1.5 + TTT, episode<100) detached launcher.
#
# Phase 1 eval 결과 본 후 user 가 ckpt 결정하면 이걸로 finetune 을 백그라운드 detach.
# 터미널/세션 끊겨도 학습 계속됨 (setsid + nohup → host process detach,
# docker exec child 는 docker daemon 이 관리).
#
# 사용:
#   bash scripts/utils/launch_phase2_n1d5_ep100.sh \
#     <predictor_path> <inner_model> [container_name]
#
#   예 (linear_preLN 우세, default 컨테이너):
#     bash scripts/utils/launch_phase2_n1d5_ep100.sh \
#       /temporal_vla/outputs/train/phase1_groot_robocasa/20260513_0041_target15_n1d5_linear_preLN/epoch_08.pt \
#       linear_preLN
#
# 옵션 env var (그대로 통과):
#   OUTPUT_DIR   default /temporal_vla/outputs/groot_ttt_n1d5_target15_ep100
#   MAX_STEPS=20000
#   SAVE_STEPS=5000
#   BATCH_SIZE=32
#   MAX_EPISODES_PER_TASK=100
#
# 진행 확인:
#   tail -f <LOG_PATH>
#   ls outputs/groot_ttt_n1d5_target15_ep100/   # 5000 step 마다 ckpt
#   wandb run name = output_dir basename

set -u

PREDICTOR_PATH="${1:?predictor_path required}"
INNER_MODEL="${2:?inner_model required (linear or linear_preLN)}"
CONTAINER="${3:-groot_extract}"

OUTPUT_DIR="${OUTPUT_DIR:-/temporal_vla/outputs/groot_ttt_n1d5_target15_ep100}"
# Defaults — GPU 2 vanilla N1.5 finetune 과 동일 (gr00t_n15_finetune_subset.py).
MAX_STEPS="${MAX_STEPS:-5895}"
SAVE_STEPS="${SAVE_STEPS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-12}"
MAX_EPISODES_PER_TASK="${MAX_EPISODES_PER_TASK:-100}"

stamp=$(date '+%Y%m%d_%H%M%S')
LOG_DIR=/home/junhyeong/pkt_ws/temporal_vla/outputs
mkdir -p "$LOG_DIR"
LOG_PATH="$LOG_DIR/groot_ttt_n1d5_ep100_${stamp}.log"

echo "=========================================="
echo "Phase 2 N1.5 + TTT (episode<${MAX_EPISODES_PER_TASK}) detached launch"
echo "  container:             $CONTAINER"
echo "  predictor:             $PREDICTOR_PATH"
echo "  inner_model:           $INNER_MODEL"
echo "  output_dir:            $OUTPUT_DIR"
echo "  max_steps:             $MAX_STEPS  (save every $SAVE_STEPS)"
echo "  batch_size:            $BATCH_SIZE"
echo "  max_episodes_per_task: $MAX_EPISODES_PER_TASK"
echo "  log:                   $LOG_PATH"
echo "=========================================="

# Container 확인
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
  echo "[ERR] container '$CONTAINER' not running"
  echo "      docker ps 로 GPU1 컨테이너 띄워두기 (핸드오프 §6 참고)"
  exit 1
fi

# Predictor 존재 (host 경로 변환)
HOST_PREDICTOR="${PREDICTOR_PATH/#\/temporal_vla\//\/home\/junhyeong\/pkt_ws\/temporal_vla\/}"
if [ ! -f "$HOST_PREDICTOR" ]; then
  echo "[ERR] predictor file not found: $HOST_PREDICTOR"
  exit 1
fi

# WANDB_API_KEY: host env 또는 lerobot 컨테이너에서 fetch (Phase 1 학습 컨테이너에 있음).
WANDB_API_KEY="${WANDB_API_KEY:-$(docker exec lerobot printenv WANDB_API_KEY 2>/dev/null || true)}"
if [ -z "$WANDB_API_KEY" ]; then
  echo "[warn] WANDB_API_KEY 없음 — wandb 비활성 (report_to=none) 로 학습 시작"
  REPORT_TO_FLAG="-e REPORT_TO=none"
else
  echo "[ok] WANDB_API_KEY found (len=${#WANDB_API_KEY}) — wandb 로깅 활성"
  REPORT_TO_FLAG=""
fi

# setsid + nohup + disown = detach (host shell 끊겨도 host wrapper 살아남음)
# docker exec 의 학습 process 는 docker daemon 이 관리 → 더욱 안전.
setsid nohup docker exec \
    -e TTT_PREDICTOR_PATH="$PREDICTOR_PATH" \
    -e TTT_INNER_MODEL="$INNER_MODEL" \
    -e OUTPUT_DIR="$OUTPUT_DIR" \
    -e MAX_STEPS="$MAX_STEPS" \
    -e SAVE_STEPS="$SAVE_STEPS" \
    -e BATCH_SIZE="$BATCH_SIZE" \
    -e DATALOADER_NUM_WORKERS="$DATALOADER_NUM_WORKERS" \
    -e MAX_EPISODES_PER_TASK="$MAX_EPISODES_PER_TASK" \
    -e WANDB_API_KEY="$WANDB_API_KEY" \
    $REPORT_TO_FLAG \
    "$CONTAINER" bash /temporal_vla/scripts/train/groot_ttt_n1d5_target_finetune_ep100.sh \
    > "$LOG_PATH" 2>&1 < /dev/null &

CHILD_PID=$!
disown "$CHILD_PID" 2>/dev/null || true

echo ""
echo "[ok] launched detached. host wrapper PID=$CHILD_PID"
echo "[ok] tail log:"
echo "       tail -f $LOG_PATH"
echo "[ok] 학습 process 는 docker daemon 이 관리하므로 터미널 끊겨도 학습 계속됨"
echo ""
echo "10초 후 wrapper 상태 확인 (CTRL-C 해도 무관):"
sleep 10
if ps -p "$CHILD_PID" > /dev/null 2>&1; then
  ps -p "$CHILD_PID" -o pid,sid,stat,etime,command 2>&1 | head -3
  echo "[ok] wrapper alive."
else
  echo "[warn] wrapper exited within 10s — log 확인:"
  echo "  tail -30 $LOG_PATH"
  tail -30 "$LOG_PATH" 2>&1 || true
fi

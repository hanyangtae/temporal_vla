#!/usr/bin/env bash
# 10 task 의 Eagle pre-LLM 추출을 task 별 별도 process 로 병렬 실행.
#
# 각 process 가 자기 Eagle backbone (~6.5 GB GPU mem) 을 로드해서 1 task 처리.
# LLM-skip 적용된 forward 라 GPU 1장에서 10 process queue 해도 throughput 손해 적음.
#
# 사용법:
#   docker compose exec groot bash /temporal_vla/scripts/extract/extract_eagle_parallel.sh
#
# 옵션 환경변수:
#   NUM_PROCS=10       동시 process 수 (default 10, task 수 == 10)
#   BATCH=32           per-process Eagle batch
#   CPU_WORKERS=4      per-process processor threads
#   LOG_DIR=/temporal_vla/outputs/eagle_logs

set -e

cd /temporal_vla

NUM_PROCS="${NUM_PROCS:-10}"
BATCH="${BATCH:-32}"
CPU_WORKERS="${CPU_WORKERS:-4}"
LOG_DIR="${LOG_DIR:-/temporal_vla/outputs/eagle_logs}"

TASKS=(
  OpenDrawer CloseDrawer
  OpenCabinet CloseCabinet
  OpenFridge CloseFridge
  OpenMicrowave CloseMicrowave
  PickPlaceCounterToStove PickPlaceCounterToSink
)

mkdir -p "$LOG_DIR"
ts=$(date +%Y%m%d_%H%M%S)
echo "[parallel] $NUM_PROCS procs × $(( ${#TASKS[@]} / NUM_PROCS + (${#TASKS[@]} % NUM_PROCS != 0) )) tasks each, logs → $LOG_DIR/${ts}_*.log"

pids=()
for i in "${!TASKS[@]}"; do
  task="${TASKS[$i]}"
  log="$LOG_DIR/${ts}_${task}.log"
  echo "[spawn] $task → $log"
  python scripts/extract/extract_eagle_pre_llm_robocasa.py \
    --batch_size "$BATCH" \
    --cpu_workers "$CPU_WORKERS" \
    --tasks "$task" \
    > "$log" 2>&1 &
  pids+=($!)
  # 5 process 단위 stagger — 동시 모델 로드 peak 완화
  if (( (i + 1) % 5 == 0 && i + 1 < ${#TASKS[@]} )); then
    sleep 5
  fi
done

echo "[parallel] PIDs: ${pids[@]}"
echo "[parallel] waiting for all to finish ..."
wait
echo "[parallel] done"

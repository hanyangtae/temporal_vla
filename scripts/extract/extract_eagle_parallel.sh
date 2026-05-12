#!/usr/bin/env bash
# RoboCasa atomic task 들의 Eagle pre-LLM 추출을 task queue 방식으로 병렬 실행.
#
# 각 process = 1 task. Eagle backbone (~6.5-13 GB GPU mem) 로드 후 LLM-skip forward.
# 한 task 끝나면 즉시 다음 task spawn (wave wait 없음) — 작은 task / 큰 task 가 섞여 있을
# 때 GPU idle 시간 최소화. extract_eagle_pre_llm_robocasa.py 가 cache 이미 있으면 skip.
#
# 사용법:
#   docker compose exec groot bash /temporal_vla/scripts/extract/extract_eagle_parallel.sh
#   # 또는 새로 받은 task 만:
#   TASKS_OVERRIDE="OpenOven OpenDishwasher ..." bash .../extract_eagle_parallel.sh
#
# 옵션 환경변수:
#   MAX_CONCURRENT=5   동시 process 수 (default 5 → 5 × ~8GB = ~40GB GPU mem peak with batch=64)
#                       (이전 이름 WAVE_SIZE 도 호환)
#   BATCH=32           per-process Eagle forward batch (크면 GPU 효율 ↑)
#   CPU_WORKERS=4      per-process processor threads
#   LOG_DIR=/temporal_vla/outputs/eagle_logs
#   TASKS_OVERRIDE     공백 구분 task list (지정 시 default 30 list 무시)

set -e

cd /temporal_vla

# MAX_CONCURRENT (또는 backwards-compat 으로 WAVE_SIZE) — 동시 실행 process 수
MAX_CONCURRENT="${MAX_CONCURRENT:-${WAVE_SIZE:-5}}"
BATCH="${BATCH:-32}"
CPU_WORKERS="${CPU_WORKERS:-4}"
LOG_DIR="${LOG_DIR:-/temporal_vla/outputs/eagle_logs}"
DATA_ROOT="${DATA_ROOT:-/temporal_vla/data/robocasa/v1.0/pretrain/atomic}"
SAVE_PATH="${SAVE_PATH:-/temporal_vla/data/robocasa_eagle_pre_llm}"
MAX_EPISODES="${MAX_EPISODES:-}"   # 빈 값이면 전체 episode 추출

# default — 30 task. 이미 cache 있는 기존 10 도 list 에 있지만 extract script 가 skip.
DEFAULT_TASKS=(
  # 기존 10
  OpenDrawer CloseDrawer
  OpenCabinet CloseCabinet
  OpenFridge CloseFridge
  OpenMicrowave CloseMicrowave
  PickPlaceCounterToStove PickPlaceCounterToSink
  # 신규 20
  OpenOven OpenDishwasher OpenFridgeDrawer OpenToasterOvenDoor
  OpenBlenderLid OpenElectricKettleLid OpenStandMixerHead
  PickPlaceCounterToCabinet PickPlaceCabinetToCounter PickPlaceCounterToDrawer
  PickPlaceCounterToMicrowave PickPlaceCounterToOven
  PickPlaceSinkToCounter PickPlaceStoveToCounter
  TurnOnSinkFaucet TurnOnStove TurnOnMicrowave
  TurnOnBlender TurnOnElectricKettle TurnOnToaster
)

if [ -n "${TASKS_OVERRIDE:-}" ]; then
  # shellcheck disable=SC2206
  TASKS=( ${TASKS_OVERRIDE} )
else
  TASKS=( "${DEFAULT_TASKS[@]}" )
fi

mkdir -p "$LOG_DIR"
ts=$(date +%Y%m%d_%H%M%S)
n_total=${#TASKS[@]}
echo "[parallel] tasks=$n_total  max_concurrent=$MAX_CONCURRENT  batch=$BATCH"
echo "[parallel] logs → $LOG_DIR/${ts}_*.log"

# Task queue 방식: 한 task 끝나면 즉시 다음 spawn. wave wait 없음 → 작은/큰 task 섞여
# 있을 때 GPU idle 최소화.
spawn_task() {
  local task="$1"
  local log="$LOG_DIR/${ts}_${task}.log"
  echo "  [spawn] $task → $log"
  local extra_args=()
  if [ -n "${MAX_EPISODES}" ]; then
    extra_args+=(--max_episodes "${MAX_EPISODES}")
  fi
  python scripts/extract/extract_eagle_pre_llm_robocasa.py \
    --data_root "$DATA_ROOT" \
    --save_path "$SAVE_PATH" \
    --batch_size "$BATCH" \
    --cpu_workers "$CPU_WORKERS" \
    --tasks "$task" \
    "${extra_args[@]}" \
    > "$log" 2>&1
}
export -f spawn_task
export DATA_ROOT SAVE_PATH BATCH CPU_WORKERS LOG_DIR ts MAX_EPISODES

# xargs -P 로 동시 process 수 제한하면서 task queue 처리.
# 한 task 끝날 때마다 새 task spawn → GPU 가 항상 N process 로 가득찬 상태 유지.
printf '%s\n' "${TASKS[@]}" | xargs -n 1 -P "$MAX_CONCURRENT" -I {} bash -c 'spawn_task "$@"' _ {}

echo "[parallel] all waves done"

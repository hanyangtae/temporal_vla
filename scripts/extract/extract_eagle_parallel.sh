#!/usr/bin/env bash
# RoboCasa atomic task 들의 Eagle pre-LLM 추출을 wave 단위 병렬 실행.
#
# 각 process = 1 task. Eagle backbone (~6.5 GB GPU mem) 로드 후 LLM-skip forward.
# WAVE_SIZE 만큼 동시 실행 → 끝나면 다음 wave 시작 (한 GPU 의 메모리 한계 보호).
# extract_eagle_pre_llm_robocasa.py 가 cache 이미 있으면 skip 하므로 30 task 전체 list
# 해도 새 task 만 실제 추출.
#
# 사용법:
#   docker compose exec groot bash /temporal_vla/scripts/extract/extract_eagle_parallel.sh
#   # 또는 새로 받은 task 만:
#   TASKS_OVERRIDE="OpenOven OpenDishwasher ..." bash .../extract_eagle_parallel.sh
#
# 옵션 환경변수:
#   WAVE_SIZE=5        wave 당 동시 process (default 5 → 5 × 6.5GB = ~33GB GPU mem peak)
#   BATCH=32           per-process Eagle batch
#   CPU_WORKERS=4      per-process processor threads
#   LOG_DIR=/temporal_vla/outputs/eagle_logs
#   TASKS_OVERRIDE     공백 구분 task list (지정 시 default 30 list 무시)

set -e

cd /temporal_vla

WAVE_SIZE="${WAVE_SIZE:-5}"
BATCH="${BATCH:-32}"
CPU_WORKERS="${CPU_WORKERS:-4}"
LOG_DIR="${LOG_DIR:-/temporal_vla/outputs/eagle_logs}"

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
n_waves=$(( (n_total + WAVE_SIZE - 1) / WAVE_SIZE ))
echo "[parallel] tasks=$n_total  wave_size=$WAVE_SIZE  → $n_waves waves, logs → $LOG_DIR/${ts}_*.log"

# wave 단위 spawn + wait
for ((s=0; s<n_total; s+=WAVE_SIZE)); do
  end=$(( s + WAVE_SIZE ))
  (( end > n_total )) && end=$n_total
  echo "[wave $(( s / WAVE_SIZE + 1 ))/$n_waves] tasks $s..$((end-1))"
  pids=()
  for ((i=s; i<end; i++)); do
    task="${TASKS[$i]}"
    log="$LOG_DIR/${ts}_${task}.log"
    echo "  [spawn] $task → $log"
    python scripts/extract/extract_eagle_pre_llm_robocasa.py \
      --batch_size "$BATCH" \
      --cpu_workers "$CPU_WORKERS" \
      --tasks "$task" \
      > "$log" 2>&1 &
    pids+=($!)
  done
  # wave 끝까지 대기 (GPU 메모리 보호)
  for pid in "${pids[@]}"; do
    wait "$pid" || echo "  [warn] pid=$pid failed (continuing)"
  done
done

echo "[parallel] all waves done"

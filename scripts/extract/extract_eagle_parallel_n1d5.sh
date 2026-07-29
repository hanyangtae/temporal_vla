#!/usr/bin/env bash
# GR00T **N1.5** Eagle pre-LLM 추출 — RoboCasa atomic per-task, task queue 병렬.
#
# N1.6 의 ``extract_eagle_parallel.sh`` 와 별개 — N1.5 codebase 사용한
# ``extract_eagle_pre_llm_robocasa_n1d5.py`` 를 호출. PYTHONPATH 조작 / 별도
# processor / pytorch3d stub 등 N1.5 specific 처리가 그 python script 안에서 다 됨.
#
# 사용법:
#   docker exec groot_extract bash /temporal_vla/scripts/extract/extract_eagle_parallel_n1d5.sh
#
# 옵션 환경변수:
#   MAX_CONCURRENT=4   동시 process 수 (default 4 — GPU SM 100% saturation 도달).
#   BATCH=16           Eagle forward batch (N1.5 의 batched forward 효과 작지만 메모리는 ↑).
#   MAX_EPISODES=200   task 별 episode 상한 (앞에서부터 N개).
#   LOG_DIR=/temporal_vla/outputs/eagle_logs
#   DATA_ROOT=/temporal_vla/data/robocasa/v1.0/target/atomic
#   SAVE_PATH=/temporal_vla/data/robocasa_eagle_pre_llm_target_n1d5
#   MODEL_PATH=/temporal_vla/checkpoints/nvidia/GR00T-N1.5-3B
#   TASKS_OVERRIDE     공백 구분 task list

set -e
cd /temporal_vla

MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
BATCH="${BATCH:-16}"
LOG_DIR="${LOG_DIR:-/temporal_vla/outputs/eagle_logs}"
DATA_ROOT="${DATA_ROOT:-/temporal_vla/data/robocasa/v1.0/target/atomic}"
SAVE_PATH="${SAVE_PATH:-/temporal_vla/data/robocasa_eagle_pre_llm_target_n1d5}"
MODEL_PATH="${MODEL_PATH:-/temporal_vla/checkpoints/nvidia/GR00T-N1.5-3B}"
MAX_EPISODES="${MAX_EPISODES:-200}"

# target/atomic 15 task default
DEFAULT_TASKS=(
  CloseFridge CloseToasterOvenDoor CoffeeSetupMug OpenCabinet OpenDrawer
  PickPlaceCounterToCabinet PickPlaceCounterToStove PickPlaceDrawerToCounter
  PickPlaceSinkToCounter PickPlaceToasterToCounter SlideDishwasherRack
  TurnOffStove TurnOnElectricKettle TurnOnMicrowave TurnOnSinkFaucet
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
echo "[parallel-n1d5] tasks=$n_total  max_concurrent=$MAX_CONCURRENT  batch=$BATCH  max_episodes=$MAX_EPISODES"
echo "[parallel-n1d5] model=$MODEL_PATH"
echo "[parallel-n1d5] logs → $LOG_DIR/${ts}_*.log"

spawn_task() {
  local task="$1"
  local log="$LOG_DIR/${ts}_${task}.log"
  echo "  [spawn] $task → $log"
  python scripts/extract/extract_eagle_pre_llm_robocasa_n1d5.py \
    --data_root "$DATA_ROOT" \
    --save_path "$SAVE_PATH" \
    --model_path "$MODEL_PATH" \
    --batch_size "$BATCH" \
    --max_episodes "$MAX_EPISODES" \
    --tasks "$task" \
    > "$log" 2>&1
}
export -f spawn_task
export DATA_ROOT SAVE_PATH MODEL_PATH BATCH MAX_EPISODES LOG_DIR ts

printf '%s\n' "${TASKS[@]}" | xargs -n 1 -P "$MAX_CONCURRENT" -I {} bash -c 'spawn_task "$@"' _ {}

echo "[parallel-n1d5] all waves done"

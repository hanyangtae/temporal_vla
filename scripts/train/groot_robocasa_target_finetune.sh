#!/usr/bin/env bash
# GR00T N1.6 baseline fine-tune on RoboCasa target/atomic (15 tasks, per-task dates).
#
# Wrapper: 각 task 마다 demo 생성 date 가 달라서 (`DATE_TAG` 단일값 미지원)
# DATASET_PATH 를 직접 조립한 뒤 baseline sh 에 위임.
#
# GPU 지정 (호스트):
#   CUDA_VISIBLE_DEVICES=3 bash scripts/train/groot_robocasa_target_finetune.sh
#
# Docker exec (컨테이너 내부):
#   docker exec -e CUDA_VISIBLE_DEVICES=3 groot bash /temporal_vla/scripts/train/groot_robocasa_target_finetune.sh
#
# 평가 시 주의: 이 학습 데이터는 NVIDIA 가 특정 seed 로 생성한 demo 들임.
# 평가 시 --seed 는 학습 demo seed 범위와 겹치지 않게 선택할 것 (관례적으로 >= 100000 안전).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
source "${REPO_ROOT}/scripts/utils/cache_env.sh"

# 데이터셋은 cache 로 분리됨 (컨테이너=/cache/datasets, 호스트=~/.cache/temporal_vla/datasets).
ATOMIC_ROOT="${ATOMIC_ROOT:-${VLA_DATASETS_ROOT}/robocasa/v1.0/target/atomic}"

# 각 task 의 demo 생성 date (디스크 검사로 확인된 값).
TASK_DATE_PAIRS=(
    "CloseFridge:20250816"
    "CloseToasterOvenDoor:20250818"
    "CoffeeSetupMug:20250813"
    "OpenCabinet:20250813"
    "OpenDrawer:20250816"
    "PickPlaceCounterToCabinet:20250811"
    "PickPlaceCounterToStove:20250818"
    "PickPlaceDrawerToCounter:20250820"
    "PickPlaceSinkToCounter:20250813"
    "PickPlaceToasterToCounter:20250817"
    "SlideDishwasherRack:20250820"
    "TurnOffStove:20250812"
    "TurnOnElectricKettle:20250817"
    "TurnOnMicrowave:20250813"
    "TurnOnSinkFaucet:20250812"
)

parts=()
for pair in "${TASK_DATE_PAIRS[@]}"; do
    task="${pair%%:*}"
    date="${pair##*:}"
    path="${ATOMIC_ROOT}/${task}/${date}/lerobot"
    if [ ! -d "${path}" ]; then
        echo "ERROR: dataset path not found: ${path}" >&2
        exit 1
    fi
    parts+=("${path}")
done

# 콜론 join (launch_finetune.py 가 ':' 로 split).
DATASET_PATH=$(IFS=":"; echo "${parts[*]}")

export DATASET_PATH
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/groot_robocasa_target_15tasks}"

# Baseline sh 에 위임 (override_pretraining_statistics=True 는 launch_finetune.py 에 hard-set).
exec bash "${SCRIPT_DIR}/groot_robocasa_finetune.sh" "$@"

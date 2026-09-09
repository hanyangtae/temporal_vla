#!/bin/bash
# compose-only(rephrase 없이 QAM 합성만) 라운드. GPU 하나가 seed 하나를 맡아
# compose(상시) → compose_gated(SAFE 발화 후만, α=0.2 고정, 재학습 SAFE) 순서로 실행.
# 사용: bash run_compose_round.sh <gpu> <seed>
set -euo pipefail
GPU=${1:?gpu}
SEED=${2:?seed}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL2="$(cd "$HERE/../../.." && pwd)/RL2-VLA"
export SAFE_DIR_OVERRIDE="$RL2/third_party/SAFE/logs/open_pizero-bridge-lstm-ours_cpTrue/20260807/123421"

# 게이팅(rephrase 없이 발화 후만 합성)을 먼저 — 우선순위 높은 질문
LANE_TAG=ours bash "$HERE/run_arm.sh" compose_gated "$GPU" OOD "$SEED" 50 0.2
bash "$HERE/run_arm.sh" compose "$GPU" OOD "$SEED" 50
echo "[compose_round] seed $SEED (GPU $GPU) 전체 완료"

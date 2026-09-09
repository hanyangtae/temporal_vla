#!/bin/bash
# resample-only + verifier 라운드 (연산자 설계 세션 요청, 2026-08-21)
# GPU 하나가 seed 하나를 맡아 resample(상시) → resample_gated(SAFE 발화 후만, α=0.2 고정,
# 재학습 SAFE) 순서로 실행. 사용: bash run_resample_round.sh <gpu> <seed>
set -euo pipefail
GPU=${1:?gpu}
SEED=${2:?seed}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL2="$(cd "$HERE/../../.." && pwd)/RL2-VLA"
export SAFE_DIR_OVERRIDE="$RL2/third_party/SAFE/logs/open_pizero-bridge-lstm-ours_cpTrue/20260807/123421"
export RESAMPLE_N=40

bash "$HERE/run_arm.sh" resample "$GPU" OOD "$SEED" 50
LANE_TAG=ours bash "$HERE/run_arm.sh" resample_gated "$GPU" OOD "$SEED" 50 0.2
echo "[resample_round] seed $SEED (GPU $GPU) 전체 완료"

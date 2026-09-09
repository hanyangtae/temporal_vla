#!/bin/bash
# 단일 합성(선별·rephrase 없음) 게이팅 라운드. GPU 하나가 seed 하나를 맡아
# w=0.5 → w=0.75 순서로 실행. 사용: bash run_compose1_round.sh <gpu> <seed>
set -euo pipefail
GPU=${1:?gpu}
SEED=${2:?seed}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL2="$(cd "$HERE/../../.." && pwd)/RL2-VLA"
export SAFE_DIR_OVERRIDE="$RL2/third_party/SAFE/logs/open_pizero-bridge-lstm-ours_cpTrue/20260807/123421"

MERGE_W=0.5  LANE_TAG=w05_ours bash "$HERE/run_arm.sh" compose1_gated "$GPU" OOD "$SEED" 50 0.2
MERGE_W=0.75 LANE_TAG=w075_ours bash "$HERE/run_arm.sh" compose1_gated "$GPU" OOD "$SEED" 50 0.2
echo "[compose1_round] seed $SEED (GPU $GPU) 완료"

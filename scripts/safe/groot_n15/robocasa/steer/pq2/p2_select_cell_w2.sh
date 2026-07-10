#!/usr/bin/env bash
# P2 선택 rollout 의 worker2 변형 — pq2_cell_runner_w2.sh(호스트 serve) 사용. w2 에서 실행.
# usage: SCENE=<scene> PA=8480 PB=8481 SINGLE_L=<layer> [FITN=30] bash p2_select_cell_w2.sh
set -uo pipefail
: "${SCENE:?}" "${PA:?}" "${PB:?}" "${SINGLE_L:?}"
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../../.." && pwd)"
cd "$REPO"
source scripts/safe/groot_n15/robocasa/steer/pq/pq_lib.sh
IFS='|' read -r c task env ci seed instr <<<"$(row_of "$SCENE")"
[ -n "$c" ] || { echo "unknown scene: $SCENE"; exit 2; }
FITN="${FITN:-30}"
MAN="$REPO/outputs/eval/robocasa/groot_n15/pq2_manifests/$SCENE"
NPZ="$REPO/outputs/eval/robocasa/groot_n15/pq2_analysis/conceptors/$SCENE/per_scene_fit${FITN}/global"
[ -f "$MAN/split.json" ] || { echo "split.json 없음: $MAN"; exit 2; }
EPLIST=$(python3 -c "import json; print(' '.join(map(str, json.load(open('$MAN/split.json'))['select_half'])))")

run_one() { # tag layers beta
  CELL_ID=$c TASK=$task ENVN=$env CELL_INDEX=$ci SEED=$seed INSTR="$instr" \
  ARM_TAG="$1" STEER_MODE=perm NPZ_DIR="$NPZ" STEER_LAYERS="$2" STEER_BETA="$3" \
  EPLIST="$EPLIST" OUT_TIER=p2 GPUS_L="2 2" PORTS_L="$PA $PB" \
  bash scripts/safe/groot_n15/robocasa/steer/pq2/pq2_cell_runner_w2.sh
}

echo "[$SCENE] w2 P2 select: single=L$SINGLE_L fit$FITN, eps=$(echo $EPLIST | wc -w)판"
run_one "p2_single_L${SINGLE_L}_b01" "$SINGLE_L" 0.1
run_one "p2_single_L${SINGLE_L}_b03" "$SINGLE_L" 0.3
run_one "p2_multi_4_8_12_b01" "4,8,12" 0.1
run_one "p2_multi_4_8_12_b03" "4,8,12" 0.3
echo "[$SCENE] w2 P2 select rollout 완료"

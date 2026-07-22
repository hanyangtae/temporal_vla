#!/usr/bin/env bash
# P2 선택 rollout 의 srv50(구 w2/worker2) 변형 — exp2_cell_runner_srv50.sh(호스트 serve) 사용. srv50 에서 실행.
# usage: SCENE=<scene> PA=8480 PB=8481 SINGLE_L=<layer> [FITN=30] bash p2_select_cell_srv50.sh
set -uo pipefail
: "${SCENE:?}" "${PA:?}" "${PB:?}" "${SINGLE_L:?}"
MYREPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../../.." && pwd)"
cd "$MYREPO"
# 주의: queue_lib.sh 가 REPO 를 로컬 절대경로로 하드코딩 → source 후 MYREPO 만 신뢰 (srv50 사고 실증)
source scripts/safe/groot_n15/robocasa/steer/queue/queue_lib.sh
IFS='|' read -r c task env ci seed instr <<<"$(row_of "$SCENE")"
[ -n "$c" ] || { echo "unknown scene: $SCENE"; exit 2; }
FITN="${FITN:-30}"
MAN="$MYREPO/outputs/eval/robocasa/groot_n15/pq2_manifests/$SCENE"  # srv50 원격 — 원격은 구명(pq) 유지
NPZ="$MYREPO/outputs/eval/robocasa/groot_n15/pq2_analysis/conceptors/$SCENE/per_scene_fit${FITN}/global"  # srv50 원격 — 원격은 구명(pq) 유지
[ -f "$MAN/split.json" ] || { echo "split.json 없음: $MAN"; exit 2; }
EPLIST=$(python3 -c "import json; print(' '.join(map(str, json.load(open('$MAN/split.json'))['select_half'])))")

run_one() { # tag layers beta
  CELL_ID=$c TASK=$task ENVN=$env CELL_INDEX=$ci SEED=$seed INSTR="$instr" \
  ARM_TAG="$1" STEER_MODE=perm NPZ_DIR="$NPZ" STEER_LAYERS="$2" STEER_BETA="$3" \
  EPLIST="$EPLIST" OUT_TIER=p2 GPUS_L="2 2" PORTS_L="$PA $PB" \
  bash scripts/safe/groot_n15/robocasa/steer/exp2/exp2_cell_runner_srv50.sh
}

echo "[$SCENE] srv50 P2 select: single=L$SINGLE_L fit$FITN, eps=$(echo $EPLIST | wc -w)판"
run_one "p2_single_L${SINGLE_L}_b01" "$SINGLE_L" 0.1
run_one "p2_single_L${SINGLE_L}_b03" "$SINGLE_L" 0.3
run_one "p2_multi_4_8_12_b01" "4,8,12" 0.1
run_one "p2_multi_4_8_12_b03" "4,8,12" 0.3
echo "[$SCENE] srv50 P2 select rollout 완료"

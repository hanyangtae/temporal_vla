#!/usr/bin/env bash
# P2 선택 rollout (scene 1개): 후보 config × β 를 select-half 에서 평가 (보고 금지 — 선택 전용).
# 후보 = {single(top1, p1_candidates fit30 기준), multi 4-8-12} × β{0.1, 0.3} = 4 arm.
#   - α 는 각 NPZ metadata 의 selected_alpha 를 loader 가 자동 적용 (5615da9).
#   - closest-α top1 은 band 후보가 없을 때만 사용 (Gate1 사전 규칙) — 여기선 fit30 기준이라
#     호출 전 p1_candidates 에서 확인.
#   - 선택 rollout 전략 = perm(기계식) 고정 — 선택된 (ℓ,β)는 P3 의 perm/gated 공통 이식.
# 선택 규칙(하방-위험, Gate1): base(select-half 라벨 SR)보다 2 episode 이상 낮은 후보 탈락
#   → 잔여 중 SE 이내 동률이면 보수(작은 β, 적은 layer) — 판정은 p2_decide.py.
# usage: SCENE=ppcc_bread GPU=4 PA=8470 PB=8471 [SINGLE_L=4] bash p2_select_cell.sh
set -uo pipefail
: "${SCENE:?}" "${GPU:?}" "${PA:?}" "${PB:?}"
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../../.." && pwd)"
cd "$REPO"
source scripts/safe/groot_n15/robocasa/steer/pq/pq_lib.sh
IFS='|' read -r c task env ci seed instr <<<"$(row_of "$SCENE")"
[ -n "$c" ] || { echo "unknown scene: $SCENE"; exit 2; }
MAN="outputs/eval/robocasa/groot_n15/pq2_manifests/$SCENE"
NPZ30="/temporal_vla/outputs/eval/robocasa/groot_n15/pq2_analysis/conceptors/$SCENE/per_scene_fit30/global"
[ -f "$MAN/split.json" ] || { echo "split.json 없음 — 승준서 pull 필요: $MAN"; exit 2; }
EPLIST=$(python3 -c "import json; print(' '.join(map(str, json.load(open('$MAN/split.json'))['select_half'])))")
SINGLE_L="${SINGLE_L:?p1_candidates fit30 top1 layer 지정 필요}"

run_one() { # tag layers beta
  CELL_ID=$c TASK=$task ENVN=$env CELL_INDEX=$ci SEED=$seed INSTR="$instr" \
  ARM_TAG="$1" STEER_MODE=perm NPZ_DIR="$NPZ30" STEER_LAYERS="$2" STEER_BETA="$3" \
  EPLIST="$EPLIST" OUT_TIER=p2 GPUS_L="$GPU $GPU" PORTS_L="$PA $PB" \
  bash scripts/safe/groot_n15/robocasa/steer/pq2/pq2_cell_runner.sh
}

echo "[$SCENE] P2 select 시작: single=L$SINGLE_L, eps=$(echo $EPLIST | wc -w)판"
run_one "p2_single_L${SINGLE_L}_b01" "$SINGLE_L" 0.1
run_one "p2_single_L${SINGLE_L}_b03" "$SINGLE_L" 0.3
run_one "p2_multi_4_8_12_b01" "4,8,12" 0.1
run_one "p2_multi_4_8_12_b03" "4,8,12" 0.3
echo "[$SCENE] P2 select rollout 완료 → p2_decide.py 로 판정"

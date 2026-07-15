#!/usr/bin/env bash
# pq3 β sweep (계획서 v9 §D) — COAST Stage3 faithful: fit seed 15판 재-steering.
# arm 별(perm/gated) 독립 sweep, 각 arm 의 eval 배선 그대로(β 만 {0.1,0.3}),
# 참조 = 수집 base 라벨(sweep_manifest 의 base_label 열) paired.
#
# env (필수): CELL_ID TASK ENVN CELL_INDEX INSTR GPUS_L PORTS_L
#   PERM_NPZ PERM_LAYERS PERM_NPZ_SHAS  GATED_NPZ GATED_LAYERS GATED_NPZ_SHAS GATED_PHASES
#   (Stage1 layer·Gate D sha·성립 게이트 phase 확정본 — 러너 preflight 필수값, R2 높음#1)
# env (선택): BETAS="0.1 0.3" ARMS="perm gated" MANIFEST(기본 pq3 manifests 경로)
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/pq3_lib.sh"
: "${CELL_ID:?}" "${TASK:?}" "${ENVN:?}" "${CELL_INDEX:?}" "${INSTR:?}"
BETAS="${BETAS:-0.1 0.3}"
ARMS="${ARMS:-perm gated}"
MANIFEST="${MANIFEST:-$(pq3_manifest_of "$CELL_ID" sweep_manifest)}"
N_SWEEP=$(grep -cv -e '^$' -e '^#' -e '^ep_idx' "$MANIFEST" || true)

for arm in $ARMS; do
  GPH=""
  case "$arm" in
    perm)  NPZ="${PERM_NPZ:?}";  LAYERS="${PERM_LAYERS:?}";  SHAS="${PERM_NPZ_SHAS:?}" ;;
    gated) NPZ="${GATED_NPZ:?}"; LAYERS="${GATED_LAYERS:?}"; SHAS="${GATED_NPZ_SHAS:?}"; GPH="${GATED_PHASES:?}" ;;
    *) echo "unknown arm $arm" >&2; exit 2 ;;
  esac
  for beta in $BETAS; do
    echo "[beta-sweep] ${CELL_ID} ${arm} β=${beta} (${N_SWEEP}판, fit seed 재사용)"
    CELL_ID="$CELL_ID" TASK="$TASK" ENVN="$ENVN" CELL_INDEX="$CELL_INDEX" INSTR="$INSTR" \
    ARM_TAG="sweep_${arm}_b${beta/./}" STEER_MODE="$arm" MANIFEST="$MANIFEST" \
    NPZ_DIR="$NPZ" STEER_LAYERS="$LAYERS" STEER_BETA="$beta" NPZ_SHAS="$SHAS" \
    ${GPH:+GATED_PHASES="$GPH"} \
    OUT_TIER=sweep EXPECT_N="$N_SWEEP" GPUS_L="${GPUS_L:?}" PORTS_L="${PORTS_L:?}" \
      bash "$HERE/pq3_cell_runner.sh"
  done
done
echo "[beta-sweep] ${CELL_ID} 완료 — beta_decide.py 로 판정"

#!/usr/bin/env bash
# exp3(구 pq3) 로컬 lane 러너 — 큐에서 HOST∈{"",any,local} 행을 pop 해 exp3_cell_runner.sh 실행.
# 사용: GPU=0 PORTS="8410 8411" LANE=local-g0 bash exp3_lane_local.sh
# (exp2 p3_lane_local.sh 파생 — exp3 큐 형식(KEY=VAL, HOST 필드)과 manifest 주입)
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/exp3_lib.sh"
RUNNER="$HERE/exp3_cell_runner.sh"
: "${GPU:?}" "${PORTS:?}"
LANE="${LANE:-local-g${GPU}}"
CLS="${CLS:-local}"
EXPECT_N="${EXPECT_N:-30}"

while :; do
  line=$(pop_exp3 "$CLS" "$LANE") || {
    if [ ! -s "$EXP3_QROOT/queue.tsv" ] && [ -z "$(ls -A "$EXP3_QROOT/running" 2>/dev/null)" ]; then
      echo "[$LANE] 큐 소진 — 종료"; break
    fi
    sleep 60; continue
  }
  body=${line%|*}
  declare -A F=()
  for kv in $body; do F[${kv%%=*}]=${kv#*=}; done
  row=$(exp3_row_of "${F[CELL]}")
  [ -n "$row" ] || { echo "[$LANE] unknown cell ${F[CELL]}"; requeue_exp3 "$line" "$LANE"; continue; }
  IFS='|' read -r cell task envn cidx _seed instr <<<"$row"
  MANIFEST=$(exp3_manifest_of "$cell" eval_manifest)
  [ -f "$MANIFEST" ] || { echo "[$LANE] manifest 없음: $MANIFEST"; requeue_exp3 "$line" "$LANE"; sleep 30; continue; }

  ENVV=(CELL_ID="$cell" TASK="$task" ENVN="$envn" CELL_INDEX="$cidx" INSTR="$instr"
        ARM_TAG="${F[TAG]}" STEER_MODE="${F[MODE]}" MANIFEST="$MANIFEST"
        EXPECT_N="$EXPECT_N" GPUS_L="$GPU $GPU" PORTS_L="$PORTS" MACHINE_TAG="local-g${GPU}")
  if [ "${F[MODE]}" != base ]; then
    ENVV+=(NPZ_DIR="/temporal_vla/${F[NPZ]}" STEER_LAYERS="${F[LAYERS]}" STEER_BETA="${F[BETA]}")
    [ "${F[SHAS]:--}" != "-" ] && ENVV+=(NPZ_SHAS="${F[SHAS]//,/ }")
    # gated: gate report 의 phase 집합을 serve --steering-phases 까지 전달 (R2 치명#1)
    [ "${F[PHASES]:--}" != "-" ] && ENVV+=(GATED_PHASES="${F[PHASES]}")
  fi
  echo "[$LANE] $(date '+%F %T') RUN ${F[CELL]}/${F[TAG]}"
  if env "${ENVV[@]}" bash "$RUNNER"; then
    d="$EXP3_REPO/outputs/eval/robocasa/groot_n15/steer_eval_exp3/e1/${cell}/${F[TAG]}/raw_rollouts/${task}/${cell}"
    cnt=$(ls "$d"/*succ*.json 2>/dev/null | wc -l)
    finish_exp3 "$line" "$LANE" "local-g${GPU}" "$cnt"
    mark_machine "$d" "local-g${GPU}" "$cnt"
  else
    echo "[$LANE] FAIL ${F[CELL]}/${F[TAG]} — requeue"
    requeue_exp3 "$line" "$LANE"
    sleep 30
  fi
done

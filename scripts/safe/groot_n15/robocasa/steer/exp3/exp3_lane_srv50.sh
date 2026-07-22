#!/usr/bin/env bash
# exp3(구 pq3) 원격 워커 lane 러너 (srv50(구 w2/worker2)/.50 기본, srv48(구 w48) 은 CLS=srv48
# HF_HOME_OVERRIDE=... 로 동일 사용).
# 큐에서 HOST∈{"",any,$CLS} 행을 pop 해 exp3_cell_runner_srv50.sh 실행. 워커 호스트에서 구동.
# 사용(srv50):  GPU=2 PORTS="8410 8411" CLS=srv50 bash exp3_lane_srv50.sh
# 사용(srv48): GPU=2 PORTS="8410 8411" CLS=srv48 MACHINE_TAG=srv48-48 HF_HOME_OVERRIDE=$HOME/.cache/hf_pq3 bash exp3_lane_srv50.sh
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXP3_REPO="${EXP3_REPO:-$(cd -- "$HERE/../../../../../.." && pwd)}"
export EXP3_REPO
source "$HERE/exp3_lib.sh"
RUNNER="$HERE/exp3_cell_runner_srv50.sh"
: "${GPU:?}" "${PORTS:?}"
CLS="${CLS:-srv50}"
LANE="${LANE:-${CLS}-g${GPU}}"
MACHINE_TAG="${MACHINE_TAG:-${CLS}-g${GPU}}"
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
        EXPECT_N="$EXPECT_N" GPUS_L="$GPU $GPU" PORTS_L="$PORTS" MACHINE_TAG="$MACHINE_TAG")
  [ -n "${HF_HOME_OVERRIDE:-}" ] && ENVV+=(HF_HOME_OVERRIDE="$HF_HOME_OVERRIDE")
  if [ "${F[MODE]}" != base ]; then
    ENVV+=(NPZ_DIR="${EXP3_REPO}/${F[NPZ]}" STEER_LAYERS="${F[LAYERS]}" STEER_BETA="${F[BETA]}")
    [ "${F[SHAS]:--}" != "-" ] && ENVV+=(NPZ_SHAS="${F[SHAS]//,/ }")
    [ "${F[PHASES]:--}" != "-" ] && ENVV+=(GATED_PHASES="${F[PHASES]}")
  fi
  echo "[$LANE] $(date '+%F %T') RUN ${F[CELL]}/${F[TAG]}"
  if env "${ENVV[@]}" bash "$RUNNER"; then
    d="$EXP3_REPO/outputs/eval/robocasa/groot_n15/steer_eval_pq3/e1/${cell}/${F[TAG]}/raw_rollouts/${task}/${cell}"  # 워커 원격 dir — 원격은 구명(pq) 유지
    cnt=$(ls "$d"/*succ*.json 2>/dev/null | wc -l)
    finish_exp3 "$line" "$LANE" "$MACHINE_TAG" "$cnt"
    mark_machine "$d" "$MACHINE_TAG" "$cnt"
  else
    echo "[$LANE] FAIL ${F[CELL]}/${F[TAG]} — requeue"
    requeue_exp3 "$line" "$LANE"
    sleep 30
  fi
done

#!/usr/bin/env bash
# pq3 원격 워커 lane 러너 (w2/.50 기본, w48 은 CLS=w48 HF_HOME_OVERRIDE=... 로 동일 사용).
# 큐에서 HOST∈{"",any,$CLS} 행을 pop 해 pq3_cell_runner_w2.sh 실행. 워커 호스트에서 구동.
# 사용(w2):  GPU=2 PORTS="8410 8411" CLS=w2 bash pq3_lane_w2.sh
# 사용(w48): GPU=2 PORTS="8410 8411" CLS=w48 MACHINE_TAG=worker1-48 HF_HOME_OVERRIDE=$HOME/.cache/hf_pq3 bash pq3_lane_w2.sh
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PQ3_REPO="${PQ3_REPO:-$(cd -- "$HERE/../../../../../.." && pwd)}"
export PQ3_REPO
source "$HERE/pq3_lib.sh"
RUNNER="$HERE/pq3_cell_runner_w2.sh"
: "${GPU:?}" "${PORTS:?}"
CLS="${CLS:-w2}"
LANE="${LANE:-${CLS}-g${GPU}}"
MACHINE_TAG="${MACHINE_TAG:-${CLS}-g${GPU}}"
EXPECT_N="${EXPECT_N:-30}"

while :; do
  line=$(pop_pq3 "$CLS" "$LANE") || {
    if [ ! -s "$PQ3_QROOT/queue.tsv" ] && [ -z "$(ls -A "$PQ3_QROOT/running" 2>/dev/null)" ]; then
      echo "[$LANE] 큐 소진 — 종료"; break
    fi
    sleep 60; continue
  }
  body=${line%|*}
  declare -A F=()
  for kv in $body; do F[${kv%%=*}]=${kv#*=}; done
  row=$(pq3_row_of "${F[CELL]}")
  [ -n "$row" ] || { echo "[$LANE] unknown cell ${F[CELL]}"; requeue_pq3 "$line" "$LANE"; continue; }
  IFS='|' read -r cell task envn cidx _seed instr <<<"$row"
  MANIFEST=$(pq3_manifest_of "$cell" eval_manifest)
  [ -f "$MANIFEST" ] || { echo "[$LANE] manifest 없음: $MANIFEST"; requeue_pq3 "$line" "$LANE"; sleep 30; continue; }

  ENVV=(CELL_ID="$cell" TASK="$task" ENVN="$envn" CELL_INDEX="$cidx" INSTR="$instr"
        ARM_TAG="${F[TAG]}" STEER_MODE="${F[MODE]}" MANIFEST="$MANIFEST"
        EXPECT_N="$EXPECT_N" GPUS_L="$GPU $GPU" PORTS_L="$PORTS" MACHINE_TAG="$MACHINE_TAG")
  [ -n "${HF_HOME_OVERRIDE:-}" ] && ENVV+=(HF_HOME_OVERRIDE="$HF_HOME_OVERRIDE")
  if [ "${F[MODE]}" != base ]; then
    ENVV+=(NPZ_DIR="${PQ3_REPO}/${F[NPZ]}" STEER_LAYERS="${F[LAYERS]}" STEER_BETA="${F[BETA]}")
    [ "${F[SHAS]:--}" != "-" ] && ENVV+=(NPZ_SHAS="${F[SHAS]}")
  fi
  echo "[$LANE] $(date '+%F %T') RUN ${F[CELL]}/${F[TAG]}"
  if env "${ENVV[@]}" bash "$RUNNER"; then
    d="$PQ3_REPO/outputs/eval/robocasa/groot_n15/steer_eval_pq3/e1/${cell}/${F[TAG]}/raw_rollouts/${task}/${cell}"
    cnt=$(ls "$d"/*succ*.json 2>/dev/null | wc -l)
    finish_pq3 "$line" "$LANE" "$MACHINE_TAG" "$cnt"
    mark_machine "$d" "$MACHINE_TAG" "$cnt"
  else
    echo "[$LANE] FAIL ${F[CELL]}/${F[TAG]} — requeue"
    requeue_pq3 "$line" "$LANE"
    sleep 30
  fi
done

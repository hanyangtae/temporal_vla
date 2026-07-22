#!/bin/bash
# P3 lane runner (로컬 GPU): work_queue_p3 에서 arm pop → exp2_cell_runner.sh.
# usage: p3_lane_local.sh <LANE> <GPU> <PA> <PB>
set -u
LANE=$1 GPU=$2 PA=$3 PB=$4
MYREPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.." && pwd)
cd "$MYREPO"
source scripts/safe/groot_n15/robocasa/steer/queue/queue_lib.sh  # CELLS/row_of 만 사용 (REPO 덮임 — MYREPO 신뢰)
Q=$MYREPO/outputs/eval/robocasa/groot_n15/work_queue_p3
RUNNER=scripts/safe/groot_n15/robocasa/steer/exp2/exp2_cell_runner.sh
echo "[$LANE] p3 local start gpu=$GPU ports=$PA,$PB $(date -u '+%FT%T')"

pop() { ( flock 9
    line=$(head -n 1 "$Q/queue.tsv" 2>/dev/null)
    [ -n "$line" ] || exit 1
    sed -i '1d' "$Q/queue.tsv"
    printf '%s\n' "$line" | tee "$Q/running/${LANE}.job"
  ) 9>>"$Q/lock"; }
requeue() { ( flock 9
    IFS='|' read -r spec retry <<<"$1"
    if [ "${retry:-0}" -lt 2 ]; then printf '%s|%d\n' "$spec" $((retry+1)) >> "$Q/queue.tsv"
    else printf '%s RETRY_EXCEEDED %s\n' "$1" "$(date -u '+%FT%T')" >> "$Q/failed/failed.tsv"; fi
    rm -f "$Q/running/${LANE}.job" ) 9>>"$Q/lock"; }
finish() { ( flock 9
    printf '%s\t%s\t%s\t%s\n' "$(date -u '+%FT%T')" "local-gpu$GPU" "$1" "$2" >> "$Q/ledger.tsv"
    rm -f "$Q/running/${LANE}.job" ) 9>>"$Q/lock"; }

while true; do
  line=$(pop) || { [ -s "$Q/queue.tsv" ] || [ -n "$(ls "$Q/running" 2>/dev/null)" ] || { echo "[$LANE] 큐 소진 — 종료"; break; }; sleep 120; continue; }
  IFS='|' read -r spec retry <<<"$line"
  declare -A F=()
  for kv in $spec; do F[${kv%%=*}]=${kv#*=}; done
  IFS='|' read -r c task env ci seed instr <<<"$(row_of "${F[CELL]}")"
  [ -n "$c" ] || { echo "[$LANE] unknown cell ${F[CELL]}"; requeue "$line"; continue; }
  echo "[$LANE] ARM ${F[CELL]}/${F[TAG]} (retry=$retry) $(date -u '+%FT%T')"
  ENVV=(CELL_ID="$c" TASK="$task" ENVN="$env" CELL_INDEX="$ci" SEED="$seed" INSTR="$instr"
        ARM_TAG="${F[TAG]}" STEER_MODE="${F[MODE]}" EP0=60 EP1=119 OUT_TIER=p3
        GPUS_L="$GPU $GPU" PORTS_L="$PA $PB")
  if [ "${F[MODE]}" != base ]; then
    ENVV+=(NPZ_DIR="/temporal_vla/${F[NPZ]}" STEER_LAYERS="${F[LAYERS]}" STEER_BETA="${F[BETA]}")
  fi
  env "${ENVV[@]}" bash "$RUNNER" >> "$Q/logs_${LANE}.log" 2>&1
  d="$MYREPO/outputs/eval/robocasa/groot_n15/steer_eval_exp2/p3/$c/${F[TAG]}/raw_rollouts/$task/$c"
  n=$(find "$d" -name '*.pkl' 2>/dev/null | wc -l)
  if [ "$n" -ge 60 ]; then finish "$spec" "$n"; else echo "[$LANE] ${F[TAG]} $n/60 → 재큐"; requeue "$line"; sleep 30; fi
done

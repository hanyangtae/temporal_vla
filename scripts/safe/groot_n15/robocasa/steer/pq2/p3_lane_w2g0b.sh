#!/bin/bash
# P3 lane runner (w2 오케스트레이션): 로컬에서 실행, arm 을 w2 에 ssh 로 위임.
# 산출 pkl 은 w2 보관, 로컬은 파일명 manifest(판정 출처)+MACHINE 만 — v2 직송 패턴의 경량판.
# usage: p3_lane_w2.sh <LANE> <P1> <P2> <P3port>
set -u
LANE=$1 W_P1=$2 W_P2=$3
MYREPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.." && pwd)
cd "$MYREPO"
source scripts/safe/groot_n15/robocasa/steer/pq/pq_lib.sh
HREPO=/home/junhyeong/pkt_ws/temporal_vla
SSH="ssh -o BatchMode=yes -o ServerAliveInterval=60 -o ServerAliveCountMax=10 $W2"
Q=$MYREPO/outputs/eval/robocasa/groot_n15/work_queue_p3
echo "[$LANE] p3 w2-gpu0 start ports=$W_P1,$W_P2 $(date -u '+%FT%T')"

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
    printf '%s\t%s\t%s\t%s\n' "$(date -u '+%FT%T')" "worker2-a100-gpu2" "$1" "$2" >> "$Q/ledger.tsv"
    rm -f "$Q/running/${LANE}.job" ) 9>>"$Q/lock"; }

while true; do
  free_kb=$($SSH "df -k / | tail -1 | awk '{print \$4}'" 2>/dev/null || echo 0)
  [ "${free_kb:-0}" -lt 10485760 ] && { echo "[$LANE] w2 디스크/ssh 문제 — 10분 대기"; sleep 600; continue; }
  line=$(pop) || { [ -s "$Q/queue.tsv" ] || [ -n "$(ls "$Q/running" 2>/dev/null)" ] || { echo "[$LANE] 큐 소진 — 종료"; break; }; sleep 120; continue; }
  IFS='|' read -r spec retry <<<"$line"
  declare -A F=()
  for kv in $spec; do F[${kv%%=*}]=${kv#*=}; done
  IFS='|' read -r c task env ci seed instr <<<"$(row_of "${F[CELL]}")"
  [ -n "$c" ] || { echo "[$LANE] unknown cell ${F[CELL]}"; requeue "$line"; continue; }
  echo "[$LANE] ARM ${F[CELL]}/${F[TAG]} (retry=$retry) $(date -u '+%FT%T')"
  STEER_ENV=""
  [ "${F[MODE]}" != base ] && STEER_ENV="NPZ_DIR=$HREPO/${F[NPZ]} STEER_LAYERS=${F[LAYERS]} STEER_BETA=${F[BETA]}"
  RCMD="cd $HREPO && env CELL_ID=$c TASK=$task ENVN=$env CELL_INDEX=$ci SEED=$seed INSTR=$(printf '%q' "$instr") \
ARM_TAG=${F[TAG]} STEER_MODE=${F[MODE]} EP0=60 EP1=119 OUT_TIER=p3 GPUS_L='0 0' PORTS_L='$W_P1 $W_P2' $STEER_ENV \
bash scripts/safe/groot_n15/robocasa/steer/pq2/pq2_cell_runner_w2.sh"
  $SSH "$RCMD" >> "$Q/logs_${LANE}.log" 2>&1
  RRD="outputs/eval/robocasa/groot_n15/steer_eval_pq2/p3/$c/${F[TAG]}/raw_rollouts/$task/$c"
  LD="$MYREPO/outputs/eval/robocasa/groot_n15/steer_eval_pq2/p3/$c/${F[TAG]}"; mkdir -p "$LD"
  $SSH "cd $HREPO/$RRD 2>/dev/null && find . -name '*.pkl' | sort" > "$LD/manifest_w2.txt" 2>/dev/null
  n=$(grep -c pkl "$LD/manifest_w2.txt" 2>/dev/null || echo 0)
  printf '%s machine=worker2-a100-gpu2 eps=%s\n' "$(date -u '+%FT%T')" "$n" >> "$LD/MACHINE.txt"
  if [ "$n" -ge 60 ]; then finish "$spec" "$n"; else echo "[$LANE] ${F[TAG]} $n/60 → 재큐"; requeue "$line"; sleep 30; fi
done

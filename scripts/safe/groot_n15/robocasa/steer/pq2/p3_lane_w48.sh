#!/bin/bash
# P3 lane runner (.48 오케스트레이션): 로컬에서 실행, arm 을 .48(worker1 A100 GPU2)에 ssh 로 위임.
# p3_lane_w2g0.sh 의 .48 변형 — 차이: 호스트, GPU2, HF_HOME 주입(체크포인트가 전용 캐시
# ~/.cache/temporal_vla/datasets/huggingface 에 있음 — HF hub 캐시는 타인 자산이라 미사용),
# HF_HUB_OFFLINE=1(네트워크 resolve 차단), machine 태그.
# usage: p3_lane_w48.sh <LANE> <P1> <P2> <P3port>
set -u
LANE=$1 W_P1=$2 W_P2=$3 W_P3=$4
MYREPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.." && pwd)
cd "$MYREPO"
source scripts/safe/groot_n15/robocasa/steer/pq/pq_lib.sh
W48="junhyeong@166.104.35.48"
HREPO=/home/junhyeong/pkt_ws/temporal_vla
MACHINE=worker48-a100-gpu2
HFENV="HF_HOME=/home/junhyeong/.cache/temporal_vla/datasets/huggingface HF_HUB_OFFLINE=1 MACHINE_TAG=$MACHINE"
SSH="ssh -o BatchMode=yes -o ServerAliveInterval=60 -o ServerAliveCountMax=10 $W48"
Q=$MYREPO/outputs/eval/robocasa/groot_n15/work_queue_p3
echo "[$LANE] p3 w48-gpu2 start ports=$W_P1,$W_P2,$W_P3 $(date -u '+%FT%T')"

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
    printf '%s\t%s\t%s\t%s\n' "$(date -u '+%FT%T')" "$MACHINE" "$1" "$2" >> "$Q/ledger.tsv"
    rm -f "$Q/running/${LANE}.job" ) 9>>"$Q/lock"; }

while true; do
  free_kb=$($SSH "df -k / | tail -1 | awk '{print \$4}'" 2>/dev/null || echo 0)
  [ "${free_kb:-0}" -lt 10485760 ] && { echo "[$LANE] w48 디스크/ssh 문제 — 10분 대기"; sleep 600; continue; }
  line=$(pop) || { [ -s "$Q/queue.tsv" ] || [ -n "$(ls "$Q/running" 2>/dev/null)" ] || { echo "[$LANE] 큐 소진 — 종료"; break; }; sleep 120; continue; }
  IFS='|' read -r spec retry <<<"$line"
  declare -A F=()
  for kv in $spec; do F[${kv%%=*}]=${kv#*=}; done
  IFS='|' read -r c task env ci seed instr <<<"$(row_of "${F[CELL]}")"
  [ -n "$c" ] || { echo "[$LANE] unknown cell ${F[CELL]}"; requeue "$line"; continue; }
  echo "[$LANE] ARM ${F[CELL]}/${F[TAG]} (retry=$retry) $(date -u '+%FT%T')"
  STEER_ENV=""
  [ "${F[MODE]}" != base ] && STEER_ENV="NPZ_DIR=$HREPO/${F[NPZ]} STEER_LAYERS=${F[LAYERS]} STEER_BETA=${F[BETA]}"
  RCMD="cd $HREPO && env $HFENV CELL_ID=$c TASK=$task ENVN=$env CELL_INDEX=$ci SEED=$seed INSTR=$(printf '%q' "$instr") \
ARM_TAG=${F[TAG]} STEER_MODE=${F[MODE]} EP0=60 EP1=119 OUT_TIER=p3 GPUS_L='2 2 2' PORTS_L='$W_P1 $W_P2 $W_P3' $STEER_ENV \
bash scripts/safe/groot_n15/robocasa/steer/pq2/pq2_cell_runner_w2.sh"
  $SSH "$RCMD" >> "$Q/logs_${LANE}.log" 2>&1
  RRD="outputs/eval/robocasa/groot_n15/steer_eval_pq2/p3/$c/${F[TAG]}/raw_rollouts/$task/$c"
  LD="$MYREPO/outputs/eval/robocasa/groot_n15/steer_eval_pq2/p3/$c/${F[TAG]}"; mkdir -p "$LD"
  $SSH "cd $HREPO/$RRD 2>/dev/null && find . -name '*.pkl' | sort" > "$LD/manifest_w48.txt" 2>/dev/null
  n=$(grep -c pkl "$LD/manifest_w48.txt" 2>/dev/null || echo 0)
  printf '%s machine=%s eps=%s\n' "$(date -u '+%FT%T')" "$MACHINE" "$n" >> "$LD/MACHINE.txt"
  if [ "$n" -ge 60 ]; then finish "$spec" "$n"; else echo "[$LANE] ${F[TAG]} $n/60 → 재큐"; requeue "$line"; sleep 30; fi
done

#!/bin/bash
# srv50(구 w2/worker2) lane runner (로컬에서 실행): 큐 pop → ssh로 srv50에서 heldout_round_cell_host.sh 실행
# → 완료 arm rsync pull → 검증 → srv50 원본 pkl/mp4 삭제 → MACHINE.txt 기록.
# usage: lane_runner_srv50.sh <lane_id> <p1> <p2> <p3>   (GPU2 고정, 3-worker)
set -u
source "$(dirname "${BASH_SOURCE[0]}")/queue_lib.sh"
LANE=$1 P1=$2 P2=$3 P3=$4
HREPO=/home/junhyeong/pkt_ws/temporal_vla
SSH="ssh -o BatchMode=yes -o ServerAliveInterval=60 -o ServerAliveCountMax=10 $SRV50"
cd "$REPO"
echo "[$LANE] start srv50 GPU2 ports=$P1,$P2,$P3 $(date -u '+%FT%T')"

while true; do
  # 디스크 가드 (srv50 여유 <10GB → 대기)
  free_kb=$($SSH "df -k / | tail -1 | awk '{print \$4}'" 2>/dev/null || echo 0)
  if [ "${free_kb:-0}" -lt 10485760 ]; then
    echo "[$LANE] srv50 디스크/ssh 문제(free_kb=$free_kb) — 10분 대기"; sleep 600; continue
  fi

  line=$(pop_job srv50 "$LANE") || {
    if [ ! -s "$QROOT/queue.tsv" ] && [ -z "$(ls "$QROOT"/running 2>/dev/null)" ]; then
      echo "[$LANE] queue 소진 — 종료 $(date -u '+%FT%T')"; break
    fi
    sleep 180; continue; }
  IFS='|' read -r jt cell suf arms npzsub layers cons retry <<<"$line"
  IFS='|' read -r c task env ci seed instr <<<"$(row_of "$cell")"
  echo "[$LANE] JOB $jt $cell $arms$suf (retry=$retry) $(date -u '+%FT%T')"

  tag=$(tag_of "$arms" "$suf")
  before=$(pkl_count "$cell" "$tag")

  RCMD="cd $HREPO && CELL_ID=$c TASK=$task ENVN=$env CELL_INDEX=$ci SEED=$seed INSTR=$(printf '%q' "$instr") \
GPUS_L='2 2 2' PORTS_L='$P1 $P2 $P3' EP0=60 EP1=119 PROX=1 SUF=$suf ARMS=$arms \
NPZ_ROOT=$HREPO/outputs/eval/robocasa/groot_n15/phase_event_6p/analysis/${npzsub:-final_ps60} \
${layers:+STEER_LAYERS=$layers} bash scripts/safe/groot_n15/robocasa/steer/heldout_round_cell_host.sh"
  $SSH "$RCMD" >> "$QROOT/logs/srv50_${LANE}.log" 2>&1
  rc=$?

  # 결과 pull (부분 실패여도 있는 만큼 회수) → 검증 → 원격 정리
  RDIR="$HREPO/outputs/eval/robocasa/groot_n15/steer_eval/$c/$tag/raw_rollouts/"
  LDIR="$REPO/$SE/$c/$tag/raw_rollouts/"
  mkdir -p "$LDIR"
  rsync -a --min-size=1 "$SRV50:$RDIR" "$LDIR" 2>/dev/null
  pull_rc=$?
  after=$(pkl_count "$cell" "$tag")
  mark_machine "$REPO/$SE/$c/$tag" "srv50-a100-gpu2" "$((after - before))"
  if [ "$after" -ge 60 ] && [ "$pull_rc" -eq 0 ]; then
    $SSH "find $HREPO/outputs/eval/robocasa/groot_n15/steer_eval/$c/$tag -type f \( -name '*.pkl' -o -name '*.mp4' \) -delete" 2>/dev/null
    finish_job "$line" "$LANE" "srv50-a100-gpu2" "$((after - before))" "$after"
  else
    echo "[$LANE] $tag ${after}/60 (rc=$rc pull=$pull_rc) → local-only 재큐"
    requeue_job "$line" "$LANE" local-only
    sleep 60
  fi
done

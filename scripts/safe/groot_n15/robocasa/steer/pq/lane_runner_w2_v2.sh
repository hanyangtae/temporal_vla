#!/bin/bash
# worker2 lane runner v2: 산출 pkl/mp4를 로컬에 안 쌓고 승준으로 직송(로컬은 파이프만).
#  - 로컬 수신물 = 경량 메타(파일명 manifest, sr tsv, MACHINE.txt, ledger)뿐
#  - 무거운 것(pkl/mp4) = ssh w2 tar | ssh 승준 tar 스트림 → 승준 검증 → w2 삭제
# usage: lane_runner_w2_v2.sh <lane_id> <p1> <p2> <p3>
set -u
source "$(dirname "${BASH_SOURCE[0]}")/pq_lib.sh"
LANE=$1 P1=$2 P2=$3 P3=$4
HREPO=/home/junhyeong/pkt_ws/temporal_vla
SJ="kimseungjun@166.104.146.37"; SJP=11112; SJREPO=workspace/temporal_vla
SSH="ssh -o BatchMode=yes -o ServerAliveInterval=60 -o ServerAliveCountMax=10 $W2"
cd "$REPO"
echo "[$LANE] start w2-v2 GPU2 ports=$P1,$P2,$P3 $(date -u '+%FT%T')"

while true; do
  free_kb=$($SSH "df -k / | tail -1 | awk '{print \$4}'" 2>/dev/null || echo 0)
  if [ "${free_kb:-0}" -lt 10485760 ]; then
    echo "[$LANE] worker2 디스크/ssh 문제(free_kb=$free_kb) — 10분 대기"; sleep 600; continue
  fi

  line=$(pop_job w2 "$LANE") || {
    if [ ! -s "$QROOT/queue.tsv" ] && [ -z "$(ls "$QROOT"/running 2>/dev/null)" ]; then
      echo "[$LANE] queue 소진 — 종료 $(date -u '+%FT%T')"; break
    fi
    sleep 180; continue; }
  IFS='|' read -r jt cell suf arms npzsub layers cons retry <<<"$line"
  IFS='|' read -r c task env ci seed instr <<<"$(row_of "$cell")"
  echo "[$LANE] JOB $jt $cell $arms$suf (retry=$retry) $(date -u '+%FT%T')"

  if [ "$jt" = collect ]; then
    # P0 fit 재료 수집 (ep0-59) → 승준 직송 (fit 은 승준에서 — pq2 재설계)
    RCMD="cd $HREPO && CELL_ID=$c TASK=$task ENVN=$env CELL_INDEX=$ci SEED=$seed INSTR=$(printf '%q' "$instr") \
GPUS_L='2 2 2' PORTS_L='$P1 $P2 $P3' bash scripts/safe/groot_n15/robocasa/steer/pq2/w2_collect_cell.sh"
    $SSH "$RCMD" >> "$QROOT/logs/w2_${LANE}.log" 2>&1
    rc=$?
    FRD="outputs/eval/robocasa/groot_n15/phase_event_6p/raw_rollouts/$task/$c"
    LFIT="$REPO/$SRC6/$task/$c"; mkdir -p "$LFIT"
    $SSH "cd $HREPO/$FRD 2>/dev/null && find . -name '*.pkl' | sort" > "$LFIT/manifest_w2.txt" 2>/dev/null
    n_pkl=$(grep -c pkl "$LFIT/manifest_w2.txt" 2>/dev/null || echo 0)
    mark_machine "$LFIT" "worker2-a100-gpu2(direct-sj)" "$n_pkl"
    if [ "$n_pkl" -ge 60 ]; then
      $SSH "cd $HREPO && tar cf - $FRD" | ssh -o BatchMode=yes -p $SJP $SJ "cd $SJREPO && tar xf -" \
        2>> "$QROOT/logs/w2_${LANE}.log"
      ship_rc=$?
      sj_n=$(ssh -o BatchMode=yes -p $SJP $SJ "find $SJREPO/$FRD -name '*.pkl' 2>/dev/null | wc -l" 2>/dev/null || echo 0)
      if [ "$ship_rc" -eq 0 ] && [ "${sj_n:-0}" -ge "$n_pkl" ]; then
        $SSH "find $HREPO/$FRD -type f \( -name '*.pkl' -o -name '*.mp4' \) -delete"
        echo "[$LANE] $c collect → 승준 직송 완료 (pkl=$n_pkl, sj=$sj_n)"
        finish_job "$line" "$LANE" "worker2-a100-gpu2(sj=$sj_n)" "$n_pkl" "$n_pkl"
      else
        echo "[$LANE] $c collect 직송 검증 실패(ship_rc=$ship_rc sj=$sj_n/$n_pkl) — w2 원본 보존, 재큐"
        requeue_job "$line" "$LANE"
      fi
    else
      echo "[$LANE] $c collect ${n_pkl}/60 (rc=$rc) → 재큐"
      requeue_job "$line" "$LANE"; sleep 60
    fi
    continue
  fi

  tag=$(tag_of "$arms" "$suf")

  RCMD="cd $HREPO && CELL_ID=$c TASK=$task ENVN=$env CELL_INDEX=$ci SEED=$seed INSTR=$(printf '%q' "$instr") \
GPUS_L='2 2 2' PORTS_L='$P1 $P2 $P3' EP0=60 EP1=119 PROX=1 SUF=$suf ARMS=$arms \
NPZ_ROOT=$HREPO/outputs/eval/robocasa/groot_n15/phase_event_6p/analysis/${npzsub:-final_ps60} \
${layers:+STEER_LAYERS=$layers} bash scripts/safe/groot_n15/robocasa/steer/heldout_round_cell_host.sh"
  $SSH "$RCMD" >> "$QROOT/logs/w2_${LANE}.log" 2>&1
  rc=$?

  RRD="outputs/eval/robocasa/groot_n15/steer_eval/$c/$tag/raw_rollouts"
  # 1) 경량 메타 회수: 파일명 manifest (판정은 succ0/1 파일명으로 충분)
  LTAG="$REPO/$SE/$c/$tag"; mkdir -p "$LTAG"
  $SSH "cd $HREPO/$RRD 2>/dev/null && find . -name '*.pkl' | sort" > "$LTAG/manifest_w2.txt" 2>/dev/null
  n_pkl=$(grep -c pkl "$LTAG/manifest_w2.txt" 2>/dev/null || echo 0)
  n_succ=$(grep -c 'succ1.pkl' "$LTAG/manifest_w2.txt" 2>/dev/null || echo 0)
  mark_machine "$LTAG" "worker2-a100-gpu2(direct-sj)" "$n_pkl"

  if [ "$n_pkl" -ge 60 ]; then
    # 2) 승준 직송 스트림 (로컬 디스크 미사용) — 표준 상대경로 유지
    $SSH "cd $HREPO && tar cf - $RRD" | ssh -o BatchMode=yes -p $SJP $SJ "cd $SJREPO && tar xf -" \
      2>> "$QROOT/logs/w2_${LANE}.log"
    ship_rc=$?
    # 3) 승준 검증 (pkl 개수 대조)
    sj_n=$(ssh -o BatchMode=yes -p $SJP $SJ "find $SJREPO/$RRD -name '*.pkl' 2>/dev/null | wc -l" 2>/dev/null || echo 0)
    if [ "$ship_rc" -eq 0 ] && [ "${sj_n:-0}" -ge "$n_pkl" ]; then
      $SSH "find $HREPO/outputs/eval/robocasa/groot_n15/steer_eval/$c/$tag -type f \( -name '*.pkl' -o -name '*.mp4' \) -delete"
      echo "[$LANE] $tag → 승준 직송 완료 (pkl=$n_pkl, succ=$n_succ, sj=$sj_n)"
      finish_job "$line" "$LANE" "worker2-a100-gpu2(sj=$sj_n)" "$n_pkl" "$n_pkl"
    else
      echo "[$LANE] $tag 직송 검증 실패(ship_rc=$ship_rc sj=$sj_n/$n_pkl) — w2 원본 보존, 재큐"
      requeue_job "$line" "$LANE"
    fi
  else
    echo "[$LANE] $tag ${n_pkl}/60 (rc=$rc) → 재큐"
    requeue_job "$line" "$LANE"
    sleep 60
  fi
done

#!/usr/bin/env bash
# 수집 완료(ALL_DONE_cleaned 또는 60 pkl) 대기 → 원격 HDD로 rsync 백업.
# GPU 회수는 watch_phase_event_cleanup.sh(별도)가 serve kill로 처리. 로컬은 Rung2 분석용으로 보존.
# setsid 로 detach 실행 (세션 종료와 무관 생존).
set -uo pipefail
RUN=/home/dongkyu/pkt_ws/temporal_vla/outputs/eval/robocasa/groot_n15/phase_event_aligned_4cell
REMOTE="kimseungjun@166.104.146.37"; RPORT=11112
RDEST="workspace/temporal_vla/outputs/eval/robocasa/groot_n15/"
LOG="$RUN/logs/archive_on_done.log"
mkdir -p "$RUN/logs"
{
  echo "[archive] $(date '+%F %T') waiting for completion…"
  for i in $(seq 1 600); do   # 최대 ~100분
    n=$(find "$RUN/raw_rollouts" -name 'task*--ep*--succ*.pkl' 2>/dev/null | wc -l)
    if [ -f "$RUN/logs/ALL_DONE_cleaned" ]; then echo "[archive] ALL_DONE_cleaned (pkl=$n)"; break; fi
    if [ "$n" -ge 60 ]; then echo "[archive] 60 pkls reached; grace 30s"; sleep 30; break; fi
    sleep 10
  done
  n=$(find "$RUN/raw_rollouts" -name 'task*--ep*--succ*.pkl' 2>/dev/null | wc -l)
  echo "[archive] $(date '+%F %T') final pkl=$n; rsync → $REMOTE:$RDEST"
  ssh -o BatchMode=yes -p $RPORT "$REMOTE" "mkdir -p $RDEST" 2>&1
  rsync -az -e "ssh -o BatchMode=yes -p $RPORT" --exclude 'smoke/' --exclude 'verify/' "$RUN" "$REMOTE:$RDEST" 2>&1
  rc=$?
  echo "[archive] rsync rc=$rc $(date '+%F %T')"
  if [ "$rc" -eq 0 ]; then
    touch "$RUN/logs/ARCHIVED_REMOTE"
    echo "[archive] DONE: backed up to $REMOTE:$RDEST (local kept for Rung2)."
  else
    echo "[archive] FAILED rc=$rc — local intact; archive manually with scripts/utils/remote_compute.sh push-data."
  fi
  echo "[archive] GPU after watcher cleanup:"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed -n '6,8p'
} >> "$LOG" 2>&1

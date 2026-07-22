#!/bin/bash
# read-only 모니터 (1개): 큐/러너/디스크/에러 감시. 이벤트 시 종료(exit code로 구분).
# 0=전작업 완료 / 2=러너 전멸(큐 남음) / 3=로컬 디스크 <10GB / 4=srv50(구 w2/worker2) 접속 불가 지속 / 5=failed 발생
set -u
source "$(dirname "${BASH_SOURCE[0]}")/queue_lib.sh"
srv50fail=0
while true; do
  q=$(wc -l < "$QROOT/queue.tsv" 2>/dev/null || echo 0)
  r=$(ls "$QROOT/running" 2>/dev/null | wc -l)
  d=$(wc -l < "$QROOT/done/done.tsv" 2>/dev/null || echo 0)
  f=$(wc -l < "$QROOT/failed/failed.tsv" 2>/dev/null || echo 0)
  runners=$(pgrep -fc "lane_""runner" || true)
  freeg=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
  echo "$(date -u '+%FT%T') queue=$q running=$r done=$d failed=$f runners=$runners disk=${freeg}G"
  [ "$f" -gt 0 ] && { echo "EVENT: failed 작업 발생 ($f건)"; exit 5; }
  if [ "$q" -eq 0 ] && [ "$r" -eq 0 ]; then echo "EVENT: 큐 소진 — 전 작업 완료"; exit 0; fi
  if [ "${runners:-0}" -eq 0 ]; then echo "EVENT: 러너 전멸 (queue=$q running=$r)"; exit 2; fi
  [ "${freeg:-99}" -lt 10 ] && { echo "EVENT: 로컬 디스크 ${freeg}G"; exit 3; }
  if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$SRV50" true 2>/dev/null; then
    srv50fail=$((srv50fail+1)); [ $srv50fail -ge 3 ] && { echo "EVENT: srv50(구 w2/worker2) ssh 3연속 실패"; exit 4; }
  else srv50fail=0; fi
  sleep 600
done

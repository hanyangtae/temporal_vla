#!/bin/bash
# 승준 노드 구세대 격리 2차 (pq2 미사용 데이터 → deprecated/). 로컬·w2와 동일 분류.
# 반드시 rejudge 드레인 후 실행 — s5 shard 가 strict/aligned/coast4 구라운드 dir 을 읽는다.
# (fork waiter 의 rename/deprecate 와 대상 비중복 — 순서 무관)
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.."/outputs/eval/robocasa/groot_n15 || exit 2
if pgrep -f 'rejudge_'success >/dev/null 2>&1; then
  echo "ABORT: rejudge 실행 중 — 드레인 후 재실행"; exit 3
fi
mkdir -p deprecated
{
  echo "# $(date -u '+%FT%T') 구세대 격리 2차 (승준)"
  for d in phase_event_strict phase_event_aligned_4cell coast4_instruction_pathway_50ep \
           coast_faithful_7task_30ep coast4_steer_eval; do
    [ -e "$d" ] && mv "$d" deprecated/ && echo "moved: $d"
  done
} | tee -a deprecated/MOVED_20260710_round2.txt
echo "[done] 잔존: $(ls -d */ | tr '\n' ' ')"

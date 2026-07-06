#!/usr/bin/env bash
# Detached watcher for the phase_event_aligned_4cell collection: waits for all 3 workers
# to finish (worker*.done sentinels) or 60 pkls, writes the per-cell succ/fail split, then
# kills the 3 N1.5 serves (GPU 5/6/7) per cleanup policy. Runs under setsid (no babysitting).
set -uo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
RUN="${REPO_ROOT}/outputs/eval/robocasa/groot_n15/phase_event_aligned_4cell"
RAW="${RUN}/raw_rollouts"

while true; do
  ndone=$(ls "${RUN}"/logs/worker*.done 2>/dev/null | wc -l)
  npkl=$(find "${RAW}" -name 'task*--ep*--succ*.pkl' 2>/dev/null | wc -l)
  if [ "${ndone}" -ge 3 ] || [ "${npkl}" -ge 60 ]; then break; fi
  sleep 60
done

# per-cell succ/fail split
{
  printf "cell\ttask\tsucc\tfail\ttotal\n"
  for d in "${RAW}"/*/*/; do
    [ -d "$d" ] || continue
    cell=$(basename "$d"); task=$(basename "$(dirname "$d")")
    s=$(ls "$d"task*succ1.pkl 2>/dev/null | wc -l)
    f=$(ls "$d"task*succ0.pkl 2>/dev/null | wc -l)
    printf "%s\t%s\t%s\t%s\t%s\n" "$cell" "$task" "$s" "$f" "$((s+f))"
  done
} > "${RUN}/succ_fail_split.tsv"

# cleanup: stop the 3 collection serves (frees GPU 5/6/7)
docker exec lerobot bash -lc "pkill -f 'serve/lerobot.py.*--collect' || true" 2>/dev/null || true
touch "${RUN}/logs/ALL_DONE_cleaned"

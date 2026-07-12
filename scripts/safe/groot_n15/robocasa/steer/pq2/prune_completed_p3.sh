#!/bin/bash
# 완료 P3 arm 의 로컬 pkl 정리: 이중채점 메타(sidecar tsv) 추출 → 승준 직송 → csv 마커 대체.
# 판정·집계는 파일명(succ)+sidecar 로 충분해짐. ledger 에 있고 pkl≥60 인 arm 만 처리 (검증 전 삭제 금지).
# usage: bash prune_completed_p3.sh  (로컬에서, 멱등)
set -u
MYREPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.." && pwd)
cd "$MYREPO"
Q=outputs/eval/robocasa/groot_n15/work_queue_p3
P3=outputs/eval/robocasa/groot_n15/steer_eval_pq2/p3
SJ="kimseungjun@166.104.146.37"; SJP=11112; SJREPO=workspace/temporal_vla

awk -F'\t' '{split($3,a," "); sub("CELL=","",a[1]); sub("TAG=","",a[2]); print a[1], a[2]}' "$Q/ledger.tsv" | sort -u | \
while read -r cell tag; do
  task=PickPlaceCounterToCabinet; [[ $cell == ppcs* ]] && task=PickPlaceCounterToStove
  d="$P3/$cell/$tag/raw_rollouts/$task/$cell"
  n=$(find "$d" -name '*.pkl' 2>/dev/null | wc -l)
  [ "$n" -ge 60 ] || continue  # 로컬 pkl 없으면(w2 arm/이미 처리) skip
  side="$P3/$cell/$tag/dual_scoring.tsv"
  if [ ! -s "$side" ]; then
    docker exec -e PYTHONPATH=/temporal_vla/src/policies/Isaac-GR00T robocasa python - "$d" "$side" <<'EOF' || continue
import pickle, sys, glob, os
d, out = "/temporal_vla/" + sys.argv[1], "/temporal_vla/" + sys.argv[2]
rows = []
for p in sorted(glob.glob(d + "/task*--ep*--succ*.pkl")):
    r = pickle.load(open(p, "rb"))
    ev = r.get("succ_ever_th") or {}
    rows.append((os.path.basename(p), r.get("episode_success"), ev.get("0.07"), ev.get("0.1")))
with open(out, "w") as f:
    f.write("pkl\tsucc\tever007\tever010\n")
    for r in rows:
        f.write("\t".join(str(x) for x in r) + "\n")
print(f"[sidecar] {len(rows)} rows -> {out}")
EOF
  fi
  rows=$(($(wc -l < "$side") - 1))
  [ "$rows" -ge "$n" ] || { echo "[skip] $cell/$tag sidecar $rows < $n"; continue; }
  # 승준 직송 (아카이브) → 검증 → 마커 대체
  tar cf - "$d" | ssh -o BatchMode=yes -p $SJP $SJ "cd $SJREPO && tar xf -" || { echo "[ship-fail] $cell/$tag"; continue; }
  sjn=$(ssh -o BatchMode=yes -p $SJP $SJ "find $SJREPO/$d -name '*.pkl' | wc -l")
  if [ "${sjn:-0}" -ge "$n" ]; then
    find "$d" -name '*.pkl' | while read -r p; do : > "${p%.pkl}.csv"; rm "$p"; done
    echo "[pruned] $cell/$tag ($n pkl → 승준, sidecar $rows)"
  else
    echo "[verify-fail] $cell/$tag sj=$sjn/$n — 보존"
  fi
done
df -h / | tail -1

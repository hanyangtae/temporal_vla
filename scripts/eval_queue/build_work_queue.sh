#!/bin/bash
# 현 디스크 상태를 스캔해 남은 작업 전량을 queue.tsv로 생성 (멱등 — 60판 채워진 arm은 제외).
# 이미 일부 판이 있는 arm(구멍)은 local-only(에피소드 resume이 로컬 파일 기준이라).
set -u
source "$(dirname "${BASH_SOURCE[0]}")/queue_lib.sh"
cd "$REPO"
mkdir -p "$QROOT"/{running,done,failed,logs}
touch "$QROOT/ledger.tsv" "$QROOT/done/done.tsv" "$QROOT/failed/failed.tsv"
Q="$QROOT/queue.tsv"; : > "$Q"

emit() { # jobtype cell suf arms npzsub layers constraint
  printf '%s|%s|%s|%s|%s|%s|%s|0\n' "$1" "$2" "$3" "$4" "$5" "$6" "$7" >> "$Q"
}
need_arm() { # cell suf arm npzsub layers gate → emit if pkl<60
  local cnt; cnt=$(pkl_count "$1" "$(tag_of "$3" "$2")")
  [ "$cnt" -ge 60 ] && return 0
  local cons=$6
  [ "$cnt" -gt 0 ] && cons=local-only   # 구멍 resume은 로컬 고정
  emit arm "$1" "$2" "$3" "$4" "$5" "$cons"
}

# 1) A-S1 수집 (fit 데이터 없는 apple cell) — local-only
for cell in $APPLES; do
  [ "$(fit_count "$cell")" -ge 60 ] || emit collect "$cell" "" "" "" "" local-only
done

# 2) bread 잔여 (base + perm/gated × ps/xb + L4/L812)
for cell in $BREADS; do
  need_arm "$cell" "" base "" "" ""
  for N in 15 30 60; do
    need_arm "$cell" "ps$N" perm  "final_ps$N" "" ""
    need_arm "$cell" "ps$N" gated "final_ps$N" "" ""
    need_arm "$cell" "xb$N" perm  "final_xb$N" "" ""
    need_arm "$cell" "xb$N" gated "final_xb$N" "" ""
  done
  need_arm "$cell" L4   gated final_ps60 4    ""
  need_arm "$cell" L812 gated final_ps60 8,12 ""
done

# 3) apple base (NPZ 불요 — 즉시 가능)
for cell in $APPLES; do need_arm "$cell" "" base "" "" ""; done

# 4) apple steered armset 전부 + bread gx — npz-wait
for cell in $APPLES; do
  for N in 15 30 60; do
    for kind in ps xa gx; do
      need_arm "$cell" "${kind}${N}" perm  "final_${kind}${N}" "" npz-wait
      need_arm "$cell" "${kind}${N}" gated "final_${kind}${N}" "" npz-wait
    done
  done
done
for cell in $BREADS; do
  for N in 15 30 60; do
    need_arm "$cell" "gx$N" perm  "final_gx$N" "" npz-wait
    need_arm "$cell" "gx$N" gated "final_gx$N" "" npz-wait
  done
done

echo "[queue] $(wc -l < "$Q") jobs:"
awk -F'|' '{print $7}' "$Q" | sort | uniq -c

#!/bin/bash
# 원샷 waiter: apple 4 cell fit 데이터(각 60판) 확보 시 apple_fits.sh 실행 (성공 시 NPZ_READY까지).
# exit 0=fits+push 완료 / 2=fit 실패
set -u
source "$(dirname "${BASH_SOURCE[0]}")/pq_lib.sh"
while true; do
  ok=1
  for cell in $APPLES; do
    [ "$(fit_count "$cell")" -ge 60 ] || { ok=0; break; }
  done
  if [ "$ok" = 1 ]; then
    echo "[waiter] A-S1 완료 감지 → fits 시작 $(date -u '+%FT%T')"
    bash "$(dirname "${BASH_SOURCE[0]}")/apple_fits.sh" && exit 0 || exit 2
  fi
  sleep 300
done

#!/usr/bin/env bash
# GPU lease 래퍼 — 규약: docs/05_gpu_server_rules.md §2
# usage: with_gpu_lease.sh <machine> "<gpu ...>" <session> <purpose> -- <command...>
# 지정 GPU 전부 claim(하나라도 타 세션 점유면 exit 3, 이미 잡은 것 되돌림) → command 실행 →
# 정상/에러/kill 어느 종료에서도 release. lease pid = 이 래퍼(=오케스트레이터 수명).
set -u
[ $# -ge 6 ] || { echo "usage: with_gpu_lease.sh <machine> \"<gpu ...>\" <session> <purpose> -- <command...>" >&2; exit 2; }
M=$1; GPUS=$2; S=$3; P=$4; shift 4
[ "$1" = "--" ] || { echo "'--' 뒤에 실행 커맨드" >&2; exit 2; }; shift
L=$(cd "$(dirname "$0")" && pwd)/gpu_lease.sh
held=()
release_all() { for g in "${held[@]}"; do "$L" release "$M" "$g" "$S" >/dev/null 2>&1 || true; done; }
trap 'release_all' EXIT
for g in $GPUS; do
  if LEASE_PID=$$ "$L" claim "$M" "$g" "$S" "$P"; then held+=("$g")
  else echo "[with_gpu_lease] ${M} gpu${g} 점유 — 발사 중단(잡은 것 되돌림)" >&2; exit 3; fi
done
"$@"; rc=$?
echo "[with_gpu_lease] 종료 rc=$rc → release ${M} gpu ${held[*]}"
exit $rc

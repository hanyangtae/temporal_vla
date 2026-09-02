#!/usr/bin/env bash
# GPU 세션 간 예약 원장 — 규약: docs/05_gpu_server_rules.md §2
# usage:
#   gpu_lease.sh status
#   gpu_lease.sh claim   <machine> <gpu> <session> <purpose> [ttl_h=12]
#   gpu_lease.sh wait    <machine> <gpu> <session> <purpose> [ttl_h=12] [poll_s=60]
#   gpu_lease.sh release <machine> <gpu> <session>
# machine ∈ kanu|srv48|srv50 . 원장 = <repo>/outputs/gpu_leases/<machine>_gpu<N>/ (mkdir 원자성)
# exit: 0 ok · 2 인자오류 · 3 타 세션 점유
# LEASE_PID=<오케스트레이터 pid> 를 주면 그 pid 사망 시 stale 자동 해제, 없으면 ttl 로만 해제
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
ROOT=$REPO/outputs/gpu_leases
mkdir -p "$ROOT"
cmd=${1:-status}

lease_dir() { echo "$ROOT/${1}_gpu${2}"; }
alive() {  # pid 살아있나 (pid 미기록이면 ttl 로만 판정)
  [ -z "${1:-}" ] || [ "$1" = none ] || kill -0 "$1" 2>/dev/null
}
is_stale() {  # dir → 0 이면 stale
  local d=$1; [ -f "$d/meta" ] || return 0
  local pid ttl start; pid=$(sed -n 's/^pid=//p' "$d/meta"); ttl=$(sed -n 's/^ttl_s=//p' "$d/meta"); start=$(sed -n 's/^start=//p' "$d/meta")
  local now; now=$(date +%s)
  if ! alive "$pid"; then return 0; fi
  [ -n "$ttl" ] && [ -n "$start" ] && [ $((now - start)) -gt "$ttl" ] && return 0
  return 1
}
show() {
  local d=$1; local m; m=$(basename "$d")
  if is_stale "$d"; then echo "$m  (stale — 자동 해제 대상)"; return; fi
  printf '%-14s %s\n' "$m" "$(tr '\n' ' ' < "$d/meta")"
}

case "$cmd" in
  status)
    found=0
    for d in "$ROOT"/*_gpu*; do [ -d "$d" ] || continue; found=1; show "$d"; done
    [ "$found" = 1 ] || echo "(lease 없음)"
    ;;
  claim)
    [ $# -ge 5 ] || { echo "usage: claim <machine> <gpu> <session> <purpose> [ttl_h]" >&2; exit 2; }
    m=$2; g=$3; s=$4; p=$5; ttl_h=${6:-12}
    d=$(lease_dir "$m" "$g")
    if [ -d "$d" ] && is_stale "$d"; then echo "[lease] stale 해제: $(basename "$d")"; rm -rf "$d"; fi
    if mkdir "$d" 2>/dev/null; then
      printf 'machine=%s\ngpu=%s\nsession=%s\npurpose=%s\npid=%s\nstart=%s\nttl_s=%s\nsince=%s\n' \
        "$m" "$g" "$s" "$p" "${LEASE_PID:-none}" "$(date +%s)" "$((ttl_h*3600))" "$(date '+%F %T')" > "$d/meta"
      echo "[lease] claimed ${m} gpu${g} ← ${s} (${p}, ttl ${ttl_h}h)"
    else
      echo "[lease] BUSY ${m} gpu${g}: $(tr '\n' ' ' < "$d/meta")" >&2
      exit 3
    fi
    ;;
  wait)
    [ $# -ge 5 ] || { echo "usage: wait <machine> <gpu> <session> <purpose> [ttl_h] [poll_s]" >&2; exit 2; }
    poll=${7:-60}
    until "$0" claim "$2" "$3" "$4" "$5" "${6:-12}" 2>/dev/null; do
      echo "[lease] ${2} gpu${3} 점유 중 — ${poll}s 후 재시도 ($(date '+%T'))"; sleep "$poll"
    done
    ;;
  release)
    [ $# -ge 4 ] || { echo "usage: release <machine> <gpu> <session>" >&2; exit 2; }
    d=$(lease_dir "$2" "$3")
    [ -d "$d" ] || { echo "[lease] 없음: ${2} gpu${3}"; exit 0; }
    owner=$(sed -n 's/^session=//p' "$d/meta")
    if [ "$owner" != "$4" ] && ! is_stale "$d"; then
      echo "[lease] 거부: ${2} gpu${3} 소유자=${owner} (요청=${4})" >&2; exit 3
    fi
    rm -rf "$d"; echo "[lease] released ${2} gpu${3}"
    ;;
  *) echo "unknown cmd: $cmd" >&2; exit 2 ;;
esac

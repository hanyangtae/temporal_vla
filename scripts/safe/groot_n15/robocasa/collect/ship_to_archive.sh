#!/usr/bin/env bash
# 롤링 shipper — 로컬 staging 의 완성된 좌표 셀을 승준 아카이브로 보내고 로컬을 비운다.
# docs/04 §7.5 (전송·삭제 절차: 이름 세기 금지 · 종류 골라 include 금지 · HDD 로만).
#
# 발사 (수집과 나란히, 장시간이므로 setsid nohup):
#
#   GRID_ROOT=/home/dongkyu/pkt_ws/temporal_vla/outputs/collect/grid_staging \
#   INTERVAL=300 STAGING_CAP_GB=50 \
#   setsid nohup bash scripts/safe/groot_n15/robocasa/collect/ship_to_archive.sh \
#     > outputs/collect/logs/shipper.log 2>&1 < /dev/null &
#
# 전송 완료 셀 키는 ${GRID_ROOT}/shipped_cells.txt 에 append 된다 —
# collect_grid.sh 의 DONE_LIST 가 이 파일을 읽어 재개한다.
# 셀 키 형식은 src/collect/plan.py GridCell.key ("instruction|s<i>|n<j>") 와 같다.
set -uo pipefail

: "${GRID_ROOT:?GRID_ROOT 필요}"
GRID_ROOT="$(realpath -m -- "$GRID_ROOT")"
SSH_PORT="${SSH_PORT:-11112}"
DEST_HOST="${DEST_HOST:-kimseungjun@166.104.146.37}"
DEST_ROOT="${DEST_ROOT:-/home/kimseungjun/datasets/temporal_vla_store/groot/n15/grid}"
DEST="${DEST:-${DEST_HOST}:${DEST_ROOT}}"
INTERVAL="${INTERVAL:-300}"
STAGING_CAP_GB="${STAGING_CAP_GB:-50}"
QUIET_SEC="${QUIET_SEC:-60}"      # 수집 중인 셀 보호: 마지막 수정 후 이 초만큼 조용해야 전송
DRY_RUN="${DRY_RUN:-0}"
ONESHOT="${ONESHOT:-0}"           # 1 이면 한 사이클만 돌고 종료 (검증용)

SHIPPED="${GRID_ROOT}/shipped_cells.txt"
LOGDIR="${LOGDIR:-${GRID_ROOT}/logs}"
mkdir -p "$LOGDIR" "$GRID_ROOT"
touch "$SHIPPED"

SSH_CMD="ssh -p ${SSH_PORT}"
log() { echo "[ship] $(date '+%F %T') $*"; }

run_ssh() {  # 원격 명령. DRY_RUN 이면 echo 만.
  if [ "$DRY_RUN" = "1" ]; then echo "[dry] ${SSH_CMD} ${DEST_HOST} $*" >&2; return 0; fi
  # shellcheck disable=SC2086
  $SSH_CMD "$DEST_HOST" "$@"
}

# 셀 디렉토리(= meta.json 을 가진 디렉토리)의 상대경로 → 셀 키.
# 레이아웃: <plan_id>/<machine>/<instruction...>/s<i>/n<j>/<arm>
cell_key_of() {
  local rel="$1"
  local IFS='/'
  read -r -a c <<< "$rel"
  local n=${#c[@]}
  [ "$n" -ge 6 ] || return 1
  local ni="${c[$((n - 2))]}" si="${c[$((n - 3))]}"
  local instr=""
  local i
  for ((i = 2; i < n - 3; i++)); do
    instr="${instr:+${instr}/}${c[$i]}"
  done
  printf '%s|%s|%s\n' "$instr" "$si" "$ni"
}

local_stats() {  # dir → "count bytes" (실물 파일만, 심링크 제외)
  find "$1" -type f -printf '%s\n' 2>/dev/null \
    | awk '{n++; s+=$1} END {printf "%d %d\n", n+0, s+0}'
}

ship_cell() {  # host_cell_dir rel
  local dir="$1" rel="$2"
  local key; key="$(cell_key_of "$rel")" || { log "SKIP 레이아웃 이상: ${rel}"; return 1; }
  local lc lb; read -r lc lb <<< "$(local_stats "$dir")"
  [ "${lc:-0}" -gt 0 ] || { log "SKIP 빈 셀: ${rel}"; return 1; }

  log "SEND ${rel} (${lc} files, ${lb} bytes) key=${key}"
  if [ "$DRY_RUN" = "1" ]; then
    echo "[dry] ${SSH_CMD} ${DEST_HOST} mkdir -p ${DEST_ROOT}/${rel}"
    echo "[dry] rsync -a --partial -e \"${SSH_CMD}\" ${dir}/ ${DEST}/${rel}/"
    echo "[dry] verify + rsync --remove-source-files + prune empty dirs"
    echo "$key" >> "$SHIPPED"
    return 0
  fi

  run_ssh "mkdir -p '${DEST_ROOT}/${rel}'" || { log "FAIL mkdir ${rel}"; return 1; }
  rsync -a --partial -e "$SSH_CMD" "${dir}/" "${DEST}/${rel}/" || {
    log "FAIL rsync ${rel} — 로컬 유지"; return 1; }

  # 대조: 원격 실물 개수·바이트 (이름만 세지 않는다 — docs/04 §7.5)
  local remote; remote="$($SSH_CMD "$DEST_HOST" \
    "find '${DEST_ROOT}/${rel}' -type f -printf '%s\n' 2>/dev/null | awk '{n++; s+=\$1} END {printf \"%d %d\n\", n+0, s+0}'")"
  local rc rb; read -r rc rb <<< "$remote"
  if [ "${rc:-0}" != "$lc" ] || [ "${rb:-0}" != "$lb" ]; then
    log "MISMATCH ${rel}: local=${lc}/${lb} remote=${rc:-?}/${rb:-?} — 로컬 삭제 안 함"
    return 1
  fi

  # 대조 통과 → 로컬 비움 (전송된 파일만 삭제 + 빈 디렉토리 정리)
  rsync -a --remove-source-files -e "$SSH_CMD" "${dir}/" "${DEST}/${rel}/" || {
    log "WARN ${rel}: 삭제-패스 rsync 실패 — 로컬 유지"; return 1; }
  find "$dir" -type d -empty -delete 2>/dev/null || true
  echo "$key" >> "$SHIPPED"
  log "OK ${rel} → shipped (${lc} files, ${lb} bytes)"
  return 0
}

cycle() {
  local now cutoff n_sent=0 n_skip=0
  now=$(date +%s); cutoff=$((now - QUIET_SEC))
  # meta.json 이 있는 디렉토리만 = 쓰기가 끝난 셀
  while IFS= read -r meta; do
    local dir rel newest
    dir="$(dirname -- "$meta")"
    rel="${dir#"${GRID_ROOT}"/}"
    [ "$rel" != "$dir" ] || continue
    # ★ 수집 중 보호: 셀 안 최신 mtime 이 QUIET_SEC 이내면 건드리지 않는다
    newest=$(find "$dir" -type f -printf '%T@\n' 2>/dev/null | sort -nr | head -1)
    newest=${newest%%.*}
    if [ -z "$newest" ] || [ "$newest" -gt "$cutoff" ]; then
      n_skip=$((n_skip + 1)); continue
    fi
    # < /dev/null 필수: ship_cell 의 ssh/rsync 가 stdin 을 먹으면 while-read 의
    # 셀 목록 파이프가 소진되어 사이클당 1셀만 전송된다 (pdk 61셀 적체 실측).
    if ship_cell "$dir" "$rel" < /dev/null; then n_sent=$((n_sent + 1)); fi
  done < <(find "$GRID_ROOT" -type f -name meta.json 2>/dev/null | sort)

  local used_kb used_gb
  used_kb=$(du -sk "$GRID_ROOT" 2>/dev/null | awk '{print $1}')
  used_gb=$(awk -v k="${used_kb:-0}" 'BEGIN {printf "%.2f", k/1048576}')
  log "cycle: sent=${n_sent} skip(collecting)=${n_skip} staging=${used_gb}GB cap=${STAGING_CAP_GB}GB"
  if awk -v u="$used_gb" -v c="$STAGING_CAP_GB" 'BEGIN {exit !(u > c)}'; then
    log "WARN staging ${used_gb}GB > cap ${STAGING_CAP_GB}GB — 전송이 수집을 못 따라간다 (여기선 로그만)"
  fi
}

log "start GRID_ROOT=${GRID_ROOT} DEST=${DEST} interval=${INTERVAL}s cap=${STAGING_CAP_GB}GB quiet=${QUIET_SEC}s dry=${DRY_RUN}"
while true; do
  cycle
  [ "$ONESHOT" = "1" ] && { log "ONESHOT 종료"; break; }
  sleep "$INTERVAL"
done

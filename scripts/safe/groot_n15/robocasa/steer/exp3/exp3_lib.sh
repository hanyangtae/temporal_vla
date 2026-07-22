#!/bin/bash
# exp3(구 pq3) 공용 라이브러리 — cell 테이블은 queue/queue_lib.sh CELLS(row_of) 재사용, 큐는 exp3 전용.
# 큐 행 형식(KEY=VAL 공백연결 + "|retry"):
#   CELL=<cell> TAG=<arm_tag> MODE=<base|perm|gated|null> NPZ=<repo상대|-> LAYERS=<L|->
#     BETA=<b|-> HOST=<local|srv50|srv48|any>  (구 라벨 w2/w3/w48 도 읽기 호환 — normalize_host)
# 계획서 v9 §E: host 균형 배정 — HOST 필드를 lane 이 필터, 집계가 ledger 로 균형 assert.
EXP3_REPO=${EXP3_REPO:-/home/dongkyu/pkt_ws/temporal_vla}
# 아래 outputs 경로 default 는 구명(pq3) 유지 — 원격 워커(srv50/srv48, 구 w2/w48) dir 이 구명이고, 로컬은 symlink 로 신구 동치
EXP3_SEROOT=${EXP3_SEROOT:-outputs/eval/robocasa/groot_n15/steer_eval_pq3}
EXP3_FITROOT=${EXP3_FITROOT:-outputs/eval/robocasa/groot_n15/phase_event_pq3/raw_rollouts}
EXP3_MANIROOT=${EXP3_MANIROOT:-outputs/eval/robocasa/groot_n15/steer_eval_pq3/manifests}
EXP3_QROOT=${EXP3_QROOT:-$EXP3_REPO/outputs/eval/robocasa/groot_n15/steer_eval_pq3/work_queue}

# shellcheck source=../queue/queue_lib.sh
source "$EXP3_REPO/scripts/safe/groot_n15/robocasa/steer/queue/queue_lib.sh"

exp3_manifest_of() { # cell kind(collect_plan|eval_manifest|sweep_manifest) → 경로
  echo "$EXP3_REPO/$EXP3_MANIROOT/$1/$2.tsv"
}

# exp3_row_of <cell> — queue_lib CELLS 우선, 없으면 C0 산출 신규 cells-config 에서 파생
# (Gate 2 높음#13: C0 신규 2 cell 이 실행 테이블에 연결되도록). 실패 시 rc=1·빈 출력.
exp3_row_of() {
  local r cfg
  r=$(row_of "$1")
  if [ -n "$r" ]; then printf '%s\n' "$r"; return 0; fi
  cfg="$EXP3_REPO/configs/robocasa/exp3_ppcc_new_cells.tsv"
  [ -f "$cfg" ] || return 1
  # 열: cell_index cell_id task env_name canonical_instruction target_episodes seed_start
  r=$(awk -F'\t' -v c="$1" 'NR>1 && $2==c { print $2 "|" $3 "|" $4 "|" $1 "|100000|" $5; exit }' "$cfg")
  [ -n "$r" ] || return 1
  printf '%s\n' "$r"
}

# normalize_host <label> → srv50/srv48 정규명 (구 라벨 w2→srv50, w3/w48→srv48; 그 외는 그대로)
# — 큐 tsv 에 구명 HOST 값이 남아있는 하위호환 read 용.
normalize_host() {
  case "$1" in
    w2) echo srv50 ;;
    w3|w48) echo srv48 ;;
    *) echo "$1" ;;
  esac
}

# pop_exp3 <host_class> <lane_id> → stdout 행, 없으면 rc=1
# HOST 필드가 비었거나 any 이거나 host_class 와 일치하는 첫 행을 pop.
# (HOST 필드·host_class 모두 normalize_host 로 정규화 후 비교 — 구 큐 행의 w2/w48 라벨도 매치)
pop_exp3() {
  local cls=$1 lane=$2 n line
  cls=$(normalize_host "$cls")
  (
    flock 9
    n=$(awk -v cls="$cls" '
      function normhost(h) {
        if (h == "w2") return "srv50"
        if (h == "w3" || h == "w48") return "srv48"
        return h
      }
      {
      hv = ""
      if (match($0, /HOST=[^ |]*/)) hv = normhost(substr($0, RSTART + 5, RLENGTH - 5))
      if (hv == "" || hv == "any" || hv == cls) { print NR; exit }
    }' "$EXP3_QROOT/queue.tsv" 2>/dev/null)
    [ -z "$n" ] && exit 1
    line=$(sed -n "${n}p" "$EXP3_QROOT/queue.tsv")
    sed -i "${n}d" "$EXP3_QROOT/queue.tsv"
    printf '%s\n' "$line" > "$EXP3_QROOT/running/${lane}.job"
    printf '%s\n' "$line"
  ) 9>>"$EXP3_QROOT/lock"
}

requeue_exp3() { # line lane
  local line=$1 lane=$2 body retry
  body=${line%|*}; retry=${line##*|}; retry=$((retry + 1))
  (
    flock 9
    if [ "$retry" -le 2 ]; then
      printf '%s|%s\n' "$body" "$retry" >> "$EXP3_QROOT/queue.tsv"
    else
      printf '%s RETRY_EXCEEDED %s\n' "$line" "$(date -u '+%FT%T')" >> "$EXP3_QROOT/failed/failed.tsv"
    fi
    rm -f "$EXP3_QROOT/running/${lane}.job"
  ) 9>>"$EXP3_QROOT/lock"
}

finish_exp3() { # line lane machine count
  local line=$1 lane=$2 machine=$3 cnt=$4
  (
    flock 9
    printf '%s|machine=%s|count=%s|%s\n' "$line" "$machine" "$cnt" "$(date -u '+%FT%T')" >> "$EXP3_QROOT/done/done.tsv"
    printf '%s\t%s\t%s\t%s\n' "$(date -u '+%FT%T')" "$machine" "$line" "$cnt" >> "$EXP3_QROOT/ledger.tsv"
    rm -f "$EXP3_QROOT/running/${lane}.job"
  ) 9>>"$EXP3_QROOT/lock"
}

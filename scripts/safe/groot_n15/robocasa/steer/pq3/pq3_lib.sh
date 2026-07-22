#!/bin/bash
# pq3 공용 라이브러리 — cell 테이블은 pq/pq_lib.sh CELLS(row_of) 재사용, 큐는 pq3 전용.
# 큐 행 형식(KEY=VAL 공백연결 + "|retry"):
#   CELL=<cell> TAG=<arm_tag> MODE=<base|perm|gated|null> NPZ=<repo상대|-> LAYERS=<L|->
#     BETA=<b|-> HOST=<local|w2|w48|any>
# 계획서 v9 §E: host 균형 배정 — HOST 필드를 lane 이 필터, 집계가 ledger 로 균형 assert.
PQ3_REPO=${PQ3_REPO:-/home/dongkyu/pkt_ws/temporal_vla}
PQ3_SEROOT=${PQ3_SEROOT:-outputs/eval/robocasa/groot_n15/steer_eval_pq3}
PQ3_FITROOT=${PQ3_FITROOT:-outputs/eval/robocasa/groot_n15/phase_event_pq3/raw_rollouts}
PQ3_MANIROOT=${PQ3_MANIROOT:-outputs/eval/robocasa/groot_n15/steer_eval_pq3/manifests}
PQ3_QROOT=${PQ3_QROOT:-$PQ3_REPO/outputs/eval/robocasa/groot_n15/steer_eval_pq3/work_queue}

# shellcheck source=../pq/pq_lib.sh
source "$PQ3_REPO/scripts/safe/groot_n15/robocasa/steer/pq/pq_lib.sh"

pq3_manifest_of() { # cell kind(collect_plan|eval_manifest|sweep_manifest) → 경로
  echo "$PQ3_REPO/$PQ3_MANIROOT/$1/$2.tsv"
}

# pq3_row_of <cell> — pq_lib CELLS 우선, 없으면 C0 산출 신규 cells-config 에서 파생
# (Gate 2 높음#13: C0 신규 2 cell 이 실행 테이블에 연결되도록). 실패 시 rc=1·빈 출력.
pq3_row_of() {
  local r cfg
  r=$(row_of "$1")
  if [ -n "$r" ]; then printf '%s\n' "$r"; return 0; fi
  cfg="$PQ3_REPO/configs/robocasa/pq3_ppcc_new_cells.tsv"
  [ -f "$cfg" ] || return 1
  # 열: cell_index cell_id task env_name canonical_instruction target_episodes seed_start
  r=$(awk -F'\t' -v c="$1" 'NR>1 && $2==c { print $2 "|" $3 "|" $4 "|" $1 "|100000|" $5; exit }' "$cfg")
  [ -n "$r" ] || return 1
  printf '%s\n' "$r"
}

# pop_pq3 <host_class> <lane_id> → stdout 행, 없으면 rc=1
# HOST 필드가 비었거나 any 이거나 host_class 와 일치하는 첫 행을 pop.
pop_pq3() {
  local cls=$1 lane=$2 n line
  (
    flock 9
    n=$(awk -v cls="$cls" '{
      hv = ""
      if (match($0, /HOST=[^ |]*/)) hv = substr($0, RSTART + 5, RLENGTH - 5)
      if (hv == "" || hv == "any" || hv == cls) { print NR; exit }
    }' "$PQ3_QROOT/queue.tsv" 2>/dev/null)
    [ -z "$n" ] && exit 1
    line=$(sed -n "${n}p" "$PQ3_QROOT/queue.tsv")
    sed -i "${n}d" "$PQ3_QROOT/queue.tsv"
    printf '%s\n' "$line" > "$PQ3_QROOT/running/${lane}.job"
    printf '%s\n' "$line"
  ) 9>>"$PQ3_QROOT/lock"
}

requeue_pq3() { # line lane
  local line=$1 lane=$2 body retry
  body=${line%|*}; retry=${line##*|}; retry=$((retry + 1))
  (
    flock 9
    if [ "$retry" -le 2 ]; then
      printf '%s|%s\n' "$body" "$retry" >> "$PQ3_QROOT/queue.tsv"
    else
      printf '%s RETRY_EXCEEDED %s\n' "$line" "$(date -u '+%FT%T')" >> "$PQ3_QROOT/failed/failed.tsv"
    fi
    rm -f "$PQ3_QROOT/running/${lane}.job"
  ) 9>>"$PQ3_QROOT/lock"
}

finish_pq3() { # line lane machine count
  local line=$1 lane=$2 machine=$3 cnt=$4
  (
    flock 9
    printf '%s|machine=%s|count=%s|%s\n' "$line" "$machine" "$cnt" "$(date -u '+%FT%T')" >> "$PQ3_QROOT/done/done.tsv"
    printf '%s\t%s\t%s\t%s\n' "$(date -u '+%FT%T')" "$machine" "$line" "$cnt" >> "$PQ3_QROOT/ledger.tsv"
    rm -f "$PQ3_QROOT/running/${lane}.job"
  ) 9>>"$PQ3_QROOT/lock"
}

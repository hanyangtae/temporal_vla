#!/bin/bash
# 양-머신 병렬 arm 큐 공용 라이브러리 (2026-07-07 전환, plan: elegant-bouncing-possum)
# 큐 행 형식: jobtype|cell|suf|arms|npzsub|layers|constraint|retry
#   jobtype: arm | collect / arms: base|perm|gated 중 1개(단일 arm 단위)
#   constraint: "" (아무 lane) | local-only (수집·hole-resume) | npz-wait (A-S2 완료 대기)
REPO=/home/dongkyu/pkt_ws/temporal_vla
QROOT=$REPO/outputs/eval/robocasa/groot_n15/work_queue
SRC6=outputs/eval/robocasa/groot_n15/phase_event_6p/raw_rollouts
SE=outputs/eval/robocasa/groot_n15/steer_eval
AN=outputs/eval/robocasa/groot_n15/phase_event_6p/analysis
W2=junhyeong@166.104.35.50
W2REPO='~/pkt_ws/temporal_vla'

CELLS=(
  "ppcc_bread|PickPlaceCounterToCabinet|robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env|5|100084|Pick the bread from the counter and place it in the cabinet."
  "ppcc_bread_s300028|PickPlaceCounterToCabinet|robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env|5|300028|Pick the bread from the counter and place it in the cabinet."
  "ppcc_bread_s300033|PickPlaceCounterToCabinet|robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env|5|300033|Pick the bread from the counter and place it in the cabinet."
  "ppcc_bread_s400020|PickPlaceCounterToCabinet|robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env|5|400020|Pick the bread from the counter and place it in the cabinet."
  "ppcs_apple|PickPlaceCounterToStove|robocasa_panda_omron/PickPlaceCounterToStove_PandaOmron_Env|1|100074|Pick the apple from the plate and place it in the pan."
  "ppcs_apple_s100050|PickPlaceCounterToStove|robocasa_panda_omron/PickPlaceCounterToStove_PandaOmron_Env|1|100050|Pick the apple from the plate and place it in the pan."
  "ppcs_apple_s100084|PickPlaceCounterToStove|robocasa_panda_omron/PickPlaceCounterToStove_PandaOmron_Env|1|100084|Pick the apple from the plate and place it in the pan."
  "ppcs_apple_s100104|PickPlaceCounterToStove|robocasa_panda_omron/PickPlaceCounterToStove_PandaOmron_Env|1|100104|Pick the apple from the plate and place it in the pan."
  # --- pq2 P0 apple 신규 scene 후보 (2026-07-10, seed 순 — selected_instruction_seeds.tsv 중 미사용 상위 6) ---
  "ppcs_apple_s100106|PickPlaceCounterToStove|robocasa_panda_omron/PickPlaceCounterToStove_PandaOmron_Env|1|100106|Pick the apple from the plate and place it in the pan."
  "ppcs_apple_s100172|PickPlaceCounterToStove|robocasa_panda_omron/PickPlaceCounterToStove_PandaOmron_Env|1|100172|Pick the apple from the plate and place it in the pan."
  "ppcs_apple_s100184|PickPlaceCounterToStove|robocasa_panda_omron/PickPlaceCounterToStove_PandaOmron_Env|1|100184|Pick the apple from the plate and place it in the pan."
  "ppcs_apple_s100202|PickPlaceCounterToStove|robocasa_panda_omron/PickPlaceCounterToStove_PandaOmron_Env|1|100202|Pick the apple from the plate and place it in the pan."
  "ppcs_apple_s100215|PickPlaceCounterToStove|robocasa_panda_omron/PickPlaceCounterToStove_PandaOmron_Env|1|100215|Pick the apple from the plate and place it in the pan."
  "ppcs_apple_s100243|PickPlaceCounterToStove|robocasa_panda_omron/PickPlaceCounterToStove_PandaOmron_Env|1|100243|Pick the apple from the plate and place it in the pan."
)
BREADS="ppcc_bread ppcc_bread_s300028 ppcc_bread_s300033 ppcc_bread_s400020"
APPLES="ppcs_apple ppcs_apple_s100050 ppcs_apple_s100084 ppcs_apple_s100104"
APPLES_P0_CAND="ppcs_apple_s100106 ppcs_apple_s100172 ppcs_apple_s100184 ppcs_apple_s100202 ppcs_apple_s100215 ppcs_apple_s100243"
row_of() { local c; for c in "${CELLS[@]}"; do [[ "$c" == "$1|"* ]] && { echo "$c"; return; }; done; }
tag_of() { case "$1" in base) echo ho_base;; perm) echo "ho_perm$2";; gated) echo "ho_gated$2";; esac; }

pkl_count() { # cell tag → eval arm pkl 수
  local row; row=$(row_of "$1"); local task; task=$(echo "$row" | cut -d'|' -f2)
  find "$REPO/$SE/$1/$2/raw_rollouts/$task/$1" -name '*.pkl' 2>/dev/null | wc -l
}
fit_count() { # cell → fit 수집 pkl 수
  local row; row=$(row_of "$1"); local task; task=$(echo "$row" | cut -d'|' -f2)
  find "$REPO/$SRC6/$task/$1" -maxdepth 1 -name '*.pkl' 2>/dev/null | wc -l
}

# pop_job <class:local|w2> <lane_id> → stdout에 행, 없으면 rc=1
pop_job() {
  local cls=$1 lane=$2 n line gate=0
  [ -f "$QROOT/NPZ_READY" ] && gate=1
  (
    flock 9
    n=$(awk -F'|' -v cls="$cls" -v gate="$gate" '
      { c=$7 }
      c=="" || (c=="local-only" && cls=="local") || (c=="npz-wait" && gate==1) { print NR; exit }
    ' "$QROOT/queue.tsv")
    [ -z "$n" ] && exit 1
    line=$(sed -n "${n}p" "$QROOT/queue.tsv")
    sed -i "${n}d" "$QROOT/queue.tsv"
    printf '%s\n' "$line" > "$QROOT/running/${lane}.job"
    printf '%s\n' "$line"
  ) 9>>"$QROOT/lock"
}

requeue_job() { # line lane [force_constraint]
  local line=$1 lane=$2 fc=${3:-}
  IFS='|' read -r jt cell suf arms npzsub layers cons retry <<<"$line"
  retry=$((${retry:-0}+1))
  [ -n "$fc" ] && cons=$fc
  (
    flock 9
    if [ "$retry" -le 2 ]; then
      printf '%s|%s|%s|%s|%s|%s|%s|%s\n' "$jt" "$cell" "$suf" "$arms" "$npzsub" "$layers" "$cons" "$retry" >> "$QROOT/queue.tsv"
    else
      printf '%s  RETRY_EXCEEDED %s\n' "$line" "$(date -u '+%FT%T')" >> "$QROOT/failed/failed.tsv"
    fi
    rm -f "$QROOT/running/${lane}.job"
  ) 9>>"$QROOT/lock"
}

finish_job() { # line lane machine eps_added count
  local line=$1 lane=$2 machine=$3 added=$4 cnt=$5
  (
    flock 9
    printf '%s|machine=%s|eps_added=%s|count=%s|%s\n' "$line" "$machine" "$added" "$cnt" "$(date -u '+%FT%T')" >> "$QROOT/done/done.tsv"
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date -u '+%FT%T')" "$machine" "$line" "$added" "$cnt" >> "$QROOT/ledger.tsv"
    rm -f "$QROOT/running/${lane}.job"
  ) 9>>"$QROOT/lock"
}

mark_machine() { # dir machine eps_added  (데이터 출처 영구 기록 — 사용자 요구)
  mkdir -p "$1"
  printf '%s machine=%s eps_added=%s\n' "$(date -u '+%FT%T')" "$2" "$3" >> "$1/MACHINE.txt"
}

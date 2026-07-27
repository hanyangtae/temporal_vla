#!/bin/bash
# exp5-3 mixer scene-matched 배송 — **승준 노드에서 실행** (home→승준 직결 불가라 pull 방식).
# 5분 주기: home 산출물 rsync pull (전 종류 — 아카이브 완전성 규칙, 3분 경과분 = 쓰기 완료).
# ★home 원본은 삭제하지 않는다 (2026-07-27 사용자 지시 — 복사만). mixer_sm.DONE 후 최종 pull.
set -u
HOME_HOST="rudxo@218.152.144.220"; HP=11111
SRC="workspace/temporal_vla/outputs/eval/robocasa/groot_n15/exp5_3_mixer_sm"
DEST="$HOME/datasets/temporal_vla_outputs/eval/robocasa/groot_n15/exp5_3_mixer_sm"
LOG="$HOME/mixer_sm_pull.log"
mkdir -p "$DEST"

pull_once() {
  # 1) pkl 제외 전체 동기 (csv/json/mp4/annot/tsv/MACHINE.txt)
  rsync -a --partial -e "ssh -p $HP -o BatchMode=yes" \
    --exclude '*.pkl' --exclude '.groot_video_tmp/' \
    "$HOME_HOST:$SRC/" "$DEST/" >> "$LOG" 2>&1
  # 2) 완료 pkl (mmin+3) pull — ★home 원본 보존 (복사만, 삭제 금지)
  local lst
  lst=$(ssh -p $HP -o BatchMode=yes "$HOME_HOST" \
        "cd $SRC && find . -name '*.pkl' -mmin +3" 2>/dev/null)
  [ -z "$lst" ] && return 0
  while IFS= read -r f; do
    [ -e "$DEST/$f" ] && continue
    rsync -a --partial -e "ssh -p $HP -o BatchMode=yes" \
      "$HOME_HOST:$SRC/$f" "$DEST/$f" >> "$LOG" 2>&1 || continue
    echo "[pull] $f ($(stat -c%s "$DEST/$f" 2>/dev/null) B) $(date -u +%T)" >> "$LOG"
  done <<< "$lst"
}

echo "=== mixer_sm puller 시작 $(date -u +%FT%T) ===" >> "$LOG"
while ! ssh -p $HP -o BatchMode=yes "$HOME_HOST" "test -f ~/mixer_sm.DONE" 2>/dev/null; do
  pull_once
  sleep 300
done
sleep 240   # 마지막 판 mmin+3 대기
pull_once
echo "=== DONE: 승준 pkl $(find "$DEST" -name '*.pkl' | wc -l) / csv $(find "$DEST" -name '*.csv' | wc -l) ===" >> "$LOG"
tail -3 "$LOG"

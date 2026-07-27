#!/bin/bash
# exp5-3 mixer scene-matched 배송 — **승준 노드에서 실행** (home→승준 직결 불가라 pull 방식).
# 5분 주기: home 산출물 rsync pull (전 종류 — 아카이브 완전성 규칙) → 크기 검증된 pkl 만
# home 에서 삭제 (3분 이상 경과분 = 쓰기 완료). mixer_sm.DONE 후 최종 pull 하고 종료.
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
  # 2) 완료 pkl (mmin+3) pull → 크기 대조 → home 측 삭제
  local lst
  lst=$(ssh -p $HP -o BatchMode=yes "$HOME_HOST" \
        "cd $SRC && find . -name '*.pkl' -mmin +3" 2>/dev/null)
  [ -z "$lst" ] && return 0
  while IFS= read -r f; do
    rsync -a --partial -e "ssh -p $HP -o BatchMode=yes" \
      "$HOME_HOST:$SRC/$f" "$DEST/$f" >> "$LOG" 2>&1 || continue
    local rsz lsz
    rsz=$(ssh -p $HP -o BatchMode=yes "$HOME_HOST" "stat -c%s $SRC/$f" 2>/dev/null)
    lsz=$(stat -c%s "$DEST/$f" 2>/dev/null)
    if [ -n "$rsz" ] && [ "$rsz" = "$lsz" ]; then
      ssh -p $HP -o BatchMode=yes "$HOME_HOST" "rm $SRC/$f" 2>/dev/null
      echo "[pull+del] $f ($lsz B) $(date -u +%T)" >> "$LOG"
    else
      echo "[검증실패-보존] $f local=$lsz remote=$rsz" >> "$LOG"
    fi
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

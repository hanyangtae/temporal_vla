#!/bin/bash
# grid_v2 kanu 오염영상(caption burn-in) 클린 재생성 오케스트레이터.
# replay_orchestrate_kanu.sh(219판, 8/14)의 v2판 — 차이: 번들 소스 = 국내투고 세션의
# /tmp/kai_bundles (파일명 kanu_<instr>_s<i>_n<j>.pkl, 접두사=수집 머신), kanu분만 pull.
# 우선군: OvenRack_out 전셀 + PPCC_marshmallow (국내 투고 논문 의존 111판 포함) 먼저.
#
# 발사:
#   GPU=7 setsid nohup bash scripts/analysis/grid_phase/replay_orchestrate_v2.sh \
#     >> outputs/analysis/grid_phase/v2_replay/orchestrate.log 2>&1 < /dev/null &
# 완료 판정: V2_REPLAY_ALL_DONE 로그 + mp4 수 == 번들 수.
set -uo pipefail

REPO="${REPO:-/home/dongkyu/pkt_ws/temporal_vla}"
WORK="${WORK:-$REPO/outputs/analysis/grid_phase/v2_replay}"
BDIR="$WORK/bundles"
ODIR="$WORK/clean"
GPU="${GPU:-7}"
WORKERS="${WORKERS:-6}"
PORT="${REMOTE_PORT:-11112}"
T="${REMOTE_USER:-kimseungjun}@${REMOTE_HOST:-166.104.146.37}"
RBUNDLE="${RBUNDLE:-/tmp/kai_bundles}"
RLOG="${RLOG:-/tmp/bundle_all.log}"
EXPECT="${EXPECT:-777}"

mkdir -p "$BDIR" "$ODIR"

pull_prio() {  # 우선군 먼저
  rsync -a -e "ssh -p $PORT" --include='kanu_OvenRack_*.pkl' \
    --include='kanu_PPCC_marshmallow_*.pkl' --exclude='*' "$T:$RBUNDLE/" "$BDIR/" 2>&1 | tail -1
}
pull_all() {  # kanu분 전체
  rsync -a -e "ssh -p $PORT" --include='kanu_*.pkl' --exclude='*' \
    "$T:$RBUNDLE/" "$BDIR/" 2>&1 | tail -1
}
pass() {
  python3 "$REPO/scripts/analysis/grid_phase/replay_batch_runner.py" \
    --bundle-dir "$BDIR" --out-dir "$ODIR" --gpu "$GPU" --workers "$WORKERS"
}

echo "[orch-v2] start $(date -Is) gpu=$GPU workers=$WORKERS expect=$EXPECT"
for i in $(seq 1 300); do
  if [ "$i" -le 3 ]; then pull_prio; else pull_all; fi
  n_b=$(ls "$BDIR"/kanu_*.pkl 2>/dev/null | wc -l)
  bundles_done=$(ssh -p "$PORT" "$T" "grep -c BUNDLES_DONE $RLOG 2>/dev/null; true" | head -1)
  [[ "$bundles_done" =~ ^[0-9]+$ ]] || bundles_done=0
  echo "[orch-v2] pass=$i bundles=$n_b bundles_done=$bundles_done $(date -Is)"
  if [ "$n_b" -gt 0 ]; then pass; fi
  n_mp4=$(ls "$ODIR"/*.mp4 2>/dev/null | wc -l)
  echo "[orch-v2] pass=$i 이후 mp4=$n_mp4"
  if [ "$bundles_done" -ge 1 ] && [ "$n_b" -ge "$EXPECT" ]; then
    pass
    break
  fi
  sleep 180
done
echo "[orch-v2] 최종 mp4=$(ls "$ODIR"/*.mp4 2>/dev/null | wc -l) / 번들 $(ls "$BDIR"/kanu_*.pkl 2>/dev/null | wc -l)"
echo "V2_REPLAY_ALL_DONE"

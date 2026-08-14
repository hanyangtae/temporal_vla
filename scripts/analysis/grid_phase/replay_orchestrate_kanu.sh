#!/bin/bash
# kanu 219판 클린 영상 재생성 오케스트레이터 (로컬 kanu, setsid 분리 발사용).
#
# 원격 번들 추출(extract_bundles_kanu.sh)이 도는 동안 이미 나온 번들부터 당겨와 replay 를
# 시작하고, 추출이 끝나면 남은 번들까지 마저 처리한다. 러너는 results.jsonl 기반 resume
# 이라 패스를 반복해도 이미 성공한 판은 건너뛴다.
#
# 사용:
#   setsid nohup bash scripts/analysis/grid_phase/replay_orchestrate_kanu.sh \
#     >> outputs/analysis/grid_phase/kanu_replay/orchestrate.log 2>&1 < /dev/null &
#
# 완료 판정: 로그에 `KANU_REPLAY_ALL_DONE` (+ 직전 요약 행). 산출 mp4 수로도 교차 확인.
set -uo pipefail

REPO="${REPO:-/home/dongkyu/pkt_ws/temporal_vla/.claude/worktrees/safe-length-ablation}"
WORK="${WORK:-$REPO/outputs/analysis/grid_phase/kanu_replay}"
BDIR="$WORK/bundles"
ODIR="$WORK/clean"
GPU="${GPU:-0}"
WORKERS="${WORKERS:-6}"
PORT="${REMOTE_PORT:-11112}"
T="${REMOTE_USER:-kimseungjun}@${REMOTE_HOST:-166.104.146.37}"
RBUNDLE="${RBUNDLE:-/tmp/replay_bundles_kanu}"
RLOG="${RLOG:-/tmp/remote_compute_logs/extract_kanu.log}"
EXPECT="${EXPECT:-219}"

mkdir -p "$BDIR" "$ODIR"

pull() {  # 원격 번들 + manifest 동기화
  rsync -a -e "ssh -p $PORT" --include='*.bundle.pkl' --include='manifest.tsv' \
    --exclude='*' "$T:$RBUNDLE/" "$BDIR/" 2>&1 | tail -1
}

pass() {
  python3 "$REPO/scripts/analysis/grid_phase/replay_batch_runner.py" \
    --bundle-dir "$BDIR" --out-dir "$ODIR" --gpu "$GPU" --workers "$WORKERS"
}

echo "[orch] start $(date -Is) work=$WORK gpu=$GPU workers=$WORKERS expect=$EXPECT"

for i in $(seq 1 200); do
  pull
  n_b=$(ls "$BDIR"/*.bundle.pkl 2>/dev/null | wc -l)
  extract_done=$(ssh -p "$PORT" "$T" "grep -c '\[extract\] 총' $RLOG 2>/dev/null || echo 0")
  echo "[orch] pass=$i bundles=$n_b extract_done=$extract_done $(date -Is)"
  if [[ "$n_b" -gt 0 ]]; then
    pass
  fi
  n_mp4=$(ls "$ODIR"/*.mp4 2>/dev/null | wc -l)
  n_res=$(sort -u "$ODIR/results.jsonl" 2>/dev/null | wc -l)
  echo "[orch] pass=$i 이후 mp4=$n_mp4 result_lines=$n_res"
  if [[ "$extract_done" -ge 1 && "$n_b" -ge "$EXPECT" ]]; then
    # 추출 완료 + 전체 번들 확보 → 마지막 패스가 남은 판을 다 처리했으면 종료
    pass
    break
  fi
  sleep 120
done

echo "[orch] 최종 mp4=$(ls "$ODIR"/*.mp4 2>/dev/null | wc -l) / 번들 $(ls "$BDIR"/*.bundle.pkl 2>/dev/null | wc -l)"
echo "[orch] 교체 실행: bash $REPO/scripts/analysis/grid_phase/push_clean_videos.sh $ODIR $BDIR/manifest.tsv"
echo "KANU_REPLAY_ALL_DONE"

#!/bin/bash
# exp4-3 분리도 지도 — N1.6 승준 CPU 분석 러너.
# cell = pkl 부모 디렉토리 이름(=task). manifest 는 수집 아카이브에서 자동 생성(경로 \t succ).
# 산출: outputs/eval/robocasa/groot_n16/exp4_3/atlas/n16/<cell>.json
#       outputs/eval/robocasa/groot_n16/exp4_3/probe_whitened/<cell>.json
# usage(승준 exp4_3 worktree 에서): bash scripts/safe/exp4_3/run_atlas_n16_remote.sh [cell...]
set -u
PY="${REMOTE_PYTHON:-$HOME/anaconda3/bin/python}"
ARCH="$HOME/datasets/temporal_vla_outputs/eval/robocasa/groot_n16/exp4_3/collect"
O=outputs/eval/robocasa/groot_n16/exp4_3/atlas/n16
OP=outputs/eval/robocasa/groot_n16/exp4_3/probe_whitened
MF=outputs/eval/robocasa/groot_n16/exp4_3/atlas/n16_manifest.tsv
# 32 DiT block 중 프로파일 포착용 격자(every-other + last) — 비용 절반, peak 위치 판독 충분
LAYERS="${LAYERS:-0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,31}"
mkdir -p "$O" "$OP"

# ── manifest 자동 생성: 아카이브 전체 pkl → 경로 \t succ(파일명에서) ──
: > "$MF"
find "$ARCH" -name '*.pkl' | sort | while read -r f; do
  s=$(basename "$f" | grep -oP 'succ\K[01]')
  [ -n "$s" ] && printf '%s\t%s\n' "$f" "$s" >> "$MF"
done
echo "[manifest] $(wc -l < "$MF") rollouts → $MF"
awk -F/ '{print $(NF-1)}' "$MF" | sort | uniq -c   # cell(=parent dir)별 개수

default_cells=(OpenDrawer PickPlaceCounterToCabinet OpenStandMixerHead)
[ $# -gt 0 ] && cells=("$@") || cells=("${default_cells[@]}")
rc=0
for cell in "${cells[@]}"; do
  n=$(awk -F/ -v c="$cell" '$(NF-1)==c' "$MF" | wc -l)
  [ "$n" -eq 0 ] && { echo "[skip] $cell manifest 0행"; continue; }
  echo "=== [atlas] n16/$cell ($n rollouts) layers=$LAYERS $(date -u '+%FT%T') ==="
  "$PY" scripts/safe/exp4_3/atlas_sweep.py --model n16 --cell "$cell" \
    --manifest "$MF" --layers "$LAYERS" --out "$O/$cell.json" || { echo "[FAIL atlas] $cell"; rc=1; }
  echo "=== [probe] n16/$cell $(date -u '+%FT%T') ==="
  "$PY" scripts/safe/exp4_3/probe_whitened.py --model n16 --cell "$cell" \
    --manifest "$MF" --layers "$LAYERS" --out "$OP/$cell.json" || { echo "[FAIL probe] $cell"; rc=1; }
done
echo "=== [n16 atlas] all done rc=$rc $(date -u '+%FT%T') ==="
exit $rc

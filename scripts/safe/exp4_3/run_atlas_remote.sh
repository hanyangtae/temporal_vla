#!/bin/bash
# exp4-3 분리도 지도 — 승준 노드 CPU 분석 러너. usage: run_atlas_remote.sh <model> [cell...]
#   n15  : 기존 fit30 5셀 (bread/beer/drawer L·R/mixer). beer 는 오염 3판 제외 교정 manifest.
# 산출: outputs/eval/robocasa/groot_n15/exp4_3/atlas/<model>/<cell>.json
set -u
PY="${REMOTE_PYTHON:-$HOME/anaconda3/bin/python}"
M="$HOME/datasets/temporal_vla_outputs/eval/robocasa/groot_n15/steer_eval_pq3/manifests_fit30"
MIX="$HOME/datasets/temporal_vla_outputs/eval/robocasa/groot_n15/exp41_mixer"
MODEL="${1:?model}"; shift
O=outputs/eval/robocasa/groot_n15/exp4_3/atlas/$MODEL
rc=0
n15_cells=(pq3_ppcc_bread pq3_ppcc_beer pq3_drawer_left pq3_drawer_right exp41_mixer)
[ $# -gt 0 ] && cells=("$@") || cells=("${n15_cells[@]}")
for cell in "${cells[@]}"; do
  case "$cell" in
    pq3_ppcc_bread)  src="$M/task_PPCC_fit.tsv" ;;
    pq3_ppcc_beer)   src="$M/task_PPCC_fit_beerclean.tsv" ;;   # 오염 3판 제외판
    pq3_drawer_*)    src="$M/task_OpenDrawer_fit.tsv" ;;
    exp41_mixer)     src="$MIX/mixer_fit_manifest.tsv" ;;
    *) echo "[skip] unknown cell $cell"; continue ;;
  esac
  echo "=== [atlas] $MODEL/$cell $(date -u '+%FT%T') ==="
  "$PY" scripts/safe/exp4_3/atlas_sweep.py --model "$MODEL" --cell "$cell" \
    --manifest "$src" --out "$O/$cell.json" || { echo "[FAIL] $cell"; rc=1; }
done
echo "=== [atlas] $MODEL all done rc=$rc $(date -u '+%FT%T') ==="
exit $rc

#!/bin/bash
set -u
PY="${REMOTE_PYTHON:-$HOME/anaconda3/bin/python}"
M="$HOME/datasets/temporal_vla_outputs/eval/robocasa/groot_n15/steer_eval_pq3/manifests_fit30"
MIX="$HOME/datasets/temporal_vla_outputs/eval/robocasa/groot_n15/exp41_mixer"
O=outputs/eval/robocasa/groot_n15/exp4_3/probe_whitened
rc=0
for c in pq3_ppcc_bread:$M/task_PPCC_fit.tsv pq3_ppcc_beer:$M/task_PPCC_fit_beerclean.tsv \
         pq3_drawer_left:$M/task_OpenDrawer_fit.tsv pq3_drawer_right:$M/task_OpenDrawer_fit.tsv \
         exp41_mixer:$MIX/mixer_fit_manifest.tsv; do
  cell="${c%%:*}" src="${c##*:}"
  echo "=== [probe] $cell $(date -u '+%FT%T') ==="
  "$PY" scripts/safe/exp4_3/probe_whitened.py --model n15 --cell "$cell" \
    --manifest "$src" --out "$O/$cell.json" || { echo "[FAIL] $cell"; rc=1; }
done
echo "=== [probe] all done rc=$rc $(date -u '+%FT%T') ==="; exit $rc

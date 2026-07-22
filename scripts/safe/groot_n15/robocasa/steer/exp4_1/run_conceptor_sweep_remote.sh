#!/bin/bash
# exp4-1 A(conceptor) layer sweep 드라이버 — 승준 노드에서 실행 (run_fits_remote.sh 자매).
set -u
PY="${REMOTE_PYTHON:-$HOME/anaconda3/bin/python}"
M="$HOME/datasets/temporal_vla_outputs/eval/robocasa/groot_n15/steer_eval_pq3/manifests_fit30"
O=outputs/eval/robocasa/groot_n15/exp4_1/npz
rc=0
for c in pq3_ppcc_bread:task_PPCC_fit pq3_ppcc_beer:task_PPCC_fit \
         pq3_drawer_left:task_OpenDrawer_fit pq3_drawer_right:task_OpenDrawer_fit; do
  cell="${c%%:*}" mf="${c##*:}"
  echo "=== [conceptor_sweep] $cell $(date -u '+%FT%T') ==="
  "$PY" scripts/safe/groot_n15/robocasa/steer/exp4_1/conceptor_layer_sweep.py \
    --manifest "$M/$mf.tsv" --cell "$cell" --out "$O/$cell/A_layer_sweep.json" \
    || { echo "[FAIL] $cell"; rc=1; }
done
echo "=== [conceptor_sweep] all done rc=$rc $(date -u '+%FT%T') ==="
exit $rc

#!/bin/bash
# exp4-1 setM/setM_pl fit 드라이버 — **승준 노드에서 실행** (remote_compute.sh run-bg 경유).
# 입력: fit30 manifest(승준 datasets) + annotation_t0.tsv(push-data 로 미리 전송).
# 산출: outputs/eval/robocasa/groot_n15/exp4_1/npz/<cell>/{setM,setM_pl,setM_loo/*,setM_pl_loo/*}
set -u
PY="${REMOTE_PYTHON:-$HOME/anaconda3/bin/python}"
M="$HOME/datasets/temporal_vla_outputs/eval/robocasa/groot_n15/steer_eval_pq3/manifests_fit30"
T=outputs/eval/robocasa/groot_n15/exp4_1/annotation_t0.tsv
O=outputs/eval/robocasa/groot_n15/exp4_1/npz
rc=0
for c in pq3_ppcc_bread:task_PPCC_fit pq3_ppcc_beer:task_PPCC_fit \
         pq3_drawer_left:task_OpenDrawer_fit pq3_drawer_right:task_OpenDrawer_fit; do
  cell="${c%%:*}" mf="${c##*:}"
  echo "=== [fit_setm] $cell ($mf) $(date -u '+%FT%T') ==="
  "$PY" scripts/safe/groot_n15/robocasa/steer/exp4_1/fit_mean_diff.py \
    --manifest "$M/$mf.tsv" --cell "$cell" --targets "$T" --out-root "$O" \
    || { echo "[FAIL] $cell"; rc=1; }
done
echo "=== [fit_setm] all done rc=$rc $(date -u '+%FT%T') ==="
exit $rc

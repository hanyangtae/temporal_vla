#!/bin/bash
# exp4-1 gated fit 드라이버 (승준) — setM_gated(+placebo) & conceptor_gated, cell 별.
# 전제: permanent fit 산출물 존재 (layer·동결순열 재사용). beer 는 오염 3판 제외
# 교정 manifest(task_PPCC_fit_beerclean.tsv) 사용.
# usage: run_gated_fits_remote.sh [cell...]   (기본: pq3_ppcc_beer pq3_drawer_left)
set -u
PY="${REMOTE_PYTHON:-$HOME/anaconda3/bin/python}"
M="$HOME/datasets/temporal_vla_outputs/eval/robocasa/groot_n15/steer_eval_pq3/manifests_fit30"
O=outputs/eval/robocasa/groot_n15/exp4_1/npz
T=outputs/eval/robocasa/groot_n15/exp4_1/annotation_t0.tsv
[ $# -gt 0 ] && CELLS=("$@") || CELLS=(pq3_ppcc_beer pq3_drawer_left)
rc=0
# phase 집합은 setM_gated(fit_mean_diff --gated: 전 비-terminal phase)와 동일하게 유지 (대칭)
for cell in "${CELLS[@]}"; do
  case "$cell" in
    pq3_ppcc_beer)
      task=PickPlaceCounterToCabinet
      groups="reach-to-object,grasp,transport,place,insert-settle"
      src="$M/task_PPCC_fit_beerclean.tsv" ;;
    pq3_ppcc_bread)
      task=PickPlaceCounterToCabinet
      groups="reach-to-object,grasp,transport,place,insert-settle"
      src="$M/task_PPCC_fit.tsv" ;;
    pq3_drawer_left|pq3_drawer_right)
      task=OpenDrawer
      groups="reach-to-handle,grasp-handle,pull,disengage,push-back,wrong-grasp"  # open-done(terminal) 제외
      src="$M/task_OpenDrawer_fit.tsv" ;;
    *) echo "[skip] unknown cell $cell"; continue ;;
  esac
  cellts="/tmp/exp41_${cell}_fit.tsv"
  grep -E "^#|/${cell}/" "$src" > "$cellts"
  echo "=== [gated fits] $cell rows=$(grep -vc '^#' "$cellts") $(date -u '+%FT%T') ==="
  "$PY" scripts/safe/groot_n15/robocasa/steer/exp4_1/fit_mean_diff.py \
    --manifest "$cellts" --cell "$cell" --targets "$T" --out-root "$O" --gated \
    || { echo "[FAIL] $cell setM_gated"; rc=1; }
  lyr=$("$PY" -c "import json;print(json.load(open('$O/$cell/conceptor_layer_sweep.json'))['selected_layer'])")
  rm -rf "$O/$cell/conceptor_gated"  # 재실행 stale phase 디렉토리 방지 (quota-skip phase 는 미등록=identity)
  "$PY" scripts/safe/groot_n15/robocasa/steer/fit_phase_conceptor_n15.py \
    --manifest "$cellts" --cell "$task/$cell" --groups "$groups" --layers "$lyr" \
    --alphas table14 --denoise per_step --token-pool mean \
    --require-capture-token-mode all_token_full --quota-floor 0.01 \
    --eval-reserved "$M/$cell/eval_reserved.json" \
    --out-dir "$O/$cell/conceptor_gated" \
    || { echo "[FAIL] $cell conceptor_gated"; rc=1; }
  # fit 이 phase 를 quota 로 skip 하면 그 phase 는 미등록=identity — 등록 phase 목록 로그
  echo "[conceptor_gated phases] $(ls -d "$O/$cell/conceptor_gated"/*/dit_L"$lyr" 2>/dev/null | sed 's|.*/conceptor_gated/||;s|/dit_L.*||' | tr '\n' ' ')"
done
echo "=== [gated fits] all done rc=$rc $(date -u '+%FT%T') ==="
exit $rc

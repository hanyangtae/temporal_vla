#!/bin/bash
# exp4-1 길이-미통제(nolen) fit 드라이버 — 승준 노드에서 실행.
#
# 배경(2026-07-25 사용자 지시): COAST 는 길이 confound 를 전혀 통제하지 않고 전 timestep 을
# pool 했다(에이전트 검증). 우리도 통제 없이 전체 길이로 연산자를 만들어 steering 이 듣는지
# 본다 — 의도적으로 길이/후반-phase 신호를 포함시키는 변형 (COAST 원 논문 정렬).
#
# 대상: drawer 1(pq3_drawer_left) + ppcc 1(pq3_ppcc_beer). arm 4종 = setM perm/gated ·
# conceptor perm/gated. 위약·LOO 없음 (targets 를 eval-풀만으로 필터 → LOO 루프 자연 스킵).
# 산출: outputs/eval/robocasa/groot_n15/exp4_1/npz_nolen/<cell>/... (기존 npz 와 격리)
set -u
PY="${REMOTE_PYTHON:-$HOME/anaconda3/bin/python}"
M="$HOME/datasets/temporal_vla_outputs/eval/robocasa/groot_n15/steer_eval_pq3/manifests_fit30"
T=outputs/eval/robocasa/groot_n15/exp4_1/annotation_t0.tsv
O=outputs/eval/robocasa/groot_n15/exp4_1/npz_nolen
OLD=outputs/eval/robocasa/groot_n15/exp4_1/npz   # layer 선정 재사용(sweep 재실행 안 함)
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
mkdir -p "$O"

# targets 를 eval-풀만으로 필터 (fit-풀 제거 → LOO 스킵, 위약은 fit 파이프라인 내장이라 산출돼도 미사용)
TE=/tmp/exp41_t0_evalonly.tsv
"$PY" - "$T" "$TE" <<'PYEOF'
import sys
lines = open(sys.argv[1]).read().splitlines()
hdr = lines[0].split("\t"); pi = hdr.index("pool")
keep = [lines[0]] + [ln for ln in lines[1:] if ln.split("\t")[pi] != "fit"]
open(sys.argv[2], "w").write("\n".join(keep) + "\n")
print(f"targets eval-only: {len(keep)-1} rows")
PYEOF

rc=0
for c in pq3_drawer_left:OpenDrawer:task_OpenDrawer_fit.tsv:"reach-to-handle,grasp-handle,pull,disengage,push-back,wrong-grasp" \
         pq3_ppcc_beer:PickPlaceCounterToCabinet:task_PPCC_fit_beerclean.tsv:"reach-to-object,grasp,transport,place,insert-settle"; do
  IFS=: read -r cell task mf groups <<< "$c"
  cellts="/tmp/exp41_${cell}_nolen_fit.tsv"
  grep -E "^#|/${cell}/" "$M/$mf" > "$cellts"
  echo "=== [nolen setM] $cell rows=$(grep -vc '^#' "$cellts") $(date -u '+%FT%T') ==="
  "$PY" scripts/safe/groot_n15/robocasa/steer/exp4_1/fit_mean_diff.py \
    --manifest "$cellts" --cell "$cell" --targets "$TE" --out-root "$O" \
    --no-length-control \
    || { echo "[FAIL] $cell setM_permanent nolen"; rc=1; continue; }
  echo "=== [nolen setM gated] $cell $(date -u '+%FT%T') ==="
  "$PY" scripts/safe/groot_n15/robocasa/steer/exp4_1/fit_mean_diff.py \
    --manifest "$cellts" --cell "$cell" --targets "$TE" --out-root "$O" \
    --no-length-control --gated \
    || { echo "[FAIL] $cell setM_gated nolen"; rc=1; }

  lyr=$("$PY" -c "import json;print(json.load(open('$OLD/$cell/conceptor_layer_sweep.json'))['selected_layer'])")
  echo "=== [nolen conceptor] $cell layer=L$lyr $(date -u '+%FT%T') ==="
  # permanent (global pool) — --length-control 없이 = 전체 길이 (구 동작)
  "$PY" scripts/safe/groot_n15/robocasa/steer/fit_phase_conceptor_n15.py \
    --manifest "$cellts" --cell "$task/$cell" --groups global --layers "$lyr" \
    --alphas table14 --denoise per_step --token-pool mean \
    --require-capture-token-mode all_token_full --quota-floor 0.01 \
    --eval-reserved "$M/$cell/eval_reserved.json" \
    --out-dir "$O/$cell/conceptor_fitraw" \
    || { echo "[FAIL] $cell conceptor_permanent nolen"; rc=1; }
  NPZ="$O/$cell/conceptor_fitraw/global/dit_L$lyr/conceptors.npz"
  if [ -f "$NPZ" ]; then
    D="$O/$cell/conceptor_permanent/steer/dit_L$lyr"; mkdir -p "$D"
    cp "$NPZ" "$D/conceptors.npz"
    cp "$O/$cell/conceptor_fitraw/global/dit_L$lyr/metadata.json" "$D/metadata.json"
  else
    echo "[FAIL] $cell conceptor_permanent nolen NPZ 미산출"; rc=1
  fi
  # gated (phase 그룹) — --length-control 없이
  rm -rf "$O/$cell/conceptor_gated"
  "$PY" scripts/safe/groot_n15/robocasa/steer/fit_phase_conceptor_n15.py \
    --manifest "$cellts" --cell "$task/$cell" --groups "$groups" --layers "$lyr" \
    --alphas table14 --denoise per_step --token-pool mean \
    --require-capture-token-mode all_token_full --quota-floor 0.01 \
    --eval-reserved "$M/$cell/eval_reserved.json" \
    --out-dir "$O/$cell/conceptor_gated" \
    || { echo "[FAIL] $cell conceptor_gated nolen"; rc=1; }
  echo "[nolen conceptor_gated phases] $(ls -d "$O/$cell/conceptor_gated"/*/dit_L"$lyr" 2>/dev/null | sed 's|.*/conceptor_gated/||;s|/dit_L.*||' | tr '\n' ' ')"
done
echo "=== [nolen fits] ALL_DONE rc=$rc $(date -u '+%FT%T') ==="
exit $rc

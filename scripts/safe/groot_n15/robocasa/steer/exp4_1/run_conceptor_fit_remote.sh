#!/bin/bash
# exp4-1 A(conceptor) arm 배포 NPZ fit — 승준 노드에서 실행.
# layer = A_layer_sweep.json 의 selected_layer (분산 기준 sweep — 전 layer 퇴화 확정이어도
# legacy 참조선으로 배포). fit 은 exp3 배포 규약(per_step·table14·quota-floor 0.01) 유지,
# floor_exhausted 로 NPZ 미산출 시 --quota-floor 0 폴백(퇴화 배포 명기 — A 는 참조선).
# 출력을 serve 계약(<cell>/A/steer/dit_L{L}/conceptors.npz)으로 복사 + provenance 기록
# (원본 phase 디렉토리명 'global' 함정 방어 — R-a).
set -u
PY="${REMOTE_PYTHON:-$HOME/anaconda3/bin/python}"
MROOT="$HOME/datasets/temporal_vla_outputs/eval/robocasa/groot_n15/steer_eval_pq3/manifests_fit30"
O=outputs/eval/robocasa/groot_n15/exp4_1/npz
rc=0
for c in pq3_ppcc_bread:PickPlaceCounterToCabinet pq3_ppcc_beer:PickPlaceCounterToCabinet \
         pq3_drawer_left:OpenDrawer pq3_drawer_right:OpenDrawer; do
  cell="${c%%:*}" task="${c##*:}"
  lyr=$("$PY" -c "import json;print(json.load(open('$O/$cell/A_layer_sweep.json'))['selected_layer'])")
  echo "=== [A fit] $cell layer=L$lyr $(date -u '+%FT%T') ==="
  fit_once() {  # $1=quota_floor
    "$PY" scripts/safe/groot_n15/robocasa/steer/fit_phase_conceptor_n15.py \
      --manifest "$MROOT/$cell/fit_manifest.tsv" --cell "$task/$cell" \
      --groups global --layers "$lyr" --alphas table14 --denoise per_step \
      --token-pool mean --require-capture-token-mode all_token_full \
      --quota-floor "$1" --eval-reserved "$MROOT/$cell/eval_reserved.json" \
      --out-dir "$O/$cell/A_fitraw"
  }
  fit_once 0.01
  NPZ="$O/$cell/A_fitraw/global/dit_L$lyr/conceptors.npz"
  if [ ! -f "$NPZ" ]; then
    echo "[A fit] $cell floor_exhausted → quota-floor 0 폴백 (퇴화 배포, legacy 참조선)"
    fit_once 0
  fi
  if [ -f "$NPZ" ]; then
    D="$O/$cell/A/steer/dit_L$lyr"; mkdir -p "$D"
    cp "$NPZ" "$D/conceptors.npz"
    cp "$O/$cell/A_fitraw/global/dit_L$lyr/metadata.json" "$D/metadata.json"
    "$PY" - "$D" "$NPZ" <<'PYEOF'
import hashlib, json, sys
d, src = sys.argv[1], sys.argv[2]
sha = hashlib.sha256(open(src, "rb").read()).hexdigest()[:12]
json.dump({"source": src, "sha": sha, "note": "global→steer 캐노니컬 복사 (R-a 함정 방어)"},
          open(f"{d}/PROVENANCE.json", "w"), indent=1)
PYEOF
  else
    echo "[FAIL] $cell NPZ 미산출"; rc=1
  fi
done
echo "=== [A fit] all done rc=$rc $(date -u '+%FT%T') ==="
exit $rc

#!/usr/bin/env bash
# S5 (인터페이스 계약 하드 게이트): S4 fit 산출 NPZ 를 exp4-1 serve 계약
# (--steering-phase-npz-base <base_B> --steering-phases steer)으로 로드 + gated 1ep 실행.
# 선행: s4_fit_truncation.sh PASS. 사용: GPU=<idx> PORT=8473 bash s5_npz_serve.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/smoke_common.sh"
GPU="${GPU:?빈 GPU 번호}"
PORT="${PORT:-8473}"
S4="${SMOKE_ROOT}/s4"; S5="${SMOKE_ROOT}/s5"; mkdir -p "$S5"
FIT_OUT="$S4/fit_rs6"

# 공유문서 §4 계약 레이아웃 조립: fit 의 global/dit_L{8,12} → <base_B>/steer/dit_L{8,12}
BASE_B="$S5/base_B"
for L in 8 12; do
  [ -f "$FIT_OUT/global/dit_L${L}/conceptors.npz" ] \
    || { echo "ABORT: S4 산출물 없음 (dit_L${L}) — s4 먼저"; exit 12; }
  mkdir -p "$BASE_B/steer/dit_L${L}"
  cp -f "$FIT_OUT/global/dit_L${L}/conceptors.npz" "$BASE_B/steer/dit_L${L}/"
  [ -f "$FIT_OUT/global/metadata.json" ] && cp -f "$FIT_OUT/global/metadata.json" "$BASE_B/steer/dit_L${L}/" || true
done

start_serve "$GPU" "$PORT" --steering-phase-npz-base "$(to_cont "$BASE_B")" \
  --steering-layers 8,12 --steering-phases steer
trap 'kill_serve "$PORT"' EXIT
wait_health "$PORT"

G1=$(curl -s -m 5 -X POST "http://127.0.0.1:${PORT}/steering_phase" \
  -H 'Content-Type: application/json' -d '{"phase":"steer"}')
G0=$(curl -s -m 5 -X POST "http://127.0.0.1:${PORT}/steering_phase" \
  -H 'Content-Type: application/json' -d '{"phase":"off"}')
echo "steer→$G1"; echo "off→$G0"
echo "$G1" | grep -q '"gated":true'  && { echo "PASS steer phase gated"; N_PASS=$((N_PASS+1)); } \
  || { echo "[FAIL] steer phase 미등록"; N_FAIL=$((N_FAIL+1)); }
echo "$G0" | grep -q '"gated":false' && { echo "PASS off phase identity"; N_PASS=$((N_PASS+1)); } \
  || { echo "[FAIL] off phase 가 gated"; N_FAIL=$((N_FAIL+1)); }

# gated 1ep 실행 (steer phase 로 고정 후 --no-features rollout — hook 실트래픽 검증)
curl -s -m 5 -X POST "http://127.0.0.1:${PORT}/steering_phase" \
  -H 'Content-Type: application/json' -d '{"phase":"steer"}' >/dev/null
run_ep "$PORT" "$S5/gated_ep" 0 0 --no-features --expect-chunk-len 16
if [ -n "$(ep_file "$S5/gated_ep" json)" ]; then
  echo "PASS gated 1ep 완주"; N_PASS=$((N_PASS+1))
else
  echo "[FAIL] gated 1ep 사이드카 없음"; N_FAIL=$((N_FAIL+1))
fi
kill_serve "$PORT"; trap - EXIT

summary S5

#!/usr/bin/env bash
# S2 (Track I DiT 하드 게이트): self-donor W=3 — fired_records == [t0,t0+1,t0+2] 정확 일치,
# csv ≈ baseline (bitwise 기대, 실패 시 max|Δ| soft 보고), 캡처 record 수 정합.
# 사용: GPU=<idx> PORT=8471 bash s2_dit_patch.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/smoke_common.sh"
GPU="${GPU:?빈 GPU 번호}"
PORT="${PORT:-8471}"
T0=8; W=3; CAP="0,2,4,8,10,12,15"
S2="${SMOKE_ROOT}/s2"; mkdir -p "$S2"

start_serve "$GPU" "$PORT" --patch-layers 15 --patch-token-select all --patch-allow-collect \
  --collect --groot-dit-capture-layers "$CAP" --groot-dit-token-pool all_token_full
trap 'kill_serve "$PORT"' EXIT
wait_health "$PORT"
curl -s -m 3 "http://127.0.0.1:${PORT}/health" | grep -q '"patch"' \
  || { echo "ABORT: /health 에 patch 지문 없음"; exit 12; }

# 1) baseline (disarm 상태, full-token 캡처)
if [ -z "$(ep_file "$S2/base" pkl)" ]; then
  curl -s -m 10 -X POST "http://127.0.0.1:${PORT}/patch_disarm" | grep -q '"ok":true' || exit 13
  echo "[s2] baseline"
  run_ep "$PORT" "$S2/base" 0 0
fi
BASE_PKL="$(ep_file "$S2/base" pkl)"
[ -n "$BASE_PKL" ] || { echo "ABORT: baseline pkl 없음"; exit 12; }

# 2) self-donor 추출 (동일 ep 의 L15 full-token → 창 내 대입 = 항등 기대)
DONOR="$S2/self_L15.npz"
if [ ! -f "$DONOR" ]; then
  docker exec lerobot python \
    "${WT_CONT}/scripts/safe/groot_n15/robocasa/steer/patchceil/extract_donor_npz.py" \
    --pkl "$(to_cont "$BASE_PKL")" --out "$(to_cont "$DONOR")" --layers 15 --cap "$CAP" \
    --allow-fail || { echo "ABORT: self-donor 추출 실패"; exit 13; }
fi

# 3) patch rollout (donor_start=t0 — 같은 ep 같은 record 창 대입)
if [ -z "$(ep_file "$S2/patch" pkl)" ]; then
  curl -s -m 30 -X POST "http://127.0.0.1:${PORT}/patch_arm" -H 'Content-Type: application/json' \
    -d "{\"npz\":\"$(to_cont "$DONOR")\",\"start_record\":${T0},\"donor_start\":${T0},\"patch_len\":${W},\"tag\":\"s2_self\"}" \
    | grep -q '"ok":true' || { echo "ABORT: patch_arm 실패"; exit 13; }
  echo "[s2] self-donor patch"
  run_ep "$PORT" "$S2/patch" 0 0
  curl -s -m 10 "http://127.0.0.1:${PORT}/patch_status" > "$S2/status.json"
fi
kill_serve "$PORT"; trap - EXIT

check "fired 창 정확 일치" judge status-audit "$S2/status.json" --expect-start "$T0" --expect-len "$W" --expect-total "$((W * 4))"
check_soft "self-donor csv ≈ baseline (bf16→fp16 왕복)" \
  judge csv-bitwise "$(ep_file "$S2/base" csv)" "$(ep_file "$S2/patch" csv)"
check "record 수 정합 (판정 필드)" judge_c fields \
  "$(to_cont "$BASE_PKL")" "$(to_cont "$(ep_file "$S2/patch" pkl)")" \
  --fields n_action_steps,scenario_seed,inference_seed

summary S2

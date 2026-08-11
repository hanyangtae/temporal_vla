#!/usr/bin/env bash
# S3 (Track I VL 하드 게이트): self-VL-donor bitwise ≡ baseline / 요청당 1-fire 전창 발화 /
# T_vl 상이 donor 무에러 실행 + action 변화 실효.
# 사용: GPU=<idx> PORT=8472 bash s3_vl_patch.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/smoke_common.sh"
GPU="${GPU:?빈 GPU 번호}"
PORT="${PORT:-8472}"
S3="${SMOKE_ROOT}/s3"; mkdir -p "$S3"

start_serve "$GPU" "$PORT" --patch-pathway vl --patch-allow-collect \
  --collect --capture-vl --groot-vl-capture-point post_vl_sa_full --groot-dit-capture-layers 15
trap 'kill_serve "$PORT"' EXIT
wait_health "$PORT"

# 1) baseline (disarm, post_vl_sa_full 캡처)
if [ -z "$(ep_file "$S3/base" pkl)" ]; then
  curl -s -m 10 -X POST "http://127.0.0.1:${PORT}/patch_disarm" | grep -q '"ok":true' || exit 13
  echo "[s3] baseline"
  run_ep "$PORT" "$S3/base" 0 0
fi
BASE_PKL="$(ep_file "$S3/base" pkl)"
[ -n "$BASE_PKL" ] || { echo "ABORT: baseline pkl 없음"; exit 12; }

# 2) self-VL-donor 추출 + T_vl-trim 변형 donor (shape 경로 검증용 — 뒤 8토큰 절단)
DONOR="$S3/self_vl.npz"; TRIM="$S3/trim_vl.npz"
if [ ! -f "$DONOR" ]; then
  docker exec lerobot python \
    "${WT_CONT}/scripts/perturb/extract_vl_donor_npz.py" \
    --pkl "$(to_cont "$BASE_PKL")" --out "$(to_cont "$DONOR")" --allow-fail \
    || { echo "ABORT: self-VL-donor 추출 실패"; exit 13; }
fi
if [ ! -f "$TRIM" ]; then
  # docker exec 는 -i 없이는 heredoc(stdin)을 전달하지 않음 — python - 이 빈 입력으로 무음 종료
  docker exec -i lerobot python - "$(to_cont "$DONOR")" "$(to_cont "$TRIM")" <<'PY' || exit 13
import json, sys
import numpy as np
src, dst = sys.argv[1], sys.argv[2]
z = np.load(src, allow_pickle=False)
meta = json.loads(bytes(z["meta_json"]).decode()); vl = z["VL"][:, :-8, :]
meta["t_vl"] = int(vl.shape[1]); meta["note"] = "s3 trim(-8 tokens) — T_vl 상이 경로 검증"
np.savez(dst, meta_json=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8), VL=vl)
print(f"wrote {dst} {vl.shape}")
PY
fi

# 3) self-donor patch (전창, donor_start=0 — 항등 기대)
if [ -z "$(ep_file "$S3/selfpatch" pkl)" ]; then
  curl -s -m 30 -X POST "http://127.0.0.1:${PORT}/patch_arm" -H 'Content-Type: application/json' \
    -d "{\"npz\":\"$(to_cont "$DONOR")\",\"start_record\":0,\"donor_start\":0,\"patch_len\":-1,\"tag\":\"s3_self\"}" \
    | grep -q '"ok":true' || { echo "ABORT: VL patch_arm 실패"; exit 13; }
  echo "[s3] self-VL patch"
  run_ep "$PORT" "$S3/selfpatch" 0 0
  curl -s -m 10 "http://127.0.0.1:${PORT}/patch_status" > "$S3/status_self.json"
fi

# 4) trim donor (T_vl 상이 — 무에러 + action 변화. 창 6 records — 발산 후 timeout 으로
#    donor 고갈(exhausted) 기록이 나는 함정을 피해 창을 donor R 내로 한정)
if [ -z "$(ep_file "$S3/trim" pkl)" ]; then
  curl -s -m 30 -X POST "http://127.0.0.1:${PORT}/patch_arm" -H 'Content-Type: application/json' \
    -d "{\"npz\":\"$(to_cont "$TRIM")\",\"start_record\":0,\"donor_start\":0,\"patch_len\":6,\"tag\":\"s3_trim\"}" \
    | grep -q '"ok":true' || { echo "ABORT: trim patch_arm 실패"; exit 13; }
  echo "[s3] trim-VL patch"
  run_ep "$PORT" "$S3/trim" 0 0
  curl -s -m 10 "http://127.0.0.1:${PORT}/patch_status" > "$S3/status_trim.json"
fi
kill_serve "$PORT"; trap - EXIT

# VL 캡처는 fp16 저장인데 원값이 fp32 라 self-donor bitwise 는 원리적으로 불가 (S3 실측:
# record0 편차 ~1e-3 = 양자화 스케일, 이후 closed-loop 증폭). 하드 기준 = 첫 편차 ≤ 5e-3.
check "self-VL ≈ baseline (첫 편차 양자화 스케일)" \
  judge csv-first-diff "$(ep_file "$S3/base" csv)" "$(ep_file "$S3/selfpatch" csv)"
check_soft "self-VL bitwise (참고 기록)" judge csv-bitwise "$(ep_file "$S3/base" csv)" "$(ep_file "$S3/selfpatch" csv)"
check "self-VL 전창 1-fire"       judge status-audit "$S3/status_self.json" --expect-full
check "trim-VL 실효(≠base)"       judge csv-bitwise "$(ep_file "$S3/base" csv)" "$(ep_file "$S3/trim" csv)" --expect-diff
check "trim-VL 창 발화 일치"      judge status-audit "$S3/status_trim.json" --expect-start 0 --expect-len 6 --expect-total 6
grep -o '"donor_t_vl":[0-9]*' "$S3/status_trim.json" | head -1

summary S3

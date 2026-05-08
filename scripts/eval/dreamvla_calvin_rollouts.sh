#!/usr/bin/env bash
# DreamVLA × Calvin 롤아웃 영상 수집.
#
# 통일 API 경로:
#   serve/dreamvla.py (port 8200)  ← 서버 (dreamvla 컨테이너에서 별도 실행)
#   eval/calvin.py                 ← env loop (calvin 컨테이너에서 실행)
#
# 동작:
#   - num-sequences 만큼 procedural 5-subtask 시퀀스 생성 (debug dataset 만 있어도 OK).
#     이유: get_sequences() 가 initial-state 조합으로 procedural 하게 생성, episode 파일 미사용.
#   - ep_len = 360 × EP_LEN_MULT (default 3 → 1080)
#   - success_deadline = 360 (원래 ep_len) → deadline 초과 oracle done 은 실패 처리
#   - 실패(success_counter < 5) 영상만 outputs/rollouts/calvin/dreamvla/<TS>/ 로 저장
#
# 실행:
#   # 1) 서버 (dreamvla 컨테이너)
#   docker compose run --rm dreamvla \
#       python /temporal_vla/scripts/serve/dreamvla.py \
#       --profile /temporal_vla/configs/checkpoints/dreamvla__calvin_dynamic_depth_semantic.yaml
#
#   # 2) eval (calvin 컨테이너)
#   docker compose run --rm calvin bash /temporal_vla/scripts/eval/dreamvla_calvin_rollouts.sh
#
# 환경 변수 override:
#   NUM_SEQUENCES=200 EP_LEN_MULT=2 bash ...

set -euo pipefail

NUM_SEQUENCES="${NUM_SEQUENCES:-1000}"
EP_LEN_MULT="${EP_LEN_MULT:-3}"
DEFAULT_EP_LEN=360
EP_LEN=$((DEFAULT_EP_LEN * EP_LEN_MULT))
SUCCESS_DEADLINE="${SUCCESS_DEADLINE:-${DEFAULT_EP_LEN}}"

DATASET_PATH="${DATASET_PATH:-/temporal_vla/src/benchmarks/calvin/dataset/calvin_debug_dataset}"
SERVER_URL="${SERVER_URL:-http://localhost:8200}"

TIMESTAMP=$(date +%y%m%d%H%M%S)
ROLLOUT_DIR="/temporal_vla/outputs/rollouts/calvin/dreamvla/${TIMESTAMP}"
mkdir -p "${ROLLOUT_DIR}"

echo "=========================================="
echo " DreamVLA × Calvin Rollouts"
echo "  timestamp        : ${TIMESTAMP}"
echo "  num_sequences    : ${NUM_SEQUENCES}"
echo "  ep_len           : ${EP_LEN}  (default ${DEFAULT_EP_LEN} × ${EP_LEN_MULT})"
echo "  success_deadline : ${SUCCESS_DEADLINE}"
echo "  dataset          : ${DATASET_PATH}"
echo "  server_url       : ${SERVER_URL}"
echo "  output           : ${ROLLOUT_DIR}"
echo "=========================================="

python /temporal_vla/scripts/eval/calvin.py \
    --dataset-path "${DATASET_PATH}" \
    --server-url "${SERVER_URL}" \
    --num-sequences "${NUM_SEQUENCES}" \
    --num-videos "${NUM_SEQUENCES}" \
    --ep-len "${EP_LEN}" \
    --success-deadline "${SUCCESS_DEADLINE}" \
    --video-dir "${ROLLOUT_DIR}" \
    --output "${ROLLOUT_DIR}/results.json" \
    2>&1 | tee "${ROLLOUT_DIR}/eval.log"

echo "[DONE] rollouts saved under ${ROLLOUT_DIR}"

#!/bin/bash
# X-VLA Calvin 평가 스크립트
#
# 사전 준비:
#   1) xvla 컨테이너에서 서버 실행:
#      docker compose run --rm xvla \
#        python /temporal_vla/scripts/serve_xvla.py \
#        --pretrained-path 2toINF/X-VLA-Calvin-ABC_D
#
#   2) 이 스크립트를 calvin 컨테이너에서 실행:
#      docker compose run --rm calvin bash /temporal_vla/scripts/eval_xvla_calvin.sh
#
# 또는 한 줄로 (서버가 이미 떠 있을 때):
#   docker compose run --rm calvin python /temporal_vla/scripts/calvin_eval.py \
#     --dataset-path /temporal_vla/src/benchmarks/calvin/dataset/task_ABC_D \
#     --server-url http://localhost:8100 \
#     --num-sequences 1000

set -e

DATASET_PATH="${DATASET_PATH:-/temporal_vla/src/benchmarks/calvin/dataset/task_ABC_D}"
SERVER_URL="${SERVER_URL:-http://localhost:8100}"
NUM_SEQ="${NUM_SEQ:-1000}"
NUM_VIDEOS="${NUM_VIDEOS:-10}"

echo "=== X-VLA Calvin Evaluation ==="
echo "Dataset:    ${DATASET_PATH}"
echo "Server:     ${SERVER_URL}"
echo "Sequences:  ${NUM_SEQ}"
echo "Videos:     ${NUM_VIDEOS}"
echo "================================"

python /temporal_vla/scripts/calvin_eval.py \
  --dataset-path "${DATASET_PATH}" \
  --server-url "${SERVER_URL}" \
  --num-sequences "${NUM_SEQ}" \
  --num-videos "${NUM_VIDEOS}" \
  --output "/temporal_vla/outputs/calvin_eval/X-VLA/results.json"

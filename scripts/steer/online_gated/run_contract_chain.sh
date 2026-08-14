#!/usr/bin/env bash
# 수축 연산자 3종 체인: [sconceptor ∥ varc] (각 3 arm × serve 3 = GPU 6 serve) → conceptor.
# A100 워커(srv48/srv50) 전용 — GPU 1장에 serve 6 규칙을 채우기 위해 op 2개를 병렬로 돌린다.
# usage (대상 머신에서): GPU=0 SLUG=DishwasherRack_out bash scripts/steer/online_gated/run_contract_chain.sh
set -euo pipefail
GPU="${GPU:?GPU 번호}"
SLUG="${SLUG:?slug}"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$HERE/run_online_gated_eval.sh"

common=(
  SLUGS="$SLUG" ARMS=online,online_pl,online_fut
  STEER_OP=conceptor NPZ_VARIANT=contract_s5m5
  SERVES_PER_GPU=3 SERVE_MODE=host
  SERVE_PY="$HOME/miniconda3/envs/lerobot_050_groot/bin/python"
  SERVE_PYTHONPATH="$HOME/pkt_ws/temporal_vla/lerobot/src"
  DETECTOR_CKPT_TMPL='outputs/analysis/grid_phase/detector_sim_s5m5/detector_pertask_lstm_%SLUG%.pt'
)
OUT=outputs/eval/robocasa/groot_n15/og_contract

env "${common[@]}" GPUS="$GPU" PORT_BASE=8700 STEER_OP_NAME=sconceptor \
  OUT_ROOT=$OUT/sconceptor bash "$RUNNER" &
A=$!
# 같은 GPU 를 두 러너가 공유 — 두 번째부터는 busy 게이트 우회 (합산 6 serve = A100 규칙)
env "${common[@]}" GPUS="$GPU" ALLOW_BUSY_GPU=1 PORT_BASE=8710 STEER_OP_NAME=varc \
  OUT_ROOT=$OUT/varc bash "$RUNNER" &
B=$!
rc=0
wait "$A" || rc=$?
wait "$B" || rc=$?
env "${common[@]}" GPUS="$GPU" ALLOW_BUSY_GPU=1 PORT_BASE=8700 STEER_OP_NAME=conceptor \
  OUT_ROOT=$OUT/conceptor bash "$RUNNER" || rc=$?
echo "CONTRACT_CHAIN_${SLUG}_DONE rc=${rc}"

#!/usr/bin/env bash
# kanu drawer 잔여 체인: v2 β1.0 resume → b05_v2 → contract(sconceptor∥varc → conceptor).
# docker serve 모드, 패치된 worktree GR00T(상단 배너) PYTHONPATH 강제.
# usage: GPUS_SET=3,4,7 bash scripts/steer/online_gated/run_kanu_drawer_chain.sh
set -uo pipefail
GPUS_SET="${GPUS_SET:?빈 GPU 콤마목록}"
W="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYO="/temporal_vla/.claude/worktrees/grid-phase-sep/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla"
RUN="$W/scripts/steer/online_gated/run_online_gated_eval.sh"

common=(
  SLUGS=OpenDrawer_left EP_MODE=replay PYPATH_OVERRIDE="$PYO"
  NPZ_ROOT="$W/outputs/steer/online_pipe"
  DETECTOR_CKPT_TMPL="$W/outputs/analysis/grid_phase/detector_sim_s5m5/detector_pertask_lstm_%SLUG%.pt"
  INDEX_TSV="$W/outputs/steer/online_pipe/manifests/index_rollouts.tsv"
  PLAN_JSON="$W/configs/collect/n15_grid_v1/collection_plan.json"
)

echo "[chain] stage1: v2 b1.0 resume"
env "${common[@]}" GPUS="$GPUS_SET" ALLOW_BUSY_GPU=1 STEER_BETA=1.0 \
  ARMS=base,online,online_pl,online_fut,oracle_always NPZ_VARIANT=setM_s5m5_seg \
  OUT_ROOT="$W/outputs/eval/robocasa/groot_n15/online_gated_replay_v2" bash "$RUN"
echo "STAGE1_RC=$? (v2 b1.0)"

echo "[chain] stage2: b05_v2"
env "${common[@]}" GPUS="$GPUS_SET" ALLOW_BUSY_GPU=1 STEER_BETA=0.5 \
  ARMS=online,online_pl,online_fut,online_fut_pl NPZ_VARIANT=setM_s5m5_seg \
  OUT_ROOT="$W/outputs/eval/robocasa/groot_n15/online_gated_replay_b05_v2" bash "$RUN"
echo "STAGE2_RC=$? (b05_v2)"

echo "[chain] stage3: contract sconceptor ∥ varc"
env "${common[@]}" GPUS="$GPUS_SET" ALLOW_BUSY_GPU=1 PORT_BASE=8700 \
  ARMS=online,online_pl,online_fut STEER_OP=conceptor STEER_OP_NAME=sconceptor \
  NPZ_VARIANT=contract_s5m5 \
  OUT_ROOT="$W/outputs/eval/robocasa/groot_n15/og_contract/sconceptor" bash "$RUN" &
A=$!
env "${common[@]}" GPUS="$GPUS_SET" ALLOW_BUSY_GPU=1 PORT_BASE=8710 \
  ARMS=online,online_pl,online_fut STEER_OP=conceptor STEER_OP_NAME=varc \
  NPZ_VARIANT=contract_s5m5 \
  OUT_ROOT="$W/outputs/eval/robocasa/groot_n15/og_contract/varc" bash "$RUN" &
B=$!
wait "$A"; wait "$B"
echo "[chain] stage3b: contract conceptor"
env "${common[@]}" GPUS="$GPUS_SET" ALLOW_BUSY_GPU=1 PORT_BASE=8700 \
  ARMS=online,online_pl,online_fut STEER_OP=conceptor STEER_OP_NAME=conceptor \
  NPZ_VARIANT=contract_s5m5 \
  OUT_ROOT="$W/outputs/eval/robocasa/groot_n15/og_contract/conceptor" bash "$RUN"
echo "KANU_DRAWER_CHAIN_DONE"

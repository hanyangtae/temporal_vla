#!/usr/bin/env bash
# Run under with_gpu_lease.sh kanu "5"; one GPU, two GR00T collection servers.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MAIN_ROOT=/home/dongkyu/pkt_ws/temporal_vla
export PLAN_JSON="$REPO/configs/collect/n15_scene_topup_coffee_20260909/collection_plan.json"
export GRID_ROOT="$MAIN_ROOT/outputs/collect/grid_staging_topup_coffee_20260909"
export INSTRUCTIONS=CoffeeSetupMug GPUS=5 SERVES_PER_GPU=2 PORT_BASE=9500 NOISE_LIMIT=10
export STAGING_WAIT_GB=30 SERVE_OMP_THREADS=4
export PYPATH_PREFIX=/temporal_vla/.claude/worktrees/eval-whole-pipe
export COLLECTOR_PY="$PYPATH_PREFIX/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py"
cd "$REPO"
exec bash scripts/safe/groot_n15/robocasa/collect/collect_grid.sh

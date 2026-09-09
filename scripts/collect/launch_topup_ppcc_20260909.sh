#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MAIN_ROOT=/home/dongkyu/pkt_ws/temporal_vla
export PLAN_JSON="$REPO/configs/collect/n15_topup_ppcc_20260909/collection_plan.json"
export GRID_ROOT="$MAIN_ROOT/outputs/collect/grid_staging_topup_ppcc_20260909"
export INSTRUCTIONS='PPCC/bread,PPCC/jug,PPCC/marshmallow'
export GPUS=5 SERVES_PER_GPU=2 PORT_BASE=9500 NOISE_LIMIT=10
export PRIORITY_CELLS='PPCC/bread|s3,PPCC/bread|s4'
export STAGING_WAIT_GB=30 SERVE_OMP_THREADS=4
export PYPATH_PREFIX=/temporal_vla/.claude/worktrees/collect-topup
export COLLECTOR_PY="$PYPATH_PREFIX/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py"
mkdir -p "$GRID_ROOT"
export DONE_LIST="$GRID_ROOT/existing_and_shipped.txt"
cat "$REPO/configs/collect/n15_topup_ppcc_20260909/skip_existing_1800.txt" > "$DONE_LIST"
if [ -f "$GRID_ROOT/shipped_cells.txt" ]; then cat "$GRID_ROOT/shipped_cells.txt" >> "$DONE_LIST"; fi
cd "$REPO"
exec bash scripts/safe/groot_n15/robocasa/collect/collect_grid.sh

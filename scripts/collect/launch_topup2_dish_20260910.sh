#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MAIN_ROOT=$(dirname "$(git -C "$REPO" rev-parse --path-format=absolute --git-common-dir)")
: "${GPUS:?}" "${INSTRUCTIONS:?}" "${COLLECTION_SHARD:?}"
export PLAN_JSON="$REPO/configs/collect/n15_topup2_dish_20260910/collection_plan.json"
export GRID_ROOT="$MAIN_ROOT/outputs/collect/grid_staging_topup2_dish_20260910_${COLLECTION_SHARD}"
export SERVES_PER_GPU=6 PORT_BASE=9660 NOISE_LIMIT=10
export STAGING_WAIT_GB=30 SERVE_OMP_THREADS=4
export SERVE_MODE=host SERVE_PY="$HOME/miniconda3/envs/lerobot_050_groot/bin/python"
export SERVE_PYTHONPATH="$MAIN_ROOT/lerobot/src:$MAIN_ROOT/src/policies/Isaac-GR00T:$MAIN_ROOT"
export PYPATH_PREFIX=/temporal_vla/.claude/worktrees/collect-topup
export COLLECTOR_PY="$PYPATH_PREFIX/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py"
mkdir -p "$GRID_ROOT"
export DONE_LIST="$GRID_ROOT/existing_and_shipped.txt"
cat "$REPO/configs/collect/n15_topup2_dish_20260910/skip_existing.txt" > "$DONE_LIST"
if [ -f "$GRID_ROOT/shipped_cells.txt" ]; then cat "$GRID_ROOT/shipped_cells.txt" >> "$DONE_LIST"; fi
cd "$REPO"
exec bash scripts/safe/groot_n15/robocasa/collect/collect_grid.sh

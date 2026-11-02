#!/usr/bin/env bash
# Harness-guarded extraction shard launcher. env: GPUS=<gpu> HARNESS_SESSION=<owner>
# usage: run_extract.sh [--smoke] [--shard k --num-shards n]
set -euo pipefail
cd /home/dongkyu/pkt_ws/temporal_vla
source scripts/utils/groot_launch_guard.sh
groot_launch_guard eval "$0" "$@"
exec .venvs/rl2_iid/bin/python scripts/rl2_vla/groot_fold27_20261028/common_qam/run_extract.py "$@"

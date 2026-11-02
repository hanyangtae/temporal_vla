#!/usr/bin/env bash
set -euo pipefail
cd /home/dongkyu/pkt_ws/temporal_vla
source scripts/utils/groot_launch_guard.sh
groot_launch_guard eval "$0" "$@"
exec .venvs/rl2_iid/bin/python -u scripts/rl2_vla/groot_robocasa_adapter_20261021/run_model_input_audit.py

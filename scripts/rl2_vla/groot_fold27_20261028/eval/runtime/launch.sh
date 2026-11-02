#!/usr/bin/env bash
# fold27 MIL+QAM/TRQAM eval entry: GR00T harness admission, then launch_group.py (stdlib python3 on every host).
#   HARNESS_MACHINE=kanu|srv48|srv50 HARNESS_SESSION=<owner> GPUS=<g[,g]> [SERVES_PER_GPU=<n>] \
#     bash scripts/rl2_vla/groot_fold27_20261028/eval/runtime/launch.sh --jobs <jobs.txt> [--rows 0] [--port0 P]
# srv48/srv50 additionally need HARNESS_COORDINATOR_SSH / HARNESS_COORDINATOR_SCRIPT (runbook_groot_harness.md).
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)
cd "$REPO_ROOT"
source scripts/utils/groot_launch_guard.sh
groot_launch_guard eval "$REPO_ROOT/scripts/rl2_vla/groot_fold27_20261028/eval/runtime/launch.sh" "$@"
PY=python3; [ -x .venvs/rl2_iid/bin/python ] && PY=.venvs/rl2_iid/bin/python   # kanu: adapter docker helpers need numpy
exec "$PY" -u scripts/rl2_vla/groot_fold27_20261028/eval/runtime/launch_group.py "$@"

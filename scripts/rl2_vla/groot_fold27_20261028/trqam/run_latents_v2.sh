#!/usr/bin/env bash
# rollout_latents_v2 launcher under the groot harness (same guard pattern as run_latents.sh; run_latents.sh is unchanged).
#   GPUS=<g> HARNESS_MACHINE=srv50 HARNESS_SESSION=<s> bash run_latents_v2.sh --token-contract 49 --episodes <dir> --out <new dir> ...
set -euo pipefail
cd "$(dirname "$0")/../../../.."
source scripts/utils/groot_launch_guard.sh
groot_launch_guard eval "$0" "$@"
PY=${FOLD27_PY:-.venvs/rl2_iid/bin/python}
[ -x "$PY" ] || PY=python3
exec "$PY" scripts/rl2_vla/groot_fold27_20261028/trqam/run_latents_v2.py "$@"

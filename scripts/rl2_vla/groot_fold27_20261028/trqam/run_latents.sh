#!/usr/bin/env bash
# Harness-guarded host-matched rollout-latent job (same guard as common_qam/run_extract.sh).
# env: GPUS=<gpu> HARNESS_MACHINE=kanu|srv48|srv50 HARNESS_SESSION=<owner> [HARNESS_COORDINATOR_SSH for srv]
# usage: run_latents.sh --episodes <full extract dir, one scene, this host> --out <latents dir> [--smoke-only]
set -euo pipefail
cd "$(dirname "$0")/../../../.."
source scripts/utils/groot_launch_guard.sh
groot_launch_guard eval "$0" "$@"
PY=.venvs/rl2_iid/bin/python
[ -x "$PY" ] || PY=python3
exec "$PY" scripts/rl2_vla/groot_fold27_20261028/trqam/run_latents.py "$@"

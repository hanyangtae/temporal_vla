#!/usr/bin/env bash
# Harness-guarded v2 GPU smoke for ONE fold x arm: (1) GPU resume check on real data, (2) finetune_fold_v2 --smoke-steps 200.
# Writes only <out>/<fold>/resume_check_v2/RESUME_CHECK.json and <out>/<fold>/<tag>_smoke/SMOKE_DONE.json (never DONE.json).
# env: GPUS=<g> HARNESS_MACHINE=kanu HARNESS_SESSION=<owner> [FOLD27_PY]
# usage: smoke_v2.sh <fold_id> <basic|pbrs|mix> <episodes dir> <latents dir> [--reuse-ae-from <v1 root>]
set -euo pipefail
cd "$(dirname "$0")/../../../.."
source scripts/utils/groot_launch_guard.sh
SERVES_PER_GPU=${SERVES_PER_GPU:-1} groot_launch_guard eval "$0" "$@"
FOLD=$1 ARM=$2 EP=$3 LAT=$4; shift 4
PY=${FOLD27_PY:-.venvs/rl2_iid/bin/python}
OUT=outputs/rl2_vla/fold27_trqam_v2_20261102
INIT=outputs/rl2_vla/atomic65_common_qam_work_20261029/scratch_run/checkpoints/step_0500000
export JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES="$GPUS" XLA_PYTHON_CLIENT_PREALLOCATE=false
T=scripts/rl2_vla/groot_fold27_20261028/trqam
[ "$ARM" != basic ] || "$PY" "$T/gpu_resume_check.py" --fold "$FOLD" --episodes "$EP" --latents "$LAT" --out "$OUT"
"$PY" "$T/finetune_fold_v2.py" --spec outputs/analysis/groot_fold27_20261028/folds/fold_spec.json --fold "$FOLD" --arm "$ARM" \
  --episodes "$EP" --latents "$LAT" --out "$OUT" --dry-run "$@"
exec "$PY" "$T/finetune_fold_v2.py" --spec outputs/analysis/groot_fold27_20261028/folds/fold_spec.json --fold "$FOLD" --arm "$ARM" \
  --episodes "$EP" --latents "$LAT" --out "$OUT" --init "$INIT" --init-step 500000 --smoke-steps 200 "$@"

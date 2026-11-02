#!/usr/bin/env bash
# smoke_v2.sh with an EXPLICIT output root (smoke_v2.sh is unchanged and hard-codes the kanu v2 root). Same steps:
# (1) GPU resume check on real data (basic arm), (2) finetune_fold_v2 dry-run, (3) --smoke-steps 200. Writes only
# <FOLD27_OUT>/<fold>/resume_check_v2/RESUME_CHECK.json and <FOLD27_OUT>/<fold>/<tag>_smoke/SMOKE_DONE.json.
# env: GPUS=<g> HARNESS_MACHINE=<host> HARNESS_SESSION=<owner> FOLD27_OUT=<new out root> [FOLD27_PY]
# usage: smoke_v2b.sh <fold_id> <basic|pbrs|mix> <episodes dir> <latents dir>
set -euo pipefail
cd "$(dirname "$0")/../../../.."
source scripts/utils/groot_launch_guard.sh
SERVES_PER_GPU=${SERVES_PER_GPU:-1} groot_launch_guard eval "$0" "$@"
FOLD=$1 ARM=$2 EP=$3 LAT=$4; shift 4
PY=${FOLD27_PY:-.venvs/rl2_iid/bin/python}
OUT=${FOLD27_OUT:?FOLD27_OUT (new output root) required}
case "$OUT" in outputs/rl2_vla/fold27_trqam_v2_20261102|outputs/rl2_vla/fold27_trqam_20261102) echo "use a NEW out root" >&2; exit 2 ;; esac
INIT=outputs/rl2_vla/atomic65_common_qam_work_20261029/scratch_run/checkpoints/step_0500000
export JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES="$GPUS" XLA_PYTHON_CLIENT_PREALLOCATE=false
T=scripts/rl2_vla/groot_fold27_20261028/trqam
[ "$ARM" != basic ] || "$PY" "$T/gpu_resume_check.py" --fold "$FOLD" --episodes "$EP" --latents "$LAT" --out "$OUT"
"$PY" "$T/finetune_fold_v2.py" --spec outputs/analysis/groot_fold27_20261028/folds/fold_spec.json --fold "$FOLD" --arm "$ARM" \
  --episodes "$EP" --latents "$LAT" --out "$OUT" --dry-run "$@"
exec "$PY" "$T/finetune_fold_v2.py" --spec outputs/analysis/groot_fold27_20261028/folds/fold_spec.json --fold "$FOLD" --arm "$ARM" \
  --episodes "$EP" --latents "$LAT" --out "$OUT" --init "$INIT" --init-step 500000 --smoke-steps 200 "$@"

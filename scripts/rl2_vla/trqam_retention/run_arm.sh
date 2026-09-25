#!/usr/bin/env bash
# Called only after coordinator lease + fresh physical-GPU occupancy checks.
set -euo pipefail
: "${TRAIN_PYTHON:?}" "${ASSET_ROOT:?}" "${CUDA_VISIBLE_DEVICES:?}" "${RUN_ROOT:?}"
method=${1:?qam or trqam}
case "$method" in qam|trqam) ;; *) exit 2;; esac
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
common=(--qam-root RL2-VLA/third_party/qam --checkpoint "$ASSET_ROOT/outputs/rl2_assets/qam_bridge/rl2_vla_qam_bridge_500k.pkl" --cache "$ASSET_ROOT/outputs/rl2_iid_training_cache_20260916" --method "$method")
for seed in 42 0 7; do
  "$TRAIN_PYTHON" -u scripts/rl2_vla/trqam_retention/train.py "${common[@]}" --seed "$seed" --output "$RUN_ROOT/${method}_seed${seed}" --steps 50000
 done

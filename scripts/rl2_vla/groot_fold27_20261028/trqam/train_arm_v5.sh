#!/usr/bin/env bash
# One TRQAM production arm under its OWN single-GPU harness receipt (kind eval, SERVES_PER_GPU=1 -> 1 train per GPU).
# Launched by host_fold_coordinator_v5 through wait_launch.py with the queue's own env (TrainQueue.env(gpu)) and the queue's
# own command (TrainQueueV2.cmd(fold, arm, 'prod')); this file adds only the harness guard and a GPU-identity check.
#   GPUS=<g> CUDA_VISIBLE_DEVICES=<g> HARNESS_MACHINE=kanu HARNESS_SESSION=fold27-trqam-v2 bash train_arm_v5.sh <python> finetune_fold_v2.py ...
set -euo pipefail
cd "$(dirname "$0")/../../../.."
source scripts/utils/groot_launch_guard.sh
SERVES_PER_GPU=1 groot_launch_guard eval "$0" "$@"
case "${GPUS:-}" in *,*|'') echo '[train_arm_v5] exactly one GPU per arm' >&2; exit 2 ;; esac
[ "${CUDA_VISIBLE_DEVICES:-}" = "$GPUS" ] || { echo "[train_arm_v5] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} != GPUS=$GPUS" >&2; exit 2; }
exec "$@"

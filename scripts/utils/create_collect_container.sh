#!/usr/bin/env bash
# Create only; never replace an existing container or claim GPU ownership here.
set -euo pipefail
: "${COLLECT_CONTAINER:?name required}"
: "${COLLECT_GPU:?one GPU index required}"
: "${MOUNT_ROOT:?absolute common repo path required}"
: "${CACHE_ROOT:?absolute cache path required}"
[[ "$COLLECT_GPU" =~ ^[0-9]+$ ]] || exit 2
[[ "$MOUNT_ROOT" = /* && "$CACHE_ROOT" = /* ]] || exit 2
docker run -d --name "$COLLECT_CONTAINER" \
  --gpus "device=$COLLECT_GPU" --network host --shm-size=8g \
  --log-driver=json-file --log-opt max-size=10m --log-opt max-file=3 \
  --entrypoint /bin/sleep \
  -v "$MOUNT_ROOT:/temporal_vla" -v "$CACHE_ROOT:/cache" \
  -e NVIDIA_DRIVER_CAPABILITIES=all "${COLLECT_IMAGE:-temporal_vla-robocasa}" infinity
python3 "$(dirname "$0")/check_collect_container.py" "$COLLECT_CONTAINER"

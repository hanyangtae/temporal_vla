#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/workspace/labeler"
if fuser 8765/tcp >/dev/null 2>&1; then
  echo 'Port 8765 already in use; stop the verified labeler process before starting.' >&2
  exit 1
fi
setsid nohup python3 labeler_server_multiplan.py \
  --root "$HOME/datasets/temporal_vla_store/groot/n15/grid/08f1c9df8207" \
  --extra-root "4fa6496cd684=$HOME/datasets/temporal_vla_store/groot/n15/grid/4fa6496cd684" \
  --labels "$HOME/workspace/labeler/v6_failure_onset_labels.tsv" \
  --index "$HOME/workspace/labeler/index.html" \
  --host "${HOST:-0.0.0.0}" --port 8765 > server.log 2>&1 < /dev/null &
echo "$!" > server.pid

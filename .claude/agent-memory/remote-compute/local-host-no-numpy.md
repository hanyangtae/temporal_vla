---
name: local-host-no-numpy
description: Local host python3 (kanu, /usr/bin/python3) has NO numpy — verify NPZ/analysis artifacts on the remote before pulling, not locally
metadata:
  type: project
---

The local workstation's bare `python3` has no numpy (no conda/venv on PATH either); local numeric
work normally happens inside Docker. So a pulled NPZ/NPY cannot be inspected with a quick local
`python3 -c "import numpy"` one-liner.

**Why:** verified 2026-08-25 while validating `outputs/analysis/v4_pilot_viz/*.npz` — the local check
failed with ModuleNotFoundError, and the same check on `~/anaconda3/bin/python` remotely worked fine.

**How to apply:** run key/shape/dtype validation of result artifacts on the remote node (via
`remote_compute.sh run ${REMOTE_PYTHON} ...`) *before* `pull-results`; treat the local pull as a
byte copy only (`ls -la` / size check). See [[remote-python-scipy-sklearn-available]].

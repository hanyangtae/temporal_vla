---
name: remote-imageio-ffmpeg-pip-install
description: imageio is present on remote ~/anaconda3/bin/python but imageio-ffmpeg (bundled ffmpeg binary) is not — needed for any mp4 read/write via imageio; installs fine with pip --user, no root needed.
metadata:
  type: project
---

`~/anaconda3/bin/python` on the remote node has `imageio` 2.33.1 but not `imageio-ffmpeg`
(the package that bundles the ffmpeg binary imageio's ffmpeg plugin shells out to). Any script
that calls `imageio.get_reader`/`get_writer` on an `.mp4` will fail with a missing-plugin error
until this is installed.

**Why:** Discovered while running `scripts/safe/groot_n15/robocasa/vis/annotate_phase_video.py`
remotely (2026-07-01) — pip install was needed before the script could read/write rollout mp4s.

**How to apply:** `~/anaconda3/bin/python -m pip install --user imageio-ffmpeg` — works without
root/sudo on this shared node (installs to `~/.local/lib/python3.11/site-packages/`), ~30MB
download, no system ffmpeg needed afterward. Check first with
`~/anaconda3/bin/python -c "import imageio_ffmpeg"` before assuming it's missing — treat this the
same as [[remote-python-scipy-sklearn-available]]: verify per-package on a shared node rather than
assuming a persistent CLAUDE.md/agent-memory claim still holds.

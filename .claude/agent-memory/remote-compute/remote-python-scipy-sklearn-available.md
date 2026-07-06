---
name: remote-python-scipy-sklearn-available
description: ~/anaconda3/bin/python on the remote node now has scipy 1.11.4 and scikit-learn 1.2.2 installed (verified 2026-07-01), contradicting the older "scipy unavailable on every remote python" fact.
metadata:
  type: project
---

As of 2026-07-01, `~/anaconda3/bin/python` (the default `REMOTE_PYTHON` used by
`scripts/utils/remote_compute.sh`) has **scipy 1.11.4** and **scikit-learn 1.2.2** already
installed, in addition to torch 2.1.0 / numpy 1.26.4 / matplotlib 3.8.0 / PIL 10.2.0.

This contradicts the older documented fact ("scipy is unavailable on every remote python —
scipy-dependent code cannot run remotely", CLAUDE.md / agent_spec, dated 2026-06-05).

**Why:** Someone (likely kimseungjun, the remote node owner) installed scipy/scikit-learn into
the anaconda env sometime between 2026-06-05 and 2026-07-01. Confirmed by running
`scripts/safe/groot_n15/robocasa/vis/phase_distribution_tsne_lda.py` remotely, which imports
`sklearn.manifold.TSNE` and `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` — both
worked without installing anything.

**How to apply:** Before assuming scipy-dependent code "genuinely can't run remotely", actually
check with `~/anaconda3/bin/python -c "import scipy; print(scipy.__version__)"` (and similarly
for sklearn) rather than trusting the stale CLAUDE.md claim. Still verify per-package — this is a
shared node and packages can be added/removed by others outside this project's control, so
re-check rather than assuming persistence across sessions. If genuinely missing, `pip install
--user <pkg>` works without root (see [[remote-imageio-ffmpeg-pip-install]]).

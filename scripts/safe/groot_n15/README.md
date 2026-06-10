# GR00T N1.5 SAFE Script Bundle

This tree is the GR00T N1.5 script bundle. It is intentionally smaller than
`scripts/safe/groot_n16/` because this repo does not own a GR00T N1.5 backend
library or RoboCasa SAFE feature server.

Current domains:

- `robocasa/`: N1.5 RoboCasa eval clients, split helpers, and checkpoint/runtime
  compatibility helpers.

The canonical RoboCasa entrypoint is `robocasa/README.md`. Shared N1.5
RoboCasa path and run identity defaults live in `robocasa/run_config.py` for
Python scripts and `robocasa/run_config.sh` for shell recipes.

Do not add loader/schema/IO/serving/feature-capture library code here. If N1.5
needs a real reusable backend module later, design it as a separate module and
update the layout guard before adding it.

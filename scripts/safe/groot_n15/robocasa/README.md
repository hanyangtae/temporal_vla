# GR00T N1.5 RoboCasa Scripts

This directory is the canonical home for GR00T N1.5 RoboCasa scripts.

It is not the N1.5 counterpart of `src/policies/groot/`. This tree owns only
eval clients, dataset split helpers, and small checkpoint/runtime compatibility
helpers. It does not own model loaders, serving services, reusable RoboCasa IO
schema adapters, or SAFE feature capture code.

- `eval/native_zmq_eval.py`: N1.5 ZMQ client using this repo's N1.6 rollout helper API shape.
- `eval/native_official_zmq_eval.py`: benchmark-style RoboCasa client for the N1.5 ZMQ server.
- `eval/lerobot_http_eval.py`: benchmark-style RoboCasa client for LeRobot HTTP `/act`.
- `eval/internal_parity.py`: raw Isaac-GR00T checkpoint vs LeRobot-wrapped model parity diagnostics.
- `eval/run_target15_seedpairs.sh`: target atomic 15-task seed-pair evaluation wrapper.
- `split/prepare_seen5_trainval_cp_test_split.py`: seen5 train/val/test split preparation.
- `split/build_safe_splits.py`: SAFE split manifest builder.
- `split/merge_seen60_source.py`: seen60 source merge helper.
- `utils/runtime.py`: N1.5 LeRobot runtime compatibility helper.
- `utils/prepare_base_new_embodiment.py`: base checkpoint metadata preparation helper.

The shared LeRobot serve adapter remains at `scripts/serve/lerobot_adapters/groot.py`
because it belongs to the common serving surface. Old N1.5 RoboCasa paths under
`scripts/eval`, `scripts/data`, and `scripts/utils` are intentionally not kept as
compatibility wrappers.

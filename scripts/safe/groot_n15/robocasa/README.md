# GR00T N1.5 RoboCasa Scripts

This directory is the canonical home for GR00T N1.5 RoboCasa scripts.

It is not the N1.5 counterpart of `src/policies/groot/`. This tree owns only
eval clients, the LeRobot HTTP feature collection client, dataset split helpers,
and small checkpoint/runtime compatibility helpers. It does not own model
loaders, serving services, reusable RoboCasa IO schema adapters, or a SAFE
feature extractor backend.

- `eval/native_zmq_eval.py`: N1.5 ZMQ client using this repo's N1.6 rollout helper API shape.
- `eval/native_official_zmq_eval.py`: benchmark-style RoboCasa client for the N1.5 ZMQ server.
- `eval/lerobot_http_eval.py`: benchmark-style RoboCasa client for LeRobot HTTP `/act`.
- `collect/http_feature_collect.py`: LeRobot HTTP `/act_with_features` collector
  that writes N1.6 SAFE-style `pkl/csv/mp4` episode triplets for N1.5 value
  comparison. It requires the explicit N1.6-style `--env-name`; use
  `--ep-meta-dir` when replaying N1.6 scene manifests.
- `eval/internal_parity.py`: raw Isaac-GR00T checkpoint vs LeRobot-wrapped model parity diagnostics.
- `eval/run_target15_seedpairs.sh`: target atomic 15-task seed-pair evaluation wrapper.
- `split/prepare_seen5_trainval_cp_test_split.py`: seen5 train/val/test split preparation.
- `split/build_safe_splits.py`: SAFE split manifest builder.
- `split/merge_seen60_source.py`: seen60 source merge helper.
- `utils/runtime.py`: N1.5 LeRobot runtime compatibility helper.
- `utils/prepare_base_new_embodiment.py`: base checkpoint metadata preparation helper.
- `run_config.py`: shared host-side paths for Python helpers.
- `run_config.sh`: shared shell paths/run identity for eval recipes.

The shared LeRobot serve adapter remains at `scripts/serve/lerobot_adapters/groot.py`
because it belongs to the common serving surface. Old N1.5 RoboCasa paths under
`scripts/eval`, `scripts/data`, and `scripts/utils` are intentionally not kept as
compatibility wrappers.

`eval/lerobot_http_eval.py` intentionally reuses `src/policies/groot/robocasa/io.py`
for RoboCasa obs alias, state, and language extraction. This does not make this
tree a reusable N1.5 backend library; the eval client only maps the shared
unified camera names (`left/right/wrist`) to the LeRobot N1.5 names
(`side_0/side_1/wrist_0`). Its video output follows the ZMQ-style episode
contract: `<uuid>_s{0|1}.mp4` plus `per_episode.tsv`.

`collect/http_feature_collect.py` intentionally reuses the N1.6 SAFE artifact
writer so N1.5 HTTP feature rollouts land as
`task{id}--ep{idx}--succ{0|1}.{pkl,csv,mp4}`. The feature tensors remain
N1.5 LeRobot tensors (`groot_n15_dit_action_tokens_pre_decode`), not N1.6 SAFE
DiT tensors. Its env path uses the same shared RoboCasa
`VideoRecordingWrapper`/`MultiStepWrapper` stack as N1.6 collection, so video is
the canonical 3-view upstream recording and action chunks execute according to
`--n_action_steps`. Its `ep_meta` path can replay manifests exported by N1.6
when `--ep-meta-load-env-name` points at the N1.6 env id.

Verify N1.5 HTTP feature triplets with the same N1.6 collection verifier, but
override the model/feature expectations:

```bash
python scripts/safe/groot_n16/robocasa/collect/verify_rollout_collection.py \
  outputs/eval/robocasa/<n15-feature-run>/raw_rollouts \
  --tasks-override CloseFridge \
  --episodes-per-task 3 \
  --expected-feature-kind groot_n15_dit_action_tokens_pre_decode \
  --expected-hidden-shape 4,16,1024 \
  --expected-model-family lerobot_groot_n15 \
  --expected-policy-transport http \
  --expected-task-suite-name lerobot_groot_n15_robocasa \
  --expected-video-source groot_upstream_video_recording_wrapper \
  --expected-model-horizon 16 \
  --expected-valid-horizon none \
  --expected-feature-action-horizon 16 \
  --expected-n-action-steps 16
```

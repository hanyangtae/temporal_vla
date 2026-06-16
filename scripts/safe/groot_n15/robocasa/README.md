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
`raw_rollouts/<task>/task{id}--ep{idx}--succ{0|1}.{pkl,csv,mp4}`. Passing
`--output-dir raw_rollouts` is enough; the collector creates the task subdir to
match the N1.6 verifier layout. The feature tensors remain
N1.5 LeRobot tensors (`groot_n15_dit_action_tokens_pre_decode`), not N1.6 SAFE
DiT tensors. Its env path uses the same shared RoboCasa
`VideoRecordingWrapper`/`MultiStepWrapper` stack as N1.6 collection, so video is
the canonical 3-view upstream recording and action chunks execute according to
`--n_action_steps`. Its `ep_meta` path can replay manifests exported by N1.6
when `--ep-meta-load-env-name` points at the N1.6 env id. For N1.5/N1.6 paired
feature comparison, keep `env_name`, `scenario_seed`, `ep_meta`, `n_action_steps`,
`max_episode_steps`, `video_fps`, `steps_per_render`, `task_description`, and
`inference_seed` aligned; the pkl payload records these replay/video settings.

By default, N1.5 feature collection stores only DiT `hidden_states`. To collect
the matching VL(goal) pathway feature, start the LeRobot server with
`scripts/serve/lerobot.py --collect --capture-vl`; then
`collect/http_feature_collect.py` will store `vl_hidden_states` and
`vl_feature_*` metadata in the same pkl triplet. The VL point is the
`action_head.vlln` output after sequence mean pooling, using the shared
`groot_n15_vlln_seq_meanpool` metadata contract.

For pathway-resolved collection (the NOTALL/COAST steering line), start the
server so it also captures DiT block residuals at a fixed layer subset:

```bash
scripts/serve/lerobot.py --collect --capture-vl \
  --groot-dit-capture-layers 0,2,4,8,10,12,15
```

This replaces the single `groot_n15_dit_action_tokens_pre_decode` tap
(`[K=4, H=16, 1024]`) with per-inference DiT block residuals
`[L=7, T=token_count, 1536]` (the K=4 denoising axis is mean-pooled; the
per-token axis `T` is preserved). The 7 layers are the N1.5 (16-layer DiT)
mapping of the N1.6 7-layer subset used in `docs/steering/09`: early `{0,2,4,8}`
(L0 = NOTALL kill-switch), `10` = COAST N1.5 selected layer, late `{12,15}`
(separation peak; 15 = final block).

The current N1.5 aligned residual runtime `T` is 49, not 51:
`state(1) + future_tokens(32) + action(16)`. This differs from N1.6 full
block residuals, where `T=51 = state(1) + action(50)`. Do not pad/truncate N1.5
to N1.6 token count; keep the native model-token layout and verify
`token_count=49` with the first pkl/verifier gate. Each pkl record now also
stores the per-inference proprio `state` (the paper expert-vs-VL state-probe
target), surfaced as `states` in the pkl payload.

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
  --expected-n-action-steps 16 \
  --expected-max-episode-steps 720 \
  --expected-video-fps 20 \
  --expected-steps-per-render 2
```

For runs collected with `--capture-vl`, add:

```bash
  --require-vl-hidden-states \
  --expected-vl-hidden-shape 2048 \
  --expected-vl-feature-kind groot_n15_vlln_seq_meanpool \
  --expected-vl-feature-dim 2048
```

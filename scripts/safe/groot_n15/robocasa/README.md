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

For COAST-faithful pathway collection (the NOTALL/COAST steering line), start the
server so it captures DiT block residuals at all 16 layers:

```bash
scripts/serve/lerobot.py --collect --capture-vl \
  --groot-dit-capture-layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

This replaces the single `groot_n15_dit_action_tokens_pre_decode` tap
(`[K=4, H=16, 1024]`) with per-inference DiT block residuals captured COAST-faithfully
(COAST A.7.2 / `docs/references/COAST.txt:265-267,1456-1471`): the residual at
`transformer_blocks[i]` is **mean-pooled over the last `action_horizon` (=16) action
tokens**, keeping the **denoising step K=4** axis, giving `[L=16, K=4, D=1536]` per
env-step (`feature_axes = ["layer","denoise_step","feature_dim"]`,
`feature_kind = groot_n15_dit_block_residual_action_tokens_denoise`). All 16 layers are
captured so the COAST quota heuristic can select the action-expert layer; COAST's published
GR00T N1.5 global choices are L11 (6 tasks) and L5 (PickPlaceCounterToCabinet).

For the global strategy COAST stacks every denoising step as an independent sample, so the
downstream loader (`analyze/instruction_pathway_features.py --preserve-dit-layers`) expands
the K axis into rows. Each pkl record also stores the per-inference proprio `state`,
surfaced as `states` in the pkl payload.

(History: a prior variant pooled the K axis and kept all 49 model tokens
`[L=7, T=49, 1536]`; that diverged from COAST and is replaced by the action-token/denoise
capture above.)

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

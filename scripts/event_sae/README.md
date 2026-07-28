# Event-SAE remote media adapter

This directory contains the minimal remote-side adapter for Event-SAE Stage 3.
It decodes rollout MP4 files and emits a portable frame bundle. Vision models,
clustering, and VLM annotation stay in the local Event-SAE environment.

The adapter preserves the paper's five-frame temporal window. The paper uses
environment-step offsets `[-4, -2, 0, 2, 4]`. For exp3 (구 PQ3), RoboCasa runs at 20 Hz,
GR00T records one pre-action EEF state per five action steps, and the video
wrapper renders every two action steps. A waypoint record `r` is therefore
aligned to environment step `5r`, mapped to the nearest rendered frame, and
packaged with video-frame offsets `[-2, -1, 0, 1, 2]`. Encoded MP4 FPS affects
playback speed only; it is not used for simulation-time alignment.

exp3 videos contain a variable-height top banner with the task text and success
bit. The adapter removes that banner by keeping only the bottom 256 scene
pixels. The remaining scene is a horizontal three-view montage ordered as
`side_0` (`robot0_agentview_left`), `side_1` (`robot0_agentview_right`), and
`wrist_0` (`robot0_eye_in_hand`). Stage 3 uses only `side_0`, so the adapter
then keeps the leftmost 256 pixels and emits a 256 x 256 JPEG. This prevents
success from leaking into the local vision descriptor and gives every sample a
single consistent external camera. Source dimensions, camera identity, and
vertical/horizontal crop amounts remain in each frame record for audit.

## Environment

```bash
conda env create -f scripts/event_sae/environment-media.yml
conda run -n event-sae-media python scripts/event_sae/stage3_media.py --help
```

## Package

```bash
conda run -n event-sae-media python scripts/event_sae/stage3_media.py package \
  # 아래 pq3_stage3_* 는 기존 로컬 산출물 dir 구명, video-root 는 승준 원격 — 둘 다 구명(pq) 유지
  --waypoint-summary outputs/event_sae/groot_n15/pq3_stage3_inputs/waypoint_summary.json \
  --trajectory-manifest outputs/event_sae/groot_n15/pq3_stage3_inputs/trajectory_manifest.json \
  --video-root /home/kimseungjun/datasets/temporal_vla_outputs/eval/robocasa/groot_n15/phase_event_pq3/raw_rollouts \
  --output-dir outputs/event_sae/groot_n15/pq3_stage3_media/pilot_left_video5_v4 \
  --control-freq-hz 20 \
  --scene-height-pixels 256 \
  --view-name side_0 \
  --view-width-pixels 256 \
  --episode-num 0 2 30 31 60 61 90 91 120 128 \
  --expected-samples 58
```

The full run omits `--episode-num` and adds `--expected-samples 913`.

## Audit

```bash
conda run -n event-sae-media python scripts/event_sae/stage3_media.py audit \
  --bundle-dir outputs/event_sae/groot_n15/pq3_stage3_media/pilot_left_video5_v4 \
  --expected-samples 58
```

The output is intentionally limited to JPEG frames and JSON/JSONL provenance.
No clips, model weights, embeddings, or API credentials are written remotely.

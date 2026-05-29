# Use a dedicated ZMQ feature server for SAFE GR00T N1.6 wiring

Status: Superseded for new feature work by
[ADR-0002](0002-groot-n16-safe-feature-dual-transport.md). The historical
ZMQ collection decision remains valid for reproducing existing SAFE artifacts,
but HTTP `/act_with_features` is now also a supported SAFE feature transport.

For SAFE wiring with the GR00T N1.6 base checkpoint, we will add a dedicated ZMQ feature server under `scripts/safe/groot_n16/robocasa/serve/` instead of extending the general FastAPI GR00T server. This keeps SAFE instrumentation separate from the normal serving API, matches the working N1.5 SAFE collector path, and avoids mixing policy feature export with the user-facing `/act` contract.

The server exports the SAFE flow-matching feature from the action-token part of the DiT output immediately before the embodiment-specific Action Decoder. GR00T N1.6 has a model-level max action horizon of 50, but RoboCasa PandaOmron decodes only 16 action steps from its modality config. The default SAFE export is therefore `model_output[:, -50:, :][:, :16, :]`, stored as `[K, 16, D]` per environment step. The full `[K, 50, D]` action-token tensor remains available via `--feature-slice all`, but it is not the default because the last 34 token positions are not decoded/executed by the RoboCasa policy output. Collection must not pre-pool this tensor; SAFE aggregation (`first`, `last`, `mean`, `first&last` over action-step/denoising axes) belongs in detector training/evaluation so it can be selected on validation data.

The feature server captures this tensor with a temporary DiT forward hook around the normal GR00T action path. We avoid patching the vendored Isaac-GR00T model code unless an upstream interface change becomes necessary.

For rollout collection, we prefer the upstream RoboCasa evaluation client environment at `gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python` over the project `robocasa` container Python when reproducing GR00T N1.6 success-rate behavior. This keeps SAFE feature export intact while aligning simulator/client dependencies with NVIDIA's published RoboCasa evaluation path. The project FastAPI `/act` path remains useful for cross-policy serving, but it is not the reference path for diagnosing GR00T N1.6 RoboCasa success-rate drift.

This is an interim compatibility choice. If later we can make the project container reproduce the upstream RoboCasa client behavior exactly, SAFE collection can move back behind the project runner without changing the feature-server contract or SAFE pkl schema.

Status update: this ZMQ feature-server path has been used to collect the current GR00T N1.6 RoboCasa SAFE task-set split and train the final SAFE-LSTM detector. The selected detector uses `horizon_idx_rel=mean`, `diff_idx_rel=concat-2`, `lr=3e-4`, `lambda_reg=1`, and the `seed2` checkpoint. The final operating point is split conformal prediction calibrated on successful `val_seen` rollouts with `alpha=0.2`, `eval_time=by final end`, threshold `0.5301596522331238`. Functional CP band evaluation is also generated from the same detector scores. The pinned artifact lives at `outputs/eval/robocasa/groot_n16/safe_seen4_unseen2_100ep/final_detector`.

---
name: conceptor-pipeline
description: Use when running any stage of the conceptor steering workflow — activation collection, phase labeling, conceptor fit, steering serve wiring — or the pipeline end-to-end. Triggers: "conceptor fit 해줘", "activation 수집해줘", "steering 실험 새로 돌려", "phase 라벨 붙여", fit_phase_conceptor, steering_hooks, http_feature_collect.
---

# Conceptor Steering Pipeline (collect → label → fit → serve → eval)

## Overview

Runbook connecting the five stages with their settled standards. Method single
source: docs/steering/14. Per-directory READMEs hold the detailed commands —
this skill tells you which stage owns what and which standards are load-bearing.

## Stage 1 — Collect activations

- Chunk standard: **16 predicted / 5 executed** → features land at 5-env-step
  resolution; phase alignment downstream depends on this. Do not change it.
- N1.5 collector: `scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py`
  against `scripts/serve/lerobot.py` with capture flags (launchers in the same
  `collect/` dir show the wiring). N1.6: `scripts/safe/groot_n16/robocasa/collect/`.
- Keep features **per-timestep**. Never aggregate at collection time.

## Stage 2 — Phase labels

- Event-anchored labelers (N1.6, LIBERO/RoboCasa) and proximity 6/7-phase labels
  (N1.5, `--proximity-phases`) are the two supported label families.
- Proximity labels read simulator state = **oracle signal**. Any result gated on
  them is an upper bound; online phase identification is the project's open core
  problem (CLAUDE.md 연구 방향) — say so in reports.

## Stage 3 — Fit conceptors

- N1.5 phase fit: `scripts/safe/groot_n15/robocasa/steer/fit_phase_conceptor_n15.py`.
  N1.6: `scripts/safe/groot_n16/robocasa/steer/fit_conceptor_steering.py`.
- Truncation standard: window W = [mean, mean+1σ] of SUCCESS episode lengths —
  the fit script auto-computes it; do not hand-pick windows.
- Per-record features only — no episode pooling (confound-audit gate 5).
- NPZ contract: `alpha{a}_C_steer` / `_C_success` / `_C_failure` keys +
  `metadata.json`. **Record `selected_alpha`** — eval must pass it explicitly.
- Heavy fits against remote rollout data run on the remote node via the
  `remote-compute` agent (needs the torch-capable anaconda python; scipy is
  unavailable on every remote python).

## Stage 4 — Steering serve

- N1.5 (HTTP, lerobot container): `scripts/serve/lerobot.py` with
  `--steering-npz-dir <…>/global` (global) or `--steering-phase-npz-base <…>`
  (phase-gated), plus `--steering-layers L --steering-beta β --steering-alpha α
  --steering-key C_steer`. Hook implementation: `scripts/serve/steering_hooks.py`
  (action-head DiT blocks).
- N1.6 (native ZMQ): `scripts/safe/groot_n16/robocasa/serve/feature_server.py`.
- Always pass `--steering-alpha` explicitly (NPZ stores 2 alphas; unset picks
  whichever key comes first).

## Stage 5 — Eval and claims

- **REQUIRED**: run SR eval via the `robocasa-steer-eval` skill (seed standard,
  held-out gates, launch/monitor/completion/cleanup rules live there).
- **REQUIRED**: pass results through the `confound-audit` skill before reporting.

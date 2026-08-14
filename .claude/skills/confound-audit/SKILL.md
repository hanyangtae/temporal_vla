---
name: confound-audit
description: Use when about to report, document, or draw conclusions from any succ/fail separation, AUROC, LDA/t-SNE clustering, ΔSR, steering effect, or detector result in this project — before writing the report. Triggers: "결과 보고해줘", "분리가 보인다", "AUROC 나왔어", "steering 효과 있다", "SR 올랐다", "이 결과 해석해줘".
---

# Confound Audit — run before any result claim

## Overview

Every apparent success/failure signal in this project has, at least once, turned
out to be an artifact. A positive result is not reportable until it passes this
checklist. Attach the filled audit table to the report.

## The checklist

| # | Gate | Test | Incident it prevents |
|---|---|---|---|
| 1 | Length | Failures ≈ always timeout, successes end early → any time-pooled feature encodes length. Use fixed-t, dwell-matched, or truncated-window analysis | length alone: AUROC 0.998 (seen18) |
| 2 | Task identity | Latents separate tasks at AUROC ≈ 1.0. Redo within-task, or residualize task | N1.5 DiT t-SNE succ/fail "separation" was task clusters |
| 3 | Instruction balance | Per-instruction SR imbalance masquerades as latent signal. Check SR per instruction variant | SlideDishwasherRack VL AUROC 0.93 |
| 4 | In-sample rescue | fit episodes/seeds ∩ eval episodes = ∅, verified from artifacts (metadata.json, rollout filenames) | multilayer ΔSR +0.20 → held-out −0.067 |
| 5 | Rollout pooling | Per-record (per-timestep) features only; episode-mean pooling destroys the load-bearing phase axis | banned repo-wide |
| 6 | Phase/dwell | Compare within dwell-matched windows; phase composition differs succ vs fail | transport separation collapsed when dwell-matched |
| 7 | Observation ≠ causation | t-SNE/LDA/AUROC are geometry diagnostics. Performance claims require intervention (steering ΔSR re-measurement) | agent_spec §3 |
| 8 | Scene-local ≠ general | Improvement on one cell/scene is not "steering works"; require scene-consistent replication before generalizing | bread local +, apple significantly harmed |

## Output contract

The report (user-facing output in Korean) must contain, in this order:

1. Result numbers with n and evaluation scope.
2. The audit table with pass / fail / N-A per gate + one-line evidence each.
3. Claim-strength label (agent_spec §3): diagnostic evidence | detector
   performance | policy performance | intervention effect.
4. Results failing a gate are stated as "confounded — 판정 보류", never softened
   into "promising" or "encouraging".

## Red flags — STOP and re-audit

- "The separation is too clean to be an artifact" — here, clean = suspicious.
- "Held-out is probably fine, the seeds looked different" — verify from artifacts.
- "Deadline — report first, verify later" — a retracted claim costs more.
- "Pooling is OK just this once" — it never was.

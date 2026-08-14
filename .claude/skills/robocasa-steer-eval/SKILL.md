---
name: robocasa-steer-eval
description: Use when running RoboCasa SR evaluation — baseline vs steering comparison, ΔSR measurement, held-out steering eval, or any multi-hour eval run. Triggers: "steering eval 돌려줘", "SR 비교해줘", "baseline eval 돌려", "ΔSR 측정", "평가 돌려줘", eval_steer_compare, heldout_round_cell.
---

# RoboCasa Steering/Baseline SR Eval

## Overview

Fast-path for running SR evals correctly. Every value below is a settled decision
(2026-06-05+) — apply it, do not re-derive or "improve" it. Rationale lives in
CLAUDE.md "평가 표준" and docs/steering/RESEARCH_DIRECTION.md.

## Fixed standards

| Item | Value |
|---|---|
| EVAL_SEED | 100000 (launcher default; matches colleague collection seeds) |
| GPUs | (2026-07-21/08-10 개정, 구 "4/5/6 고정" 폐기) **빈 GPU만** 사용 — `nvidia-smi`로 확인, 타인이 쓰는 GPU에는 발사 금지. 한 번에 최대 3장, GPU당 serve ≤2. launcher `GPUS` 리스트를 현재 빈 GPU로 명시 지정 |
| Episodes | n16 compare: `N_ENVS=2 N_EP=20` per condition. n15 held-out cell: `EP0=30 EP1=59` (defaults — keep; disjoint from fit ep0-14) |
| Per-episode log | `per_episode.tsv` (episode_idx, success, language) must be produced |
| Where | SR eval runs LOCAL-only (robocasa Docker). Never on the remote node. |

## Launchers

- N1.6 multi-task compare: `scripts/safe/groot_n16/robocasa/steer/eval_steer_compare.sh`
- N1.5 single-cell held-out 3-arm (base/perm/gated): `scripts/safe/groot_n15/robocasa/steer/heldout_round_cell.sh`
  - Worker lanes: `GPUS_L`/`PORTS_L` (space-separated lists, equal length; GPU may
    repeat = 2 serves per GPU), or fallback pair `GPU_A GPU_B` / `PORT_A PORT_B`.
  - `SEED` here is the **cell's scenario seed** (from the cells table in
    `master_final_scene2.sh`), NOT EVAL_SEED=100000 — do not "correct" it.

## Pre-flight gates — verify ALL before launch

1. **Held-out separation**: episodes/seeds used to fit the conceptor must NOT
   overlap eval episodes. Verify from artifacts (NPZ `metadata.json`, fit rollout
   filenames) — not from memory. Incident: in-sample eval reported ΔSR +0.20;
   held-out re-run of the same conceptor gave −0.067.
2. **STEER_ALPHA explicit**: pass `selected_alpha` from `metadata.json`. If unset,
   serve loads whichever alpha key comes first in the NPZ (two alphas are stored;
   order is not meaningful).
3. **STEER_LAYERS exist**: only layers with `dit_L*` dirs under `NPZ_ROOT`
   (default `4,8,12`); a missing layer aborts serve startup.
4. **Phase granularity match**: `PROX=1` (`--proximity-phases`) iff the conceptor
   was fit on proximity 6/7-phase labels; a mismatch silently degrades the gated arm.
5. **Environment clean**: GPUs free (`nvidia-smi`), docker services up, no stale
   serve (`docker exec lerobot pgrep -f serve/lerobot.py`).

## Launching multi-hour runs

Launch with `setsid nohup … &` and verify detachment (`ps -o pid,ppid` → PPID=1).
**NEVER launch via agent background jobs**: the harness can kill them mid-run →
trap cleanup fires → empty results + fake "[done]" lines in the log (real incident).

## Monitoring

Exactly one read-only monitor: the `eval-monitor` agent. It observes and reports
only — it has no authority to relaunch, kill, or clean up.

## Completion criteria (all required)

- Sentinel file exists (n15: `logs/HELDOUT_<CELL_ID><SUF>_DONE`).
- Results complete by row count: one row per condition, each with the expected
  total (n15: 3 arms × 30 ep; n16: rows = tasks × conditions).
- `grep -E 'ABORT|Traceback' <log>` returns nothing.
- A "[done]" string in the log alone is NOT completion.

## Cleanup — before reporting done

`docker exec lerobot pgrep -f serve/lerobot.py` → kill strays;
`nvidia-smi` shows memory returned. Only then write the report.

## Reporting (user-facing output in Korean)

- SR table per condition + ΔSR vs base + n. At n=20-30/arm binomial noise is
  ±0.09-0.11: report direction, do not claim significance without a
  permutation test or additional rounds.
- **REQUIRED next step**: run the `confound-audit` skill before drawing any
  conclusion from the numbers.
- No LaTeX `$` delimiters. Do not post to Notion without user approval.

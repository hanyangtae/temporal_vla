#!/usr/bin/env python3
"""pq3 eval 큐 빌더 (계획서 v9 §E) — 4 arm × 5 cell × 30판, host 균형 배정.

입력 --arm-config (Gate D 산출 단일 출처, json):
{
  "cells": {"pq3_drawer_left": "OpenDrawer", ...},           # cell → task
  "arms": {
    "ho_base":              {"mode": "base"},
    "ho_coast_cross_scene": {"mode": "perm",
      "npz":    {"OpenDrawer": "outputs/.../cross_scene/OpenDrawer/global/dit_L15", ...},
      "layers": {"OpenDrawer": "15", ...},
      "beta":   {"OpenDrawer": "0.1", ...},
      "npz_shas": {"OpenDrawer": "abc123def456", ...}},      # Gate D 동결 sha12 (선택)
    "ho_gated_cross_scene": {...  "mode": "gated"},
    "ho_null_cross_scene":  {...  "mode": "null"}
  },
  "hosts": ["local", "local", "local", "w2", "w48"]          # lane 클래스 (반복 = 가중)
}

host 균형: (cell순서×4+arm순서) % len(hosts) round-robin — 같은 arm 이 특정 host 에
몰리지 않게 cell 마다 회전 (Codex R1 #9). 집계가 ledger 로 균형을 재-assert 한다.

큐 행: CELL=.. TAG=.. MODE=.. NPZ=..|- LAYERS=..|- BETA=..|- ALPHA=- SHAS=..|- HOST=..|0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ARM_ORDER = ["ho_base", "ho_coast_cross_scene", "ho_gated_cross_scene", "ho_null_cross_scene"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm-config", required=True)
    ap.add_argument("--qroot", required=True, help="work_queue 디렉토리")
    ap.add_argument("--expect-n", type=int, default=30)
    args = ap.parse_args()

    cfg = json.loads(Path(args.arm_config).read_text())
    cells = cfg["cells"]
    arms = cfg["arms"]
    hosts = cfg.get("hosts") or ["local"]
    missing = [a for a in ARM_ORDER if a not in arms]
    if missing:
        raise SystemExit(f"arm-config 에 arm 누락: {missing}")

    qroot = Path(args.qroot)
    for sub in ("running", "done", "failed"):
        (qroot / sub).mkdir(parents=True, exist_ok=True)

    rows = []
    host_tally: dict[tuple[str, str], int] = {}
    for ci, (cell, task) in enumerate(sorted(cells.items())):
        for ai, tag in enumerate(ARM_ORDER):
            spec = arms[tag]
            mode = spec["mode"]
            host = hosts[(ci * len(ARM_ORDER) + ai) % len(hosts)]
            host_tally[(tag, host)] = host_tally.get((tag, host), 0) + 1
            if mode == "base":
                npz = layers = beta = shas = "-"
            else:
                try:
                    npz = spec["npz"][task]
                    layers = spec["layers"][task]
                    beta = spec["beta"][task]
                except KeyError as exc:
                    raise SystemExit(f"arm-config {tag}: task {task} 항목 누락 ({exc})")
                shas = spec.get("npz_shas", {}).get(task, "-")
            rows.append(
                f"CELL={cell} TAG={tag} MODE={mode} NPZ={npz} LAYERS={layers} "
                f"BETA={beta} ALPHA=- SHAS={shas} HOST={host}|0"
            )

    (qroot / "queue.tsv").write_text("\n".join(rows) + "\n")
    print(f"[build-pq3-queue] {len(rows)} rows -> {qroot / 'queue.tsv'} "
          f"(expect {len(cells)}cell × {len(ARM_ORDER)}arm × {args.expect_n}판)")
    for (tag, host), n in sorted(host_tally.items()):
        print(f"  {tag:24s} {host:6s} {n}")


if __name__ == "__main__":
    main()

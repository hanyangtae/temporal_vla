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

host 배정: **cell 블록** — 한 cell 의 4 arm 전부 같은 host (paired contrast 에서 host
상쇄, Gate 2 중간#2). cell 들은 hosts 목록에 round-robin. 집계가 블록 준수를 재검증.

gated 성립 게이트: --gated-gate-report(pq3_gated_gate.py 산출 json)의 task 별 pass 가
아니면 해당 task 의 gated arm 행을 생성하지 않음 — H2/H3 N/A (Gate 2 높음#6).

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
    ap.add_argument("--gated-gate-report", default=None,
                    help="pq3_gated_gate.py 산출 json ({task: {pass: bool}}) — gated arm "
                         "생성의 필수 관문 (누락 시 gated arm 있으면 abort)")
    args = ap.parse_args()

    cfg = json.loads(Path(args.arm_config).read_text())
    cells = cfg["cells"]
    arms = cfg["arms"]
    hosts = cfg.get("hosts") or ["local"]
    missing = [a for a in ARM_ORDER if a not in arms]
    if missing:
        raise SystemExit(f"arm-config 에 arm 누락: {missing}")

    gated_pass: dict[str, bool] = {}
    gated_phases: dict[str, str] = {}
    gated_layer: dict[str, int] = {}
    report: dict = {}
    has_gated = any(arms[t]["mode"] == "gated" for t in ARM_ORDER if t in arms)
    if has_gated:
        if not args.gated_gate_report:
            raise SystemExit("gated arm 이 있는데 --gated-gate-report 누락 — 성립 게이트 필수")
        report = json.loads(Path(args.gated_gate_report).read_text())
        not_enforced = [t for t, v in report.items() if not v.get("enforce")]
        if not_enforced:
            raise SystemExit(
                f"gate report 가 enforce 실행이 아님: {not_enforced} — "
                "판정용은 pq3_gated_gate.py --enforce 산출만 허용"
            )
        gated_pass = {task: bool(v.get("pass")) for task, v in report.items()}
        # report 가 phase 집합·layer 의 단일 출처 (Gate2 R2 치명#1/#2) — 큐 행에 실어
        # lane→runner→serve --steering-phases 까지 전달, layer 는 arm-config 와 대조.
        gated_phases = {task: ",".join(v.get("phases") or []) for task, v in report.items()}
        gated_layer = {task: int(v.get("layer", -1)) for task, v in report.items()}

    qroot = Path(args.qroot)
    for sub in ("running", "done", "failed"):
        (qroot / sub).mkdir(parents=True, exist_ok=True)

    rows = []
    host_tally: dict[tuple[str, str], int] = {}
    skipped_gated = []
    for ci, (cell, task) in enumerate(sorted(cells.items())):
        host = hosts[ci % len(hosts)]  # cell 블록: 이 cell 의 전 arm 동일 host
        for tag in ARM_ORDER:
            spec = arms[tag]
            mode = spec["mode"]
            if mode == "gated" and not gated_pass.get(task, False):
                skipped_gated.append((cell, task))
                continue  # gated 성립 게이트 미통과 task → H2/H3 N/A
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
                shas_raw = spec.get("npz_shas", {}).get(task)
                if not shas_raw:
                    raise SystemExit(
                        f"arm-config {tag}/{task}: npz_shas 누락 — Gate D 동결 sha 는 필수 "
                        "(Gate 2 높음#9)"
                    )
                # 다중 sha 는 큐 행(KEY=VAL 공백분할)과 충돌하지 않게 콤마로 직렬화
                if isinstance(shas_raw, (list, tuple)):
                    shas = ",".join(shas_raw)
                else:
                    shas = str(shas_raw).replace(" ", ",")
            phases = "-"
            if mode == "gated":
                phases = gated_phases.get(task) or "-"
                if phases == "-":
                    raise SystemExit(f"gate report 에 task {task} phases 누락")
                if gated_layer.get(task) != int(str(layers)):
                    raise SystemExit(
                        f"gated layer 불일치: arm-config {layers} != gate report "
                        f"{gated_layer.get(task)} (task {task})"
                    )
                # 배포 NPZ 결박: gate 가 검증한 npz sha 집합 == arm-config 의 sha 집합
                # (다른 데이터/α 로 만든 NPZ 의 우회 배포 봉쇄 — Gate2 R3 치명#1)
                report_shas = set(report[task].get("npz_shas") or [])
                config_shas = {s for s in shas.split(",") if s}
                if not report_shas or report_shas != config_shas:
                    raise SystemExit(
                        f"gated NPZ sha 결박 실패 (task {task}): gate report "
                        f"{sorted(report_shas)} != arm-config {sorted(config_shas)}"
                    )
            rows.append(
                f"CELL={cell} TAG={tag} MODE={mode} NPZ={npz} LAYERS={layers} "
                f"BETA={beta} ALPHA=- SHAS={shas} PHASES={phases} HOST={host}|0"
            )

    (qroot / "queue.tsv").write_text("\n".join(rows) + "\n")
    print(f"[build-pq3-queue] {len(rows)} rows -> {qroot / 'queue.tsv'} "
          f"(expect {len(cells)}cell × arm × {args.expect_n}판, cell-블록 host)")
    if skipped_gated:
        print(f"  gated 성립 게이트 미통과로 제외: {sorted(set(skipped_gated))} (H2/H3 N/A)")
    for (tag, host), n in sorted(host_tally.items()):
        print(f"  {tag:24s} {host:6s} {n}")


if __name__ == "__main__":
    main()

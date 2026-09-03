#!/usr/bin/env python3
"""index_rollouts_v5.tsv 생성 — v5 재수집(plan 8daefeabf020, 전 셀 지터, base 없음).

입력: build_grid_index.py 산출 rollouts.tsv (아카이브 전체; 구 plan 껍데기 행 포함)
규칙 (docs/04 §3.1.1 · handoff_20260902_grid_recollect_v5 §0.5):
  - v5 plan 행만: cell_si = 원본 scene_idx(평탄), scene_idx = cell_si//100,
    jitter_reset_idx = cell_si%100 (정수). adopted_cells 밖 행은 제외(있으면 안 됨 → 경고).
  - base 행 없음. 결손은 plan adopted_cells 대비로 출력.
사용: make_v5_index.py <rollouts.tsv> <v5 collection_plan.json> <out.tsv>
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

src, plan_p, out_p = sys.argv[1], sys.argv[2], sys.argv[3]
P = json.loads(Path(plan_p).read_text())
pid = P["plan_id"]
adopted = set(P["extra"]["adopted_cells"])
assign = {i: m for m, il in P["extra"]["machine_assignment"].items() for i in il}

rows_in = list(csv.DictReader(open(src, newline=""), delimiter="\t"))
fields = list(rows_in[0].keys())
for c in ("cell_si", "jitter_reset_idx"):
    if c not in fields:
        fields.append(c)

out, seen, stray = [], set(), []
for r in rows_in:
    if r["plan_id"] != pid:
        continue
    si, ni = int(r["scene_idx"]), int(r["noise_idx"])
    key = f"{r['grid_instruction']}|s{si}|n{ni}"
    if key not in adopted:
        stray.append(key); continue
    if key in seen:
        sys.exit(f"중복 좌표: {key}")
    seen.add(key)
    r["cell_si"], r["scene_idx"], r["jitter_reset_idx"] = str(si), str(si // 100), str(si % 100)
    out.append(r)

missing = sorted(adopted - seen)
with open(out_p, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
    w.writeheader(); w.writerows(out)

succ = sum(1 for r in out if r.get("success") == "1")
print(f"index_v5: {len(out)}행 / 계획 {len(adopted)} (결손 {len(missing)}, 계획 밖 {len(stray)}), 성공 {succ}")
if missing:
    print("  결손:", missing[:20], "..." if len(missing) > 20 else "")
per, mach = {}, {}
for r in out:
    k = r["grid_instruction"]
    per.setdefault(k, [0, 0]); per[k][0] += 1; per[k][1] += 1 if r.get("success") == "1" else 0
    mach.setdefault(k, set()).add(r["machine"])
for k in sorted(per):
    n, s = per[k]
    print(f"  {k:<22} {n:4d} SR={s/n:.2f} machine={sorted(mach[k])} (배정 {assign.get(k)})")

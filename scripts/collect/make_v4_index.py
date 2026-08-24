#!/usr/bin/env python3
"""index_rollouts_v4.tsv 생성 — v4 지터 셀 + v2 base 셀 병합 (exp6 좌표 계약).

입력: build_grid_index.py 산출 rollouts.tsv (전 plan 포함)
규칙 (docs/04 §3.1.1 · exp6 합의):
  - v4 plan 행: cell_si = 원본 scene_idx(평탄), scene_idx = cell_si//100,
    jitter_reset_idx = cell_si%100 (정수)
  - v2 base 행 중 s0-4 × n0-4 만 포함: cell_si = scene_idx*100 + 99,
    jitter_reset_idx = "base"
  - 그 외 행(v1·v2 확장·v3 파일럿)은 제외
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

src, v4_plan_p, v2_plan_p, out_p = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
V4 = json.loads(Path(v4_plan_p).read_text())
V2 = json.loads(Path(v2_plan_p).read_text())
v4_id, v2_id = V4["plan_id"], V2["plan_id"]
adopted = set(V4["extra"]["adopted_cells"])
tasks = set(V4["instructions"])

# base 후보 plan (v3 지터 파일럿 8ae7… 은 제외 — 다른 축)
PRIORITY = [v2_id, "b8054b5e7258", "979d4833a7db"]
BASE_PLANS = set(PRIORITY)

rows_in = list(csv.DictReader(open(src, newline=""), delimiter="\t"))
fields = list(rows_in[0].keys())
for c in ("cell_si", "jitter_reset_idx"):
    if c not in fields:
        fields.append(c)

out = []
for r in rows_in:
    pid, instr = r["plan_id"], r["grid_instruction"]
    si, ni = int(r["scene_idx"]), int(r["noise_idx"])
    if pid == v4_id:
        if f"{instr}|s{si}|n{ni}" not in adopted:
            continue
        r["cell_si"], r["scene_idx"] = str(si), str(si // 100)
        r["jitter_reset_idx"] = str(si % 100)
    elif (pid in BASE_PLANS and instr in tasks and si < 5 and ni < 5):
        # base = 좌표 기준 (v1 979d/b805 + v2 3134 수집분 혼재). 같은 좌표가 여러
        # plan 에 있으면 최신(v2 > v1b > v1) 우선 — 아래 dedup 에서 처리.
        r["cell_si"] = str(si * 100 + 99)
        r["jitter_reset_idx"] = "base"
    else:
        continue
    out.append(r)

# base 좌표 중복 정리: PRIORITY 앞쪽 plan 우선
best: dict[tuple, dict] = {}
final = []
for r in out:
    if r["jitter_reset_idx"] != "base":
        final.append(r)
        continue
    key = (r["grid_instruction"], r["scene_idx"], r["noise_idx"])
    cur = best.get(key)
    if cur is None or PRIORITY.index(r["plan_id"]) < PRIORITY.index(cur["plan_id"]):
        best[key] = r
final.extend(best.values())
out = final

with open(out_p, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
    w.writeheader()
    w.writerows(out)

n_j = sum(1 for r in out if r["jitter_reset_idx"] != "base")
n_b = len(out) - n_j
succ = sum(1 for r in out if r.get("success") == "1")
print(f"index_v4: {len(out)}행 (지터 {n_j} + base {n_b}), 성공 {succ}")
per = {}
for r in out:
    per.setdefault(r["grid_instruction"], [0, 0])
    per[r["grid_instruction"]][0] += 1
    per[r["grid_instruction"]][1] += 1 if r.get("success") == "1" else 0
for k in sorted(per):
    n, s = per[k]
    print(f"  {k:<22} {n:4d} SR={s/n:.2f}")

#!/usr/bin/env python3
"""v2 replay 번들 stem → 아카이브 video.mp4 상대경로 manifest 생성.

stem 규약: <machine>_<instr_flat>_s<i>_n<j>  (국내투고 bundle_all 산출).
instr_flat 은 grid_instruction 의 '/'→'_' 평탄화 — 역매핑은 plan 의 instruction
목록에서 유도한다 (추측 금지: plan 에 있는 키만 인정).

사용: python3 make_v2_manifest.py <plan.json> <stem목록파일|-> <out.tsv>
출력 행: stem\t<plan_id>/<machine>/<instruction>/s<i>/n<j>/base/video.mp4
"""
import json
import re
import sys

plan_p, stems_p, out_p = sys.argv[1], sys.argv[2], sys.argv[3]
plan = json.load(open(plan_p))
plan_id = plan["plan_id"]
flat2instr = {k.replace("/", "_"): k for k in plan["instructions"]}

stems = (sys.stdin if stems_p == "-" else open(stems_p)).read().split()
rows, bad = [], []
pat = re.compile(r"^(kanu|worker1|worker2|dongkyu-MS-7D43)_(.+)_s(\d+)_n(\d+)$")
for s in stems:
    m = pat.match(s)
    if not m:
        bad.append(s)
        continue
    machine, flat, si, ni = m.groups()
    instr = flat2instr.get(flat)
    if instr is None:
        bad.append(s)
        continue
    rows.append(f"{s}\t{plan_id}/{machine}/{instr}/s{si}/n{ni}/base/video.mp4")

open(out_p, "w").write("\n".join(rows) + "\n")
print(f"manifest {len(rows)}행 → {out_p}; 매핑 실패 {len(bad)}: {bad[:5]}")

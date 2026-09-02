#!/usr/bin/env python3
"""n15_grid_v4 지터 확대 plan (사용자 재지시: 10 task × scene5 × noise5 × k5).

- base = 기존 v2 셀 (s0-4 × n0-4, k=base 라벨 — 신규 수집 없음).
- 신규 = k0..3 4개/scene: 평탄 si = base_scene*100 + k (docs/04 §3.1.1),
  단 k 는 k-스캔 채택분에서 앞 4개 (목표 instruction 일치; drawer 계열은 방향 필터).
- noise = v2 와 동일 대역 n0-4 (1300000..1300004) — base 와 축 정합.
- 산출: configs/collect/n15_grid_v4/collection_plan.json + kscan_adopted.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KDIR = REPO / "outputs/collect/v4_jitter/kscan"
OUT = REPO / "configs/collect/n15_grid_v4"
V2 = json.loads((REPO / "configs/collect/n15_grid_v2/collection_plan.json").read_text())

N_K = 4      # 신규 k 개수 (base 포함 5 상태)
N_SCENE = 5  # v2 s0-4
N_INF = 5    # v2 n0-4

instructions: dict[str, list[int]] = {}
adopted_cells: list[str] = []
adopted: dict[str, dict] = {}
for instr, seeds in V2["instructions"].items():
    want = V2["extra"]["instruction_text"][instr]
    slug = instr.replace("/", "_")
    cell_of: dict[int, int] = {}
    kmax = 0
    for s_i in range(N_SCENE):
        es = seeds[s_i]
        f = KDIR / f"{slug}__s{s_i}__es{es}.tsv"
        rows = [(int(l.split("\t")[0]), l.split("\t", 1)[1].strip())
                for l in f.read_text().splitlines() if l.strip()]
        ok = [k for k, lang in rows if lang == want]
        bad = [k for k, lang in rows if lang != want]
        if len(ok) < N_K:
            sys.exit(f"{instr} s{s_i} es{es}: 채택 k {len(ok)} < {N_K} — 스캔 확장 필요")
        use = ok[:N_K]
        adopted[f"{instr}|s{s_i}|es{es}"] = {"adopted_k": use, "rejected_k": bad,
                                             "scanned": len(rows)}
        for k in use:
            si = s_i * 100 + k
            cell_of[si] = es
            kmax = max(kmax, k)
            for n in range(N_INF):
                adopted_cells.append(f"{instr}|s{si}|n{n}")
    L = (N_SCENE - 1) * 100 + kmax + 1
    dummy = seeds[0]
    instructions[instr] = [cell_of.get(si, dummy) for si in range(L)]

extra = {
    "env_names": dict(V2["extra"]["env_names"]),
    "instruction_text": dict(V2["extra"]["instruction_text"]),
    "adopted_cells": sorted(adopted_cells),
    "jitter_coord": "docs/04 §3.1.1 — si=base_scene*100+k; base 상태(v2 셀)는 이 plan "
                    "밖(3134e339de4c)이며 index_v4 에서 jitter_reset_idx=base 로 병합",
    "base_plan_id": V2["plan_id"],
}
plan = {k: V2[k] for k in ("model", "version", "ckpt", "capture_layers",
                           "denoise_k", "token_mode")}
plan.update({
    "name": "n15_grid_v4_jitter",
    "note": ("v4 지터 확대(사용자 재지시): 10 task × v2 s0-4 × v2 n0-4 × 신규 k0-3 "
             "(+base 재사용 = 5상태). 수집기 --jitter-reset-idx 필수, adopted_cells 만."),
    "instructions": instructions,
    "noise_seeds": V2["noise_seeds"][:N_INF],
    "extra": extra,
})

sys.path.insert(0, str(REPO))
from src.collect.plan import CollectionPlan  # noqa: E402

cp = CollectionPlan(**{k: plan[k] for k in
                       ("name", "model", "version", "ckpt", "capture_layers",
                        "denoise_k", "token_mode", "instructions", "noise_seeds",
                        "note", "extra")})
OUT.mkdir(parents=True, exist_ok=True)
p = cp.save(OUT)
(OUT / "kscan_adopted.json").write_text(json.dumps(adopted, indent=2, ensure_ascii=False))
saved = json.loads(Path(p).read_text())
print("plan_id:", saved["plan_id"], "수집 대상 셀:", len(adopted_cells))
rej = {k: len(v["rejected_k"]) for k, v in adopted.items() if v["rejected_k"]}
print("기각 있는 scene:", len(rej), "/", len(adopted))

#!/usr/bin/env python3
"""n15_grid_v5_scenario plan (2026-09-02 재수집 계약 — handoff_20260902_grid_recollect_v5 §0).

- 10 instruction × v2 s0-4 × v2 n0-4 × **신규 k 5** = 1,250판. base(v2 셀) 재사용 없음.
- k 는 k-스캔 채택분(목표 instruction 일치)에서 앞 5개. 스캔 원본 =
  configs/collect/ledger_20260902_purge/kscan_v4/ (N=12, 50/50 scene 채택 k ≥ 5).
- 좌표 = 평탄 si = base_scene*100 + k (docs/04 §3.1.1). adopted_cells 만 수집 대상.
- 머신 배정(사용자 지정 2026-09-02, = 향후 replay 홈) 을 extra["machine_assignment"] 에 박는다.
- 산출: configs/collect/n15_grid_v5_scenario/collection_plan.json + kscan_adopted.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KDIR = REPO / "configs/collect/ledger_20260902_purge/kscan_v4"
OUT = REPO / "configs/collect/n15_grid_v5_scenario"
V2 = json.loads((REPO / "configs/collect/n15_grid_v2/collection_plan.json").read_text())

N_K = 5      # 신규 k 개수 (base 없음)
N_SCENE = 5  # v2 s0-4
N_INF = 5    # v2 n0-4

# 사용자 지정 (2026-09-02): srv50 = PPCC 4종, srv48 = drawer 2종 + coffee, kanu = rack 2종 + apple
MACHINE_ASSIGNMENT = {
    "srv50": ["PPCC/bread", "PPCC/candle", "PPCC/jug", "PPCC/marshmallow"],
    "srv48": ["OpenDrawer/left", "OpenDrawer/right", "CoffeeSetupMug"],
    "kanu": ["DishwasherRack/out", "OvenRack/out", "PPCC/apple"],
}
assigned = [i for v in MACHINE_ASSIGNMENT.values() for i in v]
assert sorted(assigned) == sorted(V2["instructions"]), (assigned, list(V2["instructions"]))

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
    "machine_assignment": MACHINE_ASSIGNMENT,
    "jitter_coord": "docs/04 §3.1.1 — si=base_scene*100+k; base 상태 재사용 없음(전 셀 k). "
                    "scene 축 = v2 plan 3134e339de4c 의 s0-4 env_seed, noise 축 = v2 n0-4.",
    "scene_seed_source_plan_id": V2["plan_id"],
    "kscan_source": "configs/collect/ledger_20260902_purge/kscan_v4 (N=12 스캔, 목표 instruction 일치 k 앞 5개)",
    "scenario": "handoff_20260902_grid_recollect_v5 §0.2 — 같은 작업장 소변화(배치 k) 에서 "
                "activation 감지→steering; 전제 데이터 ①expert ②과거 같은 scene rollout ③현재 scene 실패 rollout",
}
plan = {k: V2[k] for k in ("model", "version", "ckpt", "capture_layers",
                           "denoise_k", "token_mode")}
plan.update({
    "name": "n15_grid_v5_scenario",
    "note": ("v5 재수집(2026-09-02, 전량 폐기 후): 10 task × v2 s0-4 × v2 n0-4 × 신규 k 5 "
             "= 1,250판. 수집기 --jitter-reset-idx 필수, adopted_cells 만. 수집 경로 = eval 경로 "
             "(첫 셀 fresh replay bit 재현 게이트 통과 후 본수집)."),
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
per_m = {m: sum(1 for c in adopted_cells if c.split("|")[0] in v) for m, v in MACHINE_ASSIGNMENT.items()}
print("머신별 판수:", per_m)
rej = {k: len(v["rejected_k"]) for k, v in adopted.items() if v["rejected_k"]}
print("기각 있는 scene:", len(rej), "/", len(adopted))

#!/usr/bin/env python3
"""n15_grid_v5_scenario plan (2026-09-02 재수집 계약 — handoff_20260902_grid_recollect_v5 §0).

- 10 instruction × v2 s0-4 × 신규 k 5 × v2 n0-4 = 1,250판. base(v2 셀) 재사용 없음.
- k 는 k-스캔 채택분(목표 instruction 일치)에서 앞 5개. 스캔 원본 =
  configs/collect/ledger_20260902_purge/kscan_v4/ (N=12, 50/50 scene 채택 k >= 5).
- **좌표 = 3축 폴더층** `s<i>/k<r>/n<j>` (docs/04 §3.1.1, 2026-09-03 개정). 구 평탄
  si=base*100+k 인코딩과 `extra["adopted_cells"]` 는 폐지 — plan 의 `jitter` 가 곧
  수집 대상 전부다. 재배치 이력은 extra["supersedes_plan_id"] (구 plan 8daefeabf020).
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

N_K = 5      # scene 당 신규 k 개수 (base 상태 재사용 없음)
N_SCENE = 5  # v2 s0-4
N_INF = 5    # v2 n0-4

# 구 v5 계약 plan(평탄 si). 같은 그리드를 3축 층으로 재발급하므로 이력을 남긴다.
SUPERSEDES_PLAN_ID = "8daefeabf020"

# 사용자 지정 (2026-09-02): srv50 = PPCC 4종, srv48 = drawer 2종 + coffee, kanu = rack 2종 + apple
MACHINE_ASSIGNMENT = {
    "srv50": ["PPCC/bread", "PPCC/candle", "PPCC/jug", "PPCC/marshmallow"],
    "srv48": ["OpenDrawer/left", "OpenDrawer/right", "CoffeeSetupMug"],
    "kanu": ["DishwasherRack/out", "OvenRack/out", "PPCC/apple"],
}
assigned = [i for v in MACHINE_ASSIGNMENT.values() for i in v]
assert sorted(assigned) == sorted(V2["instructions"]), (assigned, list(V2["instructions"]))

instructions: dict[str, list[int]] = {}
jitter: dict[str, list[list[int]]] = {}
adopted: dict[str, dict] = {}
for instr, seeds in V2["instructions"].items():
    want = V2["extra"]["instruction_text"][instr]
    slug = instr.replace("/", "_")
    ks_per_scene: list[list[int]] = []
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
        ks_per_scene.append(use)
    # instructions[instr] = base scene env_seed 목록(길이 = scene 수). dummy 슬롯 없음.
    instructions[instr] = [int(seeds[s_i]) for s_i in range(N_SCENE)]
    jitter[instr] = ks_per_scene

extra = {
    "env_names": dict(V2["extra"]["env_names"]),
    "instruction_text": dict(V2["extra"]["instruction_text"]),
    "machine_assignment": MACHINE_ASSIGNMENT,
    "jitter_coord": "docs/04 §3.1.1 (2026-09-03 개정) — 좌표는 3축 폴더층 s<i>/k<r>/n<j>. "
                    "s<i> = base scene(= instructions[instr][i] 의 env_seed), "
                    "k<r> = jitter[instr][i] 의 채택 k 값 그대로(연속 아님), n<j> = noise. "
                    "base 상태 재사용 없음(전 셀 k). 평탄 si=base*100+k 인코딩은 폐지.",
    "scene_seed_source_plan_id": V2["plan_id"],
    "supersedes_plan_id": SUPERSEDES_PLAN_ID,
    "kscan_source": "configs/collect/ledger_20260902_purge/kscan_v4 (N=12 스캔, 목표 instruction 일치 k 앞 5개)",
    "scenario": "handoff_20260902_grid_recollect_v5 §0.2 — 같은 작업장 소변화(배치 k) 에서 "
                "activation 감지→steering; 전제 데이터 ①expert ②과거 같은 scene rollout ③현재 scene 실패 rollout",
}
plan = {k: V2[k] for k in ("model", "version", "ckpt", "capture_layers",
                           "denoise_k", "token_mode")}
plan.update({
    "name": "n15_grid_v5_scenario",
    "note": ("v5 재수집(2026-09-02, 전량 폐기 후): 10 task × v2 s0-4 × 신규 k 5 × v2 n0-4 "
             "= 1,250판. 수집기 --jitter-reset-idx 필수(3축 plan 이라 resolve_grid 가 강제). "
             "좌표는 s<i>/k<r>/n<j> 3축 폴더층(2026-09-03 개정, 구 평탄 si plan "
             f"{SUPERSEDES_PLAN_ID} 대체). 수집 경로 = eval 경로 "
             "(첫 셀 fresh replay bit 재현 게이트 통과 후 본수집)."),
    "instructions": instructions,
    "noise_seeds": V2["noise_seeds"][:N_INF],
    "extra": extra,
    "jitter": jitter,
})

sys.path.insert(0, str(REPO))
from src.collect.plan import CollectionPlan  # noqa: E402

cp = CollectionPlan(**{k: plan[k] for k in
                       ("name", "model", "version", "ckpt", "capture_layers",
                        "denoise_k", "token_mode", "instructions", "noise_seeds",
                        "note", "extra", "jitter")})
OUT.mkdir(parents=True, exist_ok=True)
p = cp.save(OUT)
(OUT / "kscan_adopted.json").write_text(json.dumps(adopted, indent=2, ensure_ascii=False))
saved = json.loads(Path(p).read_text())
cells = list(cp.cells())
print("plan_id:", saved["plan_id"], "수집 대상 셀:", len(cells), "(n_cells:", saved["n_cells"], ")")
per_m = {m: sum(1 for c in cells if c.instruction in v) for m, v in MACHINE_ASSIGNMENT.items()}
print("머신별 판수:", per_m)
rej = {k: len(v["rejected_k"]) for k, v in adopted.items() if v["rejected_k"]}
print("기각 있는 scene:", len(rej), "/", len(adopted))
print("샘플 rel_path:", cells[0].key, "->", cells[0].rel_path(saved["plan_id"], "<machine>"))

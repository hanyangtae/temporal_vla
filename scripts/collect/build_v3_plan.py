#!/usr/bin/env python3
"""n15_grid_v3 지터 plan 생성 (요청서 §7 + exp6 좌표 계약).

좌표 계약 (exp6 합의):
  - pkl/디렉토리 좌표(2축, 기존 인프라 무수정 재사용): scene_idx 자리 = **평탄
    cell_idx = base_scene*100 + reset_idx(k)**, noise_idx = inference 축(0..1).
  - index_rollouts_v3.tsv 정본 = 명시 3축(scene_idx 0..2, reset_idx k, noise_idx)
    — 인덱서 후처리에서 si//100, si%100 로 복원.
  - plan instructions[instr] 는 길이 (200+kmax+1) 리스트로 si=scene*100+k 위치에
    base_es 를 둔다 (나머지는 dummy — extra["adopted_cells"] 에 든 셀만 수집).

입력: outputs/collect/v3_jitter/kscan_*.tsv (k \t instruction)
출력: configs/collect/n15_grid_v3/collection_plan.json + kscan_adopted.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KDIR = REPO / "outputs/collect/v3_jitter"
OUT = REPO / "configs/collect/n15_grid_v3"
V1 = json.loads((REPO / "configs/collect/n15_grid_v1/collection_plan.json").read_text())

SPEC = {
    "OpenDrawer/left": ("Open the left drawer.",
                        [("kscan_drawer", 100001), ("kscan_drawer", 100017),
                         ("kscan_drawer", 100019)]),
    "PPCC/candle": ("Pick the candle from the counter and place it in the cabinet.",
                    [("kscan_candle", 100214), ("kscan_candle", 100741),
                     ("kscan_candle", 100154)]),
}
N_K = 20
N_INF = 2

instructions: dict[str, list[int]] = {}
adopted_cells: list[str] = []
adopted: dict[str, dict] = {}
for instr, (want, scenes) in SPEC.items():
    cell_of: dict[int, int] = {}   # si(평탄) -> base_es
    kmax = 0
    for s_i, (prefix, es) in enumerate(scenes):
        rows = [(int(l.split("\t")[0]), l.split("\t", 1)[1].strip())
                for l in (KDIR / f"{prefix}_{es}.tsv").read_text().splitlines()
                if l.strip()]
        ok = [k for k, lang in rows if lang == want]
        bad = [k for k, lang in rows if lang != want]
        if len(ok) < N_K:
            sys.exit(f"{instr} es{es}: 채택 k {len(ok)} < {N_K} — 스캔 확장 필요")
        use = ok[:N_K]
        adopted[f"{instr}|scene{s_i}|es{es}"] = {
            "adopted_k": use, "rejected_k": bad, "scanned": len(rows)}
        for k in use:
            si = s_i * 100 + k
            cell_of[si] = es
            kmax = max(kmax, k)
            for n in range(N_INF):
                adopted_cells.append(f"{instr}|s{si}|n{n}")
    L = 200 + kmax + 1
    dummy = scenes[0][1]
    instructions[instr] = [cell_of.get(si, dummy) for si in range(L)]

extra = {
    "env_names": {k: V1["extra"]["env_names"][k] for k in instructions},
    "instruction_text": {k: SPEC[k][0] for k in instructions},
    "adopted_cells": sorted(adopted_cells),
    "jitter_coord": "scene_idx(평탄 si)=base_scene*100+reset_idx; base_scene=si//100, "
                    "reset_idx=si%100; adopted_cells 밖 셀은 dummy(수집 금지)",
}
plan = {k: V1[k] for k in ("model", "version", "ckpt", "capture_layers",
                           "denoise_k", "token_mode")}
plan.update({
    "name": "n15_grid_v3_jitter",
    "note": ("v3 지터 축(요청서 §7·exp6 좌표 계약): ep_meta 고정+연속 reset, "
             "수집기 --jitter-reset-idx(si%100) 필수, adopted_cells 만 수집."),
    "instructions": instructions,
    "noise_seeds": [200000, 300000],
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
(OUT / "kscan_adopted.json").write_text(
    json.dumps(adopted, indent=2, ensure_ascii=False))
saved = json.loads(Path(p).read_text())
print("plan_id:", saved["plan_id"], "수집 대상 셀:", len(adopted_cells))
for k, v in adopted.items():
    print(f"  {k}: k {v['adopted_k'][:8]}… 기각 {len(v['rejected_k'])}/{v['scanned']}")

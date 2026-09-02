#!/usr/bin/env python3
"""n15_grid_v2 확장 plan 생성 (docs/steering/42 §7 후속 — s+5·n+5).

입력:
  - v1 plan (configs/collect/n15_grid_v1/collection_plan.json): 공통 필드·8 instruction 정본
  - index_rollouts.tsv (dedup 후): 10 task의 scene_idx→env_seed 실물 매핑 (jug·marshmallow 포함)
  - seed_scan_v2/*.tsv: 신규 seed 대역(100742~)의 seed→instruction 스캔

출력:
  - configs/collect/n15_grid_v2/collection_plan.json
      instructions = 기존 scene seed 10개 + 신규 5개 append (순서 = scene_idx 0..14)
      noise_seeds  = v1 그대로 (40개; 수집은 NOISE_LIMIT=15 로 n0..14)
  - configs/collect/n15_grid_v2/done_prefill.txt : v1 기수집 100셀(instr|sX|nY) 목록
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
V1 = REPO / "configs/collect/n15_grid_v1/collection_plan.json"
INDEX = REPO / "outputs/steer/online_pipe/manifests/index_rollouts.tsv"
SCAN = REPO / "outputs/collect/seed_scan_v2"
OUT = REPO / "configs/collect/n15_grid_v2"

ENV_OF = {  # grid_instruction → scan tsv 파일명 (env 단위)
    "PPCC/bread": "PickPlaceCounterToCabinet", "PPCC/apple": "PickPlaceCounterToCabinet",
    "PPCC/candle": "PickPlaceCounterToCabinet", "PPCC/jug": "PickPlaceCounterToCabinet",
    "PPCC/marshmallow": "PickPlaceCounterToCabinet",
    "OpenDrawer/left": "OpenDrawer", "OpenDrawer/right": "OpenDrawer",
    "CoffeeSetupMug": "CoffeeSetupMug", "DishwasherRack/out": "SlideDishwasherRack",
    "OvenRack/out": "SlideOvenRack",
}


def main() -> int:
    v1 = json.loads(V1.read_text())

    # 1) scene_idx→env_seed·instruction 복원: v1 plan(8키, pdk 삭제와 무관한 정본)
    #    + index(jug·marshmallow 등 plan 밖 task). pdk 행 삭제 후에도 OvenRack은
    #    v1 plan에서 복원된다.
    scene_seed: dict[str, dict[int, int]] = defaultdict(dict)
    instr_text: dict[str, str] = dict(v1.get("extra", {}).get("instruction_text", {}))
    env_name: dict[str, str] = dict(v1.get("extra", {}).get("env_names", {}))
    for gi, seeds in v1["instructions"].items():
        for i, s in enumerate(seeds):
            scene_seed[gi][i] = int(s)
    with INDEX.open() as f:
        rd = csv.DictReader(f, delimiter="\t")
        for r in rd:
            gi = r["grid_instruction"]
            scene_seed[gi].setdefault(int(r["scene_idx"]), int(r["env_seed"]))
            instr_text.setdefault(gi, r["instruction"])
            env_name.setdefault(gi, r["env_name"])

    # 2) scan에서 신규 scene 후보 (canonical instruction 일치, 기존 seed 제외)
    used = {s for m in scene_seed.values() for s in m.values()}
    new_scenes: dict[str, list[int]] = {}
    for gi, env in ENV_OF.items():
        if gi not in scene_seed:
            continue
        want = instr_text[gi]
        cands = []
        for tsv in (SCAN / f"{env}.tsv", SCAN / f"{env}_ext.tsv"):
            if not tsv.exists():
                continue
            with tsv.open() as f:
                for line in f:
                    seed_s, _, text = line.rstrip("\n").partition("\t")
                    if text.strip() == want and int(seed_s) not in used:
                        cands.append(int(seed_s))
        new_scenes[gi] = cands[:5]
        if len(new_scenes[gi]) < 5:
            print(f"WARN {gi}: 신규 scene 후보 {len(new_scenes[gi])}/5 (스캔 대역 확장 필요)")

    # 3) v2 plan 조립
    instructions = {}
    for gi, m in scene_seed.items():
        base = [m[i] for i in sorted(m)]
        instructions[gi] = base + new_scenes.get(gi, [])
    extra = dict(v1.get("extra") or {})
    for key, src in (("env_names", env_name), ("instruction_text", instr_text)):
        d = dict(extra.get(key) or {})
        for gi in instructions:
            d.setdefault(gi, src[gi])
        extra[key] = d

    plan = {k: v1[k] for k in ("model", "version", "ckpt", "capture_layers",
                               "denoise_k", "token_mode", "noise_seeds")}
    plan["name"] = "n15_grid_v2"
    plan["note"] = ("v1(10×10) 확장: scene +5(스캔 대역 100742~)·noise n10~14. "
                    "기존 100셀은 done_prefill로 스킵. pdk 수집 금지(42 §7 렌더 비결정).")
    plan["instructions"] = instructions
    plan["extra"] = extra

    import sys
    sys.path.insert(0, str(REPO))
    from src.collect.plan import CollectionPlan  # noqa: E402
    cp = CollectionPlan(**{k: plan[k] for k in
                           ("name", "model", "version", "ckpt", "capture_layers",
                            "denoise_k", "token_mode", "instructions", "noise_seeds",
                            "note", "extra")})
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = cp.save(OUT)

    # 4) 기수집 셀 prefill (v1 실물 기준: index의 (gi, s, n))
    #    단 pdk(dongkyu-MS-7D43) 수집분은 prefill에서 제외 → 재수집 대상 (42 §7):
    #    OvenRack 전체 100 + marshmallow 잔존 11셀이 여기 해당.
    lines = set()
    with INDEX.open() as f:
        rd = csv.DictReader(f, delimiter="\t")
        for r in rd:
            if r["machine"] == "dongkyu-MS-7D43":
                continue
            lines.add(f"{r['grid_instruction']}|s{r['scene_idx']}|n{r['noise_idx']}")
    pf = OUT / "done_prefill.txt"
    pf.write_text("\n".join(sorted(lines)) + "\n")

    saved = json.loads(Path(out_path).read_text())
    print("plan_id:", saved["plan_id"], "n_cells:", saved["n_cells"])
    print("prefill:", len(lines), "→", pf)
    for gi in sorted(instructions):
        print(f"  {gi}: scenes {len(instructions[gi])} (신규 {len(new_scenes.get(gi, []))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

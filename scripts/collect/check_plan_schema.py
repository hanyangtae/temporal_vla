#!/usr/bin/env python3
"""collection plan 스키마 검증 — k(지터) 층 리팩터링 회귀 게이트.

docs/04 §3.1.1 이 요구하는 세 가지를 코드로 확인한다:

(a) **legacy plan_id 불변** — 지터 축 도입 전에 발급된 plan(v1·v1b·v2·v3·v4)의
    plan_id 가 재계산에서도 파일에 적힌 값과 같아야 한다. `jitter=None` 일 때
    해시 payload 에서 키를 빼는 규칙이 깨지면 여기서 잡힌다(구 수집물의 경로가
    통째로 미아가 되는 사고).
(b) **v5 3축 plan** — cells() 1,250개·키 유일·instruction 당 125개, 경로가
    `<plan_id>/<machine>/<instr>/s<i>/k<r>/n<j>` 형태.
(c) **resolve_grid 축 대조** — 3축/legacy plan × 지터 인자 유무 4조합의 통과·거부.
(d) **v6 plan**(scene=주방 · jitter j · noise) — 모의 plan 을 임시 디렉토리에 만들어
    cells()·key·rel_path·env_kwargs·resolve_grid 4케이스를 확인한다. v5 이하 plan_id
    가 (a)(b) 에서 불변인 것과 함께, 신규 필드가 None 이면 해시에서 빠지는 규칙을 건다.

실행: ``python3 scripts/collect/check_plan_schema.py`` (repo 루트 아무 데서나).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.collect.plan import CollectionPlan, resolve_grid  # noqa: E402

CFG = REPO / "configs/collect"
LEGACY_PLANS = ["n15_grid_v1", "n15_grid_v1b", "n15_grid_v2", "n15_grid_v3", "n15_grid_v4"]
V5 = "n15_grid_v5_scenario"

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


# ── (a) legacy plan_id 불변 ────────────────────────────────────────────────
print("(a) legacy plan_id 불변 (jitter 키가 해시에서 빠지는지)")
for name in LEGACY_PLANS:
    path = CFG / name / "collection_plan.json"
    if not path.exists():
        check(f"{name}", False, f"파일 없음: {path.relative_to(REPO)}")
        continue
    stored = json.loads(path.read_text())["plan_id"]
    plan = CollectionPlan.load(path)
    check(f"{name} plan_id", plan.plan_id == stored,
          f"파일 {stored} / 재계산 {plan.plan_id}"
          + (" · jitter=None" if plan.jitter is None else " · jitter 있음(3축)"))

# ── (b) v5 3축 plan ───────────────────────────────────────────────────────
print("\n(b) v5 3축 plan 구조")
v5_path = CFG / V5 / "collection_plan.json"
v5 = CollectionPlan.load(v5_path)
v5_id = json.loads(v5_path.read_text())["plan_id"]
cells = list(v5.cells())
check("v5 plan_id 일치", v5.plan_id == v5_id, f"파일 {v5_id} / 재계산 {v5.plan_id}")
check("jitter 축 존재", v5.jitter is not None)
check("cells() == 1250", len(cells) == 1250, f"실제 {len(cells)}")
check("n_cells == len(cells())", v5.n_cells == len(cells), f"n_cells {v5.n_cells}")
keys = [c.key for c in cells]
check("cell key 유일", len(set(keys)) == len(keys), f"유일 {len(set(keys))} / 전체 {len(keys)}")
per_instr = {i: sum(1 for c in cells if c.instruction == i) for i in v5.instructions}
check("instruction 당 125", all(v == 125 for v in per_instr.values()), json.dumps(per_instr))
check("instruction 10종", len(per_instr) == 10, f"{len(per_instr)}종")
check("key 형식 instr|s<i>|k<r>|n<j>",
      all(re.fullmatch(r".+\|s\d+\|k\d+\|n\d+", k) for k in keys))
pat = re.compile(rf"^{re.escape(v5_id)}/MACH/.+/s\d+/k\d+/n\d+$")
check("rel_path 형식 <plan_id>/<machine>/<instr>/s<i>/k<r>/n<j>",
      all(pat.fullmatch(str(c.rel_path(v5_id, "MACH"))) for c in cells))
meta0 = cells[0].as_metadata()
check("as_metadata 에 jitter_reset_idx",
      "jitter_reset_idx" in meta0 and meta0["scene_idx"] == cells[0].scene_idx,
      json.dumps(meta0, ensure_ascii=False))
check("scene_idx 는 base(0..4)", set(c.scene_idx for c in cells) == set(range(5)),
      str(sorted(set(c.scene_idx for c in cells))))
print("  rel_path 샘플:")
seen: set[str] = set()
for c in cells:
    if c.instruction in seen:
        continue
    seen.add(c.instruction)
    print(f"    {c.key:52s} -> {c.rel_path(v5_id, 'kanu')}")
    if len(seen) >= 4:
        break
odl = next(c for c in cells if c.instruction == "OpenDrawer/left")
print(f"    OpenDrawer/left 첫 셀: {odl.key} -> {odl.rel_path(v5_id, 'srv48')}")
check("missing() 동작 (전량 수집 가정)", v5.missing(set(keys)) == [])
check("missing() 동작 (1칸 결손)", [c.key for c in v5.missing(set(keys[1:]))] == [keys[0]])

# ── (c) resolve_grid 축 대조 ───────────────────────────────────────────────
print("\n(c) resolve_grid 축 대조 4케이스")
legacy_path = CFG / "n15_grid_v2" / "collection_plan.json"
legacy = CollectionPlan.load(legacy_path)
legacy_cell = next(iter(legacy.cells()))


def ns(plan_json: Path, cell, jitter):
    return argparse.Namespace(
        grid_root="/tmp/grid", plan_json=str(plan_json),
        scene_idx=cell.scene_idx, noise_idx=cell.noise_idx,
        jitter_reset_idx=jitter, grid_instruction=cell.instruction,
    )


# 1) 3축 plan + 지터 인자 일치 → 통과
try:
    p, c = resolve_grid(ns(v5_path, cells[0], cells[0].jitter_reset_idx))
    check("1) 3축 plan + --jitter-reset-idx 일치 → 통과",
          c is not None and c.key == cells[0].key, f"cell={c.key}")
except Exception as e:  # noqa: BLE001
    check("1) 3축 plan + --jitter-reset-idx 일치 → 통과", False, f"{type(e).__name__}: {e}")

# 2) 3축 plan + 지터 인자 없음 → ValueError
try:
    resolve_grid(ns(v5_path, cells[0], None))
    check("2) 3축 plan + 지터 인자 없음 → ValueError", False, "예외가 안 났다")
except ValueError as e:
    check("2) 3축 plan + 지터 인자 없음 → ValueError", True, str(e).split(" —")[0])
except Exception as e:  # noqa: BLE001
    check("2) 3축 plan + 지터 인자 없음 → ValueError", False, f"{type(e).__name__}: {e}")

# 3) legacy plan + 지터 인자 주어짐 → ValueError
try:
    resolve_grid(ns(legacy_path, legacy_cell, 3))
    check("3) legacy plan + 지터 인자 → ValueError", False, "예외가 안 났다")
except ValueError as e:
    check("3) legacy plan + 지터 인자 → ValueError", True, str(e).split(" —")[0])
except Exception as e:  # noqa: BLE001
    check("3) legacy plan + 지터 인자 → ValueError", False, f"{type(e).__name__}: {e}")

# 4) legacy plan + 지터 인자 없음 → 통과 (구 수집기 그대로)
try:
    p, c = resolve_grid(ns(legacy_path, legacy_cell, None))
    check("4) legacy plan + 지터 인자 없음 → 통과",
          c is not None and c.key == legacy_cell.key and c.jitter_reset_idx is None,
          f"cell={c.key}")
except Exception as e:  # noqa: BLE001
    check("4) legacy plan + 지터 인자 없음 → 통과", False, f"{type(e).__name__}: {e}")

# 추가) 3축 plan + 계획에 없는 k → ValueError (계획 밖 셀 수집 금지)
try:
    bad_k = max(k for ks in v5.jitter["OpenDrawer/left"] for k in ks) + 100
    resolve_grid(ns(v5_path, cells[0], bad_k))
    check("추가) 3축 plan + 계획에 없는 k → ValueError", False, "예외가 안 났다")
except ValueError as e:
    check("추가) 3축 plan + 계획에 없는 k → ValueError", True, str(e).split(" —")[0])
except Exception as e:  # noqa: BLE001
    check("추가) 3축 plan + 계획에 없는 k → ValueError", False, f"{type(e).__name__}: {e}")

# 좌표 인자가 불완전하면 (None, None) — 구 레이아웃 수집 경로
p, c = resolve_grid(argparse.Namespace(grid_root=None, plan_json=None,
                                       scene_idx=None, noise_idx=None))
check("좌표 인자 불완전 → (None, None)", (p, c) == (None, None))

# ── (d) v6 plan (scene 주방 · jitter j · noise) ────────────────────────────
print("\n(d) v6 plan 모의 검증")
import tempfile  # noqa: E402

V6_JITTERS_PULL = [   # oven/washer 계열: reset 0 고정 + base 오프셋 5종 (핸드오프 §3)
    {"reset_idx": 0, "lat": 0.0, "back": 0.0},
    {"reset_idx": 0, "lat": 0.05, "back": 0.0},
    {"reset_idx": 0, "lat": 0.10, "back": 0.0},
    {"reset_idx": 0, "lat": 0.0, "back": 0.10},
    {"reset_idx": 0, "lat": 0.05, "back": 0.10},
]
V6_JITTERS_PP = [{"reset_idx": j, "lat": 0.0, "back": 0.0} for j in range(5)]

with tempfile.TemporaryDirectory() as _td:
    v6 = CollectionPlan(
        name="v6_mock", model="groot", version="n15", ckpt="ckpt",
        capture_layers=[0, 2], denoise_k=4, token_mode="all_token_full",
        instructions={"OvenRack/left": [100001, 100002, 100003],
                      "PPCC/apple": [100010, 100011, 100012]},
        noise_seeds=[1300000, 1300001, 1300002, 1300003, 1300004],
        scenes={
            "OvenRack/left": [
                {"layout": ly, "style": ly, "side": "left", "lang": "Fully slide the oven rack out.",
                 "fixture_group": "oven", "spawn_lat": 0.45} for ly in (4, 7, 9)],
            "PPCC/apple": [
                {"layout": ly, "style": ly, "side": None, "lang": "Pick the apple from the counter and place it in the cabinet.",
                 "fixture_group": "counter", "spawn_lat": 0.0} for ly in (4, 5, 9)],
        },
        jitters={"OvenRack/left": [list(V6_JITTERS_PULL) for _ in range(3)],
                 "PPCC/apple": [list(V6_JITTERS_PP) for _ in range(3)]},
        extra={"env_kwargs": {"layout_and_style_ids": [[i, i] for i in range(1, 11)]},
               "instruction_text": {"OvenRack/left": "Fully slide the oven rack out."}},
    )
    v6_path = v6.save(_td)
    v6 = CollectionPlan.load(v6_path)      # save/load 왕복
    v6_id = json.loads(v6_path.read_text())["plan_id"]
    v6cells = list(v6.cells())
    check("v6 save/load 왕복 plan_id 일치", v6.plan_id == v6_id, f"{v6_id}")
    check("is_v6", v6.is_v6 and not v6.jitter)
    check("cells() == 2*3*5*5 = 150", len(v6cells) == 150, f"실제 {len(v6cells)}")
    check("n_cells == len(cells())", v6.n_cells == len(v6cells), f"n_cells {v6.n_cells}")
    v6keys = [c.key for c in v6cells]
    check("cell key 유일", len(set(v6keys)) == len(v6keys))
    check("key 형식 instr|s<i>|j<j>|n<n>",
          all(re.fullmatch(r".+\|s\d+\|j\d+\|n\d+", k) for k in v6keys))
    pat6 = re.compile(rf"^{re.escape(v6_id)}/MACH/.+/s\d+/j\d+/n\d+$")
    check("rel_path 형식 <plan_id>/<machine>/<instr>/s<sid>/j<jid>/n<nid>",
          all(pat6.fullmatch(str(c.rel_path(v6_id, "MACH"))) for c in v6cells))
    check("env_kwargs = layout_and_style_ids 10주방",
          v6.env_kwargs == {"layout_and_style_ids": [[i, i] for i in range(1, 11)]},
          json.dumps(v6.env_kwargs))
    ov = [c for c in v6cells if c.instruction == "OvenRack/left"]
    c_j4 = next(c for c in ov if c.scene_idx == 0 and c.jitter_idx == 4 and c.noise_idx == 0)
    check("pull 키 j4 = (lat .05, back .10), reset 0, side/layout 전달",
          (c_j4.base_lat, c_j4.base_back, c_j4.jitter_reset_idx, c_j4.side,
           c_j4.layout_id, c_j4.style_id) == (0.05, 0.10, 0, "left", 4, 4),
          c_j4.key)
    pp = [c for c in v6cells if c.instruction == "PPCC/apple"]
    check("pickplace 키 reset_idx == j", all(c.jitter_reset_idx == c.jitter_idx for c in pp))
    check("env_seed = instructions[key][sid]",
          all(c.env_seed == v6.instructions[c.instruction][c.scene_idx] for c in v6cells))
    m6 = c_j4.as_metadata()
    check("as_metadata v6 열 전량",
          all(k in m6 for k in ("grid_instruction", "scene_idx", "noise_idx", "jitter_idx",
                                "jitter_reset_idx", "base_lat", "base_back", "side",
                                "layout_id", "style_id", "lang")),
          json.dumps(m6, ensure_ascii=False))
    check("missing() 동작 (1칸 결손)",
          [c.key for c in v6.missing(set(v6keys[1:]))] == [v6keys[0]])

    def ns6(cell, j, reset=None):
        return argparse.Namespace(
            grid_root="/tmp/grid", plan_json=str(v6_path),
            scene_idx=cell.scene_idx, noise_idx=cell.noise_idx,
            jitter_idx=j, jitter_reset_idx=reset, grid_instruction=cell.instruction)

    # 1) v6 plan + --jitter-idx → 통과
    try:
        _p, _c = resolve_grid(ns6(c_j4, 4))
        check("v6-1) --jitter-idx 만 → 통과", _c is not None and _c.key == c_j4.key, _c.key)
    except Exception as e:  # noqa: BLE001
        check("v6-1) --jitter-idx 만 → 통과", False, f"{type(e).__name__}: {e}")
    # 2) v6 plan + --jitter-idx 없음 → ValueError
    try:
        resolve_grid(ns6(c_j4, None))
        check("v6-2) --jitter-idx 없음 → ValueError", False, "예외가 안 났다")
    except ValueError as e:
        check("v6-2) --jitter-idx 없음 → ValueError", True, str(e).split(" —")[0])
    # 3) v6 plan + --jitter-reset-idx 불일치 → ValueError
    try:
        resolve_grid(ns6(pp[0], 0, reset=3))   # pp s0 j0 의 reset_idx 는 0
        check("v6-3) --jitter-reset-idx 불일치 → ValueError", False, "예외가 안 났다")
    except ValueError as e:
        check("v6-3) --jitter-reset-idx 불일치 → ValueError", True, str(e).split(" —")[0])
    # 4) v5 plan + --jitter-idx → ValueError (계획에 없는 축)
    try:
        resolve_grid(argparse.Namespace(
            grid_root="/tmp/grid", plan_json=str(v5_path),
            scene_idx=cells[0].scene_idx, noise_idx=cells[0].noise_idx,
            jitter_idx=0, jitter_reset_idx=cells[0].jitter_reset_idx,
            grid_instruction=cells[0].instruction))
        check("v6-4) v5 plan + --jitter-idx → ValueError", False, "예외가 안 났다")
    except ValueError as e:
        check("v6-4) v5 plan + --jitter-idx → ValueError", True, str(e).split(" —")[0])
    print("  rel_path 샘플:")
    for c in (ov[0], c_j4, pp[0]):
        print(f"    {c.key:44s} -> {c.rel_path(v6_id, 'kanu')}")

print("\n" + ("전부 PASS" if not failures else f"FAIL {len(failures)}건: {failures}"))
sys.exit(1 if failures else 0)

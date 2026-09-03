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

print("\n" + ("전부 PASS" if not failures else f"FAIL {len(failures)}건: {failures}"))
sys.exit(1 if failures else 0)

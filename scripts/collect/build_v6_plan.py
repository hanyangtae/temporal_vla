#!/usr/bin/env python3
"""grid v6 collection_plan 빌더 — 선택표(scene_selection.json) → CollectionPlan.

정본: ``docs/collab_within_claude/handoff_20260903_grid_v6_scene_jitter.md``
(층위 §1, scene 선택 §2, jitter 표 §3, plan 필드 §4, 머신 배정 §5) +
저장 규약 ``docs/04_data_storage_convention.md`` §3.1.1.

**이 스크립트가 하는 일은 둘뿐이다.**

1. 선택표(주방·seed·문장·side)를 그대로 plan 의 ``instructions``/``scenes`` 로 옮긴다.
   seed·layout·문장 선택은 스캔이 하는 일이고 여기서 다시 고르지 않는다.
2. 키의 ``kind`` 에 따라 **§3 jitter 표**를 생성한다 — j 정의가 코드 한 곳에만 있어야
   수집기(plan 을 읽음)와 replay/eval 이 같은 세계를 만든다.

선택표 스키마 (에이전트 D 산출):

```json
{
  "env_kwargs": {"layout_and_style_ids": [[1,1], ..., [10,10]]},
  "machine_assignment": {"kanu": ["OvenRack/left", ...], ...},
  "keys": {
    "OvenRack/left": {
      "task_env": "robocasa_panda_omron/SlideOvenRack_PandaOmron_Env",
      "task": "SlideOvenRack",
      "kind": "pull_side",              // pull_side | pull_drawer | pickplace | coffee
      "instruction_text": "...",        // (선택) 없으면 첫 scene 의 lang
      // side = fixture 가 로봇 기준 어느 쪽인가(2026-09-04 개정). spawn_lat 은 로봇
      // 스폰의 lat 부호(>0 = 로봇이 fixture 왼쪽)라 side 와 반대다.
      "scenes": [
        {"env_seed": 100001, "layout": 4, "style": 4, "side": "right",
         "lang": "Fully slide the oven rack out.",
         "fixture_group": "oven", "spawn_lat": 0.45}
      ]
    }
  }
}
```

실행::

    python3 scripts/collect/build_v6_plan.py            # 기본 선택표 → 기본 out-dir
    python3 scripts/collect/build_v6_plan.py --selection /tmp/mock.json --out-dir /tmp/v6 --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.collect.plan import CollectionPlan  # noqa: E402

DEFAULT_DIR = REPO / "configs/collect/n15_grid_v6_scene_jitter"
DEFAULT_SELECTION = DEFAULT_DIR / "scene_selection.json"

# ── 수집 규격 (v5 와 동일 — ckpt·캡처층·denoise_k·token_mode·noise seed) ──────
CKPT = "lerobot_groot_n15__robocasa365_ckpt120000"
CAPTURE_LAYERS = [0, 2, 4, 8, 10, 12, 15]
DENOISE_K = 4
TOKEN_MODE = "all_token_full"
NOISE_SEEDS = [1300000 + i for i in range(10)]   # 2026-09-04 n 5→10 (사용자 결정, plan a81f07b86371→77e745c37b0f)
N_JITTER = 5

# ── §3 jitter 표 (재현 계약의 단일 출처) ────────────────────────────────────
# pickplace·coffee: ep_meta 고정 후 연속 reset j 회 (물체 배치·팔 관절 재추첨), 오프셋 없음.
_J_RESET_ONLY = [{"reset_idx": j, "lat": 0.0, "back": 0.0} for j in range(N_JITTER)]
# drawer: 연속 reset j 회 **+** 뒤로 물러남 back (열린 문이 없어 lat 축이 없다).
_J_DRAWER_BACK = [0.0, 0.05, 0.10, 0.05, 0.10]
_J_DRAWER = [{"reset_idx": j, "lat": 0.0, "back": _J_DRAWER_BACK[j]} for j in range(N_JITTER)]
# oven·washer(side 키): reset 재추첨 없음(reset_idx=0) — 변화는 base 오프셋 뿐.
# side 는 **fixture 가 로봇 기준 어느 쪽인가**(2026-09-04 개정)이고, lat 은 **항상 side 방향
# = fixture 쪽**이다(side=left → 왼쪽(+l), side=right → 오른쪽(−l)); 부호는 수집기가 side 로 정한다.
# 2026-09-03 전수 reset 검사(12 pull scene × 후보 12종): 안쪽 lat 만 주는 오프셋(.03/.05/.10, back 0)은
# 열린 문과 새 접촉을 만들어 대부분 scene 에서 불가. 뒤로 물러난 뒤 안쪽 5cm 는 전 scene 통과.
_J_SIDE_OFFSETS = [(0.0, 0.0), (0.0, 0.05), (0.0, 0.10), (0.05, 0.10), (0.05, 0.15)]
_J_PULL_SIDE = [{"reset_idx": 0, "lat": lat, "back": back} for lat, back in _J_SIDE_OFFSETS]

JITTER_TABLES: dict[str, list[dict[str, Any]]] = {
    "pickplace": _J_RESET_ONLY,
    "coffee": _J_RESET_ONLY,
    "pull_drawer": _J_DRAWER,
    "pull_side": _J_PULL_SIDE,
}

SCENARIO = (
    "handoff_20260903_grid_v6_scene_jitter §1 — 층위 재정의: scene = 주방(layout, style), "
    "jitter j = 같은 scene 의 세계 변형(연속 reset + base 오프셋), noise = 정책 denoise seed. "
    "12키 × scene 3 × j 5 × n 5 = 900판. 좌표 폴더층 s<sid>/j<jid>/n<nid> (docs/04 §3.1.1)."
)


def _scene_entry(raw: dict[str, Any], key: str, sid: int) -> dict[str, Any]:
    """선택표 scene → plan scenes 항목. 누락 필드는 무음 통과 금지."""
    missing = [f for f in ("env_seed", "layout", "style", "lang") if raw.get(f) is None]
    if missing:
        raise ValueError(f"선택표 keys[{key!r}].scenes[{sid}] 에 필드 없음: {missing}")
    side = raw.get("side")
    if side not in (None, "left", "right"):
        raise ValueError(f"keys[{key!r}].scenes[{sid}].side 는 left|right|null: {side!r}")
    return {
        "layout": int(raw["layout"]),
        "style": int(raw["style"]),
        "side": side,
        "lang": str(raw["lang"]),
        "fixture_group": raw.get("fixture_group"),
        "spawn_lat": raw.get("spawn_lat"),
        "env_seed": int(raw["env_seed"]),   # instructions[key][sid] 와 같은 값(가독성용 사본)
    }


def build(selection: dict[str, Any], *, name: str = "n15_grid_v6_scene_jitter") -> CollectionPlan:
    """선택표 → CollectionPlan (저장하지 않는다)."""
    keys = selection.get("keys")
    if not keys:
        raise ValueError("선택표에 keys 가 없다")
    env_kwargs = selection.get("env_kwargs") or {}
    if "layout_and_style_ids" not in env_kwargs:
        raise ValueError(
            "선택표 env_kwargs.layout_and_style_ids 가 없다 — 주방 목록이 바뀌면 seed→주방 "
            "추첨이 바뀌므로 plan 이 목록의 단일 출처여야 한다 (핸드오프 §2)"
        )

    instructions: dict[str, list[int]] = {}
    scenes: dict[str, list[dict[str, Any]]] = {}
    jitters: dict[str, list[list[dict[str, Any]]]] = {}
    env_names: dict[str, str] = {}
    instruction_text: dict[str, str] = {}
    kinds: dict[str, str] = {}

    for key, spec in keys.items():
        kind = spec.get("kind")
        if kind not in JITTER_TABLES:
            raise ValueError(
                f"keys[{key!r}].kind 가 잘못됐다: {kind!r} — {sorted(JITTER_TABLES)} 중 하나"
            )
        raw_scenes = spec.get("scenes") or []
        if not raw_scenes:
            raise ValueError(f"keys[{key!r}].scenes 가 비었다")
        entries = [_scene_entry(sc, key, sid) for sid, sc in enumerate(raw_scenes)]
        if kind == "pull_side" and any(e["side"] is None for e in entries):
            raise ValueError(
                f"keys[{key!r}] 는 pull_side 인데 side 없는 scene 이 있다 — lat 오프셋 방향"
                "(= side = fixture 쪽)을 정할 수 없다"
            )
        instructions[key] = [e["env_seed"] for e in entries]
        scenes[key] = entries
        # jitter 표는 scene 마다 같은 정의를 복사한다(사본이라 scene 별 예외 수정이 가능).
        jitters[key] = [[dict(j) for j in JITTER_TABLES[kind]] for _ in entries]
        # pull_drawer: 연속 reset 이 서랍 좌/우를 다시 뽑으므로(ep_meta 로 고정 안 됨) scene 별로
        # 문장이 맞는 reset 인덱스만 채택한다 — 선택표 scene 의 reset_idx_list (v5 k-스캔과 동일 원리).
        # 모든 kind 에서 선택표 scene 의 reset_idx_list 가 있으면 우선(coffee s0 j2 관측 정지 → 교체 등).
        for sid, e in enumerate(entries):
            lst = raw_scenes[sid].get("reset_idx_list")
            if lst and kind != "pull_drawer":
                for jid, j in enumerate(jitters[key][sid]):
                    if jid < len(lst):
                        j["reset_idx"] = int(lst[jid])
        if kind == "pull_drawer":
            for sid, e in enumerate(entries):
                lst = raw_scenes[sid].get("reset_idx_list")   # _scene_entry 가 정규화하며 버리는 키 → 원본에서
                if not lst or len(lst) < len(jitters[key][sid]):
                    raise SystemExit(f"keys[{key!r}] scene {sid}: reset_idx_list 가 없거나 {len(jitters[key][sid])} 개 미만 — "
                                     "drawer 는 reset 마다 좌/우가 재추첨되므로 scene 별 채택 목록이 필수")
                for jid, j in enumerate(jitters[key][sid]):
                    j["reset_idx"] = int(lst[jid])
        if not spec.get("task_env"):
            raise ValueError(f"keys[{key!r}].task_env 가 없다")
        env_names[key] = spec["task_env"]
        instruction_text[key] = spec.get("instruction_text") or entries[0]["lang"]
        kinds[key] = kind

    extra = {
        "env_names": env_names,
        "instruction_text": instruction_text,
        "env_kwargs": env_kwargs,
        "machine_assignment": selection.get("machine_assignment") or {},
        "jitter_kinds": kinds,
        "jitter_tables": JITTER_TABLES,
        "tasks": {k: keys[k].get("task") for k in keys},
        "scenario": SCENARIO,
        "grid_coord": (
            "docs/04 §3.1.1 (v6) — 좌표는 3축 폴더층 s<sid>/j<jid>/n<nid>. s = scenes[key] "
            "순서(주방), j = jitters[key][sid] 인덱스(값이 아니다), n = noise 순서. "
            "reset 횟수·base 오프셋은 plan 이 단일 출처이며 수집기·replay 가 다시 계산한다."
        ),
        "selection_source": selection.get("source"),
    }
    return CollectionPlan(
        name=name, model="groot", version="n15", ckpt=CKPT,
        capture_layers=CAPTURE_LAYERS, denoise_k=DENOISE_K, token_mode=TOKEN_MODE,
        instructions=instructions, noise_seeds=list(NOISE_SEEDS),
        note=SCENARIO, extra=extra, scenes=scenes, jitters=jitters,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selection", default=str(DEFAULT_SELECTION),
                    help="선택표 JSON (기본: configs/collect/n15_grid_v6_scene_jitter/scene_selection.json)")
    ap.add_argument("--out-dir", default=str(DEFAULT_DIR),
                    help="plan 을 쓸 디렉토리 (collection_plan.json 이 생긴다)")
    ap.add_argument("--name", default="n15_grid_v6_scene_jitter")
    ap.add_argument("--dry-run", action="store_true", help="쓰지 않고 요약만 출력")
    args = ap.parse_args()

    sel_path = Path(args.selection)
    if not sel_path.exists():
        print(f"[build_v6_plan] 선택표가 없다: {sel_path}", file=sys.stderr)
        return 2
    plan = build(json.loads(sel_path.read_text()), name=args.name)

    print(f"[build_v6_plan] 선택표 {sel_path}")
    print(f"  키 {len(plan.instructions)}종 · scene/키 "
          f"{sorted({len(v) for v in plan.instructions.values()})} · j {N_JITTER} · "
          f"noise {len(plan.noise_seeds)} → n_cells {plan.n_cells}")
    print(f"  plan_id = {plan.plan_id} · 추정 용량 {plan.estimate_bytes() / 2**30:.0f} GiB")
    for key in plan.instructions:
        scs = plan.scenes[key]                       # type: ignore[index]
        print(f"    {key:22s} seeds={plan.instructions[key]} "
              f"layouts={[s['layout'] for s in scs]} side={scs[0]['side']!r} "
              f"kind={plan.extra['jitter_kinds'][key]}")
    sample = next(iter(plan.cells()))
    print(f"  첫 셀: {sample.key} -> {sample.rel_path(plan.plan_id, 'MACHINE')}")

    if args.dry_run:
        print("  (--dry-run: 저장 안 함)")
        return 0
    out = plan.save(args.out_dir)
    print(f"  저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""v5 아카이브 재배치 — 평탄 si → 3축 `s<i>/k<r>/n<j>` (docs/04 §3.1.1, 2026-09-03 개정).

v5(계약 plan `8daefeabf020`)는 지터 축을 scene 좌표에 **평탄 인코딩**(si = scene*100 + k)해
수집됐다. 규약이 셋째 폴더층 `k<r>` 로 개정됐으므로 이미 수집된 1,250 셀을 새 plan_id 아래로
옮긴다. **rollout 은 그대로 두고 좌표만 바꾼다** — 재수집이 아니다.

    <grid-root>/<old_plan_id>/<machine>/<instr>/s<si>/n<j>/<arm>/
        → <grid-root>/<new_plan_id>/<machine>/<instr>/s<si//100>/k<si%100>/n<j>/<arm>/

같은 파일시스템 안에서 디렉토리 단위 `os.rename` 이므로 수백 GB 라도 즉시 끝난다(복사 없음).
각 `meta.json` 은 새 좌표로 패치하고 `layout_migrated_from` 에 옛 경로(grid-root 상대,
절대경로 금지 §8)를 남긴다. `ep_meta/` 는 새 루트로 함께 옮기고, 비워진 옛 루트에는
재배치 안내 `README.txt` 만 남긴다.

**pkl 내부 `extra_metadata.scene_idx` 는 수집 당시 평탄값 그대로다** — 열어서 고치지 않는다
(pkl 안 좌표는 사후 기록용, §8). 좌표의 정본은 경로와 meta.json 이다.

stdlib 만 쓴다 — 아카이브가 있는 노드의 base python3(3.10, torch·numpy 없음)에서 돈다.

사용:
    python3 scripts/collect/migrate_grid_k_layer.py \
        --grid-root <아카이브 grid 루트> \
        --new-plan-json <새 v5 collection_plan.json> --dry-run
    (표를 확인한 뒤 --dry-run 을 --apply 로 바꿔 재실행)

종료 코드: 0 정상, 1 검증 실패(결손·초과·목적지 충돌 등), 2 인자·경로 오류.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

OLD_PLAN_ID_DEFAULT = "8daefeabf020"
EP_META_DIR = "ep_meta"
README_NAME = "README.txt"

_S_RE = re.compile(r"^s(\d+)$")
_N_RE = re.compile(r"^n(\d+)$")


# ─────────────────────────── 새 plan (최소 파싱) ───────────────────────────
# src.collect.plan 을 import 하지 않는다 — 이 스크립트는 repo 없이 한 장만 올려도 돌아야 한다.

def load_new_plan(path: Path) -> tuple[str, set[tuple[str, int, int, int]]]:
    """새 plan JSON → (plan_id, 기대 셀 키 집합 (instr, s, k, n))."""
    body: dict[str, Any] = json.loads(path.read_text())
    plan_id = body.get("plan_id") or ""
    if not plan_id:
        raise ValueError(f"{path}: plan_id 없음 — CollectionPlan.save 가 쓴 파일이 아니다")
    jitter = body.get("jitter")
    if not jitter:
        raise ValueError(
            f"{path}: jitter 필드 없음 — 2축 legacy plan 으로는 재배치할 대상이 없다 (§3.1.1)")
    instructions = body.get("instructions") or {}
    n_noise = len(body.get("noise_seeds") or [])
    want: set[tuple[str, int, int, int]] = set()
    for instr, scenes in instructions.items():
        ks_per_scene = jitter.get(instr) or []
        for si in range(len(scenes)):
            if si >= len(ks_per_scene):
                continue
            for k in ks_per_scene[si]:
                for ni in range(n_noise):
                    want.add((instr, si, int(k), ni))
    return plan_id, want


def cell_key(instr: str, si: int, k: int, ni: int) -> str:
    return f"{instr}|s{si}|k{k}|n{ni}"


# ─────────────────────────── 옛 좌표 스캔 ───────────────────────────

def scan_old(old_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """옛 루트의 `<machine>/<instr...>/s<si>/n<j>/<arm>/meta.json` 전량 → 이동 계획.

    instruction 은 ``PPCC/bread`` 처럼 '/' 를 포함하므로 machine 뒤부터 s<si> 앞까지 전부다.
    """
    plans: list[dict[str, Any]] = []
    problems: list[str] = []
    for mp in sorted(old_root.rglob("meta.json")):
        rel = mp.relative_to(old_root)
        parts = rel.parts[:-1]                    # 파일명 제거
        if parts and parts[0] == EP_META_DIR:     # ep_meta 는 좌표가 아니다
            continue
        if len(parts) < 5:                        # machine instr s n arm
            problems.append(f"{rel}: 좌표 경로가 아님 (<machine>/<instr>/s<si>/n<j>/<arm>)")
            continue
        machine, arm = parts[0], parts[-1]
        m_s, m_n = _S_RE.match(parts[-3]), _N_RE.match(parts[-2])
        if not (m_s and m_n):
            problems.append(f"{rel}: s<si>/n<j> 층을 못 읽음 — 이미 재배치된 경로인가?")
            continue
        instruction = "/".join(parts[1:-3])
        if not instruction:
            problems.append(f"{rel}: instruction 층 없음")
            continue
        si, ni = int(m_s.group(1)), int(m_n.group(1))
        plans.append({
            "machine": machine, "instruction": instruction, "arm": arm,
            "si_flat": si, "scene_idx": si // 100, "jitter_reset_idx": si % 100,
            "noise_idx": ni,
            "src_rel": Path(*parts),              # old_root 상대 (arm 디렉토리까지)
        })
    return plans, problems


def patch_meta(meta_path: Path, new_plan_id: str, scene_idx: int, jitter: int,
               migrated_from: str) -> None:
    """meta.json 의 좌표 3열만 갱신하고 나머지 키는 보존한다 (원자적 쓰기)."""
    body = json.loads(meta_path.read_text())
    body["plan_id"] = new_plan_id
    body["scene_idx"] = scene_idx
    body["jitter_reset_idx"] = jitter
    body["layout_migrated_from"] = migrated_from
    tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    tmp.write_text(json.dumps(body, indent=2, ensure_ascii=False))
    tmp.replace(meta_path)


def prune_empty(root: Path) -> list[str]:
    """빈 디렉토리를 아래에서부터 지운다. 반환 = 남은 파일들(grid-root 상대 아님, root 상대)."""
    for d in sorted((p for p in root.rglob("*") if p.is_dir()),
                    key=lambda p: len(p.parts), reverse=True):
        try:
            d.rmdir()
        except OSError:
            pass          # 비어있지 않다 — 남은 파일은 아래에서 보고한다
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


# ─────────────────────────── 본체 ───────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid-root", required=True, type=Path,
                    help="좌표 저장소 루트 (아래에 <plan_id>/<machine>/... 이 있다)")
    ap.add_argument("--old-plan-id", default=OLD_PLAN_ID_DEFAULT,
                    help=f"평탄 si 로 수집된 옛 plan_id (기본 {OLD_PLAN_ID_DEFAULT})")
    ap.add_argument("--new-plan-json", required=True, type=Path,
                    help="새 3축 collection_plan.json (plan_id 는 이 JSON 의 plan_id 키)")
    ap.add_argument("--dry-run", action="store_true", help="이동 없이 매핑 표만 출력 (기본)")
    ap.add_argument("--apply", action="store_true", help="실제로 rename·meta 패치를 수행")
    ap.add_argument("--list-limit", type=int, default=10)
    args = ap.parse_args()

    if args.apply and args.dry_run:
        print("ERROR: --dry-run 과 --apply 를 같이 줄 수 없다", file=sys.stderr)
        return 2
    if not args.grid_root.is_dir():
        print(f"ERROR: --grid-root 없음: {args.grid_root}", file=sys.stderr)
        return 2

    old_root = args.grid_root / args.old_plan_id
    if not old_root.is_dir():
        print(f"ERROR: 옛 plan 루트 없음: {old_root}", file=sys.stderr)
        return 2
    try:
        new_plan_id, want = load_new_plan(args.new_plan_json)
    except Exception as exc:                                       # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if new_plan_id == args.old_plan_id:
        print("ERROR: 새 plan_id 가 옛 plan_id 와 같다 — 재배치할 것이 없다", file=sys.stderr)
        return 2
    new_root = args.grid_root / new_plan_id

    now = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"# migrate_grid_k_layer [{mode}]  실행시각 {now}")
    print(f"# grid-root {args.grid_root}")
    print(f"# {args.old_plan_id} (평탄 si) → {new_plan_id} (3축 s/k/n)\n")

    moves, problems = scan_old(old_root)
    if problems:
        print(f"## 좌표로 못 읽은 경로 {len(problems)} 건 — 중단")
        for p in problems[:50]:
            print(f"    ✗ {p}")
        return 1

    # 목적지 충돌·중복 원본 검사. 하나라도 있으면 아무것도 옮기지 않는다.
    conflicts: list[str] = []
    dst_seen: dict[Path, Path] = {}
    for m in moves:
        dst_rel = (Path(m["machine"]) / m["instruction"] / f"s{m['scene_idx']}"
                   / f"k{m['jitter_reset_idx']}" / f"n{m['noise_idx']}" / m["arm"])
        m["dst_rel"] = dst_rel
        dst = new_root / dst_rel
        if dst.exists():
            conflicts.append(f"목적지 이미 존재: {new_plan_id}/{dst_rel.as_posix()}")
        if dst in dst_seen:
            conflicts.append(f"두 원본이 같은 목적지로: {dst_seen[dst].as_posix()} / "
                             f"{m['src_rel'].as_posix()} → {dst_rel.as_posix()}")
        dst_seen[dst] = m["src_rel"]

    if not moves and (old_root / README_NAME).is_file():
        # 이미 옮긴 뒤 다시 실행한 경우. 결손 1,250 을 쏟아내지 말고 그대로 끝낸다.
        print(f"## 옮길 셀 없음 + {README_NAME} 존재 → 이미 재배치 완료. 할 일 없음.")
        return 0

    print(f"## 이동 대상 {len(moves)} 셀 (meta.json 기준)")
    print(f"{'src (옛 plan 상대)':<58s}  →  dst (새 plan 상대)")
    for m in moves[:args.list_limit]:
        print(f"  {m['src_rel'].as_posix():<56s}  →  {m['dst_rel'].as_posix()}")
    if len(moves) > args.list_limit:
        print(f"  ... 외 {len(moves) - args.list_limit} 셀")
    per_machine = Counter(m["machine"] for m in moves)
    per_instr = Counter(m["instruction"] for m in moves)
    print(f"\n  machine 분포: {dict(sorted(per_machine.items()))}")
    for instr in sorted(per_instr):
        print(f"    {instr:<28s} {per_instr[instr]:>5d}")

    # 새 plan 기대 셀 대조 — 이동으로 생길 좌표가 계획과 맞는지 미리 본다.
    have = {(m["instruction"], m["scene_idx"], m["jitter_reset_idx"], m["noise_idx"])
            for m in moves}
    missing, stray = sorted(want - have), sorted(have - want)
    print(f"\n## 새 plan 대비  기대 {len(want)} · 이동 {len(have)} · "
          f"결손 {len(missing)} · 계획 밖 {len(stray)}")
    for k in missing[:args.list_limit]:
        print(f"    - 결손 {cell_key(*k)}")
    if len(missing) > args.list_limit:
        print(f"    ... 외 {len(missing) - args.list_limit}")
    for k in stray[:args.list_limit]:
        print(f"    ! 계획 밖 {cell_key(*k)}")
    if len(stray) > args.list_limit:
        print(f"    ... 외 {len(stray) - args.list_limit}")

    if conflicts:
        print(f"\n## 목적지 충돌 {len(conflicts)} 건 — 아무것도 옮기지 않고 중단")
        for c in conflicts[:50]:
            print(f"    ✗ {c}")
        return 1

    ep_meta_src = old_root / EP_META_DIR
    print(f"\n## ep_meta: {'있음 → 새 루트로 이동' if ep_meta_src.is_dir() else '없음'}")

    if not args.apply:
        print("\n[DRY-RUN] 이동하지 않았다. 표를 확인한 뒤 --apply 로 재실행할 것.")
        return 1 if (missing or stray) else 0

    # ── 실행 ──
    n_moved = n_patched = 0
    for m in moves:
        src = old_root / m["src_rel"]
        dst = new_root / m["dst_rel"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.rename(src, dst)
        except OSError as exc:
            print(f"\nERROR: rename 실패 ({src} → {dst}): {exc}\n"
                  f"       같은 파일시스템이 아니면 이 스크립트를 쓸 수 없다 "
                  f"(디렉토리 rename 전제).", file=sys.stderr)
            return 1
        n_moved += 1
        migrated_from = (Path(args.old_plan_id) / m["src_rel"]).as_posix()
        patch_meta(dst / "meta.json", new_plan_id, m["scene_idx"], m["jitter_reset_idx"],
                   migrated_from)
        n_patched += 1

    if ep_meta_src.is_dir():
        ep_dst = new_root / EP_META_DIR
        if ep_dst.exists():
            print(f"ERROR: ep_meta 목적지 이미 존재: {ep_dst}", file=sys.stderr)
            return 1
        ep_dst.parent.mkdir(parents=True, exist_ok=True)
        os.rename(ep_meta_src, ep_dst)
        print(f"ep_meta → {new_plan_id}/{EP_META_DIR}")

    leftovers = prune_empty(old_root)
    print(f"\n## 이동 {n_moved} 셀 · meta 패치 {n_patched} · 옛 루트 잔여 파일 {len(leftovers)}")
    for f in leftovers[:args.list_limit]:
        print(f"    · {f}")
    if not leftovers:
        (old_root).mkdir(parents=True, exist_ok=True)
        (old_root / README_NAME).write_text(
            f"이 plan_id({args.old_plan_id})의 데이터는 {new_plan_id} 로 재배치되었다.\n"
            f"재배치 일시: {now}\n"
            f"재배치 스크립트: scripts/collect/migrate_grid_k_layer.py\n"
            f"평탄 si → 3축: s<si//100>/k<si%100>/n<j> (docs/04 §3.1.1).\n"
            f"각 meta.json 의 layout_migrated_from 에 옛 경로가 남아 있다.\n",
            encoding="utf-8")
        print(f"    빈 옛 루트에 {README_NAME} 만 남겼다.")
    else:
        print(f"    잔여 파일이 있어 {README_NAME} 를 쓰지 않았다 — 확인 후 수동 정리.")

    ok = not (missing or stray)
    print(f"\n판정: {'정상' if ok else '결손/초과 있음'} "
          f"(결손 {len(missing)} · 계획 밖 {len(stray)})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

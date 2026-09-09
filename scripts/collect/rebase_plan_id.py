#!/usr/bin/env python3
"""아카이브 plan_id 디렉토리 재배치 — plan 내용(예: 특정 scene 의 jitter 정의, instruction 키 이름)만 바뀌어
plan_id 가 갱신됐을 때, 기존 셀을 새 plan_id 아래로 옮기고 meta.json 을 패치한다(pkl 불변).
무효 셀은 삭제해 재수집 결손으로 남긴다.

사용(승준 base python3):
  rebase_plan_id.py --grid-root <grid> --old <old_plan_id> --new <new_plan_id>
                    [--drop <rel_cell_dir> ...]
                    [--rename-key OLD=NEW ...] [--flip-side] [--apply]
  --drop 은 <machine>/<key>/s<sid>/j<jid>/n<nid>/<arm> 형식(옛 plan 루트 상대). --apply 없으면 dry-run.

키 rename (2026-09-04 grid v6 out-left/out-right 의미 개정)
----------------------------------------------------------
``--rename-key OLD=NEW`` 는 instruction 키 디렉토리를 옮긴다. 키는 ``OvenRack/out-left`` 처럼
``/`` 를 포함할 수 있어 경로에서 **여러 층**이 되며, 실제 셀 경로는

    <plan_root>/<machine>/<KEY>/s*/j*/n*/<arm>/meta.json

이다. ``A=B`` 와 ``B=A`` 를 함께 주는 **swap 도 지원**한다 — 모든 rename 을 임시 이름
(``__rebase_tmp_<i>__``)으로 먼저 옮긴 뒤 최종 이름으로 두 번째로 옮기기 때문이다.

meta.json 패치(rename 된 셀만): ``grid_instruction`` 을 새 키로, ``--flip-side`` 면 ``side`` 를
left↔right 로 뒤집고, ``key_renamed_from`` 에 옛 키를 남긴다. plan_id 패치는 전 셀 공통이다.
"""
from __future__ import annotations
import argparse, json, os, shutil, sys, time
from pathlib import Path

TMP_PREFIX = "__rebase_tmp_"


def _machine_dirs(root: Path) -> list[Path]:
    """plan 루트 바로 아래의 <machine> 디렉토리들 (ep_meta 등 비-머신 층은 키가 없어 자연히 무시)."""
    return sorted(d for d in root.iterdir() if d.is_dir())


def _key_dir(machine: Path, key: str) -> Path:
    """키(``OvenRack/out-left``) → 머신 아래 디렉토리 경로. 키의 ``/`` 는 경로 층이다."""
    p = machine
    for seg in key.split("/"):
        p = p / seg
    return p


def _prune_empty(start: Path, stop: Path) -> None:
    """start 부터 위로 올라가며 빈 디렉토리를 지운다(stop 은 건드리지 않는다)."""
    p = start
    while p != stop and p.is_dir() and not any(p.iterdir()):
        p.rmdir()
        p = p.parent


def parse_renames(items: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for it in items:
        if "=" not in it:
            sys.exit(f"--rename-key 형식은 OLD=NEW: {it!r}")
        old, new = it.split("=", 1)
        old, new = old.strip("/"), new.strip("/")
        if not old or not new:
            sys.exit(f"--rename-key 에 빈 키: {it!r}")
        if old == new:
            sys.exit(f"--rename-key OLD 와 NEW 가 같다: {it!r}")
        out.append((old, new))
    olds = [o for o, _ in out]
    if len(set(olds)) != len(olds):
        sys.exit(f"--rename-key 의 OLD 가 중복: {olds}")
    news = [n for _, n in out]
    if len(set(news)) != len(news):
        sys.exit(f"--rename-key 의 NEW 가 중복: {news}")
    return out


def plan_renames(root: Path, renames: list[tuple[str, str]]) -> list[tuple[Path, Path, str, str]]:
    """(머신, 키) 별 실제 이동 목록 → [(src, dst, old_key, new_key)]. 없는 키는 무시(다른 머신 홈)."""
    moves: list[tuple[Path, Path, str, str]] = []
    for machine in _machine_dirs(root):
        for old_key, new_key in renames:
            src = _key_dir(machine, old_key)
            if not src.is_dir():
                continue
            moves.append((src, _key_dir(machine, new_key), old_key, new_key))
    return moves


def apply_renames(root: Path, moves: list[tuple[Path, Path, str, str]]) -> set[Path]:
    """swap 안전 2단 이동. 반환 = rename 된 최종 디렉토리 집합(meta 패치 대상 판별용)."""
    staged: list[tuple[Path, Path, Path]] = []   # (tmp, dst, src_parent)
    for i, (src, dst, _o, _n) in enumerate(moves):
        tmp = src.parent / f"{TMP_PREFIX}{i}__"
        if tmp.exists():
            sys.exit(f"임시 이름이 이미 있다: {tmp}")
        os.rename(src, tmp)
        staged.append((tmp, dst, src.parent))
    finals: set[Path] = set()
    for tmp, dst, src_parent in staged:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            sys.exit(f"rename 대상이 이미 있다(swap 이 아닌 충돌): {dst}")
        os.rename(tmp, dst)
        finals.add(dst)
    for _tmp, _dst, src_parent in staged:
        if src_parent.is_dir() and not any(src_parent.iterdir()):
            src_parent.rmdir()
    return finals


FLIP = {"left": "right", "right": "left"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid-root", required=True, type=Path); ap.add_argument("--old", required=True); ap.add_argument("--new", required=True)
    ap.add_argument("--drop", nargs="*", default=[]); ap.add_argument("--apply", action="store_true")
    ap.add_argument("--rename-key", action="append", default=[], metavar="OLD=NEW",
                    help="instruction 키 디렉토리 rename (반복 가능; A=B 와 B=A 를 함께 주면 swap)")
    ap.add_argument("--flip-side", action="store_true",
                    help="rename 된 셀의 meta.json side 를 left↔right 로 뒤집는다")
    a = ap.parse_args()
    renames = parse_renames(a.rename_key)
    if a.flip_side and not renames:
        sys.exit("--flip-side 는 --rename-key 와 함께 쓴다(대상이 rename 된 셀이다)")
    old, new = a.grid_root / a.old, a.grid_root / a.new
    if not old.is_dir(): sys.exit(f"옛 plan 루트 없음: {old}")
    if new.exists(): sys.exit(f"새 plan 루트가 이미 있다: {new}")
    metas = sorted(old.rglob("meta.json"))
    drops = [old / d for d in a.drop]
    for d in drops:
        if not (d / "meta.json").exists(): sys.exit(f"--drop 대상에 meta.json 없음: {d}")
    moves = plan_renames(old, renames)
    n_rename_cells = sum(len(list(src.rglob("meta.json"))) for src, _d, _o, _n in moves)
    print(f"셀 {len(metas)}개, 삭제 대상 {len(drops)}개, {a.old} → {a.new}")
    if renames:
        print(f"키 rename {len(renames)}쌍 · 이동 디렉토리 {len(moves)}개 · 대상 셀 {n_rename_cells}개"
              + (" · side 반전" if a.flip_side else ""))
    if not a.apply:
        for d in drops: print("  [dry] drop", d.relative_to(old))
        for src, dst, o, n in moves:
            print(f"  [dry] rename {src.relative_to(old)} → {dst.relative_to(old)} "
                  f"(키 {o} → {n}, 셀 {len(list(src.rglob('meta.json')))})")
        if renames:
            print(f"  [dry] meta 패치: grid_instruction ← 새 키"
                  + (", side 반전" if a.flip_side else "")
                  + ", key_renamed_from 기록 (rename 셀 한정)")
        print(f"  [dry] meta 패치: plan_id {a.old} → {a.new} (전 셀)")
        print("[DRY-RUN] 변경 없음 — --apply 로 실행"); return
    for d in drops:
        shutil.rmtree(d); print("  drop", d.relative_to(old))
        _prune_empty(d.parent, old)
    renamed_roots: set[Path] = set()
    if moves:
        renamed_roots = apply_renames(old, moves)
        for src, dst, o, n in moves:
            print(f"  rename {src.relative_to(old)} → {dst.relative_to(old)} (키 {o} → {n})")
    # 옛 루트 상대 경로 → 새 키 / 옛 키 (meta 패치용). rename 후 위치 기준.
    key_of: dict[Path, tuple[str, str]] = {}
    for src, dst, o, n in moves:
        key_of[dst.relative_to(old)] = (o, n)
    os.rename(old, new)
    today = time.strftime("%Y-%m-%d")
    n_plan = n_key = 0
    for m in sorted(new.rglob("meta.json")):
        j = json.loads(m.read_text())
        touched = False
        if j.get("plan_id") == a.old:
            j["plan_id"] = a.new; j["plan_id_migrated_from"] = a.old; j["plan_id_migrated_at"] = today
            n_plan += 1; touched = True
        rel = m.relative_to(new)
        hit = next(((o, n) for pref, (o, n) in key_of.items()
                    if rel.parts[:len(pref.parts)] == pref.parts), None)
        if hit is not None:
            old_key, new_key = hit
            j["grid_instruction"] = new_key
            j["key_renamed_from"] = old_key
            j["key_renamed_at"] = today
            if a.flip_side and j.get("side") in FLIP:
                j["side"] = FLIP[j["side"]]
            n_key += 1; touched = True
        if touched:
            m.write_text(json.dumps(j, indent=2, ensure_ascii=False))
    (a.grid_root / a.old).mkdir(); (a.grid_root / a.old / "README.txt").write_text(
        f"{today} plan_id 재배치 → {a.new} (scripts/collect/rebase_plan_id.py). 삭제 셀 {len(drops)}: "
        + ", ".join(a.drop) + "\n"
        + ("키 rename: " + ", ".join(f"{o}→{n}" for o, n in renames)
           + (" (side 반전)" if a.flip_side else "") + "\n" if renames else ""))
    print(f"완료: plan_id 패치 {n_plan}, 키 패치 {n_key}, 남은 셀 {len(list(new.rglob('meta.json')))}")


if __name__ == "__main__":
    main()

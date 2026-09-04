#!/usr/bin/env python3
"""아카이브 plan_id 디렉토리 재배치 — plan 내용(예: 특정 scene 의 jitter 정의)만 바뀌어 plan_id 가 갱신됐을 때,
기존 셀을 새 plan_id 아래로 옮기고 meta.json 의 plan_id 를 패치한다(pkl 불변). 무효 셀은 삭제해 재수집 결손으로 남긴다.

사용(승준 base python3):
  rebase_plan_id.py --grid-root <grid> --old <old_plan_id> --new <new_plan_id> [--drop <rel_cell_dir> ...] [--apply]
  --drop 은 <machine>/<key>/s<sid>/j<jid>/n<nid>/<arm> 형식(옛 plan 루트 상대). --apply 없으면 dry-run.
"""
from __future__ import annotations
import argparse, json, os, shutil, sys, time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-root", required=True, type=Path); ap.add_argument("--old", required=True); ap.add_argument("--new", required=True)
    ap.add_argument("--drop", nargs="*", default=[]); ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    old, new = a.grid_root / a.old, a.grid_root / a.new
    if not old.is_dir(): sys.exit(f"옛 plan 루트 없음: {old}")
    if new.exists(): sys.exit(f"새 plan 루트가 이미 있다: {new}")
    metas = sorted(old.rglob("meta.json"))
    drops = [old / d for d in a.drop]
    for d in drops:
        if not (d / "meta.json").exists(): sys.exit(f"--drop 대상에 meta.json 없음: {d}")
    print(f"셀 {len(metas)}개, 삭제 대상 {len(drops)}개, {a.old} → {a.new}")
    if not a.apply:
        for d in drops: print("  [dry] drop", d.relative_to(old))
        print("[DRY-RUN] 변경 없음 — --apply 로 실행"); return
    for d in drops:
        shutil.rmtree(d); print("  drop", d.relative_to(old))
        p = d.parent
        while p != old and not any(p.iterdir()): p.rmdir(); p = p.parent
    os.rename(old, new)
    n = 0
    for m in sorted(new.rglob("meta.json")):
        j = json.loads(m.read_text())
        if j.get("plan_id") == a.old:
            j["plan_id"] = a.new; j["plan_id_migrated_from"] = a.old; j["plan_id_migrated_at"] = time.strftime("%Y-%m-%d")
            m.write_text(json.dumps(j, indent=2, ensure_ascii=False)); n += 1
    (a.grid_root / a.old).mkdir(); (a.grid_root / a.old / "README.txt").write_text(
        f"{time.strftime('%Y-%m-%d')} plan_id 재배치 → {a.new} (scripts/collect/rebase_plan_id.py). 삭제 셀 {len(drops)}: " + ", ".join(a.drop) + "\n")
    print(f"완료: meta 패치 {n}, 남은 셀 {len(list(new.rglob('meta.json')))}")


if __name__ == "__main__":
    main()

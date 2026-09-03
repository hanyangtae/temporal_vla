#!/usr/bin/env python3
"""아카이브의 plan_id 디렉토리를 새 plan_id 로 옮기고 meta.json 의 plan_id 를 패치한다 (stdlib).

plan 을 고치면(셀 교체 등) plan_id(내용 지문)가 바뀐다 — 좌표가 같은 기존 셀은 그대로 유효하므로
디렉토리를 rename 하고 meta.json 의 plan_id 만 갱신한다. 이력은 meta["plan_id_history"] 에 쌓는다.
사용: rebase_plan_id.py --grid-root <grid> --old <plan_id> --new <plan_id> [--apply]
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--grid-root", required=True, type=Path); ap.add_argument("--old", required=True); ap.add_argument("--new", required=True)
ap.add_argument("--apply", action="store_true")
a = ap.parse_args()
old, new = a.grid_root / a.old, a.grid_root / a.new
if not old.is_dir(): sys.exit(f"없음: {old}")
if new.exists(): sys.exit(f"이미 있음: {new}")
metas = sorted(old.rglob("meta.json"))
print(f"{a.old} → {a.new}: meta.json {len(metas)}개, ep_meta {'있음' if (old/'ep_meta').is_dir() else '없음'}")
if not a.apply:
    print("[DRY-RUN] --apply 로 실행할 것"); sys.exit(0)
os.rename(old, new)
n = 0
for m in sorted(new.rglob("meta.json")):
    d = json.loads(m.read_text())
    if d.get("plan_id") != a.new:
        hist = list(d.get("plan_id_history") or []); hist.append(d.get("plan_id")); d["plan_id_history"] = hist
        d["plan_id"] = a.new
        tmp = m.with_suffix(".json.tmp"); tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False)); os.replace(tmp, m); n += 1
(old).mkdir(exist_ok=True)
(old / "README.txt").write_text(f"{time.strftime('%F %T')} plan_id 재지정: {a.old} → {a.new} (rebase_plan_id.py). 데이터는 {a.new}/ 에 있다.\n")
print(f"완료: rename + meta 패치 {n}")

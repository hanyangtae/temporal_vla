#!/usr/bin/env python3
"""seed → fixture 그룹(배치/방향) 스캔 — 정책 무관, reset(seed) 만으로 ep_meta 를 읽는다.

배경(2026-09-03): pull 계열(SlideOvenRack·SlideDishwasherRack)은 instruction 에 좌/우가 없지만
scene(seed)마다 fixture 그룹(layout·좌/우/island)이 달라 실질 "방향" 축이 된다. v5 oven 5 scene 은
전부 oven_left_group(layout 4) 이었다. 이 스크립트는 넓은 seed 대역에서 그룹 분포를 얻는다.

출력 TSV: seed layout_id style_id fixture_key fixture_group should_pull rack_level lang base_x base_y base_yaw
사용 (robocasa 컨테이너, GPU 불필요):
  MUJOCO_GL=egl PYTHONPATH=/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla \
  python scripts/collect/scan_fixture_groups.py --env robocasa_panda_omron/SlideOvenRack_PandaOmron_Env --seeds 100000-100999 --out <tsv> [--procs 4]
"""
from __future__ import annotations
import argparse, csv, json, sys
from multiprocessing import Pool
from pathlib import Path


def parse_seeds(spec: str) -> list[int]:
    out = []
    for tok in spec.split(","):
        if "-" in tok:
            a, b = tok.split("-"); out.extend(range(int(a), int(b) + 1))
        elif tok.strip():
            out.append(int(tok))
    return out


def probe(args):
    env_name, seed, ls = args
    import gymnasium as gym
    import robocasa  # noqa: F401
    import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401
    import robosuite  # noqa: F401
    _repo = Path(__file__).resolve().parents[2]
    if str(_repo) not in sys.path:
        sys.path.insert(0, str(_repo))
    from src.collect.robocasa.event_labeler import find_robocasa_env
    try:
        kw = {"layout_and_style_ids": ls} if ls else {}
        env = gym.make(env_name, enable_render=False, seed=seed, **kw)
        env.reset(seed=seed)
        em = find_robocasa_env(env).get_ep_meta() or {}
        fr = em.get("fixture_refs") or {}
        key, grp = (next(iter(fr.items())) if fr else ("", ""))
        pos = list(em.get("init_robot_base_pos") or [None, None, None]); ori = list(em.get("init_robot_base_ori") or [None, None, None])
        row = dict(seed=seed, layout_id=em.get("layout_id"), style_id=em.get("style_id"), fixture_key=key,
                   fixture_group=grp, should_pull=em.get("should_pull"), rack_level=em.get("rack_level"),
                   lang=em.get("lang"), base_x=round(pos[0], 3) if pos[0] is not None else None,
                   base_y=round(pos[1], 3) if pos[1] is not None else None,
                   base_yaw=round(ori[2], 3) if ori[2] is not None else None, err="")
        env.close()
        return row
    except Exception as e:  # noqa: BLE001
        return dict(seed=seed, err=str(e)[:120])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True); ap.add_argument("--seeds", required=True)
    ap.add_argument("--out", required=True, type=Path); ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--layout-style", default="", help='주방 목록 "1:1,2:2,..." (기본: 래퍼 기본 5주방). target split 전체 = 1:1,...,10:10')
    a = ap.parse_args()
    seeds = parse_seeds(a.seeds)
    ls = [[int(x), int(y)] for x, y in (t.split(":") for t in a.layout_style.split(",") if t.strip())] or None
    print("layout_and_style_ids:", ls or "wrapper default", flush=True)
    cols = ["seed", "layout_id", "style_id", "fixture_key", "fixture_group", "should_pull", "rack_level", "lang", "base_x", "base_y", "base_yaw", "err"]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with Pool(a.procs, maxtasksperchild=50) as pool, a.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore"); w.writeheader()
        for i, r in enumerate(pool.imap_unordered(probe, [(a.env, s, ls) for s in seeds], chunksize=2), 1):
            w.writerow(r); f.flush()
            if i % 50 == 0: print(f"  {i}/{len(seeds)}", flush=True)
    rows = list(csv.DictReader(a.out.open(), delimiter="\t"))
    from collections import Counter
    print("fixture_group 분포:", dict(Counter((r["fixture_group"], r["should_pull"]) for r in rows)))
    print("layout 분포:", dict(Counter(r["layout_id"] for r in rows)), "err:", sum(1 for r in rows if r["err"]))


if __name__ == "__main__":
    main()

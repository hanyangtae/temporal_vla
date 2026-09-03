#!/usr/bin/env python3
"""v6 plan 의 지터 셀(base 오프셋·reset 재추첨)을 정책 없이 reset 만으로 전수 검사한다.

collector 의 `_v6_apply_jitter`(문장 대조 → 오프셋 → 주입+reset → 충돌 검사 → base 재계산 대조)를
그대로 호출해, 발사 전에 어떤 (key, scene, j) 가 충돌/불일치로 죽는지 미리 찾는다. 결과는 TSV.
사용(robocasa 컨테이너, GPU 불필요):
  PYTHONPATH=<워크트리>:/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite \
  python probe_v6_jitter_cells.py --plan <plan.json> --out <tsv> [--keys a,b] [--procs 8]
"""
from __future__ import annotations
import argparse, csv, importlib.util, sys, traceback
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]


def _load_collector():
    p = REPO / "scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py"
    spec = importlib.util.spec_from_file_location("hfc", p); m = importlib.util.module_from_spec(spec)
    sys.modules["hfc"] = m; spec.loader.exec_module(m)
    return m


def probe(args):
    plan_path, key, sid, jid = args
    try:
        sys.path.insert(0, str(REPO))
        import gymnasium as gym
        import robocasa, robocasa.utils.gym_utils.gymnasium_groot, robosuite  # noqa
        from src.collect.plan import CollectionPlan
        from src.policies.groot.robocasa.scenario_replay import get_robocasa_ep_meta
        hfc = _load_collector()
        plan = CollectionPlan.load(plan_path)
        cell = next(c for c in plan.cells() if c.instruction == key and c.scene_idx == sid and c.jitter_idx == jid and c.noise_idx == 0)
        env_name = plan.extra["env_names"][key]
        env = gym.make(env_name, enable_render=False, seed=cell.env_seed, **plan.env_kwargs)
        env.reset(seed=cell.env_seed)
        em = get_robocasa_ep_meta(env)
        em2, _ = hfc._v6_apply_jitter(env, cell, em, plan.extra.get("instruction_text", {}).get(key))
        env.close()
        return dict(key=key, sid=sid, jid=jid, env_seed=cell.env_seed, lat=cell.base_lat, back=cell.base_back,
                    reset_idx=cell.jitter_reset_idx, base=str([round(float(x), 3) for x in em2["init_robot_base_pos"]]), ok=1, err="")
    except Exception as e:  # noqa: BLE001
        return dict(key=key, sid=sid, jid=jid, ok=0, err=(str(e)[:200] or traceback.format_exc()[-200:]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--plan", required=True); ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--keys", default=""); ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    sys.path.insert(0, str(REPO))
    from src.collect.plan import CollectionPlan
    plan = CollectionPlan.load(a.plan)
    keys = [k for k in a.keys.split(",") if k] or list(plan.scenes)
    jobs = sorted({(a.plan, c.instruction, c.scene_idx, c.jitter_idx) for c in plan.cells() if c.instruction in keys})
    cols = ["key", "sid", "jid", "env_seed", "lat", "back", "reset_idx", "base", "ok", "err"]
    with Pool(a.procs, maxtasksperchild=10) as pool, a.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore"); w.writeheader()
        for r in pool.imap_unordered(probe, jobs, chunksize=1):
            w.writerow(r); f.flush()
    rows = list(csv.DictReader(a.out.open(), delimiter="\t"))
    bad = [r for r in rows if r["ok"] != "1"]
    print(f"probe {len(rows)} cells: ok {len(rows)-len(bad)} / fail {len(bad)}")
    for r in bad: print("  FAIL", r["key"], r["sid"], r["jid"], r["err"])


if __name__ == "__main__":
    main()

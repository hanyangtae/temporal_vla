#!/usr/bin/env python3
"""scene 별 서랍 손잡이 좌표 프로브 (정책·GPU 불필요, env reset 만).

목적: exp5-3 의 **scene 별 setpoint s_t 가 물리적 위치(기하) 암기인가** 를 검정하기 위한
좌표 확보. 전역 setpoint 는 read-out 0.59 로 붕괴하고 scene 별은 0.868 유지했다 →
성공 위치가 scene 마다 다르다는 뜻이고, 그게 손잡이 기하와 상관되면 "공간 암기" 확정.

fixture 접근 규약은 analyze/mixer_scene_feasibility.py:73-91 (drawer 분기) 와 동일:
  k.drawer, naming_prefix, {name}_slidejoint, q_phys_max = size[1]*0.55
출력: TSV (seed, handle_xyz, base_xyz, 상대좌표, 거리, size, prefix)
"""
import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")


def probe(env_name: str, seed: int):
    import numpy as np
    import gymnasium as gym
    import robocasa  # noqa: F401
    import robosuite  # noqa: F401
    from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401
    from robocasa_event_labeler import find_robocasa_env

    env = gym.make(env_name, enable_render=False, seed=seed)
    env.reset(seed=seed)
    k = find_robocasa_env(env)
    sim = k.sim
    fx = k.drawer
    pref = fx.naming_prefix

    # 손잡이 body 후보: 슬라이드 하위 트리에서 이름에 handle 이 든 body
    jname = f"{fx.name}_slidejoint"
    jid = sim.model.joint_name2id(jname)
    slide_bid = int(sim.model.jnt_bodyid[jid])
    handle_bids = []
    for b in range(sim.model.nbody):
        nm = sim.model.body_id2name(b) or ""
        if "handle" in nm.lower() and nm.startswith(pref):
            handle_bids.append((b, nm))
    # 손잡이 body 가 없으면 슬라이드 body 자체 사용 (fixture 변형 대비)
    if handle_bids:
        hx = np.mean([sim.data.body_xpos[b] for b, _ in handle_bids], axis=0)
        hname = ",".join(n for _, n in handle_bids)[:60]
    else:
        hx = np.array(sim.data.body_xpos[slide_bid])
        hname = f"__slide__{sim.model.body_id2name(slide_bid)}"

    try:
        bbid = sim.model.body_name2id("mobilebase0_base")
    except Exception:  # noqa: BLE001
        bbid = sim.model.body_name2id("base0_base")
    bx = np.array(sim.data.body_xpos[bbid])
    rel = hx - bx
    env.close()
    return dict(seed=seed, hx=hx, bx=bx, rel=rel,
                dist=float(np.linalg.norm(rel)),
                size=[float(v) for v in fx.size], prefix=pref, hname=hname)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-name",
                    default="robocasa_panda_omron/OpenDrawer_PandaOmron_Env")
    ap.add_argument("--seeds", required=True, help="쉼표구분")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    with open(args.out, "w") as fh:
        fh.write("seed\thx\thy\thz\tbx\tby\tbz\trelx\trely\trelz\tdist\t"
                 "size0\tsize1\tsize2\tprefix\thandle_body\n")
        for s in seeds:
            try:
                r = probe(args.env_name, s)
                fh.write("\t".join([
                    str(r["seed"]),
                    *[f"{v:.5f}" for v in r["hx"]],
                    *[f"{v:.5f}" for v in r["bx"]],
                    *[f"{v:.5f}" for v in r["rel"]],
                    f"{r['dist']:.5f}",
                    *[f"{v:.5f}" for v in r["size"][:3]],
                    r["prefix"], r["hname"]]) + "\n")
            except Exception as exc:  # noqa: BLE001
                fh.write(f"{s}\t__ERROR__\t{type(exc).__name__}: {exc}\n")
            fh.flush()
            print(f"  probed {s}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

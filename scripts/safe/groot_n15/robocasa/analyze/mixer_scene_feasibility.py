"""StandMixer scene 실현가능성 사전 판정 (정책 무관, reset 시점 기하 스윕).

배경: OpenStandMixerHead 실패판 중 일부는 **머리를 들 공간이 없어서** 못 여는 scene 이다
(위 선반/캐비닛에 막힘). 이건 latent 실패가 아니라 scene 불가능이므로 succ/fail 대조
fit 의 failure 클래스에 섞이면 안 된다.

방법: reset 직후 로봇을 건드리지 않고 head 관절만 0→1.05 로 스윕하며 매 단계에서
head body geom 이 **믹서 자신·로봇 이외의 geom** 과 접촉하는지 본다. 접촉 없이 도달 가능한
최대 정규화 각도 = ``q_max_feasible``. env 성공 임계는 0.99 이므로

    q_max_feasible < 0.99  →  그 seed 는 어떤 정책으로도 성공 불가 (제외 대상)

정책·체크포인트·chunk 설정과 무관하고 seed 만의 함수라, 모든 arm 에 동일하게 적용되며
succ/fail 어느 쪽으로도 편향을 만들지 않는다.

사용 (robocasa 컨테이너):
  python mixer_scene_feasibility.py --seeds 100000-100011 [--task OpenStandMixerHead]
"""

from __future__ import annotations

import argparse
import json
import sys

_ROBOT_PREFIXES = ("robot", "gripper", "mobilebase", "base")


def parse_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def probe_one(env_name: str, seed: int, steps: int) -> dict:
    import numpy as np
    import gymnasium as gym
    import robocasa  # noqa: F401
    import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401
    import robosuite  # noqa: F401
    from robocasa_event_labeler import find_robocasa_env

    env = gym.make(env_name, enable_render=False, seed=seed)
    env.reset(seed=seed)
    k = find_robocasa_env(env)
    fx = k.stand_mixer
    sim = k.sim
    pref = fx.naming_prefix
    jn = fx._joint_names["head"]

    head_bid = sim.model.body_name2id(pref + "head")
    head_geoms = {g for g in range(sim.model.ngeom)
                  if int(sim.model.geom_bodyid[g]) == head_bid}

    def _is_external(gid: int) -> bool:
        """믹서 자신도 로봇도 아닌 geom (= 주변 가구/벽/선반)."""
        name = sim.model.geom_id2name(gid) or ""
        if name.startswith(pref):
            return False
        return not any(name.startswith(p) for p in _ROBOT_PREFIXES)

    # 원래 상태 보존 (프로브가 episode 를 오염시키지 않게 — 별도 프로세스지만 방어적으로)
    qpos0 = np.array(sim.data.qpos)

    q_max = 0.0
    blocker = None
    blocked_at = None
    for i in range(steps + 1):
        v = i / steps          # set_joint_state 는 정규화값 [0,1] 만 허용
        fx.set_joint_state(env=k, min=v, max=v, joint_names=[jn])
        sim.forward()
        hit = None
        for c in range(sim.data.ncon):
            con = sim.data.contact[c]
            g1, g2 = int(con.geom1), int(con.geom2)
            if g1 in head_geoms and _is_external(g2):
                hit = sim.model.geom_id2name(g2)
                break
            if g2 in head_geoms and _is_external(g1):
                hit = sim.model.geom_id2name(g1)
                break
        if hit is not None:
            blocker = hit
            blocked_at = round(v, 4)
            break
        q_max = v

    sim.data.qpos[:] = qpos0
    sim.forward()
    env.close()

    return {
        "seed": seed,
        "q_max_feasible": round(min(q_max, 1.0), 4),
        "feasible": bool(q_max >= 0.99),
        "blocked_at": blocked_at,
        "blocker_geom": blocker,
        "fixture_prefix": pref,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="OpenStandMixerHead")
    ap.add_argument("--seeds", default="100000-100011")
    ap.add_argument("--steps", type=int, default=40, help="스윕 분할 수 (0→1.0)")
    ap.add_argument("--out", default=None, help="결과 JSON 경로")
    ap.add_argument("--seed", type=int, default=None, help="워커 모드(내부용)")
    args = ap.parse_args()

    env_name = f"robocasa_panda_omron/{args.task}_PandaOmron_Env"
    if args.seed is not None:  # 워커: seed 하나만 처리하고 JSON 한 줄 출력
        print("RESULT " + json.dumps(probe_one(env_name, args.seed, args.steps)))
        return

    # 드라이버: seed 당 fresh 프로세스 (한 프로세스 연속 gym.make 시 scene 오염 — docs/steering/18 §2)
    import subprocess

    rows = []
    for s in parse_seeds(args.seeds):
        r = subprocess.run(
            [sys.executable, __file__, "--task", args.task, "--seed", str(s),
             "--steps", str(args.steps)],
            capture_output=True, text=True, timeout=600,
        )
        line = next((ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")), None)
        if line is None:
            print(f"[feas] seed {s}: FAILED\n{r.stdout[-500:]}\n{r.stderr[-500:]}", flush=True)
            continue
        row = json.loads(line[len("RESULT "):])
        rows.append(row)
        mark = "OK  " if row["feasible"] else "BLOCKED"
        print(f"[feas] seed {row['seed']}  q_max={row['q_max_feasible']:.3f}  {mark}"
              f"  blocker={row['blocker_geom']}", flush=True)

    n_bad = sum(1 for r in rows if not r["feasible"])
    print(f"\n[summary] {len(rows)} seeds  infeasible={n_bad} "
          f"({n_bad / len(rows):.1%})" if rows else "[summary] no rows", flush=True)
    if args.out and rows:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"[wrote] {args.out}", flush=True)


if __name__ == "__main__":
    main()

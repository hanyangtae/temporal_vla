"""StandMixer/Drawer scene 실현가능성 사전 판정 (정책 무관, reset 시점 기하 스윕).

배경: OpenStandMixerHead 실패판 중 일부는 **머리를 들 공간이 없어서** 못 여는 scene 이다
(위 선반/캐비닛에 막힘). 이건 latent 실패가 아니라 scene 불가능이므로 succ/fail 대조
fit 의 failure 클래스에 섞이면 안 된다.

방법: reset 직후 로봇을 건드리지 않고 head 관절만 0→1.05 로 스윕하며 매 단계에서
head body geom 이 **믹서 자신·로봇 이외의 geom** 과 접촉하는지 본다. 접촉 없이 도달 가능한
최대 정규화 각도 = ``q_max_feasible``. env 성공 임계는 0.99 이므로

    q_max_feasible < 0.99  →  그 seed 는 어떤 정책으로도 성공 불가 (제외 대상)

정책·체크포인트·chunk 설정과 무관하고 seed 만의 함수라, 모든 arm 에 동일하게 적용되며
succ/fail 어느 쪽으로도 편향을 만들지 않는다.

부산물: seed 별 `ep_lang` (= `env.get_ep_meta()["lang"]`) 도 같이 기록한다. **OpenDrawer 는
seed 마다 좌/우 variant 가 바뀌므로**(kitchen_drawer.py `_place_robot` → `drawer_side`,
lang "Open the {side} drawer.") 이 필드가 exp5-2 `drawer_left` cell 의 seed 가 정말 left 인지
정책 없이 판정하는 근거다. 결과 JSON 은 `--out` 으로 지정한 경로에만 쓴다 (main-tree outputs
하드코딩 없음).

사용 (robocasa 컨테이너):
  python mixer_scene_feasibility.py --seeds 100000-100011 [--task OpenStandMixerHead] --out <path>
  python mixer_scene_feasibility.py --task OpenDrawer --fixture drawer --seeds <...> --out <path>

Drawer 이식 (exp4-1 B3, Gate2 P2): 관절 = `{name}_slidejoint` (sign −1, size[1]·0.55 로
정규화 — get_door_state 와 동일 규약), 성공역 open ≥0.95 → 임계 0.95. 함정: 서랍 **안**의
자유 물체는 상시 접촉이라 blocker 가 아님 → blocker 는 **자유관절(free joint) body 가 아닌
외부 geom** 만 인정 (정적 가구·벽·타 fixture). 물체는 서랍과 함께 밀려 나오는 동역학이라
기하 불가 판정 대상이 아니다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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


def probe_one(env_name: str, seed: int, steps: int, fixture: str = "stand_mixer") -> dict:
    import numpy as np
    import gymnasium as gym
    import robocasa  # noqa: F401
    import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401
    import robosuite  # noqa: F401
    _repo = Path(__file__).resolve().parents[5]
    if str(_repo) not in sys.path:
        sys.path.insert(0, str(_repo))
    from src.collect.robocasa.event_labeler import find_robocasa_env

    env = gym.make(env_name, enable_render=False, seed=seed)
    env.reset(seed=seed)
    k = find_robocasa_env(env)
    sim = k.sim
    try:  # instruction variant (OpenDrawer 좌/우 판정용) — 정책 무관, reset 만의 함수
        ep_lang = str((k.get_ep_meta() or {}).get("lang", ""))
    except Exception:
        ep_lang = ""

    if fixture == "stand_mixer":
        fx = k.stand_mixer
        pref = fx.naming_prefix
        jn = fx._joint_names["head"]
        threshold = 0.99
        moving_bids = {sim.model.body_name2id(pref + "head")}

        def _set_q(v: float) -> None:
            fx.set_joint_state(env=k, min=v, max=v, joint_names=[jn])

        def _movable_ok(_gid: int) -> bool:
            return True  # 믹서 머리 경로에 자유 물체 없음 (env.objects 비어 있음)
    elif fixture == "drawer":
        fx = k.drawer
        pref = fx.naming_prefix
        jname = f"{fx.name}_slidejoint"
        threshold = 0.95  # 라벨러/성공역: open ≥0.95
        q_phys_max = float(fx.size[1]) * 0.55  # get_door_state 정규화 규약과 동일
        jid = sim.model.joint_name2id(jname)
        slide_bid = int(sim.model.jnt_bodyid[jid])
        # 슬라이드 body 의 하위 트리 전부 (서랍 상자 + 손잡이)
        moving_bids = {b for b in range(sim.model.nbody)
                       if _in_subtree(sim, b, slide_bid)}

        def _set_q(v: float) -> None:
            sim.data.set_joint_qpos(jname, -v * q_phys_max)  # sign −1 (get_door_state)

        def _movable_ok(gid: int) -> bool:
            """자유관절 body(=움직이는 물체)는 blocker 로 안 침 — 서랍 안 물체 오탐 방지."""
            return not _has_free_root(sim, int(sim.model.geom_bodyid[gid]))
    else:
        raise ValueError(f"unknown fixture: {fixture}")

    head_geoms = {g for g in range(sim.model.ngeom)
                  if int(sim.model.geom_bodyid[g]) in moving_bids}

    def _is_external(gid: int) -> bool:
        """fixture 자신도 로봇도 아닌 geom (= 주변 가구/벽/선반). drawer 는 자유물체도 제외."""
        name = sim.model.geom_id2name(gid) or ""
        if name.startswith(pref):
            return False
        if any(name.startswith(p) for p in _ROBOT_PREFIXES):
            return False
        return _movable_ok(gid)

    # 원래 상태 보존 (프로브가 episode 를 오염시키지 않게 — 별도 프로세스지만 방어적으로)
    qpos0 = np.array(sim.data.qpos)

    q_max = 0.0
    blocker = None
    blocked_at = None
    for i in range(steps + 1):
        v = i / steps          # 정규화값 [0,1]
        _set_q(v)
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
        "fixture": fixture,
        "q_max_feasible": round(min(q_max, 1.0), 4),
        "feasible": bool(q_max >= threshold),
        "threshold": threshold,
        "blocked_at": blocked_at,
        "blocker_geom": blocker,
        "fixture_prefix": pref,
        "ep_lang": ep_lang,
    }


def _in_subtree(sim, body_id: int, root_id: int) -> bool:
    b = body_id
    while b != 0:
        if b == root_id:
            return True
        b = int(sim.model.body_parentid[b])
    return body_id == root_id


def _has_free_root(sim, body_id: int) -> bool:
    """body 조상 체인에 free joint 가 있으면 자유 물체 (mjJNT_FREE=0)."""
    b = body_id
    while b != 0:
        adr = int(sim.model.body_jntadr[b])
        num = int(sim.model.body_jntnum[b])
        for j in range(adr, adr + num):
            if j >= 0 and int(sim.model.jnt_type[j]) == 0:
                return True
        b = int(sim.model.body_parentid[b])
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="OpenStandMixerHead")
    ap.add_argument("--fixture", default="stand_mixer", choices=("stand_mixer", "drawer"),
                    help="스윕 대상 fixture (drawer=OpenDrawer 이식, Gate2 P2)")
    ap.add_argument("--seeds", default="100000-100011")
    ap.add_argument("--steps", type=int, default=40, help="스윕 분할 수 (0→1.0)")
    ap.add_argument("--out", default=None, help="결과 JSON 경로")
    ap.add_argument("--seed", type=int, default=None, help="워커 모드(내부용)")
    args = ap.parse_args()

    env_name = f"robocasa_panda_omron/{args.task}_PandaOmron_Env"
    if args.seed is not None:  # 워커: seed 하나만 처리하고 JSON 한 줄 출력
        print("RESULT " + json.dumps(probe_one(env_name, args.seed, args.steps, args.fixture)))
        return

    # 드라이버: seed 당 fresh 프로세스 (한 프로세스 연속 gym.make 시 scene 오염 — docs/steering/18 §2)
    import subprocess

    rows = []
    for s in parse_seeds(args.seeds):
        r = subprocess.run(
            [sys.executable, __file__, "--task", args.task, "--fixture", args.fixture,
             "--seed", str(s), "--steps", str(args.steps)],
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
              f"  blocker={row['blocker_geom']}  lang={row.get('ep_lang', '')!r}", flush=True)

    langs = sorted({r.get("ep_lang", "") for r in rows if r.get("ep_lang")})
    if len(langs) > 1:  # OpenDrawer 좌/우처럼 seed 마다 instruction 이 갈리는 경우
        print(f"[lang] instruction variant {len(langs)}종: {langs}", flush=True)
        for lg in langs:
            seeds_lg = [r["seed"] for r in rows if r.get("ep_lang") == lg]
            print(f"[lang]   {lg!r}: seeds={seeds_lg}", flush=True)

    n_bad = sum(1 for r in rows if not r["feasible"])
    print(f"\n[summary] {len(rows)} seeds  infeasible={n_bad} "
          f"({n_bad / len(rows):.1%})" if rows else "[summary] no rows", flush=True)
    if args.out and rows:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"[wrote] {args.out}", flush=True)


if __name__ == "__main__":
    main()

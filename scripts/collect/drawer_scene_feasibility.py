# 유래: n15 analyze/ (2026-08-10 기능명 재배치 — docs/review/RENAME_PLAN.md)
"""OpenDrawer scene 실현가능성 사전 판정 (정책 무관, reset 시점 기하 스윕).

`mixer_scene_feasibility.py` 의 drawer 이식판 (docs/steering/SCENE_FEASIBILITY.md §5).
원본에서 바꾼 건 네 가지뿐이다:

  ① fixture 참조 : `env.stand_mixer` → `env.drawer`
  ② 관절        : `_joint_names["head"]` → `{fixture.name}_slidejoint`
                   (sign −1, `size[1]*0.55` 로 정규화 — `Drawer.get_door_state` 규약과 동일)
  ③ 움직이는 body: head body → slide joint body 의 하위 트리 전부 (서랍 상자 + 손잡이)
  ④ 성공 임계    : 0.99 → 0.95 (라벨러 `DrawerPhaseLabeler._done`: open ≥0.95)

방법: reset 직후 로봇을 건드리지 않고 서랍 관절만 0→1.0 으로 스윕하며, 매 단계에서 서랍
body geom 이 **서랍 자신·로봇 이외의 외부 geom** 과 접촉하는지 본다. 접촉 없이 도달 가능한
최대 정규화값 = ``q_max_feasible``.

    q_max_feasible < 0.95  →  그 seed 는 어떤 정책으로도 성공 불가 (제외 대상)

함정 (drawer 고유): 서랍 **안**의 자유 물체는 상시 접촉이라 blocker 가 아니다. 물체는 서랍과
함께 밀려 나오는 동역학이므로 기하 불가 판정 대상이 아님 → blocker 는 **자유관절(free joint)
조상을 갖지 않는 외부 geom** 만 인정한다 (정적 가구·벽·타 fixture).

부산물: seed 별 `ep_lang` (= `env.get_ep_meta()["lang"]`) 과 그로부터 파싱한 `side`
(left/right/None) 를 같이 기록한다. **OpenDrawer 는 seed 마다 좌/우 variant 가 갈리므로**
(kitchen_drawer.py `_place_robot` → `drawer_side`, lang "Open the {side} drawer.") 이 필드가
나중에 left/right 필터와 feasibility 를 교차하는 근거다.

한계 (원본과 동일): 정적·관절만 검사한다. 그리퍼가 손잡이를 잡은 채 움직이다 그리퍼 자체가
걸리는 경우는 못 잡는다 (로봇 geom 은 접촉 판정에서 제외).

사용 (robocasa 컨테이너, GPU 불필요):
  MUJOCO_GL=egl PYTHONPATH="/temporal_vla/src/policies/Isaac-GR00T:\
/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla" \
  python drawer_scene_feasibility.py --seeds 100000-100011 --out <path>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROBOT_PREFIXES = ("robot", "gripper", "mobilebase", "base")

THRESHOLD = 0.95  # 라벨러/성공역: open ≥0.95 (DrawerPhaseLabeler._done)


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


def _parse_side(lang: str) -> str | None:
    """OpenDrawer instruction 에서 좌/우 variant 파싱 ("Open the left drawer.")."""
    low = (lang or "").lower()
    if "left" in low:
        return "left"
    if "right" in low:
        return "right"
    return None


def probe_one(env_name: str, seed: int, steps: int) -> dict:
    import numpy as np
    import gymnasium as gym
    import robocasa  # noqa: F401
    import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401
    import robosuite  # noqa: F401
    _repo = Path(__file__).resolve().parents[2]
    if str(_repo) not in sys.path:
        sys.path.insert(0, str(_repo))
    from src.collect.robocasa.event_labeler import find_robocasa_env

    env = gym.make(env_name, enable_render=False, seed=seed)
    env.reset(seed=seed)
    k = find_robocasa_env(env)
    sim = k.sim
    try:  # instruction variant (좌/우) — 정책 무관, reset 만의 함수
        ep_lang = str((k.get_ep_meta() or {}).get("lang", ""))
    except Exception:
        ep_lang = ""

    fx = k.drawer                       # ① fixture 참조
    pref = fx.naming_prefix
    jname = f"{fx.name}_slidejoint"     # ② 관절
    q_phys_max = float(fx.size[1]) * 0.55  # get_door_state 정규화 규약과 동일
    jid = sim.model.joint_name2id(jname)
    slide_bid = int(sim.model.jnt_bodyid[jid])
    # ③ 움직이는 body = 슬라이드 body 의 하위 트리 전부 (서랍 상자 + 손잡이)
    moving_bids = {b for b in range(sim.model.nbody) if _in_subtree(sim, b, slide_bid)}
    drawer_geoms = {g for g in range(sim.model.ngeom)
                    if int(sim.model.geom_bodyid[g]) in moving_bids}

    def _set_q(v: float) -> None:
        sim.data.set_joint_qpos(jname, -v * q_phys_max)  # sign −1 (get_door_state)

    def _is_external(gid: int) -> bool:
        """서랍 자신도 로봇도 아니고, 자유물체도 아닌 geom (= 주변 가구/벽/타 fixture)."""
        name = sim.model.geom_id2name(gid) or ""
        if name.startswith(pref):
            return False
        if any(name.startswith(p) for p in _ROBOT_PREFIXES):
            return False
        # 서랍 안 자유 물체는 blocker 가 아님 (서랍과 함께 밀려 나옴)
        return not _has_free_root(sim, int(sim.model.geom_bodyid[gid]))

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
            if g1 in drawer_geoms and _is_external(g2):
                hit = sim.model.geom_id2name(g2)
                break
            if g2 in drawer_geoms and _is_external(g1):
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
        "fixture": "drawer",
        "q_max_feasible": round(min(q_max, 1.0), 4),
        "feasible": bool(q_max >= THRESHOLD),  # ④ 성공 임계
        "threshold": THRESHOLD,
        "blocked_at": blocked_at,
        "blocker_geom": blocker,
        "fixture_prefix": pref,
        "ep_lang": ep_lang,
        "side": _parse_side(ep_lang),
    }


def _run_worker(args_task: str, seed: int, steps: int, timeout: int = 600) -> dict | None:
    import subprocess

    r = subprocess.run(
        [sys.executable, __file__, "--task", args_task, "--seed", str(seed),
         "--steps", str(steps)],
        capture_output=True, text=True, timeout=timeout,
    )
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")), None)
    if line is None:
        print(f"[feas] seed {seed}: FAILED\n{r.stdout[-500:]}\n{r.stderr[-500:]}", flush=True)
        return None
    return json.loads(line[len("RESULT "):])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="OpenDrawer")
    ap.add_argument("--seeds", default="100000-100011")
    ap.add_argument("--steps", type=int, default=40, help="스윕 분할 수 (0→1.0)")
    ap.add_argument("--jobs", type=int, default=8,
                    help="동시 워커 프로세스 수 (CPU cap: ≤16)")
    ap.add_argument("--out", default=None, help="결과 JSON 경로")
    ap.add_argument("--seed", type=int, default=None, help="워커 모드(내부용)")
    args = ap.parse_args()

    env_name = f"robocasa_panda_omron/{args.task}_PandaOmron_Env"
    if args.seed is not None:  # 워커: seed 하나만 처리하고 JSON 한 줄 출력
        print("RESULT " + json.dumps(probe_one(env_name, args.seed, args.steps)))
        return

    # 드라이버: seed 당 fresh 프로세스 (한 프로세스 연속 gym.make 시 scene 오염 —
    # docs/steering/SCENE_FEASIBILITY.md §2)
    from concurrent.futures import ProcessPoolExecutor

    seeds = parse_seeds(args.seeds)
    jobs = max(1, min(int(args.jobs), 16))
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = [(s, ex.submit(_run_worker, args.task, s, args.steps)) for s in seeds]
        for s, fut in futs:
            try:
                row = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"[feas] seed {s}: EXC {e}", flush=True)
                continue
            if row is None:
                continue
            rows.append(row)
            mark = "OK  " if row["feasible"] else "BLOCKED"
            print(f"[feas] seed {row['seed']}  q_max={row['q_max_feasible']:.3f}  {mark}"
                  f"  blocker={row['blocker_geom']}  side={row.get('side')}"
                  f"  lang={row.get('ep_lang', '')!r}", flush=True)

    rows.sort(key=lambda r: r["seed"])
    langs = sorted({r.get("ep_lang", "") for r in rows if r.get("ep_lang")})
    if len(langs) > 1:  # OpenDrawer 좌/우처럼 seed 마다 instruction 이 갈리는 경우
        print(f"[lang] instruction variant {len(langs)}종: {langs}", flush=True)
        for lg in langs:
            seeds_lg = [r["seed"] for r in rows if r.get("ep_lang") == lg]
            n_bad_lg = sum(1 for r in rows
                           if r.get("ep_lang") == lg and not r["feasible"])
            print(f"[lang]   {lg!r}: n={len(seeds_lg)} infeasible={n_bad_lg}", flush=True)

    if rows:
        n_bad = sum(1 for r in rows if not r["feasible"])
        print(f"\n[summary] {len(rows)} seeds  infeasible={n_bad} "
              f"({n_bad / len(rows):.1%})", flush=True)
        bad = [r["seed"] for r in rows if not r["feasible"]]
        if bad:
            print(f"[summary] BLOCKED seeds: {bad}", flush=True)
    else:
        print("[summary] no rows", flush=True)

    if args.out and rows:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"[wrote] {args.out}", flush=True)


if __name__ == "__main__":
    main()

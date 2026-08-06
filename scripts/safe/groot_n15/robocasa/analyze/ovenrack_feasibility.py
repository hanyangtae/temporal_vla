"""SlideOvenRack scene 실현가능성 사전 판정 (정책 무관, reset 시점 기하 스윕).

`drawer_scene_feasibility.py` 의 오븐 랙 이식판 (docs/steering/SCENE_FEASIBILITY.md §5).
템플릿에서 바꾼 건 네 가지다:

  ① fixture 참조 : `env.drawer` → `env.oven`
  ② 관절        : 대상 랙이 **에피소드마다 랜덤인 `env.rack_level`** 로 정해진다.
                   `oven.get_state(rack_level=env.rack_level)` 의 첫 `"rack*"` 키가 곧
                   body/관절 base → 관절명 `{naming_prefix}{key}_joint`
                   (oven fixture 의 `slide_rack`/`check_rack_contact` 와 동일 규약,
                   `OvenRackPhaseLabeler._rack_entry` 와 같은 경로. `get_state` 는
                   rack0 로 fallback 하므로 랙이 하나뿐인 모델도 그대로 동작).
                   정규화는 sign 을 손으로 다루지 않고 `Fixture.get_joint_state` /
                   `set_joint_state` 규약(`_joint_infos[j]["range"]`, joint_min<0 이면 1−t)을 쓴다.
  ③ 움직이는 body: 랙 관절 body 의 하위 트리 전부
  ④ 성공 임계    : 0.95 (`SlideOvenRack._check_success`: should_pull(out) → rack ≥0.95,
                   라벨러 `SlidingRackPhaseLabeler.OPEN_TH` 와 동일)

★ 스윕 시작점이 drawer 와 다르다. out 에피소드는 `_setup_scene` 이
`oven.slide_rack(self, value=0.50, rack_level=...)` 로 랙을 **q=0.5 에서 시작**시킨다
(in 에피소드는 value=1.0). 스윕은 0 이 아니라 **현재값 → 1.0**. 결과에 `q_start` 를 기록.

방법: reset 직후 로봇을 건드리지 않고 랙 관절만 현재값→1.0 으로 스윕하며, 매 단계에서 랙
body geom 이 **랙 자신·로봇 이외의 외부 geom** 과 접촉하는지 본다. 접촉 없이 도달 가능한
최대 정규화값 = ``q_max_feasible``.

    q_max_feasible < 0.95  →  그 seed 는 어떤 정책으로도 성공 불가 (제외 대상)

함정: 랙 **위**의 자유 물체는 상시 접촉이라 blocker 가 아니다 (랙과 함께 밀려 나옴) →
blocker 는 **자유관절(free joint) 조상을 갖지 않는 외부 geom** 만 인정한다.

부산물: seed 별 `ep_lang`, `rack_level`, `should_pull` 을 같이 기록한다. SlideOvenRack 은
seed 마다 out/in 방향과 층(top/bottom, 다층 오븐일 때만 문구에 노출)이 갈리므로
(kitchen_oven.py `should_pull`/`rack_level` = rng) 이 필드가 필터 교차의 근거다.

한계 (원본과 동일): 정적·관절만 검사한다. 그리퍼가 랙을 잡은 채 움직이다 그리퍼 자체가
걸리는 경우는 못 잡는다 (로봇 geom 은 접촉 판정에서 제외).

사용 (robocasa 컨테이너, GPU 불필요):
  MUJOCO_GL=egl PYTHONPATH="/temporal_vla/src/policies/Isaac-GR00T:\
/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla" \
  python ovenrack_feasibility.py --seeds 100000-100011 --out <path>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROBOT_PREFIXES = ("robot", "gripper", "mobilebase", "base")

THRESHOLD = 0.95  # should_pull(out) 성공역: rack ≥0.95 (_check_success / OPEN_TH)


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


def _norm_to_qpos(fx, jname: str, v: float) -> float:
    """정규화값 v(0~1) → 실제 qpos. `Fixture.set_joint_state(min=max=v)` 와 동일 규약."""
    jmin, jmax = fx._joint_infos[jname]["range"]
    if jmin >= 0:
        return jmin + (jmax - jmin) * v
    return jmin + (jmax - jmin) * (1.0 - v)


def _rack_joint(k, fx) -> tuple[str, str]:
    """(관절명, rack 키) — `OvenRackPhaseLabeler._rack_entry` 와 동일 경로."""
    fx.update_state(k)  # _rack 딕셔너리 최신화 (get_state 가 이걸 읽는다)
    st = fx.get_state(rack_level=int(getattr(k, "rack_level", 0)))
    keys = [key for key in st if key.startswith("rack")]
    if not keys:
        raise RuntimeError("oven get_state 에 rack 키 없음")
    key = keys[0]
    return f"{fx.naming_prefix}{key}_joint", key


def probe_one(env_name: str, seed: int, steps: int) -> dict:
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
    try:  # instruction variant (out/in, top/bottom) — 정책 무관, reset 만의 함수
        ep_lang = str((k.get_ep_meta() or {}).get("lang", ""))
    except Exception:
        ep_lang = ""
    should_pull = bool(getattr(k, "should_pull", True))
    rack_level = int(getattr(k, "rack_level", 0))

    fx = k.oven                          # ① fixture 참조
    pref = fx.naming_prefix
    jname, rack_key = _rack_joint(k, fx)  # ② 관절 (rack_level 의존)
    jid = sim.model.joint_name2id(jname)
    rack_bid = int(sim.model.jnt_bodyid[jid])
    # ③ 움직이는 body = 랙 body 의 하위 트리 전부
    moving_bids = {b for b in range(sim.model.nbody) if _in_subtree(sim, b, rack_bid)}
    rack_geoms = {g for g in range(sim.model.ngeom)
                  if int(sim.model.geom_bodyid[g]) in moving_bids}

    # ★ 스윕 시작점 = reset 직후 현재값 (out 에피소드면 slide_rack(value=0.5) 로 ≈0.5)
    q_start = float(fx.get_joint_state(k, [jname])[jname])

    def _set_q(v: float) -> None:
        sim.data.set_joint_qpos(jname, _norm_to_qpos(fx, jname, v))

    def _is_external(gid: int) -> bool:
        """랙 자신도 로봇도 아니고, 자유물체도 아닌 geom (= 주변 가구/벽/타 fixture)."""
        name = sim.model.geom_id2name(gid) or ""
        if name.startswith(pref):
            return False
        if any(name.startswith(p) for p in _ROBOT_PREFIXES):
            return False
        return not _has_free_root(sim, int(sim.model.geom_bodyid[gid]))

    # 원래 상태 보존 (프로브가 episode 를 오염시키지 않게 — 별도 프로세스지만 방어적으로)
    qpos0 = np.array(sim.data.qpos)

    q_max = q_start
    blocker = None
    blocked_at = None
    span = max(0.0, 1.0 - q_start)
    for i in range(steps + 1):
        v = q_start + span * (i / steps)     # 정규화값 [q_start, 1]
        _set_q(v)
        sim.forward()
        hit = None
        for c in range(sim.data.ncon):
            con = sim.data.contact[c]
            g1, g2 = int(con.geom1), int(con.geom2)
            if g1 in rack_geoms and _is_external(g2):
                hit = sim.model.geom_id2name(g2)
                break
            if g2 in rack_geoms and _is_external(g1):
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
        "fixture": "oven",
        "joint": jname,
        "rack_key": rack_key,
        "rack_level": rack_level,
        "q_start": round(q_start, 4),
        "q_max_feasible": round(min(q_max, 1.0), 4),
        "feasible": bool(q_max >= THRESHOLD),  # ④ 성공 임계
        "threshold": THRESHOLD,
        "blocked_at": blocked_at,
        "blocker_geom": blocker,
        "fixture_prefix": pref,
        "ep_lang": ep_lang,
        "should_pull": should_pull,
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
    ap.add_argument("--task", default="SlideOvenRack")
    ap.add_argument("--seeds", default="100000-100011")
    ap.add_argument("--steps", type=int, default=40, help="스윕 분할 수 (q_start→1.0)")
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
            print(f"[feas] seed {row['seed']}  lvl={row['rack_level']}"
                  f"  q_start={row['q_start']:.3f}"
                  f"  q_max={row['q_max_feasible']:.3f}  {mark}"
                  f"  blocker={row['blocker_geom']}"
                  f"  lang={row.get('ep_lang', '')!r}", flush=True)

    rows.sort(key=lambda r: r["seed"])
    langs = sorted({r.get("ep_lang", "") for r in rows if r.get("ep_lang")})
    if len(langs) > 1:
        print(f"[lang] instruction variant {len(langs)}종: {langs}", flush=True)
        for lg in langs:
            seeds_lg = [r["seed"] for r in rows if r.get("ep_lang") == lg]
            n_bad_lg = sum(1 for r in rows
                           if r.get("ep_lang") == lg and not r["feasible"])
            print(f"[lang]   {lg!r}: n={len(seeds_lg)} infeasible={n_bad_lg}", flush=True)

    if rows:
        starts = sorted({r["q_start"] for r in rows})
        print(f"[q_start] 관측값 {starts[:5]}{' ...' if len(starts) > 5 else ''}", flush=True)
        n_bad = sum(1 for r in rows if not r["feasible"])
        print(f"\n[summary] {len(rows)} seeds  infeasible={n_bad} "
              f"({n_bad / len(rows):.1%})", flush=True)
        bad = [(r["seed"], r["q_max_feasible"]) for r in rows if not r["feasible"]]
        if bad:
            print(f"[summary] BLOCKED seeds (seed, q_max): {bad}", flush=True)
    else:
        print("[summary] no rows", flush=True)

    if args.out and rows:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"[wrote] {args.out}", flush=True)


if __name__ == "__main__":
    main()

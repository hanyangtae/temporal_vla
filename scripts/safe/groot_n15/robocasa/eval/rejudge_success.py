"""Rollout pkl을 open-loop replay 해 apple(PnP) 성공 판정을 임계값 grid로 재판정.

배경: PickPlaceCounterToStove._check_success() = check_contact(obj,container)
∧ ‖obj_xy−container_xy‖<0.07 ∧ gripper_obj_far(>0.25). pan 반경이 7cm보다 커서
가장자리 안착이 실패로 판정되는 오판정이 실증됨(ep43). 이 스크립트는 저장된
actions(record당 chunk 앞 n_action_steps row 실행)를 그대로 재생하며 매 env-step에서
중심거리 임계값만 완화한 판정(contact·gripper 조건 유지)을 병렬 계산한다.

robocasa 컨테이너에서 실행 (MUJOCO_GL=egl, PYTHONPATH에 robocasa/robosuite/repo).
같은 cell은 scenario_seed가 같아 scene이 동일 → env를 재사용하고 reset만 반복.

사용:
  python rejudge_success.py --rollout-dirs <dir1> <dir2> ... --out-tsv <path> \
      [--th-grid 0.07,0.09,0.10,0.12,0.15] [--limit N]
각 dir 은 task*--ep*--succ{0,1}.pkl 들을 담은 디렉토리 (재귀 탐색).
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


def find_kitchen_env(env):
    e = env
    while hasattr(e, "env"):
        if hasattr(e, "obj_body_id"):
            break
        e = e.env
    if not hasattr(e, "obj_body_id"):
        e = env.unwrapped
    return e


def episode_metrics(env, kenv, pkl, th_grid):
    """저장된 action 시퀀스를 재생하며 step별 술어를 누적."""
    import robocasa.utils.object_utils as OU

    n_as = int(pkl.get("n_action_steps", 5))
    obj = kenv.objects["obj"]
    cont = kenv.objects["container"]
    obj_bid = kenv.obj_body_id["obj"]
    cont_bid = kenv.obj_body_id["container"]
    eef_sid = kenv.robots[0].eef_site_id["right"]
    pan_radius = float(getattr(cont, "horizontal_radius", np.nan))

    ever = {th: False for th in th_grid}
    first_step = {th: -1 for th in th_grid}
    # placed(contact ∧ gripper_far) 순간의 최소 중심거리 / contact 없이 근접만 한 경우 추적
    min_dxy_placed = np.inf
    min_dxy_far = np.inf  # gripper_far ∧ (contact 무관) — contact이 blocker인지 판별용
    step_idx = 0
    for rec_action in pkl["actions"]:
        for i in range(n_as):
            action = {k: np.asarray(v[i]) for k, v in rec_action.items()}
            env.step(action)
            obj_pos = np.array(kenv.sim.data.body_xpos[obj_bid])
            cont_pos = np.array(kenv.sim.data.body_xpos[cont_bid])
            d_xy = float(np.linalg.norm(obj_pos[:2] - cont_pos[:2]))
            grip_d = float(
                np.linalg.norm(kenv.sim.data.site_xpos[eef_sid] - obj_pos)
            )
            gripper_far = grip_d > 0.25
            contact = bool(kenv.check_contact(obj, cont))
            if gripper_far:
                min_dxy_far = min(min_dxy_far, d_xy)
                if contact:
                    min_dxy_placed = min(min_dxy_placed, d_xy)
                    for th in th_grid:
                        if d_xy < th and not ever[th]:
                            ever[th] = True
                            first_step[th] = step_idx
            step_idx += 1
    return {
        "pan_radius": pan_radius,
        "ever": ever,
        "first_step": first_step,
        "min_dxy_placed": min_dxy_placed,
        "min_dxy_far": min_dxy_far,
        "steps": step_idx,
        "final_dxy": d_xy,
        "final_contact": contact,
        "final_grip_d": grip_d,
    }


def process_one(pkl_path: str, th_grid, out_tsv: str):
    """단일 에피소드 재판정 (반드시 fresh 프로세스에서 — 한 프로세스에서 env를 연속
    생성하면 두 번째부터 scene 재현이 오염됨을 검증으로 확인)."""
    import gymnasium as gym
    import robocasa  # noqa: F401
    import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401
    import robosuite  # noqa: F401

    p = Path(pkl_path)
    pkl = pickle.load(open(p, "rb"))
    env = gym.make(pkl["env_name"], enable_render=True, seed=pkl["scenario_seed"])
    env.reset(seed=pkl["scenario_seed"])
    kenv = find_kitchen_env(env)
    m = episode_metrics(env, kenv, pkl, th_grid)
    env.close()
    succ = int(pkl.get("episode_success", "succ1" in p.name))
    row = (
        [str(p.parent), p.stem.split("--")[1], succ, m["steps"], f"{m['pan_radius']:.4f}",
         f"{m['min_dxy_placed']:.4f}", f"{m['min_dxy_far']:.4f}",
         f"{m['final_dxy']:.4f}", int(m["final_contact"]), f"{m['final_grip_d']:.4f}"]
        + [int(m["ever"][th]) for th in th_grid]
        + [m["first_step"][th] for th in th_grid]
    )
    with open(out_tsv, "a") as f:
        f.write("\t".join(str(x) for x in row) + "\n")
    print(f"[rejudge] {p.name} succ_rec={succ} "
          + " ".join(f"th{th:g}={int(m['ever'][th])}" for th in th_grid), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rollout-dirs", nargs="+", default=None,
                    help="드라이버 모드: 각 dir의 pkl을 에피소드당 서브프로세스로 처리")
    ap.add_argument("--pkl", default=None, help="워커 모드: 단일 pkl 처리(내부용)")
    ap.add_argument("--out-tsv", required=True)
    ap.add_argument("--th-grid", default="0.07,0.09,0.10,0.12,0.15")
    ap.add_argument("--limit", type=int, default=0, help="dir당 최대 에피소드(0=전부)")
    args = ap.parse_args()
    th_grid = [float(x) for x in args.th_grid.split(",")]

    if args.pkl:
        return process_one(args.pkl, th_grid, args.out_tsv)

    import subprocess
    out = Path(args.out_tsv)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (
        ["dir", "ep", "succ_recorded", "steps", "pan_radius",
         "min_dxy_placed", "min_dxy_far", "final_dxy", "final_contact", "final_grip_d"]
        + [f"ever@{th:g}" for th in th_grid]
        + [f"first@{th:g}" for th in th_grid]
    )
    done = set()
    if out.exists():  # resume: 이미 처리한 (dir, ep)는 건너뜀
        for line in out.read_text().splitlines()[1:]:
            c = line.split("\t")
            if len(c) > 2:
                done.add((c[0], c[1]))
    else:
        out.write_text("\t".join(header) + "\n")

    for d in args.rollout_dirs:
        pkls = sorted(Path(d).rglob("task*--ep*--succ*.pkl"))
        if args.limit:
            pkls = pkls[: args.limit]
        for p in pkls:
            if (str(p.parent), p.stem.split("--")[1]) in done:
                continue
            r = subprocess.run(
                [sys.executable, __file__, "--pkl", str(p),
                 "--out-tsv", str(out), "--th-grid", args.th_grid],
                timeout=900,
            )
            if r.returncode != 0:
                print(f"[rejudge] WORKER FAILED rc={r.returncode}: {p}", flush=True)

    # 요약: dir(=arm)별 SR per threshold + fidelity
    print("\n=== summary ===")
    rows = []
    for line in out.read_text().splitlines()[1:]:
        c = line.split("\t")
        ever = {th: int(c[10 + i]) for i, th in enumerate(th_grid)}
        rows.append((c[0], int(c[2]), ever))
    for d in args.rollout_dirs:
        sub = [(s, e) for dd, s, e in rows if Path(dd).is_relative_to(d) or dd.startswith(str(d))]
        if not sub:
            continue
        n = len(sub)
        line = f"{d} n={n} SR_recorded={sum(s for s, _ in sub)/n:.3f}"
        for th in th_grid:
            line += f" SR@{th:g}={sum(e[th] for _, e in sub)/n:.3f}"
        succ1 = [e for s, e in sub if s == 1]
        if succ1:
            line += f" fidelity(succ1@{th_grid[0]:g})={sum(e[th_grid[0]] for e in succ1)/len(succ1):.3f}"
        print(line, flush=True)


if __name__ == "__main__":
    sys.exit(main())

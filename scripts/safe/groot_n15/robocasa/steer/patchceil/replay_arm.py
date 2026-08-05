"""patchceil direct action-replay arm (PROTOCOL §대조 4) — 모델 없이 sim replay.

target 의 저장 action 을 t0 record 까지 재생 → 그 뒤 donor 의 저장 action 을
donor_start record 부터 open-loop 재생 → donor 고갈 시점에서 종료·채점.
activation 이식의 양성이 "성공 정보" 인지 "그냥 donor 행동 재생" 인지 가르는 대조.

매핑(target↔donor, t0, donor_start)은 arm_plan.tsv 의 donor 행을 그대로 사용.
actions 는 patchceil_meta/<cell>/ep{N}_actions.npz (승준 원본 추출분).

robocasa 컨테이너 실행 (에피소드당 fresh 프로세스 — env_step_gt_retro.py §scene 오염):
  docker exec -e MUJOCO_GL=egl -e PYTHONPATH=... robocasa \
    python /temporal_vla/scripts/safe/groot_n15/robocasa/steer/patchceil/replay_arm.py [--cell C] [--limit N]
출력: patchceil/<cell>/rollouts/action_replay/results.tsv (append, resume-safe)
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO = Path("/temporal_vla") if Path("/temporal_vla").is_dir() else Path(__file__).resolve().parents[6]
GROOT = REPO / "outputs/eval/robocasa/groot_n15/patchceil"
CELLS = ["ppcc_bread_s300033", "ppcc_bread_s400020"]
ENV_NAME = "robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env"
SEEDS = {"ppcc_bread_s300033": 300033, "ppcc_bread_s400020": 400020}
NAS = 5
NPZ2KEY = {
    "a_end_effector_position": "action.end_effector_position",
    "a_end_effector_rotation": "action.end_effector_rotation",
    "a_gripper_close": "action.gripper_close",
    "a_base_motion": "action.base_motion",
    "a_control_mode": "action.control_mode",
}


def replay_one(cell: str, target_ep: int, donor_ep: int, t0: int, donor_start: int) -> dict:
    """단일 replay (fresh 프로세스에서 호출됨)."""
    import numpy as np
    import gymnasium as gym
    import robocasa  # noqa: F401
    import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401
    import robosuite  # noqa: F401

    sys.path.insert(0, str(REPO))  # src.collect.* import 용
    from src.collect.robocasa.event_labeler import find_robocasa_env

    def load_acts(ep: int) -> dict:
        return dict(np.load(GROOT / "patchceil_meta" / cell / f"ep{ep}_actions.npz"))

    tgt, don = load_acts(target_ep), load_acts(donor_ep)
    env = gym.make(ENV_NAME, enable_render=True, seed=SEEDS[cell])
    env.reset(seed=SEEDS[cell])
    kenv = find_robocasa_env(env)
    succ_any, first_succ, step_i = False, None, 0

    def run_records(acts: dict, r_from: int, r_to: int) -> None:
        nonlocal succ_any, first_succ, step_i
        r_total = next(iter(acts.values())).shape[0]
        for r in range(r_from, min(r_to, r_total)):
            for i in range(NAS):
                a = {NPZ2KEY[k]: np.asarray(v[r, i]) for k, v in acts.items()}
                env.step(a)
                step_i += 1
                try:
                    if bool(kenv._check_success()):
                        succ_any = True
                        if first_succ is None:
                            first_succ = step_i
                except Exception:
                    pass

    run_records(tgt, 0, t0)                    # target 구간 재생 (실패 경로 진입)
    run_records(don, donor_start, 10 ** 9)     # donor 구간 open-loop, 고갈 시 종료
    env.close()
    return {
        "cell": cell, "target_ep": target_ep, "donor_ep": donor_ep, "t0": t0,
        "donor_start": donor_start, "steps": step_i,
        "success": int(succ_any), "first_success_step": first_succ,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--one", default=None, help="내부용: cell:target:donor:t0:ds 단일 실행")
    args = ap.parse_args()

    if args.one:  # fresh subprocess 진입점
        c, t, d, t0, ds = args.one.split(":")
        print("RESULT " + json.dumps(replay_one(c, int(t), int(d), int(t0), int(ds))))
        return 0

    cells = [args.cell] if args.cell else CELLS
    for cell in cells:
        rows = [r for r in csv.DictReader(open(GROOT / cell / "arm_plan.tsv"), delimiter="\t")
                if r["arm"] == "donor"]
        if args.limit:
            rows = rows[: args.limit]
        out = GROOT / cell / "rollouts/action_replay/results.tsv"
        out.parent.mkdir(parents=True, exist_ok=True)
        done = set()
        if out.exists():
            done = {int(r["target_ep"]) for r in csv.DictReader(open(out), delimiter="\t")}
        else:
            out.write_text("cell\ttarget_ep\tdonor_ep\tt0\tdonor_start\tsteps\tsuccess\tfirst_success_step\n")
        for r in rows:
            t = int(r["target_ep"])
            if t in done:
                continue
            spec = f"{cell}:{t}:{r['donor_ep']}:{r['t0']}:{r['donor_start']}"
            p = subprocess.run(
                [sys.executable, __file__, "--one", spec],
                capture_output=True, text=True, timeout=900,
            )
            line = next((ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")), None)
            if line is None:
                print(f"[replay] {spec} FAIL rc={p.returncode} err={p.stderr[-200:]}", flush=True)
                continue
            d = json.loads(line[len("RESULT "):])
            with open(out, "a") as f:
                f.write("\t".join(str(d[k]) if d[k] is not None else "" for k in
                        ["cell", "target_ep", "donor_ep", "t0", "donor_start",
                         "steps", "success", "first_success_step"]) + "\n")
            print(f"[replay] {cell} ep{t} donor{d['donor_ep']} succ={d['success']} steps={d['steps']}", flush=True)
    print("REPLAY_ARM_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

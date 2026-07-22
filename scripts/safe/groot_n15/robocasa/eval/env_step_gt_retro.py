"""기존 rollout pkl에 env-step 해상도 phase/성공 GT를 소급 생성 (replay 기반, sidecar 저장).

배경: env-step GT 수집 배선(env_step_phase.py)은 배선 이후 수집분에만 적용됨.
이 스크립트는 저장된 actions 를 open-loop replay 하며 매 env-step 에서
proximity 라벨러(step)와 kitchen _check_success 를 평가해 pkl 옆에
`<stem>.envstep.json` sidecar 로 저장한다 (pkl 무수정 — 아카이브 무손상).

⚠️ replay 함정 (docs/steering/18 §2 실증): 한 프로세스에서 gym.make 연속 생성 시
scene 오염 → 에피소드당 fresh 프로세스 (rejudge_success.py 와 동일 구조).

sidecar 필드: env_step_phases[N+1](k=0=reset 직후 s_0), env_step_success[N],
env_step_event_steps/grasp/drop/wrong_grasp(env-step 인덱스),
qa_record_mismatch(feature_phases[r] vs env_step_phases[nas*r] 불일치 수),
fidelity_ok(기록 성공여부 == replay any(success)).

사용 (robocasa 컨테이너, 데이터 루트 마운트 필요):
  python env_step_gt_retro.py --rollout-dirs <dir...> [--limit N]
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "scripts/safe/groot_n16/robocasa/collect"))
sys.path.insert(0, str(REPO / "scripts/safe/groot_n16/libero"))


def process_one(pkl_path: str) -> None:
    import numpy as np
    import gymnasium as gym
    import robocasa  # noqa: F401
    import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401
    import robosuite  # noqa: F401
    from robocasa_event_labeler import find_robocasa_env, make_robocasa_event_labeler

    p = Path(pkl_path)
    out = p.with_suffix(".envstep.json")
    d = pickle.load(open(p, "rb"))
    nas = int(d.get("n_action_steps", 5))
    env = gym.make(d["env_name"], enable_render=True, seed=d["scenario_seed"])
    env.reset(seed=d["scenario_seed"])
    kenv = find_robocasa_env(env)
    lab = make_robocasa_event_labeler(env, d["env_name"], proximity_phases=True)
    lab.reset()
    phases = [lab.step()]  # s_0
    succ_steps = []
    for rec_action in d["actions"]:
        for i in range(nas):
            a = {k: np.asarray(v[i]) for k, v in rec_action.items()}
            env.step(a)
            phases.append(lab.step())
            try:
                succ_steps.append(bool(kenv._check_success()))
            except Exception:
                succ_steps.append(False)
    env.close()

    fp = list(d.get("feature_phases") or [])
    qa = sum(
        1 for r, ph in enumerate(fp)
        if r * nas < len(phases) and phases[r * nas] != ph
    )
    rec_succ = bool(d.get("episode_success", "succ1" in p.name))
    side = {
        "env_step_phases": phases,
        "env_step_success": succ_steps,
        "env_step_event_steps": dict(getattr(lab, "event_steps", {}) or {}),
        "env_step_grasp_steps": list(getattr(lab, "grasp_steps", []) or []),
        "env_step_drop_steps": list(getattr(lab, "drop_steps", []) or []),
        "env_step_wrong_grasp_steps": list(getattr(lab, "wrong_grasp_steps", []) or []),
        "n_action_steps": nas,
        "qa_record_mismatch": qa,
        "recorded_success": rec_succ,
        "replay_success": bool(any(succ_steps)),
        "fidelity_ok": rec_succ == bool(any(succ_steps)),
        "source_pkl": p.name,
        "labeler": "proximity_7p(retro)",
    }
    out.write_text(json.dumps(side))
    print(f"[retro] {p.name} steps={len(succ_steps)} qa={qa} "
          f"fid={'ok' if side['fidelity_ok'] else 'MISMATCH'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rollout-dirs", nargs="+", default=None)
    ap.add_argument("--pkl", default=None, help="워커 모드(내부용)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.pkl:
        return process_one(args.pkl)

    n_done = n_run = n_fail = 0
    for droot in args.rollout_dirs:
        pkls = sorted(Path(droot).rglob("task*--ep*--succ*.pkl"))
        if args.limit:
            pkls = pkls[: args.limit]
        for p in pkls:
            if p.with_suffix(".envstep.json").exists():  # resume
                n_done += 1
                continue
            r = subprocess.run(
                [sys.executable, __file__, "--pkl", str(p)], timeout=900
            )
            if r.returncode != 0:
                print(f"[retro] WORKER FAILED: {p}", flush=True)
                n_fail += 1
            else:
                n_run += 1
    print(f"[retro done] ran={n_run} skipped={n_done} failed={n_fail}", flush=True)

    # 요약: fidelity/QA 집계
    tot = fid = 0
    qa_hist = []
    for droot in args.rollout_dirs:
        for j in sorted(Path(droot).rglob("*.envstep.json")):
            s = json.loads(j.read_text())
            tot += 1
            fid += int(s["fidelity_ok"])
            qa_hist.append(s["qa_record_mismatch"])
    if tot:
        import statistics
        print(f"[summary] sidecars={tot} fidelity={fid}/{tot}={fid/tot:.3f} "
              f"qa_mismatch mean={statistics.mean(qa_hist):.2f} max={max(qa_hist)}", flush=True)


if __name__ == "__main__":
    main()

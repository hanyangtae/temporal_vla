"""기존 rollout pkl 트리를 open-loop replay 해 env-step 해상도 phase/성공 GT 를
소급 재구성하고, 원본을 그대로 복제한 **증강 pkl**로 출력 트리에 저장한다.

env_step_gt_retro.py 의 배치판:
- retro = pkl 옆 sidecar json. batch = 원본 dict + env_step_* 필드 병합한 새 pkl 을
  out-root 아래 입력 트리와 동일 구조로 저장(입력 무손상).
- 에피소드당 **fresh 프로세스** (한 프로세스에서 gym.make 연속 생성 시 scene 오염 —
  rejudge_success.py/docs18 §2 실증). 드라이버가 워커 프로세스를 풀로 병렬 실행.

증강 필드: env_step_phases[N+1](k=0=reset 직후 s_0), env_step_success[N],
env_step_event_steps/grasp/drop/wrong_grasp(env-step 인덱스), env_step_n_action_steps,
qa_record_mismatch(feature_phases[r] vs env_step_phases[nas*r] 불일치 수),
env_step_recorded_success / env_step_replay_success / env_step_fidelity_ok.

사용 (robocasa 컨테이너, MUJOCO_GL=egl, PYTHONPATH 세팅):
  python env_step_gt_batch.py --in-root <raw_rollouts> --out-root <step_gt_rollouts> \
      [--workers 8] [--limit N]
  python env_step_gt_batch.py --pkl <one.pkl> --in-root <r> --out-root <o>   # 워커(내부용)
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))  # src.collect.* import 용

_SEED_SUFFIX = re.compile(r"_s\d+$")


def _seeded_rel(rel: Path, scenario_seed: int) -> Path:
    """출력 rel 경로의 cell 폴더명을 _s<seed>로 통일 (이미 seed 표기면 그대로).

    rel = <task>/<cell>/<file.pkl>. seedless cell(ppcc_bread, ppcs_apple 등)만
    _s{scenario_seed} 를 붙여 seeded cell 과 명명 규칙을 맞춘다.
    """
    parts = list(rel.parts)
    cell = parts[-2]
    if not _SEED_SUFFIX.search(cell):
        parts[-2] = f"{cell}_s{int(scenario_seed)}"
    return Path(*parts)


def process_one(pkl_path: str, in_root: str, out_root: str) -> None:
    import numpy as np
    import gymnasium as gym
    import robocasa  # noqa: F401
    import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401
    import robosuite  # noqa: F401
    from src.collect.robocasa.event_labeler import find_robocasa_env, make_robocasa_event_labeler

    p = Path(pkl_path)
    d = pickle.load(open(p, "rb"))
    nas = int(d.get("n_action_steps", 5))
    rel = _seeded_rel(p.relative_to(in_root), d["scenario_seed"])
    out = Path(out_root) / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    env = gym.make(d["env_name"], enable_render=True, seed=d["scenario_seed"])
    env.reset(seed=d["scenario_seed"])
    kenv = find_robocasa_env(env)
    lab = make_robocasa_event_labeler(env, d["env_name"], proximity_phases=True)
    lab.reset()
    phases = [lab.step()]  # s_0 (reset 직후)
    succ_steps: list[bool] = []
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
    replay_succ = bool(any(succ_steps))

    d["env_step_phases"] = phases
    d["env_step_success"] = succ_steps
    d["env_step_event_steps"] = dict(getattr(lab, "event_steps", {}) or {})
    d["env_step_grasp_steps"] = list(getattr(lab, "grasp_steps", []) or [])
    d["env_step_drop_steps"] = list(getattr(lab, "drop_steps", []) or [])
    d["env_step_wrong_grasp_steps"] = list(getattr(lab, "wrong_grasp_steps", []) or [])
    d["env_step_n_action_steps"] = nas
    d["qa_record_mismatch"] = int(qa)
    d["env_step_recorded_success"] = rec_succ
    d["env_step_replay_success"] = replay_succ
    d["env_step_fidelity_ok"] = bool(rec_succ == replay_succ)
    d["env_step_labeler"] = "proximity_7p(retro-batch)"

    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(d, f)
    os.replace(tmp, out)  # atomic: 부분쓰기가 완성본으로 보이지 않게
    print(
        f"[batch] {rel} steps={len(succ_steps)} qa={qa} "
        f"fid={'ok' if d['env_step_fidelity_ok'] else 'MISMATCH'}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--pkl", default=None, help="워커 모드(내부용): 단일 pkl 처리")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="처리 최대 개수(0=전부, smoke용)")
    args = ap.parse_args()

    if args.pkl:
        return process_one(args.pkl, args.in_root, args.out_root)

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    pkls = sorted(in_root.rglob("task*--ep*--succ*.pkl"))

    # cell(부모 dir) -> scenario_seed: 대표 pkl 1개만 로드해 seed 통일 출력경로 계산.
    cell_seed: dict[Path, int] = {}
    for p in pkls:
        if p.parent not in cell_seed:
            cell_seed[p.parent] = int(pickle.load(open(p, "rb"))["scenario_seed"])

    def out_path(p: Path) -> Path:
        return out_root / _seeded_rel(p.relative_to(in_root), cell_seed[p.parent])

    # resume: 이미 완성된 출력은 건너뜀
    todo = [p for p in pkls if not out_path(p).exists()]
    if args.limit:
        todo = todo[: args.limit]
    n = len(todo)
    print(f"[batch] total={len(pkls)} todo={n} workers={args.workers}", flush=True)

    def launch(p: Path) -> int:
        # 반드시 fresh 프로세스 (결정성) — subprocess.run 이 매번 새 인터프리터.
        r = subprocess.run(
            [sys.executable, __file__, "--pkl", str(p),
             "--in-root", str(in_root), "--out-root", str(out_root)],
            timeout=1200,
        )
        return r.returncode

    done = fail = 0
    mismatches: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(launch, p): p for p in todo}
        for fut in as_completed(futs):
            pk = futs[fut]
            done += 1
            try:
                rc = fut.result()
            except Exception as e:  # noqa: BLE001
                rc = -99
                print(f"[batch] WORKER EXC {pk}: {e}", flush=True)
            if rc != 0:
                fail += 1
                print(f"[batch] FAIL rc={rc}: {pk}", flush=True)
            if done % 10 == 0 or done == n:
                print(f"[batch] progress {done}/{n} fail={fail}", flush=True)

    # QA 요약: fidelity/mismatch 집계 (출력 pkl 재스캔)
    print("\n=== QA summary ===", flush=True)
    n_out = fid_bad = qa_pos = 0
    for p in pkls:
        op = out_path(p)
        if not op.exists():
            continue
        try:
            dd = pickle.load(open(op, "rb"))
        except Exception:
            continue
        n_out += 1
        if not dd.get("env_step_fidelity_ok", True):
            fid_bad += 1
            mismatches.append(str(p.relative_to(in_root)))
        if int(dd.get("qa_record_mismatch", 0)) > 0:
            qa_pos += 1
    print(f"[batch] wrote={n_out} fidelity_MISMATCH={fid_bad} qa_mismatch>0={qa_pos}", flush=True)
    for m in mismatches[:40]:
        print(f"  MISMATCH: {m}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

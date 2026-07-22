"""patchceil A2/A3 anchor 판정 (PROTOCOL §Anchor) — lerobot 컨테이너 실행.

A2 (cross-scene action-equivalence): 상대 cell env + donor inference_seed + 전창 L15
patch rollout 의 emitted actions[0:R_donor] 가 donor 저장 actions 와 일치해야 한다.
L15 대입이 upstream(장면·관측)을 완전히 지배함의 배선 증명.

A3 (sham): 자기 activation 전창 이식 rollout == baseline (actions·succ 완전 일치).

판정: max|Δ| == 0 → PASS, < 1e-3 → WARN_PASS(fp 비결정 소음, 기록), 그 외 FAIL.
"""
from __future__ import annotations

import csv
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path("/temporal_vla") if Path("/temporal_vla").is_dir() else Path(__file__).resolve().parents[6]
GROOT = REPO / "outputs/eval/robocasa/groot_n15/patchceil"
TASK = "PickPlaceCounterToCabinet"
CELLS = ["ppcc_bread_s300033", "ppcc_bread_s400020"]
KEYMAP = {
    "action.end_effector_position": "a_end_effector_position",
    "action.end_effector_rotation": "a_end_effector_rotation",
    "action.gripper_close": "a_gripper_close",
    "action.base_motion": "a_base_motion",
    "action.control_mode": "a_control_mode",
}


def actions_diff(pkl_path: Path, ref_npz: Path, n_limit: int | None = None) -> tuple[int, float]:
    d = pickle.load(open(pkl_path, "rb"))
    ref = np.load(ref_npz)
    acts = d["actions"]
    n_ref = ref[next(iter(KEYMAP.values()))].shape[0]
    n = min(len(acts), n_ref if n_limit is None else min(n_ref, n_limit))
    max_diff = 0.0
    for pk, nk in KEYMAP.items():
        new = np.stack([np.asarray(a[pk], dtype=np.float32) for a in acts[:n]], axis=0)
        max_diff = max(max_diff, float(np.abs(new - ref[nk][:n]).max()))
    return n, max_diff


def verdict(md: float) -> str:
    return "PASS" if md == 0.0 else ("WARN_PASS" if md < 1e-3 else "FAIL")


def main() -> int:
    import torch  # noqa: F401

    ok = True
    for plan_cell in CELLS:
        for r in csv.DictReader(open(GROOT / plan_cell / "arm_plan.tsv"), delimiter="\t"):
            arm = r["arm"]
            if not arm.startswith("anchor_"):
                continue
            env_cell, t_ep, d_ep = r["cell"], int(r["target_ep"]), int(r["donor_ep"])
            rdir = GROOT / plan_cell / "rollouts" / arm / "raw_rollouts" / TASK / env_cell
            hits = sorted(rdir.glob(f"task5--ep{t_ep}--succ*.pkl"))
            if not hits:
                print(f"[{plan_cell}] {arm} ep{t_ep}: MISSING")
                ok = False
                continue
            if arm == "anchor_a2":
                # donor 는 plan_cell 소속 (env 는 상대 cell)
                ref = GROOT / "patchceil_meta" / plan_cell / f"ep{d_ep}_actions.npz"
                n, md = actions_diff(hits[-1], ref)
                v = verdict(md)
                print(f"[A2 {r['tag']}] n={n} max|Δ|={md:.3e} → {v}")
            else:  # anchor_a3_sham — baseline 은 자기 자신
                ref = GROOT / "patchceil_meta" / plan_cell / f"ep{t_ep}_actions.npz"
                n, md = actions_diff(hits[-1], ref)
                succ = 1 if hits[-1].name.endswith("succ1.pkl") else 0
                v = verdict(md) if succ == 0 else "FAIL(succ!=0)"
                print(f"[A3 {r['tag']}] n={n} max|Δ|={md:.3e} succ={succ} → {v}")
            if v.startswith("FAIL"):
                ok = False
    print("ANCHORS_PASS" if ok else "ANCHORS_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""patchceil A1 결정론 게이트 — pass B 재수집분 vs 승준 원본(추출 actions npz) 대조.

PROTOCOL.md §Anchor A1: passB_manifest 16판 각각에 대해
  ① succ 플래그 = targets_fit.tsv (승준 원본 파일명) 와 일치
  ② pkl "actions" 전 record 전 key 가 patchceil_meta/ep{N}_actions.npz 와 수치 일치
    (기대 = exact; 불일치 시 max|Δ| 보고 — GPU 개체 차이 진단용)

pass B pkl 이 torch 텐서 포함 → lerobot 컨테이너에서 실행:
  docker exec lerobot python /temporal_vla/scripts/safe/groot_n15/robocasa/steer/patchceil/check_passB_determinism.py
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
KEYMAP = {  # pkl action key → 추출 npz key
    "action.end_effector_position": "a_end_effector_position",
    "action.end_effector_rotation": "a_end_effector_rotation",
    "action.gripper_close": "a_gripper_close",
    "action.base_motion": "a_base_motion",
    "action.control_mode": "a_control_mode",
}


def main() -> int:
    import torch  # noqa: F401 — unpickle 용

    all_ok = True
    for cell in CELLS:
        manifest = GROOT / cell / "passB_manifest.tsv"
        expect_succ = {}
        for r in csv.DictReader(open(GROOT / cell / "targets_fit.tsv"), delimiter="\t"):
            expect_succ[int(r["episode_idx"])] = int(r["succ"])
        rdir = GROOT / cell / "passB/raw_rollouts" / TASK / cell
        for r in csv.DictReader(open(manifest), delimiter="\t"):
            ep = int(r["episode_idx"])
            hits = sorted(rdir.glob(f"task5--ep{ep}--succ*.pkl"))
            if not hits:
                print(f"[{cell}] ep{ep}: MISSING")
                all_ok = False
                continue
            got_succ = 1 if hits[-1].name.endswith("succ1.pkl") else 0
            succ_ok = got_succ == expect_succ[ep]
            d = pickle.load(open(hits[-1], "rb"))
            ref = np.load(GROOT / "patchceil_meta" / cell / f"ep{ep}_actions.npz")
            acts = d["actions"]
            n_ok = len(acts) == ref[next(iter(KEYMAP.values()))].shape[0]
            max_diff = 0.0
            if n_ok:
                for pk, nk in KEYMAP.items():
                    new = np.stack([np.asarray(a[pk], dtype=np.float32) for a in acts], axis=0)
                    max_diff = max(max_diff, float(np.abs(new - ref[nk]).max()))
            status = "OK" if (succ_ok and n_ok and max_diff == 0.0) else "FAIL"
            if status == "FAIL":
                all_ok = False
            print(
                f"[{cell}] ep{ep}: succ {got_succ}(기대 {expect_succ[ep]}) "
                f"records {len(acts)}{'==' if n_ok else '!='}원본 "
                f"max|Δaction|={max_diff:.3e} → {status}"
            )
    print("A1_PASS" if all_ok else "A1_FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

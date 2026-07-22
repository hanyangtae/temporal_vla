"""exp4-2 B1 — VL donor NPZ 추출 (post_vl_sa_full 캡처 pkl → serve /patch_arm 용).

입력 pkl: serve ``--collect --capture-vl --groot-vl-capture-point post_vl_sa_full`` 수집분.
``vl_hidden_states`` = record 별 [T_vl, D] 텐서 리스트, ``vl_feature_kind`` =
``groot_n15_vl_post_sa_full_tokens``. 출력 NPZ (patching_hooks.load_vl_donor_npz 규약):
  - ``VL``: fp16 [R, T_vl, D]
  - ``meta_json``: uint8 — cell/instruction/seed/n_records/t_vl/feature_phases 등.

전 record 의 T_vl 동일을 강제한다 (ragged donor 금지 — 불일치 시 ABORT).
pkl 이 torch 텐서를 담아 **lerobot 컨테이너에서 실행**:
  docker exec lerobot python /temporal_vla/.claude/worktrees/exp4-2-induced-failures/scripts/safe/groot_n15/robocasa/steer/induced/extract_vl_donor_npz.py \
    --pkl <rollout.pkl> --out <donor_vl.npz> [--allow-fail]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

EXPECTED_VL_KIND = "groot_n15_vl_post_sa_full_tokens"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--allow-fail", action="store_true",
        help="실패 episode 허용 (대조 추출용). 기본은 succ1 강제.",
    )
    args = ap.parse_args()

    import torch  # noqa: F401 — pkl unpickle 에 필요 (컨테이너 전용)

    d = pickle.load(open(args.pkl, "rb"))
    succ = int(d.get("episode_success", -1))
    if succ != 1 and not args.allow_fail:
        print(f"ABORT: episode_success={succ} — donor 는 성공 episode 여야 한다 "
              "(대조 추출은 --allow-fail)", file=sys.stderr)
        return 2

    kind = d.get("vl_feature_kind")
    if kind != EXPECTED_VL_KIND:
        print(f"ABORT: vl_feature_kind={kind!r} != {EXPECTED_VL_KIND!r} "
              "(serve 를 --groot-vl-capture-point post_vl_sa_full 로 띄웠는지 확인)",
              file=sys.stderr)
        return 2

    vls = d.get("vl_hidden_states")
    if not vls:
        print("ABORT: vl_hidden_states 비어 있음", file=sys.stderr)
        return 2
    recs = []
    for i, h in enumerate(vls):
        arr = h.numpy() if hasattr(h, "numpy") else np.asarray(h)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            print(f"ABORT: record {i} shape={arr.shape} — [T_vl,D] 아님", file=sys.stderr)
            return 2
        recs.append(arr.astype(np.float16))
    t_set = {r.shape[0] for r in recs}
    if len(t_set) != 1:
        print(f"ABORT: record 간 T_vl 불일치 {sorted(t_set)} — ragged donor 금지",
              file=sys.stderr)
        return 2
    stack = np.stack(recs, axis=0)  # [R, T_vl, D]

    meta = {
        "cell": d.get("cell_id"),
        "instruction": d.get("task_description"),
        "episode_idx": d.get("episode_idx"),
        "scenario_seed": d.get("scenario_seed"),
        "inference_seed": d.get("inference_seed"),
        "episode_success": succ,
        "n_records": int(stack.shape[0]),
        "t_vl": int(stack.shape[1]),
        "vl_dim": int(stack.shape[2]),
        "vl_feature_kind": kind,
        "feature_phases": list(d.get("feature_phases") or []),
        "source_pkl": str(Path(args.pkl).name),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        meta_json=np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
        VL=stack,
    )
    print(f"wrote {out} [R,T_vl,D]={stack.shape} instruction={meta['instruction']!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

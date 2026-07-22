"""patchceil donor NPZ 추출 — pass B full-token pkl → serve /patch_arm 용 donor NPZ.

입력 pkl (collect_passB, all_token_full): ``hidden_states`` = record 별 [L,K,T,D] fp16
torch 텐서 리스트 (L = --cap 순서). 출력 NPZ (patching_hooks.load_donor_npz 규약):
  - ``L{layer}``: fp16 [R,K,T,D]
  - ``meta_json``: uint8 — cell/episode_idx/scenario_seed/inference_seed/n_records/
    feature_phases(donor phase-start 정렬용)/cap/token 모드.

pkl 이 torch 텐서를 담아 **lerobot 컨테이너에서 실행**:
  docker exec lerobot python /temporal_vla/scripts/safe/groot_n15/robocasa/steer/patchceil/extract_donor_npz.py \
    --pkl <rollout.pkl> --out <donor.npz> --layers 15 [--cap 0,2,4,8,10,12,15] [--allow-fail]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", required=True, help="추출 layer (콤마, 예: '15' 또는 '0,2,4')")
    ap.add_argument(
        "--cap", default="0,2,4,8,10,12,15",
        help="수집 시 --groot-dit-capture-layers (pkl L축 순서 정의)",
    )
    ap.add_argument(
        "--allow-fail", action="store_true",
        help="실패 episode 허용 (placebo-fail donor 추출용). 기본은 succ1 강제.",
    )
    args = ap.parse_args()

    import torch  # noqa: F401 — pkl unpickle 에 필요 (컨테이너 전용)

    d = pickle.load(open(args.pkl, "rb"))
    succ = int(d.get("episode_success", -1))
    if succ != 1 and not args.allow_fail:
        print(f"ABORT: episode_success={succ} — donor 는 성공 episode 여야 한다 "
              "(placebo 추출은 --allow-fail)", file=sys.stderr)
        return 2

    cap = [int(x) for x in args.cap.split(",") if x.strip() != ""]
    layers = [int(x) for x in args.layers.split(",") if x.strip() != ""]
    hs = d["hidden_states"]
    if not hs:
        print("ABORT: hidden_states 비어 있음", file=sys.stderr)
        return 2
    recs = []
    for i, h in enumerate(hs):
        arr = h.numpy() if hasattr(h, "numpy") else np.asarray(h)
        if arr.ndim != 4:
            print(f"ABORT: record {i} shape={arr.shape} — full-token [L,K,T,D] 아님 "
                  "(수집이 all_token_full 인지 확인)", file=sys.stderr)
            return 2
        if arr.shape[0] != len(cap):
            print(f"ABORT: record {i} L={arr.shape[0]} != len(cap)={len(cap)}", file=sys.stderr)
            return 2
        recs.append(arr.astype(np.float16))
    stack = np.stack(recs, axis=0)  # [R,L,K,T,D]

    out_arrays = {}
    for layer in layers:
        if layer not in cap:
            print(f"ABORT: layer {layer} 는 cap {cap} 에 없음", file=sys.stderr)
            return 2
        out_arrays[f"L{layer}"] = stack[:, cap.index(layer)]  # [R,K,T,D]

    meta = {
        "cell": d.get("cell_id"),
        "episode_idx": d.get("episode_idx"),
        "scenario_seed": d.get("scenario_seed"),
        "inference_seed": d.get("inference_seed"),
        "episode_success": succ,
        "n_records": int(stack.shape[0]),
        "cap": cap,
        "layers": layers,
        "shape_rktd": [int(stack.shape[0])] + [int(x) for x in stack.shape[2:]],
        "feature_phases": list(d.get("feature_phases") or []),
        "source_pkl": str(Path(args.pkl).name),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        meta_json=np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
        **out_arrays,
    )
    kt = next(iter(out_arrays.values())).shape
    print(f"wrote {out} layers={layers} [R,K,T,D]={kt} phases={len(meta['feature_phases'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

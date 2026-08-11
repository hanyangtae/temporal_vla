"""exp4-2 B4 — donor 통계 매칭 gaussian noise NPZ (dose-matched 무구조 대조).

입력: DiT donor NPZ (키 ``L{layer}`` [R,K,T,D] — extract_donor_npz.py 산출).
각 (r,k)에 대해 donor[r,k] ([T,D])의 scalar μ,σ 를 산출해
``noise[r,k] = rng.normal(μ, scale·σ)`` 로 동형 배열을 생성 (fp16, 고정 seed).
출력 NPZ 는 같은 키 구조라 serve `/patch_arm` 이 그대로 소비한다.

  python make_noise_npz.py --src <donor_L15.npz> --scale 1.0 --seed 20260722 --out <noise.npz>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="원본 donor NPZ (통계 매칭 대상)")
    ap.add_argument("--scale", type=float, required=True, help="σ 배수 (그리드 {0.5,1,2})")
    ap.add_argument("--seed", type=int, default=20260722)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src = Path(args.src)
    src_sha12 = hashlib.sha256(src.read_bytes()).hexdigest()[:12]
    rng = np.random.default_rng(args.seed)
    out_arrays: dict[str, np.ndarray] = {}
    with np.load(src, allow_pickle=False) as z:
        src_meta = (
            json.loads(bytes(z["meta_json"]).decode("utf-8")) if "meta_json" in z else {}
        )
        for key in sorted(z.keys()):
            if not key.startswith("L"):
                continue
            arr = z[key].astype(np.float32)  # [R,K,T,D]
            if arr.ndim != 4:
                print(f"ABORT: {key} shape={arr.shape} — [R,K,T,D] 아님", file=sys.stderr)
                return 2
            mu = arr.mean(axis=(2, 3), keepdims=True)   # [R,K,1,1]
            sd = arr.std(axis=(2, 3), keepdims=True)    # [R,K,1,1]
            noise = rng.normal(loc=mu, scale=args.scale * sd, size=arr.shape)
            out_arrays[key] = noise.astype(np.float16)
    if not out_arrays:
        print("ABORT: src 에 L{layer} 키 없음", file=sys.stderr)
        return 2

    meta = {
        "noise_kind": "gauss_rk_scalar_stats",
        "source_npz": str(src.name),
        "source_sha12": src_sha12,
        "scale": args.scale,
        "seed": args.seed,
        # donor phase 정렬 정보는 원본에서 승계 (start_record 산출용).
        "cell": src_meta.get("cell"),
        "episode_idx": src_meta.get("episode_idx"),
        "scenario_seed": src_meta.get("scenario_seed"),
        "inference_seed": src_meta.get("inference_seed"),
        "n_records": src_meta.get("n_records"),
        "feature_phases": src_meta.get("feature_phases", []),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        meta_json=np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
        **out_arrays,
    )
    shapes = {k: list(v.shape) for k, v in out_arrays.items()}
    print(f"wrote {out} scale={args.scale} seed={args.seed} shapes={shapes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

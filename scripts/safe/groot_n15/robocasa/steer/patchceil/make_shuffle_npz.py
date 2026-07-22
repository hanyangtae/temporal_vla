"""patchceil donor-shuffle NPZ 생성 — record 축 고정 permutation (PROTOCOL §대조 3).

seed=20260716 고정, K/T/D 축 유지 (denoise 정렬은 보존, 시간 순서만 파괴).
lerobot 컨테이너 실행: docker exec lerobot python .../make_shuffle_npz.py <in.npz> <out.npz>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SEED = 20260716


def main() -> int:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    with np.load(src, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != "meta_json"}
        meta = json.loads(bytes(z["meta_json"]).decode("utf-8")) if "meta_json" in z else {}
    rng = np.random.default_rng(SEED)
    r = next(iter(arrays.values())).shape[0]
    perm = rng.permutation(r)
    out = {k: v[perm] for k, v in arrays.items()}
    meta["shuffle_seed"] = SEED
    meta["shuffle_perm_head"] = [int(x) for x in perm[:10]]
    np.savez(dst, meta_json=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8), **out)
    print(f"wrote {dst} (perm of {r} records, seed {SEED})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

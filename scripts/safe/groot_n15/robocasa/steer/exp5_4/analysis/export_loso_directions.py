#!/usr/bin/env python3
"""exp5-4 — scene-LOSO 실패축 방향 NPZ 내보내기 (Phase A 채점용 동결 벡터).

각 scene S 에 대해 S 를 제외한 나머지 혼재 scene 각각 mean(fail)−mean(succ) → 평균 →
L2 정규화 (`selection_upper_bound.py` 의 dir_from 과 동일 계약). record 0 (W=1) 고정.

NPZ 키: dir_<scene> ([1536] float32) · layer([1]) · window([1]) ·
        fit_source(문자열) · scenes(목록) · n_ep_per_scene
`make_selection_manifest.py --direction-npz` 의 scene 별 `dir_<scene>` 인터페이스와 맞춘다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _sel_common import load, to_matrix  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", default="/home/kimseungjun/sm_npz")
    ap.add_argument("--cell", default="pq3_drawer_right")
    ap.add_argument("--layers", default="0,12")
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--out-dir", default="/home/kimseungjun/exp54_results")
    ap.add_argument("--fit-source", default="pq3_drawer_right seed0-7e6 LOSO")
    a = ap.parse_args()

    eps, layers = load(Path(a.npz_dir), a.cell)
    if not eps:
        raise SystemExit(f"npz 없음: {a.npz_dir}/{a.cell}_sh*.npz")
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)

    for L in [int(x) for x in a.layers.split(",")]:
        li = layers.index(L)
        A, Y, scenes, seeds, _E = to_matrix(eps, li, W=a.window)
        S, J, D = A.shape
        ns, nf = (Y == 1).sum(1), (Y == 0).sum(1)
        mixed = (ns > 0) & (nf > 0)
        mu_s = np.einsum("sj,sjd->sd", (Y == 1).astype(float), A) / np.maximum(ns, 1)[:, None]
        mu_f = np.einsum("sj,sjd->sd", (Y == 0).astype(float), A) / np.maximum(nf, 1)[:, None]
        Dm = (mu_f - mu_s) * mixed[:, None]              # 실패−성공 (클수록 실패)
        tot, cnt = Dm.sum(0), int(mixed.sum())

        payload = dict(layer=np.array([L]), window=np.array([a.window]),
                       fit_source=np.array([a.fit_source]),
                       scenes=np.array(scenes), n_ep_per_scene=np.array([J]),
                       n_mixed_scene=np.array([cnt]))
        used = []
        for i, s in enumerate(scenes):
            denom = cnt - int(mixed[i])
            if denom <= 0:
                continue
            w = (tot - Dm[i]) / denom
            n = np.linalg.norm(w)
            if n == 0:
                continue
            payload[f"dir_{s}"] = (w / n).astype(np.float32)
            used.append(s)
        f = Path(a.out_dir) / f"direction_L{L}_loso.npz"
        np.savez(f, **payload)
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
        print(json.dumps(dict(file=str(f), layer=L, window=a.window,
                              n_dir=len(used), dim=D, mixed_scene=cnt,
                              sha256=sha, fit_source=a.fit_source), ensure_ascii=False))


if __name__ == "__main__":
    main()

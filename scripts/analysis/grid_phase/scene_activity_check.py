#!/usr/bin/env python3
"""scene 별 활성화 '활성도' 비교 — 관측 이상(死) vs 어려운 scene 판별.

**왜**: 전패 scene 이 나왔을 때 두 해석이 갈린다 — ① 관측은 정상인데 그 배치가 어려워
정책이 실패, ② 렌더/관측 자체가 이상해 정책이 아무 정보도 못 받음. ②면 그 데이터는
분석에서 빼야 한다. 영상 육안·VL 상수화 검사와 **독립된 축**으로 활성화에서 직접 본다.

측정 (모두 layer 12 · 마지막 denoise · 49토큰 mean 좌표):
  - `time_var`   판 안에서 record 가 시간에 따라 얼마나 움직이나
                 (연속 record L2 거리의 평균 / 전체 표준편차로 정규화).
                 관측이 얼어붙으면 이 값이 무너진다.
  - `across_var` 같은 record 인덱스에서 판들 사이 분산(= 노이즈 시드 차이의 반영).
                 관측이 상수면 판 간 차이도 줄어든다.
  - `j_acc`      초기 창에서 j(초기조건) 5-class 판독 정확도. 관측이 살아 있으면
                 배치 차이가 읽히고, 死 상태면 chance 로 내려간다.
정상 scene 과 나란히 놓고 **같은 키의 다른 scene 대비 급락이 있는지**로 판단한다.
절대값 기준은 없다 — 키·task 마다 스케일이 다르기 때문이다.

사용:
    python3 scene_activity_check.py --shard <...>__s0.npz --shard <...>__s1.npz ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

LAYER_IDX, DENOISE_IDX, SEG_IDX = 5, 3, 3


def load(shard: Path):
    with np.load(shard, allow_pickle=False) as z:
        X = z["X"][:, LAYER_IDX, DENOISE_IDX, SEG_IDX, :].astype(np.float32)
        return X, z["ep_id"], z["rec_idx"], z["jitter"], z["succ"]


def nearest_centroid_acc(F, labels, folds=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(F))
    parts = np.array_split(idx, folds)
    correct = total = 0
    for f in range(folds):
        te = parts[f]
        tr = np.concatenate([parts[g] for g in range(folds) if g != f])
        classes = np.unique(labels[tr])
        if len(classes) < 2:
            continue
        cent = np.stack([F[tr][labels[tr] == c].mean(axis=0) for c in classes])
        pred = classes[((F[te][:, None, :] - cent[None]) ** 2).sum(axis=2).argmin(axis=1)]
        correct += int((pred == labels[te]).sum())
        total += len(te)
    return correct / total if total else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", required=True, type=Path, action="append")
    ap.add_argument("--early", type=int, default=10, help="j 판독에 쓸 초기 record 수")
    a = ap.parse_args()

    print(f"{'shard':>34} {'판':>4} {'succ':>5} {'rec/판':>7} "
          f"{'time_var':>9} {'across_var':>11} {'j_acc':>6}")
    for shard in a.shard:
        X, ep, rec, jit, succ = load(shard)
        eps = np.unique(ep)
        gstd = float(X.std()) or 1.0

        # 판 안 시간 변화 — 연속 record 사이 거리 (전체 std 로 정규화)
        steps = []
        for e in eps:
            m = ep == e
            xs = X[m][np.argsort(rec[m])]
            if len(xs) > 1:
                steps.append(np.linalg.norm(np.diff(xs, axis=0), axis=1).mean())
        time_var = float(np.mean(steps)) / gstd if steps else float("nan")

        # 판 사이 분산 — 같은 record 인덱스끼리 (초기 창)
        acc_var = []
        for r in range(min(a.early, int(rec.max()) + 1)):
            m = rec == r
            if m.sum() >= 5:
                acc_var.append(float(X[m].std(axis=0).mean()))
        across_var = float(np.mean(acc_var)) / gstd if acc_var else float("nan")

        # 초기 창 j 판독
        feats, labs = [], []
        for e in eps:
            m = (ep == e) & (rec < a.early)
            if m.any():
                feats.append(X[m].mean(axis=0))
                labs.append(int(jit[ep == e][0]))
        j_acc = (nearest_centroid_acc(np.stack(feats), np.array(labs))
                 if len(set(labs)) > 1 else float("nan"))

        n_succ = len({int(e) for e, s in zip(ep, succ) if s == 1})
        print(f"{shard.stem:>34} {len(eps):>4} {n_succ:>5} "
              f"{len(X) / len(eps):>7.1f} {time_var:>9.4f} {across_var:>11.4f} "
              f"{j_acc:>6.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

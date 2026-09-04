#!/usr/bin/env python3
"""초기 record 가 무엇을 담고 있나 — j 정체성 vs 성공/실패 (scene shard 진단).

**왜**: detector 가 rollout 초반(0~10 record)에 발화하고 그 셀의 "성공판 오경보"가 1.00 인
현상이 관측됐다(fail detector 세션, 2026-09-04). 가설 두 개가 같은 관측을 낸다 —
① 실패를 정말 일찍 감지한다, ② **실패가 아니라 j(초기조건) 자체를 읽는다**. v6 의 j 는
로봇 base 오프셋·물체 재배치라 t=0 부터 활성화에 보인다. 둘은 대책이 다르다(②면 밴드를
j 별로 잡거나 j 를 통제해야 한다).

**측정**: 같은 (instruction, scene) 안에서 record 창을 바꿔 가며
  - `j 정체성` 5-class 정확도 (chance = 1/5)
  - `성공/실패` AUROC (chance = 0.5)
를 **에피소드 단위 홀드아웃**으로 잰다. 분류기는 최근접 클래스 평균(공분산 미사용) —
승준 노드에 sklearn 이 없고, 여기서 알고 싶은 건 "선형적으로 드러나는가" 뿐이다.

t=0 창에서 j 정확도가 높고 succ/fail AUROC 도 높다면, 초기 판정은 실패 예측이 아니라
초기조건 판독일 수 있다(그 scene 에서 j 와 결과가 상관되기 때문). j 정확도가 chance 인데
succ/fail 만 높으면 ①이다.

사용:
    python3 early_record_probe.py --shard <...>/segA_scene/OpenDrawer_left__s0.npz \
        --windows 0:1,0:3,0:10,10:30 --folds 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LAYER_IDX = 5      # capture_layers [0,2,4,8,10,12,15] 의 layer 12
DENOISE_IDX = 3    # 마지막 denoise call
SEGMENT_IDX = 3    # "all" (49 토큰 평균)


def episode_features(shard: Path, lo: int, hi: int):
    """에피소드별 [lo, hi) record 평균 feature + 라벨. 창이 비면 그 판은 뺀다."""
    with np.load(shard, allow_pickle=False) as z:
        X = z["X"][:, LAYER_IDX, DENOISE_IDX, SEGMENT_IDX, :].astype(np.float32)
        ep, rec = z["ep_id"], z["rec_idx"]
        succ, jit = z["succ"], z["jitter"]
    feats, ys, js, eps = [], [], [], []
    for e in np.unique(ep):
        m = (ep == e) & (rec >= lo) & (rec < hi)
        if not m.any():
            continue
        feats.append(X[m].mean(axis=0))
        ys.append(int(succ[ep == e][0]))
        js.append(int(jit[ep == e][0]))
        eps.append(int(e))
    return np.stack(feats), np.array(ys), np.array(js), np.array(eps)


def nearest_centroid_acc(F, labels, folds, rng):
    """에피소드 단위 K-fold — 클래스 평균 최근접 정확도."""
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
        d = ((F[te][:, None, :] - cent[None]) ** 2).sum(axis=2)
        pred = classes[d.argmin(axis=1)]
        correct += int((pred == labels[te]).sum())
        total += len(te)
    return correct / total if total else float("nan")


def auroc(scores, labels):
    """이진 AUROC (동점은 평균 순위). labels: 1 = 양성."""
    pos, neg = labels == 1, labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    s = np.sort(scores)
    i = 0
    while i < len(s):                       # 동점 구간 평균 순위
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2
        i = j + 1
    return (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum())


def succ_auroc_holdout(F, y, folds, rng):
    """에피소드 홀드아웃 — 학습 fold 의 클래스 평균차 방향에 투영해 AUROC."""
    idx = rng.permutation(len(F))
    parts = np.array_split(idx, folds)
    sc, lb = [], []
    for f in range(folds):
        te = parts[f]
        tr = np.concatenate([parts[g] for g in range(folds) if g != f])
        if len(np.unique(y[tr])) < 2:
            continue
        w = F[tr][y[tr] == 1].mean(axis=0) - F[tr][y[tr] == 0].mean(axis=0)
        n = np.linalg.norm(w)
        if n == 0:
            continue
        sc.append(F[te] @ (w / n))
        lb.append(y[te])
    if not sc:
        return float("nan")
    return auroc(np.concatenate(sc), np.concatenate(lb))


def succ_auroc_from_j(y, j, folds, rng):
    """**대조군**: 활성화를 안 보고 j 만으로 결과를 맞히면 얼마나 되나.

    학습 fold 에서 j 별 성공률을 구해 테스트 판의 점수로 쓴다. 활성화 기반 AUROC 가 이
    값과 비슷하면 그 신호는 "결과를 읽은 것"이 아니라 **j 정체성을 읽고 j→결과 상관을
    타고 간 것**일 수 있다(초기 창에서 특히). 크게 높으면 j 로 설명되지 않는 성분이 있다.
    """
    idx = rng.permutation(len(y))
    parts = np.array_split(idx, folds)
    sc, lb = [], []
    for f in range(folds):
        te = parts[f]
        tr = np.concatenate([parts[g] for g in range(folds) if g != f])
        if len(np.unique(y[tr])) < 2:
            continue
        rate = {int(v): float(y[tr][j[tr] == v].mean())
                for v in np.unique(j[tr]) if (j[tr] == v).any()}
        base = float(y[tr].mean())
        sc.append(np.array([rate.get(int(v), base) for v in j[te]]))
        lb.append(y[te])
    if not sc:
        return float("nan")
    return auroc(np.concatenate(sc), np.concatenate(lb))


def succ_auroc_within_j(F, y, j, folds, rng):
    """**j 잔차화** 후 succ/fail AUROC — "같은 j 안에서" 결과가 갈리는가.

    배포 단위가 (instruction, scene, 대상 j) 하나면 j 는 그 안에서 상수라 j-only 대조군이
    무력하다(fail detector 세션 지적, 2026-09-04). 그 경우의 정직한 질문은 "j 를 지운 뒤에도
    성공/실패가 갈리나" 이다. 각 판의 feature 에서 **학습 fold 로 구한 그 j 의 평균**을 빼
    j 성분을 제거한 뒤 평균차 방향을 fit 한다(테스트 판의 j 평균도 학습 fold 에서 온
    것이라 누수 없음). 셀 하나로는 표본이 10판이라, j 를 지우고 scene 전체를 모아
    검정력을 확보하는 효과도 있다.
    """
    idx = rng.permutation(len(F))
    parts = np.array_split(idx, folds)
    sc, lb = [], []
    for f in range(folds):
        te = parts[f]
        tr = np.concatenate([parts[g] for g in range(folds) if g != f])
        if len(np.unique(y[tr])) < 2:
            continue
        mu = {int(v): F[tr][j[tr] == v].mean(axis=0) for v in np.unique(j[tr])}
        gmu = F[tr].mean(axis=0)
        Rtr = F[tr] - np.stack([mu.get(int(v), gmu) for v in j[tr]])
        Rte = F[te] - np.stack([mu.get(int(v), gmu) for v in j[te]])
        if len(np.unique(y[tr])) < 2:
            continue
        w = Rtr[y[tr] == 1].mean(axis=0) - Rtr[y[tr] == 0].mean(axis=0)
        n = np.linalg.norm(w)
        if n == 0:
            continue
        sc.append(Rte @ (w / n))
        lb.append(y[te])
    if not sc:
        return float("nan")
    return auroc(np.concatenate(sc), np.concatenate(lb))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", required=True, type=Path, action="append",
                    help="scene shard NPZ (여러 번 지정 가능)")
    ap.add_argument("--windows", default="0:1,0:3,0:10,10:30",
                    help="record 창 목록 lo:hi (hi 배타)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", type=Path, default=None)
    a = ap.parse_args()

    wins = []
    for w in a.windows.split(","):
        lo, hi = w.split(":")
        wins.append((int(lo), int(hi)))

    report = {}
    for shard in a.shard:
        rows = []
        print(f"\n=== {shard.stem} ===", flush=True)
        print(f"{'창':>10} {'판':>5} {'j정확도':>9} {'(chance)':>9} "
              f"{'succAUROC':>10} {'j만':>7} {'j잔차화':>8}", flush=True)
        for lo, hi in wins:
            F, y, j, _ = episode_features(shard, lo, hi)
            rng = np.random.default_rng(a.seed)
            n_j = len(np.unique(j))
            acc = nearest_centroid_acc(F, j, a.folds, rng)
            rng = np.random.default_rng(a.seed)
            au = succ_auroc_holdout(F, y, a.folds, rng)
            rng = np.random.default_rng(a.seed)
            au_j = succ_auroc_from_j(y, j, a.folds, rng)
            rng = np.random.default_rng(a.seed)
            au_r = succ_auroc_within_j(F, y, j, a.folds, rng)
            chance = 1.0 / n_j if n_j else float("nan")
            print(f"{f'{lo}:{hi}':>10} {len(F):>5} {acc:>9.3f} {chance:>9.3f} "
                  f"{au:>10.3f} {au_j:>7.3f} {au_r:>8.3f}", flush=True)
            rows.append({"window": [lo, hi], "n_episodes": int(len(F)),
                         "n_jitters": int(n_j), "j_acc": float(acc),
                         "j_chance": float(chance), "succ_auroc": float(au),
                         "succ_auroc_from_j_only": float(au_j),
                         "succ_auroc_j_residualized": float(au_r)})
        report[shard.stem] = rows

    if a.out_json:
        a.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n[probe] → {a.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

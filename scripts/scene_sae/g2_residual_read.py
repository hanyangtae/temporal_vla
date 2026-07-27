#!/usr/bin/env python
"""exp5/G2 — scene 성분 제거 후 succ/fail read 가 잔존하는가.

핸드아웃 `docs/steering/30_sae_g1_port_handout.md` §1 사다리의 **G2**:
G1(= SAE feature 가 scene 을 인코딩한다, `docs/steering/31_sae_g1_results.md` PASS)
다음 관문. 질문은 하나다 —

    **scene(암기) 성분을 명시적으로 제거해도 succ/fail 분리가 남는가.**

남으면 그 잔여 방향이 steering 후보(G3), 안 남으면 "분리는 scene 이 만든 것" = latent
steering 서사 종결의 판정 근거. 양방향 가치 실험이므로 **제거가 실제로 일어났는지**를
같은 표에서 함께 보고한다(제거 실패면 어느 쪽도 주장 못 한다 → 판정불가).

## 세 arm (같은 프로토콜, 잔차화 방식만 다름)

| arm | 잔차화 |
|---|---|
| `raw`    | 없음 (기준선 — exp5-3 within-scene 수치와 대조해 프로토콜 정합성 자가검증) |
| `sae`    | SAE 인코드 → **scene-selective feature 0** → decode 재구성 |
| `linear` | scene 라벨 선형 부분공간(between-scatter PCA / multinomial logreg 행공간) 직교사영 제거 |

`sae` 는 통제로 `sae_full`(feature 제거 없이 decode 재구성)을 자동 동반한다. 재구성 자체가
정보를 깎으므로, `sae` 를 `raw` 와 직접 비교하면 "selective feature 제거 효과"와 "SAE 재구성
손실"이 섞인다. 비교 기준은 `sae` vs `sae_full` 이다.

## 판정 프로토콜 (단일 출처 = exp5-3 within-scene 구현)

`loso()` / `within_dir()` / `perm_null()` / `auroc()` 는 exp5-3 `analyze_sm2_ref.py` 에서
**그대로 이식**했다 (아래 함수 docstring 에 원본 표시). 즉:

  - 창 `[0, window)` record 만 (실패=timeout 이라 후반 record 는 실패에만 존재 → 상태 confound).
  - **within-scene 방향** (scene 안 succ-fail 평균차의 scene 평균) + **held-out scene 내부 중심화**
    LOSO, 혼재 scene(succ·fail 둘 다 있는 scenario_seed)만 평가.
  - null = scene **내부** 라벨 순열 n_perm 회 → z = (auroc − null_mean)/null_sd.

`linear` arm 의 부분공간은 **LOSO fold 안에서 재추정**한다(held-out scene 을 절대 보지 않음).
부분공간은 라벨 y 를 쓰지 않으므로 순열 null 에서 fold 별 사영을 캐시해 재사용한다
(y 와 독립 → 캐시가 null 을 낙관적으로 만들지 않는다).

## 데이터 단위 — record 집계 방식 (설계 결정)

입력(`build_sae_inputs.py` 산출)은 **per-token 행** [N, 1536] 이고, exp5-3 참조 구현은 이미
토큰 축이 접힌 record 벡터 `[T_rec, L, D]` 를 썼다(그 빌더는 이 브랜치에 없어 접는 방식을
1차 근거로 확인 불가). 따라서 여기서는 근거가 있는 쪽을 **1차**로 고정하고 나머지를 민감도로
병기한다:

  - **1차 = `future` 세그먼트(token 1..32) 평균.** G1 발견(scene 정보가 future 토큰에 실린다)
    반영. 토큰 전체 평균은 state 1개 + action 16개를 섞어 scene 신호를 희석한다.
  - **민감도 = `all` (전 토큰 평균).** 참조 구현이 전 토큰 평균이었을 가능성에 대한 대조.

`--token-agg` 로 둘 다(기본 `future,all`) 돌려 표에 같이 낸다. 어느 쪽이든 **rollout 전체를
1벡터로 pool 하지 않는다** (memory `feedback-no-rollout-pooling`) — record 축은 살아 있고,
읽기 단위는 ① `window_mean`(episode × 창평균, 참조 구현의 창 집계) ② `fixed_t` 곡선
(record t 별, pooling 없음) 두 가지를 모두 보고한다. 순열 null 은 1차 읽기(`window_mean`)에.

## scene 제거 검증 (판정용 probe = fold-wise)

각 arm×설정마다 잔차화 후 **scene probe 정확도**를 같이 낸다. 판정에 쓰는 것은
`scene_probe_foldwise` — **read 가 실제로 쓴 fold 별 사영 Q_s 와 같은 scene 라벨
(scenario_seed)** 로, held-out scene 행에서 평가하고 fold 를 pool/평균한다. 기존
split-train 단일 사영 + `--scene-label`(layout_id) probe 는 `scene_probe` 로 **참고치**만
유지한다. 제거 전 대비 chance 위 잉여가 충분히 깎이지 않았으면 그 설정은 **판정불가**.

## 판정 전제조건·표본 등급 (리뷰 반영)

  - **전제조건**: 제거 **전** 기준선(raw / sae_full)의 window_mean null_z ≤ 3 이면 지울
    분리가 애초에 없다 → verdict = `no_baseline_separation` (제거 판정을 하지 않는다).
  - **표본 두 벌**: (a) 전체 혼재 scene = `window_mean`, (b) 사전(SAE)·train split 이 안 본
    scene 만 = `window_mean_unseen_scenes`. 판정은 (b) 가 있으면 (b), 없으면 (a) 로 하되
    verdict 에 `_exploratory` 를 붙인다.
  - **SAE 축 검증**: ckpt 가 scene-heldout 축(`split_col=split_scene`,
    `split_axis_scene_heldout=True`)이 아니면 **에러** (`--allow-nonscene-sae` 로만 우회).
    평가 scene 이 SAE train_scenes 에 포함된 비율을 json 에 기록하고, >0 이면
    verdict 에 `_exploratory`. fold 별 SAE 재학습은 비용상 하지 않는다(사유는 json).
  - **길이 confound**: `window_mean` 결과에 창 내 record 수 분포(succ/fail 평균)를 병기.
    cohort 강제는 하지 않으며, 길이-공정 판독은 `fixed_t` 곡선을 본다.

사용 예:

    python scripts/scene_sae/g2_residual_read.py \\
        --x  .../inputs/X_L12.npz --meta .../inputs/meta.npz \\
        --ckpt-dir .../L12_m6144_k64_s0 --probe-json .../probe_L12.json \\
        --layer 12 --window 38 --arms raw,sae,linear \\
        --r-list 1,2,4,8 --sel-top-list 10,20,40,all --n-perm 200 \\
        --out .../g2_L12.json

합성 자기검증(실데이터 불필요):  `--smoke`
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SEG_NAMES = {0: "state", 1: "future", 2: "action"}
AGG_SEGS = {"future": (1,), "all": (0, 1, 2), "action": (2,), "state": (0,)}

# 판정 임계 (사전 등록) — json "criteria" 에 그대로 기록된다.
Z_KEEP = 3.0            # 잔차화 후 succ/fail z > 3 → outcome 신호 잔존
SCENE_KILL_FRAC = 0.50  # scene probe 의 chance 위 잉여가 이 비율 미만으로 줄어야 "제거 성공"


# ───────────────────────────────────────────────────────────── 판정 코어 (이식)
# 아래 auroc/within_dir/loso/perm_null 은 exp5-3 within-scene 구현
# (analyze_sm2_ref.py:49-112) 에서 그대로 이식. loso 는 fold 별 사영을 받도록만 일반화했고
# (Vfolds=None 이면 원본과 동일한 계산), 나머지는 손대지 않았다.
def auroc(s, y):
    """analyze_sm2_ref.py:49 원본 (tie 평균 순위 Mann-Whitney U)."""
    s = np.asarray(s, dtype=np.float64)
    y = np.asarray(y)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    sv = s[o]
    ranks = np.arange(1, len(s) + 1, dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sv[j + 1] == sv[i]:
            j += 1
        r[o[i:j + 1]] = ranks[i:j + 1].mean()
        i = j + 1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def within_dir(V, y, sc):
    """analyze_sm2_ref.py:69 원본 — scene 안 (succ 평균 − fail 평균) 의 scene 평균."""
    ds = []
    for s in np.unique(sc):
        m = sc == s
        if (y[m] == 1).sum() == 0 or (y[m] == 0).sum() == 0:
            continue
        ds.append(V[m & (y == 1)].mean(0) - V[m & (y == 0)].mean(0))
    return np.mean(ds, axis=0) if ds else None


def loso(V, y, sc, Vfolds=None):
    """analyze_sm2_ref.py:79 원본 + fold 별 사영 훅.

    within-scene 방향 + held-out scene 내부 중심화. 혼재 scene 만 평가.
    `Vfolds[s]` 가 주어지면 held-out scene s 의 fold 에서 그 배열을 대신 쓴다
    (linear arm 의 **fold 안 재추정** 사영 — held-out scene 은 부분공간 추정에 안 들어간다).
    Vfolds=None 이면 원본과 비트 단위로 같은 계산.
    """
    sco = np.full(len(y), np.nan)
    for s in np.unique(sc):
        Vs = V if Vfolds is None else Vfolds[int(s)]
        te = sc == s
        tr = ~te
        w = within_dir(Vs[tr], y[tr], sc[tr])
        if w is None:
            continue
        n = np.linalg.norm(w)
        if n == 0:
            continue
        Vte = Vs[te] - Vs[te].mean(0, keepdims=True)
        sco[te] = Vte @ (w / n)
    keep = np.zeros(len(y), bool)
    for s in np.unique(sc):
        m = sc == s
        if (y[m] == 1).sum() > 0 and (y[m] == 0).sum() > 0:
            keep |= m
    m = keep & ~np.isnan(sco)
    return auroc(sco[m], y[m]), int(m.sum())


def perm_null(V, y, sc, n_perm, rng, Vfolds=None):
    """analyze_sm2_ref.py:102 원본 — scene **내부** 라벨 순열."""
    out = []
    for _ in range(n_perm):
        yp = y.copy()
        for s in np.unique(sc):
            idx = np.where(sc == s)[0]
            yp[idx] = rng.permutation(y[idx])
        v, _ = loso(V, yp, sc, Vfolds=Vfolds)
        if not np.isnan(v):
            out.append(v)
    return np.array(out)


def read_with_null(V, y, sc, n_perm, rng, Vfolds=None):
    a, n = loso(V, y, sc, Vfolds=Vfolds)
    res = {"auroc": None if np.isnan(a) else float(a), "n_eval": n, "n_sample": int(len(y))}
    if np.isnan(a) or n_perm <= 0:
        return res
    nul = perm_null(V, y, sc, n_perm, rng, Vfolds=Vfolds)
    if len(nul) == 0:
        return res
    sd = float(nul.std())
    res.update(null_mean=float(nul.mean()), null_sd=sd, n_perm=int(len(nul)),
               null_z=float((a - nul.mean()) / sd) if sd > 0 else None,
               p=float((nul >= a).mean()))
    return res


# ─────────────────────────────────────────────────────── scene 부분공간 (linear)
def between_scatter_basis(V, sc, r_max):
    """(A) scene 클래스평균 between-scatter 상위 r 주성분.

    scene 별 평균을 전체 평균 기준으로 중심화해 쌓은 행렬의 우특이벡터 = between-scatter
    Σ_s n_s (μ_s − μ)(μ_s − μ)ᵀ 의 상위 고유벡터 (√n_s 가중). 라벨 y 는 안 쓴다.
    """
    scs = np.unique(sc)
    if len(scs) < 2:
        return np.zeros((V.shape[1], 0), np.float64)
    mu = V.mean(0, keepdims=True)
    M = np.stack([np.sqrt((sc == s).sum()) * (V[sc == s].mean(0) - mu[0]) for s in scs])
    _u, _s, vt = np.linalg.svd(M, full_matrices=False)
    r = int(min(r_max, vt.shape[0]))
    return np.ascontiguousarray(vt[:r].T)                    # [D, r] 정규직교


def logreg_scene_basis(V, sc, r_max, seed=0, max_iter=1000, C=1.0):
    """(B) multinomial logreg(scene) 가중치 행공간 상위 r.

    coef_ [n_class, D] 를 SVD 해 상위 r 우특이벡터. 표준화는 fold train 행에서만
    fit(누수 방지) 하고, 스케일러의 1/σ 를 coef 에 되돌려 **원공간 방향**으로 만든다.
    """
    from sklearn.linear_model import LogisticRegression      # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler         # noqa: PLC0415

    scs = np.unique(sc)
    if len(scs) < 2:
        return np.zeros((V.shape[1], 0), np.float64)
    sca = StandardScaler().fit(V)
    lr = LogisticRegression(max_iter=max_iter, C=C, random_state=seed)
    lr.fit(sca.transform(V), sc)
    coef = np.asarray(lr.coef_, dtype=np.float64) / np.maximum(sca.scale_, 1e-12)
    _u, _s, vt = np.linalg.svd(coef, full_matrices=False)
    r = int(min(r_max, vt.shape[0]))
    return np.ascontiguousarray(vt[:r].T)


BASIS_FNS = {"between": between_scatter_basis, "logreg": logreg_scene_basis}


def project_out(V, Q):
    """Q(정규직교 [D,r]) 가 span 하는 부분공간을 직교 사영으로 제거."""
    if Q.shape[1] == 0:
        return V
    return V - (V @ Q) @ Q.T


def fold_bases(V, sc, basis_fn, r_max, spy=None):
    """LOSO fold(=held-out scene) 별로 **train scene 만** 써서 추정한 부분공간 기저.

    반환 {scene: Q [D, r_max]}. 기저만 캐시하면 r sweep 은 열 슬라이스로 끝난다
    (사영 결과 배열을 t·r·method 마다 들고 있으면 GB 단위로 터진다).
    누수 금지 계약: basis_fn 에 넘어가는 행에는 held-out scene 이 없다 (spy 로 검증 가능).
    """
    out = {}
    for s in np.unique(sc):
        tr = sc != int(s)
        if spy is not None:
            spy.append({"heldout": int(s),
                        "train_scenes": sorted(int(v) for v in np.unique(sc[tr]))})
        out[int(s)] = basis_fn(V[tr], sc[tr], r_max)
    return out


def fold_projections(V, sc, basis_fn, r_list, spy=None):
    """fold_bases + 사영 — 반환 {r: {scene: V_projected}} (부분공간은 y 와 무관하므로
    순열 null 에서 재사용 가능)."""
    bases = fold_bases(V, sc, basis_fn, max(r_list), spy=spy)
    return {int(r): {s: project_out(V, Q[:, :int(r)]) for s, Q in bases.items()}
            for r in r_list}


# ─────────────────────────────────────────────────────────────── scene probe
def scene_probe(Vtr, ytr, Vte, yte, seed=0, max_iter=1000, C=1.0):
    """잔차화가 실제로 scene 을 지웠는지 — 선형 probe 정확도 (train fit / held-out 평가).

    **참고치 전용** (split-train 단일 사영). 판정에 쓰는 것은 `foldwise_scene_probe` —
    read 가 실제로 쓴 fold 별 사영과 라벨을 그대로 재현하기 때문 (리뷰 #1).
    """
    from sklearn.linear_model import LogisticRegression      # noqa: PLC0415
    from sklearn.pipeline import Pipeline                    # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler         # noqa: PLC0415

    if len(np.unique(ytr)) < 2 or len(yte) == 0:
        return {"acc": None, "chance": None, "n_test": int(len(yte)), "note": "클래스 부족"}
    pipe = Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=max_iter, C=C, random_state=seed))])
    pipe.fit(Vtr, ytr)
    acc = float((pipe.predict(Vte) == yte).mean())
    _cls, cnt = np.unique(yte, return_counts=True)
    return {"acc": acc, "chance": float(cnt.max() / cnt.sum()), "n_test": int(len(yte)),
            "n_train": int(len(ytr)), "n_class": int(len(np.unique(ytr)))}


def _fit_probe(V, y, seed=0, max_iter=1000, C=1.0):
    from sklearn.linear_model import LogisticRegression      # noqa: PLC0415
    from sklearn.pipeline import Pipeline                    # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler         # noqa: PLC0415

    pipe = Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=max_iter, C=C, random_state=seed))])
    pipe.fit(V, y)
    return pipe


def scene_stratified_holdout(sc_rows, ep_rows, frac=1.0 / 3.0):
    """scene 마다 episode 의 뒤쪽 frac 을 probe-test 로 — (probe_train, probe_test) 마스크.

    meta split 을 그대로 쓸 수 없다: scene-heldout 축(`split_scene`)이면 test 행의 scene 이
    train 에 아예 없어 scene 라벨 probe 가 퇴화한다(그 클래스를 학습할 수 없음). 여기서
    묻는 것은 "사영 후에도 scene 이 읽히는가"이므로 **scene 안에서 episode 단위로** 가른다
    (같은 episode 행이 train/test 에 걸치지 않게).
    """
    sc_rows = np.asarray(sc_rows)
    ep_rows = np.asarray(ep_rows)
    te = np.zeros(len(sc_rows), bool)
    for s in np.unique(sc_rows):
        m = sc_rows == s
        eps = np.unique(ep_rows[m])
        if len(eps) < 2:
            continue                       # 나눌 수 없는 scene → 이 scene 은 평가에서 빠진다
        n_te = max(1, int(round(len(eps) * frac)))
        for e in eps[len(eps) - n_te:]:
            te |= m & (ep_rows == e)
    return ~te, te


def foldwise_scene_probe(V, sc_rows, ep_rows, basis_fn=None, r=0, seed=0, frac=1.0 / 3.0):
    """제거 검증 probe — **read 와 같은 fold 별 사영·같은 scene 라벨** (리뷰 #1).

    read 는 LOSO fold 마다 held-out scene 을 뺀 행으로 추정한 Q_s 로 사영한다. 제거 성공
    여부도 그 Q_s 로 사영된 **held-out scene 행**에서 재야 한다 (split-train 단일 Q +
    layout 라벨은 read 가 쓴 연산이 아니다).

      - fold s: 부분공간 = probe-train ∧ scene≠s 행에서 추정 (held-out scene 미열람),
        probe 는 사영된 probe-train 행 전체로 fit → **scene s 의 probe-test 행**에서 평가.
        (probe 자체는 scene s 의 다른 episode 를 봐도 된다 — 여기서 묻는 것은 "사영 후에도
        scene 이 읽히는가"이지 scene 일반화가 아니다. held-out 계약은 **부분공간**에 건다.)
      - fold 별 test 행은 서로 겹치지 않으므로 pooled accuracy 가 곧 전체 probe-test 정확도.
      - `basis_fn=None` = 사영 없음 → SAE arm(이미 masked-decode 된 표현)과 raw 기준선.
        이 경우 fold 마다 같은 probe 이므로 한 번만 fit 한다.

    라벨 = `sc_rows` (= 부분공간 추정에 쓴 scene 라벨, scenario_seed).
    """
    sc_rows = np.asarray(sc_rows)
    scenes = np.unique(sc_rows)
    tr_m, te_m = scene_stratified_holdout(sc_rows, ep_rows, frac=frac)
    if len(scenes) < 2 or not tr_m.any() or not te_m.any() \
            or len(np.unique(sc_rows[tr_m])) < 2:
        return {"acc": None, "chance": None, "n_test": int(te_m.sum()),
                "note": "scene/episode 부족"}
    shared = _fit_probe(V[tr_m], sc_rows[tr_m], seed=seed) if basis_fn is None else None
    preds, truth, per_fold = [], [], []
    for s in scenes:
        te = te_m & (sc_rows == int(s))
        if not te.any():
            continue
        if basis_fn is None:
            Vp, pipe = V, shared
        else:
            fit = tr_m & (sc_rows != int(s))
            if not fit.any() or len(np.unique(sc_rows[fit])) < 2:
                continue
            Q = basis_fn(V[fit], sc_rows[fit], int(r))
            Vp = project_out(V, Q)
            pipe = _fit_probe(Vp[tr_m], sc_rows[tr_m], seed=seed)
        p = np.asarray(pipe.predict(Vp[te]))
        preds.append(p)
        truth.append(sc_rows[te])
        per_fold.append({"scene": int(s), "acc": float((p == sc_rows[te]).mean()),
                         "n_test": int(te.sum())})
    if not per_fold:
        return {"acc": None, "chance": None, "n_test": 0, "note": "평가 가능한 fold 없음"}
    p = np.concatenate(preds)
    t = np.concatenate(truth)
    _c, cnt = np.unique(t, return_counts=True)
    return {"acc": float((p == t).mean()), "chance": float(cnt.max() / cnt.sum()),
            "acc_fold_mean": float(np.mean([f["acc"] for f in per_fold])),
            "n_test": int(len(t)), "n_train": int(tr_m.sum()), "n_folds": len(per_fold),
            "label": "scene(=scenario_seed, 부분공간 추정 라벨과 동일)",
            "projection": "fold" if basis_fn is not None else "none",
            "split": "scene 안 episode 뒤쪽 1/3 (meta split 아님)",
            "per_fold": per_fold}


def removal_ok(before: dict, after: dict) -> bool | None:
    """chance 위 잉여가 SCENE_KILL_FRAC 미만으로 줄었나 (제거 성공 여부)."""
    if not before or not after or before.get("acc") is None or after.get("acc") is None:
        return None
    base = before["acc"] - before["chance"]
    if base <= 0.02:                                   # 애초에 scene 이 안 읽히면 판정 불가
        return None
    return bool((after["acc"] - after["chance"]) < SCENE_KILL_FRAC * base)


DECISIVE = ("outcome_separable_from_scene", "separation_was_scene")


def verdict_of(read: dict, removed: bool | None, base_read: dict | None = None) -> str:
    """제거 판정. `base_read` = 제거 **전** 기준선(raw / sae_full)의 window_mean read.

    전제조건 (리뷰 #4): 기준선에 애초에 succ/fail 분리가 없으면(null_z ≤ Z_KEEP) 제거 후
    분리가 없다는 사실은 아무것도 말하지 않는다 → 판정 자체를 하지 않는다.
    """
    if base_read is not None:
        bz = base_read.get("null_z")
        if bz is None or bz <= Z_KEEP:
            return "no_baseline_separation"
    if removed is None:
        return "undecidable_scene_probe_baseline"
    if not removed:
        return "undecidable_removal_failed"
    z = read.get("null_z")
    if z is None:
        return "undecidable_no_null"
    return "outcome_separable_from_scene" if z > Z_KEEP else "separation_was_scene"


def unseen_scene_set(ep_label, split_of_ep, sae_train_scenes=None):
    """부분공간/SAE 학습에 안 쓰인 scene 집합 (리뷰 #3 의 (b) 표본).

    조건 ① meta split 에서 train(0) episode 가 하나도 없는 scene,
    조건 ② (SAE 가 있을 때) SAE train_scenes 에 없는 scene.
    둘 다 만족해야 "사전이 본 적 없는 scene".
    """
    seen_split, scenes = set(), set()
    for e, rec in ep_label.items():
        s = int(rec["scenario_seed"])
        scenes.add(s)
        if int(split_of_ep[int(e)]) == 0:
            seen_split.add(s)
    out = scenes - seen_split
    if sae_train_scenes:
        out -= {int(v) for v in sae_train_scenes}
    return out


# ────────────────────────────────────────────────────────── 행 → record 집계
class RowGrouping:
    """창 안 행을 (episode, record, token_seg) 하위그룹으로 정렬·경계화한 인덱스.

    정렬로 하위그룹을 연속화해 두면 잔차화를 청크 스트리밍하면서 `np.add.reduceat` 으로
    합을 누적할 수 있다 (np.add.at 은 수억 원소에서 감당 불가).
    """

    def __init__(self, meta, window):
        n = len(meta["episode_idx"])
        mask = np.ones(n, bool) if not window else (meta["record_idx"] < int(window))
        idx = np.nonzero(mask)[0]
        if idx.size == 0:
            raise SystemExit(f"--window {window} 적용 후 행 0")
        order = idx[np.lexsort((meta["token_idx"][idx], meta["token_seg"][idx],
                                meta["record_idx"][idx], meta["episode_idx"][idx]))]
        key = np.stack([meta["episode_idx"][order].astype(np.int64),
                        meta["record_idx"][order].astype(np.int64),
                        meta["token_seg"][order].astype(np.int64)], axis=1)
        new = np.ones(len(order), bool)
        new[1:] = (key[1:] != key[:-1]).any(axis=1)
        starts = np.nonzero(new)[0]
        self.order = order
        self.starts = starts
        self.counts = np.diff(np.append(starts, len(order)))
        self.ep = key[starts, 0]
        self.rec = key[starts, 1]
        self.seg = key[starts, 2]
        self.n_rows = len(order)

    def chunks(self, batch_rows):
        """행 수가 batch_rows 를 넘지 않는 **완결된 하위그룹** 구간 (g0, g1, r0, r1)."""
        g0, acc = 0, 0
        for g in range(len(self.starts)):
            c = int(self.counts[g])
            if acc and acc + c > batch_rows:
                yield g0, g, int(self.starts[g0]), int(self.starts[g0] + acc)
                g0, acc = g, 0
            acc += c
        if acc:
            yield g0, len(self.starts), int(self.starts[g0]), int(self.starts[g0] + acc)


def stream_group_sums(X, grp: RowGrouping, transforms: dict, batch_rows: int, verbose=True):
    """하위그룹 합 {name: [n_sub, D]} — transform 마다 한 벌. X 는 한 번만 읽는다."""
    D = X.shape[1]
    acc = {k: np.zeros((len(grp.starts), D), np.float64) for k in transforms}
    t0 = time.time()
    for ci, (g0, g1, r0, r1) in enumerate(grp.chunks(batch_rows)):
        rows = grp.order[r0:r1]
        raw = np.asarray(X[rows], dtype=np.float32)
        local = (grp.starts[g0:g1] - r0).astype(np.int64)
        for name, fn in transforms.items():
            acc[name][g0:g1] = np.add.reduceat(fn(raw).astype(np.float64), local, axis=0)
        if verbose and ci % 20 == 0:
            print(f"  [agg] chunk {ci} rows {r1}/{grp.n_rows} ({time.time()-t0:.0f}s)", flush=True)
    return acc


def record_vectors(S, grp: RowGrouping, segs):
    """하위그룹 합 → record 벡터 (선택 세그먼트만 평균). 반환 (V, ep, rec)."""
    sel = np.isin(grp.seg, np.asarray(segs))
    if not sel.any():
        raise SystemExit(f"세그먼트 {segs} 에 해당하는 행 없음")
    key = grp.ep[sel].astype(np.int64) * (int(grp.rec.max()) + 1) + grp.rec[sel]
    uk, inv = np.unique(key, return_inverse=True)
    V = np.zeros((len(uk), S.shape[1]), np.float64)
    np.add.at(V, inv, S[sel])                          # 하위그룹 수는 작다 (record×seg)
    cnt = np.bincount(inv, weights=grp.counts[sel], minlength=len(uk))
    V /= cnt[:, None]
    ep = np.zeros(len(uk), np.int64)
    rec = np.zeros(len(uk), np.int64)
    ep[inv] = grp.ep[sel]
    rec[inv] = grp.rec[sel]
    return V, ep, rec


def episode_views(V, ep, rec, window):
    """record 벡터 → 읽기 단위 2종.

    window_mean : episode × 창평균 [n_ep, D]  (참조 구현의 창 집계)
    fixed_t     : t → (V_t [n_ep_t, D], ep_t) (pooling 없음)
    """
    eps = np.unique(ep)
    wm = np.stack([V[ep == e].mean(0) for e in eps])
    fixed = {}
    for t in range(int(window)):
        m = rec == t
        if not m.any():
            continue
        fixed[t] = (V[m], ep[m])
    return eps, wm, fixed


# ───────────────────────────────────────────────────────────────── SAE arm
def load_sae_bundle(ckpt_dir: Path, stats_path: Path | None):
    from probe_scene import load_sae                   # noqa: PLC0415 (동일 디렉토리)

    model, cfg = load_sae(Path(ckpt_dir))
    st_path = Path(ckpt_dir) / "stats.npz"
    if not st_path.exists():
        if stats_path is None:
            raise SystemExit(f"표준화 통계 없음: {st_path} 도 --stats 도 없다")
        st_path = Path(stats_path)
    st = np.load(st_path)
    return model, cfg, st["mean"].astype(np.float32), st["std"].astype(np.float32), str(st_path)


def selective_features(probe_json: Path, top):
    """probe json 의 selectivity 블록 → 제거할 feature 인덱스 (gap 내림차순)."""
    res = json.loads(Path(probe_json).read_text())
    sel = res.get("selectivity", {})
    flagged = [int(f["feature"]) for f in sel.get("top_features", []) if f.get("selective")]
    info = {"n_selective_reported": int(sel.get("n_selective", -1)),
            "n_flagged_in_list": len(flagged),
            "truncated": bool(sel.get("n_selective", 0) > len(flagged))}
    n = len(flagged) if (top in (None, "all", 0)) else min(int(top), len(flagged))
    return np.asarray(flagged[:n], np.int64), info


def make_sae_transforms(model, mu, sd, sel_sets: dict, device: str, batch: int):
    """{name: rows→잔차화된 rows} — 마스크 변형마다 encode+decode 한 벌.

    변형끼리 encode 를 공유하지 않는다(= 변형 수만큼 forward 를 반복한다): 공유하려면
    청크 전체의 희소코드 h [n_rows, m=6144] 를 들고 있어야 하는데 200k 행이면 5GB 라
    메모리가 먼저 터진다. GPU 에서 encode 는 초 단위라 반복이 더 싸다.
    """
    import torch                                       # noqa: PLC0415

    dev = torch.device(device)
    model.to(dev).eval()
    mu_t = torch.as_tensor(mu, device=dev)
    sd_t = torch.as_tensor(sd, device=dev)
    masks = {}
    m = int(getattr(model.enc, "m", getattr(model.enc, "d")))
    for name, idx in sel_sets.items():
        keep = torch.ones(m, device=dev)
        if len(idx):
            keep[torch.as_tensor(np.asarray(idx), device=dev, dtype=torch.long)] = 0.0
        masks[name] = keep

    @torch.no_grad()
    def _run(raw, keep):
        out = []
        for i in range(0, len(raw), batch):
            x = torch.as_tensor(raw[i:i + batch], dtype=torch.float32, device=dev)
            h = model.latent((x - mu_t) / sd_t)
            xh = model.dec(h * keep)[0]
            out.append((xh * sd_t + mu_t).cpu().numpy())
        return np.concatenate(out, 0)

    return {name: (lambda raw, k=keep: _run(raw, k)) for name, keep in masks.items()}


# ──────────────────────────────────────────────────────────────────── 실행
def cached_folds(cache, key, V, sc, method, r_list, r):
    """fold 별 **기저** 캐시 — 부분공간 fit 은 r 과 무관하므로 한 번만 (r sweep 재사용).

    반환 (folds_for_r, spy_summary). 캐시 키 = (token_agg, 읽기단위, method).
    """
    if cache is None or key not in cache:
        spy = []
        bases = fold_bases(V, sc, BASIS_FNS[method], max(r_list), spy=spy)
        summary = {"n_folds": len(spy),
                   "heldout_never_in_train": all(s["heldout"] not in s["train_scenes"]
                                                 for s in spy)}
        if cache is None:
            return {s: project_out(V, Q[:, :int(r)]) for s, Q in bases.items()}, summary
        cache[key] = (bases, summary)
    bases, summary = cache[key]
    return {s: project_out(V, Q[:, :int(r)]) for s, Q in bases.items()}, summary


def episode_labels(eps, ep_label):
    y = np.asarray([ep_label[int(e)]["success"] for e in eps])
    sc = np.asarray([ep_label[int(e)]["scenario_seed"] for e in eps])
    return y, sc


def record_count_note(y, n_rec_per_ep):
    """창 안 record 수 분포 (리뷰 #5) — window_mean 의 길이 confound 를 눈에 보이게."""
    n_rec_per_ep = np.asarray(n_rec_per_ep, np.float64)
    def _m(mask):
        return float(n_rec_per_ep[mask].mean()) if mask.any() else None
    return {
        "succ_mean": _m(y == 1), "fail_mean": _m(y == 0),
        "succ_min": (int(n_rec_per_ep[y == 1].min()) if (y == 1).any() else None),
        "fail_min": (int(n_rec_per_ep[y == 0].min()) if (y == 0).any() else None),
        "note": "창 내 record 수가 succ/fail 간 다르면 window_mean 은 길이(=timeout) confound 를 "
                "탄다 (cohort 강제는 하지 않음). 길이-공정 판독은 fixed_t 곡선을 볼 것.",
    }


def mixed_scene_count(y, sc):
    return sum(1 for s in np.unique(sc)
               if (y[sc == s] == 1).any() and (y[sc == s] == 0).any())


def run_arm(tag, arm, cfg, V_rec, ep_rec, rec_idx, ep_label, args, rng,
            probe_pack, base_probe, token_agg=None, probe_override=None,
            fold_cache=None, r_list=None, base_read=None, read_override=None,
            unseen_scenes=None, fold_probe_base=None, fold_probe_override=None,
            probe_cache=None, extra_flags=()):
    """한 (arm, 설정, token_agg) 조합 → 결과 dict."""
    window = int(args.window) if args.window else int(rec_idx.max()) + 1
    eps, wm, fixed = episode_views(V_rec, ep_rec, rec_idx, window)
    y, sc = episode_labels(eps, ep_label)
    n_rec_per_ep = np.asarray([int((ep_rec == e).sum()) for e in eps])
    rs = r_list or [cfg.get("r", 1)]

    Vfolds, fold_summary = None, None
    if arm == "linear":
        Vfolds, fold_summary = cached_folds(fold_cache, (token_agg, "wm", cfg["method"]),
                                            wm, sc, cfg["method"], rs, cfg["r"])

    read = dict(read_override) if read_override is not None \
        else read_with_null(wm, y, sc, args.n_perm, rng, Vfolds=Vfolds)
    read["record_counts"] = record_count_note(y, n_rec_per_ep)

    # (b) 사전/부분공간이 안 본 scene 만으로 한 벌 더 (리뷰 #3)
    unseen = {int(s) for s in (unseen_scenes or ())}
    read_unseen = None
    sub = np.isin(sc, sorted(unseen)) if unseen else np.zeros(len(sc), bool)
    if sub.any() and mixed_scene_count(y[sub], sc[sub]) >= 2:
        vf = None if Vfolds is None else {int(s): Vfolds[int(s)][sub] for s in np.unique(sc[sub])}
        read_unseen = read_with_null(wm[sub], y[sub], sc[sub], args.n_perm, rng, Vfolds=vf)
        read_unseen["record_counts"] = record_count_note(y[sub], n_rec_per_ep[sub])
        read_unseen["scenes"] = sorted(int(s) for s in np.unique(sc[sub]))

    curve = []
    for t, (Vt, ept) in (fixed.items() if not getattr(args, "skip_fixed_t", False) else []):
        yt = np.asarray([ep_label[int(e)]["success"] for e in ept])
        sct = np.asarray([ep_label[int(e)]["scenario_seed"] for e in ept])
        vf = None
        if arm == "linear":
            vf, _ = cached_folds(fold_cache, (token_agg, f"t{t}", cfg["method"]),
                                 Vt, sct, cfg["method"], rs, cfg["r"])
        a, n = loso(Vt, yt, sct, Vfolds=vf)
        curve.append([int(t), None if np.isnan(a) else float(a), int(n), int(len(yt))])
    cur_ok = [c[1] for c in curve if c[1] is not None]

    # scene probe ① 참고치 (split-train 단일 사영, --scene-label 라벨)
    probe = probe_override
    if probe is None and probe_pack is not None:
        Vp, ytr_m, yte_m, y_sc, _sc_rec, _ep_rec = probe_pack
        if ytr_m.any() and yte_m.any():
            Vpr = Vp
            if arm == "linear":
                # 사영 부분공간은 **train split 행만**으로 추정 (probe test 로의 누수 금지).
                Q = BASIS_FNS[cfg["method"]](Vp[ytr_m], y_sc[ytr_m], cfg["r"])
                Vpr = project_out(Vp, Q)
            probe = scene_probe(Vpr[ytr_m], y_sc[ytr_m], Vpr[yte_m], y_sc[yte_m],
                                seed=args.seed)

    # scene probe ② 판정용 — read 와 같은 fold 사영·같은 scene 라벨 (리뷰 #1)
    fw = fold_probe_override
    if fw is None and probe_pack is not None:
        Vp, _ytr_m, _yte_m, _y_sc, sc_rec, ep_rec_p = probe_pack
        if arm == "linear":
            key = (token_agg, cfg["method"], int(cfg["r"]))
            if probe_cache is not None and key in probe_cache:
                fw = probe_cache[key]
            else:
                fw = foldwise_scene_probe(Vp, sc_rec, ep_rec_p,
                                          BASIS_FNS[cfg["method"]], cfg["r"], seed=args.seed)
                if probe_cache is not None:
                    probe_cache[key] = fw
        else:
            fw = foldwise_scene_probe(Vp, sc_rec, ep_rec_p, seed=args.seed)

    is_baseline = arm == "raw" or cfg.get("set") == "sae_full"
    removed = None if is_baseline else removal_ok(fold_probe_base, fw)
    removed_ref = None if is_baseline else removal_ok(base_probe, probe)

    flags = list(extra_flags)
    primary = read_unseen if read_unseen is not None else read
    if read_unseen is None:
        flags.append("all_scene_read")          # (b) 표본 없음 → (a) 로 판정
    verdict = "baseline_no_removal" if is_baseline \
        else verdict_of(primary, removed, base_read=base_read)
    if flags and verdict in DECISIVE:
        verdict += "_exploratory"

    out = {
        "tag": tag, "arm": arm, "config": cfg, "token_agg": token_agg,
        "window_mean": read,
        "window_mean_unseen_scenes": read_unseen,
        "verdict_scope": "unseen_scenes" if read_unseen is not None else "all_scenes",
        "verdict_flags": flags,
        "fixed_t": {"curve": curve,
                    "mean_auroc": float(np.mean(cur_ok)) if cur_ok else None,
                    "n_t": len(cur_ok)},
        "scene_probe_foldwise": fw,
        "scene_probe_foldwise_baseline": fold_probe_base if not is_baseline else None,
        "scene_probe": probe,
        "scene_probe_baseline": base_probe if not is_baseline else None,
        "scene_removed": removed,
        "scene_removed_split_train_ref": removed_ref,
        "baseline_window_mean_null_z": (base_read or {}).get("null_z"),
        "verdict": verdict,
    }
    if fold_summary:
        out["loso_folds"] = fold_summary
    return out


def synth(seed=0, n_scene=8, n_ep_per_scene=6, n_rec=6, T=6, D=32, scene_split=False):
    """합성 입력 (X, meta) — --smoke 자기검증용. scene 방향 + 직교 outcome 방향.

    scene 중심은 **공유 평면 span{u1,u2} 위 원형 배치**다 (scene 마다 독립 방향이면 LOSO
    부분공간이 held-out scene 성분을 원리적으로 못 지워 제거 검증이 항상 실패한다 —
    실데이터에서 scene 구조가 저차원 공유일 때를 모사). outcome 방향은 그 평면과 직교.
    `scene_split=True` 면 split 을 **scene 단위**로 준다 (뒤쪽 1/3 scene = test).
    """
    rng = np.random.default_rng(seed)
    Q3, _ = np.linalg.qr(rng.normal(size=(D, 3)))
    u1, u2, out_dir = Q3[:, 0], Q3[:, 1], Q3[:, 2]
    S = np.stack([np.cos(2 * np.pi * s / n_scene) * u1 + np.sin(2 * np.pi * s / n_scene) * u2
                  for s in range(n_scene)])
    rows, meta = [], {k: [] for k in ("episode_idx", "record_idx", "token_idx", "token_seg",
                                      "success", "scenario_seed", "layout_id", "style_id",
                                      "split", "phase_code")}
    e = 0
    for s in range(n_scene):
        scene_dir = S[s]
        for j in range(n_ep_per_scene):
            succ = int(j % 2 == 0)
            for r in range(n_rec):
                for t in range(T):
                    seg = 0 if t == 0 else (1 if t < T - 1 else 2)
                    x = rng.normal(scale=0.4, size=D) + 8.0 * scene_dir
                    x = x + (1.2 if succ else -1.2) * out_dir
                    rows.append(x)
                    meta["episode_idx"].append(e)
                    meta["record_idx"].append(r)
                    meta["token_idx"].append(t)
                    meta["token_seg"].append(seg)
                    meta["success"].append(succ)
                    meta["scenario_seed"].append(1000 + s)
                    meta["layout_id"].append(s)
                    meta["style_id"].append(s)
                    meta["phase_code"].append(0)
                    meta["split"].append((0 if s < n_scene - max(1, n_scene // 3) else 2)
                                         if scene_split else
                                         (0 if j < n_ep_per_scene - 2 else 2))
            e += 1
    X = np.asarray(rows, np.float32)
    meta = {k: np.asarray(v) for k, v in meta.items()}
    return X, meta


def main() -> int:
    ap = argparse.ArgumentParser(description="G2 — scene 잔차화 후 succ/fail read 잔존 검증")
    ap.add_argument("--x", type=Path)
    ap.add_argument("--meta", type=Path)
    ap.add_argument("--stats", type=Path, default=None)
    ap.add_argument("--ckpt-dir", type=Path, default=None, help="sae arm 용 SAE ckpt 디렉토리")
    ap.add_argument("--probe-json", type=Path, default=None,
                    help="probe_scene.py 산출 json (scene-selective feature 목록)")
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--window", type=int, default=38,
                    help="record_idx < N 만 사용 (exp5-3 drawer_right cap=38)")
    ap.add_argument("--arms", default="raw,sae,linear")
    ap.add_argument("--r-list", default="1,2,4,8")
    ap.add_argument("--sel-top-list", default="10,20,40,all")
    ap.add_argument("--linear-methods", default="between,logreg")
    ap.add_argument("--token-agg", default="future,all",
                    help="record 집계 세그먼트. 첫 항목이 1차(기본 future = G1 근거), 나머지는 민감도")
    ap.add_argument("--skip-fixed-t", action="store_true",
                    help="fixed_t 곡선 생략 (linear+logreg 는 t 마다 fold 별 logreg 를 fit 해 "
                         "가장 비싸다 — 1차 판정은 window_mean 이므로 급할 때 끌 수 있다)")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=424101)
    ap.add_argument("--split-col", default="split")
    ap.add_argument("--scene-label", default="layout_id",
                    choices=["layout_id", "style_id", "scenario_seed"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--agg-batch-rows", type=int, default=200_000)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--allow-nonscene-sae", action="store_true",
                    help="SAE ckpt 가 scene-heldout 축(split_scene)으로 학습되지 않았어도 진행 "
                         "(리뷰 #2 — 기본은 에러. 켜면 결과는 사전이 평가 scene 을 본 상태)")
    ap.add_argument("--smoke", action="store_true", help="합성 데이터 자기검증 (raw/linear 만)")
    ap.add_argument("--smoke-scene-split", action="store_true",
                    help="--smoke 합성의 split 을 scene 단위로 (미열람 scene 표본 (b) 경로 검증)")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    aggs = [a.strip() for a in args.token_agg.split(",") if a.strip()]
    for a in aggs:
        if a not in AGG_SEGS:
            raise SystemExit(f"--token-agg 는 {list(AGG_SEGS)} 중에서: {a}")
    r_list = [int(r) for r in str(args.r_list).split(",") if r.strip()]
    methods = [m.strip() for m in args.linear_methods.split(",") if m.strip()]
    sel_tops = [s.strip() for s in args.sel_top_list.split(",") if s.strip()]
    rng = np.random.default_rng(args.seed)

    if args.smoke:
        X, meta = synth(args.seed, scene_split=args.smoke_scene_split)
        arms = [a for a in arms if a != "sae"] or ["raw"]
        args.n_perm = min(args.n_perm, 30)
        args.window = 6
        src = "smoke"
    else:
        if args.x is None or args.meta is None:
            raise SystemExit("--x / --meta 필요 (또는 --smoke)")
        mz = np.load(args.meta, allow_pickle=False)
        n = len(mz["split"])
        meta = {k: mz[k] for k in mz.files if mz[k].ndim == 1 and mz[k].shape[0] == n}
        if args.split_col not in meta:
            raise SystemExit(f"meta 에 '{args.split_col}' 없음: "
                             f"{[k for k in meta if k.startswith('split')]}")
        meta["split"] = meta[args.split_col]
        xz = np.load(args.x)
        X = xz["X"]
        fp_x = str(xz["row_fingerprint"]) if "row_fingerprint" in xz.files else None
        fp_m = str(mz["row_fingerprint"]) if "row_fingerprint" in mz.files else None
        if fp_x and fp_m and fp_x != fp_m:
            raise SystemExit(f"row_fingerprint 불일치 — X={fp_x} vs meta={fp_m}")
        src = str(args.x)

    grp = RowGrouping(meta, args.window)
    ep_label, split_of_ep = {}, {}
    for e in np.unique(meta["episode_idx"]):
        m = meta["episode_idx"] == e
        rec = {"success": int(meta["success"][m][0]),
               "scenario_seed": int(meta["scenario_seed"][m][0])}
        for f in ("layout_id", "style_id"):
            if f in meta:
                rec[f] = int(meta[f][m][0])
        ep_label[int(e)] = rec
        split_of_ep[int(e)] = int(meta["split"][m][0])
    if args.scene_label not in next(iter(ep_label.values())):
        raise SystemExit(f"meta 에 --scene-label '{args.scene_label}' 없음")

    # ── 잔차화 transform 준비 (raw + SAE 변형들) — X 한 번 읽고 모두 계산
    transforms = {"raw": (lambda r: r)}
    sae_info = {}
    sae_train_scenes: set[int] = set()
    sae_seen_eval_scenes = False
    if "sae" in arms:
        if args.ckpt_dir is None or args.probe_json is None:
            raise SystemExit("sae arm 은 --ckpt-dir 와 --probe-json 이 필요")
        model, cfg_sae, mu, sd, st_src = load_sae_bundle(args.ckpt_dir, args.stats)
        # ── 리뷰 #2: SAE 사전이 **scene-heldout 축**으로 학습됐는지 (episode 축이면 사전이
        #    평가 scene 을 이미 봤다 → scene 제거 주장 자체가 순환). 기본은 에러.
        ck_split, ck_scene_axis = cfg_sae.get("split_col"), cfg_sae.get("split_axis_scene_heldout")
        axis_ok = (ck_split == "split_scene") and bool(ck_scene_axis)
        if not axis_ok:
            msg = (f"SAE ckpt 가 scene-heldout 축이 아니다: split_col={ck_split!r} "
                   f"split_axis_scene_heldout={ck_scene_axis!r} (기대 'split_scene'/True). "
                   f"scene 축 ckpt 로 재학습하거나 --allow-nonscene-sae 로 강행")
            if not args.allow_nonscene_sae:
                raise SystemExit(msg)
            print(f"[warn] {msg}", flush=True)
        dev = args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
        sel_sets, sel_meta = {"sae_full": np.zeros(0, np.int64)}, {}
        for topspec in sel_tops:
            idx, info = selective_features(args.probe_json, topspec)
            name = f"sae_top{topspec}"
            sel_sets[name] = idx
            sel_meta[name] = {"n_removed": int(len(idx)), **info}
        transforms.update(make_sae_transforms(model, mu, sd, sel_sets, dev, args.batch))
        sae_info = {"ckpt_dir": str(args.ckpt_dir), "probe_json": str(args.probe_json),
                    "stats_source": st_src, "device": dev, "m": int(cfg_sae.get("m", -1)),
                    "k": int(cfg_sae.get("k", -1)), "sets": sel_meta,
                    "note": "SAE selective 집합은 SAE train split 기준(LOSO fold 밖) — "
                            "linear arm 과 달리 fold 내 재추정이 아니다"}
        # in-sample 위험 실측: SAE 가 학습에서 본 episode/scene 이 G2 평가 집합에 얼마나 들어오나
        tr_eps = {int(v) for v in (cfg_sae.get("train_episodes") or [])}
        sae_train_scenes = {int(v) for v in (cfg_sae.get("train_scenes") or [])}
        eval_scenes = {int(v["scenario_seed"]) for v in ep_label.values()}
        seen_scenes = eval_scenes & sae_train_scenes
        sae_info.update(sae_train_episodes=len(tr_eps),
                        eval_episodes=len(ep_label),
                        eval_episodes_seen_by_sae=len(tr_eps & set(ep_label)),
                        split_col_of_sae=ck_split,
                        split_axis_scene_heldout=bool(ck_scene_axis),
                        scene_axis_ok=bool(axis_ok),
                        sae_train_scenes=len(sae_train_scenes),
                        eval_scenes=len(eval_scenes),
                        eval_scenes_seen_by_sae=len(seen_scenes),
                        eval_scene_seen_frac=(float(len(seen_scenes) / len(eval_scenes))
                                              if eval_scenes else None),
                        seen_scene_list=sorted(seen_scenes),
                        refit_note="fold 별 SAE 재학습은 하지 않는다 — SAE 1개 학습이 GPU-시간 "
                                   "단위라 scene 수만큼(수십) 반복하면 라운드가 성립하지 않는다. "
                                   "대신 사전이 평가 scene 을 본 비율을 기록하고, >0 이면 "
                                   "verdict 에 '_exploratory' 를 붙인다 (리뷰 #2).")
        sae_seen_eval_scenes = bool(seen_scenes)

    print(f"[g2] rows(window<{args.window})={grp.n_rows} subgroups={len(grp.starts)} "
          f"transforms={list(transforms)}", flush=True)
    sums = stream_group_sums(X, grp, transforms, args.agg_batch_rows)

    # (b) 표본 — 부분공간/SAE 사전이 학습에 안 쓴 scene (리뷰 #3)
    unseen = unseen_scene_set(ep_label, split_of_ep, sae_train_scenes)
    print(f"[g2] unseen scenes (train split·SAE train 밖) = {len(unseen)} / "
          f"{len({v['scenario_seed'] for v in ep_label.values()})}", flush=True)

    results = []
    fold_cache: dict = {}          # (agg, 읽기단위, method) → (r 별 사영, spy 요약)
    probe_cache: dict = {}         # (agg, method, r) → fold-wise scene probe
    for agg in aggs:
        segs = AGG_SEGS[agg]
        recs = {name: record_vectors(S, grp, segs) for name, S in sums.items()}
        V0, ep0, rc0 = recs["raw"]

        # scene probe 용 record 단위 train/test 마스크 (meta split 축)
        spl = np.asarray([split_of_ep[int(e)] for e in ep0])
        y_sc = np.asarray([ep_label[int(e)][args.scene_label] for e in ep0])
        sc_rec = np.asarray([ep_label[int(e)]["scenario_seed"] for e in ep0])
        tr_m, te_m = spl == 0, spl == 2
        has_probe = bool(tr_m.any() and te_m.any())

        def _pack(name):
            return (recs[name][0], tr_m, te_m, y_sc, sc_rec, ep0)

        def _probe(name):
            if not has_probe:
                return None
            V = recs[name][0]
            return scene_probe(V[tr_m], y_sc[tr_m], V[te_m], y_sc[te_m], seed=args.seed)

        def _fold_probe(name):
            """기준선 fold-wise probe — 사영 없음(raw / sae_full 표현 그대로)."""
            return foldwise_scene_probe(recs[name][0], sc_rec, ep0, seed=args.seed)

        def _base_read(name):
            """제거 전 기준선 read (리뷰 #4 전제조건 + raw/sae_full arm 재사용)."""
            V, epv, rcv = recs[name]
            win = int(args.window) if args.window else int(rcv.max()) + 1
            e_, wm_, _fx = episode_views(V, epv, rcv, win)
            y_, s_ = episode_labels(e_, ep_label)
            return read_with_null(wm_, y_, s_, args.n_perm, rng)

        # 기준 probe: raw/linear 는 raw, sae_topN 은 **sae_full** (재구성 손실을 통제)
        probe_raw, probe_sae_full = _probe("raw"), (_probe("sae_full")
                                                    if "sae_full" in recs else None)
        fw_raw, fw_sae_full = _fold_probe("raw"), (_fold_probe("sae_full")
                                                   if "sae_full" in recs else None)
        read_raw = _base_read("raw")
        read_sae_full = _base_read("sae_full") if "sae_full" in recs else None
        sae_flags = ("sae_saw_eval_scenes",) if sae_seen_eval_scenes else ()

        for arm in arms:
            if arm == "raw":
                results.append(run_arm(f"raw|{agg}", "raw", {}, V0, ep0, rc0, ep_label,
                                       args, rng, _pack("raw"), probe_raw, token_agg=agg,
                                       probe_override=probe_raw, fold_probe_base=fw_raw,
                                       fold_probe_override=fw_raw, read_override=read_raw,
                                       unseen_scenes=unseen))
            elif arm == "sae":
                for name in [n for n in recs if n.startswith("sae")]:
                    Vn, epn, rcn = recs[name]
                    cfg = {"set": name, **sae_info.get("sets", {}).get(name, {})}
                    is_full = name == "sae_full"
                    results.append(run_arm(
                        f"{name}|{agg}", "sae", cfg, Vn, epn, rcn, ep_label, args, rng,
                        _pack(name), probe_sae_full, token_agg=agg,
                        probe_override=probe_sae_full if is_full else None,
                        fold_probe_base=fw_sae_full,
                        fold_probe_override=fw_sae_full if is_full else None,
                        read_override=read_sae_full if is_full else None,
                        base_read=None if is_full else read_sae_full,
                        unseen_scenes=unseen, extra_flags=sae_flags))
            elif arm == "linear":
                for meth in methods:
                    for r in r_list:
                        results.append(run_arm(f"linear_{meth}_r{r}|{agg}", "linear",
                                               {"method": meth, "r": int(r)},
                                               V0, ep0, rc0, ep_label, args, rng,
                                               _pack("raw"), probe_raw, token_agg=agg,
                                               fold_cache=fold_cache, r_list=r_list,
                                               fold_probe_base=fw_raw, base_read=read_raw,
                                               unseen_scenes=unseen, probe_cache=probe_cache))

    out = {
        "source": src, "layer": int(args.layer), "window": int(args.window),
        "token_agg": aggs, "token_agg_primary": aggs[0],
        "n_rows": int(grp.n_rows), "n_episode": int(len(ep_label)),
        "n_scene": int(len({v["scenario_seed"] for v in ep_label.values()})),
        "seed": int(args.seed), "n_perm": int(args.n_perm),
        "protocol": {
            "source": "analyze_sm2_ref.py (exp5-3 within-scene) 의 loso/within_dir/perm_null 이식",
            "read": "window_mean(episode×창평균) 1차 + fixed_t 곡선",
            "record_agg": "세그먼트 평균 — 1차 future(G1: scene 은 future 토큰), 민감도 all",
            "linear_subspace": "LOSO fold 안 재추정 (held-out scene 미사용)",
            "removal_probe": "판정용 = fold-wise (read 와 같은 fold 사영·scene 라벨). "
                             "split-train 단일 사영 probe 는 참고치 (scene_probe 키).",
            "read_scope": "(a) 전체 혼재 scene = window_mean, "
                          "(b) 사전/train 미열람 scene = window_mean_unseen_scenes. "
                          "판정은 (b) 가 있으면 (b), 없으면 (a)+_exploratory.",
        },
        "unseen_scenes": sorted(int(s) for s in unseen),
        "criteria": {"z_keep": Z_KEEP, "scene_kill_frac": SCENE_KILL_FRAC,
                     "labels": {
                         "outcome_separable_from_scene":
                             f"scene 제거 성공 & succ/fail null_z > {Z_KEEP} → G3 후보",
                         "separation_was_scene":
                             "scene 제거와 함께 붕괴 → 분리는 scene 이 만든 것",
                         "no_baseline_separation":
                             f"제거 **전** 기준선(raw/sae_full) window_mean null_z ≤ {Z_KEEP} "
                             "→ 지울 분리가 애초에 없다. 제거 판정을 하지 않는다",
                         "undecidable_removal_failed":
                             "scene probe 가 안 죽음 → 제거 실패 (선형 r 부족 / selective 불충분)",
                         "undecidable_scene_probe_baseline":
                             "제거 전 scene probe 가 chance 수준 → 제거 여부 판정 불가",
                         "*_exploratory":
                             "판정 표본이 (a) 전체 scene 이거나 SAE 사전이 평가 scene 을 본 경우 "
                             "— 확증 아님 (verdict_flags 에 사유)"}},
        "sae": sae_info,
        "results": results,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print("[written]", args.out, flush=True)

    print(f"\n{'tag':28s} {'AUROC':>7s} {'null_z':>7s} {'z(b)':>6s} {'fixT':>6s} "
          f"{'scn_fw':>7s} {'rm':>4s}  verdict")
    for r in results:
        w = r["window_mean"]
        wb = r.get("window_mean_unseen_scenes") or {}
        sp = r.get("scene_probe_foldwise") or {}
        print(f"{r['tag']:28s} {_f(w.get('auroc')):>7s} {_f(w.get('null_z')):>7s} "
              f"{_f(wb.get('null_z')):>6s} {_f(r['fixed_t']['mean_auroc']):>6s} "
              f"{_f(sp.get('acc')):>7s} {str(r['scene_removed']):>4s}  {r['verdict']}")
    return 0


def _f(v, nd=3):
    return "-" if v is None else f"{v:.{nd}f}"


if __name__ == "__main__":
    raise SystemExit(main())

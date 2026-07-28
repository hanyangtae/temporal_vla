#!/usr/bin/env python
"""exp5/G1 게이트 — SAE feature 가 scene 을 인코딩하는가 (핸드아웃 §4 Phase D).

로컬 실행 전용(sklearn 사용 — 승준에는 sklearn/scipy 없음).

무엇을 재는가
  ① z-probe    : SAE 희소 활성 z(top-k 후) → scene 라벨 선형 probe, **episode held-out**
  ② X-probe    : 원본 activation(표준화) probe = 상한 (SAE 가 정보를 잃었는지)
  ③ null       : **episode 단위** 라벨 순열 (≥100회) → 우연 수준 분포, z-score
  ④ selectivity: feature 별 클래스-조건 활성률 → scene-selective feature 목록/비율

scene 라벨 = **layout_id** (5 클래스).
  - scenario_seed 는 **episode 당 1개**라 클래스 = 표본이 되어 probe 불가 (핸드아웃 §6-1).
  - style_id 는 fit30 drawer_left 에서 layout_id 와 **완전 공선** → 같은 분할. 결과 json 에 명기.

길이 confound 통제 (핸드아웃 §6-3): 실패 episode 는 timeout 이라 record 가 훨씬 많다
(drawer_left: fail 13ep 이 전체 행의 73%). 기본으로 **episode 당 record 수를 균등
subsample** 한다(--records-per-ep, 기본 auto = 전 episode 최소 record 수). record 단위로
뽑고 그 record 의 T 토큰은 전부 유지한다 — 토큰 평균 금지 원칙 유지.

G1 판정 (§4-D3, 사전 등록):
  (a) held-out(test split) z-probe 정확도가 순열 null 대비 z > 3
  (b) 우연보정 회복률 (acc_z − chance) / (acc_X − chance) ≥ 0.80
  (c) scene-selective feature 비율 < 0.30
  세 조건 모두 만족 → G1 pass.

사용:
  python scripts/scene_sae/probe_scene.py --ckpt-dir <L10_m6144_k32_s0> \
      --x .../X_L10.npz --stats .../stats_L10.npz --meta .../meta.npz --out probe_L10.json
  python scripts/scene_sae/probe_scene.py --smoke        # 합성 데이터 dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# probe v2 (2026-07-27) — G1 1차 probe 결함 3건 수정 (docs/steering/31 §5):
#   ① 표준화 없음 + max_iter=100 → 6144-d 희소 z 에서 미수렴(ConvergenceWarning 다수).
#      → train-split StandardScaler + max_iter 기본 1000 + episode-group CV 로 C 선택.
#   ② state(1)/future(32)/action(16) 토큰을 한 행집합에 섞음 (노름 6배 차) → 세그먼트별 probe 병행.
#   ③ 행 정확도만 보고 → episode 다수결 정확도 병기 (독립 표본 = episode).
#
# probe v3 (2026-07-27 코드리뷰) — 절차 누수·불일치 5건 수정:
#   ④ 표준화를 train+CV fold 밖에서 한 번에 하던 것 → sklearn Pipeline(StandardScaler+LR)
#      로 감싸 **fold train 에서만** scaler 를 fit (리뷰 #6).
#   ⑤ 순열 null 이 train/test 라벨을 함께 섞고 별도 max_iter 를 쓰던 것 → **train 라벨만**
#      episode 단위로 섞어 같은 Pipeline·같은 C·같은 max_iter 로 재fit 후 **진짜 test
#      라벨**로 평가 (리뷰 #7).
#   ⑥ 표준화 통계를 빌더 stats(다른 split 축)에서 읽던 것 → ckpt 의 stats.npz 우선 (리뷰 #1).
#   ⑦ ckpt 의 split_col/train 지문과 대조 (리뷰 #2), 라벨↔split 호환 가드 (리뷰 #4),
#      --window 로 record_idx 상한 통제 (리뷰 #5).
C_GRID_DEFAULT = (0.01, 0.1, 1.0, 10.0)
SEG_NAMES = {0: "state", 1: "future", 2: "action"}

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# ------------------------------------------------------------------ subsample
def balanced_record_mask(meta: dict, records_per_ep: int, seed: int) -> np.ndarray:
    """episode 당 record 수를 균등하게 맞추는 행 마스크 (record 단위 추출, 토큰 전부 유지)."""
    ep = meta["episode_idx"]
    rec = meta["record_idx"]
    rng = np.random.default_rng(seed)
    eps = np.unique(ep)
    n_avail = {int(e): int(rec[ep == e].max()) + 1 for e in eps}
    n_take = records_per_ep if records_per_ep > 0 else min(n_avail.values())
    mask = np.zeros(len(ep), dtype=bool)
    for e in eps:
        avail = n_avail[int(e)]
        take = min(n_take, avail)
        sel = np.sort(rng.choice(avail, size=take, replace=False))
        mask |= (ep == e) & np.isin(rec, sel)
    return mask, n_take


# ----------------------------------------------------------------------- SAE z
def load_sae(ckpt_dir: Path):
    """train_scene_sae.py 산출 디렉토리 → (model, cfg). src/sae 접점은 여기 하나."""
    from src.sae.models import build_model             # noqa: PLC0415
    from train_scene_sae import build_sae_cfg          # noqa: PLC0415  (동일 디렉토리)

    cfg = json.loads((ckpt_dir / "config.json").read_text())
    sae_cfg = cfg.get("sae_cfg") or build_sae_cfg(cfg["input_dim"], cfg["m"], cfg["k"])
    model = build_model(sae_cfg)
    model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location="cpu"))
    model.eval()
    return model, cfg


def compute_z(model, X: np.ndarray, rows: np.ndarray, mu, sd, device: str, batch: int):
    """SAE 희소 활성 z [n_rows, m] (float32 dense — subsample 후라 감당 가능)."""
    from src.sae.train import encode_all               # noqa: PLC0415

    a = X[rows].astype(np.float32)
    a = (a - mu) / sd
    return encode_all(model, a, batch_size=batch, device=device)


# ---------------------------------------------------------------------- probes
def make_probe_pipeline(seed: int, max_iter: int, C: float, solver: str = "lbfgs") -> Pipeline:
    """StandardScaler + 다항 로지스틱 (probe v3 ④ — 리뷰 #6).

    z 는 top-k 후라 대부분 0 이고 살아있는 열끼리도 스케일이 크게 다르다. 표준화 없이
    lbfgs 에 넣으면 조건수가 나빠 max_iter 안에 수렴하지 못한다 (31 §5-1 실측).
    분산 0(dead) 열은 sklearn 이 scale=1 로 두므로 상수 0 열로 남는다.

    **Pipeline 인 이유**: 표준화를 미리 전 train 행에 한 번 하면 CV fold 의 held-out 부분
    통계가 fold train 에 새어 든다. Pipeline 으로 감싸면 `fit` 이 호출되는 행 집합에서만
    scaler 가 fit 된다 — CV·본 fit·순열 null 이 전부 같은 절차가 된다.

    기본 solver=lbfgs — 실측 3072-d/2000행에서 saga 10.3s vs lbfgs 1.1s (동일 정확도).
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=max_iter, C=C, solver=solver,
                                   n_jobs=(-1 if solver in ("saga", "lbfgs") else None),
                                   random_state=seed, tol=1e-3)),
    ])


def fit_probe(Ztr, ytr, Zte, yte, seed: int, max_iter: int, C: float, solver: str = "lbfgs"):
    """probe 1회 fit (**원시 행렬 입력** — 표준화는 Pipeline 안에서 train 행만으로).

    [반환] (test acc, pipeline, ConvergenceWarning 개수) — 미수렴이 결과를 만든 적이 있어
    (31 §5-1) 경고 개수를 세서 결과 json 에 남긴다.
    """
    clf = make_probe_pipeline(seed, max_iter, C, solver)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(Ztr, ytr)
    n_warn = sum(1 for w in caught if issubclass(w.category, ConvergenceWarning))
    return float((clf.predict(Zte) == yte).mean()), clf, n_warn


def chance_level(y) -> float:
    _, cnt = np.unique(y, return_counts=True)
    return float(cnt.max() / cnt.sum())


def episode_majority_acc(pred, y, ep):
    """episode 다수결 정확도 (probe v2 ③).

    행은 같은 episode 안에서 강하게 자기상관이라 행 정확도는 유효표본 수를 과대표현한다.
    episode 별 예측 최빈값 vs 그 episode 의 라벨로 '독립 표본 단위' 정확도를 함께 보고한다.
    """
    eps = np.unique(ep)
    ok = 0
    for e in eps:
        m = ep == e
        vals, cnt = np.unique(pred[m], return_counts=True)
        ok += int(vals[int(np.argmax(cnt))] == y[m][0])
    return (float(ok / len(eps)) if len(eps) else None), int(len(eps))


def usable_cv_folds(splits, ytr):
    """전 클래스가 fold train 에 다 있는 fold 만 남긴다 (probe v3 — 리뷰 #6).

    클래스가 빠진 fold 에서 fit 하면 그 클래스는 예측 자체가 불가능해 held-out 정확도가
    구조적으로 깎인다 → C 비교가 fold 구성 우연에 좌우된다. 그런 fold 는 아예 뺀다.
    [반환] (사용 가능한 fold 목록, 스킵 사유 리스트)
    """
    classes = set(int(v) for v in np.unique(ytr))
    keep, skipped = [], []
    for i, (tr, va) in enumerate(splits):
        have = set(int(v) for v in np.unique(ytr[tr]))
        missing = sorted(classes - have)
        if len(have) < 2 or missing:
            skipped.append({"fold": i, "missing_classes": missing, "n_train_classes": len(have)})
            continue
        keep.append((tr, va))
    return keep, skipped


def select_C(Ztr, ytr, ep_tr, C_grid, seed, max_iter, solver, folds=3, verbose=True):
    """train 내부 **episode-group** K-fold CV 로 C 선택 (probe v2 ①, v3 ④).

    record/행 단위 fold 는 같은 episode 가 train·val 양쪽에 들어가 누수된다 → GroupKFold.
    각 fold 는 Pipeline 으로 fit 되므로 표준화도 fold train 안에서만 일어난다.
    클래스가 빠진 fold 는 스킵하고, 전부 스킵되면 C=1 로 fallback + 경고.
    동점이면 더 강한 정규화(작은 C)를 고른다(grid 오름차순 + argmax 가 첫 최대를 잡음).
    [반환] (best_C, detail dict, 경고 수)
    """
    grid = [float(c) for c in C_grid]
    n_groups = int(len(np.unique(ep_tr)))
    k = int(min(folds, n_groups))
    if len(grid) == 1 or k < 2:
        return grid[0], {"folds": k, "grid": grid, "mean_acc": None,
                         "note": "CV 생략 (C 후보 1개 또는 group 부족)"}, 0
    gkf = GroupKFold(n_splits=k)
    splits = list(gkf.split(Ztr, ytr, groups=ep_tr))
    splits, skipped = usable_cv_folds(splits, ytr)
    if not splits:
        msg = (f"CV fold 전부 스킵 (클래스 누락) → C=1.0 fallback. 스킵 {len(skipped)}개")
        print(f"  [cv] ⚠ {msg}", flush=True)
        return 1.0, {"folds": k, "grid": grid, "mean_acc": None, "skipped_folds": skipped,
                     "fallback_C": 1.0, "note": msg}, 0
    if skipped and verbose:
        print(f"  [cv] fold {len(skipped)}개 스킵 (클래스 누락): "
              f"{[s['missing_classes'] for s in skipped][:3]}", flush=True)
    warn = 0
    means = []
    for C in grid:
        accs = []
        for tr, va in splits:
            acc, _clf, w = fit_probe(Ztr[tr], ytr[tr], Ztr[va], ytr[va],
                                     seed, max_iter, C, solver)
            accs.append(acc)
            warn += w
        means.append(float(np.mean(accs)) if accs else float("nan"))
    best = int(np.nanargmax(means))
    if verbose:
        print("  [cv] " + "  ".join(f"C={c:g}:{m:.3f}" for c, m in zip(grid, means))
              + f"  → C={grid[best]:g}", flush=True)
    return grid[best], {"folds": k, "grid": grid, "mean_acc": means,
                        "n_folds_used": len(splits), "skipped_folds": skipped}, warn


def run_probe(A_tr, ytr, ep_tr, A_te, yte, ep_te, args, C=None, verbose=True):
    """표준화된 행렬 하나에 대한 probe 1회 (C 선택 → fit → 행/episode 정확도).

    [반환] (결과 dict, clf, test 예측)
    """
    warn = 0
    cv = None
    if C is None:
        C, cv, w = select_C(A_tr, ytr, ep_tr, args.C_grid, args.seed,
                            args.max_iter, args.solver, args.cv_folds, verbose)
        warn += w
    acc, clf, w = fit_probe(A_tr, ytr, A_te, yte, args.seed, args.max_iter, C, args.solver)
    warn += w
    pred_te, pred_tr = clf.predict(A_te), clf.predict(A_tr)
    ep_acc, n_ep = episode_majority_acc(pred_te, yte, ep_te)
    ep_acc_tr, n_ep_tr = episode_majority_acc(pred_tr, ytr, ep_tr)
    res = {
        "test_acc": acc, "train_acc": float((pred_tr == ytr).mean()),
        "episode_test_acc": ep_acc, "n_test_episodes": n_ep,
        "episode_train_acc": ep_acc_tr, "n_train_episodes": n_ep_tr,
        "chance_test": chance_level(yte),
        "n_train_rows": int(len(ytr)), "n_test_rows": int(len(yte)),
        "C": float(C), "cv": cv, "max_iter": int(args.max_iter),
        "n_convergence_warnings": int(warn),
    }
    return res, clf, pred_te


def permutation_null(Ztr, Zte, ep_tr, ep_te, ytr_true, yte_true, n_perm, seed, max_iter, C,
                     solver="lbfgs", verbose=True):
    """**train episode 라벨만** 순열 → 같은 Pipeline 으로 재fit → **진짜 test 라벨**로 평가.

    (probe v3 ⑤ — 리뷰 #7. 구 버전은 train·test 라벨을 같은 사상으로 함께 섞고 별도
    --null-max-iter 를 썼다. test 라벨까지 섞으면 "라벨 사상 자체를 바꾼 세계"의 정확도가
    되어 본 probe 와 다른 양을 재게 되고, max_iter 가 다르면 수렴도까지 달라진다.
    올바른 귀무가설 = "train 에서 배운 것이 진짜 라벨과 무관" → train 라벨만 섞는다.)

    행 단위가 아니라 **episode 단위** 순열인 이유: 행 순열은 같은 episode 안의 자기상관을
    깨서 null 을 과소평가한다 (핸드아웃 §6-2 동형).
    val 은 본 probe 와 동일하게 아예 등장하지 않는다 (train/test 만 사용).
    C·max_iter 는 본 probe 가 쓴 값 그대로 (순열마다 CV 재실행은 비용이 n_perm 배).
    [반환] (acc 배열, ConvergenceWarning 총 개수)
    """
    rng = np.random.default_rng(seed)
    eps = np.asarray(sorted({int(e) for e in ep_tr}))
    lab_of = {int(e): int(ytr_true[np.argmax(ep_tr == e)]) for e in eps}
    labs = np.asarray([lab_of[int(e)] for e in eps])
    accs = []
    warn = 0
    t0 = time.time()
    for i in range(n_perm):
        perm = rng.permutation(len(labs))
        m = {int(e): int(labs[p]) for e, p in zip(eps, perm)}
        ytr = np.asarray([m[int(e)] for e in ep_tr])
        if len(np.unique(ytr)) < 2:
            continue
        acc, _clf, w = fit_probe(Ztr, ytr, Zte, yte_true, 0, max_iter, C, solver)
        accs.append(acc)
        warn += w
        if verbose and (i + 1) % 10 == 0:
            print(f"  [null] {i+1}/{n_perm} mean={np.mean(accs):.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return np.asarray(accs), warn


def feature_selectivity(Z, y, min_rate: float, ratio: float, top: int):
    """feature 별 클래스-조건 활성률 → scene-selective 판정.

    selective = (최대 클래스 활성률 ≥ min_rate) and (최대 / 차순위 ≥ ratio).
    단일-feature probe 보다 싸고, top-k 코드에서 해석이 직접적이다(어떤 scene 에서만 켜지나).
    """
    classes = np.unique(y)
    act = (Z > 0)
    rates = np.stack([act[y == c].mean(axis=0) for c in classes], axis=1)  # [m, C]
    order = np.sort(rates, axis=1)[:, ::-1]
    top1, top2 = order[:, 0], order[:, 1] if rates.shape[1] > 1 else np.zeros(len(order))
    live = act.any(axis=0)
    sel = live & (top1 >= min_rate) & (top1 >= ratio * np.maximum(top2, 1e-6))
    score = np.where(live, top1 - top2, 0.0)
    idx = np.argsort(-score)[:top]
    return {
        "n_features": int(Z.shape[1]),
        "n_live": int(live.sum()),
        "n_selective": int(sel.sum()),
        "selective_frac_of_all": float(sel.sum() / Z.shape[1]),
        "selective_frac_of_live": float(sel.sum() / max(1, live.sum())),
        "criteria": {"min_rate": min_rate, "ratio": ratio},
        "classes": [int(c) for c in classes],
        "top_features": [
            {"feature": int(j), "best_class": int(classes[int(np.argmax(rates[j]))]),
             "rates": [round(float(r), 4) for r in rates[j]],
             "gap": round(float(score[j]), 4), "selective": bool(sel[j])}
            for j in idx],
    }


# ------------------------------------------------------------------- 판정/출력
def judge(res: dict, args) -> dict:
    z_score = res["z_probe"]["null_z"]
    rec = res["recovery_chance_corrected"]
    selfrac = res["selectivity"]["selective_frac_of_all"]
    crit = {
        "a_null_z_gt_3": bool(z_score is not None and z_score > 3.0),
        "b_recovery_ge_0.80": bool(rec is not None and rec >= 0.80),
        "c_selective_frac_lt_0.30": bool(selfrac < 0.30),
    }
    return {"criteria": crit, "pass": all(crit.values()),
            "values": {"null_z": z_score, "recovery": rec, "selective_frac": selfrac}}


def print_table(res: dict) -> None:
    p = res["z_probe"]
    x = res["x_probe"]
    print("\n===== G1 scene probe (라벨 = layout_id) =====")
    print(f"  클래스 {res['classes']}  chance(test) = {res['chance_test']:.3f}")
    print(f"  행: train {res['n_train_rows']}  test {res['n_test_rows']}  "
          f"(episode 당 record {res['records_per_ep']} 균등 subsample"
          + (f", window record_idx<{res['window']}" if res.get("window") else "") + ")")
    dc = res.get("data_check") or {}
    if dc:
        print(f"  split_col={dc.get('split_col')} scene_heldout={dc.get('scene_heldout')} "
              f"stats={Path(str(dc.get('stats_source'))).name if dc.get('stats_source') else '-'}")
        for w in dc.get("warnings") or []:
            print(f"  ⚠ {w}")
    print(f"  {'probe':22s} {'test acc':>9s} {'train acc':>10s} {'ep다수결':>9s} {'C':>6s}")
    print(f"  {'① SAE z':22s} {p['test_acc']:9.3f} {p['train_acc']:10.3f} "
          f"{p['episode_test_acc']:9.3f} {p['C']:6g}  (test ep {p['n_test_episodes']})")
    print(f"  {'② 원본 X (상한)':22s} {x['test_acc']:9.3f} {x['train_acc']:10.3f} "
          f"{x['episode_test_acc']:9.3f} {x['C']:6g}")
    if p["null_mean"] is not None:
        print(f"  {'③ 순열 null(z)':22s} {p['null_mean']:9.3f} ± {p['null_std']:.3f}"
              f"  → z = {p['null_z']:.2f} (n={p['n_perm']})")
    print(f"  회복률(우연보정) = {res['recovery_chance_corrected']}")
    print(f"  success 층화 test acc: succ={res['stratified']['succ_acc']} "
          f"fail={res['stratified']['fail_acc']}")
    s = res["selectivity"]
    print(f"  ④ selective feature {s['n_selective']}/{s['n_features']} "
          f"({s['selective_frac_of_all']:.3f}, live 중 {s['selective_frac_of_live']:.3f})")
    if res.get("per_segment"):
        print(f"  ⑤ 세그먼트별 (scene 이 어느 토큰에 사는가)")
        print(f"     {'seg':8s} {'토큰수':>6s} {'행(tr/te)':>15s} {'z acc':>7s} {'X acc':>7s} "
              f"{'회복률':>7s} {'z ep':>6s}")
        for name, b in res["per_segment"].items():
            zp, xp = b["z_probe"], b["x_probe"]
            rec = "None" if b["recovery_chance_corrected"] is None \
                else f"{b['recovery_chance_corrected']:.3f}"
            print(f"     {name:8s} {b['n_tokens']:6d} "
                  f"{zp['n_train_rows']:7d}/{zp['n_test_rows']:<7d} {zp['test_acc']:7.3f} "
                  f"{xp['test_acc']:7.3f} {rec:>7s} {zp['episode_test_acc']:6.3f}")
    print(f"  수렴 경고 총 {res['n_convergence_warnings']} 건 "
          f"(max_iter={res['max_iter']})")
    j = res["judgment"]
    print(f"  판정: {'PASS' if j['pass'] else 'FAIL'}  {j['criteria']}")


# -------------------------------------------------------------------- 합성 smoke
def synth(seed=0, n_ep=20, n_rec=20, T=8, D=64, m=256, k=8):
    """layout 신호가 실제로 들어 있는 합성 데이터 — 파이프라인 dry-run 용."""
    rng = np.random.default_rng(seed)
    n_lay = 4
    dirs = rng.normal(size=(n_lay, D))
    ep_lay, ep_succ, Xs, meta = {}, {}, [], {kk: [] for kk in
                                            ("episode_idx", "record_idx", "token_idx",
                                             "token_seg", "success", "layout_id", "style_id",
                                             "scenario_seed", "split")}
    for e in range(n_ep):
        lay = e % n_lay
        succ = int(e % 3 == 0)
        recs = n_rec if succ else int(n_rec * 1.8)      # 실패가 길다 (길이 confound 모사)
        ep_lay[e], ep_succ[e] = lay, succ
        x = rng.normal(size=(recs * T, D)) + 2.0 * dirs[lay]
        Xs.append(x.astype(np.float32))
        # split 은 layout 안에서 배분한다 — layout 과 split 이 얽히면 test 에 한 클래스만
        # 남아 probe 가 무의미해진다 (실제 빌더의 층화 split 과 같은 취지).
        pos = e // n_lay
        n_pos = n_ep // n_lay
        split = 0 if pos < n_pos - 2 else (1 if pos == n_pos - 2 else 2)
        meta["episode_idx"].append(np.full(recs * T, e, np.int32))
        meta["record_idx"].append(np.repeat(np.arange(recs, dtype=np.int32), T))
        meta["token_idx"].append(np.tile(np.arange(T, dtype=np.int16), recs))
        meta["token_seg"].append(np.tile(np.arange(T, dtype=np.int8) // 4, recs))
        meta["success"].append(np.full(recs * T, succ, np.int8))
        meta["layout_id"].append(np.full(recs * T, lay, np.int32))
        meta["style_id"].append(np.full(recs * T, lay, np.int32))
        meta["scenario_seed"].append(np.full(recs * T, 100000 + e, np.int64))
        meta["split"].append(np.full(recs * T, split, np.int8))
    X = np.concatenate(Xs)
    meta = {kk: np.concatenate(v) for kk, v in meta.items()}
    W = rng.normal(size=(D, m)) / np.sqrt(D)
    return X, meta, W, k


def topk_encode(Xs: np.ndarray, W: np.ndarray, k: int) -> np.ndarray:
    h = np.maximum(Xs @ W, 0.0)
    thr = np.partition(h, -k, axis=1)[:, -k][:, None]
    return np.where(h >= thr, h, 0.0)


# ------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description="G1 scene probe (SAE feature vs layout)")
    ap.add_argument("--ckpt-dir", type=Path, help="train_scene_sae.py 산출 디렉토리")
    ap.add_argument("--x", type=Path)
    ap.add_argument("--stats", type=Path)
    ap.add_argument("--meta", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="결과 json 경로")
    ap.add_argument("--label", default="layout_id",
                    choices=["layout_id", "style_id", "scenario_seed"],
                    help="scenario_seed(scene-matched 20클래스)는 --split-col split_episode "
                         "와 함께 써야 한다 — scene held-out 축에서는 test 클래스가 train 에 없다")
    ap.add_argument("--records-per-ep", type=int, default=0,
                    help="episode 당 record 수 (0=auto: 최소값). 0 미만이면 균등화 끔")
    ap.add_argument("--window", type=int, default=0,
                    help="record_idx < N 행만 사용 (0=끔·기존 동작). 실패는 timeout 이라 "
                         "후반 record 가 실패에만 존재 → 상태 confound 통제용. "
                         "drawer_right 예정값 38 (리뷰 #5)")
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--max-iter", type=int, default=1000,
                    help="probe v2: 기본 1000 (구 200 은 6144-d 에서 미수렴 — 31 §5-1)")
    # (구 --null-max-iter 제거 — 순열 null 은 본 probe 와 **동일 절차**여야 한다. 리뷰 #7)
    ap.add_argument("--solver", default="lbfgs", choices=["lbfgs", "saga", "liblinear"])
    ap.add_argument("--C", type=float, default=None,
                    help="주면 CV 를 건너뛰고 이 C 로 고정 (기본 = episode-group CV 선택)")
    ap.add_argument("--C-grid", default=",".join(f"{c:g}" for c in C_GRID_DEFAULT),
                    help="CV 후보 C 목록 (쉼표 구분)")
    ap.add_argument("--cv-folds", type=int, default=3, help="train 내부 GroupKFold fold 수")
    ap.add_argument("--no-segments", action="store_true",
                    help="세그먼트별 probe 생략 (probe v2 기본은 실행)")
    ap.add_argument("--sel-min-rate", type=float, default=0.10)
    ap.add_argument("--sel-ratio", type=float, default=3.0)
    ap.add_argument("--sel-top", type=int, default=30)
    ap.add_argument("--smoke", action="store_true", help="합성 데이터 dry-run (ckpt 불필요)")
    ap.add_argument("--split-col", default="split",
                    help="meta.npz 의 split 컬럼. scene-matched 빌드는 split_episode/split_scene "
                         "동봉. ⚠ --label scenario_seed 는 split_episode 축에서만 평가 가능 "
                         "(scene held-out 이면 test 클래스가 train 에 없음)")
    ap.add_argument("--allow-split-mismatch", action="store_true",
                    help="ckpt config.json 의 split_col/train 지문과 달라도 진행 (리뷰 #2). "
                         "SAE 가 이 probe 의 test 행을 학습에 봤을 수 있다 = in-sample 위험")
    args = ap.parse_args()
    args.C_grid = ([float(args.C)] if args.C is not None
                   else [float(c) for c in str(args.C_grid).split(",") if c.strip()])

    t0 = time.time()
    data_check: dict = {"split_col": args.split_col, "window": int(args.window)}
    if args.smoke:
        X, meta, W, k = synth(args.seed)
        mu = X[meta["split"] == 0].mean(0)
        sd = X[meta["split"] == 0].std(0)
        sd = np.where(sd < 1e-6, 1.0, sd)
        cfg, src = {"m": W.shape[1], "k": k, "layer": -1, "cell": "SMOKE"}, "smoke"
    else:
        for need in ("ckpt_dir", "x", "meta"):
            if getattr(args, need) is None:
                raise SystemExit(f"--{need.replace('_','-')} 필요 (또는 --smoke)")
        meta_npz = np.load(args.meta, allow_pickle=False)
        if args.split_col not in meta_npz.files:
            raise SystemExit(f"meta 에 '{args.split_col}' 없음. 사용 가능: "
                             f"{[k for k in meta_npz.files if k.startswith('split')]}")
        meta = {kk: meta_npz[kk] for kk in meta_npz.files
                if meta_npz[kk].ndim == 1 and meta_npz[kk].shape[0] == len(meta_npz["split"])}
        if args.split_col != "split":                 # 선택한 축을 기본 split 자리에 올린다
            meta["split"] = meta[args.split_col]
        Xz = np.load(args.x)
        X = Xz["X"]
        # 행 지문 대조 (리뷰 #10)
        fp_x = str(Xz["row_fingerprint"]) if "row_fingerprint" in Xz.files else None
        fp_m = str(meta_npz["row_fingerprint"]) if "row_fingerprint" in meta_npz.files else None
        if fp_x is not None and fp_m is not None and fp_x != fp_m:
            raise SystemExit(f"row_fingerprint 불일치 — X={fp_x} vs meta={fp_m} "
                             f"(다른 빌드 산출물이 섞였다)")
        data_check["row_fingerprint"] = fp_m
        if fp_x is None or fp_m is None:
            print("[warn] row_fingerprint 없음 (구 빌드 산출물) — 행 정합 대조 생략", flush=True)
        # 표준화 통계: ckpt 옆 stats.npz 우선 (학습이 실제로 쓴 통계 — 리뷰 #1)
        ck_stats = Path(args.ckpt_dir) / "stats.npz"
        if ck_stats.exists():
            st = np.load(ck_stats)
            mu, sd = st["mean"], st["std"]
            data_check["stats_source"] = str(ck_stats)
        elif args.stats is not None:
            st = np.load(args.stats)
            mu, sd = st["mean"], st["std"]
            data_check["stats_source"] = str(args.stats)
            print(f"[warn] ckpt 에 stats.npz 없음 (구 학습 산출물) — 빌더 stats 사용: "
                  f"{args.stats}", flush=True)
        else:
            raise SystemExit(f"표준화 통계 없음: {ck_stats} 도 --stats 도 없다")

    if args.records_per_ep >= 0:
        sub, rpe = balanced_record_mask(meta, args.records_per_ep, args.seed)
    else:
        sub, rpe = np.ones(len(meta["split"]), bool), -1
    # --window: record_idx < N 행만 (timeout 상태 confound 통제 — 리뷰 #5)
    if args.window and args.window > 0:
        sub = sub & (meta["record_idx"] < int(args.window))
        if sub.sum() == 0:
            raise SystemExit(f"--window {args.window} 적용 후 행 0")

    y_all = meta[args.label]
    ep_all = meta["episode_idx"]
    split = meta["split"]
    # style/layout 공선 진단 (라벨 선택 근거를 결과에 남긴다)
    pairs = {(int(a), int(b)) for a, b in zip(meta["layout_id"], meta["style_id"])}
    collinear = len(pairs) == len(set(int(a) for a, _ in pairs))
    seeds_per_ep = {int(e): int(np.unique(meta["scenario_seed"][ep_all == e])[0])
                    for e in np.unique(ep_all)}

    rows_tr = sub & (split == 0)
    rows_te = sub & (split == 2)
    if rows_te.sum() == 0:
        raise SystemExit("test split 행 0 — split 배정 확인")

    # ---- 리뷰 #4: 라벨 ↔ split 축 호환 가드 (scene held-out 여부는 실측으로 판정)
    has_scene = "scenario_seed" in meta
    sc_tr = {int(v) for v in meta["scenario_seed"][split == 0]} if has_scene else set()
    sc_te = {int(v) for v in meta["scenario_seed"][split == 2]} if has_scene else set()
    scene_heldout = bool(sc_te) and not (sc_tr & sc_te)
    data_check["scene_heldout"] = scene_heldout
    data_check["n_scenes_train"], data_check["n_scenes_test"] = len(sc_tr), len(sc_te)
    warns: list[str] = []
    if args.label == "scenario_seed" and scene_heldout:
        raise SystemExit(
            "라벨 scenario_seed × scene held-out split 조합 불가 — test scene 클래스가 "
            f"train 에 하나도 없다 (train {len(sc_tr)}개 ∩ test {len(sc_te)}개 = 0). "
            "--split-col split_episode 로 바꾸거나 --label layout_id 를 쓸 것.")
    if args.label == "layout_id" and has_scene and not scene_heldout:
        w = ("layout_id 라벨 × scene 공유 split — 같은 scene 이 train/test 양쪽에 있어 "
             "layout 정확도가 'scene 암기'로 과대평가될 수 있다 (--split-col split_scene 권장)")
        warns.append(w)
        print(f"[warn] {w}", flush=True)

    # ---- 리뷰 #2: ckpt 의 split 축·train 집합과 대조
    tr_eps_probe = sorted({int(v) for v in ep_all[split == 0]})
    fp_tr_probe = hashlib.sha256(
        ",".join(str(e) for e in tr_eps_probe).encode("utf-8")).hexdigest()[:12]
    data_check["train_episode_fingerprint_probe"] = fp_tr_probe
    if not args.smoke:
        ck_cfg = json.loads((Path(args.ckpt_dir) / "config.json").read_text())
        ck_split = ck_cfg.get("split_col")
        ck_fp = ck_cfg.get("train_episode_fingerprint")
        data_check["ckpt_split_col"] = ck_split
        data_check["ckpt_train_episode_fingerprint"] = ck_fp
        bad = []
        if ck_split is not None and ck_split != args.split_col:
            bad.append(f"split_col: ckpt={ck_split} vs probe={args.split_col}")
        if ck_fp is not None and ck_fp != fp_tr_probe:
            bad.append(f"train 지문: ckpt={ck_fp} vs probe={fp_tr_probe}")
        if ck_split is None and ck_fp is None:
            warns.append("구 ckpt (split_col/train 지문 없음) — 대조 생략")
            print("[warn] ckpt config.json 에 split_col/train 지문 없음 — 대조 생략", flush=True)
        if bad:
            msg = ("ckpt 와 probe 의 split 이 다르다 — " + " | ".join(bad) +
                   ". SAE 가 이 probe 의 test 행을 학습에 봤을 수 있다(in-sample).")
            if not args.allow_split_mismatch:
                raise SystemExit(msg + " 강행하려면 --allow-split-mismatch.")
            warns.append("ALLOWED MISMATCH: " + msg)
            print(f"[warn] {msg} (--allow-split-mismatch 로 강행)", flush=True)
        data_check["split_mismatch_allowed"] = bool(args.allow_split_mismatch and bad)
    data_check["warnings"] = warns

    if args.smoke:
        Xs = ((X - mu) / sd).astype(np.float32)
        Z_tr = topk_encode(Xs[rows_tr], W, k)
        Z_te = topk_encode(Xs[rows_te], W, k)
        X_tr, X_te = Xs[rows_tr], Xs[rows_te]
    else:
        model, cfg = load_sae(args.ckpt_dir)
        src = cfg.get("sae_source", "src.sae")
        Z_tr = compute_z(model, X, rows_tr, mu, sd, args.device, args.batch)
        Z_te = compute_z(model, X, rows_te, mu, sd, args.device, args.batch)
        X_tr = ((X[rows_tr].astype(np.float32) - mu) / sd)
        X_te = ((X[rows_te].astype(np.float32) - mu) / sd)

    ytr, yte = y_all[rows_tr], y_all[rows_te]
    ep_tr, ep_te = ep_all[rows_tr], ep_all[rows_te]
    seg_tr, seg_te = meta["token_seg"][rows_tr], meta["token_seg"][rows_te]
    print(f"[probe] rows train={len(ytr)} test={len(yte)} m={Z_tr.shape[1]} "
          f"classes={sorted(set(int(v) for v in y_all))}", flush=True)

    # --- probe v3 ④: 표준화는 Pipeline 안(fit 되는 행 집합)에서만 — 원시 행렬을 그대로 넘긴다
    sel = feature_selectivity(Z_tr, ytr, args.sel_min_rate, args.sel_ratio, args.sel_top)

    print("[probe] ① SAE z (전체 토큰)", flush=True)
    zres, clf_z, pred_te = run_probe(Z_tr, ytr, ep_tr, Z_te, yte, ep_te, args)
    print("[probe] ② 원본 X (상한)", flush=True)
    xres, _clf_x, _ = run_probe(X_tr, ytr, ep_tr, X_te, yte, ep_te, args)
    acc_z, acc_x = zres["test_acc"], xres["test_acc"]
    chance = chance_level(yte)
    n_warn = zres["n_convergence_warnings"] + xres["n_convergence_warnings"]

    if args.n_perm > 0:
        # 리뷰 #7: train 라벨만 순열 → 같은 C·같은 max_iter·같은 Pipeline → 진짜 test 라벨 평가
        nulls, w = permutation_null(Z_tr, Z_te, ep_tr, ep_te, ytr, yte, args.n_perm,
                                    args.seed, args.max_iter, zres["C"], args.solver)
        n_warn += w
    else:
        nulls = np.asarray([])

    # --- probe v2 ②: 세그먼트별 (state 1 / future 32 / action 16 토큰, 노름 6배 차)
    per_segment = {}
    if not args.no_segments:
        for s in sorted(set(int(v) for v in np.unique(seg_tr)) & set(int(v) for v in np.unique(seg_te))):
            name = SEG_NAMES.get(s, f"seg{s}")
            mtr, mte = seg_tr == s, seg_te == s
            if mte.sum() == 0 or len(np.unique(ytr[mtr])) < 2:
                continue
            print(f"[probe] ⑤ 세그먼트 {name} (tr {int(mtr.sum())} / te {int(mte.sum())})",
                  flush=True)
            zr, _c, _p = run_probe(Z_tr[mtr], ytr[mtr], ep_tr[mtr], Z_te[mte],
                                   yte[mte], ep_te[mte], args)
            xr, _c, _p = run_probe(X_tr[mtr], ytr[mtr], ep_tr[mtr], X_te[mte],
                                   yte[mte], ep_te[mte], args)
            ch = chance_level(yte[mte])
            den = xr["test_acc"] - ch
            per_segment[name] = {
                "seg_id": s,
                "n_tokens": int(np.unique(meta["token_idx"][rows_tr][mtr]).size),
                "chance_test": ch,
                "z_probe": zr, "x_probe": xr,
                "recovery_chance_corrected": (float((zr["test_acc"] - ch) / den)
                                              if den > 1e-6 else None),
            }
            n_warn += zr["n_convergence_warnings"] + xr["n_convergence_warnings"]
    if len(nulls):
        nm, ns = float(nulls.mean()), float(nulls.std(ddof=1))
        nz = (acc_z - nm) / ns if ns > 1e-9 else None
    else:
        nm = ns = nz = None

    denom = acc_x - chance
    recovery = float((acc_z - chance) / denom) if denom > 1e-6 else None

    # success 층화 (scene × success 얽힘 경고 대응)
    succ_te = meta["success"][rows_te]
    strat = {}
    for name, m_ in (("succ_acc", succ_te == 1), ("fail_acc", succ_te != 1)):
        strat[name] = float((pred_te[m_] == yte[m_]).mean()) if m_.sum() else None
        strat[name.replace("acc", "n")] = int(m_.sum())

    res = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sae_source": src, "ckpt_dir": str(args.ckpt_dir) if args.ckpt_dir else None,
        "config": cfg, "label": args.label,
        "label_note": ("scene 라벨로 scenario_seed 를 쓸 수 없다 — episode 당 seed 1개라 "
                       "클래스=표본이 되어 probe 불가(핸드아웃 §6-1). style_id 는 layout_id 와 "
                       + ("완전 공선 → 동일 분할." if collinear else "부분 공선.")),
        "scenario_seed_per_episode": seeds_per_ep,
        "layout_style_collinear": bool(collinear),
        "records_per_ep": rpe, "window": int(args.window),
        "data_check": data_check,
        "n_train_rows": int(len(ytr)), "n_test_rows": int(len(yte)),
        "classes": sorted(set(int(v) for v in y_all)), "chance_test": chance,
        "probe_version": 3,
        "probe_v2_note": ("표준화(train 통계 StandardScaler) + max_iter 기본 1000 + "
                          "episode-group CV 로 C 선택 + 세그먼트별 probe + episode 다수결 "
                          "정확도. 구 probe(표준화 없음·max_iter 200·C=1 고정)와 수치 비교 시 "
                          "주의 — docs/steering/31 §5-1."),
        "probe_v3_note": ("표준화를 Pipeline 안으로(CV fold train 에서만 fit) + 순열 null 은 "
                          "train 라벨만 섞어 동일 절차로 재fit 후 진짜 test 라벨 평가 + "
                          "ckpt stats/split 대조 + --window. null_z 는 v2 와 직접 비교 불가."),
        "z_probe": {**zres, "null_mean": nm, "null_std": ns, "null_z": nz,
                    "n_perm": int(len(nulls))},
        "x_probe": xres,
        "recovery_chance_corrected": recovery,
        "per_segment": per_segment,
        "stratified": strat,
        "selectivity": sel,
        "n_convergence_warnings": int(n_warn),
        "max_iter": int(args.max_iter),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    res["judgment"] = judge(res, args)
    print_table(res)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(f"[probe] → {args.out}")


if __name__ == "__main__":
    sys.exit(main())

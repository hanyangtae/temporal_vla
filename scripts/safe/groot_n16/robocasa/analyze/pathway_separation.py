"""Pathway-resolved 사전(pre-failure) succ/fail 분리력 분석 (latent steering plan Phase 3).

새 pathway pkl(`*_pathway_pertoken_100ep`: DiT subset per-token + VL seq-mean-pool)을
스트리밍으로 읽어, **길이통제(고정 t, within-task)** 하에 각 pathway/layer 의
succ vs fail 분리력(cv-AUROC)을 측정한다. 목적:

  (1) 어느 pathway(VL=goal / DiT layer=motor)·어느 t 가 실패를 **사전에** 가장 잘 가르나
      → Phase 4 steering 타깃 선택. (DiT-only 가 약했던 이유를 정량 확인.)
  (2) length-only baseline + permutation null 로 길이 confound·우연 분리 통제
      ([[seen18-rollout-length-confound]], [[seen18-genuine-failure-direction]]).

토큰 레이아웃(N1.6 PandaOmron): DiT per-step = [L, T=51, D=1536], T = state(1) + action(50).
valid action = action 첫 16 (= SAFE valid). VL per-step = [2048] (이미 seq-mean-pool).

메모리: 1000 pkl(~38GB)을 한 번에 못 올리므로 **rollout 하나씩 읽어 고정-t 벡터만 추출**하고
raw 는 버린다. CPU cap: OMP/OPENBLAS_NUM_THREADS<=16 ([[cpu-budget-cap]]).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "scripts/safe/groot_n16/robocasa/vis"))
from core.io import load_pkl  # noqa: E402  (pickle loader)

SEED = 0
VALID_ACTION = 16  # SAFE valid horizon (action 첫 16 token)


def pool_dit_step(arr: np.ndarray, token_pool: str) -> np.ndarray:
    """DiT per-step [L, T, D] → [L, D] (token 축 pool).

    T = state(1) + action(50). token_pool:
      valid16  : action 첫 16 (tokens 1:17) mean   ← 기본 (SAFE valid)
      action50 : action 전체 (tokens 1:51) mean
      all51    : 전체 token mean
    """
    a = np.asarray(arr, dtype=np.float32)  # [L, T, D]
    if token_pool == "valid16":
        sl = a[:, 1 : 1 + VALID_ACTION, :]
    elif token_pool == "action50":
        sl = a[:, 1:, :]
    elif token_pool == "all51":
        sl = a
    else:
        raise ValueError(f"token_pool: {token_pool}")
    return sl.mean(axis=1)  # [L, D]


def fixed_t_feature(per_step: np.ndarray, t: int) -> np.ndarray | None:
    """per_step [n_steps, ...] → 첫 t step mean (길이통제). 길이<t 면 None."""
    if per_step.shape[0] < t:
        return None
    return per_step[:t].mean(axis=0)


def load_rollout_features(pkl_path: Path, token_pool: str) -> dict | None:
    """pkl 1개 → {task_id, success, length, dit:[n_steps,L,D], vl:[n_steps,Dvl]}.

    VL 없으면 vl=None. DiT 없으면 None 반환(스킵).
    """
    d = load_pkl(pkl_path)
    hs = d.get("hidden_states")
    if not hs:
        return None
    dit = np.stack([pool_dit_step(s, token_pool) for s in hs], axis=0)  # [n_steps, L, D]
    vl = d.get("vl_hidden_states")
    vl_arr = (
        np.stack([np.asarray(v, dtype=np.float32) for v in vl], axis=0)
        if vl is not None
        else None
    )
    return {
        "task_id": int(d.get("task_id", -1)),
        "success": int(d.get("episode_success", 0)),
        "length": len(hs),
        "dit": dit,
        "vl": vl_arr,
    }


def _auroc_from_scores(scores: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney U 기반 AUROC (sklearn 불필요)."""
    from scipy.stats import mannwhitneyu
    pos, neg = scores[y == 1], scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    u, _ = mannwhitneyu(pos, neg, alternative="two-sided")
    return float(u / (len(pos) * len(neg)))


def _pca_np(X: np.ndarray, n: int) -> np.ndarray:
    """numpy SVD-based PCA, X 이미 zero-mean 가정."""
    n = min(n, X.shape[0] - 1, X.shape[1])
    if n <= 0:
        return X
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt[:n].T


def _lda_score(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray) -> np.ndarray:
    """equal-cov LDA 1D 투영 점수 (PCA 후 mean-diff 방향)."""
    n_pc = min(30, Xtr.shape[1], Xtr.shape[0] - 1)
    mu = Xtr.mean(axis=0)
    Xtr_c = Xtr - mu
    Xtr_p = _pca_np(Xtr_c, n_pc)
    Xte_p = (Xte - mu) @ np.linalg.svd(Xtr_c, full_matrices=False)[2][:n_pc].T
    mu1 = Xtr_p[ytr == 1].mean(axis=0)
    mu0 = Xtr_p[ytr == 0].mean(axis=0)
    direction = mu1 - mu0
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        return np.zeros(len(Xte_p))
    return Xte_p @ (direction / norm)


def cv_auroc(X: np.ndarray, y: np.ndarray, k: int = 5) -> float | None:
    """PCA→LDA→Mann-Whitney AUROC, StratifiedKFold (numpy+scipy, sklearn 불필요)."""
    y = np.asarray(y)
    if len(np.unique(y)) < 2 or np.bincount(y).min() < k:
        return None
    n = len(y)
    rng = np.random.default_rng(SEED)
    # Stratified k-fold 수동 구성
    idx0, idx1 = np.where(y == 0)[0], np.where(y == 1)[0]
    rng.shuffle(idx0); rng.shuffle(idx1)
    folds0 = np.array_split(idx0, k); folds1 = np.array_split(idx1, k)
    aucs = []
    for i in range(k):
        te_idx = np.concatenate([folds0[i], folds1[i]])
        tr_idx = np.concatenate(
            [folds0[j] for j in range(k) if j != i] +
            [folds1[j] for j in range(k) if j != i]
        )
        scores = _lda_score(X[tr_idx], y[tr_idx], X[te_idx])
        auc = _auroc_from_scores(scores, y[te_idx])
        aucs.append(auc)
    return float(np.mean(aucs))


def perm_null(X: np.ndarray, y: np.ndarray, n: int = 50) -> tuple[float, float]:
    """label permutation null 95% band (상한 추정)."""
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(n):
        yp = rng.permutation(y)
        a = cv_auroc(X, yp, k=5)
        if a is not None:
            vals.append(a)
    if not vals:
        return (0.5, 0.5)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Pathway-resolved pre-failure separation")
    ap.add_argument("--run-dir", required=True, help="raw_rollouts dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--token-pool", default="valid16", choices=("valid16", "action50", "all51"))
    ap.add_argument("--t-set", default="4,8,12,16", help="고정 t (길이통제) 콤마구분")
    ap.add_argument("--min-per-class", type=int, default=8, help="task당 succ/fail 최소 표본")
    ap.add_argument("--max-pkl-per-task", type=int, default=None, help="디버그용 상한")
    ap.add_argument("--smoke", action="store_true", help="로더 검증(소수 pkl shape만 출력)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir.parent / "analysis" / "pathway_separation"
    t_set = [int(x) for x in args.t_set.split(",")]

    pkls = sorted(run_dir.glob("*/*.pkl"))
    if not pkls:
        raise SystemExit(f"no pkl under {run_dir}")
    print(f"[load] {len(pkls)} pkl found under {run_dir}")

    if args.smoke:
        for p in pkls[:3]:
            r = load_rollout_features(p, args.token_pool)
            vlshape = None if r["vl"] is None else r["vl"].shape
            print(f"  {p.parent.name}/{p.name}: task={r['task_id']} succ={r['success']} "
                  f"len={r['length']} dit={r['dit'].shape} vl={vlshape}")
        return

    # task별 rollout feature 누적 (고정-t 벡터만; raw 버림)
    from collections import defaultdict
    per_task = defaultdict(lambda: {"rows": []})  # task_id -> list of rollout dicts(고정-t)
    counts = defaultdict(int)
    for p in pkls:
        tid = p.parent.name
        if args.max_pkl_per_task and counts[tid] >= args.max_pkl_per_task:
            continue
        counts[tid] += 1
        r = load_rollout_features(p, args.token_pool)
        if r is None:
            continue
        L = r["dit"].shape[1]
        row = {"success": r["success"], "length": r["length"], "dit": r["dit"], "vl": r["vl"], "L": L}
        per_task[tid]["rows"].append(row)

    # pathway × layer × t cv-AUROC (within-task), task 평균
    results = {"token_pool": args.token_pool, "t_set": t_set, "tasks": {}, "summary": {}}
    n_layers = None
    for tid, blob in per_task.items():
        rows = blob["rows"]
        ys = np.array([r["success"] for r in rows])
        lens = np.array([r["length"] for r in rows])
        if len(np.unique(ys)) < 2:
            continue
        n_layers = rows[0]["dit"].shape[1]
        has_vl = rows[0]["vl"] is not None
        task_res = {}
        for t in t_set:
            # 길이>=t 인 rollout만 (길이통제)
            idx = [i for i, r in enumerate(rows) if r["length"] >= t]
            yt = ys[idx]
            if len(np.unique(yt)) < 2 or np.bincount(yt).min() < args.min_per_class:
                continue
            entry = {"n": int(len(idx)), "n_fail": int((yt == 0).sum())}
            # length-only baseline
            entry["length_auroc"] = cv_auroc(lens[idx].reshape(-1, 1), yt)
            # DiT per layer
            for ell in range(n_layers):
                X = np.stack([fixed_t_feature(rows[i]["dit"][:, ell, :], t) for i in idx])
                entry[f"dit_L{ell}"] = cv_auroc(X, yt)
            # VL
            if has_vl:
                Xvl = np.stack([fixed_t_feature(rows[i]["vl"], t) for i in idx])
                entry["vl"] = cv_auroc(Xvl, yt)
                lo, hi = perm_null(Xvl, yt)
                entry["null95"] = [lo, hi]
            task_res[f"t{t}"] = entry
        if task_res:
            results["tasks"][tid] = task_res

    # summary: pathway×t task 평균
    summ = {}
    for t in t_set:
        key = f"t{t}"
        per_path = defaultdict(list)
        for tid, tr in results["tasks"].items():
            if key not in tr:
                continue
            for pk, v in tr[key].items():
                if pk.startswith(("dit_L", "vl", "length_auroc")) and isinstance(v, float):
                    per_path[pk].append(v)
        summ[key] = {pk: float(np.mean(vs)) for pk, vs in per_path.items() if vs}
    results["summary"] = summ
    results["n_layers"] = n_layers

    out.mkdir(parents=True, exist_ok=True)
    (out / "pathway_separation.json").write_text(json.dumps(results, indent=2))
    print(f"[done] tasks={len(results['tasks'])} -> {out/'pathway_separation.json'}")
    for t in t_set:
        s = summ.get(f"t{t}", {})
        if s:
            best = max(((k, v) for k, v in s.items() if k != "length_auroc"), key=lambda x: x[1], default=None)
            print(f"  t={t}: length-only={s.get('length_auroc'):.3f}  best={best}")


if __name__ == "__main__":
    main()

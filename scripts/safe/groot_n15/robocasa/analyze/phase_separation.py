"""GR00T N1.5 RoboCasa — action-phase(event)별 succ/fail 분리도 (Rung 2 / plan §Rung 2).

플랜 단일출처: ``~/.claude/plans/vla-reactive-acorn.md`` (§"Rung 2 — per-phase 분리도").
N1.6용 ``groot_n16/.../pathway_separation.py`` 의 N1.5 event-phase 이식본이다. 차이:

  1. 절대-t(첫 t step mean) 윈도 → **event-phase pooling** (feature_phases 라벨로 grouping).
  2. DiT record layout: N1.6 ``[L, T=51, D]`` action-token → N1.5 ``[L=7, K=4, D=1536]``
     (K=4 = num_inference_timesteps denoise step). denoise 축 mean-pool → ``[L, D]``.
  3. scipy ``mannwhitneyu`` → numpy rank-AUROC (원격 scipy 없음, [[remote-compute-workflow]]).
  4. 비교 단위 = **within-cell**(고정 scenario_seed/instruction) succ vs fail. layout/object/
     instruction confound 는 fixed-seed 로 이미 통제됨. 길이 confound 만 남아 아래로 통제한다.

길이 confound 통제 ([[seen18-rollout-length-confound]]):
  - phase 도달 rollout 한정 + **phase별 동일 스텝예산 pool**(그 phase 최소 record 수로 truncate).
  - **length-only baseline**(phase record 수만으로의 분리도) 병기.
  - **permutation null** 상단 밴드. 분리 판정 = latent AUROC > null 상단 AND > length-only.

표본이 작다(fail ~3–4/cell)라 k-fold 대신 **LOO out-of-sample AUROC**. COAST 계열 요구량 ≥3/class
(사용자 기준)에 맞춰 min-per-class 기본 3. 결과는 저표본 → 정성적 신호로 해석.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np

SEED = 0
# 이 순서대로 분석. VL 은 별도 layer key "VL" 로 취급.
DEFAULT_PHASES = ("reach-to-object", "transport", "insert-settle")


# --------------------------------------------------------------------------- #
# 순수 함수 (테스트 대상)
# --------------------------------------------------------------------------- #
def rank_auroc(scores: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney U 기반 AUROC = P(score[pos] > score[neg]). tie=평균순위. (scipy 불필요).

    y==1 을 positive 로 본다. pos/neg 한쪽이 비면 0.5.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(y).ravel()
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    r = np.empty(len(scores), dtype=np.float64)
    i = 0
    n = len(scores)
    while i < n:
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        r[i : j + 1] = (i + j) / 2.0 + 1.0  # 1-based 평균순위
        i = j + 1
    ranks[order] = r
    u = ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0
    return float(u / (n1 * n0))


def pool_denoise(rec: np.ndarray) -> np.ndarray:
    """DiT record [L, K, D] → [L, D] (denoise 축 K mean)."""
    a = np.asarray(rec, dtype=np.float32)
    if a.ndim != 3:
        raise ValueError(f"expected [L,K,D], got {a.shape}")
    return a.mean(axis=1)


def equal_budget_pool(vectors: list[np.ndarray], budget: int) -> np.ndarray:
    """한 rollout·한 phase 의 record 벡터 리스트에서 **첫 budget 개만** mean-pool.

    동일예산 pool: 모든 rollout 이 같은 개수(budget)만 쓰게 해 phase-내 길이차를 제거.
    """
    if budget <= 0 or len(vectors) < budget:
        raise ValueError(f"budget={budget} vs available={len(vectors)}")
    return np.stack(vectors[:budget], axis=0).mean(axis=0)


# --------------------------------------------------------------------------- #
# 분리도 (PCA→mean-diff LDA→LOO rank-AUROC + permutation null)
# --------------------------------------------------------------------------- #
def _lda_project(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray) -> np.ndarray:
    """PCA(train fit) → (mu_succ − mu_fail) 방향 1D 투영. 높을수록 success 쪽."""
    n_pc = int(min(30, Xtr.shape[1], Xtr.shape[0] - 1))
    mu = Xtr.mean(axis=0)
    Xtr_c = Xtr - mu
    if n_pc <= 0:
        return np.zeros(len(Xte))
    _, _, Vt = np.linalg.svd(Xtr_c, full_matrices=False)
    P = Vt[:n_pc].T
    Xtr_p = Xtr_c @ P
    Xte_p = (Xte - mu) @ P
    mu1 = Xtr_p[ytr == 1].mean(axis=0)
    mu0 = Xtr_p[ytr == 0].mean(axis=0)
    d = mu1 - mu0
    nd = np.linalg.norm(d)
    if nd < 1e-12:
        return np.zeros(len(Xte_p))
    return Xte_p @ (d / nd)


def loo_auroc(X: np.ndarray, y: np.ndarray) -> float | None:
    """Leave-one-out out-of-sample AUROC. 각 클래스 ≥2 필요(LOO fold 유효)."""
    y = np.asarray(y)
    if int((y == 1).sum()) < 2 or int((y == 0).sum()) < 2:
        return None
    n = len(y)
    scores = np.empty(n, dtype=np.float64)
    for i in range(n):
        tr = np.ones(n, dtype=bool)
        tr[i] = False
        if len(np.unique(y[tr])) < 2:
            scores[i] = 0.0
            continue
        scores[i] = _lda_project(X[tr], y[tr], X[i : i + 1])[0]
    return rank_auroc(scores, y)


def perm_null_upper(X: np.ndarray, y: np.ndarray, n_perm: int = 200, pct: float = 95.0) -> float:
    """label permutation LOO-AUROC null 의 상단(pct) — |auroc-0.5| 기준 대칭 상단."""
    rng = np.random.default_rng(SEED)
    seps = []
    for _ in range(n_perm):
        a = loo_auroc(X, rng.permutation(y))
        if a is not None:
            seps.append(abs(a - 0.5))
    if not seps:
        return 0.5
    return 0.5 + float(np.percentile(seps, pct))


# --------------------------------------------------------------------------- #
# 로딩
# --------------------------------------------------------------------------- #
def load_rollout(pkl_path: Path) -> dict:
    """pkl 1개 → per-record DiT [n,L,D](denoise-pooled) + VL [n,Dvl] + phases + success."""
    import pickle

    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    hs = d["hidden_states"]
    dit = np.stack([pool_denoise(np.asarray(r)) for r in hs], axis=0)  # [n, L, D]
    vl = d.get("vl_hidden_states")
    vl_arr = np.stack([np.asarray(v, dtype=np.float32) for v in vl], axis=0) if vl is not None else None
    phases = list(d.get("feature_phases") or [])
    if len(phases) != dit.shape[0]:
        raise ValueError(f"{pkl_path.name}: feature_phases {len(phases)} != records {dit.shape[0]}")
    return {
        "name": pkl_path.name,
        "success": int(d.get("episode_success", 0)),
        "length": dit.shape[0],
        "dit": dit,
        "vl": vl_arr,
        "phases": phases,
        "capture_layers": list(d.get("capture_layers") or []),
    }


def phase_records(roll: dict, phase: str, layer_key) -> list[np.ndarray]:
    """rollout 에서 phase 에 속한 record 의 (layer 별) 벡터 리스트 (raw, truncate 전)."""
    idx = [i for i, p in enumerate(roll["phases"]) if p == phase]
    if not idx:
        return []
    if layer_key == "VL":
        if roll["vl"] is None:
            return []
        return [roll["vl"][i] for i in idx]
    return [roll["dit"][i, layer_key, :] for i in idx]


# --------------------------------------------------------------------------- #
# cell 단위 분석
# --------------------------------------------------------------------------- #
def analyze_cell(rolls: list[dict], phases, layer_keys, min_per_class: int, n_perm: int) -> dict:
    out = {"n_succ": sum(r["success"] for r in rolls),
           "n_fail": sum(1 - r["success"] for r in rolls),
           "phases": {}}
    for phase in phases:
        # 이 phase 에 record ≥1 인 rollout 만 (equal-budget = 그들 중 최소 record 수)
        present = [(r, phase_records(r, phase, layer_keys[0])) for r in rolls]
        present = [(r, recs) for r, recs in present if len(recs) > 0]
        ys = np.array([r["success"] for r, _ in present])
        n1, n0 = int((ys == 1).sum()), int((ys == 0).sum())
        counts = [len(recs) for _, recs in present]
        phase_res = {"n_succ": n1, "n_fail": n0,
                     "record_counts": {"succ": [len(rc) for (r, rc) in present if r["success"]],
                                       "fail": [len(rc) for (r, rc) in present if not r["success"]]}}
        if n1 < min_per_class or n0 < min_per_class:
            phase_res["status"] = f"skip: succ={n1} fail={n0} < min {min_per_class}"
            out["phases"][phase] = phase_res
            continue
        budget = min(counts)
        phase_res["budget"] = int(budget)
        # length-only baseline: phase record 수만으로의 분리도
        lens = np.array(counts, dtype=np.float64)
        phase_res["length_auroc"] = rank_auroc(lens, ys)
        # layer 별 latent 분리도
        layer_out = {}
        rolls_present = [r for r, _ in present]
        for lk in layer_keys:
            X = []
            ok = True
            for r in rolls_present:
                recs = phase_records(r, phase, lk)
                if len(recs) < budget:
                    ok = False
                    break
                X.append(equal_budget_pool(recs, budget))
            if not ok:
                continue
            X = np.stack(X, axis=0)
            a = loo_auroc(X, ys)
            if a is None:
                continue
            null_hi = perm_null_upper(X, ys, n_perm=n_perm)
            layer_out[str(lk)] = {"auroc": a, "null95_upper": null_hi}
        phase_res["layers"] = layer_out
        out["phases"][phase] = phase_res
    return out


def discover_cells(run_dir: Path) -> dict[str, list[Path]]:
    cells: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(run_dir.glob("*/*/*.pkl")):
        cells[p.parent.name].append(p)
    return cells


def main() -> None:
    ap = argparse.ArgumentParser(description="N1.5 event-phase succ/fail 분리도 (Rung 2)")
    ap.add_argument("--run-dir", required=True, help="raw_rollouts dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cells", default=None, help="콤마구분 cell_id (기본: fail>=min 인 셀 자동)")
    ap.add_argument("--phases", default=",".join(DEFAULT_PHASES))
    ap.add_argument("--min-per-class", type=int, default=3, help="phase별 succ/fail 최소 (COAST ≥3)")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--smoke", action="store_true", help="cell별 phase 인벤토리만 출력")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir.parent / "analysis" / "phase_separation"
    phases = tuple(args.phases.split(","))
    cells = discover_cells(run_dir)
    if not cells:
        raise SystemExit(f"no pkl under {run_dir}")

    if args.smoke:
        for cid, paths in cells.items():
            print(f"\n=== {cid}: {len(paths)} rollouts ===")
            for p in paths:
                r = load_rollout(p)
                print(f"  {r['name']} succ={r['success']} len={r['length']} "
                      f"phases={dict(Counter(r['phases']))} vl={None if r['vl'] is None else r['vl'].shape}")
        return

    # DiT 7 layer(capture_layers 라벨) + VL
    sample = load_rollout(cells[next(iter(cells))][0])
    cap = sample["capture_layers"]  # e.g. [0,2,4,8,10,12,15]
    layer_keys = list(range(sample["dit"].shape[1])) + (["VL"] if sample["vl"] is not None else [])
    layer_labels = {str(i): f"DiT-L{cap[i]}" for i in range(len(cap))}
    layer_labels["VL"] = "VL"

    want = args.cells.split(",") if args.cells else None
    results = {"run_dir": str(run_dir), "capture_layers": cap, "min_per_class": args.min_per_class,
               "layer_labels": layer_labels, "cells": {}}
    for cid, paths in cells.items():
        if want and cid not in want:
            continue
        rolls = [load_rollout(p) for p in paths]
        n_fail = sum(1 - r["success"] for r in rolls)
        if not want and n_fail < args.min_per_class:
            continue
        print(f"[cell] {cid}: {len(rolls)} rollouts, fail={n_fail}")
        results["cells"][cid] = analyze_cell(rolls, phases, layer_keys, args.min_per_class, args.n_perm)

    out.mkdir(parents=True, exist_ok=True)
    (out / "phase_separation.json").write_text(json.dumps(results, indent=2))
    print(f"\n[done] -> {out / 'phase_separation.json'}")
    _print_table(results)


def _print_table(results: dict) -> None:
    lab = results["layer_labels"]
    for cid, cres in results["cells"].items():
        print(f"\n### {cid}  (succ={cres['n_succ']} fail={cres['n_fail']})")
        for phase, pr in cres["phases"].items():
            if "layers" not in pr:
                print(f"  [{phase}] {pr.get('status','(no layers)')}  "
                      f"succ={pr['n_succ']} fail={pr['n_fail']}")
                continue
            print(f"  [{phase}] succ={pr['n_succ']} fail={pr['n_fail']} budget={pr['budget']} "
                  f"length_auroc={pr['length_auroc']:.3f}")
            rows = sorted(pr["layers"].items(), key=lambda kv: abs(kv[1]["auroc"] - 0.5), reverse=True)
            for lk, v in rows:
                sep = abs(v["auroc"] - 0.5)
                lensep = abs(pr["length_auroc"] - 0.5)
                flag = "GREEN" if (sep > v["null95_upper"] - 0.5 and sep > lensep + 0.07) else ""
                print(f"      {lab.get(lk, lk):8s} auroc={v['auroc']:.3f} "
                      f"null95_hi={v['null95_upper']:.3f} {flag}")


if __name__ == "__main__":
    main()

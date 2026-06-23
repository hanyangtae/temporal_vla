"""per-pathway online 실패 detector — decision-time AUROC (length-controlled, scipy-free).

사용자 Q1(succ/fail 분리되나)·Q2(SAFE-LSTM proxy)·Q3(per-step 정상→비정상 전이)를
**per-pathway(VL / DiT-layer / both)** 로 공정하게 판정.

핵심 = **decision-time AUROC 곡선**: 각 결정시점 t_d 에서 causal feature(첫 t_d step 평균)만으로
succ/fail 을 가르는 AUROC. 곡선이 t_d 따라 오르면 = "어느 시점 이후 비정상으로 이동"(Q3) 의 모집단 증거.

공정 metric(`verify_detector_length_control.py` 정신):
  - **cross-task**: seen task 로 LDA 학습 → **unseen task** test (일반화=헤드라인). within-task CV(관대) 병기.
  - **living-rollout only**(length>t_d): "아직 살아있나"가 라벨로 새지 않게(성공은 일찍 끝남) → 길이통제.
  - **length-only baseline**(full length→failure) + **permutation null** 동반.
공정 verdict = unseen AUROC 가 length baseline·null 을 의미있게 넘나.

pkl 은 torch 텐서라 원격 전용(${REMOTE_PYTHON}), scipy 없음 → AUROC 는 pure numpy.
재사용: `pathway_separation.load_rollout_features / _pca_np / _lda_score`.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pathway_separation import load_rollout_features, _lda_score  # noqa: E402

SEED = 0
DEFAULT_TDS = (3, 5, 8, 11, 15, 20)
# pathway_pertoken DiT capture layers (block id) → local index. block31 = idx6.
CAPTURE_LAYERS = [0, 2, 4, 8, 16, 24, 31]


def auroc(scores: np.ndarray, y: np.ndarray) -> float:
    """rank 기반 AUROC (pure numpy). y==1 (failure) 를 positive."""
    pos, neg = scores[y == 1], scores[y == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    _, inv, cnt = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(cnt.size)
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def causal_feat(r: dict, t_d: int, pathway: str, dit_idx: int) -> np.ndarray | None:
    """첫 min(t_d, length) step 평균 causal feature. living 가정 시 t_eff=t_d."""
    t_eff = min(t_d, r["length"])
    if t_eff < 1:
        return None
    dit = r["dit"][:t_eff, dit_idx, :].mean(axis=0)
    if pathway == "dit":
        return dit
    if r["vl"] is None:
        return None
    vl = r["vl"][:t_eff].mean(axis=0)
    if pathway == "vl":
        return vl
    return np.concatenate([vl, dit])  # both


def _design(rollouts, t_d, pathway, dit_idx, tasks, living):
    """task 집합에서 (X, y=failure) 행렬. living=True면 length>t_d 만."""
    X, y = [], []
    for r in rollouts:
        if r["task"] not in tasks:
            continue
        if living and r["length"] <= t_d:
            continue
        f = causal_feat(r, t_d, pathway, dit_idx)
        if f is None:
            continue
        X.append(f)
        y.append(1 - int(r["success"]))  # failure=1
    return (np.asarray(X, dtype=np.float64), np.asarray(y, dtype=int)) if X else (None, None)


def cross_task(rollouts, t_d, pathway, dit_idx, seen, unseen, living, n_perm):
    """seen 학습 → unseen test AUROC + perm null(test label shuffle)."""
    Xtr, ytr = _design(rollouts, t_d, pathway, dit_idx, seen, living)
    Xte, yte = _design(rollouts, t_d, pathway, dit_idx, unseen, living)
    if Xtr is None or Xte is None:
        return None
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2 or min(np.bincount(yte)) < 3:
        return None
    scores = _lda_score(Xtr, ytr, Xte)
    obs = auroc(scores, yte)
    rng = np.random.default_rng(SEED)
    null = [auroc(scores, rng.permutation(yte)) for _ in range(n_perm)]
    lo, hi = (float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))) if null else (0.5, 0.5)
    return {"auroc": float(obs), "n_test": int(len(yte)), "n_fail": int(yte.sum()),
            "null95": [lo, hi], "sig": bool(obs > hi)}


def within_task_cv(rollouts, t_d, pathway, dit_idx, tasks, living, k=5):
    """task별 stratified k-fold AUROC 평균(관대 baseline)."""
    aucs = []
    by_task = defaultdict(list)
    for r in rollouts:
        if r["task"] in tasks:
            by_task[r["task"]].append(r)
    for task, rs in by_task.items():
        X, y = _design(rs, t_d, pathway, dit_idx, {task}, living)
        if X is None or len(np.unique(y)) < 2 or min(np.bincount(y)) < k:
            continue
        rng = np.random.default_rng(SEED)
        i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
        rng.shuffle(i0); rng.shuffle(i1)
        f0, f1 = np.array_split(i0, k), np.array_split(i1, k)
        fold_aucs = []
        for i in range(k):
            te = np.concatenate([f0[i], f1[i]])
            tr = np.concatenate([f0[j] for j in range(k) if j != i] + [f1[j] for j in range(k) if j != i])
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            fold_aucs.append(auroc(_lda_score(X[tr], y[tr], X[te]), y[te]))
        if fold_aucs:
            aucs.append(float(np.mean(fold_aucs)))
    return float(np.mean(aucs)) if aucs else None


def length_baseline(rollouts, t_d, unseen, living):
    """full rollout length → failure AUROC (confound 천장), unseen-living."""
    L, y = [], []
    for r in rollouts:
        if r["task"] not in unseen:
            continue
        if living and r["length"] <= t_d:
            continue
        L.append(r["length"])
        y.append(1 - int(r["success"]))
    L, y = np.asarray(L, float), np.asarray(y, int)
    if len(L) == 0 or len(np.unique(y)) < 2:
        return None
    return auroc(L, y)


def load_all(run_dir: Path, token_pool: str):
    rolls = []
    for td in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        for p in sorted(td.glob("*.pkl")):
            r = load_rollout_features(p, token_pool)
            if r is None:
                continue
            r["task"] = td.name
            rolls.append(r)
    return rolls


def plot_curves(results, out_path, pathways, tds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    color = {"vl": "#d95234", "dit": "#458cd6", "both": "#43a047"}
    fig, ax = plt.subplots(figsize=(9, 6))
    for pw in pathways:
        xs, ys, los, his = [], [], [], []
        for t in tds:
            c = results["cross_task"][pw].get(str(t))
            if c:
                xs.append(t); ys.append(c["auroc"]); los.append(c["null95"][0]); his.append(c["null95"][1])
        if xs:
            ax.plot(xs, ys, marker="o", lw=2, color=color.get(pw, "k"), label=f"{pw} (unseen)")
            ax.fill_between(xs, los, his, color=color.get(pw, "k"), alpha=0.12)
    lb = [(t, results["length_baseline"].get(str(t))) for t in tds]
    lb = [(t, v) for t, v in lb if v is not None]
    if lb:
        ax.plot([t for t, _ in lb], [v for _, v in lb], ls="--", color="#888", label="length-only baseline")
    ax.axhline(0.5, color="k", ls=":", lw=0.8)
    ax.set_xlabel("decision time t_d (frames seen, causal)")
    ax.set_ylabel("AUROC (failure=positive), unseen tasks")
    ax.set_ylim(0.3, 1.0)
    ax.set_title("per-pathway online detection — cross-task (seen→unseen), length-controlled")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close()


def main():
    ap = argparse.ArgumentParser(description="per-pathway online failure detection (decision-time AUROC)")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--token-pool", default="valid16", choices=("valid16", "action50", "all51"))
    ap.add_argument("--dit-block", type=int, default=31)
    ap.add_argument("--pathways", default="vl,dit,both")
    ap.add_argument("--t-ds", default=",".join(map(str, DEFAULT_TDS)))
    ap.add_argument("--seen", default="CloseToasterOvenDoor,NavigateKitchen,OpenCabinet,PickPlaceCounterToStove,PickPlaceDrawerToCounter,SlideDishwasherRack,TurnOnMicrowave,TurnOnSinkFaucet")
    ap.add_argument("--unseen", default="OpenDrawer,PickPlaceCounterToCabinet")
    ap.add_argument("--no-living", action="store_true", help="length 통제 끄기(디버그)")
    ap.add_argument("--perm", type=int, default=200)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir.parent / "analysis" / "pathway_online_detection"
    dit_idx = int(np.argmin([abs(b - args.dit_block) for b in CAPTURE_LAYERS]))
    pathways = args.pathways.split(",")
    tds = [int(x) for x in args.t_ds.split(",")]
    seen, unseen = set(args.seen.split(",")), set(args.unseen.split(","))
    living = not args.no_living

    rolls = load_all(run_dir, args.token_pool)
    print(f"[load] {len(rolls)} rollouts, DiT block={CAPTURE_LAYERS[dit_idx]} (idx{dit_idx}), "
          f"seen={len(seen)} unseen={len(unseen)}")
    if args.smoke:
        from collections import Counter
        print("  tasks:", dict(Counter(r["task"] for r in rolls)))
        r0 = rolls[0]
        print(f"  ex: task={r0['task']} succ={r0['success']} len={r0['length']} "
              f"vl={None if r0['vl'] is None else r0['vl'].shape} dit={r0['dit'].shape}")
        return

    results = {"dit_block": CAPTURE_LAYERS[dit_idx], "t_ds": tds, "seen": sorted(seen),
               "unseen": sorted(unseen), "living": living,
               "cross_task": {pw: {} for pw in pathways}, "within_task": {pw: {} for pw in pathways},
               "length_baseline": {}}
    for t in tds:
        lb = length_baseline(rolls, t, unseen, living)
        if lb is not None:
            results["length_baseline"][str(t)] = lb
        for pw in pathways:
            ct = cross_task(rolls, t, pw, dit_idx, seen, unseen, living, args.perm)
            wt = within_task_cv(rolls, t, pw, dit_idx, seen | unseen, living)
            if ct:
                results["cross_task"][pw][str(t)] = ct
            if wt is not None:
                results["within_task"][pw][str(t)] = wt
        line = " | ".join(
            f"{pw}: unseen={results['cross_task'][pw].get(str(t), {}).get('auroc', float('nan')):.3f}"
            f"{'*' if results['cross_task'][pw].get(str(t), {}).get('sig') else ' '}"
            f"/within={results['within_task'][pw].get(str(t), float('nan')):.3f}"
            for pw in pathways
        )
        print(f"  t_d={t:2d} (len-base={lb if lb is None else round(lb,3)}): {line}")

    out.mkdir(parents=True, exist_ok=True)
    (out / "pathway_online_detection.json").write_text(json.dumps(results, indent=2))
    plot_curves(results, out / "decision_time_curve.png", pathways, tds)
    print(f"[done] -> {out}/")


if __name__ == "__main__":
    main()

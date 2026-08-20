#!/usr/bin/env python
"""activation 에서 action phase 를 읽어낼 수 있는가 — held-out 판독 정확도.

논문 주장을 "GT 보다 세밀하다"가 아니라 **"activation 만으로 지금 어느 phase 인지
읽을 수 있다"** 로 세울 때 필요한 수치를 낸다. 핵심은 *held-out* 이다: 군집도 매핑도
probe 도 전부 train scene 에서만 만들고, 한 번도 보지 않은 scene 의 스텝에서 평가한다.

비교 대상 (전부 같은 split · 같은 평가 스텝):
  major   다수 클래스만 찍기 (하한)
  clock   activation 을 보지 않고 **에피소드 진행도 t/T** 만으로 판정.
          train 에서 t/T 를 n_bin 분위로 자르고 각 bin 의 최빈 phase 를 답으로 쓴다.
          조작 phase 는 대체로 순서대로 진행하므로 이게 진짜 경쟁 상대다.
  cluster 비지도: train latent 로 KMeans(k) → 각 군집의 최빈 phase (라벨은 이 매핑에만
          쓰인다) → test 스텝은 최근접 중심으로 배정해 그 phase 로 판정.
  probe   지도: train latent 로 multinomial logistic 회귀 → test 판정 (상한 참고).
  probe_t clock 과 같은 정보(t/T)만 넣은 지도 probe — "시간만으로 얼마나 되는가"의
          더 강한 버전.

입력: ae_cluster.py --dump-labels 산출 `labels_<slug>_k<k>.npz`
      (latent/phase_code/scene/ep_id/rec_idx 필요)
출력: readout.tsv + readout.json (instruction 별 정확도·macro-F1·n)

의존: numpy 만 (scipy/sklearn 없이 로지스틱 회귀를 GD 로 직접 적합).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- 유틸

def macro_f1(y_true, y_pred, classes) -> float:
    fs = []
    for c in classes:
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        fs.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(fs))


def time_fraction(ep_id, rec_idx):
    """에피소드 내 진행도 0..1."""
    out = np.zeros(len(ep_id), np.float64)
    for e in np.unique(ep_id):
        m = ep_id == e
        r = rec_idx[m].astype(np.float64)
        out[m] = (r - r.min()) / max(r.max() - r.min(), 1.0)
    return out


def kmeans(X, k, seed=0, n_init=5, iters=100):
    """numpy KMeans (k-means++ 초기화). 중심을 반환한다."""
    rng = np.random.default_rng(seed)
    best, best_inertia = None, np.inf
    for _ in range(n_init):
        c = [X[rng.integers(len(X))]]
        for _ in range(k - 1):
            d = np.min(((X[:, None, :] - np.array(c)[None]) ** 2).sum(-1), 1)
            p = d / max(d.sum(), 1e-12)
            c.append(X[rng.choice(len(X), p=p)])
        C = np.array(c)
        for _ in range(iters):
            lab = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
            newC = np.stack([X[lab == j].mean(0) if (lab == j).any() else C[j]
                             for j in range(k)])
            if np.allclose(newC, C):
                C = newC
                break
            C = newC
        inertia = float(((X - C[lab]) ** 2).sum())
        if inertia < best_inertia:
            best, best_inertia = C, inertia
    return best


def assign(X, C):
    return ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)


def majority_map(labels, y, n_lab, fallback):
    """군집/bin → 최빈 phase 매핑 (라벨을 쓰는 유일한 단계)."""
    m = np.full(n_lab, fallback, dtype=np.int64)
    for j in range(n_lab):
        sel = y[labels == j]
        if len(sel):
            v, c = np.unique(sel, return_counts=True)
            m[j] = v[c.argmax()]
    return m


def logistic(X, y, classes, epochs=400, lr=0.5, l2=1e-3, seed=0):
    """multinomial logistic 회귀 (full-batch GD, numpy). 표준화 포함."""
    rng = np.random.default_rng(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xs = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    Y = np.zeros((len(X), len(classes)))
    for i, c in enumerate(classes):
        Y[y == c, i] = 1.0
    W = rng.normal(0, 0.01, (Xs.shape[1], len(classes)))
    for _ in range(epochs):
        Z = Xs @ W
        Z -= Z.max(1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(1, keepdims=True)
        G = Xs.T @ (P - Y) / len(Xs) + l2 * W
        W -= lr * G
    return {"W": W, "mu": mu, "sd": sd, "classes": np.asarray(classes)}


def logistic_predict(mdl, X):
    Xs = np.hstack([(X - mdl["mu"]) / mdl["sd"], np.ones((len(X), 1))])
    return mdl["classes"][(Xs @ mdl["W"]).argmax(1)]


# ---------------------------------------------------------------- 행동 대조군

INSTR2SLUG = {
    "Open the left drawer.": "OpenDrawer_left",
    "Open the right drawer.": "OpenDrawer_right",
    "Fully slide the top dishwasher rack out.": "DishwasherRack_out",
    "Fully slide the oven rack out.": "OvenRack_out",
    "Pick the mug from the counter and place it under the coffee machine dispenser.":
        "CoffeeSetupMug",
}
for _obj in ("apple", "bread", "candle", "jug", "marshmallow"):
    INSTR2SLUG[f"Pick the {_obj} from the counter and place it in the cabinet."] = \
        f"PPCC_{_obj}"


def index_traj(root: Path) -> dict:
    """{(slug, scene, noise): traj.csv 경로} — meta.json 의 instruction 으로 slug 판정."""
    import re
    idx = {}
    for meta in root.rglob("meta.json"):
        try:
            d = json.loads(meta.read_text())
        except Exception:
            continue
        slug = INSTR2SLUG.get(d.get("instruction", ""))
        traj = meta.parent / "traj.csv"
        if slug is None or not traj.is_file():
            continue
        sc = no = None
        for part in meta.parts:
            m = re.fullmatch(r"s(\d+)", part)
            if m:
                sc = int(m.group(1))
            m = re.fullmatch(r"n(\d+)", part)
            if m:
                no = int(m.group(1))
        if sc is None or no is None:
            continue
        idx[(slug, sc, no)] = traj
    return idx


def action_features(traj_path: Path, n_rec: int, steps_per_rec: int = 5):
    """record 별 행동 요약 [mean(7), std(7), last(7), cumsum(7)] = 28 차원.

    정책은 record 하나마다 5 env-step 을 실행하므로, record r 은 csv 행 5r..5r+4 다.
    """
    rows = []
    with traj_path.open() as fh:
        rd = csv.reader(fh)
        head = next(rd, None)
        if not head:
            return None
        for line in rd:
            try:
                rows.append([float(v) for v in line])
            except ValueError:
                continue
    if not rows:
        return None
    A = np.asarray(rows, float)
    out = np.zeros((n_rec, A.shape[1] * 4), float)
    for r in range(n_rec):
        seg = A[r * steps_per_rec:(r + 1) * steps_per_rec]
        if len(seg) == 0:
            seg = A[-1:]
        # 누적합 = 에피소드 시작부터의 변위 적분 → 팔이 지금 어디쯤인지의 대리값
        cum = A[:(r + 1) * steps_per_rec].sum(0)
        out[r] = np.concatenate([seg.mean(0), seg.std(0), seg[-1], cum])
    return out


# ---------------------------------------------------------------- 본체

def run_one(path: Path, k: int, n_test_scene: int, seed: int, traj_idx=None):
    d = np.load(path, allow_pickle=True)
    Z = d["latent"].astype(np.float64)
    y = d["phase_code"].astype(np.int64)
    scene = d["scene"].astype(np.int64)
    ep, rec = d["ep_id"].astype(np.int64), d["rec_idx"].astype(np.int64)
    noise = (d["noise"].astype(np.int64) if "noise" in d.files
             else np.zeros(len(y), np.int64))
    slug = path.stem.replace("labels_", "").rsplit("_k", 1)[0]

    # 행동 대조군 특징 (정책이 실제로 낸 action) — 에피소드별로 붙인다
    A = None
    if traj_idx:
        A = np.full((len(y), 28), np.nan)
        for e in np.unique(d["ep_id"]):
            m = d["ep_id"] == e
            key = (slug, int(scene[m][0]), int(noise[m][0]))
            f = traj_idx.get(key)
            if f is None:
                continue
            feats = action_features(f, int(m.sum()))
            if feats is not None:
                ridx = d["rec_idx"][m].astype(int)
                ridx = np.clip(ridx - ridx.min(), 0, len(feats) - 1)
                A[m] = feats[ridx]     # npz 행 순서 그대로, record 번호로 직접 색인
        if np.isnan(A).all(1).any():
            A = None if np.isnan(A).all() else A

    ok = y >= 0
    Z, y, scene, ep, rec = Z[ok], y[ok], scene[ok], ep[ok], rec[ok]
    if A is not None:
        A = A[ok]

    scenes = np.unique(scene)
    rng = np.random.default_rng(seed)
    test_scenes = rng.choice(scenes, size=min(n_test_scene, max(len(scenes) - 1, 1)),
                             replace=False)
    te = np.isin(scene, test_scenes)
    tr = ~te
    if tr.sum() < k * 5 or te.sum() < 10:
        return None

    classes = np.unique(y)
    fallback = int(np.bincount(y[tr]).argmax())
    tfrac = time_fraction(ep, rec)

    res = {"n_train": int(tr.sum()), "n_test": int(te.sum()),
           "n_phase": int(len(classes)), "k": k,
           "test_scenes": [int(s) for s in np.sort(test_scenes)]}

    # 1) 다수 클래스
    pred = np.full(te.sum(), fallback)
    res["major_acc"] = float((pred == y[te]).mean())
    res["major_f1"] = macro_f1(y[te], pred, classes)

    # 2) clock — 진행도 분위 bin + 최빈 phase (activation 미사용)
    edges = np.quantile(tfrac[tr], np.linspace(0, 1, k + 1)[1:-1])
    b_tr, b_te = np.digitize(tfrac[tr], edges), np.digitize(tfrac[te], edges)
    m = majority_map(b_tr, y[tr], k, fallback)
    pred = m[b_te]
    res["clock_acc"] = float((pred == y[te]).mean())
    res["clock_f1"] = macro_f1(y[te], pred, classes)

    # 3) cluster — 비지도 군집 + 최빈 phase 매핑
    C = kmeans(Z[tr], k, seed=seed)
    m = majority_map(assign(Z[tr], C), y[tr], k, fallback)
    pred = m[assign(Z[te], C)]
    res["cluster_acc"] = float((pred == y[te]).mean())
    res["cluster_f1"] = macro_f1(y[te], pred, classes)

    # 4) probe — 지도 로지스틱 (activation)
    mdl = logistic(Z[tr], y[tr], classes, seed=seed)
    pred = logistic_predict(mdl, Z[te])
    res["probe_acc"] = float((pred == y[te]).mean())
    res["probe_f1"] = macro_f1(y[te], pred, classes)

    # 4b) causal time — 에피소드 길이를 모르는 온라인 상황: 절대 스텝 t 만
    edges_c = np.quantile(rec[tr].astype(float), np.linspace(0, 1, k + 1)[1:-1])
    m = majority_map(np.digitize(rec[tr].astype(float), edges_c), y[tr], k, fallback)
    pred = m[np.digitize(rec[te].astype(float), edges_c)]
    res["causal_time_acc"] = float((pred == y[te]).mean())
    res["causal_time_f1"] = macro_f1(y[te], pred, classes)

    # 4c) action — 정책이 낸 행동만 (외부 관찰자가 볼 수 있는 출력)
    if A is not None:
        good = ~np.isnan(A).any(1)
        if (good & tr).sum() > k * 5 and (good & te).sum() > 10:
            At, Ae = A[good & tr], A[good & te]
            yt, ye = y[good & tr], y[good & te]
            C = kmeans(At, k, seed=seed)
            mm = majority_map(assign(At, C), yt, k, fallback)
            pred = mm[assign(Ae, C)]
            res["action_cluster_acc"] = float((pred == ye).mean())
            res["action_cluster_f1"] = macro_f1(ye, pred, classes)
            mdl = logistic(At, yt, classes, seed=seed)
            pred = logistic_predict(mdl, Ae)
            res["action_probe_acc"] = float((pred == ye).mean())
            res["action_probe_f1"] = macro_f1(ye, pred, classes)

    # 4d) action + time — 외부에서 볼 수 있는 정보를 전부 합친 가장 강한 대조군
    if A is not None:
        good = ~np.isnan(A).any(1)
        if (good & tr).sum() > k * 5 and (good & te).sum() > 10:
            AT = np.hstack([A, tfrac[:, None], rec[:, None].astype(float)])
            mdl = logistic(AT[good & tr], y[good & tr], classes, seed=seed)
            pred = logistic_predict(mdl, AT[good & te])
            res["action_time_acc"] = float((pred == y[good & te]).mean())
            res["action_time_f1"] = macro_f1(y[good & te], pred, classes)

    # 5) probe_t — 지도 로지스틱 (진행도만)
    mdl = logistic(tfrac[tr][:, None], y[tr], classes, seed=seed)
    pred = logistic_predict(mdl, tfrac[te][:, None])
    res["probet_acc"] = float((pred == y[te]).mean())
    res["probet_f1"] = macro_f1(y[te], pred, classes)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labels-dir", type=Path, required=True,
                    help="labels_*.npz 가 있는 디렉토리")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("outputs/analysis/grid_phase/phase_readout"))
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--test-scenes", type=int, default=2)
    ap.add_argument("--seeds", default="0,1,2", help="scene split seed 목록")
    ap.add_argument("--traj-dir", type=Path, default=None,
                    help="traj.csv 트리 루트 — 있으면 행동 대조군을 함께 잰다")
    args = ap.parse_args(argv)

    files = sorted(args.labels_dir.glob("labels_*_k*.npz"))
    if not files:
        raise SystemExit(f"labels_*.npz 없음: {args.labels_dir}")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    traj_idx = index_traj(args.traj_dir) if args.traj_dir else None
    if traj_idx is not None:
        print(f'[traj] 색인 {len(traj_idx)} 에피소드', flush=True)

    per, rows = {}, []
    for f in files:
        name = f.stem.replace("labels_", "").rsplit("_k", 1)[0]
        runs = [r for r in (run_one(f, args.k, args.test_scenes, s, traj_idx)
                            for s in seeds) if r]
        if not runs:
            print(f"[skip] {name}", flush=True)
            continue
        avg = {key: float(np.mean([r[key] for r in runs]))
               for key in runs[0] if key.endswith(("_acc", "_f1"))}
        avg["n_phase"] = runs[0]["n_phase"]
        avg["n_test"] = int(np.mean([r["n_test"] for r in runs]))
        avg["n_seed"] = len(runs)
        per[name] = {"mean": avg, "runs": runs}
        rows.append((name, avg))
        print(f"[{name:<22}] major {avg['major_acc']:.3f} | clock {avg['clock_acc']:.3f} "
              f"| cluster {avg['cluster_acc']:.3f} | probe {avg['probe_acc']:.3f} "
              f"(probe_t {avg['probet_acc']:.3f})", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cols = [c for c in ("major_acc", "causal_time_acc", "clock_acc", "probet_acc",
                        "action_cluster_acc", "action_probe_acc", "action_time_acc",
                        "cluster_acc", "probe_acc",
                        "major_f1", "causal_time_f1", "clock_f1", "probet_f1",
                        "action_cluster_f1", "action_probe_f1", "action_time_f1",
                        "cluster_f1", "probe_f1")
            if any(c in a for _, a in rows)]
    with (args.out_dir / "readout.tsv").open("w") as fh:
        fh.write("instruction\tn_phase\tn_test\t" + "\t".join(cols) + "\n")
        for name, a in rows:
            fh.write(f"{name}\t{a['n_phase']}\t{a['n_test']}\t"
                     + "\t".join(f"{a[c]:.4f}" if c in a else "" for c in cols) + "\n")
    meta = {"script": "scripts/analysis/grid_phase/phase_readout.py",
            "k": args.k, "test_scenes": args.test_scenes, "seeds": seeds,
            "protocol": "scene 단위 held-out; 군집·매핑·probe 전부 train scene 에서만 적합",
            "numpy": np.__version__}
    (args.out_dir / "readout.json").write_text(
        json.dumps({"per_instruction": per, "meta": meta}, ensure_ascii=False, indent=1))

    med = {c: float(np.median([a[c] for _, a in rows if c in a])) for c in cols}
    print("\n== 중앙값 (instruction 10개) ==")
    for c in cols:
        if c.endswith("_acc"):
            f1 = c[:-4] + "_f1"
            print(f"  {c[:-4]:<14} acc {med[c]:.3f}"
                  + (f"   macroF1 {med[f1]:.3f}" if f1 in med else ""))


if __name__ == "__main__":
    main()

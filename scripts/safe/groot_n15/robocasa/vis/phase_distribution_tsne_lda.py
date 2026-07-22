"""GR00T N1.5 RoboCasa — action-phase별 latent 분포 시각화 (t-SNE + LDA).

각 record latent 벡터를 **action-phase(reach/transport/insert-settle …)로 색**, **succ/fail로 마커**
구분해 2D로 뿌린다. 목적: (1) phase별로 활성화가 분리된 군집을 이루나(=phase 구조 존재),
(2) 같은 phase 안에서 succ/fail가 갈리나 를 눈으로 확인 (Rung 2 정성 보조, 플랜 §Rung 2).

- t-SNE: 비지도 2D 임베딩(구조 그대로).
- LDA: **phase를 클래스로** 지도 투영(phase 간 최대분리 축) → phase 구조의 존재/강도 확인.
    (succ/fail LDA 분리도의 정량은 phase_separation.py 담당; 여기선 분포 가시화 전용.)

layer: DiT block-residual [0,2,4,8,10,12,15](denoise-pool) + VL. cell별 그리드 PNG 2장
(tsne_by_phase, lda_by_phase; 색=phase, 마커=succ●/fail✕).

deps: numpy, matplotlib, **scikit-learn**(t-SNE·LDA), torch(pkl). 원격에 sklearn 없으면:
  pip install --user scikit-learn
CPU cap: OMP/OPENBLAS_NUM_THREADS<=16 ([[cpu-budget-cap]]).
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHASE_COLORS = {
    "reach-to-object": "#3aa0ff",
    "reach-to-door": "#7ec8ff",
    "transport": "#ffb020",
    "insert-settle": "#38c172",
    "release-settle": "#8ce39a",
    "terminal": "#999999",
}
CAPTURE_LAYERS = [0, 2, 4, 8, 10, 12, 15]  # pkl capture_layers 로 덮어씀


def pool_denoise(rec: np.ndarray) -> np.ndarray:
    a = np.asarray(rec, dtype=np.float32)
    return a.mean(axis=1)  # [L,K,D] -> [L,D]


def load_rollout(pkl_path: Path) -> dict:
    import pickle
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    dit = np.stack([pool_denoise(np.asarray(r)) for r in d["hidden_states"]], axis=0)  # [n,L,D]
    vl = d.get("vl_hidden_states")
    vl_arr = np.stack([np.asarray(v, dtype=np.float32) for v in vl], axis=0) if vl is not None else None
    return {
        "success": int(d.get("episode_success", 0)),
        "dit": dit,
        "vl": vl_arr,
        "phases": list(d.get("feature_phases") or []),
        "capture_layers": list(d.get("capture_layers") or CAPTURE_LAYERS),
    }


def zscore(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return (X - mu) / sd


def _tsne(X: np.ndarray) -> np.ndarray:
    from sklearn.manifold import TSNE
    perp = max(5, min(30, (len(X) - 1) // 3))
    return TSNE(n_components=2, perplexity=perp, init="pca", random_state=0).fit_transform(X)


def _lda_2d(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    n_comp = min(2, len(np.unique(labels)) - 1, X.shape[1])
    if n_comp < 1:
        return np.zeros((len(X), 2))
    Z = LDA(n_components=n_comp).fit_transform(X, labels)
    if Z.shape[1] == 1:  # 2 phase → 1D, jitter 2번째 축
        Z = np.column_stack([Z[:, 0], np.random.default_rng(0).normal(0, 0.02, len(Z))])
    return Z


def _scatter(ax, emb, phases, succ, title):
    for ph in sorted(set(phases)):
        for s, marker, name in ((1, "o", "succ"), (0, "x", "fail")):
            m = np.array([(p == ph and y == s) for p, y in zip(phases, succ)])
            if not m.any():
                continue
            ax.scatter(emb[m, 0], emb[m, 1], s=14, marker=marker, alpha=0.6,
                       c=PHASE_COLORS.get(ph, "#000000"),
                       label=f"{ph}/{name}", edgecolors="none")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def gather_cell(rolls: list[dict], layer_idx) -> tuple[np.ndarray, list[str], np.ndarray]:
    """cell 전체 record → (X[N,D], phases[N], succ[N]) for given layer (int idx or 'VL')."""
    X, phases, succ = [], [], []
    for r in rolls:
        n = len(r["phases"])
        if layer_idx == "VL":
            if r["vl"] is None:
                continue
            feats = r["vl"]
        else:
            feats = r["dit"][:, layer_idx, :]
        for i in range(n):
            X.append(feats[i]); phases.append(r["phases"][i]); succ.append(r["success"])
    return np.asarray(X, dtype=np.float32), phases, np.asarray(succ)


def make_cell_figures(cid, rolls, layer_keys, layer_labels, out_dir, max_points):
    ncol = 4
    nrow = int(np.ceil(len(layer_keys) / ncol))
    for method in ("tsne", "lda"):
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow), squeeze=False)
        for k, lk in enumerate(layer_keys):
            ax = axes[k // ncol][k % ncol]
            X, phases, succ = gather_cell(rolls, lk)
            if len(X) < 5:
                ax.set_title(f"{layer_labels[lk]} (n<5)", fontsize=9); ax.axis("off"); continue
            if len(X) > max_points:  # 균등 subsample
                sel = np.random.default_rng(0).choice(len(X), max_points, replace=False)
                X, phases, succ = X[sel], [phases[i] for i in sel], succ[sel]
            Xz = zscore(X)
            emb = _tsne(Xz) if method == "tsne" else _lda_2d(Xz, np.array(phases))
            _scatter(ax, emb, phases, succ, f"{layer_labels[lk]}  (N={len(X)})")
        for k in range(len(layer_keys), nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")
        h, l = axes[0][0].get_legend_handles_labels()
        if h:
            fig.legend(h, l, loc="lower center", ncol=6, fontsize=8)
        fig.suptitle(f"{cid} — {method.upper()} by action-phase (color=phase, marker=succ●/fail✕)",
                     fontsize=12)
        fig.tight_layout(rect=(0, 0.04, 1, 0.97))
        out = out_dir / f"{cid}__{method}_by_phase.png"
        fig.savefig(out, dpi=120); plt.close(fig)
        print(f"[ok] {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="action-phase별 latent 분포 t-SNE/LDA 시각화")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cells", default=None, help="콤마구분 cell_id (기본 전체)")
    ap.add_argument("--max-points", type=int, default=4000)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cells = defaultdict(list)
    for p in sorted(run_dir.glob("*/*/*.pkl")):
        cells[p.parent.name].append(p)
    want = args.cells.split(",") if args.cells else None

    for cid, paths in cells.items():
        if want and cid not in want:
            continue
        rolls = [load_rollout(p) for p in paths]
        cap = rolls[0]["capture_layers"]
        layer_keys = list(range(rolls[0]["dit"].shape[1])) + (["VL"] if rolls[0]["vl"] is not None else [])
        layer_labels = {i: f"DiT-L{cap[i]}" for i in range(len(cap))}
        layer_labels["VL"] = "VL"
        print(f"[cell] {cid}: {len(rolls)} rollouts")
        make_cell_figures(cid, rolls, layer_keys, layer_labels, out_dir, args.max_points)
    print(f"[done] -> {out_dir}")


if __name__ == "__main__":
    main()

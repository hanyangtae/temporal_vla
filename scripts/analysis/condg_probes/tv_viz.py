"""시변 setpoint 설명 그림 2장 + bin 해상도 표 (승준 실행, PNG는 ~/tmp_tv_viz/)."""
import json
import os
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
BLUE, ORANGE = "#2a78d6", "#eb6834"
CMAP = LinearSegmentedColormap.from_list("blueseq", ["#c8dcf2", "#14477e"])
OUT = pathlib.Path(os.path.expanduser("~/tmp_tv_viz"))
OUT.mkdir(exist_ok=True)
BASE = os.path.expanduser("~/datasets/temporal_vla_store/groot/n15/analysis/grid_phase/segA/")
NQ = 4

plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.edgecolor": "#d8d7d2", "axes.grid": True,
                     "grid.color": "#eceae5", "grid.linewidth": 0.6,
                     "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "font.size": 10})


def load(slug, layer=12):
    d = np.load(BASE + slug + ".npz", allow_pickle=True)
    meta = json.loads(str(d["meta_json"]))
    cap = [int(x) for x in meta["capture_layers"]]
    X = d["X"][:, cap.index(layer), 3, 3, :].astype(np.float32)
    cb = {int(v): k for k, v in meta["phase_codebook"].items()}
    return d, X, cb


def phase_rows(d, pc, want_succ):
    succ = d["succ"].astype(bool)
    out = []
    for e in set(d["ep_id"][(d["phase_code"] == pc) & (succ == want_succ)]):
        idx = np.where((d["ep_id"] == e) & (d["phase_code"] == pc))[0]
        idx = idx[np.argsort(d["rec_idx"][idx])]
        if len(idx) >= NQ:
            out.append(idx)
    return out


def scene_center(d, X, pc):
    Xc = X.copy()
    succ = d["succ"].astype(bool)
    for s in set(d["scene"]):
        m = (d["scene"] == s)
        ref = X[m & succ & (d["phase_code"] == pc)]
        if len(ref) < 5:
            ref = X[m]
        Xc[m] = X[m] - ref.mean(0)
    return Xc


# ── Fig 1: drawer reach-to-handle 의 경로·관·실패·고정목표 ──────────────────
d, X, cb = load("OpenDrawer_left")
pc = {v: k for k, v in cb.items()}["reach-to-handle"]
Xc = scene_center(d, X, pc)
seps = phase_rows(d, pc, True)
feps = phase_rows(d, pc, False)
Xs_all = np.concatenate([Xc[i] for i in seps])
mu = Xs_all.mean(0)
U, S, Vt = np.linalg.svd(Xs_all - mu, full_matrices=False)
P = Vt[:2]
prj = lambda Z: (Z - mu) @ P.T

fig, ax = plt.subplots(figsize=(7.6, 6.2))
for idx in feps:
    p = prj(Xc[idx])
    ax.scatter(p[:, 0], p[:, 1], marker="x", s=14, c=ORANGE, alpha=0.25, lw=0.8)
for idx in seps:
    p = prj(Xc[idx])
    prog = np.linspace(0, 1, len(idx))
    ax.scatter(p[:, 0], p[:, 1], s=8, c=prog, cmap=CMAP, alpha=0.5, lw=0)
qmu, qrad = [], []
rows = [[] for _ in range(NQ)]
for idx in seps:
    for j, i in enumerate(idx):
        rows[min(NQ - 1, j * NQ // len(idx))].append(i)
for r in rows:
    Z = Xc[np.array(r)]
    qmu.append(Z.mean(0))
    qrad.append(float(np.median(np.linalg.norm(Z - Z.mean(0), axis=1))))
q2 = prj(np.stack(qmu))
for i in range(NQ - 1):
    ax.annotate("", xy=q2[i + 1], xytext=q2[i],
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=2))
for i, (x, y) in enumerate(q2):
    ax.scatter([x], [y], s=90, c=[CMAP(i / (NQ - 1))], edgecolors=INK, zorder=5)
    ax.annotate(f"q{i+1}", (x, y), textcoords="offset points", xytext=(8, 6),
                fontsize=11, color=INK, fontweight="bold")
g = prj(Xs_all.mean(0)[None])[0]
ax.scatter([g[0]], [g[1]], marker="*", s=260, c=INK, zorder=6)
ax.annotate("current fixed target\n(phase mean)", g, textcoords="offset points",
            xytext=(10, -26), fontsize=10, color=INK)
th = np.linspace(0, 2 * np.pi, 100)
r0 = qrad[1]
ax.plot(q2[1, 0] + r0 * np.cos(th), q2[1, 1] + r0 * np.sin(th),
        ls="--", c=INK2, lw=1.2)
ax.annotate("success tube radius\n(median dist. in q2)", (q2[1, 0] + r0 * 0.72, q2[1, 1] + r0 * 0.72),
            fontsize=9, color=INK2)
ax.scatter([], [], s=30, c=[CMAP(0.8)], label="success records (light→dark = progress)")
ax.scatter([], [], marker="x", s=30, c=ORANGE, label="failure records")
ax.legend(loc="lower right", frameon=False, fontsize=9)
ax.set_xlabel("PC1 (success, scene-centered)")
ax.set_ylabel("PC2")
ax.set_title("OpenDrawer/left · reach-to-handle · L12 — success moves along a path;\n"
             "the fixed phase-mean target sits mid-path", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "fig1_pca_path.png", dpi=160)
plt.close(fig)

# ── Fig 2: 진행도 τ̂ 곡선 (drawer reach / Dish reach) ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), sharey=True)
for ax, (slug, phname) in zip(
        axes, [("OpenDrawer_left", "reach-to-handle"), ("DishwasherRack_out", "reach-to-rack")]):
    d, X, cb = load(slug)
    pc = {v: k for k, v in cb.items()}[phname]
    Xc = scene_center(d, X, pc)
    seps = phase_rows(d, pc, True)
    rows = [[] for _ in range(NQ)]
    for idx in seps:
        for j, i in enumerate(idx):
            rows[min(NQ - 1, j * NQ // len(idx))].append(i)
    M = np.stack([Xc[np.array(r)].mean(0) for r in rows])

    def curves(eps_idx):
        cs = []
        for idx in eps_idx:
            tau = np.linalg.norm(Xc[idx][:, None, :] - M[None], axis=2).argmin(1) / (NQ - 1)
            pos = np.linspace(0, 1, len(idx))
            cs.append(np.interp(np.linspace(0, 1, 20), pos, tau))
        return np.stack(cs)

    for cs, col, name in [(curves(seps), BLUE, "success"),
                          (curves(phase_rows(d, pc, False)), ORANGE, "failure")]:
        for c in cs[:60]:
            ax.plot(np.linspace(0, 1, 20), c, c=col, alpha=0.06, lw=0.8)
        med = np.median(cs, 0)
        ax.plot(np.linspace(0, 1, 20), med, c=col, lw=2.6)
        ax.annotate(name, (1.0, med[-1]), textcoords="offset points", xytext=(4, 0),
                    color=col, fontsize=10, fontweight="bold", va="center")
    ax.plot([0, 1], [0, 1], ls="--", c=INK2, lw=1)
    ax.set_xlim(0, 1.12)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("position within phase (0→1)")
    ax.set_title(f"{slug} · {phname}", fontsize=10)
axes[0].set_ylabel("estimated progress τ̂ on success path")
fig.suptitle("Failure = stalling on the success path (median curves; thin lines = episodes)",
             fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "fig2_progress.png", dpi=160)
plt.close(fig)

# ── bin 해상도 표 (데이터 요구량 답변용) ────────────────────────────────────
print("slug\tphase\tQ\tbin당 n_succ\tSE(중심오차)\t인접 bin 간격\t간격/SE")
for slug, phname in [("OpenDrawer_left", "reach-to-handle"), ("OpenDrawer_left", "grasp-handle"),
                     ("DishwasherRack_out", "reach-to-rack"), ("PPCC_candle", "transport")]:
    d, X, cb = load(slug)
    pc = {v: k for k, v in cb.items()}[phname]
    Xc = scene_center(d, X, pc)
    seps = phase_rows(d, pc, True)
    for Q in (4, 8):
        rows = [[] for _ in range(Q)]
        for idx in seps:
            for j, i in enumerate(idx):
                rows[min(Q - 1, j * Q // len(idx))].append(i)
        if min(len(r) for r in rows) < 5:
            continue
        mus = [Xc[np.array(r)].mean(0) for r in rows]
        rad = np.median([np.median(np.linalg.norm(Xc[np.array(r)] - m, axis=1))
                         for r, m in zip(rows, mus)])
        n = int(np.mean([len(r) for r in rows]))
        se = rad / np.sqrt(n)
        step = float(np.mean([np.linalg.norm(mus[i + 1] - mus[i]) for i in range(Q - 1)]))
        print(f"{slug}\t{phname}\t{Q}\t{n}\t{se:.1f}\t{step:.1f}\t{step/se:.1f}")
print("VIZ_DONE")

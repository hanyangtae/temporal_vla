"""battery.log → heatmap 2장 (~/tmp_segb/figs/)."""
import os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = os.path.expanduser("~/tmp_segb/battery.log")
OUT = os.path.expanduser("~/tmp_segb/figs/")
os.makedirs(OUT, exist_ok=True)
rows = {}
scene_detail = {}
for ln in open(LOG):
    p = ln.rstrip("\n").split("\t")
    if len(p) < 7 or p[2] in ("SKIP",):
        continue
    slug, ph, met, val = p[0], p[1], p[2], float(p[3])
    rows.setdefault((slug, ph), {})[met] = val
    rows[(slug, ph)]["n"] = f"{p[4]}/{p[5]}"
    if met == "scene_margin":
        scene_detail[(slug, ph)] = p[6]

keys = [k for k in rows if "condg_margin" in rows[k]]
keys.sort(key=lambda k: (k[0], -rows[k]["condg_margin"]))
METS = ["condg_margin", "subspace_resid", "meandiff_proj", "dist_succ", "len_only", "scene_margin"]
LBL = ["condg margin\n(fixB)", "succ-subspace\nresid (fixB)", "mean-diff proj\n(fixB)",
       "dist-to-succ\n(fixB)", "length only", "WITHIN-scene\nmargin (med)"]
M = np.array([[rows[k].get(m, np.nan) for m in METS] for k in keys])
fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(keys) + 2))
im = ax.imshow(M, cmap="RdYlGn", vmin=0.3, vmax=1.0, aspect="auto")
ax.set_xticks(range(len(METS))); ax.set_xticklabels(LBL, fontsize=8)
ax.set_yticks(range(len(keys)))
ax.set_yticklabels([f"{k[0]} / {k[1]}  ({rows[k]['n']})" for k in keys], fontsize=8)
for i in range(len(keys)):
    for j in range(len(METS)):
        v = M[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="#111" if 0.42 < v < 0.93 else "#eee")
ax.set_title("grid v2 full battery - held-out AUROC (fail>succ), fixed-B fair window, 5-seed median\n"
             "gate rule: metric > length-only. Last col = same fit, ranked WITHIN each scene (deployment unit)",
             fontsize=9)
fig.colorbar(im, shrink=0.6)
fig.tight_layout()
fig.savefig(OUT + "battery_heatmap.png", dpi=150)

# per-scene scatter: pooled margin vs within-scene margins
fig2, ax2 = plt.subplots(figsize=(9, 6))
for (slug, ph), det in scene_detail.items():
    pooled = rows[(slug, ph)]["condg_margin"]
    for m_ in re.finditer(r"s(\d+):([\d.]+)\((\d+)/(\d+)\)", det):
        s_, v, nf_, ns_ = int(m_.group(1)), float(m_.group(2)), int(m_.group(3)), int(m_.group(4))
        n_min = min(nf_, ns_)
        ax2.scatter(pooled, v, s=14 + 6 * n_min, alpha=0.55,
                    c="#2a78d6" if "Drawer" in slug else "#eb6834" if "PPCC" in slug
                    else "#4a9e6b" if "Dish" in slug else "#8b5cf6" if "Oven" in slug else "#666")
ax2.axhline(0.5, color="#999", lw=0.8, ls="--"); ax2.axvline(0.5, color="#999", lw=0.8, ls="--")
ax2.plot([0, 1], [0, 1], color="#ccc", lw=0.8)
ax2.set_xlabel("pooled condg margin AUROC (task-level, fixB)")
ax2.set_ylabel("within-scene margin AUROC (per situation)")
ax2.set_title("pooled read vs within-situation read - same fit W, same window\n"
              "(blue=Drawer, orange=PPCC, green=Dish, purple=Oven, grey=Coffee; size=min class n)",
              fontsize=10)
ax2.set_xlim(0.45, 1.02); ax2.set_ylim(-0.02, 1.02)
fig2.tight_layout()
fig2.savefig(OUT + "pooled_vs_within.png", dpi=150)
print("HEATMAP_DONE")

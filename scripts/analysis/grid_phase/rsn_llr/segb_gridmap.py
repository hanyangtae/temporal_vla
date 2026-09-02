"""scene×noise succ/fail 격자 지도 10 task — PNG는 ~/tmp_segb/figs/."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.expanduser("~/datasets/temporal_vla_store/groot/n15/analysis/grid_phase/segB_v2/")
OUT = os.path.expanduser("~/tmp_segb/figs/")
os.makedirs(OUT, exist_ok=True)
SLUGS = ["OpenDrawer_left", "OpenDrawer_right", "DishwasherRack_out", "OvenRack_out",
         "PPCC_candle", "PPCC_bread", "PPCC_marshmallow", "PPCC_jug", "PPCC_apple",
         "CoffeeSetupMug"]
SURF = "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "font.size": 9})

fig, axes = plt.subplots(2, 5, figsize=(16, 7.2))
for ax, slug in zip(axes.flat, SLUGS):
    d = np.load(BASE + slug + ".npz", allow_pickle=True)
    ep, sc, no, su = d["ep_id"], d["scene"], d["noise"], d["succ"]
    G = np.full((15, 15), np.nan)
    n_ep = 0
    for e in np.unique(ep):
        i = np.where(ep == e)[0][0]
        G[int(sc[i]), int(no[i])] = int(su[i])
        n_ep += 1
    sr = np.nanmean(G)
    ax.imshow(G, cmap=matplotlib.colors.ListedColormap(["#d95f4b", "#4a9e6b"]),
              vmin=0, vmax=1, aspect="equal")
    miss = np.argwhere(np.isnan(G))
    if len(miss):
        ax.scatter(miss[:, 1], miss[:, 0], marker="x", c="#999", s=12, linewidths=0.8)
    # scene 별 s/f 수를 오른쪽에 표기
    for s_ in range(15):
        row = G[s_]
        ns_ = int(np.nansum(row)); nf_ = int(np.nansum(1 - row))
        ax.text(15.1, s_, f"{ns_}/{nf_}", va="center", fontsize=5.5, color="#52514e")
    ax.set_title(f"{slug}  SR={sr:.2f} (ep={n_ep})", fontsize=9)
    ax.set_xlabel("noise seed n"); ax.set_ylabel("scene s")
    ax.set_xticks(range(0, 15, 2)); ax.set_yticks(range(0, 15, 2))
    ax.tick_params(labelsize=6)
fig.suptitle("grid v2 - scene(situation) x noise seed succ/fail map (green=succ, red=fail, x=missing; right: s/f per scene)",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT + "grid_succfail_map.png", dpi=150)
print("GRIDMAP_DONE")
# scene 별 s/f 혼합 통계 (situation-level fit 가능성)
print("slug\tscene_mixed(3s3f+)\tscene_allS\tscene_allF")
for slug in SLUGS:
    d = np.load(BASE + slug + ".npz", allow_pickle=True)
    ep, sc, su = d["ep_id"], d["scene"], d["succ"]
    cnt = {}
    for e in np.unique(ep):
        i = np.where(ep == e)[0][0]
        cnt.setdefault(int(sc[i]), [0, 0])[int(su[i])] += 1
    mixed = sum(1 for v in cnt.values() if v[0] >= 3 and v[1] >= 3)
    allS = sum(1 for v in cnt.values() if v[0] == 0)
    allF = sum(1 for v in cnt.values() if v[1] == 0)
    print(f"{slug}\t{mixed}\t{allS}\t{allF}")

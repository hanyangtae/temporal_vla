"""phase별 경로 애니메이션 v2: 좌 = PCA 경로 그리기, 우 = 실제 rollout 3캠 세로 재생 (동기).

성공 rollout 전부 → 실패 rollout 전부 순서. 성공 = 실선, 실패 = 점선.
usage: python - <slug> <phase1,phase2,...>
"""
import json
import os
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np

SLUG = sys.argv[1]
PHASES = sys.argv[2].split(",")
LAYER, NQ, FPS = 12, 4, 8
NAS, SPR = 5, 2                      # n_action_steps / steps_per_render
SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
BLUE_BG, ORANGE_BG = "#9dbfe4", "#f2b39a"
INSTR = {
    "OpenDrawer_left": "Open the left drawer.",
    "DishwasherRack_out": "Fully slide the top dishwasher rack out.",
    "PPCC_candle": "Pick the candle from the counter and place it in the cabinet.",
}
HOME = os.path.expanduser("~")
OUT = pathlib.Path(HOME + "/tmp_tv_viz")
BASE = HOME + "/datasets/temporal_vla_store/groot/n15/analysis/grid_phase/segA/"
GRID = HOME + "/datasets/temporal_vla_store/groot/n15/grid/"

VMAP = {}
for ln in open(OUT / "vidmap.tsv"):
    s, sc, nz, rel = ln.rstrip("\n").split("\t")
    VMAP[(s, int(sc), int(nz))] = GRID + rel

d = np.load(BASE + SLUG + ".npz", allow_pickle=True)
meta = json.loads(str(d["meta_json"]))
cap = [int(x) for x in meta["capture_layers"]]
X0 = d["X"][:, cap.index(LAYER), 3, 3, :].astype(np.float64)
cb = {k: int(v) for k, v in meta["phase_codebook"].items()}
succ = d["succ"].astype(bool)

plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.edgecolor": "#d8d7d2", "axes.grid": True,
                     "grid.color": "#eceae5", "grid.linewidth": 0.6,
                     "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "font.size": 9})


CAM_CROP = 60   # 수집 원본 하단 오버레이 완전 제거(글자 최대 높이 대응)


def cam_stack(frame, height):
    """768폭 3캠 → 오버레이 제거 → 세로 스택 → height 리사이즈.

    수집 머신별 오버레이 위치가 다름: kanu(구 렌더러)=하단 글자(높이 256, 하단 60px
    crop), worker(배너판)=상단 배너(높이 256+α, 상단 α 제거·하단 무손상)."""
    h = frame.shape[0]
    if h > 256:                       # 상단 배너판
        body = frame[h - 256:]
        ch = 256
    else:                             # 하단 오버레이판
        body = frame
        ch = 256 - CAM_CROP
    cams = [body[:ch, i * 256:(i + 1) * 256] for i in range(3)]
    col = np.vstack(cams)
    w = max(2, int(round(256 * height / (3 * ch))))
    w -= w % 2
    return cv2.resize(col, (w, height))


def seg_frames(path, r0, r1):
    """record 구간 [r0,r1] 에 해당하는 video 프레임들."""
    c = cv2.VideoCapture(path)
    n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
    f0 = max(0, int(r0 * NAS / SPR))
    f1 = min(n - 1, int((r1 + 1) * NAS / SPR))
    c.set(cv2.CAP_PROP_POS_FRAMES, f0)
    out = []
    for _ in range(f0, f1 + 1):
        ok, fr = c.read()
        if not ok:
            break
        out.append(fr)
    c.release()
    return out


for phname in PHASES:
    pc = cb[phname]
    X = X0.copy()
    for s in set(d["scene"]):
        m = d["scene"] == s
        ref = X0[m & succ & (d["phase_code"] == pc)]
        X[m] = X0[m] - (ref.mean(0) if len(ref) >= 5 else X0[m].mean(0))

    def collect(want_succ):
        eps = []
        for e in set(d["ep_id"][(d["phase_code"] == pc) & (succ == want_succ)]):
            idx = np.where((d["ep_id"] == e) & (d["phase_code"] == pc))[0]
            idx = idx[np.argsort(d["rec_idx"][idx])]
            if len(idx) >= NQ:
                eps.append(idx)
        eps.sort(key=lambda i: (int(d["scene"][i[0]]), int(d["noise"][i[0]])))
        return eps

    seps, feps = collect(True), collect(False)
    if len(seps) < 5:
        print(f"[skip] {SLUG}/{phname}")
        continue

    Xs_all = np.concatenate([X[i] for i in seps])
    mu = Xs_all.mean(0)
    _, _, Vt = np.linalg.svd(Xs_all - mu, full_matrices=False)
    P = Vt[:2]
    prj = lambda Z: (Z - mu) @ P.T

    rows = [[] for _ in range(NQ)]
    for idx in seps:
        for j, i in enumerate(idx):
            rows[min(NQ - 1, j * NQ // len(idx))].append(i)
    q2 = prj(np.stack([X[np.array(r)].mean(0) for r in rows]))

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    fig.subplots_adjust(top=0.775, left=0.115, right=0.97, bottom=0.09)
    scn = sorted(set(int(d["scene"][i[0]]) for i in seps + feps))
    fig.text(0.03, 0.965, f"{SLUG} · {phname} · DiT L12", fontsize=11,
             fontweight="bold", color=INK)
    fig.text(0.03, 0.928, f'instr: "{INSTR.get(SLUG, "?")}"', fontsize=8.5, color=INK2)
    fig.text(0.03, 0.895, f"scenes: {', '.join('s'+str(s) for s in scn)}", fontsize=8.5,
             color=INK2)
    fig.text(0.03, 0.868, f"success {len(seps)} (solid) → failure {len(feps)} (dashed)",
             fontsize=8.5, color=INK2)
    dyn = fig.text(0.03, 0.833, "", fontsize=10, color=INK, fontweight="bold")

    ps = prj(Xs_all)
    ax.scatter(ps[:, 0], ps[:, 1], s=5, c=BLUE_BG, alpha=0.18, lw=0)
    fidx_all = np.where((d["phase_code"] == pc) & ~succ)[0]
    if len(fidx_all):
        pf = prj(X[fidx_all])
        ax.scatter(pf[:, 0], pf[:, 1], marker="x", s=8, c=ORANGE_BG, alpha=0.10, lw=0.7)
    for i in range(NQ - 1):
        ax.annotate("", xy=q2[i + 1], xytext=q2[i],
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.6, alpha=0.5))
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    fig.canvas.draw()
    pw, phh = fig.canvas.get_width_height()
    phh -= phh % 2
    vh = phh
    vw_cam = max(2, int(round(256 * vh / (3 * (256 - CAM_CROP)))))  # 최대폭 기준
    vw_cam -= vw_cam % 2
    W = pw + vw_cam
    W -= W % 2
    writer = cv2.VideoWriter(str(OUT / f"path2_{SLUG}_{phname}.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, vh))
    placeholder = np.full((vh, vw_cam, 3), 235, np.uint8)
    cv2.putText(placeholder, "rollout cam", (8, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (80, 80, 80), 1, cv2.LINE_AA)

    def emit(right=None, n=1):
        fig.canvas.draw()
        left = np.asarray(fig.canvas.buffer_rgba())[:vh, :pw, :3]
        left = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)
        fr = np.hstack([left, right if right is not None else placeholder])
        if fr.shape[1] < W:
            fr = np.pad(fr, ((0, 0), (0, W - fr.shape[1]), (0, 0)))
        for _ in range(n):
            writer.write(fr[:, :W])

    emit(n=FPS)
    cmap = plt.get_cmap("tab20")
    k_all = 0
    for group, eps_g, tag, ls in [(0, seps, "SUCCESS", "-"), (1, feps, "FAILURE", "--")]:
        for k, idx in enumerate(eps_g):
            p = prj(X[idx])
            col = cmap(k_all % 20)
            sc_i, nz_i = int(d["scene"][idx[0]]), int(d["noise"][idx[0]])
            dyn.set_text(f"{tag} {k+1}/{len(eps_g)} · scene s{sc_i} · noise n{nz_i}"
                         f" · {len(idx)} records")
            vpath = VMAP.get((SLUG, sc_i, nz_i))
            frames = seg_frames(vpath, int(d["rec_idx"][idx[0]]),
                                int(d["rec_idx"][idx[-1]])) if vpath else []
            steps = min(24, max(8, len(frames))) if frames else 8
            sel = (np.linspace(0, len(frames) - 1, steps).astype(int)
                   if frames else [None] * steps)
            (line,) = ax.plot([], [], c=col, lw=2.2, alpha=0.95, zorder=6, ls=ls)
            head = ax.scatter([], [], s=26, c=[col], zorder=7, edgecolors=INK, lw=0.4)
            for g, fi in enumerate(sel):
                n = max(2, int(round((g + 1) / steps * len(p))))
                line.set_data(p[:n, 0], p[:n, 1])
                head.set_offsets(p[n - 1:n])
                right = cam_stack(frames[fi], vh) if fi is not None else None
                emit(right)
            emit(cam_stack(frames[sel[-1]], vh) if frames else None, 1)
            head.remove()
            line.set_alpha(0.22)
            line.set_linewidth(1.0)
            line.set_zorder(3)
            k_all += 1
        if group == 0:
            dyn.set_text(f"--- all {len(seps)} SUCCESS done → now FAILURE rollouts ---")
            emit(n=FPS)
    dyn.set_text("success: shared path · failure: stall / off-tube")
    emit(n=FPS * 2)
    writer.release()
    plt.close(fig)
    print(f"[video] path2_{SLUG}_{phname}.mp4 (succ {len(seps)} + fail {len(feps)})")
print("VID2_DONE")

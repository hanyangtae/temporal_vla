#!/usr/bin/env python
"""rollout 영상 + AE latent 공간 궤적을 한 화면에 보여주는 mp4 렌더러.

레이아웃 (가로 배치):
    ┌───────────────────────────┬────────┐
    │  activation space (PCA-2) │  view0 │
    │   · 군집 영역·중심        │  view1 │
    │   · 지금까지의 궤적       │  view2 │
    ├───────────────────────────┴────────┤
    │  GT / AE 타임라인 띠 + 커서        │
    └────────────────────────────────────┘

수집 영상은 3개 카메라 뷰가 **가로로 이어붙어** 저장돼 있다. 이를 3등분해 오른쪽에
세로로 쌓고, 남는 자리에 latent 궤적을 그린다.

★ 규약: 장면 픽셀 위에는 어떤 텍스트·도형도 그리지 않는다. 뷰 타일은 잘라서 옮기기만
하고(픽셀 무손상), 글자는 전부 그 바깥 영역에만 그린다. 매 프레임 타일 3개가 원본과
bit-identical 인지 assert 한다.

입력
    --video   rollout mp4 (3 뷰 가로 concat)
    --labels  labels_<slug>_k8.npz  — ae_cluster.py --dump-labels 산출물
              (ep_id/rec_idx/scene/noise/succ/phase_code/cluster/latent/centers)
    --scene/--noise 또는 --ep-id 로 에피소드 선택

정렬: 1 record = 정책 추론 1회 = 환경 5 스텝, 영상 1 프레임 = 환경 2 스텝
      → record = floor(frame*2/5). 프레임/record 비가 다른 수집분은 비율로 자동 재정렬.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ENV_STEPS_PER_RECORD = 5
ENV_STEPS_PER_FRAME = 2

# 군집 색 (AE cluster id) — 타임라인·궤적·영역에서 동일하게 쓴다
CLUSTER_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]
PHASE_COLORS = [
    "#4C72B0", "#55A868", "#C44E52", "#8172B3", "#DD8452",
    "#937860", "#DA8BC3", "#CCB974",
]


def record_of_frame(f: int) -> int:
    return (f * ENV_STEPS_PER_FRAME) // ENV_STEPS_PER_RECORD


def load_episode(npz_path: Path, ep_id, scene, noise):
    d = np.load(npz_path, allow_pickle=True)
    need = ("ep_id", "rec_idx", "phase_code", "cluster", "latent")
    for k in need:
        if k not in d.files:
            raise SystemExit(f"{npz_path.name}: '{k}' 열 없음 — ae_cluster.py --dump-labels 로 재생성 필요")
    ep = d["ep_id"]
    if ep_id is not None:
        m = ep == int(ep_id)
    else:
        if scene is None:
            raise SystemExit("--ep-id 또는 --scene(+--noise) 중 하나는 필요")
        m = d["scene"] == int(scene)
        if noise is not None:
            if "noise" not in d.files:
                raise SystemExit("NPZ 에 noise 열이 없음 — --ep-id 로 지정")
            m = m & (d["noise"] == int(noise))
        ids = np.unique(ep[m])
        if len(ids) != 1:
            raise SystemExit(f"에피소드가 유일하지 않음: {ids[:10]} (--ep-id 로 지정)")
    if not m.any():
        raise SystemExit("해당 에피소드 record 없음")

    order = np.argsort(d["rec_idx"][m])
    out = {
        "ep_id": int(ep[m][0]),
        "phase": d["phase_code"][m][order],
        "cluster": d["cluster"][m][order],
        "latent_ep": d["latent"][m][order].astype(np.float64),
        "latent_all": d["latent"].astype(np.float64),
        "cluster_all": d["cluster"],
        "succ": int(d["succ"][m][0]) if "succ" in d.files else None,
        "centers": (d["centers"].astype(np.float64)
                    if "centers" in d.files and d["centers"].size else None),
    }
    cb = {}
    if "phase_codebook" in d.files:
        try:
            cb = json.loads(str(d["phase_codebook"]))
        except Exception:
            cb = {}
    out["code2phase"] = {int(v): k for k, v in cb.items()}
    return out


def pca2(X_fit, *others):
    """X_fit 으로 2차원 PCA 적합 후 others 도 같은 축으로 투영."""
    mu = X_fit.mean(0)
    Xc = X_fit - mu
    # 공분산 고유분해 (latent 차원이 작아 numpy 로 충분)
    C = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    w, V = np.linalg.eigh(C)
    W = V[:, np.argsort(w)[::-1][:2]]
    return [(np.asarray(o) - mu) @ W for o in (X_fit,) + others]


def make_space_panel(px, py, w_px, h_px, P_all, lab_all, P_ep, cl_ep, centers2, dpi=100):
    """군집 영역(배경)·중심·전체 산점도를 한 번 그려 배경 이미지로 캐시."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi)
    ax = fig.add_axes([0.10, 0.09, 0.88, 0.86])

    lo = np.percentile(P_all, 0.5, axis=0)
    hi = np.percentile(P_all, 99.5, axis=0)
    pad = 0.06 * (hi - lo)
    xlim = (lo[0] - pad[0], hi[0] + pad[0])
    ylim = (lo[1] - pad[1], hi[1] + pad[1])

    # 배경: 최근접 중심 기준 Voronoi 색칠 (= KMeans 가 그 좌표를 어느 군집으로 볼지)
    if centers2 is not None:
        gx, gy = np.meshgrid(np.linspace(*xlim, 320), np.linspace(*ylim, 320))
        G = np.stack([gx.ravel(), gy.ravel()], 1)
        d2 = ((G[:, None, :] - centers2[None, :, :]) ** 2).sum(-1)
        owner = d2.argmin(1).reshape(gx.shape)
        from matplotlib.colors import ListedColormap, to_rgb
        cmap = ListedColormap([tuple(0.82 + 0.18 * np.array(to_rgb(CLUSTER_COLORS[i % len(CLUSTER_COLORS)])))
                               for i in range(centers2.shape[0])])
        ax.imshow(owner, origin="lower", extent=(*xlim, *ylim), aspect="auto",
                  cmap=cmap, interpolation="nearest", zorder=0)

    # 이 instruction 전체 record 산점도 (옅게)
    step = max(1, len(P_all) // 6000)
    ax.scatter(P_all[::step, 0], P_all[::step, 1], s=1.2, alpha=0.18,
               c=[CLUSTER_COLORS[int(c) % len(CLUSTER_COLORS)] for c in lab_all[::step]],
               linewidths=0, zorder=1)

    if centers2 is not None:
        for i, c in enumerate(centers2):
            ax.scatter(*c, marker="X", s=42, zorder=5,
                       facecolor=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                       edgecolor="black", linewidths=0.7)
            ax.annotate(str(i), c, textcoords="offset points", xytext=(5, 4),
                        fontsize=7, color="black", zorder=6)

    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel("AE latent PC1", fontsize=8)
    ax.set_ylabel("AE latent PC2", fontsize=8)
    ax.set_title("activation trajectory in AE latent space", fontsize=9, pad=4)
    ax.tick_params(labelsize=7)
    for s in ax.spines.values():
        s.set_linewidth(0.6)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3][..., ::-1].copy()  # RGB→BGR
    # 데이터좌표 → 픽셀좌표 변환기 (캔버스 좌상단 기준)
    def to_px(P):
        pts = ax.transData.transform(np.asarray(P, float))
        H = buf.shape[0]
        return np.stack([pts[:, 0], H - pts[:, 1]], 1)
    plt.close(fig)
    return buf, to_px


def make_strip(w_px, h_px, phase, cluster, code2phase, dpi=100):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    T = len(phase)
    fig = plt.figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi)
    ax = fig.add_axes([0.055, 0.36, 0.60, 0.46])
    codes = sorted(set(int(p) for p in phase))
    cmap_p = {c: PHASE_COLORS[i % len(PHASE_COLORS)] for i, c in enumerate(codes)}
    for t in range(T):
        ax.add_patch(plt.Rectangle((t, 1.05), 1, 0.9, color=cmap_p[int(phase[t])], lw=0))
        ax.add_patch(plt.Rectangle((t, 0.05), 1, 0.9,
                                   color=CLUSTER_COLORS[int(cluster[t]) % len(CLUSTER_COLORS)], lw=0))
    ax.set_xlim(0, T); ax.set_ylim(0, 2)
    ax.set_yticks([0.5, 1.5]); ax.set_yticklabels(["AE k=8", "GT"], fontsize=8)
    ax.set_xlabel("record (inference step)", fontsize=8, labelpad=1)
    ax.tick_params(axis="x", labelsize=7)
    for s in ax.spines.values():
        s.set_visible(False)
    handles = [Patch(color=cmap_p[c], label=code2phase.get(c, f"code {c}")) for c in codes]
    fig.legend(handles=handles, loc="center left", ncol=1, frameon=False,
               fontsize=7, bbox_to_anchor=(0.68, 0.5), handlelength=1.2,
               labelspacing=0.25)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3][..., ::-1].copy()
    x0 = ax.transData.transform((0, 0))[0]
    x1 = ax.transData.transform((T, 0))[0]
    plt.close(fig)
    return buf, float(x0), float(x1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--ep-id", type=int, default=None)
    ap.add_argument("--scene", type=int, default=None)
    ap.add_argument("--noise", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--views", type=int, default=3, help="가로로 이어붙은 뷰 개수")
    ap.add_argument("--strip-height", type=int, default=110)
    ap.add_argument("--trail", type=int, default=0,
                    help="궤적 꼬리 길이(record). 0이면 처음부터 전부 표시")
    ap.add_argument("--limit-frames", type=int, default=0)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args(argv)

    import cv2

    ep = load_episode(args.labels, args.ep_id, args.scene, args.noise)
    T = len(ep["cluster"])

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"영상 열기 실패: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if W % args.views:
        raise SystemExit(f"프레임 폭 {W} 가 뷰 수 {args.views} 로 나누어떨어지지 않음")
    tw, th = W // args.views, H

    frame_to_record = record_of_frame
    if n_frames > 0 and abs(record_of_frame(n_frames - 1) + 1 - T) > 2:
        sc = T / float(n_frames)
        frame_to_record = lambda f, _s=sc: int(f * _s)
        print(f"[align] 프레임/record 비 재정렬: record = floor(frame x {sc:.4f})", flush=True)

    # 오른쪽 뷰 열: 타일을 세로로 쌓는다 (픽셀 그대로, 스케일 없음)
    right_w, right_h = tw, th * args.views
    space_w = max(right_h - 60, 420)          # 가운데 패널 폭
    canvas_w = space_w + right_w
    strip_h = int(args.strip_height)
    canvas_h = right_h + strip_h

    # PCA 축은 instruction 전체 record 로 적합 (군집 배치가 안정적으로 보이게)
    P_all, P_ep, P_cent = pca2(ep["latent_all"], ep["latent_ep"],
                               ep["centers"] if ep["centers"] is not None
                               else np.zeros((0, ep["latent_all"].shape[1])))
    centers2 = P_cent if len(P_cent) else None
    bg, to_px = make_space_panel(0, 0, space_w, right_h, P_all, ep["cluster_all"],
                                 P_ep, ep["cluster"], centers2)
    traj_px = to_px(P_ep)
    strip, sx0, sx1 = make_strip(canvas_w, strip_h, ep["phase"], ep["cluster"],
                                 ep["code2phase"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    for cc in ("avc1", "mp4v"):
        w = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*cc),
                            float(fps), (canvas_w, canvas_h))
        if w.isOpened():
            writer, codec = w, cc
            break
        w.release()
    if writer is None:
        raise SystemExit("VideoWriter 초기화 실패")

    n_written = 0
    while True:
        ok, frame = cap.read()
        if not ok or (args.limit_frames and n_written >= args.limit_frames):
            break
        r = min(frame_to_record(n_written), T - 1)

        canvas = np.full((canvas_h, canvas_w, 3), 255, np.uint8)
        # 오른쪽: 뷰 타일 세로 스택 (원본 픽셀 복사만)
        for v in range(args.views):
            tile = frame[:, v * tw:(v + 1) * tw]
            y0 = v * th
            canvas[y0:y0 + th, space_w:space_w + tw] = tile
            if not args.no_verify:
                assert (canvas[y0:y0 + th, space_w:space_w + tw] == tile).all(), \
                    "장면 픽셀 훼손 — 렌더 중단"
        # 가운데: latent 공간 배경 + 궤적
        canvas[0:right_h, 0:space_w] = bg[:right_h, :space_w]
        s = 0 if args.trail <= 0 else max(0, r - args.trail)
        pts = traj_px[s:r + 1].astype(np.int32)
        # 궤적은 그 시점의 군집 색으로 — "어느 영역을 지나 지금 군집이 됐는지"가 보이게
        for j in range(len(pts) - 1):
            cid = int(ep["cluster"][s + j])
            hexc = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
            col = tuple(int(hexc[i:i + 2], 16) for i in (5, 3, 1))
            cv2.line(canvas, tuple(pts[j]), tuple(pts[j + 1]), (255, 255, 255), 4,
                     cv2.LINE_AA)
            cv2.line(canvas, tuple(pts[j]), tuple(pts[j + 1]), col, 2, cv2.LINE_AA)
        for j, p in enumerate(pts[:-1]):
            cv2.circle(canvas, tuple(p), 2, (70, 70, 70), -1, cv2.LINE_AA)
        if len(pts):
            cur_c = CLUSTER_COLORS[int(ep["cluster"][r]) % len(CLUSTER_COLORS)]
            bgr = tuple(int(cur_c[i:i + 2], 16) for i in (5, 3, 1))
            cv2.circle(canvas, tuple(pts[-1]), 7, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, tuple(pts[-1]), 5, bgr, -1, cv2.LINE_AA)
        name = ep["code2phase"].get(int(ep["phase"][r]), f"code {int(ep['phase'][r])}")
        cv2.putText(canvas,
                    f"record {r + 1}/{T}   AE cluster {int(ep['cluster'][r])}   GT {name}",
                    (14, right_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (30, 30, 30), 1, cv2.LINE_AA)
        # 하단: 타임라인 + 커서
        row = strip.copy()
        cx = int(sx0 + (sx1 - sx0) * (r + 0.5) / T)
        cv2.line(row, (cx, 0), (cx, strip_h), (0, 0, 0), 2)
        canvas[right_h:right_h + strip_h, 0:canvas_w] = row

        writer.write(canvas)
        n_written += 1

    cap.release(); writer.release()
    print(f"[done] {args.out}  codec={codec} fps={fps:.1f} size={canvas_w}x{canvas_h} "
          f"frames={n_written} records={T} ep_id={ep['ep_id']} succ={ep['succ']}",
          flush=True)


if __name__ == "__main__":
    main()

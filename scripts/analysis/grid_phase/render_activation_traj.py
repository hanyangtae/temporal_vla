#!/usr/bin/env python3
"""rs_steer pilot: eval rollout activation trajectory video (fit-pool PCA 공간 위).

배경 = 연산자 fit-pool record 산점도(2D 사전 투영), 전경 = eval rollout의 record별
투영점 + 잔상(trail) + 한글 오버레이. 프레임 1개 = record 1개, mp4(libx264)로 인코딩.

사용 (matplotlib+ffmpeg 있는 컨테이너 안에서 실행):
  docker exec lerobot python /temporal_vla/scripts/analysis/grid_phase/render_activation_traj.py \
      --case OvenRack_out_s4_k1 --arm base \
      --pkl  /temporal_vla/outputs/.../rollout.pkl \
      --bg-npz /temporal_vla/outputs/analysis/v4_pilot_viz/OvenRack_out_s4_k1__setm.npz \
      --st 64 --trig 67 \
      --out /temporal_vla/outputs/analysis/v4_pilot_viz/videos/OvenRack_out_s4_k1__base.mp4

계약(어긋나면 즉시 에러 — 추측하지 않는다):
  * pkl['hidden_states'] 는 record 리스트, feature_axes == [layer, denoise_step, model_token, feature_dim]
  * pkl['capture_layers'] 안에 npz meta의 layer 가 있어야 한다
  * feature_dim == npz['mean'] 길이
  * record별 phase 라벨 길이 == record 수
  * (검출기) failure_scores 가 있으면 길이 == record 수. 없으면 그 줄을 생략한다
    — 다른 arm 의 계열을 빌려오지 않는다. δ(발화 임계) 는 pkl 에 저장되지 않으므로
    표시하지 않고, 저장된 trigger_step 만 '발화' 표기에 쓴다.
  * (로봇 3분할) pkl 옆 video.mp4 는 768x256 가로 3분할. record t → 영상 프레임
    round(t * n_action_steps / steps_per_render), 마지막 프레임으로 clamp.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys

import cv2

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# ---------------------------------------------------------------- 스타일 상수
BG_FIG = "#181a1e"
BG_AX = "#202329"
C_FAIL = "#e0605c"
C_SUCC = "#4fbf7b"
C_PATH = "#7e8794"
C_PRE = "#8fd3ff"      # 개입 전 궤적
C_POST = "#ffb03a"     # 개입 구간 궤적
C_TEXT = "#e8ecf2"
C_DIM = "#98a2b0"
TRAIL_N = 15

# 캔버스: 플롯 720x780 + 로봇 3분할 열 240x780 = 960x780 (yuv420p 용 짝수)
PLOT_W, PLOT_H = 720, 780
TILE = 240
COL_W = 240
GAP = 15                      # 15 + 240*3 + 15*3 = 780

FONT_CANDIDATES = [
    "/tmp/NanumGothic-Regular.ttf",
    os.path.expanduser("~/.fonts/NanumGothic-Regular.ttf"),
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]


def die(msg: str) -> "NoReturn":  # noqa: F821
    print(f"[render_activation_traj] FAIL: {msg}", file=sys.stderr)
    sys.exit(2)


def pick_font(explicit: str | None):
    cands = [explicit] if explicit else FONT_CANDIDATES
    for p in cands:
        if p and os.path.exists(p):
            fm.fontManager.addfont(p)
            return fm.FontProperties(fname=p).get_name()
    die(
        "한글 폰트를 찾지 못함 (--font 로 ttf 경로 지정). 후보: "
        + ", ".join(str(c) for c in cands)
    )


# ---------------------------------------------------------------- 데이터 적재
def load_bg(npz_path: str) -> dict:
    if not os.path.exists(npz_path):
        die(f"bg npz 없음: {npz_path}")
    z = np.load(npz_path, allow_pickle=True)
    for k in ("bg", "bg_label", "mean", "comps", "evr"):
        if k not in z.files:
            die(f"{npz_path}: 키 '{k}' 없음 (가진 키: {z.files})")
    meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    return {
        "bg": np.asarray(z["bg"], np.float32),
        "bg_label": np.asarray(z["bg_label"]).astype(int),
        "mean": np.asarray(z["mean"], np.float32).reshape(-1),
        "comps": np.asarray(z["comps"], np.float32),
        "evr": np.asarray(z["evr"], np.float32).reshape(-1),
        "meta": meta,
        "space": meta.get("space", "?"),
        "layer": meta.get("layer"),
        "denoise": meta.get("denoise"),
    }


def phase_labels(d: dict, n_rec: int) -> list[str]:
    """record별 phase 라벨. 길이가 record 수와 안 맞으면 fail-loud."""
    for key in ("feature_phases", "phase_timeline"):
        if key not in d:
            continue
        v = list(d[key])
        if len(v) == n_rec:
            return [str(x) for x in v]
        if key == "phase_timeline" and len(v) == n_rec + 1:
            # timeline 은 경계(t=0..T) 라 앞쪽 n_rec 개를 record 라벨로 쓴다
            return [str(x) for x in v[:n_rec]]
    have = {k: len(d[k]) for k in ("feature_phases", "phase_timeline") if k in d}
    die(f"record({n_rec}) 수와 맞는 phase 키 없음 — 발견: {have}")


def project_rollout(pkl_path: str, bg: dict) -> tuple[np.ndarray, dict]:
    if not os.path.exists(pkl_path):
        die(f"pkl 없음: {pkl_path}")
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    if not isinstance(d, dict) or "hidden_states" not in d:
        die(f"{pkl_path}: dict/hidden_states 아님")
    hs = d["hidden_states"]
    if len(hs) == 0:
        die(f"{pkl_path}: hidden_states 비어 있음")

    axes = list(d.get("feature_axes") or [])
    want_axes = ["layer", "denoise_step", "model_token", "feature_dim"]
    if axes != want_axes:
        die(
            f"{pkl_path}: feature_axes={axes} (기대 {want_axes}). "
            f"feature_kind={d.get('feature_kind')!r}, record_shape="
            f"{tuple(np.shape(hs[0]))} — 이 캡처는 PCA 공간(1536d block-residual)으로 "
            "투영할 수 없다."
        )

    cap = list(d.get("capture_layers") or [])
    layer = bg["layer"]
    if layer is None:
        die("npz meta 에 layer 없음")
    if layer not in cap:
        die(f"{pkl_path}: capture_layers={cap} 에 목표 layer={layer} 없음")
    li = cap.index(layer)

    dn = bg["denoise"]
    if dn is None:
        die("npz meta 에 denoise 없음")

    shp = tuple(np.shape(hs[0]))
    if len(shp) != 4:
        die(f"{pkl_path}: record ndim={len(shp)} (shape {shp}), 4 기대")
    n_dn, n_dim = shp[1], shp[3]
    dn_i = n_dn - 1 if dn in ("last", -1) else int(dn)
    if not (0 <= dn_i < n_dn):
        die(f"{pkl_path}: denoise index {dn_i} 범위 밖 (n_denoise={n_dn})")
    if n_dim != bg["mean"].shape[0]:
        die(f"{pkl_path}: feature_dim={n_dim} != npz mean dim={bg['mean'].shape[0]}")

    feats = np.empty((len(hs), n_dim), np.float32)
    for i, r in enumerate(hs):
        a = r[li, dn_i]                     # [token, dim]
        a = np.asarray(a.float().cpu().numpy() if hasattr(a, "float") else a, np.float32)
        feats[i] = a.mean(axis=0)           # float32 캐스팅 후 토큰 평균
    xy = (feats - bg["mean"][None, :]) @ bg["comps"].T
    return xy.astype(np.float32), d


# ------------------------------------------------------- 검출기 / 로봇 3분할
def detector_series(d: dict, n_rec: int):
    """per-record 검출기 점수. 없으면 None (다른 arm 것을 빌리지 않는다)."""
    fs = d.get("failure_scores")
    if fs is None:
        return None
    fs = list(fs)
    if len(fs) != n_rec:
        die(f"failure_scores 길이 {len(fs)} != record {n_rec}")
    return np.asarray(fs, np.float32)


def detector_deltas(d: dict, n_rec: int, ckpt_path: str | None):
    """발화 임계 δ_t. detector ckpt 의 cp_bands[task][α]["delta"] (band_L 넘으면 plateau).

    pkl 의 serve_failure_detector(band_L·bw) 와 대조해 같은 ckpt 인지 확인하고,
    어긋나면 다른 ckpt 이므로 즉시 실패한다 (틀린 δ 를 그리지 않는다)."""
    if not ckpt_path:
        return None
    sfd = d.get("serve_failure_detector")
    if not isinstance(sfd, dict):
        die("serve_failure_detector 없음 — δ 를 검증할 기준이 없다")
    if not os.path.exists(ckpt_path):
        die(f"detector ckpt 없음: {ckpt_path}")
    import torch  # 이 경로에서만 필요

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    bands = ck.get("cp_bands") or {}
    task = sfd.get("task")
    if task not in bands:
        die(f"{os.path.basename(ckpt_path)}: cp_bands 에 task '{task}' 없음 (있는 것: {sorted(bands)})")
    akey = f"{float(sfd.get('alpha', 0.1)):.2f}"
    if akey not in bands[task]:
        die(f"{os.path.basename(ckpt_path)}: task '{task}' 에 α={akey} 밴드 없음")
    band = bands[task][akey]
    dl = np.asarray(band["delta"], np.float32).ravel()
    if int(sfd.get("band_L", -1)) != int(dl.size):
        die(f"band_L 불일치: pkl {sfd.get('band_L')} vs ckpt {dl.size} — 다른 detector ckpt 다")
    if abs(float(band["bw"]) - float(sfd.get("bw", float("nan")))) > 1e-6:
        die(f"bw 불일치: pkl {sfd.get('bw')} vs ckpt {band['bw']} — 다른 detector ckpt 다")
    return dl[np.minimum(np.arange(n_rec), dl.size - 1)]


def robot_column(pkl_path: str, n_rec: int, d: dict, video_path: str | None):
    """diag video.mp4(768x256 가로 3분할) → record 별 세로 3단 열 이미지 [H,COL_W,3]."""
    vp = video_path or os.path.join(os.path.dirname(pkl_path), "video.mp4")
    if not os.path.exists(vp):
        die(f"로봇 영상 없음: {vp} (--video 로 지정하거나 --no-robot)")
    nas = int(d.get("n_action_steps") or 5)
    spr = int(d.get("steps_per_render") or 2)

    cap = cv2.VideoCapture(vp)
    if not cap.isOpened():
        die(f"영상 열기 실패: {vp}")
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if not frames:
        die(f"영상 프레임 0개: {vp}")
    h, w = frames[0].shape[:2]
    if w != 3 * h:
        die(f"{vp}: {w}x{h} — 가로 3분할(width == 3*height) 아님")

    want = [min(int(round(t * nas / spr)), len(frames) - 1) for t in range(n_rec)]
    cols = np.empty((n_rec, PLOT_H, COL_W, 3), np.uint8)
    cols[:] = np.array(matplotlib.colors.to_rgb(BG_FIG))[::-1] * 255  # BGR 배경
    x0 = (COL_W - TILE) // 2
    for i, fi in enumerate(want):
        fr = frames[fi]
        for j in range(3):
            tile = fr[:, j * h : (j + 1) * h]
            tile = cv2.resize(tile, (TILE, TILE), interpolation=cv2.INTER_AREA)
            y0 = GAP + j * (TILE + GAP)
            cols[i, y0 : y0 + TILE, x0 : x0 + TILE] = tile
    return cols[..., ::-1].copy()  # BGR -> RGB


# ---------------------------------------------------------------- 오버레이 문구
def intervention_label(arm: str, t: int, st: int, trig: int, phase: str) -> str:
    if arm == "base":
        return f"없음 (발화 r{trig})"
    if t < st:
        return "개입 전"
    if arm == "rs_early":
        return f"재추첨만 (r{st}~)"
    if arm == "rs_setm":
        return f"setM β1.0 · {phase}"
    if arm == "rs_condg":
        return f"condg β1.0 · {phase}"
    if arm == "rs_coast":
        return "COAST global β0.5"
    return f"{arm} (r{st}~)"


def space_label(bg: dict) -> str:
    sp = bg["space"]
    dn = bg["denoise"]
    name = {"setm": "setM", "condg": "condg"}.get(sp, sp)
    dn_s = "last" if (dn == 3 and sp == "setm") else str(dn)
    ev = bg["evr"]
    return f"{name} 공간 L{bg['layer']}/denoise {dn_s} · evr {ev[0]:.2f}/{ev[1]:.2f}"


# ---------------------------------------------------------------- 렌더
def render(args) -> dict:
    font = pick_font(args.font)
    plt.rcParams["font.family"] = [font, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    bg = load_bg(args.bg_npz)
    xy, d = project_rollout(args.pkl, bg)
    n = len(xy)
    phases = phase_labels(d, n)
    scores = detector_series(d, n)
    deltas = detector_deltas(d, n, args.detector_ckpt) if scores is not None else None
    succ = int(d.get("episode_success", 0) or 0)
    st, trig = int(args.st), int(args.trig)
    cols = None if args.no_robot else robot_column(args.pkl, n, d, args.video)

    W, H, DPI = PLOT_W, PLOT_H, 100
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG_FIG)
    ax = fig.add_axes([0.055, 0.105, 0.905, 0.795])
    axp = fig.add_axes([0.055, 0.042, 0.905, 0.022])
    for a in (ax, axp):
        a.set_facecolor(BG_AX)
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_color("#333a44")

    P, lab = bg["bg"], bg["bg_label"]
    ax.scatter(*P[lab != 1].T, s=7, c=C_FAIL, alpha=0.25, lw=0, zorder=1)
    ax.scatter(*P[lab == 1].T, s=7, c=C_SUCC, alpha=0.25, lw=0, zorder=1)

    allpts = np.vstack([P, xy])
    lo, hi = allpts.min(0), allpts.max(0)
    pad = 0.06 * np.maximum(hi - lo, 1e-6)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])

    ax.legend(
        handles=[
            Line2D([], [], ls="", marker="o", ms=6, mfc=C_SUCC, mec="none", label="fit-pool 성공"),
            Line2D([], [], ls="", marker="o", ms=6, mfc=C_FAIL, mec="none", label="fit-pool 실패"),
            Line2D([], [], ls="", marker="o", ms=9, mfc=C_PRE, mec="w", mew=1.0, label="현재 record"),
        ],
        loc="lower right", fontsize=9.5, framealpha=0.35, facecolor="#12141a",
        edgecolor="#333a44", labelcolor=C_DIM, handletextpad=0.5, borderpad=0.5,
    )

    (path_ln,) = ax.plot([], [], lw=1.0, color=C_PATH, alpha=0.35, zorder=3)
    trail = ax.scatter(np.zeros((0,)), np.zeros((0,)), s=24, lw=0, zorder=4)
    cur = ax.scatter([xy[0, 0]], [xy[0, 1]], s=170, c=[C_PRE], edgecolors="white",
                     linewidths=1.9, zorder=5)

    res = "성공" if succ else "실패"
    fig.text(0.055, 0.972, f"{args.case} · {args.arm} · 결과 {res}", color=C_TEXT,
             fontsize=14, va="top", ha="left")
    fig.text(0.055, 0.938, space_label(bg), color=C_DIM, fontsize=10.5, va="top", ha="left")

    txt = ax.text(0.022, 0.978, "", transform=ax.transAxes, color=C_TEXT, fontsize=12,
                  va="top", ha="left", linespacing=1.6, zorder=6,
                  bbox=dict(boxstyle="round,pad=0.5", fc="#12141a", ec="#3a424e", alpha=0.85))

    # 진행 바
    axp.set_xlim(0, max(n - 1, 1))
    axp.set_ylim(0, 1)
    prog = axp.axvspan(0, 0, color=C_PRE, alpha=0.85)
    show_st = (0 <= st < n) and args.arm != "base"
    show_tg = 0 <= trig < n
    if show_st:
        axp.axvline(st, color=C_POST, lw=1.6)
    if show_tg:
        axp.axvline(trig, color="#e8ecf2", lw=1.3, ls=":")
    if show_st and show_tg and abs(trig - st) < 0.10 * max(n - 1, 1):
        mid = 0.5 * (st + trig)
        ha = "left" if mid < 0.12 * n else ("right" if mid > 0.88 * n else "center")
        axp.text(mid, 1.5, f"개입 r{st} · 발화 r{trig}", color=C_POST, fontsize=9.5,
                 ha=ha, va="bottom")
    else:
        if show_st:
            axp.text(st, 1.5, f"개입 r{st}", color=C_POST, fontsize=9.5, ha="center", va="bottom")
        if show_tg:
            axp.text(trig, 1.5, f"발화 r{trig}", color=C_DIM, fontsize=9.5, ha="center", va="bottom")

    out_w = W + (0 if cols is None else COL_W)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{out_w}x{H}", "-r", str(args.fps),
        "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(args.crf),
        args.out,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    dump_at = args.dump_frame if args.dump_frame is not None else -1

    for t in range(n):
        post = (args.arm != "base") and (t >= st)
        col = C_POST if post else C_PRE
        path_ln.set_data(xy[: t + 1, 0], xy[: t + 1, 1])
        s0 = max(0, t - TRAIL_N)
        seg = xy[s0 : t + 1]
        k = len(seg)
        alphas = np.linspace(0.10, 0.75, k) if k > 1 else np.array([0.75])
        tc = np.array(
            [matplotlib.colors.to_rgba(C_POST if (args.arm != "base" and (s0 + i) >= st) else C_PRE, a)
             for i, a in enumerate(alphas)]
        )
        trail.set_offsets(seg)
        trail.set_facecolors(tc)
        trail.set_sizes(np.linspace(8, 52, k) if k > 1 else np.array([52.0]))
        cur.set_offsets(xy[t : t + 1])
        cur.set_facecolor(col)
        lines = [
            f"record {t}/{n - 1}",
            f"phase: {phases[t]}",
            f"개입: {intervention_label(args.arm, t, st, trig, phases[t])}",
        ]
        if scores is not None:
            fired = " ▲발화" if (show_tg and t >= trig) else ""
            dl = "" if deltas is None else f" / δ {deltas[t]:.3f}"
            lines.append(f"검출 p {scores[t]:.3f}{dl}{fired}")
        txt.set_text("\n".join(lines))
        prog.set_width(max(t, 1e-6))
        prog.set_color(col)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frame = buf if cols is None else np.concatenate([buf, cols[t]], axis=1)
        if t == dump_at and args.dump_png:
            plt.imsave(args.dump_png, frame)
        proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    proc.stdin.close()
    rc = proc.wait()
    plt.close(fig)
    if rc != 0:
        die(f"ffmpeg rc={rc}")
    size = os.path.getsize(args.out)
    info = {"case": args.case, "arm": args.arm, "records": n, "frames": n,
            "success": succ, "detector": scores is not None,
            "delta": deltas is not None,
            "size": f"{out_w}x{H}", "bytes": size, "out": args.out}
    print(json.dumps(info, ensure_ascii=False))
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--bg-npz", required=True)
    ap.add_argument("--st", type=int, required=True, help="개입(재추첨/steer) 시작 record")
    ap.add_argument("--trig", type=int, required=True, help="검출기 발화 record")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--crf", type=int, default=26)
    ap.add_argument("--video", default=None, help="로봇 3분할 영상 (기본: pkl 옆 video.mp4)")
    ap.add_argument("--no-robot", action="store_true", help="로봇 3분할 열 생략")
    ap.add_argument("--detector-ckpt", default=None,
                    help="발화 임계 δ_t 를 읽을 detector ckpt (pkl 의 band_L·bw 와 대조 검증)")
    ap.add_argument("--font", default=None)
    ap.add_argument("--dump-frame", type=int, default=None, help="이 record 프레임을 png 로 덤프")
    ap.add_argument("--dump-png", default=None)
    render(ap.parse_args())


if __name__ == "__main__":
    main()

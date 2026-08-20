#!/usr/bin/env python3
"""rollout 영상 + GT phase / AE cluster 타임라인 띠 렌더러.

목적
----
rollout mp4 아래에 두 줄짜리 타임라인 띠(1행 = GT phase, 2행 = AE cluster k=8)를
붙여, "전환이 어디서 일어나는지"를 눈으로 확인한다.

★ 절대 규칙 (레포 규약: caption burn-in 금지)
------------------------------------------------
영상 프레임 픽셀 위에는 텍스트도 박스도 **절대** 그리지 않는다. 모든 라벨/띠/커서는
캔버스를 아래로 확장한 영역에만 그린다. 매 프레임 원본 영역이 bit-identical 인지
assert 로 검증한다 (`--no-verify` 로만 끌 수 있으나 기본은 항상 켜짐).

정렬 규약
---------
- 1 record = 정책 추론 1회 = 환경 ENV_STEPS_PER_RECORD(=5) 스텝
- 영상은 환경 ENV_STEPS_PER_FRAME(=2) 스텝마다 1 프레임 저장
- 따라서 프레임 f ↔ record r(f) = floor(f * ENV_STEPS_PER_FRAME / ENV_STEPS_PER_RECORD)
                                = floor(2f / 5)

사용 예
-------
    ~/miniconda3/envs/lerobot_safe/bin/python \
        scripts/analysis/grid_phase/render_phase_timeline.py \
        --video .../OvenRack/out/s7/n11/base/video.mp4 \
        --labels outputs/.../labels_ovenrack_k8.npz \
        --ep-id 11 \
        --out /tmp/ovenrack_s7_n11_timeline.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- 정렬 상수
ENV_STEPS_PER_RECORD = 5   # 정책 추론 1회 = 환경 5 스텝
ENV_STEPS_PER_FRAME = 2    # 영상 1 프레임 = 환경 2 스텝


def record_of_frame(f: int) -> int:
    """프레임 인덱스 -> record 인덱스 (floor(2f/5))."""
    return (f * ENV_STEPS_PER_FRAME) // ENV_STEPS_PER_RECORD


def expected_frames(n_rec: int) -> float:
    """record 수에 대응하는 기대 프레임 수."""
    return n_rec * ENV_STEPS_PER_RECORD / ENV_STEPS_PER_FRAME


# ---------------------------------------------------------------- 팔레트
# GT phase: 이름별 고정 팔레트 (tab10 계열, 이름 정렬 순으로 안정 배정)
GT_PALETTE = [
    "#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e",
    "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
]
# AE cluster: cluster id 별 팔레트
AE_PALETTE = [
    "#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3",
    "#937860", "#da8bc3", "#8c8c8c", "#ccb974", "#64b5cd",
    "#b3446c", "#5f9ea0",
]


# ---------------------------------------------------------------- NPZ 로딩
def load_labels(path: Path):
    z = np.load(path, allow_pickle=True)
    keys = list(z.files)
    for req in ("ep_id", "rec_idx", "phase_code", "cluster"):
        if req not in keys:
            raise SystemExit(f"[error] labels NPZ 에 '{req}' 키가 없음. keys={keys}")
    d = {k: z[k] for k in keys}
    codebook = {}
    if "phase_codebook" in d:
        raw = d["phase_codebook"]
        try:
            s = raw.item() if getattr(raw, "shape", ()) == () else raw
            if isinstance(s, bytes):
                s = s.decode("utf-8")
            if isinstance(s, str):
                codebook = json.loads(s) if s.strip() else {}
            elif isinstance(s, dict):
                codebook = s
        except Exception as e:  # 코드북 없어도 렌더는 가능
            print(f"[warn] phase_codebook 파싱 실패 ({e}) — 코드 번호로 표기", flush=True)
    return d, keys, codebook


def find_noise_key(keys) -> str | None:
    """noise 열 후보 탐색 (스키마가 유동적이라 유연 처리)."""
    for cand in ("noise", "noise_id", "noise_idx", "n", "noise_level", "nid"):
        if cand in keys:
            return cand
    return None


def _as_str(v) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return str(v)


def select_episode(d, keys, args):
    n = len(np.asarray(d["ep_id"]))
    mask = np.ones(n, dtype=bool)
    desc = []

    if args.ep_id is not None:
        mask &= (np.asarray(d["ep_id"]).astype(np.int64) == int(args.ep_id))
        desc.append(f"ep_id={args.ep_id}")

    if args.scene is not None:
        if "scene" not in keys:
            raise SystemExit("[error] --scene 을 줬지만 NPZ 에 'scene' 열이 없음")
        sc = np.array([_as_str(x) for x in np.asarray(d["scene"])])
        mask &= (sc == str(args.scene))
        desc.append(f"scene={args.scene}")

    if args.noise is not None:
        nk = find_noise_key(keys)
        if nk is None:
            raise SystemExit(
                "[error] --noise 를 줬지만 NPZ 에 noise 열이 없음 "
                f"(keys={keys}). --ep-id 로 지정하라."
            )
        nv = np.array([_as_str(x) for x in np.asarray(d[nk])])
        mask &= (nv == str(args.noise))
        desc.append(f"{nk}={args.noise}")

    if args.ep_id is None and args.scene is None:
        raise SystemExit("[error] --ep-id 또는 --scene(+--noise) 중 하나는 필수")

    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise SystemExit(f"[error] 조건에 맞는 행 없음: {', '.join(desc)}")

    eps = np.unique(np.asarray(d["ep_id"])[idx])
    if eps.size > 1:
        raise SystemExit(
            f"[error] 조건이 여러 에피소드를 선택함 (ep_id={eps.tolist()}). "
            "--ep-id 로 하나만 지정하라."
        )

    order = np.argsort(np.asarray(d["rec_idx"])[idx], kind="stable")
    idx = idx[order]
    return idx, ", ".join(desc)


# ---------------------------------------------------------------- 띠 렌더
def render_band(width_px: int, height_px: int, phase_code, cluster, codebook,
                title: str, dpi: int = 100):
    """타임라인 띠를 numpy BGR 배열로 렌더하고, 커서 x 매핑 정보를 함께 반환."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    T = len(phase_code)
    # 좁은 영상에서도 글자/띠 비율이 깨지지 않도록 최소 폭으로 렌더 후 축소
    render_w = max(width_px, 720)
    render_h = max(height_px, int(round(height_px * render_w / max(width_px, 1))))

    # code -> name (codebook 은 {name: code})
    code2name = {}
    for name, code in (codebook or {}).items():
        try:
            code2name[int(code)] = str(name)
        except Exception:
            pass

    gt_codes = sorted(set(int(c) for c in phase_code))
    gt_names = [code2name.get(c, f"code {c}") for c in gt_codes]
    # 이름 기준 안정 배정 (같은 task 면 항상 같은 색)
    gt_order = sorted(range(len(gt_codes)), key=lambda i: gt_names[i])
    gt_color = {}
    for rank, i in enumerate(gt_order):
        gt_color[gt_codes[i]] = GT_PALETTE[rank % len(GT_PALETTE)]

    ae_ids = sorted(set(int(c) for c in cluster))
    ae_color = {c: AE_PALETTE[int(c) % len(AE_PALETTE)] for c in ae_ids}

    def rgba_row(vals, cmap):
        import matplotlib.colors as mcolors
        arr = np.zeros((1, T, 3), dtype=float)
        for j, v in enumerate(vals):
            arr[0, j] = mcolors.to_rgb(cmap[int(v)])
        return arr

    fig = plt.figure(figsize=(render_w / dpi, render_h / dpi), dpi=dpi)
    fig.patch.set_facecolor("white")

    # 축 배치(figure 분수 좌표) — 커서 픽셀 계산에 그대로 재사용
    # 세로 예산: title 0.90~1.0 / GT 0.66~0.86 / AE 0.42~0.62 / ticks 0.24~0.42 / legend 하단
    ax_left, ax_width = 0.10, 0.88
    ax_gt = fig.add_axes([ax_left, 0.66, ax_width, 0.20])
    ax_ae = fig.add_axes([ax_left, 0.42, ax_width, 0.20])

    for ax, row, cmap, label in (
        (ax_gt, phase_code, gt_color, "GT"),
        (ax_ae, cluster, ae_color, "AE k=8"),
    ):
        ax.imshow(rgba_row(row, cmap), aspect="auto",
                  extent=(0, T, 0, 1), interpolation="nearest")
        ax.set_xlim(0, T)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(-0.012, 0.5, label, transform=ax.transAxes, ha="right",
                va="center", fontsize=8.5, fontweight="bold")

    # x축 눈금은 아래 띠에만
    step = max(1, int(round(T / 8 / 5)) * 5) if T > 10 else max(1, T // 4)
    ticks = list(range(0, T + 1, step))
    ax_ae.set_xticks(ticks)
    ax_ae.tick_params(axis="x", labelsize=7, length=2, pad=1)
    ax_ae.set_xlabel("record (inference step)", fontsize=7, labelpad=1)

    # GT 범례 (확장 영역 안, 작게)
    handles = [Patch(facecolor=gt_color[c], label=code2name.get(c, f"code {c}"))
               for c in gt_codes]
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)),
                   fontsize=7, frameon=False, bbox_to_anchor=(0.5, 0.0),
                   handlelength=1.2, handleheight=0.8, columnspacing=1.2,
                   borderaxespad=0.2)

    if title:
        max_chars = max(20, int(render_w / 4.2))  # 대략적 폭 예산 (fontsize 7)
        if len(title) > max_chars:
            title = title[: max_chars - 1] + "…"
        fig.text(0.5, 0.995, title, ha="center", va="top", fontsize=7,
                 color="#333333")

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()  # RGB
    plt.close(fig)

    # 실제 렌더 크기 보정 (dpi 반올림)
    h, w = buf.shape[:2]
    if (w, h) != (width_px, height_px):
        import cv2
        buf = cv2.resize(buf, (width_px, height_px),
                         interpolation=cv2.INTER_AREA)

    band_bgr = buf[:, :, ::-1].copy()  # BGR

    cursor = {
        "x0": ax_left * width_px,
        "xw": ax_width * width_px,
        "T": T,
        # 커서 세로 범위: 두 띠를 모두 덮도록 (figure 분수 -> 픽셀, y 반전)
        "y_top": int(round((1.0 - 0.885) * height_px)),
        "y_bot": int(round((1.0 - 0.405) * height_px)),
    }
    return band_bgr, cursor


# ---------------------------------------------------------------- 메인
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, type=Path, help="rollout mp4")
    ap.add_argument("--labels", required=True, type=Path,
                    help="labels_<slug>_k8.npz")
    ap.add_argument("--ep-id", type=int, default=None)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--noise", default=None)
    ap.add_argument("--out", required=True, type=Path, help="출력 mp4")
    ap.add_argument("--band-height", type=int, default=110,
                    help="하단 확장 영역 높이 px (기본 110)")
    ap.add_argument("--no-auto-align", action="store_true",
                    help="프레임/record 불일치 시 비율 재정렬을 끄고 기본 규약 유지")
    ap.add_argument("--limit-frames", type=int, default=0,
                    help="디버그용: 앞쪽 N 프레임만 렌더 (0=전체)")
    ap.add_argument("--no-verify", action="store_true",
                    help="(비권장) 프레임 무손상 assert 끄기")
    args = ap.parse_args(argv)

    import cv2

    if not args.video.exists():
        raise SystemExit(f"[error] video 없음: {args.video}")
    if not args.labels.exists():
        raise SystemExit(f"[error] labels 없음: {args.labels}")

    d, keys, codebook = load_labels(args.labels)
    idx, desc = select_episode(d, keys, args)
    phase_code = np.asarray(d["phase_code"])[idx].astype(np.int64)
    cluster = np.asarray(d["cluster"])[idx].astype(np.int64)
    rec_idx = np.asarray(d["rec_idx"])[idx].astype(np.int64)
    n_rec = len(idx)
    ep_id = int(np.asarray(d["ep_id"])[idx][0])
    succ = None
    if "succ" in keys:
        succ = int(np.asarray(d["succ"])[idx][0])

    if not np.array_equal(rec_idx, np.arange(rec_idx[0], rec_idx[0] + n_rec)):
        print(f"[warn] rec_idx 가 연속이 아님 (min={rec_idx.min()}, "
              f"max={rec_idx.max()}, n={n_rec})", flush=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"[error] video 열기 실패: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    n_frames_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_to_record = record_of_frame   # 기본 규약 (5 env-step/record, 2 env-step/frame)
    # 프레임 수 ↔ record 수 정합성 (±2 record 허용)
    exp_f = expected_frames(n_rec)
    if n_frames_meta > 0:
        rec_from_frames = record_of_frame(n_frames_meta - 1) + 1
        if abs(rec_from_frames - n_rec) > 2:
            print(f"[warn] 프레임/record 불일치: frames={n_frames_meta} "
                  f"-> records≈{rec_from_frames}, NPZ records={n_rec} "
                  f"(기대 frames≈{exp_f:.1f}, 허용 ±2 record)", flush=True)
            # 수집 라운드마다 영상 저장 간격이 달라질 수 있다. 두 열이 같은
            # 에피소드 전 구간을 덮는다는 전제 하에 비율로 재정렬한다.
            if not args.no_auto_align:
                scale = n_rec / float(n_frames_meta)
                frame_to_record = lambda f, _s=scale: int(f * _s)
                print(f"[align] 자동 재정렬: record = floor(frame x {scale:.4f})",
                      flush=True)

    title = (f"{args.labels.stem}  |  ep_id={ep_id}"
             + (f"  succ={succ}" if succ is not None else "")
             + f"  |  {desc}  |  records={n_rec}, frames={n_frames_meta}")
    band_h = int(args.band_height)
    band, cur = render_band(W, band_h, phase_code, cluster, codebook, title)

    out_h = H + band_h
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    for cc in ("avc1", "mp4v"):
        w = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*cc),
                            float(fps), (W, out_h))
        if w.isOpened():
            writer, codec = w, cc
            break
        w.release()
    if writer is None:
        raise SystemExit("[error] VideoWriter 열기 실패 (avc1/mp4v 모두)")

    n_written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.limit_frames and n_written >= args.limit_frames:
            break

        canvas = np.zeros((out_h, W, 3), dtype=np.uint8)
        # 1) 원본 프레임: 그대로 복사 (절대 덮어쓰지 않음)
        canvas[0:H, 0:W] = frame
        # 2) 확장 영역: 띠 + 커서
        strip = band.copy()
        r = min(frame_to_record(n_written), cur["T"] - 1)
        # 커서는 현재 record 구간의 중앙
        x = int(round(cur["x0"] + (r + 0.5) / cur["T"] * cur["xw"]))
        x = max(0, min(W - 1, x))
        cv2.line(strip, (x, cur["y_top"]), (x, cur["y_bot"]),
                 (0, 0, 0), 2, cv2.LINE_8)
        canvas[H:out_h, 0:W] = strip

        # ★ 무손상 검증: 상단 영역이 입력 프레임과 bit-identical
        if not args.no_verify:
            assert (canvas[0:H, 0:W] == frame).all(), \
                f"frame {n_written}: 원본 영역이 훼손됨 (caption burn-in 사고)"

        writer.write(canvas)
        n_written += 1

    cap.release()
    writer.release()

    print(f"[done] {args.out}  codec={codec} fps={fps:.3f} "
          f"size={W}x{out_h} frames={n_written} records={n_rec}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

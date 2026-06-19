"""SAFE-LSTM detector 실패 검출 시각화 영상 (DiT 검출기 vs VL 검출기).

학습된 LSTM detector를 실패 rollout 영상에 입혀, **검출 시점(score>δ)부터 테두리를 빨강**으로.
- detector: DiT latent 학습 / VL latent 학습 — 각각 별도.
- 학습: seen 8 task(80%). CP 임계 δ: seen 성공 max-score (1-α)분위(FPR≈α).
- 데모 대상: task별 **non-train 실패** 5개. → task당 영상 10개(DiT 5 + VL 5).
- 정렬: detector는 inference-step k에서 검출 → 영상은 추론당 N프레임(여기 8)이라 frame k*N부터 빨강.
cv2 사용(원격 anaconda). pkl 옆 동명 .mp4 사용. 재사용: pathway_lstm_detector 학습/스코어 함수.
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

import numpy as np
import cv2

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from pathway_separation import load_rollout_features  # noqa: E402
from pathway_lstm_detector import (  # noqa: E402
    CAPTURE_LAYERS, pathway_seq, make_seqs, standardizer, train_lstm, score_seq,
    functional_cp_band,
)


def load_with_path(run_dir: Path, token_pool: str, tasks: set | None):
    rolls = []
    for td in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        if tasks and td.name not in tasks:
            continue
        for p in sorted(td.glob("*.pkl")):
            r = load_rollout_features(p, token_pool)
            if r is None:
                continue
            r["task"] = td.name
            r["pkl"] = p
            rolls.append(r)
    return rolls


def rollout_score(model, r, pathway, dit_idx, mu, sd, device):
    X = pathway_seq(r, pathway, dit_idx)
    if X is None:
        return None
    Xn = ((X - mu) / sd).astype(np.float32)
    return score_seq(model, Xn, device)


def render(mp4_in: Path, mp4_out: Path, scores: np.ndarray, band: np.ndarray, pw: str):
    cap = cv2.VideoCapture(str(mp4_in))
    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    L = len(scores)
    fpi = nf / max(L, 1)  # frames per inference step

    def bval(k):  # 시간가변 functional CP 밴드 δ_t
        return float(band[min(k, len(band) - 1)])

    fire_step = next((k for k in range(L) if scores[k] > bval(k)), None)
    fire_frame = fire_step * fpi if fire_step is not None else float("inf")
    vw = cv2.VideoWriter(str(mp4_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        step = min(int(i / fpi), L - 1)
        sc = float(scores[step])
        fired = i >= fire_frame
        color = (0, 0, 255) if fired else (200, 200, 200)
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 14)
        cv2.putText(frame, f"{pw.upper()} det  score={sc:.2f}/band={bval(step):.2f}",
                    (18, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if fired:
            cv2.putText(frame, f"FAILURE DETECTED @step {fire_step}", (18, h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        vw.write(frame)
        i += 1
    cap.release(); vw.release()
    return fire_step, L


def main():
    ap = argparse.ArgumentParser(description="detector 실패 검출 시각화 영상")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--token-pool", default="valid16")
    ap.add_argument("--dit-block", type=int, default=31)
    ap.add_argument("--detectors", default="dit,vl")
    ap.add_argument("--n-per-task", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.3, help="functional CP 유의수준(FPR 목표). 밴드 δ_t 결정")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lambda-reg", type=float, default=1e-2, help="SAFE식 L2 정규화")
    ap.add_argument("--grad-clip", type=float, default=0.0, help="grad clip(0=off=SAFE 기본)")
    ap.add_argument("--seen", default="CloseToasterOvenDoor,NavigateKitchen,OpenCabinet,PickPlaceCounterToStove,PickPlaceDrawerToCounter,SlideDishwasherRack,TurnOnMicrowave,TurnOnSinkFaucet")
    ap.add_argument("--unseen", default="OpenDrawer,PickPlaceCounterToCabinet")
    ap.add_argument("--video-tasks", default=None, help="영상 만들 task(콤마, 기본 전체)")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cpu"
    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir.parent / "analysis" / "detector_videos"
    out.mkdir(parents=True, exist_ok=True)
    dit_idx = int(np.argmin([abs(b - args.dit_block) for b in CAPTURE_LAYERS]))
    detectors = args.detectors.split(",")
    seen, unseen = set(args.seen.split(",")), set(args.unseen.split(","))
    vtasks = set(args.video_tasks.split(",")) if args.video_tasks else (seen | unseen)

    # 학습엔 전체 task 필요, 영상은 vtasks만 → 둘 다 로드
    rolls = load_with_path(run_dir, args.token_pool, seen | unseen | vtasks)
    # split: 각 seen task 80% train
    rng = np.random.default_rng(args.seed)
    by = defaultdict(list)
    for r in rolls:
        if r["task"] in seen:
            by[r["task"]].append(r)
    train_ids = set()
    seen_test = []
    for task, rs in by.items():
        idx = np.arange(len(rs)); rng.shuffle(idx)
        n = int(round(args.train_frac * len(rs)))
        for i in idx[:n]:
            train_ids.add(id(rs[i]))
        seen_test += [rs[i] for i in idx[n:]]
    train = [r for r in rolls if id(r) in train_ids]
    L_max = max((r["length"] for r in rolls), default=45)
    print(f"[load] {len(rolls)} rollouts | train={len(train)} seen_test={len(seen_test)} | L_max={L_max}")

    # detector별 학습 + functional CP 밴드 δ_t 보정(seen_test 성공)
    models = {}
    for pw in detectors:
        tr0 = make_seqs(train, pw, dit_idx)
        mu, sd = standardizer(tr0)
        tr = make_seqs(train, pw, dit_idx, mu, sd)
        st_seqs = make_seqs(seen_test, pw, dit_idx, mu, sd)
        model = train_lstm(tr, tr[0][0].shape[1], args.epochs, 1e-3, 256, device, args.seed,
                            args.lambda_reg, args.grad_clip)
        band = functional_cp_band(model, tr, st_seqs, device, args.alpha, L_max)  # [L_max]
        models[pw] = (model, mu, sd, band)
        print(f"[train] {pw}: dim={tr[0][0].shape[1]} band(α={args.alpha}) δ_t∈[{band.min():.3f},{band.max():.3f}]")

    # 영상: task별 non-train 실패 n개
    manifest = []
    for task in sorted(vtasks):
        fails = sorted([r for r in rolls if r["task"] == task and r["success"] == 0
                        and id(r) not in train_ids], key=lambda r: r["pkl"].name)
        picked = fails[: args.n_per_task]
        tdir = out / task; tdir.mkdir(parents=True, exist_ok=True)
        print(f"[video] {task}: {len(picked)} 실패 rollout × {len(detectors)} detector")
        for r in picked:
            mp4_in = r["pkl"].with_suffix(".mp4")
            if not mp4_in.exists():
                print(f"  [skip] no mp4: {mp4_in.name}"); continue
            ep = r["pkl"].stem
            for pw in detectors:
                model, mu, sd, band = models[pw]
                sc = rollout_score(model, r, pw, dit_idx, mu, sd, device)
                if sc is None:
                    continue
                mp4_out = tdir / f"{ep}--{pw}_det.mp4"
                fire, L = render(mp4_in, mp4_out, sc, band, pw)
                manifest.append({"task": task, "ep": ep, "detector": pw,
                                 "fire_step": fire, "n_steps": L, "max_score": round(float(sc.max()), 3),
                                 "band_range": [round(float(band.min()), 3), round(float(band.max()), 3)],
                                 "out": str(mp4_out.relative_to(out))})
                print(f"    {ep} [{pw}] fire_step={fire}/{L} max={sc.max():.2f}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[done] {len(manifest)} videos -> {out}/")


if __name__ == "__main__":
    main()

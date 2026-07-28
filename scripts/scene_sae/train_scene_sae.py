#!/usr/bin/env python
"""exp5/G1 — scene SAE 학습 (핸드아웃 §4 Phase C).

입력 = build_sae_inputs.py 산출 (X_L{L}.npz / stats_L{L}.npz / meta.npz).
train split 행으로만 학습하고 val split 으로 early stop, test 는 손대지 않는다
(핸드아웃 §6-2 episode 단위 split, §6-4 in-sample 아티팩트).

표준화 mean/std 는 **이 실행이 고른 --split-col 의 train 행에서 직접 계산**하고 ckpt 옆
`stats.npz` 로 저장한다 (probe 가 같은 통계를 쓴다). `--stats` 는 참고 대조용 — 값이 다르면
경고만 낸다 (리뷰 #1: 빌더 stats 는 빌더 --split-by 축 기준이라 축이 다르면 누수).

산출: outputs/eval/robocasa/groot_n15/scene_sae/<cell>/L{L}_m{m}_k{k}_s{seed}/
      model.pt(state_dict) · config.json · stats.npz · curve.tsv · metrics.json(dead 비율 등)

실행 예 (로컬 GPU 1장, 완전히 빈 GPU 확인 후):
  python scripts/scene_sae/train_scene_sae.py \
      --x  outputs/scene_sae/pq3_drawer_left/X_L10.npz \
      --stats outputs/scene_sae/pq3_drawer_left/stats_L10.npz \
      --meta outputs/scene_sae/pq3_drawer_left/meta.npz \
      --cell pq3_drawer_left --layer 10 --m 6144 --k 32 --device cuda:0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# ==========================================================================
# src/sae 인터페이스 — **이 블록이 유일한 접점. 라이브러리가 바뀌면 여기만 고친다.**
# --------------------------------------------------------------------------
# src/sae/ 는 동료 레포(robots-oh task_classification, dev 88543a2)의
# phase/{models,train} 코어를 lift 한 것 (핸드아웃 §2.1 / §4 Phase A).
# 2026-07-27 실물 확인한 계약:
#
#   src.sae.models.sae_config(input_dim, expansion=4, k=32, loss="mse") -> cfg dict
#       cfg = {"kind":"ae","input_dim":D,"latent_dim":m,
#              "encoder":{"type":"topk","k":k},"decoder":{"type":"linear_dict"},"loss":"mse"}
#   src.sae.models.build_model(cfg, input_dim=None) -> nn.Module
#       모델 계약: .loss(x) -> 0-dim 텐서, .latent(x) -> [N, m] (no_grad)
#   src.sae.train.train_sae(model, x_train, x_val, *, epochs, patience, min_epochs,
#                           batch_size, lr, weight_decay, optimizer, grad_clip,
#                           device, seed, log_every, verbose)
#       -> best {"loss","epoch","state","history"(=[(ep,tr,va)...]),"config"}
#       (데이터는 CPU numpy/텐서로 넘기고 배치만 device 로 옮긴다 — 메모리 안전)
#   src.sae.train.encode_all(model, x, batch_size, device) -> z [N, m] numpy
#   src.sae.cluster.dead_fraction(z) -> 한 번도 켜지지 않은 feature 비율
#
# expansion 대신 --m 을 직접 받으므로 sae_config 결과의 latent_dim 을 덮어쓴다.
# ==========================================================================
def sae_api():
    """(sae_config, build_model, train_sae, encode_all, dead_fraction)."""
    from src.sae.cluster import dead_fraction          # noqa: PLC0415
    from src.sae.models import build_model, sae_config  # noqa: PLC0415
    from src.sae.train import encode_all, train_sae     # noqa: PLC0415
    return sae_config, build_model, train_sae, encode_all, dead_fraction


def build_sae_cfg(input_dim: int, m: int, k: int, loss: str = "mse",
                  aux_k: int = 0, aux_alpha: float = 1.0 / 32,
                  dead_window: int = 50) -> dict:
    sae_config, _b, _t, _e, _d = sae_api()
    cfg = sae_config(input_dim, expansion=1, k=k, loss=loss,
                     aux_k=aux_k, aux_alpha=aux_alpha, dead_window=dead_window)
    cfg["latent_dim"] = int(m)          # --m 직접 지정 (expansion 우회)
    return cfg


# ------------------------------------------------------------------------ util
def load_split_array(X: np.ndarray, mask: np.ndarray, mu, sd) -> np.ndarray:
    """표준화(train 통계)한 [N, D] float32. X 는 원시값 저장이라 여기서 정규화한다."""
    a = X[mask].astype(np.float32)
    a -= mu
    a /= sd
    return a


def compute_train_stats(X: np.ndarray, tr_mask: np.ndarray, chunk: int = 200_000):
    """**이 학습이 실제로 쓰는 train split 행**에서 feature-wise mean/std 계산 (리뷰 #1).

    빌더의 stats_L*.npz 는 빌더 `--split-by` 축의 train 행으로 만들어진다. 학습이
    `--split-col split_scene` 처럼 **다른 축**을 고르면 표준화 통계가 test scene 행까지
    포함한 것이 되어 조용한 누수가 된다. 그래서 통계는 여기서 직접 만들고,
    `--stats` 는 참고 대조용으로만 쓴다.
    (fp16 X 를 통째로 float32 로 올리지 않도록 청크 누적 — 160판이면 수십만 행.)
    """
    idx = np.flatnonzero(tr_mask)
    if len(idx) == 0:
        raise SystemExit("train split 행이 0 — 표준화 통계 계산 불가 (--split-col 확인)")
    D = X.shape[1]
    s1 = np.zeros(D, np.float64)
    s2 = np.zeros(D, np.float64)
    for i in range(0, len(idx), chunk):
        a = X[idx[i:i + chunk]].astype(np.float64)
        s1 += a.sum(axis=0)
        s2 += (a * a).sum(axis=0)
    n = float(len(idx))
    mu = s1 / n
    var = np.maximum(s2 / n - mu * mu, 0.0)
    sd = np.sqrt(var)
    sd = np.where(sd < 1e-6, 1.0, sd)                    # 상수 feature 보호 (빌더와 동일 규칙)
    return mu.astype(np.float32), sd.astype(np.float32), int(len(idx))


def episode_fingerprint(eps) -> str:
    """정렬된 episode 목록의 sha256[:12] (리뷰 #2 — probe 가 같은 train 집합인지 대조)."""
    payload = ",".join(str(int(e)) for e in sorted({int(v) for v in eps}))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def main() -> None:
    ap = argparse.ArgumentParser(description="scene SAE 학습 (top-k, overcomplete)")
    ap.add_argument("--x", required=True, type=Path)
    ap.add_argument("--stats", type=Path, default=None,
                    help="빌더 stats_L*.npz — **참고 검증용**. 표준화 통계는 --split-col 의 "
                         "train 행에서 직접 계산하고, 이 파일과 다르면 경고만 낸다 (리뷰 #1)")
    ap.add_argument("--meta", required=True, type=Path)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--m", type=int, default=6144, help="사전 크기 (기본 4×D)")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--min-epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0, help="동료 관례 wd=0")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--aux-k", type=int, default=0,
                    help="AuxK dead-feature 보조손실에서 되살릴 feature 수 "
                         "(0=끔·기본, 권장 512). L0 dead 0.449 대응 — docs/steering/31 §5-1")
    ap.add_argument("--aux-alpha", type=float, default=1.0 / 32,
                    help="L_total = L_recon + aux_alpha · L_aux (원논문 1/32)")
    ap.add_argument("--dead-window", type=int, default=50,
                    help="최근 몇 학습 스텝 미활성이면 dead 로 볼지 (배치 수)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--split-col", default="split",
                    help="meta.npz 의 split 컬럼. scene-matched 빌드는 split_episode/"
                         "split_scene 도 들어 있다 (기본 split = 빌더 --split-by 축)")
    args = ap.parse_args()

    meta = np.load(args.meta, allow_pickle=False)
    if args.split_col not in meta.files:
        raise SystemExit(f"meta 에 '{args.split_col}' 없음. 사용 가능: "
                         f"{[k for k in meta.files if k.startswith('split')]}")
    split = meta[args.split_col]
    Xz = np.load(args.x)
    X = Xz["X"]
    if len(X) != len(split):
        raise SystemExit(f"X 행수 {len(X)} != meta {len(split)}")
    # 행 지문 대조 (리뷰 #10) — 같은 out-dir 에 다른 수집이 섞이면 X↔meta 행이 어긋난다
    fp_x = str(Xz["row_fingerprint"]) if "row_fingerprint" in Xz.files else None
    fp_m = str(meta["row_fingerprint"]) if "row_fingerprint" in meta.files else None
    if fp_x is not None and fp_m is not None and fp_x != fp_m:
        raise SystemExit(f"row_fingerprint 불일치 — X={fp_x} vs meta={fp_m} "
                         f"(다른 빌드 산출물이 섞였다. 재빌드 필요)")
    if fp_x is None or fp_m is None:
        print("[warn] row_fingerprint 없음 (구 빌드 산출물) — 행 정합 대조 생략", flush=True)
    D = X.shape[1]

    # 표준화 통계: **이 split_col 의 train 행**에서 직접 계산 (리뷰 #1)
    mu, sd, n_stat_rows = compute_train_stats(X, split == 0)
    stats_check = {"stats_arg": str(args.stats) if args.stats else None}
    if args.stats is not None and Path(args.stats).exists():
        st = np.load(args.stats)
        d_mu = float(np.abs(st["mean"] - mu).max())
        d_sd = float(np.abs(st["std"] - sd).max())
        stats_check.update({"max_abs_diff_mean": d_mu, "max_abs_diff_std": d_sd})
        if d_mu > 1e-3 or d_sd > 1e-3:
            print(f"[warn] --stats 파일과 자체 계산 통계 불일치 (Δmean={d_mu:.4g} Δstd={d_sd:.4g})"
                  f" — split_col={args.split_col} 축이 빌더 기본축과 다르면 정상. "
                  f"학습은 **자체 계산 통계**를 쓴다.", flush=True)
    print(f"[train] 표준화 통계 자체 계산: train rows={n_stat_rows} split_col={args.split_col}",
          flush=True)

    Xtr = load_split_array(X, split == 0, mu, sd)
    Xva = load_split_array(X, split == 1, mu, sd)
    if len(Xva) == 0:
        raise SystemExit("val split 행이 0 — early stop 불가 (split 규칙 확인)")
    print(f"[train] L{args.layer} D={D} m={args.m} k={args.k} "
          f"train={Xtr.shape} val={Xva.shape} device={args.device}", flush=True)

    _cfgf, build_model, train_sae, encode_all, dead_fraction = sae_api()
    sae_cfg = build_sae_cfg(D, args.m, args.k, aux_k=args.aux_k,
                            aux_alpha=args.aux_alpha, dead_window=args.dead_window)
    model = build_model(sae_cfg)

    t0 = time.time()
    best = train_sae(model, Xtr, Xva, epochs=args.epochs, patience=args.patience,
                     min_epochs=args.min_epochs, batch_size=args.batch_size, lr=args.lr,
                     weight_decay=args.weight_decay, optimizer="adam",
                     grad_clip=args.grad_clip, device=args.device, seed=args.seed,
                     log_every=10)
    dt = time.time() - t0
    curve = best.get("history", [])

    # ---- 진단: dead feature 비율 + 실효 희소도 (train 행 기준)
    ztr = encode_all(model, Xtr, batch_size=args.batch_size, device=args.device)
    dead_ratio = float(dead_fraction(ztr))
    density = float((ztr > 0).mean())

    out_dir = args.out_dir or (REPO / "outputs/eval/robocasa/groot_n15/scene_sae" / args.cell /
                               f"L{args.layer}_m{args.m}_k{args.k}_s{args.seed}")
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    # probe 가 **같은** 표준화 통계를 쓰도록 ckpt 옆에 저장 (리뷰 #1)
    np.savez(out_dir / "stats.npz", mean=mu, std=sd, n_train_rows=np.int64(n_stat_rows),
             split_col=np.asarray(args.split_col))

    # train 집합 지문 (리뷰 #2) — probe 가 같은 split 을 보고 있는지 대조하는 근거
    tr_eps = sorted({int(v) for v in meta["episode_idx"][split == 0]})
    tr_scenes = (sorted({int(v) for v in meta["scenario_seed"][split == 0]})
                 if "scenario_seed" in meta.files else [])
    split_axis_scene = False
    if "scenario_seed" in meta.files:
        te_scenes = {int(v) for v in meta["scenario_seed"][split == 2]}
        split_axis_scene = bool(te_scenes) and not (te_scenes & set(tr_scenes))
    cfg = {
        "cell": args.cell, "layer": args.layer, "input_dim": D, "m": args.m, "k": args.k,
        "seed": args.seed, "sae_source": "src.sae", "loss": "mse",
        "epochs": args.epochs, "patience": args.patience, "min_epochs": args.min_epochs,
        "batch_size": args.batch_size, "lr": args.lr, "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "aux_k": args.aux_k, "aux_alpha": args.aux_alpha, "dead_window": args.dead_window,
        "x": str(args.x), "stats": str(args.stats) if args.stats else None,
        "meta": str(args.meta),
        # ---- 리뷰 #1/#2/#10: split 축 + 데이터 지문 (probe 가 대조한다)
        "split_col": args.split_col,
        "split_axis_scene_heldout": split_axis_scene,
        "train_episode_fingerprint": episode_fingerprint(tr_eps),
        "n_train_episodes": len(tr_eps),
        "train_episodes": tr_eps,
        "train_scenes": tr_scenes if split_axis_scene else None,
        "row_fingerprint": fp_m,
        "stats_source": "self_computed_from_train_split",
        "stats_check_vs_arg": stats_check,
        "sae_cfg": sae_cfg,          # probe 가 이 dict 로 모델을 재조립한다
        "train_config": best.get("config"),
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    metrics = {
        "best_epoch": best["epoch"], "best_val_loss": best["loss"],
        "n_train_rows": int(len(Xtr)), "n_val_rows": int(len(Xva)),
        "dead_feature_ratio": dead_ratio, "density": density,
        "density_expected": args.k / args.m, "train_seconds": dt,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    with open(out_dir / "curve.tsv", "w") as f:
        f.write("epoch\ttrain_loss\tval_loss\n")
        for ep, tr, va in curve:
            f.write(f"{ep}\t{tr:.6f}\t{va:.6f}\n")

    print(f"[train] best ep{best['epoch']} val={best['loss']:.4f} "
          f"dead={dead_ratio:.3f} density={density:.5f} (기대 {args.k/args.m:.5f}) "
          f"{dt:.0f}s → {out_dir}", flush=True)
    if dead_ratio > 0.5:
        print("[train] ⚠ dead-feature 비율 > 0.5 — 핸드아웃 §4-C1 선택 기준 미달 (m/k 재조정 후보)")


if __name__ == "__main__":
    sys.exit(main())

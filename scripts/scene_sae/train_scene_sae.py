#!/usr/bin/env python
"""exp5/G1 — scene SAE 학습 (핸드아웃 §4 Phase C).

입력 = build_sae_inputs.py 산출 (X_L{L}.npz / stats_L{L}.npz / meta.npz).
train split 행으로만 학습하고 val split 으로 early stop, test 는 손대지 않는다
(핸드아웃 §6-2 episode 단위 split, §6-4 in-sample 아티팩트).

산출: outputs/eval/robocasa/groot_n15/scene_sae/<cell>/L{L}_m{m}_k{k}_s{seed}/
      model.pt(state_dict) · config.json · curve.tsv · metrics.json(dead-feature 비율 등)

실행 예 (로컬 GPU 1장, 완전히 빈 GPU 확인 후):
  python scripts/scene_sae/train_scene_sae.py \
      --x  outputs/scene_sae/pq3_drawer_left/X_L10.npz \
      --stats outputs/scene_sae/pq3_drawer_left/stats_L10.npz \
      --meta outputs/scene_sae/pq3_drawer_left/meta.npz \
      --cell pq3_drawer_left --layer 10 --m 6144 --k 32 --device cuda:0
"""
from __future__ import annotations

import argparse
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


def build_sae_cfg(input_dim: int, m: int, k: int, loss: str = "mse") -> dict:
    sae_config, _b, _t, _e, _d = sae_api()
    cfg = sae_config(input_dim, expansion=1, k=k, loss=loss)
    cfg["latent_dim"] = int(m)          # --m 직접 지정 (expansion 우회)
    return cfg


# ------------------------------------------------------------------------ util
def load_split_array(X: np.ndarray, mask: np.ndarray, mu, sd) -> np.ndarray:
    """표준화(train 통계)한 [N, D] float32. X 는 원시값 저장이라 여기서 정규화한다."""
    a = X[mask].astype(np.float32)
    a -= mu
    a /= sd
    return a


def main() -> None:
    ap = argparse.ArgumentParser(description="scene SAE 학습 (top-k, overcomplete)")
    ap.add_argument("--x", required=True, type=Path)
    ap.add_argument("--stats", required=True, type=Path)
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
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    meta = np.load(args.meta, allow_pickle=False)
    split = meta["split"]
    st = np.load(args.stats)
    mu, sd = st["mean"], st["std"]
    Xz = np.load(args.x)
    X = Xz["X"]
    if len(X) != len(split):
        raise SystemExit(f"X 행수 {len(X)} != meta {len(split)}")
    D = X.shape[1]

    Xtr = load_split_array(X, split == 0, mu, sd)
    Xva = load_split_array(X, split == 1, mu, sd)
    if len(Xva) == 0:
        raise SystemExit("val split 행이 0 — early stop 불가 (split 규칙 확인)")
    print(f"[train] L{args.layer} D={D} m={args.m} k={args.k} "
          f"train={Xtr.shape} val={Xva.shape} device={args.device}", flush=True)

    _cfgf, build_model, train_sae, encode_all, dead_fraction = sae_api()
    sae_cfg = build_sae_cfg(D, args.m, args.k)
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
    cfg = {
        "cell": args.cell, "layer": args.layer, "input_dim": D, "m": args.m, "k": args.k,
        "seed": args.seed, "sae_source": "src.sae", "loss": "mse",
        "epochs": args.epochs, "patience": args.patience, "min_epochs": args.min_epochs,
        "batch_size": args.batch_size, "lr": args.lr, "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "x": str(args.x), "stats": str(args.stats), "meta": str(args.meta),
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

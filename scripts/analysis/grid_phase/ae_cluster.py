#!/usr/bin/env python
"""논문 파이프라인 확정본: **raw-1536 → AE(latent 16) → KMeans** 재산출기.

동료(상우) 실험 B 종합 판정 = "전처리는 raw-1536 + AE, PCA 불필요, PCA 를 안 할 때는
whitening 금지". 같은 표가 `docs/steering/40_action_phase_readout_review.md` §4·§5 부록에
있다. 기존 `intrinsic_phase.py` / `paper_supplements.py` 는 PCA-64 whitened → KMeans 였고,
이 스크립트는 그 앞단을 AE 로 바꾼 경로 하나만 담당한다 (930 에피소드 재산출용).

    DiT layer 12 · denoise 3 · 49토큰 평균 (1536d, raw)
      → 전역 표준화(평균 제거 + **스칼라** std, 축별 whitening 아님)
      → AE encoder (1536 → 256 → 256 → 16)
      → KMeans (instruction 별 k=8 / global k=24)
      → GT phase 대비 MI / margin / purity / boundary F1

동료 AE 규격 (실제 코드 확인 — 레포 루트 `task_classification/` 체크아웃)
--------------------------------------------------------------------------
  * `phase/models/autoencoder.py: Encoder`
        Linear(D, 256) → GELU → Linear(256, 256) → GELU → Linear(256, d)
  * `phase/models/autoencoder.py: Decoder`
        Linear(d, 256) → GELU → Linear(256, 256) → GELU → Linear(256, D)
        + 차원별 학습 파라미터 `logvar` (clamp [-8, 4])
  * `phase/models/autoencoder.py: BaseAE.loss`
        loss="log_likelihood" (conf/model/ae.yaml 기본값) =
        대각 가우시안 NLL 0.5·[(x-x̂)²·exp(-lv) + lv + log2π], feature 축 **합** → 스텝 평균.
        loss="mse" 면 logvar 를 0 에 동결 (동일 식이 0.5·Σ(x-x̂)² 가 된다).
  * `conf/model/ae.yaml`      latent_dim 16, encoder/decoder hidden 256, loss log_likelihood
  * `conf/train/default.yaml` optimizer adamw(lr 1e-3, weight_decay 1e-4), grad_clip 5.0,
                              epochs 800, patience 60, min_epochs 60, batch_size null(full-batch)
  * `phase/train/_loop.py: fit`  val 손실 최소화 early stopping + best state 복원
  * `phase/train/posthoc.py`     학습 후 **train 잠재**를 KMeans (cluster K=24, conf/cluster/default.yaml)

**의도적 차이 (meta.deviations 에도 기록)**
  1. 입력이 PCA-64 whitened 가 아니라 raw-1536 (실험 B 판정). AE 입력 스케일을 맞추려고
     전역 평균 제거 + 스칼라 std 나눗셈만 한다 — 축별 분산 정규화(whitening)는 금지 조건.
  2. 동료는 GPU full-batch 800 epoch. 여기는 CPU 전용이라 미니배치(기본 4096) +
     기본 200 epoch (`--epochs`, `--batch-size`, `--patience` 로 조정). optimizer·lr·
     weight_decay·grad_clip·early stopping 규칙은 그대로.
  3. AE 는 **전 shard 를 모아 1개만** 학습하고(동료도 혼합 학습), 그 encoder 로 각
     instruction feature 를 16차원으로 인코딩한 뒤 instruction 별 KMeans 를 돌린다.

shard 로딩 · KMeans 래퍼 · align_metrics 는 새로 구현하지 않고 `intrinsic_phase.py` 를,
scene 잔차화 · 분할표 집계는 `paper_supplements.py` 를 그대로 import 해서 쓴다
(grid_phase 는 패키지가 아니라 spec_from_file_location 로 로드).

사용 예
    python scripts/analysis/grid_phase/ae_cluster.py \
        --shard-dir ~/datasets/temporal_vla_store/groot/n15/analysis/grid_phase/segA \
        --mode all \
        --kmeans-src ~/workspace/temporal_vla/task_classification/phase/clustering/gpu.py

출력 (--out-dir, 기본 outputs/analysis/grid_phase/ae_raw)
    ae_pertask_k8.json              instruction 별 align 지표 (+ AE 학습 요약)
    ae_global_k24.json              global KMeans align 지표 (per_shard + pooled)
    resid_compare_ae.tsv / .json    scene 잔차화 대비표
    contingency_pertask_k8_ae.json  instruction 별 cluster×GT phase 분할표
    contingency_global_k24_ae.json  global 분할표 (+ cluster×task)

`--export-bundle <path.npz>` 를 주면 **온라인 판정기 번들** 을 하나 더 쓴다 (기존 산출물
불변). 번들 = mu[1536] + scalar_std + encoder state_dict(`enc.*`) + instruction 별
KMeans centers(`centers.<slug>`) + arch/provenance json. 새 rollout 을 학습 때와 같은
좌표계로 배정하는 데 필요한 전부다: standardize → encoder → 최근접 center.
`--dump-labels` 와 같이 쓰면 라벨과 번들이 **같은 KMeans 결과** 인지 재배정으로 검증한다.

실행 환경: 승준 노드 `~/anaconda3/bin/python` (numpy + torch **CPU**). GPU 를 쓰지 않는다.
scipy / sklearn 없음 — numpy + torch 만 쓴다.
"""
from __future__ import annotations

import os

# BLAS/torch 스레드 cap — 공유 노드. numpy/torch import 전에 설정해야 효력이 있다.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_p = Path(__file__).resolve()
REPO_ROOT = _p.parents[3] if len(_p.parents) > 3 else Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_REL = "scripts/analysis/grid_phase/ae_cluster.py"

LOG2PI = float(np.log(2 * np.pi))

# ---- 고정 파라미터 (기준 run 과 동일: align 지표 쪽) --------------------------------
N_INIT = 5
MAX_ITER = 300
BOUNDARY_TOL = 1
N_PERM = 300

# ---- 동료 AE 규격 (출처는 모듈 docstring) ------------------------------------------
AE_HIDDEN = 256
AE_LOSS = "log_likelihood"
AE_LR = 1.0e-3
AE_WEIGHT_DECAY = 1.0e-4
AE_GRAD_CLIP = 5.0
AE_PATIENCE = 60
AE_MIN_EPOCHS = 60
AE_VAL_FRAC = 0.1


# =============================================================================
# 형제 모듈 로드 (grid_phase 는 패키지가 아니다 — paper_supplements.load_intrinsic 패턴)
# =============================================================================

def _load_sibling(fname: str, alias: str):
    f = Path(__file__).resolve().parent / fname
    if not f.is_file():
        raise SystemExit(f"{fname} 없음: 같은 디렉토리에 있어야 한다")
    spec = importlib.util.spec_from_file_location(alias, f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


IP = _load_sibling("intrinsic_phase.py", "_grid_intrinsic_phase")
# load_shard / discover_shards / KMeansRunner / align_metrics / LAYER_IDX ...
PS = _load_sibling("paper_supplements.py", "_grid_paper_supplements")
# residualize_by_scene / contingency_stats / unified_phase / phase_names / tc_root_from_src


# =============================================================================
# AE — 동료 phase/models/autoencoder.py 구조 이식 (CPU torch)
# =============================================================================

class Encoder(nn.Module):
    """동료 `Encoder` 와 동일: MLP 2층(GELU) + 선형 head. 시점별 독립(인과적)."""

    def __init__(self, input_dim: int, d: int, hidden: int = AE_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU())
        self.head = nn.Linear(hidden, d)
        self.d = d

    def forward(self, x):
        return self.head(self.net(x))


class Decoder(nn.Module):
    """동료 `Decoder` 와 동일: MLP 3층 + 차원별 학습 logvar (clamp [-8, 4])."""

    def __init__(self, d: int, out_dim: int, hidden: int = AE_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, out_dim))
        self.logvar = nn.Parameter(torch.zeros(out_dim))

    def forward(self, c):
        return self.net(c), self.logvar.clamp(-8, 4)


class BaseAE(nn.Module):
    """동료 `BaseAE`: 재구성 NLL 하나만이 목적함수. 이산 구조는 학습에 개입하지 않는다."""

    def __init__(self, encoder, decoder, loss: str = AE_LOSS):
        if loss not in ("mse", "log_likelihood"):
            raise ValueError(f"loss 는 mse|log_likelihood: {loss!r}")
        super().__init__()
        self.enc, self.dec, self.loss_name = encoder, decoder, loss
        if loss == "mse":
            nn.init.zeros_(self.dec.logvar)
            self.dec.logvar.requires_grad_(False)

    def loss(self, x):
        c = self.enc(x)
        x_hat, logvar = self.dec(c)
        se = (x - x_hat) ** 2
        nll = 0.5 * (se * torch.exp(-logvar) + logvar + LOG2PI)
        return nll.sum(-1).mean()

    @torch.no_grad()
    def latent(self, x):
        return self.enc(x)


def standardize_fit(X: np.ndarray) -> dict:
    """전역 평균 제거 + **스칼라** std. 축별 whitening 이 아니다 (실험 B 금지 조건).

    scalar std = sqrt(mean over dims of per-dim variance) — 축 간 상대 분산 구조를
    그대로 두고 전체 스케일만 O(1) 로 옮긴다.
    """
    mu = np.asarray(X, dtype=np.float64).mean(0)
    var = ((np.asarray(X, dtype=np.float64) - mu) ** 2).mean()
    sd = float(np.sqrt(max(var, 1e-12)))
    return {"mu": mu.astype(np.float32), "scalar_std": sd}


def standardize_apply(X: np.ndarray, st: dict) -> np.ndarray:
    return np.ascontiguousarray(
        (np.asarray(X, dtype=np.float32) - st["mu"]) / np.float32(st["scalar_std"]))


def train_ae(X: np.ndarray, latent: int, epochs: int, batch_size: int,
             seed: int, patience: int = AE_PATIENCE,
             min_epochs: int = AE_MIN_EPOCHS, log_every: int = 10) -> tuple[BaseAE, dict]:
    """전 shard 를 모아 AE 1개 학습 (CPU). val 손실 최소화 early stopping.

    동료 `phase/train/_loop.py: fit` 과 같은 규칙 — best state 를 보관했다가 종료 시 복원.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n, D = X.shape
    perm = rng.permutation(n)
    n_val = max(int(round(n * AE_VAL_FRAC)), 1)
    va_idx, tr_idx = perm[:n_val], perm[n_val:]

    xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    x_tr, x_va = xt[tr_idx], xt[va_idx]

    model = BaseAE(Encoder(D, latent, AE_HIDDEN), Decoder(latent, D, AE_HIDDEN), AE_LOSS)
    opt = torch.optim.AdamW(model.parameters(), lr=AE_LR, weight_decay=AE_WEIGHT_DECAY)
    n_par = int(sum(p.numel() for p in model.parameters()))
    bs = int(batch_size) if batch_size else len(x_tr)
    print(f"[ae] n_train={len(x_tr)} n_val={len(x_va)} dim={D} latent={latent} "
          f"hidden={AE_HIDDEN} loss={AE_LOSS} params={n_par} batch={bs} "
          f"epochs<={epochs}", flush=True)

    best = {"loss": float("inf"), "epoch": -1, "state": None}
    bad = 0
    hist = []
    for ep in range(int(epochs)):
        model.train(True)
        idx = torch.randperm(len(x_tr))
        tot = 0.0
        for i in range(0, len(x_tr), bs):
            j = idx[i:i + bs]
            loss = model.loss(x_tr[j])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), AE_GRAD_CLIP)
            opt.step()
            tot += float(loss.item()) * len(j)
        tr_loss = tot / max(len(x_tr), 1)

        model.train(False)
        with torch.no_grad():
            va_loss = float(model.loss(x_va).item())
        hist.append({"epoch": ep, "train": tr_loss, "val": va_loss})

        if va_loss < best["loss"]:
            best = {"loss": va_loss, "epoch": ep,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
            bad = 0
        else:
            bad += 1
            if bad >= patience and ep >= min_epochs:
                print(f"[ae] early stop ep{ep} (patience {patience})", flush=True)
                break
        if ep % max(int(log_every), 1) == 0:
            print(f"[ae] ep{ep:4d} train={tr_loss:12.4f} val={va_loss:12.4f}", flush=True)

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.eval()
    print(f"[ae] best ep{best['epoch']} val_loss={best['loss']:.4f}", flush=True)
    summary = {"best_epoch": int(best["epoch"]), "best_val_loss": float(best["loss"]),
               "n_params": n_par, "n_train": int(len(x_tr)), "n_val": int(len(x_va)),
               "epochs_ran": int(len(hist)), "batch_size": int(bs),
               "history_tail": hist[-5:]}
    return model, summary


@torch.no_grad()
def encode(model: BaseAE, X: np.ndarray, chunk: int = 8192) -> np.ndarray:
    """[N, D] → [N, latent] f32. chunk 로 나눠 peak 메모리를 묶는다."""
    out = []
    for s in range(0, len(X), chunk):
        blk = torch.from_numpy(np.ascontiguousarray(X[s:s + chunk], dtype=np.float32))
        out.append(model.latent(blk).numpy().astype(np.float32))
    return np.concatenate(out, 0) if out else np.zeros((0, model.enc.d), np.float32)


# =============================================================================
# 지표 헬퍼 (intrinsic_phase.align_metrics 재사용)
# =============================================================================

def metrics_for(shard, labels, seed):
    return IP.align_metrics(labels, shard["phase_code"], shard["scene"],
                            shard["ep_id"], shard["rec_idx"],
                            BOUNDARY_TOL, N_PERM, seed)


def pooled_metrics(shards, labels_cat, seed):
    """shard 간 ep_id 충돌을 offset 으로 피한 pooled 측정 (intrinsic_phase main 과 동일)."""
    ep_cat, off = [], 0
    for s in shards:
        e = np.asarray(s["ep_id"]).astype(np.int64) + off
        ep_cat.append(e)
        off = int(e.max()) + 1
    phase = (np.concatenate([s["phase_code"] for s in shards])
             if all(s["phase_code"] is not None for s in shards) else None)
    scene = (np.concatenate([s["scene"] for s in shards])
             if all(s["scene"] is not None for s in shards) else None)
    return IP.align_metrics(labels_cat, phase, scene, np.concatenate(ep_cat),
                            np.concatenate([s["rec_idx"] for s in shards]),
                            BOUNDARY_TOL, N_PERM, seed)


def fmt_line(tag, name, m):
    def g(k):
        return m.get(k, float("nan"))
    return (f"[{tag}] {name:<26} MI={g('mi_phase_bits'):.3f} "
            f"margin={g('margin_bits'):+.3f} purity={g('purity_phase'):.3f} "
            f"mi_scene={g('mi_scene_bits'):.3f} F1={g('boundary_f1'):.3f} "
            f"z={g('boundary_z'):+.2f}")


def split_by_shard(shards, arr):
    out, off = {}, 0
    for s in shards:
        m = len(s["feat"])
        out[s["name"]] = arr[off:off + m]
        off += m
    return out


# =============================================================================
# 출력
# =============================================================================

def build_meta(args, runner, shards, ae_summary, scaler, extra=None):
    """docs/04 규약: 산출물 안에 **절대경로 기록 금지** (kmeans source 는 basename 만)."""
    src = runner.source
    meta = {
        "script": SCRIPT_REL,
        "reuses": ["scripts/analysis/grid_phase/intrinsic_phase.py",
                   "scripts/analysis/grid_phase/paper_supplements.py"],
        "pipeline": ["raw-1536 (PCA 없음, whitening 없음)",
                     "global mean-center + scalar std",
                     f"AE(latent {args.latent}, hidden {AE_HIDDEN}, {AE_LOSS})",
                     f"KMeans(per-task k={args.k}, global k={args.global_k})"],
        "feature_spec": {"layer": 12, "denoise": 3, "segment": "all(49-token mean)",
                         "dim": 1536,
                         "layer_axis_index": IP.LAYER_IDX,
                         "denoise_axis_index": IP.DENOISE_IDX,
                         "segment_axis_index": IP.SEGMENT_IDX},
        "ae": {
            "source_spec": "task_classification/phase/models/autoencoder.py "
                           "+ conf/model/ae.yaml + conf/train/default.yaml",
            "encoder": f"Linear({1536}->{AE_HIDDEN}) GELU Linear({AE_HIDDEN}->{AE_HIDDEN}) "
                       f"GELU Linear({AE_HIDDEN}->{args.latent})",
            "decoder": f"Linear({args.latent}->{AE_HIDDEN}) GELU "
                       f"Linear({AE_HIDDEN}->{AE_HIDDEN}) GELU Linear({AE_HIDDEN}->1536) "
                       f"+ per-dim learned logvar (clamp [-8,4])",
            "loss": AE_LOSS,
            "optimizer": {"name": "adamw", "lr": AE_LR, "weight_decay": AE_WEIGHT_DECAY},
            "grad_clip": AE_GRAD_CLIP,
            "patience": args.patience, "min_epochs": AE_MIN_EPOCHS,
            "val_frac": AE_VAL_FRAC,
            "scope": "global (전 shard 합쳐 AE 1개)",
            "summary": ae_summary,
        },
        "scaler": {"kind": "global mean-center + scalar std (축별 whitening 아님)",
                   "scalar_std": float(scaler["scalar_std"])},
        "params": {"k": args.k, "global_k": args.global_k, "latent": args.latent,
                   "epochs": args.epochs, "batch_size": args.batch_size,
                   "seed": args.seed, "n_init": N_INIT, "max_iter": MAX_ITER,
                   "boundary_tol": BOUNDARY_TOL, "n_perm": N_PERM,
                   "pca_dim": None, "whiten": False, "fit_scenes": None},
        "deviations": [
            "입력이 PCA-64 whitened 가 아니라 raw-1536 — 동료 실험 B 종합 판정 "
            "(docs/steering/40 §4·§5 부록): raw + AE, PCA 불필요, PCA 미적용 시 whitening 금지.",
            "동료는 GPU full-batch 800 epoch; 여기는 CPU 전용이라 미니배치 + "
            "--epochs 로 상한을 준다. optimizer/lr/wd/grad_clip/early-stopping 규칙은 동일.",
            "AE 는 전 shard 를 모아 1개만 학습하고 encoder 를 instruction 별로 공유한다.",
        ],
        "kmeans": {"impl": runner.impl,
                   "source": Path(src).name if src and src.endswith(".py") else src,
                   "load_error": runner.load_error},
        "shards": [s["name"] for s in shards],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_device": "cpu",
    }
    if extra:
        meta.update(extra)
    return meta


def _git_info() -> dict:
    """repo commit/branch (없으면 unknown). 산출물 재현 메타용 — 경로는 넣지 않는다."""
    import subprocess
    out = {}
    for key, cmd in (("commit", ["git", "rev-parse", "HEAD"]),
                     ("branch", ["git", "rev-parse", "--abbrev-ref", "HEAD"])):
        try:
            out[key] = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True,
                                      text=True, timeout=10).stdout.strip() or "unknown"
        except Exception:
            out[key] = "unknown"
    return out


def _check_centers_match(Z: np.ndarray, labels: np.ndarray, centers: np.ndarray,
                         name: str, tol: float = 1e-4) -> None:
    """번들에 담는 centers 가 실제 라벨을 재현하는지 fail-loud 확인.

    KMeans predict = 최근접 center 이므로 (라벨, centers) 가 같은 fit 결과라면
    argmin 이 라벨과 일치해야 한다. 동률 근처의 부동소수 잡음만 tol 로 봐준다.
    """
    Zf = np.asarray(Z, dtype=np.float32)
    C = np.asarray(centers, dtype=np.float32)
    cn = (C ** 2).sum(1)[None, :]
    d2 = np.empty((len(Zf), len(C)), np.float32)
    for s in range(0, len(Zf), 8192):     # peak 억제 (chunk)
        blk = Zf[s:s + 8192]
        d2[s:s + 8192] = (blk ** 2).sum(1)[:, None] - 2.0 * (blk @ C.T) + cn
    arg = d2.argmin(1)
    lab = np.asarray(labels).astype(np.int64)
    if lab.min() < 0 or lab.max() >= len(C):
        raise SystemExit(f"[export-bundle] {name}: 라벨 범위 {lab.min()}~{lab.max()} "
                         f"가 centers {len(C)} 개와 맞지 않는다")
    bad = arg != lab
    if bad.any():
        gap = d2[np.arange(len(lab)), lab] - d2[np.arange(len(lab)), arg]
        hard = bad & (gap > tol)
        if hard.any():
            raise SystemExit(
                f"[export-bundle] {name}: centers 가 라벨을 재현하지 못한다 "
                f"(불일치 {int(hard.sum())}/{len(lab)}, max gap {float(gap.max()):.6g}) "
                "— 라벨과 번들이 다른 KMeans 결과일 수 있다")
        print(f"[export-bundle] {name}: 동률 부근 불일치 {int(bad.sum())} "
              f"(tol {tol} 이내, 허용)", flush=True)


def export_bundle(path: Path, model: BaseAE, scaler: dict, raw: dict, args,
                  shards, ae_summary: dict) -> None:
    """온라인 판정기 번들 = 표준화 + encoder + instruction 별 KMeans centers 1개 NPZ.

    이 번들 하나면 새 rollout 의 raw-1536 feature 를 학습 때와 **같은 좌표계**로
    cluster 에 배정할 수 있다 (standardize → encoder → 최근접 center).
    docs/04 규약: 절대경로 기록 금지 → shard 디렉토리는 basename 만 남긴다.
    """
    payload: dict = {}
    payload["mu"] = np.asarray(scaler["mu"], dtype=np.float32)
    payload["scalar_std"] = np.asarray(float(scaler["scalar_std"]), dtype=np.float32)

    enc_state = model.enc.state_dict()
    for k, v in enc_state.items():
        payload[f"enc.{k}"] = v.detach().cpu().numpy().astype(np.float32)

    slugs = [s["name"] for s in shards]
    for name in slugs:
        cent = raw[name]["centers"]
        if cent is None:
            raise SystemExit(f"[export-bundle] {name}: centers 가 없다 "
                             "(KMeans 구현이 centroid 를 돌려주지 않음)")
        cent = np.asarray(cent, dtype=np.float32)
        if cent.shape != (args.k, args.latent):
            raise SystemExit(f"[export-bundle] {name}: centers shape {cent.shape} != "
                             f"({args.k}, {args.latent})")
        _check_centers_match(raw[name]["latent"], raw[name]["labels"], cent, name)
        payload[f"centers.{name}"] = cent

    payload["k"] = np.asarray(int(args.k), dtype=np.int32)
    payload["latent"] = np.asarray(int(args.latent), dtype=np.int32)
    payload["slugs"] = np.asarray(slugs, dtype=np.str_)
    payload["arch"] = np.asarray(json.dumps({
        "encoder": [
            {"op": "Linear", "in": int(payload["mu"].shape[0]), "out": AE_HIDDEN,
             "state_key": "enc.net.0"},
            {"op": "GELU"},
            {"op": "Linear", "in": AE_HIDDEN, "out": AE_HIDDEN, "state_key": "enc.net.2"},
            {"op": "GELU"},
            {"op": "Linear", "in": AE_HIDDEN, "out": int(args.latent),
             "state_key": "enc.head"},
        ],
        "gelu": "exact (erf), torch nn.GELU default",
        "preprocess": "x_std = (x - mu) / scalar_std  (축별 whitening 아님)",
        "assign": "argmin_c ||enc(x_std) - centers[c]||^2",
        "hidden": AE_HIDDEN, "latent": int(args.latent),
        "input_dim": int(payload["mu"].shape[0]),
        "state_keys": sorted(f"enc.{k}" for k in enc_state),
    }, ensure_ascii=False))

    git = _git_info()
    payload["provenance"] = np.asarray(json.dumps({
        "script": SCRIPT_REL,
        "seed": int(args.seed),
        "shard_dir_basename": Path(str(args.shard_dir)).name,
        "shards": slugs,
        "k": int(args.k), "latent": int(args.latent),
        "epochs": int(args.epochs), "batch_size": int(args.batch_size),
        "patience": int(args.patience),
        "kmeans_n_init": N_INIT, "kmeans_max_iter": MAX_ITER,
        "ae_summary": ae_summary,
        "feature_spec": {"layer": 12, "denoise_index": IP.DENOISE_IDX,
                         "segment": "all(49-token mean)",
                         "layer_axis_index": IP.LAYER_IDX, "dim": 1536},
        "git_commit": git["commit"], "git_branch": git["branch"],
        "numpy": np.__version__, "torch": torch.__version__,
    }, ensure_ascii=False))

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    print(f"[export-bundle] {path.name}: enc {len(enc_state)} tensors, "
          f"centers x{len(slugs)} (k={args.k}, latent={args.latent})", flush=True)


def write_json(path: Path, payload: dict, meta: dict):
    payload = dict(payload)
    payload["meta"] = meta
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[write] {path.name}", flush=True)


# =============================================================================
# 모드
# =============================================================================

def mode_resid(shards, raw, model, scaler, runner, k, seed, out_dir):
    """scene 잔차화: raw 단계에서 scene 별 평균을 뺀 뒤 같은 AE encoder 로 인코딩.

    encoder 가 고정이라 위치(전역 평균)를 잃으면 입력이 학습 분포 밖으로 나간다 →
    잔차에 **task 전역 평균을 되더해** scene 편차만 제거하고 위치는 보존한다.
    """
    per = {}
    tsv = ["\t".join(["task", "raw_mi", "resid_mi", "raw_margin", "resid_margin",
                      "raw_mi_scene", "resid_mi_scene", "raw_purity", "resid_purity"])]
    for s in shards:
        feat = s["feat"]
        res = PS.residualize_by_scene(feat, s["scene"])
        res += feat.mean(0, dtype=np.float64).astype(np.float32)   # 위치 복원
        Z = encode(model, standardize_apply(res, scaler))
        del res
        lab, _ = runner.fit_predict(Z, Z, k)
        m = metrics_for(s, np.asarray(lab), seed)
        r = raw[s["name"]]["metrics"]
        per[s["name"]] = {
            "raw": r, "resid": m,
            "delta": {key: (float(m[key] - r[key]) if key in m and key in r else None)
                      for key in ("mi_phase_bits", "margin_bits", "purity_phase",
                                  "mi_scene_bits", "nmi_arith", "boundary_f1")}}
        tsv.append("\t".join([s["name"]] + [f"{v:.6f}" for v in (
            r.get("mi_phase_bits", float("nan")), m.get("mi_phase_bits", float("nan")),
            r.get("margin_bits", float("nan")), m.get("margin_bits", float("nan")),
            r.get("mi_scene_bits", float("nan")), m.get("mi_scene_bits", float("nan")),
            r.get("purity_phase", float("nan")), m.get("purity_phase", float("nan")))]))
        print(fmt_line("resid", s["name"], m), flush=True)
    (out_dir / "resid_compare_ae.tsv").write_text("\n".join(tsv) + "\n", encoding="utf-8")
    print("[write] resid_compare_ae.tsv", flush=True)
    return {"k": k,
            "residualization": "per-task scene demean (all records) + task mean 복원, "
                               "AE encoder 는 raw 학습본 그대로",
            "per_task": per}


def mode_contingency_pertask(shards, raw, k):
    per = {}
    for s in shards:
        st = PS.contingency_stats(raw[s["name"]]["labels"], s["phase_code"],
                                  PS.phase_names(s))
        per[s["name"]] = st
        print(f"[cont] {s['name']:<26} purity={st.get('purity_phase', float('nan')):.3f} "
              f"off_med={st.get('off_phase_rate', {}).get('median', float('nan')):.3f} "
              f"k/phase={st.get('clusters_per_phase_mean', float('nan')):.2f}", flush=True)
    return {"scope": "per-task", "k": k, "per_task": per}


def mode_contingency_global(shards, labels_global, k):
    """global 분할표. phase 코드는 shard 간 이름 기준으로 통일한다."""
    uni_list, uni_names = PS.unified_phase(shards)
    uni = np.concatenate(uni_list)
    st = PS.contingency_stats(labels_global, uni, uni_names)

    tasks = [s["name"] for s in shards]
    tidx = np.concatenate([np.full(len(s["feat"]), i, np.int32)
                           for i, s in enumerate(shards)])
    clusters = np.unique(labels_global)
    ci = {int(c): i for i, c in enumerate(clusters)}
    cbt = np.zeros((len(clusters), len(tasks)), np.int64)
    np.add.at(cbt, (np.array([ci[int(c)] for c in labels_global]), tidx), 1)
    st["cluster_by_task"] = {"tasks": tasks,
                             "cluster_ids": [int(c) for c in clusters],
                             "counts": cbt.tolist()}
    print(f"[cont] {'__global__':<26} purity={st.get('purity_phase', float('nan')):.3f} "
          f"off_med={st.get('off_phase_rate', {}).get('median', float('nan')):.3f}",
          flush=True)
    return {"scope": "global", "k": k, "global": st}


# =============================================================================
# main
# =============================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shard-dir", "--shards-dir", dest="shard_dir", required=True,
                    type=Path, help="shard NPZ 디렉토리 (intrinsic_phase.py 와 동일 규약)")
    ap.add_argument("--out-dir", type=Path,
                    default=REPO_ROOT / "outputs/analysis/grid_phase/ae_raw")
    ap.add_argument("--mode", choices=("main", "resid", "contingency", "all"),
                    default="all")
    ap.add_argument("--k", type=int, default=8, help="instruction 별 클러스터 수")
    ap.add_argument("--global-k", type=int, default=24, help="global 클러스터 수")
    ap.add_argument("--latent", type=int, default=16, help="AE 병목 차원 (동료 규격 16)")
    ap.add_argument("--epochs", type=int, default=200, help="AE 학습 epoch 상한")
    ap.add_argument("--batch-size", type=int, default=4096,
                    help="0 이면 full-batch (동료 규격). CPU 라 기본은 미니배치")
    ap.add_argument("--patience", type=int, default=AE_PATIENCE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shards", default=None, help="쉼표구분 shard stem 부분집합")
    ap.add_argument("--kmeans-src", default=None,
                    help="task_classification 의 phase/clustering/gpu.py 경로 (또는 루트)")
    ap.add_argument("--kmeans-impl", choices=("auto", "task_classification", "numpy"),
                    default="auto")
    ap.add_argument("--dump-labels", action="store_true",
                    help="instruction 별 per-record 라벨 NPZ 덤프 (영상 렌더용)")
    ap.add_argument("--export-bundle", type=Path, default=None,
                    help="온라인 판정기 번들 NPZ 경로 (mu·scalar_std·encoder state·"
                         "instruction 별 KMeans centers). 미지정 시 저장하지 않는다")
    ap.add_argument("--no-global", action="store_true",
                    help="global k24 (main·contingency 양쪽) 를 건너뛴다")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "16")))

    runner = IP.KMeansRunner(args.kmeans_impl, PS.tc_root_from_src(args.kmeans_src),
                             N_INIT, MAX_ITER, args.seed)
    print(f"[kmeans] impl={runner.impl} source={runner.source}", flush=True)
    if runner.load_error:
        print(f"[kmeans] task_classification 로드 실패 → fallback: {runner.load_error}",
              flush=True)

    paths = IP.discover_shards(args.shard_dir.expanduser(),
                               args.shards.split(",") if args.shards else None)
    shards = [IP.load_shard(p) for p in paths]
    for s in shards:
        print(f"[load] {s['name']}: n_rec={len(s['feat'])} "
              f"ep={len(np.unique(s['ep_id']))}", flush=True)

    # ---- AE: 전 shard 를 모아 1개 학습 (raw-1536, 전역 표준화만) ----
    X = np.concatenate([s["feat"] for s in shards], 0)
    scaler = standardize_fit(X)
    Xs = standardize_apply(X, scaler)
    del X
    model, ae_summary = train_ae(Xs, args.latent, args.epochs, args.batch_size, args.seed,
                                 patience=args.patience)
    Z_all = encode(model, Xs)
    del Xs
    Z_by_shard = split_by_shard(shards, Z_all)

    meta = build_meta(args, runner, shards, ae_summary, scaler)

    # ---- instruction 별 KMeans (k) — 모든 모드가 이 라벨을 쓴다 ----
    raw: dict[str, dict] = {}
    for s in shards:
        Z = Z_by_shard[s["name"]]
        lab, cent = runner.fit_predict(Z, Z, args.k)
        lab = np.asarray(lab)
        raw[s["name"]] = {"labels": lab, "metrics": metrics_for(s, lab, args.seed),
                          "latent": np.asarray(Z, dtype=np.float32),
                          "centers": (np.asarray(cent, dtype=np.float32)
                                      if cent is not None else None)}
        print(fmt_line("ae", s["name"], raw[s["name"]]["metrics"]), flush=True)

    # ---- 온라인 판정기 번들 (--dump-labels 라벨과 **같은 raw[] 객체**에서 저장) ----
    if args.export_bundle is not None:
        export_bundle(args.export_bundle, model, scaler, raw, args, shards, ae_summary)

    # ---- global KMeans (global_k) ----
    labels_global = None
    if not args.no_global:
        lab_g, _cent = runner.fit_predict(Z_all, Z_all, args.global_k)
        labels_global = np.asarray(lab_g)

    if args.dump_labels:
        # 영상용: instruction 별 per-record 라벨을 소용량 NPZ 로 덤프
        # (ep_id/rec_idx/phase_code/scene/succ + AE cluster). 절대경로 기록 없음.
        for s in shards:
            np.savez_compressed(
                args.out_dir / f"labels_{s['name']}_k{args.k}.npz",
                ep_id=s["ep_id"], rec_idx=s["rec_idx"], scene=s["scene"],
                noise=s["noise"], ep_len=s["ep_len"],
                succ=s["succ"], phase_code=s["phase_code"],
                cluster=raw[s["name"]]["labels"],
                latent=raw[s["name"]]["latent"],
                centers=(raw[s["name"]]["centers"]
                         if raw[s["name"]]["centers"] is not None
                         else np.zeros((0, 0), np.float32)),
                phase_codebook=json.dumps(
                    (s.get("meta") or {}).get("phase_codebook", {}), ensure_ascii=False),
            )
        print(f"[dump] labels_*_k{args.k}.npz x{len(shards)}", flush=True)

    if args.mode in ("main", "all"):
        payload = {"scope": "per-task", "k": args.k,
                   "per_task": {n: v["metrics"] for n, v in raw.items()},
                   "ae_summary": ae_summary}
        write_json(args.out_dir / f"ae_pertask_k{args.k}.json", payload, meta)

        if labels_global is not None:
            g_by_shard = split_by_shard(shards, labels_global)
            per = {s["name"]: metrics_for(s, g_by_shard[s["name"]], args.seed)
                   for s in shards}
            pooled = pooled_metrics(shards, labels_global, args.seed)
            print(fmt_line("ae-global", "__pooled__", pooled), flush=True)
            write_json(args.out_dir / f"ae_global_k{args.global_k}.json",
                       {"scope": "global", "k": args.global_k,
                        "per_shard": per, "pooled": pooled,
                        "ae_summary": ae_summary}, meta)

    if args.mode in ("resid", "all"):
        payload = mode_resid(shards, raw, model, scaler, runner, args.k,
                             args.seed, args.out_dir)
        write_json(args.out_dir / "resid_compare_ae.json", payload, meta)

    if args.mode in ("contingency", "all"):
        payload = mode_contingency_pertask(shards, raw, args.k)
        write_json(args.out_dir / f"contingency_pertask_k{args.k}_ae.json", payload, meta)
        if labels_global is not None:
            payload = mode_contingency_global(shards, labels_global, args.global_k)
            write_json(args.out_dir / f"contingency_global_k{args.global_k}_ae.json",
                       payload, meta)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

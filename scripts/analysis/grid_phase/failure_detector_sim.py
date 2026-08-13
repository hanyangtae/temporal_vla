#!/usr/bin/env python
"""grid shard 위 SAFE식 failure detector + 온라인 발화 시뮬레이션 (rung0 관문).

질문 하나: **"실패를 온라인으로, 개입할 시간이 남은 시점에 읽을 수 있나"**.
분리도 지도(`phase_sep_matrix.py`)는 사후 판독이고, 여기서는 causal detector
(step t 까지만 보고 점수)를 학습해 test 판에서 **첫 발화 시점**을 잰다. 발화가
전부 종반에 몰리면(relpos≈1, 남은 step≈0) steering 은 걸 곳이 없다 — 그 판정이 목적.

## 파이프라인

1. shard NPZ(`extract_grid_matrix.py` Tier A) → record 를 `ep_id` 로 묶고 `rec_idx`
   순 정렬 → episode 당 [T, D] 시퀀스. 라벨 y=1 은 **failure**(SAFE 규약).
2. scene 단위 결정적 분할 (train 6 / calib 2 / test 2). scene 을 섞지 않는다 —
   같은 scene 의 판이 train/test 에 갈리면 scene 암기가 검출로 보인다
   ([[scene-matched-succfail-verdict]]).
3. detector 학습 (SAFE-LSTM / SAFE-MLP). 구현 출처 =
   `scripts/safe/groot_n16/robocasa/analyze/pathway_lstm_detector.py` +
   `scripts/safe/groot_n16/robocasa/vis/core/lstm.py` (아키텍처·손실·정규화 동일).
   원격 노드에 그 파일들이 없을 수 있어 **import 하지 않고 이식**했다.
4. functional CP 밴드 δ_t = μ_t + bw·σ_t (성공 궤적의 시변 밴드).
   μ_t/σ_t = train 성공 판, bw = **calib 성공 판**의 max-normalized 초과량의
   (1−α) 분위수. t 가 밴드 길이를 넘으면 마지막 밴드 유지(plateau).
   `--truncate-train` 을 켜면 밴드 출처(train/calib)가 절제되어 밴드 길이가 W 로
   줄지만, **plateau 규약이 그대로 적용**되어 full test 시퀀스의 t>W 구간은 마지막
   밴드값으로 판정된다 (별도 처리 불필요).
5. 발화 시뮬: test 판마다 `score_t > δ_t` 인 **첫 t**. 기록 = fired / t_fire /
   relpos = t_fire/(T−1) / 그 순간의 GT phase 이름 / 종결까지 남은 step.
6. 길이 confound 통제 (`--truncate-train {none,rollout,phase-gt}`) + 그 판정을 위한
   무feature **timer 기준선**(t ≥ W 발화) · `tpr_before_W` · `lead_vs_W` ·
   고정 판정시각 AUROC(`auroc_td*`). 절제는 학습·보정에만 걸고 test 는 항상 full.

## arm

- `pertask` — task 별 detector 9개 (task 내부 일반화).
- `mixed`   — 전 task 풀링 detector 1개. 표준화·가중치는 공유하되 **CP 밴드는
  per-task calib 로 보정**(RL²-VLA 방식) — task 별 점수 스케일 차이를 밴드가 흡수.

## 입력 계약 (Tier A shard)
`<shard-dir>/<instr_slug>.npz`
  X fp16 [n_rec, n_layer, K, S, D] / ep_id i32 / scene i16 / noise i16 / rec_idx i16
  succ i8 / phase_code i16 / ep_len i16 / meta_json{capture_layers|layers,
  segment_names, phase_codebook}
feature 좌표 기본값 = layer 12 × 마지막 denoise(k=3) × segment "all"(49토큰 평균).
layer 는 meta 의 layer 리스트에서 **인덱스 역산**한다 (하드코딩 금지).

## 출력 (`--out`)
  sim_summary.tsv                     task × arm × model × α 행 (TPR/FPR/발화시점/phase 분포)
  sim_detail.json                     episode 별 발화 기록
  detector_<arm>_<model>_<task>.pt    state_dict + 표준화 + feature 좌표 + CP 밴드
산출물에 절대경로를 쓰지 않는다 (docs/04 §8) — 입력은 shard 파일명(basename)만 기록.

## 사용
    # 로컬 게이트 (합성 데이터로 파이프라인 검증)
    python scripts/analysis/grid_phase/failure_detector_sim.py --self-test

    # 원격 CPU 노드 (torch CPU + numpy 만)
    ~/anaconda3/bin/python scripts/analysis/grid_phase/failure_detector_sim.py \
        --shard-dir ~/workspace/.../analysis/grid_phase/segA \
        --out ~/workspace/.../analysis/grid_phase/detector_sim \
        --arm both --models lstm,mlp --threads 8

    # 길이 절제 ablation (학습·보정만 절제, test 는 full)
    ... failure_detector_sim.py --shard-dir <segA> --out <out>/trunc_rollout \
        --truncate-train rollout --arm both --models lstm,mlp --threads 8
    ... failure_detector_sim.py --shard-dir <segA> --out <out>/trunc_phasegt \
        --truncate-train phase-gt --arm both --models lstm,mlp --threads 8
"""
from __future__ import annotations

import os

# BLAS/torch 스레드 cap — 공유 노드. numpy/torch import 전에 설정해야 효력이 있다.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import argparse
import json
import tempfile
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

DEFAULT_ALPHAS = (0.05, 0.1, 0.2, 0.3)
MIN_BAND_EPS = 3          # μ/σ, bw 추정에 필요한 최소 성공 판 수 (참조 구현과 동일)
EPS = 1e-8
TD_GRID = (5, 10, 15, 20, 30)   # 고정 판정시각 AUROC 격자 (causal, full test 시퀀스 위)
MIN_TRUNC_LEN = 2         # 절제 후 이보다 짧은 시퀀스는 버린다


# =============================================================================
# detector (SAFE-LSTM / SAFE-MLP) — 참조 구현 이식 (import 의존 없이 standalone)
# =============================================================================

class LSTMDetector(nn.Module):
    """단층 LSTM + linear + sigmoid. causal: out[t] 는 x[:t+1] 에만 의존."""
    cumulative = False

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):                       # [B,T,D] → [B,T]
        out, _ = self.lstm(x)
        return torch.sigmoid(self.fc(out)).squeeze(-1)


class MLPDetector(nn.Module):
    """SAFE-MLP: per-step MLP→scalar→sigmoid (step 독립). 검출 score 는 출력 누적평균."""
    cumulative = True

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):                       # [B,T,D] → [B,T]
        return torch.sigmoid(self.net(x)).squeeze(-1)


def build_detector(kind: str, input_dim: int, hidden: int) -> nn.Module:
    if kind == "lstm":
        return LSTMDetector(input_dim, hidden)
    if kind == "mlp":
        return MLPDetector(input_dim, hidden)
    raise ValueError(f"알 수 없는 detector: {kind}")


# =============================================================================
# shard I/O
# =============================================================================

class Episode:
    """판 하나. X 는 표준화 **전** 원본 [T,D] (표준화는 arm 별로 다시 건다)."""

    __slots__ = ("task", "ep_id", "scene", "noise", "succ", "X", "phase", "T")

    def __init__(self, task, ep_id, scene, noise, succ, X, phase):
        self.task, self.ep_id, self.scene, self.noise = task, ep_id, scene, noise
        self.succ = int(succ)
        self.X = X
        self.phase = phase
        self.T = int(X.shape[0])

    @property
    def y(self) -> int:
        """SAFE 규약: failure = positive."""
        return 1 - self.succ


class ShardSpec:
    """feature 좌표(layer/denoise/seg)를 meta 에서 역산한 결과 — 계약 위반은 fail-loud."""

    def __init__(self, path: Path, layer: int, denoise: int, seg: str):
        with np.load(path, allow_pickle=False) as f:
            meta_raw = f["meta_json"] if "meta_json" in f.files else None
            shape = f["X"].shape
        if meta_raw is None:
            raise ValueError(f"{path.name}: meta_json 없음 — Tier A shard 가 맞나")
        self.meta = json.loads(str(meta_raw))
        if len(shape) != 5:
            raise ValueError(
                f"{path.name}: X.ndim={len(shape)} — Tier A [n_rec,L,K,S,D] 만 지원 "
                "(tokB shard 는 이 도구의 입력이 아니다)")
        self.n_rec, self.n_layer, self.n_denoise, self.n_seg, self.dim = shape

        layers = self.meta.get("capture_layers") or self.meta.get("layers")
        if layers is None:
            raise ValueError(f"{path.name}: meta 에 capture_layers/layers 가 없다")
        self.layers = [int(v) for v in layers]
        if len(self.layers) != self.n_layer:
            raise ValueError(
                f"{path.name}: capture_layers {len(self.layers)} != X layer축 {self.n_layer}")
        if layer not in self.layers:
            raise ValueError(f"{path.name}: layer {layer} 없음 (있는 것: {self.layers})")
        self.layer_idx = self.layers.index(layer)

        seg_names = [str(s) for s in (self.meta.get("segment_names") or [])]
        if len(seg_names) != self.n_seg:
            seg_names = ["state", "future", "action", "all"][: self.n_seg]
        self.segment_names = seg_names
        if seg not in seg_names:
            raise ValueError(f"{path.name}: segment '{seg}' 없음 (있는 것: {seg_names})")
        self.seg_idx = seg_names.index(seg)

        d = self.n_denoise - 1 if denoise < 0 else denoise
        if not (0 <= d < self.n_denoise):
            raise ValueError(f"{path.name}: denoise {denoise} 범위 밖 (K={self.n_denoise})")
        self.denoise_idx = d

        self.phase_names = {int(v): str(k) for k, v in
                            (self.meta.get("phase_codebook") or {}).items()}
        self.instruction = str(self.meta.get("instruction", path.stem))


def load_shard_episodes(path: Path, layer: int, denoise: int, seg: str
                        ) -> tuple[list[Episode], ShardSpec]:
    """shard NPZ → episode 목록. rollout pooling 없음 — per-record 시퀀스를 유지한다."""
    spec = ShardSpec(path, layer, denoise, seg)
    task = path.stem
    with np.load(path, allow_pickle=False) as f:
        X = f["X"]
        feat = np.ascontiguousarray(
            X[:, spec.layer_idx, spec.denoise_idx, spec.seg_idx, :]).astype(np.float32)
        del X
        cols = {}
        for k in ("ep_id", "scene", "noise", "rec_idx", "succ", "phase_code", "ep_len"):
            if k not in f.files:
                raise ValueError(f"{path.name}: 필수 열 '{k}' 없음")
            cols[k] = np.asarray(f[k]).ravel().astype(np.int64)
    n = len(feat)
    for k, v in cols.items():
        if len(v) != n:
            raise ValueError(f"{path.name}: {k} 길이 {len(v)} != n_rec {n}")

    eps: list[Episode] = []
    for ep in np.unique(cols["ep_id"]):
        m = np.where(cols["ep_id"] == ep)[0]
        m = m[np.argsort(cols["rec_idx"][m], kind="mergesort")]
        su = np.unique(cols["succ"][m])
        if len(su) != 1:
            raise ValueError(f"{path.name}: ep{ep} 의 succ 이 record 마다 다르다 {su}")
        sc = np.unique(cols["scene"][m])
        if len(sc) != 1:
            raise ValueError(f"{path.name}: ep{ep} 의 scene 이 record 마다 다르다 {sc}")
        if "ep_len" in cols:
            el = int(np.unique(cols["ep_len"][m])[0])
            if el != len(m):
                raise ValueError(
                    f"{path.name}: ep{ep} ep_len={el} != record 수 {len(m)}")
        eps.append(Episode(task, int(ep), int(sc[0]), int(cols["noise"][m[0]]),
                           int(su[0]), feat[m], cols["phase_code"][m]))
    if not eps:
        raise ValueError(f"{path.name}: episode 0")
    return eps, spec


def discover_shards(shard_dir: Path, only: list[str] | None) -> list[Path]:
    paths = sorted(p for p in shard_dir.glob("*.npz") if not p.name.endswith("_fit.npz"))
    if only:
        keep = set(only)
        paths = [p for p in paths if p.stem in keep]
        missing = keep - {p.stem for p in paths}
        if missing:
            raise SystemExit(f"shard 없음: {sorted(missing)}")
    if not paths:
        raise SystemExit(f"shard NPZ 없음: {shard_dir}")
    return paths


# =============================================================================
# split (scene 단위 결정적 분할)
# =============================================================================

def split_scenes(task: str, scenes: list[int], n_train: int, n_calib: int,
                 n_test: int, seed: int) -> dict[str, list[int]]:
    """정렬 후 (seed, task) 로 결정적 셔플 → train/calib/test scene.

    scene 이 모자라면 test/calib 1개씩부터 채우고 나머지를 train 에 준다 (fail-loud
    최소값 = 3 scene). 남는 scene 은 train 으로 — 학습 표본을 우선한다.
    """
    uniq = sorted(set(int(s) for s in scenes))
    if len(uniq) < 3:
        raise ValueError(f"{task}: scene {len(uniq)}개 — train/calib/test 분할 불가(최소 3)")
    rng = np.random.default_rng([seed, zlib.crc32(task.encode("utf-8"))])
    order = [uniq[i] for i in rng.permutation(len(uniq))]
    n = len(order)
    te = min(n_test, max(1, n - 2))
    ca = min(n_calib, max(1, n - te - 1))
    tr = n - te - ca
    if tr < 1:
        raise ValueError(f"{task}: scene {n}개로 train scene 을 못 만든다")
    return {"test": sorted(order[:te]), "calib": sorted(order[te:te + ca]),
            "train": sorted(order[te + ca:])}


def standardizer(eps: list[Episode]) -> tuple[np.ndarray, np.ndarray]:
    allf = np.concatenate([e.X for e in eps], axis=0)
    mu = allf.mean(axis=0)
    sd = allf.std(axis=0)
    sd[sd < 1e-6] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def apply_std(e: Episode, mu, sd) -> np.ndarray:
    return ((e.X - mu) / sd).astype(np.float32)


# =============================================================================
# 학습 데이터 길이 절제 (length confound ablation)
# =============================================================================
# 실패는 항상 timeout(길다) / 성공은 조기종료(짧다) → full 시퀀스 학습은 "길면 실패"를
# 학습할 수 있다([[seen18-rollout-length-confound]]). cap 정의는 레포 규약과 동일:
#   rollout  W    = TRAIN 성공 episode record 수의 ceil(μ+1σ)
#   phase-gt cap  = TRAIN 성공 episode 의 그 phase dwell(>0) 의 ceil(μ+1σ)
# (`scripts/fit/fit_setm.py::phase_dwell_caps`,
#  `scripts/fit/fit_phase_conceptor.py::compute_length_caps` 와 같은 식. np.std 는
#  모집단 표준편차(ddof=0) — 그쪽 구현과 동일하게 맞춘다.)
# 성공 dwell 이 없는 phase 는 fit 쪽 규약대로 **대조 불가 → skip**: 해당 record 를
# 버린다. phase_code<0 (unknown/terminal) 도 같은 규칙이 적용된다 — 성공 판에 그
# code 가 있으면 cap 이 생기고, 없으면 drop.
# 절제는 **TRAIN·CALIB 에만** 건다. TEST 는 항상 full (온라인 현실성).


def _ceil_mu_sigma(vals: list[int]) -> int:
    v = np.asarray([x for x in vals if x > 0], dtype=np.float64)
    return int(np.ceil(v.mean() + v.std()))


def rollout_cap(train_eps: list[Episode]) -> int | None:
    """W = TRAIN 성공 판 record 수의 ceil(μ+1σ). 성공 판이 없으면 None."""
    lens = [e.T for e in train_eps if e.succ == 1]
    return _ceil_mu_sigma(lens) if any(x > 0 for x in lens) else None


def phase_dwell_caps(train_eps: list[Episode]) -> dict[int, int]:
    """phase code → dwell cap. TRAIN 성공 판의 dwell(>0) 만 사용."""
    codes: set[int] = set()
    for e in train_eps:
        codes.update(int(c) for c in np.unique(e.phase))
    caps: dict[int, int] = {}
    for c in sorted(codes):
        dw = [int((e.phase == c).sum()) for e in train_eps if e.succ == 1]
        dw = [d for d in dw if d > 0]
        if dw:
            caps[c] = _ceil_mu_sigma(dw)
    return caps


def truncate_episode(e: Episode, mode: str, W: int | None,
                     caps: dict[int, int] | None) -> Episode | None:
    """절제된 사본. 절제 불가/너무 짧으면 None (호출부에서 skip 카운트)."""
    if mode == "none":
        return e
    if mode == "rollout":
        if W is None:
            return e
        idx = np.arange(min(e.T, int(W)))
    elif mode == "phase-gt":
        if not caps:
            return e
        keep: list[int] = []
        for c, cap in caps.items():
            sel = np.where(e.phase == c)[0][:cap]     # 시간순 앞쪽 우선
            keep.extend(int(i) for i in sel)
        idx = np.asarray(sorted(keep), dtype=np.int64)
    else:
        raise ValueError(f"알 수 없는 truncate 모드: {mode}")
    if len(idx) < MIN_TRUNC_LEN:
        return None
    return Episode(e.task, e.ep_id, e.scene, e.noise, e.succ,
                   np.ascontiguousarray(e.X[idx]), np.ascontiguousarray(e.phase[idx]))


def apply_truncation(split: dict, mode: str, W: int | None,
                     caps: dict[int, int] | None) -> dict:
    """TRAIN·CALIB 만 절제하고 dropped 수를 센다. TEST 는 손대지 않는다."""
    dropped = {"train": 0, "calib": 0}
    for part in ("train", "calib"):
        kept = []
        for e in split[part]:
            te = truncate_episode(e, mode, W, caps)
            if te is None:
                dropped[part] += 1
            else:
                kept.append(te)
        split[part] = kept
    return dropped


# =============================================================================
# 학습 / 채점
# =============================================================================

def train_detector(kind: str, seqs: list[tuple[np.ndarray, int]], input_dim: int,
                   epochs: int, lr: float, hidden: int, lam: float, grad_clip: float,
                   batch_size: int, seed: int, device: str = "cpu",
                   verbose: bool = True) -> nn.Module:
    """per-step BCE(episode 라벨을 전 step 에 broadcast) + λ·L2(bias 제외) + grad clip.

    참조 구현은 bs=1 이라 판 수가 많으면 느리다 → 패딩 배치 + mask 로 pad step 제외
    (mask 평균이라 bs=1 과 손실 정의가 같다).
    """
    torch.manual_seed(seed)
    model = build_detector(kind, input_dim, hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    n = len(seqs)
    for ep in range(epochs):
        order = rng.permutation(n)
        tot, nb = 0.0, 0
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            batch = [seqs[i] for i in idx]
            L = max(len(b[0]) for b in batch)
            xb = torch.zeros(len(batch), L, input_dim)
            yb = torch.zeros(len(batch), L)
            mb = torch.zeros(len(batch), L)
            for i, (X, y) in enumerate(batch):
                t = len(X)
                xb[i, :t] = torch.from_numpy(X)
                yb[i, :t] = float(y)
                mb[i, :t] = 1.0
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            sc = model(xb).clamp(1e-6, 1 - 1e-6)
            bce = -(yb * torch.log(sc) + (1 - yb) * torch.log(1 - sc))
            loss = (bce * mb).sum() / mb.sum().clamp(min=1.0)
            if lam > 0:
                loss = loss + lam * sum((p ** 2).sum() for nm, p in model.named_parameters()
                                        if "bias" not in nm)
            opt.zero_grad()
            loss.backward()
            if grad_clip:
                clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tot += float(loss.item())
            nb += 1
        if verbose and (ep == 0 or (ep + 1) % 5 == 0 or ep == epochs - 1):
            print(f"      epoch {ep+1}/{epochs} loss={tot/max(nb,1):.4f}", flush=True)
    model.eval()
    return model


@torch.no_grad()
def score_seq(model: nn.Module, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    """[T] per-step 검출 score. MLP 는 step 독립이라 출력 **누적평균**이 score."""
    xb = torch.from_numpy(np.ascontiguousarray(X)).float().unsqueeze(0).to(device)
    raw = model(xb).squeeze(0).cpu().numpy()
    if getattr(model, "cumulative", False):
        raw = np.cumsum(raw) / np.arange(1, len(raw) + 1)
    return raw.astype(np.float64)


def _pad_to(s: np.ndarray, L: int) -> np.ndarray:
    return s[:L] if len(s) >= L else np.concatenate([s, np.full(L - len(s), s[-1])])


def functional_cp_band(band_scores: list[np.ndarray], calib_scores: list[np.ndarray],
                       alpha: float) -> dict | None:
    """δ_t = μ_t + bw·σ_t. μ/σ = band_scores(성공), bw = calib 성공의 초과량 (1−α) 분위.

    궤적은 밴드 길이 L 로 forward-fill 패딩(성공 종료 후 plateau) — 짧은 성공 판이
    밴드를 끌어내려 종반에 헛발화하는 것을 막는다.
    """
    if len(band_scores) < MIN_BAND_EPS or len(calib_scores) < MIN_BAND_EPS:
        return None
    L = max(len(s) for s in band_scores)
    tr = np.stack([_pad_to(s, L) for s in band_scores])
    ca = np.stack([_pad_to(s, L) for s in calib_scores])
    mu = tr.mean(axis=0)
    sd = tr.std(axis=0, ddof=1) + EPS
    bw = float(np.quantile(np.max((ca - mu) / sd, axis=1), 1.0 - alpha))
    return {"mu": mu, "sd": sd, "bw": bw, "delta": mu + bw * sd, "L": int(L)}


def fire_step(score: np.ndarray, delta: np.ndarray) -> int | None:
    """첫 발화 t. t 가 밴드 길이를 넘으면 **마지막 밴드 유지**."""
    idx = np.minimum(np.arange(len(score)), len(delta) - 1)
    cross = np.where(score > delta[idx])[0]
    return int(cross[0]) if len(cross) else None


# =============================================================================
# 지표
# =============================================================================

def auroc(scores: np.ndarray, y: np.ndarray) -> float | None:
    """rank AUROC (tie 평균). y=1 이 positive(=failure)."""
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    _, inv, cnt = np.unique(scores, return_inverse=True, return_counts=True)
    s = np.zeros(cnt.size)
    np.add.at(s, inv, ranks)
    ranks = (s / cnt)[inv]
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def _med(v: list[float]) -> float | None:
    return round(float(np.median(v)), 4) if v else None


def td_auroc(pairs: list[tuple[float, int]]) -> tuple[float | None, int]:
    """(score, y) 목록 → AUROC + 표본 수. 단일 클래스면 None."""
    if not pairs:
        return None, 0
    sc = np.array([p[0] for p in pairs], dtype=np.float64)
    yy = np.array([p[1] for p in pairs], dtype=np.int64)
    a = auroc(sc, yy)
    return (None if a is None else round(a, 4)), len(pairs)


def td_pairs(test_scores: list[tuple[Episode, np.ndarray]], t_d: int
             ) -> list[tuple[float, int]]:
    """t_d 시점의 causal score. **t_d 에 이미 끝난 판(T ≤ t_d)은 제외** — 종료 자체가
    성공 신호라 포함하면 길이 confound 를 AUROC 에 되돌려 넣는 꼴이다."""
    return [(float(sc[t_d]), e.y) for e, sc in test_scores if e.T > t_d]


def td_metrics(test_scores: list[tuple[Episode, np.ndarray]]) -> dict:
    out: dict = {}
    for t_d in TD_GRID:
        a, n = td_auroc(td_pairs(test_scores, t_d))
        out[f"auroc_td{t_d}"] = a
        out[f"n_td{t_d}"] = n
    return out


def timer_records(test_eps: list[Episode], task: str, W: int | None,
                  phase_names: dict) -> list[dict]:
    """무feature 기준선: t ≥ W 이면 발화. W 이후로 판이 이어지는가만 본다."""
    recs = []
    for e in test_eps:
        ft = int(W) if (W is not None and e.T > int(W)) else None
        recs.append({
            "task": task, "arm": "-", "model": "timer", "alpha": None,
            "ep_id": e.ep_id, "scene": e.scene, "noise": e.noise,
            "succ": e.succ, "y": e.y, "T": e.T, "W": None if W is None else int(W),
            "fired": ft is not None, "t_fire": ft,
            "relpos": None if ft is None else round(ft / max(e.T - 1, 1), 4),
            "steps_before_end": None if ft is None else int(e.T - 1 - ft),
            "fire_phase": None if ft is None else
                          phase_names.get(int(e.phase[ft]), str(int(e.phase[ft]))),
            "max_score": None,
        })
    return recs


def summarize(records: list[dict], phase_top: int = 4) -> dict:
    """episode 발화 기록 → TSV 한 행 분량의 지표."""
    fail = [r for r in records if r["y"] == 1]
    succ = [r for r in records if r["y"] == 0]
    f_fired = [r for r in fail if r["fired"]]
    s_fired = [r for r in succ if r["fired"]]
    dist: dict[str, int] = {}
    for r in f_fired:
        k = str(r["fire_phase"])
        dist[k] = dist.get(k, 0) + 1
    top = sorted(dist.items(), key=lambda kv: (-kv[1], kv[0]))[:phase_top]
    la = auroc(np.array([r["T"] for r in records], dtype=np.float64),
               np.array([r["y"] for r in records]))
    # timer(=길이만 보는 기준선) 대비 조기성. W 는 판마다 task 별 값이 실려 있다.
    early = [r for r in f_fired if r.get("W") is not None and r["t_fire"] < r["W"]]
    lead = [float(r["W"] - r["t_fire"]) for r in f_fired if r.get("W") is not None]
    ws = sorted({r["W"] for r in records if r.get("W") is not None})
    # 시퀀스 수준(=판 끝까지 본) 분리도. causal 하지 않으므로 조기 검출 근거가 아니라
    # "길이만 학습했는지" 대조용 (length_auroc 와 나란히 읽는다).
    ms = [r["max_score"] for r in records]
    msa = (auroc(np.array(ms, dtype=np.float64), np.array([r["y"] for r in records]))
           if all(v is not None for v in ms) and records else None)
    return {
        "n_test_ep": len(records),
        "n_fail": len(fail), "n_succ": len(succ),
        "tpr": round(len(f_fired) / len(fail), 4) if fail else None,
        "fpr": round(len(s_fired) / len(succ), 4) if succ else None,
        "tpr_before_W": round(len(early) / len(fail), 4) if fail and ws else None,
        "lead_vs_W": _med(lead),
        "W": ws[0] if len(ws) == 1 else ("|".join(map(str, ws)) if ws else None),
        "median_relpos_fail": _med([r["relpos"] for r in f_fired]),
        "median_steps_before_end_fail": _med([r["steps_before_end"] for r in f_fired]),
        "median_relpos_fp": _med([r["relpos"] for r in s_fired]),
        "fire_phase_dist": "|".join(f"{k}:{v}" for k, v in top),
        "length_auroc": None if la is None else round(la, 4),
        "maxscore_auroc": None if msa is None else round(msa, 4),
    }


# =============================================================================
# arm 실행
# =============================================================================

def run_arm(arm: str, kind: str, tasks: dict, splits: dict, args,
            out_dir: Path) -> tuple[list[dict], list[dict], dict]:
    """arm × model 하나 학습 + 발화 시뮬. 반환 = (요약행, 상세기록, 체크포인트 payload들)."""
    alphas = args.alphas
    rows: list[dict] = []
    detail: list[dict] = []
    ckpts: dict[str, dict] = {}
    td_pool: dict[int, list[tuple[float, int]]] = {t_d: [] for t_d in TD_GRID}

    groups = [("__all__", sorted(tasks))] if arm == "mixed" else \
             [(t, [t]) for t in sorted(tasks)]

    for gname, gtasks in groups:
        tr_eps = [e for t in gtasks for e in splits[t]["train"]]
        if not tr_eps:
            print(f"  [skip] {arm}/{kind}/{gname}: train 판 0", flush=True)
            continue
        mu, sd = standardizer(tr_eps)
        seqs = [(apply_std(e, mu, sd), e.y) for e in tr_eps]
        input_dim = seqs[0][0].shape[1]
        n_fail_tr = sum(1 for _, y in seqs if y == 1)
        print(f"  [train] arm={arm} model={kind} group={gname} "
              f"n_train={len(seqs)} (fail {n_fail_tr}) dim={input_dim}", flush=True)
        if n_fail_tr == 0 or n_fail_tr == len(seqs):
            print(f"  [skip] {arm}/{kind}/{gname}: train 이 단일 클래스", flush=True)
            for t in gtasks:
                rows.append({"task": t, "arm": arm, "model": kind, "alpha": None,
                             "skip_reason": "train single-class"})
            continue
        model = train_detector(kind, seqs, input_dim, args.epochs, args.lr, args.hidden,
                               args.lambda_reg, args.grad_clip, args.batch_size,
                               args.seed, verbose=not args.quiet)

        bands_ck: dict[str, dict] = {}
        for t in gtasks:
            sp = splits[t]
            band_src = [e for e in sp["train"] if e.succ == 1] if args.band_mu == "train" \
                else [e for e in sp["calib"] if e.succ == 1]
            calib_src = [e for e in sp["calib"] if e.succ == 1]
            band_scores = [score_seq(model, apply_std(e, mu, sd)) for e in band_src]
            calib_scores = [score_seq(model, apply_std(e, mu, sd)) for e in calib_src]
            test_eps = sp["test"]
            if not test_eps:
                rows.append({"task": t, "arm": arm, "model": kind, "alpha": None,
                             "skip_reason": "test 판 0"})
                continue
            test_scores = [(e, score_seq(model, apply_std(e, mu, sd))) for e in test_eps]
            tdm = td_metrics(test_scores)          # α 무관 (밴드 안 씀) — 행마다 동일값
            for t_d in TD_GRID:
                td_pool[t_d].extend(td_pairs(test_scores, t_d))
            W_task = tasks[t].get("W")

            band_ck: dict[str, dict] = {}
            for a in alphas:
                band = functional_cp_band(band_scores, calib_scores, a)
                if band is None:
                    rows.append({
                        "task": t, "arm": arm, "model": kind, "alpha": a,
                        "skip_reason": f"성공 판 부족 (band {len(band_scores)} / "
                                       f"calib {len(calib_scores)} < {MIN_BAND_EPS})"})
                    continue
                recs = []
                for e, sc in test_scores:
                    ft = fire_step(sc, band["delta"])
                    rec = {
                        "task": t, "arm": arm, "model": kind, "alpha": a,
                        "ep_id": e.ep_id, "scene": e.scene, "noise": e.noise,
                        "succ": e.succ, "y": e.y, "T": e.T,
                        "W": None if W_task is None else int(W_task),
                        "fired": ft is not None,
                        "t_fire": ft,
                        "relpos": None if ft is None else
                                  round(ft / max(e.T - 1, 1), 4),
                        "steps_before_end": None if ft is None else int(e.T - 1 - ft),
                        "fire_phase": None if ft is None else
                                      tasks[t]["phase_names"].get(int(e.phase[ft]),
                                                                  str(int(e.phase[ft]))),
                        "max_score": round(float(sc.max()), 4),
                    }
                    recs.append(rec)
                    detail.append(rec)
                row = {"task": t, "instruction": tasks[t]["instruction"], "arm": arm,
                       "model": kind, "alpha": a, "band_L": band["L"],
                       "bw": round(band["bw"], 4),
                       "n_train_ep": len(seqs), "n_band_succ": len(band_scores),
                       "n_calib_succ": len(calib_scores),
                       "train_scenes": ",".join(map(str, sp["scenes"]["train"])),
                       "calib_scenes": ",".join(map(str, sp["scenes"]["calib"])),
                       "test_scenes": ",".join(map(str, sp["scenes"]["test"])),
                       "truncate": args.truncate_train,
                       "n_trunc_dropped": tasks[t].get("n_trunc_dropped", 0),
                       "skip_reason": ""}
                row.update(summarize([r for r in recs]))
                row.update(tdm)
                rows.append(row)
                band_ck[f"{a:.2f}"] = {"mu": band["mu"].astype(np.float32),
                                       "sd": band["sd"].astype(np.float32),
                                       "bw": band["bw"],
                                       "delta": band["delta"].astype(np.float32)}
            if band_ck:
                bands_ck[t] = band_ck

        ckpts[gname] = {
            "arm": arm, "model": kind, "group": gname,
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "input_dim": input_dim, "hidden": args.hidden,
            "std_mean": mu, "std_std": sd,
            "feature": {"layer": args.layer, "denoise": args.denoise, "seg": args.seg,
                        "layer_idx": tasks[gtasks[0]]["layer_idx"],
                        "denoise_idx": tasks[gtasks[0]]["denoise_idx"],
                        "seg_idx": tasks[gtasks[0]]["seg_idx"],
                        "dim": input_dim},
            "cp_bands": bands_ck,
            "tasks": gtasks,
            "shards": [tasks[t]["shard"] for t in gtasks],   # basename 만 (docs/04 §8)
            "train": {"epochs": args.epochs, "lr": args.lr, "lambda_reg": args.lambda_reg,
                      "grad_clip": args.grad_clip, "batch_size": args.batch_size,
                      "seed": args.seed, "band_mu": args.band_mu},
            "truncate": {
                "mode": args.truncate_train,
                "rollout_W": {t: tasks[t].get("W") for t in gtasks},
                "phase_caps": {t: tasks[t].get("phase_caps") for t in gtasks},
                "n_dropped": {t: tasks[t].get("n_trunc_dropped", 0) for t in gtasks},
            },
        }

    # 풀링 행 (arm/model/α 별 전체 test 합산)
    pooled_td = {}
    for t_d in TD_GRID:
        a_, n_ = td_auroc(td_pool[t_d])
        pooled_td[f"auroc_td{t_d}"], pooled_td[f"n_td{t_d}"] = a_, n_
    for a in alphas:
        recs = [r for r in detail if r["alpha"] == a]
        if recs:
            row = {"task": "__pooled__", "instruction": "", "arm": arm, "model": kind,
                   "alpha": a, "truncate": args.truncate_train, "skip_reason": ""}
            row.update(summarize(recs))
            row.update(pooled_td)
            rows.append(row)

    for gname, payload in ckpts.items():
        slug = "all" if gname == "__all__" else gname
        torch.save(payload, out_dir / f"detector_{arm}_{kind}_{slug}.pt")
    return rows, detail, ckpts


# =============================================================================
# TSV
# =============================================================================

TSV_COLS = (["task", "instruction", "arm", "model", "alpha", "truncate", "n_test_ep",
             "n_fail", "n_succ", "tpr", "fpr", "tpr_before_W", "lead_vs_W", "W",
             "median_relpos_fail", "median_steps_before_end_fail", "median_relpos_fp",
             "fire_phase_dist", "length_auroc", "maxscore_auroc"]
            + [f"auroc_td{t}" for t in TD_GRID] + [f"n_td{t}" for t in TD_GRID]
            + ["band_L", "bw", "n_train_ep", "n_band_succ", "n_calib_succ",
               "n_trunc_dropped", "train_scenes", "calib_scenes", "test_scenes",
               "skip_reason"])


def write_tsv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(TSV_COLS)]
    for r in rows:
        lines.append("\t".join("" if r.get(c) is None else str(r.get(c, ""))
                               for c in TSV_COLS))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# self-test (합성 데이터 — 원격 실행 전 로컬 게이트)
# =============================================================================

def _write_synth_shard(path: Path, n_scene=10, n_noise=6, dim=32, seed=0,
                       onset=0.4, signal=1.6):
    """shard 계약과 동일한 NPZ 를 합성한다. 실패 판은 onset 이후 일부 축이 이동."""
    rng = np.random.default_rng(seed)
    layers = [0, 2, 4, 8, 10, 12, 15]
    K, S = 4, 4
    Xs, ep_id, scene, noise, rec_idx, succ, phase, ep_len = [], [], [], [], [], [], [], []
    codebook = {"reach": 0, "grasp": 1, "transport": 2, "release": 3}
    e = 0
    for s in range(n_scene):
        for nz in range(n_noise):
            fail = bool(rng.random() < 0.45)
            T = int(rng.integers(18, 26)) if fail else int(rng.integers(10, 18))
            base = rng.normal(0, 1, size=(T, dim)).astype(np.float32)
            base += rng.normal(0, 0.5, size=(1, dim)).astype(np.float32)   # scene 효과
            if fail:
                t0 = int(onset * T)
                base[t0:, :4] += signal
            blk = np.zeros((T, len(layers), K, S, dim), dtype=np.float16)
            # layer 12 × denoise 3 × seg "all" 만 신호를 담고 나머지는 잡음
            blk[:] = rng.normal(0, 1, size=blk.shape).astype(np.float16)
            blk[:, layers.index(12), K - 1, S - 1, :] = base.astype(np.float16)
            Xs.append(blk)
            ep_id.append(np.full(T, e, np.int32))
            scene.append(np.full(T, s, np.int16))
            noise.append(np.full(T, nz, np.int16))
            rec_idx.append(np.arange(T, dtype=np.int16))
            succ.append(np.full(T, 0 if fail else 1, np.int8))
            ph = np.minimum((np.arange(T) * 4) // T, 3).astype(np.int16)
            phase.append(ph)
            ep_len.append(np.full(T, T, np.int16))
            e += 1
    meta = {"instruction": path.stem, "capture_layers": layers,
            "segment_names": ["state", "future", "action", "all"],
            "phase_codebook": codebook, "denoise_k": K, "dim": dim, "tier": "segA"}
    np.savez(path, X=np.concatenate(Xs), ep_id=np.concatenate(ep_id),
             scene=np.concatenate(scene), noise=np.concatenate(noise),
             rec_idx=np.concatenate(rec_idx), succ=np.concatenate(succ),
             phase_code=np.concatenate(phase), ep_len=np.concatenate(ep_len),
             meta_json=np.array(json.dumps(meta, ensure_ascii=False)))


def _synth_case(root: Path, tag: str, seed: int, signal: float, onset: float,
                n_scene: int, n_noise: int) -> Path:
    """합성 shard 2개(=task 2개)를 담은 디렉터리."""
    sd = root / f"segA_{tag}"
    sd.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(("SynthTaskA", "SynthTaskB")):
        _write_synth_shard(sd / f"{name}.npz", n_scene=n_scene, n_noise=n_noise,
                           seed=seed + i, onset=onset, signal=signal)
    return sd


def _synth_run(args, shard_dir: Path, out_dir: Path, truncate: str,
               epochs: int) -> list[dict]:
    """합성 케이스 1회 실행 → sim_rows.json 의 행 목록."""
    a = argparse.Namespace(**vars(args))
    a.shard_dir, a.out = shard_dir, out_dir
    a.truncate_train = truncate
    a.epochs = epochs
    a.self_test = False
    a.arm = "pertask"
    a.models = "lstm"
    a.quiet = True
    a.train_scenes, a.calib_scenes, a.test_scenes = 8, 3, 3
    rc = run(a)
    if rc != 0:
        raise RuntimeError(f"self-test 하위 실행 실패 (truncate={truncate})")
    return json.loads((out_dir / "sim_rows.json").read_text())


def _pooled(rows: list[dict]) -> list[dict]:
    return [r for r in rows
            if r.get("task") == "__pooled__" and r.get("model") != "timer"
            and r.get("tpr") is not None]


def self_test(args) -> int:
    """합성 데이터 게이트.

    (1) 기존 게이트 — 신호 있는 합성에서 TPR > FPR.
    (2) 길이 절제 게이트 — 합성 2종:
        (a) feature 신호 O + 길이 confound  → truncate=rollout 에서도 조기 AUROC 유지
        (b) feature 신호 X + 길이 confound  → truncate=rollout 이면 조기 AUROC ≈ 0.5,
            full 학습이면 (길이만 보고) 종반 검출은 살아난다.
    합성 잡음을 감안해 임계는 느슨하게 잡고, 어긋나면 이유를 찍고 fail-loud.
    """
    epochs = min(args.epochs, 8)
    t_early = 10          # (a) 의 onset(0.25·T ≈ 4~6) 이후이면서 W 보다 한참 앞
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        sig_dir = _synth_case(root, "sig", args.seed, signal=2.0, onset=0.25,
                              n_scene=14, n_noise=8)
        len_dir = _synth_case(root, "lenonly", args.seed + 100, signal=0.0, onset=0.4,
                              n_scene=14, n_noise=8)

        print("\n=== [self-test 1/4] 신호 O · truncate=none (기존 게이트) ===")
        rows_a_none = _synth_run(args, sig_dir, root / "out_a_none", "none", epochs)
        print("\n=== [self-test 2/4] 신호 O · truncate=rollout ===")
        rows_a_tr = _synth_run(args, sig_dir, root / "out_a_rollout", "rollout", epochs)
        print("\n=== [self-test 3/4] 신호 X(길이만) · truncate=none ===")
        rows_b_none = _synth_run(args, len_dir, root / "out_b_none", "none", epochs)
        print("\n=== [self-test 4/4] 신호 X(길이만) · truncate=rollout ===")
        rows_b_tr = _synth_run(args, len_dir, root / "out_b_rollout", "rollout", epochs)

        def _td(rows) -> float | None:
            v = [r.get(f"auroc_td{t_early}") for r in _pooled(rows)
                 if r.get(f"auroc_td{t_early}") is not None]
            return float(v[0]) if v else None

        def _best_gap(rows) -> float | None:
            g = [r["tpr"] - (r["fpr"] or 0.0) for r in _pooled(rows)]
            return max(g) if g else None

        def _late(rows) -> float | None:
            v = [r.get("maxscore_auroc") for r in _pooled(rows)
                 if r.get("maxscore_auroc") is not None]
            return float(v[0]) if v else None

        fails: list[str] = []

        pooled = _pooled(rows_a_none)
        if not pooled:
            fails.append("(1) 신호 O/none: pooled 유효 행 0")
        print("\n[self-test] (1) 신호 O · none — pooled (TPR > FPR 이어야 통과)")
        for r in pooled:
            good = r["tpr"] > (r["fpr"] or 0.0)
            if not good and r["alpha"] <= 0.25:
                fails.append(f"(1) α={r['alpha']}: TPR {r['tpr']} ≤ FPR {r['fpr']}")
            print(f"  arm={r['arm']:7s} model={r['model']:4s} α={r['alpha']:.2f} "
                  f"TPR={r['tpr']} FPR={r['fpr']} relpos={r['median_relpos_fail']} "
                  f"{'OK' if good else 'FAIL'}")

        a_tr, b_tr, b_none = _td(rows_a_tr), _td(rows_b_tr), _td(rows_b_none)
        gap_b_none, gap_b_tr = _best_gap(rows_b_none), _best_gap(rows_b_tr)
        late_b_none, late_b_tr = _late(rows_b_none), _late(rows_b_tr)
        print(f"\n[self-test] 절제 게이트 (t_d={t_early} 조기 AUROC / 종반 maxscore AUROC)")
        print(f"  (a) 신호 O  rollout 절제 : auroc_td{t_early}={_fmt(a_tr)}  (> 0.65 기대)")
        print(f"  (b) 신호 X  rollout 절제 : auroc_td{t_early}={_fmt(b_tr)}  (≲ 0.65 기대)"
              f" | 종반={_fmt(late_b_tr)} TPR−FPR(best α)={_fmt(gap_b_tr)}")
        print(f"  (b) 신호 X  none        : auroc_td{t_early}={_fmt(b_none)}  "
              f"| 종반={_fmt(late_b_none)} "
              f"TPR−FPR(best α)={_fmt(gap_b_none)} (> 0.10 기대)")

        if a_tr is None:
            fails.append(f"(a) 신호 O/rollout: auroc_td{t_early} 계산 불가 (표본 부족)")
        elif a_tr <= 0.65:
            fails.append(f"(a) 신호 O/rollout: 조기 AUROC {a_tr} ≤ 0.65 — 절제가 "
                         "진짜 feature 신호까지 죽였다")
        if b_tr is None:
            fails.append(f"(b) 신호 X/rollout: auroc_td{t_early} 계산 불가")
        elif b_tr > 0.65:
            fails.append(f"(b) 신호 X/rollout: 조기 AUROC {b_tr} > 0.65 — 길이만 있는 "
                         "합성인데 조기 분리가 나온다 (절제 누수 의심)")
        # (b) 에 길이 confound 가 실제로 있고 종반에는 읽힌다는 것의 근거:
        #   ① timer(무feature 길이 기준선)가 (b) 에서 거의 완벽히 분리한다 — 데이터 sanity
        #   ② full 학습 detector 도 종반 발화로 양의 TPR−FPR 를 낸다 (조기 td 는 ≈0.5)
        timer_gap = None
        for r in rows_b_none:
            if r.get("task") == "__pooled__" and r.get("model") == "timer" \
                    and r.get("tpr") is not None:
                timer_gap = r["tpr"] - (r["fpr"] or 0.0)
        print(f"  (b) timer 기준선(길이만)  : TPR−FPR={_fmt(timer_gap)} (> 0.5 기대)")
        if timer_gap is None or timer_gap <= 0.5:
            fails.append(f"(b) timer 기준선 TPR−FPR={timer_gap} — 합성에 길이 confound "
                         "가 없다. 대조 자체가 성립하지 않는다")
        if gap_b_none is None:
            fails.append("(b) 신호 X/none: pooled 유효 행 0")
        elif gap_b_none <= 0.10:
            fails.append(f"(b) 신호 X/none: TPR−FPR {gap_b_none:.3f} ≤ 0.10 — full 학습 "
                         "detector 가 길이 신호조차 못 잡았다 (대조 무의미)")

        if fails:
            print("\n[self-test] FAIL")
            for f in fails:
                print(f"  - {f}")
            return 1
        print("\n[self-test] PASS")
        return 0


# =============================================================================
# main
# =============================================================================

def run(args) -> int:
    torch.set_num_threads(max(1, int(args.threads)))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    shard_dir = Path(args.shard_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    only = [s.strip() for s in args.shards.split(",") if s.strip()] if args.shards else None
    paths = discover_shards(shard_dir, only)

    tasks: dict[str, dict] = {}
    splits: dict[str, dict] = {}
    t0 = time.time()
    for p in paths:
        eps, spec = load_shard_episodes(p, args.layer, args.denoise, args.seg)
        slug = p.stem
        sc_split = split_scenes(slug, [e.scene for e in eps], args.train_scenes,
                                args.calib_scenes, args.test_scenes, args.seed)
        part = {k: [e for e in eps if e.scene in set(v)] for k, v in sc_split.items()}
        tasks[slug] = {
            "instruction": spec.instruction, "shard": p.name,
            "phase_names": spec.phase_names, "dim": spec.dim,
            "layer_idx": spec.layer_idx, "denoise_idx": spec.denoise_idx,
            "seg_idx": spec.seg_idx, "layers": spec.layers,
        }
        splits[slug] = {**part, "scenes": sc_split}
        # W/cap 은 truncate 모드와 무관하게 **항상** 계산한다 — timer 기준선·tpr_before_W
        # 가 모든 run 에서 필요하고, task 별(mixed arm 에서도 task 별)로 잡는다.
        W = rollout_cap(part["train"])
        caps = phase_dwell_caps(part["train"])
        tasks[slug]["W"] = W
        tasks[slug]["phase_caps"] = {str(k): v for k, v in caps.items()}
        dropped = apply_truncation(splits[slug], args.truncate_train, W, caps)
        tasks[slug]["n_trunc_dropped"] = int(sum(dropped.values()))
        if args.truncate_train != "none":
            print(f"[trunc] {slug}: mode={args.truncate_train} W={W} caps={caps} "
                  f"dropped(train/calib)={dropped['train']}/{dropped['calib']}", flush=True)
        elif W is not None:
            print(f"[trunc] {slug}: mode=none (W={W} 는 timer 기준선용으로만 사용)",
                  flush=True)
        sp = splits[slug]
        print(f"[load] {slug}: ep {len(eps)} (fail {sum(1 for e in eps if e.y == 1)}) "
              f"| train {len(sp['train'])} / calib {len(sp['calib'])} "
              f"(succ {sum(1 for e in sp['calib'] if e.succ == 1)}) "
              f"/ test {len(sp['test'])} | scenes {sc_split}", flush=True)

    dims = {t["dim"] for t in tasks.values()}
    if len(dims) != 1:
        raise SystemExit(f"shard 마다 dim 이 다르다 {sorted(dims)} — mixed arm 불가")

    arms = ["pertask", "mixed"] if args.arm == "both" else [args.arm]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    bad = [m for m in models if m not in ("lstm", "mlp")]
    if bad:
        raise SystemExit(f"--models 에 알 수 없는 값 {bad} (lstm|mlp)")

    rows: list[dict] = []
    detail: list[dict] = []
    for arm in arms:
        for kind in models:
            r, d, _ = run_arm(arm, kind, tasks, splits, args, out_dir)
            rows += r
            detail += d

    # timer 기준선 (무feature): "t ≥ W 면 발화". detector 가 이것보다 못하면 학습한 것은
    # 길이뿐이다. arm/α 무관이라 task 당 한 행 + 풀링 한 행.
    timer_all: list[dict] = []
    for t in sorted(tasks):
        recs = timer_records(splits[t]["test"], t, tasks[t].get("W"),
                             tasks[t]["phase_names"])
        if not recs:
            continue
        timer_all += recs
        row = {"task": t, "instruction": tasks[t]["instruction"], "arm": "-",
               "model": "timer", "alpha": None, "truncate": args.truncate_train,
               "skip_reason": ""}
        row.update(summarize(recs))
        rows.append(row)
    if timer_all:
        row = {"task": "__pooled__", "instruction": "", "arm": "-", "model": "timer",
               "alpha": None, "truncate": args.truncate_train, "skip_reason": ""}
        row.update(summarize(timer_all))
        rows.append(row)
        detail += timer_all

    write_tsv(rows, out_dir / "sim_summary.tsv")
    (out_dir / "sim_rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    payload = {
        "config": {
            "shards": [p.name for p in paths],          # basename 만 (docs/04 §8)
            "layer": args.layer, "denoise": args.denoise, "seg": args.seg,
            "arm": args.arm, "models": models, "alphas": list(args.alphas),
            "seed": args.seed, "epochs": args.epochs, "lr": args.lr,
            "hidden": args.hidden, "lambda_reg": args.lambda_reg,
            "grad_clip": args.grad_clip, "batch_size": args.batch_size,
            "band_mu": args.band_mu,
            "truncate_train": args.truncate_train,
            "truncate_caps": {t: {"rollout_W": tasks[t].get("W"),
                                  "phase_caps": tasks[t].get("phase_caps"),
                                  "n_dropped": tasks[t].get("n_trunc_dropped", 0)}
                              for t in sorted(tasks)},
            "td_grid": list(TD_GRID),
            "split": {"train": args.train_scenes, "calib": args.calib_scenes,
                      "test": args.test_scenes, "unit": "scene"},
            "label": "y=1 은 failure (SAFE 규약)",
        },
        "scene_split": {t: splits[t]["scenes"] for t in splits},
        "episodes": detail,
    }
    (out_dir / "sim_detail.json").write_text(json.dumps(payload, ensure_ascii=False),
                                             encoding="utf-8")

    print(f"\n[summary] {time.time()-t0:.0f}s — {len(rows)} 행 → {out_dir.name}/")
    hdr = f"{'task':22s} {'arm':7s} {'model':5s} {'α':>5s} {'TPR':>5s} {'FPR':>5s} " \
          f"{'preW':>5s} {'lead':>5s} {'relpos':>6s} {'left':>5s} {'lenA':>5s} " \
          f"{'td10':>5s} {'td20':>5s}"
    print(hdr)
    for r in rows:
        if r.get("tpr") is None:
            continue
        print(f"{str(r['task'])[:22]:22s} {r['arm']:7s} {r['model']:5s} "
              f"{_fmt(r['alpha']):>5s} {r['tpr']:5.2f} {(r['fpr'] or 0):5.2f} "
              f"{_fmt(r.get('tpr_before_W')):>5s} "
              f"{_fmt(r.get('lead_vs_W'), 1):>5s} "
              f"{_fmt(r['median_relpos_fail']):>6s} "
              f"{_fmt(r['median_steps_before_end_fail'],1):>5s} "
              f"{_fmt(r['length_auroc']):>5s} "
              f"{_fmt(r.get('auroc_td10')):>5s} {_fmt(r.get('auroc_td20')):>5s}")
    return 0


def _fmt(v, nd=2):
    return "-" if v is None else f"{v:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard-dir", help="Tier A shard 디렉터리 (<slug>.npz)")
    ap.add_argument("--out", help="출력 디렉터리")
    ap.add_argument("--shards", default="", help="slug 콤마목록 (기본: 디렉터리 전부)")
    ap.add_argument("--layer", type=int, default=12, help="물리 capture layer 번호")
    ap.add_argument("--denoise", type=int, default=-1,
                    help="denoise step index (-1 = 마지막 k)")
    ap.add_argument("--seg", default="all",
                    help="token segment 이름 (state|future|action|all)")
    ap.add_argument("--arm", default="both", choices=("both", "pertask", "mixed"))
    ap.add_argument("--models", default="lstm,mlp", help="lstm,mlp")
    ap.add_argument("--alphas", default="0.05,0.1,0.2,0.3", help="CP 유의수준(FPR 목표)")
    ap.add_argument("--train-scenes", type=int, default=6)
    ap.add_argument("--calib-scenes", type=int, default=2)
    ap.add_argument("--test-scenes", type=int, default=2)
    ap.add_argument("--truncate-train", default="none",
                    choices=("none", "rollout", "phase-gt"),
                    help="학습 데이터 길이 절제 (TRAIN·CALIB 만; TEST 는 항상 full). "
                         "rollout=성공 record 수 ceil(μ+1σ) 로 앞부분만, "
                         "phase-gt=phase 별 성공 dwell ceil(μ+1σ) cap. 기본 none")
    ap.add_argument("--band-mu", default="train", choices=("train", "calib"),
                    help="δ_t 의 μ/σ 출처 (bw 는 항상 calib 성공 판)")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lambda-reg", type=float, default=1e-2)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true", help="epoch 로그 끄기")
    ap.add_argument("--self-test", action="store_true",
                    help="합성 데이터로 파이프라인 검증 (TPR > FPR)")
    args = ap.parse_args()
    args.alphas = [float(x) for x in str(args.alphas).split(",") if str(x).strip()]

    if args.self_test:
        return self_test(args)
    if not args.shard_dir or not args.out:
        ap.error("--shard-dir 와 --out 은 필수 (--self-test 제외)")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

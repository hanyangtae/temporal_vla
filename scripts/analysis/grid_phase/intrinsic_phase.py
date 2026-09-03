#!/usr/bin/env python
"""Intrinsic (label-free) action-phase clustering on N1.5 grid activations.

라벨을 전혀 쓰지 않고 activation 자체에서 phase 후보(클러스터)를 찾고, 그 결과가 GT
phase 라벨과 얼마나 정렬되는지를 사후 측정한다. 파이프라인 규격은 동료(상우) 라인을
그대로 따른다 — `docs/steering/40_action_phase_readout_review.md` §2·§5·§6.

    DiT layer 12 · denoise 3 · 49토큰 평균 (1536d)
      → PCA-64 whitened (train split 에서만 적합)
      → KMeans (k 고정)
      → GT phase 대비 MI / margin / purity / boundary F1

동료 규격과의 **의도적 차이**: 동료는 PCA-64 뒤에 AE(latent 16)를 한 단 더 둔다. 여기서는
AE를 생략한다 (문서 §5 부록 "압축기가 필요한가" — PCA-64 → KMeans 직결이 AE 경로와
margin 1.634 vs 1.692 로 사실상 동률). 이 차이는 출력 meta 의 `deviations` 에 기록된다.

입력은 추출기가 만든 shard NPZ (고정 인터페이스):

    <shards-dir>/<instr_slug>.npz
        X          fp16 [n_rec, 7, 4, 4, 1536]
                   axis1 = capture_layers [0,2,4,8,10,12,15]  → layer 12 = index 5
                   axis2 = denoise K=4                        → k=3      = index 3
                   axis3 = segment (state, future, action, all) → "all"  = index 3
                                                                 (= 49토큰 평균)
        ep_id i32 / scene i16 / noise i16 / rec_idx i16 / succ i8
        phase_code i16 / ep_len i16 / meta_json (phase_codebook 포함)

즉 동료 규격 특징은 `X[:, 5, 3, 3, :]` 하나로 얻어진다 — 별도 추출 불필요.

출력
    <out>/intrinsic_<scope>_k<K>.npz        라벨 (통계 엔진과의 계약)
        {<instr_slug>: i16 [n_rec]  — shard 행 순서와 정렬, ..., "def_name": str}
    <out>/intrinsic_<scope>_k<K>_fit.npz    fit 부산물 (PCA 성분·centroid·EVR)
    <out>/intrinsic_<scope>_k<K>_meta.json  실행 파라미터·구현 출처·deviations
    --align-out <json>                      정렬 측정 (k sweep 시 k별 표)

사용 예
    python scripts/analysis/grid_phase/intrinsic_phase.py \
        --shards-dir <dir> --out <dir> --scope global --k 24 \
        --align-out <dir>/align_global.json

    python scripts/analysis/grid_phase/intrinsic_phase.py \
        --shards-dir <dir> --out <dir> --scope per-task --k 6,8,12 \
        --fit-scenes 0-7 --align-out <dir>/align_pertask.json

실행 환경: 승준 노드 `~/anaconda3/bin/python` (numpy 1.26 + torch 2.1 CPU).
scipy / sklearn / hdbscan 없음 — 이 스크립트는 numpy(+선택적 torch)만 쓴다.
"""
from __future__ import annotations

import os

# BLAS 스레드 cap — 공유 노드(20코어). numpy import 전에 설정해야 효력이 있다.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---- 동료 규격 축 인덱스 (문서 §6 "새 rollout에서 특징 만들기") -------------------
CAPTURE_LAYERS = [0, 2, 4, 8, 10, 12, 15]
LAYER_IDX = CAPTURE_LAYERS.index(12)   # = 5
DENOISE_IDX = 3                        # denoise step 3
SEGMENT_IDX = 3                        # "all" = 49토큰 평균

EPS = 1e-8


# =============================================================================
# shard I/O
# =============================================================================

FIELDS = ("ep_id", "scene", "noise", "rec_idx", "succ", "phase_code", "ep_len")


def load_shard(path: Path) -> dict:
    """shard NPZ → {feat [n,1536] f32, ep_id, scene, ..., meta}. 행 순서 보존.

    X 는 [n, 7, 4, 4, 1536] fp16 이라 레코드당 344KB. 한 shard 를 통째로 올린 뒤 곧바로
    슬라이스하고 버려 peak 을 shard 하나로 묶는다 (npz 는 부분 읽기가 안 된다).
    """
    out = {}
    with np.load(path, allow_pickle=False) as f:
        X = f["X"]
        if X.ndim != 5:
            raise ValueError(f"{path.name}: X.ndim={X.ndim}, expected 5 "
                             f"[n_rec, 7, 4, 4, 1536]")
        if X.shape[1] != len(CAPTURE_LAYERS):
            raise ValueError(f"{path.name}: layer axis {X.shape[1]} != "
                             f"{len(CAPTURE_LAYERS)} (capture_layers 불일치)")
        out["feat"] = np.ascontiguousarray(
            X[:, LAYER_IDX, DENOISE_IDX, SEGMENT_IDX, :]).astype(np.float32)
        del X
        for k in FIELDS:
            out[k] = f[k][:] if k in f else None
        meta_raw = f["meta_json"] if "meta_json" in f else None
    out["meta"] = json.loads(str(meta_raw)) if meta_raw is not None else {}
    out["name"] = path.stem
    n = len(out["feat"])
    for k in FIELDS:
        if out[k] is not None and len(out[k]) != n:
            raise ValueError(f"{path.name}: {k} 길이 {len(out[k])} != n_rec {n}")
    return out


def discover_shards(shards_dir: Path, only: list[str] | None) -> list[Path]:
    paths = sorted(p for p in shards_dir.glob("*.npz")
                   if not p.name.endswith(("_fit.npz",)))
    if only:
        keep = set(only)
        paths = [p for p in paths if p.stem in keep]
        missing = keep - {p.stem for p in paths}
        if missing:
            raise SystemExit(f"shard 없음: {sorted(missing)}")
    if not paths:
        raise SystemExit(f"shard NPZ 없음: {shards_dir}")
    return paths


# =============================================================================
# PCA (+ whitening) — 동료 phase/data/pca.py 와 동일 절차, numpy 직접 구현
# =============================================================================

def pca_fit(x: np.ndarray, n_comp: int) -> tuple[dict, np.ndarray]:
    """[N,F] → (stats, evr). 공분산 고유분해 (N > F 일 때 F×F 가 싸다).

    동료 `phase/data/pca.py: fit` 과 같은 정의 — 평균 제거 → cov eigh(내림차순 상위
    n_comp) → whiten 은 sqrt(eigenvalue) 나눗셈. torch 대신 numpy(float64)로 계산한다.
    """
    X = np.asarray(x, dtype=np.float64)
    n, F = X.shape
    n_comp = int(min(n_comp, F, max(n - 1, 1)))
    mu = X.mean(0)
    Xc = X - mu
    cov = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    lam_all, V = np.linalg.eigh(cov)              # 오름차순
    lam = np.clip(lam_all[::-1][:n_comp], 1e-8, None)
    Vt = np.ascontiguousarray(V[:, ::-1][:, :n_comp].T)   # [n_comp, F]
    evr = lam / max(np.clip(lam_all, 0, None).sum(), 1e-30)
    stats = {"mu": mu.astype(np.float32),
             "V": Vt.astype(np.float32),
             "sqrt_lam": np.sqrt(lam).astype(np.float32)}
    return stats, evr.astype(np.float32)


def pca_apply(x: np.ndarray, stats: dict, whiten: bool = True) -> np.ndarray:
    z = (np.asarray(x, dtype=np.float32) - stats["mu"]) @ stats["V"].T
    if whiten:
        z = z / (stats["sqrt_lam"] + EPS)
    return np.ascontiguousarray(z, dtype=np.float32)


# =============================================================================
# KMeans — 우선 동료 구현(task_classification), 실패 시 numpy fallback
# =============================================================================

def _load_tc_kmeans(tc_root: Path | None):
    """task_classification/phase/clustering/gpu.py 를 파일경로로 직접 로드.

    패키지 import 가 아니라 spec_from_file_location 이므로 hydra 등 레포 의존성이
    딸려 오지 않는다 (gpu.py 자체는 math/numpy/torch 만 import).
    반환: (GPUKMeans class, source_path) 또는 (None, 이유 문자열).
    """
    roots = [tc_root] if tc_root else [REPO_ROOT / "task_classification"]
    for root in roots:
        f = Path(root) / "phase" / "clustering" / "gpu.py"
        if not f.is_file():
            continue
        try:
            spec = importlib.util.spec_from_file_location("_tc_phase_gpu", f)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            km = getattr(mod, "GPUKMeans", None)
            if km is None:
                return None, f"{f}: GPUKMeans 없음"
            return km, str(f)
        except Exception as exc:                       # torch 없음 등
            return None, f"{f}: {type(exc).__name__}: {exc}"
    tried = ", ".join(str(Path(r) / "phase/clustering/gpu.py") for r in roots)
    return None, f"gpu.py 없음 (submodule 미체크아웃?) — tried: {tried}"


def _pairwise_sqdist(X: np.ndarray, C: np.ndarray, chunk: int = 8192):
    """[N,K] squared distance, 행 chunk 로 나눠 메모리 상한을 둔다."""
    c2 = (C ** 2).sum(1)[None, :]
    for s in range(0, len(X), chunk):
        blk = X[s:s + chunk]
        d = (blk ** 2).sum(1)[:, None] + c2 - 2.0 * (blk @ C.T)
        yield s, np.maximum(d, 0.0)


def _assign(X, C, chunk=8192):
    a = np.empty(len(X), np.int32)
    inertia = 0.0
    for s, d in _pairwise_sqdist(X, C, chunk):
        idx = d.argmin(1)
        a[s:s + len(idx)] = idx
        inertia += float(d[np.arange(len(idx)), idx].sum())
    return a, inertia


def _kmpp_init_np(X, K, rng):
    """k-means++ (D² 가중 샘플링). 동료 `_kmpp_init` 과 같은 절차."""
    n = len(X)
    first = int(rng.integers(n))
    C = [X[first]]
    d2 = ((X - X[first]) ** 2).sum(1)
    for _ in range(1, K):
        tot = d2.sum()
        if tot <= 0:
            C.append(X[int(rng.integers(n))])
        else:
            C.append(X[int(rng.choice(n, p=d2 / tot))])
        d2 = np.minimum(d2, ((X - C[-1]) ** 2).sum(1))
    return np.asarray(C, dtype=np.float32)


def kmeans_numpy(X, K, n_init=5, max_iter=300, seed=0):
    """numpy fallback. 빈 클러스터는 이전 중심 유지 (동료 `_lloyd` 과 동일 처리).

    ⚠ 반환은 **centroid 배열 하나뿐** (라벨 아님) — 라벨은 `_assign(X, C)` 로 얻고,
    `_assign` 은 `(labels, inertia)` **튜플**을 돌려준다. 둘 다 라벨 단일 반환으로
    착각하기 쉬움 (실사고: cluster_share_transfer.py 초판이 두 곳 모두 오호출,
    3a457aa 에서 수정). 외부에서 쓸 때 이 시그니처를 그대로 따를 것."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    best_C, best_I = None, None
    for r in range(n_init):
        rng = np.random.default_rng(seed + r)
        C = _kmpp_init_np(X, K, rng)
        for _ in range(max_iter):
            a, _ = _assign(X, C)
            newC = C.copy()
            for k in range(K):
                m = a == k
                if m.any():
                    newC[k] = X[m].mean(0)
            if np.allclose(newC, C, atol=1e-5):
                C = newC
                break
            C = newC
        _, inertia = _assign(X, C)
        if best_I is None or inertia < best_I:
            best_C, best_I = C, inertia
    return best_C


class KMeansRunner:
    """동료 구현이 있으면 그것을, 없으면 numpy fallback 을 쓴다. 출처를 기록한다."""

    def __init__(self, impl: str = "auto", tc_root: Path | None = None,
                 n_init: int = 5, max_iter: int = 300, seed: int = 0):
        self.n_init, self.max_iter, self.seed = n_init, max_iter, seed
        self.cls = None
        self.impl = "numpy"
        self.source = "builtin numpy kmeans++ (fallback)"
        self.load_error = None
        if impl in ("auto", "task_classification"):
            cls, info = _load_tc_kmeans(tc_root)
            if cls is not None:
                self.cls, self.impl, self.source = cls, "task_classification", info
            else:
                self.load_error = info
                if impl == "task_classification":
                    raise SystemExit(f"task_classification KMeans 로드 실패: {info}")

    def fit_predict(self, X_fit, X_all, K):
        """fit 은 X_fit 에서만, 라벨은 X_all 전체에. → (labels i16, centroids f32)."""
        if self.cls is not None:
            est = self.cls(K, n_init=self.n_init, max_iter=self.max_iter,
                           seed=self.seed, device="cpu").fit(X_fit)
            lab = np.asarray(est.predict(X_all), dtype=np.int16)
            return lab, np.asarray(est.cluster_centers_, dtype=np.float32)
        C = kmeans_numpy(X_fit, K, self.n_init, self.max_iter, self.seed)
        lab, _ = _assign(np.ascontiguousarray(X_all, dtype=np.float32), C)
        return lab.astype(np.int16), C


# =============================================================================
# 정렬 측정 — MI / margin / purity / boundary F1
# =============================================================================

def _contingency(z, y):
    _, zi = np.unique(np.asarray(z), return_inverse=True)
    _, yi = np.unique(np.asarray(y), return_inverse=True)
    cm = np.zeros((zi.max() + 1, yi.max() + 1), np.float64)
    np.add.at(cm, (zi, yi), 1.0)
    return cm


def mi_bits(z, y) -> tuple[float, float, float]:
    """(MI, H(z), H(y)) — 전부 bits. 동료 `uncertainty_coef` 와 같은 joint 추정."""
    cm = _contingency(z, y)
    n = cm.sum()
    if n <= 0:
        return 0.0, 0.0, 0.0
    pij = cm / n
    pz = pij.sum(1, keepdims=True)
    py = pij.sum(0, keepdims=True)
    nz = pij > 0
    mi = float((pij[nz] * np.log2(pij[nz] / (pz @ py)[nz])).sum())
    hz = float(-(pz[pz > 0] * np.log2(pz[pz > 0])).sum())
    hy = float(-(py[py > 0] * np.log2(py[py > 0])).sum())
    return mi, hz, hy


def purity(z, y) -> float:
    """sum_k max_c |{i: z_i=k, y_i=c}| / N (동료 phase/metrics/purity.py)."""
    z, y = np.asarray(z), np.asarray(y)
    _, yi = np.unique(y, return_inverse=True)
    return float(sum(np.bincount(yi[z == k]).max() for k in np.unique(z)) / len(z))


def episode_bounds(episode):
    """[N] 에피소드 인덱스 → 시작 오프셋 [B+1] (연속 구간 전제)."""
    episode = np.asarray(episode)
    return np.concatenate([[0], np.flatnonzero(np.diff(episode)) + 1, [len(episode)]])


def time_fraction(episode):
    """[N] 에피소드 내 진행도 0..1 (동료 structure.py: time_fraction)."""
    b = episode_bounds(episode)
    out = np.empty(len(episode), np.float64)
    for s, e in zip(b[:-1], b[1:]):
        n = e - s
        out[s:e] = np.arange(n) / max(n - 1, 1)
    return out


def clock_clusters(tfrac, n_states):
    """같은 스텝 집합·같은 상태 개수의 '시계' 대조군 (동료 structure.py 그대로).

    문서 §2 :43 — margin = I(상태열;GT) − I(clock;GT). clock 의 상태 수는 **발견
    상태열의 상태 수**에 맞춘다. 그래야 k 증가에 따른 MI 자동상승이 상쇄된다.
    """
    n_states = max(int(n_states), 2)
    edges = np.quantile(tfrac, np.linspace(0, 1, n_states + 1)[1:-1])
    return np.digitize(tfrac, edges)


def boundary_f1(pred, true, episode, tol=1):
    """(f1, prec, rec). 동료 phase/metrics/boundary.py: boundary_f1 과 동일 알고리즘
    (±tol 허용 그리디 1:1 매칭, 최근접 GT 전환부터 소진)."""
    pred, true = np.asarray(pred), np.asarray(true)
    tp = fp = fn = 0
    b = episode_bounds(episode)
    for s, e in zip(b[:-1], b[1:]):
        if e - s < 2:
            continue
        p_seg, t_seg = pred[s:e], true[s:e]
        pb = np.flatnonzero(np.diff(p_seg)) + 1
        tb = np.flatnonzero(np.diff(t_seg)) + 1
        used = np.zeros(len(tb), bool)
        for p in pb:
            if len(tb) == 0:
                fp += 1
                continue
            dd = np.abs(tb - p)
            j = int(dd.argmin())
            if dd[j] <= tol and not used[j]:
                tp += 1
                used[j] = True
            else:
                fp += 1
        fn += int((~used).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if prec + rec else 0.0
    return float(f1), float(prec), float(rec)


def boundary_null(pred, true, episode, tol=1, n_perm=300, seed=0):
    """전환 **개수**는 보존하고 **위치만** 에피소드 안에서 무작위로 흩은 기준선.

    → (mean, std, z, n_transitions). 문서 §5 판정 도구와 동일 설계.
    """
    rng = np.random.default_rng(seed)
    b = episode_bounds(episode)
    segs = [(s, e) for s, e in zip(b[:-1], b[1:]) if e - s >= 2]
    n_tr = [int((np.diff(np.asarray(pred)[s:e]) != 0).sum()) for s, e in segs]
    obs = boundary_f1(pred, true, episode, tol)[0]
    fake = np.zeros(len(pred), np.int32)
    vals = np.empty(n_perm)
    for p in range(n_perm):
        for (s, e), m in zip(segs, n_tr):
            L = e - s
            pos = rng.choice(np.arange(1, L), size=min(m, L - 1), replace=False) \
                if m > 0 and L > 1 else np.array([], int)
            path = np.zeros(L, np.int32)
            if len(pos):
                path[np.sort(pos)] = 1
                path = np.cumsum(path)
            fake[s:e] = path
        vals[p] = boundary_f1(fake, true, episode, tol)[0]
    mu, sd = float(vals.mean()), float(vals.std())
    z = float((obs - mu) / sd) if sd > 1e-12 else 0.0
    return mu, sd, z, int(sum(n_tr))


def align_metrics(labels, phase, scene, ep_id, rec_idx, tol=1, n_perm=300, seed=0):
    """클러스터 라벨 vs GT phase 정렬 지표 묶음.

    행을 (ep_id, rec_idx) 로 정렬한 뷰에서 계산한다 (경계 지표는 에피소드 연속 구간 전제).
    phase < 0 (미라벨) 행은 GT 의존 지표에서만 제외한다.
    """
    order = np.lexsort((np.asarray(rec_idx), np.asarray(ep_id)))
    lab = np.asarray(labels)[order]
    ep = np.asarray(ep_id)[order]
    ph = np.asarray(phase)[order] if phase is not None else None
    sc = np.asarray(scene)[order] if scene is not None else None

    out = {"n_rec": int(len(lab)), "n_ep": int(len(np.unique(ep))),
           "n_clusters_used": int(len(np.unique(lab)))}

    if sc is not None:
        mi_sc, _, h_sc = mi_bits(lab, sc)
        out["mi_scene_bits"] = mi_sc
        out["h_scene_bits"] = h_sc
        out["u_scene"] = mi_sc / h_sc if h_sc > 1e-12 else 0.0

    if ph is None:
        return out

    ok = ph >= 0
    if not ok.any():
        out["gt_labeled"] = 0
        return out
    lab_g, ph_g, ep_g = lab[ok], ph[ok], ep[ok]
    out["gt_labeled"] = int(ok.sum())

    mi, h_c, h_p = mi_bits(lab_g, ph_g)
    out["mi_phase_bits"] = mi
    out["h_cluster_bits"] = h_c
    out["h_phase_bits"] = h_p
    # 정규화 MI: 산술평균 정규화 (0..1) + 동료 uncertainty coefficient U(phase|cluster)
    out["nmi_arith"] = 2 * mi / (h_c + h_p) if (h_c + h_p) > 1e-12 else 0.0
    out["u_phase"] = mi / h_p if h_p > 1e-12 else 0.0
    out["purity_phase"] = purity(lab_g, ph_g)

    # --- clock 대조군 (문서 :43) ---------------------------------------------
    tfrac = time_fraction(ep_g)
    n_states = len(np.unique(lab_g))
    clock = clock_clusters(tfrac, n_states)
    mi_clock, _, _ = mi_bits(clock, ph_g)
    out["n_clock_states"] = int(len(np.unique(clock)))
    out["mi_clock_bits"] = mi_clock
    out["margin_bits"] = mi - mi_clock
    # 보조: clock 상태 수를 GT phase 수에 맞춘 변형 (해석 참고용, 주 지표 아님)
    clock_gt = clock_clusters(tfrac, len(np.unique(ph_g)))
    mi_clock_gt, _, _ = mi_bits(clock_gt, ph_g)
    out["mi_clock_gtstates_bits"] = mi_clock_gt
    out["margin_gtstates_bits"] = mi - mi_clock_gt

    # --- 경계 ----------------------------------------------------------------
    f1, prec, rec = boundary_f1(lab_g, ph_g, ep_g, tol)
    mu, sd, z, n_tr = boundary_null(lab_g, ph_g, ep_g, tol, n_perm, seed)
    b = episode_bounds(ep_g)
    gt_tr = int(sum((np.diff(ph_g[s:e]) != 0).sum()
                    for s, e in zip(b[:-1], b[1:]) if e - s >= 2))
    out.update({
        "boundary_tol": int(tol),
        "boundary_f1": f1, "boundary_prec": prec, "boundary_rec": rec,
        "boundary_null_mean": mu, "boundary_null_std": sd, "boundary_z": z,
        "boundary_n_perm": int(n_perm),
        "n_transitions_pred": n_tr, "n_transitions_gt": gt_tr,
        "mean_seg_len_pred": float(len(lab_g) / max(n_tr + out["n_ep"], 1)),
        "mean_seg_len_gt": float(len(ph_g) / max(gt_tr + out["n_ep"], 1)),
    })
    return out


# =============================================================================
# scene 홀드아웃 파싱
# =============================================================================

def parse_scene_spec(spec: str | None) -> set[int] | None:
    """"0-7" / "0,1,2" / "0-3,7" → {int}. None 이면 전 scene fit."""
    if not spec:
        return None
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:]:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


# =============================================================================
# main
# =============================================================================

def run_group(feat, scene, fit_scenes, pca_dim, K, runner, whiten=True):
    """한 fit 그룹(= per-task 면 shard 하나, global 이면 전체) 처리.

    PCA·KMeans 모두 fit scene 행에서만 적합하고, 라벨은 전 행에 부여한다.
    (클러스터링은 outcome-blind 라 scene 전부로 fit 해도 하류 LOSO 는 유효하지만,
     scene 홀드아웃을 원하면 --fit-scenes 로 통제한다.)
    """
    n = len(feat)
    if fit_scenes is None or scene is None:
        mask = np.ones(n, bool)
    else:
        mask = np.isin(np.asarray(scene), sorted(fit_scenes))
        if mask.sum() < max(pca_dim + 1, K):
            raise SystemExit(f"fit 행 부족: {int(mask.sum())} (pca_dim={pca_dim}, k={K})")
    stats, evr = pca_fit(feat[mask], pca_dim)
    Z_all = pca_apply(feat, stats, whiten=whiten)
    labels, centroids = runner.fit_predict(Z_all[mask], Z_all, K)
    return labels, stats, evr, centroids, int(mask.sum())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shards-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--shards", default=None,
                    help="쉼표구분 instr_slug 부분집합 (기본: 디렉토리 전체)")
    ap.add_argument("--scope", choices=("per-task", "per-family", "global"),
                    default="global",
                    help="per-task: instruction 별 개별 fit (구명 유지 — 실제 단위는 "
                         "instruction) / per-family: task(env family) 단위 fit / "
                         "global: 전 shard 합쳐 1회")
    ap.add_argument("--k", default=None,
                    help="클러스터 수. 쉼표로 sweep 가능 (per-task 기본 6,8,12 / global 기본 24)")
    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--no-whiten", action="store_true")
    ap.add_argument("--fit-scenes", default=None,
                    help='fit 에 쓸 scene ("0-7" / "0,1,2"). 기본: 전 scene')
    ap.add_argument("--kmeans-impl", choices=("auto", "task_classification", "numpy"),
                    default="auto")
    ap.add_argument("--tc-root", type=Path, default=None,
                    help="task_classification 체크아웃 경로 override")
    ap.add_argument("--n-init", type=int, default=5)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--align-out", type=Path, default=None)
    ap.add_argument("--boundary-tol", type=int, default=1)
    ap.add_argument("--n-perm", type=int, default=300)
    args = ap.parse_args(argv)

    ks = ([int(x) for x in args.k.split(",") if x.strip()] if args.k
          else ({"per-task": [6, 8, 12], "per-family": [8, 16, 24, 40],
                 "global": [24]}[args.scope]))
    fit_scenes = parse_scene_spec(args.fit_scenes)
    args.out.mkdir(parents=True, exist_ok=True)

    runner = KMeansRunner(args.kmeans_impl, args.tc_root,
                          args.n_init, args.max_iter, args.seed)
    print(f"[kmeans] impl={runner.impl} source={runner.source}", flush=True)
    if runner.load_error:
        print(f"[kmeans] task_classification 로드 실패 → fallback: {runner.load_error}",
              flush=True)

    paths = discover_shards(args.shards_dir, args.shards.split(",") if args.shards else None)
    shards = []
    for p in paths:
        s = load_shard(p)
        print(f"[load] {s['name']}: n_rec={len(s['feat'])} "
              f"ep={len(np.unique(s['ep_id']))}", flush=True)
        shards.append(s)

    align_all = {"scope": args.scope, "pca_dim": args.pca_dim,
                 "fit_scenes": sorted(fit_scenes) if fit_scenes else None,
                 "kmeans_impl": runner.impl, "by_k": {}}

    for K in ks:
        labels_by_shard: dict[str, np.ndarray] = {}
        fit_arrays: dict[str, np.ndarray] = {}
        fit_rows: dict[str, int] = {}

        if args.scope in ("global", "per-family"):
            # per-family (2026-08-20): task = env family 단위 fit. SAFE failure
            # detector 가 task 단위이므로 phase 판독기도 같은 단위가 가능한지 비교용.
            # family = instruction 이름의 '_' 앞 접두 (PPCC_bread→PPCC,
            # OpenDrawer_left→OpenDrawer, 나머지는 단독 family).
            if args.scope == "global":
                groups = {"__global__": list(shards)}
            else:
                groups = {}
                for s in shards:
                    groups.setdefault(s["name"].split("_")[0], []).append(s)
            for g, members in sorted(groups.items()):
                feat = np.concatenate([s["feat"] for s in members], 0)
                scene = (np.concatenate([s["scene"] for s in members], 0)
                         if all(s["scene"] is not None for s in members) else None)
                lab, stats, evr, cent, nfit = run_group(
                    feat, scene, fit_scenes, args.pca_dim, K, runner,
                    not args.no_whiten)
                del feat
                off = 0
                for s in members:
                    m = len(s["feat"])
                    labels_by_shard[s["name"]] = lab[off:off + m].astype(np.int16)
                    off += m
                fit_arrays.update({f"{g}__pca_mu": stats["mu"], f"{g}__pca_V": stats["V"],
                                   f"{g}__pca_sqrt_lam": stats["sqrt_lam"],
                                   f"{g}__pca_evr": evr, f"{g}__centroids": cent})
                fit_rows[g] = nfit
        else:
            for s in shards:
                lab, stats, evr, cent, nfit = run_group(
                    s["feat"], s["scene"], fit_scenes, args.pca_dim, K, runner,
                    not args.no_whiten)
                labels_by_shard[s["name"]] = lab.astype(np.int16)
                g = s["name"]
                fit_arrays.update({f"{g}__pca_mu": stats["mu"], f"{g}__pca_V": stats["V"],
                                   f"{g}__pca_sqrt_lam": stats["sqrt_lam"],
                                   f"{g}__pca_evr": evr, f"{g}__centroids": cent})
                fit_rows[g] = nfit

        def_name = f"intrinsic_{args.scope}_k{K}"
        lab_path = args.out / f"{def_name}.npz"
        np.savez_compressed(lab_path, def_name=def_name, **labels_by_shard)
        np.savez_compressed(args.out / f"{def_name}_fit.npz", **fit_arrays)
        print(f"[write] {lab_path}", flush=True)

        # ---- 정렬 측정 ----
        if args.align_out:
            per = {}
            for s in shards:
                per[s["name"]] = align_metrics(
                    labels_by_shard[s["name"]], s["phase_code"], s["scene"],
                    s["ep_id"], s["rec_idx"], args.boundary_tol, args.n_perm, args.seed)
            # pooled: ep_id 가 shard 간 겹칠 수 있어 shard 별 offset 을 준다
            ep_cat, off = [], 0
            for s in shards:
                e = np.asarray(s["ep_id"]).astype(np.int64) + off
                ep_cat.append(e)
                off = int(e.max()) + 1
            pooled = align_metrics(
                np.concatenate([labels_by_shard[s["name"]] for s in shards]),
                np.concatenate([s["phase_code"] for s in shards])
                if all(s["phase_code"] is not None for s in shards) else None,
                np.concatenate([s["scene"] for s in shards])
                if all(s["scene"] is not None for s in shards) else None,
                np.concatenate(ep_cat),
                np.concatenate([s["rec_idx"] for s in shards]),
                args.boundary_tol, args.n_perm, args.seed)
            align_all["by_k"][str(K)] = {"per_shard": per, "pooled": pooled,
                                         "fit_rows": fit_rows}
            for name, m in list(per.items()) + [("__pooled__", pooled)]:
                print(f"  k={K:>3} {name:<28} MI={m.get('mi_phase_bits', float('nan')):.3f} "
                      f"margin={m.get('margin_bits', float('nan')):+.3f} "
                      f"purity={m.get('purity_phase', float('nan')):.3f} "
                      f"F1={m.get('boundary_f1', float('nan')):.3f} "
                      f"z={m.get('boundary_z', float('nan')):+.2f} "
                      f"mi_scene={m.get('mi_scene_bits', float('nan')):.3f}", flush=True)

    meta = {
        "script": "scripts/analysis/grid_phase/intrinsic_phase.py",
        "feature_spec": {"layer": 12, "layer_axis_index": LAYER_IDX,
                         "denoise": 3, "denoise_axis_index": DENOISE_IDX,
                         "segment": "all(49-token mean)",
                         "segment_axis_index": SEGMENT_IDX, "dim": 1536},
        "pipeline": ["PCA-%d%s" % (args.pca_dim, "" if args.no_whiten else " whitened"),
                     "KMeans(k=%s)" % ",".join(map(str, ks))],
        "deviations": [
            "동료 규격의 AE(latent 16) 단계를 생략 — PCA-64 → KMeans 직결. "
            "근거: docs/steering/40 §5 부록 (AE margin 1.692 vs 압축기 없음 1.634, "
            "seed 범위 겹침).",
            "clock 대조군의 상태 수는 발견 상태열의 상태 수에 맞춘다 (동료 "
            "structure.py: clock_clusters). GT 상태 수 기준 변형은 "
            "margin_gtstates_bits 로 병기.",
        ],
        "params": {"scope": args.scope, "k": ks, "pca_dim": args.pca_dim,
                   "whiten": not args.no_whiten,
                   "fit_scenes": sorted(fit_scenes) if fit_scenes else None,
                   "n_init": args.n_init, "max_iter": args.max_iter,
                   "seed": args.seed, "boundary_tol": args.boundary_tol,
                   "n_perm": args.n_perm},
        "kmeans": {"impl": runner.impl, "source": runner.source,
                   "load_error": runner.load_error},
        "shards": [s["name"] for s in shards],
        "numpy": np.__version__,
    }
    for K in ks:
        (args.out / f"intrinsic_{args.scope}_k{K}_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.align_out:
        args.align_out.parent.mkdir(parents=True, exist_ok=True)
        align_all["meta"] = meta
        args.align_out.write_text(json.dumps(align_all, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
        print(f"[write] {args.align_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

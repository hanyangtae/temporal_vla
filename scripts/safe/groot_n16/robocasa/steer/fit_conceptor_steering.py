#!/usr/bin/env python3
"""Fit COAST contrastive conceptor steering matrices from GR00T N1.6 rollout features.

COAST App. A.9 (Eq. 4) + A.10.2 (Efficiently Selecting Hyperparameters):
  성공/실패 rollout 의 pre-velocity hidden state 분포에서 contrastive conceptor
  ``C_steer = C_success AND NOT C_failure`` 를 fit 하고, COAST 의 3-stage 절차로
  하이퍼파라미터를 선택한다.

  - Stage 1 (layer): quota(C_steer) at alpha0=10. **단일 layer 데이터라 layer 선택은
    skip**, quota 는 진단값으로 기록(Eq. 10).
  - Stage 2 (aperture): alpha sweep 후 mean overlap(C_s, C_f) 가 sweet-spot band
    ``[0.85, 0.95]`` 에 드는 alpha retain, 없으면 가장 가까운 것(Eq. 11).
  - Stage 3 (strength beta): rollout 필요 단계라 굽지 않고 ``{0.1, 0.3}`` 권장값만 기록.

시간축(env step) 집계 모드 — seen18 의 길이 confound(실패=timeout 45 / 성공=조기종료)
때문에 3가지를 모두 fit 해 비교한다:
  - ``coast``        : per-timestep, 전 step (COAST 논문 그대로).
  - ``episode_mean`` : rollout 당 평균 1벡터 (최대 길이 confound, 아티팩트 천장 reference).
  - ``truncated``    : per-timestep, 첫 ``--max-len`` step 만 (길이 통제 = 성공/실패를 같은
                       early temporal support 에서 샘플 → genuine signal).

Fit space:
  horizon/diff 둘 다 ``mean`` pool 한 ``z in R^1024`` (DiT pre-velocity hidden state).
  M=(1-β)I+β·C_steer 가 [1024,1024] 라 inference 시 ``[K,H,D]`` 모든 (k,h) slice 에 동일
  적용 가능 → fit space = steer space (A.9.2). 위 시간축 모드는 K/H pool 과 직교.

입력: ``--cache`` (cache_pooled_features.py 산출 npz) 또는 ``--split-root`` (raw pkl).

산출물 (mode 당, group=global / ``--per-task`` 시 task 당):
  ``<out>/<mode>/<group>/conceptors.npz``  — 선택 alpha 별 C_steer/C_success/C_failure.
  ``<out>/<mode>/<group>/metadata.json``   — sweep·선택 alpha·quota·권장 beta·step3 config.

실행 (host, conda activate hyundai_aigs):
  # DiT 기존 방식
  python scripts/safe/groot_n16/robocasa/steer/fit_conceptor_steering.py \
      --cache outputs/.../analysis/feature_cache/pooled_all_hmean_dmean.npz --per-task

  # VL pathway (action_head.vlln, D=2048)
  python scripts/safe/groot_n16/robocasa/steer/fit_conceptor_steering.py \
      --rollout-dir outputs/.../target_atomic_moderate10_pathway_pertoken_100ep/raw_rollouts \
      --pathway vl --agg-mode truncated --per-task
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROBOCASA_SAFE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROBOCASA_SAFE_ROOT))

from run_config import REPO_ROOT  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))

from safe_feature_vectors import (  # noqa: E402
    load_manifest,
    parse_aggregation_command,
    pooled_hidden_states,
)
from src.conceptor import (  # noqa: E402
    and_conceptor,
    as_float32,
    compute_conceptor,
    conceptor_overlap,
    conceptor_quota,
    eigenvalue_spectrum,
    not_conceptor,
)

DEFAULT_SPLIT_ROOT = (
    REPO_ROOT
    / "outputs/eval/robocasa/groot_n16/safe_split_seen4_unseen2_openDrawer_pnpCab_100ep"
)
# COAST Table 14 — GR00T RoboCasa aperture grid (max 10).
DEFAULT_ALPHAS = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
# COAST A.10.2 Stage 2 — overlap sweet-spot band.
OVERLAP_BAND = (0.85, 0.95)
# COAST Eq. 10 / Table 18 — Stage-1 quota diagnostic at the sharpest aperture.
LAYER_QUOTA_ALPHA = 10.0
# COAST A.10.2 Stage 3 — keep these, drop 0.5 (harmful).
RECOMMENDED_BETAS = [0.1, 0.3]
# 시간축 집계 모드.
DEFAULT_AGG_MODES = ["coast", "episode_mean", "truncated"]
# truncated 모드의 기본 최대 길이(첫 W env step). success p10≈12 이하로 잡아 대부분
# rollout 이 W step 을 채우게 한다(길이 통제).
DEFAULT_MAX_LEN = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="cache_pooled_features.py 산출 npz. 주면 raw pkl 대신 캐시에서 로드.",
    )
    parser.add_argument(
        "--select-layer",
        action="store_true",
        help="COAST Stage1: multi-layer 수집(--rollout-dir)에서 layer별 quota로 ℓ* 선택 후 fit.",
    )
    parser.add_argument(
        "--force-layer",
        type=int,
        default=None,
        help="--select-layer 와 함께 사용. 지정 시 Stage1 sweep 을 건너뛰고 해당 layer 에서 직접 fit.",
    )
    parser.add_argument(
        "--token-window",
        choices=("coast49", "valid16", "all50", "T_full"),
        default=None,
        help=(
            "per-T 수집 데이터([L,T,D]) 에서 token 축 reduce 방법. "
            "coast49=마지막 49 token (COAST p24 모방), valid16=마지막 50 중 첫 16 (action), "
            "all50=마지막 50 (model_action_horizon), T_full=T 전체. "
            "이미 [L,D] 인 데이터는 무시."
        ),
    )
    parser.add_argument(
        "--rollout-dir",
        type=Path,
        default=None,
        help=(
            "--select-layer 용 multi-layer raw_rollouts 디렉토리(per-step hidden_state=[L,D]). "
            "--pathway vl 시에도 사용 (vl_hidden_states 로드)."
        ),
    )
    parser.add_argument(
        "--pathway",
        choices=("dit", "vl"),
        default="dit",
        help=(
            "Feature pathway. "
            "dit=DiT motor (기존 동작, hidden_states 키). "
            "vl=goal VL (action_head.vlln seq-mean-pool, D=2048, vl_hidden_states 키)."
        ),
    )
    parser.add_argument(
        "--instruction-filter",
        default=None,
        metavar="SUBSTR",
        help=(
            "--pathway vl 전용. ep_meta.lang 에 해당 substring(대소문자 무시) 포함 rollout 만 "
            "사용해 instruction-pure conceptor fit. 예: 'slide in', 'right'."
        ),
    )
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument(
        "--scope",
        default="all",
        choices=("all", "train", "val_seen", "val_unseen"),
        help="raw pkl 경로일 때 사용할 manifest split.",
    )
    parser.add_argument("--horizon-idx-rel", default="mean")
    parser.add_argument("--diff-idx-rel", default="mean")
    parser.add_argument(
        "--agg-mode",
        nargs="+",
        default=None,
        choices=("coast", "episode_mean", "truncated"),
        help=f"시간축 집계 모드(복수 가능). 생략 시 {DEFAULT_AGG_MODES}.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        nargs="+",
        default=None,
        help="truncated 모드의 최대 env step(복수 가능). 생략 시 success 길이 [mean, mean+1σ] 자동.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        default=None,
        help=f"aperture sweep 그리드. 생략 시 COAST GR00T RoboCasa 그리드 {DEFAULT_ALPHAS}.",
    )
    parser.add_argument(
        "--overlap-band",
        type=float,
        nargs=2,
        default=list(OVERLAP_BAND),
        metavar=("LO", "HI"),
        help=f"Stage 2 sweet-spot band. 기본 {OVERLAP_BAND}.",
    )
    parser.add_argument(
        "--per-task",
        action="store_true",
        help="global 외에 task_id 별 conceptor 도 fit (mode-conditional).",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--max-rollouts",
        type=int,
        default=None,
        help="raw pkl 경로 smoke 용. scope 앞 N rollout 만.",
    )
    args = parser.parse_args()
    if args.alpha is None:
        args.alpha = list(DEFAULT_ALPHAS)
    if args.agg_mode is None:
        args.agg_mode = list(DEFAULT_AGG_MODES)
    return args


def resolve_source_path(source_path: str, repo_root: Path) -> Path:
    """manifest 의 절대 source_path 를 현재 repo root 기준으로 재해석(pdk_ws→pkt_ws 등)."""
    original = Path(source_path)
    if original.exists():
        return original
    parts = original.parts
    if "outputs" in parts:
        candidate = repo_root / Path(*parts[parts.index("outputs") :])
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"source_path 없음: {source_path} (repo_root={repo_root})")


def load_pooled_from_cache(cache_path: Path) -> dict[str, Any]:
    """cache_pooled_features.py npz 에서 per-timestep + episode-mean 배열 로드."""
    z = np.load(cache_path, allow_pickle=True)
    id2name = {int(t): str(n) for t, n in zip(z["ep_task_id"], z["task_names"])}
    return {
        "feats": z["feats"].astype(np.float64, copy=False),
        "success": z["success"].astype(np.int64),
        "task_id": z["task_id"].astype(np.int64),
        "step_idx": z["step_idx"].astype(np.int64),
        "ep_feat_mean": z["ep_feat_mean"].astype(np.float64, copy=False),
        "ep_success": z["ep_success"].astype(np.int64),
        "ep_task_id": z["ep_task_id"].astype(np.int64),
        "ep_len": z["ep_len"].astype(np.int64),
        "id2name": id2name,
        "source": str(cache_path),
        "n_rollouts": int(z["ep_success"].shape[0]),
    }


def load_pooled_from_pkls(
    split_root: Path,
    scope: str,
    *,
    horizon_idx_rel: str,
    diff_idx_rel: str,
    max_rollouts: int | None,
) -> dict[str, Any]:
    """raw pkl 에서 per-timestep + episode-mean 배열 로드 (캐시 없을 때)."""
    horizon_cmd = parse_aggregation_command(horizon_idx_rel)
    diff_cmd = parse_aggregation_command(diff_idx_rel)
    rows = load_manifest(split_root, scope)
    if max_rollouts is not None:
        rows = rows[:max_rollouts]

    feats, success, task_id, step_idx = [], [], [], []
    ep_feat_mean, ep_success, ep_task_id, ep_len = [], [], [], []
    id2name: dict[int, str] = {}
    for row in rows:
        path = resolve_source_path(row["source_path"], REPO_ROOT)
        with path.open("rb") as f:
            record = pickle.load(f)
        z = pooled_hidden_states(record, horizon_idx_rel=horizon_cmd, diff_idx_rel=diff_cmd)
        n = z.shape[0]
        if n == 0:
            continue
        tid, succ = int(row["task_id"]), int(row["success"])
        id2name[tid] = row["task"]
        feats.append(z.astype(np.float64))
        success.append(np.full(n, succ, dtype=np.int64))
        task_id.append(np.full(n, tid, dtype=np.int64))
        step_idx.append(np.arange(n, dtype=np.int64))
        ep_feat_mean.append(z.mean(axis=0).astype(np.float64))
        ep_success.append(succ)
        ep_task_id.append(tid)
        ep_len.append(n)

    if not feats:
        raise ValueError(f"No features for scope={scope}.")
    return {
        "feats": np.concatenate(feats),
        "success": np.concatenate(success),
        "task_id": np.concatenate(task_id),
        "step_idx": np.concatenate(step_idx),
        "ep_feat_mean": np.stack(ep_feat_mean),
        "ep_success": np.asarray(ep_success, dtype=np.int64),
        "ep_task_id": np.asarray(ep_task_id, dtype=np.int64),
        "ep_len": np.asarray(ep_len, dtype=np.int64),
        "id2name": id2name,
        "source": f"{split_root}#{scope}",
        "n_rollouts": len(ep_success),
    }


def derive_mode(
    data: dict[str, Any], mode: str, max_len: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """집계 모드별 (X, success, task_id) 반환."""
    if mode == "coast":
        return data["feats"], data["success"], data["task_id"]
    if mode == "episode_mean":
        return data["ep_feat_mean"], data["ep_success"], data["ep_task_id"]
    if mode == "truncated":
        m = data["step_idx"] < max_len
        return data["feats"][m], data["success"][m], data["task_id"][m]
    raise ValueError(f"unknown agg mode: {mode}")


def select_alphas_in_band(
    sweep: list[dict[str, float]], band: tuple[float, float]
) -> tuple[list[float], str]:
    """COAST A.10.2 Stage 2 — overlap 이 band 안인 alpha retain, 없으면 가장 가까운 것."""
    lo, hi = band
    in_band = sorted(r["alpha"] for r in sweep if lo <= r["overlap"] <= hi)
    if in_band:
        return in_band, "band"

    def band_distance(row: dict[str, float]) -> float:
        o = row["overlap"]
        return 0.0 if lo <= o <= hi else min(abs(o - lo), abs(o - hi))

    return [min(sweep, key=band_distance)["alpha"]], "closest"


def fit_group(
    X_success: np.ndarray,
    X_failure: np.ndarray,
    alphas: list[float],
    band: tuple[float, float],
) -> tuple[dict[float, dict[str, Any]], dict[str, Any]]:
    """한 그룹(global 또는 task)에 alpha sweep + band 선택 + contrastive fit."""
    cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    sweep: list[dict[str, float]] = []
    for alpha in alphas:
        C_s = compute_conceptor(X_success, alpha)
        C_f = compute_conceptor(X_failure, alpha)
        cache[alpha] = (C_s, C_f)
        sweep.append(
            {
                "alpha": float(alpha),
                "overlap": conceptor_overlap(C_s, C_f),
                "quota_success": conceptor_quota(C_s),
                "quota_failure": conceptor_quota(C_f),
            }
        )

    selected, mode = select_alphas_in_band(sweep, band)

    if LAYER_QUOTA_ALPHA in cache:
        Cs0, Cf0 = cache[LAYER_QUOTA_ALPHA]
    else:
        Cs0 = compute_conceptor(X_success, LAYER_QUOTA_ALPHA)
        Cf0 = compute_conceptor(X_failure, LAYER_QUOTA_ALPHA)
    layer_quota = conceptor_quota(and_conceptor(Cs0, not_conceptor(Cf0)))

    fits: dict[float, dict[str, Any]] = {}
    per_alpha: dict[str, Any] = {}
    for alpha in selected:
        C_s, C_f = cache[alpha]
        C_steer = and_conceptor(C_s, not_conceptor(C_f))
        fits[alpha] = {"C_success": C_s, "C_failure": C_f, "C_steer": C_steer}
        per_alpha[f"{alpha:g}"] = {
            "overlap": conceptor_overlap(C_s, C_f),
            "quota_steer": conceptor_quota(C_steer),
            "spectrum_top20": eigenvalue_spectrum(C_steer)[:20].tolist(),
        }

    metadata = {
        "selected_alphas": [float(a) for a in selected],
        "selection_mode": mode,
        "overlap_band": list(band),
        "layer_quota_contrastive_at_alpha10": layer_quota,
        "recommended_betas": RECOMMENDED_BETAS,
        "step3_configs": [
            {"alpha": float(a), "beta": b} for a in selected for b in RECOMMENDED_BETAS
        ],
        "alpha_sweep": sweep,
        "per_alpha": per_alpha,
        "n_success": int(X_success.shape[0]),
        "n_failure": int(X_failure.shape[0]),
        "feature_dim": int(X_success.shape[1]),
    }
    return fits, metadata


def save_group(
    out_dir: Path,
    fits: dict[float, dict[str, Any]],
    metadata: dict[str, Any],
    base_meta: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for alpha, group in fits.items():
        key = f"alpha{alpha:g}"
        arrays[f"{key}_C_steer"] = as_float32(group["C_steer"])
        arrays[f"{key}_C_success"] = as_float32(group["C_success"])
        arrays[f"{key}_C_failure"] = as_float32(group["C_failure"])
    np.savez_compressed(out_dir / "conceptors.npz", **arrays)
    with (out_dir / "metadata.json").open("w") as f:
        json.dump({**base_meta, **metadata}, f, indent=2, ensure_ascii=False)


def fit_and_save_mode(
    data: dict[str, Any],
    mode: str,
    *,
    max_len: int,
    alphas: list[float],
    band: tuple[float, float],
    per_task: bool,
    out_root: Path,
    base_meta: dict[str, Any],
) -> None:
    """한 집계 모드에 대해 global + (옵션) per-task conceptor fit & 저장."""
    X, succ, tid = derive_mode(data, mode, max_len)
    mode_tag = f"truncated_w{max_len}" if mode == "truncated" else mode
    out_dir = out_root / mode_tag
    id2name = data["id2name"]
    s_mask = succ == 1
    mode_meta = {**base_meta, "agg_mode": mode_tag, "max_len": max_len if mode == "truncated" else None}
    print(
        f"[{mode_tag}] X={X.shape} success={int(s_mask.sum())} failure={int((~s_mask).sum())}"
    )

    def report(tag: str, m: dict[str, Any]) -> None:
        per = ", ".join(
            f"α{a}:ov={m['per_alpha'][f'{float(a):g}']['overlap']:.3f}"
            f"/q={m['per_alpha'][f'{float(a):g}']['quota_steer']:.3f}"
            for a in m["selected_alphas"]
        )
        print(
            f"  [{tag}] α={m['selected_alphas']} ({m['selection_mode']}) "
            f"lq(α10)={m['layer_quota_contrastive_at_alpha10']:.4f} | {per}"
        )

    fits, m = fit_group(X[s_mask], X[~s_mask], alphas, band)
    save_group(out_dir / "global", fits, m, mode_meta)
    report("global", m)

    if per_task:
        for t in sorted(np.unique(tid).tolist()):
            tm = tid == t
            ts, tf = tm & s_mask, tm & ~s_mask
            name = id2name.get(int(t), f"task{t}")
            if ts.sum() < 2 or tf.sum() < 2:
                print(f"  [task {t} {name}] skip — succ={int(ts.sum())} fail={int(tf.sum())}")
                continue
            fits, m = fit_group(X[ts], X[tf], alphas, band)
            save_group(out_dir / f"task_{t}_{name}", fits, m, {**mode_meta, "task_id": int(t), "task_name": name})
            report(f"task {t} {name}", m)


def _to_numpy(h: Any) -> np.ndarray:
    return h.detach().cpu().numpy() if hasattr(h, "detach") else np.asarray(h)


def _reduce_token_axis(arr: np.ndarray, window: str | None) -> np.ndarray:
    """per-step [L,T,D] (4D) 데이터에 token-window 적용 → [L,D].

    window=None 이고 4D 면 오류. 3D ([L,D]) 면 그대로 통과.
    layer_select_compare.apply_token_window 와 동일한 슬라이싱 규칙.
    """
    if arr.ndim == 3:
        return arr  # [N_step, L, D]
    if arr.ndim != 4:
        raise ValueError(f"Unexpected ndim={arr.ndim}, shape={arr.shape}")
    if window is None:
        raise ValueError("4D per-T data needs --token-window {coast49,valid16,all50,T_full}")
    T = arr.shape[2]
    model_h, valid_h = 50, 16  # GR00T N1.6 RoboCasa 기본값
    if window == "T_full":
        sub = arr
    elif window == "all50":
        sub = arr[:, :, -min(model_h, T) :, :]
    elif window == "valid16":
        start = max(0, T - model_h)
        sub = arr[:, :, start : start + valid_h, :]
    elif window == "coast49":
        sub = arr[:, :, -min(49, T) :, :]
    else:
        raise ValueError(f"Unknown token-window: {window}")
    return sub.mean(axis=2)  # [N_step, L, D]


def load_multilayer_from_rollouts(
    rollout_dir: Path, token_window: str | None = None
) -> dict[str, Any]:
    """multi-layer 수집 pkl(per-step hidden_state=[L,D] 또는 [L,T,D])에서 feats[N,L,D] + 라벨 로드.

    collect_multilayer_parallel.sh 산출 raw_rollouts/<task>/*.pkl 을 직접 glob.
    task_id 는 pkl 필드(global), task 이름은 상위 디렉토리명.
    per-T 데이터(`[L,T,D]`)면 token_window 적용 후 [L,D] 로 reduce.
    """
    files = sorted(rollout_dir.glob("*/*.pkl"))
    if not files:
        raise ValueError(f"No pkl under {rollout_dir}")
    feats, success, task_id, step_idx = [], [], [], []
    ep_feat_mean, ep_success, ep_task_id, ep_len = [], [], [], []
    id2name: dict[int, str] = {}
    layer_indices: list[int] | None = None
    for f in files:
        with f.open("rb") as fh:
            rec = pickle.load(fh)
        hs = rec.get("hidden_states") or []
        if not hs:
            continue
        arr = np.stack([_to_numpy(h).astype(np.float64) for h in hs])
        # [N_step, L, D] 또는 [N_step, L, T, D]
        arr = _reduce_token_axis(arr, token_window)
        if arr.ndim != 3:
            raise ValueError(f"After window reduce expected 3D, got {arr.shape}")
        n = arr.shape[0]
        tid, succ = int(rec["task_id"]), int(rec.get("episode_success", 0))
        id2name[tid] = f.parent.name
        layer_indices = rec.get("layer_indices", layer_indices)
        feats.append(arr)
        success.append(np.full(n, succ, dtype=np.int64))
        task_id.append(np.full(n, tid, dtype=np.int64))
        step_idx.append(np.arange(n, dtype=np.int64))
        ep_feat_mean.append(arr.mean(axis=0))  # [L, D]
        ep_success.append(succ)
        ep_task_id.append(tid)
        ep_len.append(n)
    if not feats:
        raise ValueError(f"No non-empty rollouts under {rollout_dir}")
    n_layers = feats[0].shape[1]
    return {
        "feats": np.concatenate(feats),  # [N, L, D]
        "success": np.concatenate(success),
        "task_id": np.concatenate(task_id),
        "step_idx": np.concatenate(step_idx),
        "ep_feat_mean": np.stack(ep_feat_mean),  # [R, L, D]
        "ep_success": np.asarray(ep_success, dtype=np.int64),
        "ep_task_id": np.asarray(ep_task_id, dtype=np.int64),
        "ep_len": np.asarray(ep_len, dtype=np.int64),
        "id2name": id2name,
        "layer_indices": list(layer_indices) if layer_indices is not None else list(range(n_layers)),
        "source": str(rollout_dir),
        "n_rollouts": len(files),
    }


def load_vl_from_rollouts(
    rollout_dir: Path, instruction_filter: str | None = None
) -> dict[str, Any]:
    """VL pathway (vl_hidden_states) pkl → feats[N, 2048] + 라벨 로드.

    collect_pathway_parallel.sh 산출 raw_rollouts/<task>/*.pkl 의
    ``vl_hidden_states`` 키(list of per-step [2048])를 읽는다.
    pathway_separation.py / vl_dit_lda_analysis.py 와 동일한 로드 패턴.

    instruction_filter 지정 시 ``rec["ep_meta"]["lang"]`` 에 해당 substring
    (대소문자 무시)이 포함된 rollout 만 사용 (instruction-pure conceptor fit).
    """
    files = sorted(rollout_dir.glob("*/*.pkl"))
    if not files:
        raise ValueError(f"No pkl under {rollout_dir}")
    feats, success, task_id, step_idx = [], [], [], []
    ep_feat_mean, ep_success, ep_task_id, ep_len = [], [], [], []
    id2name: dict[int, str] = {}
    n_filtered = 0
    for f in files:
        with f.open("rb") as fh:
            rec = pickle.load(fh)
        if instruction_filter:
            lang = rec.get("ep_meta", {}).get("lang", "")
            if instruction_filter.lower() not in lang.lower():
                n_filtered += 1
                continue
        vl = rec.get("vl_hidden_states")
        if not vl:
            continue
        arr = np.stack([np.asarray(v, dtype=np.float64) for v in vl])  # [n_steps, 2048]
        n = arr.shape[0]
        if n == 0:
            continue
        tid, succ = int(rec["task_id"]), int(rec.get("episode_success", 0))
        id2name[tid] = f.parent.name
        feats.append(arr)
        success.append(np.full(n, succ, dtype=np.int64))
        task_id.append(np.full(n, tid, dtype=np.int64))
        step_idx.append(np.arange(n, dtype=np.int64))
        ep_feat_mean.append(arr.mean(axis=0))
        ep_success.append(succ)
        ep_task_id.append(tid)
        ep_len.append(n)
    if not feats:
        raise ValueError(
            f"No VL features under {rollout_dir}"
            + (f" (instruction_filter={instruction_filter!r}, {n_filtered} filtered out)"
               if instruction_filter else "")
        )
    if instruction_filter:
        print(f"[vl-load] instruction_filter={instruction_filter!r} kept {len(ep_success)} rollouts, filtered {n_filtered}")
    return {
        "feats": np.concatenate(feats),           # [N, 2048]
        "success": np.concatenate(success),
        "task_id": np.concatenate(task_id),
        "step_idx": np.concatenate(step_idx),
        "ep_feat_mean": np.stack(ep_feat_mean),   # [R, 2048]
        "ep_success": np.asarray(ep_success, dtype=np.int64),
        "ep_task_id": np.asarray(ep_task_id, dtype=np.int64),
        "ep_len": np.asarray(ep_len, dtype=np.int64),
        "id2name": id2name,
        "source": str(rollout_dir),
        "n_rollouts": len(ep_success),
        "instruction_filter": instruction_filter,
    }


def _layer_view(data: dict[str, Any], li: int) -> dict[str, Any]:
    """multi-layer data 에서 layer-pos li 의 single-layer data 뷰 (derive_mode 호환)."""
    return {**data, "feats": data["feats"][:, li, :], "ep_feat_mean": data["ep_feat_mean"][:, li, :]}


def select_and_fit_layer(args: argparse.Namespace, band: tuple[float, float]) -> None:
    """COAST A.10.2 Stage1: layer별 contrastive quota(@α0=10, 길이 통제) → argmax → ℓ*.

    이후 ℓ* feature 로 표준 Stage2/3 fit(per-task)을 수행해 steering NPZ 산출.
    """
    data = load_multilayer_from_rollouts(args.rollout_dir, token_window=args.token_window)
    n_layers = data["feats"].shape[1]
    feat_dim = data["feats"].shape[2]
    layer_idx = data["layer_indices"]
    out_root = args.out_dir or (args.rollout_dir.parent / "conceptor_steering")

    if args.max_len:
        W = int(args.max_len[0])
    else:
        slen = data["ep_len"][data["ep_success"] == 1]
        W = int(round(float(slen.mean())))
    print(
        f"[select-layer] rollouts={data['n_rollouts']} N={data['feats'].shape[0]} "
        f"layers={n_layers} dim={feat_dim} truncated_W={W} (길이 통제)"
    )

    # --force-layer: Stage1 sweep 을 건너뛰고 사용자가 지정한 layer 에서 직접 fit.
    if getattr(args, "force_layer", None) is not None:
        forced = int(args.force_layer)
        if forced not in list(layer_idx):
            raise SystemExit(f"--force-layer {forced} 가 layer_indices 에 없음: {layer_idx}")
        layer_pos = list(layer_idx).index(forced)
        best = {"layer": forced, "layer_pos": layer_pos, "quota_bar": None, "overlap_bar": None}
        # token-window 정보를 출력 경로에 포함 → 같은 layer × 다른 window 충돌 방지.
        win_tag = f"_{args.token_window}" if args.token_window else ""
        layer_dir = f"layer{best['layer']}{win_tag}"
        print(
            f"[select-layer] --force-layer {forced} (pos {layer_pos}) "
            f"window={args.token_window or 'pre-pooled'} → sweep 생략"
        )
        base_meta = {
            "selected_layer": best["layer"],
            "forced_layer": True,
            "token_window": args.token_window,
            "alpha_grid": args.alpha,
            "source": data["source"],
            "repo_root": str(REPO_ROOT),
            "created": datetime.now().isoformat(timespec="seconds"),
        }
        fit_and_save_mode(
            _layer_view(data, best["layer_pos"]),
            "truncated",
            max_len=W,
            alphas=args.alpha,
            band=band,
            per_task=args.per_task,
            out_root=(args.out_dir or (args.rollout_dir.parent / "conceptor_steering"))
            / layer_dir,
            base_meta=base_meta,
        )
        print(f"[done] steering conceptors -> {out_root}/{layer_dir}/")
        return

    sweep: list[dict[str, Any]] = []
    for li in range(n_layers):
        X, succ, tid = derive_mode(_layer_view(data, li), "truncated", W)
        s = succ == 1
        qs, ovs, q_per_task = [], [], {}
        for t in sorted(np.unique(tid).tolist()):
            tm = tid == t
            ts, tf = tm & s, tm & ~s
            if ts.sum() < 2 or tf.sum() < 2:
                continue
            C_s = compute_conceptor(X[ts], LAYER_QUOTA_ALPHA)
            C_f = compute_conceptor(X[tf], LAYER_QUOTA_ALPHA)
            q = conceptor_quota(and_conceptor(C_s, not_conceptor(C_f)))
            qs.append(q)
            ovs.append(conceptor_overlap(C_s, C_f))
            q_per_task[data["id2name"].get(int(t), str(t))] = q
        sweep.append(
            {
                "layer": int(layer_idx[li]),
                "layer_pos": li,
                "quota_bar": float(np.mean(qs)) if qs else 0.0,
                "overlap_bar": float(np.mean(ovs)) if ovs else 1.0,
                "n_tasks": len(qs),
                "quota_per_task": q_per_task,
            }
        )
        print(
            f"  layer {int(layer_idx[li]):>2}: q̄={sweep[-1]['quota_bar']:.4f} "
            f"ov̄={sweep[-1]['overlap_bar']:.3f} (n_task={len(qs)})"
        )

    best = max(sweep, key=lambda r: r["quota_bar"])
    sel_dir = out_root / "layer_selection"
    sel_dir.mkdir(parents=True, exist_ok=True)
    with (sel_dir / "stage1_quota.json").open("w") as f:
        json.dump(
            {
                "alpha0": LAYER_QUOTA_ALPHA,
                "max_len": W,
                "selected_layer": best["layer"],
                "selected_layer_pos": best["layer_pos"],
                "n_layers": n_layers,
                "feature_dim": feat_dim,
                "sweep": sweep,
                "source": data["source"],
                "created": datetime.now().isoformat(timespec="seconds"),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(
        f"[select-layer] ℓ*={best['layer']} (pos {best['layer_pos']}) "
        f"q̄={best['quota_bar']:.4f} ov̄={best['overlap_bar']:.3f} -> {sel_dir}/stage1_quota.json"
    )

    base_meta = {
        "selected_layer": best["layer"],
        "alpha_grid": args.alpha,
        "source": data["source"],
        "repo_root": str(REPO_ROOT),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    fit_and_save_mode(
        _layer_view(data, best["layer_pos"]),
        "truncated",
        max_len=W,
        alphas=args.alpha,
        band=band,
        per_task=args.per_task,
        out_root=out_root / f"layer{best['layer']}",
        base_meta=base_meta,
    )
    print(f"[done] steering conceptors -> {out_root}/layer{best['layer']}/")


def main() -> None:
    args = parse_args()
    band = (args.overlap_band[0], args.overlap_band[1])

    if args.instruction_filter and args.pathway != "vl":
        raise SystemExit("--instruction-filter 는 --pathway vl 에서만 지원.")

    if args.select_layer:
        if args.rollout_dir is None:
            raise SystemExit("--select-layer 는 --rollout-dir (multi-layer raw_rollouts) 필요")
        select_and_fit_layer(args, band)
        return

    if args.pathway == "vl":
        if args.rollout_dir is None:
            raise SystemExit("--pathway vl 은 --rollout-dir (raw_rollouts 디렉토리) 필요")
        data = load_vl_from_rollouts(args.rollout_dir, instruction_filter=args.instruction_filter)
        default_out = args.rollout_dir.parent / "analysis" / "conceptor_steering_vl"
        if args.instruction_filter:
            tag = re.sub(r"[^a-z0-9]+", "_", args.instruction_filter.lower()).strip("_")
            default_out = default_out / f"instr_{tag}"
    elif args.cache is not None:
        data = load_pooled_from_cache(args.cache)
        default_out = args.cache.parent.parent / "conceptor_steering"
    else:
        data = load_pooled_from_pkls(
            args.split_root,
            args.scope,
            horizon_idx_rel=args.horizon_idx_rel,
            diff_idx_rel=args.diff_idx_rel,
            max_rollouts=args.max_rollouts,
        )
        default_out = args.split_root.parent / "conceptor_steering"
    out_root = args.out_dir or default_out

    print(
        f"[load] source={data['source']} rollouts={data['n_rollouts']} "
        f"timesteps={data['feats'].shape[0]} dim={data['feats'].shape[1]} tasks={len(data['id2name'])}"
    )

    # truncated 의 max-len: 미지정 시 success 길이 [mean, mean+1σ] 자동 (표준 기준).
    max_lens = args.max_len
    if "truncated" in args.agg_mode and max_lens is None:
        slen = data["ep_len"][data["ep_success"] == 1]
        mean, std = float(slen.mean()), float(slen.std())
        max_lens = sorted({int(round(mean)), int(round(mean + std))})
        print(
            f"[truncated] success len mean={mean:.2f} std={std:.2f} "
            f"-> W={max_lens} (mean, mean+1σ)"
        )
    elif max_lens is None:
        max_lens = [DEFAULT_MAX_LEN]

    base_meta = {
        "pathway": args.pathway,
        "instruction_filter": args.instruction_filter,
        "horizon_idx_rel": args.horizon_idx_rel,
        "diff_idx_rel": args.diff_idx_rel,
        "alpha_grid": args.alpha,
        "source": data["source"],
        "repo_root": str(REPO_ROOT),
        "created": datetime.now().isoformat(timespec="seconds"),
    }

    for mode in args.agg_mode:
        widths = max_lens if mode == "truncated" else [max_lens[0]]
        for w in widths:
            fit_and_save_mode(
                data,
                mode,
                max_len=w,
                alphas=args.alpha,
                band=band,
                per_task=args.per_task,
                out_root=out_root,
                base_meta=base_meta,
            )
    print(f"[done] artifacts -> {out_root}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""COAST Stage1 layer 선택을 두 방식으로 나란히 비교.

각 DiT transformer_block ℓ 마다 per-task contrastive conceptor quota
``q̄(ℓ) = mean_t quota(C_s ∧ ¬C_f) @α0=10`` 를 두 집계 방식으로 계산하고,
방식별 argmax ℓ* 를 보고한다 (rollout 0회, geometric).

방식
----
``coast``
    COAST A.7 원형. task t 의 성공/실패 step 을 전부 사용(절단·균형 없음).
    실패가 항상 timeout(길이 45)·성공이 조기종료(짧음)라 길이 confound 가 섞인다.

``balanced``
    길이/표본수 confound 통제. task t 마다:
      1. W_t = p10(성공 episode 길이)  (성공 길이 10퍼센타일, 반올림)
      2. 길이 < W_t 인 성공 episode 는 제거(drop)
      3. 남은 성공 + 실패 episode 를 첫 W_t step 으로 절단(truncate)
      4. 실패 episode 수를 "남은 성공 episode 수"에 맞춰 subsample → N_succ = N_fail
    → task 안에서 성공/실패가 같은 길이(W_t)·같은 episode 수가 되어, 분리 신호가
    길이/표본수 아티팩트가 아닌 genuine 차이임을 보장.

집계는 둘 다 per-task conceptor (COAST 와 동일). quota 가 큰 layer = 성공이 실패 부분공간
밖으로 가장 많이 벗어나는 layer → steering 후보 ℓ*.

출력
----
``<rollout_dir>/../conceptor_steering/layer_selection/stage1_quota_compare.json``
  {method: {selected_layer, selected_layer_pos, sweep:[{layer, quota_bar, overlap_bar,
            n_tasks, quota_per_task}], per_task_balance(balanced 만)}}

CPU 는 OMP/OPENBLAS/MKL/NUMEXPR 환경변수로 cap (기본 8). ``--dry-run`` 은 conceptor 연산
없이 방식별 per-task W/N 만 출력하고 종료(저비용 bookkeeping 검증용).
"""

from __future__ import annotations

import argparse
import os

# numpy import 전에 BLAS thread cap (공유 머신 CPU 예산 보호).
_THREADS = os.environ.get("LAYER_SELECT_THREADS", "8")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)

import glob
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.conceptor import (  # noqa: E402
    and_conceptor,
    compute_conceptor,
    conceptor_overlap,
    conceptor_quota,
    not_conceptor,
)

LAYER_QUOTA_ALPHA = 10.0
SUCCESS_PERCENTILE = 10.0  # balanced W_t = p10(success lengths)
MIN_PER_GROUP = 2  # task 당 성공/실패 episode 최소 수 (conceptor fit 가능 조건)
DEFAULT_MODEL_ACTION_HORIZON = 50  # GR00T N1.6 RoboCasa
DEFAULT_VALID_ACTION_HORIZON = 16
COAST_S = 49  # COAST p24 명세: 1 state + 32 future + 16 action
TOKEN_WINDOWS = ("coast49", "valid16", "all50", "T_full")


def apply_token_window(
    arr: np.ndarray,
    window: str,
    model_action_horizon: int = DEFAULT_MODEL_ACTION_HORIZON,
    valid_action_horizon: int = DEFAULT_VALID_ACTION_HORIZON,
) -> np.ndarray:
    """token 축 평균으로 episode array 를 [N_step, L, D] 로 reduce.

    입력:
      - [N_step, L, D]  (token 축 이미 평균, window 무시)
      - [N_step, L, T, D] (per-T 보존, window 따라 슬라이싱 후 mean)
    """
    if arr.ndim == 3:
        return arr
    if arr.ndim != 4:
        raise ValueError(f"Expected ndim==3 or 4, got {arr.shape}")
    T = arr.shape[2]
    if window == "T_full":
        sub = arr
    elif window == "all50":
        n = min(model_action_horizon, T)
        sub = arr[:, :, -n:, :]
    elif window == "valid16":
        # 마지막 model_action_horizon 슬라이스 중 첫 valid_action_horizon
        start = max(0, T - model_action_horizon)
        sub = arr[:, :, start : start + valid_action_horizon, :]
    elif window == "coast49":
        n = min(COAST_S, T)
        sub = arr[:, :, -n:, :]
    else:
        raise ValueError(f"Unknown token window: {window}")
    return sub.mean(axis=2)


def _to_numpy(h: Any) -> np.ndarray:
    return h.detach().cpu().numpy() if hasattr(h, "detach") else np.asarray(h)


def load_episodes(
    rollout_dir: Path, window: str | None = None
) -> dict[str, dict[str, list]]:
    """raw_rollouts/<task>/*.pkl → task 별 per-episode array + 성공 플래그.

    per-step hidden_state shape 에 따라 episode array dim 이 다름:
      - 기존 multilayer (K·H pooled, [L, D]): episode → [N_step, L, D]
      - per-T multilayer ([L, T, D]): episode → [N_step, L, T, D] (token 축 보존)

    ``window`` 지정 시 load 시점에 token-window reduce 적용하여 [N_step, L, D] 로 압축
    (per-T 데이터의 메모리 폭발 회피: 4D float64 → 3D float64).

    Returns: {task_name: {"succ": [arr, ...], "fail": [...]}}
    """
    files = sorted(glob.glob(str(rollout_dir / "*" / "*.pkl")))
    if not files:
        raise ValueError(f"No pkl under {rollout_dir}")
    by_task: dict[str, dict[str, list]] = {}
    for f in files:
        with open(f, "rb") as fh:
            rec = pickle.load(fh)
        hs = rec.get("hidden_states") or []
        if not hs:
            continue
        # fp16 stack 후 window 적용 → float64 promote (메모리 ~8배 감소)
        arr16 = np.stack([_to_numpy(h) for h in hs])  # fp16 [N, L, D] 또는 [N, L, T, D]
        if arr16.ndim not in (3, 4):
            raise ValueError(
                f"Expected per-step [L,D] or [L,T,D] in {f}, got step shape {arr16.shape[1:]}"
            )
        if arr16.ndim == 4 and window is not None:
            arr16 = apply_token_window(arr16, window).astype(np.float16, copy=False)
        arr = arr16.astype(np.float64)
        task = Path(f).parent.name
        grp = by_task.setdefault(task, {"succ": [], "fail": []})
        grp["succ" if int(rec.get("episode_success", 0)) else "fail"].append(arr)
    return by_task


def _coast_group(eps: dict[str, list]) -> tuple[np.ndarray, np.ndarray, dict[str, int]] | None:
    """coast: 성공/실패 step 전부 (절단·균형 없음)."""
    succ, fail = eps["succ"], eps["fail"]
    if len(succ) < MIN_PER_GROUP or len(fail) < MIN_PER_GROUP:
        return None
    Xs = np.concatenate(succ, axis=0)  # [Ns, L, D]
    Xf = np.concatenate(fail, axis=0)  # [Nf, L, D]
    info = {"n_succ_ep": len(succ), "n_fail_ep": len(fail),
            "n_succ_step": int(Xs.shape[0]), "n_fail_step": int(Xf.shape[0])}
    return Xs, Xf, info


def _balanced_group(
    eps: dict[str, list], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, dict[str, int]] | None:
    """balanced: W_t=p10(succ len), 길이<W_t 성공 제거, 첫 W_t 절단, 실패수=남은 성공수."""
    succ, fail = eps["succ"], eps["fail"]
    if len(succ) < MIN_PER_GROUP or len(fail) < MIN_PER_GROUP:
        return None
    succ_lens = np.array([s.shape[0] for s in succ])
    W = int(round(float(np.percentile(succ_lens, SUCCESS_PERCENTILE))))
    W = max(W, 1)
    kept_succ = [s[:W] for s in succ if s.shape[0] >= W]  # 길이<W 제거 + 절단
    n = len(kept_succ)
    # 실패는 timeout(길이≥W)이라 보통 전부 ≥W; 안전하게 ≥W 인 것만, 그중 n개 subsample.
    fail_okay = [fl for fl in fail if fl.shape[0] >= W]
    if n < MIN_PER_GROUP or len(fail_okay) < MIN_PER_GROUP:
        return None
    take = min(n, len(fail_okay))
    fi = rng.choice(len(fail_okay), take, replace=False)
    used_fail = [fail_okay[i][:W] for i in fi]
    # 성공도 실패 take 수에 맞춰(둘 중 작은 쪽으로) 대칭 균형.
    if n > take:
        si = rng.choice(n, take, replace=False)
        kept_succ = [kept_succ[i] for i in si]
    Xs = np.concatenate(kept_succ, axis=0)  # [take*W, L, D]
    Xf = np.concatenate(used_fail, axis=0)  # [take*W, L, D]
    info = {"W": W, "n_succ_ep_total": len(succ), "n_succ_ep_kept": len(kept_succ),
            "n_fail_ep_total": len(fail), "n_fail_ep_used": len(used_fail),
            "n_succ_step": int(Xs.shape[0]), "n_fail_step": int(Xf.shape[0])}
    return Xs, Xf, info


def layer_sweep(
    by_task: dict[str, dict[str, list]], method: str, n_layers: int, rng: np.random.Generator
) -> tuple[list[dict[str, Any]], dict[str, dict]]:
    """방식별 per-layer q̄ sweep + per-task 균형 정보."""
    # task별 (Xs, Xf, info) 한 번만 준비 (layer 는 마지막 축에서 slice).
    prepared: dict[str, tuple[np.ndarray, np.ndarray, dict]] = {}
    per_task_info: dict[str, dict] = {}
    for task, eps in sorted(by_task.items()):
        out = _coast_group(eps) if method == "coast" else _balanced_group(eps, rng)
        if out is None:
            per_task_info[task] = {"skipped": True,
                                   "n_succ": len(eps["succ"]), "n_fail": len(eps["fail"])}
            continue
        prepared[task] = out
        per_task_info[task] = out[2]

    sweep = []
    for li in range(n_layers):
        qs, ovs, q_per_task = [], [], {}
        for task, (Xs, Xf, _info) in prepared.items():
            C_s = compute_conceptor(Xs[:, li, :], LAYER_QUOTA_ALPHA)
            C_f = compute_conceptor(Xf[:, li, :], LAYER_QUOTA_ALPHA)
            q = float(conceptor_quota(and_conceptor(C_s, not_conceptor(C_f))))
            qs.append(q)
            ovs.append(float(conceptor_overlap(C_s, C_f)))
            q_per_task[task] = round(q, 4)
        sweep.append({
            "layer": li,
            "layer_pos": li,
            "quota_bar": float(np.mean(qs)) if qs else 0.0,
            "overlap_bar": float(np.mean(ovs)) if ovs else 1.0,
            "n_tasks": len(qs),
            "quota_per_task": q_per_task,
        })
    return sweep, per_task_info


def main() -> None:
    ap = argparse.ArgumentParser(description="COAST Stage1 layer select: coast vs balanced")
    ap.add_argument("--rollout-dir", type=Path, required=True,
                    help="multi-layer raw_rollouts 디렉토리 (per-step hidden_state=[L,D] 또는 [L,T,D]).")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="기본: <rollout_dir>/../conceptor_steering/layer_selection")
    ap.add_argument("--methods", nargs="+", default=["coast", "balanced"],
                    choices=("coast", "balanced"))
    ap.add_argument("--token-windows", nargs="+", default=None,
                    choices=TOKEN_WINDOWS,
                    help=(
                        "per-T 데이터일 때 적용할 token-축 슬라이스+mean window. "
                        "복수 가능. 기본=per-T 데이터면 모든 윈도우 sweep, "
                        "이미 [L,D] pooled 데이터면 window 무시."
                    ))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="conceptor 연산 없이 방식별 per-task W/N 만 출력 후 종료.")
    args = ap.parse_args()

    # Probe: 첫 pkl 만 빠르게 읽어 shape 확인 (전체 load 전).
    probe_files = sorted(glob.glob(str(args.rollout_dir / "*" / "*.pkl")))[:1]
    if not probe_files:
        raise SystemExit(f"No pkl under {args.rollout_dir}")
    with open(probe_files[0], "rb") as fh:
        probe_rec = pickle.load(fh)
    probe_h = _to_numpy(probe_rec["hidden_states"][0])
    has_token_axis = probe_h.ndim == 3  # per-step [L, T, D]
    n_layers = probe_h.shape[0]
    token_count = probe_h.shape[1] if has_token_axis else None
    print(f"[probe] per_step_shape={probe_h.shape} ({'per-T' if has_token_axis else 'pooled'}) "
          f"n_layers={n_layers} T={token_count if has_token_axis else 'N/A'} "
          f"threads={_THREADS}", flush=True)

    if has_token_axis:
        windows = args.token_windows or list(TOKEN_WINDOWS)
    else:
        windows = ["(pre-pooled)"]
        if args.token_windows:
            print(f"[warn] token-windows 무시: 데이터가 이미 pooled [L,D]", flush=True)

    if args.dry_run:
        # dry-run: window 하나만 load 해서 balance 정보 확인.
        win0 = windows[0] if has_token_axis else None
        by_task = load_episodes(args.rollout_dir, window=win0)
        rng = np.random.default_rng(args.seed)
        print(f"\n=== window={windows[0]} (dry-run) ===", flush=True)
        for method in args.methods:
            print(f"  --- [{method}] per-task 균형 ---", flush=True)
            for task, eps in sorted(by_task.items()):
                out = _coast_group(eps) if method == "coast" else _balanced_group(eps, rng)
                if out is None:
                    print(f"    {task:<28} SKIP (succ={len(eps['succ'])} fail={len(eps['fail'])})",
                          flush=True)
                    continue
                print(f"    {task:<28} {out[2]}", flush=True)
        print("\n[dry-run] conceptor 연산 생략, 종료", flush=True)
        return

    out_dir = args.out_dir or (args.rollout_dir.parent / "conceptor_steering" / "layer_selection")
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "alpha0": LAYER_QUOTA_ALPHA,
        "success_percentile": SUCCESS_PERCENTILE,
        "n_layers": n_layers,
        "token_count_raw": token_count,
        "windows": list(windows),
        "source": str(args.rollout_dir),
        "created": datetime.now().isoformat(timespec="seconds"),
        "results": {},  # results[window][method] = ...
    }
    for window in windows:
        win_load = window if has_token_axis else None
        print(f"\n[load] window={window} ...", flush=True)
        by_task = load_episodes(args.rollout_dir, window=win_load)
        result["results"][window] = {}
        for method in args.methods:
            rng = np.random.default_rng(args.seed)
            sweep, per_task_info = layer_sweep(by_task, method, n_layers, rng)
            best = max(sweep, key=lambda r: r["quota_bar"])
            result["results"][window][method] = {
                "selected_layer": best["layer"],
                "selected_layer_pos": best["layer_pos"],
                "selected_quota_bar": best["quota_bar"],
                "selected_overlap_bar": best["overlap_bar"],
                "sweep": sweep,
                "per_task_balance": per_task_info,
            }
            print(f"\n=== [{window} | {method}] layer sweep ===", flush=True)
            for r in sweep:
                print(f"  layer {r['layer']:>2}: q̄={r['quota_bar']:.4f} "
                      f"ov̄={r['overlap_bar']:.3f} (n_task={r['n_tasks']})", flush=True)
            print(f"[{window} | {method}] ℓ*={best['layer']} q̄={best['quota_bar']:.4f} "
                  f"ov̄={best['overlap_bar']:.3f}", flush=True)

    out_path = out_dir / "stage1_quota_compare.json"
    with out_path.open("w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nsaved -> {out_path}", flush=True)
    sel = {
        f"{w}|{m}": result["results"][w][m]["selected_layer"]
        for w in windows for m in args.methods
    }
    print(f"[selected] {sel}", flush=True)


if __name__ == "__main__":
    main()

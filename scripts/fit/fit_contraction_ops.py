#!/usr/bin/env python3
"""수축형(contraction) steering 연산자 3종 fit — phase × layer 격자, serve gated 계약.

배경 (docs/steering/42 + scripts/analysis/mean_var_sep.py 진단): setM(평균-차이 setpoint)
개입이 replay 사분면 eval 에서 구제 0 이었다. 진단은 "실패가 성공 평균에서 한 방향으로
밀린 것"이 아니라 **산포가 1.3~1.8 배 넓게 퍼진 것**이라고 말한다. 그래서 이동(translation)
대신 **수축(contraction)** 연산자를 같은 fit 셀(s5m5 manifest)에서 fit 해 비교한다.

연산자 3종 (전부 D×D 행렬 M 하나. serve 는 h' = h·Mᵀ 로 적용 —
steering_hooks.ConceptorSteering, M = (1−β)I + β·C 이므로 **β=1.0 에서 M = 저장 행렬**):

  1. sconceptor  순수 성공 conceptor  C_s = R_s (R_s + α⁻²I)⁻¹,  R_s = **비중심** E[hhᵀ].
     (src.conceptor.compute_conceptor 는 mean-centering 을 하므로 여기서는 쓰지 않고
      비중심 R 로 같은 수식을 계산한다 — 출처: src/conceptor/core.py compute_conceptor.)
     성공 분포가 차지하는 부분공간만 통과시키는 soft projection = 산포 수축.
  2. conceptor   기존 대조 C_steer = C_s ∧ ¬C_f (fit_phase_conceptor.fit_one 그대로 호출).
  3. varc        분산-가중 수축. 성공 whitening 공간에서 실패 과분산 축만 감쇠:
        W = Σ_s^{-1/2} (shrinkage δ),  Σ̃_f = W Σ_f W = V diag(λ) Vᵀ,
        gain g_i = 1/λ_i (λ_i>1 축만, [g_min,1] clip),  M = W⁻¹ V diag(g) Vᵀ W.
     ★ 중심화 근사: 적용이 순선형(h·Mᵀ)이라 μ_s 오프셋을 실을 수 없다. 대신 whitened
       μ_s 방향을 게인 1 로 고정(P M P + uuᵀ)해 평균 성분을 보존한다 (fit_summary 에 명시).

α(aperture) 선택은 fit_phase_conceptor 의 grid·overlap band·quota 로직을 **import 재사용**
(select_alpha / OVERLAP_BAND / TABLE14_ALPHAS). exp3 의 aperture 포화(데이터 위 C≈0 →
M≈(1−β)I 균일수축, [[conceptor-saturation-degenerate]]) 재발을 막기 위해 **비퇴화 진단**
(fit 표본 median ‖h·Mᵀ−h‖/‖h‖, C 의 유효 rank/quota)을 전 α 에 대해 계산하고, band 선택 α 가
퇴화면 비퇴화 후보 중에서 재선택한다 (전부 퇴화면 해당 (phase,layer,op) 미생성 = identity).

산출 (serve gated 등록 계약 — <base>/<phase>/dit_L{n}/conceptors.npz):
    <out_root>/<op>/<phase>/dit_L{n}/conceptors.npz      키: alpha0_C_steer  [D,D]
    <out_root>/<op>_pl/<phase>/dit_L{n}/conceptors.npz   라벨순열 위약 (같은 α·같은 절차)
    <out_root>/<op>/fit_summary.json                     비퇴화·dose·α 진단 전체
NPZ 키의 alpha0 은 **선택 α 를 구워 넣었다**는 뜻 (serve `--steering-alpha 0` 소비).
실제 aperture 는 metadata.json 의 `aperture_alpha`.

serve 배선 (러너 run_online_gated_eval.sh 가 조립):
    --steering-phase-npz-base <out_root>/<op> --steering-phases <...> --steering-layers <n>
    --steering-op conceptor --steering-alpha 0 --steering-beta 1.0 --steering-token-select all

사용 (승준 노드, pkl 있는 곳):
  ~/anaconda3/bin/python scripts/fit/fit_contraction_ops.py \
    --manifest <s5m5.tsv> --cell OpenDrawer_left --out-root outputs/steer/online_pipe/OpenDrawer_left/contract_s5m5 \
    --phase-groups auto --layers 8,12 --ops sconceptor,conceptor,varc
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/fit

from fit_phase_conceptor import (  # noqa: E402
    DEFAULT_ALPHAS,
    OVERLAP_BAND,
    TABLE14_ALPHAS,
    _write_config,
    fit_one,
    select_alpha,
)
from fit_setm import (  # noqa: E402
    GATED_MIN_EPS,
    GATED_MIN_REC,
    RNG_SEED,
    auroc,
    episode_phase_records,
    gather_phase,
    load_cell_rolls,
    mean_diff,
    phase_dwell_caps,
    resolve_phase_groups,
)
from src.conceptor import conceptor_overlap, conceptor_quota  # noqa: E402

OPS = ("sconceptor", "conceptor", "varc")

# 비퇴화 판정 (fit 표본에서 연산자가 실제로 만드는 상대 이동량 median ‖h·Mᵀ−h‖/‖h‖).
# β=1 에서 M=C 이므로 퇴화는 양쪽 끝이다:
#   C≈I → rel_move≈0 (무개입. β<1 의 exp3 포화판 M≈(1−β)I 과 같은 자리)
#   C≈0 → rel_move≈1 (활성화 소거 — aperture 포화의 반대 끝, 정책 파괴)
# CLI 로 조절 가능 (--min-rel-move/--max-rel-move).
NONDEGEN_MIN_REL_MOVE = 0.02
NONDEGEN_MAX_REL_MOVE = 0.90
NONDEGEN_MIN_QUOTA = 0.002      # tr(C)/D 하한 (sconceptor/conceptor 전용 진단)

N_PL_CAND = 3                   # 위약 순열 후보 수 (dose 가 처치에 가장 가까운 것 채택)
VARC_SHRINKAGE = 0.05           # Σ_s 대각 shrinkage δ (tr(Σ)/D 스케일)
VARC_GAIN_MIN = 0.25            # 게인 하한 (λ≤4 까지만 되돌린다 — 과수축 방지)


# ---------------------------------------------------------------- 연산자 원자
def _uncentered_corr(X: np.ndarray) -> np.ndarray:
    """비중심 상관 R = E[hhᵀ]  (src.conceptor.compute_correlation 은 mean-center 하므로 별도)."""
    X = np.asarray(X, dtype=np.float64)
    R = (X.T @ X) / X.shape[0]
    return (R + R.T) * 0.5


def conceptor_from_R(R: np.ndarray, alpha: float) -> np.ndarray:
    """C = R (R + α⁻²I)⁻¹  — src/conceptor/core.py compute_conceptor 와 같은 수식,
    입력만 비중심 R 이다."""
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    d = R.shape[0]
    C = np.linalg.solve(R + (alpha ** -2) * np.eye(d), R)
    return (C + C.T) * 0.5


def _inv_sqrt(S: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """대칭 PSD S → (S^{-1/2}, S^{1/2})."""
    w, V = np.linalg.eigh((S + S.T) * 0.5)
    w = np.clip(w, eps, None)
    return (V * (w ** -0.5)) @ V.T, (V * (w ** 0.5)) @ V.T


def fit_varc(Xs: np.ndarray, Xf: np.ndarray, *, shrinkage: float = VARC_SHRINKAGE,
             gain_min: float = VARC_GAIN_MIN,
             fail_center: str = "own") -> tuple[np.ndarray, dict]:
    """분산-가중 수축 M (성공 whitening 공간에서 실패 과분산 축만 감쇠).

    fail_center: 'own' = Σ_f 를 μ_f 로 중심화(순수 산포) | 'succ' = μ_s 기준(평균 이동 포함).
    """
    Xs = np.asarray(Xs, dtype=np.float64)
    Xf = np.asarray(Xf, dtype=np.float64)
    D = Xs.shape[1]
    mu_s = Xs.mean(axis=0)
    Ss = np.cov(Xs, rowvar=False)
    ctr = mu_s if fail_center == "succ" else Xf.mean(axis=0)
    Xf_c = Xf - ctr
    Sf = (Xf_c.T @ Xf_c) / max(len(Xf) - 1, 1)
    delta = shrinkage * float(np.trace(Ss)) / D
    Ss = (Ss + Ss.T) * 0.5 + delta * np.eye(D)
    Sf = (Sf + Sf.T) * 0.5
    W, W_inv = _inv_sqrt(Ss)                       # W = Σ_s^{-1/2}, W_inv = Σ_s^{1/2}
    Sf_w = W @ Sf @ W
    lam, V = np.linalg.eigh((Sf_w + Sf_w.T) * 0.5)
    g = np.where(lam > 1.0, 1.0 / np.maximum(lam, 1e-12), 1.0)
    g = np.clip(g, gain_min, 1.0)
    Mw = (V * g) @ V.T                             # whitened 공간 수축 (대칭)
    # ★ 중심화 근사: 순선형 적용이라 μ_s 오프셋을 실을 수 없다 → whitened μ_s 방향은
    #   게인 1 로 고정해 평균 성분을 보존한다 (P M P + uuᵀ, 대칭 유지).
    u = W @ mu_s
    nu = float(np.linalg.norm(u))
    mu_preserved = nu > 1e-12
    if mu_preserved:
        u = u / nu
        P = np.eye(D) - np.outer(u, u)
        Mw = P @ Mw @ P + np.outer(u, u)
    M = W_inv @ Mw @ W
    diag = {
        "n_axes_overdispersed": int(np.sum(lam > 1.0)),
        "lambda_max": float(lam.max()), "lambda_median": float(np.median(lam)),
        "lambda_frac_gt1": float(np.mean(lam > 1.0)),
        "gain_min": float(g.min()), "gain_median": float(np.median(g)),
        "gain_frac_clipped": float(np.mean(g <= gain_min + 1e-12)),
        "shrinkage_delta_rel": float(shrinkage), "fail_center": fail_center,
        "mu_s_direction_preserved": mu_preserved,
        "centering_note": "적용이 순선형(h·Mᵀ)이라 μ_s 오프셋 미탑재 — whitened μ_s "
                          "방향 게인 1 고정으로 근사 (중심화 근사)",
    }
    return M, diag


# ---------------------------------------------------------------- 진단
def op_diag(M: np.ndarray, X: np.ndarray) -> dict:
    """비퇴화 진단: fit 표본 상대 이동량 + 연산자 스펙트럼."""
    X = np.asarray(X, dtype=np.float64)
    d = X @ M.T - X
    nx = np.linalg.norm(X, axis=1)
    rel = np.linalg.norm(d, axis=1) / np.maximum(nx, 1e-12)
    Ms = (M + M.T) * 0.5
    w = np.linalg.eigvalsh(Ms)
    D = M.shape[0]
    return {
        "rel_move_median": float(np.median(rel)),
        "rel_move_p90": float(np.percentile(rel, 90)),
        "abs_move_median": float(np.median(np.linalg.norm(d, axis=1))),
        "quota": float(np.trace(M) / D),                 # tr(C)/D (M=C at β=1)
        "eff_rank_gt05": int(np.sum(w > 0.5)),
        "eff_rank_gt01": int(np.sum(w > 0.1)),
        "eig_min": float(w.min()), "eig_max": float(w.max()),
        "fro_M_minus_I": float(np.linalg.norm(M - np.eye(D), "fro")),
    }


def is_degenerate(diag: dict, op: str, lo: float = NONDEGEN_MIN_REL_MOVE,
                  hi: float = NONDEGEN_MAX_REL_MOVE,
                  min_quota: float = NONDEGEN_MIN_QUOTA) -> str | None:
    r = diag["rel_move_median"]
    if r < lo:
        return f"rel_move_median {r:.4f} < {lo} (무개입 — C≈I)"
    if r > hi:
        return f"rel_move_median {r:.4f} > {hi} (활성화 소거 — C≈0)"
    if op in ("sconceptor", "conceptor") and diag["quota"] < min_quota:
        return f"quota {diag['quota']:.5f} < {min_quota} (aperture 포화)"
    return None


# ---------------------------------------------------------------- op 별 후보 생성
def candidates_for_op(op: str, Xs: np.ndarray, Xf: np.ndarray, alphas, args) -> tuple[dict, list]:
    """(alpha → M, sweep 목록) 반환. varc 는 α 없음 → {0.0: M}."""
    Xs = np.asarray(Xs, dtype=np.float64)
    Xf = np.asarray(Xf, dtype=np.float64)
    if op == "varc":
        M, vd = fit_varc(Xs, Xf, shrinkage=args.varc_shrinkage,
                         gain_min=args.varc_gain_min, fail_center=args.varc_fail_center)
        return {0.0: M}, [{"alpha": 0.0, "overlap": None, "quota_steer": None, "varc": vd}]
    if op == "sconceptor":
        Rs, Rf = _uncentered_corr(Xs), _uncentered_corr(Xf)
        cands, sweep = {}, []
        for a in alphas:
            Cs = conceptor_from_R(Rs, a)
            Cf = conceptor_from_R(Rf, a)
            cands[float(a)] = Cs
            sweep.append({"alpha": float(a), "overlap": conceptor_overlap(Cs, Cf),
                          "quota_success": conceptor_quota(Cs),
                          "quota_failure": conceptor_quota(Cf),
                          # select_alpha 의 quota floor 는 "배치되는 연산자"의 quota —
                          # sconceptor 의 배치 객체는 C_s 다.
                          "quota_steer": conceptor_quota(Cs)})
        return cands, sweep
    # op == "conceptor": fit_phase_conceptor 수식 그대로 (mean-centered R, 전 α 저장)
    fits, meta = fit_one(Xs, Xf, list(alphas), OVERLAP_BAND, args.quota_floor)
    sweep = meta["alpha_sweep"]
    if fits is None:
        return {}, sweep
    # fit_one 은 선택 α + default 만 캐시한다 → 비퇴화 재선택 후보 확보를 위해 전 α 재계산
    from src.conceptor import and_conceptor, compute_conceptor, not_conceptor

    cands = {}
    for a in alphas:
        Cs = compute_conceptor(Xs, a)
        Cf = compute_conceptor(Xf, a)
        cands[float(a)] = and_conceptor(Cs, not_conceptor(Cf))
    return cands, sweep


def select_operator(op: str, cands: dict, sweep: list, X_all: np.ndarray,
                    quota_floor: float, gate: dict | None = None
                    ) -> tuple[float | None, np.ndarray | None, dict]:
    """band 선택 → 비퇴화 검사 → 필요 시 재선택. 반환 (alpha, M, sel_meta)."""
    gate = gate or {}
    _deg = lambda d: is_degenerate(d, op, **gate)  # noqa: E731
    diags = {a: op_diag(M, X_all) for a, M in cands.items()}
    ok = {a: d for a, d in diags.items() if _deg(d) is None}
    sel_meta = {
        "alpha_diag": {f"{a:g}": diags[a] for a in sorted(diags)},
        "degenerate_reason": {f"{a:g}": _deg(diags[a]) for a in sorted(diags)},
        "nondegenerate_alphas": sorted(ok),
        "nondegenerate_gate": {"min_rel_move": gate.get("lo", NONDEGEN_MIN_REL_MOVE),
                               "max_rel_move": gate.get("hi", NONDEGEN_MAX_REL_MOVE),
                               "min_quota": gate.get("min_quota", NONDEGEN_MIN_QUOTA)},
    }
    if not cands:
        return None, None, {**sel_meta, "selection_mode": "no_candidate"}
    if op == "varc":
        a = 0.0
        why = _deg(diags[a])
        if why:
            return None, None, {**sel_meta, "selection_mode": "degenerate", "skip_reason": why}
        return a, cands[a], {**sel_meta, "selection_mode": "varc(no-aperture)"}
    a_band, mode = select_alpha(sweep, OVERLAP_BAND, quota_floor)
    if a_band is not None and _deg(diags[float(a_band)]) is None:
        return float(a_band), cands[float(a_band)], {**sel_meta, "selection_mode": mode,
                                                     "band_alpha": float(a_band)}
    if not ok:
        return None, None, {
            **sel_meta, "selection_mode": "degenerate_all", "band_alpha": a_band,
            "skip_reason": "전 α 퇴화 — " + (_deg(diags[float(a_band)])
                                             if a_band is not None else "band 선택 불가")}
    # 재선택: 비퇴화 후보 중 quota(=부분공간 점유) 최대. 동률이면 α 작은 쪽.
    a_re = max(sorted(ok), key=lambda a: ok[a]["quota"])
    return a_re, cands[a_re], {**sel_meta, "selection_mode": f"{mode}→nondegenerate_requota",
                               "band_alpha": a_band,
                               "reselect_reason": (_deg(diags[float(a_band)])
                                                   if a_band is not None else "band 선택 불가")}


# ---------------------------------------------------------------- IO
def save_conceptor_npz(out_dir: Path, layer_blk: int, C: np.ndarray, meta: dict) -> None:
    """serve gated 계약 NPZ. 키는 `alpha0_C_steer` 단일 — 선택 α 를 구워 넣었다는 뜻이고
    serve 는 `--steering-alpha 0` 으로 그 키를 집는다 (α 오배선 방지)."""
    d = out_dir / f"dit_L{layer_blk}"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "conceptors.npz", alpha0_C_steer=C.astype(np.float32))
    full = {**meta, "selected_alpha": 0,
            "note_alpha": "NPZ 키 alpha0 = 선택 aperture 를 구운 슬롯. 실 aperture 는 "
                          "aperture_alpha (varc 는 aperture 개념 없음)"}
    (d / "metadata.json").write_text(json.dumps(full, indent=2, ensure_ascii=False))
    _write_config(d, full)


# ---------------------------------------------------------------- 위약
def draw_placebo_labels(labels_arr: np.ndarray, rng, n_cand: int) -> list[list[int]]:
    """episode 단위 라벨 순열 후보 (양 클래스 비지 않은 것만)."""
    out = []
    for _ in range(max(n_cand * 20, 20)):
        pl = labels_arr.copy()
        rng.shuffle(pl)
        if 0 < int(pl.sum()) < len(pl):
            out.append(pl.tolist())
        if len(out) >= n_cand:
            break
    return out


# ---------------------------------------------------------------- main 루프
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", type=Path, required=True,
                    help="fit 셀 manifest tsv (pkl\\tlabel\\tscene) — fit_setm 과 같은 규약")
    ap.add_argument("--cell", required=True, help="pkl 부모 디렉토리명 (예: OpenDrawer_left)")
    ap.add_argument("--out-root", type=Path, required=True,
                    help="산출 루트. <out-root>/<op>/<phase>/dit_L{n}/conceptors.npz")
    ap.add_argument("--phase-groups", default="auto",
                    help="콤마목록 또는 'auto'(데이터에 있는 phase 전부, terminal 동치 제외). "
                         "'global' = phase 무구분 전 rollout 풀 (COAST Global variant)")
    ap.add_argument("--layers", default=None,
                    help="DiT 물리 layer 콤마목록 (권장 — 미지정 시 capture 전부, tok 메모리 큼)")
    ap.add_argument("--ops", default=",".join(OPS), help=f"fit 할 연산자 (기본 {','.join(OPS)})")
    ap.add_argument("--token-pool", choices=["mean", "state", "future", "action"], default="mean",
                    help="fit record 의 토큰 세그먼트. 기본 mean = 49토큰 평균")
    ap.add_argument("--alphas", default="table14",
                    help="aperture grid: 'table14'(기본, COAST Table 14) | 'default' | 콤마목록")
    ap.add_argument("--quota-floor", type=float, default=0.0,
                    help="select_alpha 의 quota_steer 하한 (퇴화 α 1차 배제)")
    ap.add_argument("--min-per-class", type=int, default=GATED_MIN_EPS,
                    help=f"phase 채택 최소 episode 수/클래스 (기본 {GATED_MIN_EPS})")
    ap.add_argument("--min-records", type=int, default=GATED_MIN_REC,
                    help=f"phase 채택 최소 record 수/클래스 (기본 {GATED_MIN_REC})")
    ap.add_argument("--no-length-control", action="store_true",
                    help="phase dwell cap 해제 (길이 confound 미통제 변형)")
    ap.add_argument("--min-rel-move", type=float, default=NONDEGEN_MIN_REL_MOVE,
                    help="비퇴화 하한 (fit 표본 median ‖h·Mᵀ−h‖/‖h‖). 미만이면 무개입 취급")
    ap.add_argument("--max-rel-move", type=float, default=NONDEGEN_MAX_REL_MOVE,
                    help="비퇴화 상한. 초과면 활성화 소거(C≈0) 취급 → α 재선택")
    ap.add_argument("--min-quota", type=float, default=NONDEGEN_MIN_QUOTA,
                    help="conceptor 계열 tr(C)/D 하한 (aperture 포화 배제)")
    ap.add_argument("--n-placebo-cand", type=int, default=N_PL_CAND,
                    help="위약 순열 후보 수 (dose 가 처치에 가장 가까운 것 채택)")
    ap.add_argument("--varc-shrinkage", type=float, default=VARC_SHRINKAGE)
    ap.add_argument("--varc-gain-min", type=float, default=VARC_GAIN_MIN)
    ap.add_argument("--varc-fail-center", choices=["own", "succ"], default="own",
                    help="Σ_f 중심: own=μ_f(순수 산포, 기본) | succ=μ_s(평균 이동 포함)")
    args = ap.parse_args()

    ops = [o.strip() for o in args.ops.split(",") if o.strip()]
    bad = [o for o in ops if o not in OPS]
    if bad:
        raise SystemExit(f"알 수 없는 op {bad} (가능: {list(OPS)})")
    if args.alphas == "table14":
        alphas = list(TABLE14_ALPHAS)
    elif args.alphas == "default":
        alphas = list(DEFAULT_ALPHAS)
    else:
        alphas = [float(x) for x in args.alphas.split(",") if x.strip()]

    tok_blks = [int(x) for x in args.layers.split(",")] if args.layers else None
    rolls = load_cell_rolls(args.manifest, args.cell, args.token_pool, tok_layer_blks=tok_blks)
    labels = [r["success"] for r in rolls]
    labels_arr = np.asarray(labels)
    cap_layers = [int(x) for x in rolls[0]["capture_layers"]]
    layer_blks = tok_blks if tok_blks else cap_layers
    for b in layer_blks:
        if b not in cap_layers:
            raise SystemExit(f"layer {b} 는 capture {cap_layers} 에 없음")
    groups, excluded = resolve_phase_groups(args.phase_groups, rolls)
    dwell_caps = phase_dwell_caps(rolls, labels, groups)
    if args.no_length_control:
        dwell_caps = {ph: 10 ** 9 for ph in dwell_caps}
        print(f"[no-length-control] dwell cap 해제 — phases={sorted(dwell_caps)}", flush=True)

    n_s = int(sum(labels))
    print(f"[{args.cell}] rollouts={len(rolls)} succ={n_s} fail={len(rolls) - n_s} "
          f"phases={groups} layers={layer_blks} ops={ops} token_pool={args.token_pool} "
          f"alphas={alphas}", flush=True)
    if excluded:
        print(f"  (terminal 동치 phase 제외: {excluded})", flush=True)

    # stale 디렉토리 제거 (남은 phase 가 무음 등록되는 사고 방지 — fit_setm 과 같은 규약)
    for op in ops:
        for d in (args.out_root / op, args.out_root / f"{op}_pl"):
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)

    manifest_sha = hashlib.sha256(args.manifest.read_bytes()).hexdigest()[:12]
    base_meta = {
        "input_sigs": [r["sig"] for r in rolls],       # docs/04 — 경로 아닌 내용 지문
        "cell": args.cell, "token_pool": args.token_pool,
        "length_control": not args.no_length_control,
        "manifest_sha": manifest_sha, "rng_seed": RNG_SEED,
        "n_rollouts": len(rolls), "n_succ": n_s, "n_fail": len(rolls) - n_s,
        "feature": "dit 49-token mean (token_pool)", "beta_contract":
            "serve M=(1−β)I+β·C — β=1.0 에서 저장 행렬이 곧 연산자",
    }
    diag_rows: dict[str, list] = {op: [] for op in ops}
    written = {op: 0 for op in ops}

    for ph in groups:
        if ph not in dwell_caps:
            for op in ops:
                diag_rows[op].append({"phase": ph, "skipped_identity": True,
                                      "skip_reason": "성공 episode dwell 없음 (대조 불가)"})
            print(f"  [{ph}] SKIP(성공 dwell 없음)", flush=True)
            continue
        dcap = dwell_caps[ph]
        for blk in layer_blks:
            li = cap_layers.index(blk)
            Xs, ns_eps = gather_phase(rolls, labels, li, 1, ph, dcap)
            Xf, nf_eps = gather_phase(rolls, labels, li, 0, ph, dcap)
            head = {"phase": ph, "layer": blk, "dwell_cap": dcap,
                    "n_rec_succ": int(len(Xs)), "n_rec_fail": int(len(Xf)),
                    "n_eps_succ": ns_eps, "n_eps_fail": nf_eps}
            if not (len(Xs) >= args.min_records and len(Xf) >= args.min_records
                    and ns_eps >= args.min_per_class and nf_eps >= args.min_per_class):
                for op in ops:
                    diag_rows[op].append({**head, "skipped_identity": True,
                                          "skip_reason": f"quota 미달 (record ≥{args.min_records}"
                                                         f"/클래스, episode ≥{args.min_per_class})"})
                print(f"  [{ph}|L{blk}] Nrec s/f={len(Xs)}/{len(Xf)} "
                      f"Neps s/f={ns_eps}/{nf_eps} SKIP(quota 미달 → 무개입)", flush=True)
                continue
            Xs64 = Xs.astype(np.float64)
            Xf64 = Xf.astype(np.float64)
            X_all = np.concatenate([Xs64, Xf64], axis=0)
            # 참고 분리도 (탐색 지표 — held-out 아님): mean-diff 사영 AUROC
            try:
                v_ref, _s_ref = mean_diff(Xs64, Xf64)
                a_ref = auroc(Xf64 @ v_ref, Xs64 @ v_ref)
            except ValueError:
                a_ref = float("nan")
            # 산포비 (setM 실패 진단의 그 지표): 실패/성공 평균 거리제곱 비
            disp_ratio = float(np.trace(np.cov(Xf64, rowvar=False))
                               / max(np.trace(np.cov(Xs64, rowvar=False)), 1e-12))
            pl_labels_list = draw_placebo_labels(
                labels_arr, np.random.default_rng(RNG_SEED + 555), args.n_placebo_cand)

            for op in ops:
                entry = {**head, "op": op, "meandiff_fit_auroc": a_ref,
                         "disp_ratio_fail_over_succ": disp_ratio}
                cands, sweep = candidates_for_op(op, Xs64, Xf64, alphas, args)
                a_sel, M, sel_meta = select_operator(
                    op, cands, sweep, X_all, args.quota_floor,
                    gate={"lo": args.min_rel_move, "hi": args.max_rel_move,
                          "min_quota": args.min_quota})
                entry.update({"alpha_sweep": sweep, **sel_meta})
                if M is None:
                    entry.update({"skipped_identity": True,
                                  "skip_reason": sel_meta.get("skip_reason", "연산자 없음")})
                    diag_rows[op].append(entry)
                    print(f"  [{ph}|L{blk}|{op}] SKIP({entry['skip_reason']})", flush=True)
                    continue
                d_treat = op_diag(M, X_all)
                entry.update({"aperture_alpha": a_sel, "treat_diag": d_treat})

                # ---- 위약: 같은 절차·같은 α, 라벨만 episode 단위 순열 ----
                pl_best = None
                for pl in pl_labels_list:
                    Xs_p, _ = gather_phase(rolls, pl, li, 1, ph, dcap)
                    Xf_p, _ = gather_phase(rolls, pl, li, 0, ph, dcap)
                    if len(Xs_p) < 2 or len(Xf_p) < 2:
                        continue
                    Xs_p = Xs_p.astype(np.float64)
                    Xf_p = Xf_p.astype(np.float64)
                    if op == "varc":
                        M_p, _vd = fit_varc(Xs_p, Xf_p, shrinkage=args.varc_shrinkage,
                                            gain_min=args.varc_gain_min,
                                            fail_center=args.varc_fail_center)
                    elif op == "sconceptor":
                        M_p = conceptor_from_R(_uncentered_corr(Xs_p), a_sel)
                    else:
                        from src.conceptor import and_conceptor, compute_conceptor, not_conceptor
                        M_p = and_conceptor(compute_conceptor(Xs_p, a_sel),
                                            not_conceptor(compute_conceptor(Xf_p, a_sel)))
                    d_p = op_diag(M_p, X_all)   # dose 는 **처치와 같은 표본**에서 비교
                    gap = abs(np.log(max(d_p["rel_move_median"], 1e-12)
                                     / max(d_treat["rel_move_median"], 1e-12)))
                    if pl_best is None or gap < pl_best[0]:
                        pl_best = (gap, M_p, d_p, pl)
                if pl_best is None:
                    entry.update({"skipped_identity": True,
                                  "skip_reason": "위약 후보 0 (클래스 고갈) → 양쪽 무개입"})
                    diag_rows[op].append(entry)
                    print(f"  [{ph}|L{blk}|{op}] SKIP(위약 불가 → 양쪽 무개입)", flush=True)
                    continue
                _gap, M_pl, d_pl, pl_lab = pl_best
                entry.update({
                    "placebo_diag": d_pl,
                    "dose_ratio_pl_over_treat": (d_pl["rel_move_median"]
                                                 / max(d_treat["rel_move_median"], 1e-12)),
                    "placebo_labels": pl_lab,
                    "note_dose": "dose = fit 표본 median ‖h·Mᵀ−h‖/‖h‖ (처치·위약 같은 표본). "
                                 "행렬 연산자는 스케일을 실을 자리가 없어 dose-match 는 "
                                 "후보 선택까지만 — 잔여 비율은 dose_ratio 로 기록",
                })
                meta = {**base_meta, "operator": op, "phase": ph, "layer": blk,
                        "dwell_cap": dcap, "aperture_alpha": a_sel,
                        **{k: v for k, v in entry.items() if k != "placebo_labels"}}
                save_conceptor_npz(args.out_root / op / ph, blk, M, meta)
                save_conceptor_npz(args.out_root / f"{op}_pl" / ph, blk, M_pl,
                                   {**meta, "operator": f"{op}_pl", "placebo": True,
                                    "placebo_labels": pl_lab})
                written[op] += 1
                diag_rows[op].append(entry)
                print(f"  [{ph}|L{blk}|{op}] α={a_sel} rel_move={d_treat['rel_move_median']:.3f}"
                      f"(pl {d_pl['rel_move_median']:.3f}) quota={d_treat['quota']:.3f} "
                      f"rank>0.5={d_treat['eff_rank_gt05']} "
                      f"‖M−I‖={d_treat['fro_M_minus_I']:.1f}", flush=True)

    # ---- 요약 (op 마다 fit_summary.json + 루트 통합본) ----
    combined = {}
    for op in ops:
        rows = diag_rows[op]
        fitted = sorted({r["phase"] for r in rows if not r.get("skipped_identity")})
        per_layer = {b: sorted({r["phase"] for r in rows if r.get("layer") == b
                                and not r.get("skipped_identity")}) for b in layer_blks}
        ragged = sorted({b for b, ps in per_layer.items() if ps != fitted})
        summary = {
            **base_meta, "op": op, "mode": "contraction_phase_groups",
            "layers": layer_blks, "phase_groups_arg": args.phase_groups,
            "phases_requested": groups, "phases_excluded_terminal": excluded,
            "phases_fitted": fitted, "phases_per_layer": {str(b): p for b, p in per_layer.items()},
            "layers_with_missing_phase": ragged,
            "dwell_caps": dwell_caps if not args.no_length_control else "disabled",
            "quota": {"min_rec": args.min_records, "min_eps": args.min_per_class},
            "alphas": alphas, "quota_floor": args.quota_floor,
            "nondegenerate_gate": {"rel_move": [args.min_rel_move, args.max_rel_move],
                                   "min_quota": args.min_quota},
            "varc_params": {"shrinkage": args.varc_shrinkage, "gain_min": args.varc_gain_min,
                            "fail_center": args.varc_fail_center,
                            "centering": "중심화 근사 — μ_s 방향 게인 1 고정"},
            "n_npz_written": written[op], "placebo_dir": f"{op}_pl", "phases": rows,
            "note": "AUROC·dose 는 fit-표본 지표(held-out 아님). 미생성 (phase,layer) 는 "
                    "serve 미등록 → identity(무개입).",
        }
        (args.out_root / op / "fit_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False))
        combined[op] = {"n_npz": written[op], "phases_fitted": fitted,
                        "layers_with_missing_phase": ragged}
        if ragged:
            print(f"[warn] {op}: layer {ragged} 의 phase 집합이 다름 — serve 는 layer×phase "
                  f"완전성을 요구한다", flush=True)
        print(f"[{op} done] {args.out_root / op} (+{op}_pl) phases={fitted} "
              f"npz={written[op]}", flush=True)
    (args.out_root / "contraction_fit_summary.json").write_text(
        json.dumps({**base_meta, "ops": combined, "layers": layer_blks,
                    "phases_requested": groups}, indent=2, ensure_ascii=False))
    if not any(written.values()):
        print("[empty] 생성된 연산자 0개", flush=True)
        sys.exit(3)
    print(f"[done] -> {args.out_root}", flush=True)


if __name__ == "__main__":
    main()

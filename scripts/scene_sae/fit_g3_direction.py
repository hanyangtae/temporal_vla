#!/usr/bin/env python
"""exp5/G3(write) 준비 — within-scene 실패축 r̂ 를 뽑아 setM 계약 NPZ 로 내보낸다.

근거 = `docs/steering/32_g2_scene_residual_results.md` §5 (채택 설계):

  - **온라인 잔차화는 포기**한다 (발견 A: 새 scene 성분은 추론 중 제거 불가).
    잔차화는 **offline 에서 방향 r̂ 를 뽑는 데만** 쓰고, 추론 시에는 고정 벡터 하나만 적용.
  - [4-2] 실패 신호는 **1차원** → conceptor 같은 다차원 연산자 불요, 단일 벡터로 충분.
  - [4-1] 잔차화는 scene 별 방향 일관성(cos 0.398→0.394~0.406)을 **개선하지 못했다** →
    잔차화 유/무 두 벌을 나란히 fit 해서 비교하는 것이 이 스크립트의 목적.
    (기대치는 낮게. 이 스크립트는 판정이 아니라 재료 산출이다.)

## 수학 (부호 규약 — ★혼동 금지)

  scene s 안에서   d_s = mean(fail) − mean(succ)          (실패 쪽을 가리키는 방향)
  r̂ = normalize( mean_s d_s )                              (혼재 scene 평균 → 정규화)

  즉 **r̂ 는 실패 방향**이고, **성공 쪽으로 미는 것은 −r̂** 이다. exp4-1 `fit_mean_diff.py`
  (`mean_diff`: r̂=normalize(μ_fail−μ_succ)) 와 **같은 부호**이며, serve 의 적용식

      h' = h − β[(h·r̂) − s]·r̂        (SetpointSteering, steering_hooks.py)

  가 h 의 r̂ 좌표를 성공 setpoint s 로 끌어내리므로 (proj>s 면 delta 가 −r̂ 방향) 그대로
  호환된다.

  ★ **부호 뒤집기는 reverse arm 이 아니다.** SetpointSteering 은 (r̂, s) 쌍에 대해
  h' = h − β[(h·r̂) − s]·r̂ 이고, s = μ_succ·r̂ 도 r̂ 와 함께 뒤집히므로

      SetpointSteering(r̂, s) ≡ SetpointSteering(−r̂, −s)      (완전히 같은 연산)

  이다 — 즉 NPZ 의 부호를 통째로 뒤집어 저장해도 serve 의 동작은 **한 치도 안 바뀐다**.
  reverse/control arm (일부러 실패 쪽으로 미는 대조군) 을 원하면 부호가 아니라
  **setpoint 를 실패 평균 좌표로 바꿔야** 한다: s_t := μ_fail(h_t)·r̂ (r̂ 는 그대로).
  부호가 실제로 의미를 갖는 곳은 setpoint 없는 순수 가산형(h' = h − β·r̂) 뿐이다.

  setpoint  s_t = mean_succ(h_t) · r̂                       (토큰 위치별, 성공 평균의 r̂ 좌표)

  ★ setpoint 는 **raw(잔차화 전) 성공 평균**에서 계산한다. 잔차화 모드의 r̂ 는 제거된
  부분공간 Q 와 직교하므로 h·r̂ = (h − QQᵀh)·r̂ — 즉 온라인에서 잔차화를 하지 않아도
  사영값이 같다. §5 "고정 벡터 하나만 적용" 설계가 성립하는 이유가 이것이다.

  ★ setpoint 표본은 **r̂ 와 똑같이 혼재 scene 으로 한정**한다 (전승/전패 scene 제외).
  전패 scene 은 성공 표본이 아예 없고 전승 scene 은 그 scene 의 위치 성분을 μ_succ 에만
  얹으므로, 표본을 다르게 두면 s_t 와 gap 이 scene 구성에 끌려간다 (32 §4.5 와 같은 논리).

  ⚠ setpoint 신뢰도: 32 §4 의 setpoint 부실 경고(신뢰도 0.708)를 메타에 그대로 기록한다.
  연산자 최종형(setpoint vs 순수 가산)은 미정이므로 NPZ 메타에 **양쪽 정보**를 남긴다
  (`operator_modes`). 현재 serve(SetpointSteering)는 setpoint 형만 배선돼 있다.

## 세그먼트 확장

read 는 token-agg(기본 future) 로 방향을 뽑지만 serve 적용은 **세그먼트별**(state[0:1] /
future[1:33] / action[33:49]) 이다. 그래서 같은 r̂ 를 setM v2 계약의 `alpha0_v_seg` [S,D]
로 3행 모두 채우고(로더가 각 행 단위벡터를 요구한다), 적용 여부는 `alpha0_seg_mask` 의
**게인 승수**(0=스킵)로 준다 — exp4-1 `setM_future_only` arm 과 동일한 관례.

## fold (exp5-3 denoise-seed fold 규약)

fold k = `inference_seed == k×1e6` 인 판 전부를 fit 에서 제외 (160판 → 140판).
⚠ **인자 규약 정정**: 실측 meta 의 inference_seed 는 {0, 1e6, …, 7e6} 이라 "0=전판" 을
쓸 수 없다(fold 0 이 실재). 그래서 전판 fit 은 `-1`(기본값) 이고, fold 지정은
`--exclude-fold k` (k×1e6 제외) 또는 `--exclude-inference-seed <raw seed>` 다.

## fit 판 신원 목록 (`fit_identities`) — eval 중복 검사 절차

fit 에 실제로 들어간 판의 `(scenario_seed, inference_seed)` 쌍 전체를 `fit_meta.json` 의
`fit_identities` 에 저장한다. **eval 을 걸기 전에 `fit_identities` 와 eval 계획의
(scenario_seed, inference_seed) 집합의 교집합이 ∅ 인지 반드시 확인한다** (in-sample rescue
아티팩트 방지 — memory `multilayer-conceptor-steering-works`).

이 fit 은 **결정적**이다 (난수 없음) — 그래서 `--seed` 인자는 없다. 같은 입력·같은 인자면
NPZ sha256 이 재현된다 (`fit_meta.npz_sha256_12`).

사용:
    python scripts/scene_sae/fit_g3_direction.py \\
        --x  .../inputs/X_L12.npz --meta .../inputs/meta.npz \\
        --layer 12 --cell scene_matched_drawer_right --window 38 \\
        --residual-rank 16 --token-agg future --apply-segs future

합성 자기검증:  tests/test_g3_fit.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from g2_residual_read import (  # noqa: E402  (같은 디렉토리 — G2 관례 재사용)
    AGG_SEGS,
    RowGrouping,
    between_scatter_basis,
    project_out,
    record_vectors,
    stream_group_sums,
)

# 토큰 세그먼트 (N1.5 DiT T=49) — fit_phase_conceptor_n15 / fit_mean_diff SEGMENTS 규약.
SEGMENTS = (("state", 0, 1), ("future", 1, 33), ("action", 33, 49))

DEFAULT_OUT_ROOT = (REPO / "outputs" / "eval" / "robocasa" / "groot_n15" / "scene_sae"
                    / "g3_directions")

# 32 §4 경고를 산출물에 박아 둔다 (하류에서 setpoint 를 무비판 사용하지 않게)
SETPOINT_CAVEAT = ("32 §4: setpoint 부실 경고(신뢰도 0.708) — s_t 는 성공 평균의 r̂ 좌표일 뿐 "
                   "'성공 상태' 의 충분통계가 아니다. 연산자 최종형 미정(setpoint vs 순수 가산).")


# ─────────────────────────────────────────────────────────────── 방향 fit 코어
def scene_directions(U, y, sc):
    """혼재 scene 별 d_s = mean(fail) − mean(succ) → {scene: d_s}.

    ★부호 규약: fail − succ (= 실패 방향). G2 read 의 `within_dir` 은 succ − fail 이므로
    정확히 부호 반대다 — 여기서는 serve/setM 계약(fit_mean_diff.mean_diff)에 맞춘다.
    혼재(succ·fail 공존) scene 만 쓴다: 전패/전승 scene 은 within-scene 대조가 없어
    scene 구조를 방향에 섞는다 (32 §4.5 "within-scene 이득의 실질은 전패 scene 배제").
    """
    out = {}
    for s in np.unique(sc):
        m = sc == s
        ns, nf = int((y[m] == 1).sum()), int((y[m] == 0).sum())
        if ns == 0 or nf == 0:
            continue
        out[int(s)] = U[m & (y == 0)].mean(0) - U[m & (y == 1)].mean(0)
    return out


def unit(v):
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise SystemExit("방향 노름 0 — d_s 평균이 영벡터")
    return v / n


def fit_direction(U, y, sc, rank, r_max_basis=None):
    """(r̂ [D], 진단 dict). rank>0 이면 scene between 부분공간 상위 r 제거 후 같은 절차.

    부분공간은 **fit 에 쓰는 전 scene** 의 평균 행렬에서 뽑는다 (offline 고정 벡터 산출이
    목적이므로 LOSO 불필요 — 32 §5. G2 의 fold 내 재추정은 *read 판정*용 계약이었다).
    """
    Q = np.zeros((U.shape[1], 0))
    if rank > 0:
        Q = between_scatter_basis(U, sc, int(rank))
        U = project_out(U, Q)
    d = scene_directions(U, y, sc)
    if len(d) < 2:
        raise SystemExit(f"혼재 scene {len(d)}개 — 방향 fit 불가 (≥2 필요)")
    r = unit(np.mean(np.stack(list(d.values())), axis=0))
    cos = {int(s): float(unit(v) @ r) for s, v in d.items()}
    vals = np.asarray(sorted(cos.values()))
    # scene 간 **쌍별** cos — 32 §4.8-[4-1] 의 "scene 별 실패방향 cos 평균"(0.398) 과 같은 자.
    # (r̂ 대비 cos 은 평균벡터 쪽으로 편향돼 항상 더 크다 — 두 값을 구분해 기록한다.)
    Dn = np.stack([unit(v) for v in d.values()])
    G = Dn @ Dn.T
    iu = np.triu_indices(len(Dn), k=1)
    pair = G[iu]
    # 전역(scene 무시) mean-diff — 32 §4.5 "전역 방향 ≈ 난이도 축" 대조용
    r_glob = unit(U[y == 0].mean(0) - U[y == 1].mean(0))
    diag = {
        "n_scene_mixed": len(d),
        "scenes": sorted(d),
        "cos_scene_dir_vs_rhat": {"mean": float(vals.mean()), "median": float(np.median(vals)),
                                  "min": float(vals.min()), "max": float(vals.max()),
                                  "per_scene": cos},
        "cos_scene_pairwise": {"mean": float(pair.mean()), "median": float(np.median(pair)),
                               "min": float(pair.min()), "max": float(pair.max()),
                               "n_pairs": int(len(pair)),
                               "note": "32 §4.8-[4-1] 과 같은 통계량(쌍별). r̂ 대비 cos 과 구분"},
        "cos_rhat_vs_global_meandiff": float(r @ r_glob),
        "residual_rank": int(rank),
        "residual_basis_rank_actual": int(Q.shape[1]),
        "cos_rhat_vs_removed_basis_max": (float(np.abs(Q.T @ r).max()) if Q.shape[1] else 0.0),
    }
    return r, diag, Q


# ───────────────────────────────────────────────────────────── setpoint / 게인
def token_class_means(X, meta, rows_mask, T, batch=200_000):
    """행 마스크 안에서 **토큰 위치별** 평균 [T, D] (raw 표현). 없으면 NaN 행."""
    tok = meta["token_idx"]
    out = np.full((T, X.shape[1]), np.nan, dtype=np.float64)
    for t in range(T):
        idx = np.nonzero(rows_mask & (tok == t))[0]
        if idx.size == 0:
            continue
        acc = np.zeros(X.shape[1], dtype=np.float64)
        for i in range(0, idx.size, batch):
            acc += np.asarray(X[idx[i:i + batch]], dtype=np.float64).sum(axis=0)
        out[t] = acc / idx.size
    return out


def window_record_counts(meta, episodes, window):
    """episode → 창 `[0, window)` 안 **고유 record 수**.

    창이 실제로 꽉 차는지(= 길이 confound 가 창에서 이미 제거됐는지) 확인용. 현 데이터는
    window 가 cell 최단 episode 길이(cap)로 잡혀 있어 전 episode 가 window 를 꽉 채운다.
    """
    ep_rows, rec_rows = meta["episode_idx"], meta["record_idx"]
    inwin = rec_rows < int(window)
    out = {}
    for e in episodes:
        m = inwin & (ep_rows == e)
        out[int(e)] = int(len(np.unique(rec_rows[m])))
    return out


def seg_of_token(t: int) -> int:
    for i, (_n, lo, hi) in enumerate(SEGMENTS):
        if lo <= t < hi:
            return i
    raise ValueError(f"token {t} 세그먼트 밖 (T 계약 위반)")


def save_setm_npz(out_dir: Path, layer: int, v_seg, s_tok, seg_mask, meta: dict) -> Path:
    """setM v2 계약 NPZ (`fit_mean_diff.save_segment_npz` 와 동일 키·동일 디렉토리 구조).

    키: alpha0_v_seg [S,D] · alpha0_s_tok [T] · alpha0_seg_bounds [S,2] · alpha0_seg_mask [S]
        (+ g3 전용 부가키 g3_r_hat [D] — 로더는 무시)
    경로: <out_dir>/steer/dit_L{layer}/conceptors.npz  (serve permanent 규약)
    ★ seg_mask 는 0/1 플래그가 아니라 **세그먼트별 게인 승수** (0=스킵). serve
      `SetpointSteering._apply_segment` 가 delta 에 그대로 곱한다.
    """
    d = out_dir / "steer" / f"dit_L{layer}"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        d / "conceptors.npz",
        alpha0_v_seg=np.asarray(v_seg, dtype=np.float32),
        alpha0_s_tok=np.asarray(s_tok, dtype=np.float32),
        alpha0_seg_bounds=np.asarray([[lo, hi] for _n, lo, hi in SEGMENTS], dtype=np.int32),
        alpha0_seg_mask=np.asarray(seg_mask, dtype=np.float32),
        g3_r_hat=np.asarray(v_seg[0], dtype=np.float32),
    )
    (d / "metadata.json").write_text(json.dumps(
        {**meta, "selected_alpha": 0, "op": "setpoint_seg",
         "segments": [s[0] for s in SEGMENTS],
         "seg_mask": [float(x) for x in np.asarray(seg_mask).tolist()]},
        indent=2, ensure_ascii=False))
    return d / "conceptors.npz"


# ─────────────────────────────────────────────────────────────────────── 실행
def build_units(X, meta, args):
    """행 → 읽기 단위 (U, y, sc, ep, 부가진단).

    `--scene-agg episode` (기본) = G2 read 의 1차 단위(episode × 창평균)와 동일 —
    [4-1] cos 진단치가 이 단위에서 나왔으므로 비교 가능성을 위해 기본으로 둔다.
    `--scene-agg record` = per-record 유지(창 안 record 를 그대로 표본으로). 두 단위의
    r̂ cos 은 항상 진단에 병기한다 (memory feedback-no-rollout-pooling: record 축이
    사라지지 않았음을 보이는 근거).
    """
    segs = AGG_SEGS[args.token_agg]
    grp = RowGrouping(meta, args.window)
    sums = stream_group_sums(X, grp, {"raw": (lambda r: r)}, args.agg_batch_rows,
                             verbose=not args.quiet)
    V, ep, rec = record_vectors(sums["raw"], grp, segs)

    ep_lab = {}
    for e in np.unique(meta["episode_idx"]):
        m = meta["episode_idx"] == e
        ep_lab[int(e)] = {"success": int(meta["success"][m][0]),
                          "scenario_seed": int(meta["scenario_seed"][m][0]),
                          "inference_seed": int(meta["inference_seed"][m][0])}
    eps = np.unique(ep)
    wm = np.stack([V[ep == e].mean(0) for e in eps])
    return V, ep, rec, eps, wm, ep_lab


def units_of(mode, V, ep, wm, eps, ep_lab, keep_eps):
    """(U, y, sc) — mode 별 표본 단위. keep_eps 에 있는 episode 만."""
    if mode == "episode":
        m = np.asarray([int(e) in keep_eps for e in eps])
        U = wm[m]
        src = eps[m]
    else:
        m = np.asarray([int(e) in keep_eps for e in ep])
        U = V[m]
        src = ep[m]
    y = np.asarray([ep_lab[int(e)]["success"] for e in src])
    sc = np.asarray([ep_lab[int(e)]["scenario_seed"] for e in src])
    return U, y, sc


def main() -> int:
    ap = argparse.ArgumentParser(description="G3 — within-scene 실패축 r̂ fit (setM NPZ 산출)")
    ap.add_argument("--x", type=Path, required=True, help="build_sae_inputs 산출 X_L{L}.npz")
    ap.add_argument("--meta", type=Path, required=True, help="같은 빌드의 meta.npz")
    ap.add_argument("--layer", type=int, required=True, help="물리 DiT layer (NPZ 경로·메타)")
    ap.add_argument("--cell", required=True, help="예: scene_matched_drawer_right")
    ap.add_argument("--window", type=int, default=38, help="record_idx < N (drawer_right cap=38)")
    ap.add_argument("--residual-rank", type=int, default=0,
                    help="0=잔차화 없음(기본). r>0 이면 scene between 상위 r PC 제거 후 fit")
    ap.add_argument("--token-agg", default="future", choices=sorted(AGG_SEGS),
                    help="방향 추정용 record 집계 세그먼트 (기본 future — G1 근거)")
    ap.add_argument("--apply-segs", default="future",
                    help="serve 적용 세그먼트 콤마목록 (나머지는 seg_mask 0=스킵). 예 'future,action'")
    ap.add_argument("--scene-agg", default="episode", choices=["episode", "record"],
                    help="scene 내 표본 단위 (episode=창평균, G2 read 1차 / record=per-record)")
    ap.add_argument("--exclude-fold", type=int, default=-1,
                    help="exp5-3 fold k → inference_seed==k×1e6 판 제외. -1=전판 fit(기본)")
    ap.add_argument("--exclude-inference-seed", type=int, default=-1,
                    help="raw inference_seed 값 제외 (--exclude-fold 와 배타). -1=없음")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help=f"기본 {DEFAULT_OUT_ROOT}/<cell>/L{{L}}_r{{rank}}_w{{win}}_f{{fold}}")
    ap.add_argument("--allow-legacy-inputs", action="store_true",
                    help="X/meta 한쪽에 row_fingerprint 가 없는 구 빌드 허용 (기본 = 에러). "
                         "쓰면 fit_meta 에 기록된다")
    ap.add_argument("--agg-batch-rows", type=int, default=200_000)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.exclude_fold >= 0 and args.exclude_inference_seed >= 0:
        raise SystemExit("--exclude-fold 와 --exclude-inference-seed 는 배타")
    apply_segs = [s.strip() for s in args.apply_segs.split(",") if s.strip()]
    seg_names = [s[0] for s in SEGMENTS]
    bad = [s for s in apply_segs if s not in seg_names]
    if bad:
        raise SystemExit(f"--apply-segs 는 {seg_names} 중에서: {bad}")

    mz = np.load(args.meta, allow_pickle=False)
    n = len(mz["episode_idx"])
    meta = {k: mz[k] for k in mz.files if mz[k].ndim == 1 and mz[k].shape[0] == n}
    xz = np.load(args.x)
    X = xz["X"]
    # ---- 입력 정합성: 행 수 + row_fingerprint (한쪽만 있으면 대조 불가 → 기본 에러)
    if X.shape[0] != n:
        raise SystemExit(f"행 수 불일치 — X={X.shape[0]} vs meta={n} (같은 빌드인지 확인)")
    fp_x = str(xz["row_fingerprint"]) if "row_fingerprint" in xz.files else None
    fp_m = str(mz["row_fingerprint"]) if "row_fingerprint" in mz.files else None
    if fp_x and fp_m:
        if fp_x != fp_m:
            raise SystemExit(f"row_fingerprint 불일치 — X={fp_x} vs meta={fp_m}")
        fp_status = "both"
    elif not fp_x and not fp_m:
        fp_status = "none"
    else:
        fp_status = f"only_{'x' if fp_x else 'meta'}"
    if fp_status != "both" and not args.allow_legacy_inputs:
        raise SystemExit(
            f"row_fingerprint 결측({fp_status}) — X/meta 대조 불가. 같은 빌드 산출물을 쓰거나 "
            "의도한 구 빌드라면 --allow-legacy-inputs")
    T = int(meta["token_idx"].max()) + 1
    if T != SEGMENTS[-1][2]:
        raise SystemExit(f"T={T} 이 세그먼트 계약 {SEGMENTS[-1][2]} 과 불일치")

    V, ep, rec, eps, wm, ep_lab = build_units(X, meta, args)

    # ---- fold 제외 (fit 표본 확정)
    excl_seed = (args.exclude_fold * 1_000_000 if args.exclude_fold >= 0
                 else args.exclude_inference_seed)
    if excl_seed >= 0:
        keep_eps = {int(e) for e in eps if ep_lab[int(e)]["inference_seed"] != excl_seed}
        dropped = sorted(int(e) for e in eps if int(e) not in keep_eps)
        if not dropped:
            raise SystemExit(f"inference_seed={excl_seed} 인 판이 없다 — fold 인자 확인")
    else:
        keep_eps = {int(e) for e in eps}
        dropped = []
    fold_tag = "all" if excl_seed < 0 else str(args.exclude_fold if args.exclude_fold >= 0
                                               else excl_seed)

    U, y, sc = units_of(args.scene_agg, V, ep, wm, eps, ep_lab, keep_eps)
    r_hat, diag, Q = fit_direction(U, y, sc, args.residual_rank)

    # 대조 진단 ① 잔차화 유/무 r̂ 간 cos ([4-1] 항목), ② 다른 scene-agg 단위와의 cos
    r_ref, diag_ref = (r_hat, diag)
    if args.residual_rank > 0:
        r_ref, diag_ref, _ = fit_direction(U, y, sc, 0)
    alt = "record" if args.scene_agg == "episode" else "episode"
    Ua, ya, sca = units_of(alt, V, ep, wm, eps, ep_lab, keep_eps)
    r_alt, _diag_alt, _ = fit_direction(Ua, ya, sca, args.residual_rank)

    # ---- 창 충족 가드: 창 안 record 가 전 episode 에서 꽉 차는가 (길이 confound 점검)
    wcnt = window_record_counts(meta, sorted(keep_eps), args.window)
    succ_of = {e: ep_lab[e]["success"] for e in wcnt}
    short = sorted(e for e, c in wcnt.items() if c < int(args.window))
    win_stats = {}
    for nm, lab in (("succ", 1), ("fail", 0)):
        vv = np.asarray([c for e, c in wcnt.items() if succ_of[e] == lab], dtype=float)
        win_stats[nm] = ({"n_episodes": 0} if vv.size == 0 else
                         {"n_episodes": int(vv.size), "min": float(vv.min()),
                          "median": float(np.median(vv)), "mean": float(vv.mean()),
                          "max": float(vv.max()),
                          "n_below_window": int((vv < int(args.window)).sum()),
                          "hist": {str(int(k)): int(c) for k, c in
                                   zip(*np.unique(vv, return_counts=True))}})
    window_full = not short
    if not window_full:
        print(f"  [warn] 창 미달 episode {len(short)}개 (window={args.window}) — "
              f"창 평균이 길이에 끌릴 수 있다. 예: "
              f"{[(e, wcnt[e]) for e in short[:5]]}", file=sys.stderr)

    # ---- setpoint: **raw 성공 평균**의 토큰별 r̂ 좌표 (잔차화해도 사영값 동일 — docstring)
    #      표본은 r̂ 와 **동일하게 혼재 scene 한정** (전승/전패 scene 제외 — docstring)
    ep_rows = meta["episode_idx"]
    mixed_scenes = set(diag["scenes"])
    sp_eps = sorted(e for e in keep_eps if ep_lab[e]["scenario_seed"] in mixed_scenes)
    keep_rows = np.isin(ep_rows, np.asarray(sp_eps))
    win_rows = keep_rows & (meta["record_idx"] < int(args.window))
    mu_succ = token_class_means(X, meta, win_rows & (meta["success"] == 1), T,
                                batch=args.agg_batch_rows)
    mu_fail = token_class_means(X, meta, win_rows & (meta["success"] != 1), T,
                                batch=args.agg_batch_rows)
    if np.isnan(mu_succ).any() or np.isnan(mu_fail).any():
        raise SystemExit("창 안에 표본이 없는 토큰 위치 존재 — window/데이터 확인")
    s_tok = mu_succ @ r_hat
    gap_tok = (mu_fail - mu_succ) @ r_hat              # >0 이어야 r̂ 가 실패 방향

    # ---- 세그먼트 확장: v_seg 3행 모두 r̂ (로더가 행별 단위벡터 요구) + mask 로 적용 제어
    v_seg = np.stack([r_hat] * len(SEGMENTS))
    seg_mask = np.asarray([1.0 if nm in apply_segs else 0.0 for nm in seg_names])
    seg_gap = {nm: float(gap_tok[lo:hi].mean()) for nm, lo, hi in SEGMENTS}
    seg_s = {nm: [float(s_tok[lo:hi].min()), float(s_tok[lo:hi].max())]
             for nm, lo, hi in SEGMENTS}

    out_dir = args.out_dir or (DEFAULT_OUT_ROOT / args.cell /
                               f"L{args.layer}_r{args.residual_rank}_w{args.window}_f{fold_tag}")
    out_dir = Path(out_dir)
    fit_eps = sorted(keep_eps)
    fit_meta = {
        "operator": "g3_setM_within_scene",
        "cell": args.cell, "layer": int(args.layer), "window": int(args.window),
        "token_agg": args.token_agg, "scene_agg": args.scene_agg,
        "residual_rank": int(args.residual_rank),
        "apply_segs": apply_segs, "seg_mask": seg_mask.tolist(),
        "sign_convention": ("r̂ = normalize(mean_scene[mean(fail) − mean(succ)]) = **실패 방향**. "
                            "성공 쪽으로 미는 것은 −r̂. serve h'=h−β[(h·r̂)−s]r̂ 와 정합 "
                            "(exp4-1 fit_mean_diff.mean_diff 와 같은 부호). "
                            "★ setpoint 형에서는 (r̂,s) 를 통째로 뒤집어도 같은 연산이다 "
                            "(s=μ_succ·r̂ 도 같이 뒤집힘) → 부호는 reverse arm 이 아니다."),
        "fold": {"exclude_fold": int(args.exclude_fold),
                 "exclude_inference_seed": int(excl_seed),
                 "n_excluded_episodes": len(dropped), "excluded_episodes": dropped,
                 "note": "exp5-3 fold 규약 = inference_seed==k×1e6 제외. -1=전판 fit "
                         "(inference_seed 0 이 실재하므로 '0=전판' 규약은 쓸 수 없다)"},
        "n_fit_episodes": len(fit_eps), "fit_episodes": fit_eps,
        # ★ eval 전 중복 검사용 판 신원 목록 — eval 계획의 (scenario_seed, inference_seed)
        #   집합과 교집합이 ∅ 인지 확인할 것 (in-sample rescue 아티팩트 방지).
        "fit_identities": [[ep_lab[e]["scenario_seed"], ep_lab[e]["inference_seed"]]
                           for e in fit_eps],
        "fit_identities_note": ("fit 에 실제로 들어간 (scenario_seed, inference_seed) 쌍 전체. "
                                "eval 실행 전 eval 계획 집합과의 교집합=∅ 확인 필수."),
        "n_fit_units": int(len(y)), "n_fit_succ": int((y == 1).sum()),
        "n_fit_fail": int((y == 0).sum()),
        "fit_scenes": diag["scenes"],
        "determinism": "난수 미사용(결정적) — --seed 인자 없음. 같은 입력·인자 → 같은 npz_sha256_12",
        "window_fill": {
            "window": int(args.window), "all_episodes_full": bool(window_full),
            "n_episodes_below_window": len(short), "episodes_below_window": short[:50],
            "per_class_record_counts": win_stats,
            "note": "창 내 고유 record 수. cap=cell 최단 길이면 전 episode 가 window 를 채운다",
        },
        "inputs_integrity": {"n_rows": int(n), "row_fingerprint_status": fp_status,
                             "allow_legacy_inputs": bool(args.allow_legacy_inputs)},
        "row_fingerprint": fp_m or fp_x,
        "source": {"x": str(args.x), "meta": str(args.meta), "argv": sys.argv},
        "diagnostics": {
            **diag,
            "cos_vs_no_residual_rhat": float(r_hat @ r_ref),
            "no_residual_cos_scene_dir_mean": diag_ref["cos_scene_dir_vs_rhat"]["mean"],
            "cos_vs_alt_scene_agg": {"unit": alt, "cos": float(r_hat @ r_alt)},
            "gap_fail_minus_succ_per_segment": seg_gap,
            "s_tok_range_per_segment": seg_s,
            "gap_all_tokens_positive": bool((gap_tok > 0).all()),
        },
        "setpoint": {"source": "raw(잔차화 전) 성공 토큰평균 · fit 표본 창 내 · **혼재 scene 한정**",
                     "n_setpoint_episodes": len(sp_eps),
                     "setpoint_episodes_excluded_pure_scene":
                         sorted(set(fit_eps) - set(sp_eps)),
                     "sample_note": ("s_tok·gap 표본을 r̂ 와 동일하게 혼재 scene 으로 제한한다 "
                                     "(전승/전패 scene 제외) — 표본이 다르면 s_t 가 scene "
                                     "구성에 끌린다"),
                     "caveat": SETPOINT_CAVEAT,
                     "orthogonality_note": "r̂ ⟂ 제거 부분공간 이므로 h·r̂ 는 온라인 잔차화 "
                                           "없이도 동일 (32 §5 고정벡터 설계의 근거)"},
        "operator_modes": {
            "setpoint": {"formula": "h' = h − β[(h·r̂) − s_t]·r̂",
                         "serve_supported": True,
                         "sign_invariance": ("SetpointSteering(r̂, s) ≡ SetpointSteering(−r̂, −s) "
                                             "— NPZ 부호를 통째로 뒤집어도 연산이 동일하다. "
                                             "부호 뒤집기는 reverse arm 이 **아니다**."),
                         "how_to_reverse": ("실패 쪽으로 미는 대조군은 부호가 아니라 setpoint 를 "
                                            "실패 평균 좌표로 교체해서 만든다: "
                                            "s_t := μ_fail(h_t)·r̂ (r̂ 는 그대로). "
                                            "gap_fail_minus_succ_per_segment 가 그 차이(μ_fail−μ_succ)"
                                            "의 r̂ 좌표라 재구성에 쓸 수 있다."),
                         "wiring": "scripts/serve/steering_hooks.py SetpointSteering "
                                   "(load_steering_segment 로 이 NPZ 를 그대로 읽는다)"},
            "additive": {"formula": "h' = h − β·r̂  (setpoint 없는 순수 가산, 성공 방향 = −r̂)",
                         "serve_supported": False,
                         "sign_invariance": "없음 — 가산형에서는 부호가 곧 방향이다 "
                                            "(−r̂=성공 쪽, +r̂=실패 쪽 = reverse arm).",
                         "wiring": "미배선 — SetpointSteering 은 오차비례 항만 구현. "
                                   "additive 를 쓰려면 hook 에 상수 delta 모드 추가 필요. "
                                   "이 NPZ 에는 alpha0_v_seg(=r̂)·g3_r_hat 이 있으므로 "
                                   "정보는 충분하다 (s_tok 을 무시하면 됨)."},
        },
        "provenance": "docs/steering/32_g2_scene_residual_results.md §5 (온라인 잔차화 포기, "
                      "offline 고정 벡터), §4.8-[4-1] 방향 일관성 / [4-2] 1차원 판정",
    }
    npz_path = save_setm_npz(out_dir, args.layer, v_seg, s_tok, seg_mask,
                             {k: v for k, v in fit_meta.items() if k != "fit_episodes"})
    fit_meta["npz_sha256_12"] = hashlib.sha256(npz_path.read_bytes()).hexdigest()[:12]
    (out_dir / "fit_meta.json").write_text(json.dumps(fit_meta, indent=2, ensure_ascii=False))

    c = diag["cos_scene_dir_vs_rhat"]
    print(f"\n===== G3 fit ({args.cell} L{args.layer} r{args.residual_rank} "
          f"w{args.window} f{fold_tag}) =====")
    print(f"  fit: episodes={len(fit_eps)} units({args.scene_agg})={len(y)} "
          f"succ/fail={int((y==1).sum())}/{int((y==0).sum())} 혼재scene={diag['n_scene_mixed']}")
    pw = diag["cos_scene_pairwise"]
    print(f"  cos(d_s, r̂): mean={c['mean']:.3f} median={c['median']:.3f} "
          f"min={c['min']:.3f} max={c['max']:.3f}")
    print(f"  cos(d_s, d_s') 쌍별([4-1] 자): mean={pw['mean']:.3f} "
          f"median={pw['median']:.3f} min={pw['min']:.3f} (n={pw['n_pairs']})")
    print(f"  cos(r̂, 전역 mean-diff)={diag['cos_rhat_vs_global_meandiff']:.3f}  "
          f"cos(r̂, 잔차화無 r̂)={float(r_hat @ r_ref):.3f}  "
          f"cos(r̂, {alt}-agg r̂)={float(r_hat @ r_alt):.3f}")
    if args.residual_rank > 0:
        print(f"  제거 부분공간 rank={Q.shape[1]} · max|cos(r̂, Q)|="
              f"{diag['cos_rhat_vs_removed_basis_max']:.2e} (직교 확인)")
    print(f"  세그먼트 갭(fail−succ)@r̂: {seg_gap}  mask={seg_mask.tolist()}")
    print(f"  setpoint 표본: 혼재 scene episode {len(sp_eps)}/{len(fit_eps)} · "
          f"창 충족={'전부' if window_full else f'미달 {len(short)}판'} "
          f"(succ min={win_stats['succ'].get('min')} fail min={win_stats['fail'].get('min')})")
    print(f"  [written] {npz_path}\n            {out_dir/'fit_meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

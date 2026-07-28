#!/usr/bin/env python
"""exp5-1/G3 사전계산 — 방향(A)·차원(B)·범위(C) 3건을 한 번에 낸다.

사양 원문 = repo 루트 `exp5-1_next_computations.txt` §1/§2/§3. G3(실제 steering) 를
돌리기 전에 **확정해야 할 것**을 기존 npz + CPU 만으로 계산한다 (새 수집·새 SAE 학습 없음).

  §1 계산 A — scene 간 실패방향 일관성(쌍별 cos) 을 잔차화 전/후로 비교
              → "잔차화가 steering 입력으로서의 방향을 개선하는가"
  §2 계산 B — r̂ 성분을 순차 제거하며 read 가 언제 붕괴하는가 (1차원 vs 부분공간)
              → 연산자 가족 결정 (성분제거/고정스텝 vs 부분공간·SAE feature 집합)
  §3 계산 C — state/future/action 세그먼트별 실패축 (개입 범위 결정)

## 판정 코어는 이식물이다

`auroc` / `within_dir` / `loso` / `perm_null` / `read_with_null` / `between_scatter_basis`
/ `project_out` / `RowGrouping` / `record_vectors` / `stream_group_sums` 는 전부
`g2_residual_read.py` 에서 **import** 한다 (원본은 exp5-3 `analyze_sm2_ref.py` 이식 —
그쪽 docstring 의 출처 주석을 그대로 유지한다). 여기서 새로 쓰는 것은 그 위의 조합뿐이다.

## 앵커 게이트 (§1-4) — 먼저 통과해야 나머지가 의미 있다

exp5-3 실측 = scene 별 실패방향 쌍별 cos 평균 **0.343**. 같은 자(raw + episode 창평균)로
이 값이 재현되지 않으면 배관이 다른 것이므로 **원인 규명 전 진행 금지**다. 이 스크립트는
token_agg 두 벌(all=K·T 전체 평균 / future)을 모두 내고, 어느 집계가 앵커를 재현하는지
명시한다. 어느 쪽도 재현 못 하면 종료코드 2 로 멈춘다 (`--force-anchor` 로만 강행).

## 부호 규약

계산 A 의 d_s = mean(fail) − mean(succ) (= 실패 방향, `fit_g3_direction.scene_directions`
와 같은 부호). 쌍별 cos 은 부호 반전에 불변이 아니지만 **모든 scene 에 같은 부호 규약**을
쓰므로 raw/잔차화/세그먼트 간 비교는 정합적이다. 계산 B 의 방향은 `within_dir`
(succ − fail, g2 이식) 이며 사영 제거에는 부호가 무관하다.

## 잔차화 범위 (진단 목적)

계산 A/B/C 의 `between_r16` 은 **전 scene 기준 between-scatter 상위 16** 을 제거한다
(`fit_g3_direction.fit_direction` 관례 — offline 방향 산출용이라 LOSO 불필요).
계산 B 의 **read** 는 이와 별개로 LOSO 계약을 지킨다: 각 단계의 r̂ᵢ 는 fold(=held-out
scene) 안에서 잔차 공간을 다시 보고 재추정한다 (전역 r̂ᵢ 재사용 금지 — 사양 §7).
이 사실은 json `settings.residualization_scope` 에도 박아 둔다.

사용:
    python scripts/scene_sae/g3_precalc.py \\
        --x  .../inputs/X_L12.npz --meta .../inputs/meta.npz \\
        --ckpt-dir .../R_L12_m6144_k64_aux0_split_scene \\
        --probe-json .../probe_quick_R_L12_scene_layout.json \\
        --layer 12 --window 38 --out .../g3_precalc_L12.json

합성 자기검증:  tests/test_g3_precalc.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "12")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "12")
os.environ.setdefault("MKL_NUM_THREADS", "12")

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from g2_residual_read import (  # noqa: E402  (같은 디렉토리 — 판정 코어 단일 출처)
    AGG_SEGS,
    RowGrouping,
    between_scatter_basis,
    loso,
    project_out,
    read_with_null,
    record_vectors,
    stream_group_sums,
    within_dir,
)

# ── 사전 등록 판정 임계 (json "criteria" 에 그대로 기록) ────────────────────────
ANCHOR_COS = 0.343       # exp5-3 실측 (사양 §1-4)
ANCHOR_TOL = 0.02
Z_KEEP = 3.0             # g2 와 동일 — null_z > 3 이면 신호 잔존
DCOS_IMPROVE = 0.05      # 계산 A: cos 평균 Δ ≥ +0.05 → 개선, ≤ −0.05 → 하락, 사이 → 증분 없음
N_DEFLATE_STEPS = 4      # 계산 B 사다리 길이 (사양 §2 "3~4회")

SEG_AGGS = ("state", "future", "action")


# ───────────────────────────────────────────────────── 계산 A: 쌍별 cos 통계
def scene_fail_dirs(U, y, sc):
    """혼재 scene 별 d_s = mean(fail) − mean(succ) (정규화 전). {scene: d_s}.

    부호 = `fit_g3_direction.scene_directions` 와 동일 (실패 방향).
    혼재(succ·fail 공존) scene 만 — 전패/전승 scene 은 within-scene 대조가 없다.
    """
    out = {}
    for s in np.unique(sc):
        m = sc == s
        if (y[m] == 1).sum() == 0 or (y[m] == 0).sum() == 0:
            continue
        out[int(s)] = U[m & (y == 0)].mean(0) - U[m & (y == 1)].mean(0)
    return out


def pairwise_cos(U, y, sc):
    """scene 실패방향들의 **쌍별** cos 통계 (사양 §1-2, [4-1] 과 같은 자)."""
    d = scene_fail_dirs(U, y, sc)
    if len(d) < 2:
        return {"n_scene_mixed": len(d), "mean": None, "median": None,
                "min": None, "max": None, "n_pairs": 0, "scenes": sorted(d)}
    Dn = np.stack([v / np.linalg.norm(v) for v in d.values()])
    G = Dn @ Dn.T
    iu = np.triu_indices(len(Dn), k=1)
    p = G[iu]
    return {"n_scene_mixed": len(d), "mean": float(p.mean()), "median": float(np.median(p)),
            "min": float(p.min()), "max": float(p.max()), "n_pairs": int(len(p)),
            "scenes": sorted(d)}


# ────────────────────────────────────────── 계산 B: LOSO 사다리 (차원 반복 검사)
def fold_direction(Vs, y, sc, s):
    """fold s(=held-out scene) 의 방향 — **train scene 행만** (LOSO 계약).

    `within_dir` (succ − fail, g2 이식) 을 그대로 쓰고 단위화한다. 사영 제거에는
    부호가 무관하므로 within_dir 의 부호 규약을 바꾸지 않는다.
    """
    tr = sc != int(s)
    w = within_dir(Vs[tr], y[tr], sc[tr])
    if w is None:
        return None
    n = float(np.linalg.norm(w))
    return None if n == 0.0 else w / n


def residual_scene_mean_ratio(V, y, sc):
    """‖mean_s d_s‖ / mean_s‖d_s‖ — 사다리 퇴화 진단 (아래 `deflation_ladder` 주석 참조).

    이 값이 0 이면 scene-평균 mean-diff 리더가 쓸 신호가 **정확히** 0 이라는 뜻이다.
    """
    d = scene_fail_dirs(V, y, sc)
    if len(d) < 2:
        return None
    Dm = np.stack(list(d.values()))
    den = float(np.linalg.norm(Dm, axis=1).mean())
    return None if den == 0 else float(np.linalg.norm(Dm.mean(0)) / den)


def deflation_ladder(V, y, sc, n_steps, n_perm, rng, keep_dirs=False):
    """r̂ 를 순차 제거하며 read 를 재측정 (사양 §2 문자 그대로).

    ★LOSO 엄수: 단계 i 의 방향은 **fold 별로**, 그리고 **그 fold 의 잔차 공간 안에서**
    재추정한다. `Vfolds[s]` = fold s 전용 잔차 표현 (s 의 행은 제거방향 추정에 안 들어감).
    전역 r̂ᵢ 를 만들어 재사용하지 않는다.

    ⚠ **구조적 퇴화 경고 (실측·수식 양쪽으로 확인)**: 리더(`within_dir`)는 scene 별
    d_s 의 **평균** m 이고, 제거는 r̂=m/‖m‖ 방향의 직교사영이다. 사영은 선형이라
    잔차의 scene 평균은 m − (m·r̂)r̂ = **정확히 0** 이다 (합성·실데이터 모두 ~1e-15).
    즉 이 사다리는 데이터의 실제 차원과 **무관하게** 2단계에서 반드시 붕괴한다 —
    "1차원" 판정은 데이터가 아니라 산술이 만든다. 그래서 `subspace_capture` 를 함께
    낸다 (그쪽이 실제 차원 판정). 단계별 `residual_scene_mean_ratio` 로 퇴화를 실측한다.

    반환 (steps, fold_dirs) — steps[i] = {step, auroc, null_z, ...}.
    """
    scenes = [int(s) for s in np.unique(sc)]
    Vfolds = {s: np.array(V, dtype=np.float64, copy=True) for s in scenes}
    steps, all_dirs = [], []
    for i in range(n_steps + 1):
        res = read_with_null(V, y, sc, n_perm, rng, Vfolds=Vfolds)
        res["step"] = i
        res["n_dirs_removed"] = i
        # 퇴화 실측: fold 별 잔차 공간에서 (train scene 기준) 잔여 scene-평균 비율
        rr = [residual_scene_mean_ratio(Vfolds[s][sc != s], y[sc != s], sc[sc != s])
              for s in scenes]
        rr = [v for v in rr if v is not None]
        res["residual_scene_mean_ratio_mean"] = float(np.mean(rr)) if rr else None
        steps.append(res)
        if i == n_steps:
            break
        dirs = {}
        for s in scenes:
            w = fold_direction(Vfolds[s], y, sc, s)
            if w is None:
                continue
            dirs[s] = w
            Vfolds[s] = Vfolds[s] - np.outer(Vfolds[s] @ w, w)
        # fold 간 방향 일치도 — 이 단계에서 지운 것이 공통 축인지 fold 잡음인지 진단
        if len(dirs) >= 2:
            Dn = np.stack(list(dirs.values()))
            G = Dn @ Dn.T
            iu = np.triu_indices(len(Dn), k=1)
            steps[-1]["removed_dir_cos_across_folds_mean"] = float(G[iu].mean())
        all_dirs.append(dirs if keep_dirs else None)
    return steps, all_dirs


def ladder_verdict(steps):
    """사양 §2 판정문 그대로 (⚠ 위 퇴화 경고와 함께 읽을 것)."""
    z = [s.get("null_z") for s in steps]
    if z[0] is None or z[0] <= Z_KEEP:
        return "no_baseline_separation", "제거 전 read 가 null 수준 — 차원 판정 불가"
    if len(z) < 2 or z[1] is None or z[1] <= Z_KEEP:
        return "one_dimensional", ("r̂₁ 제거로 read 가 null 수준(z≤3)으로 붕괴 → 사양 §2 문면상 "
                                   "**1차원**. ⚠ 단 이 결과는 산술적으로 강제된다 "
                                   "(잔차 scene 평균 ≡ 0) — 실제 차원은 subspace_capture 로 판정")
    if len(z) >= 3 and z[2] is not None and z[2] > Z_KEEP:
        return "subspace", ("2개 이상 제거해도 z>3 잔존 → **부분공간**. 다차원 연산자가 정당 "
                            "(단 DiT 단일 feature steer 는 출력 붕괴 — 집합 단위로)")
    return "two_dimensional_boundary", "r̂₁ 제거 후에는 남고 r̂₂ 제거에서 붕괴 → 경계(≈2차원)"


# ───────────────────────── 계산 B (비퇴화판): held-out scene 방향의 top-k 포착률
def subspace_capture(U, y, sc, k_max=4):
    """LOSO — train scene 방향들의 top-k 부분공간이 **held-out scene 방향**을 얼마나 담나.

    사양 §2 의 질문("실패축이 1차원인가 부분공간인가")을 퇴화 없이 재정식화한 것이다.
    write 연산자 선택과 직결된다: 성분 제거(1방향) 대 부분공간 M = I − Σr̂ᵢr̂ᵢᵀ.

      fold s: D_train = train 혼재 scene 의 **단위** d_s 행렬 → SVD → V_k (상위 k 우특이벡터)
              capture_k(s) = ‖V_kᵀ d̂_s‖²          (d̂_s = held-out scene 단위 방향)
      기대 chance = k/D (등방 랜덤 부분공간).

    k=1 이 이미 높고 증분이 작으면 1차원, k 를 늘릴수록 크게 오르면 부분공간이다.
    """
    d = scene_fail_dirs(U, y, sc)
    if len(d) < 3:
        return {"note": "혼재 scene < 3 — 포착률 계산 불가", "n_scene_mixed": len(d)}
    scenes = sorted(d)
    Dn = {s: d[s] / np.linalg.norm(d[s]) for s in scenes}
    kk = list(range(1, min(k_max, len(scenes) - 1) + 1))
    per = {k: [] for k in kk}
    for s in scenes:
        M = np.stack([Dn[t] for t in scenes if t != s])
        _u, sv, vt = np.linalg.svd(M, full_matrices=False)
        for k in kk:
            P = vt[:k]
            per[k].append(float((P @ Dn[s]) @ (P @ Dn[s])))
    D = U.shape[1]
    out = {"n_scene_mixed": len(d), "dim": int(D),
           "capture_mean": {int(k): float(np.mean(v)) for k, v in per.items()},
           "capture_median": {int(k): float(np.median(v)) for k, v in per.items()},
           "chance": {int(k): float(k / D) for k in kk},
           "train_singular_energy_frac_top_k": None}
    Mall = np.stack([Dn[s] for s in scenes])
    sv = np.linalg.svd(Mall, compute_uv=False)
    e = sv ** 2 / (sv ** 2).sum()
    out["train_singular_energy_frac_top_k"] = {int(k): float(e[:k].sum()) for k in kk}
    out["participation_ratio"] = float((e.sum() ** 2) / (e ** 2).sum())
    return out


def capture_verdict(cap, gain_thresh=0.10):
    """포착률 사다리 → 연산자 가족 판정 (사양 §4 결정표에 넣을 입력)."""
    cm = cap.get("capture_mean")
    if not cm:
        return "undecidable", cap.get("note", "포착률 없음")
    ks = sorted(cm)
    c1, cmax = cm[ks[0]], cm[ks[-1]]
    gain = cmax - c1
    if gain < gain_thresh:
        return "one_dimensional_capture", (f"k=1 포착 {c1:.3f} → k={ks[-1]} {cmax:.3f} "
                                           f"(증분 {gain:.3f} < {gain_thresh}) → 방향 1개로 "
                                           "담을 수 있는 만큼 담았다 = **1차원 연산자로 충분**")
    return "subspace_capture", (f"k=1 포착 {c1:.3f} → k={ks[-1]} {cmax:.3f} "
                                f"(증분 {gain:.3f} ≥ {gain_thresh}) → 방향을 늘릴수록 held-out "
                                "scene 방향을 더 담는다 = **부분공간 연산자 정당**")


# ─────────────────────────────────────────────────────────── 표현(condition) 구성
def episode_vectors(V_rec, ep_rec):
    """record 벡터 → episode 창평균 [n_ep, D] (g2 `episode_views` 의 window_mean 과 동일)."""
    eps = np.unique(ep_rec)
    return eps, np.stack([V_rec[ep_rec == e].mean(0) for e in eps])


def residualize_between(U, sc, rank):
    """전 scene between-scatter 상위 `rank` 제거 (fit_g3_direction 관례 — 진단용)."""
    Q = between_scatter_basis(U, sc, int(rank))
    return project_out(U, Q), Q


# ──────────────────────────────────────────────────────────────────── SAE arm
def sae_transforms(ckpt_dir: Path, probe_json: Path, device: str, batch: int):
    """{name: rows→재구성} — g2 `make_sae_transforms` 경로 재사용 (selective 제거 / 전체)."""
    from g2_residual_read import load_sae_bundle, make_sae_transforms, selective_features

    model, cfg, mu, sd, st_src = load_sae_bundle(ckpt_dir, None)
    idx, info = selective_features(probe_json, "all")
    sets = {"sae_full": np.zeros(0, np.int64), "sae_sel": np.asarray(idx, np.int64)}
    tf = make_sae_transforms(model, mu, sd, sets, device, batch)
    meta = {"ckpt_dir": str(ckpt_dir), "probe_json": str(probe_json), "stats_source": st_src,
            "device": device, "m": int(cfg.get("m", -1)), "k": int(cfg.get("k", -1)),
            "split_col_of_sae": cfg.get("split_col"),
            "split_axis_scene_heldout": bool(cfg.get("split_axis_scene_heldout")),
            "n_selective_removed": int(len(idx)), **info,
            "note": "selective 집합은 SAE train split 기준(LOSO fold 밖) — 진단 목적. "
                    "sae_full(제거 없이 재구성) 은 재구성 손실 통제용 대조"}
    return tf, meta


# ──────────────────────────────────────────────────────────────────────── 출력
def _f(v, nd=3):
    return "  -  " if v is None else f"{v:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="G3 사전계산 — 방향 cos / 차원 사다리 / 세그먼트")
    ap.add_argument("--x", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--ckpt-dir", type=Path, default=None, help="SAE ckpt (조건 (c))")
    ap.add_argument("--probe-json", type=Path, default=None, help="scene-selective feature 목록")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--cell", default="scene_matched_drawer_right")
    ap.add_argument("--window", type=int, default=38)
    ap.add_argument("--residual-rank", type=int, default=16, help="between 잔차화 rank (G2 최고 arm)")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-steps", type=int, default=N_DEFLATE_STEPS)
    ap.add_argument("--seed", type=int, default=424101)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--agg-batch-rows", type=int, default=200_000)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--force-anchor", action="store_true",
                    help="앵커 게이트 실패해도 계속 (기본 = 종료코드 2 로 중단)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    # ── 입력 로드 + 행 fingerprint 대조 (fit_g3_direction 관례)
    mz = np.load(args.meta, allow_pickle=False)
    n = len(mz["episode_idx"])
    meta = {k: mz[k] for k in mz.files if mz[k].ndim == 1 and mz[k].shape[0] == n}
    xz = np.load(args.x)
    X = xz["X"]
    if X.shape[0] != n:
        raise SystemExit(f"행 수 불일치 — X={X.shape[0]} vs meta={n}")
    fp_x = str(xz["row_fingerprint"]) if "row_fingerprint" in xz.files else None
    fp_m = str(mz["row_fingerprint"]) if "row_fingerprint" in mz.files else None
    if not (fp_x and fp_m):
        raise SystemExit("row_fingerprint 결측 — X/meta 대조 불가 (같은 빌드 산출물을 쓸 것)")
    if fp_x != fp_m:
        raise SystemExit(f"row_fingerprint 불일치 — X={fp_x} vs meta={fp_m}")

    grp = RowGrouping(meta, args.window)
    ep_label = {}
    for e in np.unique(meta["episode_idx"]):
        m = meta["episode_idx"] == e
        ep_label[int(e)] = {"success": int(meta["success"][m][0]),
                            "scenario_seed": int(meta["scenario_seed"][m][0])}

    # ── 표현 준비: raw + SAE(selective 제거 / 전체 재구성). X 는 한 번만 읽는다.
    transforms = {"raw": (lambda r: r)}
    sae_meta = None
    if args.ckpt_dir is not None:
        if args.probe_json is None:
            raise SystemExit("--ckpt-dir 를 쓰면 --probe-json 도 필요")
        tf, sae_meta = sae_transforms(args.ckpt_dir, args.probe_json, args.device, args.batch)
        transforms.update(tf)
    print(f"[g3pre] rows(window<{args.window})={grp.n_rows} subgroups={len(grp.starts)} "
          f"transforms={list(transforms)}", flush=True)
    sums = stream_group_sums(X, grp, transforms, args.agg_batch_rows, verbose=not args.quiet)

    # ── (agg, cond) → episode 벡터
    aggs = ["all", "future"] + [s for s in SEG_AGGS if s not in ("future",)]
    U_of, y_of, sc_of, resid_info = {}, {}, {}, {}
    for agg in aggs:
        segs = AGG_SEGS[agg]
        base = {}
        for name, S in sums.items():
            Vr, epr, _rc = record_vectors(S, grp, segs)
            eps, U = episode_vectors(Vr, epr)
            base[name] = U
            if name == "raw":
                y = np.asarray([ep_label[int(e)]["success"] for e in eps])
                sc = np.asarray([ep_label[int(e)]["scenario_seed"] for e in eps])
                y_of[agg], sc_of[agg] = y, sc
        Ur, Q = residualize_between(base["raw"], sc_of[agg], args.residual_rank)
        base[f"between_r{args.residual_rank}"] = Ur
        resid_info[agg] = {"rank_requested": int(args.residual_rank),
                           "rank_actual": int(Q.shape[1])}
        U_of[agg] = base
    conds = ["raw", f"between_r{args.residual_rank}"] + \
            (["sae_sel", "sae_full"] if sae_meta else [])

    # ══════════════════════════════════ 계산 A — 방향 일관성 (쌍별 cos)
    calcA = {}
    for agg in ("all", "future"):
        calcA[agg] = {c: pairwise_cos(U_of[agg][c], y_of[agg], sc_of[agg]) for c in conds}

    # ── 앵커 게이트 (§1-4): raw × 각 집계가 exp5-3 0.343 을 재현하는가
    anchor_hits = {a: (calcA[a]["raw"]["mean"] is not None
                       and abs(calcA[a]["raw"]["mean"] - ANCHOR_COS) <= ANCHOR_TOL)
                   for a in ("all", "future")}
    anchor = {
        "target": ANCHOR_COS, "tol": ANCHOR_TOL,
        "measured": {a: calcA[a]["raw"]["mean"] for a in ("all", "future")},
        "spec_expected_agg": "all(K·T 전체 평균)",
        "hit_by_agg": anchor_hits,
        "passed": bool(anchor_hits["all"] or anchor_hits["future"]),
        "matches_spec_agg": bool(anchor_hits["all"]),
    }
    anchor["diagnosis"] = (
        "사양이 지목한 all(K·T 평균)이 앵커를 재현" if anchor_hits["all"] else
        ("앵커는 재현되나 집계는 **future 세그먼트 평균** 쪽이다 — exp5-3 의 episode vector 가 "
         "전 토큰 평균이 아니라 future 세그먼트 평균(또는 그와 동치인 집계)임을 뜻한다. "
         "fit_g3_direction 의 [4-1] 0.398 은 다른 집계(all)에서 나온 값으로 보인다."
         if anchor_hits["future"] else
         "어느 집계도 0.343±0.02 를 재현하지 못함 — 배관 불일치. 원인 규명 전 진행 금지"))
    print("\n===== 앵커 게이트 (§1-4) =====")
    for a in ("all", "future"):
        print(f"  raw × token_agg={a:6s} 쌍별 cos 평균 = {_f(calcA[a]['raw']['mean'])} "
              f"(target {ANCHOR_COS}±{ANCHOR_TOL}) → {'HIT' if anchor_hits[a] else 'miss'}")
    print(f"  → {anchor['diagnosis']}")
    if not anchor["passed"] and not args.force_anchor:
        print("[abort] 앵커 게이트 실패 — --force-anchor 없이는 진행하지 않는다", file=sys.stderr)
        return 2

    # ══════════════════════════════════ 계산 B — 차원 (LOSO 사다리)
    calcB = {}
    for agg in ("future", "all"):
        for cond in ("raw", f"between_r{args.residual_rank}"):
            U, y, sc = U_of[agg][cond], y_of[agg], sc_of[agg]
            steps, _ = deflation_ladder(U, y, sc, args.n_steps, args.n_perm, rng)
            v, why = ladder_verdict(steps)
            cap = subspace_capture(U, y, sc, k_max=args.n_steps)
            cv, cwhy = capture_verdict(cap)
            calcB[f"{cond}|{agg}"] = {"steps": steps, "verdict": v, "verdict_note": why,
                                      "subspace_capture": cap, "capture_verdict": cv,
                                      "capture_verdict_note": cwhy,
                                      "token_agg": agg, "cond": cond}
            print(f"  [B] {cond}|{agg} 완료 ({time.time()-t0:.0f}s)", flush=True)

    # ══════════════════════════════════ 계산 C — 세그먼트별 실패축
    calcC = {}
    for seg in SEG_AGGS:
        for cond in ("raw", f"between_r{args.residual_rank}"):
            U, y, sc = U_of[seg][cond], y_of[seg], sc_of[seg]
            r = read_with_null(U, y, sc, args.n_perm, rng)
            calcC[f"{cond}|{seg}"] = {"segment": seg, "cond": cond, "read": r,
                                      "cos": pairwise_cos(U, y, sc)}
            print(f"  [C] {cond}|{seg} 완료 ({time.time()-t0:.0f}s)", flush=True)

    # ── 자동 판정 줄
    rk = f"between_r{args.residual_rank}"
    verdicts = {}
    for agg in ("all", "future"):
        c0, c1 = calcA[agg]["raw"]["mean"], calcA[agg][rk]["mean"]
        d = None if (c0 is None or c1 is None) else c1 - c0
        verdicts[f"A|{agg}"] = {
            "delta_cos": d,
            "verdict": ("undecidable" if d is None else
                        "residualization_improves_direction" if d >= DCOS_IMPROVE else
                        "no_increment" if d > -DCOS_IMPROVE else "residualization_degrades"),
        }
    verdicts["B"] = {k: {"verdict": v["verdict"], "note": v["verdict_note"],
                         "capture_verdict": v["capture_verdict"],
                         "capture_note": v["capture_verdict_note"],
                         "ladder_degenerate": bool(
                             (v["steps"][0].get("residual_scene_mean_ratio_mean") or 1) > 1e-8
                             and all((s.get("residual_scene_mean_ratio_mean") or 1) < 1e-8
                                     for s in v["steps"][1:]))}
                     for k, v in calcB.items()}
    seg_z = {s: (calcC[f"raw|{s}"]["read"].get("null_z")) for s in SEG_AGGS}
    ok = {s: (z is not None and z > Z_KEEP) for s, z in seg_z.items()}
    if ok["action"] and not ok["future"]:
        cverd = ("outcome_in_action_only", "outcome 은 action 토큰, scene 은 future → "
                                           "토큰 축에서 분리. 'future 잔차화 + action 개입' 설계 성립")
    elif ok["action"] and ok["future"]:
        cverd = ("outcome_in_both", "action·future 둘 다 유의 — 토큰 축 분리 없음. "
                                    "개입 범위는 세그먼트 z 크기 순으로 선택")
    elif ok["future"]:
        cverd = ("outcome_in_future_same_as_scene", "outcome 이 scene 과 같은 future 토큰에 있다 "
                                                    "→ 깨끗한 토큰 분리 설계 불가")
    else:
        cverd = ("no_segment_signal", "어느 세그먼트도 z>3 미달 — 범위 판정 불가")
    verdicts["C"] = {"verdict": cverd[0], "note": cverd[1], "null_z_raw": seg_z}

    out = {
        "spec": "exp5-1_next_computations.txt §1/§2/§3",
        "cell": args.cell, "layer": int(args.layer), "window": int(args.window),
        "source": {"x": str(args.x), "meta": str(args.meta), "argv": sys.argv,
                   "row_fingerprint": fp_m},
        "settings": {
            "n_perm": int(args.n_perm), "seed": int(args.seed), "n_steps": int(args.n_steps),
            "residual_rank": int(args.residual_rank),
            "episode_vector": f"창 [0,{args.window}) record 평균 × token_agg 세그먼트 평균",
            "token_aggs": aggs,
            "residualization_scope":
                "계산 A/C 의 between_r16 과 SAE selective 제거는 **전 scene 기준** 부분공간을 "
                "쓴다 (방향 진단 목적 — fit_g3_direction 관례). 계산 B 의 read 는 별개로 "
                "LOSO 를 엄수한다: 각 단계 r̂ᵢ 는 fold 별·잔차 공간 내부에서 재추정.",
            "sign_convention": "d_s = mean(fail) − mean(succ) (실패 방향). 계산 B 사영은 "
                               "within_dir(succ−fail) — 제거에는 부호 무관",
            "protocol_source": "g2_residual_read.py (원본 exp5-3 analyze_sm2_ref.py) 이식",
        },
        "criteria": {"anchor_cos": ANCHOR_COS, "anchor_tol": ANCHOR_TOL, "z_keep": Z_KEEP,
                     "delta_cos_improve": DCOS_IMPROVE},
        "n_episode": len(ep_label), "n_rows_window": int(grp.n_rows),
        "residual_basis": resid_info,
        "sae": sae_meta,
        "anchor_gate": anchor,
        "calc_a_direction_cos": calcA,
        "calc_b_dimension_ladder": calcB,
        "calc_c_segment": calcC,
        "verdicts": verdicts,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # ─────────────────────────────────────────────────────────── 사람이 읽는 표
    print("\n===== §1 계산 A — scene 간 실패방향 쌍별 cos =====")
    print(f"{'token_agg':10s} {'condition':16s} {'mean':>7s} {'median':>7s} "
          f"{'min':>7s} {'max':>7s} {'n_scene':>7s}")
    for agg in ("all", "future"):
        for c in conds:
            s = calcA[agg][c]
            print(f"{agg:10s} {c:16s} {_f(s['mean']):>7s} {_f(s['median']):>7s} "
                  f"{_f(s['min']):>7s} {_f(s['max']):>7s} {s['n_scene_mixed']:>7d}")
    for agg in ("all", "future"):
        v = verdicts[f"A|{agg}"]
        print(f"  [판정 A|{agg}] Δcos({rk} − raw) = {_f(v['delta_cos'])} → {v['verdict']}")

    print(f"\n===== §2 계산 B — 차원 사다리 (LOSO, n_perm={args.n_perm}) =====")
    print(f"{'표현':22s} {'step':>4s} {'AUROC':>7s} {'null_mean':>9s} {'null_z':>7s} "
          f"{'p':>6s} {'cos(fold r̂)':>12s} {'resid_mean_ratio':>17s}")
    for k, blk in calcB.items():
        for st in blk["steps"]:
            rr = st.get("residual_scene_mean_ratio_mean")
            print(f"{k:22s} {st['step']:>4d} {_f(st.get('auroc')):>7s} "
                  f"{_f(st.get('null_mean')):>9s} {_f(st.get('null_z')):>7s} "
                  f"{_f(st.get('p')):>6s} "
                  f"{_f(st.get('removed_dir_cos_across_folds_mean')):>12s} "
                  f"{('  -  ' if rr is None else f'{rr:.2e}'):>17s}")
        print(f"  [판정 B|{k}] {blk['verdict']} — {blk['verdict_note']}")
    print("\n  ── 계산 B (비퇴화판) — held-out scene 방향의 top-k 포착률 (LOSO)")
    print(f"{'표현':22s} {'k=1':>7s} {'k=2':>7s} {'k=3':>7s} {'k=4':>7s} "
          f"{'PR':>6s} {'chance(k=1)':>12s}")
    for k, blk in calcB.items():
        cap = blk["subspace_capture"]
        cm = cap.get("capture_mean", {})
        cells = "".join(f"{_f(cm.get(i)):>8s}" for i in (1, 2, 3, 4))
        print(f"{k:22s}{cells} {_f(cap.get('participation_ratio'), 2):>6s} "
              f"{(cap.get('chance') or {}).get(1, float('nan')):>12.2e}")
    for k, blk in calcB.items():
        print(f"  [판정 B-capture|{k}] {blk['capture_verdict']} — {blk['capture_verdict_note']}")

    print(f"\n===== §3 계산 C — 세그먼트별 실패축 =====")
    print(f"{'표현':22s} {'AUROC':>7s} {'null_z':>7s} {'p':>6s} {'cos_mean':>9s} "
          f"{'cos_med':>8s} {'n_scene':>7s}")
    for k, blk in calcC.items():
        r, c = blk["read"], blk["cos"]
        print(f"{k:22s} {_f(r.get('auroc')):>7s} {_f(r.get('null_z')):>7s} "
              f"{_f(r.get('p')):>6s} {_f(c['mean']):>9s} {_f(c['median']):>8s} "
              f"{c['n_scene_mixed']:>7d}")
    print(f"  [판정 C] {verdicts['C']['verdict']} — {verdicts['C']['note']}")
    print(f"\n  앵커 게이트: {'PASS' if anchor['passed'] else 'FAIL'} "
          f"({anchor['diagnosis']})")
    if args.out:
        print(f"  [written] {args.out}")
    print(f"  elapsed {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

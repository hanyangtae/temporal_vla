#!/usr/bin/env python3
"""exp4-1: setM(setpoint형 mean-diff) + setM_pl(라벨순열 위약) fit — cell(instruction)별.

수학 (docs/steering/24a §4.1): r̂ = normalize(μ_fail − μ_succ) **비중심화** per-record fit,
s = μ_succ·r̂ (성공 평균의 r̂ 좌표 = setpoint). 적용은 serve SetpointSteering
(h' = h − β[(h·r̂)−s]r̂). conceptor 가 per-class 중심화로 지우는 평균차 항이 곧 신호.

exp3 fit30 수집 pkl(full-token, 승준 노드)을 소비하므로 **승준 노드에서 실행**
(~/anaconda3/bin/python, remote_compute.sh run-bg). 산출 NPZ 는 소용량 → pull-results.

Layer 선정 (2026-07-22 사용자 결정 — setM 은 activation 평균 분리도 기준):
  cell별 DiT capture layer 전부에 대해
    ① episode 5-fold CV: train fold 로 r̂ fit → held-out fold record 사영 h·r̂ 의 AUROC
       (truncation cap 내 record 만 — 길이 confound 통제)
    ② episode-level 라벨순열 N_PERM 개로 같은 CV → null 분포 → z = (AUROC−μ_null)/σ_null
    ③ episode-cluster bootstrap B 회 → r̂ 각도 안정성 (full-fit r̂ 과의 cos)
  선정 = z-score 최대 layer (동률 시 bootstrap cos 우선). SR·eval episode 무접촉.

산출 (out_root/<cell>/):
  setM/steer/dit_L{B}/conceptors.npz          — alpha0_v_steer [D] + alpha0_s (선정 layer만)
  setM_pl/steer/dit_L{B}/conceptors.npz       — dose-match 동결 순열 1개
  setM_loo/ep{E}/steer/dit_L{B}/conceptors.npz    — fit-풀 구제 대상별 leave-one-out
  setM_pl_loo/ep{E}/steer/dit_L{B}/conceptors.npz — 위 pairing 위약 (같은 순열 라벨)
  layer_sweep.json / metadata.json            — 전 진단 + 사전등록 기록

사용 (승준):
  python fit_mean_diff.py --manifest <task_PPCC_fit.tsv> --cell pq3_ppcc_bread \
    --targets <annotation_t0.tsv> --out-root outputs/eval/robocasa/groot_n15/exp4_1/npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import numpy as np

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # steer/ (fit_phase_conceptor_n15)

from fit_phase_conceptor_n15 import (  # noqa: E402
    FULLTOKEN_MODE,
    load_rollout_fulltoken,
)

N_PERM = 20        # layer-sweep null 분포용 (CV 비용 때문에 소수)
N_PERM_PL = 200    # 위약 후보 풀 (full-fit 만 — 준직교 후보 확보용)
PL_COS_MAX = 0.3   # 위약-처치 |cos| 상한: 정렬(처치 희석)·반정렬(반처치) 모두 불공정
N_BOOT = 200
RNG_SEED = 424101  # exp4-1 고정 (재현)


# ---------------------------------------------------------------------------- fit 원자
def mean_diff(Xs: np.ndarray, Xf: np.ndarray) -> tuple[np.ndarray, float]:
    """비중심화 mean-diff: r̂=normalize(μ_fail−μ_succ), s=μ_succ·r̂."""
    if len(Xs) == 0 or len(Xf) == 0:
        raise ValueError("클래스 표본 0개 — mean-diff 불가")
    mu_s = Xs.mean(axis=0)
    mu_f = Xf.mean(axis=0)
    d = mu_f - mu_s
    n = float(np.linalg.norm(d))
    if n == 0.0:
        raise ValueError("μ_fail == μ_succ — mean-diff 방향 없음")
    r = d / n
    return r, float(mu_s @ r)


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """rank AUROC (pos=fail 사영이 커야 1)."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(np.argsort(allv, kind="mergesort"), kind="mergesort") + 1.0
    r_pos = order[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def episode_records(roll, layer_idx: int, cap: int) -> np.ndarray:
    """rollout 의 layer 사영용 record 행렬 [n_cap, D] — truncation cap 적용 (per-record 유지,
    rollout pooling 금지 — memory feedback-no-rollout-pooling)."""
    X = roll["dit"][:, layer_idx, :]
    return X[:cap]


def episode_phase_records(roll, layer_idx: int, ph: str, dwell_cap: int) -> np.ndarray:
    """episode **전체 길이**에서 phase==ph 인 record 를 앞에서부터 dwell_cap 개까지.

    길이 통제는 phase 별 dwell cap 으로 한다 (2026-07-23 사용자 지적): episode 전역 cap 을
    먼저 걸면 cap 밖의 후반 phase(place·pull 등) record 가 통째로 소실. dwell cap 은
    성공 episode 들의 그 phase 체류 길이 스케일 — 실패의 timeout 체류 과대가중만 제어."""
    X = roll["dit"][:, layer_idx, :]
    idx = [i for i, p in enumerate(roll["phases"]) if p == ph][:dwell_cap]
    return X[idx]


def phase_dwell_caps(rolls, labels, phases) -> dict:
    """phase 별 dwell cap = 성공 episode 체류 길이(>0)의 ceil(μ+1σ). 성공 dwell 없는
    phase 는 미포함 (대조 불가 — 호출부에서 skip)."""
    caps = {}
    for ph in phases:
        dw = [sum(1 for p in r["phases"] if p == ph)
              for r, y in zip(rolls, labels) if y == 1]
        dw = [d for d in dw if d > 0]
        if dw:
            caps[ph] = int(np.ceil(np.mean(dw) + np.std(dw)))
    return caps


def gather_phase(rolls, labels, layer_idx, cls, ph, dwell_cap):
    out, n_eps = [], 0
    for r, y in zip(rolls, labels):
        if y != cls:
            continue
        recs = episode_phase_records(r, layer_idx, ph, dwell_cap)
        if len(recs):
            out.append(recs)
            n_eps += 1
    return (np.concatenate(out, axis=0) if out else np.empty((0, 0))), n_eps


def gather(rolls, labels, layer_idx, cap, cls):
    out = []
    for r, y in zip(rolls, labels):
        if y == cls:
            recs = episode_records(r, layer_idx, cap)
            if len(recs):
                out.append(recs)
    return np.concatenate(out, axis=0) if out else np.empty((0, 0))


def cv_auroc(rolls, labels, layer_idx, cap, rng) -> float:
    """episode 5-fold CV record AUROC (fold 마다 train fit → held-out 사영)."""
    idx = np.arange(len(rolls))
    pos = idx[np.asarray(labels) == 0]  # fail
    neg = idx[np.asarray(labels) == 1]
    rng.shuffle(pos)
    rng.shuffle(neg)
    folds = [([], []) for _ in range(5)]
    for i, e in enumerate(pos):
        folds[i % 5][0].append(e)
    for i, e in enumerate(neg):
        folds[i % 5][1].append(e)
    scores_p, scores_n = [], []
    for k in range(5):
        test = set(folds[k][0]) | set(folds[k][1])
        tr = [i for i in idx if i not in test]
        Xs = gather([rolls[i] for i in tr], [labels[i] for i in tr], layer_idx, cap, 1)
        Xf = gather([rolls[i] for i in tr], [labels[i] for i in tr], layer_idx, cap, 0)
        if len(Xs) == 0 or len(Xf) == 0:
            continue
        r, _s = mean_diff(Xs, Xf)
        for i in test:
            proj = episode_records(rolls[i], layer_idx, cap) @ r
            (scores_p if labels[i] == 0 else scores_n).extend(proj.tolist())
    return auroc(np.asarray(scores_p), np.asarray(scores_n))


# ---------------------------------------------------------------------------- IO
def load_cell_rolls(manifest: Path, cell: str):
    rows = []
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        p = Path(parts[0]).expanduser()
        if not p.is_absolute():
            p = REPO / p
        if p.parent.name != cell:
            continue
        rows.append({"pkl": p, "label": int(parts[1]), "scene": parts[2] if len(parts) > 2 else ""})
    if not rows:
        raise SystemExit(f"manifest 에 cell={cell} 행 없음: {manifest}")
    missing = [str(m["pkl"]) for m in rows if not m["pkl"].exists()]
    if missing:
        raise SystemExit(f"pkl 누락 {len(missing)}개: {missing[:3]}")
    rolls = []
    for m in rows:
        with open(m["pkl"], "rb") as f:
            d = pickle.load(f)
        if d.get("capture_token_mode") != FULLTOKEN_MODE:
            raise SystemExit(f"{m['pkl']}: full-token pkl 아님 ({d.get('capture_token_mode')})")
        r = load_rollout_fulltoken(d, m["pkl"], "mean")
        r["success"] = m["label"]  # manifest 라벨 override (fit_phase_conceptor 관례)
        r["scene"] = m["scene"]
        r["episode_idx"] = int(d.get("episode_idx", -1))
        r["inference_seed"] = int(d.get("inference_seed", -1))
        r["scenario_seed"] = int(d.get("scenario_seed", -1))
        rolls.append(r)
    return rolls


def load_targets(targets_tsv: Path, cell: str):
    """annotation_t0.tsv 에서 이 cell 의 구제 대상 — pool별 (episode_idx, scenario_seed)."""
    lines = targets_tsv.read_text().splitlines()
    header = lines[0].split("\t")
    out = {"fit": [], "eval": []}
    for ln in lines[1:]:
        if not ln.strip():
            continue
        r = dict(zip(header, ln.split("\t")))
        if r["cell"] != cell:
            continue
        out[r["pool"]].append({
            "episode_idx": int(r["episode_idx"]),
            "scenario_seed": int(r["scenario_seed"]),
            "inference_seed": int(r["inference_seed"]),
        })
    return out


def save_setpoint_npz(out_dir: Path, layer_blk: int, v: np.ndarray, s: float, meta: dict,
                      subdir: str | None = "steer"):
    """subdir='steer'=permanent 규약(<base>/steer/dit_L*), None=gated(phase 디렉토리가
    곧 phase 명 — serve 계약 <base>/<phase>/dit_L*, 추가 레벨 금지)."""
    d = (out_dir / subdir if subdir else out_dir) / f"dit_L{layer_blk}"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "conceptors.npz",
                        alpha0_v_steer=v.astype(np.float32), alpha0_s=np.float32(s))
    (d / "metadata.json").write_text(
        json.dumps({**meta, "selected_alpha": 0}, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------- main
GATED_MIN_REC = 50     # phase 등록 최소 record/클래스 — 미달 phase 는 무개
GATED_MIN_EPS = 3
TERMINAL_PHASES = {"open-done", "insert-settle-done"}  # terminal 동치 phase 제외


def fit_gated(args, rolls, labels, out_cell: Path) -> None:
    """setM_gated / setM_gated_placebo — permanent fit 산출물(선정 layer·동결 순열) 전제.

    phase 별 (r̂_ph, s_ph) fit + 유의성 진단 (사용자: 유의성부터 판단).
    quota(record ≥GATED_MIN_REC/클래스·episode ≥GATED_MIN_EPS) 미달 phase 는 **NPZ 미생성 =
    serve 미등록 → identity(무개입)** — 실패 증거 없는 phase 에 global 방향을 씌우는 외삽
    금지 (2026-07-23 사용자 결정). 재실행 시 stale phase 디렉토리 오염 방지 위해
    setM_gated{,_placebo} 를 먼저 삭제.

    placebo (2026-07-23 리뷰 반영): permanent 동결 순열의 phase 재fit 은 phase 부분표본의
    지배축을 물려받아 준직교가 깨짐(실측 cos +0.62/−0.71) → **phase 별 독립 선택** —
    순열 후보 N_PERM_PL 에서 |cos(vs r̂_ph)|≤PL_COS_MAX 필터 후 phase-record dose-match.
    pairing 은 episode 단위(arm 간 동일 episode)라 phase 마다 순열이 달라도 유지. 위약 fit
    불가 phase 는 처치·위약 양쪽 제외(dose 대칭).
    """
    meta_perm = json.loads(
        next((out_cell / "setM_permanent" / "steer").glob("dit_L*/metadata.json")).read_text())
    blk = int(meta_perm["layer"])
    cap = int(meta_perm["cap_records"])
    cap_layers = [int(x) for x in rolls[0]["capture_layers"]]
    li = cap_layers.index(blk)
    labels_arr = np.asarray(labels)

    # 순열 후보 풀 (phase 별 위약 선택용 — permanent 와 같은 RNG 스트림이지만 선택은 독립)
    prng = np.random.default_rng(RNG_SEED + 555)
    perm_pool = []
    for _ in range(N_PERM_PL):
        pl = labels_arr.copy()
        prng.shuffle(pl)
        perm_pool.append(pl.tolist())

    # cos 진단용 global 방향 (배포 폴백 아님 — 미달 phase 는 미등록=identity)
    z_g = np.load(next((out_cell / "setM_permanent" / "steer").glob("dit_L*/conceptors.npz")))
    v_g = z_g["alpha0_v_steer"].astype(np.float64)
    # 재실행 stale phase 디렉토리 방지
    import shutil
    for d in ("setM_gated", "setM_gated_placebo"):
        shutil.rmtree(out_cell / d, ignore_errors=True)

    # phase 집합·dwell cap 은 episode 전체 길이 기준 (전역 cap 미적용 — 후반 phase 보존)
    phases = sorted({p for r in rolls for p in r["phases"]} - TERMINAL_PHASES)
    dwell_caps = phase_dwell_caps(rolls, labels, phases)
    # layer-sweep null 순열 (유의성 진단용, N_PERM 개)
    rng = np.random.default_rng(RNG_SEED + 777)
    null_perms = []
    for _ in range(N_PERM):
        pl = labels_arr.copy()
        rng.shuffle(pl)
        null_perms.append(pl.tolist())

    diag = []
    for ph in phases:
        if ph not in dwell_caps:
            diag.append({"phase": ph, "skipped_identity": True,
                         "skip_reason": "성공 episode dwell 없음 (대조 불가)"})
            print(f"  [{ph}] SKIP(성공 dwell 없음)", flush=True)
            continue
        dcap = dwell_caps[ph]
        Xs, ns_eps = gather_phase(rolls, labels, li, 1, ph, dcap)
        Xf, nf_eps = gather_phase(rolls, labels, li, 0, ph, dcap)
        entry = {"phase": ph, "dwell_cap": dcap,
                 "n_rec_succ": int(len(Xs)), "n_rec_fail": int(len(Xf)),
                 "n_eps_succ": ns_eps, "n_eps_fail": nf_eps}
        quota_ok = (len(Xs) >= GATED_MIN_REC and len(Xf) >= GATED_MIN_REC
                    and ns_eps >= GATED_MIN_EPS and nf_eps >= GATED_MIN_EPS)
        if quota_ok:
            v_ph, s_ph = mean_diff(Xs, Xf)
            proj_f = np.concatenate([episode_phase_records(r, li, ph, dcap) @ v_ph
                                     for r, y in zip(rolls, labels) if y == 0 and
                                     len(episode_phase_records(r, li, ph, dcap))])
            proj_s = np.concatenate([episode_phase_records(r, li, ph, dcap) @ v_ph
                                     for r, y in zip(rolls, labels) if y == 1 and
                                     len(episode_phase_records(r, li, ph, dcap))])
            a_fit = auroc(proj_f, proj_s)  # fit-표본 (참고)
            # 순열 null: 같은 phase 절차를 순열 라벨로
            null = []
            for pl in null_perms:
                try:
                    vp, _sp = mean_diff(gather_phase(rolls, pl, li, 1, ph, dcap)[0],
                                        gather_phase(rolls, pl, li, 0, ph, dcap)[0])
                except ValueError:
                    continue
                pf = np.concatenate([episode_phase_records(r, li, ph, dcap) @ vp
                                     for r, y in zip(rolls, pl) if y == 0 and
                                     len(episode_phase_records(r, li, ph, dcap))] or [np.empty(0)])
                ps_ = np.concatenate([episode_phase_records(r, li, ph, dcap) @ vp
                                      for r, y in zip(rolls, pl) if y == 1 and
                                      len(episode_phase_records(r, li, ph, dcap))] or [np.empty(0)])
                a_n = auroc(pf, ps_)
                if np.isfinite(a_n):
                    null.append(a_n)
            mu_n, sd_n = (float(np.mean(null)), float(np.std(null))) if null else (float("nan"),) * 2
            z = (a_fit - mu_n) / sd_n if null and sd_n > 1e-9 else float("nan")
            entry.update({"fit_auroc": a_fit, "null_mean": mu_n, "null_std": sd_n,
                          "perm_z": z, "cos_vs_global": float(v_ph @ v_g),
                          "s": s_ph, "fallback": False})
            # placebo: phase 별 독립 선택 — 준직교(|cos|≤PL_COS_MAX) 필터 후 dose-match
            ph_rec = np.concatenate(
                [episode_phase_records(r, li, ph, dcap) for r in rolls
                 if len(episode_phase_records(r, li, ph, dcap))], axis=0)
            dose_ref = float(np.median(np.abs(ph_rec @ v_ph - s_ph)))
            cands = []
            for pi, pl in enumerate(perm_pool):
                try:
                    vp_c, sp_c = mean_diff(gather_phase(rolls, pl, li, 1, ph, dcap)[0],
                                           gather_phase(rolls, pl, li, 0, ph, dcap)[0])
                except ValueError:
                    continue
                cands.append({"perm_id": pi, "v": vp_c, "s": sp_c,
                              "cos": float(vp_c @ v_ph),
                              "dose": float(np.median(np.abs(ph_rec @ vp_c - sp_c)))})
            if not cands:
                # 위약 성립 불가 → 처치도 이 phase 제외 (dose 대칭 — 미등록=identity)
                entry.update({"skipped_identity": True,
                              "skip_reason": "placebo 후보 0 (순열 클래스 고갈)"})
                diag.append(entry)
                print(f"  [{ph}] SKIP(위약 불가 → 양쪽 무개입)", flush=True)
                continue
            ortho = [c for c in cands if abs(c["cos"]) <= PL_COS_MAX]
            fb = ""
            if not ortho:
                cands.sort(key=lambda c: abs(c["cos"]))
                ortho = cands[:5]
                fb = f"no-ortho(min|cos|={abs(ortho[0]['cos']):.2f})"
            band = [c for c in ortho if 0.75 * dose_ref <= c["dose"] <= 1.25 * dose_ref]
            pool_c = band if band else ortho
            pool_c.sort(key=lambda c: abs(c["dose"] - dose_ref))
            sel = pool_c[0]
            v_pp, s_pp = sel["v"], sel["s"]
            entry.update({
                "placebo_perm_id": sel["perm_id"],
                "placebo_cos_vs_setm_ph": sel["cos"],
                "placebo_dose_ratio": sel["dose"] / dose_ref if dose_ref else None,
                "placebo_pool": {"n_cand": len(cands), "n_ortho": len(ortho),
                                 "n_in_band": len(band), "fallback": fb or None},
            })
        else:
            # 미달 phase — NPZ 미생성 (미등록 → serve identity, 무개입. 사용자 결정 07-23)
            entry.update({"skipped_identity": True})
            diag.append(entry)
            print(f"  [{ph}] Nrec s/f={entry['n_rec_succ']}/{entry['n_rec_fail']} "
                  f"SKIP(quota 미달 → 무개입)", flush=True)
            continue
        save_setpoint_npz(out_cell / "setM_gated" / ph, blk, v_ph, s_ph, subdir=None, meta={
            "operator": "setM_gated", "cell": args.cell, "phase": ph, "layer": blk,
            "cap_records": cap, **{k: entry[k] for k in entry if k != "phase"},
        })
        save_setpoint_npz(out_cell / "setM_gated_placebo" / ph, blk, v_pp, s_pp, subdir=None, meta={
            "operator": "setM_gated_placebo", "cell": args.cell, "phase": ph, "layer": blk,
            "perm_id": entry["placebo_perm_id"], "cos_vs_setm_ph": entry["placebo_cos_vs_setm_ph"],
            "dose_ratio": entry["placebo_dose_ratio"], "pl_cos_max": PL_COS_MAX,
            "pool": entry["placebo_pool"],
        })
        diag.append(entry)
        print(f"  [{ph}] Nrec s/f={entry['n_rec_succ']}/{entry['n_rec_fail']} "
              f"auroc={entry['fit_auroc']:.3f} z={entry['perm_z']:.2f} "
              f"cos_g={entry['cos_vs_global']:+.2f}", flush=True)

    (out_cell / "setM_gated_diag.json").write_text(json.dumps({
        "cell": args.cell, "layer": blk, "cap_records": cap, "phases": diag,
        "quota": {"min_rec": GATED_MIN_REC, "min_eps": GATED_MIN_EPS},
        "note": "유의성(perm_z)은 fit-표본 AUROC 기준 진단 — phase 별 표본이 작아 CV 생략, "
                "배포 게이트는 quota(폴백)로만. placebo 는 permanent 동결 순열 재사용(pairing).",
    }, indent=2, ensure_ascii=False))
    print(f"[gated done] {out_cell}/setM_gated (+placebo) phases={len(phases)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True, help="fit30 tsv (pkl\\tlabel\\tscene)")
    ap.add_argument("--cell", required=True, help="예: pq3_ppcc_bread (within-instruction fit)")
    ap.add_argument("--targets", type=Path, required=True,
                    help="annotation_t0.tsv — fit-풀 대상은 LOO, eval-풀 대상은 침범 검사")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--layers", default=None,
                    help="sweep 할 DiT 물리 layer 콤마목록 (기본: capture 전부)")
    ap.add_argument("--gated", action="store_true",
                    help="setM_gated/placebo 만 fit (permanent 산출물 전제 — 같은 layer·동결 순열)")
    args = ap.parse_args()

    rng = np.random.default_rng(RNG_SEED)
    rolls = load_cell_rolls(args.manifest, args.cell)
    labels = [r["success"] for r in rolls]
    if args.gated:
        fit_gated(args, rolls, labels, args.out_root / args.cell)
        return
    targets = load_targets(args.targets, args.cell)
    cap_layers = [int(x) for x in rolls[0]["capture_layers"]]
    layer_blks = ([int(x) for x in args.layers.split(",")] if args.layers else cap_layers)
    for b in layer_blks:
        if b not in cap_layers:
            raise SystemExit(f"layer {b} 는 capture {cap_layers} 에 없음")

    # ---- 침범 검사: rollout 정체성 = (scenario_seed, inference_seed). exp3 eval 의
    # seen 블록(ep0-29)은 fit 과 같은 scene·다른 inference_seed 계열(35xx000 vs 10xx000)
    # 이라 episode_idx 비교는 오탐 — 실측 대조로 확정(2026-07-22).
    fit_ids = {(r["scenario_seed"], r["inference_seed"]) for r in rolls}
    fit_scenes = {r["scenario_seed"] for r in rolls}
    bad = [t for t in targets["eval"]
           if (t["scenario_seed"], t["inference_seed"]) in fit_ids]
    if bad:
        raise SystemExit(f"eval-풀 대상이 fit 표본과 동일 rollout(설계 위반): {bad[:3]}")
    # seen(fit scene 재사용)/unseen 층화 플래그 — 집계에서 별도 보고 (scene-level leakage 표기)
    eval_targets_flagged = [
        {**t, "seen_scene": t["scenario_seed"] in fit_scenes} for t in targets["eval"]
    ]
    # fit-풀 대상은 반드시 fit rolls 안에 있어야 LOO 가 성립
    fit_pair_ids = {(r["episode_idx"], r["inference_seed"]) for r in rolls}
    missing_loo = [t for t in targets["fit"]
                   if (t["episode_idx"], t["inference_seed"]) not in fit_pair_ids]
    if missing_loo:
        raise SystemExit(f"fit-풀 대상이 fit 표본에 없음: {missing_loo[:3]}")

    # ---- truncation 표준 (memory truncation-length-standard): cap = ceil(μ+1σ) of succ 길이
    succ_lens = [r["length"] for r, y in zip(rolls, labels) if y == 1]
    if not succ_lens:
        raise SystemExit("성공 episode 0개 — fit 불가")
    cap = int(np.ceil(np.mean(succ_lens) + np.std(succ_lens)))
    cap_mean = int(np.ceil(np.mean(succ_lens)))

    n_s = sum(labels)
    print(f"[{args.cell}] rollouts={len(rolls)} succ={n_s} fail={len(rolls)-n_s} "
          f"cap={cap} (μ={cap_mean}) layers={layer_blks} "
          f"targets fit={len(targets['fit'])} eval={len(targets['eval'])}", flush=True)

    # ---- layer sweep: CV AUROC + permutation null z + bootstrap 각도 안정성
    sweep = []
    perm_labels_list = []
    labels_arr = np.asarray(labels)
    for pi in range(N_PERM):
        perm = labels_arr.copy()
        rng.shuffle(perm)  # episode-level 순열
        perm_labels_list.append(perm.tolist())
    for blk in layer_blks:
        li = cap_layers.index(blk)
        Xs = gather(rolls, labels, li, cap, 1)
        Xf = gather(rolls, labels, li, cap, 0)
        v_full, s_full = mean_diff(Xs, Xf)
        a_cv = cv_auroc(rolls, labels, li, cap, np.random.default_rng(RNG_SEED + blk))
        null = [
            cv_auroc(rolls, pl, li, cap, np.random.default_rng(RNG_SEED + blk + 7919 * (pi + 1)))
            for pi, pl in enumerate(perm_labels_list)
        ]
        null = [x for x in null if np.isfinite(x)]
        mu_n, sd_n = float(np.mean(null)), float(np.std(null))
        z = (a_cv - mu_n) / sd_n if sd_n > 1e-9 else float("nan")
        # bootstrap 각도 안정성 (episode-cluster 재표본 → r̂ 과 full-fit r̂ 의 cos)
        cos_list = []
        idx_s = [i for i in range(len(rolls)) if labels[i] == 1]
        idx_f = [i for i in range(len(rolls)) if labels[i] == 0]
        brng = np.random.default_rng(RNG_SEED + 100 + blk)
        for _ in range(N_BOOT):
            bs = list(brng.choice(idx_s, len(idx_s))) + list(brng.choice(idx_f, len(idx_f)))
            Xsb = gather([rolls[i] for i in bs], [labels[i] for i in bs], li, cap, 1)
            Xfb = gather([rolls[i] for i in bs], [labels[i] for i in bs], li, cap, 0)
            try:
                vb, _ = mean_diff(Xsb, Xfb)
                cos_list.append(float(vb @ v_full))
            except ValueError:
                continue
        sweep.append({
            "layer": blk, "cv_auroc": a_cv, "null_mean": mu_n, "null_std": sd_n,
            "perm_z": z, "boot_cos_mean": float(np.mean(cos_list)),
            "boot_cos_p05": float(np.percentile(cos_list, 5)),
            "s": s_full, "norm_mu_diff": float(np.linalg.norm(Xf.mean(0) - Xs.mean(0))),
            "n_rec_succ": int(len(Xs)), "n_rec_fail": int(len(Xf)),
        })
        print(f"  L{blk}: cv_auroc={a_cv:.3f} null={mu_n:.3f}±{sd_n:.3f} z={z:.2f} "
              f"boot_cos={np.mean(cos_list):.3f} s={s_full:.3f}", flush=True)

    # 선정: perm_z 최대 (동률 ±0.1 이내면 boot_cos 우선). SR 무접촉 — 사전등록 기록.
    best = max(sweep, key=lambda r: (round(r["perm_z"], 1), r["boot_cos_mean"]))
    blk = best["layer"]
    li = cap_layers.index(blk)
    print(f"[select] layer=L{blk} (perm_z={best['perm_z']:.2f})", flush=True)

    # ---- full-fit setM (eval-풀 대상용)
    Xs = gather(rolls, labels, li, cap, 1)
    Xf = gather(rolls, labels, li, cap, 0)
    v_full, s_full = mean_diff(Xs, Xf)

    # ---- setM_pl: 위약 순열 선택 — ① |cos(r̂p, r̂setM)| ≤ PL_COS_MAX (준직교 — 정렬은
    # 처치 희석, 반정렬은 반처치라 양쪽 다 불공정. 실측: 20개 풀은 지배 분산축을 물려받아
    # cos ±0.6~0.8 → 후보 200개로 확장) ② 그중 dose(중앙값 |(h·r̂)−s|)-match ±25% 밴드,
    # 밴드 내 dose-closest. 후보 부족 시 최소-|cos| 폴백(기록).
    all_rec = np.concatenate([episode_records(r, li, cap) for r in rolls], axis=0)
    dose_m = float(np.median(np.abs(all_rec @ v_full - s_full)))
    prng = np.random.default_rng(RNG_SEED + 555)
    perms = []
    for pi in range(N_PERM_PL):
        pl = labels_arr.copy()
        prng.shuffle(pl)
        pl = pl.tolist()
        try:
            vp, sp = mean_diff(gather(rolls, pl, li, cap, 1), gather(rolls, pl, li, cap, 0))
        except ValueError:
            continue
        dp = float(np.median(np.abs(all_rec @ vp - sp)))
        ap_ = auroc(np.asarray([x for r, y in zip(rolls, labels) if y == 0
                                for x in episode_records(r, li, cap) @ vp]),
                    np.asarray([x for r, y in zip(rolls, labels) if y == 1
                                for x in episode_records(r, li, cap) @ vp]))
        perms.append({"perm_id": pi, "v": vp, "s": sp, "dose_median": dp,
                      "cos_setm": float(vp @ v_full), "true_label_auroc": ap_, "labels": pl})
    ortho = [p for p in perms if abs(p["cos_setm"]) <= PL_COS_MAX]
    fallback = ""
    if not ortho:
        perms.sort(key=lambda p: abs(p["cos_setm"]))
        ortho = perms[:5]
        fallback = f"no-ortho(min|cos|={abs(ortho[0]['cos_setm']):.2f})"
    in_band = [p for p in ortho if 0.75 * dose_m <= p["dose_median"] <= 1.25 * dose_m]
    pool = in_band if in_band else ortho
    pool.sort(key=lambda p: abs(p["dose_median"] - dose_m))
    pl_sel = pool[0]
    print(f"[setM_pl] perm_id={pl_sel['perm_id']} cos={pl_sel['cos_setm']:+.3f} "
          f"dose={pl_sel['dose_median']:.3f} (setM {dose_m:.3f}) "
          f"ortho={len(ortho)}/{len(perms)} in_band={len(in_band)} "
          f"true_label_auroc={pl_sel['true_label_auroc']:.3f} {fallback}", flush=True)

    # ---- 저장 (full-fit)
    out_cell = args.out_root / args.cell
    manifest_sha = hashlib.sha256(args.manifest.read_bytes()).hexdigest()[:12]
    targets_sha = hashlib.sha256(args.targets.read_bytes()).hexdigest()[:12]
    base_meta = {
        "operator": "setM_permanent", "cell": args.cell, "layer": blk, "cap_records": cap,
        "cap_mean_alt": cap_mean, "manifest_sha": manifest_sha, "targets_sha": targets_sha,
        "n_rollouts": len(rolls), "n_succ": int(n_s), "n_fail": int(len(rolls) - n_s),
        "s": s_full, "dose_median_fitset": dose_m,
        "layer_sweep_ref": "../layer_sweep.json", "rng_seed": RNG_SEED,
        "eval_targets": eval_targets_flagged,
        "n_eval_seen": sum(t["seen_scene"] for t in eval_targets_flagged),
        "n_eval_unseen": sum(not t["seen_scene"] for t in eval_targets_flagged),
        "note": "s≈0 이면 제거형(I−βr̂r̂ᵀ)과 동치 (24a §4.1); seen_scene 대상은 "
                "fit 이 같은 scene 의 다른 rollout 을 봄 — 집계에서 seen/unseen 층화",
    }
    save_setpoint_npz(out_cell / "setM_permanent", blk, v_full, s_full, base_meta)
    save_setpoint_npz(out_cell / "setM_permanent_placebo", blk, pl_sel["v"], pl_sel["s"], {
        **base_meta, "operator": "setM_permanent_placebo", "perm_id": pl_sel["perm_id"],
        "dose_median": pl_sel["dose_median"], "true_label_auroc": pl_sel["true_label_auroc"],
        "cos_setm": pl_sel["cos_setm"], "pl_cos_max": PL_COS_MAX,
        "dose_ratio_vs_setm": pl_sel["dose_median"] / dose_m if dose_m else None,
        "pl_pool": {"n_perm": N_PERM_PL, "n_ortho": len(ortho), "n_in_band": len(in_band),
                    "fallback": fallback or None},
    })

    # ---- fit-풀 대상 per-target LOO (setM + pairing 위약: 같은 순열 라벨에서 대상 제외)
    for t in targets["fit"]:
        keep = [i for i in range(len(rolls)) if rolls[i]["episode_idx"] != t["episode_idx"]]
        kl = [labels[i] for i in keep]
        kr = [rolls[i] for i in keep]
        v_t, s_t = mean_diff(gather(kr, kl, li, cap, 1), gather(kr, kl, li, cap, 0))
        save_setpoint_npz(out_cell / "setM_permanent_loo" / f"ep{t['episode_idx']}", blk, v_t, s_t, {
            **base_meta, "operator": "setM_permanent_loo", "loo_episode_idx": t["episode_idx"],
            "s": s_t,
        })
        pl_l = [pl_sel["labels"][i] for i in keep]
        try:
            v_pt, s_pt = mean_diff(gather(kr, pl_l, li, cap, 1), gather(kr, pl_l, li, cap, 0))
        except ValueError:
            v_pt, s_pt = pl_sel["v"], pl_sel["s"]  # 순열 클래스 고갈 시 full 위약 재사용
        save_setpoint_npz(out_cell / "setM_permanent_placebo_loo" / f"ep{t['episode_idx']}", blk, v_pt, s_pt, {
            **base_meta, "operator": "setM_permanent_placebo_loo", "loo_episode_idx": t["episode_idx"],
            "perm_id": pl_sel["perm_id"], "s": s_pt,
        })

    (out_cell / "layer_sweep.json").write_text(json.dumps({
        "cell": args.cell, "sweep": sweep, "selected_layer": blk,
        "selection_rule": "max perm_z (round .1 tie → boot_cos_mean)",
        "cap_records": cap, "n_perm": N_PERM, "n_boot": N_BOOT,
        "manifest_sha": manifest_sha, "rng_seed": RNG_SEED,
    }, indent=2, ensure_ascii=False))
    print(f"[done] {out_cell} — setM/setM_pl + LOO {len(targets['fit'])}쌍 "
          f"(layer L{blk})", flush=True)


if __name__ == "__main__":
    main()

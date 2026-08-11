#!/usr/bin/env python3
# 유래: exp5-3 (2026-08-10 기능명 재배치 — docs/review/RENAME_PLAN.md)
"""exp5-3 위약 fit — scene-내 라벨 순열로 전체 within-scene 파이프 재실행.

위약 규약 (exp2/exp4-1 관례의 within-scene 판):
  · 각 LOO fold 에서 **scene 안에서만** 라벨을 순열 → 방향·per-scene setpoint 를 처치와
    동일한 파이프로 fit (라벨만 무작위 — "방향 없는 같은 종류의 개입").
  · 준직교 게이트: 처치 방향과의 |cos| ≤ 0.3 (세그먼트별 최대) — 정렬(희석)·반정렬(반처치)
    모두 불공정 (fit_mean_diff PL_COS_MAX 준용). 후보 순열을 통과할 때까지 재추첨.
  · dose-match: fit 표본에서 위약의 median |proj−s| 가 처치와 일치하도록 setpoint 잔차
    스케일 보정 없이 **방향 norm 은 이미 단위** — dose 는 (proj−s) 크기에서 나오므로
    per-scene setpoint 파이프가 같으면 1차 근사로 유사. 실측 dose 비를 report 에 기록하고
    0.5~2.0 밖이면 경고 (탐색 라운드 — 사후 스케일 보정은 하지 않음).
  · mask: --fut 이면 [0,1,0] (future-only 위약).

입력: fit_within_scene_setM.py 의 캐시(fit_cache_L12.npz) + 처치 registry (cos 게이트용).
출력: placebo(_fut)/loo_seed{k}/scene{S}|steer/dit_L12/conceptors.npz + placebo_report.json
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np

SEGMENTS = (("state", 0, 1), ("future", 1, 33), ("action", 33, 49))
N_SEED = 8
SEED_STEP = 1_000_000
PL_COS_MAX = 0.3
RNG = np.random.default_rng(424101)


def seg_of_token(t):
    for i, (_n, lo, hi) in enumerate(SEGMENTS):
        if lo <= t < hi:
            return i
    raise ValueError


def load_cache(p):
    z = np.load(p, allow_pickle=True)
    eps = []
    for i in range(len(z["label"])):
        eps.append(dict(scene=int(z["scene"][i]), inf=int(z["inf"][i]),
                        label=int(z["label"][i]), M=z["M"][i]))
    return eps, int(z["layer"][0])


def within_fit(eps_tr, labels):
    """labels dict (id(ep)→lab 대신 index 기반): eps_tr 와 병렬 라벨 배열로 fit."""
    by_scene = {}
    for e, lab in zip(eps_tr, labels):
        by_scene.setdefault(e["scene"], []).append((e, lab))
    diffs = []
    for s, grp in sorted(by_scene.items()):
        Ms = [e["M"] for e, l in grp if l == 1]
        Mf = [e["M"] for e, l in grp if l == 0]
        if Ms and Mf:
            diffs.append(np.mean(Mf, axis=0) - np.mean(Ms, axis=0))
    if not diffs:
        return None
    Dbar = np.mean(diffs, axis=0)
    v_seg = []
    for _n, lo, hi in SEGMENTS:
        d = Dbar[lo:hi].mean(axis=0)
        n = np.linalg.norm(d)
        if n == 0:
            return None
        v_seg.append(d / n)
    v_seg = np.stack(v_seg)
    Ms_all = [e["M"] for e, l in zip(eps_tr, labels) if l == 1]
    mu = np.mean(Ms_all, axis=0)
    T = mu.shape[0]
    s_tok = np.asarray([float(mu[t] @ v_seg[seg_of_token(t)]) for t in range(T)])
    # per-scene setpoint (순열 라벨 기준 "성공" 평균)
    s_by_scene = {}
    for s, grp in sorted(by_scene.items()):
        Ms = [e["M"] for e, l in grp if l == 1]
        s_by_scene[s] = (np.asarray([float(np.mean(Ms, axis=0)[t] @ v_seg[seg_of_token(t)])
                                     for t in range(T)]) if Ms else s_tok)
    return v_seg, s_tok, s_by_scene


def dose(eps_tr, labels, v_seg, s_by_scene, s_tok):
    T = eps_tr[0]["M"].shape[0]
    v_tok = np.stack([v_seg[seg_of_token(t)] for t in range(T)])
    r = []
    for e in eps_tr:
        st = s_by_scene.get(e["scene"], s_tok)
        proj = np.asarray([e["M"][t] @ v_tok[t] for t in range(T)])
        r.append(float(np.median(np.abs(proj - st))))
    return float(np.median(r))


def save_npz(path, v_seg, s_tok, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, alpha0_v_seg=v_seg.astype(np.float32),
             alpha0_s_tok=s_tok.astype(np.float32),
             alpha0_seg_bounds=np.asarray([[lo, hi] for _n, lo, hi in SEGMENTS], np.int64),
             alpha0_seg_mask=np.asarray(mask, np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--treat-root", required=True, help="처치 registry (cos 게이트)")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--fut", action="store_true")
    args = ap.parse_args()

    eps, layer = load_cache(Path(args.cache).expanduser())
    mask = [0, 1, 0] if args.fut else [1, 1, 1]
    out_root = Path(args.out_root).expanduser()
    report = {"folds": {}, "mask": mask}

    for k in range(N_SEED):
        eps_tr = [e for e in eps if e["inf"] != k * SEED_STEP]
        y_true = [e["label"] for e in eps_tr]
        # 처치 방향 로드 (cos 게이트 기준)
        tz = np.load(Path(args.treat_root).expanduser() /
                     f"loo_seed{k}/steer/dit_L{layer}/conceptors.npz")
        v_treat = tz["alpha0_v_seg"].astype(np.float64)
        # 처치 dose (동일 파이프 실측)
        tfit = within_fit(eps_tr, y_true)
        d_treat = dose(eps_tr, y_true, tfit[0], tfit[2], tfit[1])

        ok = None
        for trial in range(500):
            # scene 내 라벨 순열
            perm = []
            by_scene_idx = {}
            for i, e in enumerate(eps_tr):
                by_scene_idx.setdefault(e["scene"], []).append(i)
            yp = np.asarray(y_true).copy()
            for s, idxs in by_scene_idx.items():
                yp[idxs] = RNG.permutation(yp[idxs])
            fit = within_fit(eps_tr, list(yp))
            if fit is None:
                continue
            v_pl = fit[0]
            cos = float(np.max(np.abs(np.sum(v_pl * v_treat, axis=1))))
            if cos <= PL_COS_MAX:
                ok = (fit, cos, trial)
                break
        if ok is None:
            raise SystemExit(f"fold {k}: 준직교 순열 못 찾음 (500회)")
        (v_pl, s_tok, s_by_scene), cos, trial = ok
        d_pl = dose(eps_tr, y_true, v_pl, s_by_scene, s_tok)
        ratio = d_pl / (d_treat + 1e-9)
        base = out_root / f"loo_seed{k}"
        save_npz(base / "steer" / f"dit_L{layer}" / "conceptors.npz", v_pl, s_tok, mask)
        for s, st in s_by_scene.items():
            save_npz(base / f"scene{s}" / f"dit_L{layer}" / "conceptors.npz", v_pl, st, mask)
        report["folds"][str(k)] = dict(cos_max=round(cos, 3), trials=trial + 1,
                                       dose_treat=round(d_treat, 3), dose_pl=round(d_pl, 3),
                                       dose_ratio=round(ratio, 3),
                                       warn=bool(ratio < 0.5 or ratio > 2.0))
        print(f"fold{k}: cos={cos:.3f} trial={trial+1} dose비={ratio:.2f}"
              f"{' ⚠' if report['folds'][str(k)]['warn'] else ''}", flush=True)

    (out_root / "placebo_report.json").write_text(json.dumps(report, indent=2))
    print("[written]", out_root)


if __name__ == "__main__":
    main()

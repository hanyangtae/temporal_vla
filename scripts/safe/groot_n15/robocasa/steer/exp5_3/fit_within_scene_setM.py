#!/usr/bin/env python3
"""exp5-3: within-scene mean-diff setM fit — scene-matched drawer 수집(320판)에서
permanent + phase-gated 방향 NPZ 를 LOO-by-seed 8 fold 로 산출.

배경: scene-matched 진단(2026-07-26)에서 drawer_right 의 succ/fail 분리가 scene·길이·dwell·
seed 통제 후에도 실재(L12 held-out AUROC 0.837), 그리고 **within-scene 방향(0.847)이
cross-scene 방향(0.627)보다 우월** — 기존 cross-scene fit 은 scene 구조를 학습하고 있었다.
여기서는 그 within-scene 방향을 steering 연산자(setM v2 세그먼트)로 배포한다.

fit 규약은 exp4-1 fit_mean_diff.py 와 동일 유지 (비교 가능성):
  - 부호: r̂ = normalize(μ_fail − μ_succ), setpoint s_t = 성공 평균의 그 위치 사영
    (적용식 h' = h − β(proj − s)r̂ 가 성공 쪽으로 밈 — steering_hooks._apply_segment).
  - 세그먼트: SEGMENTS = (state 0:1, future 1:33, action 33:49), 방향은 세그먼트별,
    setpoint 는 토큰 위치별.
  - **차이점 (exp5-3 의 요지)**: 방향이 전역 mean-diff 가 아니라 **scene 별
    (μ_fail,scene − μ_succ,scene) 의 혼재-scene 평균** — scene 주효과 상쇄.
  - 길이통제: episode 당 record [0, CAP) 만 (진단과 동일; 실패=timeout 과대가중 차단).
  - denoise K 축은 평균 (serve 기본 --steering-denoise global 과 정합).

in-sample 차단: **leave-one-seed-out** — inference_seed k 열을 평가할 fold 는 seed k 의
episode 전부를 fit 에서 제외 (multilayer in-sample 반전 교훈).

산출 (out-root 아래, serve phase-registry 배치 그대로):
  permanent/loo_seed{k}/scene{S}/dit_L{L}/conceptors.npz   ← 방향=fold 공유 within-scene,
      setpoint=**그 scene 의 성공 활성 평균** (per-scene setpoint — 게이트 진단 근거:
      전역 setpoint 는 scene offset 을 못 따라가 seed-out 비중심 read-out 이 0.59 로 붕괴,
      scene-중심 read-out 은 0.83 유지. exp2 의 per-scene fit 정신과 일치. fold 에 그 scene
      성공이 없으면 전역 setpoint 로 fallback 채움 — 미등록 무음 off 방지, report 에 기록).
      client 는 --steer-phase-name scene{S} 로 스위칭 (exp4-1 LOO registry 배선 재사용).
  permanent/loo_seed{k}/steer/dit_L{L}/conceptors.npz      ← 전역 setpoint 참조 arm 용
  gated/loo_seed{k}/<phase>/dit_L{L}/conceptors.npz        ← phase 별 (setpoint 는 전역)
  fit_report.json                                          게이트·진단 전부
키 계약 = scripts/serve/steering_hooks.py load_steering_segment():
  alpha0_v_seg [S,D] 단위벡터 · alpha0_s_tok [T] · alpha0_seg_bounds [S,2] · alpha0_seg_mask [S]

실행 (승준 노드, CPU): 1-pass 스트리밍 → 캐시 npz → fold 연산은 즉시.
  ~/anaconda3/bin/python fit_within_scene_setM.py \
    --pkl-root ~/datasets/temporal_vla_outputs/eval/robocasa/groot_n15/scene_matched_exp41/pq3_drawer_right \
    --task-id 7 --layer 12 --out-root ~/exp53_npz
"""
import argparse
import json
import os
import pickle
import re
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import numpy as np

SEGMENTS = (("state", 0, 1), ("future", 1, 33), ("action", 33, 49))  # fit_mean_diff.py:55
CAP = 38                     # 진단(analyze_sm)과 동일: 성공 길이 분포 기반 공통 창
N_SEED = 8
SEED_STEP = 1_000_000
PHASE_MIN_REC = 50           # phase 채택 게이트: 혼재 scene 합산 succ·fail record 각 ≥50
PHASE_MIN_SCENE = 3          # + 기여 혼재 scene ≥3
MOVE_GAP_RATIO_MAX = 3.0     # fit_mean_diff.py:56 준용
STEM_RE = re.compile(r"task(\d+)--ep(\d+)--succ([01])")


def seg_of_token(t: int) -> int:
    for i, (_n, lo, hi) in enumerate(SEGMENTS):
        if lo <= t < hi:
            return i
    raise ValueError(f"token {t} 세그먼트 밖")


# ─────────────────────────────────────────────────────── pass 1: 스트리밍 집계 → 캐시
def build_cache(pkl_root: Path, task_id: int, layer: int, cache_path: Path):
    pkls = sorted(pkl_root.rglob(f"task{task_id}--ep*--succ*.pkl"),
                  key=lambda p: int(STEM_RE.search(p.name).group(2)))
    if not pkls:
        raise SystemExit(f"pkl 없음: {pkl_root}")
    eps = []
    for i, pk in enumerate(pkls):
        with open(pk, "rb") as f:
            d = pickle.load(f)
        caps = [int(x) for x in (d.get("capture_layers") or [])]
        li = caps.index(layer)
        hs = d["hidden_states"]
        if len(hs) < CAP:
            raise SystemExit(f"{pk.name}: records {len(hs)} < CAP {CAP}")
        # [L,K,T,D] → 이 layer → K 평균 → [T,D] float32, 창 [0,CAP)
        R = np.stack([np.asarray(rec, dtype=np.float32)[li].mean(axis=0)
                      for rec in hs[:CAP]], axis=0)          # [CAP,T,D]
        ph = [str(x) for x in (d.get("feature_phases") or d.get("phase_timeline") or [])][:CAP]
        ph += [""] * (CAP - len(ph))
        m = STEM_RE.search(pk.name)
        lab_meta = d.get("success")
        lab = int(bool(lab_meta)) if lab_meta is not None else int(m.group(3))
        if int(m.group(3)) != lab:
            raise SystemExit(f"라벨 불일치 {pk.name}: meta={lab} 파일명={m.group(3)}")
        # phase 별 토큰 합/개수 (record 가중 복원용)
        ph_names = sorted(set(ph))
        ph_sum = {p: R[[j for j, q in enumerate(ph) if q == p]].sum(axis=0) for p in ph_names}
        ph_cnt = {p: sum(1 for q in ph if q == p) for p in ph_names}
        eps.append(dict(
            scene=int(d["scenario_seed"]), inf=int(d["inference_seed"]), label=lab,
            ep=int(d.get("episode_idx", int(m.group(2)))),
            M=R.mean(axis=0),                                 # [T,D] 창 평균
            Rseg=np.stack([R[:, lo:hi, :].mean(axis=1) for _n, lo, hi in SEGMENTS],
                          axis=1),                            # [CAP,S,D] record×세그먼트
            ph_names=ph_names, ph_sum=ph_sum, ph_cnt=ph_cnt,
        ))
        if (i + 1) % 20 == 0:
            print(f"  cache {i+1}/{len(pkls)}", flush=True)
    # 직렬화 (phase dict 는 object array)
    np.savez_compressed(
        cache_path,
        scene=np.array([e["scene"] for e in eps]),
        inf=np.array([e["inf"] for e in eps]),
        label=np.array([e["label"] for e in eps]),
        ep=np.array([e["ep"] for e in eps]),
        M=np.stack([e["M"] for e in eps]),
        Rseg=np.stack([e["Rseg"] for e in eps]),
        ph_blob=np.array([pickle.dumps((e["ph_names"], e["ph_sum"], e["ph_cnt"]))
                          for e in eps], dtype=object),
        layer=np.array([layer]),
    )
    print(f"[cache] {cache_path}: {len(eps)} eps", flush=True)


def load_cache(cache_path: Path):
    z = np.load(cache_path, allow_pickle=True)
    eps = []
    for i in range(len(z["label"])):
        names, psum, pcnt = pickle.loads(z["ph_blob"][i])
        eps.append(dict(scene=int(z["scene"][i]), inf=int(z["inf"][i]),
                        label=int(z["label"][i]), ep=int(z["ep"][i]),
                        M=z["M"][i], Rseg=z["Rseg"][i],
                        ph_names=names, ph_sum=psum, ph_cnt=pcnt))
    return eps, int(z["layer"][0])


# ─────────────────────────────────────────────────────────────── fold 연산
def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); sv = s[o]
    ranks = np.arange(1, len(s) + 1, dtype=float); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sv[j + 1] == sv[i]:
            j += 1
        r[o[i:j + 1]] = ranks[i:j + 1].mean(); i = j + 1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def within_scene_dir(eps_tr):
    """혼재 scene 별 (μ_fail_tok − μ_succ_tok) 의 scene 평균 → 세그먼트별 정규화."""
    by_scene = {}
    for e in eps_tr:
        by_scene.setdefault(e["scene"], []).append(e)
    diffs, used = [], []
    for s, grp in sorted(by_scene.items()):
        Ms = [e["M"] for e in grp if e["label"] == 1]
        Mf = [e["M"] for e in grp if e["label"] == 0]
        if not Ms or not Mf:
            continue
        diffs.append(np.mean(Mf, axis=0) - np.mean(Ms, axis=0))   # [T,D]
        used.append(s)
    if not diffs:
        raise SystemExit("혼재 scene 0 — within-scene 방향 불가")
    Dbar = np.mean(diffs, axis=0)                                  # [T,D]
    v_seg = []
    for _n, lo, hi in SEGMENTS:
        d = Dbar[lo:hi].mean(axis=0)
        n = float(np.linalg.norm(d))
        if n == 0:
            raise SystemExit("영방향")
        v_seg.append(d / n)
    return np.stack(v_seg, axis=0), used


def setpoint_tok(eps_tr, v_seg):
    """s_t = 훈련 성공 episode 들의 토큰 위치별 평균 사영 (창 고정이라 record 가중 = ep 가중)."""
    Ms = np.stack([e["M"] for e in eps_tr if e["label"] == 1])     # [Ns,T,D]
    mu = Ms.mean(axis=0)                                           # [T,D]
    T = mu.shape[0]
    return np.asarray([float(mu[t] @ v_seg[seg_of_token(t)]) for t in range(T)])


def move_gap(eps_tr, v_seg, s_tok):
    """record×세그먼트 해상도 move/gap (fit_mean_diff §게이트 준용, Rseg 사용)."""
    s_seg = [np.mean([s_tok[t] for t in range(lo, hi)]) for _n, lo, hi in SEGMENTS]
    by_scene = {}
    for e in eps_tr:
        by_scene.setdefault(e["scene"], []).append(e)
    ratios = []
    for si in range(len(SEGMENTS)):
        # gap 은 **within-scene gap** (게이트 검증 2026-07-27: within-scene 방향에선 전역
        # 클래스 평균 gap 이 수축해 ratio 허위 초과 — 혼재 scene 별 gap 의 평균이 정당)
        proj_all, gaps = [], []
        for e in eps_tr:
            proj_all.append(e["Rseg"][:, si, :] @ v_seg[si])
        for _s, grp in sorted(by_scene.items()):
            ps = [e["Rseg"][:, si, :].mean(0) @ v_seg[si] for e in grp if e["label"] == 1]
            pf = [e["Rseg"][:, si, :].mean(0) @ v_seg[si] for e in grp if e["label"] == 0]
            if ps and pf:
                gaps.append(float(np.mean(pf) - np.mean(ps)))
        move = float(np.median(np.abs(np.concatenate(proj_all) - s_seg[si])))
        gap = abs(float(np.mean(gaps))) + 1e-9 if gaps else 1e-9
        ratios.append(move / gap)
    return ratios


def scene_setpoints(eps_tr, v_seg, s_tok_global):
    """scene 별 setpoint: 그 scene 성공 episode 들의 토큰별 평균 사영. 성공 없으면 전역 fallback.

    근거 (게이트 진단 2026-07-27): 전역 setpoint 의 seed-out 비중심 read-out 은 0.59 로 붕괴,
    scene-중심 read-out 은 0.83 유지 → setpoint 는 scene-로컬이어야 성공 매니폴드를 가리킴.
    exp2 의 per-scene fit 정신과 일치 (방향은 fold 공유 — 표본 안정성).
    """
    T = eps_tr[0]["M"].shape[0]
    by_scene = {}
    for e in eps_tr:
        by_scene.setdefault(e["scene"], []).append(e)
    out, fallback = {}, []
    for s, grp in sorted(by_scene.items()):
        Ms = [e["M"] for e in grp if e["label"] == 1]
        if not Ms:
            out[s] = s_tok_global
            fallback.append(s)
            continue
        mu = np.mean(Ms, axis=0)
        out[s] = np.asarray([float(mu[t] @ v_seg[seg_of_token(t)]) for t in range(T)])
    return out, fallback


def phase_stats(eps_tr):
    """phase 별 within-scene 방향 재료: scene×label 별 (합, 개수)."""
    acc = {}
    for e in eps_tr:
        for p in e["ph_names"]:
            if not p or e["ph_cnt"][p] == 0:
                continue
            key = (p, e["scene"], e["label"])
            if key not in acc:
                acc[key] = [np.zeros_like(e["M"]), 0]
            acc[key][0] += e["ph_sum"][p]
            acc[key][1] += e["ph_cnt"][p]
    return acc


def phase_dirs(acc):
    """phase 별: 혼재 scene 평균 diff → v_seg, 성공 pooled 평균 → s_tok. 게이트 적용."""
    phases = sorted({k[0] for k in acc})
    out = {}
    for p in phases:
        scenes = sorted({k[1] for k in acc if k[0] == p})
        diffs, n_s, n_f = [], 0, 0
        succ_sum, succ_cnt = None, 0
        for s in scenes:
            ks, kf = (p, s, 1), (p, s, 0)
            if ks in acc:
                if succ_sum is None:
                    succ_sum = np.zeros_like(acc[ks][0])
                succ_sum += acc[ks][0]; succ_cnt += acc[ks][1]
            if ks in acc and kf in acc:
                mu_s = acc[ks][0] / acc[ks][1]
                mu_f = acc[kf][0] / acc[kf][1]
                diffs.append(mu_f - mu_s)
                n_s += acc[ks][1]; n_f += acc[kf][1]
        if len(diffs) < PHASE_MIN_SCENE or n_s < PHASE_MIN_REC or n_f < PHASE_MIN_REC:
            out[p] = dict(adopted=False, n_scene=len(diffs), n_rec_succ=n_s, n_rec_fail=n_f)
            continue
        Dbar = np.mean(diffs, axis=0)
        v_seg = []
        for _n, lo, hi in SEGMENTS:
            d = Dbar[lo:hi].mean(axis=0)
            v_seg.append(d / (np.linalg.norm(d) + 1e-12))
        v_seg = np.stack(v_seg, axis=0)
        mu_succ = succ_sum / max(succ_cnt, 1)
        s_tok = np.asarray([float(mu_succ[t] @ v_seg[seg_of_token(t)])
                            for t in range(mu_succ.shape[0])])
        out[p] = dict(adopted=True, v_seg=v_seg, s_tok=s_tok,
                      n_scene=len(diffs), n_rec_succ=n_s, n_rec_fail=n_f)
    return out


def save_npz(path: Path, v_seg, s_tok):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path,
             alpha0_v_seg=v_seg.astype(np.float32),
             alpha0_s_tok=s_tok.astype(np.float32),
             alpha0_seg_bounds=np.asarray([[lo, hi] for _n, lo, hi in SEGMENTS], dtype=np.int64),
             alpha0_seg_mask=np.ones(len(SEGMENTS), dtype=np.float32))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl-root", required=True)
    ap.add_argument("--task-id", type=int, required=True)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()

    out_root = Path(args.out_root).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)
    cache = out_root / f"fit_cache_L{args.layer}.npz"
    if not cache.exists():
        build_cache(Path(args.pkl_root).expanduser(), args.task_id, args.layer, cache)
    eps, layer = load_cache(cache)
    y = np.array([e["label"] for e in eps])
    sc = np.array([e["scene"] for e in eps])
    print(f"[fit] eps={len(eps)} succ={int((y==1).sum())} fail={int((y==0).sum())} "
          f"scene={len(set(sc.tolist()))} L={layer}", flush=True)

    # 혼재 scene (전체 데이터 기준 — 평가 대상 정의용)
    mixed_all = {s for s in set(sc.tolist())
                 if (y[sc == s] == 1).any() and (y[sc == s] == 0).any()}

    report = dict(layer=layer, cap=CAP, n_ep=len(eps), folds={}, phases={},
                  mixed_scenes=sorted(mixed_all))
    folds = list(range(N_SEED)) + ["all"]
    v_by_fold = {}
    loo_scores, loo_scores_sc, loo_labels = [], [], []
    for k in folds:
        if k == "all":
            eps_tr = eps
        else:
            eps_tr = [e for e in eps if e["inf"] != k * SEED_STEP]
        v_seg, used_scenes = within_scene_dir(eps_tr)
        s_tok = setpoint_tok(eps_tr, v_seg)
        ratios = move_gap(eps_tr, v_seg, s_tok)
        # 게이트 ① 재료: held-out(seed k, 혼재 scene) episode 점수
        rec = dict(used_mixed_scenes=len(used_scenes),
                   move_over_gap=[round(r, 3) for r in ratios])
        s_by_scene, fallback = scene_setpoints(eps_tr, v_seg, s_tok)
        rec["setpoint_fallback_scenes"] = fallback
        if k != "all":
            ev = [e for e in eps if e["inf"] == k * SEED_STEP and e["scene"] in mixed_all]
            T = eps[0]["M"].shape[0]
            v_tok = np.stack([v_seg[seg_of_token(t)] for t in range(T)])
            for e in ev:
                proj = np.asarray([float(e["M"][t] @ v_tok[t]) for t in range(T)])
                loo_scores.append(float(proj.mean()))                        # 전역 read-out
                loo_scores_sc.append(float((proj - s_by_scene[e["scene"]]).mean()))  # scene-중심
                loo_labels.append(e["label"])
            rec["n_eval"] = len(ev)
        report["folds"][str(k)] = rec
        v_by_fold[k] = v_seg
        # permanent 배치 = phase registry: scene{S}/ = per-scene setpoint, steer/ = 전역 참조
        base_p = out_root / "permanent" / ("all" if k == "all" else f"loo_seed{k}")
        save_npz(base_p / "steer" / f"dit_L{layer}" / "conceptors.npz", v_seg, s_tok)
        for s, st in s_by_scene.items():
            save_npz(base_p / f"scene{s}" / f"dit_L{layer}" / "conceptors.npz", v_seg, st)

        # gated: phase 별
        pstats = phase_dirs(phase_stats(eps_tr))
        for p, info in pstats.items():
            tag = report["phases"].setdefault(p, {})
            tag[str(k)] = {kk: vv for kk, vv in info.items()
                           if kk in ("adopted", "n_scene", "n_rec_succ", "n_rec_fail")}
            if info["adopted"]:
                base = out_root / "gated" / ("all" if k == "all" else f"loo_seed{k}")
                save_npz(base / p / f"dit_L{layer}" / "conceptors.npz",
                         info["v_seg"], info["s_tok"])
                if k == "all":
                    cosv = [float(info["v_seg"][i] @ v_by_fold["all"][i])
                            for i in range(len(SEGMENTS))]
                    tag["cos_vs_permanent"] = [round(c, 3) for c in cosv]

    # 게이트 ① LOO 합산 AUROC (fail 점수가 높아야 정방향 → succ 는 낮음).
    # 전역 read-out 은 scene offset 때문에 낮게 나옴(0.59~0.61, 구조적 — 배포 setpoint 가
    # per-scene 인 이유). 판정 기준은 **scene-중심 read-out** 이 진단값(0.83대)과 정합하는가.
    loo_auroc = auroc(-np.asarray(loo_scores), np.asarray(loo_labels))
    loo_auroc_sc = auroc(-np.asarray(loo_scores_sc), np.asarray(loo_labels))
    report["gate1_loo_auroc_global_readout"] = round(loo_auroc, 4)
    report["gate1_loo_auroc_scene_centered"] = round(loo_auroc_sc, 4)
    # 게이트 ③ fold 안정성: 각 fold vs all 의 세그먼트별 cos
    cos_folds = {str(k): [round(float(v_by_fold[k][i] @ v_by_fold["all"][i]), 4)
                          for i in range(len(SEGMENTS))]
                 for k in range(N_SEED)}
    report["gate3_cos_fold_vs_all"] = cos_folds
    worst_ratio = max(max(r["move_over_gap"]) for r in report["folds"].values())
    report["gate2_move_over_gap_worst"] = round(worst_ratio, 3)
    report["gate2_pass"] = bool(worst_ratio <= MOVE_GAP_RATIO_MAX)

    (out_root / "fit_report.json").write_text(json.dumps(report, indent=2))
    print(f"[gate1] LOO AUROC 전역={loo_auroc:.4f} scene-중심={loo_auroc_sc:.4f} "
          f"(판정: scene-중심이 0.8대면 통과)")
    print(f"[gate2] move/gap worst = {worst_ratio:.3f} (≤{MOVE_GAP_RATIO_MAX})")
    print(f"[gate3] fold-cos 최저 = "
          f"{min(min(v) for v in cos_folds.values()):.4f}")
    print("[written]", out_root / "fit_report.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

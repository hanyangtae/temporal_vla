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

N_PERM = 20
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


def save_setpoint_npz(out_dir: Path, layer_blk: int, v: np.ndarray, s: float, meta: dict):
    d = out_dir / "steer" / f"dit_L{layer_blk}"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "conceptors.npz",
                        alpha0_v_steer=v.astype(np.float32), alpha0_s=np.float32(s))
    (d / "metadata.json").write_text(
        json.dumps({**meta, "selected_alpha": 0}, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True, help="fit30 tsv (pkl\\tlabel\\tscene)")
    ap.add_argument("--cell", required=True, help="예: pq3_ppcc_bread (within-instruction fit)")
    ap.add_argument("--targets", type=Path, required=True,
                    help="annotation_t0.tsv — fit-풀 대상은 LOO, eval-풀 대상은 침범 검사")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--layers", default=None,
                    help="sweep 할 DiT 물리 layer 콤마목록 (기본: capture 전부)")
    args = ap.parse_args()

    rng = np.random.default_rng(RNG_SEED)
    rolls = load_cell_rolls(args.manifest, args.cell)
    targets = load_targets(args.targets, args.cell)
    labels = [r["success"] for r in rolls]
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

    # ---- setM_pl: dose-match 순열 선택 — held-out 없이 fit-표본 record 사영으로 근사
    # (dose = |(h·r̂)−s| 분포 중앙값이 setM 의 ±25% 내인 순열 중 중앙값 순위 중간 것)
    all_rec = np.concatenate([episode_records(r, li, cap) for r in rolls], axis=0)
    dose_m = float(np.median(np.abs(all_rec @ v_full - s_full)))
    perms = []
    for pi, pl in enumerate(perm_labels_list):
        try:
            Xsp = gather(rolls, pl, li, cap, 1)
            Xfp = gather(rolls, pl, li, cap, 0)
            vp, sp = mean_diff(Xsp, Xfp)
        except (ValueError, SystemExit):
            continue
        dp = float(np.median(np.abs(all_rec @ vp - sp)))
        ap_ = auroc(np.asarray([x for r, y in zip(rolls, labels) if y == 0
                                for x in episode_records(r, li, cap) @ vp]),
                    np.asarray([x for r, y in zip(rolls, labels) if y == 1
                                for x in episode_records(r, li, cap) @ vp]))
        perms.append({"perm_id": pi, "v": vp, "s": sp, "dose_median": dp,
                      "true_label_auroc": ap_, "labels": pl})
    in_band = [p for p in perms if 0.75 * dose_m <= p["dose_median"] <= 1.25 * dose_m]
    pool = in_band if in_band else perms
    pool.sort(key=lambda p: abs(p["dose_median"] - dose_m))
    pl_sel = pool[0]
    print(f"[setM_pl] perm_id={pl_sel['perm_id']} dose={pl_sel['dose_median']:.3f} "
          f"(setM {dose_m:.3f}) in_band={len(in_band)}/{len(perms)} "
          f"true_label_auroc={pl_sel['true_label_auroc']:.3f}", flush=True)

    # ---- 저장 (full-fit)
    out_cell = args.out_root / args.cell
    manifest_sha = hashlib.sha256(args.manifest.read_bytes()).hexdigest()[:12]
    targets_sha = hashlib.sha256(args.targets.read_bytes()).hexdigest()[:12]
    base_meta = {
        "operator": "setM", "cell": args.cell, "layer": blk, "cap_records": cap,
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
    save_setpoint_npz(out_cell / "setM", blk, v_full, s_full, base_meta)
    save_setpoint_npz(out_cell / "setM_pl", blk, pl_sel["v"], pl_sel["s"], {
        **base_meta, "operator": "setM_pl", "perm_id": pl_sel["perm_id"],
        "dose_median": pl_sel["dose_median"], "true_label_auroc": pl_sel["true_label_auroc"],
    })

    # ---- fit-풀 대상 per-target LOO (setM + pairing 위약: 같은 순열 라벨에서 대상 제외)
    for t in targets["fit"]:
        keep = [i for i in range(len(rolls)) if rolls[i]["episode_idx"] != t["episode_idx"]]
        kl = [labels[i] for i in keep]
        kr = [rolls[i] for i in keep]
        v_t, s_t = mean_diff(gather(kr, kl, li, cap, 1), gather(kr, kl, li, cap, 0))
        save_setpoint_npz(out_cell / "setM_loo" / f"ep{t['episode_idx']}", blk, v_t, s_t, {
            **base_meta, "operator": "setM_loo", "loo_episode_idx": t["episode_idx"],
            "s": s_t,
        })
        pl_l = [pl_sel["labels"][i] for i in keep]
        try:
            v_pt, s_pt = mean_diff(gather(kr, pl_l, li, cap, 1), gather(kr, pl_l, li, cap, 0))
        except ValueError:
            v_pt, s_pt = pl_sel["v"], pl_sel["s"]  # 순열 클래스 고갈 시 full 위약 재사용
        save_setpoint_npz(out_cell / "setM_pl_loo" / f"ep{t['episode_idx']}", blk, v_pt, s_pt, {
            **base_meta, "operator": "setM_pl_loo", "loo_episode_idx": t["episode_idx"],
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

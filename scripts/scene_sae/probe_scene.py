#!/usr/bin/env python
"""exp5/G1 게이트 — SAE feature 가 scene 을 인코딩하는가 (핸드아웃 §4 Phase D).

로컬 실행 전용(sklearn 사용 — 승준에는 sklearn/scipy 없음).

무엇을 재는가
  ① z-probe    : SAE 희소 활성 z(top-k 후) → scene 라벨 선형 probe, **episode held-out**
  ② X-probe    : 원본 activation(표준화) probe = 상한 (SAE 가 정보를 잃었는지)
  ③ null       : **episode 단위** 라벨 순열 (≥100회) → 우연 수준 분포, z-score
  ④ selectivity: feature 별 클래스-조건 활성률 → scene-selective feature 목록/비율

scene 라벨 = **layout_id** (5 클래스).
  - scenario_seed 는 **episode 당 1개**라 클래스 = 표본이 되어 probe 불가 (핸드아웃 §6-1).
  - style_id 는 fit30 drawer_left 에서 layout_id 와 **완전 공선** → 같은 분할. 결과 json 에 명기.

길이 confound 통제 (핸드아웃 §6-3): 실패 episode 는 timeout 이라 record 가 훨씬 많다
(drawer_left: fail 13ep 이 전체 행의 73%). 기본으로 **episode 당 record 수를 균등
subsample** 한다(--records-per-ep, 기본 auto = 전 episode 최소 record 수). record 단위로
뽑고 그 record 의 T 토큰은 전부 유지한다 — 토큰 평균 금지 원칙 유지.

G1 판정 (§4-D3, 사전 등록):
  (a) held-out(test split) z-probe 정확도가 순열 null 대비 z > 3
  (b) 우연보정 회복률 (acc_z − chance) / (acc_X − chance) ≥ 0.80
  (c) scene-selective feature 비율 < 0.30
  세 조건 모두 만족 → G1 pass.

사용:
  python scripts/scene_sae/probe_scene.py --ckpt-dir <L10_m6144_k32_s0> \
      --x .../X_L10.npz --stats .../stats_L10.npz --meta .../meta.npz --out probe_L10.json
  python scripts/scene_sae/probe_scene.py --smoke        # 합성 데이터 dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# ------------------------------------------------------------------ subsample
def balanced_record_mask(meta: dict, records_per_ep: int, seed: int) -> np.ndarray:
    """episode 당 record 수를 균등하게 맞추는 행 마스크 (record 단위 추출, 토큰 전부 유지)."""
    ep = meta["episode_idx"]
    rec = meta["record_idx"]
    rng = np.random.default_rng(seed)
    eps = np.unique(ep)
    n_avail = {int(e): int(rec[ep == e].max()) + 1 for e in eps}
    n_take = records_per_ep if records_per_ep > 0 else min(n_avail.values())
    mask = np.zeros(len(ep), dtype=bool)
    for e in eps:
        avail = n_avail[int(e)]
        take = min(n_take, avail)
        sel = np.sort(rng.choice(avail, size=take, replace=False))
        mask |= (ep == e) & np.isin(rec, sel)
    return mask, n_take


# ----------------------------------------------------------------------- SAE z
def load_sae(ckpt_dir: Path):
    """train_scene_sae.py 산출 디렉토리 → (model, cfg). src/sae 접점은 여기 하나."""
    from src.sae.models import build_model             # noqa: PLC0415
    from train_scene_sae import build_sae_cfg          # noqa: PLC0415  (동일 디렉토리)

    cfg = json.loads((ckpt_dir / "config.json").read_text())
    sae_cfg = cfg.get("sae_cfg") or build_sae_cfg(cfg["input_dim"], cfg["m"], cfg["k"])
    model = build_model(sae_cfg)
    model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location="cpu"))
    model.eval()
    return model, cfg


def compute_z(model, X: np.ndarray, rows: np.ndarray, mu, sd, device: str, batch: int):
    """SAE 희소 활성 z [n_rows, m] (float32 dense — subsample 후라 감당 가능)."""
    from src.sae.train import encode_all               # noqa: PLC0415

    a = X[rows].astype(np.float32)
    a = (a - mu) / sd
    return encode_all(model, a, batch_size=batch, device=device)


# ---------------------------------------------------------------------- probes
def fit_probe(Ztr, ytr, Zte, yte, seed: int, max_iter: int, C: float, solver: str = "lbfgs"):
    """다항 로지스틱 probe. 기본 solver=lbfgs — 실측 3072-d/2000행에서 saga 10.3s vs
    lbfgs 1.1s (동일 정확도). saga 는 희소·L1 이 필요할 때만."""
    clf = LogisticRegression(max_iter=max_iter, C=C, solver=solver,
                             n_jobs=(-1 if solver in ("saga", "lbfgs") else None),
                             random_state=seed, tol=1e-3)
    clf.fit(Ztr, ytr)
    return float((clf.predict(Zte) == yte).mean()), clf


def chance_level(y) -> float:
    _, cnt = np.unique(y, return_counts=True)
    return float(cnt.max() / cnt.sum())


def permutation_null(Ztr, Zte, ep_tr, ep_te, ep2lab, n_perm, seed, max_iter, C,
                     solver="lbfgs", verbose=True):
    """**episode 단위** 라벨 순열: episode→layout 사상을 섞고 행 라벨을 다시 만든다.

    행 단위 순열은 같은 episode 안의 자기상관을 깨서 null 을 과소평가한다(핸드아웃 §6-2 동형).
    비용이 전체 런타임을 지배한다 (probe fit × n_perm) — --n-perm / --null-max-iter 로 조절.
    """
    rng = np.random.default_rng(seed)
    eps = np.asarray(sorted(ep2lab))
    labs = np.asarray([ep2lab[int(e)] for e in eps])
    accs = []
    t0 = time.time()
    for i in range(n_perm):
        perm = rng.permutation(len(labs))
        m = {int(e): int(labs[p]) for e, p in zip(eps, perm)}
        ytr = np.asarray([m[int(e)] for e in ep_tr])
        yte = np.asarray([m[int(e)] for e in ep_te])
        if len(np.unique(ytr)) < 2:
            continue
        acc, _ = fit_probe(Ztr, ytr, Zte, yte, 0, max_iter, C, solver)
        accs.append(acc)
        if verbose and (i + 1) % 10 == 0:
            print(f"  [null] {i+1}/{n_perm} mean={np.mean(accs):.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return np.asarray(accs)


def feature_selectivity(Z, y, min_rate: float, ratio: float, top: int):
    """feature 별 클래스-조건 활성률 → scene-selective 판정.

    selective = (최대 클래스 활성률 ≥ min_rate) and (최대 / 차순위 ≥ ratio).
    단일-feature probe 보다 싸고, top-k 코드에서 해석이 직접적이다(어떤 scene 에서만 켜지나).
    """
    classes = np.unique(y)
    act = (Z > 0)
    rates = np.stack([act[y == c].mean(axis=0) for c in classes], axis=1)  # [m, C]
    order = np.sort(rates, axis=1)[:, ::-1]
    top1, top2 = order[:, 0], order[:, 1] if rates.shape[1] > 1 else np.zeros(len(order))
    live = act.any(axis=0)
    sel = live & (top1 >= min_rate) & (top1 >= ratio * np.maximum(top2, 1e-6))
    score = np.where(live, top1 - top2, 0.0)
    idx = np.argsort(-score)[:top]
    return {
        "n_features": int(Z.shape[1]),
        "n_live": int(live.sum()),
        "n_selective": int(sel.sum()),
        "selective_frac_of_all": float(sel.sum() / Z.shape[1]),
        "selective_frac_of_live": float(sel.sum() / max(1, live.sum())),
        "criteria": {"min_rate": min_rate, "ratio": ratio},
        "classes": [int(c) for c in classes],
        "top_features": [
            {"feature": int(j), "best_class": int(classes[int(np.argmax(rates[j]))]),
             "rates": [round(float(r), 4) for r in rates[j]],
             "gap": round(float(score[j]), 4), "selective": bool(sel[j])}
            for j in idx],
    }


# ------------------------------------------------------------------- 판정/출력
def judge(res: dict, args) -> dict:
    z_score = res["z_probe"]["null_z"]
    rec = res["recovery_chance_corrected"]
    selfrac = res["selectivity"]["selective_frac_of_all"]
    crit = {
        "a_null_z_gt_3": bool(z_score is not None and z_score > 3.0),
        "b_recovery_ge_0.80": bool(rec is not None and rec >= 0.80),
        "c_selective_frac_lt_0.30": bool(selfrac < 0.30),
    }
    return {"criteria": crit, "pass": all(crit.values()),
            "values": {"null_z": z_score, "recovery": rec, "selective_frac": selfrac}}


def print_table(res: dict) -> None:
    p = res["z_probe"]
    x = res["x_probe"]
    print("\n===== G1 scene probe (라벨 = layout_id) =====")
    print(f"  클래스 {res['classes']}  chance(test) = {res['chance_test']:.3f}")
    print(f"  행: train {res['n_train_rows']}  test {res['n_test_rows']}  "
          f"(episode 당 record {res['records_per_ep']} 균등 subsample)")
    print(f"  {'probe':22s} {'test acc':>9s} {'train acc':>10s}")
    print(f"  {'① SAE z':22s} {p['test_acc']:9.3f} {p['train_acc']:10.3f}")
    print(f"  {'② 원본 X (상한)':22s} {x['test_acc']:9.3f} {x['train_acc']:10.3f}")
    if p["null_mean"] is not None:
        print(f"  {'③ 순열 null(z)':22s} {p['null_mean']:9.3f} ± {p['null_std']:.3f}"
              f"  → z = {p['null_z']:.2f} (n={p['n_perm']})")
    print(f"  회복률(우연보정) = {res['recovery_chance_corrected']}")
    print(f"  success 층화 test acc: succ={res['stratified']['succ_acc']} "
          f"fail={res['stratified']['fail_acc']}")
    s = res["selectivity"]
    print(f"  ④ selective feature {s['n_selective']}/{s['n_features']} "
          f"({s['selective_frac_of_all']:.3f}, live 중 {s['selective_frac_of_live']:.3f})")
    j = res["judgment"]
    print(f"  판정: {'PASS' if j['pass'] else 'FAIL'}  {j['criteria']}")


# -------------------------------------------------------------------- 합성 smoke
def synth(seed=0, n_ep=20, n_rec=20, T=8, D=64, m=256, k=8):
    """layout 신호가 실제로 들어 있는 합성 데이터 — 파이프라인 dry-run 용."""
    rng = np.random.default_rng(seed)
    n_lay = 4
    dirs = rng.normal(size=(n_lay, D))
    ep_lay, ep_succ, Xs, meta = {}, {}, [], {kk: [] for kk in
                                            ("episode_idx", "record_idx", "token_idx",
                                             "token_seg", "success", "layout_id", "style_id",
                                             "scenario_seed", "split")}
    for e in range(n_ep):
        lay = e % n_lay
        succ = int(e % 3 == 0)
        recs = n_rec if succ else int(n_rec * 1.8)      # 실패가 길다 (길이 confound 모사)
        ep_lay[e], ep_succ[e] = lay, succ
        x = rng.normal(size=(recs * T, D)) + 2.0 * dirs[lay]
        Xs.append(x.astype(np.float32))
        # split 은 layout 안에서 배분한다 — layout 과 split 이 얽히면 test 에 한 클래스만
        # 남아 probe 가 무의미해진다 (실제 빌더의 층화 split 과 같은 취지).
        pos = e // n_lay
        n_pos = n_ep // n_lay
        split = 0 if pos < n_pos - 2 else (1 if pos == n_pos - 2 else 2)
        meta["episode_idx"].append(np.full(recs * T, e, np.int32))
        meta["record_idx"].append(np.repeat(np.arange(recs, dtype=np.int32), T))
        meta["token_idx"].append(np.tile(np.arange(T, dtype=np.int16), recs))
        meta["token_seg"].append(np.tile(np.arange(T, dtype=np.int8) // 4, recs))
        meta["success"].append(np.full(recs * T, succ, np.int8))
        meta["layout_id"].append(np.full(recs * T, lay, np.int32))
        meta["style_id"].append(np.full(recs * T, lay, np.int32))
        meta["scenario_seed"].append(np.full(recs * T, 100000 + e, np.int64))
        meta["split"].append(np.full(recs * T, split, np.int8))
    X = np.concatenate(Xs)
    meta = {kk: np.concatenate(v) for kk, v in meta.items()}
    W = rng.normal(size=(D, m)) / np.sqrt(D)
    return X, meta, W, k


def topk_encode(Xs: np.ndarray, W: np.ndarray, k: int) -> np.ndarray:
    h = np.maximum(Xs @ W, 0.0)
    thr = np.partition(h, -k, axis=1)[:, -k][:, None]
    return np.where(h >= thr, h, 0.0)


# ------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description="G1 scene probe (SAE feature vs layout)")
    ap.add_argument("--ckpt-dir", type=Path, help="train_scene_sae.py 산출 디렉토리")
    ap.add_argument("--x", type=Path)
    ap.add_argument("--stats", type=Path)
    ap.add_argument("--meta", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="결과 json 경로")
    ap.add_argument("--label", default="layout_id", choices=["layout_id", "style_id"])
    ap.add_argument("--records-per-ep", type=int, default=0,
                    help="episode 당 record 수 (0=auto: 최소값). 0 미만이면 균등화 끔")
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--null-max-iter", type=int, default=100,
                    help="순열 null 의 probe max_iter (null 이 런타임을 지배)")
    ap.add_argument("--solver", default="lbfgs", choices=["lbfgs", "saga", "liblinear"])
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--sel-min-rate", type=float, default=0.10)
    ap.add_argument("--sel-ratio", type=float, default=3.0)
    ap.add_argument("--sel-top", type=int, default=30)
    ap.add_argument("--smoke", action="store_true", help="합성 데이터 dry-run (ckpt 불필요)")
    args = ap.parse_args()

    t0 = time.time()
    if args.smoke:
        X, meta, W, k = synth(args.seed)
        mu = X[meta["split"] == 0].mean(0)
        sd = X[meta["split"] == 0].std(0)
        sd = np.where(sd < 1e-6, 1.0, sd)
        cfg, src = {"m": W.shape[1], "k": k, "layer": -1, "cell": "SMOKE"}, "smoke"
    else:
        for need in ("ckpt_dir", "x", "stats", "meta"):
            if getattr(args, need) is None:
                raise SystemExit(f"--{need.replace('_','-')} 필요 (또는 --smoke)")
        meta_npz = np.load(args.meta, allow_pickle=False)
        meta = {kk: meta_npz[kk] for kk in meta_npz.files
                if meta_npz[kk].ndim == 1 and meta_npz[kk].shape[0] == len(meta_npz["split"])}
        X = np.load(args.x)["X"]
        st = np.load(args.stats)
        mu, sd = st["mean"], st["std"]

    if args.records_per_ep >= 0:
        sub, rpe = balanced_record_mask(meta, args.records_per_ep, args.seed)
    else:
        sub, rpe = np.ones(len(meta["split"]), bool), -1

    y_all = meta[args.label]
    ep_all = meta["episode_idx"]
    split = meta["split"]
    # style/layout 공선 진단 (라벨 선택 근거를 결과에 남긴다)
    pairs = {(int(a), int(b)) for a, b in zip(meta["layout_id"], meta["style_id"])}
    collinear = len(pairs) == len(set(int(a) for a, _ in pairs))
    seeds_per_ep = {int(e): int(np.unique(meta["scenario_seed"][ep_all == e])[0])
                    for e in np.unique(ep_all)}

    rows_tr = sub & (split == 0)
    rows_te = sub & (split == 2)
    if rows_te.sum() == 0:
        raise SystemExit("test split 행 0 — split 배정 확인")

    if args.smoke:
        Xs = ((X - mu) / sd).astype(np.float32)
        Z_tr = topk_encode(Xs[rows_tr], W, k)
        Z_te = topk_encode(Xs[rows_te], W, k)
        X_tr, X_te = Xs[rows_tr], Xs[rows_te]
    else:
        model, cfg = load_sae(args.ckpt_dir)
        src = cfg.get("sae_source", "src.sae")
        Z_tr = compute_z(model, X, rows_tr, mu, sd, args.device, args.batch)
        Z_te = compute_z(model, X, rows_te, mu, sd, args.device, args.batch)
        X_tr = ((X[rows_tr].astype(np.float32) - mu) / sd)
        X_te = ((X[rows_te].astype(np.float32) - mu) / sd)

    ytr, yte = y_all[rows_tr], y_all[rows_te]
    ep_tr, ep_te = ep_all[rows_tr], ep_all[rows_te]
    print(f"[probe] rows train={len(ytr)} test={len(yte)} m={Z_tr.shape[1]} "
          f"classes={sorted(set(int(v) for v in y_all))}", flush=True)

    acc_z, clf_z = fit_probe(Z_tr, ytr, Z_te, yte, args.seed, args.max_iter, args.C, args.solver)
    acc_z_tr = float((clf_z.predict(Z_tr) == ytr).mean())
    acc_x, clf_x = fit_probe(X_tr, ytr, X_te, yte, args.seed, args.max_iter, args.C, args.solver)
    acc_x_tr = float((clf_x.predict(X_tr) == ytr).mean())
    chance = chance_level(yte)

    ep2lab = {int(e): int(y_all[ep_all == e][0]) for e in np.unique(ep_all)}
    nulls = permutation_null(Z_tr, Z_te, ep_tr, ep_te, ep2lab, args.n_perm,
                             args.seed, args.null_max_iter, args.C,
                             args.solver) if args.n_perm > 0 \
        else np.asarray([])
    if len(nulls):
        nm, ns = float(nulls.mean()), float(nulls.std(ddof=1))
        nz = (acc_z - nm) / ns if ns > 1e-9 else None
    else:
        nm = ns = nz = None

    denom = acc_x - chance
    recovery = float((acc_z - chance) / denom) if denom > 1e-6 else None

    # success 층화 (scene × success 얽힘 경고 대응)
    pred_te = clf_z.predict(Z_te)
    succ_te = meta["success"][rows_te]
    strat = {}
    for name, m_ in (("succ_acc", succ_te == 1), ("fail_acc", succ_te != 1)):
        strat[name] = float((pred_te[m_] == yte[m_]).mean()) if m_.sum() else None
        strat[name.replace("acc", "n")] = int(m_.sum())

    sel = feature_selectivity(Z_tr, ytr, args.sel_min_rate, args.sel_ratio, args.sel_top)

    res = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sae_source": src, "ckpt_dir": str(args.ckpt_dir) if args.ckpt_dir else None,
        "config": cfg, "label": args.label,
        "label_note": ("scene 라벨로 scenario_seed 를 쓸 수 없다 — episode 당 seed 1개라 "
                       "클래스=표본이 되어 probe 불가(핸드아웃 §6-1). style_id 는 layout_id 와 "
                       + ("완전 공선 → 동일 분할." if collinear else "부분 공선.")),
        "scenario_seed_per_episode": seeds_per_ep,
        "layout_style_collinear": bool(collinear),
        "records_per_ep": rpe, "n_train_rows": int(len(ytr)), "n_test_rows": int(len(yte)),
        "classes": sorted(set(int(v) for v in y_all)), "chance_test": chance,
        "z_probe": {"test_acc": acc_z, "train_acc": acc_z_tr, "null_mean": nm,
                    "null_std": ns, "null_z": nz, "n_perm": int(len(nulls))},
        "x_probe": {"test_acc": acc_x, "train_acc": acc_x_tr},
        "recovery_chance_corrected": recovery,
        "stratified": strat,
        "selectivity": sel,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    res["judgment"] = judge(res, args)
    print_table(res)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(f"[probe] → {args.out}")


if __name__ == "__main__":
    sys.exit(main())

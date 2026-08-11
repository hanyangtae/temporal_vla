#!/usr/bin/env python3
"""exp4-1 token-space 진단 (rollout 전 CPU 게이트, 2026-07-23 배선 버그 대응).

배경: fit 은 49토큰 mean 공간에서 (r̂, s) 를 냈는데 serve 는 last_horizon(action 16토큰)에
per-token 적용 → s(pooled) 를 action 토큰에 강제해 4.4σ 오프매니폴드 이동. 수정 전에
**신호가 어느 토큰에 있는지**를 먼저 측정한다 (사용자 지시 2·5).

세그먼트 (N1.5 DiT 시퀀스 T=49): state[0:1] / future[1:33] / action[33:49]
  — fit_phase_conceptor_n15._token_segments 규약과 동일.

출력 JSON:
  A. per-token: 토큰 위치별 (pooled r̂ 사영의) succ/fail 평균·갭, 성공 평균 s_t
  B. per-segment fit: 세그먼트별 r̂_seg 독립 fit → held-out(episode 5-fold) AUROC + 순열 z
  C. 방향 비교: cos(r̂_state, r̂_future, r̂_action, r̂_pooled) 행렬
  D. 배선 영향 재현: 배포 NPZ 의 (r̂, s) 로 β=1 개입 시 토큰별 이동량 / 토큰 std 대비

사용 (승준): python diag_token_space.py --manifest <fit tsv> --cell pq3_ppcc_beer \
    --layer 10 [--npz <배포 setM NPZ>] --out <...>/diag_token_space_<cell>.json
"""
from __future__ import annotations

import argparse
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fit_setm import auroc, mean_diff  # noqa: E402
from fit_phase_conceptor import FULLTOKEN_MODE  # noqa: E402

SEGMENTS = {"state": (0, 1), "future": (1, 33), "action": (33, 49)}
N_PERM = 20
RNG_SEED = 424105


def load_fulltoken_cell(manifest: Path, cell: str, layer_blk: int):
    """cell 의 full-token pkl → per-record [T, D] (선정 layer, denoise mean) 리스트."""
    rolls = []
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
        with open(p, "rb") as f:
            d = pickle.load(f)
        if d.get("capture_token_mode") != FULLTOKEN_MODE:
            raise SystemExit(f"{p}: full-token pkl 아님")
        cap_layers = [int(x) for x in (d.get("capture_layers") or [])]
        li = cap_layers.index(layer_blk)
        recs = []
        for rec in d["hidden_states"]:
            a = np.asarray(rec, dtype=np.float32)   # [L, K, T, D]
            recs.append(a[li].mean(axis=0))         # denoise mean → [T, D]
        rolls.append({
            "X": np.stack(recs, axis=0),            # [n, T, D]
            "success": int(parts[1]),
            "length": len(recs),
            "episode_idx": int(d.get("episode_idx", -1)),
        })
    if not rolls:
        raise SystemExit(f"cell={cell} 행 없음")
    return rolls


def trunc_cap(rolls) -> int:
    lens = [r["length"] for r in rolls if r["success"] == 1]
    return int(np.ceil(np.mean(lens) + np.std(lens)))


def seg_matrix(rolls, labels, cap, cls, seg):
    """클래스별 [N, D] — 세그먼트 토큰 평균 (state 는 토큰 1개)."""
    lo, hi = SEGMENTS[seg]
    out = [r["X"][:cap, lo:hi, :].mean(axis=1) for r, y in zip(rolls, labels) if y == cls]
    return np.concatenate(out, axis=0) if out else np.empty((0, 0))


def cv_auroc_seg(rolls, labels, cap, seg, rng) -> float:
    """episode 5-fold CV — 세그먼트 공간에서 fit·평가."""
    idx = np.arange(len(rolls))
    f_idx, s_idx = idx[np.asarray(labels) == 0], idx[np.asarray(labels) == 1]
    rng.shuffle(f_idx)
    rng.shuffle(s_idx)
    lo, hi = SEGMENTS[seg]
    sp, sn = [], []
    for k in range(5):
        test = set(f_idx[k::5]) | set(s_idx[k::5])
        tr = [i for i in idx if i not in test]
        Xs = seg_matrix([rolls[i] for i in tr], [labels[i] for i in tr], cap, 1, seg)
        Xf = seg_matrix([rolls[i] for i in tr], [labels[i] for i in tr], cap, 0, seg)
        if len(Xs) == 0 or len(Xf) == 0:
            continue
        r, _ = mean_diff(Xs, Xf)
        for i in test:
            proj = rolls[i]["X"][:cap, lo:hi, :].mean(axis=1) @ r
            (sp if labels[i] == 0 else sn).extend(proj.tolist())
    return auroc(np.asarray(sp), np.asarray(sn))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--layer", type=int, required=True, help="배포 setM 선정 layer (물리 blk)")
    ap.add_argument("--npz", type=Path, default=None, help="배포 setM NPZ (D. 영향 재현용)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rolls = load_fulltoken_cell(args.manifest, args.cell, args.layer)
    labels = [r["success"] for r in rolls]
    cap = trunc_cap(rolls)
    T = rolls[0]["X"].shape[1]
    rng = np.random.default_rng(RNG_SEED)
    labels_arr = np.asarray(labels)
    perms = []
    for _ in range(N_PERM):
        pl = labels_arr.copy()
        rng.shuffle(pl)
        perms.append(pl.tolist())
    print(f"[{args.cell}] L{args.layer} rollouts={len(rolls)} succ={sum(labels)} "
          f"cap={cap} T={T}", flush=True)

    # pooled r̂ (현행 fit 과 동일 공간: 49토큰 평균)
    Xs_p = np.concatenate([r["X"][:cap].mean(axis=1) for r, y in zip(rolls, labels) if y == 1])
    Xf_p = np.concatenate([r["X"][:cap].mean(axis=1) for r, y in zip(rolls, labels) if y == 0])
    r_pool, s_pool = mean_diff(Xs_p, Xf_p)

    # A. per-token (pooled 방향 사영)
    per_tok = []
    for t in range(T):
        ps = np.concatenate([r["X"][:cap, t, :] @ r_pool for r, y in zip(rolls, labels) if y == 1])
        pf = np.concatenate([r["X"][:cap, t, :] @ r_pool for r, y in zip(rolls, labels) if y == 0])
        per_tok.append({"token": t, "succ_mean": float(ps.mean()), "fail_mean": float(pf.mean()),
                        "gap_fail_minus_succ": float(pf.mean() - ps.mean()),
                        "succ_std": float(ps.std()), "s_t": float(ps.mean()),
                        "auroc_token": auroc(pf, ps)})

    # B·C. 세그먼트별 독립 fit
    seg_res, dirs = {}, {}
    for seg in SEGMENTS:
        Xs = seg_matrix(rolls, labels, cap, 1, seg)
        Xf = seg_matrix(rolls, labels, cap, 0, seg)
        r_seg, s_seg = mean_diff(Xs, Xf)
        dirs[seg] = r_seg
        a = cv_auroc_seg(rolls, labels, cap, seg, np.random.default_rng(RNG_SEED + 1))
        null = [cv_auroc_seg(rolls, pl, cap, seg, np.random.default_rng(RNG_SEED + 7 * (i + 1)))
                for i, pl in enumerate(perms)]
        null = [x for x in null if np.isfinite(x)]
        mu, sd = float(np.mean(null)), float(np.std(null))
        seg_res[seg] = {
            "cv_auroc": a, "null_mean": mu, "null_std": sd,
            "perm_z": (a - mu) / sd if sd > 1e-9 else float("nan"),
            "gap_fail_minus_succ_own_dir": float(Xf.mean(0) @ r_seg - Xs.mean(0) @ r_seg),
            "gap_in_pooled_dir": float((Xf.mean(0) - Xs.mean(0)) @ r_pool),
            "s_seg": s_seg, "n_rec_succ": int(len(Xs)), "n_rec_fail": int(len(Xf)),
        }
        print(f"  [{seg}] cv_auroc={a:.3f} z={seg_res[seg]['perm_z']:.2f} "
              f"gap(own)={seg_res[seg]['gap_fail_minus_succ_own_dir']:+.3f}", flush=True)

    dirs["pooled"] = r_pool
    cos_mat = {a: {b: float(dirs[a] @ dirs[b]) for b in dirs} for a in dirs}

    # D. 배포 NPZ 영향 재현
    impact = None
    if args.npz and args.npz.exists():
        z = np.load(args.npz)
        v_dep = z["alpha0_v_steer"].astype(np.float64)
        s_dep = float(np.asarray(z["alpha0_s"]).reshape(()))
        rows = []
        for seg, (lo, hi) in SEGMENTS.items():
            proj = np.concatenate([(r["X"][:cap, lo:hi, :] @ v_dep).ravel() for r in rolls])
            rows.append({"segment": seg, "proj_mean": float(proj.mean()),
                         "proj_std": float(proj.std()),
                         "shift_to_s": float(s_dep - proj.mean()),
                         "shift_in_sigma": float((s_dep - proj.mean()) / (proj.std() + 1e-9))})
        impact = {"deployed_s": s_dep, "cos_with_refit_pooled": float(v_dep @ r_pool),
                  "segments": rows}
        for r_ in rows:
            print(f"  [impact:{r_['segment']}] proj={r_['proj_mean']:.1f}±{r_['proj_std']:.1f} "
                  f"→ s={s_dep:.1f} 이동 {r_['shift_in_sigma']:+.2f}σ", flush=True)

    out = {"cell": args.cell, "layer": args.layer, "cap_records": cap, "T": T,
           "n_rollouts": len(rolls), "n_succ": int(sum(labels)),
           "pooled": {"s": s_pool,
                      "gap_fail_minus_succ": float((Xf_p.mean(0) - Xs_p.mean(0)) @ r_pool)},
           "per_token": per_tok, "segments": seg_res, "cos_matrix": cos_mat,
           "deployed_impact": impact, "n_perm": N_PERM, "rng_seed": RNG_SEED,
           "segment_def": {k: list(v) for k, v in SEGMENTS.items()}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[done] {args.out}", flush=True)


if __name__ == "__main__":
    main()

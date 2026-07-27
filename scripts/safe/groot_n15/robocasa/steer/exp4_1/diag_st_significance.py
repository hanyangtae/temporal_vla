#!/usr/bin/env python3
"""s_t(토큰 위치별 setpoint)의 위치별 구조가 통계적으로 실재하는지 — episode-cluster 검정.

배경 (2026-07-23 대시보드 세션 지적): 앞선 판단에서 s_t 의 추정 노이즈를 record 수로
나눠 SE≈1.2 로 봤는데, **episode 내 record 는 독립이 아니다**(같은 scene·연속 timestep).
유효 표본은 record 수가 아니라 **episode 수**에 가깝다. 그래서 s_t 의 위치별 편차가
노이즈인지 실재 구조인지 다시 판정한다.

방법:
  - 성공 episode e 마다 토큰 t 의 평균 사영 s_t^(e) 산출 (세그먼트 방향 r̂_seg 기준).
  - **episode-cluster bootstrap** (B회, episode 단위 재표본) 으로 s_t 와
    편차 d_t = s_t − mean_{t'∈seg(t)} s_{t'} 의 CI 산출.
  - 판정: d_t 의 95% CI 가 0 을 제외하는 토큰 수 (세그먼트별). 0 이면 s_seg 로 충분.
  - 참고로 record-독립 가정 SE 와 episode-cluster SE 를 함께 출력 (과소평가 배수).

사용: python diag_st_significance.py --manifest <fit tsv> --cell <cell> --layer <blk> --out <json>
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

from fit_mean_diff import SEGMENTS, mean_diff, seg_of_token  # noqa: E402
from fit_phase_conceptor_n15 import FULLTOKEN_MODE  # noqa: E402

N_BOOT = 1000
RNG_SEED = 424106


def load_cell(manifest: Path, cell: str, layer_blk: int):
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
        li = [int(x) for x in d["capture_layers"]].index(layer_blk)
        X = np.stack([np.asarray(r, dtype=np.float32)[li].mean(axis=0)
                      for r in d["hidden_states"]], axis=0)  # [n, T, D]
        rolls.append({"X": X, "success": int(parts[1]), "length": X.shape[0]})
    if not rolls:
        raise SystemExit(f"cell={cell} 없음")
    return rolls


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rolls = load_cell(args.manifest, args.cell, args.layer)
    labels = [r["success"] for r in rolls]
    lens = [r["length"] for r, y in zip(rolls, labels) if y == 1]
    cap = int(np.ceil(np.mean(lens) + np.std(lens)))
    T = rolls[0]["X"].shape[1]

    # 세그먼트 방향 (fit 과 동일 절차)
    Xs = np.concatenate([r["X"][:cap] for r, y in zip(rolls, labels) if y == 1], axis=0)
    Xf = np.concatenate([r["X"][:cap] for r, y in zip(rolls, labels) if y == 0], axis=0)
    v_seg = []
    for _n, lo, hi in SEGMENTS:
        v, _ = mean_diff(Xs[:, lo:hi, :].reshape(-1, Xs.shape[2]),
                         Xf[:, lo:hi, :].reshape(-1, Xf.shape[2]))
        v_seg.append(v)
    v_seg = np.stack(v_seg)

    # episode별 토큰 평균 사영 (성공만) [E, T]
    succ = [r for r, y in zip(rolls, labels) if y == 1]
    ep_st = np.stack([
        np.asarray([(r["X"][:cap, t, :] @ v_seg[seg_of_token(t)]).mean() for t in range(T)])
        for r in succ], axis=0)
    E = ep_st.shape[0]
    s_t = ep_st.mean(axis=0)

    # record-독립 가정 SE (구 계산 — 과소평가) vs episode-cluster SE
    rec_all = np.concatenate([r["X"][:cap] for r in succ], axis=0)
    se_rec = np.asarray([
        (rec_all[:, t, :] @ v_seg[seg_of_token(t)]).std() / np.sqrt(len(rec_all))
        for t in range(T)])
    se_ep = ep_st.std(axis=0, ddof=1) / np.sqrt(E)

    # episode-cluster bootstrap → d_t = s_t − 세그먼트 평균 의 CI
    rng = np.random.default_rng(RNG_SEED)
    boot_d = np.empty((N_BOOT, T))
    for b in range(N_BOOT):
        idx = rng.integers(0, E, E)
        m = ep_st[idx].mean(axis=0)
        for _n, lo, hi in SEGMENTS:
            boot_d[b, lo:hi] = m[lo:hi] - m[lo:hi].mean()
    lo_ci, hi_ci = np.percentile(boot_d, [2.5, 97.5], axis=0)
    sig = (lo_ci > 0) | (hi_ci < 0)

    segs = {}
    for name, lo, hi in SEGMENTS:
        n_tok = hi - lo
        segs[name] = {
            "n_tokens": n_tok,
            "n_significant_deviation": int(sig[lo:hi].sum()),
            "frac_significant": float(sig[lo:hi].mean()),
            "s_mean": float(s_t[lo:hi].mean()),
            "s_within_std": float(s_t[lo:hi].std()),
            "se_episode_median": float(np.median(se_ep[lo:hi])),
            "se_record_median": float(np.median(se_rec[lo:hi])),
            "se_underestimate_x": float(np.median(se_ep[lo:hi]) / (np.median(se_rec[lo:hi]) + 1e-12)),
        }
        print(f"  {name:7s} 토큰{n_tok:3d}  유의편차 {segs[name]['n_significant_deviation']:3d}개"
              f"({segs[name]['frac_significant']*100:4.0f}%)  s내부std={segs[name]['s_within_std']:6.2f}"
              f"  SE_ep={segs[name]['se_episode_median']:6.2f} (record가정 {segs[name]['se_record_median']:5.2f}"
              f", {segs[name]['se_underestimate_x']:.1f}배 과소)", flush=True)

    out = {"cell": args.cell, "layer": args.layer, "cap_records": cap, "T": T,
           "n_succ_episodes": E, "n_boot": N_BOOT, "rng_seed": RNG_SEED,
           "segments": segs,
           "verdict": ("s_t 필요 (세그먼트 내 유의 편차 존재)"
                       if any(v["n_significant_deviation"] > 0 for v in segs.values())
                       else "s_seg 로 충분 (위치별 편차 = 노이즈)"),
           "note": "episode-cluster bootstrap. record 단위 SE 는 episode 내 상관 때문에 "
                   "과소평가 — 위 배수 참조 (2026-07-23 대시보드 세션 지적 반영)."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[{args.cell}] E={E} → {out['verdict']}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""위약 dose-match 가 런타임 분포에서도 유지되는지 — 표본 시프트 민감도 진단 (CPU).

배경 (2026-07-24 Codex 리뷰): placebo 세그먼트 게인(dose_match_scale)은 fit truncation
window `[:cap]` 전 rollout(X_all)에서 계산한다. 처치(setM_permanent, gain 1.0)는 런타임의
자연 dose 를 내는데, 실제 개입 분포는 **t0 이후 실패 토큰**(rescue 대상)이라 calibration
창과 다르다. 창이 다르면 위약 dose 가 처치와 조용히 어긋날 수 있다.

이 진단은 여러 창에서 세그먼트별 dose 비율
  ratio_seg(W) = median_W(scale·|proj_pl − s_pl|) / median_W(|proj_treat − s_treat|)
를 계산한다. calibration 창에서는 ≈1(설계상). fail-only·t0-이후 창에서 1 근처면 런타임
분포 시프트가 dose-match 를 깨지 않는다는 직접 증거. 크게 벗어나면 위약 대조 공정성 문제.

serve·GPU 무접촉. fit pkl(토큰 보존) + 배포 NPZ(v_seg·s_tok·seg_mask=scale) 만 읽는다.
사용: python diag_dose_robustness.py --manifest <fit tsv> --cell <cell> \
        --npz-root <npz> --t0-manifest <t0 tsv> --out <json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import numpy as np

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fit_mean_diff import SEGMENTS, seg_of_token, load_cell_rolls  # noqa: E402


def _seg_dose(recs, v_seg, s_tok):
    """recs [N,T,D] → 세그먼트별 median |proj − s_tok| (게인=1 기준 raw dose)."""
    if len(recs) == 0:
        return np.full(len(SEGMENTS), np.nan)
    T = recs.shape[1]
    v_tok = np.stack([v_seg[seg_of_token(t)] for t in range(T)])          # [T,D]
    move = np.abs(np.einsum("ntd,td->nt", recs.astype(np.float64), v_tok) - s_tok[None, :])
    return np.asarray([float(np.median(move[:, lo:hi])) for _n, lo, hi in SEGMENTS])


def _load_op(npz_dir):
    p = next(Path(npz_dir).glob("dit_L*/conceptors.npz"))
    z = np.load(p)
    return (z["alpha0_v_seg"].astype(np.float64), z["alpha0_s_tok"].astype(np.float64).reshape(-1),
            z["alpha0_seg_mask"].astype(np.float64).reshape(-1))


def _load_t0(path, cell):
    out = {}
    if not path:
        return out
    lines = Path(path).read_text().splitlines()
    h = {k: i for i, k in enumerate(lines[0].split("\t"))}
    for ln in lines[1:]:
        if not ln.strip():
            continue
        r = ln.split("\t")
        if r[h["cell"]] != cell:
            continue
        t0 = r[h["t0_record"]]
        out[int(r[h["episode_idx"]])] = None if t0 in ("", "NA") else int(t0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--npz-root", type=Path, required=True)
    ap.add_argument("--t0-manifest", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    v_t, s_t, mask_t = _load_op(args.npz_root / args.cell / "setM_permanent")
    v_p, s_p, scale = _load_op(args.npz_root / args.cell / "setM_permanent_placebo")
    blk = int(json.loads(next((args.npz_root / args.cell / "setM_permanent")
                              .glob("dit_L*/metadata.json")).read_text())["layer"])
    rolls = load_cell_rolls(args.manifest, args.cell)  # [{tok:[n,L,T,D], success, episode_idx, ...}]
    li = [int(x) for x in rolls[0]["capture_layers"]].index(blk)
    succ_lens = [r["length"] for r in rolls if r["success"] == 1]
    cap = int(np.ceil(np.mean(succ_lens) + np.std(succ_lens)))
    t0map = _load_t0(args.t0_manifest, args.cell)

    def window(kind):
        chunks = []
        for r in rolls:
            X = r["tok"][:, li]                       # [n, T, D]
            n = X.shape[0]
            if kind == "calib":
                chunks.append(X[:cap])
            elif kind == "fail_all" and r["success"] == 0:
                chunks.append(X)
            elif kind == "fail_from_t0" and r["success"] == 0:
                k = t0map.get(r["episode_idx"])
                chunks.append(X[k:] if k is not None and k < n else np.empty((0, X.shape[1], X.shape[2])))
            elif kind == "fail_late" and r["success"] == 0:
                chunks.append(X[cap:] if n > cap else np.empty((0, X.shape[1], X.shape[2])))
        return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 0, 0))

    report = {"cell": args.cell, "layer": blk, "cap": cap,
              "scale": scale.tolist(), "windows": {}}
    print(f"[{args.cell}] L{blk} cap={cap} scale={[round(x, 2) for x in scale]}", flush=True)
    for kind in ("calib", "fail_all", "fail_from_t0", "fail_late"):
        W = window(kind)
        if W.size == 0 or W.shape[0] < 3:
            report["windows"][kind] = {"n_records": int(W.shape[0]) if W.size else 0, "skip": True}
            print(f"  {kind:14s} 표본 부족", flush=True)
            continue
        dt = _seg_dose(W, v_t, s_t)                   # 처치 raw dose (gain 1)
        dp = _seg_dose(W, v_p, s_p)                   # 위약 raw dose
        ratio = (scale * dp) / np.maximum(dt, 1e-9)   # 위약 실 dose / 처치 dose (≈1 이어야 공정)
        report["windows"][kind] = {"n_records": int(W.shape[0]),
                                   "treat_dose": dt.tolist(), "placebo_dose_x_scale": (scale * dp).tolist(),
                                   "dose_ratio_seg": ratio.tolist()}
        flag = "" if np.all((ratio > 0.7) & (ratio < 1.4)) else "  ⚠ dose 어긋남"
        print(f"  {kind:14s} n={W.shape[0]:5d}  dose비율(위약/처치)="
              f"{[round(x, 2) for x in ratio]}{flag}", flush=True)

    # 판정: 런타임 근사 창(fail_from_t0, 없으면 fail_late)의 비율이 [0.7,1.4] 밖이면 문제
    rt = report["windows"].get("fail_from_t0") or report["windows"].get("fail_late") or {}
    rr = rt.get("dose_ratio_seg")
    report["verdict"] = ("런타임 dose-match 유지 (시프트 무해)" if rr and all(0.7 < x < 1.4 for x in rr)
                         else "런타임 dose 어긋남 존재 — 위약 대조 공정성 재검토" if rr
                         else "런타임 창 표본 부족 — 판정 보류")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[{args.cell}] → {report['verdict']}", flush=True)


if __name__ == "__main__":
    main()

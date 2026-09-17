#!/usr/bin/env python3
"""fit 셀(succ/fail)의 평균 vs 분산 분리 진단 (exp4-3 mean_z/var_z 축의 경량판).

phase×layer 마다:
  - mean_auroc: (μ_f−μ_s) 방향 사영의 AUROC (setM 이 쓰는 축)
  - var_ratio : Tr(Σ_f)/Tr(Σ_s) (클래스 내 산포 비)
  - disp_auroc: 자기 클래스 무관, 성공 중심까지 거리 ||x−μ_s|| 의 AUROC
                (평균 이동 없이 산포만 커져도 오름 — 분산-실린 분리 지표)
  - mdist_gap : 위 거리의 클래스 중앙값 비 (fail/succ)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    # pos=fail 점수가 높으면 1.0 쪽
    s = np.concatenate([pos, neg])
    r = s.argsort().argsort().astype(np.float64) + 1
    rp = r[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=Path, required=True)
    ap.add_argument("--scenes", required=True, help="fit scene 콤마목록")
    ap.add_argument("--noises", required=True)
    ap.add_argument("--layers", required=True, help="물리 layer 콤마목록")
    ap.add_argument("--denoise", type=int, default=-1)
    ap.add_argument("--seg", default="all", choices=["state", "future", "action", "all"])
    args = ap.parse_args()

    d = np.load(args.shard, allow_pickle=True)
    meta = json.loads(str(d["meta_json"]))
    cap = [int(x) for x in meta["capture_layers"]]
    seg_i = ["state", "future", "action", "all"].index(args.seg)
    k_i = args.denoise if args.denoise >= 0 else d["X"].shape[2] - 1
    scenes = {int(x) for x in args.scenes.split(",")}
    noises = {int(x) for x in args.noises.split(",")}
    sel = np.array([(int(s) in scenes and int(n) in noises)
                    for s, n in zip(d["scene"], d["noise"])])
    codebook = {int(v): k for k, v in meta["phase_codebook"].items()}
    succ = d["succ"].astype(bool)
    phases = sorted(set(int(p) for p in d["phase_code"][sel]))
    print(f"# {args.shard.name} fit cells: rec={sel.sum()} "
          f"succ_ep 기준 rec={int((sel & succ).sum())}/{int((sel & ~succ).sum())}")
    print(f"{'phase':<18}{'L':>4}{'n_s/n_f':>12}{'mean_auroc':>11}{'var_ratio':>10}"
          f"{'disp_auroc':>11}{'mdist_gap':>10}")
    for ph in phases:
        pm = sel & (d["phase_code"] == ph)
        # 길이 공정화: episode 별 phase 첫 B record 만 (B = 성공 dwell 25퍼센타일)
        dw = []
        for e in set(d["ep_id"][pm & succ]):
            dw.append(int(((d["ep_id"] == e) & pm).sum()))
        if len(dw) < 5:
            continue
        B = max(3, sorted(dw)[len(dw) // 4])
        keep = np.zeros(len(pm), bool)
        for e in set(d["ep_id"][pm]):
            ix = np.where((d["ep_id"] == e) & pm)[0]
            ix = ix[np.argsort(d["rec_idx"][ix])][:B]
            if len(ix) >= min(3, B):
                keep[ix] = True
        pm = pm & keep
        name = codebook.get(ph, str(ph))
        for L in (int(x) for x in args.layers.split(",")):
            li = cap.index(L)
            Xs = d["X"][pm & succ, li, k_i, seg_i, :].astype(np.float64)
            Xf = d["X"][pm & ~succ, li, k_i, seg_i, :].astype(np.float64)
            if len(Xs) < 10 or len(Xf) < 10:
                print(f"{name:<18}{L:>4}{f'{len(Xs)}/{len(Xf)}':>12}  (표본 부족 skip)")
                continue
            mu_s, mu_f = Xs.mean(0), Xf.mean(0)
            v = mu_f - mu_s
            v /= np.linalg.norm(v) + 1e-12
            m_a = auroc(Xf @ v, Xs @ v)
            var_s = ((Xs - mu_s) ** 2).sum(1).mean()
            var_f = ((Xf - mu_f) ** 2).sum(1).mean()
            ds = np.linalg.norm(Xs - mu_s, axis=1)
            df = np.linalg.norm(Xf - mu_s, axis=1)
            print(f"{name:<18}{L:>4}{f'{len(Xs)}/{len(Xf)}':>12}{m_a:>11.3f}"
                  f"{var_f/var_s:>10.2f}{auroc(df, ds):>11.3f}"
                  f"{np.median(df)/np.median(ds):>10.2f}")


if __name__ == "__main__":
    main()

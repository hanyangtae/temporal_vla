#!/usr/bin/env python
"""TW-FINCH 대조군 — 기존 논문 방법을 저자 코드 그대로 돌린다.

M. S. Sarfraz et al., "Temporally-Weighted Hierarchical Clustering for Unsupervised
Action Segmentation", CVPR 2021 (arXiv:2103.11264).
저자 구현: github.com/ssarfraz/FINCH-Clustering  →  TW-FINCH/python/twfinch.py
이 스크립트는 그 파일을 **수정 없이 import** 해서 쓴다 (--twfinch 로 경로 지정).

왜 필요한가
    우리 판독 실험의 대조군이 직접 만든 것뿐이면 "네가 만든 baseline 아니냐"가 된다.
    TW-FINCH 는 비지도 action segmentation 의 표준 비교 대상이고 코드가 공개돼 있어
    같은 데이터에 그대로 얹을 수 있다.

프로토콜 주의 (원 논문 그대로 vs 우리 것)
    TW-FINCH 는 **에피소드(비디오) 하나 안에서** 프레임을 군집화하고, 평가 시 그 에피소드의
    GT 라벨과 헝가리안 매칭으로 군집↔phase 를 대응시킨 뒤 정확도(MoF)를 잰다. 즉 평가 대상
    에피소드의 라벨을 매칭 단계에서 본다 — 우리 판독(테스트 scene 라벨 미사용)보다 **유리한**
    조건이다. 이 차이를 결과에 반드시 병기한다.

입력 신호 두 가지를 각각 돌려 신호원의 효과를 분리한다:
    activation  AE latent 16 차원  (내부)
    action      정책이 낸 7 차원 행동의 record 요약 28 차원 (외부에서 관찰 가능)

실행 (sklearn·scipy 필요 — 예: event-sae-dev env)
    ~/miniconda3/envs/event-sae-dev/bin/python scripts/analysis/grid_phase/twfinch_baseline.py \
        --labels-dir <labels 디렉토리> --traj-dir <traj 트리> --twfinch /path/to/twfinch.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def hungarian_acc(pred, y):
    """군집→phase 를 헝가리안으로 1:1 대응시킨 뒤의 정확도 (TW-FINCH 논문 MoF 규약)."""
    from scipy.optimize import linear_sum_assignment

    pu, yu = np.unique(pred), np.unique(y)
    cost = np.zeros((len(pu), len(yu)))
    for i, p in enumerate(pu):
        for j, c in enumerate(yu):
            cost[i, j] = -np.sum((pred == p) & (y == c))
    r, c = linear_sum_assignment(cost)
    return float(-cost[r, c].sum() / len(y))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labels-dir", type=Path, required=True)
    ap.add_argument("--traj-dir", type=Path, default=None)
    ap.add_argument("--twfinch", type=Path, required=True,
                    help="저자 원본 twfinch.py 경로 (수정 금지)")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("outputs/analysis/grid_phase/phase_readout"))
    ap.add_argument("--req-clust", type=int, default=None,
                    help="원하는 군집 수. 기본은 그 에피소드의 GT phase 수 (CTE 이후 표준)")
    ap.add_argument("--max-ep", type=int, default=40, help="instruction 당 평가 에피소드 수")
    args = ap.parse_args(argv)

    tw = load_module(args.twfinch, "twfinch_authors")
    here = Path(__file__).resolve().parent
    pr = load_module(here / "phase_readout.py", "phase_readout_local")
    traj_idx = pr.index_traj(args.traj_dir) if args.traj_dir else None
    if traj_idx:
        print(f"[traj] {len(traj_idx)} 에피소드 색인", flush=True)

    per, rows = {}, []
    for f in sorted(args.labels_dir.glob("labels_*_k*.npz")):
        slug = f.stem.replace("labels_", "").rsplit("_k", 1)[0]
        d = np.load(f, allow_pickle=True)
        ep = d["ep_id"].astype(np.int64)
        rec = d["rec_idx"].astype(np.int64)
        y_all = d["phase_code"].astype(np.int64)
        Z_all = d["latent"].astype(np.float64)
        scene = d["scene"].astype(np.int64)
        noise = (d["noise"].astype(np.int64) if "noise" in d.files
                 else np.zeros(len(ep), np.int64))

        acc_act, acc_lat = [], []
        for e in np.unique(ep)[: args.max_ep]:
            m = ep == e
            order = np.argsort(rec[m])
            y = y_all[m][order]
            if len(np.unique(y)) < 2 or len(y) < 8:
                continue
            k_req = args.req_clust or int(len(np.unique(y)))

            for tag, X, bucket in (("latent", Z_all[m][order], acc_lat),
                                   ("action", None, acc_act)):
                if tag == "action":
                    if traj_idx is None:
                        continue
                    p = traj_idx.get((slug, int(scene[m][0]), int(noise[m][0])))
                    if p is None:
                        continue
                    X = pr.action_features(p, int(m.sum()))
                    if X is None:
                        continue
                    X = X[np.clip(rec[m][order] - rec[m].min(), 0, len(X) - 1)]
                try:                       # 저자 코드 호출 (tw_finch=True 가 TW-FINCH)
                    c, _n, req_c = tw.FINCH(np.ascontiguousarray(X, dtype=np.float64),
                                            req_clust=k_req, distance="cosine",
                                            tw_finch=True, verbose=False)
                except Exception as exc:   # 에피소드가 짧으면 병합이 실패할 수 있다
                    print(f"  [warn] {slug} ep{e} {tag}: {exc}", flush=True)
                    continue
                lab = req_c if req_c is not None else c[:, 0]
                bucket.append(hungarian_acc(np.asarray(lab).ravel(), y))

        if not acc_lat:
            continue
        entry = {"n_ep": len(acc_lat),
                 "twfinch_latent_acc": float(np.mean(acc_lat))}
        if acc_act:
            entry["twfinch_action_acc"] = float(np.mean(acc_act))
        per[slug] = entry
        rows.append((slug, entry))
        print(f"[{slug:<22}] TW-FINCH  latent {entry['twfinch_latent_acc']:.3f}"
              + (f" | action {entry['twfinch_action_acc']:.3f}" if acc_act else "")
              + f"   (ep {entry['n_ep']})", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"script": "scripts/analysis/grid_phase/twfinch_baseline.py",
            "method": "TW-FINCH (Sarfraz et al., CVPR 2021) — 저자 twfinch.py 무수정 호출",
            "protocol": "에피소드 단위 군집화 + 그 에피소드 GT 로 헝가리안 매칭 (원 논문 MoF 규약)."
                        " 평가 에피소드의 라벨을 매칭에 쓰므로 우리 판독보다 유리한 조건.",
            "req_clust": args.req_clust or "그 에피소드의 GT phase 수",
            "max_ep_per_instruction": args.max_ep}
    (args.out_dir / "twfinch.json").write_text(
        json.dumps({"per_instruction": per, "meta": meta}, ensure_ascii=False, indent=1))

    for key in ("twfinch_latent_acc", "twfinch_action_acc"):
        v = [a[key] for _, a in rows if key in a]
        if v:
            print(f"\n{key}: 중앙값 {np.median(v):.3f} (n={len(v)})")


if __name__ == "__main__":
    main()

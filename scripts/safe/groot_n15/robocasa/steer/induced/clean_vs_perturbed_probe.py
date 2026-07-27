"""clean baseline vs perturbed — perturbation별·layer별 activation 차이/분리 프로브 (07-27).

두 지표 (config × layer[0,2,4,8,10,12,15,VL]):
1. paired onset Δ: 같은 episode 의 clean↔perturbed 를 record 정렬해 온셋 창의
   ‖h_p−h_c‖/‖h_c‖ 중앙값. P1/P2 는 trigger 이전 bitwise 동일(실측)이라 순수 섭동 효과.
   C1 은 record0 부터(관측 변화 즉시). G1 은 EE 이동으로 state 자체가 달라 record 정렬
   불가 → NaN (분포 지표만).
2. held-out 분리 AUROC: clean-succ vs perturbed(전 라벨) episode-level —
   mean-diff 방향을 짝수 ep 에서 fit, 홀수 ep 에서 AUROC (사전등록 파티션과 정합).
   perturbed 쪽은 시간분리 절단(record_start 규칙: C1=0, G1=1, P1=창끝+4, P2=창끝+2) 적용.

  docker exec lerobot python clean_vs_perturbed_probe.py \
      --clean-dir <baseline_cap/raw_rollouts/...> --capture-root <p0/capture> \
      --record-start <manifests/cal+fit union tsv> --out probe.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from induced_common import load_roll_any, read_record_start  # noqa: E402

LAYER_KEYS = [0, 2, 4, 8, 10, 12, 15, "VL"]


def _ep_of(p: Path) -> int:
    return int(re.search(r"--ep(\d+)--", p.name).group(1))


def _feat(r: dict, lk) -> np.ndarray:
    if lk == "VL":
        return np.asarray(r["vl"], dtype=np.float32)
    return r["dit"][:, r["capture_layers"].index(lk), :].astype(np.float32)


def _auroc(s0, s1):
    s = np.concatenate([s0, s1]); n0, n1 = len(s0), len(s1)
    if n0 == 0 or n1 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clean-dir", required=True)
    ap.add_argument("--capture-root", required=True)
    ap.add_argument("--record-start", required=True, help="perturbed 절단 rs union tsv")
    ap.add_argument("--onset-len", type=int, default=6, help="paired Δ 온셋 창(record 수)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rs = read_record_start(args.record_start)
    clean = {}
    for p in sorted(Path(args.clean_dir).glob("task*--ep*--succ1.pkl")):
        clean[_ep_of(p)] = load_roll_any(p)
    print(f"clean-succ {len(clean)} eps")

    out = {"onset_len": args.onset_len, "configs": {}}
    for cfg_dir in sorted(Path(args.capture_root).iterdir()):
        cfg = cfg_dir.name
        eps = []
        for p in sorted(cfg_dir.glob("raw_rollouts/*/*/task*--ep*--succ*.pkl")):
            r = load_roll_any(p)
            eps.append((_ep_of(p), r, rs.get(str(p.resolve()), 0),
                        int(r.get("success", 0))))
        if not eps:
            continue
        mode = None
        res = {}
        for lk in LAYER_KEYS:
            # -- 1. paired onset Δ (C1: r0~, P1/P2: trigger~; G1 은 정렬 불가) ----------
            deltas = []
            for ep, r, start, _s in eps:
                if ep not in clean:
                    continue
                if cfg.startswith("g1"):
                    break  # state 정렬 불가
                on0 = 0 if cfg.startswith("c1") else max(start - (4 if cfg.startswith("p1") else 2), 0)
                Xc, Xp = _feat(clean[ep], lk), _feat(r, lk)
                hi = min(on0 + args.onset_len, Xc.shape[0], Xp.shape[0])
                if hi <= on0:
                    continue
                num = np.linalg.norm(Xp[on0:hi] - Xc[on0:hi], axis=1)
                den = np.clip(np.linalg.norm(Xc[on0:hi], axis=1), 1e-9, None)
                deltas.extend((num / den).tolist())
            # -- 2. held-out 분리 AUROC (짝수 fit → 홀수 eval) --------------------------
            def ep_vec(r, start, lk=lk):
                X = _feat(r, lk)[start:]
                return X.mean(axis=0) if X.shape[0] else None
            fit_c = [ep_vec(v, 0) for e, v in clean.items() if e % 2 == 0]
            fit_p = [ep_vec(r, s) for e, r, s, _ in eps if e % 2 == 0]
            ev_c = [ep_vec(v, 0) for e, v in clean.items() if e % 2 == 1]
            ev_p = [ep_vec(r, s) for e, r, s, _ in eps if e % 2 == 1]
            fit_c = [v for v in fit_c if v is not None]; fit_p = [v for v in fit_p if v is not None]
            ev_c = [v for v in ev_c if v is not None]; ev_p = [v for v in ev_p if v is not None]
            w = np.mean(fit_p, axis=0) - np.mean(fit_c, axis=0)
            w /= max(np.linalg.norm(w), 1e-12)
            auroc = _auroc(np.asarray([v @ w for v in ev_c]),
                           np.asarray([v @ w for v in ev_p]))
            res[str(lk)] = {
                "paired_onset_delta_median": (float(np.median(deltas)) if deltas else None),
                "paired_n_records": len(deltas),
                "sep_auroc_heldout": auroc,
                "n_eval": [len(ev_c), len(ev_p)],
            }
        out["configs"][cfg] = res
        row = " ".join(f"L{lk}:{res[str(lk)]['sep_auroc_heldout']:.2f}" for lk in LAYER_KEYS)
        drow = " ".join(
            f"L{lk}:{res[str(lk)]['paired_onset_delta_median']:.3f}"
            if res[str(lk)]["paired_onset_delta_median"] is not None else f"L{lk}:--"
            for lk in LAYER_KEYS)
        print(f"[{cfg}] AUROC {row}")
        print(f"[{cfg}] Δonset {drow}")
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

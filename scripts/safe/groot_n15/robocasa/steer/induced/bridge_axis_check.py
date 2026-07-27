"""exp4-2 bridge 게이트 — 유도실패 축 ↔ 자연실패 축 정렬 검사 (24b §3, eval 0판 분석).

pre-registered 지표 (판정·게이트 결정은 confound-audit 경유 사용자 보고 — 본 스크립트는 계산기):
  1. mean-diff cosine  : cos(μ_pf−μ_ps, μ_nf−μ_ns), layer×phase-bin.
     episode cluster bootstrap CI + 양측 label-permutation null.
  2. cross-AUROC       : r̂_induced 로 자연 succ/fail 점수화 — episode-level(주지표) +
     record-level(보조). 역방향(natural→induced)도 대칭 산출.
  3. 선형 분리도       : regularized LDA (sklearn 불요, WA-LQR SVM-분리도의 결정론 대용
     — 결과 필드명에 명기). induced 로 fit → natural episode-level held-out acc.
  4. mode 층화         : C1/G1/P1/P2 별 subgroup (P1 시각 불연속 별도 층 — 24b §1.1).

입력 manifest: pkl<TAB>label[<TAB>scene] (fit_manifest.tsv 규약). --induced-record-start 로
유도측 절단 적용. pkl 로드 torch 필요 — lerobot 컨테이너 실행.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from induced_common import REPO, load_roll_any, read_manifest, read_record_start  # noqa: E402

# 6p proximity 라벨 어휘 기준 (transport 라벨은 6p 에 없음 — carry 구간 ≈ "place")
DEFAULT_BINS = ("global", "reach-to-object", "grasp", "place", "insert-settle")

_read_manifest = read_manifest
_read_rs = read_record_start


def _load_side(rows, rs_map, capture_layer, mode_map=None):
    """→ list of dict(ep): {X_bin: {bin: [n,D]}, label, mode}. capture_layer: int | "VL"."""
    eps = []
    for p, label in rows:
        r = load_roll_any(p)
        start = rs_map.get(str(p.resolve()), 0)
        if capture_layer == "VL":
            if r.get("vl") is None:
                raise SystemExit(f"{p.name}: VL 없음")
            X = np.asarray(r["vl"], dtype=np.float32)[start:]
        else:
            cap = r["capture_layers"]
            if capture_layer not in cap:
                raise SystemExit(f"{p.name}: layer {capture_layer} 없음 (cap={cap})")
            li = cap.index(capture_layer)
            X = r["dit"][start:, li, :].astype(np.float32)
        phases = r["phases"][start:]
        if not X.shape[0]:
            continue
        xb = {"global": X}
        for b in DEFAULT_BINS:
            if b == "global":
                continue
            idx = [i for i, ph in enumerate(phases) if ph == b]
            if idx:
                xb[b] = X[idx]
        eps.append({"x": xb, "label": int(label),
                    "mode": (mode_map or {}).get(str(p), "all"), "pkl": str(p)})
    return eps


def _mean_diff(eps, bin_name):
    xf = [e["x"][bin_name] for e in eps if e["label"] == 0 and bin_name in e["x"]]
    xs = [e["x"][bin_name] for e in eps if e["label"] == 1 and bin_name in e["x"]]
    if not xf or not xs:
        return None, 0, 0
    # episode 동등 가중: episode 별 mean 후 클래스 mean (길이 confound 완화 — Gate1 공통 통제)
    mf = np.mean([x.mean(axis=0) for x in xf], axis=0)
    ms = np.mean([x.mean(axis=0) for x in xs], axis=0)
    return mf - ms, len(xf), len(xs)


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(a @ b / (na * nb))


def _auroc(scores0, scores1):
    """label1(fail 방향 고득점 기대) AUROC — rank 기반, 동률 0.5."""
    s = np.concatenate([scores0, scores1])
    n0, n1 = len(scores0), len(scores1)
    if n0 == 0 or n1 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # 동률 평균 rank
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    r1 = ranks[n0:].sum()
    return float((r1 - n1 * (n1 + 1) / 2) / (n0 * n1))


def _ep_scores(eps, bin_name, w):
    out0, out1 = [], []
    for e in eps:
        if bin_name not in e["x"]:
            continue
        sc = float((e["x"][bin_name] @ w).mean())  # episode-level = record 점수 평균
        (out1 if e["label"] == 0 else out0).append(sc)  # fail(0) 이 고득점 기대측
    return np.asarray(out0), np.asarray(out1)


def _lda_dir(eps, bin_name, reg=1e-3):
    xf = [e["x"][bin_name] for e in eps if e["label"] == 0 and bin_name in e["x"]]
    xs = [e["x"][bin_name] for e in eps if e["label"] == 1 and bin_name in e["x"]]
    if not xf or not xs:
        return None
    Xf, Xs = np.concatenate(xf), np.concatenate(xs)
    mu_f, mu_s = Xf.mean(axis=0), Xs.mean(axis=0)
    Xc = np.concatenate([Xf - mu_f, Xs - mu_s])
    cov = (Xc.T @ Xc) / max(len(Xc) - 2, 1)
    cov += reg * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
    return np.linalg.solve(cov, mu_f - mu_s)


def _pair_stats(ind, nat, bin_name, rng, n_boot, n_perm):
    d_ind, nf_i, ns_i = _mean_diff(ind, bin_name)
    d_nat, nf_n, ns_n = _mean_diff(nat, bin_name)
    if d_ind is None or d_nat is None:
        return None
    row = {"n_induced": [nf_i, ns_i], "n_natural": [nf_n, ns_n],
           "cosine": _cos(d_ind, d_nat)}
    # cluster bootstrap (episode 재표본, 양측 독립)
    boots = []
    for _ in range(n_boot):
        bi = [ind[i] for i in rng.integers(0, len(ind), len(ind))]
        bn = [nat[i] for i in rng.integers(0, len(nat), len(nat))]
        di, *_ = _mean_diff(bi, bin_name)
        dn, *_ = _mean_diff(bn, bin_name)
        if di is not None and dn is not None:
            boots.append(_cos(di, dn))
    if boots:
        row["cosine_ci95"] = [float(np.percentile(boots, 2.5)),
                              float(np.percentile(boots, 97.5))]
    # 양측 label-permutation null (양측 라벨 각각 순열)
    null = []
    for _ in range(n_perm):
        li = rng.permutation([e["label"] for e in ind])
        ln = rng.permutation([e["label"] for e in nat])
        pi = [{**e, "label": int(l)} for e, l in zip(ind, li)]
        pn = [{**e, "label": int(l)} for e, l in zip(nat, ln)]
        di, *_ = _mean_diff(pi, bin_name)
        dn, *_ = _mean_diff(pn, bin_name)
        if di is not None and dn is not None:
            null.append(abs(_cos(di, dn)))
    if null:
        row["perm_p_twosided"] = float(
            (np.sum(np.asarray(null) >= abs(row["cosine"])) + 1) / (len(null) + 1))
    # cross-AUROC (episode-level 주지표 + record-level 보조), 양방향
    for tag, src, dst, d_src in (("ind_to_nat", ind, nat, d_ind),
                                 ("nat_to_ind", nat, ind, d_nat)):
        w = d_src / max(np.linalg.norm(d_src), 1e-12)
        s0, s1 = _ep_scores(dst, bin_name, w)
        row[f"auroc_ep_{tag}"] = _auroc(s0, s1)
        rec0 = np.concatenate([e["x"][bin_name] @ w for e in dst
                               if e["label"] == 1 and bin_name in e["x"]] or [np.zeros(0)])
        rec1 = np.concatenate([e["x"][bin_name] @ w for e in dst
                               if e["label"] == 0 and bin_name in e["x"]] or [np.zeros(0)])
        row[f"auroc_rec_{tag}"] = _auroc(rec0, rec1)
    # 선형 분리도 (LDA-대용): induced fit → natural episode acc
    w = _lda_dir(ind, bin_name)
    if w is not None:
        s0, s1 = _ep_scores(nat, bin_name, w)
        thr = float(np.median(np.concatenate([s0, s1]))) if len(s0) + len(s1) else 0.0
        acc = (np.sum(s1 > thr) + np.sum(s0 <= thr)) / max(len(s0) + len(s1), 1)
        row["lda_ep_acc_ind_to_nat"] = float(acc)
        row["lda_note"] = "regularized-LDA (WA-LQR SVM 분리도의 결정론 대용)"
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--induced-manifest", required=True)
    ap.add_argument("--induced-record-start", default=None)
    ap.add_argument("--induced-details", default=None,
                    help="build_induced_fit_manifest details.json — mode 층화용")
    ap.add_argument("--natural-manifest", required=True)
    ap.add_argument("--layers", default="8,12")
    ap.add_argument("--bins", default=",".join(DEFAULT_BINS))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260722)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mode_map = {}
    if args.induced_details:
        det = json.loads(Path(args.induced_details).read_text())
        for r in det.get("rows", []):
            mode_map[r["pkl"]] = r["note"].split("/")[0]

    rng = np.random.default_rng(args.seed)
    rs = _read_rs(args.induced_record_start)
    ind_rows = _read_manifest(args.induced_manifest)
    nat_rows = _read_manifest(args.natural_manifest)
    bins = [b.strip() for b in args.bins.split(",") if b.strip()]
    result = {"config": vars(args), "layers": {}}
    layer_keys = ["VL" if x.strip() == "VL" else int(x)
                  for x in args.layers.split(",") if x.strip()]
    for layer in layer_keys:
        ind = _load_side(ind_rows, rs, layer, mode_map)
        nat = _load_side(nat_rows, {}, layer)
        lay = {}
        for b in bins:
            row = _pair_stats(ind, nat, b, rng, args.n_boot, args.n_perm)
            if row:
                lay[b] = row
            # mode 층화 (global bin 만 — 표본 소형)
            if b == "global":
                for mode in sorted({e["mode"] for e in ind if e["mode"] != "all"}):
                    sub = [e for e in ind if e["mode"] == mode]
                    if len(sub) >= 4:
                        r2 = _pair_stats(sub, nat, b, rng,
                                         max(args.n_boot // 4, 100),
                                         max(args.n_perm // 4, 100))
                        if r2:
                            lay[f"global[{mode}]"] = r2
        result["layers"][f"L{layer}"] = lay
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))
    for lk, lay in result["layers"].items():
        for b, row in lay.items():
            print(f"[{lk}/{b}] cos={row.get('cosine', float('nan')):.3f} "
                  f"ci={row.get('cosine_ci95')} p={row.get('perm_p_twosided')} "
                  f"AUROCep(i→n)={row.get('auroc_ep_ind_to_nat', float('nan')):.3f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Cluster 공유도 → detector 학습 단위 판단 검증 (slug 쌍 A→B directed).

질문: activation 만으로 계산한 "공유도"가 실제 failure-detector 전이 성능
(`failure_detector_sim.py --arm loto --shards A,B` 의 td AUROC)을 예측하는가.
예측이 성립하면 detector 를 새로 학습해 보기 전에 그룹핑을 정할 수 있다.

지표 2종 (사용자 결정 2026-08-19: S1 이 최종 판정, S2 는 정렬 틀·참고):

  S1. 실패방향 전이 AUROC — A 의 GT-phase 조건부 fail−succ diff-of-means 방향
      w_{A,p} 를 B 의 held-out record 에 투영해 succ/fail AUROC. LOTO 전이 실패의
      병목이 cluster 겹침 이전에 **방향 부호 반전**이었다는 실측(docs/43 §7)을
      직접 측정. + 방향 부호 일치 cos(w_{A,p}, w_{B,p}).
  S2. cluster 구조 전이 — A 에서 fit 한 PCA-64w+KMeans(k) 를 B 에 centroid 할당
      → B 에서의 margin(vs clock) 을 "B 자체 fit 대비 유지율"로. + 원공간
      centroid greedy cosine 매칭. (MI 는 과분할에 둔감하므로 margin 유지율로.)

길이·scene 통제 (docs/43 프로토콜 상속):
  - phase-gt dwell cap: TRAIN 성공 dwell(>0) ceil(μ+σ), ddof=0
    (`fit_setm.py::phase_dwell_caps` = `failure_detector_sim.py` 와 동일 규약)
  - scene-중심화 전/후 병기 (`paper_supplements.py::residualize_by_scene`)
  - 조기 한정: relpos < 0.5 record 병기 (후기 공유 = 결말 흔적)
  - length_auroc 자가감사 병기 (episode record 수만의 AUROC)
  - scene split 은 detector 러너의 `split_scenes` 와 **동일 구현** (seed, task
    crc32 결정적 셔플) — 상관 분석이 같은 fold 위에서 돌게 한다.
  - 위약: B episode 라벨을 scene 블록 내 순열 → S1 z.

사용:
    python cluster_share_transfer.py --shards-dir <segA> --out <dir> \
        --pairs "ppcc_bread:ppcc_candle,ppcc_candle:ppcc_bread" \
        --seeds 0,1,2 --k 8
    # --pairs all → 전 slug 순서쌍(자기전이 포함; A=B 는 배관 게이트)

실행 환경: 승준 `~/anaconda3/bin/python` (numpy only). BLAS cap 은
intrinsic_phase import 시 설정됨(OMP≤16).
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import intrinsic_phase as IP          # noqa: E402  (BLAS cap 포함)
import numpy as np                    # noqa: E402
from paper_supplements import residualize_by_scene  # noqa: E402

EPS = 1e-8
TD_NOTE = "전이 정답지는 failure_detector_sim.py (exp/safe-length-ablation) 산출"


# =============================================================================
# scene split — failure_detector_sim.split_scenes 와 동일 구현 (fold 일치용)
# =============================================================================

def split_scenes(task: str, scenes, n_train=6, n_calib=2, n_test=2, seed=0):
    """정렬 후 (seed, crc32(task)) 결정적 셔플 → {test, calib, train}.

    출처: failure_detector_sim.py::split_scenes (exp/safe-length-ablation) —
    상대 세션 detector fold 와 같은 분할을 재현해야 상관이 깨끗하다.
    """
    uniq = sorted(set(int(s) for s in scenes))
    if len(uniq) < 3:
        raise ValueError(f"{task}: scene {len(uniq)}개 — 분할 불가(최소 3)")
    rng = np.random.default_rng([seed, zlib.crc32(task.encode("utf-8"))])
    order = [uniq[i] for i in rng.permutation(len(uniq))]
    n = len(order)
    te = min(n_test, max(1, n - 2))
    ca = min(n_calib, max(1, n - te - 1))
    return {"test": sorted(order[:te]), "calib": sorted(order[te:te + ca]),
            "train": sorted(order[te + ca:])}


# =============================================================================
# 길이 통제 — phase-gt dwell cap (fit_setm/failure_detector_sim 규약)
# =============================================================================

def _ceil_mu_sigma(vals):
    v = np.asarray([x for x in vals if x > 0], dtype=np.float64)
    return int(np.ceil(v.mean() + v.std()))          # ddof=0 (모집단)


def phase_dwell_caps(phase, succ, ep_id) -> dict:
    """phase code → dwell cap. 성공 episode 의 dwell(>0) ceil(μ+σ).

    출처 규약: scripts/fit/fit_setm.py::phase_dwell_caps (dev 정본).
    성공 dwell 없는 phase 는 미포함 → 호출부에서 해당 record drop.
    """
    caps = {}
    succ_eps = np.unique(ep_id[succ == 1])
    for c in np.unique(phase):
        dw = [int(((ep_id == e) & (phase == c)).sum()) for e in succ_eps]
        dw = [d for d in dw if d > 0]
        if dw:
            caps[int(c)] = _ceil_mu_sigma(dw)
    return caps


def cap_mask(phase, ep_id, rec_idx, caps) -> np.ndarray:
    """episode×phase 별 시간순 앞쪽 cap 개만 True. cap 없는 phase 는 False."""
    order = np.lexsort((rec_idx, ep_id))
    keep = np.zeros(len(phase), dtype=bool)
    seen: dict = {}
    for i in order:
        key = (int(ep_id[i]), int(phase[i]))
        cnt = seen.get(key, 0)
        cap = caps.get(int(phase[i]))
        if cap is not None and cnt < cap:
            keep[i] = True
        seen[key] = cnt + 1
    return keep


def relpos(ep_id, rec_idx) -> np.ndarray:
    """episode 내 진행도 0..1 (rec_idx 기준)."""
    out = np.empty(len(ep_id), np.float64)
    for e in np.unique(ep_id):
        m = ep_id == e
        r = rec_idx[m].astype(np.float64)
        span = max(r.max() - r.min(), 1.0)
        out[m] = (r - r.min()) / span
    return out


# =============================================================================
# AUROC / 순열
# =============================================================================

def auroc(scores, y) -> float | None:
    scores = np.asarray(scores, np.float64)
    y = np.asarray(y, np.int64)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = scores.argsort().argsort().astype(np.float64) + 1.0
    # 동점 평균 순위
    order = np.argsort(scores)
    s = scores[order]
    rr = np.empty(len(s), np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        rr[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    r[order] = rr
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def perm_z(ep_scores, ep_y, ep_scene, n_perm=100, seed=0):
    """scene 블록 내 라벨 순열 null 대비 z. (scene 이 1종이면 전체 순열.)"""
    obs = auroc(ep_scores, ep_y)
    if obs is None:
        return None, None
    rng = np.random.default_rng(seed)
    null = []
    sc = np.asarray(ep_scene)
    y = np.asarray(ep_y).copy()
    for _ in range(n_perm):
        yp = y.copy()
        for s in np.unique(sc):
            m = sc == s
            yp[m] = rng.permutation(yp[m])
        a = auroc(ep_scores, yp)
        if a is not None:
            null.append(a)
    if len(null) < 10:
        return obs, None
    mu, sd = float(np.mean(null)), float(np.std(null))
    return obs, (obs - mu) / max(sd, 1e-6)


# =============================================================================
# S1 — 실패방향 전이
# =============================================================================

def fit_directions(feat, phase, succ, mask) -> dict:
    """phase → (w 단위벡터, n_fail, n_succ). fail/succ 각 3 record 미만이면 skip."""
    dirs = {}
    for c in np.unique(phase[mask]):
        m = mask & (phase == c)
        f, s = feat[m & (succ == 0)], feat[m & (succ == 1)]
        if len(f) < 3 or len(s) < 3:
            continue
        w = f.mean(0, dtype=np.float64) - s.mean(0, dtype=np.float64)
        nrm = np.linalg.norm(w)
        if nrm < EPS:
            continue
        dirs[int(c)] = (w / nrm, len(f), len(s))
    return dirs


def project_episode_scores(feat, phase, ep_id, mask, w_by_phase):
    """episode 단위 equal-budget 점수: phase 별 (cap 내) record 투영의 평균을
    phase 간 평균. 방향이 있는 phase record 가 없으면 None."""
    out = {}
    for e in np.unique(ep_id):
        vals = []
        for c, (w, _, _) in w_by_phase.items():
            m = mask & (ep_id == e) & (phase == c)
            if m.any():
                vals.append(float(feat[m].mean(0, dtype=np.float64) @ w))
        out[int(e)] = float(np.mean(vals)) if vals else None
    return out


def s1_transfer(A, B, mask_a, mask_b, n_perm, seed):
    """A 방향 → B 투영 AUROC (+부호 일치, 위약 z, length 자가감사)."""
    dirs_a = fit_directions(A["feat"], A["phase_code"], A["succ"], mask_a)
    if not dirs_a:
        return {"s1_auroc": None, "s1_note": "A 방향 없음"}
    # B 자체 방향과의 부호/코사인 (전이 없이 B 내부에서 적합 — 참고 지표)
    dirs_b = fit_directions(B["feat"], B["phase_code"], B["succ"], mask_b)
    cos, agree = [], []
    for c, (wa, _, _) in dirs_a.items():
        if c in dirs_b:
            cc = float(wa @ dirs_b[c][0])
            cos.append(cc)
            agree.append(cc > 0)
    ep_sc = project_episode_scores(B["feat"], B["phase_code"], B["ep_id"],
                                   mask_b, dirs_a)
    eps = sorted(e for e, v in ep_sc.items() if v is not None)
    if not eps:
        return {"s1_auroc": None, "s1_note": "B 겹치는 phase 없음"}
    ep_arr = np.asarray(eps)
    y = np.array([1 - int(B["succ"][B["ep_id"] == e][0]) for e in ep_arr])  # 1=fail
    scn = np.array([int(B["scene"][B["ep_id"] == e][0]) for e in ep_arr])
    sc = np.array([ep_sc[int(e)] for e in ep_arr])
    a, z = perm_z(sc, y, scn, n_perm=n_perm, seed=seed)
    # length 자가감사: episode record 수만으로 낸 AUROC
    ep_len = np.array([int((B["ep_id"] == e).sum()) for e in ep_arr], np.float64)
    return {
        "s1_auroc": None if a is None else round(a, 4),
        "s1_perm_z": None if z is None else round(z, 2),
        "s1_n_phase": len(dirs_a),
        "s1_n_phase_shared": len(cos),
        "s1_cos_med": round(float(np.median(cos)), 3) if cos else None,
        "s1_sign_agree": round(float(np.mean(agree)), 3) if agree else None,
        "s1_n_ep": len(eps), "s1_n_fail": int(y.sum()),
        "length_auroc": (lambda la: None if la is None else round(la, 4))(
            auroc(ep_len, y)),
    }


# =============================================================================
# S2 — cluster 구조 전이
# =============================================================================

def kmeans_labels(feat, K, seed):
    """PCA-64w → KMeans(K). intrinsic_phase 부품 재사용. 전 행 fit."""
    stats, _evr = IP.pca_fit(feat, 64)
    zw = IP.pca_apply(feat, stats, whiten=True)
    cent = IP.kmeans_numpy(zw, K, n_init=5, max_iter=300, seed=seed)
    lab, _ = IP._assign(zw, cent)
    return np.asarray(lab), stats, cent


def margin_bits(labels, phase, ep_id, rec_idx):
    """margin = MI(labels;phase) − MI(clock;phase). intrinsic_phase 부품."""
    order = np.lexsort((rec_idx, ep_id))
    lab, ph, ep = labels[order], phase[order], ep_id[order]
    mi, _, _ = IP.mi_bits(lab, ph)
    tfrac = IP.time_fraction(ep)
    clk = IP.clock_clusters(tfrac, len(np.unique(lab)))
    mi_c, _, _ = IP.mi_bits(clk, ph)
    return mi - mi_c


def s2_transfer(A, B, mask_a, mask_b, K, seed):
    """A 클러스터를 B 에 이식했을 때 margin 유지율 + 원공간 centroid 매칭."""
    lab_a, stats_a, cent_a = kmeans_labels(A["feat"][mask_a], K, seed)
    zb = IP.pca_apply(B["feat"][mask_b], stats_a, whiten=True)
    lab_b_trans, _ = IP._assign(zb, cent_a)
    lab_b_self, _, _ = kmeans_labels(B["feat"][mask_b], K, seed)

    ph_b = B["phase_code"][mask_b]
    ep_b = B["ep_id"][mask_b]
    ri_b = B["rec_idx"][mask_b]
    m_trans = margin_bits(np.asarray(lab_b_trans), ph_b, ep_b, ri_b)
    m_self = margin_bits(np.asarray(lab_b_self), ph_b, ep_b, ri_b)

    # 원공간 centroid (클러스터별 원 feature 평균) greedy cosine 매칭
    def _cent_raw(feat, lab, K):
        return np.stack([
            feat[lab == k].mean(0, dtype=np.float64) if (lab == k).any()
            else np.zeros(feat.shape[1]) for k in range(K)])
    ca = _cent_raw(A["feat"][mask_a], lab_a, K)
    cb = _cent_raw(B["feat"][mask_b], np.asarray(lab_b_self), K)
    ca_n = ca / (np.linalg.norm(ca, axis=1, keepdims=True) + EPS)
    cb_n = cb / (np.linalg.norm(cb, axis=1, keepdims=True) + EPS)
    sim = ca_n @ cb_n.T
    used, match = set(), []
    for _ in range(K):
        i, j = np.unravel_index(
            np.argmax(np.where(
                np.isin(np.arange(K), list(used))[:, None] |
                np.isin(np.arange(K), [m[1] for m in match])[None, :],
                -np.inf, sim)), sim.shape)
        if not np.isfinite(sim[i, j]):
            break
        used.add(int(i))
        match.append((int(i), int(j), float(sim[i, j])))
    cos_matched = [m[2] for m in match]
    return {
        "s2_margin_transfer": round(m_trans, 4),
        "s2_margin_self": round(m_self, 4),
        "s2_margin_keep": round(m_trans / m_self, 3) if abs(m_self) > 1e-6 else None,
        "s2_centroid_cos_med": round(float(np.median(cos_matched)), 3)
                               if cos_matched else None,
    }


# =============================================================================
# main
# =============================================================================

def prepare(shard, seed, centering, early):
    """전처리 마스크: train-scene 한정(방향 fit 용) / 전 scene(평가 용),
    phase-gt cap, relpos, (옵션) scene-중심화."""
    sp = split_scenes(shard["name"], shard["scene"], seed=seed)
    feat = shard["feat"]
    if centering == "scene":
        feat = residualize_by_scene(feat, shard["scene"])
    caps = phase_dwell_caps(shard["phase_code"], shard["succ"], shard["ep_id"])
    keep = cap_mask(shard["phase_code"], shard["ep_id"], shard["rec_idx"], caps)
    if early is not None:
        keep &= relpos(shard["ep_id"], shard["rec_idx"]) < early
    train_m = np.isin(shard["scene"], sp["train"]) & keep
    test_m = np.isin(shard["scene"], sp["test"]) & keep
    out = dict(shard)
    out["feat"] = feat
    return out, {"train": train_m, "test": test_m, "all": keep, "split": sp}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shards-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--pairs", required=True,
                    help='"A:B,C:D" 또는 all (전 순서쌍, 자기전이 포함)')
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--early", type=float, default=0.5,
                    help="relpos 상한 (0 이하면 전체 사용)")
    ap.add_argument("--centering", choices=("both", "scene", "none"), default="both")
    ap.add_argument("--eval-scope", choices=("test", "all"), default="all",
                    help="B 평가 범위. detector LOTO 는 held-out slug 전 episode "
                         "평가이므로 기본 all")
    ap.add_argument("--n-perm", type=int, default=100)
    args = ap.parse_args(argv)

    paths = IP.discover_shards(args.shards_dir, None)
    shards = {p.stem: IP.load_shard(p) for p in paths}
    for s in shards.values():
        print(f"[load] {s['name']}: n={len(s['feat'])}", flush=True)

    if args.pairs.strip() == "all":
        names = sorted(shards)
        pairs = [(a, b) for a in names for b in names]
    else:
        pairs = [tuple(x.split(":")) for x in args.pairs.split(",") if x.strip()]

    seeds = [int(x) for x in args.seeds.split(",")]
    early = args.early if args.early > 0 else None
    centerings = (["none", "scene"] if args.centering == "both"
                  else [args.centering])

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        prepped = {}
        for cen in centerings:
            for name, sh in shards.items():
                prepped[(name, cen)] = prepare(sh, seed, cen, early)
        for a_name, b_name in pairs:
            for cen in centerings:
                A, ma = prepped[(a_name, cen)]
                B, mb = prepped[(b_name, cen)]
                mask_a = ma["train"]
                mask_b = mb["test"] if (args.eval_scope == "test"
                                        or a_name == b_name) else mb["all"]
                row = {"seed": seed, "src": a_name, "dst": b_name,
                       "centering": cen, "early": early,
                       "eval_scope": ("test" if (args.eval_scope == "test"
                                                 or a_name == b_name) else "all")}
                try:
                    row.update(s1_transfer(A, B, mask_a, mask_b,
                                           args.n_perm, seed))
                    row.update(s2_transfer(A, B, mask_a, mask_b, args.k, seed))
                except Exception as exc:  # noqa: BLE001 — 셀 단위 fail-loud 기록
                    row["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)
                print(f"[pair] s{seed} {a_name}->{b_name} ({cen}): "
                      f"s1={row.get('s1_auroc')} z={row.get('s1_perm_z')} "
                      f"s2keep={row.get('s2_margin_keep')} "
                      f"err={row.get('error')}", flush=True)

    cols = ["seed", "src", "dst", "centering", "early", "eval_scope",
            "s1_auroc", "s1_perm_z", "s1_n_phase", "s1_n_phase_shared",
            "s1_cos_med", "s1_sign_agree", "s1_n_ep", "s1_n_fail",
            "length_auroc", "s2_margin_transfer", "s2_margin_self",
            "s2_margin_keep", "s2_centroid_cos_med", "s1_note", "error"]
    tsv = args.out / "share_transfer.tsv"
    with tsv.open("w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join("" if r.get(c) is None else str(r.get(c))
                              for c in cols) + "\n")
    (args.out / "share_transfer_meta.json").write_text(json.dumps({
        "note": TD_NOTE, "k": args.k, "early": early, "seeds": seeds,
        "centering": centerings, "eval_scope": args.eval_scope,
        "n_perm": args.n_perm, "n_rows": len(rows),
        "protocol": {
            "cap": "phase-gt dwell ceil(mu+sigma), ddof=0, 성공 dwell>0",
            "split": "failure_detector_sim.split_scenes 동일 (seed, crc32)",
            "s1": "A train-scene 방향 -> B 투영, episode equal-budget",
            "s2": "A PCA64w+KMeans -> B 할당 margin 유지율",
        }}, ensure_ascii=False, indent=2))
    print(f"[done] {len(rows)} rows -> {tsv}", flush=True)


if __name__ == "__main__":
    main()

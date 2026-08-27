#!/usr/bin/env python
"""cluster-phase 전환 **사전 진단 3종** (게이트).

phase 정의를 GT 이벤트 라벨에서 task 별 K=8 activation cluster 로 바꾸기 전에, 바꿔도
되는지를 숫자로 먼저 본다. 통과/탈락 판정을 자동으로 내리지 않고 (임계는 라운드마다
다르다) **위험 신호를 표로 드러내는** 것이 목적이다.

D1 cluster × outcome
    slug 별 8 cluster × {succ, fail} 의 record 수·episode 수. 경고 =
    "succ record < 50 또는 succ episode < 3" 인 cluster (그 cluster 에서는 성공 대조군을
    못 만든다 → setpoint/conceptor fit 불가). 그 cluster 들이 **fail record 의 몇 %**
    를 담는지도 같이 낸다 = 개입 순간을 얼마나 놓치는지의 근사.

D2 v4 OOD 점유율
    v4(지터 축) 케이스 rollout 을 같은 좌표계로 배정했을 때 cluster 점유가 한 곳에
    쏠리는지. 쏠리면 (엔트로피↓) 학습 분포 밖에서 판정기가 무너진 것이다. 대조로 같은
    slug 의 in-domain(v2 labels) 점유 엔트로피를 나란히 찍는다. 엔트로피는 log(k) 로
    정규화 (k=8 이면 log8).

D3 절제 후 길이 분포
    phase-dwell 절제(길이통제)를 GT phase 와 cluster 각각으로 시뮬한다. cluster 가
    GT 보다 실패 꼬리를 못 자르면 (= 실패 판의 종반 record 가 그대로 살아남으면)
    길이통제가 무력해진 것이고, succ/fail 대비가 다시 길이 아티팩트가 된다.
    절제 규약은 setM `scripts/fit/fit_setm.py: phase_dwell_caps` 와 **동일 식** —
    cap = 성공 episode dwell(>0) 의 ceil(μ + σ) (σ 는 ddof=0), episode 별로 각 code 의
    **시간순 앞 cap 개** 만 유지. 성공 dwell 이 없는 code 는 cap 이 없어 통째로 제외
    (fit_setm 도 그 phase 를 skip 한다).

입력
    --labels-dir   ae_cluster.py --dump-labels 출력 (labels_<slug>_k<K>.npz)
    --bundle       ae_cluster.py --export-bundle 산출 NPZ (D2 배정에 필요)
    --shard-dir    원본 shard (GT phase_code 대조용, D1/D3)
    --v4-manifest-dir + --grid-root   D2 용 (없으면 D2 skip)
    --out          결과 JSON

사용 예
    ~/anaconda3/bin/python scripts/analysis/grid_phase/cluster_prediag.py \
        --labels-dir ~/workspace/.../ae_raw --bundle ~/workspace/.../ae_bundle_k8.npz \
        --shard-dir  ~/workspace/.../segA \
        --v4-manifest-dir ~/workspace/.../manifests/v4 --grid-root ~/workspace/.../raw \
        --out ~/workspace/.../ae_raw/cluster_prediag.json

실행 환경: 승준 노드 `~/anaconda3/bin/python` (numpy + torch CPU). scipy 없음.
"""
from __future__ import annotations

import os

# BLAS/torch 스레드 cap — 공유 노드. numpy/torch import 전에 설정해야 효력이 있다.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import json
import math
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import torch

_p = Path(__file__).resolve()
REPO_ROOT = _p.parents[3] if len(_p.parents) > 3 else Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_REL = "scripts/analysis/grid_phase/cluster_prediag.py"
LABELS_RE = re.compile(r"^labels_(?P<slug>.+)_k(?P<k>\d+)\.npz$")

# D1 경고 임계 (성공 대조군을 만들 수 있는 최소 표본)
MIN_SUCC_REC = 50
MIN_SUCC_EP = 3
LATE_FRAC = 0.7            # D3 "후반 30%" 구간 시작 상대위치

EXPECT_LAYER = 12
EXPECT_K = 4               # denoise 축 — segA 특징(마지막 k=3)과 같은 좌표를 강제


# =============================================================================
# 번들 (mu / scalar_std / encoder / centers)
# =============================================================================

class Bundle:
    """ae_cluster.py --export-bundle NPZ → raw-1536 을 cluster id 로 보내는 판정기."""

    def __init__(self, path: Path):
        f = np.load(path, allow_pickle=False)
        try:
            self.mu = np.asarray(f["mu"], dtype=np.float32)
            self.scalar_std = float(np.asarray(f["scalar_std"]))
            self.k = int(np.asarray(f["k"]))
            self.latent = int(np.asarray(f["latent"]))
            self.slugs = [str(s) for s in np.asarray(f["slugs"])]
            self.arch = json.loads(str(np.asarray(f["arch"])))
            self.provenance = (json.loads(str(np.asarray(f["provenance"])))
                               if "provenance" in f.files else {})
            self.centers = {s: np.asarray(f[f"centers.{s}"], dtype=np.float32)
                            for s in self.slugs}
            state = {k[len("enc."):]: torch.from_numpy(
                        np.asarray(f[k], dtype=np.float32))
                     for k in f.files if k.startswith("enc.")}
        finally:
            f.close()
        if not state:
            raise SystemExit(f"{path.name}: encoder state (enc.*) 없음")

        from importlib.util import module_from_spec, spec_from_file_location
        ae_py = Path(__file__).resolve().parent / "ae_cluster.py"
        if not ae_py.is_file():
            raise SystemExit("ae_cluster.py 없음 (같은 디렉토리에 있어야 한다)")
        spec = spec_from_file_location("_grid_ae_cluster", ae_py)
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.enc = mod.Encoder(int(self.mu.shape[0]), self.latent,
                               int(self.arch.get("hidden", 256)))
        missing = self.enc.load_state_dict(state, strict=True)
        _ = missing
        self.enc.eval()

    def encode(self, X: np.ndarray, chunk: int = 4096) -> np.ndarray:
        Xs = (np.asarray(X, dtype=np.float32) - self.mu) / np.float32(self.scalar_std)
        out = []
        with torch.no_grad():
            for s in range(0, len(Xs), chunk):
                blk = torch.from_numpy(np.ascontiguousarray(Xs[s:s + chunk]))
                out.append(self.enc(blk).numpy().astype(np.float32))
        return (np.concatenate(out, 0) if out
                else np.zeros((0, self.latent), np.float32))

    def assign(self, X: np.ndarray, slug: str) -> np.ndarray:
        if slug not in self.centers:
            raise SystemExit(f"번들에 centers.{slug} 없음 (있는 slug: {self.slugs})")
        C = self.centers[slug]
        Z = self.encode(X)
        d2 = ((Z ** 2).sum(1)[:, None] - 2.0 * (Z @ C.T) + (C ** 2).sum(1)[None, :])
        return d2.argmin(1).astype(np.int16)


# =============================================================================
# 공통 로더
# =============================================================================

def discover_labels(labels_dir: Path, k: int | None) -> tuple[dict[str, Path], int]:
    found: dict[str, list[tuple[int, Path]]] = {}
    for p in sorted(labels_dir.glob("labels_*_k*.npz")):
        m = LABELS_RE.match(p.name)
        if not m:
            continue
        kk = int(m.group("k"))
        if k is not None and kk != k:
            continue
        found.setdefault(m.group("slug"), []).append((kk, p))
    if not found:
        raise SystemExit(f"labels NPZ 없음: {labels_dir} (k={k})")
    out, ks = {}, set()
    for slug, items in found.items():
        if len(items) > 1:
            raise SystemExit(f"{slug}: k 가 여럿 {[i[0] for i in items]} — --k 지정 필요")
        out[slug] = items[0][1]
        ks.add(items[0][0])
    if len(ks) > 1:
        raise SystemExit(f"labels 의 k 가 섞여 있다 {sorted(ks)} — --k 지정 필요")
    return out, ks.pop()


def load_labels(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as f:
        for need in ("cluster", "ep_id", "rec_idx", "succ"):
            if need not in f.files:
                raise SystemExit(f"{path.name}: `{need}` 없음")
        return {"cluster": np.asarray(f["cluster"]).astype(np.int64),
                "ep_id": np.asarray(f["ep_id"]).astype(np.int64),
                "rec_idx": np.asarray(f["rec_idx"]).astype(np.int64),
                "succ": np.asarray(f["succ"]).astype(np.int64)}


def load_shard_light(path: Path) -> dict:
    """shard 에서 X 를 빼고 소용량 배열만. NpzFile 은 접근한 멤버만 압축 해제한다."""
    with np.load(path, allow_pickle=False) as f:
        need = ("ep_id", "rec_idx", "succ", "phase_code")
        for n in need:
            if n not in f.files:
                raise SystemExit(f"{path.name}: `{n}` 없음")
        out = {n: np.asarray(f[n]).astype(np.int64) for n in need}
        meta_raw = f["meta_json"] if "meta_json" in f.files else None
    out["meta"] = json.loads(str(meta_raw)) if meta_raw is not None else {}
    return out


def check_alignment(slug: str, lab: dict, sh: dict) -> None:
    if len(lab["cluster"]) != len(sh["phase_code"]):
        raise SystemExit(f"{slug}: labels {len(lab['cluster'])} 행 != "
                         f"shard {len(sh['phase_code'])} 행")
    for key in ("ep_id", "rec_idx", "succ"):
        if not np.array_equal(lab[key], sh[key]):
            i = int(np.flatnonzero(lab[key] != sh[key])[0])
            raise SystemExit(f"{slug}: labels 와 shard 의 {key} 불일치 "
                             f"(행 {i}: {lab[key][i]} vs {sh[key][i]})")


# =============================================================================
# D1 — cluster × outcome
# =============================================================================

def d1_cluster_outcome(slug: str, cluster: np.ndarray, succ: np.ndarray,
                       ep_id: np.ndarray, k: int) -> dict:
    rows = []
    fail_total = int((succ == 0).sum())
    flagged, flagged_fail = [], 0
    for c in range(k):
        m = cluster == c
        m_s, m_f = m & (succ == 1), m & (succ == 0)
        n_rec_s, n_rec_f = int(m_s.sum()), int(m_f.sum())
        n_ep_s = int(len(np.unique(ep_id[m_s])))
        n_ep_f = int(len(np.unique(ep_id[m_f])))
        low = (n_rec_s < MIN_SUCC_REC) or (n_ep_s < MIN_SUCC_EP)
        if low:
            flagged.append(c)
            flagged_fail += n_rec_f
        rows.append({"cluster": c, "n_rec_succ": n_rec_s, "n_rec_fail": n_rec_f,
                     "n_ep_succ": n_ep_s, "n_ep_fail": n_ep_f,
                     "low_succ_support": bool(low)})
    return {
        "slug": slug, "k": k,
        "n_rec": int(len(cluster)), "n_rec_fail": fail_total,
        "n_rec_succ": int((succ == 1).sum()),
        "per_cluster": rows,
        "flagged_clusters": flagged,
        "flagged_fail_record_frac": (float(flagged_fail / fail_total)
                                     if fail_total else None),
        "criterion": f"succ record < {MIN_SUCC_REC} or succ episode < {MIN_SUCC_EP}",
    }


# =============================================================================
# D2 — v4 OOD 점유율
# =============================================================================

def occupancy(cluster: np.ndarray, k: int) -> dict:
    cnt = np.bincount(np.asarray(cluster).astype(np.int64), minlength=k)[:k]
    n = int(cnt.sum())
    p = cnt / max(n, 1)
    nz = p > 0
    h = max(float(-(p[nz] * np.log(p[nz])).sum()), 0.0)   # -0.0 방지
    return {"counts": [int(x) for x in cnt], "n": n,
            "frac": [float(x) for x in p],
            "entropy_nats": h,
            "entropy_norm": float(h / math.log(k)) if k > 1 else 0.0,
            "n_occupied": int(nz.sum()),
            "top_cluster": int(cnt.argmax()) if n else None,
            "top_frac": float(p.max()) if n else None}


def _as_f32(x) -> np.ndarray:
    """torch fp16 텐서 / ndarray / list → float32 (extract_grid_matrix._as_f32 이식)."""
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False)
    if hasattr(x, "detach"):
        x = x.detach().cpu()
        if str(x.dtype).endswith("float16"):
            x = x.float()
        return x.numpy().astype(np.float32, copy=False)
    return np.asarray(x, dtype=np.float32)


def pkl_features(pkl: Path) -> tuple[np.ndarray, int]:
    """rollout.pkl → (feat [n_rec, D] f32, episode_success).

    좌표는 shard 특징과 동일: capture_layers 에서 layer 12 인덱스 역산 · 마지막 denoise ·
    전 토큰 평균. 하드코딩 인덱스 금지 (extract_grid_matrix 규약).
    """
    with open(pkl, "rb") as f:
        d = pickle.load(f)
    layers = list(d.get("capture_layers") or [])
    if EXPECT_LAYER not in layers:
        raise SystemExit(f"{pkl.name}: capture_layers={layers} 에 layer "
                         f"{EXPECT_LAYER} 없음")
    li = layers.index(EXPECT_LAYER)
    hs = d.get("hidden_states")
    if not hs:
        raise SystemExit(f"{pkl.name}: hidden_states 없음/비어 있음")
    feats = np.zeros((len(hs), 0), np.float32)
    for i in range(len(hs)):
        a = _as_f32(hs[i])
        hs[i] = None
        if a.ndim != 4:
            raise SystemExit(f"{pkl.name}: record{i} shape {a.shape} — [L,K,T,D] 기대")
        if a.shape[0] != len(layers):
            raise SystemExit(f"{pkl.name}: record{i} L={a.shape[0]} != "
                             f"capture_layers {len(layers)}")
        if a.shape[1] != EXPECT_K:
            raise SystemExit(f"{pkl.name}: record{i} K={a.shape[1]} != {EXPECT_K} "
                             "— shard 특징과 denoise 좌표가 달라진다")
        v = a[li, -1].mean(axis=0)          # 마지막 denoise · 전 토큰 평균 → [D]
        if feats.shape[1] == 0:
            feats = np.zeros((len(hs), v.shape[0]), np.float32)
        feats[i] = v
        del a
    succ = d.get("episode_success")
    return feats, (int(succ) if succ is not None else -1)


def resolve_slug(case: str, slugs: list[str]) -> str:
    """케이스 파일명(<slug>_s<N>[...]) → bundle slug. 최장 접두 일치, 없으면 fail-loud."""
    cands = [s for s in slugs if case == s or case.startswith(s + "_")]
    if not cands:
        raise SystemExit(f"케이스 {case!r} 의 slug 를 번들에서 찾지 못했다 "
                         f"(slugs: {slugs})")
    return max(cands, key=len)


def read_case_tsv(path: Path) -> list[str]:
    """케이스 tsv → pkl 경로 목록 (첫 열). 주석(#)/헤더 제외."""
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        first = ln.split("\t")[0].strip()
        if first in ("pkl_path", "path", "rel_path"):
            continue
        out.append(first)
    if not out:
        raise SystemExit(f"{path.name}: pkl 행이 없다")
    return out


def d2_v4_occupancy(manifest_dir: Path, grid_root: Path, bundle: Bundle,
                    indomain: dict[str, dict], k: int, limit: int | None) -> dict:
    cases = sorted(manifest_dir.glob("*.tsv"))
    if not cases:
        raise SystemExit(f"케이스 tsv 없음: {manifest_dir}")
    out = {}
    for cpath in cases:
        case = cpath.stem
        slug = resolve_slug(case, bundle.slugs)
        rels = read_case_tsv(cpath)
        if limit:
            rels = rels[:limit]
        feats, n_ep, n_bad = [], 0, []
        for rel in rels:
            p = Path(rel)
            pkl = p if p.is_absolute() else grid_root / rel
            if not pkl.is_file():
                n_bad.append(rel)
                continue
            fx, _succ = pkl_features(pkl)
            feats.append(fx)
            n_ep += 1
        if not feats:
            raise SystemExit(f"{case}: 읽을 수 있는 pkl 이 없다 (누락 {len(n_bad)})")
        X = np.concatenate(feats, 0)
        del feats
        if X.shape[1] != bundle.mu.shape[0]:
            raise SystemExit(f"{case}: feature dim {X.shape[1]} != "
                             f"번들 {bundle.mu.shape[0]}")
        cl = bundle.assign(X, slug)
        occ = occupancy(cl, k)
        occ.update({"case": case, "slug": slug, "n_episode": n_ep,
                    "n_pkl_missing": len(n_bad),
                    "missing_examples": n_bad[:3]})
        ind = indomain.get(slug)
        if ind is not None:
            occ["indomain_entropy_norm"] = ind["entropy_norm"]
            occ["indomain_top_frac"] = ind["top_frac"]
            occ["delta_entropy_norm"] = occ["entropy_norm"] - ind["entropy_norm"]
        out[case] = occ
        print(f"[D2] {case:<30} slug={slug:<24} n_ep={n_ep:>4} "
              f"H={occ['entropy_norm']:.3f} (in-domain "
              f"{occ.get('indomain_entropy_norm', float('nan')):.3f}) "
              f"top c{occ['top_cluster']} {occ['top_frac']:.2f}", flush=True)
    return out


# =============================================================================
# D3 — 절제 후 길이 분포
# =============================================================================

def episode_index(ep_id: np.ndarray, rec_idx: np.ndarray) -> list[np.ndarray]:
    """episode 별 record 인덱스 (rec_idx 오름차순). 순서를 가정하지 않는다."""
    order = np.lexsort((rec_idx, ep_id))
    ep_sorted = ep_id[order]
    bounds = np.concatenate([[0], np.flatnonzero(np.diff(ep_sorted)) + 1,
                             [len(ep_sorted)]])
    return [order[s:e] for s, e in zip(bounds[:-1], bounds[1:])]


def dwell_caps(eps: list[np.ndarray], code: np.ndarray, succ: np.ndarray) -> dict:
    """cap[c] = 성공 episode 의 code c 체류 길이(>0) 의 ceil(μ + σ), σ 는 ddof=0.

    출처: scripts/fit/fit_setm.py: phase_dwell_caps (성공 dwell 없는 code 는 미포함 →
    호출부에서 skip). 여기서는 shard record 축 위에서 같은 식을 다시 쓴다.
    """
    per: dict[int, list[int]] = {}
    for idx in eps:
        if int(succ[idx[0]]) != 1:
            continue
        c, n = np.unique(code[idx], return_counts=True)
        for cc, nn in zip(c.tolist(), n.tolist()):
            per.setdefault(int(cc), []).append(int(nn))
    caps = {}
    for cc, dw in per.items():
        dw = [d for d in dw if d > 0]
        if dw:
            caps[cc] = int(math.ceil(float(np.mean(dw)) + float(np.std(dw))))
    return caps


def truncate_sim(eps: list[np.ndarray], code: np.ndarray, succ: np.ndarray,
                 caps: dict) -> dict:
    """episode 별로 각 code 의 시간순 앞 cap 개만 유지했을 때의 생존 통계.

    cap 이 없는 code(성공 dwell 0) 의 record 는 제외한다 — fit_setm 이 그 phase 를
    skip 하는 것과 같은 취급.
    """
    keep_rel, fail_keep, succ_keep = [], [], []
    late_tot = late_keep = 0
    n_drop_nocap = 0
    for idx in eps:
        T = len(idx)
        is_fail = int(succ[idx[0]]) != 1
        seen: dict[int, int] = {}
        kept = 0
        for pos, i in enumerate(idx):
            c = int(code[i])
            rel = pos / max(T - 1, 1)
            cap = caps.get(c)
            if cap is None:
                n_drop_nocap += 1
                survive = False
            else:
                n = seen.get(c, 0)
                survive = n < cap
                seen[c] = n + 1
            if survive:
                kept += 1
            if is_fail:
                if rel >= LATE_FRAC:
                    late_tot += 1
                    late_keep += int(survive)
                if survive:
                    keep_rel.append(rel)
        (fail_keep if is_fail else succ_keep).append(kept)
    fk = float(np.mean(fail_keep)) if fail_keep else float("nan")
    sk = float(np.mean(succ_keep)) if succ_keep else float("nan")
    rel = np.asarray(keep_rel, dtype=np.float64)
    return {
        "caps": {str(c): int(v) for c, v in sorted(caps.items())},
        "n_codes_with_cap": len(caps),
        "fail_survivor_relpos_median": float(np.median(rel)) if len(rel) else None,
        "fail_survivor_relpos_p90": (float(np.quantile(rel, 0.9)) if len(rel)
                                     else None),
        "fail_mean_records_kept": fk,
        "succ_mean_records_kept": sk,
        "fail_over_succ_kept_ratio": (float(fk / sk) if sk and not math.isnan(sk)
                                      and sk > 0 else None),
        "fail_late30_survival": (float(late_keep / late_tot) if late_tot else None),
        "fail_late30_records": int(late_tot),
        "n_records_dropped_no_cap": int(n_drop_nocap),
        "n_fail_episodes": len(fail_keep), "n_succ_episodes": len(succ_keep),
    }


def d3_truncation(slug: str, lab: dict, sh: dict) -> dict:
    eps = episode_index(lab["ep_id"], lab["rec_idx"])
    succ = lab["succ"]
    out = {"slug": slug, "n_episodes": len(eps)}
    for tag, code in (("gt", sh["phase_code"]), ("cluster", lab["cluster"])):
        caps = dwell_caps(eps, code, succ)
        out[tag] = truncate_sim(eps, code, succ, caps)
    return out


# =============================================================================
# 출력
# =============================================================================

def _f(v, nd=3):
    return "  n/a" if v is None or (isinstance(v, float) and math.isnan(v)) \
        else f"{v:.{nd}f}"


def print_tables(res: dict) -> None:
    print("\n== D1 cluster × outcome ==", flush=True)
    print(f"{'slug':<26} {'flag':<12} {'fail%':>7}  low-support cluster (succ_rec/succ_ep)")
    for slug, d in sorted(res["D1"].items()):
        low = [r for r in d["per_cluster"] if r["low_succ_support"]]
        detail = ", ".join(f"c{r['cluster']}({r['n_rec_succ']}/{r['n_ep_succ']})"
                           for r in low) or "-"
        print(f"{slug:<26} {len(low):>2}/{d['k']:<9} "
              f"{_f(d['flagged_fail_record_frac'])}  {detail}")

    if res.get("D2"):
        print("\n== D2 v4 OOD 점유율 (엔트로피 = log k 정규화) ==", flush=True)
        print(f"{'case':<30} {'slug':<24} {'n_ep':>5} {'H_v4':>7} {'H_in':>7} "
              f"{'ΔH':>7} {'top':>5} {'top%':>7}")
        for case, d in sorted(res["D2"].items()):
            print(f"{case:<30} {d['slug']:<24} {d['n_episode']:>5} "
                  f"{_f(d['entropy_norm'])} {_f(d.get('indomain_entropy_norm'))} "
                  f"{_f(d.get('delta_entropy_norm'))} "
                  f"{('c' + str(d['top_cluster'])):>5} {_f(d['top_frac'])}")

    print("\n== D3 절제 후 길이 분포 (GT vs cluster) ==", flush=True)
    print(f"{'slug':<26} {'def':<8} {'relpos_med':>10} {'relpos_p90':>10} "
          f"{'f/s_kept':>9} {'late30':>8} {'caps':>5}")
    for slug, d in sorted(res["D3"].items()):
        for tag in ("gt", "cluster"):
            t = d[tag]
            print(f"{slug:<26} {tag:<8} {_f(t['fail_survivor_relpos_median']):>10} "
                  f"{_f(t['fail_survivor_relpos_p90']):>10} "
                  f"{_f(t['fail_over_succ_kept_ratio']):>9} "
                  f"{_f(t['fail_late30_survival']):>8} "
                  f"{t['n_codes_with_cap']:>5}")


# =============================================================================
# main
# =============================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labels-dir", required=True, type=Path)
    ap.add_argument("--bundle", required=True, type=Path,
                    help="ae_cluster.py --export-bundle 산출 NPZ")
    ap.add_argument("--shard-dir", required=True, type=Path,
                    help="원본 shard (GT phase_code 대조용)")
    ap.add_argument("--out", required=True, type=Path, help="결과 JSON")
    ap.add_argument("--v4-manifest-dir", type=Path, default=None,
                    help="케이스별 tsv 디렉토리 (첫 열 = pkl 경로). 없으면 D2 skip")
    ap.add_argument("--grid-root", type=Path, default=None,
                    help="tsv 의 pkl 상대경로 기준 루트 (D2 필수)")
    ap.add_argument("--v4-limit", type=int, default=None,
                    help="케이스당 pkl 상한 (스모크용)")
    ap.add_argument("--k", type=int, default=None, help="labels 의 k (자동 유도)")
    ap.add_argument("--shards", default=None, help="쉼표구분 slug 부분집합")
    args = ap.parse_args(argv)

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

    lab_map, k = discover_labels(args.labels_dir.expanduser(), args.k)
    if args.shards:
        only = set(args.shards.split(","))
        missing = only - set(lab_map)
        if missing:
            raise SystemExit(f"labels 없음: {sorted(missing)}")
        lab_map = {s: p for s, p in lab_map.items() if s in only}

    bundle = Bundle(args.bundle.expanduser())
    if bundle.k != k:
        raise SystemExit(f"번들 k={bundle.k} != labels k={k}")

    res = {"D1": {}, "D2": {}, "D3": {},
           "meta": {"script": SCRIPT_REL, "k": k,
                    "labels_dir_basename": args.labels_dir.name,
                    "shard_dir_basename": args.shard_dir.name,
                    "bundle_basename": args.bundle.name,
                    "bundle_provenance": bundle.provenance,
                    "criteria": {"min_succ_record": MIN_SUCC_REC,
                                 "min_succ_episode": MIN_SUCC_EP,
                                 "late_window_start_relpos": LATE_FRAC},
                    "truncation_rule": "cap = ceil(mean+std, ddof=0) of succ-episode "
                                       "dwell(>0); keep first cap per code per episode "
                                       "(scripts/fit/fit_setm.py: phase_dwell_caps)",
                    "numpy": np.__version__, "torch": torch.__version__}}

    indomain: dict[str, dict] = {}
    for slug, lpath in sorted(lab_map.items()):
        lab = load_labels(lpath)
        shard = args.shard_dir.expanduser() / f"{slug}.npz"
        if not shard.is_file():
            raise SystemExit(f"shard 없음: {shard.name}")
        sh = load_shard_light(shard)
        check_alignment(slug, lab, sh)

        res["D1"][slug] = d1_cluster_outcome(slug, lab["cluster"], lab["succ"],
                                             lab["ep_id"], k)
        indomain[slug] = occupancy(lab["cluster"], k)
        res["D1"][slug]["indomain_occupancy"] = indomain[slug]
        res["D3"][slug] = d3_truncation(slug, lab, sh)
        print(f"[load] {slug:<26} n_rec={len(lab['cluster']):>7} "
              f"flagged={len(res['D1'][slug]['flagged_clusters'])}", flush=True)

    if args.v4_manifest_dir is not None:
        if args.grid_root is None:
            raise SystemExit("--v4-manifest-dir 를 쓰면 --grid-root 도 필요하다")
        res["D2"] = d2_v4_occupancy(args.v4_manifest_dir.expanduser(),
                                    args.grid_root.expanduser(), bundle,
                                    indomain, k, args.v4_limit)
    else:
        res["meta"]["D2_skipped"] = "--v4-manifest-dir 미지정"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[write] {args.out.name}", flush=True)
    print_tables(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())

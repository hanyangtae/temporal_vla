#!/usr/bin/env python
"""논문 보강 실험: scene 잔차화 재측정 + cluster×GT phase contingency.

국내 논문(1~2쪽)의 주장 두 가지를 방어·직접검증하기 위한 보조 실행기다.

  ① scene 잔차화 (mode=resid)
     기준 run 의 `mi_scene_bits` 는 task 에 따라 0.2~1.0 bits 로, 클러스터가 phase 가
     아니라 **scene(배치·물체 위치)** 을 읽고 있을 여지가 있다. task 안에서 scene 별
     평균을 뺀 뒤 (= scene 주효과 제거) 같은 파이프라인을 다시 태워 phase-MI/margin 이
     살아남는지 본다. 살아남으면 "phase 정보는 scene 오염의 부산물이 아니다".

  ② cluster×GT phase contingency (mode=contingency)
     "activation cluster 는 GT phase 의 하위 분할이고 각 cluster 는 phase-순수하다" 는
     주장을 요약지표(purity 하나)가 아니라 분할표 자체로 보인다. cluster 별 최빈 phase
     점유율, phase 별 cluster 개수, off-phase 출현율 분포를 낸다.

  ③ 재현 게이트 (mode=gate)
     위 둘을 하기 전에, 이 스크립트가 기준 run 과 **같은 수를 만드는지** 먼저 확인한다.
     `paper_ref/align_pertask.json` 의 k8 per-task 값과 mi_phase_bits / margin_bits /
     purity_phase 를 대조하고 어긋나면 실패로 끝낸다 (exit 1).

파이프라인·지표는 새로 구현하지 않고 `intrinsic_phase.py` 를 그대로 import 해서 쓴다
(shard 로딩 · PCA-64 whitened · KMeans 래퍼 · align_metrics · mi_bits/purity). 이 파일이
추가하는 것은 (a) scene 잔차화 한 줄, (b) 분할표 집계, (c) 기준값 대조뿐이다.

특징 규격은 기준 run 과 동일: DiT layer 12 · denoise 3 · segment "all"(49토큰 평균),
PCA-64 whitened → KMeans(per-task k=8 / global k=24), n_init 5 · max_iter 300 · seed 0,
boundary_tol 1 · n_perm 300.

사용 예
    python scripts/analysis/grid_phase/paper_supplements.py \
        --shard-dir ~/datasets/temporal_vla_store/groot/n15/analysis/grid_phase/segA \
        --mode all \
        --kmeans-src ~/workspace/temporal_vla/task_classification/phase/clustering/gpu.py

출력 (--out-dir, 기본 outputs/analysis/grid_phase/paper_supp)
    gate_check.json                 기준값 대조 결과
    resid_pertask_k8.json           잔차화 후 per-task align 지표 (+원본)
    resid_compare.tsv               원본 vs 잔차화 요약표
    contingency_pertask_k8.json     per-task 분할표
    contingency_global_k24.json     global 분할표

실행 환경: 승준 노드 `~/anaconda3/bin/python` (numpy 1.26 + torch CPU).
scipy / sklearn 없음 — numpy 만 쓴다. OMP 스레드 cap 은 호출측(remote_compute.sh)이 준다.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_REL = "scripts/analysis/grid_phase/paper_supplements.py"

# 기준 run 과 동일 고정 파라미터 (align_pertask.json meta.params 와 일치해야 한다)
PCA_DIM = 64
WHITEN = True
N_INIT = 5
MAX_ITER = 300
SEED = 0
BOUNDARY_TOL = 1
N_PERM = 300

GATE_KEYS = ("mi_phase_bits", "margin_bits", "purity_phase")


# =============================================================================
# intrinsic_phase.py 재사용 (파이프라인·지표의 정본)
# =============================================================================

def load_intrinsic():
    """같은 디렉토리의 intrinsic_phase.py 를 파일경로로 로드한다.

    grid_phase 는 패키지가 아니라서 (``__init__.py`` 없음) 일반 import 가 안 된다.
    spec_from_file_location 이면 sys.path 조작 없이 형제 모듈을 그대로 쓸 수 있다.
    """
    f = Path(__file__).resolve().parent / "intrinsic_phase.py"
    if not f.is_file():
        raise SystemExit(f"intrinsic_phase.py 없음: {f.name} (같은 디렉토리 필요)")
    spec = importlib.util.spec_from_file_location("_grid_intrinsic_phase", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


IP = load_intrinsic()   # load_shard / discover_shards / pca_* / KMeansRunner / align_metrics


def tc_root_from_src(src: str | Path | None) -> Path | None:
    """--kmeans-src → KMeansRunner 가 기대하는 task_classification 루트.

    gpu.py 파일경로(기준 run 이 쓴 형태)와 루트 디렉토리 둘 다 받는다.
    """
    if not src:
        return None
    p = Path(src).expanduser()
    if p.name.endswith(".py"):
        # <root>/phase/clustering/gpu.py → <root>
        return p.parent.parent.parent
    return p


# =============================================================================
# shard 유틸
# =============================================================================

def phase_names(shard: dict) -> dict[int, str]:
    """shard meta 의 phase_codebook {name: code} → {code: name}."""
    cb = (shard.get("meta") or {}).get("phase_codebook") or {}
    out: dict[int, str] = {}
    for name, code in cb.items():
        try:
            out[int(code)] = str(name)
        except (TypeError, ValueError):
            continue
    return out


def residualize_by_scene(feat: np.ndarray, scene: np.ndarray) -> np.ndarray:
    """task 안에서 scene 별 평균을 뺀다 (scene 주효과 제거).

    전 record 를 대상으로 하며 (fit/eval 분리 없음), 이후 PCA 는 잔차에 **재적합** 한다.
    """
    if scene is None:
        raise SystemExit("scene 열이 없어 잔차화 불가")
    out = np.asarray(feat, dtype=np.float32).copy()
    sc = np.asarray(scene)
    for s in np.unique(sc):
        m = sc == s
        out[m] -= out[m].mean(0, dtype=np.float64).astype(np.float32)
    return out


def cluster_one(feat, scene, K, runner):
    """PCA-64(w) → KMeans(K) → 라벨. 전 행에서 fit (기준 run 의 fit_scenes=null 과 동일)."""
    labels, _stats, _evr, _cent, _nfit = IP.run_group(
        feat, scene, None, PCA_DIM, K, runner, WHITEN)
    return np.asarray(labels)


def metrics_for(shard, labels):
    return IP.align_metrics(labels, shard["phase_code"], shard["scene"],
                            shard["ep_id"], shard["rec_idx"],
                            BOUNDARY_TOL, N_PERM, SEED)


def fmt_line(tag, name, m):
    def g(k):
        return m.get(k, float("nan"))
    return (f"[{tag}] {name:<26} MI={g('mi_phase_bits'):.3f} "
            f"margin={g('margin_bits'):+.3f} purity={g('purity_phase'):.3f} "
            f"mi_scene={g('mi_scene_bits'):.3f} F1={g('boundary_f1'):.3f}")


# =============================================================================
# 분할표
# =============================================================================

def contingency_stats(labels, phase, code2name) -> dict:
    """cluster × GT phase 분할표와 파생 통계.

    phase < 0 (미라벨) 행은 제외한다. 반환 dict 는 전부 JSON 직렬화 가능 (그림은 로컬).
    """
    lab = np.asarray(labels)
    ph = np.asarray(phase)
    ok = ph >= 0
    lab, ph = lab[ok], ph[ok]
    n = int(len(lab))
    if n == 0:
        return {"n_rec": 0}

    clusters = np.unique(lab)
    codes = np.unique(ph)
    ci = {int(c): i for i, c in enumerate(clusters)}
    pi = {int(p): i for i, p in enumerate(codes)}
    cm = np.zeros((len(clusters), len(codes)), np.int64)
    np.add.at(cm, (np.array([ci[int(c)] for c in lab]),
                   np.array([pi[int(p)] for p in ph])), 1)

    names = [code2name.get(int(c), str(int(c))) for c in codes]
    row_n = cm.sum(1)
    occ = cm / np.maximum(row_n, 1)[:, None]              # row-normalized 점유율
    top_j = cm.argmax(1)
    top_share = occ[np.arange(len(clusters)), top_j]
    off_rate = 1.0 - top_share

    per_cluster = []
    for i, c in enumerate(clusters):
        per_cluster.append({
            "cluster": int(c),
            "n_rec": int(row_n[i]),
            "top_phase": names[int(top_j[i])],
            "top_phase_code": int(codes[int(top_j[i])]),
            "top_share": float(top_share[i]),
            "off_phase_rate": float(off_rate[i]),
            "occupancy": [float(v) for v in occ[i]],
        })

    per_phase = []
    col_n = cm.sum(0)
    for j, p in enumerate(codes):
        owned = [int(clusters[i]) for i in range(len(clusters)) if top_j[i] == j]
        per_phase.append({
            "phase": names[j],
            "phase_code": int(p),
            "n_rec": int(col_n[j]),
            "n_clusters_argmax": len(owned),
            "clusters_argmax": owned,
            # 이 phase 행 중 '자기 phase 를 최빈으로 갖는 cluster' 에 들어간 비율
            "covered_by_own_clusters": float(
                cm[[ci[c] for c in owned], j].sum() / max(int(col_n[j]), 1)) if owned else 0.0,
        })

    q1, med, q3 = (float(x) for x in np.quantile(off_rate, [0.25, 0.5, 0.75]))
    return {
        "n_rec": n,
        "n_clusters": int(len(clusters)),
        "n_phases": int(len(codes)),
        "cluster_ids": [int(c) for c in clusters],
        "phase_codes": [int(p) for p in codes],
        "phase_names": names,
        "counts": cm.tolist(),                     # [n_cluster, n_phase] 정수
        "occupancy_rownorm": [[float(v) for v in r] for r in occ],
        "per_cluster": per_cluster,
        "per_phase": per_phase,
        "off_phase_rate": {
            "median": med, "q1": q1, "q3": q3,
            "min": float(off_rate.min()), "max": float(off_rate.max()),
            "mean": float(off_rate.mean()),
            # record 가중 = 1 − purity (요약지표와의 정합 확인용)
            "record_weighted": float(1.0 - cm.max(1).sum() / n),
        },
        "purity_phase": float(cm.max(1).sum() / n),
        "clusters_per_phase_mean": float(len(clusters) / max(len(codes), 1)),
    }


def unified_phase(shards):
    """shard 마다 다를 수 있는 phase_codebook 을 **이름 기준**으로 통일한다.

    → (per-shard 통일코드 배열 리스트, {통일코드: 이름}). global scope 에서 필수.
    """
    name2uni: dict[str, int] = {}
    out = []
    for s in shards:
        c2n = phase_names(s)
        ph = np.asarray(s["phase_code"])
        uni = np.full(len(ph), -1, np.int32)
        for code in np.unique(ph):
            code = int(code)
            if code < 0:
                continue
            nm = c2n.get(code, f"code{code}")
            uni[ph == code] = name2uni.setdefault(nm, len(name2uni))
        out.append(uni)
    return out, {v: k for k, v in name2uni.items()}


# =============================================================================
# 모드
# =============================================================================

def load_ref(ref_dir: Path, fname: str, k: int):
    path = ref_dir / fname
    if not path.is_file():
        raise SystemExit(f"기준 JSON 없음: {fname} (--ref-dir 확인)")
    ref = json.loads(path.read_text(encoding="utf-8"))
    by_k = ref.get("by_k", {})
    if str(k) not in by_k:
        raise SystemExit(f"{fname}: by_k 에 k={k} 없음 (있는 k: {sorted(by_k)})")
    return ref, by_k[str(k)]


def mode_gate(shards, raw, ref_dir, k, tol):
    """기준 run 과 동일한 수가 나오는지 대조. 어긋나면 실패."""
    _ref, ref_k = load_ref(ref_dir, "align_pertask.json", k)
    ref_per = ref_k.get("per_shard", {})
    rows, ok_all = {}, True
    for s in shards:
        name = s["name"]
        new = raw[name]["metrics"]
        r = ref_per.get(name)
        if r is None:
            rows[name] = {"status": "missing_ref"}
            ok_all = False
            print(f"[gate] {name:<26} 기준값 없음", flush=True)
            continue
        entry, ident = {"status": "identical"}, True
        for key in GATE_KEYS:
            a, b = r.get(key), new.get(key)
            d = float("nan") if (a is None or b is None) else float(b - a)
            entry[key] = {"ref": a, "new": b, "diff": d}
            if a is None or b is None or not (abs(d) <= tol):
                ident = False
        if not ident:
            entry["status"] = "mismatch"
            ok_all = False
        rows[name] = entry
        worst = max((abs(entry[key]["diff"]) for key in GATE_KEYS
                     if entry[key]["diff"] == entry[key]["diff"]), default=float("nan"))
        print(f"[gate] {name:<26} {entry['status']:<10} max|diff|={worst:.3e}", flush=True)
    return {"k": k, "tol": tol, "passed": ok_all,
            "ref_file": "align_pertask.json",
            "ref_kmeans_impl": _ref.get("kmeans_impl"),
            "keys": list(GATE_KEYS), "per_task": rows}, ok_all


def mode_resid(shards, raw, runner, k, out_dir):
    """① scene 잔차화 후 재측정."""
    per, tsv = {}, ["\t".join([
        "task", "raw_mi", "resid_mi", "raw_margin", "resid_margin",
        "raw_mi_scene", "resid_mi_scene", "raw_purity", "resid_purity"])]
    for s in shards:
        name = s["name"]
        res_feat = residualize_by_scene(s["feat"], s["scene"])
        lab = cluster_one(res_feat, s["scene"], k, runner)
        del res_feat
        m = metrics_for(s, lab)
        r = raw[name]["metrics"]
        per[name] = {"raw": r, "resid": m,
                     "delta": {key: (float(m[key] - r[key])
                                     if key in m and key in r else None)
                               for key in ("mi_phase_bits", "margin_bits",
                                           "purity_phase", "mi_scene_bits",
                                           "nmi_arith", "boundary_f1")}}
        tsv.append("\t".join([name] + [
            f"{v:.6f}" for v in (
                r.get("mi_phase_bits", float("nan")), m.get("mi_phase_bits", float("nan")),
                r.get("margin_bits", float("nan")), m.get("margin_bits", float("nan")),
                r.get("mi_scene_bits", float("nan")), m.get("mi_scene_bits", float("nan")),
                r.get("purity_phase", float("nan")), m.get("purity_phase", float("nan")))]))
        print(fmt_line("resid", name, m), flush=True)
    (out_dir / "resid_compare.tsv").write_text("\n".join(tsv) + "\n", encoding="utf-8")
    return {"k": k, "residualization": "per-task scene demean (all records)",
            "per_task": per}


def mode_contingency(shards, raw, k):
    """② per-task 분할표 (원본 feature 라벨 재사용)."""
    per = {}
    for s in shards:
        st = contingency_stats(raw[s["name"]]["labels"], s["phase_code"], phase_names(s))
        per[s["name"]] = st
        print(f"[cont] {s['name']:<26} purity={st.get('purity_phase', float('nan')):.3f} "
              f"off_med={st.get('off_phase_rate', {}).get('median', float('nan')):.3f} "
              f"k/phase={st.get('clusters_per_phase_mean', float('nan')):.2f}", flush=True)
    return {"scope": "per-task", "k": k, "per_task": per}


def mode_contingency_global(shards, runner, k):
    """global k24 분할표. phase 는 이름 기준으로 통일한다."""
    feat = np.concatenate([s["feat"] for s in shards], 0)
    scene = (np.concatenate([s["scene"] for s in shards], 0)
             if all(s["scene"] is not None for s in shards) else None)
    lab = cluster_one(feat, scene, k, runner)
    del feat
    uni_list, uni_names = unified_phase(shards)
    uni = np.concatenate(uni_list)
    st = contingency_stats(lab, uni, uni_names)

    # cluster × task 도 함께 (cluster 가 task 전용인지 = 공유되는지)
    tasks = [s["name"] for s in shards]
    tidx = np.concatenate([np.full(len(s["feat"]), i, np.int32)
                           for i, s in enumerate(shards)])
    clusters = np.unique(lab)
    ci = {int(c): i for i, c in enumerate(clusters)}
    cbt = np.zeros((len(clusters), len(tasks)), np.int64)
    np.add.at(cbt, (np.array([ci[int(c)] for c in lab]), tidx), 1)
    st["cluster_by_task"] = {"tasks": tasks, "cluster_ids": [int(c) for c in clusters],
                            "counts": cbt.tolist()}
    print(f"[cont] {'__global__':<26} purity={st.get('purity_phase', float('nan')):.3f} "
          f"off_med={st.get('off_phase_rate', {}).get('median', float('nan')):.3f}",
          flush=True)
    return {"scope": "global", "k": k, "global": st}


# =============================================================================
# main
# =============================================================================

def build_meta(args, runner, shards, extra=None):
    """docs/04 규약: 산출물 안에 절대경로 금지 (kmeans source 는 basename 만)."""
    src = runner.source
    meta = {
        "script": SCRIPT_REL,
        "reuses": "scripts/analysis/grid_phase/intrinsic_phase.py",
        "feature_spec": {"layer": 12, "denoise": 3, "segment": "all(49-token mean)",
                         "layer_axis_index": IP.LAYER_IDX,
                         "denoise_axis_index": IP.DENOISE_IDX,
                         "segment_axis_index": IP.SEGMENT_IDX},
        "params": {"k": args.k, "global_k": args.global_k, "pca_dim": PCA_DIM,
                   "whiten": WHITEN, "n_init": N_INIT, "max_iter": MAX_ITER,
                   "seed": SEED, "boundary_tol": BOUNDARY_TOL, "n_perm": N_PERM,
                   "fit_scenes": None},
        "kmeans": {"impl": runner.impl,
                   "source": Path(src).name if src and src.endswith(".py") else src,
                   "load_error": runner.load_error},
        "shards": [s["name"] for s in shards],
        "numpy": np.__version__,
    }
    if extra:
        meta.update(extra)
    return meta


def write_json(path: Path, payload: dict, meta: dict):
    payload = dict(payload)
    payload["meta"] = meta
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[write] {path.name}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shard-dir", "--shards-dir", dest="shard_dir", required=True,
                    type=Path, help="segA shard NPZ 디렉토리 (intrinsic_phase.py 와 동일 규약)")
    ap.add_argument("--out-dir", type=Path,
                    default=REPO_ROOT / "outputs/analysis/grid_phase/paper_supp")
    ap.add_argument("--ref-dir", type=Path,
                    default=REPO_ROOT / "outputs/analysis/grid_phase/paper_ref",
                    help="재현 게이트 기준 JSON 디렉토리 (align_pertask.json 등)")
    ap.add_argument("--mode", choices=("gate", "resid", "contingency", "all"),
                    default="all")
    ap.add_argument("--k", type=int, default=8, help="per-task 클러스터 수")
    ap.add_argument("--global-k", type=int, default=24,
                    help="contingency 의 global scope 클러스터 수")
    ap.add_argument("--shards", default=None, help="쉼표구분 shard stem 부분집합")
    ap.add_argument("--kmeans-src", default=None,
                    help="task_classification 의 phase/clustering/gpu.py 경로 "
                         "(또는 그 루트). 기준 run 과 같은 구현을 쓰기 위한 것")
    ap.add_argument("--kmeans-impl", choices=("auto", "task_classification", "numpy"),
                    default="auto")
    ap.add_argument("--gate-tol", type=float, default=1e-6)
    ap.add_argument("--no-global", action="store_true",
                    help="contingency 에서 global k24 를 건너뛴다 (메모리 절약)")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runner = IP.KMeansRunner(args.kmeans_impl, tc_root_from_src(args.kmeans_src),
                             N_INIT, MAX_ITER, SEED)
    print(f"[kmeans] impl={runner.impl} source={runner.source}", flush=True)
    if runner.load_error:
        print(f"[kmeans] task_classification 로드 실패 → fallback: {runner.load_error}",
              flush=True)

    paths = IP.discover_shards(args.shard_dir.expanduser(),
                               args.shards.split(",") if args.shards else None)
    shards = [IP.load_shard(p) for p in paths]
    for s in shards:
        print(f"[load] {s['name']}: n_rec={len(s['feat'])} "
              f"ep={len(np.unique(s['ep_id']))}", flush=True)

    need_raw = True   # gate / resid 비교 / contingency 모두 원본 라벨이 필요
    raw: dict[str, dict] = {}
    if need_raw:
        for s in shards:
            lab = cluster_one(s["feat"], s["scene"], args.k, runner)
            raw[s["name"]] = {"labels": lab, "metrics": metrics_for(s, lab)}
            print(fmt_line("raw", s["name"], raw[s["name"]]["metrics"]), flush=True)

    meta = build_meta(args, runner, shards)
    rc = 0

    if args.mode in ("gate", "all"):
        payload, ok = mode_gate(shards, raw, args.ref_dir, args.k, args.gate_tol)
        write_json(args.out_dir / "gate_check.json", payload, meta)
        if not ok:
            print("[gate] FAIL — 기준 run 과 불일치. 이후 단계 결과는 신뢰 불가.",
                  flush=True)
            if args.mode == "all":
                return 1
            rc = 1

    if args.mode in ("resid", "all"):
        payload = mode_resid(shards, raw, runner, args.k, args.out_dir)
        write_json(args.out_dir / f"resid_pertask_k{args.k}.json", payload, meta)

    if args.mode in ("contingency", "all"):
        payload = mode_contingency(shards, raw, args.k)
        write_json(args.out_dir / f"contingency_pertask_k{args.k}.json", payload, meta)
        if not args.no_global:
            payload = mode_contingency_global(shards, runner, args.global_k)
            write_json(args.out_dir / f"contingency_global_k{args.global_k}.json",
                       payload, meta)

    return rc


if __name__ == "__main__":
    raise SystemExit(main())

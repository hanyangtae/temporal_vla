#!/usr/bin/env python3
"""condg — 상태-조건부 대조 guidance 연산자 fit (phase별 W_s/W_f + 게이트 τ).

스펙 단일 출처: `docs/steering/44_cond_guidance_operator.md` (§1 연산자, §2 등록 게이트,
§3 대조군, §4 저장 규약). 정확식(전처리·릿지·고정B·AUROC)의 원본 구현은
`scripts/analysis/grid_phase/cond_margin.py` 이고, 이 스크립트는 그 수식을 **그대로**
재사용한다 (아래 "cond_margin 대비 차이" 참조).

연산자 (44 §1):
  φ(s) = [eef_pos_rel(3), eef_quat_rel(4), gripper_qpos(2)] 9 + 인접 record 차분 속도 9 = 18
         (첫 record 속도 = 0). scene 별 z-score (mp, sp+1e-8; train record 풀).
  h    = DiT L12 · 마지막 denoise(step 3) · 49토큰 mean. scene 별 중심화 (train 성공 record
         평균 mh; 해당 scene 에 성공이 없으면 그 scene 전체 평균).
  릿지(무절편)  W = (PᵀP + λI)⁻¹ PᵀX,  λ = 1e-3·n  (n = 그 클래스 train record 행수).
  margin        m = ‖h̃ − φ̃W_s‖² − ‖h̃ − φ̃W_f‖²   (m 클수록 실패 쪽 — 부호 고정, 뒤집지 않음)
  개입(serve)   m > τ 일 때만  d̂ = normalize(φ̃W_f − φ̃W_s),
                h̃′ = h̃ − β·⟨h̃ − φ̃W_s, d̂⟩·d̂

등록 게이트 (44 §2): held-out 에서 **고정B margin AUROC > 길이단독 AUROC (strict)** 이고
양 클래스 train episode ≥ 15 일 때만 registered=True. 미달 cell 은 registered=False 로
기록만 하고 serve 는 무시(identity). B = max(3, train 성공 dwell 25퍼센타일),
τ = held-out 성공 episode 고정B margin 의 90퍼센타일. 배포 연산자 = train-split W +
held-out τ (재적합 없음).

변형 (44 §3):
  condg     처치.
  condg_pl  episode 라벨을 **scene-층화 순열** 후 동일 절차. 등록 게이트는 우회 —
            처치가 등록한 phase 는 위약도 강제 등록(τ 는 자기 캘리브레이션). AUROC 는 기록.
  condg_hs  ablation "성공-모방 단독" h̃′ = (1−β)h̃ + β·φ̃W_s. W·τ·B 가 처치와 **동일**하고
            적용식만 다르므로 NPZ 내용은 condg 와 같고 meta 의 mode 만 "hs" 다.

cond_margin.py 대비 차이 (전부 44 문서가 요구한 것):
  1. split 이 scene **층화** 6:4 (cond_margin 은 비층화 전역 순열) — 44 §2.
  2. glob 이 `n*` (cond_margin 은 `n[0-4]`) 이고, 실제 셀 선택은 fit manifest 로 한다.
     manifest 가 없으면 s0–4 × n0–4 fallback = cond_margin 과 동일 집합.
  3. phase 별 rng 를 (seed, crc32(phase)) 로 고정 — 처치/위약이 **같은 split** 을 쓰게
     하기 위함 (cond_margin 은 phase 순서에 의존하는 단일 rng 스트림).
  4. AUROC 를 3종(고정B margin / 길이단독 / record) 다 산출하고 등록 판정에 쓴다.
  전처리·릿지·B·margin·auroc 함수는 라인 단위로 동일하다.

사용 (승준 노드, pkl 있는 곳):
  ~/anaconda3/bin/python scripts/steer/online_gated/fit_cond_guidance.py \
      --slug OpenDrawer_right --phases reach,grasp,pull

산출: <out-dir>/<variant>/condg.npz + metadata.json,  <out-dir>/fit_summary.json
"""
from __future__ import annotations

import os

# BLAS 스레드 cap — 공유 노드. numpy import 전에 설정해야 효력이 있다.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse  # noqa: E402
import glob as globmod  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

try:  # rollout pkl 은 torch 텐서를 담고 있어 unpickle 에 torch 가 필요할 수 있다.
    import torch  # noqa: F401
except Exception:  # pragma: no cover - torch 없는 환경에서도 import 자체는 통과시킨다
    torch = None

# cond_margin.py 와 동일한 slug → grid 경로 매핑
TASKMAP = {"OpenDrawer_left": "OpenDrawer/left", "OpenDrawer_right": "OpenDrawer/right",
           "DishwasherRack_out": "DishwasherRack/out", "OvenRack_out": "OvenRack/out",
           "PPCC_candle": "PPCC/candle", "PPCC_bread": "PPCC/bread",
           "PPCC_marshmallow": "PPCC/marshmallow", "PPCC_jug": "PPCC/jug",
           "CoffeeSetupMug": "CoffeeSetupMug"}

DEFAULT_GRID = "~/datasets/temporal_vla_store/groot/n15/grid/"
DEFAULT_VARIANTS = "condg,condg_pl,condg_hs"
FALLBACK_CELLS = {(s, n) for s in range(5) for n in range(5)}   # s5m5
MIN_PHASE_RECORDS = 4       # cond_margin: episode 당 phase record 4개 미만이면 제외
MIN_EPS_PER_CLASS = 7       # cond_margin: 클래스당 episode 7 미만이면 phase 자체를 건너뜀
MIN_TRAIN_EPS = 15          # 44 §2 최소 표본 (등록 게이트)
TRAIN_FRAC = 0.6
CELL_RE = re.compile(r"/s(\d+)/n(\d+)/")
SPEC = "44"


# ------------------------------------------------------------------ 정확식 (cond_margin 이식)
def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """rank AUROC (pos 가 높을 확률). cond_margin.auroc 와 동일."""
    if not len(pos) or not len(neg):
        return float("nan")
    s = np.concatenate([pos, neg])
    r = s.argsort().argsort().astype(np.float64) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def ridge(X_list, P_list) -> np.ndarray:
    """무절편 릿지 W = (PᵀP + λI)⁻¹PᵀX, λ = 1e-3·n. cond_margin.ridge 와 동일."""
    X = np.concatenate(X_list)
    P = np.concatenate(P_list)
    lam = 1e-3 * len(P)
    return np.linalg.solve(P.T @ P + lam * np.eye(P.shape[1]), P.T @ X)


# ------------------------------------------------------------------ 입력
def load_fit_cells(path: Path | None) -> tuple[set[tuple[int, int]], str]:
    """fit manifest 에서 (scene_idx, noise_idx) 집합을 뽑는다.

    manifest 포맷이 레포 안에 하나로 굳어 있지 않아 3종을 자동 판별한다.
      (a) 헤더에 scene_idx/noise_idx 열이 있는 index 계열 tsv
      (b) fit_setm 계열 headerless `pkl경로 \\t label \\t scene` → 경로에서 s*/n* 파싱
      (c) replay_cells 계열 headerless `scene_idx \\t noise_idx \\t env_seed ...`
    파일이 없으면 s0–4 × n0–4 fallback (+경고).
    """
    if path is None or not path.exists():
        print(f"[warn] fit manifest 없음 ({path}) — s0-4 × n0-4 fallback 사용", flush=True)
        return set(FALLBACK_CELLS), "fallback_s5m5"
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.startswith("#")]
    if not lines:
        print(f"[warn] fit manifest 비어 있음 ({path}) — s0-4 × n0-4 fallback", flush=True)
        return set(FALLBACK_CELLS), "fallback_s5m5"
    head = lines[0].split("\t")
    cells: set[tuple[int, int]] = set()
    if "scene_idx" in head and "noise_idx" in head:      # (a)
        ci, ni = head.index("scene_idx"), head.index("noise_idx")
        for ln in lines[1:]:
            p = ln.split("\t")
            cells.add((int(p[ci]), int(p[ni])))
        return cells, "header_scene_noise"
    first = head[0]
    if first.endswith(".pkl") or "/" in first:           # (b)
        for ln in lines:
            m = CELL_RE.search(ln.split("\t")[0])
            if m:
                cells.add((int(m.group(1)), int(m.group(2))))
        if cells:
            return cells, "path_scene_noise"
    try:                                                  # (c)
        for ln in lines:
            p = ln.split("\t")
            cells.add((int(p[0]), int(p[1])))
        return cells, "leading_two_ints"
    except (ValueError, IndexError):
        pass
    print(f"[warn] fit manifest 포맷 판별 실패 ({path}) — s0-4 × n0-4 fallback", flush=True)
    return set(FALLBACK_CELLS), "fallback_s5m5"


def load_episodes(slug: str, grid_root: Path, cells: set[tuple[int, int]],
                  layer: int, denoise_step: int) -> list[dict]:
    """rollout pkl → episode 레코드. 특징 추출식은 cond_margin 과 동일."""
    pat = str(grid_root / f"*/*/{TASKMAP[slug]}/s*/n*/base/rollout.pkl")
    pkls = sorted(globmod.glob(pat))
    # 경로 기반 사전 필터 (pkl 로드 비용 절감). 최종 판정은 pkl 안의 scene_idx/noise_idx.
    pre = []
    for p in pkls:
        m = CELL_RE.search(p)
        if m is None or (int(m.group(1)), int(m.group(2))) in cells:
            pre.append(p)
    print(f"# {slug}: pkl {len(pkls)}개 → fit cell 후보 {len(pre)}개", flush=True)

    eps: list[dict] = []
    seen_cells: set[tuple[int, int]] = set()
    for p in pre:
        with open(p, "rb") as f:
            d = pickle.load(f)
        cell = (int(d["scene_idx"]), int(d["noise_idx"]))
        if cell not in cells:
            continue
        if cell in seen_cells:      # 다중 머신 수집 task 의 중복 셀 제거 (첫 등장만)
            continue
        seen_cells.add(cell)
        cap = [int(x) for x in d["capture_layers"]]
        H = np.stack([np.asarray(h[cap.index(layer), denoise_step], dtype=np.float32).mean(0)
                      for h in d["hidden_states"]])
        st = [np.concatenate([np.asarray(s["observation.state.eef_pos_rel"]),
                              np.asarray(s["observation.state.eef_quat_rel"]),
                              np.asarray(s["observation.state.gripper_qpos"])])
              for s in d["states"]]
        P0 = np.stack(st).astype(np.float64)
        P = np.hstack([P0, np.vstack([np.zeros((1, 9)), np.diff(P0, axis=0)])])
        eps.append({
            "scene": int(d["scene_idx"]), "noise": int(d["noise_idx"]),
            "success": int(d["episode_success"]),
            "H": H.astype(np.float64), "P": P,
            "phases": list(d["feature_phases"]),
            # docs/04 규약 — 출처는 절대경로가 아니라 내용 지문으로 남긴다.
            "sig": hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16],
        })
    return eps


# ------------------------------------------------------------------ 분할·라벨
def stratified_train_idx(rows: list[tuple], seed: int, phase: str) -> set[int]:
    """scene 층화 6:4 episode 분할 (44 §2). scene 마다 순열 후 앞 60% 를 train.

    rng 는 (seed, crc32(phase)) 로 고정 — phase 처리 순서와 무관하고, 처치/위약이
    같은 split 을 쓴다 (라벨만 다른 대조가 되도록)."""
    rng = np.random.default_rng([seed, zlib.crc32(phase.encode("utf-8"))])
    tr: set[int] = set()
    for sc in sorted({r[0] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r[0] == sc]
        order = rng.permutation(len(idx))
        n_tr = int(len(idx) * TRAIN_FRAC)
        tr.update(idx[j] for j in order[:n_tr])
    return tr


def scene_stratified_permute(eps: list[dict], seed: int) -> list[int]:
    """episode 성공 라벨을 scene 안에서만 순열 (44 §3 위약). 반환 = episode 순 라벨 리스트."""
    rng = np.random.default_rng([seed, 12345])
    labels = [e["success"] for e in eps]
    out = list(labels)
    for sc in sorted({e["scene"] for e in eps}):
        idx = [i for i, e in enumerate(eps) if e["scene"] == sc]
        vals = np.array([labels[i] for i in idx])
        vals = vals[rng.permutation(len(vals))]
        for i, v in zip(idx, vals):
            out[i] = int(v)
    return out


# ------------------------------------------------------------------ phase fit
def fit_phase(eps: list[dict], labels: list[int], phase: str, seed: int,
              min_train_eps: int) -> dict | None:
    """한 phase 의 W_s/W_f/B/τ/AUROC. 미달이면 registered=False 인 dict, 아예 불가면 None."""
    rows = []
    for k, e in enumerate(eps):
        idx = [i for i, ph in enumerate(e["phases"]) if ph == phase]
        if len(idx) >= MIN_PHASE_RECORDS:
            rows.append((e["scene"], int(labels[k]), e["H"][idx], e["P"][idx], k))
    if not rows:
        return None
    ns = sum(r[1] for r in rows)
    nf = len(rows) - ns
    base = {"phase": phase, "n_eps": len(rows), "n_eps_succ": int(ns), "n_eps_fail": int(nf)}
    if ns < MIN_EPS_PER_CLASS or nf < MIN_EPS_PER_CLASS:
        return {**base, "registered": False,
                "skip_reason": f"episode 부족 s/f={ns}/{nf} (<{MIN_EPS_PER_CLASS})"}

    tr = stratified_train_idx(rows, seed, phase)

    # scene-중심화 파라미터는 train 에서만 (성공 기준). cond_margin 과 동일.
    stats: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for sc in sorted({r[0] for r in rows}):
        g = [rows[i] for i in tr if rows[i][0] == sc] or [r for r in rows if r[0] == sc]
        ref_h = np.concatenate([r[2] for r in g if r[1] == 1]) if any(
            r[1] == 1 for r in g) else np.concatenate([r[2] for r in g])
        ref_p = np.concatenate([r[3] for r in g])
        stats[sc] = (ref_h.mean(0), ref_p.mean(0), ref_p.std(0) + 1e-8)

    def prep(i):
        sc, su, H, P, _ = rows[i]
        mh, mp, sp = stats[sc]
        return su, H - mh, (P - mp) / sp

    tr_s = [i for i in tr if rows[i][1] == 1]
    tr_f = [i for i in tr if rows[i][1] == 0]
    n_tr_s, n_tr_f = len(tr_s), len(tr_f)
    if n_tr_s < 1 or n_tr_f < 1:
        return {**base, "registered": False, "n_train_succ": n_tr_s, "n_train_fail": n_tr_f,
                "skip_reason": "train 에 한쪽 클래스 없음 — 대조 불가"}
    prep_s = [prep(i) for i in tr_s]
    prep_f = [prep(i) for i in tr_f]
    Ws = ridge([p[1] for p in prep_s], [p[2] for p in prep_s])
    Wf = ridge([p[1] for p in prep_f], [p[2] for p in prep_f])

    # 길이 공정화: 고정 예산 B = train 성공 dwell 의 25퍼센타일 (전 episode 동일 초반 창)
    succ_dw = sorted(len(rows[i][2]) for i in tr_s)
    B = max(3, succ_dw[len(succ_dw) // 4])

    ep_m = {0: [], 1: []}
    fixB_m = {0: [], 1: []}
    rec_m = {0: [], 1: []}
    dwell = {0: [], 1: []}
    for i in range(len(rows)):
        if i in tr:
            continue
        su, X, P = prep(i)
        m = ((X - P @ Ws) ** 2).sum(1) - ((X - P @ Wf) ** 2).sum(1)
        ep_m[su].append(float(np.mean(m)))
        if len(m) >= B:
            fixB_m[su].append(float(np.mean(m[:B])))
        rec_m[su].append(m)
        dwell[su].append(len(m))

    a_fixB = auroc(np.array(fixB_m[0]), np.array(fixB_m[1]))
    a_len = auroc(np.array(dwell[0], float), np.array(dwell[1], float))
    a_ep = auroc(np.array(ep_m[0]), np.array(ep_m[1]))
    a_rec = auroc(np.concatenate(rec_m[0]) if rec_m[0] else np.array([]),
                  np.concatenate(rec_m[1]) if rec_m[1] else np.array([]))
    # τ = held-out 성공 episode 고정B margin 의 90퍼센타일 (44 §2)
    tau = float(np.percentile(np.array(fixB_m[1]), 90)) if fixB_m[1] else float("nan")

    # 등록 게이트 (44 §2): 고정B margin AUROC > 길이단독 AUROC (strict) AND 표본 하한.
    # 부호는 뒤집지 않는다 — 역전 cell(Dish contact 0.28)은 이 게이트가 자동 배제한다.
    reasons = []
    if not (np.isfinite(a_fixB) and np.isfinite(a_len) and a_fixB > a_len):
        reasons.append(f"고정B AUROC {a_fixB:.3f} ≤ 길이단독 {a_len:.3f}")
    if min(n_tr_s, n_tr_f) < min_train_eps:
        reasons.append(f"train episode {n_tr_s}/{n_tr_f} < {min_train_eps}")
    if not np.isfinite(tau):
        reasons.append("held-out 성공 고정B 표본 0 — τ 산출 불가")
    return {
        **base, "registered": not reasons,
        "skip_reason": "; ".join(reasons) if reasons else "",
        "W_s": Ws, "W_f": Wf, "B": int(B), "tau": tau, "sign": 1.0,
        "auroc_margin_fixB": a_fixB, "auroc_len": a_len, "auroc_record": a_rec,
        "auroc_episode": a_ep,
        "n_train": len(tr), "n_train_succ": n_tr_s, "n_train_fail": n_tr_f,
        "n_heldout_succ": len(ep_m[1]), "n_heldout_fail": len(ep_m[0]),
        "n_fixB_succ": len(fixB_m[1]), "n_fixB_fail": len(fixB_m[0]),
        "stats": stats,
        # global fallback (미지 scene 용): train 전체 record 풀
        "global": _global_stats(rows, tr),
    }


def _global_stats(rows: list[tuple], tr: set[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """unseen scene 용 fallback 중심화 파라미터 — train 전체(성공 기준 mh)."""
    g = [rows[i] for i in tr] or list(rows)
    ref_h = np.concatenate([r[2] for r in g if r[1] == 1]) if any(
        r[1] == 1 for r in g) else np.concatenate([r[2] for r in g])
    ref_p = np.concatenate([r[3] for r in g])
    return ref_h.mean(0), ref_p.mean(0), ref_p.std(0) + 1e-8


# ------------------------------------------------------------------ 저장
def save_npz(out_dir: Path, variant: str, mode: str, results: dict[str, dict],
             meta: dict) -> Path:
    """NPZ (docs/04 규약: 절대경로 미기록, 입력 출처는 sig 리스트로)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    arr: dict[str, np.ndarray] = {}
    phases_out = []
    for ph, r in results.items():
        phases_out.append(ph)
        arr[f"{ph}__registered"] = np.array(bool(r.get("registered", False)))
        for key in ("auroc_margin_fixB", "auroc_len", "auroc_record", "auroc_episode"):
            arr[f"{ph}__{key}"] = np.array(float(r.get(key, np.nan)))
        arr[f"{ph}__tau"] = np.array(float(r.get("tau", np.nan)))
        arr[f"{ph}__B"] = np.array(int(r.get("B", 0)))
        arr[f"{ph}__sign"] = np.array(float(r.get("sign", 1.0)))
        if "W_s" not in r:
            continue
        arr[f"{ph}__W_s"] = r["W_s"].astype(np.float32)
        arr[f"{ph}__W_f"] = r["W_f"].astype(np.float32)
        mh_g, mp_g, sp_g = r["global"]
        arr[f"{ph}__mh_global"] = mh_g.astype(np.float32)
        arr[f"{ph}__mp_global"] = mp_g.astype(np.float32)
        arr[f"{ph}__sp_global"] = sp_g.astype(np.float32)
        scenes = sorted(r["stats"])
        arr[f"{ph}__scenes"] = np.array(scenes, dtype=np.int32)
        for sc in scenes:
            mh, mp, sp = r["stats"][sc]
            arr[f"{ph}__scene{sc}__mh"] = mh.astype(np.float32)
            arr[f"{ph}__scene{sc}__mp"] = mp.astype(np.float32)
            arr[f"{ph}__scene{sc}__sp"] = sp.astype(np.float32)
    arr["phases"] = np.array(phases_out, dtype="<U64")
    arr["registered_phases"] = np.array(
        [p for p in phases_out if results[p].get("registered")], dtype="<U64")
    full_meta = {**meta, "variant": variant, "mode": mode}
    arr["meta_json"] = np.array(json.dumps(full_meta, ensure_ascii=False))
    path = out_dir / "condg.npz"
    np.savez_compressed(path, **arr)
    (out_dir / "metadata.json").write_text(
        json.dumps(full_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def print_table(variant: str, results: dict[str, dict]) -> None:
    print(f"\n[{variant}] phase 표 (held-out; fail>succ 방향)", flush=True)
    print(f"  {'phase':<16}{'고정B':>7}{'길이':>7}{'record':>8}{'ep':>7}"
          f"{'B':>5}{'tau':>10}  {'reg':<5} 표본(train s/f, heldout s/f)", flush=True)
    for ph, r in results.items():
        if "W_s" not in r:
            print(f"  {ph:<16}{'-':>7}{'-':>7}{'-':>8}{'-':>7}{'-':>5}{'-':>10}  "
                  f"{'no':<5} {r.get('skip_reason', '')}", flush=True)
            continue
        print(f"  {ph:<16}{r['auroc_margin_fixB']:>7.2f}{r['auroc_len']:>7.2f}"
              f"{r['auroc_record']:>8.2f}{r['auroc_episode']:>7.2f}{r['B']:>5d}"
              f"{r['tau']:>10.2f}  {'YES' if r['registered'] else 'no':<5} "
              f"({r['n_train_succ']}/{r['n_train_fail']}, "
              f"{r['n_heldout_succ']}/{r['n_heldout_fail']})"
              + (f"  ← {r['skip_reason']}" if r["skip_reason"] else ""), flush=True)


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--slug", required=True, help=f"task slug (가능: {sorted(TASKMAP)})")
    ap.add_argument("--phases", required=True, help="콤마 목록 (예: reach,grasp,pull)")
    ap.add_argument("--grid-root", type=Path, default=Path(DEFAULT_GRID),
                    help=f"grid rollout 루트 (기본 {DEFAULT_GRID})")
    ap.add_argument("--cells-tsv", type=Path, default=None,
                    help="정본 fit cell manifest (기본 "
                         "outputs/steer/online_pipe/manifests/<slug>_s5m5.tsv)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="기본 outputs/steer/online_pipe/<slug>/condg_s5m5/")
    ap.add_argument("--variants", default=DEFAULT_VARIANTS,
                    help=f"생성할 변형 (기본 {DEFAULT_VARIANTS})")
    ap.add_argument("--seed", type=int, default=0, help="split·위약 순열 seed (기본 0)")
    ap.add_argument("--layer", type=int, default=12, help="DiT 물리 layer (기본 12)")
    ap.add_argument("--denoise-step", type=int, default=3,
                    help="denoise step 인덱스 (기본 3 = 마지막)")
    ap.add_argument("--min-train-eps", type=int, default=MIN_TRAIN_EPS,
                    help=f"등록 게이트의 클래스당 train episode 하한 (기본 {MIN_TRAIN_EPS})")
    args = ap.parse_args()

    if args.slug not in TASKMAP:
        raise SystemExit(f"알 수 없는 slug {args.slug} (가능: {sorted(TASKMAP)})")
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    bad = [v for v in variants if v not in ("condg", "condg_pl", "condg_hs")]
    if bad:
        raise SystemExit(f"알 수 없는 variant {bad}")
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    grid_root = args.grid_root.expanduser()
    cells_tsv = (args.cells_tsv or
                 Path(f"outputs/steer/online_pipe/manifests/{args.slug}_s5m5.tsv")).expanduser()
    out_dir = (args.out_dir or
               Path(f"outputs/steer/online_pipe/{args.slug}/condg_s5m5")).expanduser()

    cells, cells_mode = load_fit_cells(cells_tsv)
    eps = load_episodes(args.slug, grid_root, cells, args.layer, args.denoise_step)
    if not eps:
        raise SystemExit(f"fit cell 에 해당하는 rollout 0개 (slug={args.slug})")
    n_s = sum(e["success"] for e in eps)
    used_cells = sorted((e["scene"], e["noise"]) for e in eps)
    print(f"[{args.slug}] episode {len(eps)} (succ {n_s} / fail {len(eps) - n_s}) "
          f"scene {sorted({e['scene'] for e in eps})} cells_mode={cells_mode}", flush=True)

    base_meta = {
        "spec": SPEC, "lambda": "1e-3*n", "seed": args.seed, "slug": args.slug,
        "phases_requested": phases, "layer": args.layer, "denoise_step": args.denoise_step,
        "feature": "DiT L%d · denoise %d · 49-token mean" % (args.layer, args.denoise_step),
        "state_phi": "[eef_pos_rel(3), eef_quat_rel(4), gripper_qpos(2)] + 차분속도 = 18d",
        "split": "episode 6:4, scene 층화, rng([seed, crc32(phase)])",
        "gate": "고정B margin AUROC > 길이단독 AUROC (strict) AND "
                f"클래스당 train episode ≥ {args.min_train_eps}",
        "tau": "held-out 성공 episode 고정B margin 90퍼센타일",
        "B": "max(3, train 성공 dwell 25퍼센타일)",
        "sign_note": "margin 부호 고정(m 클수록 실패) — 역전 cell 은 게이트가 배제",
        "fit_cells": [[int(s), int(n)] for s, n in used_cells],
        "fit_cells_source": cells_mode,
        "input_sigs": [e["sig"] for e in eps],   # docs/04 — 경로 아닌 내용 지문
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": "docs/steering/44_cond_guidance_operator.md + "
                  "scripts/analysis/grid_phase/cond_margin.py",
    }

    labels_treat = [e["success"] for e in eps]
    res_treat: dict[str, dict] = {}
    for ph in phases:
        r = fit_phase(eps, labels_treat, ph, args.seed, args.min_train_eps)
        if r is None:
            print(f"  [{ph}] phase 없음(codebook) — 건너뜀", flush=True)
            continue
        res_treat[ph] = r
    if not res_treat:
        raise SystemExit("처리 가능한 phase 0개")
    print_table("condg", res_treat)
    reg_phases = {ph for ph, r in res_treat.items() if r.get("registered")}
    print(f"  등록 phase: {sorted(reg_phases) or '없음'}", flush=True)

    written = []
    summary: dict[str, dict] = {}
    if "condg" in variants:
        p = save_npz(out_dir / "condg", "condg", "margin_gated_contrast", res_treat, base_meta)
        written.append(p)
    if "condg_hs" in variants:
        # W·τ·B 는 처치와 동일. 적용식만 h̃′ = (1−β)h̃ + β·φ̃W_s (44 §3) → mode 로만 구분.
        p = save_npz(out_dir / "condg_hs", "condg_hs", "success_mimic", res_treat,
                     {**base_meta, "note": "condg 와 동일한 W/τ/B — 적용식만 성공-모방 단독"})
        written.append(p)
    if "condg_pl" in variants:
        labels_pl = scene_stratified_permute(eps, args.seed)
        res_pl: dict[str, dict] = {}
        for ph in res_treat:
            r = fit_phase(eps, labels_pl, ph, args.seed, args.min_train_eps)
            if r is None:
                continue
            # 44 §3: 위약은 자기 등록 게이트를 **우회**하고, 등록 집합을 처치와 일치시킨다.
            #   - 처치가 등록한 phase → 강제 등록 (게이트에 걸려 identity 가 되면
            #     타이밍-섭동 대조 기능을 잃는다).
            #   - 처치가 등록 안 한 phase → 자기 AUROC 가 좋아도 미등록 (처치가 손대지
            #     않는 cell 에 위약만 개입하면 짝이 맞지 않는다).
            # 자기 게이트 판정은 own_gate_* 로 남긴다.
            r["own_gate_registered"] = bool(r.get("registered"))
            r["own_gate_reason"] = r.get("skip_reason", "")
            r["registered"] = ph in reg_phases and "W_s" in r
            r["forced_register"] = bool(r["registered"] and not r["own_gate_registered"])
            r["skip_reason"] = "" if r["registered"] else (
                "처치 미등록 phase — 위약도 미등록(짝맞춤)" if ph not in reg_phases
                else r.get("skip_reason", ""))
            if ph in reg_phases and "W_s" not in r:
                print(f"  [warn] 위약 {ph}: W 산출 실패로 강제 등록 불가 "
                      f"({r.get('own_gate_reason')})", flush=True)
            res_pl[ph] = r
        print_table("condg_pl", res_pl)
        p = save_npz(out_dir / "condg_pl", "condg_pl", "margin_gated_contrast",
                     res_pl, {**base_meta,
                              "placebo": "episode 라벨 scene-층화 순열, 등록 게이트 우회",
                              "placebo_labels": [int(x) for x in labels_pl]})
        written.append(p)
        summary["condg_pl"] = {ph: _row_summary(r) for ph, r in res_pl.items()}

    summary["condg"] = {ph: _row_summary(r) for ph, r in res_treat.items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fit_summary.json").write_text(
        json.dumps({**base_meta, "variants": variants,
                    "registered_phases": sorted(reg_phases),
                    "results": summary}, indent=2, ensure_ascii=False), encoding="utf-8")
    for p in written:
        print(f"[write] {p.relative_to(out_dir.parent) if out_dir.parent in p.parents else p.name}",
              flush=True)
    print(f"FIT_CONDG_DONE {args.slug}", flush=True)


def _row_summary(r: dict) -> dict:
    """JSON 요약 (행렬 제외)."""
    return {k: v for k, v in r.items() if k not in ("W_s", "W_f", "stats", "global")}


if __name__ == "__main__":
    sys.exit(main())

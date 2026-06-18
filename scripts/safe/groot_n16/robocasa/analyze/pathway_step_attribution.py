"""per-step phase-resolved VL/DiT 실패 귀인 (length 통제, scipy-free).

기존 vl_anomaly_score / vl_dit_lda_analysis 의 **t<=8 mean-pool 귀인** 을 교정한다.
그 풀링은 (a) "전체 이상 크기" 공통인자를 만들어 VL·DiT 이탈도를 co-linear 로 보이게 하고
(b) Phase 3 가 찾은 phase 구조(VL 이른 신호 t<=8 / DiT 늦은 신호 t>=12)를 뭉갠다
([[n16-pathway-attribution-colinear]]). 여기서는 rollout-t 를 풀지 않고 **step 별로** 본다.

방법 (각 env-step t):
  1. 그 t 에 살아있는(length>t) **성공 rollout 만** 으로 성공 분포(대각 mu/std) fit
     → length / survivorship 통제. 성공은 일찍 끝나므로 living 성공이 충분한 t 까지만 유효.
  2. 각 rollout 의 step-t feature 가 VL / DiT 각각에서 성공 분포로부터 벗어난 정도
     = 대각 Mahalanobis d_VL(t), d_DiT(t).
  3. pathway 간 스케일이 다르므로(VL D=2048 / DiT D=1536) 직접 비교 금지 →
     living-success 이탈도 분포 기준 **percentile-rank** r_VL(t), r_DiT(t) in [0,1] 로 환산.
     δ(t) = r_VL - r_DiT. δ>0 = VL 쪽이 (상대적으로) 더 이상 → goal-lean, δ<0 = motor-lean.
  4. phase 검정: 실패들의 mean δ 가 early(t<=early_t) 에서 양(VL)→ late([late_lo,Tmax]) 에서
     음(DiT) 으로 가나. permutation(succ/fail label shuffle) null 로 우연 통제.
  보너스: 각 t 에서 VL/DiT succ-vs-fail AUROC(pure numpy) — Phase 3 와 직접 대조.

confound: living-rollout(길이), --instruction-filter / --split-instruction(ep_meta.lang) 로 통제.
pkl 은 원격 전용(torch 텐서) → remote-compute, ${REMOTE_PYTHON}. 출력 JSON/plot 소용량.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "scripts/safe/groot_n16/robocasa/vis"))
from core.io import load_pkl  # noqa: E402

# moderate10 pathway_pertoken DiT capture layers (block id) → local index.
CAPTURE_LAYERS = [0, 2, 4, 8, 16, 24, 31]
VALID_ACTION = (1, 17)  # action token slice [1:17] (state=token0), SAFE valid16.
SEED = 0


# ----------------------------------------------------------------------------- loading

def _instr(d: dict) -> str:
    return str(d.get("ep_meta", {}).get("lang", ""))


def load_task(task_dir: Path, dit_local_idx: int, instr_filter: str | None) -> list[dict]:
    """task_dir/*.pkl → [{success,length,vl:[n,Dvl],dit:[n,D],lang}]. VL/DiT 없으면 skip."""
    out: list[dict] = []
    for p in sorted(task_dir.glob("*.pkl")):
        d = load_pkl(p)
        vl, hs = d.get("vl_hidden_states"), d.get("hidden_states")
        if vl is None or hs is None:
            continue
        lang = _instr(d)
        if instr_filter and instr_filter.lower() not in lang.lower():
            continue
        vl_arr = np.stack([np.asarray(v, dtype=np.float32) for v in vl])          # [n, Dvl]
        hs_arr = np.stack([np.asarray(h, dtype=np.float32) for h in hs])          # [n, L, T, D]
        dit_arr = hs_arr[:, dit_local_idx, VALID_ACTION[0]:VALID_ACTION[1], :].mean(axis=1)  # [n, D]
        out.append({
            "name": p.stem,
            "success": int(d.get("episode_success", 0)),
            "length": int(len(vl)),
            "vl": vl_arr,
            "dit": dit_arr,
            "lang": lang,
        })
    return out


# ----------------------------------------------------------------------------- metrics

def mahal_diag(x: np.ndarray, mu: np.ndarray, std: np.ndarray) -> float:
    """대각 공분산 Mahalanobis distance. std~0 차원은 무시."""
    m = std > 1e-8
    if not m.any():
        return 0.0
    return float(np.sqrt(np.mean(((x[m] - mu[m]) / std[m]) ** 2)))


def pct_rank(value: float, ref: np.ndarray) -> float:
    """ref 분포에서 value 의 percentile-rank in [0,1] (ref 중 <=value 비율)."""
    if ref.size == 0:
        return 0.5
    return float(np.mean(ref <= value))


def auroc(scores: np.ndarray, y: np.ndarray) -> float:
    """rank 기반 AUROC (pure numpy, scipy 불필요). y=1 을 positive."""
    pos, neg = scores[y == 1], scores[y == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # tie 평균 rank 보정
    _, inv, cnt = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(cnt.size); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    r_pos = ranks[y == 1].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ----------------------------------------------------------------------------- per-step δ

def step_table(rollouts: list[dict], *, min_succ: int, min_fail: int, max_t: int):
    """각 t 에서 living-success 분포로 d_VL,d_DiT → percentile r → δ. 유효 t 만 반환.

    returns dict t -> {n_succ,n_fail, vl_auroc,dit_auroc, mean_delta, frac_vl_dom,
                       per_fail_delta:[...]} (living 성공/실패 충분한 t 만).
    """
    succ = [r for r in rollouts if r["success"] == 1]
    fail = [r for r in rollouts if r["success"] == 0]
    res: dict[int, dict] = {}
    for t in range(max_t):
        s_live = [r for r in succ if r["length"] > t]
        f_live = [r for r in fail if r["length"] > t]
        if len(s_live) < min_succ or len(f_live) < min_fail:
            continue
        out: dict[str, dict] = {}
        ranks = {}
        for key in ("vl", "dit"):
            S = np.stack([r[key][t] for r in s_live])           # [n_succ, D]
            mu, std = S.mean(axis=0), S.std(axis=0)
            d_succ = np.array([mahal_diag(v, mu, std) for v in S])
            d_fail = np.array([mahal_diag(r[key][t], mu, std) for r in f_live])
            out[key] = (d_succ, d_fail)
            ranks[key] = np.array([pct_rank(df, d_succ) for df in d_fail])  # [n_fail]
        # succ-vs-fail AUROC (Phase 3 대조)
        vl_scores = np.concatenate([out["vl"][0], out["vl"][1]])
        dit_scores = np.concatenate([out["dit"][0], out["dit"][1]])
        y = np.concatenate([np.zeros(len(s_live)), np.ones(len(f_live))])
        delta = ranks["vl"] - ranks["dit"]                      # [n_fail]
        res[t] = {
            "n_succ": len(s_live), "n_fail": len(f_live),
            "vl_auroc": auroc(vl_scores, y), "dit_auroc": auroc(dit_scores, y),
            "mean_delta": float(delta.mean()),
            "frac_vl_dom": float(np.mean(delta > 0)),
            "delta": delta,
        }
    return res


def phase_contrast(res: dict[int, dict], early_t: int, late_lo: int) -> dict | None:
    """early(t<=early_t) mean δ - late(t>=late_lo) mean δ. 양수 = VL이 early에 더 우세(가설)."""
    e = [v["mean_delta"] for t, v in res.items() if t < early_t]
    l = [v["mean_delta"] for t, v in res.items() if t >= late_lo]
    if not e or not l:
        return None
    return {"early_mean_delta": float(np.mean(e)), "late_mean_delta": float(np.mean(l)),
            "contrast": float(np.mean(e) - np.mean(l)),
            "n_early_t": len(e), "n_late_t": len(l)}


def perm_null(rollouts, *, n_perm, min_succ, min_fail, max_t, early_t, late_lo):
    """succ/fail label 을 섞어 phase contrast null 분포(95% band)."""
    rng = np.random.default_rng(SEED)
    base = [r.copy() for r in rollouts]
    vals = []
    labels = np.array([r["success"] for r in base])
    for _ in range(n_perm):
        perm = rng.permutation(labels)
        for r, s in zip(base, perm):
            r["success"] = int(s)
        res = step_table(base, min_succ=min_succ, min_fail=min_fail, max_t=max_t)
        pc = phase_contrast(res, early_t, late_lo)
        if pc:
            vals.append(pc["contrast"])
    if not vals:
        return None
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


# ----------------------------------------------------------------------------- run

def run_one(rollouts, label, args) -> dict | None:
    succ = sum(r["success"] for r in rollouts)
    fail = len(rollouts) - succ
    if succ < args.min_succ or fail < args.min_fail:
        print(f"  [skip] {label}: succ={succ} fail={fail} (< min)")
        return None
    res = step_table(rollouts, min_succ=args.min_succ, min_fail=args.min_fail, max_t=args.max_t)
    if not res:
        print(f"  [skip] {label}: living t 없음")
        return None
    pc = phase_contrast(res, args.early_t, args.late_lo)
    null = perm_null(rollouts, n_perm=args.perm, min_succ=args.min_succ, min_fail=args.min_fail,
                     max_t=args.max_t, early_t=args.early_t, late_lo=args.late_lo) if pc else None
    valid_t = sorted(res)
    sig = (pc and null and (pc["contrast"] < null[0] or pc["contrast"] > null[1]))
    print(f"  {label}: valid t={valid_t[0]}..{valid_t[-1]} | "
          f"phase contrast={pc['contrast']:+.3f} null95={null} "
          f"{'SIG' if sig else 'ns'}" if pc else f"  {label}: phase contrast 계산불가")
    return {
        "label": label, "n_succ": succ, "n_fail": fail,
        "valid_t": valid_t,
        "per_t": {str(t): {k: v for k, v in d.items() if k != "delta"} for t, d in res.items()},
        "phase_contrast": pc, "perm_null95": null, "significant": bool(sig),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="per-step phase-resolved VL/DiT failure attribution")
    ap.add_argument("--run-dir", required=True, help="raw_rollouts dir (task subdirs)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dit-block", type=int, default=31, help="DiT block id (capture 중 최근접)")
    ap.add_argument("--early-t", type=int, default=8)
    ap.add_argument("--late-lo", type=int, default=12)
    ap.add_argument("--max-t", type=int, default=30, help="이 t 까지만 검사(이후 living 성공 부족)")
    ap.add_argument("--min-succ", type=int, default=8)
    ap.add_argument("--min-fail", type=int, default=8)
    ap.add_argument("--perm", type=int, default=200)
    ap.add_argument("--split-instruction", action="store_true",
                    help="task 내 instruction(ep_meta.lang) 변형별로도 따로 분석")
    ap.add_argument("--min-instr-rollouts", type=int, default=20)
    ap.add_argument("--tasks", default=None, help="콤마구분 task 이름 필터(기본 전체)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir.parent / "analysis" / "pathway_step_attribution"
    local_idx = int(np.argmin([abs(b - args.dit_block) for b in CAPTURE_LAYERS]))
    dit_block = CAPTURE_LAYERS[local_idx]

    task_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir()])
    if args.tasks:
        keep = set(args.tasks.split(","))
        task_dirs = [p for p in task_dirs if p.name in keep]
    if not task_dirs:
        raise SystemExit(f"no task dir under {run_dir}")
    print(f"[load] {len(task_dirs)} tasks, DiT block={dit_block} (local L{local_idx})")

    if args.smoke:
        for td in task_dirs[:2]:
            rs = load_task(td, local_idx, None)
            langs = Counter(r["lang"] for r in rs)
            print(f"  {td.name}: n={len(rs)} succ={sum(r['success'] for r in rs)} "
                  f"vl={rs[0]['vl'].shape if rs else None} dit={rs[0]['dit'].shape if rs else None} "
                  f"instr={dict(langs)}")
        return

    results = {"dit_block": dit_block, "early_t": args.early_t, "late_lo": args.late_lo,
               "max_t": args.max_t, "tasks": {}}
    for td in task_dirs:
        rs = load_task(td, local_idx, None)
        if not rs:
            print(f"  [skip] {td.name}: VL feature 없음")
            continue
        print(f"[task] {td.name} (n={len(rs)})")
        entry = {"mixed": run_one(rs, f"{td.name}/mixed", args)}
        if args.split_instruction:
            by_lang: dict[str, list] = defaultdict(list)
            for r in rs:
                by_lang[r["lang"]].append(r)
            entry["by_instruction"] = {}
            for lang, grp in by_lang.items():
                if len(grp) < args.min_instr_rollouts:
                    continue
                entry["by_instruction"][lang] = run_one(grp, f"{td.name}/[{lang}]", args)
        results["tasks"][td.name] = entry

    out.mkdir(parents=True, exist_ok=True)
    (out / "pathway_step_attribution.json").write_text(json.dumps(results, indent=2))
    print(f"[done] -> {out/'pathway_step_attribution.json'}")
    # 요약: phase contrast SIG task
    sig = [(t, e["mixed"]["phase_contrast"]["contrast"])
           for t, e in results["tasks"].items()
           if e.get("mixed") and e["mixed"].get("significant")]
    print(f"[summary] phase-contrast SIG (mixed): {sig}")


if __name__ == "__main__":
    main()

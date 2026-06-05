"""Step별 VL/DiT anomaly trajectory 분석.

각 rollout의 step t마다 VL·DiT feature가 성공 분포에서 얼마나 벗어나는지를
시간 흐름대로 추적. 목적: "언제 이상 신호가 시작되는가" 확인.

출력:
  - step별 anomaly 곡선 (성공 평균 vs 사분면별 실패 평균)
  - 각 사분면 대표 ep의 step별 곡선
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "scripts/safe/groot_n16/robocasa/vis"))
from core.io import load_pkl  # noqa: E402

CAPTURE_LAYERS = [0, 2, 4, 8, 16, 24, 31]


def mahal(x, mu, std):
    mask = std > 1e-8
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(((x[mask] - mu[mask]) / std[mask]) ** 2) ** 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--early-t", type=int, default=8)
    ap.add_argument("--dit-block", type=int, default=31)
    ap.add_argument("--max-steps", type=int, default=45)
    args = ap.parse_args()

    task_dir = Path(args.task_dir)
    out = Path(args.out) if args.out else \
        task_dir.parent.parent / "analysis" / "vl_dit_stepwise" / task_dir.name
    out.mkdir(parents=True, exist_ok=True)

    local_idx = int(np.argmin([abs(b - args.dit_block) for b in CAPTURE_LAYERS]))
    dit_block_actual = CAPTURE_LAYERS[local_idx]
    print(f"[load] {task_dir.name}  DiT block={dit_block_actual} (local L{local_idx})")

    # feature 로드
    records = []
    for p in sorted(task_dir.glob("*.pkl")):
        d = load_pkl(p)
        vl = d.get("vl_hidden_states")
        hs = d.get("hidden_states")
        if vl is None or hs is None:
            continue
        vl_arr = np.stack([np.asarray(v, dtype=np.float32) for v in vl])
        hs_arr = np.stack([np.asarray(h, dtype=np.float32) for h in hs])
        dit_arr = hs_arr[:, local_idx, 1:17, :].mean(axis=1)
        records.append({
            "mp4": p.with_suffix(".mp4").name,
            "success": int(d.get("episode_success", 0)),
            "length": len(vl),
            "vl": vl_arr,
            "dit": dit_arr,
        })

    succ = [r for r in records if r["success"] == 1]
    fail = [r for r in records if r["success"] == 0]
    print(f"  success={len(succ)}  fail={len(fail)}")

    T = args.early_t

    # 성공 분포 (step별 mu/std — 성공 ep들의 각 step)
    MAX = args.max_steps

    def fit_stepwise(key, dim):
        """step t마다 성공 ep들의 feature 분포를 추정."""
        mu_list, std_list = [], []
        for t in range(MAX):
            vecs = []
            for r in succ:
                if r["length"] > t:
                    vecs.append(r[key][t])
                elif r["length"] > 0:
                    vecs.append(r[key][-1])  # 마지막 step 복사
            mat = np.stack(vecs)
            mu_list.append(mat.mean(axis=0))
            std_list.append(mat.std(axis=0))
        return np.stack(mu_list), np.stack(std_list)  # [MAX, D]

    print("  fitting success distribution per step...")
    vl_mu, vl_std   = fit_stepwise("vl",  records[0]["vl"].shape[1])
    dit_mu, dit_std = fit_stepwise("dit", records[0]["dit"].shape[1])

    # 각 rollout × step anomaly score
    def step_scores(r, key, mu_list, std_list):
        arr = r[key]
        scores = []
        for t in range(MAX):
            vec = arr[t] if t < r["length"] else arr[-1]
            scores.append(mahal(vec, mu_list[t], std_list[t]))
        return np.array(scores)

    print("  computing per-step anomaly scores...")
    for r in records:
        r["vl_steps"]  = step_scores(r, "vl",  vl_mu,  vl_std)
        r["dit_steps"] = step_scores(r, "dit", dit_mu, dit_std)
        # early score (첫 T step mean) → 사분면 분류용
        r["vl_score"]  = float(r["vl_steps"][:T].mean())
        r["dit_score"] = float(r["dit_steps"][:T].mean())

    # 사분면 분류
    fail_recs = [r for r in records if r["success"] == 0]
    vl_med  = float(np.median([r["vl_score"]  for r in fail_recs]))
    dit_med = float(np.median([r["dit_score"] for r in fail_recs]))

    QUAD = {
        "A_both":  ("VL↑ DiT↑ (둘다 혼란)",   "red"),
        "B_goal":  ("VL↑ DiT↓ (goal-type)", "orange"),
        "C_motor": ("VL↓ DiT↑ (motor-type)", "blue"),
        "D_none":  ("VL↓ DiT↓ (예측불가)",   "gray"),
    }

    def get_quad(r):
        hi_vl  = r["vl_score"]  >= vl_med
        hi_dit = r["dit_score"] >= dit_med
        if hi_vl  and hi_dit:  return "A_both"
        if hi_vl  and not hi_dit: return "B_goal"
        if not hi_vl and hi_dit:  return "C_motor"
        return "D_none"

    quads = {q: [] for q in QUAD}
    for r in fail_recs:
        quads[get_quad(r)].append(r)

    steps = np.arange(MAX)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── 그림 1: VL anomaly step trajectory (사분면별 평균) ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, mu_list, label_y in [
        (axes[0], "vl_steps",  vl_mu,  f"VL anomaly"),
        (axes[1], "dit_steps", dit_mu, f"DiT-b{dit_block_actual} anomaly"),
    ]:
        # 성공 평균
        succ_traj = np.stack([r[key] for r in succ]).mean(axis=0)
        ax.plot(steps, succ_traj, color="green", linewidth=2.5, label="success (mean)")
        # 사분면별 실패 평균
        for qkey, (qlabel, qcolor) in QUAD.items():
            recs = quads[qkey]
            if not recs:
                continue
            traj = np.stack([r[key] for r in recs]).mean(axis=0)
            ax.plot(steps, traj, color=qcolor, linewidth=2, label=f"{qlabel} (n={len(recs)})")
        ax.axvline(T, color="black", linestyle=":", linewidth=1, label=f"t={T}")
        ax.set_xlabel("env step"); ax.set_ylabel(label_y)
        ax.set_title(f"{task_dir.name} — {label_y}")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "stepwise_trajectory.png", dpi=130)
    plt.close()

    # ── 그림 2: 사분면별 개별 ep step 곡선 (VL+DiT 동시, 대표 3개씩) ──
    n_rep = 3
    fig, axes = plt.subplots(len(QUAD), 2, figsize=(12, 4 * len(QUAD)))
    for qi, (qkey, (qlabel, qcolor)) in enumerate(QUAD.items()):
        recs = sorted(quads[qkey], key=lambda r: r["vl_score"], reverse=True)[:n_rep]
        for ax, key, title in [(axes[qi, 0], "vl_steps", "VL anomaly"),
                               (axes[qi, 1], "dit_steps", f"DiT-b{dit_block_actual} anomaly")]:
            # 성공 평균 (배경)
            succ_traj = np.stack([r[key] for r in succ]).mean(axis=0)
            ax.fill_between(steps, 0, succ_traj, alpha=0.15, color="green")
            ax.plot(steps, succ_traj, color="green", linewidth=1.5,
                    linestyle="--", label="success mean")
            for r in recs:
                ax.plot(steps, r[key], linewidth=1.2, alpha=0.8,
                        label=r["mp4"].replace("succ0.mp4", "").replace("task7--", "").replace("task2--", ""))
            ax.axvline(T, color="black", linestyle=":", linewidth=0.8)
            ax.set_title(f"{qlabel} — {title}", fontsize=9)
            ax.set_xlabel("step"); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "stepwise_per_quad.png", dpi=130)
    plt.close()

    print(f"[plot] {out}/stepwise_trajectory.png")
    print(f"[plot] {out}/stepwise_per_quad.png")

    # JSON 요약
    summary = {
        "task": task_dir.name, "early_t": T, "dit_block": dit_block_actual,
        "vl_median": vl_med, "dit_median": dit_med,
        "quadrant_counts": {q: len(v) for q, v in quads.items()},
        "quadrant_eps": {
            q: [{"mp4": r["mp4"], "vl": round(r["vl_score"],3),
                 "dit": round(r["dit_score"],3)} for r in v]
            for q, v in quads.items()
        },
    }
    (out / "stepwise_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] -> {out}/")


if __name__ == "__main__":
    main()

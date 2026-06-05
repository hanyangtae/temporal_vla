"""VL pathway anomaly score 분석.

각 rollout의 VL feature(action_head.vlln seq-mean-pool, [n_steps, 2048])가
성공 rollout 분포에서 얼마나 벗어났는지(Mahalanobis distance)를 계산.

목적: "VL anomaly가 높은 실패"와 "낮은 실패"를 분리해서
      goal-type failure vs motor-type failure 가설을 검증.

출력:
  - per-rollout: episode_idx, success, vl_anomaly_mean, vl_anomaly_early(t<=8), mp4 파일명
  - VL anomaly 상위 실패 ep 목록 (비디오 확인용)
  - step별 VL anomaly trajectory (성공 평균 vs 실패 상위/하위 그룹)
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


def mahalanobis_diag(x: np.ndarray, mu: np.ndarray, std: np.ndarray) -> float:
    """대각 공분산 Mahalanobis distance (std 기반, 0 std는 skip)."""
    mask = std > 1e-8
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(((x[mask] - mu[mask]) / std[mask]) ** 2) ** 0.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", required=True, help="raw_rollouts/<task> 경로")
    ap.add_argument("--out", default=None)
    ap.add_argument("--early-t", type=int, default=8, help="early window 길이")
    ap.add_argument("--dit-block", type=int, default=31,
                    help="DiT anomaly 에 사용할 block (capture_layers 중 가장 가까운 것 선택)")
    ap.add_argument("--top-n", type=int, default=8, help="각 사분면 출력 수")
    args = ap.parse_args()

    task_dir = Path(args.task_dir)
    out = Path(args.out) if args.out else \
        task_dir.parent.parent / "analysis" / "vl_dit_quadrant" / task_dir.name
    out.mkdir(parents=True, exist_ok=True)

    pkls = sorted(task_dir.glob("*.pkl"))
    print(f"[load] {len(pkls)} pkl in {task_dir.name}")

    # capture_layers 매핑 (0,2,4,8,16,24,31) → local index
    CAPTURE_LAYERS = [0, 2, 4, 8, 16, 24, 31]

    # 1. feature 로드
    records = []
    for p in pkls:
        d = load_pkl(p)
        vl = d.get("vl_hidden_states")
        hs = d.get("hidden_states")
        if vl is None or hs is None:
            continue
        vl_arr = np.stack([np.asarray(v, dtype=np.float32) for v in vl])   # [n_steps, Dvl]
        # DiT: [n_steps, L, T, D] → best block local index → action token mean → [n_steps, D]
        hs_arr = np.stack([np.asarray(h, dtype=np.float32) for h in hs])   # [n_steps, L, T, D]
        # 가장 가까운 captured layer 선택
        local_idx = int(np.argmin([abs(b - args.dit_block) for b in CAPTURE_LAYERS]))
        dit_arr = hs_arr[:, local_idx, 1:17, :].mean(axis=1)               # [n_steps, D] valid16 mean
        records.append({
            "mp4": p.with_suffix(".mp4").name,
            "success": int(d.get("episode_success", 0)),
            "length": len(vl),
            "vl": vl_arr,
            "dit": dit_arr,
        })

    if not records:
        raise SystemExit("VL feature 없음 — --capture-vl 없이 수집된 데이터")

    succ = [r for r in records if r["success"] == 1]
    fail = [r for r in records if r["success"] == 0]
    print(f"  success={len(succ)} fail={len(fail)}  "
          f"DiT block={CAPTURE_LAYERS[int(np.argmin([abs(b-args.dit_block) for b in CAPTURE_LAYERS]))]} "
          f"(local L{int(np.argmin([abs(b-args.dit_block) for b in CAPTURE_LAYERS]))})")

    T = args.early_t

    # 2. 성공 분포로 mu/std 추정 (VL, DiT 각각)
    def fit_dist(key):
        vecs = np.stack([r[key][:T].mean(axis=0) for r in succ if r["length"] >= T])
        return vecs.mean(axis=0), vecs.std(axis=0)

    vl_mu, vl_std   = fit_dist("vl")
    dit_mu, dit_std = fit_dist("dit")

    # 3. 각 rollout anomaly score
    def score(r, key, mu, std):
        arr = r[key]
        vec = arr[:T].mean(axis=0) if r["length"] >= T else arr.mean(axis=0)
        return mahalanobis_diag(vec, mu, std)

    for r in records:
        r["vl_score"]  = round(score(r, "vl",  vl_mu,  vl_std),  4)
        r["dit_score"] = round(score(r, "dit", dit_mu, dit_std), 4)

    # 4. 실패 ep 2x2 분류 (median split)
    fail_recs = [r for r in records if r["success"] == 0]
    vl_med  = float(np.median([r["vl_score"]  for r in fail_recs]))
    dit_med = float(np.median([r["dit_score"] for r in fail_recs]))

    def quadrant(r):
        hi_vl  = r["vl_score"]  >= vl_med
        hi_dit = r["dit_score"] >= dit_med
        if     hi_vl and     hi_dit: return "A_both_high"
        if     hi_vl and not hi_dit: return "B_goal_type"
        if not hi_vl and     hi_dit: return "C_motor_type"
        return                              "D_both_low"

    QUAD_LABELS = {
        "A_both_high": "A: VL↑ DiT↑  (둘 다 혼란)",
        "B_goal_type": "B: VL↑ DiT↓  (goal-type failure)",
        "C_motor_type":"C: VL↓ DiT↑  (motor-type failure)",
        "D_both_low":  "D: VL↓ DiT↓  (예측불가형)",
    }

    quads = {q: [] for q in QUAD_LABELS}
    for r in fail_recs:
        quads[quadrant(r)].append(r)

    print(f"\n  median split:  VL_med={vl_med:.3f}  DiT_med={dit_med:.3f}")
    print(f"\n{'='*60}")
    for qkey, qlabel in QUAD_LABELS.items():
        eps = sorted(quads[qkey], key=lambda x: x["vl_score"], reverse=True)
        print(f"\n=== {qlabel}  (n={len(eps)}) ===")
        for r in eps[:args.top_n]:
            print(f"  {r['mp4']}  vl={r['vl_score']:.3f}  dit={r['dit_score']:.3f}  len={r['length']}")

    # 5. 산점도: 실패(●)/성공(○) VL-score vs DiT-score
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COLORS = {"A_both_high": "red", "B_goal_type": "orange",
              "C_motor_type": "blue", "D_both_low": "gray"}

    fig, ax = plt.subplots(figsize=(7, 6))
    # 성공
    ax.scatter([r["vl_score"] for r in succ],
               [r["dit_score"] for r in succ],
               c="green", alpha=0.3, s=30, label="success", marker="o")
    # 실패 4 그룹
    for qkey, qlabel in QUAD_LABELS.items():
        eps = quads[qkey]
        if eps:
            ax.scatter([r["vl_score"] for r in eps],
                       [r["dit_score"] for r in eps],
                       c=COLORS[qkey], alpha=0.7, s=50,
                       label=f"{qlabel} (n={len(eps)})", marker="x")
    ax.axvline(vl_med,  color="orange", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(dit_med, color="blue",   linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel(f"VL anomaly (early t={T})"); ax.set_ylabel(f"DiT-b{args.dit_block} anomaly")
    ax.set_title(f"Failure quadrant — {task_dir.name}")
    ax.legend(fontsize=8, loc="upper left"); fig.tight_layout()
    fig.savefig(out / "quadrant_scatter.png", dpi=130)
    plt.close()
    print(f"\n[plot] {out/'quadrant_scatter.png'}")

    # 6. JSON
    out_data = {
        "task": task_dir.name, "early_t": T,
        "dit_block": CAPTURE_LAYERS[int(np.argmin([abs(b-args.dit_block) for b in CAPTURE_LAYERS]))],
        "vl_median": vl_med, "dit_median": dit_med,
        "n_succ": len(succ), "n_fail": len(fail_recs),
        "quadrants": {q: [{"mp4":r["mp4"],"vl":r["vl_score"],"dit":r["dit_score"],"len":r["length"]}
                          for r in eps] for q, eps in quads.items()},
    }
    (out / "quadrant.json").write_text(json.dumps(out_data, indent=2))
    print(f"[done] -> {out/'quadrant.json'}")


if __name__ == "__main__":
    main()

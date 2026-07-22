#!/usr/bin/env python3
"""cam-attention 수집 pkl → per-record 뷰별 attention mass CSV + phase 집계 JSON.

입력: collect_cam_attn.sh 산출 pkl 트리 (cross_attn [N, n_blocks, K, qgroup, kgroup]).
핵심 지표 = **step별 wrist attention mass** (action-query 그룹, cross-block·denoise mean)
과 wrist 의 vision-mass 내 share (uniform 기대 = 1/3 — 뷰당 256 token 동수라 정확).

출력:
  <out>/cam_attn_records.csv   — record(=inference, 5 env-step) 단위 long-format
  <out>/cam_attn_phase_agg.json — (cell × phase × succ) wrist share mean±bootstrap CI

caveat: DiT 앞의 vl_self_attention 이 VL 시퀀스를 한 번 섞음 — 위치 기반 귀인은
유효하나 내용 혼입 가능. 관측 지표로만 해석 (인과 아님).
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

KGROUPS = ("text", "left", "right", "wrist")
QGROUPS = ("state", "future", "action")


def load_episode(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    ca = np.asarray(d["cross_attn"], dtype=np.float32)  # [N, L, K, 3, 4]
    if d.get("cross_attn_kgroups") != list(KGROUPS):
        raise RuntimeError(f"{pkl_path}: unexpected kgroups {d.get('cross_attn_kgroups')}")
    if d.get("cross_attn_qgroups") != list(QGROUPS):
        raise RuntimeError(f"{pkl_path}: unexpected qgroups {d.get('cross_attn_qgroups')}")
    fp = list(d.get("feature_phases") or [])
    if len(fp) != ca.shape[0]:
        raise RuntimeError(
            f"{pkl_path}: feature_phases {len(fp)} != records {ca.shape[0]} (qa mismatch)"
        )
    nas = int(d.get("n_action_steps") or 5)
    esp = d.get("env_step_phases")
    # env_step_phases 는 steps+1 (조기 성공 종료 시 steps < N*nas 일 수 있음).
    if esp is not None and len(esp) > ca.shape[0] * nas + 1:
        raise RuntimeError(
            f"{pkl_path}: env_step_phases {len(esp)} > records*nas+1 ({ca.shape[0] * nas + 1})"
        )
    return {
        "cross_attn": ca,
        "feature_phases": fp,
        "n_action_steps": nas,
        "success": int(d.get("episode_success", 0)),
        "cell_id": d.get("cell_id"),
        "task": d.get("robocasa_task") or d.get("task_description"),
        "episode_idx": int(d.get("episode_idx", -1)),
        "blocks": list(d.get("cross_attn_blocks") or []),
    }


def episode_rows(ep: dict, qgroup: str) -> list[dict]:
    qi = QGROUPS.index(qgroup)
    ca = ep["cross_attn"]  # [N, L, K, 3, 4]
    mass = ca[:, :, :, qi, :].mean(axis=(1, 2))  # [N, 4] block/denoise mean
    vision = mass[:, 1:].sum(axis=1)  # left+right+wrist
    rows = []
    for r in range(mass.shape[0]):
        rows.append(
            {
                "cell_id": ep["cell_id"],
                "task": ep["task"],
                "episode_idx": ep["episode_idx"],
                "success": ep["success"],
                "record_idx": r,
                "env_step": r * ep["n_action_steps"],
                "phase": ep["feature_phases"][r],
                "mass_text": float(mass[r, 0]),
                "mass_left": float(mass[r, 1]),
                "mass_right": float(mass[r, 2]),
                "mass_wrist": float(mass[r, 3]),
                "wrist_share_vision": float(mass[r, 3] / vision[r]) if vision[r] > 0 else float("nan"),
            }
        )
    return rows


def bootstrap_ci(ep_means: list[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """에피소드 단위 평균의 percentile bootstrap 95% CI (record 는 에피소드 내 상관)."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(ep_means, dtype=np.float64)
    if len(arr) < 2:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boots = arr[idx].mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rollout-root",
        default="outputs/eval/robocasa/groot_n15/cam_attn/raw_rollouts",
    )
    parser.add_argument("--out-dir", default="outputs/eval/robocasa/groot_n15/cam_attn/analysis")
    parser.add_argument("--qgroup", choices=QGROUPS, default="action")
    args = parser.parse_args()

    root = Path(args.rollout_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pkls = sorted(root.glob("*/*/task*--ep*--succ*.pkl"))
    if not pkls:
        raise SystemExit(f"no pkl found under {root}")
    print(f"[cam_attn_phase] {len(pkls)} episodes")

    all_rows: list[dict] = []
    for p in pkls:
        ep = load_episode(p)
        all_rows.extend(episode_rows(ep, args.qgroup))

    csv_path = out_dir / "cam_attn_records.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[cam_attn_phase] wrote {csv_path} ({len(all_rows)} rows)")

    # (cell × phase × succ) 집계: 에피소드별 phase-mean → across-episode mean + CI
    per_ep_phase: dict[tuple, dict[tuple, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in all_rows:
        key = (row["cell_id"], row["phase"], row["success"])
        per_ep_phase[key][(row["episode_idx"],)].append(row["wrist_share_vision"])
    agg: dict[str, dict] = {}
    for (cell, phase, succ), ep_map in sorted(per_ep_phase.items()):
        ep_means = [float(np.nanmean(v)) for v in ep_map.values()]
        lo, hi = bootstrap_ci(ep_means)
        agg[f"{cell}|{phase}|succ{succ}"] = {
            "n_episodes": len(ep_means),
            "n_records": int(sum(len(v) for v in ep_map.values())),
            "wrist_share_vision_mean": float(np.mean(ep_means)),
            "ci95": [lo, hi],
        }
    json_path = out_dir / "cam_attn_phase_agg.json"
    json_path.write_text(json.dumps(agg, indent=2, ensure_ascii=False))
    print(f"[cam_attn_phase] wrote {json_path}")
    print(f"{'cell|phase|succ':<48} {'n_ep':>4} {'wrist_share':>11} {'CI95':>20}  (uniform=0.333)")
    for k, v in agg.items():
        print(
            f"{k:<48} {v['n_episodes']:>4} {v['wrist_share_vision_mean']:>11.3f} "
            f"[{v['ci95'][0]:.3f},{v['ci95'][1]:.3f}]"
        )


if __name__ == "__main__":
    main()

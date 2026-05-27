#!/usr/bin/env python3
"""Pool every rollout's per-step hidden states once and cache to a single npz.

Reading the 1800 raw pkls (~10 GB) is the dominant cost for every downstream
embedding plot. This script does it once for a chosen aggregation and writes a
flat per-timestep cache plus a per-rollout (episode-mean) cache, so the
visualization/clustering/conceptor tools can load instantly.

Per-timestep arrays (one row per env step, concatenated across rollouts):
    feats        [N, D]    pooled feature
    success      [N]       episode-level success label (0/1), broadcast
    task_id      [N]
    rollout_idx  [N]       0..R-1 index into the manifest order
    episode_idx  [N]
    step_idx     [N]       timestep within the rollout
    progress     [N]       step_idx / (T-1) in [0, 1]

Per-rollout arrays (one row per rollout):
    ep_feat_mean [R, D]    mean over time
    ep_feat_last [R, D]    last timestep
    ep_success   [R], ep_task_id [R], ep_episode_idx [R], ep_len [R]
    task_names   [R]       (object array)
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROBOCASA_SAFE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROBOCASA_SAFE_ROOT))

from safe_feature_vectors import (  # noqa: E402
    load_manifest,
    parse_aggregation_command,
    pooled_hidden_states,
)


def parse_args() -> argparse.Namespace:
    run_root = (
        Path(__file__).resolve().parents[5]
        / "outputs/eval/robocasa/groot_n16"
        / "target_atomic_seen18_ckpt120000_robocasa365_100ep"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-root", type=Path, default=run_root / "analysis" / "split")
    p.add_argument("--out-root", type=Path, default=run_root / "analysis" / "feature_cache")
    p.add_argument("--scope", default="all")
    p.add_argument("--horizon-idx-rel", default="mean")
    p.add_argument("--diff-idx-rel", default="mean")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def slug(value: str) -> str:
    return value.replace(".", "p").replace("-", "_").replace(":", "_")


def main() -> None:
    args = parse_args()
    h_cmd = parse_aggregation_command(args.horizon_idx_rel)
    d_cmd = parse_aggregation_command(args.diff_idx_rel)

    rows = load_manifest(args.split_root, args.scope)
    print(f"manifest rows: {len(rows)}")

    agg = f"h{slug(args.horizon_idx_rel)}_d{slug(args.diff_idx_rel)}"
    out_dir = args.out_root
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pooled_{args.scope}_{agg}.npz"
    if out_path.exists() and not args.force:
        print(f"cache exists (use --force to overwrite): {out_path}")
        return

    feats, success, task_id, rollout_idx = [], [], [], []
    episode_idx, step_idx, progress = [], [], []
    ep_feat_mean, ep_feat_last = [], []
    ep_success, ep_task_id, ep_episode_idx, ep_len, task_names = [], [], [], [], []

    t0 = time.time()
    for i, row in enumerate(rows):
        with Path(row["source_path"]).open("rb") as f:
            rec = pickle.load(f)
        feat = pooled_hidden_states(rec, horizon_idx_rel=h_cmd, diff_idx_rel=d_cmd)  # [T, D]
        T = feat.shape[0]
        if T == 0:
            continue
        succ = int(row["success"])
        tid = int(row["task_id"])
        eidx = int(row["episode_idx"])

        feats.append(feat)
        success.append(np.full(T, succ, np.int64))
        task_id.append(np.full(T, tid, np.int64))
        rollout_idx.append(np.full(T, i, np.int64))
        episode_idx.append(np.full(T, eidx, np.int64))
        step_idx.append(np.arange(T, dtype=np.int64))
        progress.append((np.arange(T, dtype=np.float32) / max(T - 1, 1)))

        ep_feat_mean.append(feat.mean(axis=0))
        ep_feat_last.append(feat[-1])
        ep_success.append(succ)
        ep_task_id.append(tid)
        ep_episode_idx.append(eidx)
        ep_len.append(T)
        task_names.append(row["task"])

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(rows)}  ({time.time() - t0:.0f}s)")

    feats = np.concatenate(feats).astype(np.float32)
    print(f"per-timestep feats: {feats.shape} {feats.dtype}  ({time.time() - t0:.0f}s)")

    np.savez(
        out_path,
        feats=feats,
        success=np.concatenate(success),
        task_id=np.concatenate(task_id),
        rollout_idx=np.concatenate(rollout_idx),
        episode_idx=np.concatenate(episode_idx),
        step_idx=np.concatenate(step_idx),
        progress=np.concatenate(progress),
        ep_feat_mean=np.stack(ep_feat_mean).astype(np.float32),
        ep_feat_last=np.stack(ep_feat_last).astype(np.float32),
        ep_success=np.asarray(ep_success, np.int64),
        ep_task_id=np.asarray(ep_task_id, np.int64),
        ep_episode_idx=np.asarray(ep_episode_idx, np.int64),
        ep_len=np.asarray(ep_len, np.int64),
        task_names=np.asarray(task_names, dtype=object),
        horizon_idx_rel=args.horizon_idx_rel,
        diff_idx_rel=args.diff_idx_rel,
    )
    print(f"wrote {out_path}  ({out_path.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()

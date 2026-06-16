"""cache — build the pooled-feature npz from the manifest (per-timestep + per-rollout).

Reads manifest.tsv rollouts, pools each pkl's hidden states (horizon/diff agg) to
[T, D], and writes the npz consumed by all other subcommands.
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np

# safe_feature_vectors lives in the robocasa/ dir (parents[2] of this file)
_ROBOCASA = Path(__file__).resolve().parents[2]
if str(_ROBOCASA) not in sys.path:
    sys.path.insert(0, str(_ROBOCASA))
from safe_feature_vectors import (  # noqa: E402
    load_manifest,
    parse_aggregation_command,
    pooled_hidden_states,
    tensor_to_numpy,
)

from analyses._common import RUN_ROOT

NAME = "cache"
HELP = "build pooled-feature npz from manifest (run once before other subcommands)"


def add_args(p):
    p.add_argument("--split-root", type=Path, default=RUN_ROOT / "analysis" / "split")
    p.add_argument("--out-root", type=Path, default=RUN_ROOT / "analysis" / "feature_cache")
    p.add_argument("--scope", default="all")
    p.add_argument("--horizon-idx-rel", default="mean")
    p.add_argument("--diff-idx-rel", default="mean")
    p.add_argument(
        "--pathway",
        default="dit",
        choices=["dit", "vl"],
        help=(
            "dit=DiT motor hidden_states [K,H,D] (default, horizon/diff pooled). "
            "vl=goal VL vl_hidden_states (already seq-mean-pooled [D]; horizon/diff ignored)."
        ),
    )
    p.add_argument("--force", action="store_true")
    p.add_argument("--seed", type=int, default=0)  # unused; for CLI uniformity


def _slug(v):
    return v.replace(".", "p").replace("-", "_").replace(":", "_")


def _pool_vl(record) -> np.ndarray:
    """Stack the pre-pooled VL per-step vectors (vl_hidden_states) to [T, D]."""
    vl = record.get("vl_hidden_states") or []
    feats = [tensor_to_numpy(h).astype(np.float32, copy=False).reshape(-1) for h in vl]
    if not feats:
        return np.empty((0, 0), dtype=np.float32)
    return np.stack(feats, axis=0)


def run(args):
    h_cmd = parse_aggregation_command(args.horizon_idx_rel)
    d_cmd = parse_aggregation_command(args.diff_idx_rel)
    rows = load_manifest(args.split_root, args.scope)
    print(f"manifest rows: {len(rows)}")
    agg = f"h{_slug(args.horizon_idx_rel)}_d{_slug(args.diff_idx_rel)}"
    args.out_root.mkdir(parents=True, exist_ok=True)
    # Keep the DiT (default) filename backward-compatible; prefix only for non-dit pathways.
    pref = "" if args.pathway == "dit" else f"{args.pathway}_"
    out_path = args.out_root / f"pooled_{pref}{args.scope}_{agg}.npz"
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
        if args.pathway == "vl":
            feat = _pool_vl(rec)
        else:
            feat = pooled_hidden_states(rec, horizon_idx_rel=h_cmd, diff_idx_rel=d_cmd)
        T = feat.shape[0]
        if T == 0:
            continue
        succ, tid, eidx = int(row["success"]), int(row["task_id"]), int(row["episode_idx"])
        feats.append(feat)
        success.append(np.full(T, succ, np.int64))
        task_id.append(np.full(T, tid, np.int64))
        rollout_idx.append(np.full(T, i, np.int64))
        episode_idx.append(np.full(T, eidx, np.int64))
        step_idx.append(np.arange(T, dtype=np.int64))
        progress.append(np.arange(T, dtype=np.float32) / max(T - 1, 1))
        ep_feat_mean.append(feat.mean(axis=0)); ep_feat_last.append(feat[-1])
        ep_success.append(succ); ep_task_id.append(tid); ep_episode_idx.append(eidx)
        ep_len.append(T); task_names.append(row["task"])
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(rows)}  ({time.time() - t0:.0f}s)")

    feats = np.concatenate(feats).astype(np.float32)
    print(f"per-timestep feats: {feats.shape} {feats.dtype}  ({time.time() - t0:.0f}s)")
    np.savez(out_path, feats=feats, success=np.concatenate(success), task_id=np.concatenate(task_id),
             rollout_idx=np.concatenate(rollout_idx), episode_idx=np.concatenate(episode_idx),
             step_idx=np.concatenate(step_idx), progress=np.concatenate(progress),
             ep_feat_mean=np.stack(ep_feat_mean).astype(np.float32),
             ep_feat_last=np.stack(ep_feat_last).astype(np.float32),
             ep_success=np.asarray(ep_success, np.int64), ep_task_id=np.asarray(ep_task_id, np.int64),
             ep_episode_idx=np.asarray(ep_episode_idx, np.int64), ep_len=np.asarray(ep_len, np.int64),
             task_names=np.asarray(task_names, dtype=object),
             horizon_idx_rel=args.horizon_idx_rel, diff_idx_rel=args.diff_idx_rel)
    print(f"wrote {out_path}  ({out_path.stat().st_size / 1e6:.0f} MB)")

#!/usr/bin/env python3
"""Run SAFE feature visualization for GR00T N1.6 RoboCasa outputs.

This is a thin adapter around SAFE's ``scripts/visualize_features.py``.  The
upstream script writes to ``./notebooks/feat_vis`` relative to the current
working directory; this adapter pins the final artifact location under the
GR00T N1.6 evaluation output tree and writes a small manifest beside the plots.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import runpy
import shutil
import sys
import types
from typing import Any


ROBOCASA_SAFE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROBOCASA_SAFE_ROOT))

from run_config import (  # noqa: E402
    EXPERIMENT_ID,
    FEATURE_SPACE_ROOT,
    FINAL_DIFF_IDX_REL,
    FINAL_HORIZON_IDX_REL,
    SAFE_ROOT,
    SAFE_SUBSET_NAME,
    SPLIT_ROOT,
)


DEFAULT_SPLIT_ROOT = SPLIT_ROOT
DEFAULT_OUT_ROOT = FEATURE_SPACE_ROOT
DEFAULT_SUBSET_NAME = SAFE_SUBSET_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAFE t-SNE/UMAP feature visualization for GR00T N1.6 RoboCasa."
    )
    parser.add_argument(
        "--scope",
        default="val_unseen",
        choices=("all", "train", "val_seen", "val_unseen"),
        help="Which physical split to visualize. Use 'all' for the split root.",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Optional task directory inside the selected split, for example OpenDrawer.",
    )
    parser.add_argument("--projector", default="tsne")
    parser.add_argument("--horizon-idx-rel", default=FINAL_HORIZON_IDX_REL)
    parser.add_argument("--diff-idx-rel", default=FINAL_DIFF_IDX_REL)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--safe-root", type=Path, default=SAFE_ROOT)
    parser.add_argument("--subset-name", default=DEFAULT_SUBSET_NAME)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_data_path(split_root: Path, scope: str, task: str | None) -> Path:
    if scope == "all":
        if task is not None:
            raise ValueError("--task cannot be used with --scope all")
        data_path = split_root
    else:
        data_path = split_root / scope
        if task:
            data_path = data_path / task
    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    return data_path


def scope_name(scope: str, task: str | None) -> str:
    return scope if task is None else f"{scope}_{task}"


def install_umap_stub_if_needed(projector: str) -> None:
    if projector == "umap":
        return
    try:
        __import__("umap")
    except ModuleNotFoundError:
        umap_mod = types.ModuleType("umap")

        class _UnavailableUMAP:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                raise RuntimeError("umap is not installed; use another projector or install umap-learn")

        umap_mod.UMAP = _UnavailableUMAP
        sys.modules["umap"] = umap_mod


def count_rollouts(data_path: Path) -> int:
    return sum(1 for _ in data_path.rglob("*.pkl"))


def read_feature_count(artifact_dir: Path) -> int | None:
    pkl_path = artifact_dir / "feats_projected_skip1.pkl"
    if not pkl_path.exists():
        return None
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    feats = data.get("feats_projected")
    return int(feats.shape[0]) if hasattr(feats, "shape") else None


def run_safe_visualizer(args: argparse.Namespace, data_path: Path, name: str) -> Path:
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    install_umap_stub_if_needed(args.projector)

    work_root = args.out_root / ".visualize_features_work"
    work_root.mkdir(parents=True, exist_ok=True)

    suffix = f"groot_n16_{name}_{args.horizon_idx_rel}_{args.diff_idx_rel}_{args.projector}"
    generated_folder = f"groot_n16-{args.subset_name}-{suffix}-{args.projector}"
    generated_dir = work_root / "notebooks/feat_vis" / generated_folder

    old_cwd = Path.cwd()
    old_argv = sys.argv[:]
    try:
        os.chdir(work_root)
        sys.argv = [
            str(args.safe_root / "scripts/visualize_features.py"),
            "dataset=groot_n16",
            f"dataset.data_path={data_path}",
            f"dataset.horizon_idx_rel={args.horizon_idx_rel}",
            f"dataset.diff_idx_rel={args.diff_idx_rel}",
            "dataset.load_to_cuda=False",
            f"projector={args.projector}",
            f"save_video={str(args.save_video)}",
            f"train.exp_suffix={suffix}",
        ]
        runpy.run_path(str(args.safe_root / "scripts/visualize_features.py"), run_name="__main__")
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)

    if not generated_dir.exists():
        raise FileNotFoundError(f"SAFE visualizer did not produce expected output: {generated_dir}")
    return generated_dir


def write_manifest(
    artifact_dir: Path,
    args: argparse.Namespace,
    data_path: Path,
    name: str,
    rollout_count: int,
    feature_count: int | None,
) -> None:
    manifest = {
        "artifact_type": "safe_feature_visualization",
        "dataset": "groot_n16",
        "experiment_id": args.experiment_id,
        "source_data_path": str(data_path),
        "scope": args.scope,
        "task": args.task,
        "scope_name": name,
        "projector": args.projector,
        "horizon_idx_rel": args.horizon_idx_rel,
        "diff_idx_rel": args.diff_idx_rel,
        "rollout_count": rollout_count,
        "feature_count": feature_count,
        "safe_script": str(args.safe_root / "scripts/visualize_features.py"),
        "generated_by": str(Path(__file__).relative_to(REPO_ROOT)),
        "outputs": sorted(p.name for p in artifact_dir.iterdir() if p.is_file()),
    }
    with (artifact_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> None:
    args = parse_args()
    data_path = resolve_data_path(args.split_root, args.scope, args.task)
    name = scope_name(args.scope, args.task)

    target_dir = (
        args.out_root
        / args.experiment_id
        / name
        / f"{args.projector}_{args.horizon_idx_rel}_{args.diff_idx_rel}"
    )
    if target_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Target already exists; pass --overwrite to replace: {target_dir}")
        shutil.rmtree(target_dir)

    generated_dir = run_safe_visualizer(args, data_path, name)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(generated_dir), str(target_dir))

    write_manifest(
        artifact_dir=target_dir,
        args=args,
        data_path=data_path,
        name=name,
        rollout_count=count_rollouts(data_path),
        feature_count=read_feature_count(target_dir),
    )

    print(target_dir)


if __name__ == "__main__":
    main()

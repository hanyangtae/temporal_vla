"""End-to-end post-processing for downloaded RoboCasa LeRobot datasets.

For each `lerobot/` dataset directory found, this script runs:

  1. v2.1 → v3.0 conversion (LeRobot built-in, called as subprocess against
     the submodule script to avoid the pip-installed lerobot's
     `convert_dataset` quirk that appends `repo_id` to `--root`).
     - original `lerobot/` → `lerobot_old/` (backup)
     - `lerobot/` becomes v3.0
  2. Add per-frame `progress` feature (rounded to 2 decimals). The result is
     written by `add_features()` to a sibling `lerobot_with_progress/` dir,
     then swapped in place of `lerobot/` so the final layout is:
         <base>/lerobot/      ← v3.0 + progress (final)
         <base>/lerobot_old/  ← v2.1 backup
  3. Copy `embodiment.json` and `modality.json` from `lerobot_old/meta/` to
     `lerobot/meta/` (LeRobot converter and `add_features()` do not migrate
     RoboCasa-specific meta).

Assumes step 0 (download) was done separately via robocasa container's
`download_datasets.py`. Run this in the lerobot container.

Default behavior: auto-discover all RoboCasa atomic tasks under
`/temporal_vla/src/benchmarks/robocasa/datasets/v1.0/<split>/atomic/*/*/lerobot`.

Usage:
    # Auto-discover all downloaded atomic tasks (default split=pretrain):
    python scripts/extract/prepare_robocasa_dataset.py

    # Filter by task name(s):
    python scripts/extract/prepare_robocasa_dataset.py \
        --tasks PickPlaceCounterToSink PickPlaceSinkToCounter

    # Use a different split:
    python scripts/extract/prepare_robocasa_dataset.py --split target
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from lerobot.datasets.dataset_tools import add_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset

META_FILES_TO_MIGRATE = ["embodiment.json", "modality.json"]
SUBMODULE_CONVERT_SCRIPT = (
    "/temporal_vla/lerobot/src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py"
)
DEFAULT_BASE_TPL = "/temporal_vla/src/benchmarks/robocasa/datasets/v1.0/{split}/atomic"


def discover_dataset_roots(split: str, tasks: list[str] | None = None) -> list[Path]:
    """Find every `<base>/<task>/<date>/lerobot` directory under the atomic base.

    If `tasks` is given, restrict to those task names.
    """
    base = Path(DEFAULT_BASE_TPL.format(split=split))
    if not base.is_dir():
        raise FileNotFoundError(f"Atomic base not found: {base}")

    if tasks:
        candidates: list[Path] = []
        for task in tasks:
            task_dir = base / task
            matches = sorted(task_dir.glob("*/lerobot"))
            if not matches:
                raise FileNotFoundError(
                    f"No dataset for task '{task}' under {task_dir}/. "
                    f"Run download_datasets.py first."
                )
            candidates.extend(matches)
        return candidates

    return sorted(base.glob("*/*/lerobot"))


def derive_repo_id(root: Path) -> str:
    """Infer a repo_id label from the dataset path.

    For RoboCasa datasets at `.../atomic/<TaskName>/<date>/lerobot`,
    returns `robocasa/<TaskName>`.
    """
    parts = root.parts
    if "atomic" in parts:
        idx = parts.index("atomic")
        if idx + 1 < len(parts):
            return f"robocasa/{parts[idx + 1]}"
    return f"robocasa/{root.parent.parent.name}"


def codebase_version(root: Path) -> str | None:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        return None
    return json.loads(info_path.read_text()).get("codebase_version")


def has_progress_feature(root: Path) -> bool:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        return False
    return "progress" in json.loads(info_path.read_text()).get("features", {})


def step_convert_v21_to_v30(root: Path, repo_id: str) -> None:
    if codebase_version(root) == "v3.0":
        print(f"  [skip convert] {root.name} is already v3.0")
        return
    print(f"  [convert] {root.name} → v3.0 (backup at {root.name}_old)")
    subprocess.run(
        [
            "python",
            SUBMODULE_CONVERT_SCRIPT,
            "--repo-id",
            repo_id,
            "--root",
            str(root),
            "--push-to-hub=false",
        ],
        check=True,
    )


def step_copy_robocasa_meta(root: Path) -> None:
    src_dir = root.parent / f"{root.name}_old" / "meta"
    if not src_dir.exists():
        print(f"  [skip meta copy] no backup dir {src_dir}")
        return
    for fname in META_FILES_TO_MIGRATE:
        src, dst = src_dir / fname, root / "meta" / fname
        if dst.exists():
            print(f"  [skip meta copy] {fname} already in v3.0 meta")
            continue
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  [copy meta] {fname}")


def compute_progress(ds: LeRobotDataset) -> np.ndarray:
    """Return per-frame linear progress rounded to 2 decimals."""
    ep_lengths = {
        int(e): int(length)
        for e, length in zip(
            ds.meta.episodes["episode_index"], ds.meta.episodes["length"]
        )
    }
    data = ds.hf_dataset.with_format(None)
    frame_idx = np.asarray(data["frame_index"], dtype=np.int64)
    ep_idx = np.asarray(data["episode_index"], dtype=np.int64)
    lengths = np.asarray([ep_lengths[int(e)] for e in ep_idx], dtype=np.int64)
    raw = frame_idx / np.maximum(lengths - 1, 1)
    return np.round(raw, 2).astype(np.float32)


def step_add_progress(root: Path, repo_id: str) -> None:
    if has_progress_feature(root):
        print(f"  [skip add_progress] {root.name} already has 'progress' feature")
        return

    tmp_dir = root.parent / f"{root.name}_with_progress"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    ds = LeRobotDataset(repo_id=repo_id, root=str(root))
    progress = compute_progress(ds)
    print(
        f"  [add_progress] {ds.num_frames} frames, "
        f"min={progress.min():.2f}, max={progress.max():.2f}, "
        f"unique={len(np.unique(progress))}"
    )

    add_features(
        ds,
        features={
            "progress": (
                progress,
                # shape=(1,) tuple ⇒ HF Value (scalar). list [1] would map to
                # Sequence and require per-row arrays.
                {"dtype": "float32", "shape": (1,), "names": None},
            ),
        },
        output_dir=str(tmp_dir),
        repo_id=f"{repo_id}_with_progress",
    )

    # Swap: discard intermediate v3.0 (no progress), promote tmp to canonical.
    shutil.rmtree(root)
    shutil.move(str(tmp_dir), str(root))
    print(f"  [add_progress] swapped → {root}")


def process(root: Path) -> None:
    if not root.exists():
        raise FileNotFoundError(root)
    repo_id = derive_repo_id(root)
    print(f"\n=== {repo_id} ===")
    print(f"  root: {root}")
    step_convert_v21_to_v30(root, repo_id)
    # add_features() rewrites the dataset and only knows about LeRobot-standard
    # files, so RoboCasa-specific meta must be copied AFTER the swap.
    step_add_progress(root, repo_id)
    step_copy_robocasa_meta(root)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--split",
        default="pretrain",
        choices=["pretrain", "target", "real"],
        help="RoboCasa dataset split to scan. Default: pretrain.",
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Restrict to specific task names. Default: all tasks under the split.",
    )
    args = p.parse_args()

    roots = discover_dataset_roots(args.split, args.tasks)

    if not roots:
        print("No datasets found.")
        return

    print(f"Found {len(roots)} dataset(s) to process:")
    for r in roots:
        print(f"  - {r}")

    for root in roots:
        process(root)

    print("\nDone.")


if __name__ == "__main__":
    main()

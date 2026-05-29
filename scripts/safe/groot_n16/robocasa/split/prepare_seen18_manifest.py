#!/usr/bin/env python3
"""Build a flat manifest.tsv for the seen18 GR00T N1.6 RoboCasa rollout run.

Unlike ``prepare_seen4_unseen2_split.py`` (which hardcodes 6 tasks and builds a
train/val_seen/val_unseen symlink tree for SAFE detector training), this script
just enumerates every rollout under ``raw_rollouts/`` and writes a single
manifest the visualization/clustering tools can consume directly.

The manifest schema matches ``safe_feature_vectors.load_manifest``:

    split  task_id  task  category  episode_idx  success  source_path

``split`` is set to ``all`` for every row (no detector split is needed for
embedding visualization); pass ``--scope all`` to the downstream tools.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ROLLOUT_RE = re.compile(r"task(?P<task_id>\d+)--ep(?P<episode_idx>\d+)--succ(?P<success>[01])\.pkl$")

DEFAULT_RUN_ROOT = (
    Path(__file__).resolve().parents[5]
    / "outputs/eval/robocasa/groot_n16"
    / "target_atomic_seen18_ckpt120000_robocasa365_100ep"
)


def category_of(task_name: str) -> str:
    name = task_name.lower()
    if name.startswith("pickplace") or name.startswith("pnp"):
        return "pnp"
    if name.startswith("open"):
        return "open"
    if name.startswith("close"):
        return "close"
    if name.startswith("turn"):
        return "turn"
    if name.startswith("navigate"):
        return "navigate"
    if name.startswith("slide"):
        return "slide"
    return "other"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Defaults to <run-root>/raw_rollouts.",
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=None,
        help="Destination directory for manifest.tsv. Defaults to <run-root>/analysis/split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root or (args.run_root / "raw_rollouts")
    split_root = args.split_root or (args.run_root / "analysis" / "split")
    if not source_root.is_dir():
        raise FileNotFoundError(f"Missing source root: {source_root}")

    task_dirs = sorted(p for p in source_root.iterdir() if p.is_dir())
    rows: list[tuple] = []
    for task_dir in task_dirs:
        task_name = task_dir.name
        category = category_of(task_name)
        pkls = sorted(task_dir.glob("task*--ep*--succ*.pkl"))
        for pkl in pkls:
            m = ROLLOUT_RE.match(pkl.name)
            if m is None:
                raise ValueError(f"Unexpected rollout filename: {pkl}")
            rows.append(
                (
                    int(m.group("task_id")),
                    task_name,
                    category,
                    int(m.group("episode_idx")),
                    int(m.group("success")),
                    str(pkl.resolve()),
                )
            )

    rows.sort(key=lambda r: (r[0], r[3]))
    split_root.mkdir(parents=True, exist_ok=True)
    manifest_path = split_root / "manifest.tsv"
    with manifest_path.open("w") as f:
        f.write("split\ttask_id\ttask\tcategory\tepisode_idx\tsuccess\tsource_path\n")
        for task_id, task_name, category, episode_idx, success, source_path in rows:
            f.write(
                f"all\t{task_id}\t{task_name}\t{category}\t{episode_idx}\t{success}\t{source_path}\n"
            )

    n_succ = sum(1 for r in rows if r[4] == 1)
    n_tasks = len({r[0] for r in rows})
    print(f"wrote {manifest_path}")
    print(f"  rollouts={len(rows)}  tasks={n_tasks}  success={n_succ}  fail={len(rows) - n_succ}")


if __name__ == "__main__":
    main()

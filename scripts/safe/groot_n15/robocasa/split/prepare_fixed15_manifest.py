#!/usr/bin/env python3
"""Build a flat manifest.tsv for a GR00T N1.5 fixed-instruction RoboCasa rollout run.

Mirrors ``groot_n16/.../split/prepare_seen18_manifest.py`` but handles the N1.5
collection's extra nesting: pkls live at

    raw_rollouts/<Task>/<cell>/task<id>--ep<n>--succ<s>.pkl

where each ``<cell>`` is one fixed-instruction/object variant and carries a unique
``task_id`` (the COAST per-task granularity). The manifest schema matches
``safe_feature_vectors.load_manifest`` so the existing cache / fit tools consume it
directly via ``--split-root``:

    split  task_id  task  category  episode_idx  success  source_path

``task`` is set to the cell name (distinct per task_id); ``category`` is derived from
the parent Task dir. ``split`` defaults to ``all``; pass ``--scope all`` downstream.
Per-cell holdout splits (COAST 15-train/rest-test) can be layered on later.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROLLOUT_RE = re.compile(r"task(?P<task_id>\d+)--ep(?P<episode_idx>\d+)--succ(?P<success>[01])\.pkl$")


def category_of(task_name: str) -> str:
    name = task_name.lower()
    if name.startswith("pickplace") or name.startswith("pnp") or name.startswith("pp"):
        return "pnp"
    if name.startswith("open"):
        return "open"
    if name.startswith("close"):
        return "close"
    if name.startswith("turn"):
        return "turn"
    if name.startswith("navigate"):
        return "navigate"
    if name.startswith("coffee"):
        return "coffee"
    return "other"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="<run-root>/raw_rollouts of the N1.5 collection.",
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=None,
        help="Destination for manifest.tsv. Defaults to <source-root>/../analysis/split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root
    split_root = args.split_root or (source_root.parent / "analysis" / "split")
    if not source_root.is_dir():
        raise FileNotFoundError(f"Missing source root: {source_root}")

    rows: list[tuple] = []
    for pkl in sorted(source_root.rglob("task*--ep*--succ*.pkl")):
        m = ROLLOUT_RE.match(pkl.name)
        if m is None:
            raise ValueError(f"Unexpected rollout filename: {pkl}")
        cell = pkl.parent.name           # fixed-instruction unit (distinct per task_id)
        task_dir = pkl.parent.parent.name  # robocasa Task
        rows.append(
            (
                int(m.group("task_id")),
                cell,
                category_of(task_dir),
                int(m.group("episode_idx")),
                int(m.group("success")),
                str(pkl.resolve()),
            )
        )

    if not rows:
        raise ValueError(f"No rollout pkls found under {source_root}")

    rows.sort(key=lambda r: (r[0], r[3]))
    split_root.mkdir(parents=True, exist_ok=True)
    manifest_path = split_root / "manifest.tsv"
    with manifest_path.open("w") as f:
        f.write("split\ttask_id\ttask\tcategory\tepisode_idx\tsuccess\tsource_path\n")
        for task_id, cell, category, episode_idx, success, source_path in rows:
            f.write(f"all\t{task_id}\t{cell}\t{category}\t{episode_idx}\t{success}\t{source_path}\n")

    n_succ = sum(1 for r in rows if r[4] == 1)
    task_ids = sorted({r[0] for r in rows})
    print(f"wrote {manifest_path}")
    print(f"  rollouts={len(rows)}  task_ids={len(task_ids)}  success={n_succ}  fail={len(rows) - n_succ}")
    for tid in task_ids:
        sub = [r for r in rows if r[0] == tid]
        s = sum(1 for r in sub if r[4] == 1)
        print(f"    task_id={tid:<2} {sub[0][1]:<24} n={len(sub):<4} succ={s:<4} fail={len(sub) - s}")


if __name__ == "__main__":
    main()

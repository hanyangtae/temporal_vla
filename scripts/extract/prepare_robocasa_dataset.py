"""Add a per-frame `progress` feature to RoboCasa LeRobot v2.1 datasets in place.

For each downloaded `lerobot/` dataset, this script:

  1. Reads each `data/chunk-XXX/episode_YYYYYY.parquet`.
  2. Computes `progress = frame_index / max(length-1, 1)` rounded to 2 decimals,
     using per-episode lengths from `meta/episodes.jsonl`.
  3. Appends the `progress` column to the parquet (overwriting in place).
  4. Registers `progress` in `meta/info.json` `features`.
  5. Adds per-episode progress stats to `meta/episodes_stats.jsonl`.
  6. Adds aggregate progress stats to `meta/stats.json`.

The dataset stays at codebase_version v2.1 — no schema upgrade. groot finetuning
expects v2.1, so we keep the format.

Idempotent: a finished dataset is detected via `info.json` and skipped. Per-file
checks also skip already-processed parquet rows / stats entries, so an interrupted
run can simply be re-executed.

Run inside the lerobot container.

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
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_BASE_TPL = "/temporal_vla/data/robocasa/v1.0/{split}/atomic"

# LeRobot scalar-feature convention: shape=[1] with no `names`.
PROGRESS_FEATURE = {"dtype": "float32", "shape": [1], "names": None}


def discover_dataset_roots(split: str, tasks: list[str] | None) -> list[Path]:
    base = Path(DEFAULT_BASE_TPL.format(split=split))
    if not base.is_dir():
        raise FileNotFoundError(f"Atomic base not found: {base}")

    if tasks:
        out: list[Path] = []
        for task in tasks:
            task_dir = base / task
            matches = sorted(task_dir.glob("*/lerobot"))
            if not matches:
                raise FileNotFoundError(
                    f"No dataset for task '{task}' under {task_dir}/. "
                    f"Run download_robocasa_pretrain_human.sh first."
                )
            out.extend(matches)
        return out

    return sorted(base.glob("*/*/lerobot"))


def load_info(root: Path) -> dict:
    return json.loads((root / "meta" / "info.json").read_text())


def has_progress(root: Path) -> bool:
    return "progress" in load_info(root).get("features", {})


def load_episode_lengths(root: Path) -> dict[int, int]:
    out: dict[int, int] = {}
    with open(root / "meta" / "episodes.jsonl") as f:
        for line in f:
            d = json.loads(line)
            out[int(d["episode_index"])] = int(d["length"])
    return out


def compute_progress(length: int) -> np.ndarray:
    raw = np.arange(length, dtype=np.float32) / max(length - 1, 1)
    return np.round(raw, 2).astype(np.float32)


def episode_parquet_path(root: Path, ep_idx: int, chunks_size: int, tpl: str) -> Path:
    chunk = ep_idx // chunks_size
    return root / tpl.format(episode_chunk=chunk, episode_index=ep_idx)


def add_progress_column(parquet_path: Path, progress: np.ndarray) -> None:
    table = pq.read_table(str(parquet_path))
    if "progress" in table.column_names:
        return
    if len(progress) != table.num_rows:
        raise ValueError(
            f"progress length {len(progress)} != parquet rows {table.num_rows} "
            f"for {parquet_path}"
        )
    table = table.append_column("progress", pa.array(progress, type=pa.float32()))
    # Atomic-ish write: write to .tmp then rename.
    tmp_path = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
    pq.write_table(table, str(tmp_path))
    os.replace(tmp_path, parquet_path)


def update_episodes_stats(root: Path, ep_lengths: dict[int, int]) -> None:
    path = root / "meta" / "episodes_stats.jsonl"
    if not path.exists():
        print(f"  [skip episodes_stats] not found: {path}")
        return

    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    changed = False
    for row in rows:
        ep_idx = int(row["episode_index"])
        if "progress" in row["stats"]:
            continue
        prog = compute_progress(ep_lengths[ep_idx])
        row["stats"]["progress"] = {
            "min": [float(prog.min())],
            "max": [float(prog.max())],
            "mean": [float(prog.mean())],
            "std": [float(prog.std())],
            "count": [int(prog.size)],
        }
        changed = True

    if changed:
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")


def update_stats_json(root: Path, all_progress: np.ndarray) -> None:
    path = root / "meta" / "stats.json"
    if not path.exists():
        print(f"  [skip stats.json] not found: {path}")
        return
    stats = json.loads(path.read_text())
    if "progress" in stats:
        return
    stats["progress"] = {
        "mean": [float(all_progress.mean())],
        "std": [float(all_progress.std())],
        "min": [float(all_progress.min())],
        "max": [float(all_progress.max())],
        "q01": [float(np.quantile(all_progress, 0.01))],
        "q99": [float(np.quantile(all_progress, 0.99))],
    }
    path.write_text(json.dumps(stats, indent=4))


def update_info_json(root: Path) -> None:
    p = root / "meta" / "info.json"
    info = json.loads(p.read_text())
    info["features"]["progress"] = PROGRESS_FEATURE
    p.write_text(json.dumps(info, indent=4))


def process(root: Path) -> None:
    print(f"\n=== {root} ===")
    if has_progress(root):
        print("  [skip] already has 'progress' feature")
        return

    info = load_info(root)
    version = info.get("codebase_version")
    if version != "v2.1":
        raise RuntimeError(
            f"expected codebase_version v2.1, got {version!r}. "
            f"This script targets v2.1 only (groot finetuning constraint)."
        )

    chunks_size = int(info["chunks_size"])
    data_path_tpl = info["data_path"]
    ep_lengths = load_episode_lengths(root)

    all_chunks: list[np.ndarray] = []
    for ep_idx, length in sorted(ep_lengths.items()):
        prog = compute_progress(length)
        pq_path = episode_parquet_path(root, ep_idx, chunks_size, data_path_tpl)
        add_progress_column(pq_path, prog)
        all_chunks.append(prog)

    all_progress = np.concatenate(all_chunks)
    print(
        f"  [add_progress] {all_progress.size} frames over {len(ep_lengths)} episodes, "
        f"min={all_progress.min():.2f}, max={all_progress.max():.2f}, "
        f"mean={all_progress.mean():.2f}"
    )

    # Order matters: info.json is the "done" marker — update last so a crash
    # before completion leaves has_progress() == False and re-running resumes.
    update_episodes_stats(root, ep_lengths)
    update_stats_json(root, all_progress)
    update_info_json(root)
    print("  [done]")


def main() -> None:
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

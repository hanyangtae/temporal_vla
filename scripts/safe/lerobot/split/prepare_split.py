#!/usr/bin/env python3
"""lerobot SAFE 수집물(raw_rollouts) → manifest.tsv 생성 (범용, task-map 무관).

groot_n16 의 robocasa 전용 split helper(prepare_seen4_unseen2_split.py)는 TASKS dict 가
하드코딩돼 있어 LIBERO/VLABench 에 못 쓴다. 이 스크립트는 raw_rollouts 디렉토리 구조
(``{task_name}/task{task_id}--ep{episode_idx}--succ{0|1}.pkl``)에서 직접 메타데이터를
뽑아 ``manifest.tsv`` 를 만든다 → 기존 시각화/분석(safe_feature_vectors.load_manifest,
plot_safe_style_feature_space.py)이 그대로 읽는다.

manifest.tsv 컬럼 (groot_n16 splitter 와 동일):
  split, task_id, episode_idx, source_path, success, task

split 배정:
  --val-ratio 0      → 전부 train (기본; scope=all 로 보면 됨)
  --val-ratio 0.2    → task 별로 20% 를 val_seen 으로 홀드아웃, 나머지 train
"""
from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path

_PKL_RE = re.compile(r"task(\d+)--ep(\d+)--succ([01])\.pkl$")


def scan_rollouts(rollouts_root: Path) -> list[dict]:
    """raw_rollouts/{task}/task{id}--ep{idx}--succ{s}.pkl 를 스캔해 row 리스트 반환."""
    rows: list[dict] = []
    for task_dir in sorted(p for p in rollouts_root.iterdir() if p.is_dir()):
        for pkl in sorted(task_dir.glob("task*--ep*--succ*.pkl")):
            m = _PKL_RE.search(pkl.name)
            if m is None:
                continue
            task_id, episode_idx, succ = int(m[1]), int(m[2]), int(m[3])
            rows.append(
                {
                    "task_id": task_id,
                    "episode_idx": episode_idx,
                    "source_path": str(pkl.resolve()),
                    "success": succ,
                    "task": task_dir.name,
                }
            )
    return rows


def assign_splits(rows: list[dict], val_ratio: float, seed: int) -> None:
    """task 별로 val_ratio 만큼 val_seen, 나머지 train 으로 split 컬럼 채움 (in-place)."""
    if val_ratio <= 0:
        for r in rows:
            r["split"] = "train"
        return
    rng = random.Random(seed)
    by_task: dict[int, list[dict]] = {}
    for r in rows:
        by_task.setdefault(r["task_id"], []).append(r)
    for task_rows in by_task.values():
        idxs = list(range(len(task_rows)))
        rng.shuffle(idxs)
        n_val = max(1, round(len(task_rows) * val_ratio)) if len(task_rows) > 1 else 0
        val_set = set(idxs[:n_val])
        for i, r in enumerate(task_rows):
            r["split"] = "val_seen" if i in val_set else "train"


def write_manifest(rows: list[dict], out_root: Path) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = out_root / "manifest.tsv"
    cols = ["split", "task_id", "episode_idx", "source_path", "success", "task"]
    rows_sorted = sorted(rows, key=lambda r: (r["task_id"], r["episode_idx"]))
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in rows_sorted:
            w.writerow({c: r[c] for c in cols})
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--rollouts-root",
        type=Path,
        required=True,
        help="raw_rollouts 디렉토리 (예: outputs/eval/libero/pi05/<run>/raw_rollouts).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="split 출력 디렉토리 (manifest.tsv). 미지정 시 <run>/split.",
    )
    ap.add_argument("--val-ratio", type=float, default=0.0, help="task별 val_seen 비율.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rollouts_root = args.rollouts_root.resolve()
    if not rollouts_root.is_dir():
        raise SystemExit(f"rollouts-root not found: {rollouts_root}")
    out_root = args.out or rollouts_root.parent / "split"

    rows = scan_rollouts(rollouts_root)
    if not rows:
        raise SystemExit(f"No rollout pkl found under {rollouts_root}")
    assign_splits(rows, args.val_ratio, args.seed)
    manifest = write_manifest(rows, out_root)

    n_succ = sum(r["success"] for r in rows)
    n_task = len({r["task_id"] for r in rows})
    n_train = sum(r["split"] == "train" for r in rows)
    n_val = sum(r["split"] == "val_seen" for r in rows)
    print(f"manifest: {manifest}")
    print(f"  rollouts={len(rows)}  tasks={n_task}  succ={n_succ}  fail={len(rows) - n_succ}")
    print(f"  split: train={n_train}  val_seen={n_val}")


if __name__ == "__main__":
    main()

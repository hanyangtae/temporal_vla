#!/usr/bin/env python3
"""Classify natural-collected N1.5 RoboCasa rollouts by instruction (post-hoc).

Natural collection (no --cell-id) writes ``raw_root/<task>/*.pkl`` where the actual
per-episode instruction lives in ``payload['task_description']`` (the env's sampled
language). This groups them and builds symlink trees that the instruction-pathway loader
globs as ``root/<task>/<cell>/<file>.pkl`` (loader falls back to path levels for
task/cell when the payload lacks robocasa_task/cell_id):

  <out_root>/by_task/<task>/<task>/<file>.pkl                 COAST task-level (all instr pooled)
  <out_root>/by_instruction/<task>/<task>__<slug>/<file>.pkl  top-K instructions (succ+fail)

Also writes ``classification.tsv`` and prints a per-task summary.
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path


def _norm(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _slug(text: str | None) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", _norm(text)).strip("_")
    return s[:40] or "empty"


def _read_meta(path: Path) -> tuple[str, int]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    instr = payload.get("task_description") or ""
    succ = int(payload.get("episode_success", 0))
    return instr, succ


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("raw_root", type=Path, help="natural raw_rollouts root (task/<*.pkl>)")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=4, help="instructions per task for by_instruction")
    ap.add_argument(
        "--min-per-class",
        type=int,
        default=2,
        help="min successes AND min failures required for an instruction cell to be eligible",
    )
    args = ap.parse_args()

    task_dirs = sorted(d for d in args.raw_root.iterdir() if d.is_dir())
    if not task_dirs:
        print(f"no task dirs under {args.raw_root}", file=sys.stderr)
        return 1

    by_task_root = args.out_root / "by_task"
    by_instr_root = args.out_root / "by_instruction"
    rows: list[tuple[str, str, int, int, int, int]] = []
    summary: list[tuple[str, int, int, int]] = []

    for tdir in task_dirs:
        task = tdir.name
        pkls = sorted(tdir.glob("*.pkl"))
        if not pkls:
            continue
        groups: dict[str, list[tuple[Path, int, str]]] = defaultdict(list)
        for pk in pkls:
            instr, succ = _read_meta(pk)
            groups[_norm(instr)].append((pk, succ, instr))

        # by_task: cell = task, link ALL episodes (COAST pools every instruction).
        cell_task = by_task_root / task / task
        cell_task.mkdir(parents=True, exist_ok=True)
        for pk in pkls:
            link = cell_task / pk.name
            if not link.exists():
                link.symlink_to(pk.resolve())

        ranked = []
        for ninstr, items in groups.items():
            n = len(items)
            ns = sum(s for _, s, _ in items)
            nf = n - ns
            ranked.append((ninstr, n, ns, nf, items[0][2]))
        # eligible (both classes present) first, then by total count.
        ranked.sort(key=lambda r: (min(r[2], r[3]) >= args.min_per_class, r[1]), reverse=True)
        chosen = [r for r in ranked if min(r[2], r[3]) >= args.min_per_class][: args.top_k]
        chosen_norm = {r[0] for r in chosen}

        for ninstr, n, ns, nf, raw in ranked:
            rows.append((task, raw, n, ns, nf, int(ninstr in chosen_norm)))

        # by_instruction: link only chosen instruction cells.
        for ninstr, n, ns, nf, raw in chosen:
            cell = by_instr_root / task / f"{task}__{_slug(raw)}"
            cell.mkdir(parents=True, exist_ok=True)
            for pk, _succ, _raw in groups[ninstr]:
                link = cell / pk.name
                if not link.exists():
                    link.symlink_to(pk.resolve())

        summary.append((task, len(pkls), len(groups), len(chosen)))

    args.out_root.mkdir(parents=True, exist_ok=True)
    tsv = args.out_root / "classification.tsv"
    with tsv.open("w") as f:
        f.write("task\tinstruction\tn\tn_succ\tn_fail\tpicked\n")
        for task, raw, n, ns, nf, picked in rows:
            f.write(f"{task}\t{raw}\t{n}\t{ns}\t{nf}\t{picked}\n")

    print(f"=== classification summary (top_k={args.top_k}, min_per_class={args.min_per_class}) ===")
    print(f"{'task':<34} {'pkls':>5} {'#instr':>7} {'#cells':>7}")
    for task, npk, ninstr, ncells in summary:
        print(f"{task:<34} {npk:>5} {ninstr:>7} {ncells:>7}")
    print(f"\nby_task tree:        {by_task_root}")
    print(f"by_instruction tree: {by_instr_root}")
    print(f"classification tsv:  {tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

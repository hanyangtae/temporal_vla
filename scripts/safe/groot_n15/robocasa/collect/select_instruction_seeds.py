#!/usr/bin/env python3
"""Select RoboCasa scenario seeds whose sampled instruction matches a fixed cell.

The selector is intentionally policy-free: it only creates the RoboCasa env,
resets candidate scenario seeds, records the resulting ep_meta, and keeps seeds
whose language matches the canonical instruction cell.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_CELLS_CONFIG = REPO_ROOT / "configs/robocasa/n15_instruction_fixed_cells.tsv"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.policies.groot.robocasa.scenario_replay import (  # noqa: E402
    ep_meta_manifest_path,
    get_robocasa_ep_meta,
    write_ep_meta_manifest,
)


SELECTED_SEED_COLUMNS = [
    "cell_index",
    "cell_id",
    "task",
    "env_name",
    "scenario_seed",
    "instruction",
    "canonical_instruction",
    "ep_meta_path",
]

SCAN_PROGRESS_COLUMNS = [
    "cell_id",
    "task",
    "env_name",
    "last_scanned_seed",
    "scanned",
    "found",
    "target_count",
]

SCAN_SAMPLE_COLUMNS = [
    "cell_id",
    "task",
    "env_name",
    "scenario_seed",
    "instruction",
    "canonical_instruction",
    "is_match",
]


@dataclass(frozen=True)
class InstructionCell:
    cell_index: int
    cell_id: str
    task: str
    env_name: str
    canonical_instruction: str
    target_episodes: int
    seed_search_start: int

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "InstructionCell":
        env_kwargs_json = row.get("env_kwargs_json", "").strip()
        if env_kwargs_json:
            raise ValueError(
                f"{row['cell_id']}: env_kwargs_json is not supported for "
                "instruction seed selection; preselect matching scenario seeds instead"
            )
        return cls(
            cell_index=int(row["cell_index"]),
            cell_id=row["cell_id"],
            task=row["task"],
            env_name=row["env_name"],
            canonical_instruction=row["canonical_instruction"],
            target_episodes=int(row["target_episodes"]),
            seed_search_start=int(row["seed_search_start"]),
        )


def normalize_instruction(value: str) -> str:
    return " ".join(value.strip().split())


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return _as_text(value[0])
        return " ".join(str(part) for part in value)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _as_text(tolist())
    return str(value)


def extract_instruction(obs: dict[str, Any], ep_meta: dict[str, Any]) -> str:
    for value in (
        ep_meta.get("lang"),
        obs.get("annotation.human.task_description"),
        obs.get("annotation.human.action.task_description"),
    ):
        text = _as_text(value)
        if text:
            return text
    raise RuntimeError("Could not find instruction in ep_meta or reset observation")


def load_instruction_cells(path: Path) -> list[InstructionCell]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    cells = [InstructionCell.from_row(row) for row in rows]

    cell_ids = [cell.cell_id for cell in cells]
    if len(cell_ids) != len(set(cell_ids)):
        raise ValueError(f"Duplicate cell_id in {path}")

    cell_indexes = [cell.cell_index for cell in cells]
    if len(cell_indexes) != len(set(cell_indexes)):
        raise ValueError(f"Duplicate cell_index in {path}")

    return cells


def make_env(
    env_name: str,
    *,
    task: str,
    scenario_seed: int,
    enable_render: bool,
):
    del enable_render

    from http_feature_collect import make_env as make_collect_env

    return make_collect_env(
        task,
        "target",
        env_name=env_name,
        scenario_seed=scenario_seed,
        video_dir=None,
    )


def _close_env(env: Any) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


def _selected_row(
    *,
    cell: InstructionCell,
    scenario_seed: int,
    instruction: str,
    ep_meta_path: Path,
) -> dict[str, Any]:
    return {
        "cell_index": cell.cell_index,
        "cell_id": cell.cell_id,
        "task": cell.task,
        "env_name": cell.env_name,
        "scenario_seed": scenario_seed,
        "instruction": instruction,
        "canonical_instruction": cell.canonical_instruction,
        "ep_meta_path": str(ep_meta_path),
    }


def scan_cell(
    cell: InstructionCell,
    *,
    target_count: int,
    seed_start: int,
    max_seeds: int,
    ep_meta_dir: Path,
    enable_render: bool,
    make_env_fn: Callable[..., Any] = make_env,
    robocasa_env_source: str = "robocasa365",
    progress_interval: int = 25,
    excluded_seed_keys: set[tuple[str, int]] | None = None,
    on_selected: Callable[[dict[str, Any]], None] | None = None,
    on_progress: Callable[[int, int, int], None] | None = None,
    on_sample: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    canonical = normalize_instruction(cell.canonical_instruction)
    excluded_seed_keys = excluded_seed_keys or set()

    for attempts, scenario_seed in enumerate(range(seed_start, seed_start + max_seeds), start=1):
        env = make_env_fn(
            cell.env_name,
            task=cell.task,
            scenario_seed=scenario_seed,
            enable_render=enable_render,
        )
        try:
            obs, _info = env.reset(seed=scenario_seed)
            ep_meta = get_robocasa_ep_meta(env)
            instruction = extract_instruction(obs, ep_meta)
        finally:
            _close_env(env)

        is_match = normalize_instruction(instruction) == canonical
        if on_sample is not None:
            on_sample(
                {
                    "cell_id": cell.cell_id,
                    "task": cell.task,
                    "env_name": cell.env_name,
                    "scenario_seed": scenario_seed,
                    "instruction": instruction,
                    "canonical_instruction": cell.canonical_instruction,
                    "is_match": int(is_match),
                }
            )

        if not is_match:
            if progress_interval > 0 and attempts % progress_interval == 0:
                if on_progress is not None:
                    on_progress(scenario_seed, attempts, len(selected))
                print(
                    f"[scan] {cell.cell_id}: scanned={attempts} "
                    f"found={len(selected)}/{target_count} seed={scenario_seed}",
                    flush=True,
                )
            continue

        if (cell.cell_id, scenario_seed) in excluded_seed_keys:
            print(
                f"[exclude] {cell.cell_id}: seed={scenario_seed} already reserved",
                flush=True,
            )
            continue

        manifest_dir = ep_meta_dir / cell.task / cell.cell_id
        manifest_path = ep_meta_manifest_path(manifest_dir, cell.env_name, scenario_seed)
        write_ep_meta_manifest(
            manifest_path,
            env_name=cell.env_name,
            scenario_seed=scenario_seed,
            ep_meta=ep_meta,
            robocasa_env_source=robocasa_env_source,
            sort_keys=True,
        )
        row = _selected_row(
            cell=cell,
            scenario_seed=scenario_seed,
            instruction=instruction,
            ep_meta_path=manifest_path,
        )
        selected.append(row)
        if on_selected is not None:
            on_selected(row)
        if on_progress is not None:
            on_progress(scenario_seed, attempts, len(selected))
        print(
            f"[match] {cell.cell_id}: found={len(selected)}/{target_count} "
            f"seed={scenario_seed} instruction={instruction!r}",
            flush=True,
        )
        if len(selected) >= target_count:
            break

    if len(selected) < target_count:
        raise RuntimeError(
            f"{cell.cell_id}: found {len(selected)}/{target_count} matching seeds "
            f"in [{seed_start}, {seed_start + max_seeds - 1}]"
        )

    return selected


def load_selected_seed_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    normalized: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = dict(row)
        item["cell_index"] = int(item["cell_index"])
        item["scenario_seed"] = int(item["scenario_seed"])
        normalized.append(item)
    return normalized


def write_selected_seed_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SELECTED_SEED_COLUMNS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_excluded_seed_keys(paths: list[Path] | None) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    for path in paths or []:
        for row in load_selected_seed_manifest(path):
            keys.add((str(row["cell_id"]), int(row["scenario_seed"])))
    return keys


def _default_scan_progress_path(output_tsv: Path) -> Path:
    return output_tsv.with_name(f"{output_tsv.stem}_scan_progress.tsv")


def _default_scan_samples_path(output_tsv: Path) -> Path:
    return output_tsv.with_name(f"{output_tsv.stem}_scan_samples.tsv")


def load_scan_progress(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    progress: dict[str, dict[str, Any]] = {}
    for row in rows:
        item: dict[str, Any] = dict(row)
        item["last_scanned_seed"] = int(item["last_scanned_seed"])
        item["scanned"] = int(item["scanned"])
        item["found"] = int(item["found"])
        item["target_count"] = int(item["target_count"])
        progress[str(item["cell_id"])] = item
    return progress


def write_scan_progress(path: Path, progress: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(progress.values(), key=lambda row: str(row["cell_id"]))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCAN_PROGRESS_COLUMNS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_scan_samples(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    normalized: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = dict(row)
        item["scenario_seed"] = int(item["scenario_seed"])
        item["is_match"] = int(item["is_match"])
        normalized.append(item)
    return normalized


def write_scan_samples(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCAN_SAMPLE_COLUMNS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def select_instruction_seeds(
    *,
    cells_config: Path,
    output_tsv: Path,
    ep_meta_dir: Path,
    max_seeds_per_cell: int,
    target_per_cell: int | None,
    seed_start: int | None,
    cell_ids: list[str] | None,
    exclude_selected_seeds: list[Path] | None = None,
    enable_render: bool,
    progress_interval: int = 25,
    resume: bool = False,
    scan_progress_tsv: Path | None = None,
    scan_samples_tsv: Path | None = None,
    make_env_fn: Callable[..., Any] = make_env,
) -> list[dict[str, Any]]:
    requested = None if cell_ids is None else set(cell_ids)
    cells = [
        cell
        for cell in load_instruction_cells(cells_config)
        if requested is None or cell.cell_id in requested
    ]
    if requested is not None and len(cells) != len(requested):
        found = {cell.cell_id for cell in cells}
        missing = ", ".join(sorted(requested - found))
        raise ValueError(f"Unknown cell_id(s): {missing}")

    rows: list[dict[str, Any]] = load_selected_seed_manifest(output_tsv) if resume else []
    seen = {(str(row["cell_id"]), int(row["scenario_seed"])) for row in rows}
    excluded_seed_keys = load_excluded_seed_keys(exclude_selected_seeds)
    overlap = seen & excluded_seed_keys
    if overlap:
        preview = ", ".join(f"{cell_id}/{seed}" for cell_id, seed in sorted(overlap)[:10])
        extra = "" if len(overlap) <= 10 else f" ... +{len(overlap) - 10}"
        raise ValueError(f"existing selected seeds overlap excluded manifests: {preview}{extra}")
    write_selected_seed_manifest(output_tsv, rows)
    scan_progress_path = scan_progress_tsv or _default_scan_progress_path(output_tsv)
    scan_progress = load_scan_progress(scan_progress_path) if resume else {}
    scan_samples_path = scan_samples_tsv or _default_scan_samples_path(output_tsv)
    scan_samples = load_scan_samples(scan_samples_path) if resume else []
    sampled = {(str(row["cell_id"]), int(row["scenario_seed"])) for row in scan_samples}

    for cell in cells:
        target_count = target_per_cell or cell.target_episodes
        existing = [
            row
            for row in rows
            if str(row["cell_id"]) == cell.cell_id
            and normalize_instruction(str(row["canonical_instruction"]))
            == normalize_instruction(cell.canonical_instruction)
        ]
        if len(existing) >= target_count:
            print(
                f"[skip] {cell.cell_id}: existing={len(existing)}/{target_count}",
                flush=True,
            )
            continue

        cell_seed_start = seed_start if seed_start is not None else cell.seed_search_start
        if resume and existing:
            cell_seed_start = max(
                cell_seed_start,
                max(int(row["scenario_seed"]) for row in existing) + 1,
            )
        if resume and cell.cell_id in scan_progress:
            cell_seed_start = max(
                cell_seed_start,
                int(scan_progress[cell.cell_id]["last_scanned_seed"]) + 1,
            )
        sampled_seeds = [
            int(row["scenario_seed"])
            for row in scan_samples
            if str(row["cell_id"]) == cell.cell_id
        ]
        if resume and sampled_seeds:
            cell_seed_start = max(cell_seed_start, max(sampled_seeds) + 1)
        print(
            f"[cell] {cell.cell_id}: target={target_count} "
            f"existing={len(existing)} start_seed={cell_seed_start}",
            flush=True,
        )

        def _record(row: dict[str, Any]) -> None:
            key = (str(row["cell_id"]), int(row["scenario_seed"]))
            if key in seen:
                return
            seen.add(key)
            rows.append(row)
            write_selected_seed_manifest(output_tsv, rows)

        def _progress(last_scanned_seed: int, scanned: int, found: int) -> None:
            scan_progress[cell.cell_id] = {
                "cell_id": cell.cell_id,
                "task": cell.task,
                "env_name": cell.env_name,
                "last_scanned_seed": last_scanned_seed,
                "scanned": scanned,
                "found": len(existing) + found,
                "target_count": target_count,
            }
            write_scan_progress(scan_progress_path, scan_progress)

        def _sample(row: dict[str, Any]) -> None:
            key = (str(row["cell_id"]), int(row["scenario_seed"]))
            if key in sampled:
                return
            sampled.add(key)
            scan_samples.append(row)
            write_scan_samples(scan_samples_path, scan_samples)

        scan_cell(
            cell,
            target_count=target_count - len(existing),
            seed_start=cell_seed_start,
            max_seeds=max_seeds_per_cell,
            ep_meta_dir=ep_meta_dir,
            enable_render=enable_render,
            make_env_fn=make_env_fn,
            progress_interval=progress_interval,
            excluded_seed_keys=excluded_seed_keys,
            on_selected=_record,
            on_progress=_progress,
            on_sample=_sample,
        )

    write_selected_seed_manifest(output_tsv, rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells-config", type=Path, default=DEFAULT_CELLS_CONFIG)
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--ep-meta-dir", type=Path, required=True)
    parser.add_argument("--cell-id", action="append", dest="cell_ids")
    parser.add_argument("--target-per-cell", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument(
        "--exclude-selected-seeds",
        type=Path,
        action="append",
        default=None,
        help=(
            "Existing selected-seed manifest(s) to exclude per cell. Use this "
            "for held-out eval manifests so collection/training seeds are not reused."
        ),
    )
    parser.add_argument("--max-seeds-per-cell", type=int, default=10000)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--scan-progress-tsv",
        type=Path,
        default=None,
        help="Optional scan progress checkpoint. Defaults next to --output-tsv.",
    )
    parser.add_argument(
        "--scan-samples-tsv",
        type=Path,
        default=None,
        help="Optional per-seed instruction sample log. Defaults next to --output-tsv.",
    )
    parser.add_argument(
        "--enable-render",
        action="store_true",
        help=(
            "Deprecated compatibility flag. The selector now uses the collector "
            "env path, which enables the same RoboCasa wrapper stack as collection."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = select_instruction_seeds(
        cells_config=args.cells_config,
        output_tsv=args.output_tsv,
        ep_meta_dir=args.ep_meta_dir,
        max_seeds_per_cell=args.max_seeds_per_cell,
        target_per_cell=args.target_per_cell,
        seed_start=args.seed_start,
        cell_ids=args.cell_ids,
        exclude_selected_seeds=args.exclude_selected_seeds,
        enable_render=args.enable_render,
        progress_interval=args.progress_interval,
        resume=args.resume,
        scan_progress_tsv=args.scan_progress_tsv,
        scan_samples_tsv=args.scan_samples_tsv,
    )
    print(f"selected {len(rows)} instruction-fixed seeds -> {args.output_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

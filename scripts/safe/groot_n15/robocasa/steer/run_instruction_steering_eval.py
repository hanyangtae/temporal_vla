#!/usr/bin/env python3
"""Run selected N1.5 instruction steering eval matrix rows sequentially."""

from __future__ import annotations

import argparse
import csv
import pickle
import shlex
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_RUN_ROOT = REPO_ROOT / "outputs/eval/robocasa/groot_n15/target_instruction_fixed15_pathway_50ep"
DEFAULT_EVAL_MATRIX = (
    DEFAULT_RUN_ROOT
    / "steer_eval/steer_eval_instruction_fixed15_alpha_all_b01_b03/eval_matrix.tsv"
)


@dataclass(frozen=True)
class EvalRow:
    row_index: int
    cell_id: str
    cell_index: int
    task: str
    condition: str
    pathway: str
    beta: str
    alpha: str
    steering_npz: str
    server_command: str
    collect_command: str


def load_eval_rows(
    path: Path,
    *,
    cell_ids: set[str] | None = None,
    conditions: set[str] | None = None,
    pathways: set[str] | None = None,
    row_indices: set[int] | None = None,
) -> list[EvalRow]:
    selected: list[EvalRow] = []
    with path.open(newline="") as f:
        for idx, raw in enumerate(csv.DictReader(f, delimiter="\t")):
            row = EvalRow(
                row_index=idx,
                cell_id=raw["cell_id"],
                cell_index=int(raw["cell_index"]),
                task=raw["task"],
                condition=raw["condition"],
                pathway=raw["pathway"],
                beta=raw["beta"],
                alpha=raw["alpha"],
                steering_npz=raw["steering_npz"],
                server_command=raw["server_command"],
                collect_command=raw["collect_command"],
            )
            if row_indices is not None and row.row_index not in row_indices:
                continue
            if cell_ids is not None and row.cell_id not in cell_ids:
                continue
            if conditions is not None and row.condition not in conditions:
                continue
            if pathways is not None and row.pathway not in pathways:
                continue
            selected.append(row)
    return selected


def _arg_after(argv: list[str], flag: str) -> str | None:
    try:
        idx = argv.index(flag)
    except ValueError:
        return None
    if idx + 1 >= len(argv):
        raise ValueError(f"{flag} has no value in command: {shlex.join(argv)}")
    return argv[idx + 1]


def command_port(command: str) -> int:
    value = _arg_after(shlex.split(command), "--port")
    if value is None:
        raise ValueError(f"--port missing from server command: {command}")
    return int(value)


def server_container_name(command: str) -> str | None:
    return _arg_after(shlex.split(command), "--name")


def collect_output_dir(command: str) -> Path:
    value = _arg_after(shlex.split(command), "--output-dir")
    if value is None:
        raise ValueError(f"--output-dir missing from collect command: {command}")
    path = Path(value)
    if path.is_absolute() and path.parts[:2] == ("/", "temporal_vla"):
        return REPO_ROOT / path.relative_to("/temporal_vla")
    return path


def wait_http_health(
    port: int,
    *,
    timeout_s: float,
    interval_s: float = 2.0,
    process: subprocess.Popen | None = None,
) -> None:
    deadline = time.time() + timeout_s
    url = f"http://127.0.0.1:{port}/health"
    last_error: Exception | None = None
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"server process exited before health check passed: returncode={process.returncode}"
            )
        try:
            with urllib.request.urlopen(url, timeout=3.0) as response:
                if 200 <= response.status < 300:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
        time.sleep(interval_s)
    raise TimeoutError(f"server did not become healthy at {url}: {last_error}")


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def summarize_rollouts(output_dir: Path, *, task: str, cell_id: str) -> dict[str, int]:
    cell_dir = output_dir / task / cell_id
    files = sorted(cell_dir.glob("*.pkl"))
    success = 0
    for path in files:
        payload = _load_pickle(path)
        success += int(payload.get("episode_success", 0))
    return {"pkl_count": len(files), "success_count": success, "failure_count": len(files) - success}


def _terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=20)


def _remove_docker_container(container_name: str, *, log_path: Path) -> None:
    with log_path.open("a") as log_f:
        log_f.write(f"\n[runner] docker rm -f {container_name}\n")
        try:
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                cwd=REPO_ROOT,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            log_f.write(f"[runner] docker rm -f {container_name} timed out\n")


def run_eval_row(
    row: EvalRow,
    *,
    log_dir: Path,
    health_timeout_s: float,
    dry_run: bool,
) -> dict[str, Any]:
    output_dir = collect_output_dir(row.collect_command)
    server_log = log_dir / f"row{row.row_index:04d}_{row.cell_id}_{row.condition}.server.log"
    collect_log = log_dir / f"row{row.row_index:04d}_{row.cell_id}_{row.condition}.collect.log"
    base = {
        "row_index": row.row_index,
        "cell_id": row.cell_id,
        "cell_index": row.cell_index,
        "task": row.task,
        "condition": row.condition,
        "pathway": row.pathway,
        "beta": row.beta,
        "alpha": row.alpha,
        "steering_npz": row.steering_npz,
        "output_dir": str(output_dir),
        "server_log": str(server_log),
        "collect_log": str(collect_log),
    }
    if dry_run:
        return {
            **base,
            "status": "dry-run",
            "server_returncode": "",
            "collect_returncode": "",
            "pkl_count": "",
            "success_count": "",
            "failure_count": "",
            "error": "",
        }

    log_dir.mkdir(parents=True, exist_ok=True)
    server_process: subprocess.Popen | None = None
    container_name = server_container_name(row.server_command)
    try:
        with server_log.open("w") as server_f:
            server_process = subprocess.Popen(
                shlex.split(row.server_command),
                cwd=REPO_ROOT,
                stdout=server_f,
                stderr=subprocess.STDOUT,
            )
        wait_http_health(
            command_port(row.server_command),
            timeout_s=health_timeout_s,
            process=server_process,
        )
        with collect_log.open("w") as collect_f:
            collect = subprocess.run(
                shlex.split(row.collect_command),
                cwd=REPO_ROOT,
                stdout=collect_f,
                stderr=subprocess.STDOUT,
                check=False,
            )
        server_code = server_process.poll()
        summary = summarize_rollouts(output_dir, task=row.task, cell_id=row.cell_id)
        status = "ok" if collect.returncode == 0 and summary["pkl_count"] > 0 else "failed"
        return {
            **base,
            "status": status,
            "server_returncode": "" if server_code is None else server_code,
            "collect_returncode": collect.returncode,
            **summary,
            "error": "",
        }
    except Exception as exc:
        server_code = "" if server_process is None or server_process.poll() is None else server_process.poll()
        return {
            **base,
            "status": "failed",
            "server_returncode": server_code,
            "collect_returncode": "",
            "pkl_count": "",
            "success_count": "",
            "failure_count": "",
            "error": str(exc),
        }
    finally:
        if server_process is not None:
            _terminate_process(server_process)
        if container_name is not None:
            _remove_docker_container(container_name, log_path=server_log)


def append_results(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_index",
        "cell_id",
        "cell_index",
        "task",
        "condition",
        "pathway",
        "beta",
        "alpha",
        "steering_npz",
        "status",
        "server_returncode",
        "collect_returncode",
        "pkl_count",
        "success_count",
        "failure_count",
        "output_dir",
        "server_log",
        "collect_log",
        "error",
    ]
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-matrix", type=Path, default=DEFAULT_EVAL_MATRIX)
    parser.add_argument("--cell-id", action="append", dest="cell_ids")
    parser.add_argument("--condition", action="append", dest="conditions")
    parser.add_argument(
        "--pathway",
        action="append",
        dest="pathways",
        help="Matrix pathway/feature key filter, e.g. baseline, dit, vl, or dit_layer0.",
    )
    parser.add_argument("--row-index", type=int, action="append", dest="row_indices")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--results-tsv", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--health-timeout-s", type=float, default=600.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_eval_rows(
        args.eval_matrix,
        cell_ids=None if args.cell_ids is None else set(args.cell_ids),
        conditions=None if args.conditions is None else set(args.conditions),
        pathways=None if args.pathways is None else set(args.pathways),
        row_indices=None if args.row_indices is None else set(args.row_indices),
    )
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    if not rows:
        raise SystemExit("no eval rows selected")
    eval_root = args.eval_matrix.parent
    results_tsv = args.results_tsv or eval_root / "results.tsv"
    log_dir = args.log_dir or eval_root / "logs"
    results = [
        run_eval_row(
            row,
            log_dir=log_dir,
            health_timeout_s=args.health_timeout_s,
            dry_run=bool(args.dry_run),
        )
        for row in rows
    ]
    append_results(results_tsv, results)
    print(f"wrote {len(results)} rows -> {results_tsv}")
    return 0 if all(row["status"] in {"ok", "dry-run"} for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

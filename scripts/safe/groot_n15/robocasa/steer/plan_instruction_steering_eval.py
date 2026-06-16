#!/usr/bin/env python3
"""Plan N1.5 instruction-cell steering eval conditions from conceptor fits."""

from __future__ import annotations

import argparse
import csv
import shlex
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_RUN_ROOT = REPO_ROOT / "outputs/eval/robocasa/groot_n15/target_instruction_fixed15_pathway_50ep"
DEFAULT_PROFILE = "/temporal_vla/configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml"
DEFAULT_SELECTED_SEEDS = DEFAULT_RUN_ROOT / "manifests/selected_instruction_seeds.tsv"
DEFAULT_CONCEPTOR_ROOT = DEFAULT_RUN_ROOT / "conceptor"
DEFAULT_RUN_TAG = "steer_eval_instruction_fixed15"
DEFAULT_BETAS = [0.1, 0.3]
DEFAULT_CONTAINER_REPO_ROOT = Path("/temporal_vla")
SELECTED_SEED_COLUMNS = {"cell_id", "scenario_seed"}


@dataclass(frozen=True)
class FitRow:
    pathway: str
    agg_mode: str
    cell_id: str
    cell_index: int
    task: str
    selected_alphas: tuple[float, ...]
    output_dir: Path


@dataclass(frozen=True)
class EvalCondition:
    cell_id: str
    cell_index: int
    task: str
    condition: str
    pathway: str
    beta: float | None
    alpha: float | None
    steering_npz: Path | None
    server_command: str
    collect_command: str


def _format_float(value: float) -> str:
    return f"{value:g}"


def _format_float_tag(value: float) -> str:
    return _format_float(value).replace(".", "p")


def _docker_name_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())
    return token.strip("_.-").lower()


def _server_container_name(*, run_tag: str, cell_id: str, condition: str) -> str:
    return "_".join(
        [
            "temporal_vla",
            "lerobot",
            _docker_name_token(run_tag),
            _docker_name_token(cell_id),
            _docker_name_token(condition),
        ]
    )


def _steering_server_args(pathway_key: str) -> list[str]:
    layer_prefix = "dit_layer"
    if pathway_key == "dit":
        return ["--steering-pathway", "dit"]
    if pathway_key == "vl":
        return ["--steering-pathway", "vl"]
    if pathway_key.startswith(layer_prefix):
        suffix = pathway_key[len(layer_prefix) :]
        if not suffix.isdigit():
            raise ValueError(f"invalid DiT layer pathway key: {pathway_key}")
        return ["--steering-pathway", "dit", "--steering-layer", suffix]
    raise ValueError(
        f"unsupported steering pathway key {pathway_key!r}; "
        "expected dit, vl, or dit_layer<layer_id>"
    )


def to_container_path(path: Path, *, container_repo_root: Path) -> Path:
    if not path.is_absolute():
        return container_repo_root / path
    try:
        return container_repo_root / path.relative_to(REPO_ROOT)
    except ValueError:
        return path


def _parse_alphas(value: str) -> tuple[float, ...]:
    if not value:
        return ()
    return tuple(float(part) for part in value.split(",") if part)


def _read_fit_summary(path: Path, pathway: str) -> list[FitRow]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    fits: list[FitRow] = []
    for row in rows:
        if row["status"] != "fit":
            continue
        fits.append(
            FitRow(
                pathway=pathway,
                agg_mode=row["agg_mode"],
                cell_id=row["cell_id"],
                cell_index=int(row["cell_index"]),
                task=row["task"],
                selected_alphas=_parse_alphas(row["selected_alphas"]),
                output_dir=Path(row["output_dir"]),
            )
        )
    return fits


def _load_selected_seed_keys(path: Path) -> set[tuple[str, int]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        missing = SELECTED_SEED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing selected seed columns: {', '.join(sorted(missing))}")
        return {
            (str(row["cell_id"]), int(row["scenario_seed"]))
            for row in reader
        }


def validate_selected_seed_no_overlap(
    *,
    selected_seeds: Path,
    forbidden_selected_seeds: list[Path] | None,
) -> None:
    if not forbidden_selected_seeds:
        return
    selected = _load_selected_seed_keys(selected_seeds)
    overlaps: set[tuple[str, int]] = set()
    for path in forbidden_selected_seeds:
        overlaps.update(selected & _load_selected_seed_keys(path))
    if not overlaps:
        return
    preview = ", ".join(f"{cell_id}/{seed}" for cell_id, seed in sorted(overlaps)[:10])
    extra = "" if len(overlaps) <= 10 else f" ... +{len(overlaps) - 10}"
    raise ValueError(f"selected eval seeds overlap forbidden manifests: {preview}{extra}")


def load_fit_rows(
    conceptor_root: Path,
    *,
    pathways: Iterable[str],
    agg_mode: str,
    cell_ids: set[str] | None,
) -> list[FitRow]:
    fits: list[FitRow] = []
    for pathway in pathways:
        summary = conceptor_root / pathway / agg_mode / "fit_summary.tsv"
        if not summary.exists():
            raise FileNotFoundError(f"missing fit summary: {summary}")
        for row in _read_fit_summary(summary, pathway):
            if cell_ids is not None and row.cell_id not in cell_ids:
                continue
            fits.append(row)
    if not fits:
        raise ValueError("no fit rows selected")
    return fits


def _select_alphas(row: FitRow, policy: str, explicit_alpha: float | None) -> tuple[float, ...]:
    if policy == "explicit":
        if explicit_alpha is None:
            raise ValueError("--alpha is required when --alpha-policy=explicit")
        if explicit_alpha not in row.selected_alphas:
            raise ValueError(
                f"alpha {explicit_alpha:g} not selected for {row.pathway}/{row.cell_id}: "
                f"{','.join(_format_float(a) for a in row.selected_alphas)}"
            )
        return (explicit_alpha,)
    if policy == "first":
        if not row.selected_alphas:
            raise ValueError(f"no selected alpha for {row.pathway}/{row.cell_id}")
        return (row.selected_alphas[0],)
    if policy == "all":
        return row.selected_alphas
    raise ValueError(f"unknown alpha policy: {policy}")


def _server_command(
    *,
    port: int,
    profile: str,
    condition: EvalCondition | None,
    docker_service: str,
    device: str,
    container_name: str,
    container_repo_root: Path,
) -> str:
    argv = [
        "docker",
        "compose",
        "run",
        "--rm",
        "--no-deps",
        "-T",
        "--name",
        container_name,
        docker_service,
        "python",
        "/temporal_vla/scripts/serve/lerobot.py",
        "--profile",
        profile,
        "--device",
        device,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--collect",
        "--capture-vl",
    ]
    if condition is not None and condition.steering_npz is not None:
        argv.extend(
            [
                "--steering-npz",
                str(
                    to_container_path(
                        condition.steering_npz,
                        container_repo_root=container_repo_root,
                    )
                ),
                *_steering_server_args(condition.pathway),
                "--steering-beta",
                _format_float(float(condition.beta)),
                "--steering-alpha",
                _format_float(float(condition.alpha)),
                "--steering-key",
                "C_steer",
            ]
        )
    return shlex.join(argv)


def _collect_command(
    *,
    selected_seeds: Path,
    output_dir: Path,
    ep_meta_dir: Path,
    cell_id: str,
    max_episodes_per_cell: int,
    robocasa_container: str,
    port: int,
    container_repo_root: Path,
) -> str:
    argv = [
        "docker",
        "exec",
        "-e",
        "MUJOCO_GL=egl",
        robocasa_container,
        "python",
        "/temporal_vla/scripts/safe/groot_n15/robocasa/collect/collect_instruction_fixed_http_features.py",
        "--selected-seeds",
        str(to_container_path(selected_seeds, container_repo_root=container_repo_root)),
        "--output-dir",
        str(to_container_path(output_dir, container_repo_root=container_repo_root)),
        "--ep-meta-dir",
        str(to_container_path(ep_meta_dir, container_repo_root=container_repo_root)),
        "--vla-server",
        f"http://127.0.0.1:{port}",
        "--cell-id",
        cell_id,
        "--max-episodes-per-cell",
        str(max_episodes_per_cell),
        "--n-action-steps",
        "16",
        "--max-episode-steps",
        "720",
        "--video-fps",
        "20",
        "--steps-per-render",
        "2",
        "--wait-ready",
    ]
    return shlex.join(argv)


def build_eval_conditions(
    fits: list[FitRow],
    *,
    selected_seeds: Path,
    run_root: Path,
    run_tag: str,
    betas: list[float],
    alpha_policy: str,
    alpha: float | None,
    max_episodes_per_cell: int,
    port: int,
    profile: str,
    docker_service: str,
    robocasa_container: str,
    device: str,
    container_repo_root: Path,
) -> list[EvalCondition]:
    by_cell: dict[str, FitRow] = {}
    for row in fits:
        by_cell.setdefault(row.cell_id, row)

    conditions: list[EvalCondition] = []
    for cell_id, ref in sorted(by_cell.items(), key=lambda item: item[1].cell_index):
        condition_name = "baseline"
        out_root = run_root / "steer_eval" / run_tag / cell_id / condition_name
        baseline = EvalCondition(
            cell_id=cell_id,
            cell_index=ref.cell_index,
            task=ref.task,
            condition=condition_name,
            pathway="baseline",
            beta=None,
            alpha=None,
            steering_npz=None,
            server_command="",
            collect_command="",
        )
        conditions.append(
            EvalCondition(
                **{
                    **baseline.__dict__,
                    "server_command": _server_command(
                        port=port,
                        profile=profile,
                        condition=None,
                        docker_service=docker_service,
                        device=device,
                        container_name=_server_container_name(
                            run_tag=run_tag,
                            cell_id=cell_id,
                            condition=condition_name,
                        ),
                        container_repo_root=container_repo_root,
                    ),
                    "collect_command": _collect_command(
                        selected_seeds=selected_seeds,
                        output_dir=out_root / "raw_rollouts",
                        ep_meta_dir=out_root / "ep_meta",
                        cell_id=cell_id,
                        max_episodes_per_cell=max_episodes_per_cell,
                        robocasa_container=robocasa_container,
                        port=port,
                        container_repo_root=container_repo_root,
                    ),
                }
            )
        )

        for row in sorted(
            [fit for fit in fits if fit.cell_id == cell_id],
            key=lambda fit: fit.pathway,
        ):
            npz = row.output_dir / "conceptors.npz"
            if not npz.exists():
                raise FileNotFoundError(f"missing conceptor npz: {npz}")
            for selected_alpha in _select_alphas(row, alpha_policy, alpha):
                for beta in betas:
                    condition_name = (
                        f"{row.pathway}_a{_format_float_tag(selected_alpha)}"
                        f"_b{_format_float_tag(beta)}"
                    )
                    out_root = run_root / "steer_eval" / run_tag / cell_id / condition_name
                    partial = EvalCondition(
                        cell_id=cell_id,
                        cell_index=row.cell_index,
                        task=row.task,
                        condition=condition_name,
                        pathway=row.pathway,
                        beta=beta,
                        alpha=selected_alpha,
                        steering_npz=npz,
                        server_command="",
                        collect_command="",
                    )
                    conditions.append(
                        EvalCondition(
                            **{
                                **partial.__dict__,
                                "server_command": _server_command(
                                    port=port,
                                    profile=profile,
                                    condition=partial,
                                    docker_service=docker_service,
                                    device=device,
                                    container_name=_server_container_name(
                                        run_tag=run_tag,
                                        cell_id=cell_id,
                                        condition=condition_name,
                                    ),
                                    container_repo_root=container_repo_root,
                                ),
                                "collect_command": _collect_command(
                                    selected_seeds=selected_seeds,
                                    output_dir=out_root / "raw_rollouts",
                                    ep_meta_dir=out_root / "ep_meta",
                                    cell_id=cell_id,
                                    max_episodes_per_cell=max_episodes_per_cell,
                                    robocasa_container=robocasa_container,
                                    port=port,
                                    container_repo_root=container_repo_root,
                                ),
                            }
                        )
                    )
    return conditions


def write_eval_matrix(path: Path, conditions: list[EvalCondition]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cell_id",
        "cell_index",
        "task",
        "condition",
        "pathway",
        "beta",
        "alpha",
        "steering_npz",
        "server_command",
        "collect_command",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in conditions:
            writer.writerow(
                {
                    "cell_id": row.cell_id,
                    "cell_index": row.cell_index,
                    "task": row.task,
                    "condition": row.condition,
                    "pathway": row.pathway,
                    "beta": "" if row.beta is None else _format_float(row.beta),
                    "alpha": "" if row.alpha is None else _format_float(row.alpha),
                    "steering_npz": "" if row.steering_npz is None else str(row.steering_npz),
                    "server_command": row.server_command,
                    "collect_command": row.collect_command,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--selected-seeds", type=Path, default=DEFAULT_SELECTED_SEEDS)
    parser.add_argument(
        "--forbid-selected-overlap",
        type=Path,
        action="append",
        default=None,
        help=(
            "Selected-seed manifest(s) that eval seeds must not overlap by "
            "cell_id/scenario_seed. Use collection/conceptor manifests here "
            "when planning held-out SR eval."
        ),
    )
    parser.add_argument("--conceptor-root", type=Path, default=DEFAULT_CONCEPTOR_ROOT)
    parser.add_argument("--agg-mode", default="truncated_w10")
    parser.add_argument(
        "--pathway",
        action="append",
        dest="pathways",
        help="Feature key(s) to plan, e.g. dit, vl, or aligned keys like dit_layer0.",
    )
    parser.add_argument("--cell-id", action="append", dest="cell_ids")
    parser.add_argument("--beta", type=float, action="append", dest="betas")
    parser.add_argument(
        "--alpha-policy",
        choices=("all", "first", "explicit"),
        default="all",
        help="Which selected alpha(s) to evaluate per cell/pathway.",
    )
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--max-episodes-per-cell", type=int, default=20)
    parser.add_argument("--run-tag", default=DEFAULT_RUN_TAG)
    parser.add_argument("--port", type=int, default=8400)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--docker-service", default="lerobot")
    parser.add_argument("--robocasa-container", default="temporal_vla-robocasa-run-3705634bbbf6")
    parser.add_argument("--container-repo-root", type=Path, default=DEFAULT_CONTAINER_REPO_ROOT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output-tsv", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pathways = args.pathways or ["dit", "vl"]
    betas = args.betas or list(DEFAULT_BETAS)
    if args.max_episodes_per_cell < 1:
        raise SystemExit("--max-episodes-per-cell must be >= 1")
    validate_selected_seed_no_overlap(
        selected_seeds=args.selected_seeds,
        forbidden_selected_seeds=args.forbid_selected_overlap,
    )
    fits = load_fit_rows(
        args.conceptor_root,
        pathways=pathways,
        agg_mode=args.agg_mode,
        cell_ids=None if args.cell_ids is None else set(args.cell_ids),
    )
    conditions = build_eval_conditions(
        fits,
        selected_seeds=args.selected_seeds,
        run_root=args.run_root,
        run_tag=args.run_tag,
        betas=betas,
        alpha_policy=args.alpha_policy,
        alpha=args.alpha,
        max_episodes_per_cell=args.max_episodes_per_cell,
        port=args.port,
        profile=args.profile,
        docker_service=args.docker_service,
        robocasa_container=args.robocasa_container,
        device=args.device,
        container_repo_root=args.container_repo_root,
    )
    output_tsv = args.output_tsv or args.run_root / "steer_eval" / args.run_tag / "eval_matrix.tsv"
    write_eval_matrix(output_tsv, conditions)
    print(f"wrote {len(conditions)} rows -> {output_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

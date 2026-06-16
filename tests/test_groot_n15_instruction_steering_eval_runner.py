from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "safe"
    / "groot_n15"
    / "robocasa"
    / "steer"
    / "run_instruction_steering_eval.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "groot_n15_instruction_steering_eval_runner_under_test",
        RUN_SCRIPT,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_eval_matrix(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "cell_id\tcell_index\ttask\tcondition\tpathway\tbeta\talpha\tsteering_npz\t"
        "server_command\tcollect_command\n"
        "open_drawer_right\t7\tOpenDrawer\tbaseline\tbaseline\t\t\t\t"
        "docker compose run --rm --no-deps -T lerobot python /temporal_vla/scripts/serve/lerobot.py "
        "--profile /temporal_vla/profile.yaml --device cuda --host 127.0.0.1 --port 8400 --collect --capture-vl\t"
        "docker exec -e MUJOCO_GL=egl robocasa python /temporal_vla/scripts/safe/groot_n15/robocasa/collect/collect_instruction_fixed_http_features.py "
        "--selected-seeds /temporal_vla/selected.tsv --output-dir /temporal_vla/out/baseline/raw_rollouts "
        "--ep-meta-dir /temporal_vla/out/baseline/ep_meta --vla-server http://127.0.0.1:8400 "
        "--cell-id open_drawer_right --max-episodes-per-cell 1\n"
        "open_drawer_right\t7\tOpenDrawer\tvl_a1_b0p1\tvl\t0.1\t1\t/temporal_vla/conceptors.npz\t"
        "docker compose run --rm --no-deps -T lerobot python /temporal_vla/scripts/serve/lerobot.py "
        "--profile /temporal_vla/profile.yaml --device cuda --host 127.0.0.1 --port 8401 --collect --capture-vl "
        "--steering-npz /temporal_vla/conceptors.npz --steering-pathway vl --steering-beta 0.1 --steering-alpha 1 --steering-key C_steer\t"
        "docker exec -e MUJOCO_GL=egl robocasa python /temporal_vla/scripts/safe/groot_n15/robocasa/collect/collect_instruction_fixed_http_features.py "
        "--selected-seeds /temporal_vla/selected.tsv --output-dir /temporal_vla/out/vl/raw_rollouts "
        "--ep-meta-dir /temporal_vla/out/vl/ep_meta --vla-server http://127.0.0.1:8401 "
        "--cell-id open_drawer_right --max-episodes-per-cell 1\n"
    )


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def test_load_eval_rows_filters_by_pathway_and_row_index(tmp_path: Path):
    module = _load_module()
    matrix = tmp_path / "eval_matrix.tsv"
    _write_eval_matrix(matrix)

    rows = module.load_eval_rows(
        matrix,
        pathways={"vl"},
        row_indices={1},
    )

    assert len(rows) == 1
    assert rows[0].condition == "vl_a1_b0p1"
    assert rows[0].row_index == 1


def test_command_helpers_parse_port_and_container_output_dir():
    module = _load_module()

    assert module.command_port("python serve.py --port 8401 --collect") == 8401
    assert (
        module.server_container_name("docker compose run --name n15_eval server python serve.py")
        == "n15_eval"
    )
    assert module.collect_output_dir(
        "python collect.py --output-dir /temporal_vla/outputs/x/raw_rollouts"
    ) == REPO_ROOT / "outputs/x/raw_rollouts"


def test_dry_run_appends_result_rows(tmp_path: Path):
    module = _load_module()
    matrix = tmp_path / "eval_matrix.tsv"
    _write_eval_matrix(matrix)
    rows = module.load_eval_rows(matrix, row_indices={1})

    result = module.run_eval_row(
        rows[0],
        log_dir=tmp_path / "logs",
        health_timeout_s=1.0,
        dry_run=True,
    )
    results_tsv = tmp_path / "results.tsv"
    module.append_results(results_tsv, [result])

    written = _read_tsv(results_tsv)
    assert written[0]["status"] == "dry-run"
    assert written[0]["cell_id"] == "open_drawer_right"
    assert written[0]["condition"] == "vl_a1_b0p1"
    assert written[0]["output_dir"] == str(REPO_ROOT / "out/vl/raw_rollouts")


def test_wait_http_health_fails_fast_when_server_process_exits():
    module = _load_module()
    process = subprocess.Popen([sys.executable, "-c", "raise SystemExit(7)"])
    process.wait(timeout=5)

    try:
        module.wait_http_health(65530, timeout_s=10.0, interval_s=0.01, process=process)
    except RuntimeError as exc:
        assert "server process exited" in str(exc)
        assert "returncode=7" in str(exc)
    else:
        raise AssertionError("wait_http_health should fail when process already exited")

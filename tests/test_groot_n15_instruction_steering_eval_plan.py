from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "safe"
    / "groot_n15"
    / "robocasa"
    / "steer"
    / "plan_instruction_steering_eval.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "groot_n15_instruction_steering_eval_plan_under_test",
        PLAN_SCRIPT,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_fit_summary(root: Path, pathway: str) -> Path:
    summary = root / pathway / "truncated_w10" / "fit_summary.tsv"
    cell_dir = root / pathway / "truncated_w10" / "open_drawer_right"
    cell_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        cell_dir / "conceptors.npz",
        **{"alpha0.1_C_steer": np.eye(2), "alpha0.3_C_steer": np.eye(2)},
    )
    summary.write_text(
        "pathway\tagg_mode\tcell_id\tcell_index\ttask\tstatus\t"
        "n_success_episodes\tn_failure_episodes\tn_success_samples\tn_failure_samples\t"
        "feature_dim\tselected_alphas\tselection_mode\toutput_dir\treason\n"
        f"{pathway}\ttruncated_w10\topen_drawer_right\t7\tOpenDrawer\tfit\t"
        f"15\t35\t150\t350\t2\t0.1,0.3\tband\t{cell_dir}\t\n"
    )
    return summary


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _write_selected_manifest(path: Path, rows: list[tuple[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cell_index",
        "cell_id",
        "task",
        "env_name",
        "scenario_seed",
        "instruction",
        "canonical_instruction",
        "ep_meta_path",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for cell_id, scenario_seed in rows:
            writer.writerow(
                {
                    "cell_index": 7,
                    "cell_id": cell_id,
                    "task": "OpenDrawer",
                    "env_name": "robocasa_panda_omron/OpenDrawer_PandaOmron_Env",
                    "scenario_seed": scenario_seed,
                    "instruction": "Open the right drawer.",
                    "canonical_instruction": "Open the right drawer.",
                    "ep_meta_path": f"ep_meta/{cell_id}/{scenario_seed}.json",
                }
            )


def test_validate_selected_seed_no_overlap_rejects_cell_seed_leakage(tmp_path: Path):
    module = _load_module()
    eval_manifest = tmp_path / "heldout.tsv"
    training_manifest = tmp_path / "collection.tsv"
    _write_selected_manifest(eval_manifest, [("open_drawer_right", 400000)])
    _write_selected_manifest(training_manifest, [("open_drawer_right", 400000)])

    try:
        module.validate_selected_seed_no_overlap(
            selected_seeds=eval_manifest,
            forbidden_selected_seeds=[training_manifest],
        )
    except ValueError as exc:
        assert "open_drawer_right/400000" in str(exc)
    else:
        raise AssertionError("planner should reject eval/collection seed overlap")


def test_validate_selected_seed_no_overlap_allows_same_seed_in_other_cell(tmp_path: Path):
    module = _load_module()
    eval_manifest = tmp_path / "heldout.tsv"
    training_manifest = tmp_path / "collection.tsv"
    _write_selected_manifest(eval_manifest, [("open_drawer_right", 400000)])
    _write_selected_manifest(training_manifest, [("open_drawer_left", 400000)])

    module.validate_selected_seed_no_overlap(
        selected_seeds=eval_manifest,
        forbidden_selected_seeds=[training_manifest],
    )


def test_build_eval_conditions_expands_baseline_and_all_selected_alphas(tmp_path: Path):
    module = _load_module()
    conceptor_root = tmp_path / "conceptor"
    _write_fit_summary(conceptor_root, "dit")
    fits = module.load_fit_rows(
        conceptor_root,
        pathways=["dit"],
        agg_mode="truncated_w10",
        cell_ids=None,
    )

    conditions = module.build_eval_conditions(
        fits,
        selected_seeds=tmp_path / "selected.tsv",
        run_root=tmp_path / "run",
        run_tag="eval",
        betas=[0.1, 0.3],
        alpha_policy="all",
        alpha=None,
        max_episodes_per_cell=20,
        port=8400,
        profile="/temporal_vla/configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml",
        docker_service="lerobot",
        robocasa_container="robocasa",
        device="cuda",
        container_repo_root=Path("/temporal_vla"),
    )

    assert [c.condition for c in conditions] == [
        "baseline",
        "dit_a0p1_b0p1",
        "dit_a0p1_b0p3",
        "dit_a0p3_b0p1",
        "dit_a0p3_b0p3",
    ]
    baseline = conditions[0]
    assert baseline.pathway == "baseline"
    assert "--name temporal_vla_lerobot_eval_open_drawer_right_baseline" in baseline.server_command
    assert "--steering-npz" not in baseline.server_command
    steered = conditions[1]
    assert "--steering-npz" in steered.server_command
    assert "--name temporal_vla_lerobot_eval_open_drawer_right_dit_a0p1_b0p1" in steered.server_command
    assert "--steering-alpha 0.1" in steered.server_command
    assert "--steering-beta 0.1" in steered.server_command
    assert "/temporal_vla/" in steered.server_command
    assert "--cell-id open_drawer_right" in steered.collect_command
    assert "--max-episodes-per-cell 20" in steered.collect_command
    assert "/temporal_vla/" in steered.collect_command


def test_alpha_policy_first_uses_one_selected_alpha_per_pathway(tmp_path: Path):
    module = _load_module()
    conceptor_root = tmp_path / "conceptor"
    _write_fit_summary(conceptor_root, "dit")
    _write_fit_summary(conceptor_root, "vl")
    fits = module.load_fit_rows(
        conceptor_root,
        pathways=["dit", "vl"],
        agg_mode="truncated_w10",
        cell_ids={"open_drawer_right"},
    )

    conditions = module.build_eval_conditions(
        fits,
        selected_seeds=tmp_path / "selected.tsv",
        run_root=tmp_path / "run",
        run_tag="eval",
        betas=[0.3],
        alpha_policy="first",
        alpha=None,
        max_episodes_per_cell=5,
        port=8500,
        profile="/profile.yaml",
        docker_service="lerobot",
        robocasa_container="robocasa",
        device="cuda",
        container_repo_root=Path("/temporal_vla"),
    )

    assert [c.condition for c in conditions] == [
        "baseline",
        "dit_a0p1_b0p3",
        "vl_a0p1_b0p3",
    ]
    assert conditions[1].server_command != conditions[2].server_command
    assert "--steering-pathway dit" in conditions[1].server_command
    assert "--steering-pathway vl" in conditions[2].server_command


def test_aligned_dit_layer_pathway_maps_to_steering_layer(tmp_path: Path):
    module = _load_module()
    conceptor_root = tmp_path / "conceptor"
    _write_fit_summary(conceptor_root, "dit_layer0")
    fits = module.load_fit_rows(
        conceptor_root,
        pathways=["dit_layer0"],
        agg_mode="truncated_w10",
        cell_ids={"open_drawer_right"},
    )

    conditions = module.build_eval_conditions(
        fits,
        selected_seeds=tmp_path / "selected.tsv",
        run_root=tmp_path / "run",
        run_tag="eval",
        betas=[0.1],
        alpha_policy="first",
        alpha=None,
        max_episodes_per_cell=5,
        port=8500,
        profile="/profile.yaml",
        docker_service="lerobot",
        robocasa_container="robocasa",
        device="cuda",
        container_repo_root=Path("/temporal_vla"),
    )

    steered = conditions[1]
    assert steered.pathway == "dit_layer0"
    assert steered.condition == "dit_layer0_a0p1_b0p1"
    assert "--steering-pathway dit" in steered.server_command
    assert "--steering-layer 0" in steered.server_command
    assert "--steering-pathway dit_layer0" not in steered.server_command


def test_write_eval_matrix_outputs_commands(tmp_path: Path):
    module = _load_module()
    row = module.EvalCondition(
        cell_id="open_drawer_right",
        cell_index=7,
        task="OpenDrawer",
        condition="baseline",
        pathway="baseline",
        beta=None,
        alpha=None,
        steering_npz=None,
        server_command="serve",
        collect_command="collect",
    )

    out = tmp_path / "eval_matrix.tsv"
    module.write_eval_matrix(out, [row])

    rows = _read_tsv(out)
    assert rows[0]["cell_id"] == "open_drawer_right"
    assert rows[0]["beta"] == ""
    assert rows[0]["server_command"] == "serve"
    assert rows[0]["collect_command"] == "collect"


def test_main_rejects_forbidden_selected_overlap_before_writing_matrix(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    module = _load_module()
    conceptor_root = tmp_path / "conceptor"
    _write_fit_summary(conceptor_root, "dit")
    eval_manifest = tmp_path / "heldout.tsv"
    training_manifest = tmp_path / "collection.tsv"
    _write_selected_manifest(eval_manifest, [("open_drawer_right", 400000)])
    _write_selected_manifest(training_manifest, [("open_drawer_right", 400000)])
    output_tsv = tmp_path / "eval_matrix.tsv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plan_instruction_steering_eval.py",
            "--run-root",
            str(tmp_path / "run"),
            "--selected-seeds",
            str(eval_manifest),
            "--forbid-selected-overlap",
            str(training_manifest),
            "--conceptor-root",
            str(conceptor_root),
            "--pathway",
            "dit",
            "--output-tsv",
            str(output_tsv),
        ],
    )

    try:
        module.main()
    except ValueError as exc:
        assert "open_drawer_right/400000" in str(exc)
    else:
        raise AssertionError("planner should reject forbidden overlap before writing matrix")

    assert not output_tsv.exists()
    assert capsys.readouterr().out == ""

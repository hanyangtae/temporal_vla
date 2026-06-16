"""SAFE rollout collection verifier contract tests."""

from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFY_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "safe"
    / "groot_n16"
    / "robocasa"
    / "collect"
    / "verify_rollout_collection.py"
)


def _write_rollout_triplet(task_dir: Path, stem: str, payload: dict) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    with (task_dir / f"{stem}.pkl").open("wb") as f:
        pickle.dump(payload, f)
    (task_dir / f"{stem}.csv").write_text("eef_pos_x\n0.0\n")
    (task_dir / f"{stem}.mp4").write_bytes(b"video")


def test_verifier_accepts_n15_http_safe_triplet_contract(tmp_path: Path):
    task_dir = tmp_path / "CloseFridge"
    stem = "task0--ep0--succ1"
    payload = {
        "task_suite_name": "lerobot_groot_n15_robocasa",
        "model_family": "lerobot_groot_n15",
        "policy_transport": "http",
        "task_id": 0,
        "task_description": "close the fridge",
        "episode_idx": 0,
        "scenario_seed": 100000,
        "episode_success": 1,
        "ep_meta": {"lang": "close the fridge"},
        "hidden_states": [np.zeros((4, 16, 1024), dtype=np.float16)],
        "action_vectors": np.zeros((1, 12), dtype=np.float32),
        "feature_kind": "groot_n15_dit_action_tokens_pre_decode",
        "exported_action_token_count": 16,
        "feature_action_horizon": 16,
        "n_action_steps": 16,
        "max_episode_steps": 720,
        "video_fps": 20,
        "steps_per_render": 2,
        "inference_seed": 424242,
        "valid_action_horizon": None,
        "model_action_horizon": 16,
        "env_name": "robocasa_panda_omron/CloseFridge_PandaOmron_Env",
        "robocasa_env_source": "robocasa365",
        "video_source": "groot_upstream_video_recording_wrapper",
    }
    _write_rollout_triplet(task_dir, stem, payload)

    result = subprocess.run(
        [
            sys.executable,
            str(VERIFY_SCRIPT),
            str(tmp_path),
            "--tasks-override",
            "CloseFridge",
            "--episodes-per-task",
            "1",
            "--expected-feature-kind",
            "groot_n15_dit_action_tokens_pre_decode",
            "--expected-hidden-shape",
            "4,16,1024",
            "--expected-model-family",
            "lerobot_groot_n15",
            "--expected-policy-transport",
            "http",
            "--expected-task-suite-name",
            "lerobot_groot_n15_robocasa",
            "--expected-video-source",
            "groot_upstream_video_recording_wrapper",
            "--expected-model-horizon",
            "16",
            "--expected-valid-horizon",
            "none",
            "--expected-feature-action-horizon",
            "16",
            "--expected-n-action-steps",
            "16",
            "--expected-max-episode-steps",
            "720",
            "--expected-video-fps",
            "20",
            "--expected-steps-per-render",
            "2",
            "--expected-inference-seed",
            "424242",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "status=ok" in result.stdout


def test_verifier_accepts_n15_http_safe_triplet_with_required_vl(tmp_path: Path):
    task_dir = tmp_path / "CloseFridge"
    stem = "task0--ep0--succ1"
    payload = {
        "task_suite_name": "lerobot_groot_n15_robocasa",
        "model_family": "lerobot_groot_n15",
        "policy_transport": "http",
        "task_id": 0,
        "episode_idx": 0,
        "episode_success": 1,
        "hidden_states": [np.zeros((4, 16, 1024), dtype=np.float16)],
        "action_vectors": np.zeros((1, 12), dtype=np.float32),
        "feature_kind": "groot_n15_dit_action_tokens_pre_decode",
        "exported_action_token_count": 16,
        "feature_action_horizon": 16,
        "n_action_steps": 16,
        "valid_action_horizon": None,
        "model_action_horizon": 16,
        "robocasa_env_source": "robocasa365",
        "vl_hidden_states": [np.zeros((2048,), dtype=np.float16)],
        "vl_feature_kind": "groot_n15_vlln_seq_meanpool",
        "vl_feature_dim": 2048,
    }
    _write_rollout_triplet(task_dir, stem, payload)

    result = subprocess.run(
        [
            sys.executable,
            str(VERIFY_SCRIPT),
            str(tmp_path),
            "--tasks-override",
            "CloseFridge",
            "--episodes-per-task",
            "1",
            "--expected-feature-kind",
            "groot_n15_dit_action_tokens_pre_decode",
            "--expected-hidden-shape",
            "4,16,1024",
            "--expected-model-family",
            "lerobot_groot_n15",
            "--expected-policy-transport",
            "http",
            "--expected-task-suite-name",
            "lerobot_groot_n15_robocasa",
            "--expected-model-horizon",
            "16",
            "--expected-valid-horizon",
            "none",
            "--expected-feature-action-horizon",
            "16",
            "--expected-n-action-steps",
            "16",
            "--require-vl-hidden-states",
            "--expected-vl-hidden-shape",
            "2048",
            "--expected-vl-feature-kind",
            "groot_n15_vlln_seq_meanpool",
            "--expected-vl-feature-dim",
            "2048",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "status=ok" in result.stdout


def test_verifier_accepts_n15_instruction_cell_nested_layout(tmp_path: Path):
    config_path = tmp_path / "cells.tsv"
    canonical_instruction = "Open the right drawer."
    config_path.write_text(
        "cell_index\tcell_id\ttask\tenv_name\tcanonical_instruction\t"
        "target_episodes\tseed_search_start\n"
        "7\topen_drawer_right\tOpenDrawer\t"
        "robocasa_panda_omron/OpenDrawer_PandaOmron_Env\t"
        f"{canonical_instruction}\t1\t100000\n"
    )
    task_dir = tmp_path / "OpenDrawer" / "open_drawer_right"
    stem = "task7--ep0--succ1"
    payload = {
        "task_suite_name": "lerobot_groot_n15_robocasa",
        "model_family": "lerobot_groot_n15",
        "policy_transport": "http",
        "task_id": 7,
        "cell_id": "open_drawer_right",
        "cell_index": 7,
        "robocasa_task": "OpenDrawer",
        "canonical_instruction": canonical_instruction,
        "task_description": canonical_instruction,
        "episode_idx": 0,
        "scenario_seed": 100001,
        "episode_success": 1,
        "ep_meta": {"lang": canonical_instruction},
        "hidden_states": [np.zeros((4, 16, 1024), dtype=np.float16)],
        "action_vectors": np.zeros((1, 12), dtype=np.float32),
        "feature_kind": "groot_n15_dit_action_tokens_pre_decode",
        "exported_action_token_count": 16,
        "feature_action_horizon": 16,
        "n_action_steps": 16,
        "valid_action_horizon": None,
        "model_action_horizon": 16,
        "robocasa_env_source": "robocasa365",
        "vl_hidden_states": [np.zeros((2048,), dtype=np.float16)],
        "vl_feature_kind": "groot_n15_vlln_seq_meanpool",
        "vl_feature_dim": 2048,
    }
    _write_rollout_triplet(task_dir, stem, payload)

    result = subprocess.run(
        [
            sys.executable,
            str(VERIFY_SCRIPT),
            str(tmp_path),
            "--layout",
            "task_instruction",
            "--cell-config",
            str(config_path),
            "--tasks-override",
            "OpenDrawer",
            "--episodes-per-task",
            "1",
            "--expected-feature-kind",
            "groot_n15_dit_action_tokens_pre_decode",
            "--expected-hidden-shape",
            "4,16,1024",
            "--expected-model-family",
            "lerobot_groot_n15",
            "--expected-policy-transport",
            "http",
            "--expected-task-suite-name",
            "lerobot_groot_n15_robocasa",
            "--expected-model-horizon",
            "16",
            "--expected-valid-horizon",
            "none",
            "--expected-feature-action-horizon",
            "16",
            "--expected-n-action-steps",
            "16",
            "--require-vl-hidden-states",
            "--expected-vl-hidden-shape",
            "2048",
            "--expected-vl-feature-kind",
            "groot_n15_vlln_seq_meanpool",
            "--expected-vl-feature-dim",
            "2048",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "layout=task_instruction" in result.stdout
    assert "open_drawer_right\tOpenDrawer\tcount=1\tsuccess=1" in result.stdout
    assert "status=ok" in result.stdout


def test_verifier_requires_vl_hidden_states_only_when_requested(tmp_path: Path):
    task_dir = tmp_path / "CloseFridge"
    stem = "task0--ep0--succ1"
    payload = {
        "task_id": 0,
        "episode_idx": 0,
        "episode_success": 1,
        "hidden_states": [np.zeros((4, 16, 1024), dtype=np.float16)],
        "action_vectors": np.zeros((1, 12), dtype=np.float32),
        "feature_kind": "groot_n16_dit_valid_action_tokens_pre_velocity",
        "valid_action_horizon": 16,
        "model_action_horizon": 50,
        "robocasa_env_source": "robocasa365",
    }
    _write_rollout_triplet(task_dir, stem, payload)

    result = subprocess.run(
        [
            sys.executable,
            str(VERIFY_SCRIPT),
            str(tmp_path),
            "--tasks-override",
            "CloseFridge",
            "--episodes-per-task",
            "1",
            "--require-vl-hidden-states",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "missing vl_hidden_states" in result.stdout


def test_verifier_accepts_required_vl_hidden_states(tmp_path: Path):
    task_dir = tmp_path / "CloseFridge"
    stem = "task0--ep0--succ1"
    payload = {
        "task_id": 0,
        "episode_idx": 0,
        "episode_success": 1,
        "hidden_states": [
            np.zeros((7, 51, 1536), dtype=np.float16),
            np.zeros((7, 51, 1536), dtype=np.float16),
        ],
        "action_vectors": np.zeros((2, 12), dtype=np.float32),
        "feature_kind": "groot_n16_dit_block_residual_kmean_perT_multilayer",
        "valid_action_horizon": 16,
        "model_action_horizon": 50,
        "robocasa_env_source": "robocasa365",
        "vl_hidden_states": [
            np.zeros((2048,), dtype=np.float16),
            np.zeros((2048,), dtype=np.float16),
        ],
        "vl_feature_kind": "groot_n16_vlln_seq_meanpool",
        "vl_feature_dim": 2048,
    }
    _write_rollout_triplet(task_dir, stem, payload)

    result = subprocess.run(
        [
            sys.executable,
            str(VERIFY_SCRIPT),
            str(tmp_path),
            "--tasks-override",
            "CloseFridge",
            "--episodes-per-task",
            "1",
            "--expected-feature-kind",
            "groot_n16_dit_block_residual_kmean_perT_multilayer",
            "--expected-hidden-shape",
            "7,51,1536",
            "--require-vl-hidden-states",
            "--expected-vl-hidden-shape",
            "2048",
            "--expected-vl-feature-kind",
            "groot_n16_vlln_seq_meanpool",
            "--expected-vl-feature-dim",
            "2048",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "status=ok" in result.stdout


def test_verifier_accepts_block_residual_metadata_expectations(tmp_path: Path):
    task_dir = tmp_path / "CloseFridge"
    stem = "task0--ep0--succ1"
    payload = {
        "task_id": 0,
        "episode_idx": 0,
        "episode_success": 1,
        "hidden_states": [
            np.zeros((7, 51, 1536), dtype=np.float16),
            np.zeros((7, 51, 1536), dtype=np.float16),
        ],
        "action_vectors": np.zeros((2, 12), dtype=np.float32),
        "feature_kind": "groot_n15_dit_block_residual_tokens",
        "feature_axes": ["layer", "model_token", "feature_dim"],
        "capture_layers": [0, 2, 4, 8, 16, 24, 31],
        "layer_count": 7,
        "token_count": 51,
        "valid_action_horizon": None,
        "model_action_horizon": 16,
        "n_action_steps": 16,
        "robocasa_env_source": "robocasa365",
        "vl_hidden_states": [
            np.zeros((2048,), dtype=np.float16),
            np.zeros((2048,), dtype=np.float16),
        ],
        "vl_feature_kind": "groot_n15_vlln_seq_meanpool",
        "vl_feature_dim": 2048,
    }
    _write_rollout_triplet(task_dir, stem, payload)

    command = [
        sys.executable,
        str(VERIFY_SCRIPT),
        str(tmp_path),
        "--tasks-override",
        "CloseFridge",
        "--episodes-per-task",
        "1",
        "--expected-feature-kind",
        "groot_n15_dit_block_residual_tokens",
        "--expected-feature-axes",
        "layer,model_token,feature_dim",
        "--expected-hidden-shape",
        "7,51,1536",
        "--expected-model-horizon",
        "16",
        "--expected-valid-horizon",
        "none",
        "--expected-n-action-steps",
        "16",
        "--expected-capture-layers",
        "0,2,4,8,16,24,31",
        "--expected-layer-count",
        "7",
        "--expected-token-count",
        "51",
        "--require-vl-hidden-states",
        "--expected-vl-hidden-shape",
        "2048",
        "--expected-vl-feature-kind",
        "groot_n15_vlln_seq_meanpool",
        "--expected-vl-feature-dim",
        "2048",
    ]

    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "status=ok" in result.stdout

    bad_command = command.copy()
    bad_command[bad_command.index("51")] = "50"
    bad_result = subprocess.run(
        bad_command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert bad_result.returncode != 0
    assert "token_count mismatch" in bad_result.stdout

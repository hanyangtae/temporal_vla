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


def test_verifier_accepts_n15_http_safe_triplet_contract(tmp_path: Path):
    task_dir = tmp_path / "CloseFridge"
    task_dir.mkdir()
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
        "valid_action_horizon": None,
        "model_action_horizon": 16,
        "env_name": "robocasa_panda_omron/CloseFridge_PandaOmron_Env",
        "robocasa_env_source": "robocasa365",
        "video_source": "groot_upstream_video_recording_wrapper",
    }
    with (task_dir / f"{stem}.pkl").open("wb") as f:
        pickle.dump(payload, f)
    (task_dir / f"{stem}.csv").write_text("eef_pos_x\n0.0\n")
    (task_dir / f"{stem}.mp4").write_bytes(b"video")

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
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "status=ok" in result.stdout

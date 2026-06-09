from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


CANONICAL_GROOT_N15_ROBOCASA_FILES = {
    "scripts/safe/groot_n15/robocasa/README.md",
    "scripts/safe/groot_n15/robocasa/eval/native_official_zmq_eval.py",
    "scripts/safe/groot_n15/robocasa/eval/native_zmq_eval.py",
    "scripts/safe/groot_n15/robocasa/eval/lerobot_http_eval.py",
    "scripts/safe/groot_n15/robocasa/eval/internal_parity.py",
    "scripts/safe/groot_n15/robocasa/eval/run_target15_seedpairs.sh",
    "scripts/safe/groot_n15/robocasa/split/build_safe_splits.py",
    "scripts/safe/groot_n15/robocasa/split/merge_seen60_source.py",
    "scripts/safe/groot_n15/robocasa/utils/runtime.py",
    "scripts/safe/groot_n15/robocasa/utils/prepare_base_new_embodiment.py",
    "scripts/safe/groot_n15/robocasa/split/prepare_seen5_trainval_cp_test_split.py",
}

REMOVED_GROOT_N15_ROBOCASA_FILES = {
    "scripts/eval/groot_n15_official_robocasa_eval.py",
    "scripts/eval/groot_n15_robocasa_zmq_eval.py",
    "scripts/eval/lerobot_groot_n15_internal_parity.py",
    "scripts/eval/lerobot_groot_n15_official_robocasa_eval.py",
    "scripts/eval/run_groot_n15_target15_seedpairs.sh",
    "scripts/data/build_safe_groot_n15_splits.py",
    "scripts/data/merge_safe_groot_n15_seen60_source.py",
    "scripts/utils/lerobot_groot_n15_runtime.py",
    "scripts/utils/prepare_groot_n15_base_new_embodiment.py",
}


def test_groot_n15_robocasa_files_live_under_safe_tree() -> None:
    missing = [
        path
        for path in sorted(CANONICAL_GROOT_N15_ROBOCASA_FILES)
        if not (REPO_ROOT / path).is_file()
    ]
    assert missing == []


def test_old_groot_n15_robocasa_script_paths_are_removed() -> None:
    remaining = [
        path
        for path in sorted(REMOVED_GROOT_N15_ROBOCASA_FILES)
        if (REPO_ROOT / path).exists()
    ]
    assert remaining == []

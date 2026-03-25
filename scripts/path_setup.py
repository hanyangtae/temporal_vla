from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_UTILS_DIR = REPO_ROOT / "scripts" / "utils"
ROBOCASA_ROOT = REPO_ROOT / "src" / "benchmarks" / "robocasa"
ROBOSUITE_ROOT = REPO_ROOT / "src" / "benchmarks" / "robosuite"
DREAMVLA_ROOT = REPO_ROOT / "src" / "policies" / "dreamvla"
DATASETS_ROOT = REPO_ROOT / "src" / "datasets"


def _prepend(path: Path) -> None:
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)


def configure_repo_paths(
    *,
    include_script_utils: bool = False,
    include_robocasa: bool = False,
    include_dreamvla: bool = False,
    include_datasets: bool = False,
) -> Path:
    _prepend(REPO_ROOT)

    if include_script_utils:
        _prepend(SCRIPTS_UTILS_DIR)

    if include_robocasa:
        _prepend(ROBOCASA_ROOT)
        _prepend(ROBOSUITE_ROOT)

    if include_dreamvla:
        _prepend(DREAMVLA_ROOT)

    if include_datasets:
        _prepend(DATASETS_ROOT)

    return REPO_ROOT

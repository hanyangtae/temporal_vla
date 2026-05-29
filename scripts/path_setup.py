from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_UTILS_DIR = REPO_ROOT / "scripts" / "utils"
ROBOCASA_ROOT = REPO_ROOT / "src" / "benchmarks" / "robocasa"
ROBOSUITE_ROOT = REPO_ROOT / "src" / "benchmarks" / "robosuite"
DREAMVLA_ROOT = REPO_ROOT / "src" / "policies" / "dreamvla"
DATASETS_ROOT = REPO_ROOT / "src" / "datasets"
GROOT_ROOT = REPO_ROOT / "src" / "policies" / "Isaac-GR00T"
GROOT_ROBOCASA_V02_ROOT = GROOT_ROOT / "external_dependencies" / "robocasa"

# ── Cache roots (체크포인트/데이터셋 물리 저장소) ──────────────────────
# 체크포인트와 데이터셋은 repo 트리 밖 cache 로 분리한다. 컨테이너 안에서는
# docker-compose 가 호스트 cache 를 /cache 로 bind-mount 하고 VLA_CACHE_ROOT=/cache
# 를 주입한다. 호스트(uv venv 등)에서는 기본값 ~/.cache/temporal_vla 를 쓴다.
#   - CHECKPOINTS_ROOT: 베이스/사전학습/모듈 체크포인트 (학습 산출물은 outputs/ 에 둠)
#   - DATA_ROOT       : robocasa/datasets/eagle 등 학습·추출 데이터셋
CACHE_ROOT = Path(
    os.environ.get("VLA_CACHE_ROOT", Path.home() / ".cache" / "temporal_vla")
)
CHECKPOINTS_ROOT = CACHE_ROOT / "checkpoints"
DATA_ROOT = CACHE_ROOT / "datasets"


def _prepend(path: Path) -> None:
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)


def prepend_path(path: Path | str) -> None:
    _prepend(Path(path))


def configure_repo_paths(
    *,
    include_script_utils: bool = False,
    include_robocasa: bool = False,
    include_dreamvla: bool = False,
    include_datasets: bool = False,
    include_groot: bool = False,
    include_groot_robocasa_v02: bool = False,
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

    if include_groot:
        _prepend(GROOT_ROOT)

    if include_groot_robocasa_v02:
        _prepend(GROOT_ROBOCASA_V02_ROOT)

    return REPO_ROOT

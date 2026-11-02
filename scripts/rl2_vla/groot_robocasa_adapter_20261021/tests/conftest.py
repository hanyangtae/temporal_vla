import ast
import os
import sys
import textwrap
from pathlib import Path

import pytest

ADAPTER_DIR = Path(__file__).resolve().parents[1]
# mutation 점검용 /tmp 사본은 env 로 repo root 를 넘긴다 (기본: 이 파일 기준 4단계 위).
REPO_ROOT = Path(os.environ.get("GROOT_ADAPTER_REPO_ROOT") or Path(__file__).resolve().parents[4])
for p in (ADAPTER_DIR, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

LEROBOT_HEAD = (REPO_ROOT / "lerobot/src/lerobot/policies/groot/action_head/"
                "flow_matching_action_head.py")
UPSTREAM_HEAD = (REPO_ROOT / "src/policies/Isaac-GR00T-N1.5/gr00t/model/action_head/"
                 "flow_matching_action_head.py")
UPSTREAM_STATE_ACTION = (REPO_ROOT / "src/policies/Isaac-GR00T-N1.5/gr00t/data/transform/"
                         "state_action.py")


def extract_source(path: Path, name: str, kind=ast.FunctionDef) -> str:
    """실제 소스 파일에서 함수/클래스 정의만 꺼낸다 (무거운 import 없이 parity 비교용)."""
    text = path.read_text()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, kind) and node.name == name:
            return textwrap.dedent(ast.get_source_segment(text, node))
    raise LookupError(f"{name} not in {path}")


@pytest.fixture(scope="session")
def cached_checkpoint_dir():
    from contract import ContractError, resolve_cached_checkpoint_dir

    try:
        return resolve_cached_checkpoint_dir()
    except ContractError as exc:
        pytest.skip(f"cached checkpoint-120000 metadata 없음: {exc}")

"""exp4-2 분석 공용 로더 — pkl capture 모드 자동 디스패치 (bridge/diag 공유).

action_token_mean pkl → phase_separation.load_rollout, all_token_full pkl →
fit_phase_conceptor_n15.load_rollout_fulltoken(token_pool="mean") (fit 과 동일 경로 —
로더 이원화로 인한 수치 불일치 방지). 반환 dict 키: dit [n,L,D], phases, capture_layers,
success, (fulltoken 시 dit_k 등).
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]
for _p in (REPO, REPO / "scripts/safe/groot_n15/robocasa/analyze",
           REPO / "scripts/safe/groot_n15/robocasa/steer"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from fit_phase_conceptor_n15 import FULLTOKEN_MODE, load_rollout_fulltoken  # noqa: E402
from phase_separation import load_rollout  # noqa: E402


def load_roll_any(pkl_path: Path) -> dict:
    """capture_token_mode 디스패치 로더 (fit main() 의 로더 분기와 동일 규약)."""
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    if d.get("capture_token_mode") == FULLTOKEN_MODE:
        r = load_rollout_fulltoken(d, pkl_path, "mean")
        r.setdefault("success", int(d.get("episode_success", 0)))
        return r
    return load_rollout(pkl_path)


def read_manifest(path: str) -> list[tuple[Path, int]]:
    """fit_manifest.tsv 규약 (pkl\tlabel[\tscene], # 주석)."""
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        p = Path(parts[0]).expanduser()
        if not p.is_absolute():
            p = REPO / p
        rows.append((p, int(parts[1])))
    return rows


def read_record_start(path: str | None) -> dict[str, int]:
    """record_start.tsv 규약 (pkl\tstart)."""
    if not path:
        return {}
    out = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        p = Path(parts[0]).expanduser()
        if not p.is_absolute():
            p = REPO / p
        out[str(p.resolve())] = int(parts[1])
    return out

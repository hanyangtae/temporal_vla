"""연산자 저장 규약 헬퍼 — docs/04_data_storage_convention.md 의 코드 강제 지점.

연산자(conceptor / setM / SAE / direction)를 저장하는 모든 스크립트는 실체 파일
(`conceptors.npz` / `model.pt` / ...) 을 쓴 직후 :func:`write_operator_config` 를
호출해 같은 디렉토리에 ``config.json`` 을 남긴다.

왜 필요한가 (2026-08 아카이브 실측):

- 연산자 604 개 중 169 개가 메타 파일 없이 저장돼 무엇인지 식별 불가였다.
- 입력 rollout 기록이 있는 것은 100 개뿐. 나머지 504 개(10.44G)는 재현 불가라 폐기했다.
  ``exp4_1/fit_mean_diff.py`` 가 입력을 전혀 남기지 않아 그중 350 개가 나왔다.
- ``fit_phase_conceptor_n15.py`` 는 입력을 arm 루트의 ``fit_inputs.json`` 에만 남겨,
  연산자 디렉토리(``<arm>/<phase>/<layer>/``)만 옮기면 출처가 끊겼다.

규약: **입력 sig 목록은 연산자 디렉토리 안에 있어야 한다.** 상위 디렉토리에 두면
디렉토리 단위 이동·복사에서 분리된다.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

CONFIG_NAME = "config.json"


def input_fingerprint(sigs: Sequence[str]) -> str:
    """입력 집합의 지문. 순서 무관 — 정렬 후 해시."""
    payload = "\n".join(sorted(sigs))
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def write_operator_config(
    out_dir: str | Path,
    *,
    op_type: str,
    input_sigs: Iterable[str],
    params: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    """연산자 디렉토리에 ``config.json`` 을 쓴다.

    Parameters
    ----------
    out_dir:
        실체 파일(``conceptors.npz`` 등)이 있는 디렉토리.
    op_type:
        ``conceptor`` | ``setm`` | ``sae`` | ``direction``.
    input_sigs:
        입력 rollout 의 ``sig`` (pkl 의 sha256[:16]). **경로가 아니라 sig** —
        경로는 도커/머신이 바뀌면 끊어진다(기존 기록의 35% 가 그렇게 끊겼다).
    params:
        연산 파라미터 전부. 이것만으로 재현할 수 있어야 한다.
    extra:
        진단·부가 정보(수렴 지표 등). 재현에 불필요한 것은 여기에.

    Raises
    ------
    ValueError
        ``input_sigs`` 가 비었을 때. 출처 없는 연산자는 만들지 않는다.
    """
    sigs = [str(s) for s in input_sigs if s]
    if not sigs:
        raise ValueError(
            "input_sigs 가 비었다. 출처를 기록하지 않은 연산자는 저장하지 않는다 "
            "(docs/04 §1). 입력 rollout 의 sig 를 넘겨라."
        )
    if op_type not in ("conceptor", "setm", "sae", "direction"):
        raise ValueError(f"알 수 없는 op_type: {op_type!r}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg: dict[str, Any] = {
        "op_type": op_type,
        **params,
        "n_train_episodes": len(sigs),
        "train_episode_fingerprint": input_fingerprint(sigs),
        "train_episode_sigs": sigs,
    }
    if extra:
        cfg["extra"] = extra
    path = out / CONFIG_NAME
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False, sort_keys=False))
    return path


def read_operator_config(op_dir: str | Path) -> dict[str, Any] | None:
    """연산자 디렉토리의 ``config.json`` 을 읽는다. 없으면 ``None``."""
    p = Path(op_dir) / CONFIG_NAME
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

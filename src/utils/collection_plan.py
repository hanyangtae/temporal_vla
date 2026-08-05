"""수집 그리드 계획 — docs/04 §5 의 코드 강제 지점.

계획한 그리드(instruction × scene n × noise m)를 **수집 전에** 파일로 박아두고,
각 rollout 에 그리드 좌표(`scene_idx`·`noise_idx`)를 함께 기록한다.

왜 필요한가:

- 좌표가 없으면 `env_seed=100010` 이 그리드의 몇 번째 scene 인지 역산할 수 없다.
  1,200 판을 목표했는데 1,187 판만 있을 때 **무엇이 빠졌는지 알 수 없다.**
- 계획이 기록되지 않으면 "이 셀은 수집 실패인가, 애초에 계획에 없었나"를
  구분할 수 없다. 2026-08 정리에서 activation 526 판의 머신을 사후 복원하려다
  실패한 것과 같은 종류의 손실이다 — 사후에는 채울 수 없다.

사용:

    plan = CollectionPlan(
        name="n15_grid_v1", model="groot", version="n15",
        ckpt="lerobot_groot_n15__robocasa365_ckpt120000",
        capture_layers=[0, 2, 4, 8, 10, 12, 15], denoise_k=4, token_mode="all_token_full",
        instructions={"OpenDrawer/left": [100010, 100011, ...]},   # instruction -> scene seed 목록
        noise_seeds=[1300000, 1300001, ...],                        # 전 instruction 공통
    )
    plan.save(out_dir)
    for cell in plan.cells():        # (instruction, scene_idx, env_seed, noise_idx, inference_seed)
        ...  # 수집 실행, cell.as_metadata() 를 pkl extra_metadata 로
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

PLAN_NAME = "collection_plan.json"


@dataclass(frozen=True)
class GridCell:
    """그리드의 한 칸 = rollout 하나."""

    instruction: str
    scene_idx: int
    env_seed: int
    noise_idx: int
    inference_seed: int

    def as_metadata(self) -> dict[str, Any]:
        """pkl `extra_metadata` 로 실을 좌표. 인덱스의 동명 열이 된다."""
        return {
            "grid_instruction": self.instruction,
            "scene_idx": self.scene_idx,
            "noise_idx": self.noise_idx,
        }

    @property
    def key(self) -> str:
        return f"{self.instruction}|s{self.scene_idx}|n{self.noise_idx}"


@dataclass
class CollectionPlan:
    name: str
    model: str
    version: str
    ckpt: str
    capture_layers: list[int]
    denoise_k: int
    token_mode: str
    instructions: dict[str, list[int]]   # instruction -> scene seed 목록 (순서 = scene_idx)
    noise_seeds: list[int]               # 순서 = noise_idx
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def cells(self) -> Iterator[GridCell]:
        for instr, scenes in self.instructions.items():
            for si, env_seed in enumerate(scenes):
                for ni, inf_seed in enumerate(self.noise_seeds):
                    yield GridCell(instr, si, int(env_seed), ni, int(inf_seed))

    @property
    def n_cells(self) -> int:
        return sum(len(s) for s in self.instructions.values()) * len(self.noise_seeds)

    @property
    def plan_id(self) -> str:
        """계획의 지문 — 그리드가 바뀌면 값이 바뀐다."""
        payload = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def estimate_bytes(self, mb_per_layer_per_record: float = 0.66,
                       records_per_rollout: int = 94) -> int:
        """저장 예산 추정. 기본값은 2026-08 실측([7,4,49,1536] fp16, 판당 432MB)."""
        per_rollout = mb_per_layer_per_record * len(self.capture_layers) * records_per_rollout
        return int(per_rollout * self.n_cells * 1024 * 1024)

    def save(self, out_dir: str | Path) -> Path:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        body = {**asdict(self), "plan_id": self.plan_id, "n_cells": self.n_cells,
                "estimated_bytes": self.estimate_bytes()}
        p = d / PLAN_NAME
        p.write_text(json.dumps(body, indent=2, ensure_ascii=False))
        return p

    @staticmethod
    def load(path: str | Path) -> "CollectionPlan":
        p = Path(path)
        if p.is_dir():
            p = p / PLAN_NAME
        raw = json.loads(p.read_text())
        for k in ("plan_id", "n_cells", "estimated_bytes"):
            raw.pop(k, None)
        return CollectionPlan(**raw)

    def missing(self, collected: set[str]) -> list[GridCell]:
        """계획 대비 결손 셀. `collected` 는 수집된 셀의 :attr:`GridCell.key` 집합."""
        return [c for c in self.cells() if c.key not in collected]

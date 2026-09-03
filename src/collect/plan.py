"""수집 그리드 계획 — docs/04 §5 의 코드 강제 지점.

계획한 그리드(instruction × scene × [지터 k] × noise)를 **수집 전에** 파일로 박아두고,
각 rollout 에 그리드 좌표(`scene_idx`·`jitter_reset_idx`·`noise_idx`)를 함께 기록한다.

왜 필요한가:

- 좌표가 없으면 `env_seed=100010` 이 그리드의 몇 번째 scene 인지 역산할 수 없다.
  1,200 판을 목표했는데 1,187 판만 있을 때 **무엇이 빠졌는지 알 수 없다.**
- 계획이 기록되지 않으면 "이 셀은 수집 실패인가, 애초에 계획에 없었나"를
  구분할 수 없다. 2026-08 정리에서 activation 526 판의 머신을 사후 복원하려다
  실패한 것과 같은 종류의 손실이다 — 사후에는 채울 수 없다.

**축은 셋이다 (docs/04 §3.1.1, 2026-09-03 개정)**: base scene `s<i>` · 지터
`k<r>`(=`jitter_reset_idx`, ep_meta 고정 후 연속 reset 횟수) · noise `n<j>`.
지터 축이 없는 구 plan(v1·v2 계열)은 `jitter=None` 인 **legacy 2축** plan 이며
경로·키에서 `k` 층이 통째로 빠진다. 구 v3·v4 가 쓰던 "평탄 si = scene*100+k"
인코딩은 폐지됐다 — k 는 폴더층이지 scene 인덱스에 섞어 넣는 값이 아니다.

사용:

    plan = CollectionPlan(
        name="n15_grid_v5_scenario", model="groot", version="n15",
        ckpt="lerobot_groot_n15__robocasa365_ckpt120000",
        capture_layers=[0, 2, 4, 8, 10, 12, 15], denoise_k=4, token_mode="all_token_full",
        instructions={"OpenDrawer/left": [100010, 100011, ...]},   # instruction -> base scene seed 목록
        noise_seeds=[1300000, 1300001, ...],                        # 전 instruction 공통
        jitter={"OpenDrawer/left": [[1, 3, 4, 7, 9], ...]},         # [instr][scene_idx] = 채택 k 목록
    )
    plan.save(out_dir)
    for cell in plan.cells():        # (instruction, scene_idx, env_seed, noise_idx, inference_seed, k)
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
    """그리드의 한 칸 = rollout 하나.

    ``jitter_reset_idx`` 가 ``None`` 이면 legacy 2축 셀(구 v1·v2 plan)이다.
    """

    instruction: str
    scene_idx: int
    env_seed: int
    noise_idx: int
    inference_seed: int
    jitter_reset_idx: int | None = None

    def as_metadata(self) -> dict[str, Any]:
        """pkl `extra_metadata` 로 실을 좌표. 인덱스의 동명 열이 된다.

        ``scene_idx`` 는 **base scene 인덱스**다(평탄 인코딩 금지). 지터 축이 있는
        3축 plan 에서만 ``jitter_reset_idx`` 가 추가로 실린다 — legacy 셀에서는
        키 자체가 없어야 인덱스가 빈 값으로 구분할 수 있다.
        """
        meta: dict[str, Any] = {
            "grid_instruction": self.instruction,
            "scene_idx": self.scene_idx,
            "noise_idx": self.noise_idx,
        }
        if self.jitter_reset_idx is not None:
            meta["jitter_reset_idx"] = self.jitter_reset_idx
        return meta

    @property
    def key(self) -> str:
        """셀 키 — 3축 ``instr|s<i>|k<r>|n<j>`` / legacy ``instr|s<i>|n<j>``."""
        if self.jitter_reset_idx is None:
            return f"{self.instruction}|s{self.scene_idx}|n{self.noise_idx}"
        return (f"{self.instruction}|s{self.scene_idx}"
                f"|k{self.jitter_reset_idx}|n{self.noise_idx}")

    def rel_path(self, plan_id: str, machine: str) -> Path:
        """docs/04 §3.1.1 좌표 경로 — ``<plan_id>/<machine>/<instruction>/s<i>/k<r>/n<j>``.

        지터 축이 없는 legacy 셀은 ``k`` 층을 생략한다(``s<i>/n<j>``) — 인덱서·shipper
        는 양쪽을 다 읽는다.

        instruction 은 ``OpenDrawer/left`` 처럼 ``/`` 를 포함할 수 있어 그대로 하위 경로가
        된다. 경로 생성은 여기 하나뿐이다 — 수집기가 문자열을 조립하면 규약과 어긋난다.
        """
        p = Path(plan_id) / machine / self.instruction / f"s{self.scene_idx}"
        if self.jitter_reset_idx is not None:
            p = p / f"k{self.jitter_reset_idx}"
        return p / f"n{self.noise_idx}"


GRID_ARG_NAMES = ("grid_root", "plan_json", "scene_idx", "noise_idx")

# 지터 좌표 인자. GRID_ARG_NAMES 에 넣지 않는 이유: legacy 2축 plan 은 이 인자가 없어야
# 정상이고, 인자 자체는 수집기(`--jitter-reset-idx`)가 이미 갖고 있다(add_grid_args 가
# 중복 정의하면 argparse 충돌).
JITTER_ARG_NAME = "jitter_reset_idx"


def resolve_grid(args: Any) -> "tuple[CollectionPlan, GridCell] | tuple[None, None]":
    """CLI 인자 → (plan, cell). 좌표 인자가 온전할 때만 좌표 레이아웃을 쓴다.

    수집기(n15 HTTP · n16 ZMQ · 이후 cosmos 등)가 공유한다 — 좌표 해석이 수집기마다
    갈리면 같은 그리드가 다른 자리에 떨어진다. 경로 조립은 :meth:`GridCell.rel_path` 가,
    해석은 여기가 단일 출처다.

    지터 축(docs/04 §3.1.1)도 여기서 대조한다:

    - 3축 plan(``jitter`` 있음)인데 ``--jitter-reset-idx`` 가 없으면 **거부**한다.
      k 없이 수집하면 같은 (scene, noise) 셀에 서로 다른 상태가 겹쳐 쌓인다.
    - legacy 2축 plan 인데 ``--jitter-reset-idx`` 가 오면 **거부**한다. 계획에 없는
      축으로 수집된 판은 계획 대비 결손 계산에서 사라진다.

    계획에 없는 좌표는 거부한다(docs/04 §5.1 — 계획에 없는 셀은 수집하지 않는다).
    """
    vals = [getattr(args, n, None) for n in GRID_ARG_NAMES]
    if any(v is None for v in vals):
        return None, None
    grid_root, plan_json, scene_idx, noise_idx = vals
    del grid_root  # 경로 조립은 호출자가 rel_path 로 한다
    jitter_idx = getattr(args, JITTER_ARG_NAME, None)
    plan = CollectionPlan.load(plan_json)
    instr = getattr(args, "grid_instruction", None) or getattr(args, "canonical_instruction", None)
    if plan.jitter is not None and jitter_idx is None:
        raise ValueError(
            f"3축(지터) plan 인데 --jitter-reset-idx 가 없다: {plan_json} "
            "— docs/04 §3.1.1 은 k 를 좌표 폴더층으로 요구한다(k 없이 수집하면 같은 "
            "(scene, noise) 칸에 서로 다른 지터 상태가 겹쳐 쌓인다)."
        )
    if plan.jitter is None and jitter_idx is not None:
        raise ValueError(
            f"legacy 2축 plan 인데 --jitter-reset-idx={jitter_idx} 가 주어졌다: {plan_json} "
            "— 계획에 없는 축으로 수집하면 계획 대비 결손 계산에서 사라진다. "
            "지터를 쓰려면 jitter 를 가진 plan 을 쓸 것."
        )
    for cell in plan.cells():
        if (cell.scene_idx == scene_idx and cell.noise_idx == noise_idx
                and cell.jitter_reset_idx == jitter_idx
                and (instr is None or cell.instruction == instr)):
            return plan, cell
    kdesc = "" if jitter_idx is None else f"k{jitter_idx} "
    raise ValueError(
        f"계획에 없는 좌표: instruction={instr!r} s{scene_idx} {kdesc}n{noise_idx} "
        f"— {plan_json} 확인 (docs/04 §5.1: 계획에 없는 셀은 수집하지 않는다)"
    )


BASE_ARM = "base"

# armsig 에 반드시 들어가야 하는 개입 파라미터 (docs/04 §3.3).
# 하나라도 빠지면 서로 다른 arm 이 같은 지문을 얻는다 — §2 opsig 충돌(604 중 393)과 같은 사고.
ARM_PARAM_KEYS = (
    "op",                 # 연산자 종류 (conceptor | setpoint_seg | setpoint_vl | …)
    "bindings",           # [(phase, layer, opsig)] — layer 복수·phase 별 다른 NPZ 를 담는다
    "beta",               # 개입 정도
    "token_select",       # exp5-3: 같은 β 인데 full 0.025 vs future-only 0.350
    "denoise",            # global | per_step
    "steer_from_record",  # 개입 시점 (latch). gated 면 None
    "gated_phases",       # gated 개입 phase 집합. latch 면 None
)


def arm_signature(params: dict[str, Any]) -> str:
    """개입 파라미터 → armsig(8자). ``ARM_PARAM_KEYS`` 전량이 있어야 한다."""
    missing = [k for k in ARM_PARAM_KEYS if k not in params]
    if missing:
        raise ValueError(
            f"armsig 계산에 필요한 키 없음: {missing} — docs/04 §3.3 ARM_PARAM_KEYS 참조"
        )
    norm = {k: params[k] for k in ARM_PARAM_KEYS}
    payload = json.dumps(norm, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def add_grid_args(parser: Any) -> None:
    """수집기 공용 좌표 인자 (docs/04 §3). 넷을 다 주면 좌표 레이아웃, 아니면 구 레이아웃.

    ``--jitter-reset-idx`` 는 여기서 정의하지 않는다 — 수집기가 이미 갖고 있고(지터
    실행 방식 자체의 인자다) 여기서 또 만들면 argparse 가 충돌한다. :func:`resolve_grid`
    는 그 값을 ``args`` 에서 읽어 plan 과 대조한다.
    """
    parser.add_argument("--grid-root", default=None,
                        help="좌표 저장소 루트. 아래에 <plan_id>/<machine>/... 이 생긴다")
    parser.add_argument("--plan-json", default=None,
                        help="collection_plan.json 경로 (plan_id·좌표 역산)")
    parser.add_argument("--scene-idx", type=int, default=None,
                        help="그리드 base scene 좌표 (평탄 si 인코딩 금지)")
    parser.add_argument("--noise-idx", type=int, default=None, help="그리드 noise 좌표")
    parser.add_argument("--grid-instruction", default=None,
                        help="그리드 instruction 키. 미지정 시 --canonical-instruction 사용")
    parser.add_argument("--arm-dir", default=None,
                        help="arm 디렉토리명(arm_dirname 산출). 미지정 시 base")


def grid_dir_for(args: Any, plan: "CollectionPlan", cell: "GridCell", machine: str | None) -> Path:
    """좌표 + arm → 최종 쓰기 디렉토리. 수집기 공용."""
    arm = getattr(args, "arm_dir", None) or BASE_ARM
    return Path(args.grid_root) / cell.rel_path(plan.plan_id, machine or "unknown") / arm


def arm_dirname(armsig: str, hint: str = "") -> str:
    """``<armsig>__<hint>``. hint 는 사람용이고 진실은 config.json 이다(docs/04 §3.3)."""
    if armsig == BASE_ARM:
        return BASE_ARM
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in hint)
    return f"{armsig}__{safe}" if safe else armsig


@dataclass
class CollectionPlan:
    name: str
    model: str
    version: str
    ckpt: str
    capture_layers: list[int]
    denoise_k: int
    token_mode: str
    instructions: dict[str, list[int]]   # instruction -> base scene seed 목록 (순서 = scene_idx)
    noise_seeds: list[int]               # 순서 = noise_idx
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    # 지터 축 (docs/04 §3.1.1). jitter[instruction][scene_idx] = 채택 k 목록.
    # None 이면 legacy 2축 plan — plan_id 해시에서도 통째로 빠진다(구 plan_id 보존).
    jitter: dict[str, list[list[int]]] | None = None

    def cells(self) -> Iterator[GridCell]:
        """계획된 셀 전량. 3축이면 (instruction, scene, k, noise) 순서로 순회한다."""
        for instr, scenes in self.instructions.items():
            ks_per_scene = self._jitter_for(instr, len(scenes))
            for si, env_seed in enumerate(scenes):
                ks: list[int | None] = ([None] if ks_per_scene is None
                                        else [int(k) for k in ks_per_scene[si]])
                for k in ks:
                    for ni, inf_seed in enumerate(self.noise_seeds):
                        yield GridCell(instr, si, int(env_seed), ni, int(inf_seed), k)

    def _jitter_for(self, instr: str, n_scene: int) -> list[list[int]] | None:
        """instruction 의 scene 별 채택 k 목록. legacy plan 이면 None."""
        if self.jitter is None:
            return None
        if instr not in self.jitter:
            raise ValueError(
                f"jitter 에 instruction 이 없다: {instr!r} — 3축 plan 은 전 instruction 에 "
                "scene 별 채택 k 목록을 가져야 한다 (docs/04 §3.1.1)"
            )
        ks = self.jitter[instr]
        if len(ks) != n_scene:
            raise ValueError(
                f"jitter[{instr!r}] 길이 {len(ks)} != scene 수 {n_scene} — "
                "jitter[instr][scene_idx] 는 instructions[instr] 와 같은 길이여야 한다"
            )
        return ks

    @property
    def n_cells(self) -> int:
        if self.jitter is None:
            return sum(len(s) for s in self.instructions.values()) * len(self.noise_seeds)
        n = 0
        for instr, scenes in self.instructions.items():
            ks_per_scene = self._jitter_for(instr, len(scenes))
            assert ks_per_scene is not None
            n += sum(len(ks) for ks in ks_per_scene)
        return n * len(self.noise_seeds)

    @property
    def plan_id(self) -> str:
        """계획의 지문 — 그리드가 바뀌면 값이 바뀐다.

        ``jitter`` 가 없는 legacy plan 은 해시 payload 에서 그 키를 아예 뺀다 —
        지터 축 도입 이전에 발급된 plan_id(3134e339de4c 등)가 그대로 유지된다.
        """
        payload_obj = asdict(self)
        if self.jitter is None:
            payload_obj.pop("jitter", None)
        payload = json.dumps(payload_obj, sort_keys=True, ensure_ascii=False)
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
        if self.jitter is None:
            body.pop("jitter", None)   # legacy plan JSON 은 이 키를 갖지 않는다
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
        return CollectionPlan(**raw)   # jitter 키가 없으면 legacy(None)

    def missing(self, collected: set[str]) -> list[GridCell]:
        """계획 대비 결손 셀. `collected` 는 수집된 셀의 :attr:`GridCell.key` 집합."""
        return [c for c in self.cells() if c.key not in collected]

"""수집 그리드 계획 — docs/04 §5 의 코드 강제 지점.

계획한 그리드(instruction × scene × [지터 k] × noise)를 **수집 전에** 파일로 박아두고,
각 rollout 에 그리드 좌표(`scene_idx`·`jitter_reset_idx`·`noise_idx`)를 함께 기록한다.

왜 필요한가:

- 좌표가 없으면 `env_seed=100010` 이 그리드의 몇 번째 scene 인지 역산할 수 없다.
  1,200 판을 목표했는데 1,187 판만 있을 때 **무엇이 빠졌는지 알 수 없다.**
- 계획이 기록되지 않으면 "이 셀은 수집 실패인가, 애초에 계획에 없었나"를
  구분할 수 없다. 2026-08 정리에서 activation 526 판의 머신을 사후 복원하려다
  실패한 것과 같은 종류의 손실이다 — 사후에는 채울 수 없다.

**축은 셋이다 (docs/04 §3.1.1)**. v6(2026-09-03, handoff_20260903_grid_v6_scene_jitter)
부터 층 정의가 바뀌었다:

- **v6**: scene `s<sid>` = **주방(layout, style)** · jitter `j<jid>` = 같은 scene 의 세계
  변형 하나(연속 reset 횟수 + base 오프셋 lat/back, 정의는 plan 의 `jitters`) · noise `n<nid>`.
- **v5(legacy)**: scene `s<i>` = base seed · `k<r>`(=`jitter_reset_idx`) · noise `n<j>`.
- **legacy 2축**(v1·v2 계열): `jitter=None` — 경로·키에서 지터 층이 통째로 빠진다.

구 v3·v4 가 쓰던 "평탄 si = scene*100+k" 인코딩은 폐지됐다. legacy·v5 plan 은 읽기
호환을 유지하며 plan_id 도 불변이다(신규 필드가 None 이면 해시 payload 에서 제거).

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

    좌표 형태가 셋이다 (docs/04 §3.1.1):

    - **v6** (``jitter_idx`` 있음): scene = 주방(layout, style), jitter j = 세계 변형
      하나(연속 reset 횟수 + base 오프셋). 경로·키에 ``j<jid>`` 층이 온다.
    - **v5** (``jitter_reset_idx`` 만 있음): scene = base seed, k = 연속 reset 횟수.
      경로·키에 ``k<r>`` 층.
    - **legacy 2축** (둘 다 ``None``): 지터 축 없음.
    """

    instruction: str
    scene_idx: int
    env_seed: int
    noise_idx: int
    inference_seed: int
    # 연속 reset 횟수. legacy k 층에서는 k 값 자체, v6 에서는 jitters[..].reset_idx.
    jitter_reset_idx: int | None = None
    # v6 j 인덱스 = 경로 j<jid>. None 이면 v6 셀이 아니다.
    jitter_idx: int | None = None
    base_lat: float = 0.0     # v6 base 오프셋 (m) — fixture 중심 쪽이 양수
    base_back: float = 0.0    # v6 base 오프셋 (m) — 뒤쪽이 양수
    lang: str | None = None   # v6 scene 문장 (canonical instruction 대조용)
    side: str | None = None   # "left" | "right" | None
    layout_id: int | None = None
    style_id: int | None = None

    @property
    def is_v6(self) -> bool:
        """v6 (scene·jitter·noise) 셀인가."""
        return self.jitter_idx is not None

    def as_metadata(self) -> dict[str, Any]:
        """pkl `extra_metadata` 로 실을 좌표. 인덱스의 동명 열이 된다.

        ``scene_idx`` 는 **scene 인덱스**다(평탄 인코딩 금지). 축이 없는 셀에서는
        해당 키가 **아예 없어야** 인덱스가 빈 값으로 구분할 수 있다.
        """
        meta: dict[str, Any] = {
            "grid_instruction": self.instruction,
            "scene_idx": self.scene_idx,
            "noise_idx": self.noise_idx,
        }
        if self.jitter_reset_idx is not None:
            meta["jitter_reset_idx"] = self.jitter_reset_idx
        if self.is_v6:
            meta.update({
                "jitter_idx": self.jitter_idx,
                "base_lat": float(self.base_lat),
                "base_back": float(self.base_back),
                "side": self.side,
                "layout_id": self.layout_id,
                "style_id": self.style_id,
                "lang": self.lang,
            })
        return meta

    @property
    def key(self) -> str:
        """셀 키 — v6 ``instr|s<sid>|j<jid>|n<nid>`` / v5 ``instr|s<i>|k<r>|n<j>`` /
        legacy ``instr|s<i>|n<j>``."""
        if self.is_v6:
            return (f"{self.instruction}|s{self.scene_idx}"
                    f"|j{self.jitter_idx}|n{self.noise_idx}")
        if self.jitter_reset_idx is None:
            return f"{self.instruction}|s{self.scene_idx}|n{self.noise_idx}"
        return (f"{self.instruction}|s{self.scene_idx}"
                f"|k{self.jitter_reset_idx}|n{self.noise_idx}")

    def rel_path(self, plan_id: str, machine: str) -> Path:
        """docs/04 §3.1.1 좌표 경로.

        - v6: ``<plan_id>/<machine>/<instruction>/s<sid>/j<jid>/n<nid>``
        - v5: ``.../s<i>/k<r>/n<j>`` · legacy 2축: ``.../s<i>/n<j>``

        instruction 은 ``OpenDrawer/left`` 처럼 ``/`` 를 포함할 수 있어 그대로 하위 경로가
        된다. 경로 생성은 여기 하나뿐이다 — 수집기가 문자열을 조립하면 규약과 어긋난다.
        """
        p = Path(plan_id) / machine / self.instruction / f"s{self.scene_idx}"
        if self.is_v6:
            p = p / f"j{self.jitter_idx}"
        elif self.jitter_reset_idx is not None:
            p = p / f"k{self.jitter_reset_idx}"
        return p / f"n{self.noise_idx}"


GRID_ARG_NAMES = ("grid_root", "plan_json", "scene_idx", "noise_idx")

# 지터 좌표 인자. GRID_ARG_NAMES 에 넣지 않는 이유: legacy 2축 plan 은 이 인자가 없어야
# 정상이고, 인자 자체는 수집기(`--jitter-reset-idx`)가 이미 갖고 있다(add_grid_args 가
# 중복 정의하면 argparse 충돌).
JITTER_ARG_NAME = "jitter_reset_idx"

# v6 지터 좌표 인자(`--jitter-idx`). reset_idx·오프셋은 plan 이 갖고, 수집기는 j 인덱스만
# 넘긴다. add_grid_args 가 정의하지 않는 이유는 JITTER_ARG_NAME 과 같다(수집기 소유).
JITTER_IDX_ARG_NAME = "jitter_idx"


def resolve_grid(args: Any) -> "tuple[CollectionPlan, GridCell] | tuple[None, None]":
    """CLI 인자 → (plan, cell). 좌표 인자가 온전할 때만 좌표 레이아웃을 쓴다.

    수집기(n15 HTTP · n16 ZMQ · 이후 cosmos 등)가 공유한다 — 좌표 해석이 수집기마다
    갈리면 같은 그리드가 다른 자리에 떨어진다. 경로 조립은 :meth:`GridCell.rel_path` 가,
    해석은 여기가 단일 출처다.

    지터 축(docs/04 §3.1.1)도 여기서 대조한다:

    - **v6 plan**(``scenes``·``jitters`` 있음)인데 ``--jitter-idx`` 가 없으면 거부.
      ``--jitter-reset-idx`` 를 함께 주면 plan 이 정한 셀 값과 **같아야** 한다
      (수집기가 계획과 다른 지터를 돌면 좌표와 내용이 어긋난다).
    - v5 3축 plan(``jitter`` 있음)인데 ``--jitter-reset-idx`` 가 없으면 **거부**한다.
      k 없이 수집하면 같은 (scene, noise) 셀에 서로 다른 상태가 겹쳐 쌓인다.
    - legacy 2축 plan 인데 지터 인자가 오면 **거부**한다. 계획에 없는 축으로 수집된
      판은 계획 대비 결손 계산에서 사라진다.

    계획에 없는 좌표는 거부한다(docs/04 §5.1 — 계획에 없는 셀은 수집하지 않는다).
    """
    vals = [getattr(args, n, None) for n in GRID_ARG_NAMES]
    grid_root, plan_json, scene_idx, noise_idx = vals
    # grid_root 는 **쓰기 위치**일 뿐 셀 해석에는 불필요 — 캡처 OFF eval(좌표 트리에 안 씀)도
    # plan+좌표만으로 셀(지터 정의 포함)을 해석해야 한다(2026-09-03 v6: 아니면 지터가 무음 무시됨).
    if plan_json is None or scene_idx is None or noise_idx is None:
        return None, None
    del grid_root  # 경로 조립은 호출자가 rel_path 로 한다
    reset_idx = getattr(args, JITTER_ARG_NAME, None)      # --jitter-reset-idx
    j_idx = getattr(args, JITTER_IDX_ARG_NAME, None)      # --jitter-idx (v6)
    plan = CollectionPlan.load(plan_json)
    instr = getattr(args, "grid_instruction", None) or getattr(args, "canonical_instruction", None)

    if plan.is_v6:
        if j_idx is None:
            raise ValueError(
                f"v6 plan 인데 --jitter-idx 가 없다: {plan_json} — v6 좌표는 "
                "s<sid>/j<jid>/n<nid> 3층이며 j 는 plan 의 jitters 인덱스다 "
                "(reset 횟수·base 오프셋은 plan 이 갖는다)."
            )
        for cell in plan.cells():
            if (cell.scene_idx == scene_idx and cell.noise_idx == noise_idx
                    and cell.jitter_idx == j_idx
                    and (instr is None or cell.instruction == instr)):
                if reset_idx is not None and reset_idx != cell.jitter_reset_idx:
                    raise ValueError(
                        f"--jitter-reset-idx={reset_idx} 가 plan 의 셀 값 "
                        f"{cell.jitter_reset_idx} 와 다르다 ({cell.key}) — v6 는 plan 이 "
                        "reset 횟수의 단일 출처다. 인자를 빼거나 plan 값과 맞출 것."
                    )
                return plan, cell
        raise ValueError(
            f"계획에 없는 좌표: instruction={instr!r} s{scene_idx} j{j_idx} n{noise_idx} "
            f"— {plan_json} 확인 (docs/04 §5.1: 계획에 없는 셀은 수집하지 않는다)"
        )

    if j_idx is not None:
        raise ValueError(
            f"v6 plan 이 아닌데 --jitter-idx={j_idx} 가 주어졌다: {plan_json} — "
            "j 층은 scenes/jitters 를 가진 v6 plan 에서만 뜻이 있다."
        )
    if plan.jitter is not None and reset_idx is None:
        raise ValueError(
            f"3축(지터) plan 인데 --jitter-reset-idx 가 없다: {plan_json} "
            "— docs/04 §3.1.1 은 k 를 좌표 폴더층으로 요구한다(k 없이 수집하면 같은 "
            "(scene, noise) 칸에 서로 다른 지터 상태가 겹쳐 쌓인다)."
        )
    if plan.jitter is None and reset_idx is not None:
        raise ValueError(
            f"legacy 2축 plan 인데 --jitter-reset-idx={reset_idx} 가 주어졌다: {plan_json} "
            "— 계획에 없는 축으로 수집하면 계획 대비 결손 계산에서 사라진다. "
            "지터를 쓰려면 jitter 를 가진 plan 을 쓸 것."
        )
    for cell in plan.cells():
        if (cell.scene_idx == scene_idx and cell.noise_idx == noise_idx
                and cell.jitter_reset_idx == reset_idx
                and (instr is None or cell.instruction == instr)):
            return plan, cell
    kdesc = "" if reset_idx is None else f"k{reset_idx} "
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
    # ── v6 (2026-09-03, handoff_20260903_grid_v6_scene_jitter) ──────────────
    # scenes[key][sid] = {layout, style, side, lang, fixture_group, spawn_lat, ...}
    #   — sid 순서는 instructions[key] 의 env_seed 순서와 1:1 이다.
    # jitters[key][sid][jid] = {reset_idx, lat, back} — j 는 인덱스(값이 아니다).
    # 둘 다 None 이면 v6 plan 이 아니며, plan_id 해시·JSON 에서 키가 통째로 빠진다
    # (legacy·v5 plan_id 불변).
    scenes: dict[str, list[dict[str, Any]]] | None = None
    jitters: dict[str, list[list[dict[str, Any]]]] | None = None

    @property
    def is_v6(self) -> bool:
        """v6 plan 인가 (scene 주방 + jitter j 층)."""
        return self.scenes is not None and self.jitters is not None

    @property
    def env_kwargs(self) -> dict[str, Any]:
        """env 생성 인자 (``layout_and_style_ids`` 등). 수집기가 gym.make 에 전달한다.

        주방 목록이 바뀌면 seed→주방 추첨이 바뀌므로 **plan 이 단일 출처**여야 한다
        (legacy 5주방 목록으로 만든 v5 이하 좌표는 v6 과 호환되지 않는다).
        """
        return dict((self.extra or {}).get("env_kwargs") or {})

    def _v6_scene(self, instr: str, sid: int) -> dict[str, Any]:
        """v6 scenes[instr][sid] — 없으면 계약 위반이라 즉시 중단."""
        assert self.scenes is not None
        if instr not in self.scenes:
            raise ValueError(
                f"scenes 에 instruction 이 없다: {instr!r} — v6 plan 은 전 instruction 에 "
                "scene 목록을 가져야 한다"
            )
        scenes = self.scenes[instr]
        if not 0 <= sid < len(scenes):
            raise ValueError(f"scenes[{instr!r}] 범위 밖 sid={sid} (len={len(scenes)})")
        return scenes[sid]

    def _v6_jitters(self, instr: str, sid: int) -> list[dict[str, Any]]:
        """v6 jitters[instr][sid] — scene 수와 길이가 어긋나면 즉시 중단."""
        assert self.jitters is not None
        if instr not in self.jitters:
            raise ValueError(
                f"jitters 에 instruction 이 없다: {instr!r} — v6 plan 은 전 instruction 에 "
                "scene 별 jitter 목록을 가져야 한다"
            )
        per_scene = self.jitters[instr]
        n_scene = len(self.instructions[instr])
        if len(per_scene) != n_scene:
            raise ValueError(
                f"jitters[{instr!r}] 길이 {len(per_scene)} != scene 수 {n_scene} — "
                "jitters[instr][scene_idx] 는 instructions[instr] 와 같은 길이여야 한다"
            )
        return per_scene[sid]

    def cells(self) -> Iterator[GridCell]:
        """계획된 셀 전량.

        v6 = (instruction, scene, jitter j, noise), v5 = (instruction, scene, k, noise),
        legacy = (instruction, scene, noise) 순서.
        """
        if self.is_v6:
            yield from self._cells_v6()
            return
        for instr, scenes in self.instructions.items():
            ks_per_scene = self._jitter_for(instr, len(scenes))
            for si, env_seed in enumerate(scenes):
                ks: list[int | None] = ([None] if ks_per_scene is None
                                        else [int(k) for k in ks_per_scene[si]])
                for k in ks:
                    for ni, inf_seed in enumerate(self.noise_seeds):
                        yield GridCell(instr, si, int(env_seed), ni, int(inf_seed), k)

    def _cells_v6(self) -> Iterator[GridCell]:
        """v6 순회 — scene(주방) × jitter j × noise. env_seed 는 instructions[key][sid]."""
        for instr, seeds in self.instructions.items():
            for sid, env_seed in enumerate(seeds):
                scene = self._v6_scene(instr, sid)
                jits = self._v6_jitters(instr, sid)
                for jid, jit in enumerate(jits):
                    for ni, inf_seed in enumerate(self.noise_seeds):
                        yield GridCell(
                            instruction=instr,
                            scene_idx=sid,
                            env_seed=int(env_seed),
                            noise_idx=ni,
                            inference_seed=int(inf_seed),
                            jitter_reset_idx=int(jit["reset_idx"]),
                            jitter_idx=jid,
                            base_lat=float(jit.get("lat", 0.0) or 0.0),
                            base_back=float(jit.get("back", 0.0) or 0.0),
                            lang=scene.get("lang"),
                            side=scene.get("side"),
                            layout_id=scene.get("layout"),
                            style_id=scene.get("style"),
                        )

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
        if self.is_v6:
            n = 0
            for instr, seeds in self.instructions.items():
                n += sum(len(self._v6_jitters(instr, sid)) for sid in range(len(seeds)))
            return n * len(self.noise_seeds)
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

        ``None`` 인 신규 필드(``jitter``·``scenes``·``jitters``)는 해시 payload 에서
        아예 뺀다 — 그 축 도입 이전에 발급된 plan_id(legacy 3134e339de4c, v5
        e82e99cb666b 등)가 그대로 유지된다.
        """
        payload_obj = asdict(self)
        for key in ("jitter", "scenes", "jitters"):
            if getattr(self, key) is None:
                payload_obj.pop(key, None)
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
        for key in ("jitter", "scenes", "jitters"):
            if getattr(self, key) is None:
                body.pop(key, None)   # 그 축이 없는 plan JSON 은 키 자체를 갖지 않는다
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
        # jitter/scenes/jitters 키가 없으면 각각 None (legacy·v5 읽기 호환)
        return CollectionPlan(**raw)

    def missing(self, collected: set[str]) -> list[GridCell]:
        """계획 대비 결손 셀. `collected` 는 수집된 셀의 :attr:`GridCell.key` 집합."""
        return [c for c in self.cells() if c.key not in collected]

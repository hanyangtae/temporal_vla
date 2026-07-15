"""COAST conceptor steering hook (groot serve 용).

action head DiT 출력의 action token 을 multiplicative gate 로 steer 한다
(COAST A.9.2): ``M = (1-β)I + β·C_steer``, ``h' = h·Mᵀ``. 주입 지점은
``safe_hooks.py`` 의 groot 추출 지점과 동일한 ``action_head.model``(DiT) 출력이며,
출력 토큰 중 **마지막 action_horizon 개(action token)** 만 steer 한다.

steering matrix M 은 ``src.conceptor.build_steering_matrix`` 로 만든 [D,D] (D=1024).
conceptor NPZ(fit_conceptor_steering.py 산출)의 ``alpha{a}_C_steer`` 를 읽어 β 와
함께 M 으로 변환한다. β=0 이면 M=I → forward 무변경(baseline).

런타임 비용: forward step 마다 [B, horizon, D] @ [D, D] 한 번 (COAST B.2 와 동일, 무시 가능).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# src.conceptor 재사용 (repo root 가 path 에 있어야 함; serve 컨테이너는 /temporal_vla).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.conceptor import build_steering_matrix  # noqa: E402

__all__ = [
    "load_steering_matrix",
    "load_steering_matrices_per_step",
    "ConceptorSteering",
    "Pi05ConceptorSteering",
]


def load_steering_matrix(
    npz_path: str | Path,
    beta: float,
    *,
    alpha: float | None = None,
    key: str = "C_steer",
) -> np.ndarray:
    """conceptor NPZ 에서 C_steer 를 골라 steering matrix M=(1-β)I+β·C_steer 반환.

    Args:
        npz_path: fit_conceptor_steering.py 가 저장한 ``conceptors.npz``.
        beta: steering 강도 [0,1].
        alpha: 사용할 aperture. None 이면 sibling metadata.json 의 selected_alpha,
            그것도 없으면 NPZ 첫 키 (구 NPZ 는 키 순서가 비결정 — [[alpha-wiring-audit]]
            오배선 원인이라 어느 α 가 적용됐는지 preflight 로그로 반드시 남긴다).
        key: ``C_steer`` | ``C_success`` | ``C_failure`` (positive-only 실험은 C_success).

    Returns:
        M: (D, D) float64.
    """
    z = np.load(npz_path)
    steer_keys = [k for k in z.files if k.endswith(f"_{key}")]
    if not steer_keys:
        raise KeyError(f"{npz_path} 에 *_{key} 없음 (keys={z.files})")
    alpha_src = "explicit"
    if alpha is None:
        meta_path = Path(npz_path).with_name("metadata.json")
        if meta_path.exists():
            alpha = json.loads(meta_path.read_text()).get("selected_alpha")
            alpha_src = "meta"
    if alpha is not None:
        want = f"alpha{alpha:g}_{key}"
        if want not in steer_keys:
            raise KeyError(f"{want} 없음 (있는 키={steer_keys}, alpha_src={alpha_src})")
        chosen = want
    else:
        chosen = steer_keys[0]
        alpha_src = "first-key"
    # preflight 로그: 러너가 serve 로그의 이 라인을 arm manifest 와 대조 (불일치 시 rollout 전 실패)
    sha = hashlib.sha256(Path(npz_path).read_bytes()).hexdigest()[:12]
    print(f"[steer-preflight] npz={npz_path} key={chosen} alpha_src={alpha_src} "
          f"beta={beta:g} sha={sha}", flush=True)
    C = z[chosen].astype(np.float64)
    return build_steering_matrix(C, beta)


def load_steering_matrices_per_step(
    npz_path: str | Path,
    beta: float,
    *,
    alpha: float | None = None,
    key: str = "C_steer",
    num_steps: int | None = None,
) -> list[np.ndarray]:
    """Per-Step conceptor NPZ 에서 step별 M_k=(1-β)I+β·C_steer^{(k)} 리스트 반환.

    NPZ 키 계약 (pq3 fit --denoise per_step): ``step{k}_alpha{a}_{key}``, k=0..K-1.
    α 선택: explicit ``alpha``(전 step 공통) > sibling metadata.json 의
    ``selected_alpha_per_step``({"0": a0, ...}) > 없으면 KeyError (α 오배선 방지 —
    per-step 은 첫-키 폴백을 두지 않는다).

    Args:
        num_steps: 기대 step 수 (serve 의 num_inference_timesteps). 지정 시 불일치 abort.
    """
    import re

    z = np.load(npz_path)
    step_re = re.compile(rf"^step(\d+)_alpha([0-9.eE+-]+)_{re.escape(key)}$")
    by_step: dict[int, dict[str, str]] = {}
    for name in z.files:
        m = step_re.match(name)
        if m:
            by_step.setdefault(int(m.group(1)), {})[m.group(2)] = name
    if not by_step:
        raise KeyError(
            f"{npz_path} 에 step{{k}}_alpha*_{key} 없음 (per-step NPZ 아님? keys={z.files[:6]}...)"
        )
    steps = sorted(by_step)
    if steps != list(range(len(steps))):
        raise KeyError(f"step 인덱스 불연속: {steps} ({npz_path})")
    if num_steps is not None and len(steps) != int(num_steps):
        raise KeyError(
            f"per-step NPZ step 수 {len(steps)} != 기대 num_steps {num_steps} ({npz_path})"
        )
    alpha_per_step: dict[int, float] = {}
    alpha_src = "explicit"
    if alpha is not None:
        alpha_per_step = {k: float(alpha) for k in steps}
    else:
        meta_path = Path(npz_path).with_name("metadata.json")
        sel = None
        if meta_path.exists():
            sel = json.loads(meta_path.read_text()).get("selected_alpha_per_step")
        if not sel:
            raise KeyError(
                f"per-step α 미지정: --steering-alpha 도 없고 metadata.json 의 "
                f"selected_alpha_per_step 도 없음 ({npz_path})"
            )
        alpha_per_step = {int(k): float(v) for k, v in sel.items()}
        alpha_src = "meta"
    sha = hashlib.sha256(Path(npz_path).read_bytes()).hexdigest()[:12]
    mats: list[np.ndarray] = []
    for k in steps:
        a = alpha_per_step.get(k)
        if a is None:
            raise KeyError(f"step {k} 의 α 없음 (alpha_src={alpha_src}, {npz_path})")
        want = f"step{k}_alpha{a:g}_{key}"
        if want not in z.files:
            raise KeyError(
                f"{want} 없음 (있는 step{k} 키={sorted(by_step[k].values())}, "
                f"alpha_src={alpha_src})"
            )
        # preflight 로그: 러너가 step별 key 를 arm manifest 와 대조 (불일치 시 rollout 전 실패)
        print(
            f"[steer-preflight] npz={npz_path} denoise=per_step step={k} key={want} "
            f"alpha_src={alpha_src} beta={beta:g} sha={sha}",
            flush=True,
        )
        mats.append(build_steering_matrix(z[want].astype(np.float64), beta))
    return mats


PATHWAYS: tuple[str, ...] = ("dit", "vl")
TOKEN_SELECTS: tuple[str, ...] = ("last_horizon", "all")


class ConceptorSteering:
    """groot 의 hidden state 에 forward hook 으로 steering 을 거는 context manager.

    ``pathway`` 로 NOTALL pathway 를 고른다 — motor(DiT) vs goal(VL):

    - ``pathway="dit"`` (motor "how", COAST A.9.2 "forward hook on selected
      action-expert layer's output"). ``layer`` 로 세부 주입 지점:
        * ``layer=None``: ``action_head.model`` 출력(= DiT 최종, pre-velocity, D=1024).
        * ``layer=i``:    ``action_head.model.transformer_blocks[i]`` 출력(residual
          stream, D=input_embedding_dim=1536). h'=hMᵀ 후 나머지 block 통과.
      action token(마지막 ``horizon`` 개)만 steer. denoising step(K) 마다 1회 발화.
    - ``pathway="vl"`` (goal "what"). 주입 지점은 ``action_head.vlln`` 출력
      (post-LayerNorm VL features, D=backbone_embedding_dim=2048). DiT 가 cross-attend
      하는 바로 그 텐서다. ``get_action`` 당 **1회만** 발화하고 결과가 K denoising step
      전부로 전파된다 (gr00t_n1d6.py: vl_embeds 가 loop 밖에서 1회 계산·재사용). 따라서
      VL token 전체(``token_select="all"``)를 steer 하며 horizon slicing 안 함.

    Args:
        groot_model: ``Gr00tN1d6`` (``.action_head.{model, vlln}``).
        M: (D, D) steering matrix **또는 list[(D, D)]** (Per-Step — denoise call k 에
            M_seq[k] 적용, pq3 신규 배선). D 는 주입 지점 hidden dim 과 일치해야 함
            (dit_final=1024, dit_block=1536, vl=2048).
        pathway: ``"dit"`` | ``"vl"``. Per-Step(list)은 dit 전용 — vl 은 get_action 당
            1회만 발화하므로 list 를 주면 배선 오류로 ValueError.
        layer: dit pathway 의 transformer_block 인덱스. None 이면 DiT 최종 출력. vl 은 None.
        horizon: dit pathway 에서 steer 할 마지막 action token 수. None 이면
            action_head.action_horizon.
        token_select: ``"last_horizon"`` (dit 기본) | ``"all"`` (vl 기본). 명시 시 우선.

    Per-Step 규약: hook 은 자체 카운터 ``_k`` 로 요청 내 denoise call index 를 세고
    (call 마다 +1), serve 가 **요청 시작 시** ``reset_step_counter()`` 를 호출한다.
    카운터가 len(M_seq) 를 넘는 fire 는 RuntimeError (무음 오적용 방지 — fail loud).
    """

    def __init__(
        self,
        groot_model: Any,
        M: np.ndarray | list | tuple,
        *,
        pathway: str = "dit",
        layer: int | None = None,
        horizon: int | None = None,
        token_select: str | None = None,
    ):
        if pathway not in PATHWAYS:
            raise ValueError(f"Unsupported pathway: {pathway} (expected {PATHWAYS})")
        head = groot_model.action_head
        self.pathway = pathway
        self.layer = layer
        if pathway == "dit":
            if layer is None:
                self.module = head.model
            else:
                self.module = head.model.transformer_blocks[layer]
            default_select = "last_horizon"
        else:  # vl
            if layer is not None:
                raise ValueError("pathway='vl' 는 layer 를 받지 않는다 (vlln 단일 지점).")
            self.module = head.vlln
            default_select = "all"
        self.token_select = token_select or default_select
        if self.token_select not in TOKEN_SELECTS:
            raise ValueError(
                f"Unsupported token_select: {self.token_select} (expected {TOKEN_SELECTS})"
            )
        self.horizon = int(horizon if horizon is not None else head.action_horizon)
        self._handle = None
        self._k = 0
        self.set_matrices(M)

    # -- Per-Step 상태 ---------------------------------------------------------
    @property
    def per_step(self) -> bool:
        return len(self._M_seq) > 1

    @property
    def M(self) -> np.ndarray:
        """단일-M 호환 접근자 (구 코드가 ``hook.M`` 을 읽는 경로 보존)."""
        return self._M_seq[0]

    @M.setter
    def M(self, value: Any) -> None:
        self.set_matrices(value)

    def set_matrices(self, M: np.ndarray | list | tuple) -> None:
        """단일 (D,D) 또는 list[(D,D)] 를 설정하고 캐시·카운터를 리셋."""
        if isinstance(M, (list, tuple)):
            seq = [np.asarray(m) for m in M]
        else:
            arr = np.asarray(M)
            if arr.ndim == 3:  # [K, D, D] 스택도 허용
                seq = [arr[i] for i in range(arr.shape[0])]
            else:
                seq = [arr]
        if not seq:
            raise ValueError("empty steering matrix sequence")
        if len(seq) > 1 and self.pathway != "dit":
            raise ValueError(
                "Per-Step M_seq 는 pathway='dit' 전용 (vl 은 요청당 1회 발화)."
            )
        self._M_seq = seq
        self._Mt_cache: list[torch.Tensor | None] = [None] * len(seq)
        self._k = 0

    def reset_step_counter(self) -> None:
        """요청(get_action) 시작 시 serve 가 호출 — denoise call index 리셋."""
        self._k = 0

    def _hook(self, _module: Any, _args: tuple, output: Any) -> Any:
        is_tuple = isinstance(output, tuple)
        out = output[0] if is_tuple else output
        if self.per_step:
            k = self._k
            if k >= len(self._M_seq):
                raise RuntimeError(
                    f"per-step steering: denoise call index {k} >= M_seq "
                    f"{len(self._M_seq)} (요청 시작 reset 누락 또는 K 불일치 — "
                    "reset_step_counter() 배선을 확인하라)"
                )
            self._k += 1
        else:
            k = 0
        Mt = self._Mt_cache[k]
        if Mt is None or Mt.device != out.device or Mt.dtype != out.dtype:
            Mt = torch.as_tensor(self._M_seq[k], device=out.device, dtype=out.dtype)
            self._Mt_cache[k] = Mt
        steered = out.clone()
        # h' = h @ Mᵀ (마지막 D 축). token_select 로 적용 토큰 결정.
        if self.token_select == "last_horizon":
            steered[..., -self.horizon :, :] = steered[..., -self.horizon :, :] @ Mt.T
        else:  # "all" — 전체 토큰 (COAST 49토큰 정렬 / VL goal pathway)
            steered = steered @ Mt.T
        if is_tuple:
            return (steered, *output[1:])
        return steered

    def register(self) -> "ConceptorSteering":
        """forward hook 등록 (서버 수명 동안 영구 적용 시 사용)."""
        if self._handle is None:
            self._handle = self.module.register_forward_hook(self._hook)
        return self

    def unregister(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self) -> "ConceptorSteering":
        return self.register()

    def __exit__(self, *_exc: Any) -> bool:
        self.unregister()
        return False


class Pi05ConceptorSteering:
    """pi05 action expert 의 residual stream 에 forward hook 으로 steering 을 거는 CM.

    COAST A.7.1 global strategy: π0.5 action expert(Gemma2, 18 layer, d=1024) 의
    decoder layer ℓ(default 11) 출력에서 마지막 ``chunk_size`` action token 만
    ``h' = h·Mᵀ`` 로 steer 한다. 주입 지점은 ``ConceptorSteering`` 의 groot DiT block
    경로와 동등하게 ``policy.model.paligemma_with_expert.gemma_expert.model.layers[ℓ]``
    출력(residual stream)이며, denoise step(K) 마다 1회 발화한다.

    HF Gemma decoder layer 는 출력을 tuple ``(hidden_states, ...)`` 로 내므로 hook 은
    tuple 의 첫 원소만 steer 하고 나머지는 그대로 재조립한다.

    Args:
        policy: pi05 LeRobot policy (``.model.paligemma_with_expert.gemma_expert``,
            ``.model.config.chunk_size`` 를 노출).
        M: (D, D) steering matrix. D 는 expert hidden dim(=1024) 과 일치해야 한다.
        layer: steer 할 decoder layer 인덱스. 기본 11 (COAST default ℓ).
        chunk_size: steer 할 마지막 action token 수. None 이면
            ``policy.model.config.chunk_size``.
    """

    def __init__(
        self,
        policy: Any,
        M: np.ndarray,
        *,
        layer: int = 11,
        chunk_size: int | None = None,
    ):
        self.layer = int(layer)
        layers = policy.model.paligemma_with_expert.gemma_expert.model.layers
        self.module = layers[self.layer]
        if chunk_size is None:
            chunk_size = int(policy.model.config.chunk_size)
        self.chunk_size = int(chunk_size)
        self.M = np.asarray(M)
        self._Mt: torch.Tensor | None = None
        self._handle = None

    def _hook(self, _module: Any, _args: tuple, output: Any) -> Any:
        is_tuple = isinstance(output, tuple)
        out = output[0] if is_tuple else output
        if (
            self._Mt is None
            or self._Mt.device != out.device
            or self._Mt.dtype != out.dtype
        ):
            self._Mt = torch.as_tensor(self.M, device=out.device, dtype=out.dtype)
        steered = out.clone()
        # h' = h @ Mᵀ — 마지막 chunk_size action token 위치만 (앞쪽 토큰 불변).
        steered[..., -self.chunk_size :, :] = (
            steered[..., -self.chunk_size :, :] @ self._Mt.T
        )
        if is_tuple:
            return (steered, *output[1:])
        return steered

    def register(self) -> "Pi05ConceptorSteering":
        """forward hook 등록 (서버 수명 동안 영구 적용 시 사용)."""
        if self._handle is None:
            self._handle = self.module.register_forward_hook(self._hook)
        return self

    def unregister(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self) -> "Pi05ConceptorSteering":
        return self.register()

    def __exit__(self, *_exc: Any) -> bool:
        self.unregister()
        return False

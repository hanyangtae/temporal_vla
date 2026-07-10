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

__all__ = ["load_steering_matrix", "ConceptorSteering", "Pi05ConceptorSteering"]


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
        M: (D, D) steering matrix. D 는 주입 지점 hidden dim 과 일치해야 함
            (dit_final=1024, dit_block=1536, vl=2048).
        pathway: ``"dit"`` | ``"vl"``.
        layer: dit pathway 의 transformer_block 인덱스. None 이면 DiT 최종 출력. vl 은 None.
        horizon: dit pathway 에서 steer 할 마지막 action token 수. None 이면
            action_head.action_horizon.
        token_select: ``"last_horizon"`` (dit 기본) | ``"all"`` (vl 기본). 명시 시 우선.
    """

    def __init__(
        self,
        groot_model: Any,
        M: np.ndarray,
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
        # h' = h @ Mᵀ (마지막 D 축). token_select 로 적용 토큰 결정.
        if self.token_select == "last_horizon":
            steered[..., -self.horizon :, :] = steered[..., -self.horizon :, :] @ self._Mt.T
        else:  # "all" — VL token 전체 (goal pathway)
            steered = steered @ self._Mt.T
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

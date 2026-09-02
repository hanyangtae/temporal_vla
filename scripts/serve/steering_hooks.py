"""COAST conceptor steering hook (groot serve 용).

action head DiT 출력의 action token 을 multiplicative gate 로 steer 한다
(COAST A.9.2): ``M = (1-β)I + β·C_steer``, ``h' = h·Mᵀ``. 주입 지점은
``safe_hooks.py`` 의 groot 추출 지점과 동일한 ``action_head.model``(DiT) 출력이며,
출력 토큰 중 **마지막 action_horizon 개(action token)** 만 steer 한다.

steering matrix M 은 ``src.conceptor.build_steering_matrix`` 로 만든 [D,D] (D=1024).
conceptor NPZ(구 n16 fit 산출 — 파일은 archive 됨, NPZ 계약은 유지)의 ``alpha{a}_C_steer`` 를 읽어 β 와
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
    "load_steering_setpoint",
    "load_cond_guidance",
    "ConceptorSteering",
    "SetpointSteering",
    "CondGuidanceSteering",
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
        npz_path: conceptor fit 이 저장한 ``conceptors.npz``.
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

    NPZ 키 계약 (exp3(구 pq3) fit --denoise per_step): ``step{k}_alpha{a}_{key}``, k=0..K-1.
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


def load_steering_setpoint(
    npz_path: str | Path,
    *,
    alpha: float | None = None,
) -> tuple[np.ndarray, float]:
    """setpoint형 mean-diff(setM) NPZ 에서 (r̂, s) 를 반환 (exp4-1, docs/steering/24a §4.1).

    NPZ 키 계약 (fit_mean_diff.py 산출): ``alpha{a}_v_steer`` [D] 단위벡터 +
    ``alpha{a}_s`` 스칼라(성공 평균의 r̂ 좌표 = setpoint). α 선택 규칙은
    ``load_steering_matrix`` 와 동일 (explicit > metadata selected_alpha > 첫 키).
    β 는 hook(``SetpointSteering``) 쪽 인자 — M 에 굽는 conceptor 경로와 달리
    적용식 h' = h − β[(h·r̂)−s]r̂ 에서 실행 시 곱한다.
    """
    z = np.load(npz_path)
    v_keys = [k for k in z.files if k.endswith("_v_steer")]
    if not v_keys:
        raise KeyError(f"{npz_path} 에 *_v_steer 없음 (keys={z.files})")
    alpha_src = "explicit"
    if alpha is None:
        meta_path = Path(npz_path).with_name("metadata.json")
        if meta_path.exists():
            alpha = json.loads(meta_path.read_text()).get("selected_alpha")
            alpha_src = "meta"
    if alpha is not None:
        chosen = f"alpha{alpha:g}_v_steer"
        if chosen not in v_keys:
            raise KeyError(f"{chosen} 없음 (있는 키={v_keys}, alpha_src={alpha_src})")
    else:
        chosen = v_keys[0]
        alpha_src = "first-key"
    s_key = chosen.replace("_v_steer", "_s")
    if s_key not in z.files:
        raise KeyError(f"{npz_path} 에 {s_key} 없음 (setpoint 스칼라 누락)")
    v = z[chosen].astype(np.float64).reshape(-1)
    nrm = float(np.linalg.norm(v))
    if not (0.99 < nrm < 1.01):
        raise ValueError(f"v_steer 는 단위벡터여야 함: ‖v‖={nrm:.4f} ({npz_path})")
    s = float(np.asarray(z[s_key]).reshape(()))
    sha = hashlib.sha256(Path(npz_path).read_bytes()).hexdigest()[:12]
    print(f"[steer-preflight] npz={npz_path} key={chosen} alpha_src={alpha_src} "
          f"op=setpoint s={s:.4f} dim={v.shape[0]} sha={sha}", flush=True)
    return v, s


def load_steering_segment(npz_path: str | Path):
    """세그먼트 setpoint NPZ → (v_seg [S,D], s_tok [T], seg_bounds [S,2], seg_mask [S]).

    exp4-1 v2 규약 (2026-07-23, fit_mean_diff.save_segment_npz 산출):
    방향은 토큰 **종류(state/future/action)별**, setpoint 는 토큰 **위치별**.
    구 pooled 규약(단일 r̂ + 스칼라 s)은 fit 공간(49토큰 평균)과 적용 공간(action 16토큰)이
    달라 β=1 이 최대 4σ 오프매니폴드 이동을 일으켰다 — 그 회귀의 교정본.
    """
    z = np.load(npz_path)
    need = ("alpha0_v_seg", "alpha0_s_tok", "alpha0_seg_bounds", "alpha0_seg_mask")
    missing = [k for k in need if k not in z.files]
    if missing:
        raise KeyError(f"{npz_path}: 세그먼트 키 누락 {missing} (keys={z.files})")
    v_seg = z["alpha0_v_seg"].astype(np.float64)
    s_tok = z["alpha0_s_tok"].astype(np.float64).reshape(-1)
    bounds = z["alpha0_seg_bounds"].astype(int)
    mask = z["alpha0_seg_mask"].astype(np.float64).reshape(-1)
    nrms = np.linalg.norm(v_seg, axis=1)
    if not np.all((nrms > 0.99) & (nrms < 1.01)):
        raise ValueError(f"v_seg 각 행은 단위벡터여야 함: ‖v‖={nrms} ({npz_path})")
    if int(bounds[-1, 1]) != s_tok.shape[0]:
        raise ValueError(f"seg_bounds 끝 {bounds[-1, 1]} != T {s_tok.shape[0]}")
    sha = hashlib.sha256(Path(npz_path).read_bytes()).hexdigest()[:12]
    print(f"[steer-preflight] npz={npz_path} op=setpoint_seg S={v_seg.shape[0]} "
          f"T={s_tok.shape[0]} dim={v_seg.shape[1]} mask={mask.tolist()} "
          f"s_tok[min,max]=[{s_tok.min():.2f},{s_tok.max():.2f}] sha={sha}", flush=True)
    return v_seg, s_tok, bounds, mask


PATHWAYS: tuple[str, ...] = ("dit", "vl")
# "future": setM future_only 정렬 — future 세그먼트 토큰([1 : T-horizon]) 만 steer
# (state[0:1]·action[마지막 horizon] 제외). conceptor_*_future_only arm 용.
TOKEN_SELECTS: tuple[str, ...] = ("last_horizon", "all", "future")


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
            M_seq[k] 적용, exp3 신규 배선). D 는 주입 지점 hidden dim 과 일치해야 함
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
        elif self.token_select == "future":
            # future 세그먼트만 ([1 : T-horizon]) — state[0:1]·action[마지막 horizon] 제외.
            # setM future_only(seg_mask[0,1,0]) 와 정렬한 conceptor future_only.
            _t = steered.shape[-2]
            steered[..., 1 : _t - self.horizon, :] = (
                steered[..., 1 : _t - self.horizon, :] @ Mt.T
            )
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


class SetpointSteering:
    """setpoint형 mean-diff(setM) affine hook (exp4-1, docs/steering/24a §4.1).

    적용식: ``h' = h − β[(h·r̂) − s]·r̂`` — 오차 비례 개입, (h·r̂)=s 도달 시 개입량 0
    (자기 소멸), β≤1 이면 목표 초과 불가. 선행: ACE(2411.09003)·LEACE·WA-LQR setpoint.
    conceptor 경로(h'=hMᵀ, β를 M 에 굽기)와 달리 bias 항이 있는 affine 이라 별도 hook.

    주입 지점·발화 규약은 ``ConceptorSteering`` 의 dit 경로와 동일
    (``transformer_blocks[layer]`` 출력 residual stream D=1536, denoise call 마다 발화,
    ``last_horizon`` 이면 마지막 action token 만). per-step vec 시퀀스는 미지원.

    ★ ``pathway="vl"`` (exp5-2 setM VL 확장): 주입 지점은 ``action_head.vlln`` 출력
    (D=2048, get_action 당 1회 발화 — ConceptorSteering vl 경로와 동일 지점).
    fit 이 record 당 **VL 토큰-평균**(캡처 ``vl_hidden_states`` = vlln seq-mean-pool)
    공간에서 이뤄지므로 적용도 **토큰-평균 이동**으로 한다::

        m = mean_t h_t,   δ = β[(m·r̂) − s],   h'_t = h_t − δ·r̂  (모든 토큰 공통)

    → 토큰 평균이 정확히 setpoint 쪽으로 이동하고 토큰 내 분산은 보존된다.
    per-token 개별 setpoint 적용(h'_t = h_t − β[(h_t·r̂)−s]r̂)은 fit 공간(토큰 평균)과
    적용 공간(개별 토큰)이 달라 **금지** — exp4-1 v1 pooled 회귀와 동종.

    phase-gated 스위칭 API: ``set_vector(v, s)`` 활성 / ``set_vector(None)`` 비활성.
    비활성이면 hook 은 **출력 텐서를 그대로 반환** (clone 없음) — no-hook 과 구성상
    동일해 off≡identity smoke(24a §8-2·4)가 구조적으로 성립한다.
    ``reset_step_counter`` 는 registry 폴리모픽 순회 호환용 no-op.
    """

    def __init__(
        self,
        groot_model: Any,
        vec_s: tuple[np.ndarray, float] | None,
        beta: float,
        *,
        pathway: str = "dit",
        layer: int | None = None,
        horizon: int | None = None,
        token_select: str | None = None,
    ):
        if pathway not in PATHWAYS:
            raise ValueError(
                f"SetpointSteering pathway 는 {PATHWAYS} (got {pathway})"
            )
        head = groot_model.action_head
        self.pathway = pathway
        self.layer = layer
        self.beta = float(beta)
        self.expected_dim: int | None = None
        if pathway == "vl":
            # exp5-2: vlln 출력(토큰-평균 공간에서 fit) — layer 없음, 전 토큰 공통 이동.
            if layer is not None:
                raise ValueError("pathway='vl' 는 layer 를 받지 않는다 (vlln 단일 지점).")
            if token_select not in (None, "all"):
                raise ValueError(
                    "pathway='vl' setpoint 은 token_select='all' 전용 (토큰-평균 이동)."
                )
            self.module = head.vlln
            self.token_select = "all"
            _ns = getattr(head.vlln, "normalized_shape", None)
            if _ns:
                self.expected_dim = int(_ns[-1])
        else:
            self.module = head.model if layer is None else head.model.transformer_blocks[layer]
            self.token_select = token_select or "last_horizon"
        if self.token_select not in TOKEN_SELECTS:
            raise ValueError(
                f"Unsupported token_select: {self.token_select} (expected {TOKEN_SELECTS})"
            )
        self.horizon = int(horizon if horizon is not None else head.action_horizon)
        self._handle = None
        self._seg = None
        self._seg_cache = None
        self.set_vector(*(vec_s if vec_s is not None else (None,)))

    @property
    def per_step(self) -> bool:
        return False

    def set_segment(self, spec) -> None:
        """세그먼트 연산자 활성화: spec=(v_seg [S,D], s_tok [T], bounds [S,2], mask [S]) 또는 None.

        토큰 위치 t 마다 그 세그먼트의 방향으로 h'_t = h_t − β·mask·[(h_t·r̂_seg)−s_t]r̂_seg.
        ★ mask 는 0/1 플래그가 아니라 **세그먼트별 게인 승수**(_apply_segment 참조):
          처치=1.0, future_only=0(스킵), 위약=dose-match 스케일(예 1.95). 0/1 로 강제하면
          위약 dose 매칭이 깨진다. set_vector(구 pooled)와 상호배타.
        """
        if spec is not None and self.pathway == "vl":
            raise ValueError("세그먼트 연산자는 pathway='dit' 전용 (vl 은 pooled setpoint).")
        if spec is None:
            self._seg = None
        else:
            v_seg, s_tok, bounds, mask = spec
            self._seg = (np.asarray(v_seg, dtype=np.float64),
                         np.asarray(s_tok, dtype=np.float64),
                         np.asarray(bounds, dtype=int),
                         np.asarray(mask, dtype=np.float64))
        self._v = None
        self._s = 0.0
        self._seg_cache = None
        self._vt_cache = None

    def set_vector(self, v: np.ndarray | None, s: float | None = None) -> None:
        """(r̂, s) 활성화 또는 None 비활성화. 텐서 캐시 리셋."""
        self._seg = None
        self._seg_cache = None
        if v is None:
            self._v = None
            self._s = 0.0
        else:
            if isinstance(v, (list, tuple)):
                raise ValueError("per-step vec 시퀀스는 미지원 (exp4-1 은 denoise global).")
            arr = np.asarray(v, dtype=np.float64).reshape(-1)
            if s is None:
                raise ValueError("set_vector(v, s): 활성화에는 setpoint s 필요.")
            if self.expected_dim is not None and arr.shape[0] != self.expected_dim:
                raise ValueError(
                    f"setpoint 벡터 차원 {arr.shape[0]} != 주입 지점 hidden dim "
                    f"{self.expected_dim} (pathway={self.pathway}) — NPZ/pathway 불일치"
                )
            self._v = arr
            self._s = float(s)
        self._vt_cache: torch.Tensor | None = None

    def reset_step_counter(self) -> None:
        """요청 시작 훅 호환용 no-op (setpoint 는 denoise step 무관 동일 적용)."""

    def _hook(self, _module: Any, _args: tuple, output: Any) -> Any:
        if self._v is None and self._seg is None:  # off — 원본 그대로 (no-hook 과 동일)
            return output
        is_tuple = isinstance(output, tuple)
        out = output[0] if is_tuple else output
        if self._seg is not None:
            steered = self._apply_segment(out)
            return (steered, *output[1:]) if is_tuple else steered
        vt = self._vt_cache
        if vt is None or vt.device != out.device or vt.dtype != out.dtype:
            vt = torch.as_tensor(self._v, device=out.device, dtype=out.dtype)
            self._vt_cache = vt
        if self.pathway == "vl":
            # 토큰-평균 이동 (fit 공간 = vlln seq-mean). δ 는 토큰 공통 스칼라라
            # 평균만 setpoint 로 옮기고 토큰 내 분산은 그대로 보존된다.
            if self.expected_dim is not None and out.shape[-1] != self.expected_dim:
                raise RuntimeError(
                    f"setpoint_vl: hook 텐서 dim {out.shape[-1]} != 기대 "
                    f"{self.expected_dim} — 주입 지점 불일치"
                )
            m = out.mean(dim=-2)                       # [..., D]
            proj = (m * vt).sum(dim=-1)                # [...]
            delta = self.beta * (proj - self._s)       # [...]
            steered = out - delta[..., None, None] * vt
            if is_tuple:
                return (steered, *output[1:])
            return steered
        steered = out.clone()
        if self.token_select == "last_horizon":
            hs = steered[..., -self.horizon :, :]
            proj = hs @ vt  # [..., horizon]
            steered[..., -self.horizon :, :] = hs - self.beta * (proj - self._s).unsqueeze(-1) * vt
        else:  # "all"
            proj = steered @ vt
            steered = steered - self.beta * (proj - self._s).unsqueeze(-1) * vt
        if is_tuple:
            return (steered, *output[1:])
        return steered

    def _apply_segment(self, out):
        """토큰 위치별 세그먼트 연산자 적용 — h'_t = h_t − β[(h_t·r̂_seg)−s_t]r̂_seg.

        토큰 축은 마지막에서 두 번째(out [..., T, D]). NPZ 의 T 와 실제 T 가 다르면
        무음 오적용이 되므로 RuntimeError (fail loud).
        """
        v_seg, s_tok, bounds, mask = self._seg
        T = out.shape[-2]
        if T != s_tok.shape[0]:
            raise RuntimeError(
                f"setpoint_seg: NPZ T={s_tok.shape[0]} != 실제 토큰 수 {T} — "
                "fit 캡처 토큰 규약(all_token_full)과 serve 주입 지점 불일치")
        cache = self._seg_cache
        if (cache is None or cache[0].device != out.device or cache[0].dtype != out.dtype):
            vt = torch.as_tensor(v_seg, device=out.device, dtype=out.dtype)      # [S,D]
            st = torch.as_tensor(s_tok, device=out.device, dtype=out.dtype)      # [T]
            # 토큰→세그먼트 인덱스, mask 를 토큰 단위로 펼침
            idx = torch.zeros(T, dtype=torch.long, device=out.device)
            mk = torch.zeros(T, device=out.device, dtype=out.dtype)
            for si, (lo, hi) in enumerate(bounds):
                idx[int(lo):int(hi)] = si
                # mask 는 세그먼트별 **게인 승수**(0=스킵·1=처치·위약=dose-match 스케일).
                # 아래 delta 에 그대로 곱해진다 — bool 로 강등하면 위약 dose 매칭이 깨짐.
                mk[int(lo):int(hi)] = float(mask[si])
            v_tok = vt[idx]                                                       # [T,D]
            self._seg_cache = cache = (v_tok, st, mk)
        v_tok, st, mk = cache
        proj = (out * v_tok).sum(dim=-1)                    # [..., T]
        delta = (self.beta * mk * (proj - st)).unsqueeze(-1) * v_tok
        return out - delta

    def register(self) -> "SetpointSteering":
        if self._handle is None:
            self._handle = self.module.register_forward_hook(self._hook)
        return self

    def unregister(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self) -> "SetpointSteering":
        return self.register()

    def __exit__(self, *_exc: Any) -> bool:
        self.unregister()
        return False


def load_cond_guidance(npz_path: str | Path) -> dict:
    """상태-조건부 대조 guidance(condg) NPZ → 파라미터 dict (docs/steering/44 §4).

    NPZ 키 계약 (fit_cond_guidance.py ``_save_npz`` 가 단일 출처 — ``__`` 구분):

    - ``{phase}__W_s`` [P, D], ``{phase}__W_f`` [P, D] — 릿지 계수
      (P=상태 차원 18 = raw 9 + 속도 9, D=주입 지점 hidden dim).
    - ``{phase}__tau`` 스칼라 — margin 게이트 임계 (held-out 성공 90퍼센타일).
    - ``{phase}__B`` 스칼라, ``{phase}__registered`` bool — 미등록 phase 는 identity.
    - ``{phase}__scene{sc}__mh`` [D] / ``__mp`` [P] / ``__sp`` [P] — **phase별**·scene별
      중심화/z-score (fit 이 phase 마다 stats 를 재계산).
    - ``{phase}__mh_global`` / ``__mp_global`` / ``__sp_global`` — 미지 scene 폴백 (필수).
    - ``phases`` / ``registered_phases`` / ``meta_json`` — 목록·메타.

    Returns:
        {"phases": {name: {"W_s","W_f","tau","B","registered",
                           "scenes": {int: {"mh","mp","sp"}},
                           "global": {"mh","mp","sp"}}}}   (모든 배열 float64)
    """
    z = np.load(npz_path, allow_pickle=False)
    if "phases" not in z.files:
        raise KeyError(f"{npz_path}: 'phases' 키 없음 (condg NPZ 아님? keys={z.files[:8]})")
    out_phases: dict[str, dict] = {}
    for ph in [str(p) for p in z["phases"]]:
        registered = bool(np.asarray(z[f"{ph}__registered"]).reshape(())) \
            if f"{ph}__registered" in z.files else False
        if f"{ph}__W_s" not in z.files:
            if registered:
                raise KeyError(f"{npz_path}: phase '{ph}' registered 인데 W_s 없음")
            continue  # 표본 미달로 W 자체가 없는 미등록 phase
        W_s = np.asarray(z[f"{ph}__W_s"], dtype=np.float64)
        W_f = np.asarray(z[f"{ph}__W_f"], dtype=np.float64)
        if W_s.shape != W_f.shape:
            raise ValueError(f"{npz_path}: phase '{ph}' W_s{W_s.shape} != W_f{W_f.shape}")
        for suf in ("mh_global", "mp_global", "sp_global"):
            if f"{ph}__{suf}" not in z.files:
                raise KeyError(f"{npz_path}: {ph}__{suf} 없음 (미지 scene 폴백 필수)")
        glob = {k: np.asarray(z[f"{ph}__{k}_global"], dtype=np.float64).reshape(-1)
                for k in ("mh", "mp", "sp")}
        scenes: dict[int, dict] = {}
        for sc in [int(s) for s in z[f"{ph}__scenes"]] if f"{ph}__scenes" in z.files else []:
            scenes[sc] = {k: np.asarray(z[f"{ph}__scene{sc}__{k}"],
                                        dtype=np.float64).reshape(-1)
                          for k in ("mh", "mp", "sp")}
        out_phases[ph] = {
            "W_s": W_s, "W_f": W_f,
            "tau": float(np.asarray(z[f"{ph}__tau"]).reshape(())),
            "B": int(np.asarray(z[f"{ph}__B"]).reshape(())) if f"{ph}__B" in z.files else None,
            "registered": registered,
            "scenes": scenes, "global": glob,
        }
    if not out_phases:
        raise KeyError(f"{npz_path}: W 보유 phase 0개 (전 phase 표본 미달?)")
    reg = sorted(p for p, v in out_phases.items() if v["registered"])
    sha = hashlib.sha256(Path(npz_path).read_bytes()).hexdigest()[:12]
    _any = next(iter(out_phases.values()))
    print(f"[steer-preflight] npz={npz_path} op=condg phases={sorted(out_phases)} "
          f"registered={reg} "
          f"state_dim={_any['W_s'].shape[0]} dim={_any['W_s'].shape[1]} sha={sha}",
          flush=True)
    return {"phases": out_phases}


COND_GUIDANCE_MODES: tuple[str, ...] = ("condg", "hs")


class CondGuidanceSteering:
    """상태-조건부 대조 guidance(condg) hook (docs/steering/44 §1·§5).

    성공/실패 각각의 상태→활성화 릿지 예측 ĥ_s=φ̃Wₛ, ĥ_f=φ̃W_f 로 현재 활성화가
    어느 쪽에 가까운지(margin m = ‖h̃−ĥ_s‖² − ‖h̃−ĥ_f‖²)를 재고, m>τ 일 때만
    실패 방향 성분을 깎는다::

        d̂ = normalize(ĥ_f − ĥ_s),  Δ = β·⟨h̃−ĥ_s, d̂⟩·d̂,  h ← h − Δ   (mode="condg")
        Δ = β·(h̃ − ĥ_s)  → h' = (1−β)h̃ + β·ĥ_s                       (mode="hs" ablation)

    setM(SetpointSteering)과 달리 setpoint 가 **상태 φ 의 함수**라 요청마다 갱신된다:
    serve 가 매 /act 에서 ``set_state(p9)`` 로 raw proprio 9차원(eef_pos_rel 3 +
    eef_quat_rel 4 + gripper_qpos 2)을 주입하고, 속도 9차원은 hook 내부 버퍼의
    직전 record 차분으로 만든다(첫 record=0, ``reset_state()`` 로 episode 경계 초기화).

    주입 지점·발화 규약:

    - ``action_head.model.transformer_blocks[layer]`` 출력 residual stream (dit 전용,
      기본 layer=12 — fit 이 L12 에서 이뤄짐).
    - **마지막 denoise call 한정**: fit 표적이 step K−1(=3, K=num_inference_timesteps)
      의 활성화라 그 call 에서만 적용한다. 요청 시작마다 serve 가 부르는
      ``reset_step_counter()`` 로 call index 를 리셋하고, hook 은 call 마다 +1 하며
      index==K−1 일 때만 발화한다. K 는 생성자에서 action_head 에서 읽는다.
    - margin·투영은 **전 토큰 mean 공간**에서 계산한다 (fit 이 49토큰 mean 이라
      τ 캘리브레이션이 그 공간에 묶여 있음). ``token_select="future"`` 는 계산 공간은
      그대로 두고 **Δ 를 빼는 토큰만** future 세그먼트([1:T−horizon])로 제한한다
      (setM all-token eef 진동 전례 대응 arm).

    phase/scene 스위칭: ``set_phase(phase, scene)``. 미등록 phase·None → 완전 identity
    (출력 텐서 그대로 반환, clone 없음 — off≡identity 가 구조적으로 성립).
    scene 이 NPZ 에 없으면 global fallback 중심화 파라미터를 쓴다.
    """

    def __init__(
        self,
        groot_model: Any,
        params: dict,
        beta: float,
        *,
        layer: int = 12,
        mode: str = "condg",
        token_select: str = "all",
        gate: bool = True,
        horizon: int | None = None,
        num_denoise: int | None = None,
        apply_call: str = "last",
    ):
        if apply_call not in ("last", "first"):
            raise ValueError(f"condg apply_call 은 ('last','first') (got {apply_call})")
        if mode not in COND_GUIDANCE_MODES:
            raise ValueError(f"condg mode 는 {COND_GUIDANCE_MODES} (got {mode})")
        if token_select not in ("all", "future"):
            raise ValueError(f"condg token_select 는 ('all','future') (got {token_select})")
        head = groot_model.action_head
        self.pathway = "dit"
        self.layer = None if layer is None else int(layer)
        self.module = head.model if layer is None else head.model.transformer_blocks[int(layer)]
        self.mode = mode
        self.token_select = token_select
        self.gate = bool(gate)
        self.beta = float(beta)
        self.horizon = int(horizon if horizon is not None else head.action_horizon)
        self.num_denoise = int(
            num_denoise if num_denoise is not None else head.num_inference_timesteps
        )
        # 개입 call: "last"=k==K−1 (fit denoise_step 3 표적), "first"=k==0
        # (fit --denoise-step 0 재적합 NPZ 전용 — τ·W 가 step-0 공간 캘리브레이션이어야 함).
        self.apply_call = apply_call
        self._apply_k = 0 if apply_call == "first" else self.num_denoise - 1
        self.params = params
        self._phases = params["phases"]
        _any = next(iter(self._phases.values()))
        self.state_dim = int(_any["W_s"].shape[0])   # 18 (raw 9 + 속도 9)
        self.expected_dim = int(_any["W_s"].shape[1])
        self._handle = None
        self._k = 0                    # 요청 내 denoise call index
        self._prev_p9: np.ndarray | None = None
        self._phi: np.ndarray | None = None      # 18차원 raw 상태(+속도), 정규화 전
        self._active: dict | None = None         # 현재 phase 파라미터
        self._scene = None
        self._stats: dict | None = None          # 활성 phase 의 (scene|global) 중심화
        self._cache: tuple | None = None         # (device, dtype, W_s, W_f, mh, mp, sp)
        self._phi_cache: torch.Tensor | None = None
        # 진단용 (러너가 /health·로그로 dose 를 재구성)
        self.last_margin: float | None = None
        self.n_fired = 0               # 마지막 denoise call 에서 실제 개입한 횟수
        self.n_gated_off = 0           # m ≤ τ 로 no-op 한 횟수

    # -- 폴리모픽 호환 (registry 순회) -----------------------------------------
    @property
    def per_step(self) -> bool:
        return False

    def reset_step_counter(self) -> None:
        """요청 시작 시 serve 가 호출 — denoise call index 리셋."""
        self._k = 0

    # -- 상태/phase 주입 --------------------------------------------------------
    def set_state(self, p9: np.ndarray) -> None:
        """raw proprio 9차원 주입 → 내부 버퍼 차분으로 18차원 φ 구성 (요청마다 1회)."""
        arr = np.asarray(p9, dtype=np.float64).reshape(-1)
        if arr.shape[0] * 2 != self.state_dim:
            raise ValueError(
                f"condg set_state: raw 상태 {arr.shape[0]}차원, 기대 {self.state_dim // 2} "
                f"(W 의 상태 차원 {self.state_dim} = raw+속도)"
            )
        vel = np.zeros_like(arr) if self._prev_p9 is None else arr - self._prev_p9
        self._prev_p9 = arr
        self._phi = np.concatenate([arr, vel])
        self._phi_cache = None

    def reset_state(self) -> None:
        """episode 경계(/reset): 속도 차분 버퍼 초기화 (다음 record 속도=0)."""
        self._prev_p9 = None
        self._phi = None
        self._phi_cache = None

    def set_phase(self, phase: str | None, scene: int | None = None) -> None:
        """현재 phase·scene 설정. 미등록 phase 또는 None → identity(off).

        중심화/z-score 파라미터는 **phase별**(fit 이 phase 마다 재계산) — 활성 phase
        엔트리 안의 scene 별 stats 에서 고르고, 미지 scene 은 그 phase 의 global 폴백.
        """
        ent = None if not phase else self._phases.get(str(phase))
        self._active = ent if (ent is not None and ent["registered"]) else None
        self._scene = None if scene is None else int(scene)
        if self._active is None:
            self._stats = None
        else:
            stats = self._active["scenes"].get(self._scene) \
                if self._scene is not None else None
            self._stats = stats if stats is not None else self._active["global"]
        self._cache = None
        self._phi_cache = None

    @property
    def armed(self) -> bool:
        return self._active is not None

    def status(self) -> dict:
        return {
            "op": "condg", "mode": self.mode, "beta": self.beta,
            "layer": self.layer, "token_select": self.token_select,
            "gate": self.gate, "armed": self.armed, "scene": self._scene,
            "scene_fallback": (self._active is not None
                               and self._stats is self._active["global"]),
            "num_denoise": self.num_denoise, "apply_call": self.apply_call,
            "last_margin": self.last_margin,
            "n_fired": self.n_fired, "n_gated_off": self.n_gated_off,
        }

    # -- hook -------------------------------------------------------------------
    def _tensors(self, out: torch.Tensor):
        """device/dtype 별 텐서 캐시 (W_s/W_f/mh/mp/sp). 계산은 float32 고정."""
        dev = out.device
        if self._cache is not None and self._cache[0] == dev:
            return self._cache[1:]
        ent = self._active
        st = self._stats
        t = lambda a: torch.as_tensor(a, device=dev, dtype=torch.float32)  # noqa: E731
        pack = (t(ent["W_s"]), t(ent["W_f"]), t(st["mh"]), t(st["mp"]), t(st["sp"]))
        self._cache = (dev, *pack)
        self._phi_cache = None
        return pack

    def _hook(self, _module: Any, _args: tuple, output: Any) -> Any:
        k = self._k
        self._k += 1
        if self._active is None or self._phi is None:
            return output                      # off — 원본 그대로 (no-hook 과 동일)
        if k != self._apply_k:
            return output                      # 지정 denoise call(기본 마지막)에서만 개입
        is_tuple = isinstance(output, tuple)
        out = output[0] if is_tuple else output
        if out.shape[-1] != self.expected_dim:
            raise RuntimeError(
                f"condg: hook 텐서 dim {out.shape[-1]} != NPZ dim {self.expected_dim} "
                f"(layer={self.layer}) — fit 주입 지점 불일치"
            )
        W_s, W_f, mh, mp, sp = self._tensors(out)
        phi = self._phi_cache
        if phi is None:
            phi = torch.as_tensor(self._phi, device=out.device, dtype=torch.float32)
            self._phi_cache = phi
        phi_z = (phi - mp) / sp                       # [P] z-score (scene 통계)
        hs = phi_z @ W_s                              # [D]
        hf = phi_z @ W_f                              # [D]
        # fit 공간 = 전 토큰 mean (τ 캘리브레이션이 이 공간에 묶여 있음)
        hbar = out.to(torch.float32).mean(dim=-2)     # [..., D]
        ht = hbar - mh                                # scene-중심화
        es, ef = ht - hs, ht - hf
        margin = (es * es).sum(-1) - (ef * ef).sum(-1)          # [...]
        self.last_margin = float(margin.reshape(-1)[0].item())
        if self.mode == "condg":
            d = hf - hs
            nrm = torch.linalg.norm(d)
            if float(nrm) < 1e-8:                      # 성공/실패 예측이 동일 → 개입 불가
                return output
            d = d / nrm
            delta = self.beta * (es * d).sum(-1, keepdim=True) * d   # [..., D]
        else:                                          # "hs" — 성공-모방 단독
            delta = self.beta * es
        if self.gate:
            fire = (margin > self._active["tau"]).to(delta.dtype).unsqueeze(-1)
            if float(fire.reshape(-1)[0].item()) == 0.0:
                self.n_gated_off += 1
                return output
            delta = delta * fire
        self.n_fired += 1
        delta = delta.to(out.dtype).unsqueeze(-2)      # [..., 1, D] — 토큰 공통 벡터
        if self.token_select == "all":
            steered = out - delta
        else:                                          # "future" — [1 : T−horizon] 만
            steered = out.clone()
            _t = steered.shape[-2]
            steered[..., 1 : _t - self.horizon, :] = (
                steered[..., 1 : _t - self.horizon, :] - delta
            )
        if is_tuple:
            return (steered, *output[1:])
        return steered

    def register(self) -> "CondGuidanceSteering":
        if self._handle is None:
            self._handle = self.module.register_forward_hook(self._hook)
        return self

    def unregister(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self) -> "CondGuidanceSteering":
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

"""GR00T N1.5 flow-matching action head + QAM velocity 합성 sampler (torch).

기준 코드 (실제 serve 경로): lerobot/src/lerobot/policies/groot/action_head/
flow_matching_action_head.py ``get_action`` (lerobot 1396b9f; Isaac-GR00T-N1.5 784fb74 와 동치).
그 루프를 그대로 옮기고 velocity 한 줄만 합성으로 바꾼다:

    GR00T:  t_k = k/4 (0→1), x_{k+1} = x_k + dt * v,  v = action - noise 방향
    합성:   v = v_vla                                  (lam == 0, QAM 호출 없음)
            v[..., :12] = (1-lam) * v_vla[..., :12] + lam * v_qam   (lam > 0)
            v[..., 12:] = v_vla[..., 12:]              (padding 은 VLA 그대로)

RL2 SIMPLER(pi0) 코드는 t 1→0, dt<0 이라 ``w*v_vla - (1-w)*v_qam`` 부호를 쓴다. GR00T 는 QAM 과
시간 방향이 같으므로 부호가 ``+`` 다 (lam = 1 - w). 이 차이 때문에 pi0 합성 코드를 복사하지 않는다.

QAM 관측 feature (확정, 2026-10-21 review):
    action_head.action_decoder 의 **입력** = DiT proj_out_2 출력 (B, 49, 1024) 중
    마지막 16 개 action token 만 평균 → (B, 1024). 같은 호출의 **첫 denoise call(flow step 0)** 에서
    한 번만 캡처하고 4 flow step 내내 그대로 쓴다. state(0:1)·future(1:33) token 은 섞지 않는다.
    DiT block residual(1536)·VLM context(2048) 는 다른 인터페이스라 거부한다 (차원 변환 대체 금지).
실제 GPU 모델에서의 실행은 검증되지 않았다 (CPU fake head + upstream 소스 parity 만 검증).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from contract import (
    ACTION_HORIZON,
    CONFIG_SHA256,
    ContractError,
    DIT_NUM_LAYERS,
    DIT_OUTPUT_DIM,
    DIT_SEQ_LEN,
    DIT_WIDTH,
    ENV_ACTION_DIM,
    FEATURE_INTERFACE,
    MODEL_ACTION_DIM,
    NUM_INFERENCE_STEPS,
    NUM_TIMESTEP_BUCKETS,
    VLM_CONTEXT_DIM,
)

QAMVelocityFn = Callable[[np.ndarray, np.ndarray, int], Any]
GateFn = Callable[[np.ndarray], float]

FEATURE_CAPTURE_POINT = "action_head.action_decoder.input"  # == DiT proj_out_2 출력
FEATURE_TOKEN_SELECTION = "last16_action_tokens"
FEATURE_POOL = "mean"
FEATURE_DENOISE_STEP = 0
FEATURE_DIM = DIT_OUTPUT_DIM


@dataclass(frozen=True)
class FeatureSpec:
    """확정된 QAM 관측 계약. 필드는 기록용이며 다른 값은 생성 단계에서 거부한다."""

    capture_point: str = FEATURE_CAPTURE_POINT
    token_selection: str = FEATURE_TOKEN_SELECTION
    pool: str = FEATURE_POOL
    denoise_step: int = FEATURE_DENOISE_STEP
    dim: int = FEATURE_DIM

    def __post_init__(self):
        if int(self.dim) in (DIT_WIDTH, VLM_CONTEXT_DIM) or "transformer_blocks" in self.capture_point:
            raise ContractError(
                f"{self.capture_point} dim {self.dim}: DiT block residual(1536)/VLM context(2048) 는 "
                "action_decoder 입력과 다른 인터페이스 — 이 계약의 대체 입력이 아니다")
        want = {"capture_point": FEATURE_CAPTURE_POINT, "token_selection": FEATURE_TOKEN_SELECTION,
                "pool": FEATURE_POOL, "denoise_step": FEATURE_DENOISE_STEP, "dim": FEATURE_DIM}
        got = {k: getattr(self, k) for k in want}
        if got != want:
            raise ContractError(f"FeatureSpec 은 확정 계약만 허용: got {got} want {want}")

    @property
    def feature_kind(self) -> str:
        return f"groot_n15_{FEATURE_INTERFACE}"

    def signature(self) -> str:
        payload = {"feature_kind": self.feature_kind, "capture_point": self.capture_point,
                   "token_selection": self.token_selection, "pool": self.pool,
                   "denoise_step": self.denoise_step, "dim": self.dim,
                   "config_sha256": CONFIG_SHA256}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def pool_decoder_input(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 49, 1024) action_decoder 입력 → 마지막 16 token 평균 (B, 1024)."""
        if x.ndim != 3 or x.shape[-1] != FEATURE_DIM:
            if x.ndim == 3 and x.shape[-1] in (DIT_WIDTH, VLM_CONTEXT_DIM):
                raise ContractError(
                    f"입력 폭 {x.shape[-1]} 거부: 1536 block residual/2048 VLM 은 다른 인터페이스")
            raise ContractError(f"action_decoder 입력 shape {tuple(x.shape)} != (B, 49, 1024)")
        if x.shape[1] != DIT_SEQ_LEN:
            raise ContractError(f"action_decoder 입력 token 수 {x.shape[1]} != {DIT_SEQ_LEN}")
        # 앞 token 을 하나 버리는 pi0 식 slicing 이 아니라 정확히 마지막 H=16 개.
        return x[:, -ACTION_HORIZON:].float().mean(dim=1)


SPEC = FeatureSpec()


class FirstCallCapture:
    """action_decoder forward pre-hook. 첫 호출 입력만 보관하고 이후 호출은 세기만 한다.

    write-once: 보관 슬롯을 다시 쓰려 하면 ContractError (첫 denoise feature 덮어쓰기 금지).
    """

    def __init__(self):
        self._first: torch.Tensor | None = None
        self.calls = 0

    def _store(self, x: torch.Tensor) -> None:
        if self._first is not None:
            raise ContractError("첫 denoise call feature 는 write-once — 덮어쓰기 거부")
        self._first = x.detach().clone()

    def __call__(self, _module, args, kwargs=None):
        x = args[0] if args else (kwargs or {}).get("x")
        if self.calls == 0:
            self._store(x)
        self.calls += 1
        return None

    @property
    def first(self) -> torch.Tensor:
        if self._first is None:
            raise ContractError("action_decoder 가 호출되지 않아 feature 없음")
        return self._first


@dataclass(frozen=True)
class FeatureRecord:
    value: np.ndarray       # (B, 1024) float32
    env_step: int           # 이 feature 를 만든 get_action 호출의 env step
    flow_step: int          # 그 호출 안의 denoise step
    signature: str


def validate_feature_record(rec: FeatureRecord, *, spec: FeatureSpec, decision_env_step: int,
                            batch_size: int) -> np.ndarray:
    if rec.signature != spec.signature():
        raise ContractError(f"feature signature {rec.signature} != {spec.signature()}")
    if rec.env_step > decision_env_step:
        raise ContractError(f"미래 feature 거부: env_step {rec.env_step} > 결정 {decision_env_step}")
    if rec.env_step < decision_env_step:
        raise ContractError(f"stale feature 거부: env_step {rec.env_step} < 결정 {decision_env_step}")
    if rec.flow_step != spec.denoise_step:
        raise ContractError(f"flow_step {rec.flow_step} != {spec.denoise_step} (합성 후 feature 금지)")
    v = np.asarray(rec.value)
    if v.ndim == 2 and v.shape[-1] in (DIT_WIDTH, VLM_CONTEXT_DIM):
        raise ContractError(f"feature 폭 {v.shape[-1]} 거부: block residual/VLM 은 다른 인터페이스")
    if v.shape != (batch_size, FEATURE_DIM):
        raise ContractError(f"feature shape {v.shape} != ({batch_size}, {FEATURE_DIM})")
    if not np.all(np.isfinite(v)):
        raise ContractError("feature 에 비유한 값")
    return v.astype(np.float32, copy=False)


def _check_velocity(v: torch.Tensor, batch_size: int, what: str) -> torch.Tensor:
    if tuple(v.shape) != (batch_size, ACTION_HORIZON, MODEL_ACTION_DIM):
        raise ContractError(f"{what} shape {tuple(v.shape)} != ({batch_size}, 16, 32)")
    if not bool(torch.isfinite(v).all()):
        raise ContractError(f"{what} 에 비유한 velocity")
    return v


def qam_velocity_to_tensor(v_qam: Any, batch_size: int, like: torch.Tensor) -> torch.Tensor:
    a = np.asarray(v_qam, dtype=np.float32)
    if a.shape == (batch_size, ACTION_HORIZON * ENV_ACTION_DIM):
        a = a.reshape(batch_size, ACTION_HORIZON, ENV_ACTION_DIM)
    if a.shape != (batch_size, ACTION_HORIZON, ENV_ACTION_DIM):
        raise ContractError(f"QAM velocity shape {a.shape} != ({batch_size}, 192|16x12)")
    if not np.all(np.isfinite(a)):
        raise ContractError("QAM velocity 에 비유한 값")
    return torch.as_tensor(a, device=like.device).to(like.dtype)


def compose_velocity(v_vla: torch.Tensor, v_qam12: torch.Tensor | None, lam: float) -> torch.Tensor:
    if lam == 0.0:
        return v_vla
    if v_qam12 is None:
        raise ContractError("lam > 0 인데 QAM velocity 없음")
    v = v_vla.clone()
    v[..., :ENV_ACTION_DIM] = (1.0 - lam) * v_vla[..., :ENV_ACTION_DIM] + lam * v_qam12
    return v


def check_head_contract(head: Any) -> None:
    cfg = head.config
    got = {
        "num_inference_timesteps": head.num_inference_timesteps,
        "num_timestep_buckets": head.num_timestep_buckets,
        "action_horizon": cfg.action_horizon,
        "action_dim": cfg.action_dim,
        "n_blocks": len(head.model.transformer_blocks),
        "input_embedding_dim": getattr(cfg, "input_embedding_dim", None),
        "hidden_size": getattr(cfg, "hidden_size", None),
    }
    want = {
        "num_inference_timesteps": NUM_INFERENCE_STEPS,
        "num_timestep_buckets": NUM_TIMESTEP_BUCKETS,
        "action_horizon": ACTION_HORIZON,
        "action_dim": MODEL_ACTION_DIM,
        "n_blocks": DIT_NUM_LAYERS,
        "input_embedding_dim": DIT_WIDTH,      # DiT 내부 폭 (관측 아님)
        "hidden_size": DIT_OUTPUT_DIM,         # action_decoder 입력 폭 = QAM 관측 폭
    }
    if got != want:
        raise ContractError(f"action head 계약 불일치: got {got} want {want}")


class ComposedFlowSampler:
    def __init__(self, spec: FeatureSpec = SPEC, *, qam_fn: QAMVelocityFn | None, qam_weight: float,
                 gate_fn: GateFn | None = None):
        if not isinstance(spec, FeatureSpec):
            raise ContractError("spec 은 FeatureSpec (decoder 입력 계약) 이어야 함")
        if not 0.0 <= float(qam_weight) <= 1.0:
            raise ContractError(f"qam_weight {qam_weight} not in [0, 1]")
        self.spec = spec
        self.qam_fn = qam_fn
        self.qam_weight = float(qam_weight)
        self.gate_fn = gate_fn

    def _resolve_lam(self, gate: float | None, feature: np.ndarray | None) -> float:
        if gate is None:
            if self.gate_fn is None or feature is None:
                raise ContractError("gate 미지정 (fixed gate 또는 gate_fn 필요)")
            gate = float(self.gate_fn(feature))
        if gate not in (0.0, 1.0):
            raise ContractError(f"gate {gate} 는 0/1 만 허용")
        lam = gate * self.qam_weight
        if lam > 0.0 and self.qam_fn is None:
            raise ContractError("gate 가 켜졌는데 QAM 없음")
        return lam

    @torch.no_grad()
    def sample(self, head: Any, backbone_output: Any, action_input: Any, *,
               decision_env_step: int, gate: float | None = None,
               external_feature: FeatureRecord | None = None) -> tuple[torch.Tensor, dict]:
        check_head_contract(head)
        # ---- 이하 upstream get_action 과 같은 순서 (RNG 소비: torch.randn 1회) ----
        backbone_output = head.process_backbone_output(backbone_output)
        vl_embs = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id
        state_features = head.state_encoder(action_input.state, embodiment_id)
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, head.config.action_horizon, head.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )
        num_steps = head.num_inference_timesteps
        dt = 1.0 / num_steps

        feature: np.ndarray | None = None
        if external_feature is not None:
            feature = validate_feature_record(external_feature, spec=self.spec,
                                              decision_env_step=decision_env_step,
                                              batch_size=batch_size)
        lam: float | None = None
        qam_steps: list[int] = []
        capture = FirstCallCapture()
        hook = head.action_decoder.register_forward_pre_hook(capture)
        try:
            for t in range(num_steps):
                t_cont = t / float(num_steps)
                t_discretized = int(t_cont * head.num_timestep_buckets)
                timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized,
                                              device=device)
                action_features = head.action_encoder(actions, timesteps_tensor, embodiment_id)
                if head.config.add_pos_embed:
                    pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                    pos_embs = head.position_embedding(pos_ids).unsqueeze(0)
                    action_features = action_features + pos_embs
                future_tokens = head.future_tokens.weight.unsqueeze(0).expand(
                    vl_embs.shape[0], -1, -1)
                sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

                model_output = head.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embs,
                    timestep=timesteps_tensor,
                )
                pred = head.action_decoder(model_output, embodiment_id)
                pred_velocity = _check_velocity(pred[:, -head.action_horizon:], batch_size, "VLA")

                if t == self.spec.denoise_step and feature is None:
                    # hook 이 본 decoder 입력이 루프 변수와 같은 텐서인지 확인 후 pool.
                    if not torch.equal(capture.first, model_output):
                        raise ContractError("hook 캡처 ≠ action_decoder 입력 (배선 불일치)")
                    rec = FeatureRecord(
                        value=self.spec.pool_decoder_input(capture.first).cpu().numpy(),
                        env_step=decision_env_step, flow_step=t, signature=self.spec.signature())
                    feature = validate_feature_record(rec, spec=self.spec,
                                                      decision_env_step=decision_env_step,
                                                      batch_size=batch_size)
                    feature.setflags(write=False)  # 4 step 동안 불변
                if lam is None:
                    lam = self._resolve_lam(gate, feature)

                v_qam = None
                if lam > 0.0:
                    noisy = actions[..., :ENV_ACTION_DIM].float().cpu().numpy().reshape(
                        batch_size, -1)
                    v_qam = qam_velocity_to_tensor(self.qam_fn(feature, noisy, t), batch_size,
                                                   pred_velocity)
                    qam_steps.append(t)
                # Update actions using euler integration.
                actions = actions + dt * compose_velocity(pred_velocity, v_qam, lam)
        finally:
            hook.remove()
        if capture.calls != num_steps:
            raise ContractError(f"action_decoder 호출 {capture.calls} != {num_steps}")

        info = {"lam": lam, "qam_steps": qam_steps, "feature": feature,
                "feature_signature": self.spec.signature(), "decision_env_step": decision_env_step,
                "decoder_calls": capture.calls}
        return actions, info

    def make_get_action(self, head: Any, context: Callable[[], dict]) -> Callable:
        """head.get_action 대체 callable. context() → {decision_env_step, gate?}.

        실제 lerobot GrootPolicy 에 끼우는 동작은 미검증 (GPU/모델 로드 필요).
        """
        def get_action(backbone_output, action_input):
            ctx = context()
            actions, info = self.sample(head, backbone_output, action_input,
                                        decision_env_step=int(ctx["decision_env_step"]),
                                        gate=ctx.get("gate"))
            out = type(backbone_output)(data={"action_pred": actions})
            get_action.last_info = info
            return out

        get_action.last_info = None
        return get_action

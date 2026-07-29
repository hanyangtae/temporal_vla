"""TTT integration for **GR00T N1.5** — mirrors ``Gr00tN1d6WithTTT`` for N1.5 base.

N1.5 와 N1.6 의 차이:
- Model class: ``GR00T_N1_5`` (in ``Isaac-GR00T-N1.5/gr00t/model/gr00t_n1.py``).
- Backbone embedding dim: 1536 (vs N1.6 의 2048). DiT KV concat 시 proj_dim 1536 매칭.
- Forward path: ``model.backbone(backbone_inputs) → BatchFeature{backbone_features,
  backbone_attention_mask} → model.action_head(backbone_outputs, action_inputs)``.

설계:
- ``attach_ttt_to_n1d5(base_model, ...)`` 가 **in-place** TTT 추가 (재인스턴스화 X).
- ``base_model.predictor`` = ProgressPredictor (input_dim=2048, proj_dim=1536).
- forward 시 batch 에 ``ttt_z_seq`` / ``ttt_valid_mask`` 가 있으면 episode-prefix replay
  으로 LHT (Latent History Token) 만들어 ``backbone_features`` 에 token 추가 후 action_head 호출.
- TTT 모듈 outer params (P_K/V/Q/W_0 등) 는 frozen, inner-loop SSL 만 작동
  (``create_graph=False``, gradient 차단).
- linear_preLN 모드일 때 cache-based incremental update 추후 추가 가능 (TODO 자리).
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from types import MethodType
from typing import Optional

import torch
from transformers.feature_extraction_utils import BatchFeature

# N1.5 codebase path 보장 (N1.6 의 gr00t 와 충돌 가능 → 호출자가 PYTHONPATH 또는 sys.path 관리)
_N1d5_ROOT = "/temporal_vla/src/policies/Isaac-GR00T-N1.5"
if _N1d5_ROOT not in sys.path:
    sys.path.insert(0, _N1d5_ROOT)

from src.ttt.predictor import ProgressPredictor  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# attach_ttt_to_n1d5: base_model 에 TTT in-place 추가
# ─────────────────────────────────────────────────────────────────────
def attach_ttt_to_n1d5(
    base_model,                                 # GR00T_N1_5 instance (already from_pretrained)
    predictor_input_dim: int = 2048,            # Phase 1 cache dim (N1.5 의 pre-LLM hidden)
    predictor_proj_dim: int = 1536,             # N1.5 DiT KV / backbone_embedding_dim
    predictor_inner_model: str = "linear",      # "linear" | "linear_preLN"
    predictor_eta_base: float = 0.1,
    predictor_state_path: Optional[str] = None,
    ttt_update_in_train: bool = True,
    ttt_update_at_inference: bool = True,
):
    """N1.5 base_model 에 TTT 모듈 + forward override 를 in-place 로 부착.

    Returns
    -------
    base_model (modified): forward 와 get_action 이 ttt_z_seq/valid_mask 를 받아 LHT
    토큰을 backbone_features 에 concat 후 action_head 로 전달.
    """
    device = next(base_model.parameters()).device

    # 1) Predictor (fp32) 생성 + freeze
    predictor = ProgressPredictor(
        input_dim=predictor_input_dim,
        proj_dim=predictor_proj_dim,
        inner_model_type=predictor_inner_model,
        eta_base=predictor_eta_base,
        learnable_eta=False,
    ).to(device=device, dtype=torch.float32)

    base_model.predictor = predictor
    base_model.ttt_update_in_train = ttt_update_in_train
    base_model.ttt_update_at_inference = ttt_update_at_inference
    base_model._predictor_loaded = False
    base_model.predictor_inner_model = predictor_inner_model

    for p in base_model.predictor.parameters():
        p.requires_grad = False

    # 2) predictor ckpt 로드 (Phase 1 결과)
    if predictor_state_path is not None:
        sp = Path(predictor_state_path)
        if not sp.exists():
            raise FileNotFoundError(f"predictor ckpt not found: {sp}")
        state = torch.load(str(sp), map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        base_model.predictor.load_state_dict(state)
        base_model.predictor.save_init()
        base_model._predictor_loaded = True
        print(f"[n1d5+ttt] predictor loaded: {sp}")

    # 3) Method bind
    _bind_ttt_methods(base_model)
    return base_model


# ─────────────────────────────────────────────────────────────────────
# Method 정의 (free functions; MethodType 으로 bind)
# ─────────────────────────────────────────────────────────────────────
@contextlib.contextmanager
def _enable_inner_grad(self):
    """ttt_step 의 autograd.grad 호출 위해 f_adapt 만 transient 하게 requires_grad=True.

    추가로 ``torch.enable_grad()`` 컨텍스트도 push — caller 가 ``no_grad`` 안에 있어도
    inner update 가능. (단 ``torch.inference_mode`` 는 외부에서 비활성화돼있어야 함 —
    inference tensor 는 autograd 불가, ZMQ server entry 가 그걸 처리.)
    """
    flipped = []
    for n, p in self.predictor.named_parameters():
        if "f_adapt" in n and not p.requires_grad:
            p.requires_grad = True
            flipped.append(p)
    with torch.enable_grad():
        try:
            yield
        finally:
            for p in flipped:
                p.requires_grad = False


def _load_predictor_state(self, state_dict_path: str) -> None:
    state = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    self.predictor.load_state_dict(state)
    self.predictor.save_init()
    self._predictor_loaded = True


def _reset_predictor(self) -> None:
    """Episode 시작 시 inner state reset (θ_0 으로 복원)."""
    self.predictor.reset()


def _ttt_token_from_zseq(self, z_seq: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Episode-prefix replay → LHT.

    Args:
        z_seq:      [B, T_max, D_in=2048]  Eagle pre-LLM cache slice (0..t prefix)
        valid_mask: [B, T_max] bool. True = real frame.

    Returns:
        ttt_token: [B, 1, proj_dim=1536]  detached LHT for concat into backbone_features.
    """
    B, T_max, D = z_seq.shape
    target_dtype = z_seq.dtype
    z_seq_t = z_seq.permute(1, 0, 2).contiguous().float()    # [T, B, D]
    valid_mask_t = valid_mask.permute(1, 0).contiguous()      # [T, B]

    with self._enable_inner_grad():
        result = self.predictor.meta_forward(
            z_seq_t, create_graph=False, valid_mask=valid_mask_t,
        )
    ttt_outputs_seq = result["ttt_outputs_seq"]               # [T_max, B, proj_dim]

    last_valid = valid_mask.long().sum(dim=1) - 1             # [B]
    last_valid = last_valid.clamp(min=0)
    arange_B = torch.arange(B, device=z_seq.device)
    lht = ttt_outputs_seq[last_valid, arange_B, :]            # [B, proj_dim]
    ttt_token = lht.detach().unsqueeze(1).to(target_dtype)    # [B, 1, proj_dim]
    return ttt_token


def _augment_backbone_features(backbone_outputs: BatchFeature, ttt_token: torch.Tensor) -> BatchFeature:
    """N1.5 backbone output 의 backbone_features 에 ttt_token concat + mask 확장."""
    feats = backbone_outputs["backbone_features"]                # [B, T_vl, D=1536]
    attn = backbone_outputs["backbone_attention_mask"]           # [B, T_vl]
    new_feats = torch.cat([feats, ttt_token.to(feats.dtype)], dim=1)
    ones = torch.ones(feats.size(0), 1, dtype=attn.dtype, device=attn.device)
    new_attn = torch.cat([attn, ones], dim=1)
    return BatchFeature(data={
        "backbone_features": new_feats,
        "backbone_attention_mask": new_attn,
    })


def _pop_ttt_inputs(inputs):
    """ttt_z_seq / ttt_valid_mask 를 batch 에서 pop (prepare_input 에 unknown key 안 가게)."""
    z_seq = inputs.pop("ttt_z_seq", None) if isinstance(inputs, dict) else None
    valid_mask = inputs.pop("ttt_valid_mask", None) if isinstance(inputs, dict) else None
    return z_seq, valid_mask


def _ttt_forward(self, inputs):
    """TTT-aware forward — N1.5 의 ``GR00T_N1_5.forward`` 대체."""
    z_seq, valid_mask = _pop_ttt_inputs(inputs)
    backbone_inputs = self.backbone.prepare_input(inputs)
    action_inputs = self.action_head.prepare_input(inputs)
    backbone_outputs = self.backbone(backbone_inputs)

    if z_seq is not None and valid_mask is not None:
        ttt_token = self._ttt_token_from_zseq(z_seq, valid_mask)
        backbone_outputs = _augment_backbone_features(backbone_outputs, ttt_token)
    # else: fallback — TTT 없이 N1.5 vanilla

    return self.action_head(backbone_outputs, action_inputs)


def _backbone_forward_with_pre_llm(self, vl_input):
    """N1.5 ``backbone.forward_eagle`` 를 한 번만 호출하여 backbone_features 와
    pre-LLM z_t (hidden_states[0]) 를 동시에 추출.

    학습 시에는 z_t 가 dataloader 의 Eagle cache 에서 옴 (`ttt_z_seq`).
    추론 시에는 cache 없이 backbone 자체에서 매 step z_t 를 추출 → predictor 의
    inner state 를 streaming 으로 갱신.

    Returns
    -------
    backbone_outputs : BatchFeature
        ``backbone_features`` [B, T, 1536], ``backbone_attention_mask`` [B, T]
    z_t : Tensor
        [B, 2048] — mean-pooled pre-LLM hidden (eagle_model hidden_states[0]).
    """
    bb = self.backbone
    eagle_prefix = "eagle_"
    eagle_input = {
        k.removeprefix(eagle_prefix): v
        for k, v in vl_input.items()
        if k.startswith(eagle_prefix)
    }
    eagle_input.pop("image_sizes", None)

    out = bb.eagle_model(
        **eagle_input, output_hidden_states=True, return_dict=True,
    )

    # hidden_states[0]: pre-LLM (embedding 통과 후 LLM 입력 직전).
    # hidden_states[bb.select_layer]: backbone_features 가 되는 LLM hidden.
    z_per_token = out.hidden_states[0]                              # [B, N, 2048]
    eagle_features = out.hidden_states[bb.select_layer]
    eagle_features = bb.eagle_linear(eagle_features)                # → [B, T, 1536]

    attn_mask = eagle_input["attention_mask"]
    backbone_outputs = BatchFeature(data={
        "backbone_features": eagle_features,
        "backbone_attention_mask": attn_mask,
    })

    # z_t pooling — attention_mask 로 valid token mean (extract script 와 동일).
    m = attn_mask.to(z_per_token.dtype).unsqueeze(-1)
    z_t = (z_per_token * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)   # [B, 2048]
    return backbone_outputs, z_t


def _ttt_get_action(self, inputs):
    """TTT-aware get_action — N1.5 의 ``GR00T_N1_5.get_action`` 대체.

    두 경로:
      A) Streaming (default 추론 path): ttt_z_seq 가 inputs 에 없을 때.
         backbone 의 eagle_model 을 1회 호출하여 z_t (pre-LLM) 와 backbone_features 둘
         다 추출 → ``predictor.forward(z_t, update=True)`` 로 in-place inner update +
         LHT 추출 → backbone_features 에 concat → DiT.
         episode 시작 시 ``/reset`` 이 ``reset_predictor()`` 를 호출하여 θ_0 복원.

      B) Legacy (호환성): caller 가 명시적으로 ``ttt_z_seq`` (prefix replay) 를 전달했을 때.
         학습 path 와 동일하게 episode 0..t 의 cached z 로 LHT 계산.
    """
    z_seq, valid_mask = _pop_ttt_inputs(inputs)
    backbone_inputs = self.backbone.prepare_input(inputs)
    action_inputs = self.action_head.prepare_input(inputs)

    if z_seq is not None and valid_mask is not None:
        # B) Legacy episode-prefix replay path
        backbone_outputs = self.backbone(backbone_inputs)
        ttt_token = self._ttt_token_from_zseq(z_seq, valid_mask)
    else:
        # A) Streaming inner-loop TTT path — 매 step z_t 추출 + inner update.
        backbone_outputs, z_t = self._backbone_forward_with_pre_llm(backbone_inputs)
        if self.ttt_update_at_inference:
            with self._enable_inner_grad():
                result = self.predictor.forward(z_t, update=True)
        else:
            result = self.predictor.forward(z_t, update=False)
        lht = result["ttt_output"]                                       # [B, proj_dim]
        ttt_token = lht.detach().unsqueeze(1).to(
            backbone_outputs["backbone_features"].dtype,
        )

    backbone_outputs = _augment_backbone_features(backbone_outputs, ttt_token)
    return self.action_head.get_action(backbone_outputs, action_inputs)


def _bind_ttt_methods(model):
    """method 들 model instance 에 bind. Gr00tN1d6WithTTT 의 mirror pattern."""
    # original 보존 (debug / fallback 용)
    if not hasattr(model, "_orig_forward"):
        model._orig_forward = model.forward
        model._orig_get_action = model.get_action

    # instance methods
    model._enable_inner_grad = MethodType(_enable_inner_grad, model)
    model.load_predictor_state = MethodType(_load_predictor_state, model)
    model.reset_predictor = MethodType(_reset_predictor, model)
    model._ttt_token_from_zseq = MethodType(_ttt_token_from_zseq, model)
    model._backbone_forward_with_pre_llm = MethodType(_backbone_forward_with_pre_llm, model)
    model.forward = MethodType(_ttt_forward, model)
    model.get_action = MethodType(_ttt_get_action, model)

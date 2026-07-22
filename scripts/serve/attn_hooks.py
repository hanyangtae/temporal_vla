"""GR00T N1.5 DiT cross-attention 카메라 뷰별 attention 캡처 (serve-side 주입).

목적
----
"wrist cam 이 rollout 의 어느 step 에서 실제로 읽히는가"를 보기 위해, DiT
cross-attention 블록(짝수 idx, key=Eagle VL 시퀀스 전체)의 attention weight 를
카메라 뷰별 mass 로 축약해 캡처한다.

구조적 근거 (lerobot v0.5.1 소스에서 검증)
  - Eagle 시퀀스: ``eagle_input_ids == image_token_index`` 위치에 vision token 이
    뷰당 256개 연속 블록으로 채워짐 (``modeling_eagle2_5_vl.py:246-259``). 뷰 순서는
    ``processor_groot.py:275`` 의 ``sorted(img_keys)`` = side_0(left), side_1(right),
    wrist_0(wrist) — **wrist = 마지막 블록**.
  - backbone_features = LM hidden state 전체 시퀀스 (길이 = len(input_ids)),
    vlln → vl_self_attention 을 지나도 길이·위치 보존 → DiT
    ``encoder_hidden_states`` 의 key 컬럼 인덱스가 input_ids 위치와 1:1.
  - DiT query = [state | future(num_target_vision_tokens) | action(action_horizon)]
    (``flow_matching_action_head.py:384``).

캡처 방식
  - diffusers ``Attention.set_processor`` 로 cross-attn 블록(``cross_attention_dim
    is not None``)의 ``attn1`` 에 wrapper processor 를 **serve boot 시 1회** 설치.
    출력 경로는 원본 processor(AttnProcessor2_0/SDPA)에 그대로 위임 → action 은
    캡처 ON/OFF 와 무관하게 bit-identical.
  - 부수 계산으로 softmax(q·kᵀ·scale) 를 fp32 로 명시 계산 후 **즉시 그룹 축약**:
    key 축 → (text, left, right, wrist) 4그룹 mass 합, query 축 → (state, future,
    action) 3그룹 mean, head 축 → mean (``keep_heads`` 로 보존 가능).
    per-request 결과 = ``[n_cross_blocks, K_denoise, n_qgroups, n_kgroups]``.

caveat (해석 시 명시할 것)
  - DiT 앞의 ``vl_self_attention`` 이 VL 시퀀스를 한 번 섞으므로, 위치 기반 귀인은
    유효하나 각 위치의 내용에는 다른 뷰 정보가 혼입될 수 있다. attention mass 는
    "그 카메라 위치를 참조" 이지 "그 카메라 정보만 사용"이 아니다.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

# sorted(img_keys) = side_0, side_1, wrist_0 (processor_groot.py:275) → 고정 순서.
VIEW_ORDER = ("left", "right", "wrist")
KGROUPS = ("text",) + VIEW_ORDER
QGROUPS = ("state", "future", "action")
# Eagle2.5 224px / patch14 / pixel_shuffle off → 뷰당 256 vision token.
EXPECTED_TOKENS_PER_VIEW = 256


def resolve_view_spans(
    input_ids: "torch.Tensor | np.ndarray",
    image_token_index: int,
    *,
    n_views: int = len(VIEW_ORDER),
    tokens_per_view: int = EXPECTED_TOKENS_PER_VIEW,
) -> list[tuple[int, int]]:
    """``input_ids`` 에서 image token 위치를 뷰별 ``(start, end)`` span 으로 반환.

    vision embedding 은 image token 위치에 **순서대로** 채워지므로
    (``modeling_eagle2_5_vl.py:246-259`` masked fill), 전체 image 위치를 순서대로
    ``tokens_per_view`` 개씩 chunk 하면 곧 뷰 경계다. 무음 오귀인을 막기 위해
    총 개수 불일치·chunk 비연속(뷰 토큰이 interleave 된 경우) 은 즉시 raise 한다.
    """
    if isinstance(input_ids, torch.Tensor):
        ids = input_ids.detach().reshape(-1).cpu().numpy()
    else:
        ids = np.asarray(input_ids).reshape(-1)
    pos = np.flatnonzero(ids == int(image_token_index))
    if len(pos) != n_views * tokens_per_view:
        raise ValueError(
            f"expected {n_views}x{tokens_per_view} image tokens, got {len(pos)}"
        )
    spans: list[tuple[int, int]] = []
    for v in range(n_views):
        chunk = pos[v * tokens_per_view : (v + 1) * tokens_per_view]
        if int(chunk[-1]) - int(chunk[0]) != tokens_per_view - 1:
            raise ValueError(
                f"view {v} image tokens are not contiguous: "
                f"[{int(chunk[0])}, {int(chunk[-1])}]"
            )
        spans.append((int(chunk[0]), int(chunk[-1]) + 1))
    return spans


class _CrossAttnWrapProcessor:
    """원본 processor 를 감싸 출력은 그대로 두고 attention mass 만 곁가지 계산."""

    def __init__(self, block_idx: int, sink: "CrossAttnCapture", inner: Any):
        self.block_idx = block_idx
        self.sink = sink
        self.inner = inner

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if encoder_hidden_states is not None and self.sink.active:
            self.sink.record(self.block_idx, attn, hidden_states, encoder_hidden_states)
        return self.inner(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )


class CrossAttnCapture:
    """DiT cross-attn 블록에 wrapper processor 를 설치하고 per-request 캡처를 관리.

    serve boot 시 1회 ``install()`` (per-request 설치/해제는 torch.compile 캐시와
    상호작용할 수 있어 피한다). 요청마다::

        spans = resolve_view_spans(batch["eagle_input_ids"], cap.image_token_index)
        cap.begin(spans, seq_len)
        ... policy.predict_action_chunk(batch) ...
        mass, maps = cap.finish()   # mass: [n_blocks, K, n_qgroups, n_kgroups]
    """

    def __init__(
        self,
        policy: Any,
        *,
        keep_heads: bool = False,
        full_maps: bool = False,
    ):
        model = policy._groot_model
        head = model.action_head
        blocks = head.model.transformer_blocks
        self.cross_block_indices = [
            i
            for i, b in enumerate(blocks)
            if getattr(b, "cross_attention_dim", None) is not None
        ]
        if not self.cross_block_indices:
            raise RuntimeError("no cross-attention blocks found in DiT transformer_blocks")
        self.image_token_index = int(model.backbone.eagle_model.image_token_index)
        self.n_future = int(head.config.num_target_vision_tokens)
        self.n_action = int(head.action_horizon)
        self.keep_heads = keep_heads
        self.full_maps = full_maps
        self.qgroups = QGROUPS
        self.kgroups = KGROUPS
        self._attns = [(i, blocks[i].attn1) for i in self.cross_block_indices]
        self._installed = False
        self.active = False
        self._spans: list[tuple[int, int]] | None = None
        self._seq_len: int | None = None
        self._mass_bufs: dict[int, list[torch.Tensor]] = {}
        self._map_bufs: dict[int, list[torch.Tensor]] = {}

    # ---- 설치/해제 (boot 시 1회) ----

    def install(self) -> "CrossAttnCapture":
        if self._installed:
            return self
        for i, attn in self._attns:
            attn.set_processor(_CrossAttnWrapProcessor(i, self, attn.processor))
        self._installed = True
        return self

    def uninstall(self) -> None:
        if not self._installed:
            return
        for _, attn in self._attns:
            proc = attn.processor
            if isinstance(proc, _CrossAttnWrapProcessor):
                attn.set_processor(proc.inner)
        self._installed = False

    # ---- per-request ----

    def begin_from_batch(self, batch: dict[str, Any]) -> None:
        """serve batch 의 ``eagle_input_ids`` 로 뷰 span 을 해석하고 캡처 시작."""
        ids = batch.get("eagle_input_ids")
        if ids is None:
            raise RuntimeError("cross-attn capture requires 'eagle_input_ids' in batch")
        if ids.ndim == 2 and ids.shape[0] != 1:
            raise RuntimeError(f"cross-attn capture expects B=1, got {tuple(ids.shape)}")
        spans = resolve_view_spans(ids, self.image_token_index)
        self.begin(spans, int(ids.shape[-1]))

    @property
    def view_spans(self) -> list[tuple[int, int]] | None:
        return None if self._spans is None else list(self._spans)

    def begin(self, spans: list[tuple[int, int]], seq_len: int) -> None:
        if not self._installed:
            raise RuntimeError("CrossAttnCapture.begin() before install()")
        self._spans = list(spans)
        self._seq_len = int(seq_len)
        self._mass_bufs = {i: [] for i in self.cross_block_indices}
        self._map_bufs = {i: [] for i in self.cross_block_indices}
        self.active = True

    def record(
        self,
        block_idx: int,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> None:
        spans, seq_len = self._spans, self._seq_len
        if spans is None or seq_len is None:
            raise RuntimeError("record() without begin()")
        # 출력 경로와 동일한 q/k projection 을 곁가지로 재계산 (순수 read-only).
        q = attn.to_q(hidden_states)
        enc = encoder_hidden_states
        if getattr(attn, "norm_cross", None):
            enc = attn.norm_encoder_hidden_states(enc)
        k = attn.to_k(enc)
        bsz, t_q, _ = q.shape
        s = k.shape[1]
        if bsz != 1:
            raise RuntimeError(f"cross-attn capture expects B=1 serve, got B={bsz}")
        if s != seq_len:
            raise RuntimeError(
                f"encoder seq len {s} != eagle_input_ids len {seq_len} "
                "(view span misalignment)"
            )
        n_state = t_q - self.n_future - self.n_action
        if n_state < 1:
            raise RuntimeError(
                f"unexpected query layout: T_q={t_q}, future={self.n_future}, "
                f"action={self.n_action}"
            )
        head_dim = q.shape[-1] // attn.heads
        # [B, heads, T_q, S], softmax 는 fp32 로 (bf16 autocast 안전).
        qh = q.view(bsz, t_q, attn.heads, head_dim).transpose(1, 2).float()
        kh = k.view(bsz, s, attn.heads, head_dim).transpose(1, 2).float()
        probs = torch.softmax(qh @ kh.transpose(-1, -2) * (head_dim**-0.5), dim=-1)

        # key 축 그룹 mass: text = 나머지 전부 (system prompt + instruction + special).
        view_mass = [probs[..., s0:s1].sum(dim=-1) for s0, s1 in spans]  # each [B,h,T_q]
        text_mass = probs.sum(dim=-1) - torch.stack(view_mass, dim=0).sum(dim=0)
        kg = torch.stack([text_mass, *view_mass], dim=-1)  # [B, h, T_q, n_kgroups]
        # query 축 그룹 mean: [state | future | action].
        bounds = (0, n_state, n_state + self.n_future, t_q)
        qg = torch.stack(
            [kg[:, :, bounds[j] : bounds[j + 1], :].mean(dim=2) for j in range(3)],
            dim=2,
        )  # [B, h, n_qgroups, n_kgroups]
        out = qg[0] if self.keep_heads else qg[0].mean(dim=0)
        self._mass_bufs[block_idx].append(out.detach().cpu())
        if self.full_maps:
            self._map_bufs[block_idx].append(probs[0].mean(dim=0).detach().cpu())

    def finish(self) -> tuple[np.ndarray, np.ndarray | None]:
        """캡처 종료 → (mass, maps).

        mass: ``[n_cross_blocks, K, n_qgroups, n_kgroups]`` float32
              (``keep_heads`` 시 ``[n_blocks, K, heads, n_qgroups, n_kgroups]``).
        maps: ``full_maps`` 시 ``[n_blocks, K, T_q, S]`` float16, 아니면 None.
        """
        self.active = False
        counts = {i: len(b) for i, b in self._mass_bufs.items()}
        uniq = set(counts.values())
        if len(uniq) != 1 or 0 in uniq:
            raise RuntimeError(f"uneven cross-attn capture counts per block: {counts}")
        mass = torch.stack(
            [torch.stack(self._mass_bufs[i], dim=0) for i in self.cross_block_indices],
            dim=0,
        ).numpy().astype(np.float32)
        maps = None
        if self.full_maps:
            maps = torch.stack(
                [torch.stack(self._map_bufs[i], dim=0) for i in self.cross_block_indices],
                dim=0,
            ).numpy().astype(np.float16)
        self._mass_bufs = {}
        self._map_bufs = {}
        return mass, maps

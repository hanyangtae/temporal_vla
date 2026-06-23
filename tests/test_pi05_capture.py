"""pi05 expert-block residual 멀티레이어 capture 단위 테스트 (GPU 불필요).

COAST A.7.1: π0.5 action expert(Gemma2, 18 layer, d=1024)의 residual stream을
``policy.model.paligemma_with_expert.gemma_expert.model.layers[i]`` 출력에서 캡처한다.
suffix sequence의 마지막 ``chunk_size`` 토큰(= action token)을 mean-pool 하여
denoise step 마다 1개 벡터를 얻고, K denoise step × L layer 로 ``[L, K, 1024]``를 만든다.

가짜 pi05 정책의 layer ModuleList 를 K 번 발화시켜
  (1) 결합된 capture shape 이 ``[L, K, D]`` 인지,
  (2) pooling 이 마지막 ``chunk_size`` 토큰의 mean 인지(known value 로 수학 검증)를 확인한다.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "serve"))

from safe_hooks import SafeFeatureCapture, run_with_features  # noqa: E402


class _FakeLayer(torch.nn.Module):
    """[B, S_full, D] residual stream을 그대로 통과시키는 가짜 decoder layer.

    HF Gemma decoder layer 처럼 출력을 tuple ``(hidden_states,)`` 로 감싼다
    (SafeForwardCapture._post_hook 가 tuple[0] 을 캡처하는지 검증).
    """

    def forward(self, hidden_states: torch.Tensor, *_a, **_k):
        return (hidden_states,)


class _FakeModelInner(torch.nn.Module):
    def __init__(self, n_layers: int):
        super().__init__()
        self.layers = torch.nn.ModuleList([_FakeLayer() for _ in range(n_layers)])


class _FakeGemmaExpert(torch.nn.Module):
    def __init__(self, n_layers: int):
        super().__init__()
        self.model = _FakeModelInner(n_layers)


class _FakePaliWithExpert(torch.nn.Module):
    def __init__(self, n_layers: int):
        super().__init__()
        self.gemma_expert = _FakeGemmaExpert(n_layers)


class _FakeConfig:
    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size


class _FakePi05Inner(torch.nn.Module):
    def __init__(self, n_layers: int, chunk_size: int):
        super().__init__()
        self.paligemma_with_expert = _FakePaliWithExpert(n_layers)
        self.config = _FakeConfig(chunk_size)
        # 일반 pi05 경로(_resolve_target)는 action_out_proj pre-hook 을 건다.
        self.action_out_proj = torch.nn.Identity()


class _FakePi05Policy(torch.nn.Module):
    """``.model.paligemma_with_expert.gemma_expert.model.layers`` 와
    ``.model.config.chunk_size`` 를 노출하는 가짜 pi05 정책."""

    def __init__(self, n_layers: int = 18, chunk_size: int = 10):
        super().__init__()
        self.model = _FakePi05Inner(n_layers, chunk_size)

    def fire_denoise(self, layer_indices, inputs):
        """선택한 layer 들을 각 denoise step 입력으로 순차 발화."""
        layers = self.model.paligemma_with_expert.gemma_expert.model.layers
        for x in inputs:  # 각 denoise step
            for i in layer_indices:
                layers[i](x)


def _make_inputs(K: int, B: int, S_full: int, D: int) -> list[torch.Tensor]:
    """K denoise step 입력. step k 의 텐서를 다른 값으로 채워 step 별 분리 검증 가능."""
    rng = np.random.default_rng(0)
    return [
        torch.tensor(rng.standard_normal((B, S_full, D)), dtype=torch.float32)
        for _ in range(K)
    ]


def test_assemble_shape_L_K_D():
    """4개 layer × K denoise step 발화 → capture shape == [L, K, D]."""
    L_layers = [0, 5, 11, 17]
    K, B, S_full, D, chunk = 10, 1, 12, 1024, 10
    policy = _FakePi05Policy(n_layers=18, chunk_size=chunk)
    cap = SafeFeatureCapture(policy, "pi05", pi05_expert_layers=L_layers)
    inputs = _make_inputs(K, B, S_full, D)
    with cap:
        policy.fire_denoise(L_layers, inputs)
    hidden = cap.assemble_expert_blocks()
    assert hidden is not None
    assert hidden.shape == (len(L_layers), K, D)


def test_pooling_is_mean_over_last_chunk_tokens():
    """pooling = 마지막 chunk_size 토큰의 mean (known value 로 수학 검증)."""
    L_layers = [0, 1]
    K, B, S_full, D, chunk = 3, 1, 12, 8, 10
    policy = _FakePi05Policy(n_layers=4, chunk_size=chunk)
    inputs = _make_inputs(K, B, S_full, D)
    cap = SafeFeatureCapture(policy, "pi05", pi05_expert_layers=L_layers)
    with cap:
        policy.fire_denoise(L_layers, inputs)
    hidden = cap.assemble_expert_blocks()
    assert hidden is not None
    # 두 layer 모두 identity passthrough 라 같은 입력 → 같은 pooled 값.
    for li in range(len(L_layers)):
        for k in range(K):
            expected = inputs[k][0, -chunk:, :].mean(dim=0).numpy().astype(np.float16)
            np.testing.assert_allclose(hidden[li, k], expected, rtol=0, atol=0)


def test_only_last_chunk_tokens_used():
    """앞쪽(non-action) 토큰을 바꿔도 pooled 결과 불변 (마지막 chunk 만 사용)."""
    L_layers = [0]
    K, B, S_full, D, chunk = 1, 1, 12, 8, 10
    policy = _FakePi05Policy(n_layers=2, chunk_size=chunk)
    base = torch.tensor(
        np.random.default_rng(1).standard_normal((B, S_full, D)), dtype=torch.float32
    )
    perturbed = base.clone()
    perturbed[0, : S_full - chunk, :] += 100.0  # 앞 2개 토큰만 크게 변경

    cap_a = SafeFeatureCapture(policy, "pi05", pi05_expert_layers=L_layers)
    with cap_a:
        policy.fire_denoise(L_layers, [base])
    hidden_a = cap_a.assemble_expert_blocks()

    cap_b = SafeFeatureCapture(policy, "pi05", pi05_expert_layers=L_layers)
    with cap_b:
        policy.fire_denoise(L_layers, [perturbed])
    hidden_b = cap_b.assemble_expert_blocks()

    np.testing.assert_allclose(hidden_a, hidden_b, rtol=0, atol=0)


def test_run_with_features_metadata():
    """run_with_features(pi05, pi05_expert_layers=...) → [L,K,D] + 올바른 meta."""
    L_layers = [0, 5, 11, 17]
    K, B, S_full, D, chunk = 10, 1, 12, 1024, 10
    inputs = _make_inputs(K, B, S_full, D)

    class _Policy(_FakePi05Policy):
        def select_action(self, batch):
            self.fire_denoise(L_layers, inputs)
            return torch.zeros((1, chunk, 7))

    policy = _Policy(n_layers=18, chunk_size=chunk)
    action, hidden, axes, meta = run_with_features(
        policy, {"task": ""}, "pi05", pi05_expert_layers=L_layers
    )
    assert hidden is not None
    assert hidden.shape == (len(L_layers), K, D)
    assert axes == ["layer", "denoise_step", "feature_dim"]
    assert meta["denoise_step_count"] == K
    assert meta["layer_count"] == len(L_layers)
    assert meta["feature_dim"] == D
    assert meta["capture_layers"] == L_layers


def test_groot_path_untouched():
    """pi05_expert_layers 미지정 시 일반 pi05 경로(action_out_proj pre-hook) 사용."""
    # capture target 이 pi05_expert_layers 분기로 새지 않는지 — 미지정이면 block 캡처 None.
    policy = _FakePi05Policy(n_layers=18, chunk_size=10)
    cap = SafeFeatureCapture(policy, "pi05", pi05_expert_layers=None)
    assert cap.pi05_expert_layers is None
    assert cap.assemble_expert_blocks() is None

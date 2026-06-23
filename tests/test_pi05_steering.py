"""Pi05ConceptorSteering forward-hook 단위 테스트 (GPU 불필요).

COAST A.7.1 steering(global): π0.5 action expert 의 decoder layer ℓ(default 11) 출력
residual stream에서 마지막 ``chunk_size`` action token 만 ``h' = h·Mᵀ`` 로 steer 한다.
주입 지점은 ``policy.model.paligemma_with_expert.gemma_expert.model.layers[ℓ]``.

검증:
  (1) M=I (β=0) → 출력 무변경,
  (2) 비-identity M → 마지막 chunk_size 토큰만 ``h·Mᵀ``, 앞 토큰 불변,
  (3) context 종료 후 hook 제거.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "serve"))

from steering_hooks import Pi05ConceptorSteering  # noqa: E402


class _FakeLayer(torch.nn.Module):
    """[B, S_full, D] 고정 출력을 tuple 로 내는 가짜 decoder layer (HF Gemma 모사)."""

    def __init__(self, out: torch.Tensor):
        super().__init__()
        self._out = out

    def forward(self, *_a, **_k):
        return (self._out,)


class _FakeModelInner(torch.nn.Module):
    def __init__(self, layers: list):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)


class _FakeGemmaExpert(torch.nn.Module):
    def __init__(self, layers: list):
        super().__init__()
        self.model = _FakeModelInner(layers)


class _FakePaliWithExpert(torch.nn.Module):
    def __init__(self, layers: list):
        super().__init__()
        self.gemma_expert = _FakeGemmaExpert(layers)


class _FakeConfig:
    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size


class _FakePi05Inner(torch.nn.Module):
    def __init__(self, layers: list, chunk_size: int):
        super().__init__()
        self.paligemma_with_expert = _FakePaliWithExpert(layers)
        self.config = _FakeConfig(chunk_size)


class _FakePi05Policy(torch.nn.Module):
    def __init__(self, layers: list, chunk_size: int):
        super().__init__()
        self.model = _FakePi05Inner(layers, chunk_size)


def _make(n_layers: int = 18, S_full: int = 12, chunk: int = 10, d: int = 16):
    rng = np.random.default_rng(0)
    outs = [
        torch.tensor(rng.standard_normal((2, S_full, d)), dtype=torch.float64)
        for _ in range(n_layers)
    ]
    layers = [_FakeLayer(o) for o in outs]
    model = _FakePi05Policy(layers, chunk_size=chunk)
    return model, outs


def _fire(model, layer):
    layers = model.model.paligemma_with_expert.gemma_expert.model.layers
    return layers[layer]()[0]


def test_only_last_chunk_tokens_steered():
    """default layer=11 에서 마지막 chunk_size 토큰만 h·Mᵀ, 앞 토큰 불변."""
    chunk, d = 10, 16
    model, outs = _make(chunk=chunk, d=d)
    rng = np.random.default_rng(1)
    M = rng.standard_normal((d, d))
    Mt = torch.tensor(M, dtype=torch.float64)
    out = outs[11]
    with Pi05ConceptorSteering(model, M):  # default layer=11
        y = _fire(model, 11)
    torch.testing.assert_close(y[:, :-chunk], out[:, :-chunk])
    torch.testing.assert_close(y[:, -chunk:], out[:, -chunk:] @ Mt.T)


def test_identity_matrix_no_change():
    """M=I (β=0) → 출력 무변경."""
    d = 16
    model, outs = _make(d=d)
    out = outs[11]
    with Pi05ConceptorSteering(model, np.eye(d)):
        y = _fire(model, 11)
    torch.testing.assert_close(y, out)


def test_scaled_identity_doubles_last_chunk():
    """M=2I 면 마지막 chunk 토큰만 2배, 앞 토큰 불변 (known value)."""
    chunk, d = 10, 16
    model, outs = _make(chunk=chunk, d=d)
    out = outs[11]
    with Pi05ConceptorSteering(model, np.eye(d) * 2.0):
        y = _fire(model, 11)
    torch.testing.assert_close(y[:, -chunk:], out[:, -chunk:] * 2.0)
    torch.testing.assert_close(y[:, :-chunk], out[:, :-chunk])


def test_custom_layer():
    """layer=5 지정 시 그 layer 만 steer, 다른 layer 불변."""
    chunk, d = 10, 16
    model, outs = _make(chunk=chunk, d=d)
    M = np.eye(d) * 3.0
    with Pi05ConceptorSteering(model, M, layer=5):
        y5 = _fire(model, 5)
        y11 = _fire(model, 11)
    torch.testing.assert_close(y5[:, -chunk:], outs[5][:, -chunk:] * 3.0)
    torch.testing.assert_close(y11, outs[11])  # 미적용


def test_hook_removed_after_context():
    """context 종료 후 hook 제거되어 steering 미적용."""
    d = 16
    model, outs = _make(d=d)
    with Pi05ConceptorSteering(model, np.eye(d) * 2.0):
        pass
    y = _fire(model, 11)
    torch.testing.assert_close(y, outs[11])


def test_chunk_size_from_config():
    """chunk_size 를 명시 안 하면 policy.model.config.chunk_size 사용."""
    chunk, d = 4, 16
    model, outs = _make(chunk=chunk, d=d)
    M = np.eye(d) * 5.0
    out = outs[11]
    with Pi05ConceptorSteering(model, M):
        y = _fire(model, 11)
    # config.chunk_size=4 → 마지막 4 토큰만 5배.
    torch.testing.assert_close(y[:, -chunk:], out[:, -chunk:] * 5.0)
    torch.testing.assert_close(y[:, :-chunk], out[:, :-chunk])

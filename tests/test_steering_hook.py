"""ConceptorSteering forward-hook 단위 테스트 (GPU 불필요).

가짜 DiT 모듈에 hook 을 걸어 (1) action token 만 h'=h·Mᵀ 로 변형되고 나머지
토큰은 불변인지, (2) β=0(M=I)이면 출력 무변경인지 검증한다.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "serve"))

from steering_hooks import ConceptorSteering, SetpointSteering  # noqa: E402


class _FakeDiT(torch.nn.Module):
    """[B, T, D] 고정 출력을 내는 가짜 DiT (state+future+action 토큰 모사)."""

    def __init__(self, out: torch.Tensor, blocks: list | None = None):
        super().__init__()
        self._out = out
        self.transformer_blocks = torch.nn.ModuleList(blocks or [])

    def forward(self, *_a, **_k) -> torch.Tensor:
        return self._out


class _FakeBlock(torch.nn.Module):
    """[B,T,D] 고정 출력 가짜 transformer_block (residual stream 모사)."""

    def __init__(self, out: torch.Tensor):
        super().__init__()
        self._out = out

    def forward(self, *_a, **_k) -> torch.Tensor:
        return self._out


class _FakeVLLN(torch.nn.Module):
    """[B, seq, D] 고정 출력 가짜 vlln (post-LayerNorm VL features 모사)."""

    def __init__(self, out: torch.Tensor):
        super().__init__()
        self._out = out

    def forward(self, *_a, **_k) -> torch.Tensor:
        return self._out


class _FakeHead(torch.nn.Module):
    def __init__(self, dit: _FakeDiT, horizon: int, vlln: _FakeVLLN | None = None):
        super().__init__()
        self.model = dit
        self.action_horizon = horizon
        if vlln is not None:
            self.vlln = vlln


class _FakeGroot(torch.nn.Module):
    def __init__(self, head: _FakeHead):
        super().__init__()
        self.action_head = head


def _make(horizon: int = 4, n_tokens: int = 10, d: int = 16):
    rng = np.random.default_rng(0)
    out = torch.tensor(rng.standard_normal((2, n_tokens, d)), dtype=torch.float64)
    model = _FakeGroot(_FakeHead(_FakeDiT(out), horizon))
    return model, out


def test_only_action_tokens_steered():
    """마지막 horizon 토큰만 h·Mᵀ, 앞 토큰은 불변."""
    horizon, d = 4, 16
    model, out = _make(horizon=horizon, d=d)
    rng = np.random.default_rng(1)
    M = rng.standard_normal((d, d))
    with ConceptorSteering(model, M, horizon=horizon):
        y = model.action_head.model(out)
    Mt = torch.tensor(M, dtype=torch.float64)
    # 앞쪽(non-action) 토큰 불변
    torch.testing.assert_close(y[:, :-horizon], out[:, :-horizon])
    # action 토큰 = out @ Mᵀ
    torch.testing.assert_close(y[:, -horizon:], out[:, -horizon:] @ Mt.T)


def test_identity_matrix_no_change():
    """M=I 이면 출력 무변경 (β=0 대응)."""
    d = 16
    model, out = _make(d=d)
    with ConceptorSteering(model, np.eye(d)):
        y = model.action_head.model(out)
    torch.testing.assert_close(y, out)


def test_hook_removed_after_context():
    """context 종료 후 hook 제거되어 steering 미적용."""
    d = 16
    model, out = _make(d=d)
    M = np.eye(d) * 2.0
    with ConceptorSteering(model, M):
        pass
    y = model.action_head.model(out)
    torch.testing.assert_close(y, out)


def test_layer_targets_transformer_block():
    """layer=i 면 transformer_blocks[i] 출력만 h·Mᵀ, 다른 block은 불변 (COAST 중간 layer)."""
    horizon, d, n_blocks = 4, 16, 3
    rng = np.random.default_rng(8)
    block_out = torch.tensor(rng.standard_normal((2, 10, d)), dtype=torch.float64)
    dit_out = torch.tensor(rng.standard_normal((2, 10, d)), dtype=torch.float64)
    blocks = [_FakeBlock(block_out) for _ in range(n_blocks)]
    model = _FakeGroot(_FakeHead(_FakeDiT(dit_out, blocks), horizon))
    M = rng.standard_normal((d, d))
    Mt = torch.tensor(M, dtype=torch.float64)

    with ConceptorSteering(model, M, layer=1, horizon=horizon):
        y1 = model.action_head.model.transformer_blocks[1](block_out)  # hooked
        y0 = model.action_head.model.transformer_blocks[0](block_out)  # not hooked
    # block 1: action token만 steer, 앞 토큰 불변
    torch.testing.assert_close(y1[:, -horizon:], block_out[:, -horizon:] @ Mt.T)
    torch.testing.assert_close(y1[:, :-horizon], block_out[:, :-horizon])
    # block 0: 미적용
    torch.testing.assert_close(y0, block_out)


def _make_vl(seq: int = 12, d: int = 16):
    """vlln 출력 [B, seq, D] 가짜 모델 (goal pathway)."""
    rng = np.random.default_rng(3)
    vl_out = torch.tensor(rng.standard_normal((2, seq, d)), dtype=torch.float64)
    dit_out = torch.tensor(rng.standard_normal((2, 10, d)), dtype=torch.float64)
    model = _FakeGroot(_FakeHead(_FakeDiT(dit_out), horizon=4, vlln=_FakeVLLN(vl_out)))
    return model, vl_out


def test_vl_pathway_steers_all_tokens():
    """pathway='vl' 면 vlln 출력의 모든 VL token 이 h·Mᵀ (horizon slicing 없음)."""
    d, seq = 16, 12
    model, vl_out = _make_vl(seq=seq, d=d)
    rng = np.random.default_rng(4)
    M = rng.standard_normal((d, d))
    Mt = torch.tensor(M, dtype=torch.float64)
    with ConceptorSteering(model, M, pathway="vl"):
        y = model.action_head.vlln(vl_out)
    # 전체 token steer
    torch.testing.assert_close(y, vl_out @ Mt.T)


def test_vl_pathway_identity_no_change():
    """vl pathway 에서도 M=I 면 무변경 (β=0 대응)."""
    d = 16
    model, vl_out = _make_vl(d=d)
    with ConceptorSteering(model, np.eye(d), pathway="vl"):
        y = model.action_head.vlln(vl_out)
    torch.testing.assert_close(y, vl_out)


def test_vl_pathway_does_not_touch_dit():
    """vl pathway hook 은 DiT 출력에 영향 없음 (분리 검증)."""
    d = 16
    model, _ = _make_vl(d=d)
    dit_out = model.action_head.model._out
    M = np.eye(d) * 3.0
    with ConceptorSteering(model, M, pathway="vl"):
        y_dit = model.action_head.model(dit_out)
    torch.testing.assert_close(y_dit, dit_out)


def test_vl_pathway_rejects_layer():
    """pathway='vl' + layer 지정은 에러 (vlln 단일 지점)."""
    d = 16
    model, _ = _make_vl(d=d)
    try:
        ConceptorSteering(model, np.eye(d), pathway="vl", layer=0)
    except ValueError:
        return
    raise AssertionError("pathway='vl' 에 layer 를 주면 ValueError 여야 함")


# --------------------------------------------------------------------------- #
# 세그먼트 setpoint(setM) — seg_mask 는 0/1 플래그가 아니라 게인 승수 (회귀 방지)
# 배경: Codex 리뷰가 "serve 가 mask 를 0/1 스킵으로만 해석 → 위약 dose 매칭 붕괴"
# 라고 지적(false positive). 실제로는 mask 를 float 게인으로 곱한다. 아래 테스트가
# 그 반의미를 잠근다 — 누군가 bool 로 강등하면 즉시 실패.
# --------------------------------------------------------------------------- #
def _seg_op(beta: float, v_seg, s_tok, bounds, mask):
    """모델 없이 _apply_segment 만 검증하는 최소 SetpointSteering 객체."""
    op = object.__new__(SetpointSteering)
    op.beta = float(beta)
    op._seg = (np.asarray(v_seg, float), np.asarray(s_tok, float),
               np.asarray(bounds, int), np.asarray(mask, float))
    op._seg_cache = None
    return op


def _seg_fixture(T=49, D=8, seed=0):
    rng = np.random.default_rng(seed)
    bounds = np.array([[0, 1], [1, 33], [33, T]])
    v = rng.normal(size=(3, D)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    s = rng.normal(size=T)
    x = torch.as_tensor(rng.normal(size=(4, T, D)) * 5.0, dtype=torch.float64)
    return bounds, v, s, x


def test_seg_mask_is_gain_multiplier():
    """mask=[2,2,2] 의 delta 는 mask=[1,1,1] 의 정확히 2배 (게인 승수)."""
    bounds, v, s, x = _seg_fixture()
    d1 = x - _seg_op(1.0, v, s, bounds, [1, 1, 1])._apply_segment(x)
    d2 = x - _seg_op(1.0, v, s, bounds, [2, 2, 2])._apply_segment(x)
    ratio = float(d2.abs().sum() / d1.abs().sum())
    assert abs(ratio - 2.0) < 1e-6, f"seg_mask 가 게인 승수가 아님 (비율 {ratio}); bool 강등 의심"


def test_seg_mask_zero_skips_segment():
    """mask=[0,1,0] 은 state·action 세그먼트를 무개입, future 만 개입 (future_only arm)."""
    bounds, v, s, x = _seg_fixture()
    d = (x - _seg_op(1.0, v, s, bounds, [0, 1, 0])._apply_segment(x)).abs()
    assert float(d[:, 0:1].sum()) == 0.0 and float(d[:, 33:].sum()) == 0.0
    assert float(d[:, 1:33].sum()) > 0.0


def test_seg_mask_fractional_gain_scales_dose():
    """위약 dose-match: mask=scale 이면 세그먼트 이동량이 scale 배 (raw×scale=처치)."""
    bounds, v, s, x = _seg_fixture()
    scale = [1.951, 2.0, 1.178]
    d_raw = (x - _seg_op(1.0, v, s, bounds, [1, 1, 1])._apply_segment(x)).abs()
    d_scaled = (x - _seg_op(1.0, v, s, bounds, scale)._apply_segment(x)).abs()
    for si, (lo, hi) in enumerate(bounds):
        got = float(d_scaled[:, lo:hi].sum() / d_raw[:, lo:hi].sum())
        assert abs(got - scale[si]) < 1e-6, f"seg {si} 이동량 배율 {got} != scale {scale[si]}"

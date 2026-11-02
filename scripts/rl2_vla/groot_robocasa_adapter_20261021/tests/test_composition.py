"""Composed sampler vs 실제 lerobot/upstream get_action 소스 (fake 모듈, CPU)."""
import types

import numpy as np
import pytest
import torch
from torch import nn

from conftest import LEROBOT_HEAD, UPSTREAM_HEAD, extract_source
from composition import (
    SPEC,
    ComposedFlowSampler,
    FeatureRecord,
    FeatureSpec,
    FirstCallCapture,
    compose_velocity,
)
from contract import ContractError

B = 2


class BF(dict):
    """transformers BatchFeature 최소 stub (attr 접근 + data= 생성자)."""

    def __init__(self, data=None):
        super().__init__(data or {})

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e


class Block(nn.Module):
    def __init__(self, i):
        super().__init__()
        self.scale = nn.Parameter(torch.full((1536,), 0.9 + 0.01 * i))
        self.bias = nn.Parameter(torch.randn(1536) * 0.01)

    def forward(self, h, enc, temb):
        return torch.tanh(h * self.scale + self.bias + enc.mean(1, keepdim=True) + temb[:, None])


class FakeDiT(nn.Module):
    def __init__(self, n_blocks=16):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(Block(i) for i in range(n_blocks))
        self.proj = nn.Linear(1536, 1024)

    def forward(self, hidden_states, encoder_hidden_states, timestep):
        temb = (timestep.float() / 1000.0)[:, None].expand(-1, 1536)
        h = hidden_states
        for blk in self.transformer_blocks:
            h = blk(h, encoder_hidden_states, temb)
        return self.proj(h)


class FakeDecoder(nn.Module):
    def __init__(self, action_dim, nan_velocity):
        super().__init__()
        self.dec = nn.Linear(1024, action_dim)
        self.nan_velocity = nan_velocity

    def forward(self, x, emb):
        out = self.dec(x)
        if self.nan_velocity:
            out = out.clone()
            out[:, -1, 0] = float("nan")
        return out


class FakeHead(nn.Module):
    def __init__(self, *, steps=4, horizon=16, action_dim=32, n_blocks=16, nan_velocity=False,
                 hidden_size=1024):
        super().__init__()
        self.config = types.SimpleNamespace(action_horizon=horizon, action_dim=action_dim,
                                            add_pos_embed=True, input_embedding_dim=1536,
                                            hidden_size=hidden_size)
        self.num_inference_timesteps = steps
        self.num_timestep_buckets = 1000
        self.action_horizon = horizon
        self.state_lin = nn.Linear(64, 1536)
        self.act_lin = nn.Linear(action_dim, 1536)
        self.position_embedding = nn.Embedding(horizon, 1536)
        self.future_tokens = nn.Embedding(32, 1536)
        self.model = FakeDiT(n_blocks)
        # 실제 action_decoder 는 nn.Module(CategorySpecificMLP, 입력 hidden_size 1024).
        self.action_decoder = FakeDecoder(action_dim, nan_velocity)

    def process_backbone_output(self, bo):
        bo["backbone_features"] = bo["backbone_features"] * 1.0
        return bo

    def state_encoder(self, state, emb):
        return self.state_lin(state)

    def action_encoder(self, actions, t, emb):
        return self.act_lin(actions) + (t.float() / 1000.0)[:, None, None]



@pytest.fixture(scope="module", params=[LEROBOT_HEAD, UPSTREAM_HEAD], ids=["lerobot", "upstream"])
def upstream_get_action(request):
    ns = {"torch": torch, "BatchFeature": BF}
    exec(extract_source(request.param, "get_action"), ns)
    return ns["get_action"]


def make_inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    bo = BF({"backbone_features": torch.randn(B, 7, 1536, generator=g)})
    ai = BF({"state": torch.randn(B, 1, 64, generator=g), "embodiment_id": torch.zeros(B, dtype=torch.long)})
    return bo, ai


@pytest.fixture(scope="module")
def head():
    torch.manual_seed(123)
    return FakeHead().eval()



class CountingQAM:
    def __init__(self, value=0.0, shape=None, nan=False):
        self.calls = []
        self.value, self.shape, self.nan = value, shape, nan

    def __call__(self, feature, noisy, i):
        self.calls.append((feature.copy(), noisy.copy(), i))
        out = np.full(self.shape or noisy.shape, self.value, np.float32)
        if self.nan:
            out[0, 0] = np.nan
        return out


def run_upstream(get_action, head, seed=7):
    bo, ai = make_inputs()
    torch.manual_seed(seed)
    out = get_action(head, bo, ai)["action_pred"]
    return out, torch.get_rng_state()


def run_ours(head, sampler, seed=7, **kw):
    bo, ai = make_inputs()
    torch.manual_seed(seed)
    out, info = sampler.sample(head, bo, ai, **kw)
    return out, torch.get_rng_state(), info


# ---------------------------------------------------------------- gate0 identity
@pytest.mark.parametrize("gate,weight", [(0.0, 1.0), (0.0, 0.5), (1.0, 0.0)])
def test_gate0_bit_identical_and_rng_preserved(upstream_get_action, head, gate, weight):
    ref, ref_rng = run_upstream(upstream_get_action, head)
    qam = CountingQAM(value=5.0)
    np_state = np.random.get_state()[1].copy()
    out, rng, info = run_ours(head, ComposedFlowSampler(SPEC, qam_fn=qam, qam_weight=weight),
                              decision_env_step=10, gate=gate)
    assert torch.equal(out, ref)
    assert torch.equal(rng, ref_rng)
    assert np.array_equal(np.random.get_state()[1], np_state)
    assert qam.calls == [] and info["lam"] == 0.0
    assert info["feature"].shape == (B, 1024)  # gate0 에서도 feature 는 기록


def test_gate1_preserves_rng_sequence(upstream_get_action, head):
    _, ref_rng = run_upstream(upstream_get_action, head)
    _, rng, _ = run_ours(head, ComposedFlowSampler(SPEC, qam_fn=CountingQAM(1.0), qam_weight=0.7),
                         decision_env_step=0, gate=1.0)
    assert torch.equal(rng, ref_rng)  # 다음 호출/episode noise 정렬 유지


# ---------------------------------------------------------------- composition semantics
def _initial_noise(seed=7):
    torch.manual_seed(seed)
    return torch.randn(B, 16, 32)


def test_full_qam_weight_sign_time_and_padding(upstream_get_action, head):
    noise = _initial_noise()
    ref, _ = run_upstream(upstream_get_action, head)
    qam = CountingQAM(value=1.0)
    out, _, info = run_ours(head, ComposedFlowSampler(SPEC, qam_fn=qam, qam_weight=1.0),
                            decision_env_step=0, gate=1.0)
    # active12: v=+1 for 4 steps of dt=1/4 → noise + 1 (GR00T t:0→1, 부호 +)
    torch.testing.assert_close(out[..., :12], noise[..., :12] + 1.0)
    assert [c[2] for c in qam.calls] == [0, 1, 2, 3]  # QAM t=i/4 == GR00T t_cont
    assert all(c[1].shape == (B, 192) for c in qam.calls)
    np.testing.assert_allclose(qam.calls[0][1], noise[..., :12].reshape(B, -1).numpy())
    # feature 는 4 step 모두 동일 (flow step 0 고정) & gate0 feature 와 같음
    assert all(np.array_equal(c[0], qam.calls[0][0]) for c in qam.calls)
    _, _, info0 = run_ours(head, ComposedFlowSampler(SPEC, qam_fn=None, qam_weight=1.0),
                           decision_env_step=0, gate=0.0)
    np.testing.assert_array_equal(info["feature"], info0["feature"])


def test_compose_velocity_formula():
    v = torch.randn(B, 16, 32)
    q = torch.randn(B, 16, 12)
    c = compose_velocity(v, q, 0.3)
    torch.testing.assert_close(c[..., :12], 0.7 * v[..., :12] + 0.3 * q)
    assert torch.equal(c[..., 12:], v[..., 12:])
    assert compose_velocity(v, None, 0.0) is v
    with pytest.raises(ContractError):
        compose_velocity(v, None, 0.3)


def _decoder_inputs(head):
    """독립 hook 으로 action_decoder 입력 4개를 모두 기록 (비교 기준)."""
    seen = []
    h = head.action_decoder.register_forward_pre_hook(lambda m, a: seen.append(a[0].detach().clone()))
    try:
        _, _, info = run_ours(head, ComposedFlowSampler(SPEC, qam_fn=None, qam_weight=0.0),
                              decision_env_step=0, gate=0.0)
    finally:
        h.remove()
    return seen, info


def test_feature_is_first_call_decoder_input_last16_mean(head):
    seen, info = _decoder_inputs(head)
    assert len(seen) == 4 and info["decoder_calls"] == 4
    assert all(x.shape == (B, 49, 1024) for x in seen)
    expect = seen[0][:, -16:].mean(1).numpy()  # 첫 denoise call, 마지막 16 action token
    np.testing.assert_allclose(info["feature"], expect, rtol=1e-6)
    np.testing.assert_allclose(info["feature"], seen[0][:, 33:49].mean(1).numpy(), rtol=1e-6)
    for later in seen[1:]:  # 이후 call 값으로 바뀌지 않았음
        assert not np.allclose(info["feature"], later[:, -16:].mean(1).numpy())
    assert info["feature"].flags.writeable is False
    assert len(head.action_decoder._forward_pre_hooks) == 0  # hook 정리


def test_feature_excludes_state_and_future_tokens(head):
    _, base = _decoder_inputs(head)

    def corrupt(_m, _i, out):
        out = out.clone()
        out[:, :33] += 1e3  # state(0:1) + future(1:33)
        return out

    h = head.model.register_forward_hook(corrupt)
    try:
        _, _, info = run_ours(head, ComposedFlowSampler(SPEC, qam_fn=None, qam_weight=0.0),
                              decision_env_step=0, gate=0.0)
    finally:
        h.remove()
    np.testing.assert_array_equal(info["feature"], base["feature"])


def test_feature_retained_across_four_flow_steps(head):
    qam = CountingQAM(value=0.3)
    _, _, info = run_ours(head, ComposedFlowSampler(SPEC, qam_fn=qam, qam_weight=0.5),
                          decision_env_step=0, gate=1.0)
    seen, info0 = _decoder_inputs(head)
    assert len(qam.calls) == 4
    for f, _, _ in qam.calls:
        np.testing.assert_array_equal(f, info0["feature"])  # 합성 후 x 에 의존하지 않음


def test_first_call_capture_is_write_once():
    cap = FirstCallCapture()
    a, b = torch.ones(1, 49, 1024), torch.zeros(1, 49, 1024)
    cap(None, (a,))
    cap(None, (b,))
    cap(None, (b,))
    assert cap.calls == 3 and torch.equal(cap.first, a)
    with pytest.raises(ContractError, match="write-once"):
        cap._store(b)
    a.fill_(5.0)  # 원본 텐서가 바뀌어도 보관값은 복사본
    assert torch.equal(cap.first, torch.ones(1, 49, 1024))
    with pytest.raises(ContractError, match="호출되지"):
        FirstCallCapture().first


@pytest.mark.parametrize("kw", [
    dict(capture_point="action_head.model.transformer_blocks[11]", dim=1536),
    dict(dim=1536),
    dict(dim=2048),
    dict(token_selection="all_49_tokens"),
    dict(token_selection="last15_drop_first"),
    dict(pool="max"),
    dict(denoise_step=1),
])
def test_feature_spec_rejects_other_interfaces(kw):
    with pytest.raises(ContractError):
        FeatureSpec(**kw)


def test_pool_rejects_block_residual_tensor():
    with pytest.raises(ContractError, match="1536"):
        SPEC.pool_decoder_input(torch.zeros(1, 49, 1536))
    with pytest.raises(ContractError):
        SPEC.pool_decoder_input(torch.zeros(1, 48, 1024))  # token 하나 빠진 입력


# ---------------------------------------------------------------- rejections
def _rec(**kw):
    base = dict(value=np.zeros((B, 1024), np.float32), env_step=10, flow_step=0,
                signature=SPEC.signature())
    base.update(kw)
    return FeatureRecord(**base)


@pytest.mark.parametrize("rec,match", [
    (_rec(env_step=15), "미래"),
    (_rec(env_step=5), "stale"),
    (_rec(flow_step=1), "flow_step"),
    (_rec(signature="deadbeefdeadbeef"), "signature"),
    (_rec(value=np.zeros((B, 1536), np.float32)), "block residual"),
    (_rec(value=np.zeros((B, 2048), np.float32)), "block residual"),
    (_rec(value=np.zeros((B, 512), np.float32)), "shape"),
    (_rec(value=np.full((B, 1024), np.nan, np.float32)), "비유한"),
])
def test_external_feature_rejections(head, rec, match):
    s = ComposedFlowSampler(SPEC, qam_fn=CountingQAM(), qam_weight=1.0)
    with pytest.raises(ContractError, match=match):
        run_ours(head, s, decision_env_step=10, gate=1.0, external_feature=rec)


def test_external_feature_valid_is_used(head):
    qam = CountingQAM()
    rec = _rec(value=np.ones((B, 1024), np.float32))
    run_ours(head, ComposedFlowSampler(SPEC, qam_fn=qam, qam_weight=1.0),
             decision_env_step=10, gate=1.0, external_feature=rec)
    assert np.all(qam.calls[0][0] == 1.0)


@pytest.mark.parametrize("qam", [CountingQAM(nan=True), CountingQAM(shape=(B, 28)),
                                 CountingQAM(shape=(B, 4, 7)), CountingQAM(shape=(B, 16, 32))])
def test_bad_qam_velocity_rejected(head, qam):
    with pytest.raises(ContractError):
        run_ours(head, ComposedFlowSampler(SPEC, qam_fn=qam, qam_weight=0.5),
                 decision_env_step=0, gate=1.0)


def test_nonfinite_vla_velocity_rejected():
    torch.manual_seed(0)
    h = FakeHead(nan_velocity=True).eval()
    with pytest.raises(ContractError, match="비유한"):
        run_ours(h, ComposedFlowSampler(SPEC, qam_fn=None, qam_weight=0.0),
                 decision_env_step=0, gate=0.0)


@pytest.mark.parametrize("kw", [dict(steps=10), dict(horizon=4), dict(action_dim=7), dict(n_blocks=8),
                                dict(hidden_size=1536)])
def test_wrong_head_contract_rejected(kw):
    torch.manual_seed(0)
    h = FakeHead(**kw).eval()
    with pytest.raises(ContractError, match="계약"):
        run_ours(h, ComposedFlowSampler(SPEC, qam_fn=None, qam_weight=0.0),
                 decision_env_step=0, gate=0.0)


def test_gate_and_spec_validation(head):
    with pytest.raises(ContractError):
        ComposedFlowSampler(SPEC, qam_fn=None, qam_weight=1.5)
    s = ComposedFlowSampler(SPEC, qam_fn=CountingQAM(), qam_weight=1.0)
    with pytest.raises(ContractError, match="0/1"):
        run_ours(head, s, decision_env_step=0, gate=0.5)
    with pytest.raises(ContractError, match="gate 미지정"):
        run_ours(head, s, decision_env_step=0)
    with pytest.raises(ContractError, match="QAM 없음"):
        run_ours(head, ComposedFlowSampler(SPEC, qam_fn=None, qam_weight=1.0),
                 decision_env_step=0, gate=1.0)


def test_gate_fn_sees_step0_feature(head):
    seen = []
    s = ComposedFlowSampler(SPEC, qam_fn=CountingQAM(), qam_weight=1.0,
                            gate_fn=lambda f: seen.append(f.copy()) or 0.0)
    _, _, info = run_ours(head, s, decision_env_step=0)
    assert len(seen) == 1 and np.array_equal(seen[0], info["feature"]) and info["lam"] == 0.0


def test_make_get_action_returns_batchfeature(upstream_get_action, head):
    s = ComposedFlowSampler(SPEC, qam_fn=None, qam_weight=0.0)
    fn = s.make_get_action(head, lambda: {"decision_env_step": 5, "gate": 0.0})
    ref, _ = run_upstream(upstream_get_action, head)
    bo, ai = make_inputs()
    torch.manual_seed(7)
    out = fn(bo, ai)
    assert isinstance(out, BF) and torch.equal(out["action_pred"], ref)
    assert fn.last_info["decision_env_step"] == 5

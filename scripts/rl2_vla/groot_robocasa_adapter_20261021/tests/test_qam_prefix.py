"""prefix-critic QAM trainer (CPU, JAX, synthetic). 실제 데이터셋 학습·GPU parity 아님."""
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from composition import SPEC, ComposedFlowSampler
from contract import ContractError, validate_qam_flags
from qam_prefix import (
    PREFIX_CONTRACT_KEY,
    PrefixCriticQAMAgent,
    PrefixQAMTrainer,
    QAMAgent,
    make_config,
    prefix_mask,
    to_qam_batch,
)
from smoke_prefix_qam import SYNTHETIC_LABEL, synthetic_sets
from transitions import TrainSampler

GAMMA = 0.99
SIG = SPEC.signature()


@pytest.fixture(scope="module")
def sets():
    return synthetic_sets(0, GAMMA)


@pytest.fixture(scope="module")
def trainer(sets):
    tr = PrefixQAMTrainer.create(profile="smoke_reduced_network", gamma=GAMMA, seed=0,
                                 data_label=SYNTHETIC_LABEL)
    tr.fit(TrainSampler(sets[0], val=sets[1], seed=1), steps=3, batch_size=8)  # 초기값 대칭 깨기
    return tr


def _all_rows(ts):
    return {k: v for k, v in ts.data.items()}


@pytest.fixture(scope="module")
def batch(trainer, sets):
    b = trainer.batch(_all_rows(sets[0]))
    m = b["exec_mask"].sum(-1)
    assert (m < 5).any() and (m == 5).any()                      # partial 행 포함
    assert (b["discounts"] == 0).any() and (b["chunk_valid"] == 0).any()
    return b


def _critic(agent, name, obs, acts, mask):
    return agent.network.select(name)(obs, acts, mask)


def _perturb_tail(acts_flat, mask, seed):
    """mask 밖 (실행되지 않은) 위치만 큰 값으로 바꾼다."""
    am = np.repeat(np.asarray(mask), 12, axis=-1).astype(bool)
    noise = np.random.default_rng(seed).normal(scale=5.0, size=acts_flat.shape).astype(np.float32)
    return np.where(am, acts_flat, noise)


# --------------------------------------------------------------------- config / fail-closed

def test_config_profiles_keep_contract():
    prod = make_config("production", gamma=GAMMA)
    smoke = make_config("smoke_reduced_network", gamma=GAMMA)
    for c in (prod, smoke):
        assert (c["horizon_length"], c["flow_steps"], c["best_of_n"], c["execute_steps"]) == (16, 4, 1, 5)
        assert c["action_chunking"] and c["fql_alpha"] == 0 and c["edit_scale"] == 0 and not c["residual"]
        assert c["inv_temp"] == 0.1 and c["discount"] == GAMMA
    assert prod["actor_hidden_dims"] == (512,) * 4 and prod["num_qs"] == 10
    assert smoke["actor_hidden_dims"] == (32, 32) and smoke["profile"] == "smoke_reduced_network"
    for bad in ({"actor_hidden_dims": (8,)}, {"horizon_length": 5}, {"flow_steps": 10},
                {"best_of_n": 4}, {"fql_alpha": 0.1}):
        with pytest.raises(ContractError, match="override"):
            make_config("production", gamma=GAMMA, **bad)
    with pytest.raises(ContractError, match="override"):
        make_config("smoke_reduced_network", gamma=GAMMA, horizon_length=5)
    with pytest.raises(ContractError):
        make_config("full", gamma=GAMMA)


@pytest.mark.parametrize("patch,match", [
    ({"fql_alpha": 0.1}, "fql_alpha"),
    ({"edit_scale": 0.1}, "edit_scale"),
    ({"residual": True}, "residual"),
    ({"action_chunking": False}, "action_chunking"),
    ({"execute_steps": 0}, "execute_steps"),
    ({"execute_steps": 17}, "execute_steps"),
])
def test_unsupported_branches_fail_closed(patch, match):
    cfg = dict(make_config("smoke_reduced_network", gamma=GAMMA), **patch)
    with pytest.raises(ContractError, match=match):
        PrefixCriticQAMAgent.create(0, jnp.zeros(1024), jnp.zeros(12), cfg)


def test_unmasked_critic_access_fails_closed(trainer, batch):
    """mask 없이 critic 을 부르는 upstream 경로는 실행 자체가 실패한다."""
    a = trainer.agent
    obs, acts = batch["observations"][:4], batch["actions"][:4].reshape(4, -1)
    with pytest.raises(TypeError):
        a.network.select("critic")(obs, acts)
    with pytest.raises(TypeError):
        a.network.select("target_critic")(obs, acts)
    up_batch = {"observations": obs, "actions": batch["actions"][:4],
                "next_observations": batch["next_observations"][:4, None],
                "rewards": batch["rewards"][:4, None], "masks": np.ones((4, 1), np.float32),
                "valid": np.ones((4, 1), np.float32)}
    with pytest.raises(TypeError):
        QAMAgent.critic_loss(a, up_batch, a.network.params, jax.random.PRNGKey(0))


# --------------------------------------------------------------------- tail invariance

@pytest.mark.parametrize("m", [5, 3, 1])
def test_critic_tail_perturbation_invariance_and_zero_tail_gradient(trainer, batch, m):
    a = trainer.agent
    obs = batch["observations"][:6]
    acts = batch["actions"][:6].reshape(6, -1)
    mask = np.asarray(prefix_mask(np.full(6, m)))
    for name in ("critic", "target_critic"):
        q0 = np.asarray(_critic(a, name, obs, acts, mask))
        q1 = np.asarray(_critic(a, name, obs, _perturb_tail(acts, mask, m), mask))
        assert np.array_equal(q0, q1)
        g = np.asarray(jax.grad(lambda y: _critic(a, name, obs, y, mask).mean(0).sum())(jnp.asarray(acts)))
        g = g.reshape(6, 16, 12)
        assert np.all(g[:, m:] == 0.0)
        assert np.all(np.abs(g[:, :m]).sum((1, 2)) > 0)
    # mask/duration 조건: 같은 prefix 값이라도 duration 이 다르면 다른 Q (tail 0 과 '미실행' 구분)
    other = np.asarray(prefix_mask(np.full(6, max(1, m - 1) if m > 1 else 2)))
    assert not np.allclose(np.asarray(_critic(a, "critic", obs, acts * np.repeat(mask, 12, -1), mask)),
                           np.asarray(_critic(a, "critic", obs, acts * np.repeat(mask, 12, -1), other)))


def test_critic_loss_invariant_to_unexecuted_tail_per_row_partial(trainer, batch):
    a = trainer.agent
    rng = jax.random.PRNGKey(3)
    l0, _ = a.critic_loss(batch, a.network.params, rng)
    pert = dict(batch)
    pert["actions"] = _perturb_tail(batch["actions"].reshape(len(batch["actions"]), -1),
                                    batch["exec_mask"], 9).reshape(batch["actions"].shape)
    l1, _ = a.critic_loss(pert, a.network.params, rng)
    assert float(l0) == float(l1)
    # critic loss 의 dataset action gradient: 각 행의 실행 prefix (partial 포함) 밖은 정확히 0
    def f(acts):
        return a.critic_loss(dict(batch, actions=acts), a.network.params, rng)[0]
    g = np.asarray(jax.grad(f)(jnp.asarray(batch["actions"])))
    em = batch["exec_mask"].astype(bool)
    assert np.all(g[~em] == 0.0)
    partial = em.sum(-1) < 5
    assert partial.any() and np.all(np.abs(g[partial][em[partial]]).sum(-1) > 0)


def test_target_critic_invariant_to_next_action_tail(trainer, batch):
    a = trainer.agent
    na = np.asarray(a.sample_actions(batch["next_observations"], rng=jax.random.PRNGKey(5)))
    t0 = np.asarray(a.td_target(batch, None, next_actions=na))
    na2 = _perturb_tail(na, batch["next_exec_mask"], 11)
    t1 = np.asarray(a.td_target(batch, None, next_actions=na2))
    assert np.array_equal(t0, t1)
    # 명목 m 밖을 바꿔도 같고, 실행 prefix 를 바꾸면 bootstrap 행에서 달라진다
    na3 = na.copy()
    na3[:, :12] = -na3[:, :12]
    t3 = np.asarray(a.td_target(batch, None, next_actions=na3))
    boot = batch["discounts"] > 0
    assert not np.allclose(t0[boot], t3[boot]) and np.array_equal(t0[~boot], t3[~boot])


def test_ranking_and_adjoint_invariant_to_tail_and_zero_tail(trainer, batch):
    a = trainer.agent
    obs = batch["observations"][:5]
    cand = np.random.default_rng(0).uniform(-1, 1, size=(5, 3, 192)).astype(np.float32)
    obs_n = np.repeat(obs[:, None], 3, axis=1)
    nom = np.asarray(a.nominal_mask((5, 3)))
    s0 = np.asarray(a.rank_candidates(obs_n, cand))
    s1 = np.asarray(a.rank_candidates(obs_n, _perturb_tail(cand, nom, 1)))
    assert np.array_equal(s0, s1) and np.array_equal(s0.argmax(-1), s1.argmax(-1))

    x1 = cand[:, 0]
    adj = np.asarray(a.terminal_adjoint(obs, x1)).reshape(5, 16, 12)
    assert np.all(adj[:, 5:] == 0.0) and np.all(np.abs(adj[:, :5]).sum((1, 2)) > 0)
    adj2 = np.asarray(a.terminal_adjoint(obs, _perturb_tail(x1, nom[:, 0], 2))).reshape(5, 16, 12)
    assert np.array_equal(adj, adj2)

    # 독립 대조: prefix 60 차원만 받는 함수의 gradient 와 같다 (tail 은 critic 에서 자유도가 아님)
    name = "target_critic" if a.config["use_target_grad"] else "critic"
    def q_prefix(p):
        full = jnp.concatenate([p, jnp.zeros((5, 132))], -1)
        return _critic(a, name, obs, jnp.clip(full, -1, 1), nom[:, 0]).mean(0).sum()
    ref = -np.asarray(jax.grad(q_prefix)(jnp.asarray(x1[:, :60]))) * a.config["inv_temp"]
    np.testing.assert_allclose(adj.reshape(5, -1)[:, :60], ref, rtol=1e-6, atol=1e-9)


def test_adj_matching_uses_masked_terminal_adjoint(trainer, batch):
    """adj_matching 의 adjoint 는 masked terminal 조건에서 시작한다 (pre_adj 통계로 대조)."""
    a = trainer.agent
    obs = batch["observations"][:4]
    xs, adjs, ts, info = a.adj_matching(obs, jax.random.PRNGKey(4))
    assert xs.shape == (4, 4, 192) and adjs.shape == (4, 4, 192)
    assert np.all(np.isfinite(np.asarray(adjs)))
    # 마지막 ODE step 을 재현해 x1 을 만들고 terminal adjoint 의 통계와 비교
    h = 0.25
    slow = a.network.select("target_actor_slow" if a.config["target_actor"] else "actor_slow")
    x1 = xs[-1] + h * slow(obs, xs[-1], ts[-1])
    adj_T = np.asarray(a.terminal_adjoint(obs, x1))
    np.testing.assert_allclose(float(info["adj_max"]), np.abs(adj_T).max(), rtol=1e-5)
    assert np.all(adj_T.reshape(4, 16, 12)[:, 5:] == 0.0)


# --------------------------------------------------------------------- Bellman / discount

def test_bellman_target_applies_supplied_discount_once(trainer, batch):
    a = trainer.agent
    na = np.asarray(a.sample_actions(batch["next_observations"], rng=jax.random.PRNGKey(6)))
    tgt = np.asarray(a.td_target(batch, None, next_actions=na))
    nq = np.asarray(_critic(a, "target_critic", batch["next_observations"], np.clip(na, -1, 1),
                            batch["next_exec_mask"]))
    nq = nq.mean(0) - a.config["rho"] * nq.std(0)
    np.testing.assert_allclose(tgt, batch["rewards"] + batch["discounts"] * nq, rtol=1e-6, atol=1e-6)
    boot = batch["discounts"] > 0
    wrong = batch["rewards"] + GAMMA ** 16 * batch["discounts"] * nq   # γ^H 재곱
    assert not np.allclose(tgt[boot], wrong[boot])
    term = batch["discounts"] == 0
    np.testing.assert_array_equal(tgt[term], batch["rewards"][term])
    # config['discount'] 는 target 에 쓰이지 않는다 (기록용)
    a2 = a.replace(config=a.config.copy({"discount": 0.5}))
    np.testing.assert_array_equal(tgt, np.asarray(a2.td_target(batch, None, next_actions=na)))
    # partial truncated 행: d = γ^{m_i}
    em = batch["exec_mask"].sum(-1)
    part = (em < 5) & boot
    assert part.any()
    np.testing.assert_allclose(batch["discounts"][part], GAMMA ** em[part], rtol=1e-6)


def _mut(**kw):
    def f(rows):
        rows = {k: np.array(v, copy=True) for k, v in rows.items()}
        for k, fn in kw.items():
            rows[k] = fn(rows)
        return rows
    return f


@pytest.mark.parametrize("mutate,match", [
    (_mut(discounts=lambda r: np.where(r["terminals"], 0.0, GAMMA ** 16)), "discounts"),
    (_mut(discounts=lambda r: r["discounts"] * GAMMA ** 16), "discounts"),
    (_mut(discounts=lambda r: np.where(r["terminals"], 0.0, GAMMA ** 5)), "discounts"),
    (_mut(next_executed_m=lambda r: np.where(r["has_next"], 3, 0)), "next_executed_m"),
    (_mut(next_executed_m=lambda r: np.full(len(r["has_next"]), 5)), "next_executed_m"),
    (_mut(executed_mask=lambda r: np.ones_like(r["executed_mask"])), "executed_mask"),
    (_mut(executed_m=lambda r: np.full_like(r["executed_m"], 6)), "executed_m"),
    (_mut(chunk_valid=lambda r: np.zeros_like(r["chunk_valid"])), "chunk_valid"),
    (_mut(has_next=lambda r: np.ones_like(r["has_next"])), "terminal"),
    (_mut(next_observations=lambda r: r["next_observations"] + 1.0), "next_observations"),
    (_mut(rewards=lambda r: r["rewards"] * np.nan), "rewards"),
])
def test_batch_contract_fail_closed(sets, mutate, match):
    with pytest.raises(ContractError, match=match):
        to_qam_batch(mutate(_all_rows(sets[0])), execute_steps=5, gamma=GAMMA)


# --------------------------------------------------------------------- BC target

def test_bc_loss_ignores_padded_expert_tail_and_matches_upstream_when_all_valid(trainer, batch):
    a = trainer.agent
    rng = jax.random.PRNGKey(7)
    p = a.network.params
    l0 = float(a.bc_flow_loss(batch, p, rng))
    pad = ~batch["chunk_valid"].astype(bool)
    assert pad.any()
    acts = batch["actions"].copy()
    acts[pad] = 7.0
    assert float(a.bc_flow_loss(dict(batch, actions=acts), p, rng)) == l0
    acts2 = batch["actions"].copy()
    acts2[~pad] += 0.5
    assert float(a.bc_flow_loss(dict(batch, actions=acts2), p, rng)) != l0
    # 16 전체 감독: 실행 prefix 밖이어도 valid 면 loss 에 들어간다
    beyond = batch["chunk_valid"].astype(bool) & ~batch["exec_mask"].astype(bool)
    assert beyond.any()
    acts3 = batch["actions"].copy()
    acts3[beyond] += 0.5
    assert float(a.bc_flow_loss(dict(batch, actions=acts3), p, rng)) != l0
    # 독립 참조식: padding 차원은 loss 의 합·분모 어디에도 들어가지 않는다
    x1 = batch["actions"].reshape(len(acts), -1) * np.repeat(batch["chunk_valid"], 12, -1)
    xr, tr_ = jax.random.split(rng)
    x0 = np.asarray(jax.random.normal(xr, x1.shape))
    t = np.asarray(jax.random.uniform(tr_, (x1.shape[0], 1)))
    pred = np.asarray(a.network.select("actor_slow")(batch["observations"], (1 - t) * x0 + t * x1, t, params=p))
    vm = np.repeat(batch["chunk_valid"], 12, -1)
    ref = np.mean((np.square(pred - (x1 - x0)) * vm).sum(-1) / vm.sum(-1))
    np.testing.assert_allclose(l0, ref, rtol=1e-5)
    assert not np.isclose(l0, np.mean(np.square(pred - (x1 - x0))), rtol=1e-3)
    # chunk_valid 전부 1 이면 upstream 식 (mean over dims) 과 같다
    full = dict(batch, chunk_valid=np.ones_like(batch["chunk_valid"]))
    x1 = batch["actions"].reshape(len(acts), -1)
    xr, tr_ = jax.random.split(rng)
    x0 = jax.random.normal(xr, x1.shape)
    t = jax.random.uniform(tr_, (x1.shape[0], 1))
    pred = a.network.select("actor_slow")(batch["observations"], (1 - t) * x0 + t * x1, t, params=p)
    up = float(jnp.mean(jnp.square(pred - (x1 - x0)).mean(axis=-1)))
    np.testing.assert_allclose(float(a.bc_flow_loss(full, p, rng)), up, rtol=1e-6)


# --------------------------------------------------------------------- training / checkpoint / callback

def test_synthetic_update_smoke_changes_params_finite(sets):
    tr = PrefixQAMTrainer.create(profile="smoke_reduced_network", gamma=GAMMA, seed=1,
                                 data_label=SYNTHETIC_LABEL)
    p0 = jax.tree_util.tree_map(np.asarray, tr.agent.network.params)
    hist = tr.fit(TrainSampler(sets[0], val=sets[1], seed=2), steps=10, batch_size=8,
                  val_set=sets[1], eval_every=10, log_every=5)
    assert tr.steps_done == 10
    for h in hist:
        assert all(np.isfinite(v) for k, v in h.items() if k != "step")
    assert any("val/critic_loss" in h for h in hist)
    p1 = tr.agent.network.params
    for k in ("modules_critic", "modules_actor_slow", "modules_actor_fast",
              "modules_target_critic", "modules_target_actor_slow"):
        d = max(float(np.abs(np.asarray(x) - np.asarray(y)).max())
                for x, y in zip(jax.tree_util.tree_leaves(p0[k]), jax.tree_util.tree_leaves(p1[k])))
        assert d > 0, k
    # target_actor_fast 는 upstream 과 같이 갱신되지 않는다 (upstream target_update 대상 아님)
    k = "modules_target_actor_fast"
    assert all(np.array_equal(np.asarray(x), np.asarray(y))
               for x, y in zip(jax.tree_util.tree_leaves(p0[k]), jax.tree_util.tree_leaves(p1[k])))


def test_val_split_not_accepted_by_optimizer_sampler(sets):
    with pytest.raises(ContractError):
        TrainSampler(sets[1], val=None, seed=0)


def test_checkpoint_save_reload_roundtrip(trainer, tmp_path):
    path = trainer.save(tmp_path / "ck", feature_signature=SIG)
    assert path.name == f"params_{trainer.steps_done}.pkl"
    re = PrefixQAMTrainer.load(tmp_path / "ck", trainer.steps_done, feature_signature=SIG)
    for x, y in zip(jax.tree_util.tree_leaves(trainer.agent.network.params),
                    jax.tree_util.tree_leaves(re.agent.network.params)):
        assert np.array_equal(np.asarray(x), np.asarray(y))
    assert re.data_label == SYNTHETIC_LABEL and re.steps_done == trainer.steps_done
    obs = np.ones((2, 1024), np.float32) * 0.1
    k = jax.random.PRNGKey(0)
    assert np.array_equal(np.asarray(trainer.agent.sample_actions(obs, k)), np.asarray(re.agent.sample_actions(obs, k)))
    flags = json.loads((tmp_path / "ck" / f"flags_{trainer.steps_done}.json").read_text())
    validate_qam_flags(flags, feature_signature=SIG)
    assert flags[PREFIX_CONTRACT_KEY]["execute_steps"] == 5

    # prefix 계약 없는 (full-chunk critic) checkpoint · bridge 표식 · 다른 feature signature 거부
    for mutate, match in (
        (lambda f: f.pop(PREFIX_CONTRACT_KEY), "prefix"),
        (lambda f: f.update(run_group="bridge_latents"), "Bridge"),
        (lambda f: f["agent"].update(horizon_length=4), "계약"),
    ):
        g = json.loads(json.dumps(flags))
        mutate(g)
        (tmp_path / "ck" / f"flags_{trainer.steps_done}.json").write_text(json.dumps(g))
        with pytest.raises(ContractError, match=match):
            PrefixQAMTrainer.load(tmp_path / "ck", trainer.steps_done, feature_signature=SIG)
    with pytest.raises(ContractError):
        PrefixQAMTrainer.load(tmp_path / "ck", trainer.steps_done, feature_signature="0" * 16)


def test_velocity_callback_drives_composed_flow_sampler(trainer):
    from test_composition import B, FakeHead, make_inputs

    torch.manual_seed(5)
    head = FakeHead().eval()
    bo, ai = make_inputs(1)
    qam_fn = trainer.velocity_fn()
    sampler = ComposedFlowSampler(qam_fn=qam_fn, qam_weight=1.0)
    torch.manual_seed(77)
    noise = torch.randn((B, 16, 32))
    torch.manual_seed(77)
    actions, info = sampler.sample(head, bo, ai, decision_env_step=0, gate=1.0)
    assert info["qam_steps"] == [0, 1, 2, 3] and info["lam"] == 1.0
    # lam=1: active12 는 QAM velocity 의 Euler 적분 그 자체 (t = i/4)
    x = noise[..., :12].numpy().reshape(B, -1)
    for i in range(4):
        x = x + 0.25 * np.asarray(trainer.agent.flow_velocity(jnp.asarray(info["feature"]), jnp.asarray(x), i,
                                                              model="fast"))
    np.testing.assert_allclose(actions[..., :12].numpy().reshape(B, -1), x, rtol=1e-5, atol=1e-5)
    # gate 0: QAM 미호출
    calls = []
    s0 = ComposedFlowSampler(qam_fn=lambda *a: calls.append(1) or qam_fn(*a), qam_weight=1.0)
    s0.sample(head, bo, ai, decision_env_step=0, gate=0.0)
    assert calls == []
    with pytest.raises(ContractError):
        qam_fn(info["feature"], x, 4)

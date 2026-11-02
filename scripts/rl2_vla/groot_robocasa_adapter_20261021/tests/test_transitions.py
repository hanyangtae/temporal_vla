"""Chunk transition builder: 시간순·실행 m·partial chunk·terminal/truncation·episode 경계·split 분리."""
import numpy as np
import pytest

from contract import ActionCodec, ContractError, raw_to_model_index, reorder_raw_to_model
from transitions import (
    EpisodeInput,
    TrainSampler,
    build_episode_transitions,
    build_split,
    expert_chunk_targets,
    sparse_success_rewards,
    val_batches,
)

G = 0.9
M = 5


def make_ep(eid, T, *, end="terminal", split="train", group=None, final=True, rewards=None,
            offset=0.0):
    steps = np.arange(0, T, M)
    n = len(steps)
    # latent 첫 성분 = 결정 env step (+episode offset) 로 두어 next 정렬을 값으로 검증
    lat = np.zeros((n, 1024), np.float32)
    lat[:, 0] = steps + offset
    lat[:, 1] = offset
    acts = np.random.default_rng(int(offset)).normal(size=(n, 16, 12)).astype(np.float32)
    fl = None
    if final:
        fl = np.zeros(1024, np.float32)
        fl[0], fl[1] = T + offset, offset
    r = rewards if rewards is not None else np.arange(1, T + 1, dtype=float)
    return EpisodeInput(eid, group or eid, split, T, end, steps, lat, acts, r, final_latent=fl)


def test_chronological_rewards_next_and_discount_with_partial_final_chunk():
    T = 13  # 결정 0,5,10 → m = 5,5,3 (마지막 partial)
    ep = make_ep("a", T, end="terminal")
    tr = build_episode_transitions(ep, execute_steps=M, gamma=G)
    assert tr["decision_step"].tolist() == [0, 5, 10]
    assert tr["executed_m"].tolist() == [5, 5, 3]
    r = ep.rewards
    for i, (t, m) in enumerate([(0, 5), (5, 5), (10, 3)]):
        assert tr["rewards"][i] == pytest.approx(sum(G ** k * r[t + k] for k in range(m)))
        assert tr["executed_mask"][i].sum() == m and tr["executed_mask"][i][:m].all()
    # next latent = latent(t+m) 같은 episode
    assert tr["next_observations"][0, 0] == 5 and tr["next_observations"][1, 0] == 10
    np.testing.assert_allclose(tr["discounts"], [G ** 5, G ** 5, 0.0])
    assert tr["terminals"].tolist() == [False, False, True]
    assert tr["has_next"].tolist() == [True, True, False]
    assert not tr["next_observations"][2].any()


def test_truncation_bootstraps_with_gamma_pow_m_and_final_latent():
    T = 13
    tr = build_episode_transitions(make_ep("a", T, end="truncated"), execute_steps=M, gamma=G)
    assert tr["terminals"].tolist() == [False, False, False]
    assert tr["truncated_end"].tolist() == [False, False, True]
    np.testing.assert_allclose(tr["discounts"], [G ** 5, G ** 5, G ** 3])  # 실제 m=3
    assert tr["next_observations"][2, 0] == T and tr["has_next"][2]


def test_truncation_without_next_latent_rejected():
    with pytest.raises(ContractError, match="bootstrap"):
        build_episode_transitions(make_ep("a", 13, end="truncated", final=False),
                                  execute_steps=M, gamma=G)
    # terminal 이면 latent(T) 불필요
    build_episode_transitions(make_ep("a", 13, end="terminal", final=False),
                              execute_steps=M, gamma=G)


def test_exact_multiple_length_and_m16():
    tr = build_episode_transitions(make_ep("a", 15, end="terminal"), execute_steps=M, gamma=G)
    assert tr["executed_m"].tolist() == [5, 5, 5] and tr["terminals"].tolist() == [0, 0, 1]
    ep = make_ep("b", 20)
    ep16 = EpisodeInput("b", "b", "train", 20, "terminal", np.array([0, 16]), ep.latents[:2],
                        ep.action_chunks[:2], ep.rewards)
    tr = build_episode_transitions(ep16, execute_steps=16, gamma=G)
    assert tr["executed_m"].tolist() == [16, 4]
    np.testing.assert_allclose(tr["discounts"], [G ** 16, 0.0])


@pytest.mark.parametrize("kw", [
    dict(end="success"), dict(end=None),
])
def test_end_reason_must_be_explicit(kw):
    with pytest.raises(ContractError, match="end_reason"):
        build_episode_transitions(make_ep("a", 13, **kw), execute_steps=M, gamma=G)


def test_bad_decision_grid_and_shapes_rejected():
    ep = make_ep("a", 13)
    bad = EpisodeInput("a", "a", "train", 13, "terminal", np.array([0, 5, 11]), ep.latents,
                       ep.action_chunks, ep.rewards)
    with pytest.raises(ContractError, match="decision_steps"):
        build_episode_transitions(bad, execute_steps=M, gamma=G)
    bad = EpisodeInput("a", "a", "train", 13, "terminal", ep.decision_steps,
                       np.zeros((3, 1536), np.float32), ep.action_chunks, ep.rewards)
    with pytest.raises(ContractError, match="latents"):
        build_episode_transitions(bad, execute_steps=M, gamma=G)
    for m in (0, 17):
        with pytest.raises(ContractError):
            build_episode_transitions(ep, execute_steps=m, gamma=G)


def test_episodes_never_crossed_and_shuffle_safe():
    a = make_ep("a", 13, end="truncated", offset=1000.0)
    b = make_ep("b", 22, end="terminal", offset=2000.0)
    ts = build_split([a, b], split="train", execute_steps=M, gamma=G)
    d = ts.data
    for i in range(len(ts)):
        if d["has_next"][i]:
            # next latent 은 같은 episode(offset 일치)의 t+m
            assert d["next_observations"][i, 1] == d["observations"][i, 1]
            assert d["next_observations"][i, 0] == d["observations"][i, 0] + d["executed_m"][i]
    # a 마지막 행 다음 배열 행은 b 첫 행이지만 next 로 쓰이지 않음
    last_a = np.flatnonzero(d["episode_id"] == "a")[-1]
    assert d["next_observations"][last_a, 1] == 1000.0
    # 섞은 뒤에도 행 단위 저장값만 쓰므로 정렬 불변
    batch = TrainSampler(ts, val=None, seed=0).sample(64)
    ok = batch["has_next"]
    np.testing.assert_array_equal(batch["next_observations"][ok, 1], batch["observations"][ok, 1])
    np.testing.assert_array_equal(batch["next_observations"][ok, 0],
                                  batch["observations"][ok, 0] + batch["executed_m"][ok])


def test_train_val_separation():
    tr = build_split([make_ep("a", 13)], split="train", execute_steps=M, gamma=G)
    va = build_split([make_ep("v", 13, split="val")], split="val", execute_steps=M, gamma=G)
    TrainSampler(tr, val=va, seed=0).sample(4)
    with pytest.raises(ContractError, match="train split"):
        TrainSampler(va, val=None, seed=0)
    with pytest.raises(ContractError, match="혼입"):
        build_split([make_ep("a", 13), make_ep("v", 13, split="val")], split="train",
                    execute_steps=M, gamma=G)
    va_dup = build_split([make_ep("a", 13, split="val")], split="val", execute_steps=M, gamma=G)
    with pytest.raises(ContractError, match="episode"):
        TrainSampler(tr, val=va_dup, seed=0)
    va_grp = build_split([make_ep("w", 13, split="val", group="a")], split="val",
                         execute_steps=M, gamma=G)
    with pytest.raises(ContractError, match="run_group"):
        TrainSampler(tr, val=va_grp, seed=0)
    assert sum(len(b["rewards"]) for b in val_batches(va, 2)) == len(va)
    with pytest.raises(ContractError):
        list(val_batches(tr, 2))


def test_sparse_success_reward_contract():
    assert sparse_success_rewards(4, True).tolist() == [0, 0, 0, 1]
    assert sparse_success_rewards(4, False).tolist() == [0, 0, 0, 0]


def test_expert_targets_are_dataset_actions_reordered_and_normalized():
    rng = np.random.default_rng(3)
    T = 20
    raw = rng.uniform(-1, 1, size=(T, 12))
    raw[:, 4] = -1.0
    raw[:, 11] = rng.choice([-1.0, 1.0], size=T)
    codec = ActionCodec(min=-np.ones(12), max=np.ones(12),
                        modes=tuple(["min_max"] * 6 + ["binary"] + ["min_max"] * 4 + ["binary"]))
    idx = raw_to_model_index()
    chunks, valid = expert_chunk_targets(raw, [0, 5, 10, 15], codec=codec, raw_index=idx)
    expect = codec.normalize(reorder_raw_to_model(raw, idx))
    np.testing.assert_allclose(chunks[0], expect[0:16], rtol=1e-6)
    np.testing.assert_allclose(chunks[2][:10], expect[10:20], rtol=1e-6)
    assert valid[0].all() and valid[2].sum() == 10 and not chunks[2][10:].any()
    assert valid[3].sum() == 5
    # gripper(model 6) ← raw 11, mode(model 11) ← raw 4 의 binary 정규화
    np.testing.assert_array_equal(chunks[0][:, 6], (raw[:16, 11] > 0.5).astype(np.float32))
    assert not chunks[0][:, 11].any()
    # target 은 정책 입력과 무관 — 같은 raw 면 같은 target
    chunks2, _ = expert_chunk_targets(raw.copy(), [0, 5, 10, 15], codec=codec, raw_index=idx)
    np.testing.assert_array_equal(chunks, chunks2)
    # expert chunk 를 transition 으로: 실행 위치는 valid 여야 하고 target 은 그대로 보존
    lat = np.zeros((4, 1024), np.float32)
    ep = EpisodeInput("x", "x", "train", T, "terminal", np.array([0, 5, 10, 15]), lat, chunks,
                      sparse_success_rewards(T, True), chunk_valid=valid)
    tr = build_episode_transitions(ep, execute_steps=M, gamma=G)
    np.testing.assert_array_equal(tr["actions"], chunks)
    np.testing.assert_array_equal(tr["chunk_valid"], valid)
    assert tr["rewards"][-1] == pytest.approx(G ** 4)

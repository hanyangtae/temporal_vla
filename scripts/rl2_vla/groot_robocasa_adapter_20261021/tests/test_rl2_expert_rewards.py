import numpy as np
import pytest
from contract import ContractError
from transitions import EpisodeInput
from rl2_expert_rewards import reward_fields, build_rl2_split

def ep(T=21):
    steps=np.arange(0,T,5); r=np.zeros(T);r[-16:]=1
    return EpisodeInput("episode_000000","episode_000000","train",T,"terminal",steps,
        np.repeat(steps[:,None],1024,axis=1).astype(np.float32),
        np.ones((len(steps),16,12),np.float32),r,
        np.arange(16)[None,:] < (T-steps[:,None]))

def test_raw_and_derived_are_distinct_and_immutable():
    e=ep(); d=np.zeros(e.length,bool);d[-1]=True;r=e.rewards.copy()
    f=reward_fields(e.rewards,d,source_role="successful_expert")
    assert np.array_equal(e.rewards,r)
    assert np.array_equal(f["raw_done"],d)
    assert np.array_equal(f["raw_reward"],r)
    assert np.all(f["rl2_reward"][:-3]==-1) and np.all(f["rl2_reward"][-3:]==0)
    assert np.all(f["rl2_td_mask"][-3:]==0)
    assert f["rl2_td_mask_last_step"].sum()==e.length-1
    f["raw_reward"][:]=99
    assert np.array_equal(e.rewards,r)

@pytest.mark.parametrize("role",["changed_failure","historical_base","unknown"])
def test_failure_unknown_refused(role):
    e=ep();d=np.zeros(e.length,bool);d[-1]=True
    with pytest.raises(ContractError):reward_fields(e.rewards,d,source_role=role)

@pytest.mark.parametrize("T",[17,18,19,20,21,22,23,24,25,26])
def test_td_mask_prefix_chronology_and_partial(T):
    e=ep(T);out,fields,audit=build_rl2_split([e],split="train")
    b=out.data
    for i,t in enumerate(e.decision_steps):
        raw_m=min(5,T-int(t));first_zero=max(0,T-3-int(t))
        terminal=first_zero<raw_m; m=min(raw_m,first_zero+1)
        assert b["executed_m"][i]==m
        assert np.array_equal(b["executed_mask"][i],np.arange(16)<m)
        assert b["terminals"][i]==terminal
        expected=-sum(.99**k for k in range(min(m,first_zero)))
        assert b["rewards"][i]==pytest.approx(expected)
        assert b["discounts"][i]==pytest.approx(0 if terminal else .99**m)
        assert b["has_next"][i]==(not terminal)
        assert np.all(b["next_observations"][i]==(0 if terminal else t+5))
        assert np.array_equal(b["actions"][i],e.action_chunks[i])
        assert np.array_equal(b["chunk_valid"][i],e.chunk_valid[i])
        assert audit[i]["observed_prefix"]==raw_m
    assert np.array_equal(fields['episode_000000__raw_reward'],e.rewards)

def test_split_leakage_refused():
    with pytest.raises(ContractError):build_rl2_split([ep()],split="val")

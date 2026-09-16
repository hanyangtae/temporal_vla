import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

STAGE2 = Path(__file__).resolve().parents[1]/'scripts/rl2_vla/stage2_cluster_reward'
sys.path.insert(0, str(STAGE2))
from data import ACTION_CONTRACT, FEATURE_CONTRACT, TASKS, assign_splits, base_rewards, load_episode, transitions
from clusters import fit, ClusterPotential


def episode(success=False, length=10, idx=0):
    chunks = []
    for t in range(0,length,4):
        m = min(4,length-t)
        chunks.append(dict(context=np.full(1024, t+idx,dtype=float).tolist(),start_step=t,
            executed_actions=np.zeros((m,7)).tolist(),postprocessed_actions=np.zeros((4,7)).tolist(),
            normalized_actions=np.ones((4,7)).tolist(),step_reward=[0.]*m,
            step_success=[False]*(m-1)+[bool(success and t+m==length)],step_truncated=[False]*m,
            safe_trigger=False))
    return dict(schema_version=1,feature_contract=FEATURE_CONTRACT,action_contract=ACTION_CONTRACT,
        episode_id=f'e{idx}',reset_id=f'r{idx}',task_id=TASKS[0],success=success,
        end_reason='success' if success else 'horizon',chunks=chunks)


def test_original_bridge_reward_and_failure_extension():
    r,m = base_rewards(8,True)
    np.testing.assert_array_equal(r,[-1,-1,-1,-1,-1,0,0,0])
    np.testing.assert_array_equal(m,[1,1,1,1,1,0,0,0])
    r,m = base_rewards(8,False)
    assert np.all(r==-1) and np.array_equal(m,[1]*7+[0])
    assert np.array_equal(base_rewards(2,True)[0],[0,0])


def test_partial_terminal_and_zero_shaping_identity():
    eps=[episode()]
    a=transitions(eps)
    b=transitions(eps,lambda *_: 123.,scale=0)
    for k in a:
        np.testing.assert_array_equal(a[k],b[k])
    assert a['actual_steps'].tolist()==[4,4,2]
    assert a['masks'][-1,0]==0
    assert a['discounts'][-1]==np.float32(.99**2)
    assert not a['actions'][-1,2:].any()
    assert not a['action_mask'][-1,2:].any()
    assert a['rewards'][-1,0]==np.float32(-1-.99)


def test_potential_telescopes_with_terminal_zero():
    e=episode()
    base=transitions([e])
    shaped=transitions([e],lambda _,h: float(h[0]+2),scale=.1)
    delta=(shaped['rewards']-base['rewards'])[:,0]
    actual=np.dot(.99**np.array([0,4,8]),delta)
    assert actual==pytest.approx(-.1*2,abs=1e-6)


def test_reject_missing_step_log_and_mismatched_action(tmp_path):
    e=episode()
    path=tmp_path/'episode.json'
    path.write_text(json.dumps(e))
    load_episode(path)
    e['chunks'][0]['executed_actions'][0][0]=1
    path.write_text(json.dumps(e))
    with pytest.raises(ValueError,match='mismatch'):
        load_episode(path)
    e=episode(); e['chunks'][0]['step_success']=[]
    path.write_text(json.dumps(e))
    with pytest.raises(ValueError,match='missing per-step'):
        load_episode(path)


def test_reset_siblings_share_split():
    eps=[episode(idx=i) for i in range(10)]
    eps += [dict(e,episode_id=e['episode_id']+'seed2') for e in eps]
    a=assign_splits(eps)
    assert list(a.values()).count('train')==6
    assert len(a)==10
    assert a==assign_splits(list(reversed(eps)))


def test_cluster_scores_roundtrip_and_single_class(tmp_path):
    rng=np.random.default_rng(42)
    eps=[episode(success=i%2==0,length=16,idx=i) for i in range(8)]
    for e in eps:
        for c in e['chunks']:
            c['context']=(rng.normal(size=1024)+int(e['success'])*3).tolist()
    b=fit(eps,'manifest',k=2,dim=2)
    assert np.max(np.abs(b.tasks[TASKS[0]]['scores']))<=1
    p=tmp_path/'cluster.json'; b.save(p)
    assert b(TASKS[0],eps[0]['chunks'][0]['context'])==ClusterPotential.load(p)(TASKS[0],eps[0]['chunks'][0]['context'])
    disabled=fit([eps[0]],'manifest',k=48,dim=16)
    assert disabled(TASKS[0],np.zeros(1024))==0


def test_qam_prefix_update_smoke():
    pytest.importorskip('jax')
    from train_qam import agent_class
    root=STAGE2.parents[2]/'RL2-VLA/third_party/qam'
    if not root.exists():
        pytest.skip('initialize QAM submodule for numerical smoke')
    cls=agent_class(root)
    from agents.qam import get_config
    import jax
    config=get_config().to_dict()
    config.update(horizon_length=4,action_chunking=True,actor_hidden_dims=(16,16),
                  value_hidden_dims=(16,16),num_qs=2,flow_steps=2,inv_temp=.1)
    agent=cls.create(42,np.zeros(1024,np.float32),np.zeros(7,np.float32),config)
    batch=transitions([episode(False,length=6)])
    before=copy.deepcopy(agent.network.params['modules_actor_slow'])
    warm,info=agent.warmup(batch)
    for a,b in zip(jax.tree_util.tree_leaves(before),jax.tree_util.tree_leaves(warm.network.params['modules_actor_slow'])):
        np.testing.assert_array_equal(a,b)
    updated,info=warm.update(batch)
    assert all(np.isfinite(float(v)) for v in info.values())
    # Padding is never a supervised action target.
    tampered={k:v.copy() for k,v in batch.items()}
    tampered['actions'][-1,2:]=999
    loss1,_=agent.critic_loss(batch,agent.network.params,agent.rng)
    loss2,_=agent.critic_loss(tampered,agent.network.params,agent.rng)
    np.testing.assert_allclose(loss1,loss2)


def test_analysis_counts_detected_denominators_and_rejects_missing():
    from analyze import summarize
    arms={name:{} for name in ('vanilla','qam_original','qam_base','qam_shaped')}
    for task in TASKS:
        for i in range(2):
            e=episode(success=bool(i),idx=i)
            e.update(task_id=task,episode_id=f'{task}-{i}',reset_id=f'{task}-r{i}')
            for arm in arms:
                arms[arm][e['episode_id']]=copy.deepcopy(e)
            shaped=arms['qam_shaped'][e['episode_id']]
            shaped['success']=not e['success']
            shaped['chunks'][0]['safe_trigger']=True
    result=summarize(arms,bootstraps=10)['all']['qam_shaped']
    assert result['rescued']==result['destroyed']==4
    assert result['baseline_failures']==result['detected_baseline_failures']==4
    assert result['baseline_successes']==result['detected_baseline_successes']==4
    arms['qam_base'].pop(next(iter(arms['qam_base'])))
    with pytest.raises(ValueError,match='unpaired'):
        summarize(arms,bootstraps=10)

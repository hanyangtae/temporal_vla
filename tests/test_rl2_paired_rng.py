"""CPU-only regression: intervention/episode order must not shift policy noise."""
import ast
import contextlib
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import pytest
torch=pytest.importorskip('torch')
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'RL2-VLA/RL2_CoVer_VLA/simpler/run_simpler_eval_with_openpi.py'
ns=dict(contextlib=contextlib,hashlib=hashlib,json=json,random=random,np=np,torch=torch)
tree=ast.parse(source.read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_stage2_seed','_stage2_rng')],type_ignores=[]),str(source),'exec'),ns)
seed,scope=ns['_stage2_seed'],ns['_stage2_rng']

def sample(episode,step,stream='policy',size=8):
    with scope(seed('task',episode,42,stream,step)):
        return torch.randn(size),np.random.normal(size=size),random.random()

def test_noise_independent_of_previous_episodes_and_intervention():
    expected=sample(10001,0)
    for step in range(0,88,4):
        sample(10000,step)
        sample(10000,step,'intervention',99)
    actual=sample(10001,0)
    for x,y in zip(expected,actual): np.testing.assert_array_equal(x,y)
    expected=sample(10001,4)
    sample(10001,0,'intervention',199)
    for x,y in zip(expected,sample(10001,4)): np.testing.assert_array_equal(x,y)
    assert not torch.equal(sample(10001,0)[0],sample(10001,0,'intervention')[0])
    assert seed('task',10001,42,'policy',0)!=seed('task',10001,7,'policy',0)

def test_rng_restored_even_on_exception():
    torch.manual_seed(17); np.random.seed(17); random.seed(17)
    expected=(torch.randn(3),np.random.rand(3),random.random())
    torch.manual_seed(17); np.random.seed(17); random.seed(17)
    with pytest.raises(RuntimeError):
        with scope(234):
            torch.randn(77); np.random.rand(33); random.random()
            raise RuntimeError('simulated compose failure')
    actual=(torch.randn(3),np.random.rand(3),random.random())
    for x,y in zip(expected,actual):np.testing.assert_array_equal(x,y)


def test_analysis_rejects_legacy_and_nonintervention_drift():
    import sys,copy
    sys.path.insert(0,str(ROOT/'scripts/rl2_vla/stage2_cluster_reward'))
    from analyze import validate_paired_rng
    e=dict(rng_contract='paired_rng_v2',task_id='task',reset_id='task/env1',env_seed=1,policy_seed=42,episode_rng_seed=12,
           policy_checkpoint='pi0',action_contract='v1',feature_contract='v1',success=True,
           chunks=[dict(start_step=0,context=[1.],safe_trigger=False,proposed_actions=[[1.]],executed_actions=[[1.]])])
    arms={'vanilla':{'id':e},'qam_base':{'id':copy.deepcopy(e)}}
    validate_paired_rng(arms)
    arms['qam_base']['id']['chunks'][0]['proposed_actions']=[[2.]]
    with pytest.raises(ValueError,match='proposed_actions'):validate_paired_rng(arms)
    arms['qam_base']['id']['chunks'][0]['safe_trigger']=True
    validate_paired_rng(arms)  # Real intervention may change actions, not its input context.
    arms['qam_base']['id']['chunks'][0]['context']=[2.]
    with pytest.raises(ValueError,match='context'):validate_paired_rng(arms)
    arms['qam_base']['id'].pop('rng_contract')
    with pytest.raises(ValueError,match='RNG contract'):validate_paired_rng(arms)

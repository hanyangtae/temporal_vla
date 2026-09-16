"""Execute the actual opt-in loop with a fake simulator and real CPU tensors."""
import ast
from collections import deque
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/rl2_vla/stage2_cluster_reward'))
from data import load_episode


@pytest.mark.parametrize('stop_at,gated', [(2,False),(None,False),(6,True)])
def test_actual_loop_records_only_executed_actions(tmp_path,stop_at,gated):
    source=ROOT/'RL2-VLA/RL2_CoVer_VLA/simpler/run_simpler_eval_with_openpi.py'
    tree=ast.parse(source.read_text())
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and (n.name.startswith('_stage2') or n.name in ('_atomic_write_stage2_episode','_eval_stage2_single_candidate'))]
    class Env:
        def __init__(self): self.actions=[]
        def reset(self,seed): self.seed=seed; return {},{}
        def get_language_instruction(self): return 'test'
        def step(self,a):
            self.actions.append(a)
            return {},0., bool(stop_at and len(self.actions)==stop_at),False,{}
    env=Env()
    class Policy:
        config=SimpleNamespace(device='cpu',image_features={'observation.images.top':None})
        def reset(self): pass
        def normalize_targets(self,b): return b
        def select_action(self,observation,**kwargs):
            assert observation['observation.images.top'].ndim==4
            return deque([torch.tensor([[float(i),0,0,0,0,0,1]]) for i in range(4)]),np.zeros((1,10,5,1024),np.float32)
    adapter=SimpleNamespace(preprocess=lambda _: {'observation.images.top':torch.zeros(1,3,2,2),'observation.state':torch.zeros(1,7)})
    calls=[]
    def compose(**kw):
        calls.append(kw['hidden_states_np'].copy())
        return None,deque([torch.tensor([[10.+i,0,0,0,0,0,1]]) for i in range(4)]),.5
    def failure(**kw):
        assert kw['hidden_states_last_token'].device==next(kw['failure_model'].parameters()).device
        kw['all_features'].append(kw['hidden_states_last_token'])
        return (.9,kw['timestep']==0)
    namespace=dict(np=np,torch=torch,deque=deque,Path=Path,os=os,json=json,time=time,
        GenerateConfig=SimpleNamespace,tqdm=SimpleNamespace(tqdm=lambda it:it),
        create_bridge_adapter_wrapper=lambda _:adapter,get_simpler_env=lambda *_:env,
        get_image_from_maniskill2_obs_dict=lambda *_:np.zeros((2,2,3)),
        SAFE_TASK_MAP_DICT={},QAMInference=lambda **_:object(),
        load_failure_detection_model=lambda *_:(torch.nn.Linear(1024,1),np.full(150,.5)),
        check_failure_prediction_lstm=failure,compute_composed_actions=compose,
        process_inputs=lambda n,q,**_: [torch.stack(list(q),dim=1)[0].numpy()])
    exec(compile(ast.Module(body=funcs,type_ignores=[]),str(source),'exec'),namespace)
    cfg=SimpleNamespace(stage2_save_activations=True,stage2_min_free_gb=0,use_verifier=False,use_verifier_always=False,lang_transform_type='no_transform',num_steps_wait=0,
        action_samples_prefail=1,action_samples=1,lang_rephrase_num_prefail=1,lang_rephrase_num=1,composed_samples_prefail=0,
        composed_samples=int(gated),n_action_steps=4,stage2_rollout_dir=str(tmp_path),action_ensemble_temp=-.8,
        qam_ckpt='fake',task_suite_name='simpler_spoon_on_towel',model_family='openvla',num_trials_per_task=1,
        env_seed_start=10000,seed=42,use_failure_prediction=gated,use_rephrased_latents_for_qam=False,pretrained_checkpoint='synthetic')
    suite=SimpleNamespace(n_tasks=1,get_task=lambda _: 'test')
    namespace['_eval_stage2_single_candidate'](cfg,Policy(),suite)
    files=list(tmp_path.glob('episode_*.json')); assert len(files)==1
    e=load_episode(files[0]); n=stop_at or 150
    assert sum(len(c['executed_actions']) for c in e['chunks'])==n
    assert len(env.actions)==n
    assert env.actions[0][0]==(10 if gated else 0)
    assert env.actions[1][0]==(11 if gated else 1)
    if gated:
        assert len(calls)==1 and calls[0].shape==(1,1024)
        assert env.actions[4][0]==0  # gate clears; no latch
    else:
        assert not calls
    assert e['elapsed_seconds']>=0

    import hashlib
    sidecar=tmp_path/e['activation_file']['path']
    assert hashlib.sha256(sidecar.read_bytes()).hexdigest()==e['activation_file']['sha256']
    with np.load(sidecar) as archive:
        assert archive['action_embeds'].shape==(len(e['chunks']),1,10,5,1024)

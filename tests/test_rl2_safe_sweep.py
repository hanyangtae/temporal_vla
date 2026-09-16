"""Scientific contracts and original SAFE numerical equivalence."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('safe_sweep',ROOT/'scripts/rl2_vla/stage2_cluster_reward/safe_sweep.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def test_full_grid_and_seed_contract():
    jobs=m.grid()
    assert len(jobs)==576
    assert len({(j['horizon'],j['denoise']) for j in jobs})==16
    assert {j['seed'] for j in jobs}=={0,1,2}
    assert {j['lr'] for j in jobs}=={.0001,.0003,.001}
    assert {j['reg'] for j in jobs}=={.001,.01,.1,1}
    assert m.model_config(jobs[0])['batch_size']==512


def test_band_matches_upstream_tfunc():
    sys.path.insert(0,str(ROOT/'RL2-VLA/third_party/SAFE'))
    from failure_prob.utils.conformal.functional_predictor import FunctionalPredictor, ModulationType, RegressionType
    rng=np.random.default_rng(51); train=rng.random((7,38)); residual=rng.random((19,38))
    original=FunctionalPredictor(ModulationType.Tfunc,RegressionType.Mean)
    for a in m.ALPHAS:
        expected=original.get_one_sided_prediction_band(train,residual,a,False).reshape(-1)
        np.testing.assert_allclose(m.original_band(train,residual,a),expected,rtol=1e-12,atol=1e-12)


def test_grouped_calibration_no_sibling_overlap():
    rows=[{'reset_id':f'r{i}','success':True} for i in range(10) for seed in range(3)]
    a,b=m.calibration_indices(rows)
    assert len(a)==9 and len(b)==21
    assert not {rows[i]['reset_id'] for i in a}&{rows[i]['reset_id'] for i in b}
    with pytest.raises(ValueError,match='four'): m.calibration_indices(rows[:9])


def test_edge_extension_and_real_stop_detection():
    r={'length':2,'chunk_lengths':[4,2],'success':True}
    score=m.expanded_scores(np.array([.2,.5]),r,10)
    np.testing.assert_array_equal(score,[.2]*4+[.5]*6)
    # A threshold breach only in terminal padding must not be called an alarm.
    band=np.ones(10); band[6:]=0
    fail=dict(r,success=False)
    assert m.balanced_accuracy(np.stack([score,score]),[r,fail],band,observed_only=True)==.5


def test_top3_includes_ties_original_rounding():
    scores={.1:.80,.2:.9,.3:.8,.4:.80000001,.5:.6}
    assert m.top_alphas(scores)==[.2,.1,.3,.4]


def test_cache_rejects_group_leakage_and_hash_corruption(tmp_path):
    rows=[dict(episode_id=f'e{i}',task='task',reset_id=f'r{i}',success=i%2==0,split=split,length=2,chunk_lengths=[4,2]) for i,split in enumerate(m.SPLITS)]
    features=np.zeros((4,2,3),np.float32); np.savez_compressed(tmp_path/'f.npz',features=features)
    payload={'episodes':rows,'features':{'k':'f.npz'},'features_sha256':{'k':m.sha(tmp_path/'f.npz')}}
    cachepath=tmp_path/'cache.json'; cachepath.write_text(json.dumps(payload)); cache=m.load_cache(cachepath)
    np.testing.assert_array_equal(m.load_features(cachepath,cache,'k'),features)
    cache['features_sha256']['k']='0'*64
    with pytest.raises(ValueError,match='hash'): m.load_features(cachepath,cache,'k')
    payload['episodes'][1]['reset_id']='r0'; cachepath.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='crosses'): m.load_cache(cachepath)


def test_auc_maximum_is_only_over_observed_chunks():
    rows=[{'success':True,'length':1},{'success':False,'length':2}]
    assert m.sequence_auc(np.array([[.1,.99],[.2,.8]]),rows)==1


def test_early_auc_task_min_ignores_holdout():
    rows=[dict(task='t',split='train',length=3),dict(task='t',split='selection',length=2),dict(task='t',split='holdout',length=1)]
    assert m.task_min_steps(rows)=={'t':2}
    selection=[dict(task='t',success=True,length=3),dict(task='t',success=False,length=3)]
    scores=np.array([[.1,.2,.9],[.5,.4,.6]])
    assert m.sequence_auc(scores,selection,{'t':2})==1
    assert m.sequence_auc(scores,selection)==0


def test_finalize_rejects_stale_contracts_and_incomplete_metrics():
    import copy
    cache={'_sha256':'a'*64,'episodes':[dict(task='t',split='train',length=3)]}
    spec=m.grid()[0]
    valid={'contract':m.job_contract(spec,cache,[1000,2000],512),'metrics':[dict(epoch=e,selection_auc=.5) for e in (1000,2000)]}
    assert len(m.checked_metrics(valid,spec,cache))==2
    for key,value in [('spec',dict(spec,lr=.5)),('epochs',[1,2]),('batch_size',32),('cache_sha256','b'*64),('selection_metric','endpoint')]:
        bad=copy.deepcopy(valid); bad['contract'][key]=value
        with pytest.raises(ValueError,match='stale'): m.checked_metrics(bad,spec,cache)
    bad=copy.deepcopy(valid); bad['metrics'].pop()
    with pytest.raises(ValueError,match='endpoint'): m.checked_metrics(bad,spec,cache)


def test_real_lstm_worker_resume_after_last_checkpoint(tmp_path,monkeypatch):
    import argparse
    import torch
    torch.set_num_threads(2)
    rows=[dict(episode_id=f'e{i}',task='t',reset_id=f'r{i}',success=bool(i%2),split=m.SPLITS[i//2],length=2,chunk_lengths=[4,2]) for i in range(8)]
    np.savez_compressed(tmp_path/'f.npz',features=np.random.default_rng(1).normal(size=(8,2,4)).astype(np.float32))
    (tmp_path/'cache.json').write_text(json.dumps({'episodes':rows,'features':{'h=0.0__d=0.0':'f.npz'},'features_sha256':{'h=0.0__d=0.0':m.sha(tmp_path/'f.npz')}}))
    args=argparse.Namespace(cache=tmp_path/'cache.json',safe_root=ROOT/'RL2-VLA/third_party/SAFE',output=tmp_path/'run',device='cpu',shards=1,shard=0,job=0,epochs=[1,2],batch_size=512,save_every=1)
    real=m.atomic_json
    def interrupt_done(path,value):
        if Path(path).name=='done.json': raise RuntimeError('simulate interrupted completion')
        real(path,value)
    monkeypatch.setattr(m,'atomic_json',interrupt_done)
    with pytest.raises(RuntimeError,match='interrupted'): m.worker(args)
    folder=args.output/'job_0000'; assert (folder/'resume.pt').exists()
    before=m.sha(folder/'model_2.ckpt')
    monkeypatch.setattr(m,'atomic_json',real); m.worker(args)
    assert not (folder/'resume.pt').exists()
    assert m.sha(folder/'model_2.ckpt')==before
    assert len(json.loads((folder/'done.json').read_text())['metrics'])==2

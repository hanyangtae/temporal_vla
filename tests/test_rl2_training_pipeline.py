from pathlib import Path
from types import SimpleNamespace
import sys,json
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rl2_vla/stage2_cluster_reward'))
import run_training_pipeline as pipeline

def args(tmp_path,run=False):
 return SimpleNamespace(cache=str(tmp_path/'cache.json'),rl2_root=str(tmp_path/'rl2'),output=str(tmp_path/'out'),policy=str(tmp_path/'policy'),qam_initial=str(tmp_path/'initial.pkl'),python=sys.executable,safe_gpus=['2','3','4','6','7'],qam_gpus=['0','1'],eval_gpus=['0','1','2','3','4','6','7'],run=run)

def test_full_plan_serializes_and_correct_dependency_paths(tmp_path):
 a=args(tmp_path);assert pipeline.run(a)==0
 plan=json.loads((Path(a.output)/'plan.json').read_text())['stages']
 safe=next(x for x in plan if x['name']=='safe_sweep')
 assert len(safe['jobs'])==10
 assert sorted(j['shard'] for j in safe['jobs'])==list(range(10))
 qam=next(x for x in plan if x['name']=='qam')
 for j in qam['jobs']:
  c=j['command'];assert c[c.index('--qam-root')+1]==str(tmp_path/'rl2/third_party/qam')
 follow=next(x for x in plan if x['name']=='eval_followup')['jobs'][0]['command']
 assert follow[follow.index('--seeds')+1:follow.index('--seeds')+3]==['0','7']

def test_failed_concurrent_training_blocks_finalization_and_eval(tmp_path,monkeypatch):
 a=args(tmp_path,True);monkeypatch.setattr(pipeline,'_contract',lambda _: {'verified':'test'})
 monkeypatch.setattr(pipeline,'_verify_qam',lambda *_:False)
 called=[]
 def fail(stage,jobs,*_):
  called.append(stage)
  assert len(jobs)==12  # 10SAFE and2QAM submitted together
  assert sum(j.get('mode') in ('base','cluster_potential') for j in jobs)==2
  raise RuntimeError('injected training failure')
 monkeypatch.setattr(pipeline,'_run_jobs',fail)
 assert pipeline.run(a)==1 and called==['training']
 assert json.loads((Path(a.output)/'status.json').read_text())['stages']['training']['state']=='failed'

def test_verified_qam_is_not_relaunched(tmp_path,monkeypatch):
 a=args(tmp_path,True);monkeypatch.setattr(pipeline,'_contract',lambda _: {'verified':'test'})
 monkeypatch.setattr(pipeline,'_verify_qam',lambda *_:True)
 called=[]
 def fail(stage,jobs,*_):
  called.append(stage);assert len(jobs)==10
  raise RuntimeError('stop after proving skipped QAM')
 monkeypatch.setattr(pipeline,'_run_jobs',fail)
 assert pipeline.run(a)==1 and called==['training']

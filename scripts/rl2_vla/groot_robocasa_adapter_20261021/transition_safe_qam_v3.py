from pathlib import Path
import os,time,json,subprocess,hashlib
R=Path('/home/dongkyu/pkt_ws/temporal_vla');A=R/'scripts/rl2_vla/groot_robocasa_adapter_20261021';O=R/'outputs/eval/robocasa/groot_n15/safe_qam_opendrawer_heldout_20261021'
def alive(pid):
 p=Path('/proc')/str(pid)/'stat'
 return p.exists() and p.read_text().split()[2]!='Z'
limit=time.time()+2400
while any(alive(p) for p in [3862954,3862965,3863663]):
 if time.time()>limit:raise RuntimeError('old runner/verifier did not reach boundary; no duplicate launch')
 time.sleep(10)
assert (O/'FAILED.json').exists()
failure=json.loads((O/'FAILED.json').read_text());assert 'exit status 75' in failure['error'],failure
for src,dst in [('FAILED.json','BASELINE_REUSE_BOUNDARY_EXIT75.json'),('VERIFIER_EXIT_V2.json','VERIFIER_EXIT_BEFORE_V3.json'),('PAUSE_BEFORE_NEXT_EPISODE.json','PAUSE_CLEARED_BASELINE_REUSE.json')]:
 assert (O/src).exists() and not (O/dst).exists();(O/src).rename(O/dst)
(O/'resume_receipt_v3.json').write_text(json.dumps({'time':time.time(),'old_pids_exited':[3862954,3862965,3863663],'primary_original_baseline':30,'safe_expected':30,'interrupted_episodes':0,'runner_sha256':hashlib.sha256((A/'run_safe_qam_eval_v3.py').read_bytes()).hexdigest()},indent=2))
env=os.environ.copy();env.update(HARNESS_MACHINE='kanu',HARNESS_SESSION='groot-qam-20261021',GPUS='6',SERVES_PER_GPU='1')
with (O/'runner_v3.log').open('x') as f:
 p=subprocess.Popen(['bash',str(A/'run_safe_qam_eval_v3.sh')],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 with (O/'verifier_controller_v3.log').open('x') as g:
  v=subprocess.Popen([str(R/'.venvs/rl2_iid/bin/python'),'-u',str(A/'watch_safe_qam_eval_v3.py')],cwd=R,stdout=g,stderr=subprocess.STDOUT,start_new_session=True)
 (O/'v3_pids.json').write_text(json.dumps({'harness_pid':p.pid,'verifier_pid':v.pid,'time':time.time()},indent=2));print('v3 launched',p.pid,v.pid,flush=True)

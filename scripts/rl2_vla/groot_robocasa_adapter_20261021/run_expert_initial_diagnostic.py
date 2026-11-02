import json,os,subprocess,time
from paired_eval import ROOT,MODEL_IMAGE,ENV_IMAGE,PROFILE,docker_args,inside,ensure_absent,stop_owned,get_json
from baseline_data import write_json_exclusive
assert os.environ.get('HARNESS_RECEIPT')
gpu=int(os.environ['GPUS']);out=ROOT/'outputs/analysis/groot_robocasa_lineage_20261021/expert_initial_state_diagnostic_v2'
out.mkdir(exist_ok=False)
write_json_exclusive(out/'launch.json',{'gpu':gpu,'pid':os.getpid(),'episode':1,'purpose':'Held-out expert replay and frozen GR00T same saved state; no training or checkpoint reselection','time':time.time()})
def worker(mode):
 name=f'groot-qam-expert-diag-{mode}-{os.getpid()}';ensure_absent(name)
 with open(out/f'{mode}.log','x') as f:
  try:subprocess.run(docker_args(out,gpu,name,ENV_IMAGE,network='host' if mode=='groot' else 'none')+['scripts/rl2_vla/groot_robocasa_adapter_20261021/expert_initial_state_diagnostic.py','--mode',mode,'--out',inside(out),'--port','8896'],stdout=f,stderr=subprocess.STDOUT,timeout=1200,check=True)
  finally:stop_owned(name)
try:
 worker('expert')
 result=json.loads((out/'expert_result.json').read_text());assert result['initial_state_max_abs_error']<=1e-10
 name=f'groot-qam-expert-diag-serve-{os.getpid()}';ensure_absent(name)
 with open(out/'serve.log','x') as f:
  proc=subprocess.Popen(docker_args(out,gpu,name,MODEL_IMAGE)+['scripts/rl2_vla/groot_robocasa_adapter_20261021/serve_qam.py','--qam-arm','base','--profile',PROFILE,'--host','127.0.0.1','--port','8896'],stdout=f,stderr=subprocess.STDOUT)
  try:
   deadline=time.monotonic()+180
   while True:
    if proc.poll() is not None:raise RuntimeError('Base server exited before ready')
    try:
     q=get_json(8896,'qam_contract')
     if q['ready'] and q['arm']=='base':break
    except OSError:pass
    if time.monotonic()>deadline:raise RuntimeError('Base server readiness timeout')
    time.sleep(2)
   worker('groot')
  finally:stop_owned(name);proc.wait(timeout=20)
 expert=json.loads((out/'expert_result.json').read_text());groot=json.loads((out/'groot_result.json').read_text())
 if expert['initial_state_sha256']!=groot['initial_state_sha256']:raise RuntimeError('Expert and GR00T initial states differ')
 write_json_exclusive(out/'COMPLETE.json',{'expert':expert,'groot':groot,'same_saved_initial_state':True,'no_checkpoint_reselection':True})
 print('EXPERT_INITIAL_DIAGNOSTIC_COMPLETE',flush=True)
except Exception as e:
 write_json_exclusive(out/'BLOCKED.json',{'error':str(e),'type':type(e).__name__,'time':time.time()});raise

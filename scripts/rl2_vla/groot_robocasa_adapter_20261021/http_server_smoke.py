"""Readiness check of the real HTTP wrapper; no rollout or performance claim."""
import os,subprocess,time
from pathlib import Path
from paired_eval import ROOT,MODEL_IMAGE,PROFILE,docker_args,get_json,stop_owned,ensure_absent
from baseline_data import write_json_exclusive
assert os.environ.get('HARNESS_RECEIPT')
out=ROOT/'outputs/analysis/groot_robocasa_lineage_20261021/http_qam_server_smoke_v1'
out.mkdir(exist_ok=False)
name=f'groot-qam-http-smoke-{os.getpid()}'
ensure_absent(name)
args=docker_args(out,4,name,MODEL_IMAGE)+['scripts/rl2_vla/groot_robocasa_adapter_20261021/serve_qam.py',
 '--qam-arm','gate1','--qam-actor','/temporal_vla/outputs/rl2_vla/groot_rl2_actor_smoke_export_20261021',
 '--profile',PROFILE,'--host','127.0.0.1','--port','8896']
with open(out/'serve.log','x') as log:
 proc=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT)
 try:
  deadline=time.monotonic()+180
  while True:
   if proc.poll() is not None:raise RuntimeError(f'server exited {proc.returncode}')
   try:
    q=get_json(8896,'qam_contract');health=get_json(8896,'health')
    if q['ready']:
     assert q['arm']=='gate1' and q['actor']['source_checkpoint']['step']==4
     write_json_exclusive(out/'result.json',{'status':'passed','qam':q,'health':health,'rollout_performed':False});print('HTTP_READY_PASS',flush=True);break
   except OSError:pass
   if time.monotonic()>deadline:raise RuntimeError('readiness timeout')
   time.sleep(2)
 except Exception as e:
  write_json_exclusive(out/'FAILED.json',{'error':str(e)});raise
 finally:
  stop_owned(name);proc.wait(timeout=20)

import os,subprocess
from paired_eval import ROOT,ENV_IMAGE,docker_args,ensure_absent,stop_owned,inside
assert os.environ.get('HARNESS_RECEIPT')
out=ROOT/'outputs/analysis/groot_robocasa_lineage_20261021/input_preprocess_audit_v1'
out.mkdir(exist_ok=False)
name=f'groot-input-capture-{os.getpid()}';ensure_absent(name)
with open(out/'capture.log','x') as f:
 try:subprocess.run(docker_args(out,int(os.environ['GPUS']),name,ENV_IMAGE,network='none')+['scripts/rl2_vla/groot_robocasa_adapter_20261021/capture_initial_inputs.py','--mode','expert','--out',inside(out)],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
 finally:stop_owned(name)
print('CAPTURE_DONE',flush=True)

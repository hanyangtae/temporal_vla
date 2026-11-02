import os,subprocess
from paired_eval import ROOT,ENV_IMAGE,docker_args,ensure_absent,stop_owned
assert os.environ.get('HARNESS_RECEIPT')
out=ROOT/'outputs/analysis/groot_robocasa_lineage_20261021/env_reset_smoke_v2'
out.mkdir(exist_ok=False)
name=f'groot-qam-env-smoke-{os.getpid()}'
ensure_absent(name)
with open(out/'reset.log','x') as f:
 try:
  subprocess.run(docker_args(out,4,name,ENV_IMAGE,network='none')+['scripts/rl2_vla/groot_robocasa_adapter_20261021/env_reset_smoke.py'],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=240)
 finally:stop_owned(name)
print('ENV_RESET_PREFLIGHT_PASS',flush=True)

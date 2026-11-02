import argparse,os,subprocess
from pathlib import Path
from paired_eval import ROOT,MODEL_IMAGE,docker_args,ensure_absent,stop_owned,inside
ap=argparse.ArgumentParser();ap.add_argument('--actor',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--gpu',type=int,required=True);a=ap.parse_args()
assert os.environ.get('HARNESS_RECEIPT') and int(os.environ['GPUS'])==a.gpu
a.out=a.out.resolve();a.out.mkdir(exist_ok=False)
name=f'groot-qam-final-smoke-{os.getpid()}';ensure_absent(name)
with open(a.out/'smoke.log','x') as f:
 try:
  subprocess.run(docker_args(a.out,a.gpu,name,MODEL_IMAGE,network='none')+['scripts/rl2_vla/groot_robocasa_adapter_20261021/smoke_gate1.py',
   '--data-root','/cache/datasets','--actor',inside(a.actor),'--out',inside(a.out)],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=240)
 finally:stop_owned(name)
print('FINAL_GATE1_SMOKE_PASS',flush=True)

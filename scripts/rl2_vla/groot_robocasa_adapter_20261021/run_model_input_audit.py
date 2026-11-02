import os,subprocess
from paired_eval import ROOT,MODEL_IMAGE,docker_args,ensure_absent,stop_owned
assert os.environ.get('HARNESS_RECEIPT')
out=ROOT/'outputs/analysis/groot_robocasa_lineage_20261021/input_preprocess_audit_v1'
name=f'groot-input-model-{os.getpid()}';ensure_absent(name)
with open(out/'model.log','x') as f:
 try:subprocess.run(docker_args(out,int(os.environ['GPUS']),name,MODEL_IMAGE,network='none')+['scripts/rl2_vla/groot_robocasa_adapter_20261021/audit_model_inputs.py'],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
 finally:stop_owned(name)
print('MODEL_INPUT_AUDIT_DONE',flush=True)

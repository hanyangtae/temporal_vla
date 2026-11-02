import time,json,subprocess
from pathlib import Path
R=Path('/home/dongkyu/pkt_ws/temporal_vla');O=R/'outputs/eval/robocasa/groot_n15/safe_qam_opendrawer_heldout_20261021'
last=-1;deadline=time.time()+12*3600
while time.time()<deadline:
 p=O/'progress.jsonl';n=len(p.read_text().splitlines()) if p.exists() else 0
 done=(O/'RUN_COMPLETE.json').exists();failed=(O/'FAILED.json').exists()
 if n!=last and n>0 or done or failed:
  result=subprocess.run([str(R/'.venvs/rl2_iid/bin/python'),'scripts/rl2_vla/groot_robocasa_adapter_20261021/verify_safe_qam_eval_v3.py'],cwd=R,capture_output=True,text=True,timeout=120)
  (O/'latest_verification_v3.log').write_text(result.stdout+result.stderr);last=n
  print(time.time(),n,'verified_return',result.returncode,flush=True)
  if done or failed:
   (O/'VERIFIER_EXIT_V3.json').write_text(json.dumps({'time':time.time(),'completed':done,'run_failed':failed,'verifier_exit':result.returncode,'episodes':n}));break
 time.sleep(30)
else:(O/'VERIFIER_TIMEOUT_V3.json').write_text(json.dumps({'time':time.time(),'episodes':last}))

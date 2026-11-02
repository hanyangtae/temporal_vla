import json,os,time,subprocess,socket
from pathlib import Path
from paired_eval_audited import ROOT,MODEL_IMAGE,ENV_IMAGE,PROFILE,docker_args,inside,ensure_absent,stop_owned,get_json
from baseline_data import sha256_path
import shutil
_original_docker_args=docker_args
def docker_args(*args,**kwargs):
 argv=_original_docker_args(*args,**kwargs)
 # New containers only: avoid root-disk Docker logs and temporary cache writes.
 return argv[:2]+['--log-driver','none','--tmpfs','/tmp:rw,size=1g']+argv[2:]
def storage_guard():
 if shutil.disk_usage(ROOT).free<2*1024**3:
  raise RuntimeError('Root disk reserve below2GiB; evaluation paused before new episode; no deletion')

O=ROOT/'outputs/eval/robocasa/groot_n15/safe_qam_multitask_scratch_20261021'
M=json.loads((O/'manifest.json').read_text());actor=ROOT/'outputs/rl2_vla/groot_multitask_scratch_actor_20261021';port=8898
assert os.environ.get('HARNESS_RECEIPT') and int(os.environ['GPUS'])==M['gpu']
assert sha256_path(actor/'actor.npz')==M['actor_sha256']
for name,digest in M['code_hashes'].items():assert sha256_path(Path(__file__).parent/name)==digest
assert sha256_path(ROOT/'src/failure_online/online_failure.py')==M['detector_code_sha256']
with socket.socket() as s:s.bind(('127.0.0.1',port))
existing=[json.loads(x) for x in (O/'progress.jsonl').read_text().splitlines() if x.strip()];done_keys={(x['cell'],x['n'],x['arm']) for x in existing};assert len(done_keys)==len(existing);progress=(O/'progress.jsonl').open('a',buffering=1);start=time.time()
try:
 for c in M['cells']:
  assert sha256_path(ROOT/c['detector'])==c['detector_sha256']
  for arm in M['arms']:
   assert arm=='safe'
   pending=[row for row in c['test_rows'] if (c['scene_idx'],int(row['noise_idx']),arm) not in done_keys]
   if not pending:continue
   storage_guard()
   tag=f's{c["scene_idx"]}_{arm}';name=f'safe-qam-{tag}-{os.getpid()}';ensure_absent(name)
   argv=docker_args(O,M['gpu'],name,MODEL_IMAGE)+['scripts/rl2_vla/groot_robocasa_adapter_20261021/serve_safe_qam.py','--qam-arm',arm,'--qam-actor',inside(actor),'--detector',inside(ROOT/c['detector']),'--profile',PROFILE,'--host','127.0.0.1','--port',str(port)]
   log=(O/(tag+'.serve.v3.log')).open('x');proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT)
   try:
    deadline=time.monotonic()+240
    while True:
     if proc.poll() is not None:raise RuntimeError(f'{tag} server exit{proc.returncode}')
     try:
      state=get_json(port,'qam_contract')
      if state['ready']:break
     except (OSError,ValueError,KeyError):pass
     if time.monotonic()>deadline:raise RuntimeError('server timeout')
     time.sleep(2)
    (O/(tag+'.identity.v3.json')).write_text(json.dumps(state,indent=2))
    for row in pending:
     storage_guard()
     ni=int(row['noise_idx']);ct=f'{tag}_n{ni}';cn=f'safe-qam-collect-{ct}-{os.getpid()}';ensure_absent(cn)
     cmd=docker_args(O,M['gpu'],cn,ENV_IMAGE)+['scripts/rl2_vla/groot_robocasa_adapter_20261021/collect_safe_qam.py','--vla-server',f'http://127.0.0.1:{port}','--task','OpenDrawer','--env-name','robocasa_panda_omron/OpenDrawer_PandaOmron_Env','--split','target','--output-dir',inside(O/'raw_rollouts'),'--episode-start-idx','0','--n-episodes','1','--seed',row['env_seed'],'--inference-seed',row['inference_seed'],'--n-action-steps','5','--max-episode-steps','720','--video-fps','20','--steps-per-render','2','--wait-ready','--no-features','--expect-chunk-len','16','--plan-json','configs/collect/n15_grid_v6_scene_jitter/collection_plan.json','--scene-idx',str(c['scene_idx']),'--jitter-idx',str(c['target_j']),'--noise-idx',str(ni),'--grid-instruction','OpenDrawer/left','--grid-root',inside(O/'grid'),'--arm-dir',arm,'--run-tag',sha256_path(O/'manifest.json')]
     before=time.time()
     with (O/(ct+'.collect.v3.log')).open('x') as f:
      try:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=1800,check=True)
      finally:stop_owned(cn)
     q=get_json(port,'qam_contract');assert q['off_identity_verified']
     metas=[p for p in (O/'grid').rglob('meta.json') if p.parent.name==arm and f'/s{c["scene_idx"]}/' in str(p) and f'/n{ni}/' in str(p)];assert len(metas)==1,metas
     meta=json.loads(metas[0].read_text());item={'time':time.time(),'cell':c['scene_idx'],'j':c['target_j'],'n':ni,'arm':arm,'success':meta.get('episode_success',meta.get('success')),'seconds':time.time()-before,'meta':str(metas[0].relative_to(O)),'state':q};progress.write(json.dumps(item)+'\n');print(json.dumps(item),flush=True)
   finally:
    stop_owned(name);proc.wait(timeout=30);log.close()
 (O/'RUN_COMPLETE.json').write_text(json.dumps({'primary_episodes':60,'original_baseline_records':30,'safe_episodes':30,'baseline_replays_expected':0,'legacy_always':0,'seconds':time.time()-start,'validation':'pending independent hash/paired audit'},indent=2))
except Exception as e:
 (O/'FAILED.json').write_text(json.dumps({'error':str(e),'time':time.time()},indent=2));raise
finally:progress.close()

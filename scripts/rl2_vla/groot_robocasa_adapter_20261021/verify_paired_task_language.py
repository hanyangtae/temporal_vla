import json,time,hashlib,collections
from pathlib import Path
R=Path('/home/dongkyu/pkt_ws/temporal_vla');O=R/'outputs/eval/robocasa/groot_n15/qam_opendrawer_pilot_audited_20261021';OLD=R/'outputs/eval/robocasa/groot_n15/qam_opendrawer_pilot_20261021'
def readlines(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,data):
 with (O/name).open('x') as f:json.dump(data,f,indent=2)
def verify():
 rows=[];actions={};initials={}
 for arm in ['base','gate0','gate1']:
  received_files=list(O.glob(f'groot-qam-{arm}-*.received.jsonl'));assert len(received_files)==1,(arm,received_files)
  received=readlines(received_files[0]);sent=[]
  for i,seed in enumerate([100000,100001]):
   files=list(O.glob(f'*collect-{arm}-{i}-*.task_trace.jsonl'));assert len(files)==1,(arm,i,files)
   events=readlines(files[0]);assert events[-1]['event']=='complete'
   reset=[e for e in events if e['event']=='reset'];assert len(reset)==1
   after=[e for e in events if e['event']=='after_step'];before=[e for e in events if e['event']=='before_step'];assert len(after)==len(before)==events[-1]['steps']
   first=reset[0]['task'];assert all(e['task']==first for e in after+before)
   requests=[e for e in events if e['event']=='http_send'];responses=[e for e in events if e['event']=='http_response'];assert len(requests)==len(responses) and all(x['status']==200 for x in responses)
   assert all(e['task']==first['language'] for e in requests);sent.extend(requests)
   actions[arm,seed]=[e['actions'] for e in before];initials[arm,seed]=reset[0]['state_sha256']
   newmeta=next(p for p in (O/'grid').rglob('meta.json') if p.parent.name=='qam_'+arm and json.loads(p.read_text())['scenario_seed']==seed)
   oldmeta=next(p for p in (OLD/'grid').rglob('meta.json') if p.parent.name=='qam_'+arm and json.loads(p.read_text())['scenario_seed']==seed)
   nm=json.loads(newmeta.read_text());om=json.loads(oldmeta.read_text())
   assert first['fixture']==om['ep_meta']['fixture_refs']['drawer'] and first['language']==om['task_description']
   rows.append({'arm':arm,'seed':seed,'steps':len(after),'requests':len(requests),'success':nm['episode_success'],'task':first,'original_trajectory_byte_equal':sha(newmeta.parent/'traj.csv')==sha(oldmeta.parent/'traj.csv'),'trace_sha256':sha(files[0])})
  assert [e['payload_sha256'] for e in sent]==[e['payload_sha256'] for e in received],f'{arm} actual sent/received payload mismatch'
 for seed in [100000,100001]:
  assert len({initials[arm,seed] for arm in ['base','gate0','gate1']})==1,'Paired initial-state mismatch'
  assert actions['base',seed]==actions['gate0',seed],'Full action dictionary base/gate0 mismatch'
 save('task_language_verified.json',{'status':'verified','episodes':rows,'all_env_step_task_language_consistent':True,'all_sent_received_payload_hashes_equal':True,'base_gate0_full_action_dictionaries_equal':True,'paired_initial_states_equal':True,'goals_modified':False,'weights_modified':False,'no_checkpoint_reselection':True,'claim':'Two seeds per arm, diagnostic only; no generalization or statistical improvement claim.'})
 print('TASK_LANGUAGE_VERIFIED',flush=True)
deadline=time.monotonic()+5400
try:
 while not (O/'evaluation_complete.json').exists():
  if (O/'FAILED.json').exists():raise RuntimeError((O/'FAILED.json').read_text())
  if time.monotonic()>deadline:raise TimeoutError('5400-second bounded verifier wait expired; no process terminated')
  time.sleep(20)
 verify()
except Exception as e:
 if O.exists():save('task_language_verification_failed.json',{'error':str(e),'type':type(e).__name__})
 raise

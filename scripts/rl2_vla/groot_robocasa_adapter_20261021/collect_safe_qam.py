from pathlib import Path as _PausePath
if _PausePath('/temporal_vla/outputs/eval/robocasa/groot_n15/safe_qam_opendrawer_heldout_20261021/PAUSE_BEFORE_NEXT_EPISODE.json').exists():
 print('PAUSED_AT_EPISODE_BOUNDARY: reproducibility audit required; no environment episode started',flush=True)
 raise SystemExit(75)
import json,os,runpy,sys,time,hashlib
from pathlib import Path
import numpy as np
from readonly_assets import install
from task_language_guard import canonical_hash
install();ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'scripts/utils'))
plan=json.loads((ROOT/'configs/collect/n15_grid_v6_scene_jitter/collection_plan.json').read_text());sid=int(sys.argv[sys.argv.index('--scene-idx')+1]);key=sys.argv[sys.argv.index('--grid-instruction')+1];expected=plan['scenes'][key][sid]
stream=Path(os.environ['PDK_LOG_FILE']).with_suffix('.task_trace.jsonl').open('x',buffering=1);ctx={'env':None,'started':False,'steps':0,'initial':None,'latest_obs':None}
def emit(event,**kw):stream.write(json.dumps({'event':event,'time':time.time(),'step':ctx['steps'],**kw},sort_keys=True)+'\n')
def check():
 e=ctx['env'].env;m=e.get_ep_meta();task={'language':m['lang'],'fixture':e.drawer.name,'drawer_side':e.drawer_side,'layout':m['layout_id'],'style':m['style_id']}
 assert task['language']==expected['lang']==f"Open the {task['drawer_side']} drawer."
 assert task['layout']==expected['layout'] and task['style']==expected['style']
 group=expected['fixture_group'];assert task['fixture']==group or task['fixture'].startswith(group+'_'),(task,group)
 assert ctx['latest_obs']['annotation.human.action.task_description']==task['language']
 if not ctx['started']:
  ctx['initial']=task.copy();ctx['started']=True;state=np.asarray(e.sim.get_state().flatten());emit('reset',task=task,state_sha256=hashlib.sha256(state.tobytes()).hexdigest())
 assert task==ctx['initial'];return task
from robocasa.utils.gym_utils.gymnasium_groot import GrootRoboCasaEnv
reset0=GrootRoboCasaEnv.reset;step0=GrootRoboCasaEnv.step

def reset(self,*args,**kwargs):
 assert not ctx['started'],'reset after episode start';obs,info=reset0(self,*args,**kwargs);ctx['env']=self;ctx['latest_obs']=obs;return obs,info

def step(self,action):
 t=check();emit('before_step',task=t,actions={k:np.asarray(v).tolist() for k,v in action.items()})
 obs,reward,terminated,truncated,info=step0(self,action);ctx['latest_obs']=obs;ctx['steps']+=1;t=check();emit('after_step',task=t,success=bool(self.env._check_success()),reward=np.asarray(reward).tolist(),terminated=bool(terminated),truncated=bool(truncated));return obs,reward,terminated,truncated,info
GrootRoboCasaEnv.reset=reset;GrootRoboCasaEnv.step=step
import requests
send0=requests.sessions.Session.send

def send(self,request,**kw):
 if request.method=='POST' and request.url.split('?')[0].endswith('/act_with_features'):
  payload=json.loads(request.body);t=check();assert payload['task']==t['language'];emit('http_send',task=payload['task'],payload_sha256=canonical_hash(payload),inference_seed=payload.get('inference_seed'));response=send0(self,request,**kw);emit('http_response',status=response.status_code);return response
 return send0(self,request,**kw)
requests.sessions.Session.send=send
try:runpy.run_path(str(ROOT/'scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py'),run_name='__main__');emit('complete',steps=ctx['steps'])
except Exception as exc:emit('failed',error=str(exc));raise
finally:stream.close()

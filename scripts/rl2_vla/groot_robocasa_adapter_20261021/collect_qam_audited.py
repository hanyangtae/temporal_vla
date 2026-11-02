import json,os,runpy,sys,time
from pathlib import Path
import numpy as np
from readonly_assets import install
from task_language_guard import TaskGuard,canonical_hash
install()
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'scripts/utils'))
seed=sys.argv[sys.argv.index('--seed')+1]
expected=json.loads((Path(__file__).parent/'paired_original_targets.json').read_text())[seed]
guard=TaskGuard(expected)
trace=Path(os.environ['PDK_LOG_FILE']).with_suffix('.task_trace.jsonl');stream=trace.open('x',buffering=1)
def emit(event,**kw):stream.write(json.dumps({'event':event,'time':time.time(),'step':guard.steps,**kw},sort_keys=True)+'\n')
def task(wrapper,obs):
 e=wrapper.env;m=e.get_ep_meta();lang=obs['annotation.human.action.task_description']
 return {'language':m['lang'],'observed_language':lang,'fixture':e.drawer.name,'drawer_side':e.drawer_side,'layout':m['layout_id'],'style':m['style_id']}
from robocasa.utils.gym_utils.gymnasium_groot import GrootRoboCasaEnv
reset0=GrootRoboCasaEnv.reset;step0=GrootRoboCasaEnv.step

def reset(self,*args,**kwargs):
 if guard.initial is not None:raise RuntimeError('Unexpected second environment reset')
 obs,info=reset0(self,*args,**kwargs);t=task(self,obs);guard.check(t)
 import hashlib
 state=np.asarray(self.env.sim.get_state().flatten());emit('reset',task=t,state_sha256=hashlib.sha256(state.tobytes()).hexdigest())
 return obs,info

def step(self,action):
 before={'language':self.env.get_ep_meta()['lang'],'observed_language':guard.latest['observed_language'],'fixture':self.env.drawer.name,'drawer_side':self.env.drawer_side,'layout':self.env.get_ep_meta()['layout_id'],'style':self.env.get_ep_meta()['style_id']};guard.check(before)
 action_values={k:np.asarray(v).tolist() for k,v in action.items()};emit('before_step',task=before,actions=action_values)
 obs,reward,terminated,truncated,info=step0(self,action);guard.steps+=1;t=task(self,obs);guard.check(t);emit('after_step',task=t,success=bool(self.env._check_success()))
 return obs,reward,terminated,truncated,info
GrootRoboCasaEnv.reset=reset;GrootRoboCasaEnv.step=step
# Inspect the bytes actually passed to requests.Session.send, not only a payload builder.
import requests
send0=requests.sessions.Session.send

def send(self,request,**kw):
 if request.method=='POST' and request.url.split('?')[0].endswith('/act_with_features'):
  payload=json.loads(request.body);guard.sent(payload);emit('http_send',task=payload['task'],payload_sha256=canonical_hash(payload),inference_seed=payload.get('inference_seed'))
  response=send0(self,request,**kw);emit('http_response',status=response.status_code);return response
 return send0(self,request,**kw)
requests.sessions.Session.send=send
try:
 runpy.run_path(str(ROOT/'scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py'),run_name='__main__')
 emit('complete',steps=guard.steps)
except Exception as exc:emit('failed',error=str(exc));raise
finally:stream.close()

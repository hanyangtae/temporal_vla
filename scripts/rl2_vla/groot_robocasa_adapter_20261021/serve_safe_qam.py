import argparse,sys,json,os,time,copy
from pathlib import Path
import numpy as np,torch
from extract_opendrawer_latents import import_serve_module
from composition import ComposedFlowSampler,SPEC
from qam_torch_actor import TorchQAMVelocity
from task_language_guard import canonical_hash

def main():
 p=argparse.ArgumentParser(add_help=False);p.add_argument('--qam-arm',choices=['base','always','safe'],required=True);p.add_argument('--qam-actor',type=Path,required=True);p.add_argument('--detector',type=Path,required=True);a,rest=p.parse_known_args()
 amendment=Path('/temporal_vla/outputs/eval/robocasa/groot_n15/safe_qam_opendrawer_heldout_20261021/queue_amendment_v2.json')
 if a.qam_arm=='always' and amendment.exists():
  print('QUEUE_WITHDRAWN: user removed remaining always-on evaluations; no model or episode started',flush=True)
  raise SystemExit(75)
 sv=import_serve_module()
 from src.failure_online import OnlineFailureDetector
 from lerobot_adapters.groot import GrootPolicyAdapter
 original=GrootPolicyAdapter.load
 stream=Path(os.environ['PDK_LOG_FILE']).with_suffix('.decisions.jsonl').open('x',buffering=1)
 received=Path(os.environ['PDK_LOG_FILE']).with_suffix('.received.jsonl').open('x',buffering=1)
 state={'ready':False,'arm':a.qam_arm,'requests':0,'gate_calls':0,'off_identity_verified':False}
 seed0=sv._apply_inference_seed
 def seed(payload):
  if 'task' in payload:received.write(json.dumps({'payload_sha256':canonical_hash(payload),'task':payload['task'],'inference_seed':payload.get('inference_seed'),'time':time.time()})+'\n')
  return seed0(payload)
 sv._apply_inference_seed=seed
 def load(self,*args,**kwargs):
  loaded=original(self,*args,**kwargs);loaded.policy.eval();loaded.policy.requires_grad_(False)
  det=OnlineFailureDetector.from_checkpoint(a.detector,alpha=.1,device='cpu');state['detector']=det.spec()
  actor=TorchQAMVelocity(a.qam_actor,device='cpu');head=loaded.policy._groot_model.action_head;native=head.get_action
  sampler=ComposedFlowSampler(SPEC,qam_fn=actor,qam_weight=.5)
  reset0=loaded.policy.reset
  def reset(*args,**kwargs):det.reset();return reset0(*args,**kwargs)
  loaded.policy.reset=reset
  def cloned(x):return type(x)(data={k:v.clone() if isinstance(v,torch.Tensor) else copy.deepcopy(v) for k,v in x.items()})
  def get_action(backbone_output,action_input):
   started=time.perf_counter();buf=[]
   def hook(m,args,out):buf.append(out.detach().float().cpu().numpy().astype(np.float16)[0])
   handle=head.model.transformer_blocks[12].register_forward_hook(hook)
   cpu=torch.random.get_rng_state();gpu=torch.cuda.get_rng_state()
   def rewind():torch.random.set_rng_state(cpu);torch.cuda.set_rng_state(gpu)
   try:
    base=native(cloned(backbone_output),action_input);assert len(buf)==4
    hidden=np.stack(buf)[None];snap=det.snapshot();pre=det.step(det.feature_from_hidden(hidden,0))
    if not state['off_identity_verified']:
     rewind();buf.clear();off,_=sampler.sample(head,cloned(backbone_output),action_input,decision_env_step=0,gate=0.)
     if not torch.equal(off,base['action_pred']):raise RuntimeError('detector-off base identity failed')
     state['off_identity_verified']=True
    gate=a.qam_arm=='always' or (a.qam_arm=='safe' and pre['fired'])
    result=base;post=pre;delta=0.
    if gate:
     rewind();buf.clear();actions,info=sampler.sample(head,cloned(backbone_output),action_input,decision_env_step=pre['t']*5,gate=1.)
     assert len(buf)==4
     det.restore(snap);post=det.step(det.feature_from_hidden(np.stack(buf)[None],0))
     result=type(backbone_output)(data={'action_pred':actions});delta=(actions[...,:12]-base['action_pred'][...,:12]).float().abs().max().item();state['gate_calls']+=1
    state['requests']+=1
    stream.write(json.dumps({'time':time.time(),'request':state['requests'],'arm':a.qam_arm,'detector':pre,'post_detector':post,'gate':bool(gate),'qam_delta_max_normalized':delta,'latency_ms':(time.perf_counter()-started)*1000,'action_finite':bool(torch.isfinite(result['action_pred']).all()),'off_identity_verified':state['off_identity_verified']})+'\n')
    return result
   finally:handle.remove()
  head.get_action=get_action;state['ready']=True;return loaded
 GrootPolicyAdapter.load=load
 @sv.app.get('/qam_contract')
 def contract():return state
 sys.argv=[str(Path(sv.__file__)),*rest];sv.main()
if __name__=='__main__':main()

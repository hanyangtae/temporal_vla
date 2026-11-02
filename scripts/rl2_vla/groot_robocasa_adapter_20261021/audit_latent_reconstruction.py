import json,copy,inspect,hashlib
from pathlib import Path
import numpy as np,torch,av,cv2
from extract_opendrawer_latents import FrozenGroot,import_serve_module,png_b64,state_payload_map,video_payload_map
ROOT=Path('/temporal_vla');O=ROOT/'outputs/analysis/groot_robocasa_lineage_20261021/input_preprocess_audit_v2'
D=Path('/cache/datasets/robocasa/v1.0/pretrain/atomic/OpenDrawer/20250819/lerobot')
CK=Path('/cache/datasets/huggingface/hub/models--robocasa--robocasa365_checkpoints/snapshots/c484448aba1a9b60a04c9b0ca117241518ea69f3/gr00t_n1-5/multitask_learning/checkpoint-120000')
serve=import_serve_module();model=FrozenGroot(serve,profile_path=ROOT/'configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml',pretrained=CK,device='cuda:0')
import pandas as pd
mod=json.loads((D/'meta/modality.json').read_text());state=np.asarray(pd.read_parquet(D/'data/chunk-000/episode_000001.parquet')['observation.state'].iloc[0]);language=json.loads((D/'extras/episode_000001/ep_meta.json').read_text())['lang']
sm=state_payload_map(mod,model.profile);vm=video_payload_map(mod)
rest=dict(np.load(O/'restored_observation.npz',allow_pickle=False));payloads={};images={};report={'samples':{},'camera_map':vm,'state_map':sm,'language':language,'embodiment':model.profile.model_specific['embodiment_tag']}
source={}
for video,key in vm.items():
 with av.open(str(D/'videos/chunk-000'/video/'episode_000001.mp4')) as c:source[key]=next(c.decode(video=0)).to_ndarray(format='rgb24')
payloads['source']={**{k:png_b64(v) for k,v in source.items()},**{k:state[lo:hi].tolist() for k,(lo,hi) in sm.items()},'task':language}
from src.policies.groot.robocasa.io import prepare_groot_robocasa_http_request
rest['annotation.human.action.task_description']=json.loads((O/'capture_complete.json').read_text())['language']
request=prepare_groot_robocasa_http_request(rest,strict=False)
payloads['restored']={**{('observation.images.'+k if not k.startswith('observation.') else k):png_b64(v) for k,v in request.images.items()},**{k:v.tolist() for k,v in request.states.items()},'task':request.instruction}

O=ROOT/'outputs/analysis/groot_robocasa_lineage_20261021/latent_reconstruction_v1'
report={'samples':{},'archive_quantization':'float16','scope':'two actual observations; simultaneous last-block and decoder-input hooks; no rollout'}
dit=model.head.model
for name,payload in payloads.items():
 captured=[];inputs=[];times=[]
 def bh(m,args,out): captured.append(out.detach().clone())
 def dh(m,args): inputs.append(args[0].detach().clone())
 def th(m,args,out): times.append(out.detach().clone())
 hooks=[dit.transformer_blocks[-1].register_forward_hook(bh),model.head.action_decoder.register_forward_pre_hook(dh),dit.timestep_encoder.register_forward_hook(th)]
 try:
  batch=model.preprocessor(serve._apply_input_remap(serve.parse_payload(payload)))
  serve._apply_inference_seed({'inference_seed':1400000})
  with torch.inference_mode(): action=model.policy.predict_action_chunk(batch)
 finally:
  for h in hooks:h.remove()
 assert len(captured)==len(inputs)==len(times)==4
 result=[]
 with torch.inference_mode():
  for k,(last,expected,temb) in enumerate(zip(captured,inputs,times)):
   shift,scale=dit.proj_out_1(torch.nn.functional.silu(temb)).chunk(2,dim=1)
   def convert(x):return dit.proj_out_2(dit.norm_out(x)*(1+scale[:,None])+shift[:,None])
   rebuilt=convert(last)
   quant=last.to(torch.float16).cpu().to(last.device,dtype=last.dtype)
   recovered=convert(quant)
   def metrics(x,y):
    diff=(x.float()-y.float()).abs();return {'max_abs':diff.max().item(),'mean_abs':diff.mean().item(),'rmse':diff.square().mean().sqrt().item(),'exact':torch.equal(x,y)}
   result.append({'denoise':k,'dtype':str(last.dtype),'last_shape':list(last.shape),'decoder_shape':list(expected.shape),'native':metrics(rebuilt,expected),'fp16_tokens':metrics(recovered,expected),'fp16_qam_latent':metrics(recovered[:,-16:].float().mean(1),expected[:,-16:].float().mean(1))})
 report['samples'][name]=result
report['native_all_exact']=all(x['native']['exact'] for v in report['samples'].values() for x in v)
(O/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

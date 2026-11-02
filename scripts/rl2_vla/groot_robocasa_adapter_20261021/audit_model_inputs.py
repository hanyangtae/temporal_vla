import json,copy,inspect,hashlib
from pathlib import Path
import numpy as np,torch,av,cv2
from extract_opendrawer_latents import FrozenGroot,import_serve_module,png_b64,state_payload_map,video_payload_map
ROOT=Path('/temporal_vla');O=ROOT/'outputs/analysis/groot_robocasa_lineage_20261021/input_preprocess_audit_v1'
D=Path('/cache/datasets/robocasa/v1.0/pretrain/atomic/OpenDrawer/20250819/lerobot')
CK=Path('/cache/datasets/huggingface/hub/models--robocasa--robocasa365_checkpoints/snapshots/c484448aba1a9b60a04c9b0ca117241518ea69f3/gr00t_n1-5/multitask_learning/checkpoint-120000')
serve=import_serve_module();model=FrozenGroot(serve,profile_path=ROOT/'configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml',pretrained=CK,device='cuda:0')
import pandas as pd
mod=json.loads((D/'meta/modality.json').read_text());state=np.asarray(pd.read_parquet(D/'data/chunk-000/episode_000001.parquet')['observation.state'].iloc[0]);language=json.loads((O/'capture_complete.json').read_text())['language']
sm=state_payload_map(mod,model.profile);vm=video_payload_map(mod)
rest=dict(np.load(O/'restored_observation.npz',allow_pickle=False));payloads={};images={};report={'samples':{},'camera_map':vm,'state_map':sm,'language':language,'embodiment':model.profile.model_specific['embodiment_tag']}
source={}
for video,key in vm.items():
 with av.open(str(D/'videos/chunk-000'/video/'episode_000001.mp4')) as c:source[key]=next(c.decode(video=0)).to_ndarray(format='rgb24')
payloads['source']={**{k:png_b64(v) for k,v in source.items()},**{k:state[lo:hi].tolist() for k,(lo,hi) in sm.items()},'task':language}
from src.policies.groot.robocasa.io import prepare_groot_robocasa_http_request
rest['annotation.human.action.task_description']=language
request=prepare_groot_robocasa_http_request(rest,strict=False)
payloads['restored']={**{('observation.images.'+k if not k.startswith('observation.') else k):png_b64(v) for k,v in request.images.items()},**{k:v.tolist() for k,v in request.states.items()},'task':request.instruction}
# Preserve actual preprocessor intermediate arrays via the existing pack-step implementation.
from lerobot.policies.groot.processor_groot import GrootPackInputsStep
from lerobot.types import TransitionKey
orig=GrootPackInputsStep.__call__;current=['']
def pack(self,transition):
 result=orig(self,transition);obs=result[TransitionKey.OBSERVATION];comp=result[TransitionKey.COMPLEMENTARY_DATA]
 arr={k:(v.detach().cpu().numpy() if isinstance(v,torch.Tensor) else v) for k,v in obs.items() if isinstance(v,(torch.Tensor,np.ndarray))}
 np.savez_compressed(O/(current[0]+'_packed.npz'),**arr)
 report['samples'][current[0]]['packed_language']=comp.get('language');report['samples'][current[0]]['embodiment_id']=comp['embodiment_id'].tolist()
 return result
GrootPackInputsStep.__call__=pack
for name,payload in payloads.items():
 current[0]=name;report['samples'][name]={}
 batch=serve._apply_input_remap(serve.parse_payload(payload))
 arrays={k:v.detach().cpu().numpy() for k,v in batch.items() if isinstance(v,torch.Tensor)}
 np.savez_compressed(O/(name+'_before_preprocessor.npz'),**arrays)
 report['samples'][name]['ordered_image_keys']=sorted(k for k in arrays if 'images' in k)
 report['samples'][name]['state_before']=arrays['observation.state'].tolist()
 report['samples'][name]['task_before']=batch['task']
 processed=model.preprocessor(batch)
 arr={k:v.detach().float().cpu().numpy() for k,v in processed.items() if isinstance(v,torch.Tensor)}
 np.savez_compressed(O/(name+'_model_inputs.npz'),**arr)
 report['samples'][name]['model_shapes']={k:list(v.shape) for k,v in arr.items()}
 serve._apply_inference_seed({'inference_seed':1400000})
 with torch.inference_mode():action=model.policy.predict_action_chunk(processed)
 report['samples'][name]['forward_finite']=bool(torch.isfinite(action).all());report['samples'][name]['forward_shape']=list(action.shape)
 print(name,'INPUT_CAPTURE_AND_FORWARD_PASS',flush=True)
GrootPackInputsStep.__call__=orig
report['image_metrics']={}
for cam,v in source.items():
 alias=cam.split('.')[-1];rv=request.images.get(alias,request.images.get(cam));assert rv is not None
 if v.shape!=rv.shape:cv=cv2.resize(v,(rv.shape[1],rv.shape[0]),interpolation=cv2.INTER_AREA)
 else:cv=v
 report['image_metrics'][alias]={'source_shape':list(v.shape),'restored_shape':list(rv.shape),'mae_rgb':float(np.abs(cv.astype(float)-rv).mean()),'mae_bgr_swap':float(np.abs(cv[...,::-1].astype(float)-rv).mean()),'mae_vertical_flip':float(np.abs(cv[::-1].astype(float)-rv).mean())}
 images['source_'+alias]=v;images['restored_'+alias]=rv
np.savez_compressed(O/'comparison_rgb.npz',**images)
with (O/'model_preprocess_audit.json').open('x') as f:json.dump(report,f,indent=2)
print('MODEL_PREPROCESS_AUDIT_COMPLETE',flush=True)

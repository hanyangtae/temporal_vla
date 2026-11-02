"""No-install float32 PyTorch deployment of the pinned QAM actor.
Only the actual plain Dense/GELU actor profile is supported; other architectures fail closed.
The final QAM action clipping is NOT part of velocity; GR00T composition preserves its base integration.
"""
import argparse
import io
import json
from pathlib import Path
import numpy as np
from baseline_data import write_exclusive, write_json_exclusive, sha256_path
from composition import SPEC
from contract import ContractError, validate_qam_flags

VERSION="groot_qam_torch_velocity/v1"

def export_actor(trainer, checkpoint_dir, out, *, selection=None, diagnostic=False):
    cfg=trainer.agent.config
    if cfg['actor_layer_norm'] or cfg.get('encoder') is not None:
        raise ContractError('Only pinned plain Dense/GELU actor is supported')
    flags=json.loads((Path(checkpoint_dir)/f'flags_{trainer.steps_done}.json').read_text()); validate_qam_flags(flags,feature_signature=SPEC.signature())
    model='slow' if cfg['inv_temp']==0 else 'fast'
    tree=trainer.agent.network.params['modules_actor_'+model]
    if set(tree)!={'mlp'}:raise ContractError('Unexpected actor encoder/fourier parameters')
    widths=[1024+192+1,*list(cfg['actor_hidden_dims']),192]
    mlp=tree['mlp']; expected={'Dense_'+str(i) for i in range(len(widths)-1)}
    if set(mlp)!=expected:raise ContractError('Unsupported actor modules')
    arrays={}
    for i,(a,b) in enumerate(zip(widths[:-1],widths[1:])):
        layer=mlp['Dense_'+str(i)]
        if set(layer)!={'kernel','bias'}:raise ContractError('Unexpected dense params')
        for key,shape in [('kernel',(a,b)),('bias',(b,))]:
            x=np.asarray(layer[key],np.float32)
            if x.shape!=shape or not np.isfinite(x).all():raise ContractError('Invalid actor tensor')
            arrays[f'{i}_{key}']=x
    checkpoint_dir=Path(checkpoint_dir);source=checkpoint_dir/f'params_{trainer.steps_done}.pkl'
    source_sha=sha256_path(source)
    selected=None
    if selection is not None:
        selection=Path(selection);doc=json.loads(selection.read_text())
        if doc['selected_step']!=trainer.steps_done:raise ContractError('Checkpoint not validation selected')
        if doc['selected_checkpoint']['files'][source.name]['sha256']!=source_sha:
            raise ContractError('Selected checkpoint hash mismatch')
        selected={'receipt_sha256':sha256_path(selection),'step':trainer.steps_done,
                  'val_metric':doc['val_metric'],'production_run':doc['production_run']}
    elif not diagnostic:raise ContractError('Selection receipt required unless explicitly diagnostic')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    buf=io.BytesIO();np.savez(buf,**arrays);write_exclusive(out/'actor.npz',buf.getvalue())
    meta={'version':VERSION,'feature_signature':SPEC.signature(),'flags':flags,'widths':widths,
          'actor':model,'activation':'gelu_approximate_tanh','dtype':'float32',
          'layer_norm':False,'flow_steps':4,'time_input':'i/4, i in 0..3',
          'velocity_only':True,'final_action_clipping':False,
          'source_checkpoint':{'file':source.name,'sha256':source_sha,'step':trainer.steps_done},
          'validation_selection':selected,'diagnostic_only':selected is None or not selected['production_run'],
          'weights_sha256':sha256_path(out/'actor.npz')}
    write_json_exclusive(out/'actor.json',meta)
    return meta

class TorchQAMVelocity:
    def __init__(self,path,device='cpu'):
        import torch
        self.torch=torch;self.device=torch.device(device);self.calls=0
        path=Path(path);m=json.loads((path/'actor.json').read_text());self.meta=m
        if m['version']!=VERSION or m['feature_signature']!=SPEC.signature():raise ContractError('Actor contract mismatch')
        validate_qam_flags(m['flags'],feature_signature=SPEC.signature())
        if m['activation']!='gelu_approximate_tanh' or m['layer_norm'] or m['flow_steps']!=4:
            raise ContractError('Unsupported actor architecture')
        if sha256_path(path/'actor.npz')!=m['weights_sha256']:raise ContractError('Actor weights hash mismatch')
        widths=m['widths']
        if widths[0]!=1217 or widths[-1]!=192:raise ContractError('Actor dimensions mismatch')
        self.layers=[]
        with np.load(path/'actor.npz',allow_pickle=False) as z:
            expected={f'{i}_{key}' for i in range(len(widths)-1) for key in ['kernel','bias']}
            if set(z.files)!=expected:raise ContractError('Actor parameter keys mismatch')
            for i,(a,b) in enumerate(zip(widths[:-1],widths[1:])):
                w,bias=z[f'{i}_kernel'],z[f'{i}_bias']
                if w.shape!=(a,b) or bias.shape!=(b,) or not np.isfinite(w).all() or not np.isfinite(bias).all():raise ContractError('Actor tensor invalid')
                self.layers.append((torch.tensor(w,dtype=torch.float32,device=self.device),torch.tensor(bias,dtype=torch.float32,device=self.device)))
    def __call__(self,feature,noisy,t):
        torch=self.torch;f=np.asarray(feature);x=np.asarray(noisy)
        if int(t)!=t or not 0<=int(t)<4:raise ContractError('flow step must be integer0..3')
        if f.ndim!=2 or f.shape[1]!=1024 or x.shape!=(len(f),192):raise ContractError('velocity input dimensions')
        if not np.isfinite(f).all() or not np.isfinite(x).all():raise ContractError('nonfinite velocity input')
        with torch.inference_mode(),torch.autocast(device_type=self.device.type,enabled=False):
            inp=np.concatenate([f,x,np.full((len(f),1),int(t)/4)],axis=1).astype(np.float32)
            y=torch.tensor(inp,device=self.device)
            for i,(w,b) in enumerate(self.layers):
                y=y@w+b
                if i<len(self.layers)-1:y=torch.nn.functional.gelu(y,approximate='tanh')
            out=y.cpu().numpy().copy()
        if not np.isfinite(out).all():raise ContractError('nonfinite QAM velocity')
        self.calls+=1
        return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint-dir',type=Path,required=True)
    ap.add_argument('--step',type=int,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--selection',type=Path);ap.add_argument('--diagnostic',action='store_true');a=ap.parse_args()
    from qam_prefix import PrefixQAMTrainer
    tr=PrefixQAMTrainer.load(a.checkpoint_dir,a.step,feature_signature=SPEC.signature())
    print(json.dumps(export_actor(tr,a.checkpoint_dir,a.out,selection=a.selection,diagnostic=a.diagnostic)))
if __name__=='__main__':main()

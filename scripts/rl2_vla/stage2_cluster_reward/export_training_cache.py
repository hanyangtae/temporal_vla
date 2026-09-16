"""Validate archive in place and export derived SAFE features/QAM transitions only."""
import argparse,hashlib,itertools,json
from pathlib import Path
import numpy as np
from data import TASKS,assign_splits,load_episode,sha256,transitions
from clusters import fit
COMMANDS=('0.0','1.0','mean','concat-2')

def reduce_axis(x,mode):
    if mode=='mean':return x.mean(axis=-2)
    if mode=='concat-2':
        y=x[..., [0,x.shape[-2]-1],:]
        return y.reshape(*y.shape[:-2],-1)
    return x[...,round((x.shape[-2]-1)*float(mode)),:]

def feature(x,h,d):
    return reduce_axis(reduce_axis(x[:,0,:,1:,:],h),d)

def main():
    p=argparse.ArgumentParser();p.add_argument('--rollouts',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    paths=sorted(args.rollouts.glob('collect/seed*/*/episode_*.json'))
    if len(paths)!=1200:raise ValueError(f'expected1200 complete records, got{len(paths)}')
    eps=[load_episode(x) for x in paths]
    identities={(e['task_id'],e['env_seed'],e['policy_seed']) for e in eps}
    expected=set(itertools.product(TASKS,range(1000,1100),(0,7,42)))
    if identities!=expected or len({e['episode_id'] for e in eps})!=1200:raise ValueError('collection grid mismatch')
    if len({e['policy_checkpoint'] for e in eps})!=1:raise ValueError('mixed checkpoint')
    splits=assign_splits(eps,42)
    safe_splits=dict(splits)
    for task in TASKS:
        ids=sorted({e['reset_id'] for e in eps if e['task_id']==task and splits[(task,e['reset_id'])]=='validation'},key=lambda r:hashlib.sha256(f'safe-selection:{r}'.encode()).hexdigest())
        assert len(ids)==20
        for i,r in enumerate(ids):safe_splits[(task,r)]='selection' if i<10 else 'calibration'
    rows=[];safe_rows=[]
    for e in eps:
        row={k:e[k] for k in ('episode_id','reset_id','task_id','success','env_seed','policy_seed')}
        rows.append(dict(row,path=e['_path'],sha256=e['_sha256'],split=splits[(e['task_id'],e['reset_id'])]))
        safe_rows.append(dict(row,task=e['task_id'],split=safe_splits[(e['task_id'],e['reset_id'])],length=len(e['chunks']),chunk_lengths=[len(c['executed_actions']) for c in e['chunks']],source_sha256=e['_sha256']))
    manifest=out/'manifest.json';manifest.write_text(json.dumps({'schema_version':1,'seed':42,'episodes':rows},indent=2))
    train=[e for e in eps if splits[(e['task_id'],e['reset_id'])]=='train']
    bundle=fit(train,sha256(manifest),seed=42);bundle.save(out/'clusters.json')
    cache={'schema_version':1,'manifest_sha256':sha256(manifest),'episodes':safe_rows,'features':{},'features_sha256':{},'qam_transitions':{},'source_grid':str(args.rollouts.resolve()),'feature_contract':'pi0_full_action_embeddings_lossless_v1'}
    for mode in ('base','cluster_potential'):
        values=transitions(train,bundle if mode=='cluster_potential' else None,.1 if mode=='cluster_potential' else 0.,.99)
        f=out/f'transitions_{mode}.npz';np.savez_compressed(f,**values)
        cache['qam_transitions'][mode]={'path':f.name,'sha256':sha256(f),'manifest_sha256':sha256(manifest),'cluster_sha256':sha256(out/'clusters.json') if mode=='cluster_potential' else None,'shaping_scale':.1 if mode=='cluster_potential' else 0.,'discount':.99,'episode_ids':[e['episode_id'] for e in train],'transitions':len(values['observations'])}
    max_t=max(len(e['chunks']) for e in eps)
    features={}
    for h,d in itertools.product(COMMANDS,repeat=2):
        dim=1024*(2 if h=='concat-2' else 1)*(2 if d=='concat-2' else 1)
        features[(h,d)]=np.zeros((len(eps),max_t,dim),np.float32)
    for i,e in enumerate(eps):
        meta=e['activation_file'];path=Path(e['_path']).parent/meta['path']
        if Path(meta['path']).name!=meta['path']:raise ValueError('unsafe sidecar path')
        if path.stat().st_size!=meta['size_bytes'] or sha256(path)!=meta['sha256']:raise ValueError(f'sidecar identity mismatch:{path}')
        with np.load(path,allow_pickle=False) as a:x=a['action_embeds']
        if x.shape!=(len(e['chunks']),1,10,5,1024) or not np.isfinite(x).all():raise ValueError(f'activation invalid:{path}:{x.shape}')
        baseline=feature(x,'mean','0.0')
        if not np.allclose(baseline,np.asarray([c['context'] for c in e['chunks']]),atol=1e-6):raise ValueError('QAM context/activation mismatch')
        for (h,d),v in features.items():v[i,:len(x)]=feature(x,h,d)
        if i%100==0:print('validated',i,flush=True)
    for (h,d),v in features.items():
        key=f'h={h}__d={d}';path=out/f'{key}.npz';np.savez_compressed(path,features=v)
        cache['features'][key]=path.name;cache['features_sha256'][key]=sha256(path)
        print('feature_saved',key,path.stat().st_size,flush=True)
    (out/'cache.json').write_text(json.dumps(cache,indent=2))
    summary={t:{s:{'episodes':sum(e['task']==t and e['split']==s for e in safe_rows),'successes':sum(e['task']==t and e['split']==s and e['success'] for e in safe_rows)} for s in ('train','selection','calibration','holdout')} for t in TASKS}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'DONE.json').write_text(json.dumps({'episodes':1200,'cache_sha256':sha256(out/'cache.json')}))
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()

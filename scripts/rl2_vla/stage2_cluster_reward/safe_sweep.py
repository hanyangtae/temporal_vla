#!/usr/bin/env python3
"""Full original SAFE grid; isolated selection/calibration; deterministic IID deployment.

Cache: cache.json contains episodes with episode_id/task/reset_id/success/split,
length (#chunks), chunk_lengths and policy_seed; features maps h=X__d=Y to .npy
or .npz (features key) [N,T,D] float32 arrays. features_sha256 records digests.
Train 576 trajectories (16 features*12 lr/reg*3 seeds), save metrics at1000/2000.
Select hyperparameters by mean three-seed selection AUROC; deploy seed0.
This preserves upstream numerical CP, NOT a finite-sample coverage claim.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import numpy as np

VARIANTS = ('0.0', '1.0', 'mean', 'concat-2')
ALPHAS = (.02,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.6,.7,.8,.9)
SPLITS = ('train','selection','calibration','holdout')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def atomic_json(path, value):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n'); os.replace(tmp,path)


def grid():
    return [dict(horizon=h,denoise=d,lr=lr,reg=reg,seed=seed)
            for h,d,lr,reg,seed in itertools.product(VARIANTS,VARIANTS,(1e-4,3e-4,1e-3),(.001,.01,.1,1.),(0,1,2))]


def feature_key(spec): return f"h={spec['horizon']}__d={spec['denoise']}"


def load_cache(path):
    path=Path(path); cache=json.loads(path.read_text()); seen=set(); groups={}
    for row in cache['episodes']:
        if row['episode_id'] in seen: raise ValueError('duplicate episode')
        seen.add(row['episode_id'])
        if row['split'] not in SPLITS: raise ValueError('unknown split')
        group=(row['task'],row['reset_id'])
        if groups.setdefault(group,row['split']) != row['split']: raise ValueError('reset group crosses splits')
        if int(row['length']) != len(row['chunk_lengths']) or not row['length']: raise ValueError('invalid chunk lengths')
        if any(not 1<=int(n)<=4 for n in row['chunk_lengths']): raise ValueError('invalid executed prefix')
        if row['success'] not in (True,False,0,1): raise ValueError('invalid success label')
    if set(r['split'] for r in cache['episodes']) != set(SPLITS): raise ValueError('all four splits required')
    cache['_sha256']=sha(path)
    return cache


def load_features(path, cache, key):
    p=Path(path).parent/cache['features'][key]
    expected=cache.get('features_sha256',cache.get('feature_sha256',{})).get(key)
    if not expected or sha(p)!=expected: raise ValueError(f'feature cache hash mismatch/missing: {key}')
    loaded=np.load(p,mmap_mode='r',allow_pickle=False)
    if isinstance(loaded,np.lib.npyio.NpzFile):
        x=loaded['features']; loaded.close()
    else: x=loaded
    if x.dtype!=np.float32 or x.ndim!=3 or len(x)!=len(cache['episodes']): raise ValueError('expected float32 [N,T,D]')
    for i,row in enumerate(cache['episodes']):
        if row['length']>x.shape[1] or not np.isfinite(x[i,:row['length']]).all(): raise ValueError('invalid features')
    return x


def model_config(spec):
    return dict(name='lstm',n_layers=1,hidden_dim=256,n_history_steps=-1,one_loss_per_seq=False,
                lr=spec['lr'],lambda_reg=spec['reg'],lambda_hard_heg=0.,hard_neg_margin=.1,hard_neg_beta=50.,
                cumsum=False,rmean=False,use_time_weighting=False,optimizer='adam',lr_step_size=300,lr_gamma=1.,
                weight_decay=.01,warmup_steps=0,lambda_success=1.,lambda_fail=1.,init_weight_scale=1.,
                grad_max_norm=None,dropout=0.,batch_size=512)


def make_model(safe_root,spec,dim,device):
    sys.path.insert(0,str(Path(safe_root).resolve()))
    from failure_prob.model.lstm import LstmModel
    from omegaconf import OmegaConf
    return LstmModel(OmegaConf.create({'model':model_config(spec)}),dim).to(device)


def task_min_steps(rows):
    result={}
    for row in rows:
        if row['split'] not in ('train','selection'): continue
        result[row['task']]=min(result.get(row['task'],row['length']),row['length'])
    return result


def sequence_auc(scores, rows, earliest=None):
    from sklearn.metrics import roc_auc_score
    labels=np.array([1-int(r['success']) for r in rows])
    if len(np.unique(labels))!=2: raise ValueError('selection requires both labels')
    maxima=np.array([s[:min(r['length'],earliest[r['task']]) if earliest else r['length']].max() for s,r in zip(scores,rows)])
    return float(roc_auc_score(labels,maxima))


def predict(model,x,device,batch_size=512):
    import torch
    model.eval(); out=[]
    with torch.no_grad():
        for start in range(0,len(x),batch_size):
            inp=torch.as_tensor(np.array(x[start:start+batch_size]),device=device)
            out.append(model({'features':inp}).squeeze(-1).cpu().numpy())
    return np.concatenate(out)


def worker(args):
    import torch
    cache=load_cache(args.cache); rows=cache['episodes']; output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    train_idx=np.array([i for i,r in enumerate(rows) if r['split']=='train'])
    select_idx=np.array([i for i,r in enumerate(rows) if r['split']=='selection'])
    labels=np.array([float(rows[i]['success']) for i in train_idx],dtype=np.float32)
    if len(np.unique(labels))!=2: raise ValueError('training requires both labels')
    weights=[len(labels)/(np.sum(labels==v)+1) for v in (0,1)]
    features=None; current_key=None; earliest=task_min_steps(rows)
    jobs=[(i,s) for i,s in enumerate(grid()) if i%args.shards==args.shard]
    if args.job is not None: jobs=[(args.job,grid()[args.job])]
    for job,spec in jobs:
        folder=output/f'job_{job:04d}'; folder.mkdir(exist_ok=True)
        contract=job_contract(spec,cache,args.epochs,args.batch_size)
        done=folder/'done.json'
        if done.exists():
            if json.loads(done.read_text())['contract']!=contract: raise ValueError('resume contract mismatch')
            continue
        lock=folder/'worker.lock'
        fd=os.open(lock,os.O_CREAT|os.O_WRONLY,0o600)
        import fcntl
        try: fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd); continue
        try:
            key=feature_key(spec)
            if key!=current_key:
                features=load_features(args.cache,cache,key); current_key=key
                tx=torch.as_tensor(np.array(features[train_idx]),device=args.device)
                masks=np.arange(features.shape[1])[None,:]<np.array([rows[i]['length'] for i in train_idx])[:,None]
                tm=torch.as_tensor(masks.astype(np.float32),device=args.device)
                ty=torch.as_tensor(labels,device=args.device)
            random.seed(spec['seed']); np.random.seed(spec['seed']); torch.manual_seed(spec['seed'])
            model=make_model(args.safe_root,spec,features.shape[-1],args.device)
            optimizer,scheduler=model.get_optimizer(); start_epoch=0; metrics=[]
            resume=folder/'resume.pt'
            if resume.exists():
                state=torch.load(resume,map_location=args.device,weights_only=False)
                if state['contract']!=contract: raise ValueError('resume contract mismatch')
                model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
                start_epoch=state['epoch']; metrics=state['metrics']; torch.set_rng_state(state['torch_rng'].cpu())
                if torch.cuda.is_available() and state.get('cuda_rng') is not None: torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
                del state
            began=time.monotonic()
            for epoch in range(start_epoch+1,max(args.epochs)+1):
                model.train(); order=torch.randperm(len(tx),device='cpu')
                losses=[]
                for start in range(0,len(order),args.batch_size):
                    ix=order[start:start+args.batch_size].to(args.device)
                    loss,_=model.forward_compute_loss({'features':tx[ix],'valid_masks':tm[ix],'success_labels':ty[ix]},weights)
                    reg,_=model.compute_regularization_loss(spec['reg']); total=loss+reg
                    if not torch.isfinite(total): raise FloatingPointError('nonfinite SAFE loss')
                    optimizer.zero_grad(); total.backward(); optimizer.step(); losses.append(float(total.detach()))
                scheduler.step()
                if epoch in args.epochs:
                    scores=predict(model,features[select_idx],args.device,args.batch_size)
                    auc=sequence_auc(scores,[rows[i] for i in select_idx],earliest)
                    metric=dict(epoch=epoch,selection_auc=auc,selection_endpoint_auc=sequence_auc(scores,[rows[i] for i in select_idx]),elapsed_seconds=time.monotonic()-began,loss=float(np.mean(losses)))
                    metrics.append(metric); atomic_json(folder/f'metric_{epoch}.json',dict(contract=contract,**metric))
                    # Seed0 is prespecified deployment; other seeds are selection replications.
                    if spec['seed']==0:
                        tmp=folder/f'model_{epoch}.tmp'; torch.save(model.state_dict(),tmp); os.replace(tmp,folder/f'model_{epoch}.ckpt')
                if epoch%args.save_every==0 or epoch in args.epochs:
                    state=dict(contract=contract,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,
                               metrics=metrics,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
                    tmp=folder/'resume.tmp'; torch.save(state,tmp); os.replace(tmp,resume); del state
                    print(json.dumps(dict(job=job,epoch=epoch,loss=float(np.mean(losses)),elapsed_seconds=time.monotonic()-began)),flush=True)
            atomic_json(done,dict(contract=contract,metrics=metrics)); resume.unlink(missing_ok=True)
            del model,optimizer,scheduler
        finally:
            os.close(fd)


def original_band(regression,residual,alpha):
    """Exact SAFE FunctionalPredictor Tfunc + one-sided linear np.quantile."""
    x=np.asarray(regression,dtype=np.float64); cal=np.asarray(residual,dtype=np.float64)
    if len(x)<1 or len(cal)<1: raise ValueError('empty CP regression/residual groups')
    center=x.mean(axis=0); deviations=np.abs(x-center); rank=int(np.ceil((len(x)+1)*(1-alpha)))
    subset=deviations if rank>len(x) else deviations[deviations.max(axis=1)<=np.sort(deviations.max(axis=1))[rank-1]]
    modulation=subset.max(axis=0)+1e-8
    width=np.quantile(((cal-center)/modulation).max(axis=1),1-alpha)
    return center+width*modulation


def expanded_scores(score,row,horizon=150):
    result=np.repeat(np.asarray(score)[:row['length']],row['chunk_lengths'])
    if not 0<len(result)<=horizon: raise ValueError('invalid episode horizon')
    return np.pad(result,(0,horizon-len(result)),mode='edge')


def top_alphas(values,k=3):
    ordered=sorted(values,key=lambda a:(-round(values[a],6),a)); cutoff=round(values[ordered[min(k,len(ordered))-1]],6)
    return [a for a in ordered if round(values[a],6)>=cutoff]


def calibration_indices(rows,seed=0):
    groups=sorted({r['reset_id'] for r in rows if r['success']})
    if len(groups)<4: raise ValueError('fewer than four successful reset groups for 30/70 CP split')
    np.random.default_rng(seed).shuffle(groups); cut=max(1,int(len(groups)*.3)); regression=set(groups[:cut])
    return ([i for i,r in enumerate(rows) if r['success'] and r['reset_id'] in regression],
            [i for i,r in enumerate(rows) if r['success'] and r['reset_id'] not in regression])


def balanced_accuracy(scores,rows,band,observed_only=False):
    truth=np.array([not r['success'] for r in rows])
    pred=np.array([np.any(s[:sum(r['chunk_lengths'])]>=band[:sum(r['chunk_lengths'])]) if observed_only else np.any(s>=band) for s,r in zip(scores,rows)])
    if len(np.unique(truth))<2: raise ValueError('alpha selection requires both outcome classes')
    return float((pred[truth].mean()+(~pred[~truth]).mean())/2)



def job_contract(spec,cache,epochs,batch_size):
    return dict(spec=spec,cache_sha256=cache['_sha256'],epochs=list(epochs),batch_size=batch_size,
                selection_metric='falert_early_roc_auc_train_selection_min_v1',task_min_steps=task_min_steps(cache['episodes']))


def checked_metrics(payload,spec,cache):
    expected=job_contract(spec,cache,[1000,2000],512)
    if payload.get('contract')!=expected: raise ValueError('stale sweep spec/epochs/batch/cache/selection contract')
    metrics=payload.get('metrics',[])
    if sorted(m.get('epoch',-1) for m in metrics)!=[1000,2000]: raise ValueError('missing/duplicate endpoint metrics')
    if any(not np.isfinite(m['selection_auc']) or not 0<=m['selection_auc']<=1 for m in metrics): raise ValueError('invalid selection AUC')
    return metrics

def finalize(args):
    import torch
    cache=load_cache(args.cache); root=Path(args.output); aggregates={}; specs=grid()
    for job,spec in enumerate(specs):
        done=root/f'job_{job:04d}'/'done.json'
        if not done.exists(): raise ValueError(f'full sweep incomplete: {done}')
        payload=json.loads(done.read_text())
        for metric in checked_metrics(payload,spec,cache):
            key=(spec['horizon'],spec['denoise'],spec['lr'],spec['reg'],metric['epoch'])
            aggregates.setdefault(key,[]).append((spec['seed'],metric['selection_auc'],job))
    for key,values in aggregates.items():
        if sorted(v[0] for v in values)!=[0,1,2]: raise ValueError('incomplete three-seed selection')
    best=sorted(aggregates,key=lambda k:(-np.mean([v[1] for v in aggregates[k]]),str(k)))[0]
    job=next(v[2] for v in aggregates[best] if v[0]==0); spec=specs[job]
    output=root/'selected'; output.mkdir(exist_ok=True)
    x=load_features(args.cache,cache,feature_key(spec)); rows=cache['episodes']
    model=make_model(args.safe_root,spec,x.shape[-1],args.device)
    model.load_state_dict(torch.load(root/f'job_{job:04d}'/f'model_{best[-1]}.ckpt',map_location=args.device,weights_only=True))
    # Never infer holdout during selection or calibration.
    permitted=[i for i,r in enumerate(rows) if r['split'] in ('selection','calibration')]
    raw=predict(model,x[permitted],args.device,args.batch_size)
    score_by_id={rows[i]['episode_id']:expanded_scores(score,rows[i]) for i,score in zip(permitted,raw)}
    bands={}; selected={}; diagnostics={}; per_seed={}; cp_groups={}
    for task in sorted({r['task'] for r in rows}):
        cal=[r for r in rows if r['task']==task and r['split']=='calibration']; sel=[r for r in rows if r['task']==task and r['split']=='selection']
        reg_idx,res_idx=calibration_indices(cal); cscore=np.stack([score_by_id[r['episode_id']] for r in cal]); sscore=np.stack([score_by_id[r['episode_id']] for r in sel])
        bands[task]={a:original_band(cscore[reg_idx],cscore[res_idx],a) for a in ALPHAS}
        ba={a:balanced_accuracy(sscore,sel,bands[task][a]) for a in ALPHAS}
        ranked=top_alphas(ba); selected[task]=ranked[0]; diagnostics[task]={'balanced_accuracy_padded_original':ba,'balanced_accuracy_actual_stop':{a:balanced_accuracy(sscore,sel,bands[task][a],True) for a in ALPHAS},'top3_including_ties':ranked}
        cp_groups[task]={'regression':[cal[i]['episode_id'] for i in reg_idx],'residual':[cal[i]['episode_id'] for i in res_idx]}
        per_seed[task]={}
        for seed in sorted({str(r.get('policy_seed','unknown')) for r in sel}):
            ix=[i for i,r in enumerate(sel) if str(r.get('policy_seed','unknown'))==seed]; sub=[sel[i] for i in ix]
            if len({r['success'] for r in sub})<2:
                per_seed[task][seed]={'unavailable':'only one outcome class'}; continue
            subcal=[r for r in cal if str(r.get('policy_seed','unknown'))==seed]
            try:
                sr,sc=calibration_indices(subcal)
            except ValueError as exc:
                per_seed[task][seed]={'unavailable':str(exc)}; continue
            subcs=np.stack([score_by_id[r['episode_id']] for r in subcal])
            subbands={a:original_band(subcs[sr],subcs[sc],a) for a in ALPHAS}
            subba={a:balanced_accuracy(sscore[ix],sub,subbands[a]) for a in ALPHAS}
            per_seed[task][seed]={'top3_including_ties':top_alphas(subba),'balanced_accuracy_padded_original':subba,'balanced_accuracy_actual_stop':{a:balanced_accuracy(sscore[ix],sub,subbands[a],True) for a in ALPHAS}}
    feature_contract='pi0_safe_original_grid_v1'
    config={'model':model_config(spec),'input_dim':x.shape[-1],'dataset':{'horizon_idx_rel':spec['horizon'],'diff_idx_rel':spec['denoise'],'dim_features':x.shape[-1]},
            'horizon_idx_rel':spec['horizon'],'diff_idx_rel':spec['denoise'],'cp_band_time_unit':'environment_step','selected_cp_alpha_by_task':selected,
            'feature_contract':feature_contract,'train':{'seed':0,'epochs':best[-1]},'cp_recipe':'original_Tfunc_quantile_edgepad_group_disjoint_30_70'}
    shutil.copyfile(root/f'job_{job:04d}'/f'model_{best[-1]}.ckpt',output/'model_final.ckpt')
    atomic_json(output/'config.yaml',config); np.save(output/'cp_bands_by_task.npy',bands,allow_pickle=True)
    atomic_json(output/'selection_report.json',{'selected_job':job,'selected_spec':spec,'epoch':best[-1],'mean_seed_auc':float(np.mean([v[1] for v in aggregates[best]])),
        'cache_sha256':cache['_sha256'],'selection_metric':'falert_early_roc_auc','task_min_steps':task_min_steps(rows),'alpha_diagnostics':diagnostics,'per_collection_seed_diagnostics':per_seed,'cp_groups':cp_groups,
        'departures':['reset-group-disjoint 60/10/10/20 splits','early-AUC task_min_step uses train+selection only, not calibration/holdout','CP30/70 partition by reset, not individual trajectory','deployment top1 task aggregate BA; upstream exports top3 lists','original np.quantile band: no finite-sample coverage assertion','original padded final-end BA ranking can include alarms after actual termination; actual-stop BA reported separately']})
    for split,name in [('train','training_manifest'),('selection','selection_manifest'),('calibration','calibration_manifest')]:
        atomic_json(output/f'{name}.json',{'cache_sha256':cache['_sha256'],'episodes':[r for r in rows if r['split']==split]})
    artifacts=[{'path':p.name,'sha256':sha(p)} for p in sorted(output.iterdir()) if p.is_file() and p.name!='provenance.json']
    provenance={'owner':'temporal_vla','feature_contract':feature_contract,'cache_sha256':cache['_sha256'],'artifacts':artifacts}
    for name in ('training_manifest','calibration_manifest'): provenance[name]={'path':name+'.json','sha256':sha(output/(name+'.json'))}
    atomic_json(output/'provenance.json',provenance)
    print(output,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--cache',type=Path,required=True); p.add_argument('--safe-root',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=('worker','finalize','grid'),default='worker'); p.add_argument('--device',default='cuda'); p.add_argument('--shards',type=int,default=1); p.add_argument('--shard',type=int,default=0)
    p.add_argument('--job',type=int); p.add_argument('--epochs',type=int,nargs='+',default=[1000,2000]); p.add_argument('--batch-size',type=int,default=512); p.add_argument('--save-every',type=int,default=100)
    args=p.parse_args()
    if args.shards<1 or not 0<=args.shard<args.shards or args.batch_size<1 or min(args.epochs)<1 or args.save_every<1 or (args.job is not None and not 0<=args.job<len(grid())): p.error('invalid worker settings')
    if args.mode=='grid': print(json.dumps(grid(),indent=2))
    elif args.mode=='worker': worker(args)
    else: finalize(args)

if __name__=='__main__': main()

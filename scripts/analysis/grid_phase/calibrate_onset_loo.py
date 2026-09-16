"""True model-refit LOO scores on the training jitters; target jitter stays blind.

Frozen upstream k8 assignments are retained. Each fold recomputes the scaler,
length caps and LSTM from 39 episodes. The held episode is scored at full length.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import failure_detector_sim as det
from compare_failure_onset import prefix_count


def atomic_json(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj)+'\n');tmp.replace(path)


def training_data(pool,mode,annotations):
    cap=None
    if mode in ('phase_ck8','phase_gt'):
        caps=det.phase_dwell_caps(pool)
        fit=[det.truncate_episode(e,mode.replace('_','-'),det.rollout_cap(pool),caps) for e in pool]
        fit=[e for e in fit if e is not None]
    else:
        if mode=='min_success_prefix':cap=min(e.T for e in pool if e.succ==1)
        else:
            times=[annotations[e.ep_id]['event_env_step'] for e in pool
                   if annotations[e.ep_id]['event_env_step'] is not None]
            if not times:raise ValueError('no training event after fold exclusion')
            cap=prefix_count(min(times),5)
        caps={}
        fit=[det.Episode(e.task,e.ep_id,e.scene,e.noise,e.succ,
             np.ascontiguousarray(e.X[:cap]),np.ascontiguousarray(e.phase[:cap]),jitter=e.jitter) for e in pool]
    assert len({e.y for e in fit})==2
    return fit,dict(prefix_records=cap,phase_caps=caps,used_records={e.ep_id:e.T for e in fit})


def fit_model(pool,mode,annotations):
    fit,audit=training_data(pool,mode,annotations)
    mu,sd=det.standardizer(fit)
    seq=[(det.apply_std(e,mu,sd),e.y) for e in fit]
    model=det.train_detector('lstm',seq,seq[0][0].shape[1],25,.001,256,.01,1.,8,0,verbose=False)
    return model,mu,sd,audit


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--threads',type=int,default=8);ap.add_argument('--cell')
    ap.add_argument('--modes',nargs='+',default=['phase_ck8','first_event_prefix','phase_gt','min_success_prefix'])
    ap.add_argument('--alphas',nargs='+',type=float,default=[.1,.05])
    a=ap.parse_args();torch.set_num_threads(a.threads)
    cells=json.loads(a.manifest.read_text());digest=hashlib.sha256(a.manifest.read_bytes()).hexdigest()
    a.out.mkdir(parents=True,exist_ok=True)
    for c in cells:
        if a.cell and c['cell_id']!=a.cell:continue
        eps,_=det.load_shard_episodes(Path(c['prepared']),12,-1,'all')
        ann={e['ep_id']:e for e in c['episodes']}
        tr=[e for e in eps if e.jitter!=c['target']];test=[e for e in eps if e.jitter==c['target']]
        assert len(tr)==40 and len(test)==10
        assert not {e.ep_id for e in tr}&{e.ep_id for e in test}
        with np.load(c['prepared'],allow_pickle=False) as z:
            gt={}
            for e in tr:
                ix=np.flatnonzero(z['ep_id']==e.ep_id);ix=ix[np.argsort(z['rec_idx'][ix],kind='stable')]
                assert np.array_equal(z['rec_idx'][ix],np.arange(e.T))
                gt[e.ep_id]=np.ascontiguousarray(z['gt_phase_code'][ix])
        for mode in a.modes:
            if mode=='first_event_prefix' and not c['prefix_eligible']:continue
            root=a.out/c['cell_id']/mode;root.mkdir(parents=True,exist_ok=True)
            pool=tr if mode!='phase_gt' else [det.Episode(e.task,e.ep_id,e.scene,e.noise,e.succ,e.X,gt[e.ep_id],jitter=e.jitter) for e in tr]
            fold_scores=[]
            for held in pool:
                path=root/f'fold_{held.ep_id}.json'
                if path.exists():
                    r=json.loads(path.read_text());assert r['manifest_sha256']==digest
                else:
                    train=[e for e in pool if e.ep_id!=held.ep_id]
                    model,mu,sd,audit=fit_model(train,mode,ann)
                    score=det.score_seq(model,det.apply_std(held,mu,sd))
                    r=dict(manifest_sha256=digest,held_ep_id=held.ep_id,held_success=held.succ,
                           trained_ep_ids=[e.ep_id for e in train],audit=audit,scores=score.tolist(),
                           score_length=held.T,held_jitter=held.jitter,held_noise=held.noise)
                    atomic_json(path,r)
                assert len(r['trained_ep_ids'])==39 and held.ep_id not in r['trained_ep_ids']
                assert not set(r['trained_ep_ids'])&{e.ep_id for e in test}
                assert len(r['scores'])==held.T
                fold_scores.append(r)
                print('[fold]',c['cell_id'],mode,len(fold_scores),'/40',flush=True)
            # Full-length OOF successes calibrate an empirical band. This is not
            # exact split conformal coverage for the different all-40 final model.
            model,mu,sd,audit=fit_model(pool,mode,ann)
            test_scores={e.ep_id:det.score_seq(model,det.apply_std(e,mu,sd)) for e in test}
            for alpha in a.alphas:
                band=det.loo_cp_band([np.asarray(r['scores']) for r in fold_scores if r['held_success']==1],alpha,0)
                assert band is not None
                rows=[]
                for e in test:
                    scores=test_scores[e.ep_id];fire=det.fire_step(scores,band['delta'])
                    event=ann[e.ep_id]['event_env_step'];fire_env=None if fire is None else fire*5
                    rows.append(dict(ep_id=e.ep_id,noise=e.noise,success=e.succ,event_env_step=event,
                                     fire_record=fire,fire_env_step=fire_env,
                                     lag_env_steps=None if event is None or fire is None else fire_env-event,
                                     scores=scores.tolist()))
                atomic_json(root/f'result_alpha{alpha:.2f}.json',dict(manifest_sha256=digest,cell=c['cell_id'],mode=mode,
                    alpha=alpha,train_ep_ids=[e.ep_id for e in pool],test_ep_ids=[e.ep_id for e in test],
                    band_length=int(band['L']),threshold=band['delta'].tolist(),final_fit=audit,episodes=rows,
                    calibration='model-refit LOO full sequences; empirical band; no exact final-model coverage claim'))
            print('[done]',c['cell_id'],mode,flush=True)
    atomic_json(a.out/('DONE_'+(a.cell or 'all')+'.json'),dict(complete=True,manifest_sha256=digest))


if __name__=='__main__':main()

"""Compare existing k8 dwell training with a train-only first-event prefix.

Inputs are a frozen, signature-joined manifest. No test labels enter fitting,
standardization, truncation or empirical success-band calibration.
"""
import argparse
import hashlib
import json
from pathlib import Path


def prefix_count(event_env_step, action_steps):
    # Inference r sees state after r*action_steps executed actions. Exclude onset.
    if event_env_step <= 0 or action_steps <= 0:
        raise ValueError('positive event step/action period required')
    return (event_env_step + action_steps - 1) // action_steps


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--threads', type=int, default=8)
    a = ap.parse_args()
    import numpy as np
    import torch
    import failure_detector_sim as det
    torch.set_num_threads(a.threads)
    cells = json.loads(a.manifest.read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    for c in cells:
        root = a.out / c['cell_id']
        root.mkdir(exist_ok=True)
        eps, spec = det.load_shard_episodes(Path(c['prepared']), 12, -1, 'all')
        tr = [e for e in eps if e.jitter != c['target']]
        te = [e for e in eps if e.jitter == c['target']]
        assert len(tr) == 40 and len(te) == 10
        assert len({e.y for e in tr}) == 2
        assert not ({e.ep_id for e in tr} & {e.ep_id for e in te})
        annotations = {r['ep_id']: r for r in c['episodes']}
        for e in eps:
            r = annotations[e.ep_id]
            assert (e.jitter,e.noise,e.succ,e.T)==(r['jitter'],r['noise'],r['succ'],r['ep_len'])
        onsets = [annotations[e.ep_id]['event_env_step'] for e in tr
                  if annotations[e.ep_id]['event_env_step'] is not None]
        cutoff = min(onsets) if onsets else None
        full = []
        for mode in ['phase_ck8', 'first_event_prefix']:
            dst = root / (mode + '.json')
            if dst.exists():
                old=json.loads(dst.read_text())
                assert old['manifest_sha256']==hashlib.sha256(a.manifest.read_bytes()).hexdigest()
                print('[reuse]',dst,flush=True)
                continue
            if mode == 'first_event_prefix' and (not c['prefix_eligible'] or cutoff is None):
                print('[skip prefix]',c['cell_id'],c['incomplete_reason'],flush=True)
                continue
            W=det.rollout_cap(tr);caps=det.phase_dwell_caps(tr)
            if mode == 'phase_ck8':
                fit=[det.truncate_episode(e,'phase-ck8',W,caps) for e in tr]
                assert all(e is not None for e in fit)
            else:
                n=prefix_count(cutoff,c['action_steps'])
                fit=[det.Episode(e.task,e.ep_id,e.scene,e.noise,e.succ,
                     np.ascontiguousarray(e.X[:n]),np.ascontiguousarray(e.phase[:n]),jitter=e.jitter) for e in tr]
                assert all(0<e.T<=n for e in fit)
            mu,sd=det.standardizer(fit)
            seq=[(det.apply_std(e,mu,sd),e.y) for e in fit]
            print('[fit]',c['cell_id'],mode,'records',sum(e.T for e in fit),'cutoff',cutoff,flush=True)
            model=det.train_detector('lstm',seq,seq[0][0].shape[1],25,.001,256,.01,1.,8,0,verbose=False)
            calib=[det.score_seq(model,det.apply_std(e,mu,sd)) for e in fit if e.succ==1]
            band=det.loo_cp_band(calib,.1,0)
            assert band is not None
            results=[]
            for e in te:
                sc=det.score_seq(model,det.apply_std(e,mu,sd));ft=det.fire_step(sc,band['delta'])
                event=annotations[e.ep_id]['event_env_step']
                fire=None if ft is None else ft*c['action_steps']
                results.append(dict(ep_id=e.ep_id,noise=e.noise,jitter=e.jitter,success=e.succ,
                    event_env_step=event,fire_record=ft,fire_env_step=fire,
                    lag_env_steps=None if event is None or fire is None else fire-event,
                    T=e.T,scores=sc.tolist(),threshold=band['delta'].tolist()))
            out=dict(cell=c['cell_id'],mode=mode,manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
                     train_ep_ids=[e.ep_id for e in tr],test_ep_ids=[e.ep_id for e in te],
                     training_records={str(e.ep_id):e.T for e in fit},cutoff_env_step=cutoff,
                     fit_outcome_labels='original rollout outcome, not manual event label',
                     calibration='training successes; empirical LOO band alpha=.1; last threshold held beyond band',
                     band_length=int(band['L']),episodes=results)
            tmp=dst.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2)+'\n');tmp.replace(dst)
            # Newly trained weights only; no pickle checkpoint is loaded.
            torch.save(dict(state_dict=model.state_dict(),std_mean=torch.from_numpy(mu),
                            std_std=torch.from_numpy(sd),delta=torch.from_numpy(band['delta']),
                            hidden=256,input_dim=seq[0][0].shape[1]),root/(mode+'.pt'))
            print('[done]',dst,flush=True)
    (a.out/'DONE.json').write_text(json.dumps({'complete':True,'cells':len(cells)})+'\n')


if __name__ == '__main__':
    main()

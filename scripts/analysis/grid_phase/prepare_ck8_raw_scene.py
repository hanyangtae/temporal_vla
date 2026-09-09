"""Prepare a new scene beside its raw archive, without fitting GT phases.

Match the existing segA -> prepare_ck8_scene numerical path: float32 mean
over all 49 tokens, float16 storage roundtrip, then production k8 assignment.
Keep one raw episode in memory. Verify the frozen index hash before unpickling.
"""
import argparse
import csv
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.failure_online.cluster_phase import ClusterPhaseAssigner


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grid-root', type=Path, required=True)
    p.add_argument('--index-tsv', type=Path, required=True)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--slug', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--wait-for-archive', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(4)
    rows = list(csv.DictReader(a.index_tsv.open(), delimiter='\t'))
    assert len(rows) == 50
    assert len({(r['plan_id'], r['machine'], r['grid_instruction'], r['scene_idx']) for r in rows}) == 1
    assert {(int(r['jitter_idx']), int(r['noise_idx'])) for r in rows} == {(j,n) for j in range(5) for n in range(10)}
    rows.sort(key=lambda r: (int(r['noise_idx']), int(r['jitter_idx'])))
    assigner = ClusterPhaseAssigner.from_bundle(a.bundle, task=a.slug, device='cpu')
    assert assigner.centers.shape[0] == 8
    assert int(assigner.feature.get('layer',12)) == 12
    assert int(assigner.feature.get('denoise_index',3)) == 3
    assert assigner.feature.get('seg','all') == 'all'
    fields = {k: [] for k in ('ep_id','scene','noise','jitter','rec_idx','succ','ep_len','phase_code')}
    features = []
    sources = []
    assert not a.out.exists(), a.out
    for e,r in enumerate(rows):
        path = a.grid_root/r['rel_path']/'rollout.pkl'
        deadline = time.monotonic() + 12 * 3600
        while True:
            digest = sha256(path) if path.is_file() else None
            if digest == r['pkl_sha256']:
                break
            if not a.wait_for_archive or time.monotonic() >= deadline:
                raise ValueError(f'{path}: missing or incomplete archive hash')
            print(f'[waiting archive] {path}',flush=True)
            time.sleep(30)
        assert digest == r['pkl_sha256'], (path,'sha256 mismatch')
        meta = json.loads(path.with_name('meta.json').read_text())
        for k in ('env_seed','inference_seed','success','scene_idx','jitter_idx','noise_idx','layout_id','style_id'):
            assert int(meta[k]) == int(r[k]), (path,k)
        for k in ('plan_id','machine','grid_instruction','ckpt','env_name'):
            assert meta[k] == r[k], (path,k)
        with path.open('rb') as f:
            d = pickle.load(f)
        layers = list(d['capture_layers'])
        assert layers == [0,2,4,8,10,12,15]
        assert int(d['episode_success']) == int(r['success'])
        hs = d['hidden_states']; assert len(hs) > 0
        li = assigner.resolve_layer_index(layers)
        out = []
        for i,h in enumerate(hs):
            assert tuple(h.shape) == (7,4,49,1536), (path,i,h.shape)
            selected = h[li,3]
            if hasattr(selected,'detach'):
                selected = selected.detach().cpu().float().numpy()
            else:
                selected = np.asarray(selected,dtype=np.float32)
            assert np.isfinite(selected).all(), (path,i,'nonfinite')
            out.append(selected.mean(axis=0,dtype=np.float32).astype(np.float16).astype(np.float32))
            hs[i] = None
        x = np.asarray(out,dtype=np.float32)
        assert np.isfinite(x).all()
        phases = np.asarray([z['idx'] for z in assigner.assign_batch(x)],dtype=np.int16)
        n = len(x)
        features.append(x)
        for k,v in dict(ep_id=e,scene=int(r['scene_idx']),noise=int(r['noise_idx']),
                        jitter=int(r['jitter_idx']),succ=int(r['success']),ep_len=n).items():
            fields[k].append(np.full(n,v,dtype=np.int32))
        fields['rec_idx'].append(np.arange(n,dtype=np.int32))
        fields['phase_code'].append(phases)
        sources.append(dict(rel_path=r['rel_path'],sha256=digest,records=n))
        del d,hs
        print(f'[prepare] {a.slug} {e+1}/50 {r["rel_path"]} records={n}',flush=True)
    x = np.concatenate(features)
    X = np.zeros((len(x),1,4,1,1536),dtype=np.float32)
    X[:,0,3,0] = x
    meta = dict(instruction=rows[0]['grid_instruction'],plan_id=[rows[0]['plan_id']],
                machine=rows[0]['machine'],capture_layers=[12],segment_names=['all'],
                n_episodes=50,denoise_k=4,dim=1536,jitter_axis=True,
                phase_codebook={f'c{i}':i for i in range(8)},
                phase_source=dict(kind='ck8',bundle=a.bundle.name,bundle_sha256=sha256(a.bundle),
                    token_read='all49',feature_layer=12,denoise_index=3,
                    numerical_path='float32_token_mean_float16_roundtrip',gt_phase_used=False),
                sources=sources)
    payload = {k:np.concatenate(v) for k,v in fields.items()}
    payload.update(X=X,meta_json=np.asarray(json.dumps(meta)))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    tmp = a.out.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez(f,**payload)
    tmp.replace(a.out)
    print(f'[prepared] {a.out} records={len(x)}',flush=True)


if __name__ == '__main__':
    main()

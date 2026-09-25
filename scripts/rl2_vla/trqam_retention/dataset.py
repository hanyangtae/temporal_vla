"""Select whole successful train episodes using audited cache row ordering."""
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'stage2_cluster_reward'))
from data import sha256


def load_success_cache(root):
    root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text())
    cache=json.loads((root/'cache.json').read_text())
    digest=sha256(root/'manifest.json')
    if cache['manifest_sha256'] != digest:
        raise ValueError('manifest mismatch')
    meta=cache['qam_transitions']['cluster_potential']
    if meta['manifest_sha256'] != digest or meta['discount'] != .99 or meta['shaping_scale'] != .1:
        raise ValueError('reward contract mismatch')
    if sha256(root/'clusters.json') != meta['cluster_sha256']:
        raise ValueError('frozen potential mismatch')
    path=root/meta['path']
    if sha256(path) != meta['sha256']:
        raise ValueError('transition hash mismatch')
    with np.load(path,allow_pickle=False) as f:
        data={k:f[k] for k in f.files}
    rows={r['episode_id']:r for r in manifest['episodes']}
    lengths={r['episode_id']:r for r in cache['episodes']}
    if len(rows)!=len(manifest['episodes']) or len(lengths)!=len(cache['episodes']):
        raise ValueError('duplicate episodes')
    train=[r['episode_id'] for r in manifest['episodes'] if r['split']=='train']
    if train != meta['episode_ids']:
        raise ValueError('transition episode order mismatch')
    selected=[]; ids=[]; cursor=0
    for eid in train:
        row=rows[eid]; cr=lengths[eid]; n=cr['length']
        if row['success']!=cr['success'] or row['sha256']!=cr['source_sha256'] or cr['split']!='train':
            raise ValueError('cache episode mismatch')
        if len(cr['chunk_lengths']) != n or not np.array_equal(data['actual_steps'][cursor:cursor+n],cr['chunk_lengths']):
            raise ValueError('cache episode boundary mismatch')
        if row['success']:
            selected.extend(range(cursor,cursor+n)); ids.append(eid)
        cursor+=n
    if cursor!=meta['transitions'] or any(len(v)!=cursor for v in data.values()):
        raise ValueError('cache row count mismatch')
    if not selected or not all(np.isfinite(v).all() for v in data.values()):
        raise ValueError('empty or nonfinite transitions')
    provenance=dict(manifest_sha256=digest, transitions_sha256=meta['sha256'],
                    cluster_sha256=meta['cluster_sha256'], episode_ids=ids,
                    episodes=len(ids), transitions=len(selected),
                    selection='train AND success; fixed potential fitted on original train successes+failures')
    return {k:v[selected] for k,v in data.items()},provenance

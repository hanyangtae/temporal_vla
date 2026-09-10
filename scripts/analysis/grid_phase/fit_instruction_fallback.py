"""Fit scene-local phase-agnostic plain setM without length adjustment.

Successes of ALL evaluation target cells are excluded; target failures are retained.
All eligible records contribute equally; no cap, truncation or episode reweighting.
No raw pickle loading and no detector refit.
"""
import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
import numpy as np



def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    cells = json.loads(a.manifest.read_text())['cells']
    assert len(cells) == 23
    grouped = defaultdict(list)
    for r in cells:
        grouped[(r['instruction'], r['scene'], r['target'])].append(r)
    assert not a.out.exists(), 'immutable fit output already exists'
    a.out.mkdir(parents=True)
    ready = {'complete': False, 'instructions': [], 'failed': None}
    def publish():
        p = a.out/'ready.tmp'; p.write_text(json.dumps(ready, indent=2)+'\n');p.replace(a.out/'ready.json')
    publish()
    try:
        for key, group in sorted(grouped.items()):
            instruction = key[0]
            vectors, epis, labels, jitters, provenance = [], [], [], [], []
            for idx, r in enumerate(group):
                path = Path(r['prepared'])
                with np.load(path, allow_pickle=False) as z:
                    meta = json.loads(str(z['meta_json']))
                    assert meta['phase_source']['kind'] == 'ck8'
                    X = z['X'][:, list(meta['capture_layers']).index(12), 3,
                               list(meta['segment_names']).index('all'), :].astype(np.float32)
                    ep, c, su = z['ep_id'], z['phase_code'], z['succ']
                    j = z['jitter'] if 'jitter' in z else z['jitter_idx']
                    rec = z['rec_idx']
                    assert set(z['scene'].tolist()) == {r['scene']}
                assert np.isfinite(X).all() and len(np.unique(ep)) == 50
                order = np.lexsort((rec, ep))
                X, ep, c, su, j = (v[order] for v in (X, ep, c, su, j))
                starts = [np.flatnonzero(ep == e)[0] for e in np.unique(ep)]
                assert sum(su[i] == 1 and j[i] == r['target'] for i in starts) == r['target_success']
                allow = (j != r['target']) | (su == 0)
                keep = allow.copy()
                caps, contrib = {}, []
                assert not np.any(keep & (j == r['target']) & (su == 1))
                vectors.append(X[keep]); labels.append(su[keep]); jitters.append(j[keep])
                # IDs are local to prepared shards; remap by scene BEFORE pooling.
                ids = np.unique(ep); mapped = np.searchsorted(ids, ep) + idx*50
                epis.append(mapped[keep])
                provenance.append(dict(r, prepared_sha256=sha(path), caps=caps,
                    contributions=contrib, excluded_target_success=r['target_success'],
                    included_episode_ids=np.unique(ep[keep]).tolist()))
            X, ep, su, j = (np.concatenate(v) for v in (vectors, epis, labels, jitters))
            assert min(np.sum(su == 1), np.sum(su == 0)) >= 20
            mu_s = X[su == 1].mean(0, dtype=np.float64)
            mu_f = X[su == 0].mean(0, dtype=np.float64)
            delta = mu_s - mu_f
            norm = np.linalg.norm(delta)
            assert np.isfinite(norm) and norm > 1e-12
            v = delta / norm
            setpoint = float(mu_s @ v)
            metadata = dict(op='setpoint', setm_form='plain', layer=12, denoise=3,
                token_read='all_49', phase='instruction', instruction=instruction,
                fit_scope='scene_local_k8_partition_removed', length_control='none',
                mean_weighting='equal_record_within_outcome', sources=provenance,
                n_success_episodes=int(len(np.unique(ep[su == 1]))), n_failure_episodes=int(len(np.unique(ep[su == 0]))),
                beta=.8, target_success_exclusion='all_evaluated_target_cells',
                phase_source='scene_all_records_no_phase_partition')
            for r in group:
                for suffix, mode in [('instruction_plain','instruction_only'), ('cluster_instruction_plain','cluster_instruction_fallback')]:
                    base = a.out/'outputs/steer/online_pipe_v4_pilot'/f'instr_setm_v6_ck8dwell_ck8_{suffix}'/r['artifact_slug']/f's{r["scene"]}'/f'j{r["target"]}'
                    dest = base/'instruction/dit_L12';dest.mkdir(parents=True)
                    np.savez_compressed(dest/'conceptors.npz', alpha0_v_steer=v.astype(np.float32), alpha0_s=np.float32(setpoint))
                    (dest/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
                    originals = {}
                    if mode == 'cluster_instruction_fallback':
                        source = Path(r['cluster_operator'])
                        for c in sorted(source.glob('c[0-7]')):
                            if (c/'dit_L12/conceptors.npz').is_file():
                                shutil.copytree(c, base/c.name)
                                originals[c.name] = sha(c/'dit_L12/conceptors.npz')
                    routing = dict(version='instruction_setm_routing_v1', mode=mode,
                        instruction=instruction, instruction_npz_sha256=sha(dest/'conceptors.npz'),
                        cluster_npz_sha256=originals, seed_policy='same_first_pass_noise')
                    (base/'instruction_routing.json').write_text(json.dumps(routing, indent=2)+'\n')
            ready['instructions'].append(list(key));publish()
        ready['complete'] = True;publish()
    except Exception as e:
        ready['failed'] = repr(e);publish();raise

if __name__ == '__main__': main()

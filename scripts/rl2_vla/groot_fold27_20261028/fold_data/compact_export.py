"""Run ON 승준 (CPU, numpy): full extract_scene dir -> compact dir without last_block_d0 (MIL / TRQAM CPU inputs only).

Per episode (episode-by-episode, resumable, write-once): every array except last_block_d0 is copied byte-identically into
a new npz; the record keeps episode_id / pkl_sha256 and adds source_npz / source_npz_sha256 so the compact file is traceable
to the full extraction (which stays on 승준 for GPU latent reconstruction). contract.json of the compact dir records
source_contract_sha256 = sha256(full contract.json) and the dropped keys.

    python compact_export.py --src <full extract dir> --out <compact dir> [--glob '*.json']
"""
import argparse, hashlib, json, os, sys
from pathlib import Path

import numpy as np

DROP = ('last_block_d0',)
KEEP_REQUIRED = ('tokmean_l12d3', 'actions_model12', 'success', 'n_inferences', 'env_steps', 'first_success_step',
                 'max_episode_steps', 'env_step_success')


class CompactError(RuntimeError):
    pass


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def export(src, out, pattern='*.json', log=print):
    src_contract = src / 'contract.json'
    contract = {'kind': 'fold27_compact_v1', 'source_contract_sha256': sha(src_contract),
                'source_contract': json.loads(src_contract.read_text()), 'dropped': list(DROP),
                'exporter_sha256': sha(Path(__file__))}
    out.mkdir(parents=True, exist_ok=True)
    cp = out / 'contract.json'
    if cp.exists() and json.loads(cp.read_text()) != contract:
        raise CompactError('compact dir contract differs; use a new --out')
    cp.write_text(json.dumps(contract, indent=1))
    n = 0
    for jp in sorted(src.glob(pattern)):
        if jp.name in ('contract.json', 'COMPLETE.json'):
            continue
        rec = json.loads(jp.read_text())
        oj = out / jp.name
        if oj.exists():
            o = json.loads(oj.read_text())
            if o['episode_id'] != rec['episode_id'] or o['source_npz_sha256'] != rec['npz_sha256'] or sha(out / o['npz']) != o['npz_sha256']:
                raise CompactError(f'committed compact record inconsistent {rec["episode_id"]}')
            n += 1
            continue
        if sha(src / rec['npz']) != rec['npz_sha256']:
            raise CompactError(f'source npz sha {rec["episode_id"]}')
        with np.load(src / rec['npz']) as z:
            missing = [k for k in KEEP_REQUIRED if k not in z.files]
            if missing:
                raise CompactError(f'source lacks {missing} ({rec["episode_id"]}); re-extract with the v2 extractor')
            arrays = {k: z[k] for k in z.files if k not in DROP}
        dst = out / rec['npz']
        tmp = out / f'.{rec["npz"]}.tmp.npz'
        np.savez(tmp, **arrays)
        os.replace(tmp, dst)
        new = {k: v for k, v in rec.items() if k not in ('npz_sha256', 'seconds')}
        new.update(npz_sha256=sha(dst), source_npz=rec['npz'], source_npz_sha256=rec['npz_sha256'])
        with open(oj, 'x') as f:
            f.write(json.dumps(new, indent=1))
        n += 1
        log(json.dumps({'compact': n, 'episode': rec['episode_id']}))
    return n


def verify_equal(src_npz, compact_npz):
    """Every kept array is identical to the source array (used by tests and spot checks)."""
    with np.load(src_npz) as a, np.load(compact_npz) as b:
        if set(b.files) != set(a.files) - set(DROP):
            return False
        return all(np.array_equal(a[k], b[k]) and a[k].dtype == b[k].dtype for k in b.files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--glob', default='*.json')
    a = ap.parse_args()
    print(json.dumps({'compact_episodes': export(a.src, a.out, a.glob)}))


if __name__ == '__main__':
    try:
        main()
    except CompactError as e:
        print('COMPACT_ERROR', e, file=sys.stderr); sys.exit(2)

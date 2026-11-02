"""Completeness audit of a rollout_latents_v2 output dir against its episode dir, by EXPLICIT episode identity.
Shared by rollout_latents_v2.py (end of conversion) and finalize_latents_v2.py (in-place audit of an existing dir).

Fix for the 61a66541 run on srv50: the converter counted every *.json except contract/COMPLETE/PARITY_SMOKE_V2, so
DIAG_V2.json made 201 != 200 and no COMPLETE.json was written. Here a record is only an out json whose NAME is an
expected episode record name; known non-record files are listed explicitly; anything else is reported as unexpected.

Checks (all must hold for ok):
  expected set   = episode records in the episode dir (contract.json / COMPLETE.json excluded); episode_ids unique;
                   == --expect-records if given
  per expected   out/<name> exists; episode_id, pkl_sha256 match; source_npz_sha256 == episode npz_sha256;
                   npz exists, sha == npz_sha256; n == n_inferences; token_contract == contract; latents float32
                   (n,1024) finite
  no duplicates  no two out records with the same episode_id
  no unexpected  every other out json is in NON_RECORD; every out *.npz is referenced by exactly one record;
                 leftover *.tmp.npz reported (never deleted)
"""
import hashlib, json
from pathlib import Path

import numpy as np

NON_RECORD = {'contract.json', 'COMPLETE.json', 'PARITY_SMOKE_V2.json', 'DIAG_V2.json', 'COMPLETE_AUDIT_V2.json',
              'PARITY_SMOKE.json'}
EPISODE_NON_RECORD = {'contract.json', 'COMPLETE.json'}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def expected_records(episodes_dir):
    recs = {}
    for jp in sorted(Path(episodes_dir).glob('*.json')):
        if jp.name in EPISODE_NON_RECORD:
            continue
        recs[jp.name] = json.loads(jp.read_text())
    return recs


def audit(episodes_dir, out_dir, token_contract, expect=None, check_arrays=True):
    episodes_dir, out_dir = Path(episodes_dir), Path(out_dir)
    exp = expected_records(episodes_dir)
    problems = {'missing': [], 'mismatch': [], 'duplicate_episode_id': [], 'unexpected_json': [], 'unreferenced_npz': [],
                'tmp_leftover': [], 'episode_set': []}
    ids = [r['episode_id'] for r in exp.values()]
    if len(set(ids)) != len(ids):
        problems['episode_set'].append('duplicate episode_id among episode records')
    if expect is not None and len(exp) != expect:
        problems['episode_set'].append(f'{len(exp)} episode records != expected {expect}')
    seen_ids, referenced, ok_records = {}, set(), 0
    for name, rec in sorted(exp.items()):
        p = out_dir / name
        if not p.exists():
            problems['missing'].append(name); continue
        lr = json.loads(p.read_text())
        why = []
        if lr.get('episode_id') != rec['episode_id']:
            why.append('episode_id')
        if lr.get('pkl_sha256') != rec.get('pkl_sha256'):
            why.append('pkl_sha256')
        if lr.get('source_npz_sha256') != rec.get('npz_sha256'):
            why.append('source_npz_sha256')
        if lr.get('token_contract') != token_contract:
            why.append(f'token_contract {lr.get("token_contract")}')
        npz = out_dir / str(lr.get('npz'))
        if not npz.exists():
            why.append('npz missing')
        else:
            referenced.add(npz.name)
            if sha(npz) != lr.get('npz_sha256'):
                why.append('npz sha')
            elif check_arrays:
                with np.load(npz) as z:
                    lat = z['latents']
                if lat.dtype != np.float32 or lat.shape != (rec['n_inferences'], 1024) or not np.isfinite(lat).all():
                    why.append(f'latents {lat.dtype} {lat.shape}')
        if lr.get('n') != rec.get('n_inferences'):
            why.append('n')
        seen_ids.setdefault(lr.get('episode_id'), []).append(name)
        if why:
            problems['mismatch'].append({'record': name, 'fields': why})
        else:
            ok_records += 1
    for eid, names in seen_ids.items():
        if len(names) > 1:
            problems['duplicate_episode_id'].append({'episode_id': eid, 'records': names})
    for jp in sorted(out_dir.glob('*.json')):
        if jp.name not in exp and jp.name not in NON_RECORD:
            problems['unexpected_json'].append(jp.name)
    for z in sorted(out_dir.glob('*.npz')):
        if z.name.endswith('.tmp.npz'):
            problems['tmp_leftover'].append(z.name)
        elif z.name not in referenced:
            problems['unreferenced_npz'].append(z.name)
    blocking = {k: v for k, v in problems.items() if v and k != 'tmp_leftover'}
    return {'ok': not blocking and ok_records == len(exp), 'expected': len(exp), 'verified_records': ok_records,
            'non_record_present': sorted(n for n in NON_RECORD if (out_dir / n).exists()), 'problems': problems}

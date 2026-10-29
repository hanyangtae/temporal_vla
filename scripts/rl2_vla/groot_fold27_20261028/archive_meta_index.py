"""Run ON 승준 (stdlib, read-only): archive_coords.tsv -> archive_index.tsv for folds.py --archive-index.

Schema (2026-10-29 remote check by main session):
  store index groot/n15/index/rollouts.tsv: coordinate is `rel_path`
      (<plan>/<machine>/<instruction key, may contain '/'>/s<sid>/j<jid>/n<nid>/<arm>); jitter_idx may be absent,
      so coordinates are ALWAYS parsed from rel_path (other coord columns, if present, must agree).
      steps / n_inferences / sig / pkl_sha256 / success / has_pkl come from the index.
  coordinate meta.json: has env_seed, inference_seed, success (no steps).
Index and meta must agree on success (and on seeds when the index has them). pkl bytes are not re-hashed
here (1800 x ~0.4 GB); the rollout-latent step verifies pkl_sha256 when it reads a pkl.

usage (승준): python3 archive_meta_index.py --coords archive_coords.tsv \
    --store /home/kimseungjun/datasets/temporal_vla_store --out archive_index.tsv
"""
import argparse, csv, json, re, sys
from pathlib import Path

ALIASES = {'kanu': 'kanu', 'worker1': 'worker1', 'srv48': 'worker1', 'worker2': 'worker2', 'srv50': 'worker2'}
COLS = ['plan_id', 'machine', 'instruction', 'scene', 'jitter', 'noise', 'env_seed', 'inference_seed',
        'success', 'steps', 'n_inferences', 'sig', 'pkl_sha256', 'has_pkl']
REL = re.compile(r'^(?P<plan>[0-9a-f]{12})/(?P<machine>[^/]+)/(?P<key>.+)/s(?P<s>\d+)/j(?P<j>\d+)/n(?P<n>\d+)/(?P<arm>[^/]+)/?$')


class IndexError_(RuntimeError):
    pass


def as_bool(v):
    if isinstance(v, bool): return int(v)
    s = str(v).strip().lower()
    if s in ('1', 'true', '1.0'): return 1
    if s in ('0', 'false', '0.0'): return 0
    raise IndexError_(f'not a bool: {v!r}')


def opt_int(v):
    s = '' if v is None else str(v).strip()
    return int(float(s)) if s not in ('', 'None', 'nan') else ''


def parse_rel(rel):
    m = REL.match(rel.strip())
    if not m:
        raise IndexError_(f'unparseable rel_path {rel!r}')
    return (m['plan'], ALIASES.get(m['machine'], m['machine']), m['key'], int(m['s']), int(m['j']), int(m['n'])), m['arm']


def load_index(rows, wanted=None):
    """Index rows -> {coord: row} for arm 'base'. Coords from rel_path; optional coord columns must agree.

    `wanted` = set of (plan_id, canonical machine, instruction) prefixes of the target coords. Rows outside
    it (older plans without a jitter level, other hosts/instructions) are skipped BEFORE parsing; rows inside it
    are parsed strictly and any inconsistency aborts.
    """
    out = {}
    for r in rows:
        if 'rel_path' not in r:
            raise IndexError_('index has no rel_path column')
        if wanted is not None:
            parts = r['rel_path'].strip().split('/')
            if len(parts) < 3 or not any(parts[0] == p and ALIASES.get(parts[1], parts[1]) == m
                                         and r['rel_path'].strip().startswith(f'{parts[0]}/{parts[1]}/{k}/s')
                                         for p, m, k in wanted):
                continue
        coord, arm = parse_rel(r['rel_path'])
        if r.get('armsig', arm) != arm:
            raise IndexError_(f'armsig/rel_path disagree {r["rel_path"]}')
        if arm != 'base':
            continue
        for col, pos in (('plan_id', 0), ('grid_instruction', 2), ('scene_idx', 3), ('jitter_idx', 4), ('noise_idx', 5)):
            if r.get(col, '') not in ('', None) and str(r[col]) != str(coord[pos]):
                raise IndexError_(f'{col}={r[col]} disagrees with rel_path {r["rel_path"]}')
        if r.get('machine') and ALIASES.get(r['machine'], r['machine']) != coord[1]:
            raise IndexError_(f'machine disagrees with rel_path {r["rel_path"]}')
        if coord in out and out[coord].get('sig') != r.get('sig'):
            raise IndexError_(f'conflicting index rows {coord}')
        out[coord] = r
    return out


def join(coord, ix, meta, has_pkl_file):
    """One output row; raises on any meta/index disagreement."""
    succ = as_bool(meta['success'] if 'success' in meta else meta['episode_success'])
    if ix.get('success', '') != '' and as_bool(ix['success']) != succ:
        raise IndexError_(f'success meta/index disagree {coord}')
    env, inf = int(meta.get('env_seed', ix.get('env_seed'))), int(meta.get('inference_seed', ix.get('inference_seed')))
    for col, val in (('env_seed', env), ('inference_seed', inf)):
        if str(ix.get(col, '')).strip() not in ('', str(val)):
            raise IndexError_(f'{col} meta/index disagree {coord}')
    if ix.get('has_pkl', '') != '' and as_bool(ix['has_pkl']) != has_pkl_file:
        raise IndexError_(f'has_pkl index/file disagree {coord}')
    return [coord[0], coord[1], coord[2], coord[3], coord[4], coord[5], env, inf, succ,
            opt_int(ix.get('steps')), opt_int(ix.get('n_inferences')), ix.get('sig', ''), ix.get('pkl_sha256', ''), has_pkl_file]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--coords', type=Path, required=True)
    ap.add_argument('--store', type=Path, required=True)
    ap.add_argument('--index', type=Path, help='default <store>/groot/n15/index/rollouts.tsv')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        sys.exit(f'refuse overwrite {a.out}')
    with open(a.coords, newline='') as f:
        coords = list(csv.DictReader(f, delimiter='\t'))
    wanted = {(c['plan_id'], ALIASES.get(c['machine'], c['machine']), c['instruction']) for c in coords}
    with open(a.index or a.store / 'groot/n15/index/rollouts.tsv', newline='') as f:
        index = load_index(csv.DictReader(f, delimiter='\t'), wanted)
    rows, errors = [], []
    for c in coords:
        coord = (c['plan_id'], ALIASES.get(c['machine'], c['machine']), c['instruction'], int(c['scene']), int(c['jitter']), int(c['noise']))
        rel = Path('groot/n15/grid', coord[0], coord[1], coord[2], f's{coord[3]}', f'j{coord[4]}', f'n{coord[5]}', 'base')
        try:
            ix = index.get(coord)
            if ix is None:
                raise IndexError_(f'missing index row {coord}')
            meta_p = a.store / rel / 'meta.json'
            if not meta_p.is_file():
                raise IndexError_(f'missing meta {rel}')
            rows.append(join(coord, ix, json.loads(meta_p.read_text()), int((a.store / rel / 'rollout.pkl').is_file())))
        except IndexError_ as e:
            errors.append(str(e))
    if errors or len(rows) != len(coords):
        sys.exit('ARCHIVE_INDEX_FAILED ' + json.dumps({'errors': errors[:20], 'n_errors': len(errors), 'rows': len(rows), 'coords': len(coords)}))
    with open(a.out, 'x') as f:
        f.write('\t'.join(COLS) + '\n')
        for r in rows:
            f.write('\t'.join(map(str, r)) + '\n')
    print(json.dumps({'rows': len(rows), 'success': sum(r[8] for r in rows), 'has_pkl': sum(r[13] for r in rows),
                      'steps_missing': sum(r[9] == '' for r in rows)}))


if __name__ == '__main__':
    main()

"""Run ON 승준 (stdlib, read-only): archive_coords.tsv -> archive_index.tsv for folds.py --archive-index.

Primary source = the coordinate's own meta.json (src/collect/artifacts.py write_safe_triplet writes
sig, pkl_sha256, success, env_seed, inference_seed, capture_layers, record_shape, plan_id/machine/
grid_instruction/scene_idx/jitter_idx/noise_idx). The store index groot/n15/index/rollouts.tsv was found
stale for these 1800 coords (2026-10-29), so it is NOT required and is NOT regenerated (other experiments'
scope); pass --index only to cross-check rows that exist there.
steps/length is not in meta.json and is NOT estimated here (empty); the rollout-latent/MIL-shard step reads
n_inferences from the pkl itself. --hash-pkl re-hashes rollout.pkl against meta pkl_sha256 (slow, ~0.4 GB each).

usage (승준): python3 archive_meta_index.py --coords archive_coords.tsv \
    --store /home/kimseungjun/datasets/temporal_vla_store --out archive_index.tsv [--hash-pkl] [--index <rollouts.tsv>]
"""
import argparse, csv, hashlib, json, re, sys
from pathlib import Path

ALIASES = {'kanu': 'kanu', 'worker1': 'worker1', 'srv48': 'worker1', 'worker2': 'worker2', 'srv50': 'worker2'}
COLS = ['plan_id', 'machine', 'instruction', 'scene', 'jitter', 'noise', 'env_seed', 'inference_seed',
        'success', 'steps', 'n_inferences', 'sig', 'pkl_sha256', 'has_pkl', 'pkl_hash_verified', 'index_crosscheck']
REL = re.compile(r'^(?P<plan>[0-9a-f]{12})/(?P<machine>[^/]+)/(?P<key>.+)/s(?P<s>\d+)/j(?P<j>\d+)/n(?P<n>\d+)/(?P<arm>[^/]+)/?$')
EXPECT_LAYERS = [0, 2, 4, 8, 10, 12, 15]
EXPECT_SHAPE = [7, 4, 49, 1536]


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
    """Optional cross-check source. Rows outside `wanted` (plan, canonical machine, instruction) prefixes are
    skipped BEFORE parsing (old non-jitter plans etc.); rows inside are parsed strictly."""
    out = {}
    for r in rows:
        if 'rel_path' not in r:
            raise IndexError_('index has no rel_path column')
        rel = r['rel_path'].strip()
        if wanted is not None:
            parts = rel.split('/')
            if len(parts) < 3 or not any(parts[0] == p and ALIASES.get(parts[1], parts[1]) == m
                                         and rel.startswith(f'{parts[0]}/{parts[1]}/{k}/s') for p, m, k in wanted):
                continue
        coord, arm = parse_rel(rel)
        if r.get('armsig', arm) != arm:
            raise IndexError_(f'armsig/rel_path disagree {rel}')
        if arm != 'base':
            continue
        for col, pos in (('plan_id', 0), ('grid_instruction', 2), ('scene_idx', 3), ('jitter_idx', 4), ('noise_idx', 5)):
            if r.get(col, '') not in ('', None) and str(r[col]) != str(coord[pos]):
                raise IndexError_(f'{col}={r[col]} disagrees with rel_path {rel}')
        if r.get('machine') and ALIASES.get(r['machine'], r['machine']) != coord[1]:
            raise IndexError_(f'machine disagrees with rel_path {rel}')
        if coord in out and out[coord].get('sig') != r.get('sig'):
            raise IndexError_(f'conflicting index rows {coord}')
        out[coord] = r
    return out


def from_meta(coord, meta, has_pkl_file, ix=None, pkl_sha=None):
    """One output row from meta.json; strict on coordinate/capture contract; optional index cross-check."""
    plan, mach, key, s, j, n = coord
    for k, want in (('plan_id', plan), ('grid_instruction', key), ('scene_idx', s), ('jitter_idx', j), ('noise_idx', n)):
        if k in meta and str(meta[k]) != str(want):
            raise IndexError_(f'meta {k}={meta[k]} != coord {coord}')
    if 'machine' in meta and ALIASES.get(str(meta['machine']).split(':')[0], str(meta['machine']).split(':')[0]) != mach:
        raise IndexError_(f'meta machine {meta["machine"]} != {mach}')
    if meta.get('capture_layers') is not None and list(meta['capture_layers']) != EXPECT_LAYERS:
        raise IndexError_(f'capture_layers {meta["capture_layers"]} {coord}')
    if meta.get('record_shape') is not None and list(meta['record_shape']) != EXPECT_SHAPE:
        raise IndexError_(f'record_shape {meta["record_shape"]} {coord}')
    succ = as_bool(meta['success'] if 'success' in meta else meta['episode_success'])
    env, inf = int(meta['env_seed']), int(meta['inference_seed'])
    sha = str(meta.get('pkl_sha256', ''))
    sig = str(meta.get('sig', ''))
    if sha and sig and not sha.startswith(sig):
        raise IndexError_(f'sig is not prefix of pkl_sha256 {coord}')
    if has_pkl_file and not sha:
        raise IndexError_(f'pkl present but meta has no pkl_sha256 {coord}')
    verified = ''
    if pkl_sha is not None:
        if pkl_sha != sha:
            raise IndexError_(f'pkl bytes sha != meta pkl_sha256 {coord}')
        verified = 1
    cross = 'absent'
    if ix is not None:
        if ix.get('success', '') != '' and as_bool(ix['success']) != succ:
            raise IndexError_(f'success meta/index disagree {coord}')
        for col, val in (('env_seed', env), ('inference_seed', inf), ('pkl_sha256', sha)):
            if str(ix.get(col, '')).strip() not in ('', str(val)):
                raise IndexError_(f'{col} meta/index disagree {coord}')
        cross = 'agree'
    steps = opt_int(meta.get('steps', ix.get('steps') if ix else None))
    n_inf = opt_int(meta.get('n_inferences', ix.get('n_inferences') if ix else None))
    return [plan, mach, key, s, j, n, env, inf, succ, steps, n_inf, sig, sha, has_pkl_file, verified, cross]


def file_sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 23), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--coords', type=Path, required=True)
    ap.add_argument('--store', type=Path, required=True)
    ap.add_argument('--index', type=Path, help='optional cross-check (store index found stale for these coords)')
    ap.add_argument('--hash-pkl', action='store_true')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        sys.exit(f'refuse overwrite {a.out}')
    with open(a.coords, newline='') as f:
        coords = list(csv.DictReader(f, delimiter='\t'))
    index = {}
    if a.index:
        wanted = {(c['plan_id'], ALIASES.get(c['machine'], c['machine']), c['instruction']) for c in coords}
        with open(a.index, newline='') as f:
            index = load_index(csv.DictReader(f, delimiter='\t'), wanted)
    rows, errors = [], []
    for c in coords:
        coord = (c['plan_id'], ALIASES.get(c['machine'], c['machine']), c['instruction'], int(c['scene']), int(c['jitter']), int(c['noise']))
        rel = Path('groot/n15/grid', coord[0], coord[1], coord[2], f's{coord[3]}', f'j{coord[4]}', f'n{coord[5]}', 'base')
        try:
            meta_p, pkl_p = a.store / rel / 'meta.json', a.store / rel / 'rollout.pkl'
            if not meta_p.is_file():
                raise IndexError_(f'missing meta {rel}')
            has = int(pkl_p.is_file())
            rows.append(from_meta(coord, json.loads(meta_p.read_text()), has, index.get(coord),
                                  file_sha256(pkl_p) if (a.hash_pkl and has) else None))
        except (IndexError_, KeyError, ValueError) as e:
            errors.append(f'{type(e).__name__}: {e}')
    if errors or len(rows) != len(coords):
        sys.exit('ARCHIVE_INDEX_FAILED ' + json.dumps({'errors': errors[:20], 'n_errors': len(errors), 'rows': len(rows), 'coords': len(coords)}))
    with open(a.out, 'x') as f:
        f.write('\t'.join(COLS) + '\n')
        for r in rows:
            f.write('\t'.join(map(str, r)) + '\n')
    print(json.dumps({'rows': len(rows), 'success': sum(r[8] for r in rows), 'has_pkl': sum(r[13] for r in rows),
                      'pkl_hash_verified': sum(r[14] == 1 for r in rows), 'index_crosscheck_agree': sum(r[15] == 'agree' for r in rows)}))


if __name__ == '__main__':
    main()

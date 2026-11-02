"""27 heldout-j fold spec (CPU, stdlib only).

Fold = (instruction, scene, target_j). fit = other 4 j x n0..39 (160), eval = target j x n0..39 (40).
Other folds' target j are NOT excluded (user 2026-10-28). Success-0 targets (OvenRack s2 j4) stay in.

Source plan per noise range (join key source_plan+machine+instruction+s+j+n, n20 README):
  n0-9   = the scene's original plan = n20 requests.tsv `source_plan` column (e.g. Dish-left s3 ->
           4ab360df2b71, bread/marshmallow s4 -> 4fa6496cd684); must exist in archive inventory.
  n10-19 = 748863ce9233 (n20 requests)      n20-39 = 2f26fb046fc4 (n40 requests)
Machine aliases: srv48/AISem_48_junhyeong == worker1, srv50/AISem_50_junhyeong == worker2 (same physical
host); everything is canonicalized to archive names (kanu/worker1/worker2). Plans are never substituted.
n0-9 seeds are derived (scene env_seed, inference 1300000+n) and flagged until archive meta.json
confirms them (--archive-index, built on 승준 by archive_meta_index.py).

usage: folds.py --out outputs/analysis/groot_fold27_20261028 [--archive-index <tsv>]
"""
import argparse, csv, hashlib, json, sys
from pathlib import Path

R = Path(__file__).resolve().parents[3]
SEL = 'outputs/analysis/rescue27_n40_selection_20261027/selection.json'
TGT = 'outputs/analysis/rescue27_n40_selection_20261027/targets.tsv'
INV = 'outputs/analysis/collection_inventory_20261027/archive_inventory.jsonl'
CFG = '.claude/worktrees/collect-topup/configs/collect'
N20 = CFG + '/n15_target9_n20_20261013/requests.tsv'
N40 = CFG + '/n15_target9_n40_20261025/requests.tsv'
N20_PLAN, N40_PLAN = '748863ce9233', '2f26fb046fc4'
ARCHIVE_TO_HOST = {'kanu': 'kanu', 'worker1': 'srv48', 'worker2': 'srv50'}
# 같은 물리 머신의 두 표기 (n20 requests/inventory=worker*, n40 requests=srv*).
MACHINE_ALIASES = {'kanu': 'kanu', 'worker1': 'worker1', 'srv48': 'worker1', 'AISem_48_junhyeong': 'worker1',
                   'worker2': 'worker2', 'srv50': 'worker2', 'AISem_50_junhyeong': 'worker2'}
# user order 2026-10-28; first j of each row = original target
USER_TARGETS = [('OpenDrawer/left', 2, (2, 4, 3)), ('OpenDrawer/right', 1, (4, 0, 2)),
                ('DishwasherRack/out-left', 3, (2, 0, 1)), ('DishwasherRack/out-right', 1, (2, 0, 4)),
                ('OvenRack/out-left', 2, (4, 1, 2)), ('PPCC/bread', 4, (1, 4, 3)), ('PPCC/candle', 0, (1, 0, 2)),
                ('PPCC/jug', 1, (1, 4, 3)), ('PPCC/marshmallow', 4, (2, 3, 0))]
JS, NS = range(5), range(40)


class FoldError(RuntimeError):
    pass


def canonical_machine(name):
    """archive 표기(kanu/worker1/worker2)로 정규화. 모르는 이름은 추측하지 않고 거부."""
    try:
        return MACHINE_ALIASES[str(name).strip()]
    except KeyError:
        raise FoldError(f'unknown machine alias {name!r}') from None


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fold_id(instruction, scene, j):
    return f"{instruction.replace('/', '-')}_s{scene}_j{j}"


def noise_range(n):
    if n in range(0, 10): return 'n0_9'
    if n in range(10, 20): return 'n10_19'
    if n in range(20, 40): return 'n20_39'
    raise FoldError(f'noise {n} outside n0..39')


def read_tsv(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def load_inventory(root):
    """{(plan, machine, key, scene, j, n)} with canonical machine."""
    have = set()
    for line in (root / INV).read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            m = canonical_machine(d['machine'])
            for j, ns in d['j'].items():
                for n in ns:
                    k = (d['plan'], m, d['key'], int(d['scene']), int(j), int(n))
                    if k in have:
                        raise FoldError(f'duplicate inventory coord {k}')
                    have.add(k)
    return have


def load_seeds(root, inventory):
    """(instruction, scene, j, n) -> {env_seed, inference_seed, plan_id, machine, seed_source}."""
    seeds, source_plan = {}, {}
    def put(key, rec):
        if key in seeds:
            raise FoldError(f'duplicate request row {key}')
        seeds[key] = rec
    for r in read_tsv(root / N20):
        k = (r['instruction'], int(r['scene']), int(r['jitter']), int(r['noise']))
        if r['plan_id'] != N20_PLAN or noise_range(k[3]) != 'n10_19':
            raise FoldError(f'n20 row outside contract {k} {r["plan_id"]}')
        sp = source_plan.setdefault(k[:2], r['source_plan'])
        if sp != r['source_plan']:
            raise FoldError(f'source_plan not unique for {k[:2]}')
        put(k, {'env_seed': int(r['env_seed']), 'inference_seed': int(r['inference_seed']), 'plan_id': N20_PLAN,
                'machine': canonical_machine(r['machine']), 'seed_source': 'n20_requests'})
    for r in read_tsv(root / N40):
        k = (r['instruction'], int(r['scene']), int(r['jitter']), int(r['noise']))
        if r['plan_id'] != N40_PLAN or noise_range(k[3]) != 'n20_39':
            raise FoldError(f'n40 row outside contract {k} {r["plan_id"]}')
        put(k, {'env_seed': int(r['env_seed']), 'inference_seed': int(r['inference_seed']), 'plan_id': N40_PLAN,
                'machine': canonical_machine(r['machine']), 'seed_source': 'n40_requests'})
    for instr, scene, _ in USER_TARGETS:
        cell = [seeds[(instr, scene, j, n)] for j in JS for n in range(10, 40)]
        env = {c['env_seed'] for c in cell}; mach = {c['machine'] for c in cell}
        if len(env) != 1 or len(mach) != 1:
            raise FoldError(f'{instr} s{scene}: env_seed {env} / machine {mach} not unique')
        env, mach, plan = env.pop(), mach.pop(), source_plan[(instr, scene)]
        for j in JS:
            for n in range(10):
                if (plan, mach, instr, scene, j, n) not in inventory:
                    raise FoldError(f'source plan {plan} lacks {mach} {instr} s{scene} j{j} n{n} in inventory')
                put((instr, scene, j, n), {'env_seed': env, 'inference_seed': 1300000 + n, 'plan_id': plan,
                                           'machine': mach, 'seed_source': 'derived_pending_archive_meta'})
    return seeds, source_plan


def load_archive_index(p):
    """Rows built on 승준 by archive_meta_index.py; key (plan_id, machine, instruction, s, j, n)."""
    out = {}
    for r in read_tsv(p):
        k = (r['plan_id'], canonical_machine(r['machine']), r['instruction'], int(r['scene']), int(r['jitter']), int(r['noise']))
        if k in out:
            raise FoldError(f'duplicate archive row {k}')
        out[k] = r
    return out


def build(root, archive_index=None):
    sel = json.loads((root / SEL).read_text())
    by_instr = {t['instruction']: t for t in sel['targets']}
    targets_tsv = {(r['instruction'], int(r['scene']), int(r['jitter'])): r for r in read_tsv(root / TGT)}
    inventory = load_inventory(root)
    seeds, source_plan = load_seeds(root, inventory)
    folds, sources = [], {}
    for instr, scene, tjs in USER_TARGETS:
        t = by_instr[instr]
        if t['scene'] != scene or t['original_j'] != tjs[0] or list(t['additional_j']) != list(tjs[1:]):
            raise FoldError(f'user targets disagree with selection.json for {instr}')
        counts = t['counts']  # [success, failure] per j0..j4
        if any(sum(c) != 40 for c in counts):
            raise FoldError(f'{instr}: per-j counts not 40')
        mach = canonical_machine(t['machine'])
        def item(j, n):
            s = seeds[(instr, scene, j, n)]
            if s['machine'] != mach:
                raise FoldError(f'machine mismatch {instr} s{scene} j{j} n{n}: {s["machine"]} != {mach}')
            key = (s['plan_id'], mach, instr, scene, j, n)
            if key not in inventory:
                raise FoldError(f'coord absent from archive inventory {key}')
            rec = {'plan_id': s['plan_id'], 'j': j, 'n': n, 'env_seed': s['env_seed'],
                   'inference_seed': s['inference_seed'], 'seed_source': s['seed_source']}
            if archive_index is not None:
                a = archive_index.get(key)
                if a is None:
                    raise FoldError(f'archive index missing {key}')
                if (int(a['env_seed']), int(a['inference_seed'])) != (s['env_seed'], s['inference_seed']):
                    raise FoldError(f'archive seeds disagree {key}: {a["env_seed"]},{a["inference_seed"]}')
                steps = str(a.get('steps', '')).strip()   # index steps may be empty (meta.json has none)
                rec.update(seed_source='archive_meta', success=int(a['success']), steps=int(steps) if steps else None,
                           n_inferences=int(a['n_inferences']) if str(a.get('n_inferences', '')).strip() else None,
                           sig=a['sig'], pkl_sha256=a['pkl_sha256'], has_pkl=int(a['has_pkl']))
            sources[key] = rec
            return dict(rec)
        for tj in tjs:
            row = targets_tsv[(instr, scene, tj)]
            if (int(row['success']), int(row['failure'])) != tuple(counts[tj]):
                raise FoldError(f'targets.tsv vs selection.json mismatch {instr} j{tj}')
            fit = [item(j, n) for j in JS if j != tj for n in NS]
            ev = [item(tj, n) for n in NS]
            fit_s = sum(counts[j][0] for j in JS if j != tj)
            fold = {'fold_id': fold_id(instr, scene, tj), 'instruction': instr, 'scene': scene,
                    'archive_machine': mach, 'eval_host': ARCHIVE_TO_HOST[mach],
                    'source_plan_n0_9': source_plan[(instr, scene)],
                    'target_j': tj, 'target_role': 'original' if tj == tjs[0] else 'additional',
                    'expected': {'eval_success': counts[tj][0], 'eval_failure': counts[tj][1],
                                 'fit_success': fit_s, 'fit_failure': 160 - fit_s,
                                 'fit_per_j': {str(j): counts[j] for j in JS if j != tj}},
                    'fit': fit, 'eval': ev}
            validate_fold(fold)
            if archive_index is not None:
                got = (sum(r['success'] for r in ev), sum(r['success'] for r in fit))
                if got != (counts[tj][0], fit_s):
                    raise FoldError(f'{fold["fold_id"]}: archive success {got} != selection {(counts[tj][0], fit_s)}')
                if any(r['has_pkl'] != 1 for r in fit):
                    raise FoldError(f'{fold["fold_id"]}: fit rollout without pkl')
            folds.append(fold)
    validate_spec(folds, sources)
    pending = sum(r['seed_source'] == 'derived_pending_archive_meta' for r in sources.values())
    scenes = {(i, s) for i, s, _ in USER_TARGETS}
    return {'version': 'groot_fold27_v2', 'created_by': 'scripts/rl2_vla/groot_fold27_20261028/folds.py',
            'rule': 'fit = 4 non-target j x n0..39 (160); eval = target j x n0..39 (40); other folds target j not excluded; '
                    'user instruction 2026-10-28 overrides n40 README sealed_test/reserved_not_train rule for this experiment',
            'source_plans_n0_9': {f'{k[0]}|s{k[1]}': v for k, v in sorted(source_plan.items()) if k in scenes},
            'noise_ranges': {'n0_9': 'scene source_plan', 'n10_19': N20_PLAN, 'n20_39': N40_PLAN},
            'inputs_sha256': {p: sha256(root / p) for p in (SEL, TGT, INV, N20, N40)},
            'archive_index_sha256': None, 'unique_sources': len(sources),
            'source_rows_pending_archive_confirmation': pending,
            'production_ready': pending == 0 and archive_index is not None, 'folds': folds}, sources


def validate_fold(f):
    """Leakage/shape gate reused by trainers and eval builders."""
    fit = {(r['plan_id'], r['j'], r['n']) for r in f['fit']}
    ev = {(r['plan_id'], r['j'], r['n']) for r in f['eval']}
    if len(fit) != 160 or len(ev) != 40 or len(f['fit']) != 160 or len(f['eval']) != 40:
        raise FoldError(f'{f["fold_id"]}: counts fit {len(fit)} eval {len(ev)}')
    if fit & ev:
        raise FoldError(f'{f["fold_id"]}: fit/eval overlap')
    if any(r['j'] == f['target_j'] for r in f['fit']) or any(r['j'] != f['target_j'] for r in f['eval']):
        raise FoldError(f'{f["fold_id"]}: target j leakage')
    if sorted(r['n'] for r in f['eval']) != list(NS):
        raise FoldError(f'{f["fold_id"]}: eval must be n0..39')
    if sorted((r['j'], r['n']) for r in f['fit']) != [(j, n) for j in JS if j != f['target_j'] for n in NS]:
        raise FoldError(f'{f["fold_id"]}: fit must be 4 j x n0..39')
    for r in f['fit'] + f['eval']:
        want = {'n10_19': N20_PLAN, 'n20_39': N40_PLAN}.get(noise_range(r['n']), f['source_plan_n0_9'])
        if r['plan_id'] != want:
            raise FoldError(f'{f["fold_id"]}: n{r["n"]} plan {r["plan_id"]} != {want}')
    return True


def validate_spec(folds, sources):
    if len(folds) != 27 or len({f['fold_id'] for f in folds}) != 27:
        raise FoldError('expected 27 unique folds')
    if len(sources) != 9 * 5 * 40:
        raise FoldError(f'expected 1800 unique source rollouts, got {len(sources)}')
    per_scene = {}
    for f in folds:
        per_scene.setdefault((f['instruction'], f['scene']), []).append(f['target_j'])
    if len(per_scene) != 9 or any(len(set(v)) != 3 for v in per_scene.values()):
        raise FoldError('expected 9 scenes x 3 distinct target j')
    if sum(len(f['fit']) for f in folds) != 27 * 160 or sum(len(f['eval']) for f in folds) != 27 * 40:
        raise FoldError('total fit/eval rows')


def write_outputs(out, spec, sources, final):
    path = out / ('fold_spec.json' if final else 'fold_spec_draft.json')
    text = json.dumps(spec, indent=1)
    if path.exists() and path.read_text() != text:
        raise FoldError(f'{path} exists with different content; refuse overwrite (new version dir needed)')
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    cols = ['plan_id', 'machine', 'instruction', 'scene', 'jitter', 'noise', 'env_seed', 'inference_seed', 'seed_source']
    with (out / 'archive_coords.tsv').open('w') as f:
        f.write('\t'.join(cols) + '\n')
        for k in sorted(sources):
            r = sources[k]
            f.write('\t'.join(map(str, [*k, r['env_seed'], r['inference_seed'], r['seed_source']])) + '\n')
    with (out / 'fold_membership.tsv').open('w') as f:
        f.write('fold_id\trole\tplan_id\tmachine\tinstruction\tscene\tjitter\tnoise\n')
        for fo in spec['folds']:
            for role in ('fit', 'eval'):
                for r in fo[role]:
                    f.write('\t'.join(map(str, [fo['fold_id'], role, r['plan_id'], fo['archive_machine'], fo['instruction'], fo['scene'], r['j'], r['n']])) + '\n')
    return path


def load_spec(path):
    """Trainer/eval entry: re-validate every fold before use."""
    spec = json.loads(Path(path).read_text())
    for f in spec['folds']:
        validate_fold(f)
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--archive-index', type=Path)
    a = ap.parse_args()
    idx = load_archive_index(a.archive_index) if a.archive_index else None
    spec, sources = build(R, idx)
    if a.archive_index:
        spec['archive_index_sha256'] = sha256(a.archive_index)
    path = write_outputs(a.out, spec, sources, idx is not None)
    print(json.dumps({'out': str(path), 'folds': len(spec['folds']), 'unique_sources': spec['unique_sources'],
                      'pending_seed_rows': spec['source_rows_pending_archive_confirmation'],
                      'production_ready': spec['production_ready']}))


if __name__ == '__main__':
    try:
        main()
    except FoldError as e:
        print('FOLD_ERROR', e, file=sys.stderr); sys.exit(2)

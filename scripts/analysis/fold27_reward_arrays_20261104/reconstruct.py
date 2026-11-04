"""Reconstruct the exact per-inference basic and MIX reward arrays of ONE production Fold27 fold.

    OMP_NUM_THREADS=4 <python> scripts/analysis/fold27_reward_arrays_20261104/reconstruct.py \
        --spec outputs/analysis/groot_fold27_20261028/folds/fold_spec.json --fold OpenDrawer-left_s2_j2 \
        --episodes <compact/full extract dir> --latents <rollout_latents dir> \
        --fold-dir <training out root>/<fold> --out-root <new derived root>

What is reused unchanged (sha-pinned, the code recorded in the DONE fingerprints):
  finetune_fold.load_fit_episodes  spec linkage, pkl/npz sha, latent provenance contract, fit (j,n) sort order
  ae_potential.load_fold_ae        saved fold ae_potential.json + encoder (encoder sha verified, never refit)
  rewards.episode_transitions/concat  the reward rows exactly as build_rows builds them
Actions are not used by the rewards or observations: an identity codec is passed (the GR00T checkpoint codec is
not needed); this changes only rows['actions'], which is neither hashed in the fingerprint nor written here.

Gate (all must hold for sha_matched): rows and observations_sha256 == both DONE fingerprints, basic rewards_sha256 ==
basic DONE, mix rewards_sha256 == mix DONE, float32 term-wise MIX recomposition bitwise == mix rewards.
Otherwise the fold goes to <out-root>/contract_failure/<fold>/ (never mixed with <out-root>/sha_matched/).
No pickle is loaded (np.load allow_pickle=False everywhere; no resume/checkpoint files are opened).
Existing outputs are never overwritten; inputs are only read.
"""
import argparse, hashlib, json, os, resource, sys, time, traceback
from pathlib import Path

MAX_THREADS = 4
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_k, str(MAX_THREADS))
    if int(os.environ[_k]) > MAX_THREADS:
        sys.exit(f'{_k}={os.environ[_k]} > {MAX_THREADS}')

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core  # noqa: E402

PINNED = {'rewards.py': '6f3e780b05279dba3e03dfc38b602f304b769fa356f9447d0e8e6b3af3c5f6d7',
          'ae_potential.py': '312d0492ccb52db8038644c0a8bc89f6ca1502456f107aff6cc09972213ba9e9'}
REPO = TRQAM = CODE_PATHS = None


def configure(code_root):
    """code_root = checkout holding the pinned training code (default: this repo). Imported read-only; sha-verified."""
    global REPO, TRQAM, CODE_PATHS
    REPO = Path(code_root).resolve()
    TRQAM = REPO / 'scripts/rl2_vla/groot_fold27_20261028/trqam'
    CODE_PATHS = {'finetune_fold.py': TRQAM / 'finetune_fold.py', 'rewards.py': TRQAM / 'rewards.py',
                  'trqam_prefix.py': TRQAM / 'trqam_prefix.py', 'folds.py': TRQAM.parent / 'folds.py',
                  'finetune_fold_v2.py': TRQAM / 'finetune_fold_v2.py', 'resume_state.py': TRQAM / 'resume_state.py',
                  'ae_potential.py': TRQAM / 'ae_potential.py'}
    for q in (str(REPO), str(REPO / 'scripts/rl2_vla/groot_robocasa_adapter_20261021'), str(TRQAM.parent), str(TRQAM)):
        if q not in sys.path:
            sys.path.insert(1, q)


ARM_DIRS = {'basic': 'basic', 'mix': 'mix_a0.1_b0.0396'}
ALPHA, BETA = 0.1, 0.0396


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def rel(p):
    """Artifact paths: repo-relative, or '<external>/<last 3 parts>' (no absolute paths in artifacts)."""
    p = Path(p).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return '<external>/' + '/'.join(p.parts[-3:])


class IdentityCodec:
    def normalize(self, a):
        return a


def check_code(done_fps):
    """Local code == pinned values == every DONE fingerprint's code record."""
    local = {k: sha(p) for k, p in CODE_PATHS.items()}
    for k, v in PINNED.items():
        if local[k] != v:
            raise core.ContractError(f'{k} sha {local[k]} != pinned {v}')
    for arm, fp in done_fps.items():
        for k, v in fp['code'].items():
            if local.get(k) != v:
                raise core.ContractError(f'{arm} DONE code {k} {v[:12]} != local {str(local.get(k))[:12]}')
    return local


def load_done(fold_dir, fold_id):
    out = {}
    for arm, d in ARM_DIRS.items():
        p = fold_dir / d / 'DONE.json'
        if not p.exists():
            raise core.ContractError(f'missing input {rel(p)}')
        dj = json.loads(p.read_text())
        fp = dj['fingerprint']
        if fp['fold_id'] != fold_id or fp['arm'] != arm or fp.get('potential') != 'ae16_k48' or dj.get('smoke'):
            raise core.ContractError(f'{rel(p)}: fold/arm/potential/smoke mismatch')
        if arm == 'mix' and (fp['alpha'], fp['beta']) != (ALPHA, BETA):
            raise core.ContractError(f'mix alpha/beta {fp["alpha"]}/{fp["beta"]} != {ALPHA}/{BETA}')
        if dj.get('reward_confirmation', {}).get('lambda_prior', 5.0) != 5.0:
            raise core.ContractError('lambda prior != 5')
        out[arm] = dict(path=p, sha=sha(p), fp=fp, done=dj)
    return out


def load_potential(fold_dir, eps, instruction):
    import ae_potential as AP
    pj, pe = fold_dir / 'ae_potential.json', fold_dir / 'ae_potential_encoder.npz'
    for p in (pj, pe):
        if not p.exists():
            raise core.ContractError(f'missing input {rel(p)}')
    p = AP.load_fold_ae(pj)                      # verifies encoder sha; no refit
    ids = sorted(p.metadata.get('episode_ids', []) + p.metadata.get('ae_validation_episode_ids', []))
    if ids != sorted(e['episode_id'] for e in eps):     # finetune_fold.build_rows check
        raise core.ContractError('saved ae_potential.json was fit on different episodes')
    task = p.tasks.get(instruction, {})
    prov = {'ae_potential.json': sha(pj), 'ae_potential_encoder.npz': sha(pe),
            'prior_module_sha256': {n: sha(AP.PRIOR / f'{n}.py') for n in ('potential', 'cache_io', 'tc_import')},
            'disabled': bool(task.get('disabled')), 'support_prior': 5.0}
    reuse = fold_dir / 'AE_REUSED_FROM.json'
    if reuse.exists():
        r = json.loads(reuse.read_text())['files']
        prov['ae_reused_from_match'] = (r['ae_potential.json']['sha256'] == prov['ae_potential.json'] and
                                        r['ae_potential_encoder.npz']['sha256'] == prov['ae_potential_encoder.npz'])
        if not prov['ae_reused_from_match']:
            raise core.ContractError('ae_potential files differ from AE_REUSED_FROM.json record')
    return AP.potential_fn(p, instruction), prov


def load_fit_episodes_json(FF, fold, episodes_dir, latents_dir):
    """finetune_fold.load_fit_episodes with the extract-NPZ read replaced by the record JSON scalars (success,
    env_steps, max_episode_steps, n_inferences). For staging dirs whose extract NPZs were removed after training
    (srv48 dish-right cleanup 2026-11-03). Same validate_fold, (j,n) order, linkage, latent npz sha, latent
    provenance, count checks. Actions are not loaded (zeros; rewards/observations do not use them). Exactness of
    the JSON scalars is then proven only by the rewards_sha256 gate (they are the only non-latent reward inputs)."""
    FF.validate_fold(fold)
    FF.check_latent_provenance(episodes_dir, latents_dir)
    eps = []
    for r in sorted(fold['fit'], key=lambda r: (r['j'], r['n'])):
        eid = f'{r["plan_id"]}/{fold["archive_machine"]}/{fold["instruction"]}/s{fold["scene"]}/j{r["j"]}/n{r["n"]}'
        st = FF.episode_stem(eid)
        rec = json.loads((episodes_dir / f'{st}.json').read_text())
        lrec = json.loads((latents_dir / f'{st}.json').read_text())
        FF.check_linkage(eid, r, rec, lrec)
        if 'source_npz_sha256' in rec and lrec.get('source_npz_sha256') not in (None, rec['source_npz_sha256']):
            raise core.ContractError(f'compact/latent source npz mismatch {eid}')
        if sha(latents_dir / lrec['npz']) != lrec['npz_sha256']:
            raise core.ContractError(f'latent npz sha {eid}')
        with np.load(latents_dir / lrec['npz'], allow_pickle=False) as zl:
            lat = zl['latents'].astype(np.float32)
        succ = bool(rec['success'])
        if succ != bool(r['success']) or len(lat) != int(rec['n_inferences']):
            raise core.ContractError(f'success/length mismatch {eid}')
        eps.append({'episode_id': eid, 'latents': lat, 'actions': np.zeros((len(lat), 16, 12), np.float32),
                    'success': succ, 'steps': int(rec['env_steps']), 'max_episode_steps': int(rec['max_episode_steps'])})
    if len(eps) != 160 or sum(e['success'] for e in eps) != fold['expected']['fit_success']:
        raise core.ContractError('fit episode count/success differs from fold spec')
    return eps


def build(a, receipt):
    import finetune_fold as FF
    import rewards as RW
    from folds import load_spec
    fold_dir = a.fold_dir
    dones = load_done(fold_dir, a.fold)
    receipt['inputs']['done'] = {arm: {'path': rel(d['path']), 'sha256': d['sha']} for arm, d in dones.items()}
    receipt['code_sha256'] = check_code({arm: d['fp'] for arm, d in dones.items()})
    spec = load_spec(a.spec)
    if not spec.get('production_ready'):
        raise core.ContractError('fold spec not production_ready')
    fold = next(f for f in spec['folds'] if f['fold_id'] == a.fold)
    receipt['inputs']['spec'] = {'path': rel(a.spec), 'sha256': sha(a.spec)}
    receipt['inputs']['episodes_dir'] = {'path': rel(a.episodes), 'contract_sha256': sha(a.episodes / 'contract.json')}
    receipt['inputs']['latents_dir'] = {'path': rel(a.latents), 'contract_sha256': sha(a.latents / 'contract.json')}
    receipt['episode_scalars'] = a.episode_scalars
    if a.episode_scalars == 'npz':
        eps = FF.load_fit_episodes(fold, a.episodes, a.latents, IdentityCodec())     # linkage + sha + (j,n) order
    else:
        eps = load_fit_episodes_json(FF, fold, a.episodes, a.latents)
    pot, prov = load_potential(fold_dir, eps, fold['instruction'])
    receipt['inputs']['potential'] = prov

    rows_b = RW.concat([RW.episode_transitions(e, 'basic') for e in eps])
    rows_m = RW.concat([RW.episode_transitions(e, 'mix', pot, alpha=ALPHA, beta=BETA) for e in eps])
    RW.recompute_check(rows_b, 'basic')
    got = {'rows': int(len(rows_b['rewards'])),
           'observations_sha256': core.sha_bytes(rows_b['observations']),
           'basic_rewards_sha256': core.sha_bytes(rows_b['rewards']),
           'mix_rewards_sha256': core.sha_bytes(rows_m['rewards']),
           'mix_observations_sha256': core.sha_bytes(rows_m['observations'])}
    want = {'rows': dones['basic']['fp']['rows'], 'mix_rows': dones['mix']['fp']['rows'],
            'basic_rewards_sha256': dones['basic']['fp']['rewards_sha256'],
            'mix_rewards_sha256': dones['mix']['fp']['rewards_sha256'],
            'basic_observations_sha256': dones['basic']['fp']['observations_sha256'],
            'mix_observations_sha256': dones['mix']['fp']['observations_sha256']}
    gates = {'rows_basic': got['rows'] == want['rows'], 'rows_mix': got['rows'] == want['mix_rows'],
             'observations_basic': got['observations_sha256'] == want['basic_observations_sha256'],
             'observations_mix': got['mix_observations_sha256'] == want['mix_observations_sha256'],
             'rewards_basic': got['basic_rewards_sha256'] == want['basic_rewards_sha256'],
             'rewards_mix': got['mix_rewards_sha256'] == want['mix_rewards_sha256']}
    receipt['sha'] = {'reconstructed': got, 'done_fingerprint': want, 'gates': gates}
    del rows_b['observations'], rows_b['next_observations'], rows_m['next_observations'], rows_m['observations']

    terms, rows = [], []
    for i, e in enumerate(eps):
        phi = pot(e['latents'])
        t = core.episode_terms(RW, e, phi)
        n = len(e['latents'])
        jn = core.parse_jn(e['episode_id'])
        t.update(episode_index=np.full(n, i, np.int32), step=np.arange(n, dtype=np.int32),
                 env_step_start=(RW.EXECUTE * np.arange(n)).astype(np.int32),
                 success=np.full(n, bool(e['success'])), j=np.full(n, jn[0], np.int16), n=np.full(n, jn[1], np.int16))
        terms.append(t)
    T = {k: np.concatenate([t[k] for t in terms]) for k in terms[0]}
    if not np.array_equal(T['phi'].astype(np.float32), rows_m['phi']):
        raise core.ContractError('re-evaluated phi != phi used in mix rows')
    if not np.array_equal(T['m'], rows_m['executed_m']) or not np.array_equal(T['b'] == 0, rows_m['terminals']):
        raise core.ContractError('m / terminal != reward rows')
    dec = core.decompose(T, rows_b['rewards'], rows_m['rewards'], ALPHA, BETA)
    gates['mix_termwise_recomposition'] = True        # decompose raised otherwise
    arrays = dict(basic=rows_b['rewards'], mix=rows_m['rewards'], alpha_term=dec['alpha_term'],
                  beta_cost=dec['beta_cost'], delta=dec['delta'], delta_exact=dec['delta_exact'],
                  phi=T['phi'], anchor=T['anchor'], C=T['C'], episode_index=T['episode_index'], step=T['step'],
                  env_step_start=T['env_step_start'], m=T['m'].astype(np.int32), terminal=T['b'] == 0,
                  final=T['final'], success=T['success'], j=T['j'], n=T['n'],
                  episode_id=np.asarray([e['episode_id'] for e in eps]),
                  ep_j=np.asarray([int(x['j'][0]) for x in terms], np.int16),
                  ep_n=np.asarray([int(x['n'][0]) for x in terms], np.int16),
                  ep_success=np.asarray([bool(e['success']) for e in eps]),
                  ep_steps=np.asarray([int(e['steps']) for e in eps], np.int32),
                  ep_max_steps=np.asarray([int(e['max_episode_steps']) for e in eps], np.int32),
                  ep_n_decisions=np.asarray([len(e['latents']) for e in eps], np.int32),
                  ep_row_start=np.concatenate([[0], np.cumsum([len(e['latents']) for e in eps])[:-1]]).astype(np.int64))
    receipt['decomposition'] = {'delta_float32_rounding_max_abs': dec['delta_float32_rounding_max_abs'],
                                'alpha': ALPHA, 'beta': BETA, 'gamma': core.GAMMA,
                                'episodes': len(eps), 'fit_success': int(sum(e['success'] for e in eps)),
                                'order': 'fold fit rows sorted by (j, n) (finetune_fold.load_fit_episodes)'}
    return arrays, all(gates.values())


def write_npz(path, arrays):
    if path.exists():
        raise FileExistsError(f'{rel(path)} exists; never overwritten')
    tmp = path.with_name('.' + path.name + '.tmp.npz')
    np.savez(tmp, **arrays)
    size = tmp.stat().st_size
    with np.load(tmp, allow_pickle=False) as z:          # reload gate, no pickle
        for k, v in arrays.items():
            if not np.array_equal(z[k], v):
                raise core.ContractError(f'npz reload mismatch {k}')
    os.replace(tmp, path)
    return size


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', type=Path, required=True); ap.add_argument('--fold', required=True)
    ap.add_argument('--episodes', type=Path, required=True); ap.add_argument('--latents', type=Path, required=True)
    ap.add_argument('--fold-dir', type=Path, required=True); ap.add_argument('--out-root', type=Path, required=True)
    ap.add_argument('--host', default='unknown')
    ap.add_argument('--code-root', type=Path, default=HERE.parents[2],
                    help='checkout with the pinned training code (sha-verified against DONE)')
    ap.add_argument('--episode-scalars', choices=('npz', 'json'), default='npz',
                    help='npz = finetune_fold.load_fit_episodes unchanged (default); json = record JSON scalars when '
                         'the extract NPZs are absent (exactness proven by the rewards_sha256 gate)')
    a = ap.parse_args(argv)
    sys.dont_write_bytecode = True
    configure(a.code_root)
    t0 = time.time()
    receipt = {'fold': a.fold, 'host': a.host, 'code_root': rel(a.code_root), 'started': t0, 'inputs': {},
               'threads': {k: os.environ[k] for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
               'no_pickle_loaded': True, 'originals_modified': False}
    for d in ('sha_matched', 'contract_failure'):
        if (a.out_root / d / a.fold).exists():
            print(f'REFUSED {a.fold}: {rel(a.out_root / d / a.fold)} exists', file=sys.stderr); return 3
    try:
        import torch
        torch.set_num_threads(MAX_THREADS)
        receipt['versions'] = {'python': sys.version.split()[0], 'numpy': np.__version__, 'torch': torch.__version__}
        arrays, ok = build(a, receipt)
        status = 'sha_matched' if ok else 'contract_failure'
    except Exception as e:
        arrays, status = None, 'contract_failure'
        receipt['error'] = repr(e); receipt['trace'] = traceback.format_exc()
    out = a.out_root / status / a.fold
    out.mkdir(parents=True, exist_ok=False)
    receipt['status'] = status
    if arrays is not None:
        receipt['estimated_output_bytes'] = int(sum(v.nbytes for v in arrays.values()))
        receipt['temporary_peak_bytes'] = write_npz(out / 'reward_arrays.npz', arrays)
        receipt['outputs'] = {'reward_arrays.npz': {'sha256': sha(out / 'reward_arrays.npz'),
                                                    'bytes': (out / 'reward_arrays.npz').stat().st_size,
                                                    'keys': {k: [str(v.dtype), list(v.shape)] for k, v in arrays.items()}}}
    receipt['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    receipt['seconds'] = time.time() - t0
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=1))
    print(json.dumps({'fold': a.fold, 'status': status, 'gates': receipt.get('sha', {}).get('gates'),
                      'error': receipt.get('error'), 'peak_rss_mib': round(receipt['peak_rss_mib'])}))
    return 0 if status == 'sha_matched' else 2


if __name__ == '__main__':
    sys.exit(main())

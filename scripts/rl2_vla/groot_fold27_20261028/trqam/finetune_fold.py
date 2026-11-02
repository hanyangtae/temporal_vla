"""Per-fold TRQAM fine-tuning of the common atomic65 prefix-QAM (basic / pbrs / mix), with a CPU dry-run.

    # CPU dry-run (no JAX, no GPU): data, potential, rewards, leakage, batch contract
    python finetune_fold.py --spec fold_spec.json --fold ID --arm basic --episodes <extract_scene v2> \
        --latents <rollout_latents out> --out <run root> --dry-run
    # production (host .venvs/rl2_iid, under groot harness receipt; GPU launch is the main session's call)
    HARNESS_RECEIPT=... python finetune_fold.py ... --init <atomic65 checkpoint dir> --init-step 500000 \
        [--reward-confirmation reward_confirmation.json] [--alpha A] [--beta B]

Data = the fold's 160 fit rollouts only (folds.validate_fold; eval-j rows are never read). Actions = GR00T's own
predicted 16-step chunk (normalized with the checkpoint codec), latents = decoder-input reconstruction.
Recipe (trqam_retention/train.py, RetentionAgent): restore every weight of the init checkpoint, frozen actor_fast
reference, 50k joint updates, batch 256, seed 42, resume state every 5k (current + previous kept), final-only export.
Reward arms: basic (no shaping), pbrs, mix. Without --alpha/--beta the existing recipe values are used and recorded as
decision=main_default_existing_recipe (RECIPE_DEFAULTS; PBRX==PBRS confirmed by user, alpha/beta question unanswered).
An optional --reward-confirmation json upgrades the record to user_confirmed; it is never fabricated.
Failure propagation: any exception -> FAILED.json + exit 2; restart resumes from the newest verified state; finished
runs (DONE.json with matching sha) are skipped; a code/data fingerprint change refuses to resume (new --out needed).
"""
import argparse, hashlib, json, os, sys, time, traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(HERE), str(HERE.parent), str(ROOT / 'scripts/rl2_vla/groot_robocasa_adapter_20261021'), str(ROOT)]
import rewards as RW            # noqa: E402
from folds import load_spec, validate_fold   # noqa: E402

STEPS, SAVE_EVERY, BATCH, SEED, GAMMA = 50000, 5000, 256, 42, 0.99
FEATURE_SIGNATURE = 'ccbfb51ca01724cb'


class FinetuneError(RuntimeError):
    pass


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def episode_stem(eid):
    return hashlib.sha256(eid.encode()).hexdigest()[:24]


def check_linkage(eid, r, rec, lrec):
    """All three must agree: fold spec row (archive meta), extract_scene record, rollout_latents record."""
    if not (rec['episode_id'] == lrec['episode_id'] == eid):
        raise FinetuneError(f'episode id linkage {eid}')
    if not (r['pkl_sha256'] == rec['pkl_sha256'] == lrec['pkl_sha256']):
        raise FinetuneError(f'pkl sha linkage {eid}: spec {r["pkl_sha256"][:12]} extract {rec["pkl_sha256"][:12]} '
                            f'latent {lrec["pkl_sha256"][:12]}')


def check_latent_provenance(episodes_dir, latents_dir):
    """Latents must have been reconstructed from exactly this extraction (rollout_latents contract)."""
    c = json.loads((latents_dir / 'contract.json').read_text())
    ec = json.loads((episodes_dir / 'contract.json').read_text())
    if ec.get('kind') == 'fold27_compact_v1':      # compact_export.py: traceable to the full extraction
        full_sha, full = ec['source_contract_sha256'], ec['source_contract']
    else:
        full_sha, full = sha(episodes_dir / 'contract.json'), ec
    if c.get('episodes_contract_sha256') != full_sha:
        raise FinetuneError('latents were built from a different extract_scene contract')
    if not full.get('with_last_block'):
        raise FinetuneError('extraction lacks last_block_d0; latents cannot come from it')
    return c


def check_init(init, step):
    """Production init = full common atomic65 run only (no smoke / partial checkpoints)."""
    flags = json.loads((init / f'flags_{step}.json').read_text())
    label = str(flags.get('data_label', ''))
    if 'smoke' in label.lower() or int(flags.get('steps_done', -1)) != int(step):
        raise FinetuneError(f'init checkpoint not a full run: data_label={label!r} steps_done={flags.get("steps_done")} step={step}')
    return label


def load_fit_episodes(fold, episodes_dir, latents_dir, codec):
    """160 fit rollouts -> rewards.episode_transitions inputs. Verifies record linkage, sha, counts, steps."""
    validate_fold(fold)
    check_latent_provenance(episodes_dir, latents_dir)
    eps = []
    for r in sorted(fold['fit'], key=lambda r: (r['j'], r['n'])):
        eid = f'{r["plan_id"]}/{fold["archive_machine"]}/{fold["instruction"]}/s{fold["scene"]}/j{r["j"]}/n{r["n"]}'
        st = episode_stem(eid)
        rec = json.loads((episodes_dir / f'{st}.json').read_text())
        lrec = json.loads((latents_dir / f'{st}.json').read_text())
        check_linkage(eid, r, rec, lrec)
        if 'source_npz_sha256' in rec and lrec.get('source_npz_sha256') not in (None, rec['source_npz_sha256']):
            raise FinetuneError(f'compact/latent source npz mismatch {eid}')
        for d, x in ((episodes_dir, rec), (latents_dir, lrec)):
            if sha(d / x['npz']) != x['npz_sha256']:
                raise FinetuneError(f'npz sha {eid}')
        with np.load(episodes_dir / rec['npz']) as z, np.load(latents_dir / lrec['npz']) as zl:
            act, succ, steps, mx = z['actions_model12'], bool(z['success']), int(z['env_steps']), int(z['max_episode_steps'])
            lat = zl['latents'].astype(np.float32)
        if succ != bool(r['success']) or len(lat) != len(act):
            raise FinetuneError(f'success/length mismatch {eid}')
        eps.append({'episode_id': eid, 'latents': lat, 'actions': codec.normalize(act).astype(np.float32),
                    'success': succ, 'steps': steps, 'max_episode_steps': mx})
    if len(eps) != 160 or sum(e['success'] for e in eps) != fold['expected']['fit_success']:
        raise FinetuneError('fit episode count/success differs from fold spec')
    return eps


POTENTIALS = ('pca16_k48', 'ae16_k48')
# pca16_k48 = the potential every validated TRQAM reward run used (pinned clusters.json from
#   stage2_cluster_reward/clusters.fit; riskmix contract "No refit"). ae16_k48 = task_classification AE16 -> K48
#   (ae_cluster_reward/, comparison cache only, never trained/evaluated as a TRQAM arm). Choice is decision D-POT.
# Input of both = the QAM observation (here GR00T decoder-input 1024; SIMPLER: pi0 context 1024). MIL uses a different
#   input (DiT layer-12 denoise-3 token mean 1536).


def build_rows(fold, eps, arm, alpha, beta, potential='ae16_k48', out=None):
    """Potential is fit on the fold's 160 fit rollouts only (AE val split also inside them); basic needs none."""
    pot, bundle = None, None
    if arm != 'basic':
        if potential == 'ae16_k48':
            import ae_potential as AP
            saved = None if out is None else out / 'ae_potential.json'
            if saved is not None and saved.exists():
                # the dry-run (or an earlier attempt) already fit it on the same 160 fit rollouts: reuse, never refit,
                # so dry-run, resume and production all use one potential (rewards_sha256 in the fingerprint matches)
                p = AP.load_fold_ae(saved)
                ids = sorted(p.metadata.get('episode_ids', []) + p.metadata.get('ae_validation_episode_ids', []))
                if ids != sorted(e['episode_id'] for e in eps):
                    raise FinetuneError('saved ae_potential.json was fit on different episodes')
            else:
                p = AP.fit_fold_ae(eps, fold['instruction'])
                if saved is not None:
                    p.save(saved)
            bundle, pot = {'kind': 'ae16_k48', 'metadata': p.metadata}, AP.potential_fn(p, fold['instruction'])
        elif potential == 'pca16_k48':
            bundle = RW.fit_potential([(e['latents'], e['success']) for e in eps])
            pot = RW.potential_fn(bundle)
        else:
            raise FinetuneError(f'unknown potential {potential}')
    rows = RW.concat([RW.episode_transitions(e, arm, pot, alpha=alpha, beta=beta) for e in eps])
    resid = RW.recompute_check(rows, arm, alpha) if arm in ('basic', 'pbrs') else None
    return rows, bundle, resid


# Main-session default (2026-10-29; user reward question still unanswered, PBRX==PBRS confirmed). Values are the
# existing SIMPLER recipe, recorded with their source — not a user confirmation:
#   pbrs: alpha .1, lambda 5, PCA16/K48  = original TRQAM cluster arm (export_training_cache.py:47 shaping_scale .1;
#         clusters.py prior 5)
#   mix : alpha .1, beta .03960         = riskmix arm mix_a10_b03960 (build_riskmix_cache.py:45-48)
RECIPE_DEFAULTS = {'pbrs': {'alpha': 0.1, 'beta': None, 'source': 'TRQAM cluster arm (stage2 export_training_cache shaping .1, lambda5)'},
                   'mix': {'alpha': 0.1, 'beta': 0.0396, 'source': 'riskmix_contract arm mix_a10_b03960'}}


def reward_gate(a):
    """Fill recipe defaults (recorded as main_default, never as a user confirmation); optional user confirmation json."""
    if a.arm == 'basic':
        if a.alpha is not None or a.beta is not None:
            raise FinetuneError('basic takes no alpha/beta')
        return {'arm': 'basic', 'decision': 'n/a'}
    d = RECIPE_DEFAULTS[a.arm]
    used_default = a.alpha is None and a.beta is None
    if used_default:
        a.alpha, a.beta = d['alpha'], d['beta']
    if a.alpha is None or (a.arm == 'mix' and a.beta is None) or (a.arm == 'pbrs' and a.beta is not None):
        raise FinetuneError(f'{a.arm}: alpha (and beta for mix only) must be given together')
    rec = {'arm': a.arm, 'alpha': a.alpha, 'beta': a.beta, 'lambda_prior': 5.0,
           'decision': 'main_default_existing_recipe' if used_default else 'explicit_cli', 'source': d['source']}
    if a.reward_confirmation is not None:
        c = json.loads(a.reward_confirmation.read_text())
        if (c.get('arm'), c.get('alpha'), c.get('beta')) != (a.arm, a.alpha, a.beta) or not c.get('confirmed_by'):
            raise FinetuneError(f'reward confirmation does not match {a.arm}/{a.alpha}/{a.beta}')
        rec.update(decision='user_confirmed', confirmation=c)
    return rec


def fingerprint(a, fold, rows):
    files = {p.name: sha(p) for p in (HERE / 'finetune_fold.py', HERE / 'rewards.py', HERE / 'trqam_prefix.py', HERE.parent / 'folds.py')}
    return {'fold_id': fold['fold_id'], 'arm': a.arm, 'alpha': a.alpha, 'beta': a.beta, 'potential': a.potential, 'code': files,
            'rows': int(len(rows['rewards'])), 'rewards_sha256': hashlib.sha256(rows['rewards'].tobytes()).hexdigest(),
            'observations_sha256': hashlib.sha256(rows['observations'].tobytes()).hexdigest(),
            'init': None if a.init is None else {'dir_name': a.init.name, 'step': a.init_step,
                                                 'params_sha256': sha(a.init / f'params_{a.init_step}.pkl')},
            'steps': a.smoke_steps or STEPS, 'batch': BATCH, 'seed': SEED, 'gamma': GAMMA}


def launch_receipt():
    """GPU reservation evidence: groot harness receipt, or the kanu GPU lease held by the training scheduler
    (with_gpu_lease.sh, the precedent for JAX training in common_qam). Recorded as DONE.harness_receipt."""
    if os.environ.get('HARNESS_RECEIPT'):
        return os.environ['HARNESS_RECEIPT']
    lease = os.environ.get('FOLD27_GPU_LEASE')
    if lease:
        return f'gpu_lease:{lease}'
    raise FinetuneError('production/smoke training needs HARNESS_RECEIPT or FOLD27_GPU_LEASE (run via trqam/train_queue.py)')


def train(a, fold, rows, fp, run, steps=STEPS):
    import jax, pickle
    from qam_prefix import PrefixQAMTrainer, to_qam_batch
    from trqam_prefix import RETENTION_DEFAULTS, migrate, inference_agent
    if jax.default_backend() != 'gpu':
        raise FinetuneError(f'expected GPU backend, got {jax.default_backend()}')
    base = PrefixQAMTrainer.load(a.init, a.init_step, feature_signature=FEATURE_SIGNATURE)
    agent = migrate(base.agent, SEED, RETENTION_DEFAULTS)
    resume = run / 'resume'; resume.mkdir(exist_ok=True)
    states = sorted(resume.glob('state_*.pkl'), key=lambda p: int(p.stem.split('_')[1]))
    start = 0
    if states:
        st = pickle.loads(states[-1].read_bytes())
        if st['fingerprint'] != fp:
            raise FinetuneError('resume state fingerprint differs; use a new --out')
        agent = agent.replace(network=st['network'], rng=st['rng'], lam=st['lam'], kl_ema=st['kl_ema'])
        start = st['step']
    rng = np.random.default_rng(SEED + start)
    n = len(rows['rewards'])
    with (run / 'progress.jsonl').open('a') as log:
        for step in range(start + 1, steps + 1):
            idx = rng.integers(n, size=BATCH)
            batch = to_qam_batch({k: v[idx] for k, v in rows.items()}, execute_steps=RW.EXECUTE, gamma=GAMMA)
            agent, info = agent.update(batch)
            if step % 1000 == 0 or step == steps:
                out = {k: float(v) for k, v in info.items()}
                if not all(np.isfinite(list(out.values()))):
                    raise FinetuneError(f'non-finite metrics at {step}')
                log.write(json.dumps({'step': step, 'time': time.time(), **out}) + '\n'); log.flush(); os.fsync(log.fileno())
            if step % SAVE_EVERY == 0 or step == steps:
                tmp = resume / f'.state_{step}.tmp'
                tmp.write_bytes(pickle.dumps({'step': step, 'fingerprint': fp, 'network': agent.network, 'rng': agent.rng,
                                              'lam': agent.lam, 'kl_ema': agent.kl_ema}))
                os.replace(tmp, resume / f'state_{step}.pkl')
                for old in sorted(resume.glob('state_*.pkl'), key=lambda p: int(p.stem.split('_')[1]))[:-2]:
                    old.unlink()      # keep current + previous (trqam_retention retention policy)
    final = PrefixQAMTrainer(inference_agent(agent, base.agent.config), gamma=GAMMA, data_label=f'fold27_{fold["fold_id"]}_{a.arm}')
    final.steps_done = steps
    ck = final.save(run / 'checkpoint', feature_signature=FEATURE_SIGNATURE)
    PrefixQAMTrainer.load(run / 'checkpoint', steps, feature_signature=FEATURE_SIGNATURE)   # reload gate
    return ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', type=Path, required=True); ap.add_argument('--fold', required=True)
    ap.add_argument('--arm', choices=RW.ARMS, required=True)
    ap.add_argument('--alpha', type=float); ap.add_argument('--beta', type=float)
    ap.add_argument('--potential', choices=POTENTIALS, default='ae16_k48',
                    help='ae16_k48 = task_classification AE clustering (user instruction, default); pca16_k48 = SIMPLER TRQAM recipe')
    ap.add_argument('--episodes', type=Path, required=True); ap.add_argument('--latents', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--init', type=Path); ap.add_argument('--init-step', type=int, default=500000)
    ap.add_argument('--reward-confirmation', type=Path); ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--smoke-steps', type=int, help='GPU smoke: N updates into <tag>_smoke, SMOKE_DONE.json (never DONE.json)')
    a = ap.parse_args()
    if a.smoke_steps is not None and not 0 < a.smoke_steps < STEPS:
        print('FINETUNE_FAILED --smoke-steps must be in (0, 50000)', file=sys.stderr); sys.exit(2)
    try:
        confirmation = reward_gate(a)          # fills recipe defaults before the run tag is formed
    except FinetuneError as e:
        print('FINETUNE_FAILED', repr(e), file=sys.stderr); sys.exit(2)
    tag = a.arm if a.arm == 'basic' else f'{a.arm}_a{a.alpha}' + (f'_b{a.beta}' if a.beta is not None else '')
    run = a.out / a.fold / (tag + ('_smoke' if a.smoke_steps else ''))
    run.mkdir(parents=True, exist_ok=True)
    try:
        spec = load_spec(a.spec)
        if not spec.get('production_ready'):
            raise FinetuneError('fold spec not production_ready')
        fold = next(f for f in spec['folds'] if f['fold_id'] == a.fold)
        from contract import load_checkpoint_contract, resolve_cached_checkpoint_dir
        codec = load_checkpoint_contract(resolve_cached_checkpoint_dir()).codec
        eps = load_fit_episodes(fold, a.episodes, a.latents, codec)
        # one AE potential per fold, shared by pbrs and mix (SIMPLER: both shaped arms used the same task clusters);
        # fit by whichever arm's dry-run comes first, reloaded by the other arm and by smoke/production
        rows, bundle, resid = build_rows(fold, eps, a.arm, a.alpha, a.beta, a.potential, a.out / a.fold)
        from qam_prefix import to_qam_batch
        to_qam_batch(rows, execute_steps=RW.EXECUTE, gamma=GAMMA)          # whole-table contract check
        if bundle is not None and bundle.get('kind') != 'ae16_k48' and not (run / 'potential.npz').exists():
            np.savez(run / 'potential.npz', **{k: np.asarray(v) for k, v in bundle.items() if k != 'disabled'},
                     disabled=np.asarray(bundle.get('disabled', '')))
        fp = fingerprint(a, fold, rows)
        summary = {'fold_id': a.fold, 'arm': a.arm, 'alpha': a.alpha, 'beta': a.beta, 'rows': int(len(rows['rewards'])),
                   'episodes': len(eps), 'fit_success': int(sum(e['success'] for e in eps)), 'reward_recompute_residual': resid,
                   'reward_mean': float(rows['rewards'].mean()), 'terminal_rows': int(rows['terminals'].sum()),
                   'potential': None if bundle is None else {'method': a.potential, 'fit_on': '160 fit rollouts only',
                                                             'disabled': bool(bundle.get('disabled'))},
                   'reward_confirmation': confirmation}
        if a.dry_run:
            (run / 'DRYRUN.json').write_text(json.dumps(summary, indent=1)); print(json.dumps(summary)); return
        receipt = launch_receipt()
        if a.init is None:
            raise FinetuneError('--init (common atomic65 checkpoint) required')
        summary['init_data_label'] = check_init(a.init, a.init_step)
        steps = a.smoke_steps or STEPS
        done = run / ('SMOKE_DONE.json' if a.smoke_steps else 'DONE.json')
        if done.exists():
            d = json.loads(done.read_text())
            if d['fingerprint'] == fp and sha(run / 'checkpoint' / f'params_{steps}.pkl') == d['params_sha256']:
                print('done', run); return
            raise FinetuneError(f'{done.name} exists but fingerprint/checkpoint differ')
        (run / 'fingerprint.json').write_text(json.dumps(fp, indent=1))
        train(a, fold, rows, fp, run, steps)
        done.write_text(json.dumps({**summary, 'fingerprint': fp, 'harness_receipt': receipt, 'smoke': bool(a.smoke_steps),
                                    'params_sha256': sha(run / 'checkpoint' / f'params_{steps}.pkl')}, indent=1))
    except Exception as e:
        (run / 'FAILED.json').write_text(json.dumps({'error': repr(e), 'trace': traceback.format_exc(), 'time': time.time()}, indent=1))
        print('FINETUNE_FAILED', repr(e), file=sys.stderr); sys.exit(2)


if __name__ == '__main__':
    main()

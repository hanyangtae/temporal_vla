"""finetune_fold v2 = v1 (frozen, imported unchanged: data, rewards, AE potential, recipe, gates, DONE/FAILED) with ONLY
the training-state persistence replaced (resume_state.py): full Flax numerical state + NumPy minibatch generator.
No recipe change, no critic reset: the init is the same migrate(common 500k) and every update is the same agent.update.

Extra (optional) --reuse-ae-from <v1 out root>: copy (never move) the v1 fold-level ae_potential.json + encoder npz into
the v2 fold dir before the run, so pbrs/mix keep the potential the v1 dry-run fit. Accepted only if (a) encoder sha verifies
(AEClusterPotential.load), (b) its episode ids == this fold's 160 fit ids (v1 build_rows check), and (c) for a shaped-arm
dry-run, rows and reward_mean equal the v1 dry-run of the same arm exactly (proves same latents/rows/potential). Otherwise
AE_REUSE_REJECTED.json + failure (the v2 fold dir is not used further until inspected). Provenance: AE_REUSED_FROM.json.
"""
import json, os, shutil, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parents[1] / 'groot_robocasa_adapter_20261021')]
import finetune_fold as FF     # noqa: E402  (frozen v1, imported unchanged)
import resume_state as RS      # noqa: E402

V2_CODE = {'finetune_fold_v2.py': Path(__file__).resolve(), 'resume_state.py': HERE / 'resume_state.py'}
_v1_fingerprint = FF.fingerprint


def fingerprint_v2(a, fold, rows):
    fp = _v1_fingerprint(a, fold, rows)
    fp['code'] = {**fp['code'], **{k: FF.sha(p) for k, p in V2_CODE.items()}}
    fp['trainer_version'] = 'v2_flax_state_dict_resume'
    return fp


def train_v2(a, fold, rows, fp, run, steps=FF.STEPS):
    import jax
    from qam_prefix import PrefixQAMTrainer, to_qam_batch
    from trqam_prefix import RETENTION_DEFAULTS, migrate, inference_agent
    if jax.default_backend() != 'gpu':           # no CPU escape hatch: smoke and production DONE are GPU-only
        raise FF.FinetuneError(f'expected GPU backend, got {jax.default_backend()}')
    base = PrefixQAMTrainer.load(a.init, a.init_step, feature_signature=FF.FEATURE_SIGNATURE)
    agent = migrate(base.agent, FF.SEED, RETENTION_DEFAULTS)     # template: same init as v1, also the restore target
    resume = run / 'resume_v2'; resume.mkdir(exist_ok=True)
    states = sorted(resume.glob('state_*.pkl'), key=lambda p: int(p.stem.split('_')[1]))
    np_rng, start = np.random.default_rng(FF.SEED), 0             # v1 bug (SEED + start on resume) not reproduced
    if states:
        agent, np_rng, start, sfp = RS.load_state(states[-1], agent)
        if sfp != fp:
            raise FF.FinetuneError('resume state fingerprint differs; use a new --out')
    to_batch = lambda b: to_qam_batch(b, execute_steps=FF.RW.EXECUTE, gamma=FF.GAMMA)
    log = (run / 'progress.jsonl').open('a')
    ledger = (run / 'resume_v2' / 'saves.jsonl').open('a')

    def on_step(step, ag, info, idx):
        if step % 1000 == 0 or step == steps:
            out = {k: float(v) for k, v in info.items()}
            if not all(np.isfinite(list(out.values()))):
                raise FF.FinetuneError(f'non-finite metrics at {step}')
            log.write(json.dumps({'step': step, 'time': time.time(), **out}) + '\n'); log.flush(); os.fsync(log.fileno())
        if step % FF.SAVE_EVERY == 0 or step == steps:
            p = resume / f'state_{step}.pkl'
            digest = RS.save_state(p, ag, np_rng, step, fp)
            ledger.write(json.dumps({'step': step, 'sha256': digest, 'time': time.time()}) + '\n'); ledger.flush()
            for old in sorted(resume.glob('state_*.pkl'), key=lambda q: int(q.stem.split('_')[1]))[:-2]:
                old.unlink()      # keep current + previous (trqam_retention retention policy); own resume files only
    try:
        agent, _ = RS.update_loop(agent, rows, np_rng, start, steps, FF.BATCH, to_batch, on_step)
    finally:
        log.close(); ledger.close()
    final = PrefixQAMTrainer(inference_agent(agent, base.agent.config), gamma=FF.GAMMA, data_label=f'fold27_{fold["fold_id"]}_{a.arm}')
    final.steps_done = steps
    ck = final.save(run / 'checkpoint', feature_signature=FF.FEATURE_SIGNATURE)
    PrefixQAMTrainer.load(run / 'checkpoint', steps, feature_signature=FF.FEATURE_SIGNATURE)   # reload gate
    return ck


def reuse_ae(src_root, out_root, fold_id):
    """Copy v1 fold-level AE files into the v2 fold dir (if absent). Returns provenance or None."""
    src, dst = Path(src_root) / fold_id, Path(out_root) / fold_id
    if (dst / 'AE_REUSE_REJECTED.json').exists():
        raise FF.FinetuneError(f'{dst}: AE reuse was rejected before; inspect before rerunning')
    files = ('ae_potential.json', 'ae_potential_encoder.npz')
    if (dst / files[0]).exists() or not (src / files[0]).exists():
        return None
    dst.mkdir(parents=True, exist_ok=True)
    prov = {'source_root': str(src_root), 'files': {}}
    for n in files:
        shutil.copy2(src / n, dst / n)
        prov['files'][n] = {'sha256': FF.sha(dst / n), 'source_sha256': FF.sha(src / n)}
    (dst / 'AE_REUSED_FROM.json').write_text(json.dumps(prov, indent=1))
    return prov


def verify_reuse(src_root, out_root, fold_id, tag):
    """After a shaped-arm dry-run in v2: rows/reward_mean must equal the v1 dry-run of the same tag."""
    dst = Path(out_root) / fold_id
    if not (dst / 'AE_REUSED_FROM.json').exists():
        return None
    old, new = Path(src_root) / fold_id / tag / 'DRYRUN.json', dst / tag / 'DRYRUN.json'
    if not old.exists():
        return 'no v1 dry-run for this arm (reuse verified by encoder sha + episode ids only)'
    o, n = json.loads(old.read_text()), json.loads(new.read_text())
    if (o['rows'], o['reward_mean'], o['episodes']) != (n['rows'], n['reward_mean'], n['episodes']):
        (dst / 'AE_REUSE_REJECTED.json').write_text(json.dumps({'v1': o, 'v2': n, 'tag': tag}, indent=1))
        raise FF.FinetuneError(f'AE reuse rejected: v1 vs v2 dry-run differ for {tag}')
    rec = json.loads((dst / 'AE_REUSED_FROM.json').read_text())
    rec.setdefault('verified_by_dry_run', {})[tag] = {'rows': n['rows'], 'reward_mean': n['reward_mean']}
    (dst / 'AE_REUSED_FROM.json').write_text(json.dumps(rec, indent=1))
    return 'verified'


def main():
    reuse_from = None
    if '--reuse-ae-from' in sys.argv:
        i = sys.argv.index('--reuse-ae-from'); reuse_from = Path(sys.argv[i + 1]); del sys.argv[i:i + 2]
    FF.train, FF.fingerprint = train_v2, fingerprint_v2
    if reuse_from is not None:
        g = lambda k: sys.argv[sys.argv.index(k) + 1]
        out, fold, arm = Path(g('--out')), g('--fold'), g('--arm')
        try:
            if arm != 'basic':
                reuse_ae(reuse_from, out, fold)
        except FF.FinetuneError as e:
            print('FINETUNE_FAILED', repr(e), file=sys.stderr); sys.exit(2)
        FF.main()                     # exits non-zero on its own failures
        if arm != 'basic' and '--dry-run' in sys.argv:
            tag = {'pbrs': 'pbrs_a0.1', 'mix': 'mix_a0.1_b0.0396'}[arm]
            try:
                print(json.dumps({'ae_reuse': verify_reuse(reuse_from, out, fold, tag)}))
            except FF.FinetuneError as e:
                print('FINETUNE_FAILED', repr(e), file=sys.stderr); sys.exit(2)
        return
    FF.main()


if __name__ == '__main__':
    main()

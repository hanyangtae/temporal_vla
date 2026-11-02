"""Restore a QAM / TRQAM checkpoint into the torch serve actor (actor.npz + actor.json + runtime_provenance.json).

Runs on CPU in .venvs/rl2_iid (JAX_PLATFORMS=cpu). Export itself = groot_robocasa_adapter_20261021/qam_torch_actor.export_actor
(unchanged); this file only decides WHICH checkpoint is admissible and writes the selection receipt it requires.

    # production common atomic65 (fixed final 500k step)
    restore_actor.py common --run outputs/rl2_vla/atomic65_common_qam_work_20261029/scratch_run --out <actors>/common_qam65
    # production fold TRQAM (basic / pbrs / mix); --common-run pins the init chain
    restore_actor.py trqam --run <trqam root>/<fold>/<tag> --fold <fold> --arm basic|pbrs|mix --common-run <scratch_run> --out <actors>/<fold>/<arm>
    # integration smoke ONLY (diagnostic actor, refused by every production path)
    restore_actor.py smoke --run outputs/rl2_vla/atomic65_common_qam_work_20261029/scratch_smoke --step 20 --out <smoke root>/actors/smoke20
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

TRQAM_STEPS = 50_000


def _ckpt(run, step):
    return Path(run) / 'checkpoints' / f'step_{step:07d}'


def check_common_run(run):
    """Full fresh production atomic65 run at its fixed final 500k step; no smoke, no target-split data."""
    run = Path(run)
    C.need((run / 'COMPLETE.json').is_file(), f'{run}: COMPLETE.json missing (training not finished)')
    rm = json.loads((run / 'run_manifest.json').read_text())
    C.need(rm.get('mode') == 'production_atomic65', f'{run}: mode {rm.get("mode")} is not production_atomic65')
    C.need(int(rm.get('steps', -1)) == C.COMMON_STEPS, f'{run}: steps {rm.get("steps")} != {C.COMMON_STEPS}')
    C.need(rm.get('warm_start_step') is None and rm.get('optimizer_state') == 'fresh', 'common run is not fresh')
    done = json.loads((run / 'COMPLETE.json').read_text())
    C.need(int(done['steps']) == C.COMMON_STEPS and int(done['selection']['step']) == C.COMMON_STEPS, 'selection is not the final 500k step')
    C.need(done['selection']['archive'].get('verified') is True, 'final checkpoint archive not hash-verified')
    manifest = C.R / C.COMMON_QAM_WORK / 'metadata/manifest.json'
    C.need(manifest.is_file() and C.sha256_path(manifest) == rm['manifest_sha256'], 'production manifest missing or sha differs')
    check_manifest_leakage(json.loads(manifest.read_text()))
    ck = _ckpt(run, C.COMMON_STEPS)
    flags = json.loads((ck / f'flags_{C.COMMON_STEPS}.json').read_text())
    C.need('smoke' not in str(flags.get('data_label', '')).lower() and int(flags['steps_done']) == C.COMMON_STEPS,
           f'checkpoint flags not production: {flags.get("data_label")} {flags.get("steps_done")}')
    return ck, C.COMMON_STEPS


def check_manifest_leakage(m):
    """Common QAM data = official pretrain human demos only; no RoboCasa target split, no fold rollouts."""
    C.need(m.get('experiment') == 'atomic65_common_qam_20261029' and m.get('mode') == 'full', 'manifest is not the full atomic65 manifest')
    C.need(int(m['n_tasks']) == 65, 'manifest task count != 65')
    for e in m['episodes']:
        C.need(e['episode_id'].startswith('robocasa/v1.0/pretrain/') and e['split'] in ('train', 'val'),
               f'non-pretrain or test-split episode in common QAM manifest: {e["episode_id"]} {e["split"]}')


def check_fold_leakage(fold):
    fit = {r['sig'] for r in fold['fit']}
    ev = {r['sig'] for r in fold['eval']}
    C.need(len(fold['fit']) == 160 and len(fold['eval']) == 40, 'fold sizes != 160/40')
    C.need(not fit & ev, f'{fold["fold_id"]}: fit/eval rollouts overlap')
    C.need(all(int(r['j']) != int(fold['target_j']) for r in fold['fit']), 'fit contains target j')


def check_trqam_run(run, fold_id, arm, common_run):
    run = Path(run)
    C.need(arm in ('basic', 'pbrs', 'mix'), f'arm {arm}')
    done = json.loads((run / 'DONE.json').read_text())
    fp = done['fingerprint']
    C.need(done['fold_id'] == fold_id == fp['fold_id'] and done['arm'] == arm == fp['arm'], 'DONE fold/arm mismatch')
    C.need(int(done['episodes']) == 160 and int(fp['steps']) == TRQAM_STEPS, 'TRQAM run is not the 160-fit 50k production run')
    C.need(done.get('harness_receipt'), 'TRQAM run lacks a harness receipt')
    common_ck, step = check_common_run(common_run)
    C.need(fp['init'] and int(fp['init']['step']) == step and
           fp['init']['params_sha256'] == C.sha256_path(common_ck / f'params_{step}.pkl'), 'TRQAM init != verified common 500k checkpoint')
    ck = run / 'checkpoint'
    C.need(C.sha256_path(ck / f'params_{TRQAM_STEPS}.pkl') == done['params_sha256'], 'TRQAM params sha differs from DONE.json')
    spec = json.loads((C.R / C.FOLD_SPEC).read_text())
    check_fold_leakage(next(f for f in spec['folds'] if f['fold_id'] == fold_id))
    return ck, TRQAM_STEPS, done


def export(ck, step, out, *, production, provenance):
    sys.path.insert(0, str(C.ADAPTER))
    from qam_prefix import PrefixQAMTrainer
    from qam_torch_actor import export_actor
    out = Path(out)
    C.need(not out.exists(), f'{out} exists; actors are write-once')
    tr = PrefixQAMTrainer.load(ck, step, feature_signature=C.FEATURE_SIGNATURE)
    C.need(tr.steps_done == step, 'loaded steps_done mismatch')
    selection = None
    if production:
        selection = out.parent / f'{out.name}.selection.json'
        C.write_json_new(selection, {'selected_step': step, 'production_run': True, 'val_metric': 'fixed final step (no selection)',
                                     'selected_checkpoint': {'dir': str(ck), 'files': {f'params_{step}.pkl': {'sha256': C.sha256_path(ck / f'params_{step}.pkl')}}}})
    meta = export_actor(tr, ck, out, selection=selection, diagnostic=not production)
    C.need(bool(meta['diagnostic_only']) is (not production), 'diagnostic flag inconsistent')
    C.write_json_new(out / 'runtime_provenance.json', {**provenance, 'production': production, 'smoke': not production,
                                                       'actor_sha256': C.sha256_path(out / 'actor.npz')})
    print(json.dumps({'out': str(out), 'actor_sha256': C.sha256_path(out / 'actor.npz'), 'production': production}))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='kind', required=True)
    p = sub.add_parser('common'); p.add_argument('--run', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('trqam'); p.add_argument('--run', type=Path, required=True); p.add_argument('--fold', required=True)
    p.add_argument('--arm', required=True); p.add_argument('--common-run', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('smoke'); p.add_argument('--run', type=Path, required=True); p.add_argument('--step', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(argv)
    if a.kind == 'common':
        ck, step = check_common_run(a.run)
        export(ck, step, a.out, production=True, provenance={'kind': 'common', 'common_steps': step, 'run': C.rel(a.run),
                                                              'params_sha256': C.sha256_path(ck / f'params_{step}.pkl')})
    elif a.kind == 'trqam':
        ck, step, done = check_trqam_run(a.run, a.fold, a.arm, a.common_run)
        export(ck, step, a.out, production=True, provenance={
            'kind': f'trqam_{a.arm}', 'fold_id': a.fold, 'common_steps': C.COMMON_STEPS, 'trqam_steps': step, 'run': C.rel(a.run),
            'alpha': done['alpha'], 'beta': done['beta'], 'params_sha256': done['params_sha256'],
            'init_params_sha256': done['fingerprint']['init']['params_sha256']})
    else:
        C.need('smoke' in str(a.out).lower(), 'smoke actor output path must contain "smoke"')
        flags = json.loads((_ckpt(a.run, a.step) / f'flags_{a.step}.json').read_text())
        C.need('smoke' in flags['data_label'].lower(), 'smoke export only from a smoke-labelled checkpoint')
        ck = _ckpt(a.run, a.step)
        export(ck, a.step, a.out, production=False, provenance={'kind': 'smoke', 'common_steps': a.step, 'run': C.rel(a.run),
                                                                'params_sha256': C.sha256_path(ck / f'params_{a.step}.pkl')})


if __name__ == '__main__':
    try:
        main()
    except C.RuntimeContractError as e:
        print('RESTORE_REFUSED', e, file=sys.stderr); sys.exit(2)

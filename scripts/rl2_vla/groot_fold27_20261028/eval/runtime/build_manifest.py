"""Freeze one fold's eval manifest: the same 40 archive rows, per-noise-range plans, MIL + actor pins, code hashes.

    build_manifest.py --fold OpenDrawer-left_s2_j2 --mil-run <fold MIL run>/OpenDrawer-left_s2_j2/attn/raw/s0 \
        --actor mil_qam65=<actors>/common_qam65 --actor mil_trqam_basic=<actors>/OpenDrawer-left_s2_j2/basic ... \
        [--out-root outputs/eval/robocasa/groot_n15/fold27_mil_qam_20261030]
Smoke (smoke_integration.py calls build(..., smoke=True)): smoke_* arms, any MIL ckpt, row subset, SMOKE_ROOT only.
Every pin missing -> refuse. Writes <out-root>/<fold_id>/manifest.json (write-once; identical rebuild accepted).
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
from folds import N20_PLAN, N40_PLAN, canonical_machine, noise_range, validate_fold  # noqa: E402

RUNTIME_FILES = ('common.py', 'gated_action.py', 'serve_mil_qam.py', 'collect_task.py', 'run_eval_arm.py', 'build_manifest.py')
ADAPTER_FILES = ('composition.py', 'qam_torch_actor.py', 'contract.py', 'task_language_guard.py', 'readonly_assets.py',
                 'extract_opendrawer_latents.py', 'paired_eval_audited.py', 'baseline_data.py')
OTHER_FILES = ('scripts/rl2_vla/groot_fold27_20261028/mil/mil_gate.py', 'scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py',
               'scripts/serve/lerobot.py', 'configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml')


def code_hashes():
    out = {f'runtime/{f}': C.sha256_path(HERE / f) for f in RUNTIME_FILES}
    out.update({f'adapter/{f}': C.sha256_path(C.ADAPTER / f) for f in ADAPTER_FILES})
    out.update({f: C.sha256_path(C.R / f) for f in OTHER_FILES})
    return out


def check_code_hashes(pinned):
    now = code_hashes()
    C.need(set(now) == set(pinned), 'code hash key set differs')
    bad = [k for k in now if now[k] != pinned[k]]
    C.need(not bad, f'code changed since manifest: {bad}')


def expected_plan(fold, n):
    return {'n0_9': fold['source_plan_n0_9'], 'n10_19': N20_PLAN, 'n20_39': N40_PLAN}[noise_range(n)]


def fold_rows(fold, rows_subset=None):
    validate_fold(fold)
    key, scene = fold['instruction'], int(fold['scene'])
    rows, idents, plans = [], {}, {}
    ev = sorted(fold['eval'], key=lambda r: r['n'])
    C.need([r['n'] for r in ev] == list(range(40)), f'{fold["fold_id"]}: eval rows are not n0..39')
    C.need({int(r['j']) for r in ev} == {int(fold['target_j'])}, 'eval rows not all target j')
    C.need(sum(int(r['success']) for r in ev) == fold['expected']['eval_success'], 'archive success differs from selection')
    for r in ev:
        if rows_subset is not None and r['n'] not in rows_subset:
            continue
        C.need(r.get('seed_source') == 'archive_meta', f'n{r["n"]}: seeds not archive-confirmed')
        C.need(r['plan_id'] == expected_plan(fold, r['n']), f'n{r["n"]}: plan {r["plan_id"]} != range plan {expected_plan(fold, r["n"])}')
        path, plan = C.load_plan(r['plan_id'])
        C.check_row_against_plan(plan, key, scene, r)
        idents[r['plan_id']] = C.task_identity(plan, key, scene)
        plans[r['plan_id']] = {'path': C.rel(path), 'sha256': C.sha256_path(path)}
        rows.append({'n': int(r['n']), 'j': int(r['j']), 'plan_id': r['plan_id'], 'env_seed': int(r['env_seed']),
                     'inference_seed': int(r['inference_seed']), 'base_success': int(r['success']), 'sig': r['sig'],
                     'pkl_sha256': r['pkl_sha256'], 'noise_range': noise_range(r['n'])})
    C.need(rows, 'no rows selected')
    first = next(iter(idents.values()))
    C.need(all(v == first for v in idents.values()), f'task identity differs between noise-range plans: {idents}')
    return rows, first, plans


def actor_pin(arm, path, fold_id, smoke):
    mode, kind = (C.SMOKE_ARMS if smoke else C.PRODUCTION_ARMS)[arm]
    if kind is None:
        C.need(path is None, f'{arm} takes no actor')
        return {'mode': mode, 'actor': None}
    C.need(path is not None, f'{arm}: actor pin missing')
    path = Path(path)
    prov = json.loads((path / 'runtime_provenance.json').read_text())
    meta = json.loads((path / 'actor.json').read_text())
    C.need(prov['actor_sha256'] == C.sha256_path(path / 'actor.npz'), 'actor sha mismatch')
    if smoke:
        C.need(prov['smoke'] is True, 'smoke arms use the smoke actor only')
    else:
        C.need(prov['production'] is True and prov['smoke'] is False and not meta['diagnostic_only'], f'{arm}: actor not production')
        C.need(prov['kind'] == kind and int(prov['common_steps']) == C.COMMON_STEPS, f'{arm}: actor kind {prov["kind"]} != {kind}')
        if kind.startswith('trqam'):
            C.need(prov['fold_id'] == fold_id, f'{arm}: TRQAM actor trained for {prov["fold_id"]}, not {fold_id}')
    return {'mode': mode, 'actor': C.rel(path), 'actor_sha256': prov['actor_sha256'],
            'actor_json_sha256': C.sha256_path(path / 'actor.json'), 'provenance': prov}


def mil_pin(mil_run, fold, smoke):
    p = Path(mil_run)
    model = p / 'model.pt' if p.is_dir() else p
    C.need(model.is_file(), f'MIL model missing: {model}')
    sha = C.sha256_path(model)
    pin = {'model': C.rel(model), 'sha256': sha, 'target_j': int(fold['target_j']), 'fold_contract': None}
    fc = model.parent / 'FOLD_CONTRACT.json'
    if not smoke:
        C.need(fc.is_file(), 'production MIL requires FOLD_CONTRACT.json (fold_mil train output); shared48 is smoke-only')
        d = json.loads(fc.read_text())
        C.need(d['fold_id'] == fold['fold_id'] and int(d['target_j']) == int(fold['target_j']) and d['model_sha256'] == sha
               and d['shared_checkpoint_reused'] is False, 'FOLD_CONTRACT does not match this fold')
        pin['fold_contract'] = {'path': C.rel(fc), 'sha256': C.sha256_path(fc)}
    return pin


def build(fold_id, mil_run, actors, out_root, *, smoke=False, rows_subset=None, spec_path=None):
    spec_path = Path(spec_path or C.R / C.FOLD_SPEC)
    spec = json.loads(spec_path.read_text())
    C.need(spec.get('production_ready') is True, 'fold spec not production_ready')
    fold = next((f for f in spec['folds'] if f['fold_id'] == fold_id), None)
    C.need(fold is not None, f'unknown fold {fold_id}')
    out_root = Path(out_root).resolve()
    if smoke:
        C.need(str(out_root).startswith(str((C.R / C.SMOKE_ROOT).resolve())), 'smoke manifests only under SMOKE_ROOT')
    else:
        C.need(rows_subset is None, 'production evaluates all 40 rows')
        C.need(not str(out_root).startswith(str((C.R / C.SMOKE_ROOT).resolve())), 'production manifest under SMOKE_ROOT')
        C.need(set(actors) == set(C.PRODUCTION_ARMS), f'production arms must be exactly {sorted(C.PRODUCTION_ARMS)}')
    host = fold['eval_host']
    C.need(C.HOSTS[host]['archive'] == canonical_machine(fold['archive_machine']), 'eval host != archive collection machine')
    rows, ident, plans = fold_rows(fold, rows_subset)
    m = {'version': 'fold27_mil_qam_runtime_v1', 'smoke': bool(smoke), 'fold_id': fold_id, 'instruction': fold['instruction'],
         'scene': int(fold['scene']), 'target_j': int(fold['target_j']), 'host': host, 'archive_machine': fold['archive_machine'],
         'fold_spec': {'path': C.rel(spec_path), 'sha256': C.sha256_path(spec_path)}, 'identity': ident, 'plans': plans,
         'rows': rows, 'mil': mil_pin(mil_run, fold, smoke),
         'arms': {arm: actor_pin(arm, actors.get(arm), fold_id, smoke) for arm in sorted(actors)},
         'policy': {'qam_weight': C.QAM_WEIGHT, 'n_action_steps': C.N_ACTION_STEPS, 'chunk': C.CHUNK, 'max_steps': C.MAX_STEPS,
                    'flow_steps': C.FLOW_STEPS, 'feature_signature': C.FEATURE_SIGNATURE, 'tc_commit': C.TC_COMMIT,
                    'gate': 'MIL w_F > 0.5 every inference; composed denoise from the same initial noise when fired'},
         'baseline': 'archive success of the same rows (no new base sweep)', 'code': code_hashes()}
    C.need(all(a in (C.SMOKE_ARMS if smoke else C.PRODUCTION_ARMS) for a in m['arms']), 'arm set not allowed in this mode')
    path = out_root / fold_id / 'manifest.json'
    C.write_json_new(path, m)
    return path, m


def parse_actors(items):
    out = {}
    for it in items or []:
        arm, _, path = it.partition('=')
        C.need(arm not in out, f'duplicate actor {arm}')
        out[arm] = Path(path) if path else None
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', required=True); ap.add_argument('--mil-run', type=Path, required=True)
    ap.add_argument('--actor', action='append', help='arm=actor_dir (repeat)')
    ap.add_argument('--out-root', type=Path, default=C.R / C.EVAL_ROOT)
    a = ap.parse_args(argv)
    path, m = build(a.fold, a.mil_run, parse_actors(a.actor), a.out_root)
    print(json.dumps({'manifest': str(path), 'sha256': C.sha256_path(path), 'rows': len(m['rows']), 'arms': sorted(m['arms'])}))


if __name__ == '__main__':
    try:
        main()
    except C.RuntimeContractError as e:
        print('MANIFEST_REFUSED', e, file=sys.stderr); sys.exit(2)

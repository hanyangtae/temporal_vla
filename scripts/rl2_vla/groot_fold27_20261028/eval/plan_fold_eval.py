"""Eval dry-run planner (CPU, no launch): per-fold "same 40" manifests + host capacity schedule.

Each fold's eval = target j x n0..39 with the exact archive (plan_id, env_seed, inference_seed); runs only on the
collection machine (kanu / srv48=worker1 / srv50=worker2; replay is bit-exact only there, docs/04).
Arms (user contract): base (archive result; replay QA on qa_noise must be byte-identical), mil_qam65 (common atomic65
QAM gated by the fold's own MIL), mil_trqam_basic / mil_trqam_pbrs / mil_trqam_mix (fold fine-tunes, same MIL gate).
Pins required before launch: fold MIL model.pt sha (fold_mil train), actor sha per arm (export), plan sha per noise range.

    python plan_fold_eval.py --spec fold_spec.json --out outputs/analysis/groot_fold27_20261028/eval_plan [--arms ...]
"""
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from folds import load_spec, validate_fold, noise_range, N20_PLAN, N40_PLAN   # noqa: E402

DEFAULT_ARMS = ('base', 'mil_qam65', 'mil_trqam_basic', 'mil_trqam_pbrs', 'mil_trqam_mix')
# docs/05_gpu_server_rules.md: all-session caps
HOSTS = {'kanu': {'gpus': 3, 'serves_per_gpu': 2, 'port0': 8860}, 'srv48': {'gpus': 1, 'serves_per_gpu': 6, 'port0': 8890},
         'srv50': {'gpus': 1, 'serves_per_gpu': 6, 'port0': 8890}}
PLAN_DIRS = {N20_PLAN: '.claude/worktrees/collect-topup/configs/collect/n15_target9_n20_20261013/collection_plan.json',
             N40_PLAN: '.claude/worktrees/collect-topup/configs/collect/n15_target9_n40_20261025/collection_plan.json'}
QA_NOISE = (0, 10, 20)       # one base replay per noise range (byte-identity QA); decision D-QA may widen


class PlanError(RuntimeError):
    pass


def fold_manifest(f, arms):
    validate_fold(f)
    rows = []
    for r in sorted(f['eval'], key=lambda r: r['n']):
        if r.get('seed_source') != 'archive_meta':
            raise PlanError(f'{f["fold_id"]}: eval row n{r["n"]} not archive-confirmed')
        rows.append({'plan_id': r['plan_id'], 'noise_range': noise_range(r['n']), 'j': r['j'], 'n': r['n'],
                     'env_seed': r['env_seed'], 'inference_seed': r['inference_seed'], 'base_success': r['success'],
                     'sig': r['sig']})
    if sum(x['base_success'] for x in rows) != f['expected']['eval_success']:
        raise PlanError(f'{f["fold_id"]}: base success differs from selection')
    plan_json = {rng: PLAN_DIRS.get(pid, f'<source plan {pid} collection_plan.json — locate before launch>')
                 for rng, pid in (('n0_9', f['source_plan_n0_9']), ('n10_19', N20_PLAN), ('n20_39', N40_PLAN))}
    return {'fold_id': f['fold_id'], 'instruction': f['instruction'], 'scene': f['scene'], 'target_j': f['target_j'],
            'host': f['eval_host'], 'archive_machine': f['archive_machine'], 'arms': list(arms),
            'expected_per_arm': 40, 'new_episodes': 40 * (len(arms) - 1) + len(QA_NOISE) * ('base' in arms),
            'qa_noise': list(QA_NOISE), 'plan_json': plan_json, 'test_rows': rows,
            'pins_required': {'mil_model_sha256': None, **{f'actor_sha256/{a}': None for a in arms if a != 'base'}}}


def schedule(manifests, minutes_per_episode=None):
    per_host = {}
    for m in manifests:
        per_host.setdefault(m['host'], []).append(m['fold_id'])
    out = {}
    for h, fs in per_host.items():
        eps = sum(m['new_episodes'] for m in manifests if m['host'] == h)
        slots = HOSTS[h]['gpus'] * HOSTS[h]['serves_per_gpu']
        out[h] = {'folds': fs, 'episodes': eps, 'max_parallel_serves': slots, 'episodes_per_slot': -(-eps // slots),
                  'ports': [HOSTS[h]['port0'] + i for i in range(slots)],
                  'hours_if_minutes_per_episode': None if minutes_per_episode is None else round(eps / slots * minutes_per_episode / 60, 1)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--arms', nargs='+', default=list(DEFAULT_ARMS)); ap.add_argument('--minutes-per-episode', type=float)
    a = ap.parse_args()
    spec = load_spec(a.spec)
    if not spec.get('production_ready'):
        raise PlanError('fold spec not production_ready')
    if 'base' not in a.arms:
        raise PlanError('base arm (archive + replay QA) is mandatory')
    ms = [fold_manifest(f, a.arms) for f in spec['folds']]
    a.out.mkdir(parents=True, exist_ok=True)
    for m in ms:
        p = a.out / f'{m["fold_id"]}.manifest.json'
        if p.exists() and json.loads(p.read_text()) != m:
            raise PlanError(f'{p} exists with different content')
        p.write_text(json.dumps(m, indent=1))
    summary = {'folds': len(ms), 'arms': a.arms, 'new_episodes_total': sum(m['new_episodes'] for m in ms),
               'schedule': schedule(ms, a.minutes_per_episode), 'status': 'DRY_RUN_PLAN_ONLY (no launch)'}
    (a.out / 'PLAN_SUMMARY.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ('folds', 'new_episodes_total')}))


if __name__ == '__main__':
    try:
        main()
    except PlanError as e:
        print('PLAN_ERROR', e, file=sys.stderr); sys.exit(2)

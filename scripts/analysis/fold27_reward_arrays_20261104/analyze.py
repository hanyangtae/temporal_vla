"""Distribution analysis of reconstructed Fold27 basic vs MIX per-inference rewards.

    <python> scripts/analysis/fold27_reward_arrays_20261104/analyze.py --derived <out-root of reconstruct.py> \
        --out outputs/analysis/fold27_reward_arrays_20261104/<run> [--expect-folds 18]

Only folds under <derived>/sha_matched/ whose receipt has status sha_matched and every gate true are analysed and
pooled. Folds under <derived>/contract_failure/ are listed separately with their errors/gates and NEVER pooled.
Arrays are read with allow_pickle=False. The output directory must not exist (no overwrite).
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402

LOAD_KEYS = ('basic', 'mix', 'alpha_term', 'beta_cost', 'delta', 'episode_index', 'terminal', 'final', 'success')


def load_matched(derived):
    folds, rejected = {}, {}
    for d in sorted((derived / 'sha_matched').glob('*/')):
        r = json.loads((d / 'receipt.json').read_text())
        gates = r.get('sha', {}).get('gates', {})
        if r.get('status') != 'sha_matched' or not gates or not all(gates.values()):
            rejected[d.name] = 'receipt in sha_matched/ but status/gates not all true'
            continue
        with np.load(d / 'reward_arrays.npz', allow_pickle=False) as z:
            folds[d.name] = {k: z[k] for k in LOAD_KEYS}
        folds[d.name]['_receipt'] = r
    failures = {}
    for d in sorted((derived / 'contract_failure').glob('*/')):
        r = json.loads((d / 'receipt.json').read_text())
        failures[d.name] = {'error': r.get('error'), 'gates': r.get('sha', {}).get('gates')}
    return folds, rejected, failures


def pool(folds):
    out, off = {k: [] for k in LOAD_KEYS}, 0
    for name in sorted(folds):
        f = folds[name]
        for k in LOAD_KEYS:
            out[k].append(f[k] + off if k == 'episode_index' else f[k])
        off += int(f['episode_index'].max()) + 1
    return {k: np.concatenate(v) for k, v in out.items()}


def fmt(x, nd=4):
    return '—' if x is None else f'{x:.{nd}g}'


def markdown(res):
    L = ['# Fold27 basic vs MIX per-inference reward 분포', '',
         f"SHA 일치 fold {len(res['folds'])}개만 집계. 계약 실패 fold {len(res['contract_failure'])}개는 별도 표(집계 제외).", '',
         'delta = mix − basic = α·(anchor − φ) − β·C (α=0.1, β=0.0396). cancellation = 1 − |Σdelta| / Σ|delta|.',
         '비율 = |delta|/|basic|, basic=0 행은 제외하고 별도 표기.', '',
         '| fold | rows | |Δ| p0 | p50 | p90 | p95 | p99 | p100 | ΣΔ | Σ|Δ| | +/−/0 | cancel | Σ|Δ|/Σ|basic| (basic≠0) | basic=0 행 |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for name, rep in list(res['folds'].items()) + [('**POOLED**', res['pooled'])] if res['folds'] else []:
        a = rep['all']; q = a['abs_delta']; s = a['sign_counts']
        L.append(f"| {name} | {a['rows']} | " + ' | '.join(fmt(q[f'p{p}']) for p in core.PCTS) +
                 f" | {fmt(a['sums']['delta'])} | {fmt(a['sums']['abs_delta'])} | {s['pos']}/{s['neg']}/{s['zero']} | "
                 f"{fmt(a['cancellation'])} | {fmt(a['ratio_nonzero_basic']['sum_abs_delta_over_sum_abs_basic'])} | "
                 f"{a['zero_basic']['rows']} |")
    if res['folds']:
        p = res['pooled']
        L += ['', '## pooled 분해', '',
              '| 부분집합 | rows | Σα항 | Σβ비용 | ΣΔ | Σ|Δ| | cancel | |Δ| p50 | p99 |', '|---|---|---|---|---|---|---|---|---|']
        subsets = [('all', p['all'])] + [(f'outcome={k}', v) for k, v in p['by_outcome'].items()] + \
                  [(f'row={k}', v) for k, v in p['by_row'].items()]
        for k, a in subsets:
            L.append(f"| {k} | {a['rows']} | {fmt(a['sums']['alpha_term'])} | {fmt(a['sums']['beta_cost'])} | "
                     f"{fmt(a['sums']['delta'])} | {fmt(a['sums']['abs_delta'])} | {fmt(a['cancellation'])} | "
                     f"{fmt(a['abs_delta']['p50'])} | {fmt(a['abs_delta']['p99'])} |")
        z = p['all']['zero_basic']
        L += ['', f"basic=0 행 (비율 제외): {z['rows']}행, ΣΔ={fmt(z['sum_delta'])}, Σ|Δ|={fmt(z['sum_abs_delta'])}, "
                  f"|Δ| p50={fmt(z['abs_delta']['p50'])} p100={fmt(z['abs_delta']['p100'])}"]
    L += ['', '## 계약 실패 / 미집계 (집계 제외)', '']
    if not res['contract_failure'] and not res['rejected'] and not res['missing']:
        L.append('없음')
    for k, v in res['contract_failure'].items():
        L.append(f"- {k}: {v['error'] or ''} gates={v['gates']}")
    for k, v in res['rejected'].items():
        L.append(f'- {k}: {v}')
    if res['missing']:
        L.append(f"- 기대 {res['expect_folds']}개 중 미도착: {res['expect_folds'] - len(res['folds']) - len(res['contract_failure'])}개")
    return '\n'.join(L) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--derived', type=Path, nargs='+', required=True, help='one or more reconstruct.py out-roots')
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--expect-folds', type=int, default=18)
    a = ap.parse_args(argv)
    if a.out.exists():
        print(f'REFUSED: {a.out} exists', file=sys.stderr); return 3
    folds, rejected, failures = {}, {}, {}
    for d in a.derived:
        f, r, c = load_matched(d)
        for src, dst in ((f, folds), (r, rejected), (c, failures)):
            dup = set(src) & set(dst)
            if dup:
                raise SystemExit(f'duplicate fold across derived roots: {sorted(dup)}')
            dst.update(src)
    both = set(folds) & set(failures)
    if both:
        raise SystemExit(f'fold both sha_matched and contract_failure: {sorted(both)}')
    res = {'expect_folds': a.expect_folds, 'folds': {k: core.fold_report(v) for k, v in sorted(folds.items())},
           'pooled': core.fold_report(pool(folds)) if folds else None,
           'contract_failure': failures, 'rejected': rejected,
           'missing': len(folds) + len(failures) < a.expect_folds,
           'receipts': {k: {'host': v['_receipt'].get('host'), 'sha': v['_receipt']['sha'],
                            'outputs': v['_receipt'].get('outputs')} for k, v in folds.items()}}
    a.out.mkdir(parents=True)
    (a.out / 'analysis.json').write_text(json.dumps(res, indent=1))
    (a.out / 'REPORT.md').write_text(markdown(res))
    print(json.dumps({'sha_matched': len(folds), 'contract_failure': len(failures), 'rejected': len(rejected),
                      'out': str(a.out)}))
    return 0


if __name__ == '__main__':
    sys.exit(main())

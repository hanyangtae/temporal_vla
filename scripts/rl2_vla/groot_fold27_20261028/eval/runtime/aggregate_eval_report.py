"""READ-ONLY fold27 eval aggregation (stdlib only). Original results are never modified; outputs are NEW files only.

  snapshot  run on each host against its eval root: every production manifest (<root>/<fold>/manifest.json, smoke
            manifests skipped and listed), every arm in it. Rows come from run_eval_arm.load_progress (unchanged: re-hashes
            every finished row; any inconsistency is RECORDED as load_error for that arm, never hidden) and are audited
            with verify_fold.audit_decisions (unchanged: MIL-gated per-inference decision log). Rows named in an audit
            error are marked audit_ok=false and EXCLUDED from all rates, but listed. Unfinished rows (no progress line)
            are not counted; arm completeness (rows_done / planned rows, RUN_COMPLETE) is reported.
  merge     combine host snapshots -> report.json, per_policy.csv, per_episode.csv, report.md in a NEW --out-dir:
            per host x policy and global per policy: N (audited finished rows), success, SR, plannedN (given), coverage,
            archived baseline success on the SAME rows, rescue (base 0 -> 1), damage (base 1 -> 0), fired/total
            inferences, composed rows, failed-audit rows, load errors; plus the COMMON SUBSET of (host, fold, n) audited in
            every policy, with the same statistics per policy on that subset.
            Policies are the EXPECTED production set (common.PRODUCTION_ARMS, 4 arms, or --policies): a policy with no
            rows still appears (N=0) and the common subset is then reported INCOMPLETE (n_keys 0, missing listed) --
            never silently computed over the policies that happen to exist. Unexpected arm names are listed and excluded.
            Duplicate snapshots (same host / root / file) and duplicate (fold, policy, n) rows are refused (MergeError).
            Pins (MIL model/sha256/fold-contract sha256, actor path/sha256/json sha256/provenance kind) come from the
            manifest; a production arm missing any of them gets pin_error (its rows excluded and listed).

    python3 aggregate_eval_report.py snapshot --root <eval root> --host srv48 --out <new snapshot.json>
    python3 aggregate_eval_report.py merge --snapshot a.json b.json c.json --planned kanu=240 srv48=360 srv50=480 \
        --planned-total 1080 --out-dir outputs/analysis/groot_fold27_20261028/eval_report_<utc>
"""
import argparse, csv, json, re, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                   # noqa: E402  (unchanged)
from run_eval_arm import load_progress               # noqa: E402  (unchanged)
from verify_fold import audit_decisions              # noqa: E402  (unchanged)

PIN_KEYS = ('actor', 'actor_sha256', 'actor_json_sha256')
N_RE = re.compile(r'^n(\d+)\b')
LIMITATIONS = [
    'Archived baseline = original collection outcome of the same rows (manifest base_success). Equivalence of the archived '
    'initialisation to the current eval initialisation is NOT established by this tool (see init_contract work); '
    'rescue/damage are paired diagnostics, not causal effects.',
    'AE / cluster configuration is not stored in eval manifests; this report identifies policies by arm name, mode and '
    'actor sha256 only. The AE64/K48 statement comes from main, not from this tool.',
    'Video/frame content is not audited here (only per-inference decision logs, row file hashes and success flags).',
    'Partial folds/arms are included row-wise; plannedN/coverage show how far each host is from its plan.',
    'Folds of different hosts/scenes share no rows but use overlapping episode pools; no cross-fold independence claim.',
]


def snapshot(root, host, now=time.time):
    root = Path(root)
    out = {'kind': 'fold27_eval_snapshot_v1', 'host': host, 'root': str(root), 'time': now(), 'folds': [], 'skipped_smoke': []}
    for mp in sorted(root.glob('*/manifest.json')):
        m = json.loads(mp.read_text())
        if m.get('smoke'):
            out['skipped_smoke'].append(str(mp)); continue
        fold = {'fold_id': m['fold_id'], 'manifest_sha256': C.sha256_path(mp), 'host': m.get('host'), 'target_j': m.get('target_j'),
                'instruction': m.get('instruction'), 'scene': m.get('scene'), 'mil': m.get('mil'), 'policy': m.get('policy'),
                'planned_rows': len(m['rows']), 'arms': {}}
        mil = m.get('mil') or {}
        mil_miss = [k for k in ('model', 'sha256') if not mil.get(k)] + ([] if (mil.get('fold_contract') or {}).get('sha256')
                                                                          else ['fold_contract.sha256'])
        for arm in sorted(m['arms']):
            d = mp.parent / arm
            pin = m['arms'][arm]
            prov = pin.get('provenance') or {}
            miss = ['mil.' + k for k in mil_miss] + [k for k in PIN_KEYS if not pin.get(k)] + ([] if prov.get('kind') else ['provenance.kind'])
            a = {'pin': {k: v for k, v in pin.items() if k != 'provenance'}, 'actor_kind': prov.get('kind'),
                 'actor_fold_id': prov.get('fold_id'), 'pin_error': f'missing pins: {miss}' if miss else None,
                 'run_complete': (d / 'RUN_COMPLETE.json').exists(), 'rows': [], 'load_error': None,
                 'audit_errors': [], 'started': (d / 'progress.jsonl').exists()}
            if a['started']:
                try:
                    done = load_progress(d, m, arm)
                except Exception as e:   # noqa: BLE001 - recorded, never hidden
                    a['load_error'] = f'{type(e).__name__}: {e}'
                    done = None
                if done is not None:
                    try:
                        errs = audit_decisions(done, d)
                    except Exception as e:   # noqa: BLE001
                        errs = [f'audit exception {type(e).__name__}: {e}']
                    a['audit_errors'] = errs
                    bad = {int(x.group(1)) for x in (N_RE.match(e) for e in errs) if x}
                    whole = any(not N_RE.match(e) for e in errs)
                    rows = {r['n']: r for r in m['rows']}
                    for n, it in sorted(done.items()):
                        r = rows[n]
                        a['rows'].append({'n': n, 'j': r.get('j'), 'env_seed': r.get('env_seed'), 'inference_seed': r.get('inference_seed'),
                                          'plan_id': r.get('plan_id'), 'base_success': int(r['base_success']), 'success': int(it['success']),
                                          'composed': it.get('composed'), 'fired': it.get('fired'), 'http_sends': it.get('http_sends'),
                                          'first_fire_t': it.get('first_fire_t'), 'attempt': it.get('attempt'),
                                          'audit_ok': not whole and n not in bad and not miss})
            fold['arms'][arm] = a
        out['folds'].append(fold)
    return out


def stats(rows):
    ok = [r for r in rows if r['audit_ok']]
    n = len(ok)
    s = sum(r['success'] for r in ok)
    b = sum(r['base_success'] for r in ok)
    fail0 = [r for r in ok if not r['base_success']]
    succ0 = [r for r in ok if r['base_success']]
    return {'N': n, 'success': s, 'SR': (s / n) if n else None, 'base_success_archived': b, 'base_SR': (b / n) if n else None,
            'base_success_N': b, 'base_fail_N': n - b,
            'rescue': sum(1 for r in fail0 if r['success']),
            'damage': sum(1 for r in succ0 if not r['success']),
            # rows of each denominator where the detector fired at least once (the only rows the policy could change)
            'rescue_detected_den': sum(1 for r in fail0 if (r['fired'] or 0) > 0),
            'damage_detected_den': sum(1 for r in succ0 if (r['fired'] or 0) > 0),
            'fired_inferences': sum(r['fired'] or 0 for r in ok), 'inferences': sum(r['http_sends'] or 0 for r in ok),
            'composed_rows': sum(1 for r in ok if (r['composed'] or 0) > 0), 'failed_audit_rows': len(rows) - n}


class MergeError(ValueError):
    pass


def merge(snaps, planned, planned_total=None, policies=None):
    policies = sorted(policies or C.PRODUCTION_ARMS)
    for k in ('host', 'root'):
        vals = [s[k] for s in snaps]
        if len(set(vals)) != len(vals):
            raise MergeError(f'duplicate snapshot {k}: {sorted({v for v in vals if vals.count(v) > 1})}')
    rows, arms_meta, errors, seen = [], [], [], {}
    for sn in snaps:
        for f in sn['folds']:
            for arm, a in f['arms'].items():
                if arm not in policies:
                    errors.append({'host': sn['host'], 'fold_id': f['fold_id'], 'policy': arm, 'unexpected_policy': 'excluded'})
                    continue
                mil = f.get('mil') or {}
                arms_meta.append({'host': sn['host'], 'fold_id': f['fold_id'], 'policy': arm, 'mode': a['pin'].get('mode'),
                                  'actor': a['pin'].get('actor'), 'actor_sha256': a['pin'].get('actor_sha256'),
                                  'actor_json_sha256': a['pin'].get('actor_json_sha256'), 'actor_kind': a.get('actor_kind'),
                                  'actor_fold_id': a.get('actor_fold_id'), 'mil_model': mil.get('model'),
                                  'mil_sha256': mil.get('sha256'), 'mil_fold_contract_sha256': (mil.get('fold_contract') or {}).get('sha256'),
                                  'manifest_sha256': f['manifest_sha256'], 'started': a['started'], 'run_complete': a['run_complete'],
                                  'rows_done': len(a['rows']), 'planned_rows': f['planned_rows'], 'load_error': a['load_error'],
                                  'pin_error': a.get('pin_error'), 'audit_errors': len(a['audit_errors'])})
                for k in ('load_error', 'pin_error'):
                    if a.get(k):
                        errors.append({'host': sn['host'], 'fold_id': f['fold_id'], 'policy': arm, k: a[k]})
                for e in a['audit_errors']:
                    errors.append({'host': sn['host'], 'fold_id': f['fold_id'], 'policy': arm, 'audit_error': e})
                for r in a['rows']:
                    dk = (f['fold_id'], arm, r['n'])        # a fold has ONE eval host: same fold on two hosts = duplicate
                    if dk in seen:
                        raise MergeError(f'duplicate row fold={dk[0]} policy={dk[1]} n={dk[2]} (hosts {seen[dk]}, {sn["host"]})')
                    seen[dk] = sn['host']
                    rows.append({'host': sn['host'], 'fold_id': f['fold_id'], 'policy': arm, **r})
    hosts = sorted({sn['host'] for sn in snaps})
    per = []
    for h in hosts:
        for p in policies:
            st = stats([r for r in rows if r['host'] == h and r['policy'] == p])
            pn = planned.get(h)
            per.append({'host': h, 'policy': p, **st, 'plannedN': pn, 'coverage': (st['N'] / pn) if pn else None})
    for p in policies:
        st = stats([r for r in rows if r['policy'] == p])
        per.append({'host': 'ALL', 'policy': p, **st, 'plannedN': planned_total, 'coverage': (st['N'] / planned_total) if planned_total else None})
    key = lambda r: (r['host'], r['fold_id'], r['n'])   # noqa: E731
    ok_by = {p: {key(r) for r in rows if r['policy'] == p and r['audit_ok']} for p in policies}
    missing = [p for p in policies if not ok_by[p]]
    common = set() if missing else set.intersection(*ok_by.values())
    sub = [{'policy': p, **stats([r for r in rows if r['policy'] == p and key(r) in common])} for p in policies]
    status = f'INCOMPLETE: no audited rows for {missing}' if missing else f'complete over all {len(policies)} policies'
    return {'kind': 'fold27_eval_report_v2', 'policies': policies, 'hosts': hosts, 'per_host_policy': per,
            'common_subset': {'n_keys': len(common), 'policies_required': policies, 'missing_policies': missing,
                              'status': status, 'per_policy': sub}, 'arms': arms_meta, 'errors': errors,
            'snapshots': [{'host': s['host'], 'root': s['root'], 'time': s['time'], 'skipped_smoke': s['skipped_smoke']} for s in snaps],
            'limitations': LIMITATIONS, 'rows': rows}


def fmt(x):
    return '' if x is None else (f'{x:.3f}' if isinstance(x, float) else str(x))


def write_outputs(rep, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / 'report.json').write_text(json.dumps(rep, indent=1, default=str))
    cols = ['host', 'policy', 'N', 'success', 'SR', 'plannedN', 'coverage', 'base_success_N', 'base_fail_N', 'base_SR', 'rescue',
            'rescue_detected_den', 'damage', 'damage_detected_den', 'fired_inferences', 'inferences', 'composed_rows', 'failed_audit_rows']
    with open(out_dir / 'per_policy.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(cols); w.writerows([[fmt(r[c]) for c in cols] for r in rep['per_host_policy']])
    ecols = ['host', 'fold_id', 'policy', 'n', 'j', 'env_seed', 'inference_seed', 'plan_id', 'base_success', 'success', 'composed',
             'fired', 'http_sends', 'first_fire_t', 'attempt', 'audit_ok']
    with open(out_dir / 'per_episode.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(ecols); w.writerows([[fmt(r.get(c)) for c in ecols] for r in rep['rows']])
    md = ['# fold27 eval report (read-only aggregate)', '',
          f'policies: {", ".join(rep["policies"])}; hosts: {", ".join(rep["hosts"])}', '',
          '## Per host x policy (audited finished rows only)', '',
          '| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    md += ['| ' + ' | '.join(fmt(r[c]) for c in cols) + ' |' for r in rep['per_host_policy']]
    cs = rep['common_subset']
    md += ['', f'## Common subset ({cs["n_keys"]} episodes audited in every one of {len(cs["policies_required"])} policies; {cs["status"]})', '',
           '| policy | N | success | SR | base_success_N | base_fail_N | rescue | rescue_detected_den | damage | damage_detected_den |',
           '|---|---|---|---|---|---|---|---|---|---|']
    md += [f'| {r["policy"]} | {r["N"]} | {r["success"]} | {fmt(r["SR"])} | {r["base_success_N"]} | {r["base_fail_N"]} | {r["rescue"]} | '
           f'{r["rescue_detected_den"]} | {r["damage"]} | {r["damage_detected_den"]} |' for r in cs['per_policy']]
    md += ['', f'## Errors ({len(rep["errors"])}; excluded rows are listed, never hidden)', '']
    md += [f'- {json.dumps(e)}' for e in rep['errors'][:200]] or ['- none']
    md += ['', '## Arms (completeness, pins; full sha256 in report.json)', '',
           '| host | fold | policy | mode | actor_kind | rows_done/planned | complete | load_error | pin_error | actor_sha256 | mil_sha256 |',
           '|---|---|---|---|---|---|---|---|---|---|---|']
    md += [f'| {a["host"]} | {a["fold_id"]} | {a["policy"]} | {a["mode"]} | {a["actor_kind"] or ""} | {a["rows_done"]}/{a["planned_rows"]} | '
           f'{a["run_complete"]} | {a["load_error"] or ""} | {a["pin_error"] or ""} | {(a["actor_sha256"] or "")[:16]} | '
           f'{(a["mil_sha256"] or "")[:16]} |' for a in rep['arms']]
    md += ['', '## Limitations', ''] + [f'- {x}' for x in rep['limitations']]
    (out_dir / 'report.md').write_text('\n'.join(md) + '\n')
    return out_dir


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('snapshot'); s.add_argument('--root', type=Path, required=True); s.add_argument('--host', required=True)
    s.add_argument('--out', type=Path, required=True)
    g = sub.add_parser('merge'); g.add_argument('--snapshot', type=Path, nargs='+', required=True)
    g.add_argument('--planned', nargs='*', default=[]); g.add_argument('--planned-total', type=int)
    g.add_argument('--out-dir', type=Path, required=True)
    g.add_argument('--policies', nargs='+', help='expected policy set (default: common.PRODUCTION_ARMS)')
    a = ap.parse_args(argv)
    if a.cmd == 'snapshot':
        r = snapshot(a.root, a.host)
        with open(a.out, 'x') as f:
            f.write(json.dumps(r, indent=1, default=str))
        print(json.dumps({'folds': len(r['folds']), 'arms_with_load_error': sum(1 for f in r['folds'] for x in f['arms'].values() if x['load_error'])}))
        return 0
    planned = {k: int(v) for k, v in (x.split('=', 1) for x in a.planned)}
    if len({p.resolve() for p in a.snapshot}) != len(a.snapshot):
        raise MergeError('the same snapshot file was given twice')
    rep = merge([json.loads(p.read_text()) for p in a.snapshot], planned, a.planned_total, a.policies)
    d = write_outputs(rep, a.out_dir)
    print(json.dumps({'out_dir': str(d), 'policies': rep['policies'], 'errors': len(rep['errors']),
                      'common_subset': rep['common_subset']['n_keys'], 'common_status': rep['common_subset']['status']}))
    return 0


if __name__ == '__main__':
    sys.exit(main())

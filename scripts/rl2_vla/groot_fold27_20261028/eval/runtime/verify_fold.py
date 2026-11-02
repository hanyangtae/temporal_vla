"""Audit a finished fold (production only): same 40 rows per arm, hashes, per-inference gate log, paired vs archive base.

    verify_fold.py --manifest <out>/<fold>/manifest.json [--arms mil_qam65 ...]
Writes <fold dir>/verification/<ns>.json and per_episode.tsv. Exit 1 on any error. Smoke manifests are refused.
Claim scope: policy-intervention SR on the target j only; archive base = same rows' original outcome (diagnostic pairing).
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
from run_eval_arm import lines, load_progress, verify_manifest  # noqa: E402


def audit_decisions(rows_done, out):
    """Every inference was MIL-scored; gate == fired; ungated output untouched; composed from the same noise."""
    errors, logs = [], {}
    for it in rows_done.values():
        if it['decisions_log'] not in logs:
            logs[it['decisions_log']] = {d['request']: d for d in lines(out / it['decisions_log'])}
        ds = logs[it['decisions_log']]
        lo, hi = it['requests']
        got = [ds.get(i) for i in range(lo, hi + 1)]
        if any(d is None for d in got) or hi - lo + 1 != it['http_sends']:
            errors.append(f'n{it["n"]}: decision log incomplete'); continue
        for d in got:
            if d['mode'] != 'gated' or 'score' not in d['mil']:
                errors.append(f'n{it["n"]} r{d["request"]}: not a gated MIL decision')
            if d['gate'] != bool(d['mil']['fired']) or d['mil']['fired'] != (d['mil']['score'] > 0.5):
                errors.append(f'n{it["n"]} r{d["request"]}: gate rule mismatch')
            if not d['gate'] and (d['qam_delta_max_normalized'] != 0 or d['noise_sha256_composed'] is not None):
                errors.append(f'n{it["n"]} r{d["request"]}: ungated action modified')
            if d['gate'] and (d['noise_sha256_composed'] != d['noise_sha256_first_pass'] or d['qam_steps'] != [0, 1, 2, 3]):
                errors.append(f'n{it["n"]} r{d["request"]}: composed pass noise/QAM steps mismatch')
            if not (d['action_finite'] and d['off_identity_verified'] and d['rng_restored_equal']):
                errors.append(f'n{it["n"]} r{d["request"]}: finite/identity/RNG failure')
        if sum(d['gate'] for d in got) != it['composed']:
            errors.append(f'n{it["n"]}: composed count mismatch')
    return errors


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True); ap.add_argument('--arms', nargs='+')
    a = ap.parse_args(argv)
    m = json.loads(a.manifest.read_text())
    C.need(not m['smoke'], 'smoke manifests are never verified as production')
    fold_dir = a.manifest.resolve().parent
    result = {'fold_id': m['fold_id'], 'host': m['host'], 'arms': {}, 'errors': []}
    table = []
    for arm in a.arms or sorted(m['arms']):
        verify_manifest(a.manifest, arm, check_files=False)
        out = fold_dir / arm
        done = load_progress(out, m, arm)
        errs = [] if (out / 'RUN_COMPLETE.json').exists() else ['RUN_COMPLETE missing']
        if sorted(done) != list(range(40)):
            errs.append(f'rows {sorted(done)} != n0..39')
        errs += audit_decisions(done, out)
        pairs = [(r['base_success'], done[r['n']]['success']) for r in m['rows'] if r['n'] in done]
        result['arms'][arm] = {'n': len(pairs), 'success': sum(q for _, q in pairs), 'base_success_archive': sum(b for b, _ in pairs),
                               'rescue': sum(1 for b, q in pairs if not b and q), 'damage': sum(1 for b, q in pairs if b and not q),
                               'fired_inferences': sum(it['fired'] for it in done.values()),
                               'inferences': sum(it['http_sends'] for it in done.values()), 'errors': errs}
        result['errors'] += [f'{arm}: {e}' for e in errs]
        table += [[m['fold_id'], arm, r['n'], r['base_success'], done[r['n']]['success'], done[r['n']]['composed'],
                   done[r['n']]['first_fire_t']] for r in m['rows'] if r['n'] in done]
    vdir = fold_dir / 'verification'; vdir.mkdir(exist_ok=True)
    (vdir / f'{time.time_ns()}.json').write_text(json.dumps(result, indent=1))
    if not result['errors']:
        with (fold_dir / 'per_episode.tsv').open('w') as f:
            w = csv.writer(f, delimiter='\t')
            w.writerow(['fold_id', 'arm', 'noise', 'base_success_archive', 'success', 'composed_inferences', 'first_fire_t'])
            w.writerows(table)
    print(json.dumps(result, indent=1))
    return 1 if result['errors'] else 0


if __name__ == '__main__':
    sys.exit(main())

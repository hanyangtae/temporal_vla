#!/usr/bin/env python3
"""CPU diagnostics of saved LR operators against their exact source activations.

No detector fitting or policy evaluation. Reconstruct saved cap-controlled fits,
then use equal first-five phase windows as a separate dwell-matched diagnostic.
Episode means below are within one semantic phase, never full-rollout pooling.
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import fit


def cosine(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 1e-10 else None


def class_mean(eps, phase, side, label, window=None):
    groups = {}
    for e in eps:
        if e.side != side or e.success != label:
            continue
        x = e.X[e.phase == phase]
        if not len(x) or (window and len(x) < window):
            continue
        if window:
            x = x[:window]
        groups.setdefault(e.row['jitter'], []).append(x.mean(0))
    if not groups:
        return None
    return np.mean([np.mean(xs, axis=0) for xs in groups.values()], axis=0)


def projected_records(eps, phase, side, label, v, window=5):
    xs = [e.X[e.phase == phase] for e in eps if e.side == side and e.success == label]
    xs = [x[:window] for x in xs if len(x) >= window]
    return np.concatenate(xs) @ v if xs else np.array([])


def run(root, output):
    shards = fit.Shards(json.loads((root / 'shard_map.json').read_text()))
    rows, checks = [], []
    for path in sorted((root / 'fit').glob('*/pair_report.json')):
        p = json.loads(path.read_text())
        r = p['request']
        records, reports = {}, {}
        for arm in p['arms']:
            report = json.loads((path.parent / arm / 'fit_report.json').read_text())
            reports[arm] = report
            records[arm] = [fit.truncate(shards.load(e), p['common_phase_caps']) for e in report['episodes']]
        solo = next(a for a in records if 'LRmix' not in a)
        mixed = next(a for a in records if 'LRmix' in a)
        target = r['target_side']
        for phase in p['common_phase_caps']:
            vectors = {}
            for arm, eps in records.items():
                op = path.parent / arm / 'operators/plain' / phase / 'dit_L12/conceptors.npz'
                if not op.exists():
                    continue
                z = np.load(op)
                v, s = z['alpha0_v_steer'], float(z['alpha0_s'])
                sides = sorted({e.side for e in eps})
                means = {(side, lab): class_mean(eps, phase, side, lab) for side in sides for lab in (0, 1)}
                mu_s = np.mean([means[side, 1] for side in sides], axis=0)
                delta = mu_s - np.mean([means[side, 0] for side in sides], axis=0)
                if not np.allclose(v, delta / np.linalg.norm(delta), atol=2e-5) or not np.isclose(s, mu_s @ v, atol=2e-4):
                    raise ValueError(f'operator reconstruction mismatch: {op}')
                checks.append(str(op.relative_to(root)))
                vectors[arm] = (v, s)
            row = dict(pair=p['pair_id'], **r, phase=phase, cap=p['common_phase_caps'][phase],
                       solo_registered=solo in vectors, mixed_registered=mixed in vectors)
            if solo in vectors and mixed in vectors:
                vs, ss = vectors[solo]
                vm, sm = vectors[mixed]
                eps = records[mixed]
                target_s = class_mean(eps, phase, target, 1)
                target_f = class_mean(eps, phase, target, 0)
                other = 'R' if target == 'L' else 'L'
                other_s = class_mean(eps, phase, other, 1)
                other_f = class_mean(eps, phase, other, 0)
                row['solo_mixed_cosine'] = cosine(vs, vm)
                row['within_mixed_side_delta_cosine'] = cosine(target_s-target_f, other_s-other_f)
                row['target_delta_norm'] = float(np.linalg.norm(target_s-target_f))
                row['other_delta_norm'] = float(np.linalg.norm(other_s-other_f))
                row['target_setpoint_offset'] = float(sm - target_s @ vm)
                row['target_gap_projected'] = float((target_s-target_f) @ vm)
                row['offset_over_target_gap'] = float((sm-target_s @ vm) / abs((target_s-target_f) @ vm)) if abs((target_s-target_f) @ vm) > 1e-6 else None
                # Equal first-five records per phase, retaining per-record scores.
                means5 = {(side, lab): class_mean(eps, phase, side, lab, 5) for side in (target, other) for lab in (0, 1)}
                if all(x is not None for x in means5.values()):
                    row['first5_side_delta_cosine'] = cosine(means5[target, 1]-means5[target, 0], means5[other, 1]-means5[other, 0])
                score_s = projected_records(eps, phase, target, 1, vm)
                score_f = projected_records(eps, phase, target, 0, vm)
                row['first5_target_success_records'] = len(score_s)
                row['first5_target_failure_records'] = len(score_f)
                if len(score_s):
                    lo, hi = np.quantile(score_s, [.1, .9])
                    row['mixed_goal_outside_target_success_q10_q90'] = bool(sm < lo or sm > hi)
                    row['mixed_goal_success_z'] = float((sm-score_s.mean()) / max(score_s.std(), 1e-6))
                    row['mean_abs_success_kick_beta08'] = float(.8*np.abs(score_s-sm).mean())
                if len(score_f):
                    row['mean_abs_failure_kick_beta08'] = float(.8*np.abs(score_f-sm).mean())
                # Same target failure frames under solo/mix isolates operator geometry.
                frames = [e.X[e.phase == phase][:5] for e in records[solo] if not e.success and np.sum(e.phase == phase) >= 5]
                if frames:
                    x = np.concatenate(frames)
                    ks = (ss-x@vs)[:, None]*vs
                    km = (sm-x@vm)[:, None]*vm
                    denom = np.linalg.norm(ks, axis=1)*np.linalg.norm(km, axis=1)
                    valid = denom > 1e-8
                    row['first5_failure_kick_cosine_median'] = float(np.median(np.sum(ks*km, axis=1)[valid]/denom[valid])) if valid.any() else None
                    row['first5_failure_kick_norm_ratio'] = float(np.linalg.norm(km, axis=1).mean()/max(np.linalg.norm(ks, axis=1).mean(),1e-6))
            rows.append(row)
        print(json.dumps({'pair': p['pair_id'], 'rows': len(rows), 'shards_loaded': len(shards.cache)}), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    result = {'source_root': str(root), 'revision': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'verified_plain_operators': len(checks), 'rows': rows,
              'scope': 'Exact selected training activations; first-five-per-phase subset diagnostic; not held-out prediction or online intervention.',
              'source_sha256': {k: next(iter(v.values()))[3]['sha256'] for k, v in shards.cache.items()}}
    (output/'geometry.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())

"""fold27 success -> success timing: archived baseline vs each eval policy, on the SAME manifest coordinates. READ-ONLY,
stdlib only, no pickle: eval side = per-episode meta.json named by the audited progress line; baseline side = the
derived full_episodes/*.json written by fold_data/extract_scene.py (pkl/npz sha recorded there, never opened here).
Outputs are NEW files only.

  eval      on each host:  --root <eval root> --host H --out <new eval_snap.json>
            Every production manifest/arm is snapshotted with aggregate_eval_report.snapshot (unchanged: progress re-hash +
            MIL decision audit); for each audited finished row the meta.json named by its progress line is read.
  baseline  on each host:  --dir <full_episodes/<machine>_<slug>_s<scene>> [...] --out <new base_index.json>
  table     anywhere:      --eval a.json b.json ... --baseline x.json y.json ... --out-dir <new dir>
            report.json, pairs.csv (one row per succ->succ pair), per_policy.csv / per_instruction.csv / per_fold.csv, report.md

TIME UNIT (identical signal on both sides, nothing estimated):
  Both sides were produced by http_feature_collect.py with n_action_steps 5 / max_episode_steps 720. Its loop calls
  MultiStepWrapper.step once per decision (5 env steps), tests the chunk-level step_success(reward, info) and breaks on
  the first success (http_feature_collect.py:1479-1487). The time compared is d = the 1-based DECISION index of that first
  chunk-level success:
    eval      d = meta.first_success_step   (asserted == meta.steps, the loop count at the break; 1 <= d <= 144)
    baseline  d = env_steps / 5             (env_steps = executed raw steps = len(env_step_success) in the pkl; asserted
                                             == n_inferences and env_steps % 5 == 0, i.e. the same break, counted in env steps)
  env-step equivalent = 5 d = end of the success chunk (chunk resolution, 5 env steps). No seconds: control frequency is not
  used and wall time is never compared. The baseline per-env-step probe (first_success_step in full_episodes, 0-based
  _check_success() right after each raw step) exists only for the baseline and is reported separately (probe_chunk =
  ceil((probe+1)/5), lag = env_steps - 5*probe_chunk in {0, 5}); it is NOT compared with the eval side.
Any unit/contract violation is an error row (listed, excluded), never coerced.

SELECTION: succ->succ pairs are a selected subset (both the archived run and the policy run succeeded) and differ per
policy; the common subset (base and ALL policies succeeded on the coordinate) is reported as well. The archived baseline
is the original collection, not a re-run under the eval initialisation (equivalence not established): timing differences
are descriptive, not causal.
"""
import argparse, csv, hashlib, json, statistics, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RT = ROOT / 'scripts/rl2_vla/groot_fold27_20261028/eval/runtime'
K, MAX_STEPS = 5, 720
MAX_D = MAX_STEPS // K


class TimingError(ValueError):
    pass


def need(c, msg):
    if not c:
        raise TimingError(msg)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def runtime():
    sys.path.insert(0, str(RT))
    import aggregate_eval_report as AG     # noqa: E402  (unchanged)
    from run_eval_arm import load_progress  # noqa: E402  (unchanged)
    return AG.snapshot, load_progress


# ------------------------------------------------------------------ eval side
def eval_d(meta):
    """Decision index of the first chunk-level success, or None for a failure. Raises on any unit inconsistency."""
    need(int(meta.get('n_action_steps', -1)) == K, f'n_action_steps {meta.get("n_action_steps")} != {K}')
    need(int(meta.get('max_episode_steps', -1)) == MAX_STEPS, f'max_episode_steps {meta.get("max_episode_steps")} != {MAX_STEPS}')
    succ = meta.get('episode_success', meta.get('success'))
    need(succ is not None, 'meta lacks success')
    d, steps = meta.get('first_success_step'), meta.get('steps')
    if not int(bool(succ)):
        need(d is None, f'failure with first_success_step {d}')
        return None
    need(isinstance(d, int) and 1 <= d <= MAX_D, f'success with first_success_step {d!r}')
    need(steps == d, f'first_success_step {d} != steps {steps} (loop breaks on first success)')
    return d


def collect_eval(root, host, snapshot=None, load_progress=None):
    if snapshot is None or load_progress is None:
        snapshot, load_progress = runtime()
    root = Path(root)
    snap = snapshot(root, host)
    out = {'kind': 'fold27_success_timing_eval_v1', 'host': host, 'root': str(root), 'snapshot_time': snap['time'],
           'skipped_smoke': snap['skipped_smoke'], 'rows': [], 'errors': []}
    for f in snap['folds']:
        mp = root / f['fold_id'] / 'manifest.json'
        m = json.loads(mp.read_text())
        need(sha(mp) == f['manifest_sha256'], f'{mp} changed during collection')
        rows = {r['n']: r for r in m['rows']}
        for arm, a in sorted(f['arms'].items()):
            if not a['rows']:
                continue
            done = load_progress(root / f['fold_id'] / arm, m, arm)
            for r in a['rows']:
                base = {'host': host, 'fold_id': f['fold_id'], 'instruction': f['instruction'], 'scene': f['scene'],
                        'archive_machine': m.get('archive_machine'), 'policy': arm, 'n': r['n'], 'j': r['j'],
                        'plan_id': r['plan_id'], 'base_success': r['base_success'], 'success': r['success'],
                        'audit_ok': r['audit_ok'], 'fired': r.get('fired'), 'first_fire_t': r.get('first_fire_t')}
                try:
                    need(r['audit_ok'], 'row failed the aggregate audit (pins/decision log)')
                    it = done[r['n']]
                    need(rows[r['n']]['plan_id'] == r['plan_id'], 'manifest row drift')
                    mpath = root / f['fold_id'] / arm / it['meta']
                    meta = json.loads(mpath.read_text())
                    d = eval_d(meta)
                    need(int(d is not None) == r['success'], f'meta success {d is not None} != progress success {r["success"]}')
                    out['rows'].append({**base, 'd': d, 'meta': str(Path(f['fold_id']) / arm / it['meta']), 'meta_sha256': sha(mpath)})
                except (TimingError, KeyError, OSError, ValueError, TypeError) as e:
                    out['errors'].append({**base, 'error': f'{type(e).__name__}: {e}'})
    return out


# ------------------------------------------------------------------ baseline side
def base_d(rec):
    """(d, probe) for a success, (None, None) for a failure, from a full_episodes JSON. Raises on inconsistency.
    Older extractions (e.g. kanu_drawer_left_s2) carry no probe fields: d still comes from env_steps / n_inferences
    (the collector break), probe is None ('absent'), never taken from another source."""
    need(int(rec['max_episode_steps']) == MAX_STEPS, f'max_episode_steps {rec["max_episode_steps"]}')
    steps, ninf = int(rec['env_steps']), int(rec['n_inferences'])
    has_probe = 'first_success_step' in rec
    probe = int(rec['first_success_step']) if has_probe else None
    lag = int(rec['success_chunk_lag_steps']) if 'success_chunk_lag_steps' in rec else None
    need(has_probe == (lag is not None), 'probe fields incomplete (first_success_step / success_chunk_lag_steps)')
    if not int(rec['success']):
        need(steps == MAX_STEPS and ninf == MAX_D, f'failure: steps {steps} n_inferences {ninf}')
        need(not has_probe or (probe == -1 and lag == -1), f'failure: probe {probe} lag {lag}')
        return None, None
    need(steps % K == 0 and steps // K == ninf, f'success: env_steps {steps} vs n_inferences {ninf} (k={K})')
    if has_probe:
        need(0 <= probe < steps, f'success: probe {probe} outside [0, {steps})')
        pc = -(-(probe + 1) // K)
        need(lag == steps - K * pc and lag in (0, K), f'success: lag {lag} vs steps {steps} probe chunk {pc}')
    return steps // K, probe


def collect_baseline(dirs):
    out = {'kind': 'fold27_success_timing_baseline_v1', 'dirs': [str(d) for d in dirs], 'episodes': {}, 'errors': []}
    for d in dirs:
        for p in sorted(Path(d).glob('*.json')):
            rec = json.loads(p.read_text())
            if 'episode_id' not in rec:                    # extractor spec file in the same dir, not an episode
                out.setdefault('skipped_non_episode', []).append(str(p)); continue
            try:
                key = '|'.join(map(str, (rec['plan_id'], rec['instruction'], rec['scene'], rec['jitter'], rec['noise'])))
                need(key not in out['episodes'], f'duplicate baseline episode {key}')
                dd, probe = base_d(rec)
                out['episodes'][key] = {'machine': rec['machine'], 'episode_id': rec['episode_id'], 'success': int(rec['success']),
                                        'd': dd, 'probe_first_env_step0': probe,
                                        'probe_chunk': None if probe is None else -(-(probe + 1) // K),
                                        'json': p.name, 'json_sha256': sha(p), 'pkl_sha256': rec.get('pkl_sha256')}
            except (TimingError, KeyError, ValueError, TypeError) as e:
                out['errors'].append({'json': str(p), 'error': f'{type(e).__name__}: {e}'})
    return out


# ------------------------------------------------------------------ pairing + tables
def summ(xs):
    if not xs:
        return {'N': 0}
    xs = sorted(xs)
    q = statistics.quantiles(xs, n=4, method='inclusive') if len(xs) > 1 else [xs[0]] * 3
    return {'N': len(xs), 'mean': statistics.fmean(xs), 'median': float(statistics.median(xs)), 'q1': q[0], 'q3': q[2],
            'min': xs[0], 'max': xs[-1]}


def group_stats(pairs):
    db, de = [p['d_base'] for p in pairs], [p['d_eval'] for p in pairs]
    dl = [p['delta'] for p in pairs]
    return {'pairs': len(pairs), 'base_d': summ(db), 'eval_d': summ(de), 'delta_d': summ(dl),
            'faster': sum(x < 0 for x in dl), 'same': sum(x == 0 for x in dl), 'slower': sum(x > 0 for x in dl)}


def build(evals, bases):
    idx, berr = {}, []
    for b in bases:
        berr += b['errors']
        for k, v in b['episodes'].items():
            need(k not in idx, f'baseline episode {k} in two indexes')
            idx[k] = v
    rows, errors, seen = [], [], set()
    for e in evals:
        errors += [{**x, 'source': e['host']} for x in e['errors']]
        for r in e['rows']:
            key3 = (r['host'], r['fold_id'], r['policy'], r['n'])
            need(key3 not in seen, f'duplicate eval row {key3}')
            seen.add(key3)
            k = '|'.join(map(str, (r['plan_id'], r['instruction'], r['scene'], r['j'], r['n'])))
            b = idx.get(k)
            if b is None:
                errors.append({**r, 'error': 'baseline full_episodes JSON not found (not filled from any other run)'}); continue
            if b['success'] != r['base_success']:
                errors.append({**r, 'error': f'manifest base_success {r["base_success"]} != full_episodes success {b["success"]}'}); continue
            if r['archive_machine'] and b['machine'] != r['archive_machine']:
                errors.append({**r, 'error': f'archive machine {b["machine"]} != manifest {r["archive_machine"]}'}); continue
            rows.append({**r, 'd_base': b['d'], 'base_episode_id': b['episode_id'], 'base_json_sha256': b['json_sha256'],
                         'probe_chunk': b['probe_chunk']})
    pairs = [{**r, 'd_eval': r['d'], 'delta': r['d'] - r['d_base']} for r in rows
             if r['base_success'] and r['success']]
    policies = sorted({r['policy'] for r in rows})
    by = lambda keyf: {k: group_stats(v) for k, v in sorted(_group(pairs, keyf).items())}
    coords = defaultdict(dict)
    for r in rows:
        coords[(r['host'], r['fold_id'], r['n'])][r['policy']] = r
    common = [c for c, v in coords.items() if set(v) == set(policies) and all(x['base_success'] and x['success'] for x in v.values())]
    cset = set(common)
    common_pairs = [p for p in pairs if (p['host'], p['fold_id'], p['n']) in cset]
    counts = {pol: {'rows': sum(r['policy'] == pol for r in rows),
                    'base_succ': sum(r['policy'] == pol and r['base_success'] for r in rows),
                    'eval_succ': sum(r['policy'] == pol and r['success'] for r in rows),
                    'succ_succ': sum(p['policy'] == pol for p in pairs)} for pol in policies}
    return {'unit': 'decision index d of first chunk-level success (1 decision = 5 env steps; env-step equivalent 5d)',
            'policies': policies, 'counts': counts,
            'per_policy': by(lambda p: (p['policy'],)),
            'per_policy_by_fired': by(lambda p: (p['policy'], 'fired>0' if (p['fired'] or 0) > 0 else 'fired=0')),
            'per_instruction': by(lambda p: (p['policy'], p['instruction'])),
            'per_fold': by(lambda p: (p['policy'], p['host'], p['fold_id'])),
            'common_subset': {'coordinates': len(common), 'per_policy': {k[0]: v for k, v in
                              {k: group_stats(v) for k, v in _group(common_pairs, lambda p: (p['policy'],)).items()}.items()}},
            'pairs': pairs, 'errors': errors, 'baseline_errors': berr}


def _group(xs, keyf):
    g = defaultdict(list)
    for x in xs:
        g[keyf(x)].append(x)
    return g


def fmt(x):
    return '' if x is None else (f'{x:.2f}' if isinstance(x, float) else str(x))


def flat(key, st):
    return {'key': ' / '.join(map(str, key)) if isinstance(key, tuple) else key, 'pairs': st['pairs'],
            **{f'{w}_{s}': st[w].get(s) for w in ('base_d', 'eval_d', 'delta_d') for s in ('mean', 'median', 'q1', 'q3')},
            'faster': st['faster'], 'same': st['same'], 'slower': st['slower']}


def write(rep, out_dir, sources):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    rep = {**rep, 'sources': sources}
    (out / 'report.json').write_text(json.dumps({k: (v if k not in ('per_instruction', 'per_fold', 'per_policy', 'per_policy_by_fired')
                                                     else {' / '.join(map(str, kk)): vv for kk, vv in v.items()})
                                                 for k, v in rep.items()}, indent=1, default=str))
    pc = ['host', 'fold_id', 'instruction', 'scene', 'j', 'n', 'plan_id', 'policy', 'd_base', 'd_eval', 'delta', 'probe_chunk',
          'fired', 'first_fire_t',
          'base_episode_id', 'meta', 'meta_sha256', 'base_json_sha256']
    with open(out / 'pairs.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, pc, extrasaction='ignore'); w.writeheader(); w.writerows(rep['pairs'])
    for name in ('per_policy', 'per_policy_by_fired', 'per_instruction', 'per_fold'):
        rs = [flat(k, v) for k, v in rep[name].items()]
        with open(out / f'{name}.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, list(flat(('x',), group_stats([])).keys()) if not rs else list(rs[0])); w.writeheader(); w.writerows(rs)
    md = ['# fold27 success->success timing (archived baseline vs policy, same coordinates)', '',
          f'Unit: {rep["unit"]}. delta = d_eval - d_base (negative = policy succeeded earlier). Chunk resolution; no seconds.',
          'Selected subset: only coordinates where BOTH the archived baseline and the policy succeeded (differs per policy). '
          'Archived baseline = original collection, init equivalence to the eval not established: descriptive, not causal.', '',
          '| policy | rows | base succ | policy succ | succ->succ |', '|---|---|---|---|---|']
    md += [f'| {p} | {c["rows"]} | {c["base_succ"]} | {c["eval_succ"]} | {c["succ_succ"]} |' for p, c in rep['counts'].items()]

    def table(title, d):
        t = ['', f'## {title}', '', '| group | pairs | base d mean / median | policy d mean / median | delta mean / median [q1, q3] | faster / same / slower |',
             '|---|---|---|---|---|---|']
        for k, st in d.items():
            kk = ' / '.join(map(str, k)) if isinstance(k, tuple) else k
            b, e, dl = st['base_d'], st['eval_d'], st['delta_d']
            t.append(f'| {kk} | {st["pairs"]} | {fmt(b.get("mean"))} / {fmt(b.get("median"))} | {fmt(e.get("mean"))} / {fmt(e.get("median"))} | '
                     f'{fmt(dl.get("mean"))} / {fmt(dl.get("median"))} [{fmt(dl.get("q1"))}, {fmt(dl.get("q3"))}] | {st["faster"]} / {st["same"]} / {st["slower"]} |')
        return t
    md += table('All (per policy)', rep['per_policy'])
    md += table('Per policy x intervention fired in the episode (fired=0: the policy never changed an action, '
                'so equality with the archived run is expected)', rep['per_policy_by_fired'])
    md += table(f'Common subset ({rep["common_subset"]["coordinates"]} coordinates: base and every policy succeeded)',
                rep['common_subset']['per_policy'])
    md += table('Per instruction', rep['per_instruction'])
    md += table('Per fold', rep['per_fold'])
    md += ['', f'Errors (excluded, listed in report.json): eval/pairing {len(rep["errors"])}, baseline index {len(rep["baseline_errors"])}.']
    (out / 'report.md').write_text('\n'.join(md) + '\n')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('eval'); e.add_argument('--root', type=Path, required=True); e.add_argument('--host', required=True)
    e.add_argument('--out', type=Path, required=True)
    b = sub.add_parser('baseline'); b.add_argument('--dir', type=Path, nargs='+', required=True); b.add_argument('--out', type=Path, required=True)
    t = sub.add_parser('table'); t.add_argument('--eval', type=Path, nargs='+', required=True)
    t.add_argument('--baseline', type=Path, nargs='+', required=True); t.add_argument('--out-dir', type=Path, required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd in ('eval', 'baseline'):
            need(not a.out.exists(), f'{a.out} exists (new files only)')
            r = collect_eval(a.root, a.host) if a.cmd == 'eval' else collect_baseline(a.dir)
            a.out.parent.mkdir(parents=True, exist_ok=True)
            with open(a.out, 'x') as f:
                f.write(json.dumps(r, indent=1, default=str))
            print(json.dumps({'out': str(a.out), 'sha256': sha(a.out),
                              'n': len(r.get('rows', r.get('episodes', []))), 'errors': len(r['errors'])}))
        else:
            need(not a.out_dir.exists(), f'{a.out_dir} exists (new dir only)')
            ev = [json.loads(p.read_text()) for p in a.eval]
            bs = [json.loads(p.read_text()) for p in a.baseline]
            rep = build(ev, bs)
            write(rep, a.out_dir, {'eval': [[str(p), sha(p)] for p in a.eval], 'baseline': [[str(p), sha(p)] for p in a.baseline]})
            print(json.dumps({'out_dir': str(a.out_dir), 'pairs': len(rep['pairs']), 'errors': len(rep['errors'])}))
    except (TimingError, OSError) as ex:
        print(f'[success-timing] REFUSED: {ex}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())

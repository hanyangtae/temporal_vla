"""Focused stdlib tests for analyze.py (synthetic data only; no eval roots touched).

    python3 -m unittest discover -s scripts/analysis/fold27_paired_recovery_20261106/tests -v
"""
import hashlib, json, sys, tempfile, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import analyze as A  # noqa: E402

sys.path.insert(0, str(A.ST))
import success_timing as S  # noqa: E402  (real eval_d / summ)

SHA = 'ab' * 32


def dec(req, gate=False, fired=None, comp='same', steps=None, delta=0.3):
    fired = gate if fired is None else fired
    return {'request': req, 'mode': 'gated', 'mil': {'score': 0.9 if fired else 0.1, 'fired': fired, 't': 0},
            'gate': gate, 'qam_steps': ([0, 1, 2, 3] if gate else []) if steps is None else steps,
            'qam_delta_max_normalized': delta if gate else 0.0, 'noise_sha256_first_pass': SHA,
            'noise_sha256_composed': (SHA if comp == 'same' else comp) if gate else None,
            'action_finite': True, 'off_identity_verified': True, 'rng_restored_equal': True}


class TimingCore(unittest.TestCase):
    def test_inclusive_exclusive_off_by_one(self):
        self.assertEqual(A.timing(1, 10, 0), ('valid', 10, 9))   # composed at the first inference
        self.assertEqual(A.timing(1, 10, 9), ('valid', 1, 0))    # composed chunk is the success chunk
        self.assertEqual(A.timing(1, 10, 3), ('valid', 7, 6))
        with self.assertRaises(A.AnalysisError):
            A.timing(1, 10, 10)                                  # composition after the success: impossible

    def test_failure_and_noncomposed_are_null(self):
        self.assertEqual(A.timing(0, None, 4), ('failure', None, None))
        self.assertEqual(A.timing(0, None, None), ('failure', None, None))
        self.assertEqual(A.timing(1, 12, None), ('no_composition', None, None))
        with self.assertRaises(A.AnalysisError):
            A.timing(0, 7, None)                                 # failure must have null d

    def test_fired_is_not_composition(self):
        got = [dec(10), dec(11, fired=True), dec(12), dec(13, gate=True)]
        tc, tf, n, d = A.first_events(got, 1)
        self.assertEqual((tc, tf, n, d['request']), (3, 1, 1, 13))

    def test_gate_true_without_composed_noise_is_invalid(self):
        for bad in (dec(5, gate=True, comp=None), dec(5, gate=True, comp='cd' * 32), dec(5, gate=True, steps=[0, 1])):
            with self.assertRaises(A.AnalysisError):
                A.first_events([dec(4), bad], 1)
        with self.assertRaises(A.AnalysisError):            # gate=false carrying a composed hash
            A.actual_composed({**dec(4), 'noise_sha256_composed': SHA})

    def test_composed_count_must_match_progress(self):
        with self.assertRaises(A.AnalysisError):
            A.first_events([dec(1, gate=True), dec(2, gate=True)], 1)

    def test_delta_zero_still_composition(self):
        tc, _, _, d = A.first_events([dec(1), dec(2, gate=True, delta=0.0)], 1)
        self.assertEqual((tc, d['qam_delta_max_normalized']), (1, 0.0))

    def test_episode_join_uses_exact_range(self):
        recs = {i: dec(i) for i in range(1, 30)}
        got = A.episode_decisions(recs, set(), 11, 20)
        self.assertEqual([d['request'] for d in got], list(range(11, 21)))
        with self.assertRaises(A.AnalysisError):
            A.episode_decisions(recs, set(), 25, 35)            # incomplete log
        with self.assertRaises(A.AnalysisError):
            A.episode_decisions(recs, {15}, 11, 20)             # duplicated request id


# ------------------------------------------------------------------ synthetic eval root for collect()
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class FakeC:
    sha256_path = staticmethod(sha)


def make_root(tmp, spec, fold='Task-x_s1_j0', scene=1):
    """spec: {arm: {n: (success, d, composed_ts, fired_ts) or None(=no progress line) or 'badmeta'}}."""
    root = Path(tmp) / 'root'
    fdir = root / fold
    fdir.mkdir(parents=True)
    rows = [{'n': n, 'j': 0, 'env_seed': 100 + n, 'inference_seed': 900 + n, 'plan_id': 'p1', 'base_success': n % 2}
            for n in range(3)]
    m = {'fold_id': fold, 'smoke': False, 'instruction': 'Task/x', 'scene': scene, 'rows': rows,
         'arms': {a: {'mode': 'gated'} for a in A.POLICIES}}
    (fdir / 'manifest.json').write_text(json.dumps(m))
    progress = {}
    for arm in A.POLICIES:
        adir = fdir / arm
        adir.mkdir()
        decs, lines, req = [], [], 0
        for r in rows:
            s = spec.get(arm, {}).get(r['n'], (0, None, [], []))
            if s is None:
                continue
            bad = s == 'badmeta'
            success, d, cts, fts = (1, 5, [], []) if bad else s
            steps = d if success else 144
            lo, hi = req + 1, req + steps
            req = hi
            for i in range(steps):
                decs.append(dec(lo + i, gate=i in cts, fired=(i in cts) or (i in fts)))
            mdir = adir / f'rows/n{r["n"]:02d}/a0/grid/p1/h/Task/x/s{scene}/j0/n{r["n"]}/{arm}'
            mdir.mkdir(parents=True)
            meta = {'n_action_steps': 5, 'max_episode_steps': 720, 'episode_success': success,
                    'first_success_step': None if bad else d, 'steps': steps, 'plan_id': 'p1', 'scene_idx': scene,
                    'jitter_idx': 0, 'noise_idx': r['n'], 'scenario_seed': r['env_seed'], 'inference_seed': r['inference_seed']}
            (mdir / 'meta.json').write_text(json.dumps(meta))
            (mdir / 'video.mp4').write_bytes(b'v')
            tr = f'trace-n{r["n"]}.jsonl'
            (adir / tr).write_text(''.join(json.dumps({'event': 'http_send'}) + '\n' for _ in range(steps))
                                   + json.dumps({'event': 'complete'}) + '\n')
            lines.append({'n': r['n'], 'success': success, 'meta': str((mdir / 'meta.json').relative_to(adir)), 'attempt': 'a0',
                          'requests': [lo, hi], 'composed': len(cts), 'fired': len(set(cts) | set(fts)), 'http_sends': steps,
                          'first_fire_t': None, 'trace': tr, 'decisions_log': 'serve.decisions.jsonl'})
        (adir / 'serve.decisions.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in decs))
        (adir / 'progress.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in lines))
        progress[arm] = {x['n']: x for x in lines}
    return root, progress


def fake_rt(progress, audit=lambda done, out: []):
    return (FakeC, lambda out, m, arm: progress[arm], lambda mp, arm, check_files=False: None, audit, S)


class Collect(unittest.TestCase):
    def test_rows_timing_and_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = {A.QAM: {0: (0, None, [3], [])},
                    'mil_trqam_mix': {0: (1, 20, [4, 6], [2]), 1: (1, 9, [], [1]), 2: None},
                    'mil_trqam_basic': {1: 'badmeta'}}
            root, prog = make_root(tmp, spec)
            out = A.collect(root, 'h', fake_rt(prog))
            self.assertEqual(len(out['rows']), 3 * 4)                          # every manifest row x 4 arms
            r = {(x['policy'], x['n']): x for x in out['rows']}
            mix0 = r[('mil_trqam_mix', 0)]
            self.assertEqual((mix0['success_d'], mix0['first_composed_t'], mix0['first_fired_t']), (20, 4, 2))
            self.assertEqual((mix0['inclusive_count'], mix0['exclusive_after_composed_chunk']), (16, 15))
            self.assertEqual(mix0['timing_status'], 'valid')
            self.assertTrue(mix0['video'].endswith('video.mp4') and mix0['decisions_sha256'] and mix0['trace_http_sends'] == 20)
            mix1 = r[('mil_trqam_mix', 1)]                                    # fired but never composed
            self.assertEqual((mix1['timing_status'], mix1['first_fired_t'], mix1['inclusive_count']), ('no_composition', 1, None))
            q0 = r[(A.QAM, 0)]
            self.assertEqual((q0['timing_status'], q0['first_composed_t'], q0['inclusive_count']), ('failure', 3, None))
            miss = r[('mil_trqam_mix', 2)]
            self.assertEqual(miss['timing_status'], 'missing')
            self.assertIn('no progress line', miss['errors'][0])
            bad = r[('mil_trqam_basic', 1)]                                    # success with null first_success_step
            self.assertEqual(bad['timing_status'], 'invalid')
            self.assertTrue(any('meta timing schema' in e and 'meta.json' in e for e in bad['errors']))

    def test_audit_error_blocks_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, prog = make_root(tmp, {'mil_trqam_pbrs': {0: (1, 10, [2], [])}})
            audit = lambda done, out: ['n0 r3: gate rule mismatch'] if out.name == 'mil_trqam_pbrs' else []  # noqa: E731
            out = A.collect(root, 'h', fake_rt(prog, audit))
            r = {(x['policy'], x['n']): x for x in out['rows']}
            p0 = r[('mil_trqam_pbrs', 0)]
            self.assertEqual((p0['timing_status'], p0['audit_ok'], p0['inclusive_count']), ('invalid', False, None))
            self.assertTrue(r[('mil_trqam_pbrs', 1)]['audit_ok'])


# ------------------------------------------------------------------ merge
def row(fold, policy, n, success, status='valid', inc=5, base=0, audit_ok=True, host='h'):
    return {'host': host, 'root': 'r', 'fold_id': fold, 'instruction': 'I', 'scene': 1, 'policy': policy, 'n': n, 'j': 0,
            'env_seed': n, 'inference_seed': n, 'plan_id': 'p', 'base_success': base, 'success': success, 'audit_ok': audit_ok,
            'success_d': (inc + 1) if status == 'valid' else (9 if success else None),
            'first_composed_t': 1 if status in ('valid', 'failure') else None, 'inclusive_count': inc if status == 'valid' else None,
            'exclusive_after_composed_chunk': inc - 1 if status == 'valid' else None, 'timing_status': status,
            'meta': f'{fold}/{policy}/m{n}', 'video': f'{fold}/{policy}/v{n}', 'trace': 't', 'decisions': 'd', 'errors': []}


def col(rows, host='h', root='r'):
    return {'kind': A.KIND, 'host': host, 'root': root, 'fold_errors': [], 'unexpected_arms': [], 'skipped_smoke': [],
            'rows': rows}


class Merge(unittest.TestCase):
    def base_rows(self):
        rs = []
        for n in range(4):
            rs.append(row('F', A.QAM, n, 0 if n < 3 else 1, status='failure' if n < 3 else 'valid'))
        rs += [row('F', 'mil_trqam_basic', 0, 1, inc=3, base=1), row('F', 'mil_trqam_basic', 1, 1, 'no_composition'),
               row('F', 'mil_trqam_basic', 2, 1, inc=8), row('F', 'mil_trqam_basic', 3, 1, inc=2),   # n3: QAM succeeded
               row('F', 'mil_trqam_mix', 0, 1, inc=4), row('F', 'mil_trqam_mix', 1, 0, 'failure'),
               row('F', 'mil_trqam_mix', 2, 1, 'invalid', audit_ok=False), row('F', 'mil_trqam_mix', 3, 0, 'failure')]
        rs += [row('F', 'mil_trqam_pbrs', n, 0, 'failure') for n in range(4)]
        return rs

    def test_pair_selection_and_summary(self):
        rep = A.merge([col(self.base_rows())], 16, {'mil_trqam_basic': 3, 'mil_trqam_mix': 1, 'mil_trqam_pbrs': 0}, None, S.summ)
        self.assertEqual(rep['selected_counts'], {'mil_trqam_basic': 3, 'mil_trqam_pbrs': 0, 'mil_trqam_mix': 1})
        sel = {(c['policy'], c['n']) for c in rep['selected']}
        self.assertEqual(sel, {('mil_trqam_basic', 0), ('mil_trqam_basic', 1), ('mil_trqam_basic', 2), ('mil_trqam_mix', 0)})
        b = rep['summary']['policy'][('mil_trqam_basic',)]
        self.assertEqual((b['selected_N'], b['valid_timing_N'], b['no_composition_N'], b['archived_base_fail_N']), (3, 2, 1, 2))
        self.assertEqual((b['inclusive_count']['N'], b['inclusive_count']['median'], b['inclusive_count']['min']), (2, 5.5, 3))
        self.assertEqual(rep['summary']['policy'][('mil_trqam_pbrs',)]['selected_N'], 0)       # empty group kept
        self.assertEqual(rep['summary']['policy'][('mil_trqam_pbrs',)]['inclusive_count'], {'N': 0})
        self.assertEqual(rep['representatives']['mil_trqam_pbrs'], {})
        rb = rep['representatives']['mil_trqam_basic']
        self.assertEqual((rb['min']['n'], rb['median']['n'], rb['max']['n']), (0, 0, 2))     # lower median of 2
        self.assertTrue(any(u['n'] == 2 and u['policy'] == 'mil_trqam_mix' for u in rep['selection_unresolved']))
        self.assertTrue(any('archived_base_fail' in str(k) for k in rep['summary']['policy_archived_base']))
        self.assertEqual(rep['status'], 'ISSUES')                     # 1 fold != 27 etc.: never a silent PASS
        self.assertTrue(any('folds 1 != 27' in x for x in rep['issues']))

    def test_expected_count_mismatch_reported(self):
        rep = A.merge([col(self.base_rows())], 16, {'mil_trqam_basic': 65}, None, S.summ)
        self.assertTrue(any('selected mil_trqam_basic: 3 != expected 65' in x for x in rep['issues']))

    def test_duplicate_coordinate_refused_across_hosts(self):
        rs = self.base_rows()
        dup = dict(rs[0], host='other')
        with self.assertRaises(A.AnalysisError):
            A.merge([col(rs), col([dup], host='other', root='r2')], 17, None, None, S.summ)

    def test_identity_mismatch_across_policies(self):
        rs = self.base_rows()
        rs[5] = dict(rs[5], env_seed=999)
        rep = A.merge([col(rs)], 16, None, None, S.summ)
        self.assertTrue(any('identity differs' in x for x in rep['issues']))

    def test_reference_check(self):
        ref = {'rows': [{'fold_id': r['fold_id'], 'n': r['n'], 'policy': r['policy'], 'success': r['success'], 'audit_ok': True}
                        for r in self.base_rows()]}
        rep = A.merge([col(self.base_rows())], 16, None, ref, S.summ)
        # mix n2 failed OUR audit but is a success in the reference -> explicit mismatch, not silently dropped
        self.assertTrue(any(x.startswith('reference mismatch mil_trqam_mix') for x in rep['issues']))
        self.assertFalse(any(x.startswith('reference mismatch mil_trqam_basic') for x in rep['issues']))

    def test_write_outputs_and_refuse_overwrite(self):
        rep = A.merge([col(self.base_rows())], 16, None, None, S.summ)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'o'
            A.write(rep, out, {'input': []})
            for f in ('rows.json', 'rows.csv', 'selected_cases.json', 'selected_cases.csv', 'summary.json',
                      'summary_policy.csv', 'summary_task.csv', 'summary_fold.csv', 'report.json', 'report.md'):
                self.assertTrue((out / f).is_file(), f)
            self.assertIn('Status: ISSUES', (out / 'report.md').read_text())
            with self.assertRaises(FileExistsError):
                A.write(rep, out, {'input': []})
            inp = Path(tmp) / 'c.json'
            inp.write_text(json.dumps(col(self.base_rows())))
            self.assertEqual(A.main(['merge', '--input', str(inp), '--expected-rows', '16', '--out-dir', str(out)]), 2)
            self.assertEqual(A.main(['collect', '--root', tmp, '--host', 'h', '--out', str(inp)]), 2)

    def test_runtime_imports_unchanged_functions(self):
        C, load_progress, verify_manifest, audit_decisions, SS = A._rt()
        self.assertEqual(set(A.POLICIES), set(C.PRODUCTION_ARMS))
        self.assertEqual((load_progress.__module__, audit_decisions.__module__), ('run_eval_arm', 'verify_fold'))
        self.assertEqual((SS.K, SS.MAX_STEPS), (5, 720))


if __name__ == '__main__':
    unittest.main()

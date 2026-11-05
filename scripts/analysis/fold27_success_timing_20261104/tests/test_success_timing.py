"""success_timing: unit contracts (decision index on both sides), eval collection through fake snapshot/progress, baseline
index (old/new extractor JSON), pairing errors are listed not filled, selection subsets, CLI new-paths-only."""
import hashlib, json, sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import success_timing as ST   # noqa: E402

INS, PLAN = 'OpenDrawer/right', 'p1'


def meta(succ, d=None, **kw):
    m = {'episode_success': int(succ), 'first_success_step': d, 'steps': d if succ else 144, 'n_action_steps': 5,
         'max_episode_steps': 720}
    m.update(kw)
    return m


def base_rec(n, succ, d=None, probe=True, lag=0, **kw):
    r = {'episode_id': f'{PLAN}/kanu/{INS}/s1/j4/n{n}', 'plan_id': PLAN, 'machine': 'kanu', 'instruction': INS, 'scene': 1,
         'jitter': 4, 'noise': n, 'success': int(succ), 'n_inferences': d if succ else 144, 'env_steps': 5 * d if succ else 720,
         'max_episode_steps': 720}
    if probe:
        r['first_success_step'] = (5 * d - lag - 3) if succ else -1
        r['success_chunk_lag_steps'] = lag if succ else -1
    r.update(kw)
    return r


class Units(unittest.TestCase):
    def test_eval_d(self):
        self.assertEqual(ST.eval_d(meta(1, 40)), 40); self.assertIsNone(ST.eval_d(meta(0)))
        for m in (meta(1, 40, steps=41), meta(1, 40, n_action_steps=16), meta(1, 40, max_episode_steps=500), meta(1, 0),
                  meta(1, 145, steps=145), meta(0, 12), meta(1, None)):
            with self.assertRaises(ST.TimingError):
                ST.eval_d(m)

    def test_base_d(self):
        self.assertEqual(ST.base_d(base_rec(1, 1, 30)), (30, 147))
        self.assertEqual(ST.base_d(base_rec(1, 1, 30, lag=5))[0], 30)              # probe one chunk earlier than the stop
        self.assertEqual(ST.base_d(base_rec(1, 1, 30, probe=False)), (30, None))    # older extractor JSON: no probe fields
        self.assertEqual(ST.base_d(base_rec(1, 0)), (None, None))
        bad = [base_rec(1, 1, 30, env_steps=151), base_rec(1, 1, 30, n_inferences=29), base_rec(1, 1, 30, lag=10),
               base_rec(1, 1, 30, success_chunk_lag_steps=5), base_rec(1, 0, env_steps=700), base_rec(1, 1, 30, max_episode_steps=500)]
        r = base_rec(1, 1, 30); del r['success_chunk_lag_steps']; bad.append(r)
        for b in bad:
            with self.assertRaises(ST.TimingError):
                ST.base_d(b)


class Pipeline(unittest.TestCase):
    def setUp(self):
        self.t = Path(tempfile.mkdtemp())
        self.root = self.t / 'root'; fd = self.root / 'F1'; fd.mkdir(parents=True)
        self.m = {'fold_id': 'F1', 'instruction': INS, 'scene': 1, 'archive_machine': 'kanu',
                  'rows': [{'n': n, 'j': 4, 'plan_id': PLAN, 'base_success': b} for n, b in ((0, 1), (1, 1), (2, 0), (3, 1))]}
        (fd / 'manifest.json').write_text(json.dumps(self.m))
        self.metas = {'A': {0: meta(1, 30), 1: meta(1, 41), 2: meta(1, 50), 3: meta(0)},
                      'B': {0: meta(1, 28), 1: meta(0), 2: meta(0), 3: meta(1, 60)}}
        for arm, ms in self.metas.items():
            for n, mm in ms.items():
                p = fd / arm / f'rows/n{n}/meta.json'; p.parent.mkdir(parents=True); p.write_text(json.dumps(mm))
        self.bdir = self.t / 'fe'; self.bdir.mkdir()
        for i, r in enumerate([base_rec(0, 1, 30), base_rec(1, 1, 40, probe=False), base_rec(2, 0), base_rec(3, 1, 60)]):
            (self.bdir / f'e{i}.json').write_text(json.dumps(r))
        (self.bdir / 'spec.json').write_text(json.dumps({'extractor_sha256': 'x'}))

    def snapshot(self, root, host):
        mp = Path(root) / 'F1' / 'manifest.json'
        arms = {a: {'rows': [{'n': n, 'j': 4, 'plan_id': PLAN, 'base_success': self.m['rows'][n]['base_success'],
                              'success': int(bool(mm['episode_success'])), 'audit_ok': not (a == 'B' and n == 2),
                              'fired': n, 'first_fire_t': None} for n, mm in ms.items()]} for a, ms in self.metas.items()}
        return {'time': 1.0, 'skipped_smoke': [], 'folds': [{'fold_id': 'F1', 'instruction': INS, 'scene': 1, 'arms': arms,
                                                              'manifest_sha256': hashlib.sha256(mp.read_bytes()).hexdigest()}]}

    def progress(self, d, m, arm):
        return {n: {'meta': f'rows/n{n}/meta.json'} for n in self.metas[arm]}

    def test_end_to_end(self):
        ev = ST.collect_eval(self.root, 'kanu', self.snapshot, self.progress)
        self.assertEqual(len(ev['rows']), 7)
        self.assertEqual([e['n'] for e in ev['errors']], [2])                   # failed audit row excluded, listed
        bi = ST.collect_baseline([self.bdir])
        self.assertEqual((len(bi['episodes']), bi['errors'], len(bi['skipped_non_episode'])), (4, [], 1))
        rep = ST.build([ev], [bi])
        got = sorted((p['policy'], p['n'], p['d_base'], p['d_eval'], p['delta']) for p in rep['pairs'])
        self.assertEqual(got, [('A', 0, 30, 30, 0), ('A', 1, 40, 41, 1), ('B', 0, 30, 28, -2), ('B', 3, 60, 60, 0)])
        self.assertEqual(rep['counts']['A'], {'rows': 4, 'base_succ': 3, 'eval_succ': 3, 'succ_succ': 2})
        self.assertEqual(rep['common_subset']['coordinates'], 1)                 # n0 only: base + A + B all succeeded
        self.assertEqual(rep['per_policy'][('B',)]['faster'], 1)
        out = self.t / 'tab'
        ST.write(rep, out, {})
        self.assertIn('Common subset (1 coordinates', (out / 'report.md').read_text())
        self.assertEqual(len((out / 'pairs.csv').read_text().strip().splitlines()), 5)

    def test_pairing_errors_not_filled(self):
        ev = ST.collect_eval(self.root, 'kanu', self.snapshot, self.progress)
        bi = ST.collect_baseline([self.bdir])
        del bi['episodes'][f'{PLAN}|{INS}|1|4|3']                                 # missing baseline JSON
        bi['episodes'][f'{PLAN}|{INS}|1|4|0']['success'] = 0                     # disagreement with manifest base_success
        bi['episodes'][f'{PLAN}|{INS}|1|4|1']['machine'] = 'srv48'
        rep = ST.build([ev], [bi])
        errs = sorted((e['policy'], e['n'], e['error'].split(' ')[0]) for e in rep['errors'])
        self.assertEqual(errs, [('A', 0, 'manifest'), ('A', 1, 'archive'), ('A', 3, 'baseline'),
                                ('B', 0, 'manifest'), ('B', 1, 'archive'), ('B', 2, 'TimingError:'), ('B', 3, 'baseline')])
        self.assertEqual(rep['pairs'], [])

    def test_meta_unit_violation_is_error(self):
        self.metas['A'][0] = meta(1, 30, steps=35)
        (self.root / 'F1/A/rows/n0/meta.json').write_text(json.dumps(self.metas['A'][0]))
        ev = ST.collect_eval(self.root, 'kanu', self.snapshot, self.progress)
        self.assertIn('steps', next(e for e in ev['errors'] if e['policy'] == 'A' and e['n'] == 0)['error'])

    def test_duplicates_refused(self):
        ev = ST.collect_eval(self.root, 'kanu', self.snapshot, self.progress)
        bi = ST.collect_baseline([self.bdir])
        with self.assertRaises(ST.TimingError):
            ST.build([ev, ev], [bi])
        with self.assertRaises(ST.TimingError):
            ST.build([ev], [bi, bi])
        b2 = ST.collect_baseline([self.bdir, self.bdir])
        self.assertTrue(all('duplicate' in e['error'] for e in b2['errors'])); self.assertEqual(len(b2['errors']), 4)

    def test_cli_new_paths_only(self):
        (self.t / 'x.json').write_text('keep')
        self.assertEqual(ST.main(['baseline', '--dir', str(self.bdir), '--out', str(self.t / 'x.json')]), 2)
        self.assertEqual((self.t / 'x.json').read_text(), 'keep')
        self.assertEqual(ST.main(['baseline', '--dir', str(self.bdir), '--out', str(self.t / 'b.json')]), 0)
        self.assertEqual(ST.main(['table', '--eval', str(self.t / 'b.json'), '--baseline', str(self.t / 'b.json'),
                                  '--out-dir', str(self.t)]), 2)


if __name__ == '__main__':
    unittest.main()

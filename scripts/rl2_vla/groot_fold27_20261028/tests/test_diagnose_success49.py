"""diagnose_success49 (read-only) on synthetic pkls: the candle-type transient (first 452, end 455) passes the historical
60b65fe8 rule and fails the current rule with the exact message; continuous / one-chunk-lag / failure episodes;
sha mismatch reported (not raised); --missing-from picks PROGRESS 'missing' in extraction (sorted-stem) order;
compare-16 reads the original series; inputs are never modified; --out is exclusive."""
import hashlib, json, pickle, shutil, sys, tempfile, unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'fold_data'))
import diagnose_success49 as D    # noqa: E402
import extract_scene as ES        # noqa: E402
import extract_scene49 as E49     # noqa: E402

INSTR, SC, MX = 'PPCC/candle', 0, 720


def series(steps, true_idx):
    return [i in true_idx for i in range(steps)]


CASES = {   # n -> (episode_success, env_step_success)
    0: (True, series(455, {452})),                       # candle-type transient: true only at 452, stop at 455
    1: (True, series(140, set(range(134, 140)))),         # OvenRack observed: continuous, one-chunk lag
    2: (True, series(205, set(range(200, 205)))),         # continuous, lag 0
    3: (False, series(MX, set())),                        # failure at horizon
}


class DiagTest(unittest.TestCase):
    def setUp(self):
        self.t = Path(tempfile.mkdtemp()); self.store = self.t / 'store'
        rows = []
        for n, (succ, es) in CASES.items():
            k = ('p', 'worker2', INSTR, SC, 0, n)
            payload = {'env_step_success': es, 'episode_success': succ, 'max_episode_steps': MX, 'env_step_n_action_steps': 5,
                       'env_step_phases': ['x'] * (len(es) + 1), 'hidden_states': [None] * (-(-len(es) // 5)),
                       'actions': [None] * (-(-len(es) // 5)), 'first_success_step': (es.index(True) + 1) if succ else None,
                       'rewards_unrelated': np.zeros(3)}
            p = E49.source_path(self.store, k); p.parent.mkdir(parents=True)
            b = pickle.dumps(payload); p.write_bytes(b)
            rows.append({'plan_id': 'p', 'j': 0, 'n': n, 'success': int(succ), 'pkl_sha256': hashlib.sha256(b).hexdigest()})
        rows[3]['pkl_sha256'] = '0' * 64                        # sha mismatch must be REPORTED
        self.spec = {'folds': [{'fold_id': 'f', 'instruction': INSTR, 'scene': SC, 'archive_machine': 'worker2', 'target_j': 0,
                                'fit': rows[:2], 'eval': rows[2:]}]}
        (self.t / 'spec.json').write_text(json.dumps(self.spec))
        self.eid = {n: ES.episode_id('p', 'worker2', INSTR, SC, 0, n) for n in CASES}
        self.before = {p: p.read_bytes() for p in self.store.rglob('*.pkl')}

    def tearDown(self):
        for p, b in self.before.items():
            assert p.read_bytes() == b, 'source modified'
        shutil.rmtree(self.t)

    def run_all(self, **kw):
        return D.run(self.t / 'spec.json', self.store, f'{INSTR}|s{SC}', all_=True, **kw)

    def ep(self, r, n):
        return next(x for x in r['episodes'] if x['episode_id'] == self.eid[n])

    def test_transient_passes_historical_fails_current(self):
        r = self.run_all(); e = self.ep(r, 0)
        self.assertEqual(e['rule_60b65fe8'], 'OK')
        self.assertEqual(e['rule_current_90e69e53'], 'FAIL probe success not continuous from first 452 to end 455; transient success unexplained')
        self.assertEqual(e['probe_true_runs'], [[452, 452]]); self.assertFalse(e['final_probe']); self.assertEqual(e['first_success_chunk_end'], 455)
        self.assertEqual(e['probe_window']['bits'][-3:], '100')
        s = r['summary']
        self.assertEqual(s['fail_rule_current'], [self.eid[0]]); self.assertEqual(s['fail_rule_60b65fe8'], [self.eid[1]])
        self.assertEqual(s['non_continuous_success'], [self.eid[0]]); self.assertEqual(s['sha_mismatch'], [self.eid[3]])
        self.assertIn('first_success_step', e['related_fields'])

    def test_other_cases(self):
        r = self.run_all()
        self.assertEqual(self.ep(r, 1)['rule_current_90e69e53'], 'OK lag=5')
        self.assertTrue(self.ep(r, 1)['rule_60b65fe8'].startswith('FAIL'))       # why 01a528a3 relaxed the lag
        self.assertEqual(self.ep(r, 2)['rule_current_90e69e53'], 'OK lag=0'); self.assertEqual(self.ep(r, 2)['rule_60b65fe8'], 'OK')
        self.assertEqual(self.ep(r, 3)['rule_current_90e69e53'], 'OK lag=-1'); self.assertFalse(self.ep(r, 3)['sha_ok'])

    def test_missing_from_order_and_compare16(self):
        d = self.t / 'derived'; d.mkdir()
        stems = {n: E49.stem_of(self.eid[n]) for n in CASES}
        (d / 'PROGRESS.json').write_text(json.dumps({'done': 2, 'missing': [stems[0], stems[2]]}))
        o = self.t / 'orig16'; o.mkdir()
        np.savez(o / 'a.npz', env_step_success=np.asarray(CASES[0][1], np.int8), env_steps=np.int32(455), first_success_step=np.int32(452))
        (o / f'{stems[0]}.json').write_text(json.dumps({'npz': 'a.npz', 'pkl_sha256': 'x'}))
        r = D.run(self.t / 'spec.json', self.store, f'{INSTR}|s{SC}', missing_from=d, compare_16=o)
        self.assertEqual([x['stem'] for x in r['episodes']], sorted([stems[0], stems[2]]))
        self.assertEqual(r['summary']['first_in_extract_order'], r['episodes'][0]['episode_id'])
        c = self.ep(r, 0)['compare_16']
        self.assertTrue(c['env_step_success_identical'] and c['env_steps_equal'] and c['first_equal'])
        self.assertEqual(self.ep(r, 2)['compare_16'], {'present': False})
        self.assertEqual((d / 'PROGRESS.json').read_text(), json.dumps({'done': 2, 'missing': [stems[0], stems[2]]}))

    def test_cli_out_exclusive(self):
        out = self.t / 'rep.json'
        args = ['--spec', str(self.t / 'spec.json'), '--store', str(self.store), '--scene', f'{INSTR}|s{SC}', '--episode', self.eid[0], '--out', str(out)]
        self.assertEqual(D.main(args), 0)
        self.assertEqual(json.loads(out.read_text())['summary']['episodes'], 1)
        with self.assertRaises(FileExistsError):
            D.main(args)


if __name__ == '__main__':
    unittest.main()

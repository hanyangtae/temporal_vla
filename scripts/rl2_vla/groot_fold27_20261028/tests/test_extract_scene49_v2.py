"""extract_scene49_v2 (collector-semantics success contract, NEW version/root) on synthetic pickles; CPU only.
Covers: the real candle j0 n4 series (True 452,453 / False 454 / end 455) passes v2 and still fails v1; every other
v1 rule unchanged; extract_scene.success_lag binding restored (also on error); arrays bit-identical to the original
convert for non-transient episodes; compare-16 against a historical-style record; labels never changed; v1 root/contract
refused; space guard = remaining raw upper bound + one temp record + metadata + reserve, re-measured before every
record (disk drop -> stop with committed records kept -> resume recomputes only the rest); tampered resume refused;
missing raw termination fields named, not invented."""
import hashlib, json, pickle, shutil, sys, tempfile, types, unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'fold_data'))
import extract_scene as ES       # noqa: E402
import extract_scene49 as E49    # noqa: E402
import extract_scene49_v2 as V2  # noqa: E402

SCENE = 'PPCC/candle|s0'


def series(steps, true_idx):
    return [i in true_idx for i in range(steps)]


def payload(seed, success, es, max_steps, n=2):
    rng = np.random.default_rng(seed)
    return {'capture_layers': ES.LAYERS, 'hidden_states': [rng.standard_normal((7, 4, 49, 1536)).astype(np.float32) for _ in range(n)],
            'actions': [{f'action.{k}': rng.standard_normal((16, d)).astype(np.float32) for k, d in ES.ACTION_KEYS} for _ in range(n)],
            'env_step_success': es, 'env_step_n_action_steps': 5, 'max_episode_steps': max_steps, 'episode_success': success,
            'env_step_phases': ['x'] * (len(es) + 1)}


# (j, n) -> (success, env_step_success, max_episode_steps); all have 10 steps = 2 records
CASES = {(0, 4): (True, series(10, {7, 8}), 20),          # candle-type transient: True 7,8 / False 9 / end 10
         (0, 0): (True, series(10, {5, 6, 7, 8, 9}), 20),  # continuous, lag 0
         (1, 0): (False, series(10, set()), 10),           # failure at horizon
         (1, 1): (True, series(10, {4, 5, 6, 7, 8, 9}), 20)}  # continuous, one-chunk lag (OvenRack type)


class RuleTest(unittest.TestCase):
    def test_real_candle_series(self):
        es = series(455, {452, 453})
        self.assertEqual(V2.success_lag_v2(es, 452, 455, 5, 720, True), 0)
        with self.assertRaisesRegex(ES.ExtractError, 'not continuous from first 452 to end 455'):
            ES.success_lag(es, 452, 455, 5, 720, True)                         # v1 / original unchanged

    def test_other_rules_unchanged(self):
        ok = [(series(205, set(range(200, 205))), 200, 205, 0), (series(140, set(range(134, 140))), 134, 140, 5),
              (series(140, {134}), 134, 140, 5)]
        for es, first, steps, lag in ok:
            self.assertEqual(V2.success_lag_v2(es, first, steps, 5, 720, True), lag)
        bad = [(series(145, {134}), 134, 145, True, 'stop lag 10'), (series(143, {134}), 134, 143, True, 'chunk boundary'),
               (series(140, set()), -1, 140, True, 'without any probe'), (series(130, {120}), 120, 120, True, 'ended at'),
               (series(720, {3}), 3, 720, False, 'failure episode'), (series(700, set()), -1, 700, False, 'failure episode')]
        for es, first, steps, succ, msg in bad:
            with self.assertRaisesRegex(ES.ExtractError, msg):
                V2.success_lag_v2(es, first, steps, 5, 720, succ)
        self.assertEqual(V2.success_lag_v2(series(720, set()), -1, 720, 5, 720, False), -1)
        self.assertEqual(V2.success_lag_v2(series(720, {718}), 718, 720, 5, 720, True), 0)     # horizon end

    def test_binding_restored_and_v1_still_refuses(self):
        orig = ES.success_lag
        p = payload(0, True, series(10, {7, 8}), 20)
        V2.convert49_v2(p)
        self.assertIs(ES.success_lag, orig)
        with self.assertRaisesRegex(ES.ExtractError, 'not continuous'):
            E49.convert49(p)                                                     # v1 path unaffected
        bad = payload(1, True, series(10, set()), 20)
        with self.assertRaises(ES.ExtractError):
            V2.convert49_v2(bad)
        self.assertIs(ES.success_lag, orig)                                      # restored on error

    def test_space_need_numbers(self):
        self.assertEqual(V2.space_need(200, 3.0), 7_757_329_744)
        self.assertEqual(V2.space_need(1, 3.0) - V2.space_need(0, 3.0) - 2 * 22_233_808, 67_108_864 + 3 * 1024 ** 3)
        self.assertEqual(V2.space_need(0, 3.0), 0)


class RunTest(unittest.TestCase):
    def setUp(self):
        self.t = Path(tempfile.mkdtemp()); self.store = self.t / 'store'
        rows, self.payloads = [], {}
        for i, ((j, n), (succ, es, mx)) in enumerate(sorted(CASES.items())):
            p = payload(i, succ, es, mx); k = ('p', 'worker2', 'PPCC/candle', 0, j, n)
            path = E49.source_path(self.store, k); path.parent.mkdir(parents=True)
            b = pickle.dumps(p); path.write_bytes(b)
            rows.append({'plan_id': 'p', 'j': j, 'n': n, 'pkl_sha256': hashlib.sha256(b).hexdigest(), 'success': int(succ)})
            self.payloads[k] = p
        self.rows = rows
        self.write_spec()
        self.out = self.t / 'derived' / 'fold27_extract49_v2_20261103' / 'srv50_candle_s0'
        self.big = lambda p: types.SimpleNamespace(free=10 ** 13)
        self.src_bytes = {p: p.read_bytes() for p in self.store.rglob('*.pkl')}

    def write_spec(self):
        spec = {'production_ready': True, 'folds': [
            {'fold_id': 'PPCC-candle_s0_j0', 'instruction': 'PPCC/candle', 'scene': 0, 'archive_machine': 'worker2',
             'fit': [r for r in self.rows if r['j'] != 0], 'eval': [r for r in self.rows if r['j'] == 0]}]}
        self.spec = self.t / 'spec.json'; self.spec.write_text(json.dumps(spec))

    def tearDown(self):
        for p, b in self.src_bytes.items():
            assert p.read_bytes() == b, 'source pkl modified'
        shutil.rmtree(self.t)

    def run2(self, out=None, **kw):
        kw.setdefault('disk', self.big)
        return V2.run_scene(self.spec, self.store, out or self.out, SCENE, kw.pop('expect', 4), log=lambda s: None, **kw)

    def stem(self, k):
        return E49.stem_of(ES.episode_id(*k))

    def make_orig16(self):
        """historical-style 16-token dir (60b65fe8 had no continuity rule and no lag key)."""
        d = self.t / 'orig16'; d.mkdir()
        (d / 'contract.json').write_text(json.dumps({'with_last_block': True, 'last_block_tokens': [33, 49]}))
        for k, p in self.payloads.items():
            s = self.stem(k)
            with V2.collector_rule():
                a = ES.convert(p, True)
            a.pop('success_chunk_lag_steps')
            np.savez(d / f'{s}.npz', **a)
            b = E49.source_path(self.store, k).read_bytes()
            (d / f'{s}.json').write_text(json.dumps({'episode_id': ES.episode_id(*k), 'pkl_sha256': hashlib.sha256(b).hexdigest(),
                                                     'npz': f'{s}.npz', 'npz_sha256': ES.sha256_bytes((d / f'{s}.npz').read_bytes())}))
        return d

    def test_full_scene_v2_contract_records_and_bitwise_arrays(self):
        r = self.run2(compare_16=self.make_orig16())
        self.assertEqual((r['computed'], r['episodes']), (4, 4))
        tk = ('p', 'worker2', 'PPCC/candle', 0, 0, 4)
        self.assertEqual(r['transient_success_episodes'], [ES.episode_id(*tk)])
        c = json.loads((self.out / 'COMPLETE.json').read_text())
        self.assertEqual(c['contract']['kind'], 'fold27_extract49_v2'); self.assertEqual(c['contract']['success_rule'], 'collector_v2')
        self.assertEqual(c['contract']['last_block_tokens'], [0, 49]); self.assertTrue(c['contract']['with_last_block'])
        self.assertEqual(c['transient_success_episodes'], [ES.episode_id(*tk)])
        rec = json.loads((self.out / f'{self.stem(tk)}.json').read_text())
        self.assertEqual((rec['success'], rec['first_success_step'], rec['env_steps'], rec['success_chunk_lag_steps']), (1, 7, 10, 0))
        self.assertEqual((rec['probe_true_runs'], rec['final_probe'], rec['transient_success']), ([[7, 8]], False, True))
        self.assertIn('terminated', rec['raw_termination']['missing_standard'])
        self.assertIn('truncated', rec['raw_termination']['missing_standard'])
        for k, p in self.payloads.items():
            rr = json.loads((self.out / f'{self.stem(k)}.json').read_text())
            self.assertEqual(rr['success'], int(p['episode_success']))                          # labels untouched
            with np.load(self.out / f'{self.stem(k)}.npz') as z:
                self.assertEqual((z['last_block_d0'].dtype, z['last_block_d0'].shape), (np.dtype(np.float16), (2, 49, 1536)))
                if k != tk:
                    ref = ES.convert(p, False)                                                    # original code, v1 rule
                    for key in ref:
                        self.assertTrue(np.array_equal(z[key], ref[key]), key)
                self.assertEqual([bool(v) for v in z['env_step_success']], p['env_step_success'])
        self.assertEqual(json.loads((self.out / 'PROGRESS.json').read_text())['missing'], [])

    def test_compare16_mismatch_refused(self):
        o = self.make_orig16()
        tk = ('p', 'worker2', 'PPCC/candle', 0, 0, 4); s = self.stem(tk)
        with np.load(o / f'{s}.npz') as z:
            a = {k: z[k] for k in z.files}
        a['env_step_success'] = a['env_step_success'].copy(); a['env_step_success'][9] = 1
        np.savez(o / f'{s}.npz', **a)
        j = json.loads((o / f'{s}.json').read_text()); j['npz_sha256'] = ES.sha256_bytes((o / f'{s}.npz').read_bytes())
        (o / f'{s}.json').write_text(json.dumps(j))
        with self.assertRaisesRegex(E49.Extract49Error, 'env_step_success'):
            self.run2(compare_16=o)
        self.assertFalse((self.out / f'{s}.json').exists()); self.assertFalse((self.out / 'COMPLETE.json').exists())

    def test_label_mismatch_with_spec_refused(self):
        for r in self.rows:
            if (r['j'], r['n']) == (0, 4):
                r['success'] = 0
        self.write_spec()
        with self.assertRaisesRegex(V2.Extract49V2Error, 'pkl success 1 != spec 0'):
            self.run2()

    def test_v1_root_and_v1_contract_refused(self):
        with self.assertRaisesRegex(V2.Extract49V2Error, 'v1 root'):
            self.run2(out=self.t / 'fold27_extract49_v1_20261103' / 'srv50_candle_s0')
        d = self.t / 'other'; d.mkdir()
        (d / 'contract.json').write_text(json.dumps({'kind': 'fold27_extract49_v1'}))
        with self.assertRaisesRegex(V2.Extract49V2Error, 'different contract'):
            self.run2(out=d)

    def test_space_guard_upfront_drop_and_resume(self):
        need4 = V2.space_need(4, 3.0)
        plan = self.run2(estimate=True, disk=lambda p: types.SimpleNamespace(free=need4))
        self.assertTrue(plan['fits']); self.assertEqual(plan['headroom_bytes'], 0); self.assertFalse(self.out.exists())
        with self.assertRaisesRegex(V2.Extract49V2Error, 'insufficient space'):
            self.run2(disk=lambda p: types.SimpleNamespace(free=need4 - 1))
        self.assertFalse(self.out.exists())                                         # nothing written
        calls = {'n': 0}

        def dropping(p):
            calls['n'] += 1
            return types.SimpleNamespace(free=10 ** 13 if calls['n'] <= 3 else V2.space_need(2, 3.0) - 1)
        with self.assertRaisesRegex(V2.Extract49V2Error, 'space guard before'):
            self.run2(disk=dropping)                                                # plan + 2 records, then disk drop
        prog = json.loads((self.out / 'PROGRESS.json').read_text())
        self.assertEqual((prog['done'], len(prog['missing'])), (2, 2))
        self.assertEqual(list(self.out.glob('*.tmp.npz')), []); self.assertFalse((self.out / 'COMPLETE.json').exists())
        kept = {p.name: p.read_bytes() for p in self.out.glob('*.npz')}
        r = self.run2()
        self.assertEqual((r['computed'], r['skipped'], r['episodes']), (2, 2, 4))
        for name, b in kept.items():
            self.assertEqual((self.out / name).read_bytes(), b)                      # resume never rewrites

    def test_resume_refuses_tampered_record(self):
        self.run2()
        tk = ('p', 'worker2', 'PPCC/candle', 0, 0, 4); jp = self.out / f'{self.stem(tk)}.json'
        rec = json.loads(jp.read_text()); rec['success_rule'] = 'v1'; jp.write_text(json.dumps(rec))
        with self.assertRaisesRegex(V2.Extract49V2Error, 'committed record mismatch'):
            self.run2()


if __name__ == '__main__':
    unittest.main()

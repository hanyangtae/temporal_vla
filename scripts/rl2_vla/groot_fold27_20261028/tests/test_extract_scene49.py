"""extract_scene49 on synthetic rollout pickles (original layout, numpy instead of torch; CPU only)."""
import hashlib, json, pickle, shutil, sys, tempfile, types, unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'fold_data'))
import extract_scene as ES     # noqa: E402
import extract_scene49 as X    # noqa: E402

SCENE = 'PPCC/jug|s1'


def payload(seed, success, n=2, shape=(7, 4, 49, 1536)):
    rng = np.random.default_rng(seed)
    hs = [rng.standard_normal(shape).astype(np.float32) for _ in range(n)]
    steps = 10
    es = [False] * steps if not success else [False] * 7 + [True] * 3
    return {'capture_layers': ES.LAYERS, 'hidden_states': hs,
            'actions': [{f'action.{k}': rng.standard_normal((16, d)).astype(np.float32) for k, d in ES.ACTION_KEYS} for _ in range(n)],
            'env_step_success': es, 'env_step_n_action_steps': 5, 'max_episode_steps': 10, 'episode_success': success,
            'env_step_phases': ['x'] * (steps + 1)}


class Extract49Test(unittest.TestCase):
    def setUp(self):
        self.t = Path(tempfile.mkdtemp())
        self.store = self.t / 'store'
        rows, self.payloads = [], {}
        for i, (j, n, succ) in enumerate(((0, 0, False), (0, 1, True), (1, 0, False), (2, 0, True))):
            p = payload(i, succ)
            k = ('p', 'worker2', 'PPCC/jug', 1, j, n)
            path = X.source_path(self.store, k); path.parent.mkdir(parents=True)
            blob = pickle.dumps(p); path.write_bytes(blob)
            rows.append({'plan_id': 'p', 'j': j, 'n': n, 'pkl_sha256': hashlib.sha256(blob).hexdigest(), 'success': int(succ)})
            self.payloads[k] = p
        spec = {'production_ready': True, 'folds': [
            {'fold_id': 'PPCC-jug_s1_j0', 'instruction': 'PPCC/jug', 'scene': 1, 'archive_machine': 'worker2', 'fit': rows[2:], 'eval': rows[:2]},
            {'fold_id': 'PPCC-jug_s1_j1', 'instruction': 'PPCC/jug', 'scene': 1, 'archive_machine': 'worker2', 'fit': rows[:2] + rows[3:], 'eval': rows[2:3]},
            {'fold_id': 'PPCC-bread_s4_j1', 'instruction': 'PPCC/bread', 'scene': 4, 'archive_machine': 'worker2', 'fit': [], 'eval': []}]}
        self.spec = self.t / 'spec.json'; self.spec.write_text(json.dumps(spec))
        self.out = self.t / 'out49'
        self.big = lambda p: types.SimpleNamespace(free=10 ** 13)

    def tearDown(self):
        shutil.rmtree(self.t)

    def run49(self, **kw):
        kw.setdefault('disk', self.big)
        return X.run_scene(self.spec, self.store, self.out, SCENE, kw.pop('expect', 4), log=lambda s: None, **kw)

    def make_orig16(self):
        """original-style 16-token extraction dir (extract_scene.convert with_last_block=True)."""
        d = self.t / 'orig16'; d.mkdir()
        (d / 'contract.json').write_text(json.dumps({'with_last_block': True, 'last_block_tokens': [33, 49]}))
        for k, p in self.payloads.items():
            s = X.stem_of(ES.episode_id(*k))
            np.savez(d / f'{s}.npz', **ES.convert(p, True))
            blob = X.source_path(self.store, k).read_bytes()
            (d / f'{s}.json').write_text(json.dumps({'episode_id': ES.episode_id(*k), 'pkl_sha256': hashlib.sha256(blob).hexdigest(),
                                                     'npz': f'{s}.npz', 'npz_sha256': ES.sha256_bytes((d / f'{s}.npz').read_bytes())}))
        return d

    # ------------------------------------------------------------------------------------------------------------
    def test_exact49_indexing_dtype_and_original_cross_check(self):
        r = self.run49(compare_16=self.make_orig16())
        self.assertEqual((r['computed'], r['episodes']), (4, 4))
        c = json.loads((self.out / 'COMPLETE.json').read_text())
        self.assertEqual((c['episodes'], c['decisions'], c['compare_16']), (4, 8, True))
        self.assertEqual(c['contract']['kind'], X.KIND); self.assertEqual(c['contract']['last_block_tokens'], [0, 49])
        for k, p in self.payloads.items():
            s = X.stem_of(ES.episode_id(*k))
            rec = json.loads((self.out / f'{s}.json').read_text())
            self.assertEqual(rec['source_hidden_dtype'], 'float32')
            with np.load(self.out / f'{s}.npz') as z:
                lb = z['last_block_d0']
                self.assertEqual((lb.dtype, lb.shape), (np.dtype(np.float16), (2, 49, 1536)))
                want = np.stack([h[ES.LAYERS.index(15), 0, 0:49].astype(np.float16) for h in p['hidden_states']])
                self.assertTrue(np.array_equal(lb, want))
                self.assertFalse(np.array_equal(lb, np.stack([h[ES.LAYERS.index(15), 1, 0:49].astype(np.float16) for h in p['hidden_states']])))
                ref = ES.convert(p, False)
                for key in ref:
                    self.assertTrue(np.array_equal(z[key], ref[key]), key)          # original arrays unchanged
        self.assertEqual(json.loads((self.out / 'PROGRESS.json').read_text())['missing'], [])

    def test_compare16_mismatch_refuses_commit(self):
        orig = self.make_orig16()
        k = sorted(self.payloads)[0]; s = X.stem_of(ES.episode_id(*k))
        with np.load(orig / f'{s}.npz') as z:
            arrs = {n: z[n] for n in z.files}
        arrs['last_block_d0'] = arrs['last_block_d0'].copy(); arrs['last_block_d0'][0, 0, 0] += np.float16(1)
        np.savez(orig / f'{s}.npz', **arrs)
        j = json.loads((orig / f'{s}.json').read_text()); j['npz_sha256'] = ES.sha256_bytes((orig / f'{s}.npz').read_bytes())
        (orig / f'{s}.json').write_text(json.dumps(j))
        with self.assertRaises(X.Extract49Error):
            self.run49(compare_16=orig)
        self.assertFalse((self.out / f'{s}.json').exists()); self.assertFalse((self.out / 'COMPLETE.json').exists())

    def test_source_tamper_and_missing_source(self):
        k = sorted(self.payloads)[1]
        path = X.source_path(self.store, k); keep = path.read_bytes()
        path.write_bytes(keep + b'\0')
        with self.assertRaises(X.Extract49Error) as e:
            self.run49()
        self.assertIn('pkl sha mismatch', str(e.exception))
        self.assertFalse((self.out / 'COMPLETE.json').exists())
        bad_stem = X.stem_of(ES.episode_id(*k))
        prog = json.loads((self.out / 'PROGRESS.json').read_text())
        self.assertEqual(prog['status'], 'incomplete'); self.assertIn(bad_stem, prog['missing'])
        self.assertFalse((self.out / f'{bad_stem}.json').exists()); self.assertFalse((self.out / f'{bad_stem}.npz').exists())
        committed = 4 - len(prog['missing'])
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.run49()
        self.assertFalse((self.out / f'{bad_stem}.json').exists())
        path.write_bytes(keep)
        r = self.run49()
        self.assertEqual(r['computed'] + r['skipped'], 4); self.assertGreaterEqual(r['skipped'], committed)
        self.assertTrue((self.out / 'COMPLETE.json').exists())

    def test_wrong_shape_and_count(self):
        k = sorted(self.payloads)[0]
        bad = payload(9, False, shape=(7, 4, 16, 1536))
        blob = pickle.dumps(bad); X.source_path(self.store, k).write_bytes(blob)
        spec = json.loads(self.spec.read_text())
        for f in spec['folds']:
            for r in f['fit'] + f['eval']:
                if (r['j'], r['n']) == (k[4], k[5]):
                    r['pkl_sha256'] = hashlib.sha256(blob).hexdigest()
        self.spec.write_text(json.dumps(spec))
        with self.assertRaises((X.Extract49Error, ES.ExtractError)):
            self.run49()
        with self.assertRaises(X.Extract49Error):
            self.run49(expect=200)

    def test_last_block49_rejects_mixed_dtype_and_shape(self):
        p = payload(3, False)
        p['hidden_states'][1] = p['hidden_states'][1].astype(np.float64)
        with self.assertRaises(X.Extract49Error):
            X.last_block49(p)

    def test_resume_no_recompute_and_committed_checks(self):
        calls = []

        def counting(p):
            calls.append(1); return X.convert49(p)
        self.run49(convert=counting)
        before = {f.name: f.read_bytes() for f in self.out.iterdir() if f.suffix == '.npz'}
        calls.clear()
        r = self.run49(convert=counting)
        self.assertEqual((r['computed'], r['skipped'], len(calls)), (0, 4, 0))
        self.assertEqual({f.name: f.read_bytes() for f in self.out.iterdir() if f.suffix == '.npz'}, before)
        s = X.stem_of(ES.episode_id(*sorted(self.payloads)[0]))
        with np.load(self.out / f'{s}.npz') as z:
            arrs = {n: z[n] for n in z.files}
        arrs['last_block_d0'] = arrs['last_block_d0'].astype(np.float32)          # dtype tamper with a consistent sha
        np.savez(self.out / f'{s}.npz', **arrs)
        j = json.loads((self.out / f'{s}.json').read_text()); j['npz_sha256'] = ES.sha256_bytes((self.out / f'{s}.npz').read_bytes())
        (self.out / f'{s}.json').write_text(json.dumps(j))
        with self.assertRaises(X.Extract49Error):
            self.run49()

    def test_contract_and_orphan_guards(self):
        self.out.mkdir()
        (self.out / 'contract.json').write_text(json.dumps({'with_last_block': True, 'last_block_tokens': [33, 49]}))
        with self.assertRaises(X.Extract49Error):
            self.run49()                                                          # original-contract dir is never reused
        (self.out / 'contract.json').unlink()
        (self.out / 'deadbeef.npz').write_bytes(b'x')
        with self.assertRaises(X.Extract49Error) as e:
            self.run49()
        self.assertIn('orphan', str(e.exception))

    def test_space_bound_and_refusal_before_writing(self):
        b = X.raw_upper_bound_bytes(200)
        self.assertTrue(4.3e9 < b < 4.5e9, b)                                     # ~4.45e9 B raw for 200 x 144 decisions
        small = lambda p: types.SimpleNamespace(free=10 ** 6)
        plan = self.run49(estimate=True, disk=small)
        self.assertFalse(plan['fits']); self.assertEqual(plan['records'], 4); self.assertFalse(self.out.exists())
        with self.assertRaises(X.Extract49Error):
            self.run49(disk=small)
        self.assertFalse(self.out.exists())


if __name__ == '__main__':
    unittest.main()

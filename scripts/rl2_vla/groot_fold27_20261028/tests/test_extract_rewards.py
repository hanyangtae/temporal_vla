"""CPU tests: extract_scene step contract + rewards (basic/pbrs/mix) + transition rows vs qam_prefix.to_qam_batch.

    python3 -m unittest scripts/rl2_vla/groot_fold27_20261028/tests/test_extract_rewards.py
"""
import importlib.util, sys, unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'fold_data'), str(ROOT / 'trqam')]
import extract_scene as E   # noqa: E402
import rewards as RW        # noqa: E402

G = RW.GAMMA


def payload(steps, first, success, max_steps=720, k=5):
    es = [False] * steps
    if first is not None:
        for i in range(first, steps):
            es[i] = True
    return {'env_step_success': es, 'env_step_phases': ['x'] * (steps + 1), 'env_step_n_action_steps': k,
            'max_episode_steps': max_steps, 'episode_success': success}


class StepContractTest(unittest.TestCase):
    def test_real_success_case_chunk_boundary(self):          # OpenDrawer/left s2 j0 n0: first 200, steps 205
        self.assertEqual(E.episode_steps(payload(205, 200, True), 41), (205, 200, 720, 0))
        self.assertEqual(E.episode_steps(payload(205, 204, True), 41), (205, 204, 720, 0))

    def test_real_one_chunk_stop_lag(self):                   # OvenRack/out-left s2 j0 n3: probe true 134..139, steps 140
        self.assertEqual(E.episode_steps(payload(140, 134, True), 28), (140, 134, 720, 5))

    def test_success_rejections(self):
        with self.assertRaises(E.ExtractError):
            E.episode_steps(payload(145, 134, True), 29)          # two-chunk lag: outside observed behaviour
        p = payload(140, 134, True); p['env_step_success'][137] = False
        with self.assertRaises(E.ExtractError):
            E.episode_steps(p, 28)                                # success not continuous to the end
        with self.assertRaises(E.ExtractError):
            E.episode_steps(payload(138, 134, True), 28)          # not on a chunk boundary
        with self.assertRaises(E.ExtractError):
            E.episode_steps(payload(205, None, True), 41)         # success flag but no success step
        with self.assertRaises(E.ExtractError):
            E.episode_steps(payload(205, 200, True), 40)          # record count mismatch

    def test_adopt_hardlink_revalidates_and_records_provenance(self):
        import json, os, tempfile, hashlib
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); src, dst = t / 'v3', t / 'v4'; src.mkdir(); dst.mkdir()
            base = {'spec_sha256': 's', 'with_last_block': True, 'layers': E.LAYERS, 'extractor_sha256': 'old'}
            (src / 'contract.json').write_text(json.dumps(base))
            es = np.zeros(140, np.int8); es[134:] = 1
            np.savez(src / 'a.npz', env_step_success=es, env_steps=np.int32(140), first_success_step=np.int32(134),
                     max_episode_steps=np.int32(720), success=np.int8(1), last_block_d0=np.zeros((28, 16, 4), np.float16))
            sha = hashlib.sha256((src / 'a.npz').read_bytes()).hexdigest()
            (src / 'a.json').write_text(json.dumps({'episode_id': 'e', 'pkl_sha256': 'p', 'npz': 'a.npz', 'npz_sha256': sha}))
            new = E.adopt_record(src, dst, src / 'a.json', {'pkl_sha256': 'p', 'success': 1}, dict(base, extractor_sha256='new'))
            self.assertEqual(new['success_chunk_lag_steps'], 5)
            self.assertEqual(os.stat(src / 'a.npz').st_ino, os.stat(dst / 'a.npz').st_ino)     # hard link, no copy
            self.assertEqual(new['adopted_from_extractor_sha256'], 'old')
            with self.assertRaises(FileExistsError):
                E.adopt_record(src, dst, src / 'a.json', {'pkl_sha256': 'p', 'success': 1}, base)
            with self.assertRaises(E.ExtractError):
                E.adopt_record(src, t, src / 'a.json', {'pkl_sha256': 'p', 'success': 1}, dict(base, spec_sha256='x'))

    def test_main_shards_adopt_only_and_race_guard(self):
        import json, pickle, tempfile
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); store = t / 'store'
            rows, fold = [], {'fold_id': 'F', 'instruction': 'PPCC/bread', 'scene': 4, 'archive_machine': 'worker2', 'fit': [], 'eval': []}
            rng = np.random.default_rng(1)
            for i in range(3):
                succ = i % 2
                p = dict(payload(10, 6 if succ else None, bool(succ), max_steps=10), capture_layers=E.LAYERS,
                         hidden_states=[rng.normal(size=(7, 4, 49, 1536)).astype(np.float16) for _ in range(2)],
                         actions=[{f'action.{k}': np.zeros((16, d), np.float32) for k, d in E.ACTION_KEYS}] * 2)
                blob = pickle.dumps(p)
                d = store / 'groot/n15/grid/pl/worker2/PPCC/bread/s4/j0' / f'n{i}' / 'base'; d.mkdir(parents=True)
                (d / 'rollout.pkl').write_bytes(blob)
                fold['fit'].append({'plan_id': 'pl', 'j': 0, 'n': i, 'pkl_sha256': E.sha256_bytes(blob), 'success': succ})
            sp = t / 'spec.json'; sp.write_text(json.dumps({'production_ready': True, 'folds': [fold]}))
            run = lambda *extra: (setattr(sys, 'argv', ['x', '--spec', str(sp), '--store', str(store), *extra]), E.main())
            run('--out', str(t / 'v3'), '--shard', '0/2')
            self.assertFalse((t / 'v3/COMPLETE.json').exists())                 # other shard pending
            run('--out', str(t / 'v3'), '--shard', '1/2')
            self.assertTrue((t / 'v3/COMPLETE.json').exists())
            # newer extractor version dir: refuse adopt+shard; serial adopt-only links all three
            with self.assertRaises(SystemExit):
                run('--out', str(t / 'v4'), '--adopt-from', str(t / 'v3'), '--shard', '0/2')
            c3 = json.loads((t / 'v3/contract.json').read_text())        # simulate: v3 written by an older extractor
            (t / 'v3/contract.json').write_text(json.dumps(dict(c3, extractor_sha256='old')))
            run('--out', str(t / 'v4'), '--adopt-from', str(t / 'v3'), '--adopt-only')
            adopted = [json.loads(p.read_text()) for p in (t / 'v4').glob('*.json') if p.name != 'contract.json']
            self.assertEqual({r['adopted_from_extractor_sha256'] for r in adopted}, {'old'})
            self.assertEqual(len([p for p in (t / 'v4').glob('*.json') if p.name != 'contract.json']), 3)
            self.assertFalse((t / 'v4/COMPLETE.json').exists())

    def test_failure_timeout_from_pkl_max(self):
        self.assertEqual(E.episode_steps(payload(720, None, False), 144), (720, -1, 720, -1))
        self.assertEqual(E.episode_steps(payload(500, None, False, max_steps=500), 100)[0], 500)
        with self.assertRaises(E.ExtractError):
            E.episode_steps(payload(715, None, False), 143)
        with self.assertRaises(E.ExtractError):
            E.episode_steps(payload(720, 300, False), 144)        # failure with a success step

    def test_phase_length_and_missing(self):
        p = payload(205, 200, True); p['env_step_phases'] = ['x'] * 205
        with self.assertRaises(E.ExtractError):
            E.episode_steps(p, 41)
        p = payload(205, 200, True); del p['env_step_success']
        with self.assertRaises(E.ExtractError):
            E.episode_steps(p, 41)

    def test_convert_stores_action_tokens_of_layer15_d0(self):
        rng = np.random.default_rng(5)
        n = 41
        recs = [rng.normal(size=(7, 4, 49, 1536)).astype(np.float16) for _ in range(n)]
        chunk = {f'action.{k}': np.zeros((16, d), np.float32) for k, d in E.ACTION_KEYS}
        p = dict(payload(205, 200, True), hidden_states=recs, actions=[chunk] * n,
                 capture_layers=[0, 2, 4, 8, 10, 12, 15])
        out = E.convert(p, True)
        self.assertEqual(out['last_block_d0'].shape, (n, 16, 1536))
        self.assertTrue(np.array_equal(out['last_block_d0'][3], recs[3][6, 0, 33:49]))
        self.assertTrue(np.array_equal(out['tokmean_l12d3'][3], E.tokmean_f16(recs[3][5, 3])))
        self.assertEqual((int(out['env_steps']), int(out['first_success_step'])), (205, 200))

    def test_action_model_order(self):
        chunk = {f'action.{k}': np.full((16, d), i, np.float32) for i, (k, d) in enumerate(E.ACTION_KEYS)}
        a = E.action_model12(chunk)
        self.assertEqual(a.shape, (16, 12))
        self.assertEqual(a[0].tolist(), [0, 0, 0, 1, 1, 1, 2, 3, 3, 3, 3, 4])


class RewardTest(unittest.TestCase):
    def test_basic_success_partial_last_chunk(self):
        m, R, b, final = RW.chunk_terms(203, True, 41)            # last chunk 3 steps, last 3 steps reward 0
        self.assertEqual(m[-1], 1)                                  # cut at first masked step (step 200)
        self.assertEqual((m[:-1] == 5).all(), True)
        self.assertAlmostEqual(R[0], -sum(G ** i for i in range(5)))
        self.assertEqual((b[:-1] == 1).all(), True); self.assertEqual(b[-1], 0)
        self.assertAlmostEqual(R[-1], 0.0)
        m2, R2, b2, _ = RW.chunk_terms(202, True, 41)              # masked steps 199,200,201 span two chunks
        self.assertEqual((m2[-2], b2[-2], m2[-1], b2[-1]), (5, 0, 1, 0))   # cut AT the first masked step (inclusive)
        self.assertAlmostEqual(R2[-2], -sum(G ** i for i in range(4)))

    def test_basic_failure(self):
        m, R, b, final = RW.chunk_terms(720, False, 144)
        self.assertTrue((m == 5).all()); self.assertEqual(b[-1], 0); self.assertTrue((b[:-1] == 1).all())
        with self.assertRaises(RW.RewardContractError):
            RW.chunk_terms(None, False, 144)

    def test_pbrs_and_mix_formulas(self):
        m = np.array([5, 5, 2]); R = np.array([-1., -1., 0.]); b = np.array([1., 1., 0.]); final = np.array([0, 0, 1], bool)
        phi = np.array([.2, -.4, .5]); nxt = np.array([-.4, .5, 0.])
        p = RW.shaped('pbrs', R, m, b, final, True, phi, nxt, alpha=.1)
        np.testing.assert_allclose(p, R + .1 * (G ** m * b * nxt - phi))
        x = RW.shaped('mix', R, m, b, final, False, phi, nxt, alpha=.1, beta=.0396)
        C = np.maximum(0, -phi) * (1 - G ** m) / (1 - G)
        np.testing.assert_allclose(x, R + .1 * (np.array([-.4, .5, -1.]) - phi) - .0396 * C)
        for bad in (dict(arm='mix', alpha=.1), dict(arm='pbrs'), dict(arm='basic', alpha=.1)):
            with self.assertRaises(RW.RewardContractError):
                RW.shaped(bad.pop('arm'), R, m, b, final, True, phi, nxt, **bad)

    def episode(self, success, steps, n, seed=0):
        rng = np.random.default_rng(seed)
        return {'episode_id': f'e{seed}', 'latents': rng.normal(size=(n, 1024)).astype(np.float32),
                'actions': rng.uniform(-1, 1, size=(n, 16, 12)).astype(np.float32), 'success': success,
                'steps': steps, 'max_episode_steps': 720}

    def test_rows_and_recompute(self):
        eps = [self.episode(True, 205, 41, 1), self.episode(False, 720, 144, 2)]
        pot = lambda lat: np.tanh(lat[:, 0])
        for arm, kw in (('basic', {}), ('pbrs', {'alpha': .1})):
            rows = RW.concat([RW.episode_transitions(e, arm, pot if arm != 'basic' else None, **kw) for e in eps])
            self.assertLess(RW.recompute_check(rows, arm, kw.get('alpha')), 1e-5)
            self.assertTrue(np.array_equal(rows['terminals'], ~rows['has_next']))
            self.assertTrue((rows['next_observations'][rows['terminals']] == 0).all())

    @unittest.skipUnless(importlib.util.find_spec('jax') is not None, 'jax absent (run in .venvs/rl2_iid)')
    def test_rows_pass_to_qam_batch(self):
        sys.path.insert(0, str(ROOT.parents[1] / 'rl2_vla/groot_robocasa_adapter_20261021'))
        from qam_prefix import to_qam_batch
        rows = RW.episode_transitions(self.episode(True, 202, 41, 3), 'basic')
        to_qam_batch(rows, execute_steps=5, gamma=G)

    @unittest.skipUnless(importlib.util.find_spec('sklearn') is not None, 'sklearn absent')
    def test_potential_fit(self):
        rng = np.random.default_rng(0)
        eps = [(rng.normal(size=(60, 8)) + (1 if i % 2 else -1), bool(i % 2)) for i in range(10)]
        bun = RW.fit_potential(eps, k=4, dim=3)
        phi = RW.potential_fn(bun)(eps[1][0])
        self.assertEqual(phi.shape, (60,)); self.assertTrue(np.all(np.abs(phi) <= 1))
        self.assertEqual(RW.fit_potential([(e, True) for e, _ in eps])['disabled'][:3], 'one')


if __name__ == '__main__':
    unittest.main()

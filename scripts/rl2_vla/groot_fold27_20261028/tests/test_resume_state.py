"""Resume equality for the v2 TRQAM trainer state (CPU, JAX).

continuous N+M updates  ==  N updates -> save_state -> load_state into a FRESH template -> M updates
(bitwise: every Flax state leaf incl. actor/critic/target/reference params, optimizer moments+count, network.step,
JAX rng, dual lambda, KL EMA; identical minibatch indices; identical NumPy generator state at the end).

    JAX_PLATFORMS=cpu .venvs/rl2_iid/bin/python -m pytest -q scripts/rl2_vla/groot_fold27_20261028/tests/test_resume_state.py
"""
import importlib.util, json, pickle, sys, tempfile, unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
sys.path[:0] = [str(ROOT / 'trqam'), str(ROOT), str(REPO / 'scripts/rl2_vla/groot_robocasa_adapter_20261021'), str(REPO)]
HAS_JAX = importlib.util.find_spec('jax') is not None and importlib.util.find_spec('flax') is not None
COMMON = REPO / 'outputs/rl2_vla/atomic65_common_qam_work_20261029/scratch_run/checkpoints/step_0500000'


def synthetic_rows(seed=0):
    import rewards as RW
    rng = np.random.default_rng(seed)
    eps = []
    for i, (succ, steps, n) in enumerate([(True, 205, 41), (False, 720, 144), (True, 140, 28)]):
        eps.append({'episode_id': f'e{i}', 'latents': rng.normal(size=(n, 1024)).astype(np.float32),
                    'actions': rng.uniform(-1, 1, size=(n, 16, 12)).astype(np.float32), 'success': succ,
                    'steps': steps, 'max_episode_steps': 720})
    return RW.concat([RW.episode_transitions(e, 'basic') for e in eps])


@unittest.skipUnless(HAS_JAX, 'jax/flax absent (run with .venvs/rl2_iid)')
class ResumeEqualityTest(unittest.TestCase):
    N, M, B = 3, 2, 16

    def make_template(self, base_agent):
        from trqam_prefix import RETENTION_DEFAULTS, migrate
        return migrate(base_agent, 42, RETENTION_DEFAULTS)

    def check(self, base_agent):
        import resume_state as RS
        from qam_prefix import to_qam_batch
        rows = synthetic_rows()
        to_batch = lambda b: to_qam_batch(b, execute_steps=5, gamma=0.99)
        idx_a, idx_b = [], []
        # continuous N+M
        rng_a = np.random.default_rng(42)
        cont, _ = RS.update_loop(self.make_template(base_agent), rows, rng_a, 0, self.N + self.M, self.B, to_batch,
                                 lambda s, ag, info, idx: idx_a.append(idx.copy()))
        # N -> save -> restore into a fresh template -> M
        rng_b = np.random.default_rng(42)
        part, _ = RS.update_loop(self.make_template(base_agent), rows, rng_b, 0, self.N, self.B, to_batch,
                                 lambda s, ag, info, idx: idx_b.append(idx.copy()))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / f'state_{self.N}.pkl'
            RS.save_state(p, part, rng_b, self.N, {'fp': 1})
            restored, rng_r, step, fp = RS.load_state(p, self.make_template(base_agent))
        self.assertEqual((step, fp), (self.N, {'fp': 1}))
        self.assertTrue(RS.state_equal(restored, part))                       # restore is exact
        self.assertFalse(RS.state_equal(restored, self.make_template(base_agent)))  # and not the fresh template
        resumed, _ = RS.update_loop(restored, rows, rng_r, self.N, self.N + self.M, self.B, to_batch,
                                    lambda s, ag, info, idx: idx_b.append(idx.copy()))
        self.assertEqual(len(idx_a), len(idx_b))
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(idx_a, idx_b)))     # same minibatch stream
        self.assertTrue(RS.state_equal(cont, resumed))                        # bitwise identical trajectory
        self.assertEqual(rng_a.bit_generator.state, rng_r.bit_generator.state)
        self.assertEqual(float(cont.lam), float(resumed.lam)); self.assertEqual(float(cont.kl_ema), float(resumed.kl_ema))
        self.assertEqual(int(cont.network.step), self.N + self.M)

    def test_reduced_network(self):
        from qam_prefix import PrefixQAMTrainer
        self.check(PrefixQAMTrainer.create(profile='smoke_reduced_network', gamma=0.99, seed=0, data_label='resume_test').agent)

    @unittest.skipUnless((COMMON / 'params_500000.pkl').exists(), 'common 500k checkpoint absent')
    def test_real_common_checkpoint(self):
        from qam_prefix import PrefixQAMTrainer
        self.check(PrefixQAMTrainer.load(COMMON, 500000, feature_signature='ccbfb51ca01724cb').agent)

    def test_v1_network_pickle_fails_as_observed(self):
        from qam_prefix import PrefixQAMTrainer
        agent = self.make_template(PrefixQAMTrainer.create(profile='smoke_reduced_network', gamma=0.99, seed=0, data_label='t').agent)
        with self.assertRaises((AttributeError, pickle.PicklingError, TypeError)):
            pickle.dumps(agent.network)                                       # the v1 failure mode


class ReuseAETest(unittest.TestCase):
    def test_copy_verify_and_reject(self):
        if not HAS_JAX:
            self.skipTest('finetune_fold import needs the rl2 env')
        import finetune_fold_v2 as V2
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); src, out = d / 'v1', d / 'v2'
            (src / 'F/pbrs_a0.1').mkdir(parents=True)
            (src / 'F/ae_potential.json').write_text('{"a":1}'); (src / 'F/ae_potential_encoder.npz').write_bytes(b'enc')
            dry = {'rows': 10, 'reward_mean': -1.5, 'episodes': 160}
            (src / 'F/pbrs_a0.1/DRYRUN.json').write_text(json.dumps(dry))
            prov = V2.reuse_ae(src, out, 'F')
            self.assertEqual(prov['files']['ae_potential.json']['sha256'], prov['files']['ae_potential.json']['source_sha256'])
            self.assertTrue((src / 'F/ae_potential.json').exists())          # copied, never moved
            self.assertIsNone(V2.reuse_ae(src, out, 'F'))                     # already present: no overwrite
            (out / 'F/pbrs_a0.1').mkdir(parents=True); (out / 'F/pbrs_a0.1/DRYRUN.json').write_text(json.dumps(dry))
            self.assertEqual(V2.verify_reuse(src, out, 'F', 'pbrs_a0.1'), 'verified')
            (out / 'F/pbrs_a0.1/DRYRUN.json').write_text(json.dumps(dict(dry, reward_mean=-1.4)))
            with self.assertRaises(V2.FF.FinetuneError):
                V2.verify_reuse(src, out, 'F', 'pbrs_a0.1')
            self.assertTrue((out / 'F/AE_REUSE_REJECTED.json').exists())
            with self.assertRaises(V2.FF.FinetuneError):
                V2.reuse_ae(src, out, 'F')                                    # rejected stays rejected


if __name__ == '__main__':
    unittest.main()

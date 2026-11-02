"""CPU tests for the MIL adapter (needs numpy; torch/pandas tests skip if absent; real shared checkpoints optional).

    python3 -m unittest scripts/rl2_vla/groot_fold27_20261028/tests/test_mil_gate.py
"""
import importlib.util, sys, tempfile, unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'mil')); sys.path.insert(0, str(ROOT / 'fold_data'))
HAS_TORCH = importlib.util.find_spec('torch') is not None
HAS_PANDAS = importlib.util.find_spec('pandas') is not None


def trainer_token_mean(feature):   # verbatim math of MilPoolingTrainer._token_mean (task_classification 33e5e04)
    return np.asarray(feature).astype(np.float32).mean(1).astype(np.float16).astype(np.float32)


class TokenMeanTest(unittest.TestCase):
    def test_stored_f16_tokenmean_is_bit_exact(self):
        import extract_scene as E
        rng = np.random.default_rng(0)
        h = rng.normal(0, 3, size=(37, 49, 1536)).astype(np.float16)
        direct = trainer_token_mean(h)                         # trainer on full [N,49,D] shard
        stored = E.tokmean_f16(h)                              # what extract_scene writes
        via_shard = trainer_token_mean(stored[:, None, :])     # trainer on our [N,1,D] shard
        self.assertTrue(np.array_equal(direct, via_shard))

    def test_hook_layout_matches_pkl_layout(self):
        import mil_gate as G
        rng = np.random.default_rng(1)
        rec = rng.normal(size=(7, 4, 49, 1536)).astype(np.float16)    # pkl record [layer, denoise, token, dim]
        hook = rec[5]                                                   # serve hook on transformer_blocks[12] -> [4,49,1536]
        self.assertEqual([0, 2, 4, 8, 10, 12, 15].index(12), 5)
        self.assertTrue(np.array_equal(G.token_mean(hook[3][None])[0], trainer_token_mean(rec[5, 3][None])[0]))


@unittest.skipUnless(HAS_TORCH, 'torch not installed in this python')
class GateTest(unittest.TestCase):
    def make_gate(self, d):
        import torch, mil_gate as G
        sys.path.insert(0, str(G.TC))
        from failure_detection_multi_instance_learning.model.mil_pooling_model import MilPoolingModel
        torch.manual_seed(0)
        net = MilPoolingModel(1536, 'attn')
        rng = np.random.default_rng(2)
        tf = {'kind': 'raw', 'mu': rng.normal(size=1536).astype(np.float32), 'sd': np.abs(rng.normal(size=1536)).astype(np.float32) + .5,
              'z_mu': rng.normal(size=1536).astype(np.float32) * .1, 'z_sd': np.abs(rng.normal(size=1536)).astype(np.float32) + .5}
        p = Path(d) / 'model.pt'
        net.save_model(p, config={'fire_threshold': 0.5, 'train': {'optimizer': 'sgd', 'lr': .3}}, transform=tf, input='raw',
                       cell={'cells': ['x'], 'target_jitter': [1], 'train_jitter': [0, 2, 3, 4]}, seed=0, best={'epoch': 0})
        return G.MilGate.from_run(Path(d)), net, tf

    def test_online_equals_example_w_fail(self):
        import torch, mil_gate as G
        with tempfile.TemporaryDirectory() as d:
            g, net, tf = self.make_gate(d)
            hid = np.random.default_rng(3).normal(size=(9, 4, 49, 1536)).astype(np.float16)
            x = trainer_token_mean(hid[:, 3])
            z = ((x - tf['mu']) / tf['sd']); z = ((z - tf['z_mu']) / tf['z_sd']).astype(np.float32)   # examples/score_test_episodes.w_fail
            with torch.no_grad():
                ref = torch.sigmoid(net.eval().frame_logits(torch.from_numpy(z))).numpy()
            out = []
            for t in range(9):
                r = g.step(g.feature_from_hidden(hid[t]))
                self.assertEqual(r['t'], t); self.assertEqual(r['fired'], r['score'] > 0.5)
                out.append(r['score'])
            self.assertTrue(np.allclose(out, ref, atol=0, rtol=0) or np.max(np.abs(np.float32(out) - ref)) < 1e-6)
            snap = g.snapshot(); g.step(x[0]); g.restore(snap); self.assertEqual(g.step(x[0])['t'], 9)

    def test_rejects_threshold_override_and_bad_shape(self):
        import mil_gate as G
        with tempfile.TemporaryDirectory() as d:
            g, _, _ = self.make_gate(d)
            with self.assertRaises(G.MilContractError):
                G.MilGate(g.net, g.ck, Path(d) / 'model.pt', threshold=0.3)
            with self.assertRaises(G.MilContractError):
                g.feature_from_hidden(np.zeros((7, 4, 49, 1536), np.float16))
            with self.assertRaises(G.MilContractError):
                G.MilGate.from_run(Path(d), require_target_jitter=2)

    @unittest.skipUnless((ROOT.parents[2] / 'task_classification/output/share/grid_n10_n19/runs/cell-n10-bread_s4/attn/raw/s0/model.pt').exists(), 'shared ckpt absent')
    def test_real_bread_s4_s0(self):
        import mil_gate as G
        g = G.MilGate.from_run(G.TC / 'output/share/grid_n10_n19/runs/cell-n10-bread_s4/attn/raw/s0', require_target_jitter=1)
        self.assertEqual(g.threshold, 0.5)
        self.assertEqual(g.ck['config']['train']['lr'], 0.3)
        self.assertEqual(g.ck['config']['train']['optimizer'], 'sgd')


class PreflightTest(unittest.TestCase):
    def test_only_dish_right_j0_blocked(self):
        import folds as F
        import importlib
        sys.modules.setdefault('numpy', np)
        fm = importlib.import_module('fold_mil')
        spec, _ = F.build(F.R)
        rows = fm.preflight(spec)
        self.assertEqual([r['fold_id'] for r in rows if not r['mil_trainable_unmodified']], ['DishwasherRack-out-right_s1_j0'])
        self.assertEqual([r['fold_id'] for r in rows if r['split_exception']], ['DishwasherRack-out-right_s1_j0'])
        self.assertEqual([r for r in rows if r['blocker']], [])
        self.assertTrue(fm.splitter_feasible(6, 154)); self.assertFalse(fm.splitter_feasible(3, 157))

    @unittest.skipUnless(HAS_PANDAS, 'pandas absent')
    def test_exception_splitter_disjoint_and_failure_draw_unchanged(self):
        import importlib, pandas as pd
        fm = importlib.import_module('fold_mil')

        class Shard:   # minimal DataShard stand-in: 160 episodes, 3 successes
            pass
        succ_eps = {7, 50, 120}
        ep = np.repeat(np.arange(160), 4)
        sh = Shard(); sh.data_meta = pd.DataFrame({'episode_idx': ep}); sh.data_success = np.isin(ep, list(succ_eps)).astype(np.int8)
        tr, va = fm.ExceptionSplitter(0.2, 0, 1).split_rows(sh)
        vs, ts = set(ep[va]), set(ep[tr])
        self.assertFalse(vs & ts)
        self.assertEqual(len(vs & succ_eps), 1); self.assertEqual(len(ts & succ_eps), 2)
        # failure side = exactly DataSplitter's draw (rng consumed failure-side first, same k)
        rng = np.random.default_rng(0)
        fails = np.array(sorted(set(range(160)) - succ_eps))
        want = set(rng.choice(fails, max(2, int(round(len(fails) * .2))), replace=False).tolist())
        self.assertEqual(vs - succ_eps, want)


if __name__ == '__main__':
    unittest.main()

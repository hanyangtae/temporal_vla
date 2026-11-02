"""CPU tests for trqam/train_queue_v2.py (v2 wiring only; queue logic is the frozen v1, tested in test_train_queue.py)."""
import importlib.util, json, shutil, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
sys.path[:0] = [str(ROOT / 'trqam'), str(ROOT), str(REPO / 'scripts/rl2_vla/groot_robocasa_adapter_20261021'), str(REPO)]
import train_queue_v2 as Q2   # noqa: E402
import train_queue as TQ      # noqa: E402

SPEC = REPO / 'outputs/analysis/groot_fold27_20261028/folds/fold_spec.json'


@unittest.skipUnless(SPEC.exists(), 'production fold_spec.json absent')
class QueueV2Test(unittest.TestCase):
    def setUp(self):
        self.t = Path(tempfile.mkdtemp())
        self.init = self.t / 'init'; self.init.mkdir(); (self.init / 'params_500000.pkl').write_bytes(b'p')
        e, l = self.t / 'ep', self.t / 'lat'; e.mkdir(); l.mkdir(); (e / 'contract.json').write_text('{}')
        (l / 'COMPLETE.json').write_text(json.dumps({'episodes': 200}))
        (self.t / 's.json').write_text(json.dumps({'OpenDrawer/left|s2': {'episodes': str(e), 'latents': str(l)}}))

    def tearDown(self):
        shutil.rmtree(self.t)

    def base_args(self, *extra):
        return ['--spec', str(SPEC), '--sources', str(self.t / 's.json'), '--init', str(self.init), '--dry-only', *extra]

    def test_cmd_pins_and_roots(self):
        a, reuse = Q2.parse(self.base_args('--out', str(self.t / 'v2'), '--reuse-ae-from', '/old/root'))
        self.assertEqual(reuse, Path('/old/root'))
        q = Q2.TrainQueueV2(a, reuse_ae_from=reuse)
        f = next(f for f in q.spec['folds'] if f['fold_id'] == 'OpenDrawer-left_s2_j2')
        c = q.cmd(f, 'pbrs', 'smoke')
        self.assertTrue(c[1].endswith('finetune_fold_v2.py'))
        self.assertEqual(c[-2:], ['--reuse-ae-from', '/old/root'])
        self.assertIn('--smoke-steps', c)
        self.assertIn('finetune_fold_v2.py', q.pins['fingerprint_code']); self.assertIn('resume_state.py', q.pins['fingerprint_code'])
        self.assertIn('train_queue_v2.py', q.pins['extra_code'])
        a2, _ = Q2.parse(self.base_args())
        self.assertEqual(a2.out, Q2.V2_OUT)
        a3, _ = Q2.parse(self.base_args('--out', str(REPO / 'outputs/rl2_vla/fold27_trqam_20261102')))
        with self.assertRaises(TQ.QueueError):
            Q2.TrainQueueV2(a3)

    @unittest.skipUnless(importlib.util.find_spec('jax') is not None, 'rl2 env needed to import finetune_fold_v2')
    def test_pins_match_fingerprint_v2(self):
        import argparse, finetune_fold_v2 as V2
        a, _ = Q2.parse(self.base_args('--out', str(self.t / 'v2')))
        q = Q2.TrainQueueV2(a)
        ns = argparse.Namespace(fold='F', arm='basic', alpha=None, beta=None, potential='ae16_k48', init=None, init_step=500000, smoke_steps=None)
        import numpy as np
        fp = V2.fingerprint_v2(ns, {'fold_id': 'F'}, {'rewards': np.zeros(1, np.float32), 'observations': np.zeros((1, 2), np.float32)})
        self.assertEqual(fp['code'], q.pins['fingerprint_code'])      # done() comparison will match v2 runs


if __name__ == '__main__':
    unittest.main()

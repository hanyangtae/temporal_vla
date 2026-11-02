"""rollout_latents_v2 gates on CPU (no GPU, no checkpoint): explicit token contracts never admit each other's data,
the smoke receipt cannot be satisfied by editing it, provenance drift refuses, conversion runs at batch 1."""
import json, shutil, sys, tempfile, unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'trqam'))
import rollout_latents_v2 as R2   # noqa: E402

ROW = {'episode': 0, 'block_shape': [1, 49, 1536], 'block_dtype': 'torch.bfloat16', 'decoder_input_shape': [1, 49, 1536],
       'decoder_input_dtype': 'torch.bfloat16', 'device': 'cuda:0', 'exact_49': True, 'exact_16': False,
       'max_abs_49': 0.0, 'max_abs_16': 0.0078}


class GateTest(unittest.TestCase):
    def test_record_contracts_are_disjoint(self):
        self.assertTrue(R2.check_record(np.zeros((3, 49, 1536), np.float16), 49))
        self.assertTrue(R2.check_record(np.zeros((3, 16, 1536), np.float16), 16))
        for arr, c in ((np.zeros((3, 16, 1536), np.float16), 49), (np.zeros((3, 49, 1536), np.float16), 16),
                       (np.zeros((3, 49, 1536), np.float32), 49), (np.zeros((49, 1536), np.float16), 49)):
            with self.assertRaises(SystemExit):
                R2.check_record(arr, c)

    def test_smoke_verdict_49_does_not_need_16_but_16_needs_both(self):
        self.assertTrue(R2.smoke_verdict([ROW, dict(ROW, episode=1)], 49)['passed'])
        with self.assertRaises(SystemExit):
            R2.smoke_verdict([ROW], 16)                                     # exact_16 False -> 16-token data not admitted
        self.assertTrue(R2.smoke_verdict([dict(ROW, exact_16=True)], 16)['passed'])
        with self.assertRaises(SystemExit):
            R2.smoke_verdict([dict(ROW, exact_49=False)], 49)
        with self.assertRaises(SystemExit):
            R2.smoke_verdict([], 49)

    def test_smoke_must_be_batch1_full_block(self):
        for bad in (dict(ROW, block_shape=[2, 49, 1536]), dict(ROW, block_shape=[1, 16, 1536]),
                    dict(ROW, decoder_input_shape=[3, 49, 1536]), {k: v for k, v in ROW.items() if k != 'block_shape'}):
            with self.assertRaises(SystemExit):
                R2.smoke_verdict([bad], 49)


class OutAndReceiptTest(unittest.TestCase):
    def setUp(self):
        self.t = Path(tempfile.mkdtemp())
        self.prov = {'version': R2.VERSION, 'token_contract': 49, 'kernel': {'allow_tf32_matmul': False}, 'script_sha256': 'a'}

    def tearDown(self):
        shutil.rmtree(self.t)

    def test_guard_out_refuses_v1_artefacts(self):
        R2.guard_out(self.t)                                                # empty: fine
        (self.t / 'contract.json').write_text(json.dumps({'version': R2.VERSION}))
        R2.guard_out(self.t)
        (self.t / 'contract.json').write_text(json.dumps({'tokens': [33, 49]}))   # v1 contract
        with self.assertRaises(SystemExit):
            R2.guard_out(self.t)
        (self.t / 'contract.json').unlink(); (self.t / 'PARITY_SMOKE.json').write_text('{}')
        with self.assertRaises(SystemExit):
            R2.guard_out(self.t)

    def test_require_smoke(self):
        with self.assertRaises(SystemExit):
            R2.require_smoke(self.t, self.prov)                             # missing
        p = self.t / 'PARITY_SMOKE_V2.json'
        p.write_text(json.dumps({'provenance': self.prov, 'rows': [ROW], 'verdict': {'passed': True}}))
        self.assertEqual(len(R2.require_smoke(self.t, self.prov)), 64)
        drift = dict(self.prov, kernel={'allow_tf32_matmul': True})
        with self.assertRaises(SystemExit) as e:
            R2.require_smoke(self.t, drift)
        self.assertIn('kernel', str(e.exception))
        p.write_text(json.dumps({'provenance': self.prov, 'rows': [dict(ROW, exact_49=False)], 'verdict': {'passed': True}}))
        with self.assertRaises(SystemExit):
            R2.require_smoke(self.t, self.prov)                             # edited 'passed' does not admit failing rows
        p.write_text(json.dumps({'provenance': dict(self.prov, token_contract=16), 'rows': [ROW]}))
        with self.assertRaises(SystemExit):
            R2.require_smoke(self.t, self.prov)                             # a 16-contract smoke never serves 49 (and vice versa)


class DeviceCpuTest(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        torch.manual_seed(0)
        self.dit = types_dit(torch)
        self.shapes = []

        def tail(x):
            self.shapes.append(tuple(x.shape))
            return self.dit.proj_out_2(self.dit.norm_out(x))
        self.tail = tail

    def test_reconstruct_batch1_and_shape(self):
        lb = np.random.default_rng(0).standard_normal((5, 49, 1536)).astype(np.float16)
        lat = R2.reconstruct(lb, self.tail, self.torch, 'cpu', self.torch.float32)
        self.assertEqual(lat.shape, (5, 1024)); self.assertEqual(lat.dtype, np.float32)
        self.assertEqual(set(self.shapes), {(1, 49, 1536)})
        x0 = self.torch.from_numpy(lb[0:1]).float()
        with self.torch.no_grad():
            ref = self.tail(x0)[:, -16:].float().mean(1).numpy()[0]
        np.testing.assert_array_equal(lat[0], ref)

    def test_diagnose_reports_stages_per_batch(self):
        d = R2.diagnose(self.dit, self.tail, self.torch, 'cpu', self.torch.float32)
        self.assertEqual(sorted(d), ['synthetic_b1', 'synthetic_b3'])
        for v in d.values():
            self.assertEqual(set(v), {'layernorm_equal', 'tail_equal', 'tail_max_abs', 'gemm_rows'})
            self.assertTrue(v['layernorm_equal'])                           # LayerNorm is per token
        self.assertEqual(d['synthetic_b3']['gemm_rows'], [147, 48])

    def test_kernel_settings_fields(self):
        k = R2.kernel_settings(self.torch)
        for f in ('allow_tf32_matmul', 'allow_tf32_cudnn', 'float32_matmul_precision', 'cudnn_deterministic',
                  'deterministic_algorithms', 'env', 'torch', 'device_name'):
            self.assertIn(f, k)


class LauncherTest(unittest.TestCase):
    def setUp(self):
        import run_latents_v2 as L2
        self.L2 = L2
        self.t = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.t)

    def episodes(self, tokens, dtype=np.float16, n=2):
        d = self.t / f'ep{tokens}'; d.mkdir()
        (d / 'contract.json').write_text('{}')
        for i in range(n):
            np.savez(d / f'r{i}.npz', last_block_d0=np.zeros((4, tokens, 1536), dtype))
            (d / f'r{i}.json').write_text(json.dumps({'episode_id': f'x/worker2/{i}', 'npz': f'r{i}.npz'}))
        return d

    def test_peek_records_per_contract(self):
        p = self.L2.peek_records(self.episodes(49), 49)
        self.assertEqual([r['shape'] for r in p], [[4, 49, 1536]] * 2)
        with self.assertRaises(SystemExit):
            self.L2.peek_records(self.episodes(16), 49)
        with self.assertRaises(SystemExit):
            self.L2.peek_records(self.t / 'ep49', 16)

    def test_smoke_passed_requires_same_contract(self):
        out = self.t / 'out'; out.mkdir()
        self.assertFalse(self.L2.smoke_passed(out, 49))
        (out / 'PARITY_SMOKE_V2.json').write_text(json.dumps({'provenance': {'token_contract': 49}, 'verdict': {'passed': True}}))
        self.assertTrue(self.L2.smoke_passed(out, 49)); self.assertFalse(self.L2.smoke_passed(out, 16))
        (out / 'PARITY_SMOKE_V2.json').write_text(json.dumps({'provenance': {'token_contract': 49}, 'verdict': {'passed': False}}))
        self.assertFalse(self.L2.smoke_passed(out, 49))

    def test_srv_command_targets_v2_script(self):
        V1R = self.L2.V1R
        old = V1R.REL
        try:
            V1R.REL = self.L2.REL2
            cmd, env = V1R.build_cmd('srv50', 2, self.t, self.t / 'o', lambda ep: ['--token-contract', '49', '--parity-smoke'],
                                     data_root=Path('/data'))
        finally:
            V1R.REL = old
        self.assertTrue(cmd[1].endswith('trqam/rollout_latents_v2.py'))
        self.assertEqual(cmd[-3:], ['--token-contract', '49', '--parity-smoke'])
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '2')
        self.assertEqual(V1R.REL, 'scripts/rl2_vla/groot_fold27_20261028/trqam/rollout_latents.py')


def types_dit(torch):
    import types
    return types.SimpleNamespace(norm_out=torch.nn.LayerNorm(1536, elementwise_affine=False), proj_out_2=torch.nn.Linear(1536, 1024))


if __name__ == '__main__':
    unittest.main()

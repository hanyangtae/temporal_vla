"""CPU tests for the host-matched latent launcher (trqam/run_latents.py) and rollout_latents pins.

    .venvs/rl2_iid/bin/python -m pytest -q scripts/rl2_vla/groot_fold27_20261028/tests/test_trqam_launch.py
"""
import json, os, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'trqam')]
import run_latents as RL       # noqa: E402
import rollout_latents as RO   # noqa: E402


def episodes_dir(d, machine='kanu', n=3):
    d.mkdir(parents=True)
    (d / 'contract.json').write_text('{}')
    for i in range(n):
        (d / f'{i}.json').write_text(json.dumps({'episode_id': f'p/{machine}/OpenDrawer/left/s2/j0/n{i}'}))
    return d


class LauncherTest(unittest.TestCase):
    def setUp(self):
        self.t = Path(tempfile.mkdtemp())
        self.data = self.t / 'datasets'
        (self.data / RL.CKPT_REL).mkdir(parents=True); (self.data / RL.DEMO_REL).mkdir(parents=True)

    def test_preflight_host_and_inputs(self):
        ep = episodes_dir(self.t / 'ep')
        self.assertEqual(RL.preflight(ep, self.t / 'out', self.data, 'kanu'), 3)
        with self.assertRaises(RL.LatentLaunchError):
            RL.preflight(ep, self.t / 'out', self.data, 'srv50')          # kanu records on srv50
        ep2 = episodes_dir(self.t / 'ep2', machine='worker2')
        self.assertEqual(RL.preflight(ep2, self.t / 'out', self.data, 'srv50'), 3)
        with self.assertRaises(RL.LatentLaunchError):
            RL.preflight(ep, self.t / 'out', self.t / 'empty', 'kanu')    # no checkpoint / demo on this host

    def test_kanu_docker_command(self):
        ep, out = self.t / 'ep', self.t / 'out'
        cmd, env = RL.build_cmd('kanu', 6, ep, out, lambda e: ['--episodes', str(e), '--expect-records', '200'],
                                image='sha256:img', cache_root=Path('/c'))
        self.assertIsNone(env)
        self.assertIn('device=6', cmd); self.assertIn('sha256:img', cmd)
        self.assertIn(f'{ep}:/episodes:ro', cmd); self.assertIn(f'{out}:/out:rw', cmd); self.assertIn('/c:/cache:ro', cmd)
        self.assertEqual(cmd[-4:], ['--episodes', '/episodes', '--expect-records', '200'])
        self.assertIn('/cache/datasets', cmd)
        for k in ('HARNESS_RECEIPT', 'HARNESS_MACHINE'):
            self.assertIn(k, cmd)

    def test_srv_host_command(self):
        cmd, env = RL.build_cmd('srv50', 2, self.t / 'ep', self.t / 'out', lambda e: ['--parity-smoke'], data_root=Path('/d'))
        self.assertEqual(cmd[0], str(RL.HOST_PY)); self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '2')
        self.assertIn('/lerobot/src', env['PYTHONPATH']); self.assertEqual(cmd[-3:], ['--data-root', '/d', '--parity-smoke'])

    def test_conversion_requires_expected_count(self):
        os.environ.update(HARNESS_RECEIPT='r', HARNESS_MACHINE='kanu', GPUS='6')
        try:
            sys.argv = ['x', '--episodes', str(self.t / 'ep'), '--out', str(self.t / 'out')]
            with self.assertRaises(RL.LatentLaunchError):
                RL.main()                                                  # conversion without --expect-records
        finally:
            for k in ('HARNESS_RECEIPT', 'HARNESS_MACHINE', 'GPUS'):
                os.environ.pop(k, None)

    def test_smoke_passed_and_audit_pin(self):
        out = self.t / 'out'; out.mkdir()
        self.assertFalse(RL.smoke_passed(out))
        (out / 'PARITY_SMOKE.json').write_text(json.dumps({'passed': True}))
        self.assertTrue(RL.smoke_passed(out))
        self.assertEqual(RO.audit_gate(), RO.AUDIT_SHA256)                 # real repo audit file
        bad = self.t / 'audit.json'; bad.write_text('{}')
        with self.assertRaises(SystemExit):
            RO.audit_gate(bad)


if __name__ == '__main__':
    unittest.main()

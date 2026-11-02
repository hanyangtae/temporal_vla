"""CPU tests for trqam/host_fold_coordinator.py (real frozen TrainQueue verification; launches stubbed).

    .venvs/rl2_iid/bin/python -m pytest -q scripts/rl2_vla/groot_fold27_20261028/tests/test_host_coordinator.py
"""
import json, shutil, sys, tempfile, types, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'trqam'), str(ROOT)]
import host_fold_coordinator as HC   # noqa: E402
import train_queue as TQ             # noqa: E402

SPEC = ROOT.parents[2] / 'outputs/analysis/groot_fold27_20261028/folds/fold_spec.json'
FIRST = 'OpenDrawer-left_s2_j2'


@unittest.skipUnless(SPEC.exists(), 'production fold_spec.json absent')
class HostCoordTest(unittest.TestCase):
    def setUp(self):
        self.t = Path(tempfile.mkdtemp())
        self.init = self.t / 'init'; self.init.mkdir(); (self.init / 'params_500000.pkl').write_bytes(b'p')
        src = {}
        for scene in ('OpenDrawer/left|s2', 'OpenDrawer/right|s1'):
            e, l = self.t / 'ep' / scene.replace('/', '_'), self.t / 'lat' / scene.replace('/', '_')
            e.mkdir(parents=True); l.mkdir(parents=True); (e / 'contract.json').write_text('{}')
            src[scene] = {'episodes': str(e), 'latents': str(l)}
        self.src = src
        (self.t / 'sources.json').write_text(json.dumps(src))
        self.out = self.t / 'out'
        self.launched, self.rc, self.write_done = [], 0, True

    def tearDown(self):
        shutil.rmtree(self.t)

    def latents_complete(self, scene, n=200):
        (Path(self.src[scene]['latents']) / 'COMPLETE.json').write_text(json.dumps({'episodes': n}))

    def queue(self):
        return TQ.TrainQueue(TQ.parse(['--spec', str(SPEC), '--sources', str(self.t / 'sources.json'), '--out', str(self.out),
                                       '--init', str(self.init), '--dry-only']))

    def mark_done(self, fid):
        q = self.queue()
        for arm in TQ.ARMS:
            m = q.marker(fid, arm, 'prod'); ck = m.parent / 'checkpoint'; ck.mkdir(parents=True, exist_ok=True)
            (ck / 'params_50000.pkl').write_bytes(f'{fid}{arm}'.encode())
            m.write_text(json.dumps({'harness_receipt': 'r', 'params_sha256': TQ.sha(ck / 'params_50000.pkl'),
                                     'fingerprint': {'steps': 50000, 'code': q.pins['fingerprint_code'],
                                                     'init': {'step': 500000, 'params_sha256': q.pins['init_params_sha256']}}}))

    def fake_run(self, cmd, **kw):
        fid = cmd[cmd.index('--folds') + 1]
        self.launched.append(fid)
        self.assertIn('wait_launch.py', cmd[1]); self.assertIn('GPUS=5', cmd); self.assertIn('HARNESS_MACHINE=kanu', cmd)
        if self.rc == 0 and self.write_done:
            self.mark_done(fid)
        return types.SimpleNamespace(returncode=self.rc)

    def coord(self, *extra, sleep=None):
        clock = [0.0]

        def default_sleep(s):
            clock[0] += s
        a = HC.parse(['--machine', 'kanu', '--spec', str(SPEC), '--sources', str(self.t / 'sources.json'), '--out', str(self.out),
                      '--init', str(self.init), '--after-fold', FIRST, '--poll-s', '1', '--max-wait-h', '1', *extra])
        return HC.HostCoordinator(a, run=self.fake_run, sleep=sleep or default_sleep, now=lambda: clock[0])

    def test_waits_first_then_serial_bounded_per_fold(self):
        self.latents_complete('OpenDrawer/left|s2'); self.latents_complete('OpenDrawer/right|s1')
        polls = []

        def sleep(s):
            polls.append(s)
            if len(polls) == 3:
                self.queue(); self.mark_done(FIRST)                 # first fold finishes elsewhere
        res = self.coord(sleep=sleep).run()
        self.assertEqual(len(polls), 3)
        self.assertNotIn(FIRST, self.launched)                      # never re-launched
        self.assertEqual(len(self.launched), 5); self.assertEqual(len(set(self.launched)), 5)
        self.assertEqual(res['done_folds'], self.launched)
        n = len(self.launched)
        self.coord().run()
        self.assertEqual(len(self.launched), n)                     # all verified DONE -> nothing relaunched

    def test_inputs_gate_and_fatal_inputs(self):
        self.queue(); self.mark_done(FIRST); self.latents_complete('OpenDrawer/left|s2')
        c = self.coord('--folds', 'OpenDrawer-right_s1_j0', '--max-wait-h', '0.0005')
        with self.assertRaises(HC.CoordError):
            c.run()                                                 # right latents never COMPLETE -> deadline, not infinite
        self.assertEqual(self.launched, [])
        self.assertTrue((self.out / 'queue/host_kanu_COORD_FAILED.json').exists())
        self.latents_complete('OpenDrawer/right|s1', n=199)
        with self.assertRaises(HC.CoordError):
            self.coord('--folds', 'OpenDrawer-right_s1_j0', '--retry-failed').run()   # wrong count: immediate
        self.assertEqual(len(list((self.out / 'queue').glob('host_kanu_COORD_FAILED.prev.*.json'))), 1)

    def test_failure_markers_rc75_and_unverified(self):
        self.queue(); self.mark_done(FIRST)
        self.latents_complete('OpenDrawer/left|s2'); self.latents_complete('OpenDrawer/right|s1')
        (self.out / 'queue/FAILED.json').write_text('{"job":"x"}')
        with self.assertRaises(HC.CoordError):
            self.coord().run()                                      # train_queue failure detected, nothing launched
        self.assertEqual(self.launched, [])
        (self.out / 'queue/FAILED.json').unlink()
        self.rc = 75
        with self.assertRaises(HC.CoordError):
            self.coord('--retry-failed').run()
        self.assertEqual(len(self.launched), 1)
        self.rc, self.write_done = 0, False
        with self.assertRaises(HC.CoordError):
            self.coord('--retry-failed').run()                      # rc 0 but no verified DONE
        marker = self.t / 'LATENT_FAILED'; marker.write_text('x')
        with self.assertRaises(HC.CoordError):
            self.coord('--retry-failed', '--fail-marker', str(marker)).run()
        with self.assertRaises(HC.CoordError):
            self.coord().run()                                      # previous failure kept until --retry-failed

    def test_host_scope(self):
        with self.assertRaises(HC.CoordError):
            self.coord('--folds', 'PPCC-bread_s4_j1').run()        # srv50 fold / not in kanu sources


if __name__ == '__main__':
    unittest.main()

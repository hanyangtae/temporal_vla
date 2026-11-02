"""CPU tests for trqam/host_fold_coordinator_v2.py (real v2 queue verification; launches stubbed)."""
import fcntl, json, shutil, sys, tempfile, types, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'trqam'), str(ROOT)]
import host_fold_coordinator_v2 as H2   # noqa: E402

TQ = H2.TQ
SPEC = ROOT.parents[2] / 'outputs/analysis/groot_fold27_20261028/folds/fold_spec.json'
SMOKE = 'OpenDrawer-left_s2_j2'


@unittest.skipUnless(SPEC.exists(), 'production fold_spec.json absent')
class HostCoordV2Test(unittest.TestCase):
    def setUp(self):
        self.t = Path(tempfile.mkdtemp())
        self.init = self.t / 'init'; self.init.mkdir(); (self.init / 'params_500000.pkl').write_bytes(b'p')
        src = {}
        for scene in ('OpenDrawer/left|s2', 'OpenDrawer/right|s1'):
            e, l = self.t / 'ep' / scene.replace('/', '_'), self.t / 'lat' / scene.replace('/', '_')
            e.mkdir(parents=True); l.mkdir(parents=True); (e / 'contract.json').write_text('{}')
            (l / 'COMPLETE.json').write_text(json.dumps({'episodes': 200}))
            src[scene] = {'episodes': str(e), 'latents': str(l)}
        (self.t / 'sources.json').write_text(json.dumps(src))
        self.out = self.t / 'v2'
        self.launched = []

    def tearDown(self):
        shutil.rmtree(self.t)

    def args(self, *extra):
        return H2.parse(['--machine', 'kanu', '--spec', str(SPEC), '--sources', str(self.t / 'sources.json'), '--out', str(self.out),
                         '--init', str(self.init), '--gpus', '5', '6', '7', '--smoke-fold', SMOKE, '--poll-s', '1', '--max-wait-h', '1',
                         '--reuse-ae-from', '/v1root', *extra])

    def coord(self, *extra, sleep=None, run=None):
        clock = [0.0]
        return H2.HostCoordinatorV2(self.args(*extra), run=run or self.fake_run,
                                    sleep=sleep or (lambda s: clock.__setitem__(0, clock[0] + s)), now=lambda: clock[0])

    def write(self, fid, arm, stage):
        q = H2.HostCoordinatorV2(self.args()).queue()
        m = q.marker(fid, arm, stage); steps = 200 if stage == 'smoke' else 50000
        ck = m.parent / 'checkpoint'; ck.mkdir(parents=True, exist_ok=True); (ck / f'params_{steps}.pkl').write_bytes(f'{fid}{arm}{stage}'.encode())
        m.write_text(json.dumps({'harness_receipt': 'r', 'params_sha256': TQ.sha(ck / f'params_{steps}.pkl'),
                                 'fingerprint': {'steps': steps, 'code': q.pins['fingerprint_code'],
                                                 'init': {'step': 500000, 'params_sha256': q.pins['init_params_sha256']}}}))

    def resume_check(self, ok=True):
        d = self.out / SMOKE / 'resume_check_v2'; d.mkdir(parents=True, exist_ok=True)
        r = {g: True for g in H2.RESUME_GATES}; r['backend'] = 'gpu'
        if not ok:
            r['trajectory_bitwise_equal'] = False; r['trajectory_max_abs_diff'] = 1e-7
        (d / 'RESUME_CHECK.json').write_text(json.dumps(r))

    def fake_run(self, cmd, **kw):
        fid = cmd[cmd.index('--folds') + 1]
        self.launched.append(fid)
        self.assertIn('GPUS=5,6,7', cmd); self.assertTrue(any(c.endswith('train_queue_v2.sh') for c in cmd))
        self.assertEqual(cmd[cmd.index('--per-gpu') + 1], '1'); self.assertIn('/v1root', cmd)
        self.assertIn('HARNESS_SESSION=fold27-trqam-v2', cmd)
        for arm in TQ.ARMS:
            self.write(fid, arm, 'prod')
        return types.SimpleNamespace(returncode=0)

    def test_strict_smoke_gate_then_all_six_folds(self):
        polls = []

        def sleep(s):
            polls.append(s)
            if len(polls) == 1:
                self.write(SMOKE, 'basic', 'smoke'); self.write(SMOKE, 'pbrs', 'smoke')
            if len(polls) == 2:
                self.resume_check()                                 # still waits: mix smoke missing
            if len(polls) == 3:
                self.write(SMOKE, 'mix', 'smoke')
        res = self.coord(sleep=sleep).run()
        self.assertEqual(len(polls), 3)
        self.assertEqual(len(self.launched), 6); self.assertIn(SMOKE, self.launched)    # smoke fold's production included
        self.assertEqual(res['done_folds'], self.launched)
        n = len(self.launched); self.coord().run(); self.assertEqual(len(self.launched), n)

    def test_failed_resume_check_or_smoke_failure_stops(self):
        for arm in TQ.ARMS:
            self.write(SMOKE, arm, 'smoke')
        self.resume_check(ok=False)
        with self.assertRaises(H2.HC.CoordError):
            self.coord().run()
        self.assertEqual(self.launched, [])
        self.assertTrue((self.out / 'queue/host_kanu_COORD_V2_FAILED.json').exists())
        self.resume_check(ok=True)
        (self.out / SMOKE / 'mix_a0.1_b0.0396_smoke' / 'FAILED.json').write_text('{}')
        with self.assertRaises(H2.HC.CoordError):
            self.coord('--retry-failed').run()
        self.assertEqual(self.launched, [])

    def test_single_coordinator_lock(self):
        (self.out / 'queue').mkdir(parents=True, exist_ok=True)
        with open(self.out / 'queue/host_coord_v2.lock', 'a+') as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(H2.HC.CoordError):
                self.coord().run()

    def test_requires_smoke_fold(self):
        with self.assertRaises(SystemExit):
            H2.parse(['--machine', 'kanu', '--sources', str(self.t / 'sources.json'), '--init', str(self.init)])


if __name__ == '__main__':
    unittest.main()

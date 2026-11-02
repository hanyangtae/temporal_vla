"""CPU tests for the fold-boundary handover: live v2 (real class, launches stubbed) stopped by handover_boundary.arm,
then host_fold_coordinator_v3 takes over with the eval-yield gate."""
import fcntl, json, shutil, sys, tempfile, types, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'trqam'), str(ROOT)]
import host_fold_coordinator_v2 as H2   # noqa: E402
import host_fold_coordinator_v3 as H3   # noqa: E402
import handover_boundary as HB          # noqa: E402

TQ = H2.TQ
SPEC = ROOT.parents[2] / 'outputs/analysis/groot_fold27_20261028/folds/fold_spec.json'
SMOKE = 'OpenDrawer-left_s2_j2'


@unittest.skipUnless(SPEC.exists(), 'production fold_spec.json absent')
class HandoverV3Test(unittest.TestCase):
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
        self.out = self.t / 'v2'; self.q = self.out / 'queue'
        self.launched, self.events = [], []
        for arm in TQ.ARMS:
            self.write(SMOKE, arm, 'smoke')
        d = self.out / SMOKE / 'resume_check_v2'; d.mkdir(parents=True)
        (d / 'RESUME_CHECK.json').write_text(json.dumps({**{g: True for g in H2.RESUME_GATES}, 'backend': 'gpu'}))

    def tearDown(self):
        shutil.rmtree(self.t)

    def argv(self):
        return ['--machine', 'kanu', '--spec', str(SPEC), '--sources', str(self.t / 'sources.json'), '--out', str(self.out),
                '--init', str(self.init), '--gpus', '5', '6', '7', '--smoke-fold', SMOKE, '--poll-s', '1', '--max-wait-h', '1',
                '--reuse-ae-from', '/v1root']

    def write(self, fid, arm, stage):
        q = H2.HostCoordinatorV2(H2.parse(self.argv())).queue()
        m = q.marker(fid, arm, stage); steps = 200 if stage == 'smoke' else 50000
        ck = m.parent / 'checkpoint'; ck.mkdir(parents=True, exist_ok=True); (ck / f'params_{steps}.pkl').write_bytes(f'{fid}{arm}{stage}'.encode())
        m.write_text(json.dumps({'harness_receipt': 'r', 'params_sha256': TQ.sha(ck / f'params_{steps}.pkl'),
                                 'fingerprint': {'steps': steps, 'code': q.pins['fingerprint_code'],
                                                 'init': {'step': 500000, 'params_sha256': q.pins['init_params_sha256']}}}))

    def clock(self):
        c = [0.0]
        return c, (lambda s: c.__setitem__(0, c[0] + s)), (lambda: c[0])

    def train(self, cmd, during=None, rc=0):
        fid = cmd[cmd.index('--folds') + 1]
        self.launched.append(fid); self.events.append(('launch', fid))
        self.assertIn('GPUS=5,6,7', cmd); self.assertTrue(any(c.endswith('train_queue_v2.sh') for c in cmd))
        self.assertEqual(cmd[cmd.index('--per-gpu') + 1], '1'); self.assertIn('/v1root', cmd)
        if during:
            during(fid)
        if rc == 0:
            for arm in TQ.ARMS:
                self.write(fid, arm, 'prod')
        return types.SimpleNamespace(returncode=rc)

    def live_v2_until_handover(self, during=None, rc=0):
        """Real v2 run; the sentinel is armed while its FIRST fold trains (exactly the live situation)."""
        c, sleep, now = self.clock()

        def run(cmd, **kw):
            first = not self.launched
            return self.train(cmd, during=(during or self.arm) if first else None, rc=rc if first else 0)
        v2 = H2.HostCoordinatorV2(H2.parse(self.argv()), run=run, sleep=sleep, now=now)
        with self.assertRaises(H2.HC.CoordError) as e:
            v2.run()
        v2._lock.close()                                      # v2 process exits
        return str(e.exception)

    def arm(self, fid):
        st = HB.status(self.out)
        self.assertEqual(st['coordinator_state']['status'], 'training'); self.assertTrue(st['lock_holder_pid'])
        HB.arm(self.out, 'test', now=lambda: 7.0)

    def v3(self, pending, *extra):
        c, sleep, now = self.clock()

        def pend():
            r = pending(); self.events.append(('pending', r[0])); return r
        co = H3.HostCoordinatorV3(H3.parse(self.argv() + list(extra)), pending=pend,
                                  run=lambda cmd, **kw: self.train(cmd), sleep=sleep, now=now)
        self.addCleanup(lambda: co._lock and co._lock.close())
        orig = co.run

        def run():                                            # a failed v3 process exits -> its lock is released
            try:
                return orig()
            finally:
                if co._lock:
                    co._lock.close(); co._lock = None
        co.run = run
        return co

    def snapshot(self):
        return {p.name: p.read_bytes() for p in sorted(self.q.iterdir()) if p.is_file()}

    # ---------------------------------------------------------------- tests
    def test_boundary_handover_eval_first_then_v3_finishes(self):
        msg = self.live_v2_until_handover()
        first = self.launched[0]
        self.assertEqual(self.launched, [first])                          # v2 stopped BEFORE claiming GPUs for fold 2
        self.assertIn('inputs:', msg); self.assertIn('handover_sentinel', msg)
        v2_files = {k: v for k, v in self.snapshot().items() if 'COORD_V2' in k}
        seq = iter([(0, 'ready'), (0, 'running'), (3, '')] + [(3, '')] * 20)
        v3 = self.v3(lambda: next(seq))
        res = v3.run()
        self.assertEqual(len(self.launched), 6); self.assertEqual(len(set(self.launched)), 6)
        self.assertEqual(self.events[1:4], [('pending', 0), ('pending', 0), ('pending', 3)])   # eval got the GPUs first
        self.assertEqual(self.events[4][0], 'launch')
        self.assertTrue(all(self.events[i - 1] == ('pending', 3) for i, e in enumerate(self.events) if e[0] == 'launch' and i > 1))
        self.assertFalse((self.q / 'FAILED.json').exists())
        moved = list(self.q.glob('HANDOVER_STOP.*.json')); self.assertEqual(len(moved), 1)
        self.assertTrue(json.loads(moved[0].read_text())['handover_sentinel'])
        self.assertEqual({k: v for k, v in self.snapshot().items() if 'COORD_V2' in k}, v2_files)   # v2 evidence untouched
        self.assertEqual(json.loads((self.q / 'host_kanu_COORD_V3_STATE.json').read_text())['status'], 'done')
        self.assertEqual(res['done_folds'][0], first)

    def test_no_write_while_waiting_for_live_lock(self):
        self.q.mkdir(parents=True, exist_ok=True)
        (self.q / 'host_kanu_COORD_V2_STATE.json').write_text(json.dumps({'status': 'training', 'phase': 'train:X'}))
        before = self.snapshot()
        with open(self.q / 'host_coord_v2.lock', 'a+') as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB); held.write('4242'); held.flush()
            before = self.snapshot()
            with self.assertRaises(H2.HC.CoordError):
                self.v3(lambda: (3, ''), '--lock-wait-h', '0.01').run()
            self.assertEqual(self.snapshot(), before)                     # nothing written pre-lock, lock pid kept
        self.assertEqual(self.launched, [])

    def test_real_failure_never_consumed(self):
        self.q.mkdir(parents=True, exist_ok=True)
        (self.q / 'FAILED.json').write_text(json.dumps({'job': ['f', 'basic', 'prod'], 'rc': 1}))
        before = (self.q / 'FAILED.json').read_bytes()
        with self.assertRaises(H2.HC.CoordError):
            self.v3(lambda: (3, '')).run()
        self.assertEqual((self.q / 'FAILED.json').read_bytes(), before)
        self.assertEqual(self.launched, []); self.assertEqual(list(self.q.glob('HANDOVER_STOP.*')), [])

    def test_real_train_failure_overwrites_sentinel_and_stops_v3(self):
        def arm_then_fail(fid):
            self.arm(fid)
            (self.q / 'FAILED.json').write_text(json.dumps({'job': [fid, 'mix', 'prod'], 'rc': 1}))   # train_queue.py:231
        self.live_v2_until_handover(during=arm_then_fail, rc=2)
        with self.assertRaises(H2.HC.CoordError):
            self.v3(lambda: (3, '')).run()
        self.assertEqual(json.loads((self.q / 'FAILED.json').read_text())['rc'], 1)
        self.assertEqual(len(self.launched), 1)

    def test_sentinel_cannot_mask_train_failure(self):
        msg = self.live_v2_until_handover(rc=2)                           # train rc!=0, sentinel still in place
        self.assertIn('handover_sentinel', msg)                           # v2 reports it as FAILED.json present at train:
        with self.assertRaises(H2.HC.CoordError) as e:
            self.v3(lambda: (3, '')).run()
        self.assertIn('NOT consumed', str(e.exception))
        self.assertTrue(json.loads((self.q / 'FAILED.json').read_text())['handover_sentinel'])
        self.assertEqual(len(self.launched), 1)

    def test_arm_refusals(self):
        with self.assertRaises(SystemExit):
            HB.arm(self.out, 'x')                                         # no coordinator
        self.q.mkdir(parents=True, exist_ok=True)
        (self.q / 'FAILED.json').write_text('{"rc": 1}')
        self.assertEqual(HB.status(self.out)['failed_json'], 'REAL_FAILURE')
        with self.assertRaises(SystemExit):
            HB.arm(self.out, 'x')
        self.assertEqual((self.q / 'FAILED.json').read_text(), '{"rc": 1}')

    def test_eval_yield_fail_closed_and_bounded(self):
        with self.assertRaises(H2.HC.CoordError):
            self.v3(lambda: (2, 'boom')).run()
        self.assertEqual(self.launched, [])
        (self.q / 'host_kanu_COORD_V3_FAILED.json').unlink()
        with self.assertRaises(H2.HC.CoordError) as e:
            self.v3(lambda: (0, 'pending'), '--yield-max-h', '0.001').run()
        self.assertIn('eval_yield', str(e.exception)); self.assertEqual(self.launched, [])
        self.assertFalse((self.q / 'host_kanu_COORD_V2_FAILED.json').exists())   # v3 failures never land in v2 files


if __name__ == '__main__':
    unittest.main()

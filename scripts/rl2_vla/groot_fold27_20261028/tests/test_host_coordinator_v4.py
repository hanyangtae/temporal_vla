"""CPU tests for host_fold_coordinator_v4 (CPU dry/AE prep one fold ahead, outside the GPU claim) + handover_boundary_v2."""
import json, shutil, sys, tempfile, threading, types, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'trqam'), str(ROOT)]
import host_fold_coordinator_v3 as H3    # noqa: E402
import host_fold_coordinator_v4 as H4    # noqa: E402
import handover_boundary_v2 as HB2       # noqa: E402

H2, TQ, Q2 = H4.H2, H4.TQ, H4.Q2
SPEC = ROOT.parents[2] / 'outputs/analysis/groot_fold27_20261028/folds/fold_spec.json'
SMOKE = 'OpenDrawer-left_s2_j2'
TO = 20


@unittest.skipUnless(SPEC.exists(), 'production fold_spec.json absent')
class CoordV4Test(unittest.TestCase):
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
        self.ev, self.lk = [], threading.Lock()
        self.training, self.drying = set(), set()
        self.violations = []
        for arm in TQ.ARMS:
            self.write(SMOKE, arm, 'smoke')
            self.write(SMOKE, arm, 'prod')                    # first fold already done (as live)
        d = self.out / SMOKE / 'resume_check_v2'; d.mkdir(parents=True)
        (d / 'RESUME_CHECK.json').write_text(json.dumps({**{g: True for g in H2.RESUME_GATES}, 'backend': 'gpu'}))
        self.folds = H2.HostCoordinatorV2(H2.parse(self.argv())).host_folds()

    def tearDown(self):
        shutil.rmtree(self.t)

    def argv(self, *extra):
        return ['--machine', 'kanu', '--spec', str(SPEC), '--sources', str(self.t / 'sources.json'), '--out', str(self.out),
                '--init', str(self.init), '--gpus', '5', '6', '7', '--smoke-fold', SMOKE, '--poll-s', '0.01', '--max-wait-h', '1',
                '--reuse-ae-from', '/v1root', *extra]

    def queue(self):
        return H2.HostCoordinatorV2(H2.parse(self.argv())).queue()

    def write(self, fid, arm, stage):
        q = self.queue()
        m = q.marker(fid, arm, stage); steps = 200 if stage == 'smoke' else 50000
        ck = m.parent / 'checkpoint'; ck.mkdir(parents=True, exist_ok=True); (ck / f'params_{steps}.pkl').write_bytes(f'{fid}{arm}{stage}'.encode())
        m.write_text(json.dumps({'harness_receipt': 'r', 'params_sha256': TQ.sha(ck / f'params_{steps}.pkl'),
                                 'fingerprint': {'steps': steps, 'code': q.pins['fingerprint_code'],
                                                 'init': {'step': 500000, 'params_sha256': q.pins['init_params_sha256']}}}))

    def log(self, *e):
        with self.lk:
            self.ev.append(e)

    # fake CPU dry job (the prep path): writes DRYRUN.json like finetune_fold --dry-run
    def fake_popen(self, fail=None, gate=None):
        def popen(cmd, stdout=None, stderr=None, env=None):
            fid, arm = cmd[cmd.index('--fold') + 1], cmd[cmd.index('--arm') + 1]
            self.assertIn('--dry-run', cmd); self.assertEqual(cmd[cmd.index('--reuse-ae-from') + 1], '/v1root')
            self.assertTrue(cmd[1].endswith('finetune_fold_v2.py'))
            self.assertEqual((env['JAX_PLATFORMS'], env['CUDA_VISIBLE_DEVICES'], env['OMP_NUM_THREADS']), ('cpu', '', '2'))
            with self.lk:
                if fid in self.training:
                    self.violations.append(('dry while same fold trains', fid, arm))
                if self.drying:
                    self.violations.append(('two dry jobs at once', fid, arm, set(self.drying)))
                self.drying.add(fid)
            self.log('dry_start', fid, arm, threading.current_thread().name)
            if gate and fid in gate:
                self.assertTrue(gate[fid].wait(TO), 'prep never overlapped a GPU launch')
            rc = 1 if fail == (fid, arm) else 0
            if rc == 0:
                m = self.queue().marker(fid, arm, 'dry'); m.parent.mkdir(parents=True, exist_ok=True); m.write_text('{}')
            with self.lk:
                self.drying.discard(fid)
            return types.SimpleNamespace(wait=lambda: rc)
        return popen

    def fake_train(self, gate=None):
        def run(cmd, **kw):
            fid = cmd[cmd.index('--folds') + 1]
            self.assertIn('GPUS=5,6,7', cmd); self.assertTrue(any(c.endswith('train_queue_v2.sh') for c in cmd))
            q = self.queue()
            missing = [a for a in TQ.ARMS if not q.done(fid, a, 'dry')]
            if missing:
                self.violations.append(('GPU claimed before dry', fid, missing))
            with self.lk:
                self.training.add(fid)
            self.log('launch', fid)
            if gate:
                nxt = self.folds[self.folds.index(fid) + 1] if self.folds.index(fid) + 1 < len(self.folds) else None
                if nxt in gate:
                    gate[nxt].set()
                    deadline = threading.Event()
                    for _ in range(TO * 100):           # stay "training" until the next fold's prep has started
                        if any(e[:2] == ('dry_start', nxt) for e in self.ev):
                            break
                        deadline.wait(0.01)
            for arm in TQ.ARMS:
                self.write(fid, arm, 'prod')
            with self.lk:
                self.training.discard(fid)
            return types.SimpleNamespace(returncode=0)
        return run

    def v4(self, popen, run, *extra, pending=lambda: (3, '')):
        clock = [0.0]
        co = H4.HostCoordinatorV4(H4.parse(self.argv(*extra)), popen=popen, pending=pending, run=run,
                                  sleep=lambda s: clock.__setitem__(0, clock[0] + s), now=lambda: clock[0])
        orig = co.run

        def go():
            try:
                return orig()
            finally:
                if co._lock:
                    co._lock.close(); co._lock = None
        co.run = go
        return co

    # ------------------------------------------------------------------ tests
    def test_prep_ahead_overlaps_gpu_and_never_races(self):
        todo = [f for f in self.folds if f != SMOKE]
        gate = {f: threading.Event() for f in todo[1:]}
        res = self.v4(self.fake_popen(gate=gate), self.fake_train(gate=gate)).run()
        self.assertEqual(self.violations, [])
        launches = [e[1] for e in self.ev if e[0] == 'launch']
        self.assertEqual(launches, todo); self.assertEqual(res['done_folds'][0], SMOKE)
        # first remaining fold is prepared synchronously (main thread); later folds by the prep thread during the previous launch
        first_dry = [e for e in self.ev if e[0] == 'dry_start' and e[1] == todo[0]]
        self.assertTrue(all(e[3] == 'MainThread' for e in first_dry))
        for prev, nxt in zip(todo, todo[1:]):
            # overlap: nxt's dry cannot finish before launch(prev) opens its gate, and train(prev) waits for nxt's dry start
            i_dry = min(i for i, e in enumerate(self.ev) if e[:2] == ('dry_start', nxt))
            self.assertTrue(gate[nxt].is_set())
            self.assertTrue(all(e[3].startswith('prep-') for e in self.ev if e[0] == 'dry_start' and e[1] == nxt))
            self.assertLess(i_dry, self.ev.index(('launch', nxt)))
        st = json.loads((self.q / 'host_kanu_COORD_V4_STATE.json').read_text())
        self.assertEqual(st['status'], 'done'); self.assertTrue(all(r.get('ok') for r in st['prep'].values()))
        self.assertFalse((self.q / 'FAILED.json').exists()); self.assertFalse((self.q / 'QUEUE_STATE.json').exists())
        self.assertEqual(len(list((self.q / 'prep_v4').glob('*.log'))), 3 * len(todo))

    def test_prep_failure_stops_before_gpu_claim(self):
        todo = [f for f in self.folds if f != SMOKE]
        with self.assertRaises(H2.HC.CoordError) as e:
            self.v4(self.fake_popen(fail=(todo[1], 'pbrs')), self.fake_train()).run()
        self.assertIn(f'prep:{todo[1]}', str(e.exception))
        self.assertEqual([x[1] for x in self.ev if x[0] == 'launch'], [todo[0]])
        self.assertTrue((self.q / 'host_kanu_COORD_V4_FAILED.json').exists())
        self.assertFalse((self.q / 'FAILED.json').exists())                    # queue failure file never written by prep
        self.assertEqual(self.violations, [])

    def test_dry_command_is_the_queue_command(self):
        co = self.v4(self.fake_popen(), self.fake_train())
        f = next(x for x in co.spec['folds'] if x['fold_id'] == self.folds[1])
        args, _ = Q2.parse(['--spec', str(SPEC), '--sources', str(self.t / 'sources.json'), '--out', str(self.out),
                            '--init', str(self.init), '--folds', f['fold_id'], '--per-gpu', '1', '--gpus', '5'])
        ref = Q2.TrainQueueV2(args, reuse_ae_from=Path('/v1root'))
        for arm in TQ.ARMS:
            self.assertEqual(co.dry_queue().cmd(f, arm, 'dry'), ref.cmd(f, arm, 'dry'))
        self.assertEqual({k: v for k, v in co.dry_queue().env(None).items() if 'THREADS' in k or k.startswith(('JAX', 'CUDA'))},
                         {k: v for k, v in ref.env(None).items() if 'THREADS' in k or k.startswith(('JAX', 'CUDA'))})

    def test_v3_to_v4_handover_and_stale_sentinel_refused(self):
        todo = [f for f in self.folds if f != SMOKE]
        clock = [0.0]

        def v3_run(cmd, **kw):
            fid = cmd[cmd.index('--folds') + 1]
            if fid == todo[0]:
                st = HB2.status(self.out)
                self.assertEqual(st['live_version'], 'V3'); self.assertTrue(st['lock_holder_pid'])
                HB2.arm(self.out, 'test', now=lambda: 0.0)
            for arm in TQ.ARMS:
                self.write(fid, arm, 'dry'.replace('dry', 'prod'))
            self.log('launch', fid)
            return types.SimpleNamespace(returncode=0)
        v3 = H3.HostCoordinatorV3(H3.parse(self.argv()), pending=lambda: (3, ''), run=v3_run,
                                  sleep=lambda s: clock.__setitem__(0, clock[0] + s), now=lambda: clock[0])
        with self.assertRaises(H2.HC.CoordError):
            v3.run()
        v3._lock.close()
        self.assertEqual([e[1] for e in self.ev if e[0] == 'launch'], [todo[0]])
        # stale: sentinel armed AFTER the recorded stop -> refused
        body = json.loads((self.q / 'FAILED.json').read_text())
        (self.q / 'FAILED.json').write_text(json.dumps({**body, 'time': 1e12}))
        with self.assertRaises(H2.HC.CoordError) as e:
            self.v4(self.fake_popen(), self.fake_train()).run()
        self.assertIn('NOT consumed', str(e.exception))
        (self.q / 'FAILED.json').write_text(json.dumps(body))
        self.v4(self.fake_popen(), self.fake_train()).run()
        self.assertEqual([e[1] for e in self.ev if e[0] == 'launch'], todo)
        self.assertEqual(len(list(self.q.glob('HANDOVER_STOP.*.json'))), 1)
        self.assertTrue((self.q / 'host_kanu_COORD_V3_FAILED.json').exists())       # v3 evidence kept
        self.assertEqual(self.violations, [])

    def test_exact_route_live_v2_armed_sentinel_then_v4(self):
        """Deployed route: v2 trains a fold, handover_boundary.py (v1 tool) arms during it, v3 never took the lock,
        v4 lock-waits, v2 stops at the boundary, v4 consumes the SAME sentinel and continues with CPU prep first."""
        import handover_boundary as HB
        todo = [f for f in self.folds if f != SMOKE]
        clock = [0.0]

        def v2_run(cmd, **kw):
            fid = cmd[cmd.index('--folds') + 1]
            if fid == todo[0]:
                HB.arm(self.out, 'eval READY priority', now=lambda: clock[0])
            for arm in TQ.ARMS:
                self.write(fid, arm, 'dry'); self.write(fid, arm, 'prod')     # live queue did its own dry + prod
            self.log('launch', fid)
            return types.SimpleNamespace(returncode=0)
        v2 = H2.HostCoordinatorV2(H2.parse(self.argv()), run=v2_run,
                                  sleep=lambda s: clock.__setitem__(0, clock[0] + s), now=lambda: clock[0])
        with self.assertRaises(H2.HC.CoordError) as e:
            v2.run()
        v2._lock.close()
        self.assertIn(f'inputs:{todo[1]}', str(e.exception))
        self.assertFalse((self.q / 'host_kanu_COORD_V3_STATE.json').exists())
        self.v4(self.fake_popen(), self.fake_train()).run()
        self.assertEqual([x[1] for x in self.ev if x[0] == 'launch'], todo)
        moved = list(self.q.glob('HANDOVER_STOP.*.json')); self.assertEqual(len(moved), 1)
        self.assertEqual(json.loads(moved[0].read_text())['expected_stop_after'], f'train:{todo[0]}')
        self.assertFalse((self.q / 'FAILED.json').exists()); self.assertEqual(self.violations, [])
        dry_folds = {x[1] for x in self.ev if x[0] == 'dry_start'}
        self.assertEqual(dry_folds, set(todo[1:]))                               # v2's fold never re-dried

    def test_arm_v2_refuses_when_not_mid_fold(self):
        with self.assertRaises(SystemExit):
            HB2.arm(self.out, 'x')
        self.assertFalse((self.q / 'FAILED.json').exists())


if __name__ == '__main__':
    unittest.main()

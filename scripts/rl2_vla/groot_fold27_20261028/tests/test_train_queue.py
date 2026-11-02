"""CPU tests for trqam/train_queue.py (fake launcher writing real-format markers; production fold spec).

    .venvs/rl2_iid/bin/python -m pytest -q scripts/rl2_vla/groot_fold27_20261028/tests/test_train_queue.py
"""
import json, os, shutil, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'trqam'), str(ROOT)]
import train_queue as TQ   # noqa: E402

SPEC = ROOT.parents[2] / 'outputs/analysis/groot_fold27_20261028/folds/fold_spec.json'


class FakeProc:
    def __init__(self, rc, on_done):
        self.rc, self.on_done, self.n = rc, on_done, 0

    def poll(self):
        self.n += 1
        if self.n < 2:
            return None
        self.on_done(); return self.rc


@unittest.skipUnless(SPEC.exists(), 'production fold_spec.json absent')
class QueueTest(unittest.TestCase):
    def setUp(self):
        self.t = Path(tempfile.mkdtemp())
        self.init = self.t / 'init'; self.init.mkdir(); (self.init / 'params_500000.pkl').write_bytes(b'p')
        src = {}
        for scene, ready in (('OpenDrawer/left|s2', True), ('OpenDrawer/right|s1', True), ('PPCC/bread|s4', False)):
            e, l = self.t / 'ep' / scene.replace('/', '_'), self.t / 'lat' / scene.replace('/', '_')
            e.mkdir(parents=True); l.mkdir(parents=True); (e / 'contract.json').write_text('{}')
            if ready:
                (l / 'COMPLETE.json').write_text('{}')
            src[scene] = {'episodes': str(e), 'latents': str(l)}
        (self.t / 'sources.json').write_text(json.dumps(src))
        self.ledger = self.t / 'slots.json'; self.ledger.write_text('{}')
        self.launched, self.fail_key, self.peak = [], None, 0
        self.live = set()
        os.environ['FOLD27_GPU_LEASE'] = 'kanu:6,7:test'

    def tearDown(self):
        os.environ.pop('FOLD27_GPU_LEASE', None); shutil.rmtree(self.t)

    def args(self, *extra):
        return TQ.parse(['--spec', str(SPEC), '--sources', str(self.t / 'sources.json'), '--init', str(self.init),
                         '--out', str(self.t / 'out'), '--gpus', '6', '7', '--per-gpu', '2', '--poll-s', '0',
                         '--harness-ledger', str(self.ledger), *extra])

    def popen(self, cmd, stdout=None, stderr=None, env=None):
        fold, arm = cmd[cmd.index('--fold') + 1], cmd[cmd.index('--arm') + 1]
        stage = 'dry' if '--dry-run' in cmd else ('smoke' if '--smoke-steps' in cmd else 'prod')
        if stage == 'dry':
            self.assertEqual(env['JAX_PLATFORMS'], 'cpu'); self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
        else:
            self.assertIn(env['CUDA_VISIBLE_DEVICES'], ('6', '7')); self.assertIn('--init', cmd)
        self.launched.append((fold, arm, stage)); key = (fold, arm, stage); self.live.add(key)
        gpu_live = sum(1 for k in self.live if k[2] != 'dry'); self.peak = max(self.peak, gpu_live)
        q = self

        def done():
            q.live.discard(key)
            if key == q.fail_key:
                return
            m = q.queue.marker(fold, arm, stage); m.parent.mkdir(parents=True, exist_ok=True)
            if stage == 'dry':
                m.write_text('{}'); return
            steps = 200 if stage == 'smoke' else 50000
            ck = m.parent / 'checkpoint'; ck.mkdir(exist_ok=True); (ck / f'params_{steps}.pkl').write_bytes(f'{fold}{arm}'.encode())
            pins = q.queue.pins
            m.write_text(json.dumps({'harness_receipt': 'gpu_lease:x', 'params_sha256': TQ.sha(ck / f'params_{steps}.pkl'),
                                     'fingerprint': {'steps': steps, 'code': pins['fingerprint_code'],
                                                     'init': {'step': 500000, 'params_sha256': pins['init_params_sha256']}}}))
        return FakeProc(1 if key == self.fail_key else 0, done)

    def run_queue(self, *extra):
        self.queue = TQ.TrainQueue(self.args(*extra), popen=self.popen, sleep=lambda s: None)
        return self.queue.run()

    def test_full_order_smoke_gate_and_resume(self):
        res = self.run_queue()
        ready = {f for f, _, _ in self.launched}
        self.assertTrue(all(f.startswith('OpenDrawer') for f in ready))           # bread latents not COMPLETE
        self.assertEqual(res['prod_done'], 6 * 3); self.assertIn('PPCC/bread|s4', res['not_ready'])
        for arm in TQ.ARMS:
            stages = [(f, s) for f, a, s in self.launched if a == arm]
            first_prod = next(i for i, (_, s) in enumerate(stages) if s == 'prod')
            self.assertEqual([s for _, s in stages].count('smoke'), 1)
            self.assertLess(next(i for i, (_, s) in enumerate(stages) if s == 'smoke'), first_prod)
        self.assertLessEqual(self.peak, 4)                                        # 2 GPUs x 2
        n = len(self.launched)
        self.run_queue()
        self.assertEqual(len(self.launched), n)                                   # everything verified done -> nothing reruns

    def test_failure_stops_and_needs_retry(self):
        self.fail_key = ('OpenDrawer-left_s2_j2', 'basic', 'dry')
        with self.assertRaises(TQ.QueueError):
            self.run_queue()
        self.assertTrue((self.t / 'out/queue/FAILED.json').exists())
        self.assertFalse(any(s == 'prod' for _, _, s in self.launched))
        with self.assertRaises(TQ.QueueError):
            self.run_queue()
        self.fail_key = None
        self.assertEqual(self.run_queue('--retry-failed')['prod_done'], 18)

    def test_bounded_fold_bundles_reuse_smoke(self):
        self.run_queue('--folds', 'OpenDrawer-left_s2_j2')
        first = list(self.launched)
        self.assertEqual({f for f, _, _ in first}, {'OpenDrawer-left_s2_j2'})
        self.assertEqual(sum(s == 'smoke' for _, _, s in first), 3)
        self.launched.clear()
        self.run_queue('--folds', 'OpenDrawer-right_s1_j0')
        self.assertEqual({s for _, _, s in self.launched}, {'dry', 'prod'})     # smoke from the first bundle reused
        self.assertEqual({f for f, _, _ in self.launched}, {'OpenDrawer-right_s1_j0'})

    def test_dry_only_needs_no_lease(self):
        os.environ.pop('FOLD27_GPU_LEASE')
        with self.assertRaises(TQ.QueueError):
            self.run_queue()
        self.run_queue('--dry-only')
        self.assertTrue(self.launched and all(s == 'dry' for _, _, s in self.launched))

    def test_harness_preflight(self):
        self.ledger.write_text(json.dumps({'r1': {'machine': 'kanu', 'gpus': [6, 7], 'session': 'collect-topup-fork', 'kind': 'collect'}}))
        with self.assertRaises(TQ.QueueError):
            TQ.harness_preflight([6], self.ledger)                                # reserved GPU
        self.ledger.write_text(json.dumps({'r1': {'machine': 'kanu', 'gpus': [5, 6], 'session': 'a', 'kind': 'eval'},
                                           'r2': {'machine': 'srv50', 'gpus': [2], 'session': 'b', 'kind': 'eval'}}))
        TQ.harness_preflight([7], self.ledger)                                    # 3 kanu GPUs total: ok
        with self.assertRaises(TQ.QueueError):
            TQ.harness_preflight([1, 7], self.ledger)                             # would be 4 > cap
        with self.assertRaises(SystemExit):
            TQ.parse(['--sources', 'x', '--init', str(self.init), '--gpus', '1', '2', '3', '4'])

    def test_freeze_verified_done_and_no_empty_success(self):
        self.run_queue('--folds', 'OpenDrawer-left_s2_j2')
        fz = json.loads((self.t / 'out/queue/QUEUE_FREEZE.json').read_text())
        self.assertIn('finetune_fold.py', fz['fingerprint_code']); self.assertIn('init_params_sha256', fz)
        q = self.queue
        self.assertTrue(q.done('OpenDrawer-left_s2_j2', 'basic', 'prod'))
        ck = q.marker('OpenDrawer-left_s2_j2', 'basic', 'prod').parent / 'checkpoint/params_50000.pkl'
        ck.write_bytes(b'tampered')
        self.assertFalse(q.done('OpenDrawer-left_s2_j2', 'basic', 'prod'))      # hash, not presence
        with self.assertRaises(TQ.QueueError):
            self.run_queue('--folds', 'PPCC-bread_s4_j1')                         # latents not COMPLETE: no empty success
        (self.init / 'params_500000.pkl').write_bytes(b'other init')
        with self.assertRaises(TQ.QueueError):
            self.run_queue('--folds', 'OpenDrawer-left_s2_j2')                    # frozen inputs changed -> refuse

    def test_gpus_from_harness_and_machine_cap(self):
        os.environ.update(GPUS='5', HARNESS_MACHINE='srv50')
        try:
            a = TQ.parse(['--spec', str(SPEC), '--sources', str(self.t / 'sources.json'), '--init', str(self.init)])
            self.assertEqual(a.gpus, [5])
            os.environ['GPUS'] = '1,2'
            with self.assertRaises(SystemExit):
                TQ.parse(['--spec', str(SPEC), '--sources', str(self.t / 'sources.json'), '--init', str(self.init)])
        finally:
            for k in ('GPUS', 'HARNESS_MACHINE'):
                os.environ.pop(k, None)


@unittest.skipUnless(SPEC.exists(), 'production fold_spec.json absent')
class WaitReadyTest(unittest.TestCase):
    def test_waits_then_ready_and_bad_inputs(self):
        import wait_ready as WR
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); e, l = d / 'ep', d / 'lat'; e.mkdir(); l.mkdir(); (e / 'contract.json').write_text('{}')
            (d / 's.json').write_text(json.dumps({'OpenDrawer/left|s2': {'episodes': str(e), 'latents': str(l)}}))
            clock, polls = [0.0], []

            def sleep(s):
                clock[0] += s; polls.append(s)
                if len(polls) == 2:
                    (l / 'COMPLETE.json').write_text(json.dumps({'episodes': 200}))
            args = ['--spec', str(SPEC), '--sources', str(d / 's.json'), '--folds', 'OpenDrawer-left_s2_j2', '--poll-s', '1']
            self.assertEqual(WR.main(args, sleep=sleep, now=lambda: clock[0]), 0)
            self.assertEqual(len(polls), 2)
            (l / 'COMPLETE.json').write_text(json.dumps({'episodes': 199}))
            self.assertEqual(WR.main(args, sleep=sleep, now=lambda: clock[0]), 2)       # wrong count: no waiting
            self.assertEqual(WR.main(args[:-4] + ['--folds', 'PPCC-bread_s4_j1', '--poll-s', '1'], sleep=sleep,
                                     now=lambda: clock[0]), 2)                           # scene not in sources
            (l / 'COMPLETE.json').unlink()
            self.assertEqual(WR.main(args + ['--max-wait-h', '0.0001'], sleep=lambda s: None, now=lambda: 0.0), 75)
            (l / 'COMPLETE.json').write_text(json.dumps({'episodes': 200}))
            gate = d / 'GATED_SMOKE_AUDIT_PASS.json'
            self.assertEqual(WR.main(args + ['--require', str(gate), '--max-wait-h', '0.0001'], sleep=lambda s: None,
                                     now=lambda: 0.0), 75)                               # latents ready but gate missing
            gate.write_text('{}')
            self.assertEqual(WR.main(args + ['--require', str(gate)], sleep=lambda s: None, now=lambda: 0.0), 0)


class WaitLaunchTest(unittest.TestCase):
    def test_retries_only_on_harness_block(self):
        import types, wait_launch as WL
        seq = [(2, b'[harness] BLOCKED: kanu: all-session GPU limit exceeded\n'), (2, b'[harness] BLOCKED: x\n'), (0, b'')]
        calls, clock = [], [0.0]

        def run(cmd, stdout=None, stderr=None):
            rc, err = seq[len(calls)]; calls.append(cmd); return types.SimpleNamespace(returncode=rc, stderr=err)
        with tempfile.TemporaryDirectory() as d:
            rc = WL.wait_launch(['x'], Path(d), 1, 10, run=run, sleep=lambda s: clock.__setitem__(0, clock[0] + s), now=lambda: clock[0])
            self.assertEqual((rc, len(calls)), (0, 3))
            calls.clear(); seq[:] = [(2, b'Traceback: real error\n')]
            self.assertEqual(WL.wait_launch(['x'], Path(d), 1, 10, run=run, sleep=lambda s: None, now=lambda: 0.0), 2)
            self.assertEqual(len(calls), 1)                                           # real failure: no retry
            calls.clear(); seq[:] = [(2, b'[harness] BLOCKED: y\n')] * 10
            self.assertEqual(WL.wait_launch(['x'], Path(d), 0.001, 10, run=run, sleep=lambda s: None, now=lambda: 0.0), 75)


if __name__ == '__main__':
    unittest.main()

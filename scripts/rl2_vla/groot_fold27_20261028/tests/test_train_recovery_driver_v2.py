"""train_recovery_driver_v2 on the same fake host/fixtures as test_train_recovery_driver (no real signal/process/file
outside the temp dir): a COMPLETED eval whose consumer exited 'yielded to trainer' transitions (v1 waits forever on it);
an incomplete yielded eval keeps waiting; failures / audit errors / RUN_COMPLETE hash mismatch are refused; an alive eval
consumer or a remaining harness receipt keeps the boundary closed (no signal, no spawn)."""
import json, signal, sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tests'), str(ROOT / 'trqam'), str(ROOT / 'eval/runtime'), str(ROOT.parents[2] / 'scripts/utils')]
import test_train_recovery_driver as TD     # noqa: E402  (fixtures: fake host, audited snapshot, RUN_COMPLETE files)
import train_recovery_driver_v2 as V2       # noqa: E402

RD = TD.RD
YIELD = 'yielded to trainer'


class Stop(BaseException):
    pass


class DriverV2Test(TD.DriverTest):
    def finish(self, exit_=YIELD, failures=(), consumer_gone=True, receipt_gone=True):
        """The eval completes all 480 rows; the consumer writes its exit record."""
        self.done_rows = 480
        if consumer_gone:
            self.h.starts.pop(3088084, None)
            self.p.apps = [a for a in self.p.apps if a['pid'] != 4000]      # its serve left the GPU
        if receipt_gone:
            self.h.led = {}
        self.qs.write_text(json.dumps({'exit': exit_, 'failures_this_run': list(failures)}))

    def v2(self, max_sleeps=None, cls=None, **finish_kw):
        def sleep(s):
            self.sleeps.append(s); self.h.clock += s
            if len(self.sleeps) == 2:
                self.finish(**finish_kw)
            if max_sleeps and len(self.sleeps) >= max_sleeps:
                raise Stop()
        return (cls or V2.DriverV2)(self.cfg, self.h, sleep=sleep, eval_snapshot=self.snap)

    def assert_untouched(self):
        self.assertEqual(self.h.kills, []); self.assertEqual(self.h.spawned, [])

    # ---------------------------------------------------------------- allowed
    def test_completed_yield_transitions(self):
        self.assertEqual(self.v2().run(), 0)
        self.assertEqual(self.h.kills, [(631152, signal.SIGTERM), (627282, signal.SIGTERM)])     # only the two pinned ids
        self.assertEqual(len(self.h.spawned), 2)
        self.assertIn('DONE.json', self.outcome())
        steps = [json.loads(x) for x in (Path(self.cfg['state_dir']) / 'steps.jsonl').read_text().splitlines()]
        self.assertTrue(any(s['step'] == 'boundary_ready' for s in steps))

    def test_completed_plain_exit_still_transitions(self):
        self.assertEqual(self.v2(exit_='no runnable job').run(), 0)

    def test_v1_waits_forever_on_completed_yield(self):                     # documents the bug v2 fixes
        with self.assertRaises(Stop):
            self.v2(max_sleeps=6, cls=RD.Driver).run()
        self.assert_untouched()
        last = [json.loads(x) for x in (Path(self.cfg['state_dir']) / 'steps.jsonl').read_text().splitlines()][-1]
        self.assertIn('yielded to trainer', json.dumps(last['reasons']))

    # ---------------------------------------------------------------- keep waiting
    def test_incomplete_yield_keeps_waiting(self):
        drv = self.v2(max_sleeps=6)
        orig = self.finish

        def partial(**kw):
            orig(**kw); self.done_rows = 479                                 # one row short (e.g. failed/skipped)
        self.finish = partial
        with self.assertRaises(Stop):
            drv.run()
        self.assert_untouched(); self.assertEqual(self.outcome(), {})

    def test_unknown_or_missing_exit_keeps_waiting(self):
        for ex in ('stopped', None):
            self.tearDown(); self.setUp()
            with self.assertRaises(Stop):
                self.v2(max_sleeps=5, exit_=ex).run()
            self.assert_untouched()

    def test_active_consumer_cannot_transition(self):
        with self.assertRaises(Stop):
            self.v2(max_sleeps=6, consumer_gone=False).run()                 # completed + yielded, consumer still alive
        self.assert_untouched()

    def test_remaining_receipt_cannot_transition(self):
        with self.assertRaises(Stop):
            self.v2(max_sleeps=6, receipt_gone=False).run()
        self.assert_untouched()

    def test_unknown_gpu_process_cannot_transition(self):
        orig = self.finish

        def leftover(**kw):
            orig(**kw); self.p.apps.append({'pid': 4001, 'used_mib': 6000})  # a serve still on the GPU
        self.finish = leftover
        with self.assertRaises(Stop):
            self.v2(max_sleeps=6).run()
        self.assert_untouched()

    # ---------------------------------------------------------------- refused (no signal)
    def test_yield_with_failures_refused(self):
        self.assertEqual(self.v2(failures=[{'job': ['F', 'a'], 'marker': 'm'}]).run(), 2)
        self.assert_untouched(); self.assertIn('failures', self.outcome()['REFUSED.json']['reason'])

    def test_yield_with_audit_error_refused(self):
        self.audit_bad = True
        self.assertEqual(self.v2().run(), 2)
        self.assert_untouched(); self.assertIn('audit', self.outcome()['REFUSED.json']['reason'])

    def test_yield_with_run_complete_hash_mismatch_refused(self):
        def tamper(root):
            p = root / TD.FOLDS[2] / TD.ARMS[1] / 'RUN_COMPLETE.json'
            if p.exists():
                b = json.loads(p.read_text()); b['progress_sha256'] = '0' * 64; p.write_text(json.dumps(b))
        self.rc_tamper = tamper
        self.assertEqual(self.v2().run(), 2)
        self.assert_untouched(); self.assertIn('RUN_COMPLETE', self.outcome()['REFUSED.json']['reason'])

    def test_yield_with_other_coordinate_set_refused(self):
        self.extra_fold = True
        self.assertEqual(self.v2().run(), 2)
        self.assert_untouched(); self.assertIn('pinned set', self.outcome()['REFUSED.json']['reason'])


for _n in [n for n in dir(TD.DriverTest) if n.startswith('test')]:
    if getattr(DriverV2Test, _n) is getattr(TD.DriverTest, _n):
        setattr(DriverV2Test, _n, None)                                     # v1 suite runs in its own module


if __name__ == '__main__':
    unittest.main()

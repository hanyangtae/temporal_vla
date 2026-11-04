"""deadline_stop on the v1/v2 fake fixture (CPU only; no real signal / nvidia-smi / harness). The fixture's tearDown asserts
that progress, claims, registry and partial outputs are byte-identical afterwards (completed/partial results preserved).
Covers: dry-run = observe-only act (no signal), full stop with v3 outer follower + root container collector + release,
contract-approved process and other foreign never signalled, old guardian frozen first then ended (only if pinned),
replaced consumer / receipt mismatch / follower mismatch -> no signal, consumer already gone -> no signal,
old-guardian pin mismatch -> that pid untouched, receipt kept when wrapper stays, duplicate run (lock), CLI new paths."""
import json, signal, sys, unittest
from pathlib import Path

HD = Path(__file__).resolve().parents[1]
F27 = HD.parents[1]
sys.path[:0] = [str(HD), str(F27 / 'tests'), str(F27 / 'eval/runtime'), str(F27.parents[2] / 'scripts/utils')]
import test_foreign_gpu_guardian as T     # noqa: E402  (unchanged)
import test_foreign_gpu_guardian_v2 as T2  # noqa: E402  (unchanged)
import deadline_stop as DS                 # noqa: E402

V3_OUTER = 'bash scripts/rl2_vla/groot_fold27_20261028/eval/runtime/eval_gpu_queue_v3.sh'


class DeadlineStopTest(T2.SharingTest):
    def setUp(self):
        super().setUp()
        self.o.self_ident = lambda: {'pid': 970, 'start': 97}
        self.o.procs[100].cmd = V3_OUTER                        # the real srv50 follower is the v3 outer loop
        self.o.ledger[T.RECEIPT]['consumer_pid'] = 120
        self.o.add(980, 98, 1, 'python foreign_gpu_guardian_v3.py --config srv50_candle.json --state-dir st watch')
        self.foreign_other = self.o.add(501, 51, 1, 'python someone_elses_job.py', uid=1000)   # same uid, not ours
        self.pins = {'consumer_pid': 120, 'consumer_start': 12, 'receipt': T.RECEIPT, 'old_guardian_pid': 980,
                     'old_guardian_start': 98, 'follower_pid': None, 'follower_start': None}
        self.n = 0

    def go(self, dry=False, **pins):
        self.n += 1
        cfg = {**self.cfg, 'sharing_authorization': str(self.ap)}
        return DS.run(cfg, self.o, self.t / f'ds{self.n}', {**self.pins, **pins}, dry, probe=self.probe)

    def no_signal(self):
        self.assertEqual(self.o.kills, []); self.assertEqual(self.o.cexec, [])

    def test_dry_run_is_real_act_observe_only(self):
        ev, rc = self.go(dry=True)
        self.assertEqual((ev['result'], rc), ('dry_run_ok', 0)); self.no_signal()
        self.assertEqual(ev['follower'], (100, 10)); self.assertTrue(ev['old_guardian_verified'])
        self.assertTrue(list((self.t / 'ds1').glob('ACTION.*.json')))
        self.assertIn(T.RECEIPT, self.o.ledger)

    def test_full_stop_releases_and_spares_others(self):
        ev, rc = self.go()
        self.assertEqual((ev['result'], rc), ('released', 0))
        k = self.o.kills
        self.assertEqual(k[0], (980, int(signal.SIGSTOP)))                     # old guardian frozen first
        self.assertEqual(k[1:4], [(100, int(signal.SIGSTOP)), (120, int(signal.SIGSTOP)), (130, int(signal.SIGSTOP))])
        self.assertIn(('fold27-iso-env-gpu1', 42, int(signal.SIGTERM)), self.o.cexec)   # root env-container collector
        for p in (100, 110, 120, 130, 140, 150, 160, 980):
            self.assertNotIn(p, self.o.procs)
        self.never(600, 501)                                                    # approved + other foreign untouched
        self.assertNotIn(T.RECEIPT, self.o.ledger)
        rel = [c for c in self.o.harness_calls if c[0] == 'release']
        self.assertEqual(rel, [('release', '--receipt', T.RECEIPT, '--session', T.SESSION)])     # official path, once
        post = ev['post']
        self.assertFalse(post['consumer_alive'] or post['follower_alive'] or post['old_guardian_alive'])
        self.assertEqual(post['our_pids_on_gpu'], []); self.assertEqual(post['approved_alive'], [[600, True]])
        self.assertFalse(post['receipt_in_ledger'])

    def test_replaced_consumer_is_not_retargeted(self):
        ev, rc = self.go(consumer_pid=121, consumer_start=12)
        self.assertEqual(rc, 7); self.no_signal()                               # pinned identity is not running
        self.o.procs[120].start = 13                                            # same pid, other process
        ev, rc = self.go()
        self.assertEqual(rc, 7); self.no_signal()

    def test_registration_or_receipt_mismatch_no_signal(self):
        ev, rc = self.go(receipt='X' * 32)
        self.assertEqual((ev['result'], rc), ('refused_pin_mismatch', 6)); self.no_signal()
        self.o.ledger[T.RECEIPT]['consumer_pid'] = 999
        ev, rc = self.go()
        self.assertEqual(rc, 6); self.assertIn('consumer_pid', ev['reason']); self.no_signal()

    def test_follower_pin_mismatch_no_signal(self):
        ev, rc = self.go(follower_pid=100, follower_start=99)
        self.assertEqual(rc, 6); self.no_signal()
        ev, rc = self.go(dry=True, follower_pid=100, follower_start=10)
        self.assertEqual(rc, 0)

    def test_consumer_already_gone(self):
        for p in (160, 150, 140, 130, 120):
            self.o.remove(p)
        ev, rc = self.go()
        self.assertEqual((ev['result'], rc), ('consumer_not_running', 7)); self.no_signal()

    def test_old_guardian_pin_mismatch_untouched(self):
        ev, rc = self.go(old_guardian_start=77)
        self.assertEqual(rc, 0); self.assertFalse(ev['old_guardian_verified'])
        self.assertIn(980, self.o.procs); self.assertNotIn(980, {k[0] for k in self.o.kills})

    def test_receipt_kept_when_not_proven_absent(self):
        self.o.wrapper_stays = True
        ev, rc = self.go()
        self.assertEqual((ev['result'], rc), ('kept', 5)); self.assertIn(T.RECEIPT, self.o.ledger)

    def test_duplicate_run_refused_by_lock(self):
        import guardian_v4 as G4
        holder = G4.GuardianV4({**self.cfg, 'sharing_authorization': str(self.ap)}, self.o, self.t / 'holder', probe=self.probe)
        try:
            ev, rc = self.go()
            self.assertEqual(rc, 6); self.no_signal()
        finally:
            holder.close()

    def test_cli_refuses_existing_paths(self):
        (self.t / 'exists.json').write_text('{}'); self.files[self.t / 'exists.json'] = b'{}'
        cfgp = self.t / 'c.json'; cfgp.write_text(json.dumps(self.cfg)); self.files[cfgp] = cfgp.read_bytes()
        rc = DS.main(['--config', str(cfgp), '--state-dir', str(self.t / 'new'), '--out', str(self.t / 'exists.json'),
                      '--consumer-pid', '120', '--consumer-start', '12', '--receipt', T.RECEIPT,
                      '--old-guardian-pid', '980', '--old-guardian-start', '98', '--dry-run'])
        self.assertEqual(rc, 2); self.no_signal()



for _n in [n for n in dir(T2.SharingTest) if n.startswith('test_') and n not in DeadlineStopTest.__dict__]:
    setattr(DeadlineStopTest, _n, None)      # fixture only: inherited v1/v2 tests run in their own files (with their setUp)

if __name__ == '__main__':
    unittest.main()

"""train_guardian_v2: every v1 test re-run on v2 (except the probe-failure case, whose v1 behaviour 'no action' is exactly
what v2 changes), plus: NVML failure while training -> ALERT + ADMISSION_UNSAFE + stop OUR training + receipt KEPT;
NVML failure idle -> ALERT/UNSAFE only; ledger unreachable with a valid cached proof -> monitoring continues, stop uses
cached proof, release impossible -> receipt kept; ledger unreachable without proof -> UNSAFE, no signal; own-PID-only
measurement separate from the provisional cap budget. Fake host only."""
import json, signal, sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tests'), str(ROOT / 'trqam')]
import test_train_guardian as TT     # noqa: E402
import train_guardian_v2 as TG2     # noqa: E402


class V2(TT.TrainGuardianTest):
    def setUp(self):
        super().setUp()
        self.auth['sessions'][TT.SESSION]['per_serve_basis'] = 'provisional_cap'
        self.ap.write_text(json.dumps(self.auth)); self.files[self.ap] = self.ap.read_bytes()

    def g(self, **kw):
        return TG2.TrainGuardianV2({**self.cfg, 'kind': TG2.KIND, **kw}, self.o, self.t / 'state', probe=self.probe)

    def unsafe(self):
        p = self.t / 'state' / 'ADMISSION_UNSAFE.json'
        return json.loads(p.read_text()) if p.exists() else None

    def break_ledger(self):
        orig = self.o.harness

        def h(*a):
            if a[0] == 'status':
                raise RuntimeError('ssh to coordinator failed')
            return orig(*a)
        self.o.harness = h

    # v1 expected 'no_action' on NVML failure; v2 must NOT treat it as safe
    def test_probe_failure_no_action(self):
        self.probe.fail = True
        r = self.g().run(max_polls=4)
        self.assertEqual(r, 'kept')                                                   # own stopped, release unprovable
        self.assertIn((240, int(signal.SIGTERM)), self.o.kills); self.never(600, 700, 701)
        self.assertIn(TT.RID, self.o.ledger)                                         # receipt kept (blocks other admissions)
        self.assertIn('unverifiable', self.unsafe()['reason'] + json.dumps(self.unsafe()))

    def test_gpu_failure_idle_is_unsafe_not_pass(self):
        self.o.ledger.clear(); self.probe.fail = True
        self.assertEqual(self.g().run(max_polls=3), 'no_action'); self.assertEqual(self.o.kills, [])
        self.assertIn('idle', self.unsafe()['reason'])

    def test_single_gpu_failure_alerts_without_stopping(self):
        g = self.g(); g.poll()
        self.probe.fail = True
        self.assertIsNone(g.poll())                                                   # 1 < confirm_polls
        self.assertIsNotNone(self.unsafe()); self.assertEqual(self.o.kills, [])
        self.probe.fail = False
        self.assertIsNone(g.poll()); self.assertEqual(g.fails['gpu'], 0)

    def test_ledger_unreachable_uses_cached_proof_and_keeps_receipt(self):
        g = self.g(); self.assertIsNone(g.poll())                                    # proof via ledger
        self.break_ledger()
        self.o.add(900, 90, 1, 'python other.py', uid=1019, cgroup='0::/user.slice'); self.o.gpu.append(900)
        self.assertIsNone(g.poll()); r = g.poll(); self.assertTrue(r)
        self.assertEqual(g.act(r), 'kept')
        self.assertIn((240, int(signal.SIGTERM)), self.o.kills); self.never(600, 700, 701, 900)
        self.assertIn(TT.RID, self.o.ledger)
        acts = list((self.t / 'state').glob('ACTION.*.json'))
        self.assertEqual(json.loads(acts[0].read_text())['proof'], 'cached')

    def test_ledger_unreachable_without_proof_never_signals(self):
        self.break_ledger()
        self.o.add(900, 90, 1, 'python other.py', uid=1019); self.o.gpu.append(900)
        self.assertEqual(self.g().run(max_polls=4), 'no_action'); self.assertEqual(self.o.kills, [])
        self.assertIn('unproven', self.unsafe()['reason'])

    def test_cached_proof_invalid_if_identity_changed(self):
        g = self.g(); g.poll()
        self.break_ledger()
        self.o.procs[220].start = 999                                               # launcher identity changed
        self.assertIsNone(g.poll()); self.assertEqual(self.o.kills, [])
        self.assertIn('unproven', self.unsafe()['reason'])

    def test_measurement_own_pids_only_and_cap_separate(self):
        g = self.g(); g.poll()
        self.probe.used[240] = 9500; g.poll()
        m = json.loads((self.t / 'state' / 'MEASURE.json').read_text())
        self.assertEqual((m['measured']['n_samples'], m['measured']['own_used_peak_mib'], m['measured']['own_used_last_mib']), (2, 9500, 9500))
        self.assertEqual(m['budget'], {'cap_mib': 12000, 'basis': 'provisional_cap', 'session': TT.SESSION})
        self.assertNotIn(20970, [s[1] for s in m['measured']['samples']])          # approved root never counted
        self.assertEqual([s[1] for s in m['measured']['samples']], [9000, 9500])

    def test_provisional_cap_is_enforced(self):
        self.probe.used[240] = 12001
        self.assertIsNone(self.g().poll())
        g = self.g(); g.poll(); r = g.poll()
        self.assertIn('own_mem breach', r)
        self.assertEqual(g.act(r), 'released'); self.never(600, 700, 701)

    def test_v1_kind_refused(self):
        with self.assertRaises(SystemExit):
            TG2.TrainGuardianV2(self.cfg, self.o, self.t / 'state', probe=self.probe)


if __name__ == '__main__':
    unittest.main()

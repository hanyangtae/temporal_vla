"""persist_guardian_v4 on the guardian_v4 fake fixture (FakeOS + fake NVML probe; no real signal/GPU/harness).
Covers: explicit config required, a new foreign process alone never stops our consumer and is never signalled, memory
protection (foreign present + free < reserve for confirm_polls) stops only ours with official release, low memory without
foreign is not acted on, NVML failure acts on nothing, adoption with a foreign present refused unless adopt_with_foreign,
watchdog still stops the consumer when this guardian is lost."""
import sys, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_guardian_handoff as TH          # noqa: E402  (fixture only, unchanged)
import persist_guardian_v4 as PG            # noqa: E402
import consumer_watchdog as CW              # noqa: E402

R1, R2 = TH.R1, TH.R2
PERSIST = {'foreign_policy': 'persist', 'persist_approval': '일단 돌렸으면 다른애 들어와도 뺐기지말고 계속 해', 'reserve_mib': 16384}


class PersistTest(TH.Base):
    def setUp(self):
        super().setUp()
        self.probe = TH.T2.GProbe(self.o); self.probe.free = 40000

    def pg(self, **kw):
        g = PG.PersistGuardianV4({**self.cfg, **PERSIST, **kw}, self.o, self.sd, probe=self.probe)
        self.gs.append(g); return g

    def steps(self, g, n=6):
        return [r for r in (g.step() for _ in range(n)) if r]

    def test_explicit_config_required(self):
        for bad in ({'foreign_policy': 'stop'}, {'persist_approval': ''}, {'reserve_mib': 0}, {'reserve_mib': '16384'},
                    {'sharing_authorization': 'x.json'}):
            with self.assertRaises(SystemExit):
                PG.PersistGuardianV4({**self.cfg, **PERSIST, **bad}, self.o, self.sd, probe=self.probe)

    def test_new_foreign_alone_never_stops_ours(self):
        g = self.pg(); pid = self.adopt(g)
        self.foreign()
        self.assertEqual(self.steps(g, 20), [])
        self.assertEqual(self.o.kills, []); self.assertIn(pid, self.o.procs); self.assertIn(R1, self.o.ledger)
        self.assertEqual([f[0] for f in g.foreign_now], [500])

    def test_memory_protection_stops_only_ours(self):
        g = self.pg(); pid = self.adopt(g)
        self.foreign(); self.probe.free = 16000
        self.assertEqual(self.steps(g, 3), ['released'])
        self.assertTrue(all(p not in self.o.procs for p in (pid, pid + 10, pid + 20)))
        self.assertNotIn(500, self.kills()); self.assertIn(500, self.o.procs)
        self.assertNotIn(R1, self.o.ledger)
        rel = [c for c in self.o.harness_calls if c[0] == 'release']
        self.assertEqual(rel, [('release', '--receipt', R1, '--session', TH.S)])

    def test_one_low_poll_is_not_enough(self):
        g = self.pg(); self.adopt(g); self.foreign()
        self.probe.free = 16000; self.assertIsNone(g.step())
        self.probe.free = 40000; self.assertEqual(self.steps(g, 5), []); self.assertEqual(self.o.kills, [])

    def test_low_memory_without_foreign_is_ours_not_acted(self):
        g = self.pg(); self.adopt(g)
        self.probe.free = 1000
        self.assertEqual(self.steps(g, 6), []); self.assertEqual(self.o.kills, [])

    def test_nvml_failure_acts_on_nothing(self):
        g = self.pg(); self.adopt(g); self.foreign(); self.probe.free = 1000
        self.probe.fail = True
        self.assertEqual(self.steps(g, 6), []); self.assertEqual(self.o.kills, [])
        self.assertFalse(g.last_ok)

    def test_adoption_with_foreign_present(self):
        g = self.pg(); g.step(); self.foreign(); g.step()
        pid = self.launch(100, R1)
        ok, why = self.req(g, pid, R1, timeout=5)
        self.assertFalse(ok); self.assertIn('not clean', why)
        g2 = None
        g.close(); self.gs.remove(g)
        g2 = self.pg(adopt_with_foreign=True); g2.step()
        pid2 = self.launch(200, R2)
        self.assertTrue(self.req(g2, pid2, R2)[0])
        self.assertEqual(self.o.kills, [])

    def test_watchdog_still_covers_lost_persist_guardian(self):
        g = self.pg(); pid = self.adopt(g)
        w = CW.Watchdog({**self.cfg, **PERSIST}, self.o, self.sd, pid, R1, self.last_nonce)
        try:
            self.assertIsNone(w.check()); self.assertTrue(w.armed)
            self.foreign()
            g.close(); self.o.remove(990)
            r = None
            for _ in range(10):
                r = w.check()
                if r:
                    break
                self.o.sleep(5)
            self.assertEqual(r, 'watchdog_released'); self.assertNotIn(500, self.kills())
        finally:
            w.close()


if __name__ == '__main__':
    unittest.main()

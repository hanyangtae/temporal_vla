"""train_guardian_v3 on the train_guardian fake host with a kind-V1 contract (one pinned foreign root, one training
session), plus: v2 contracts still accepted; fail-closed additions of v3 (contract validity every poll even with no
foreign process; NVML-listed PID with unreadable /proc counted as foreign). Fake host only, no real signal."""
import hashlib, json, signal, sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tests'), str(ROOT / 'trqam')]
import test_train_guardian as TT          # noqa: E402
import test_foreign_gpu_guardian_v2 as T2  # noqa: E402
import train_guardian_v3 as TG3          # noqa: E402


class V3V1Contract(TT.TrainGuardianTest):
    def setUp(self):
        super().setUp()
        o = self.o
        o.gpu = [240, 600]                                                    # pinned root + our trainer (no container)
        self.v2auth = dict(self.auth)
        self.auth = {'kind': 'groot_gpu_sharing_exception_v1', 'machine': 'srv50', 'hostname': 'worker2', 'gpu': 1,
                     'gpu_uuid': T2.UUID, 'session': TT.SESSION, 'experiment': 'fold27', 'allowed_kind': 'eval', 'max_serves': 1,
                     'per_serve_mib': 12000, 'per_serve_basis': 'measured_test', 'reserve_mib': 8192, 'expires_at': o.t + 3600,
                     'boot_id': o.bootv, 'prepared_at': o.t, 'approval': 'user one-time',
                     'foreign': [{'pid': 600, 'uid': 0, 'start_ticks': 60, 'cmdline': T2.APPROVED_CMD, 'max_used_mib': 22000,
                                  'cmdline_sha256': hashlib.sha256(T2.APPROVED_CMD.encode()).hexdigest()}]}
        self.ap.write_text(json.dumps(self.auth)); self.files[self.ap] = self.ap.read_bytes()
        self.v2p = self.t / 'auth_v2.json'; self.v2p.write_text(json.dumps(self.v2auth)); self.files[self.v2p] = self.v2p.read_bytes()

    def g(self, **kw):
        return TG3.TrainGuardianV3({**self.cfg, 'kind': TG3.KIND, **kw}, self.o, self.t / 'state', probe=self.probe)

    def unsafe(self):
        p = self.t / 'state' / 'ADMISSION_UNSAFE.json'
        return json.loads(p.read_text()) if p.exists() else None

    def test_v1_no_action_with_pinned_root(self):
        self.assertEqual(self.g().run(max_polls=4), 'no_action'); self.assertEqual(self.o.kills, [])
        self.assertIsNone(self.unsafe())

    def test_v1_new_foreign_stops_only_ours(self):
        self.o.add(900, 90, 1, 'python other.py', uid=1019); self.o.gpu.append(900)
        self.assertEqual(self.act(self.g()), 'released')
        self.never(600, 900); self.assertNotIn(220, self.signalled())
        self.assertNotIn(TT.RID, self.o.ledger)

    def test_v1_pinned_pid_reuse_is_foreign(self):
        self.o.procs[600].start = 61
        self.act(self.g()); self.never(600)

    def test_v1_pinned_over_max_used_is_foreign(self):
        self.probe.used[600] = 22001
        self.act(self.g()); self.never(600)

    def test_v1_container_member_not_approved(self):
        self.o.gpu.append(701)                                                # approved only by the old v2 container pin
        self.act(self.g()); self.never(600, 700, 701)

    def test_v1_own_memory_cap_and_measurement_basis(self):
        g = self.g(); g.poll()
        m = json.loads((self.t / 'state' / 'MEASURE.json').read_text())
        self.assertEqual(m['budget'], {'cap_mib': 12000, 'basis': 'measured_test', 'session': TT.SESSION})
        self.assertEqual(m['measured']['own_used_last_mib'], 9000)            # root 600 never counted
        self.probe.used[240] = 12001
        self.assertIsNone(g.poll()); r = g.poll(); self.assertIn('own_mem breach', r)
        self.assertEqual(g.act(r), 'released'); self.never(600)

    def test_v1_headroom(self):
        self.probe.free = 8191
        self.act(self.g()); self.never(600)

    def test_expired_contract_without_foreign_stops_ours(self):
        g = self.g(); self.assertIsNone(g.poll())
        self.o.t = self.auth['expires_at'] + 1
        self.o.gpu = [240]                                                     # foreign gone: nothing to classify
        self.assertIsNone(g.poll()); self.assertIn('no longer valid', self.unsafe()['reason'])
        r = g.poll(); self.assertIn('contract breach', r)
        self.assertEqual(g.act(r), 'released'); self.never(600)

    def test_boot_change_detected_without_foreign(self):
        g = self.g(); g.poll()
        self.auth['boot_id'] = 'other'; g.auth = dict(g.auth, boot_id='other')  # contract of another boot
        self.o.gpu = [240]
        g.poll(); r = g.poll(); self.assertIn('contract breach', r)

    def test_invalid_contract_idle_is_unsafe_only(self):
        self.o.ledger.clear(); self.o.t = self.auth['expires_at'] + 1
        self.assertEqual(self.g().run(max_polls=3), 'no_action'); self.assertEqual(self.o.kills, [])
        self.assertIn('no longer valid', self.unsafe()['reason'])

    def test_unreadable_listed_pid_counts_as_foreign(self):
        orig = self.probe.gpu

        def gpu(i):
            r = orig(i); r['apps'].append({'pid': 950, 'used_mib': 100}); return r   # NVML lists it, /proc has nothing
        self.probe.gpu = gpu
        r = self.act(self.g()); self.assertEqual(r, 'released'); self.never(600)

    def test_identity_query_exception_unsafe_then_stop_ours(self):
        g = self.g(); self.assertIsNone(g.poll())

        def boom(pid):
            raise PermissionError('proc unreadable')
        self.probe.proc = boom                                                  # pinned identity check raises
        self.assertIsNone(g.poll()); self.assertIn('raised', self.unsafe()['reason'])
        r = g.poll(); self.assertEqual(r, ['identity unverifiable'])
        self.assertEqual(g.act(r), 'released'); self.never(600)

    def test_identity_exception_idle_never_signals(self):
        self.o.ledger.clear()

        def boom():
            raise OSError('boot_id unreadable')
        self.probe.boot_id = boom
        self.assertEqual(self.g().run(max_polls=3), 'no_action'); self.assertEqual(self.o.kills, [])
        self.assertIn('raised', self.unsafe()['reason'])

    def test_single_identity_exception_recovers(self):
        g = self.g(); g.poll()
        orig = self.probe.proc
        self.probe.proc = lambda pid: (_ for _ in ()).throw(OSError('x'))
        self.assertIsNone(g.poll())
        self.probe.proc = orig
        self.assertIsNone(g.poll()); self.assertEqual(g.fails['identity'], 0); self.assertEqual(self.o.kills, [])

    def test_gpu_failure_while_training_unsafe_and_stop(self):
        self.probe.fail = True
        self.assertEqual(self.g().run(max_polls=4), 'kept')
        self.assertIn((240, int(signal.SIGTERM)), self.o.kills); self.never(600)

    def test_scope_mismatch_refused(self):
        for kw in ({'session': 'fold27-other'}, {'gpu': 2}, {'host': 'srv48'}):
            with self.assertRaises(SystemExit):
                self.g(**kw)
        with self.assertRaises(SystemExit):
            TG3.TrainGuardianV3({**self.cfg, 'kind': 'fold27_train_guardian_v2'}, self.o, self.t / 'state', probe=self.probe)

    def test_v2_contract_still_supported(self):
        self.o.gpu = [240, 600, 701]
        g = self.g(sharing_authorization=str(self.v2p))
        self.assertEqual(g.run(max_polls=4), 'no_action'); self.assertEqual(self.o.kills, [])


for _n in [n for n in dir(TT.TrainGuardianTest) if n.startswith('test')]:
    if getattr(V3V1Contract, _n) is getattr(TT.TrainGuardianTest, _n):
        setattr(V3V1Contract, _n, None)                                      # v1/v2 container fixtures not re-run here


if __name__ == '__main__':
    unittest.main()

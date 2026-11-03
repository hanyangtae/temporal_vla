"""foreign_gpu_guardian_v2: (a) without 'sharing_authorization' every v1 test passes unchanged on v2; (b) with the
one-time contract only the exactly pinned approved process is tolerated and NEVER signalled; extra foreign, PID reuse,
memory growth, expiry -> v1 stop procedure for OUR jobs only; low headroom stops our jobs; config scope mismatch refused.
Fake OS / fake NVML only (no signal reaches a real process)."""
import hashlib, json, signal, sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tests'), str(ROOT / 'eval/runtime'), str(ROOT.parents[2] / 'scripts/utils')]
import test_foreign_gpu_guardian as T     # noqa: E402  (v1 fixture + tests)
import foreign_gpu_guardian_v2 as G2      # noqa: E402

UUID = 'GPU-9fdab52c-2f3e-38e1-1fcd-855c290ef8c3'
APPROVED_CMD = '/workspace/isaaclab/_isaac_sim/kit/python/bin/python3 flow_policy_prev/flow.py'


class V1BehaviourOnV2(T.GuardianTest):
    """All v1 guardian tests, run against GuardianV2 with no sharing contract."""

    def g(self, **kw):
        return G2.GuardianV2({**self.cfg, **kw}, self.o, self.t / 'state', probe=None)


class GProbe:
    def __init__(self, o):
        self.o, self.free, self.used, self.uuid, self.fail = o, 60057, {600: 20970}, UUID, False

    def boot_id(self): return self.o.bootv
    def now(self): return self.o.t
    def hostname(self): return 'worker2'

    def proc(self, pid):
        p = self.o.procs.get(pid)
        return None if p is None else {'uid': p.uid, 'start_ticks': p.start, 'cmdline': p.cmd,
                                       'cmdline_sha256': hashlib.sha256(p.cmd.encode()).hexdigest()}

    def gpu(self, index):
        if self.fail:
            raise RuntimeError('nvidia-smi failed')
        return {'uuid': self.uuid, 'free_mib': self.free, 'total_mib': 81920,
                'apps': [{'pid': p, 'used_mib': self.used.get(p, 6118)} for p in self.o.gpu if p in self.o.procs]}


class SharingTest(T.GuardianTest):
    def setUp(self):
        super().setUp()
        self.o.add(600, 60, 1, APPROVED_CMD, uid=0)
        self.o.gpu.append(600)
        self.auth = {'kind': 'groot_gpu_sharing_exception_v1', 'machine': 'srv50', 'hostname': 'worker2', 'gpu': 1, 'gpu_uuid': UUID,
                     'session': T.SESSION, 'experiment': 'fold27', 'allowed_kind': 'eval', 'max_serves': 2, 'per_serve_mib': 6118,
                     'reserve_mib': 8192, 'expires_at': self.o.t + 3600, 'boot_id': self.o.bootv, 'prepared_at': self.o.t,
                     'approval': 'user one-time', 'foreign': [{'pid': 600, 'uid': 0, 'start_ticks': 60, 'cmdline': APPROVED_CMD,
                                                               'cmdline_sha256': hashlib.sha256(APPROVED_CMD.encode()).hexdigest(),
                                                               'max_used_mib': 22000}]}
        self.ap = self.t / 'state_auth.json'; self.ap.write_text(json.dumps(self.auth))
        self.files[self.ap] = self.ap.read_bytes()
        self.probe = GProbe(self.o)

    def g(self, **kw):
        return G2.GuardianV2({**self.cfg, 'sharing_authorization': str(self.ap), **kw}, self.o, self.t / 'state', probe=self.probe)

    def never(self, *pids):
        hit = {k[0] if not isinstance(k[0], tuple) else k[0][1] for k in self.o.kills}
        for p in pids:
            self.assertNotIn(p, hit); self.assertIn(p, self.o.procs)

    def run_to_act(self, g):
        g.poll(); conf = g.poll()
        self.assertTrue(conf)
        return g.act(conf)

    def test_approved_task_tolerated_no_action(self):
        self.assertEqual(self.g().run(max_polls=4), 'no_action')
        self.assertEqual(self.o.kills, []); self.never(600)
        ev = (self.t / 'state' / 'events.jsonl').read_text()
        self.assertIn('sharing_tolerated', ev)

    def test_extra_foreign_still_stops_own_jobs_only(self):
        self.foreign()                                               # 500 root + 501 same-uid appear
        self.assertEqual(self.run_to_act(self.g()), 'released')
        self.never(600, 500, 501)
        self.assertTrue(all(p not in self.o.procs for p in (120, 130, 140, 160)))

    def test_pid_reuse_is_foreign(self):
        self.o.procs[600].start = 61
        self.run_to_act(self.g()); self.never(600)

    def test_memory_growth_is_foreign(self):
        self.probe.used[600] = 22001
        self.run_to_act(self.g()); self.never(600)

    def test_expired_is_foreign(self):
        self.o.t = self.auth['expires_at'] + 1
        self.run_to_act(self.g()); self.never(600)

    def test_uuid_change_is_foreign(self):
        self.probe.uuid = 'GPU-other'
        self.run_to_act(self.g()); self.never(600)

    def test_low_headroom_stops_own_jobs(self):
        self.probe.free = 8191
        r = self.run_to_act(self.g())
        self.assertEqual(r, 'released'); self.never(600)
        self.assertIn('sharing_headroom_low', (self.t / 'state' / 'events.jsonl').read_text())

    def test_probe_failure_means_no_action(self):
        self.probe.fail = True
        self.assertEqual(self.g().run(max_polls=4), 'no_action'); self.assertEqual(self.o.kills, [])

    def test_scope_mismatch_refused(self):
        for k, v in (('gpu', 2), ('session', 'other'), ('machine', 'srv48')):
            bad = dict(self.auth); bad[k] = v
            if k == 'machine':
                bad['hostname'] = 'worker1'
            p = self.t / f'bad_{k}.json'; p.write_text(json.dumps(bad)); self.files[p] = p.read_bytes()
            with self.assertRaises(SystemExit):
                G2.GuardianV2({**self.cfg, 'sharing_authorization': str(p)}, self.o, self.t / 'state', probe=self.probe)

    def test_without_contract_approved_task_is_foreign(self):
        g = G2.GuardianV2(self.cfg, self.o, self.t / 'state', probe=self.probe)
        g.poll(); self.assertEqual(g.poll(), [(600, 60)])


class WaitResourcesSharedTest(unittest.TestCase):
    def test_default_is_unchanged_gpu_busy(self):
        import wait_resources_shared as W
        self.assertIs(W.busy_fn_for('srv50', 2, T.SESSION, None), W.EP.gpu_busy)

    def test_contract_free_only_when_verified(self):
        import tempfile, wait_resources_shared as W
        t = Path(tempfile.mkdtemp())
        o = T.FakeOS(str(t)); o.add(600, 60, 1, APPROVED_CMD, uid=0); o.gpu = [600]
        auth = {'kind': 'groot_gpu_sharing_exception_v1', 'machine': 'srv50', 'hostname': 'worker2', 'gpu': 3, 'gpu_uuid': UUID,
                'session': T.SESSION, 'experiment': 'fold27', 'allowed_kind': 'eval', 'max_serves': 2, 'per_serve_mib': 6118,
                'reserve_mib': 8192, 'expires_at': o.t + 3600, 'boot_id': o.bootv, 'prepared_at': o.t, 'approval': 'user one-time',
                'foreign': [{'pid': 600, 'uid': 0, 'start_ticks': 60, 'cmdline': APPROVED_CMD,
                             'cmdline_sha256': hashlib.sha256(APPROVED_CMD.encode()).hexdigest(), 'max_used_mib': 22000}]}
        p = t / 'auth.json'; p.write_text(json.dumps(auth))
        pr = GProbe(o)
        busy = W.busy_fn_for('srv50', 2, T.SESSION, str(p), probe=pr)
        self.assertEqual(busy([3]), [])                              # exactly the approved task: free for us
        self.assertEqual(busy([2]), [2])                             # other GPU: refused
        o.add(777, 77, 1, 'python other.py'); o.gpu.append(777)
        self.assertEqual(busy([3]), [3])                             # any other process: busy
        o.gpu.remove(777); pr.free = 100
        self.assertEqual(busy([3]), [3])                             # headroom: busy
        self.assertEqual(W.busy_fn_for('srv50', 3, T.SESSION, str(p), probe=GProbe(o))([3]), [3])   # serves > max_serves


if __name__ == '__main__':
    unittest.main()

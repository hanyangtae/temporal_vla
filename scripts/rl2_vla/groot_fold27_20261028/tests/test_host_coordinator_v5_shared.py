"""host_fold_coordinator_v5_shared on the v5 fixture (real fold spec + v2 queue freeze/DONE verification; fake harness
ledger, eval demand, spawn, clock) restricted to ONE contract GPU. Fake contract probe / guardian /proc only."""
import hashlib, json, sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tests'), str(ROOT / 'trqam'), str(ROOT), str(ROOT.parents[2] / 'scripts/utils')]
import test_host_coordinator_v5 as T5          # noqa: E402
import test_host_coordinator_v4 as T4          # noqa: E402
import host_fold_coordinator_v5 as H5          # noqa: E402
import host_fold_coordinator_v5_shared as HS   # noqa: E402

HC = H5.HC
ROOT_CMD = '/workspace/isaaclab/_isaac_sim/kit/python/bin/python3 flow_policy_prev/flow.py'


class SProbe:
    def __init__(self):
        self.boot, self.t, self.free, self.extra, self.fail = 'B', 0.0, 60000, [], False

    def boot_id(self): return self.boot
    def hostname(self): return 'kanu'
    def now(self): return self.t

    def proc(self, pid):
        if pid == 600:
            return {'uid': 0, 'start_ticks': 60, 'cmdline': ROOT_CMD, 'cmdline_sha256': hashlib.sha256(ROOT_CMD.encode()).hexdigest()}
        if pid in [e['pid'] for e in self.extra]:
            return {'uid': 1019, 'start_ticks': 1, 'cmdline': 'x', 'cmdline_sha256': 'x'}
        return None

    def status(self, pid): return None
    def cgroup(self, pid): return None

    def gpu(self, index):
        if self.fail:
            raise HS.SA.AuthError('nvidia-smi failed')
        return {'uuid': 'GPU-u', 'free_mib': self.free, 'total_mib': 81920, 'apps': [{'pid': 600, 'used_mib': 20970}, *self.extra]}


class GProbe:
    def __init__(self, cfg, sd):
        self.st, self.bootv = 77, 'B'
        self.argvv = ['python', 'scripts/rl2_vla/groot_fold27_20261028/trqam/train_guardian_v2.py', '--config', cfg, '--state-dir', sd]
        self.hook = None

    def boot(self): return self.bootv
    def start(self, pid): return self.st if pid == 4242 else None

    def argv(self, pid):
        if self.hook:
            self.hook()
        return list(self.argvv) if pid == 4242 else []

    def read_json(self, path):
        try:
            return json.loads(Path(path).read_text())
        except (OSError, ValueError):
            return None

    def exists(self, path): return Path(path).exists()


class SharedTest(T5.CoordV5Test):
    def setUp(self):
        super().setUp()
        self.auth = {'kind': 'groot_gpu_sharing_exception_v2', 'machine': 'kanu', 'hostname': 'kanu', 'gpu': 5, 'gpu_uuid': 'GPU-u',
                     'experiment': 'fold27', 'sessions': {'fold27-trqam-v2': {'allowed_kind': 'eval', 'max_serves': 1, 'per_serve_mib': 7168,
                                                                               'per_serve_basis': 'measured'}},
                     'reserve_mib': 16384, 'expires_at': 1e12, 'boot_id': 'B', 'prepared_at': 0, 'approval': 'user one-time',
                     'foreign': [{'pid': 600, 'uid': 0, 'start_ticks': 60, 'cmdline': ROOT_CMD, 'max_used_mib': 22000,
                                  'cmdline_sha256': hashlib.sha256(ROOT_CMD.encode()).hexdigest()}], 'containers': []}
        self.ap = self.t / 'contract.json'; self.ap.write_text(json.dumps(self.auth))
        self.gcfg = str(self.t / 'guardian.json'); self.gsd = self.t / 'gstate'; self.gsd.mkdir()
        self.gconf = {'kind': 'fold27_train_guardian_v2', 'host': 'kanu', 'gpu': 5, 'session': 'fold27-trqam-v2',
                      'sharing_authorization': str(self.ap), 'poll_s': 10, 'confirm_polls': 2}
        Path(self.gcfg).write_text(json.dumps(self.gconf))
        self.sp, self.gp, self.envs = SProbe(), GProbe(self.gcfg, str(self.gsd)), []

    def shared(self, demand=lambda: 0, harness=None, extra=()):
        self.clock = clock = [0.0]

        def sleep(s):
            clock[0] += s
            self.conc.append((clock[0], len([k for k, v in self.ledger.items() if v['session'] == 'fold27-trqam-v2'])))
        sp = self.spawn()

        def spawn(cmd, **kw):
            self.envs.append(kw['env']); return sp(cmd, **kw)
        argv = self.argv('--gpus', '5', *extra) + ['--eval-root', str(self.t / 'eval'), '--sharing-auth', str(self.ap),
                                                   '--train-guardian-pid', '4242', '--train-guardian-start', '77',
                                                   '--train-guardian-config', self.gcfg]
        co = HS.HostCoordinatorV5Shared(HS.parse(argv), popen=self.fake_popen(), run=None, spawn=spawn, demand=demand,
                                         harness=harness or (lambda: self.ledger), pending=lambda: (3, ''), sleep=sleep,
                                         now=lambda: clock[0], probe=self.sp, gprobe=self.gp)
        orig = co.run

        def go():
            try:
                return orig()
            finally:
                if co._lock:
                    co._lock.close(); co._lock = None
        co.run = go
        return co

    def todo(self):
        return [f for f in self.folds if f != T4.SMOKE]

    def test_default_v5_probe_unchanged(self):
        co = H5.HostCoordinatorV5(H5.parse(self.argv() + ['--eval-root', str(self.t / 'eval')]), popen=self.fake_popen(), run=None)
        self.assertIs(co.busy, H5.EP.gpu_busy)                                      # default deny path untouched

    def test_approved_sharing_trains_all_arms_on_the_one_gpu(self):
        res = self.shared().run()
        self.assertEqual(res['done_folds'][1:], self.todo())
        self.assertEqual(len(self.units), 3 * len(self.todo())); self.assertTrue(all(u[2] == 5 for u in self.units))
        self.assertLessEqual(max(c for _, c in self.conc), 1)                      # one arm at a time on one GPU
        self.assertTrue(all(e['HARNESS_SHARING_AUTH'] == str(self.ap) for e in self.envs))

    def refused(self, *extra):
        with self.assertRaises(HC.CoordError):
            self.shared(extra=('--poll-s', '60', *extra)).run()
        self.assertEqual(self.units, [])

    def test_new_foreign_pid_never_launches(self):
        self.sp.extra = [{'pid': 777, 'used_mib': 100}]; self.refused()

    def test_low_memory_never_launches(self):
        self.sp.free = 7168 + 16384 - 1; self.refused()

    def test_query_failure_never_launches(self):
        self.sp.fail = True; self.refused()

    def test_pid_reuse_of_approved_never_launches(self):
        orig = self.sp.proc
        self.sp.proc = lambda pid: {**orig(pid), 'start_ticks': 61} if pid == 600 else orig(pid)
        self.refused()

    def conf(self, **kw):
        Path(self.gcfg).write_text(json.dumps({**self.gconf, **kw}))

    def test_guardian_invalid_never_launches(self):
        muts = [('restarted', lambda: setattr(self.gp, 'st', 78)),                              # pid reused / restarted
                ('other_boot', lambda: setattr(self.gp, 'bootv', 'other')),                     # != contract boot_id
                ('observe_only_argv', lambda: self.gp.argvv.append('--observe-only')),
                ('not_train_guardian', lambda: self.gp.argvv.__setitem__(1, 'foreign_gpu_guardian_v3.py')),
                ('other_config_argv', lambda: self.gp.argvv.__setitem__(3, str(self.t / 'other.json'))),
                ('trainmem_session', lambda: self.conf(session='fold27-trainmem-srv50')),        # the measurement guardian
                ('gpu', lambda: self.conf(gpu=6)), ('host', lambda: self.conf(host='srv50')),
                ('measure_contract', lambda: self.conf(sharing_authorization=str(self.t / 'SHARING_MEASURE.json'))),
                ('v1_kind', lambda: self.conf(kind='fold27_train_guardian_v1')),
                ('observe_only_cfg', lambda: self.conf(observe_only=True)),
                ('alert', lambda: (self.gsd / 'ALERT.json').write_text('{}')),
                ('admission_unsafe', lambda: (self.gsd / 'ADMISSION_UNSAFE.json').write_text('{}'))]
        for name, m in muts:
            with self.subTest(mutation=name):
                self.tearDown(); self.setUp(); m()
                self.refused()

    def test_alert_mid_run_stops_new_admissions(self):
        def hook():
            if self.units:
                (self.gsd / 'ALERT.json').write_text('{}')
        self.gp.hook = hook
        with self.assertRaises(HC.CoordError):
            self.shared(extra=('--poll-s', '60')).run()
        self.assertEqual(len(self.units), 1)                                                 # nothing after the alert

    def test_boot_change_mid_run_stops_new_admissions(self):
        def hook():
            if self.units:
                self.gp.bootv = 'rebooted'
        self.gp.hook = hook
        with self.assertRaises(HC.CoordError):
            self.shared(extra=('--poll-s', '60')).run()
        self.assertEqual(len(self.units), 1)

    def test_eval_receipt_on_gpu_blocks_until_released(self):
        ev = {'e5': {'machine': 'kanu', 'gpus': [5], 'serves': 2, 'session': 'fold27-v2-eval', 'kind': 'eval'}}

        def harness():
            return {**(ev if self.clock[0] < 30 else {}), **self.ledger}
        res = self.shared(harness=harness, extra=('--poll-s', '1')).run()
        self.assertEqual(res['done_folds'][1:], self.todo())
        self.assertGreaterEqual(min(u[3] for u in self.units), 30)                  # no overlap with the eval receipt

    def test_scope_mismatch_refused(self):
        with self.assertRaises(HC.CoordError):
            self.shared(extra=('--gpus', '5', '6'))
        with self.assertRaises(HC.CoordError):
            self.shared(extra=('--session', 'fold27-other'))

    def test_done_arms_skipped(self):
        f0 = self.todo()[0]
        self.write(f0, 'basic', 'prod')
        self.shared().run()
        self.assertNotIn((f0, 'basic'), [(u[0], u[1]) for u in self.units])


for _n in [n for n in dir(T5.CoordV5Test) if n.startswith('test')]:
    if not hasattr(SharedTest, _n) or getattr(SharedTest, _n) is getattr(T5.CoordV5Test, _n):
        setattr(SharedTest, _n, None)                                                  # inherited 3-GPU tests not re-run here


if __name__ == '__main__':
    unittest.main()

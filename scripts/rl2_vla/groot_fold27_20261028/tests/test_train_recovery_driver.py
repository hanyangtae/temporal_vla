"""train_recovery_driver on a fake host (no real signal/process): waits read-only while eval runs; at the boundary stops
ONLY the two pinned identities, derives the contract live (real gpu_sharing_derive_v1 on the fake probe), starts the v3
guardian and the g3 coordinator with the old argv; refuses on failed eval, identity change, boundary loss, survivors,
guardian alert, re-run (O_EXCL intent)."""
import json, shutil, signal, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tests'), str(ROOT / 'trqam'), str(ROOT / 'eval/runtime'), str(ROOT.parents[2] / 'scripts/utils')]
import test_train_recovery_boundary as TB   # noqa: E402
import train_recovery_driver as RD          # noqa: E402

OLD_ARGV = ['.fold27_env/bin/python', '/tmp/x/trqam/host_fold_coordinator_v5_shared.py', '--machine', 'srv50', '--gpus', '3',
            '--session', 'fold27-t49-srv50-candle-v2', '--out', 'outputs/rl2_vla/candle', '--eval-root', 'E',
            '--sharing-auth', 'OLD.json', '--train-guardian-pid', '627282', '--train-guardian-start', '33',
            '--train-guardian-config', 'old_g.json']
FOLDS = ['PPCC-bread_s4_j1', 'PPCC-bread_s4_j3', 'PPCC-bread_s4_j4']
ARMS = ('mil_qam65', 'mil_trqam_basic', 'mil_trqam_mix', 'mil_trqam_pbrs')


class Host(TB.Host):
    def __init__(self, probe, t):
        super().__init__(probe)
        self.kills, self.spawned, self.uids, self.shas, self.clock, self.survive = [], [], {631152: 1000, 627282: 1000}, \
            {631152: 'c' * 64, 627282: 'g' * 64}, 0.0, set()
        self.gsd_alert = False

    def uid(self, pid): return self.uids.get(pid) if pid in self.starts else None
    def cmd_sha(self, pid): return self.shas.get(pid) if pid in self.starts else None
    def now(self): return self.clock

    def kill(self, pid, sig):
        self.kills.append((pid, sig))
        if pid not in self.survive:
            self.starts.pop(pid, None)

    def spawn(self, argv, log, env):
        pid = 50000 + len(self.spawned); self.spawned.append((argv, env)); self.starts[pid] = 900 + len(self.spawned)
        if self.gsd_alert and 'train_guardian_v3.py' in argv[1]:
            gsd = Path(argv[argv.index('--state-dir') + 1]); gsd.mkdir(parents=True); (gsd / 'ALERT.json').write_text('{}')
        return pid

    def exists(self, p): return Path(p).exists()


class DriverTest(TB.Boundary):
    def setUp(self):
        super().setUp()
        self.h = Host(self.p, self.t)
        self.h.led = {'E' * 32: {'machine': 'srv50', 'gpus': [3], 'session': 'fold27-v2-eval', 'kind': 'eval'}}
        self.qs = self.t / 'QUEUE_STATE.gpu3.json'
        self.cfg = {'kind': RD.KIND, 'state_dir': str(self.t / 'driver'), 'boot_id': TB.BOOT, 'eval_queue_state': str(self.qs),
                    'boundary': dict(machine='srv50', gpu=3, boot_id=TB.BOOT, session='fold27-t49-srv50-candle-v2',
                                     contract=str(self.ap), eval_consumer='3088084:11', old_coordinator='631152:22',
                                     old_guardian='627282:33', old_out=str(self.out)),
                    'old_coordinator': {'pid': 631152, 'start': 22, 'uid': 1000, 'cmdline_sha256': 'c' * 64, 'argv': OLD_ARGV},
                    'old_guardian': {'pid': 627282, 'start': 33, 'uid': 1000, 'cmdline_sha256': 'g' * 64},
                    'derive': dict(source=self.ap, session='fold27-t49-srv50-candle-v2', max_serves=1, per_serve_mib=7168,
                                   per_serve_basis='measured_4584_x1.5_rounded_7168', measured_peak_mib=4584, reserve_mib=16384,
                                   expires_in_h=24.0, expect_pid=2795788, expect_uid=1019, expect_start_ticks=1377893575,
                                   expect_cmdline_sha256=TB.SHA, expect_boot_id=TB.BOOT, expect_gpu_uuid=TB.UUID, approval='user'),
                    'guardian': {'python': 'py', 'config': {'host': 'srv50', 'gpu': 3, 'session': 'fold27-t49-srv50-candle-v2',
                                                            'confirm_polls': 2, 'poll_s': 10}},
                    'eval_complete': {'root': str(self.t / 'evalroot'), 'host': 'srv50', 'rows_per_arm': 40,
                                      'expected': {f: list(ARMS) for f in FOLDS}},
                    'poll_s': 60, 'term_wait_s': 10, 'settle_s': 1}
        self.sleeps, self.done_rows, self.audit_bad, self.extra_fold = [], 351, False, False
        self.settle_hook, self.rc_tamper = None, None

    def snap(self, root, host):                       # audited snapshot: 3 folds x 4 arms x 40 + real RUN_COMPLETE files
        folds, left = [], self.done_rows
        for fid in FOLDS + (['PPCC-bread_s4_j9'] if self.extra_fold else []):
            arms = {}
            for arm in ARMS:
                n = max(0, min(40, left)); left -= n
                d = Path(root) / fid / arm; d.mkdir(parents=True, exist_ok=True)
                (d / 'progress.jsonl').write_text('x\n' * n)
                if n == 40 and not (d / 'RUN_COMPLETE.json').exists():
                    (d / 'RUN_COMPLETE.json').write_text(json.dumps({'fold_id': fid, 'arm': arm, 'rows': 40,
                                                                     'progress_sha256': RD.Driver.file_sha(d / 'progress.jsonl')}))
                arms[arm] = {'load_error': None, 'audit_errors': ['n3: gate mismatch'] if self.audit_bad and fid == FOLDS[0] else [],
                             'rows': [{'audit_ok': True}] * n, 'run_complete': n == 40}
            folds.append({'fold_id': fid, 'planned_rows': 40, 'arms': arms})
        if self.rc_tamper:
            self.rc_tamper(Path(root))
        return {'folds': folds}

    def drv(self):
        def sleep(s):
            self.sleeps.append(s); self.h.clock += s
            if len(self.sleeps) == 2:                                          # eval finishes during the wait
                self.done_rows = 480
                self.eval_done(); self.qs.write_text(json.dumps({'exit': 'no runnable job', 'failures_this_run': []}))
            if self.settle_hook:
                self.settle_hook()
        return RD.Driver(self.cfg, self.h, sleep=sleep, eval_snapshot=self.snap)

    def outcome(self):
        d = Path(self.cfg['state_dir'])
        return {n: json.loads((d / n).read_text()) for n in ('DONE.json', 'REFUSED.json') if (d / n).exists()}

    def test_waits_then_replaces_only_ours(self):
        self.assertEqual(self.drv().run(), 0)
        self.assertEqual(self.h.kills, [(631152, signal.SIGTERM), (627282, signal.SIGTERM)])     # nothing else signalled
        (g_argv, _), (c_argv, c_env) = self.h.spawned
        self.assertTrue(g_argv[1].endswith('train_guardian_v3.py'))
        gcfg = json.loads(Path(g_argv[g_argv.index('--config') + 1]).read_text())
        self.assertEqual(gcfg['kind'], 'fold27_train_guardian_v3')
        self.assertTrue(c_argv[1].endswith('host_fold_coordinator_v5_shared_g3.py'))
        new = Path(self.cfg['state_dir']) / 'SHARING_TRAIN.json'
        self.assertEqual(c_argv[c_argv.index('--sharing-auth') + 1], str(new))
        gc = str(Path(self.cfg['state_dir']) / 'guardian_v3.json')
        want = list(OLD_ARGV)
        want[1] = str(RD.HERE / 'host_fold_coordinator_v5_shared_g3.py')
        for k, v in (('--sharing-auth', str(new)), ('--train-guardian-pid', '50000'), ('--train-guardian-start', '901'),
                     ('--train-guardian-config', gc)):
            want[want.index(k) + 1] = v
        self.assertEqual(c_argv, want)                                          # old argv otherwise byte-identical
        self.assertEqual(c_env['HARNESS_SHARING_AUTH'], str(new))
        self.assertEqual(json.loads(new.read_text())['per_serve_mib'], 7168)
        self.assertIn('DONE.json', self.outcome())
        self.assertGreaterEqual(len(self.sleeps), 2)                           # really waited while eval ran

    def test_failed_eval_refuses_without_signals(self):
        def sleep(s):
            self.sleeps.append(s)
            self.eval_done(); self.qs.write_text(json.dumps({'exit': 'no runnable job', 'failures_this_run': [{'job': ['F', 'a']}]}))
        self.assertEqual(RD.Driver(self.cfg, self.h, sleep=sleep, eval_snapshot=self.snap).run(), 2)
        self.assertEqual(self.h.kills, []); self.assertEqual(self.h.spawned, []); self.assertIn('REFUSED.json', self.outcome())

    def test_incomplete_eval_keeps_waiting_even_if_exit_record_says_done(self):
        self.eval_done(); self.qs.write_text(json.dumps({'exit': 'no runnable job', 'failures_this_run': []}))
        self.done_rows = 479                                                   # e.g. one row missing/skipped
        calls = []
        Path(self.cfg['state_dir']).mkdir(parents=True)

        def sleep(s):
            calls.append(s)
            if len(calls) == 3:
                raise KeyboardInterrupt                                        # stop the test loop
        with self.assertRaises(KeyboardInterrupt):
            RD.Driver(self.cfg, self.h, sleep=sleep, eval_snapshot=self.snap)._run()
        self.assertEqual(self.h.kills, [])

    def test_other_coordinate_set_refused(self):
        self.extra_fold = True                                                 # 4 folds: never "480 by another layout"
        self.assertEqual(self.drv().run(), 2); self.assertEqual(self.h.kills, [])
        self.assertIn('pinned set', self.outcome()['REFUSED.json']['reason'])

    def test_run_complete_not_binding_progress_refused(self):
        def tamper(root):
            p = root / FOLDS[1] / ARMS[2] / 'RUN_COMPLETE.json'
            if p.exists():
                b = json.loads(p.read_text()); b['progress_sha256'] = '0' * 64; p.write_text(json.dumps(b))
        self.rc_tamper = tamper
        self.assertEqual(self.drv().run(), 2); self.assertEqual(self.h.kills, [])
        self.assertIn('RUN_COMPLETE', self.outcome()['REFUSED.json']['reason'])

    def test_audit_failure_refuses_without_signals(self):
        self.audit_bad = True
        self.assertEqual(self.drv().run(), 2)
        self.assertEqual(self.h.kills, []); self.assertIn('audit', self.outcome()['REFUSED.json']['reason'])

    def test_guardian_pid_reused_by_other_identity_refuses(self):
        self.h.shas[627282] = 'z' * 64                                         # same pid alive, other command
        self.assertEqual(self.drv().run(), 2); self.assertEqual(self.h.kills, [])

    def test_spawned_guardian_replaced_during_settle_refuses(self):
        def hook():
            if self.h.spawned and 50000 in self.h.starts:
                self.h.starts[50000] = 7777                                    # pid reused during settle
        self.settle_hook = hook
        self.assertEqual(self.drv().run(), 2); self.assertEqual(len(self.h.spawned), 1)

    def test_derive_error_recorded_as_refusal(self):
        self.cfg['derive']['expect_start_ticks'] = 1                           # live re-proof fails (AuthError)
        self.assertEqual(self.drv().run(), 2)
        self.assertIn('AuthError', self.outcome()['REFUSED.json']['reason']); self.assertEqual(self.h.spawned, [])

    def test_coordinator_identity_change_refuses(self):
        self.h.shas[631152] = 'd' * 64                                          # PID reused by another command
        self.assertEqual(self.drv().run(), 2); self.assertEqual(self.h.kills, [])

    def test_guardian_already_gone_is_allowed(self):
        del self.h.starts[627282]
        self.assertEqual(self.drv().run(), 0); self.assertEqual(self.h.kills, [(631152, signal.SIGTERM)])

    def test_survivor_refuses_no_sigkill(self):
        self.h.survive.add(631152)
        self.assertEqual(self.drv().run(), 2)
        self.assertNotIn(signal.SIGKILL, [s for _, s in self.h.kills]); self.assertEqual(self.h.spawned, [])

    def test_guardian_alert_refuses_before_coordinator(self):
        self.h.gsd_alert = True
        self.assertEqual(self.drv().run(), 2); self.assertEqual(len(self.h.spawned), 1)   # coordinator never spawned

    def test_rerun_refused(self):
        self.assertEqual(self.drv().run(), 0)
        with self.assertRaises(FileExistsError):
            self.drv().run()

    def test_argv_without_shared_flags_refused(self):
        self.cfg['old_coordinator']['argv'] = OLD_ARGV[:-8]
        self.assertEqual(self.drv().run(), 2)
        self.assertIn('lacks', self.outcome()['REFUSED.json']['reason'])


import train_recovery_config as RC          # noqa: E402


class CfgHost:
    def __init__(self):
        self.starts = {631152: 1376533122, 627282: 1376530452, 3088084: 555}
        self.argvs = {631152: list(OLD_ARGV), 627282: ['py', 'trqam/train_guardian_v2.py', '--config', 'g.json', '--state-dir', 's']}
        self.cwds, self.bootv, self.uids = {631152: str(RD.REPO.resolve())}, TB.BOOT, {}

    def boot(self): return self.bootv
    def start(self, pid): return self.starts.get(pid)
    def argv(self, pid): return self.argvs.get(pid)
    def uid(self, pid): return self.uids.get(pid, 1000)
    def cmd_sha(self, pid): return f'{pid:064d}'
    def cwd(self, pid): return self.cwds.get(pid)


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.t = Path(tempfile.mkdtemp()); self.h = CfgHost()
        self.args = ['--out', str(self.t / 'driver.json'), '--state-dir', str(self.t / 'state'), '--machine', 'srv50', '--gpu', '3',
                     '--session', 'fold27-t49-srv50-candle-v2', '--boot-id', TB.BOOT, '--old-coordinator', '631152:1376533122',
                     '--old-guardian', '627282:1376530452', '--eval-consumer', '3088084:555', '--old-out', 'outputs/rl2_vla/candle',
                     '--eval-root', 'E', '--expect-owner-uid', '1000',
                     '--eval-queue-state', 'Q', '--eval-folds', *FOLDS, '--eval-arms', *ARMS, '--rows-per-arm', '40',
                     '--contract-source', 'SHARING_PID_1735.json', '--foreign-pid', '2795788', '--foreign-uid', '1019',
                     '--foreign-start', '1377893575', '--foreign-cmdline-sha256', TB.SHA, '--gpu-uuid', TB.UUID,
                     '--per-serve-mib', '7168', '--per-serve-basis', 'measured_4584_x1.5_rounded_7168', '--measured-peak-mib', '4584',
                     '--reserve-mib', '16384', '--expires-in-h', '24', '--approval', 'user', '--guardian-python', 'py']

    def tearDown(self):
        shutil.rmtree(self.t)

    def test_builds_pinned_config(self):
        OLD = list(OLD_ARGV); OLD[OLD.index('--gpus') + 1] = '3'; self.h.argvs[631152] = OLD
        self.assertEqual(RC.main(self.args, host=self.h), 0)
        c = json.loads((self.t / 'driver.json').read_text())
        self.assertEqual((c['old_coordinator']['start'], c['old_coordinator']['argv']), (1376533122, OLD))
        self.assertEqual(c['old_guardian']['cmdline_sha256'], f'{627282:064d}')
        self.assertEqual(c['eval_complete']['expected'], {f: list(ARMS) for f in FOLDS})
        self.assertEqual((c['derive']['per_serve_mib'], c['derive']['reserve_mib'], c['derive']['measured_peak_mib']), (7168, 16384, 4584))
        RD.Driver(c, self.h)                                                   # schema accepted by the driver
        self.assertEqual(RC.main(self.args, host=self.h), 2)                   # never overwritten

    def test_refusals_write_nothing(self):
        OLD = list(OLD_ARGV); OLD[OLD.index('--gpus') + 1] = '3'
        muts = [lambda: self.h.starts.__setitem__(631152, 1),                  # PID reuse / other start
                lambda: self.h.cwds.__setitem__(631152, '/elsewhere'),
                lambda: self.h.argvs.__setitem__(631152, [x.replace('candle', 'other') for x in OLD]),
                lambda: self.h.argvs.__setitem__(627282, ['py', 'train_guardian.py']),
                lambda: self.h.starts.pop(3088084),                            # eval consumer must be live now
                lambda: setattr(self.h, 'bootv', 'other'),
                lambda: self.h.uids.__setitem__(631152, 1019),                 # another user's process
                lambda: self.h.uids.__setitem__(627282, None),                 # unreadable uid
                lambda: self.h.argvs.__setitem__(631152, [x if x != 'E' else 'E2' for x in OLD]),          # other eval root
                lambda: self.h.argvs.__setitem__(631152, [x if x != 'outputs/rl2_vla/candle' else 'outputs/rl2_vla/jug'
                                                          for x in OLD])]                                # other experiment out
        for m in muts:
            self.h = CfgHost(); self.h.argvs[631152] = list(OLD); m()
            self.assertEqual(RC.main(self.args, host=self.h), 2)
            self.assertFalse((self.t / 'driver.json').exists())


for _n in [n for n in dir(TB.Boundary) if n.startswith('test')]:
    setattr(DriverTest, _n, None)                                              # boundary tests run in their own module


if __name__ == '__main__':
    unittest.main()

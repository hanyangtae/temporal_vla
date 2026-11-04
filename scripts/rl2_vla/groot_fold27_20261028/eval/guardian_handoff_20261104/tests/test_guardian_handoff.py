"""guardian_v4 + handoff_gate + eval_gpu_queue_v4.sh on the v1 fake process table (CPU only: no signal reaches a real
process, no nvidia-smi, no harness, no GPU). The launcher side (handoff_gate.request) runs in-line; its sleep advances the
guardian loop, so guardian and launcher interleave deterministically.

Regression for the srv50 GPU3 restart gap: v1 follower marks miss eval_gpu_queue_v3.sh; v1 run() returns after act();
a relaunched consumer (rc 241 retry) must not run unless adopted. Plus: PID reuse, new foreign, low VRAM headroom,
old receipt returned, duplicate guardian / launcher, stale OWNED / state files, guardian / launcher crash, HOLD."""
import hashlib, json, os, shutil, signal, subprocess, sys, tempfile, unittest
from pathlib import Path

HD = Path(__file__).resolve().parents[1]
F27 = HD.parents[1]
sys.path[:0] = [str(HD), str(F27 / 'tests'), str(F27 / 'eval/runtime'), str(F27.parents[2] / 'scripts/utils')]
import test_foreign_gpu_guardian as T     # noqa: E402  (v1 fake OS, unchanged)
import test_foreign_gpu_guardian_v2 as T2  # noqa: E402  (fake NVML probe, unchanged)
import foreign_gpu_guardian as G1          # noqa: E402
import guardian_v4 as G4                   # noqa: E402
import handoff_gate as HG                  # noqa: E402
import consumer_watchdog as CW             # noqa: E402

S, R1, R2 = T.SESSION, 'A' * 32, 'B' * 32
OUTER = 'bash scripts/rl2_vla/groot_fold27_20261028/eval/guardian_handoff_20261104/eval_gpu_queue_v4.sh'
INNER = OUTER + ' --inner'
CONSUMER = '.venvs/rl2_iid/bin/python -u eval/runtime/eval_gpu_queue_v2.py --pool 1 run --min-jobs 1'


class FakeOS4(T.FakeOS):
    me = {'pid': 990, 'start': 99}

    def self_ident(self):
        return dict(self.me)


class Base(unittest.TestCase):
    def setUp(self):
        self.t = Path(tempfile.mkdtemp())
        self.q = self.t / 'queue/srv50'; (self.q / 'consumers').mkdir(parents=True); (self.q / 'claims').mkdir()
        (self.q / 'claims/fold.arm.claim').write_text('{"claim": 1}')
        o = self.o = FakeOS4(str(self.t))
        o.container = None
        o.ledger = {}
        o.add(990, 99, 1, 'python guardian_v4.py --config c.json --state-dir st')
        self.cfg = {'kind': G1.KIND, 'host': 'srv50', 'gpu': 1, 'session': S, 'queue_dir': str(self.q), 'deploy': str(self.t),
                    'poll_s': 5, 'confirm_polls': 2, 'grace_s': 20, 'kill_wait_s': 10, 'release_wait_s': 30}
        self.sd = self.t / 'state'
        self.files = {p: p.read_bytes() for p in self.t.rglob('*') if p.is_file()}
        self.gs = []

    def tearDown(self):
        for g in self.gs:
            g.close()
        for p, b in self.files.items():
            assert p.read_bytes() == b, f'modified {p}'
        shutil.rmtree(self.t)

    def g(self, probe=None, **kw):
        g = G4.GuardianV4({**self.cfg, **kw}, self.o, self.sd, probe=probe)
        self.gs.append(g); return g

    # ---- launcher tree helpers (pids: outer o, harness o+10, inner/consumer o+20, arm o+30, serve o+40)
    def launch(self, base, receipt, outer=True):
        o = self.o
        if outer and base not in o.procs:
            o.add(base, base // 10, 1, OUTER)
        o.add(base + 10, base // 10 + 1, base if outer else 1, f'python3 scripts/utils/groot_harness.py run --gpus 1 -- bash {INNER}')
        o.add(base + 20, base // 10 + 2, base + 10, INNER)
        o.ledger[receipt] = {'machine': 'srv50', 'gpus': [1], 'serves': 6, 'session': S, 'kind': 'eval', 'consumer_pid': base + 20}
        return base + 20

    def exec_consumer(self, pid, receipt, register=True):
        """what happens after an ack: exec (same pid) + registration + arm/serve on the GPU"""
        o = self.o
        o.procs[pid].cmd = CONSUMER
        if register:
            (self.q / 'consumers/gpu1.json').write_text(json.dumps({'gpu': 1, 'receipt': receipt, 'boot_id': 'BOOT',
                                                                    'consumer': {'pid': pid, 'start': o.procs[pid].start}}))
        o.add(pid + 10, o.procs[pid].start + 1, pid, 'python -u eval/runtime/run_eval_arm_v2.py --gpu 1')
        o.add(pid + 20, o.procs[pid].start + 2, pid + 10, 'python eval/runtime/serve_mil_qam.py --port 8996')
        o.gpu += [pid + 20]

    def watchdog_proc(self, pid, receipt, nonce, ppid=1):
        wp = 8000 + pid
        self.o.add(wp, wp // 10, ppid, f'python consumer_watchdog.py --config c.json --state-dir st --consumer-pid {pid} '
                                       f'--receipt {receipt} --nonce {nonce}')
        return wp

    def req(self, g, pid, receipt, timeout=30, nonce=None, watchdog=True):
        nonce = nonce or f'n{pid}x{len(self.o.kills)}{int(self.o.t)}'
        wp = self.watchdog_proc(pid, receipt, nonce) if watchdog else None
        self.last_nonce = nonce
        return HG.request(self.sd, 'srv50', 1, S, receipt, pid, timeout, now_fn=lambda: self.o.t,
                          sleep_fn=lambda s: (self.o.sleep(1), g.step()), start_fn=self.o.start, boot_fn=lambda: self.o.bootv,
                          nonce=nonce, watchdog_pid=wp)

    def ready(self):
        return HG.check_ready(self.sd, 'srv50', 1, S, self.o.t, self.o.start, lambda: self.o.bootv)

    def foreign(self, pid=500, start=50):
        self.o.add(pid, start, 1, 'python someone_elses_job.py', uid=0); self.o.gpu.append(pid)

    def kills(self):
        return {k[0] if not isinstance(k[0], tuple) else k[0][1] for k in self.o.kills}

    def adopt(self, g, base=100, receipt=R1):
        g.step()                                   # guardian loop is running before any launch
        pid = self.launch(base, receipt)
        ok, info = self.req(g, pid, receipt)
        self.assertTrue(ok, info)
        self.exec_consumer(pid, receipt)
        g.step()
        return pid

    def act_on_foreign(self, g):
        self.foreign()
        res = [g.step() for _ in range(3)]
        return [r for r in res if r]


class CauseTest(Base):
    def test_v1_follower_mark_misses_v3_and_v4_finds_outer_loop(self):
        v3 = 'bash scripts/rl2_vla/groot_fold27_20261028/eval/runtime/eval_gpu_queue_v3.sh'
        self.assertFalse(any(m in v3 for m in G1.FOLLOWER_MARK))             # the original gap
        self.o.add(100, 10, 1, v3); self.o.add(110, 11, 100, 'python3 scripts/utils/groot_harness.py run -- bash x --inner')
        self.o.add(120, 12, 110, CONSUMER)
        cons = {'pid': 120, 'start': 12}
        self.assertIsNone(G1.Guardian(self.cfg, self.o, self.t / 'v1state').follower(cons))
        self.assertEqual(self.g().follower(cons), (100, 10))

    def test_v1_run_returns_after_one_action_v4_keeps_guarding(self):
        g = self.g(); self.adopt(g)
        self.assertEqual(self.act_on_foreign(g), ['released'])
        self.assertIsNone(self.ready())                                     # still alive, heartbeat fresh, not on hold
        self.assertIsNone(g.adopted); self.assertIn(R1, g.retired)


class HandoffTest(Base):
    def test_adopted_consumer_is_protected_and_follower_stopped(self):
        g = self.g(); pid = self.adopt(g)
        self.assertEqual(g.adopted['pid'], pid)
        owned = {r for (_, _), r in g.owned.items()}
        self.assertTrue({'consumer', 'arm', 'serve'} <= owned)
        self.act_on_foreign(g)
        self.assertNotIn(500, self.kills()); self.assertIn(500, self.o.procs)       # foreign never signalled
        for p in (100, 110, pid, pid + 10, pid + 20):
            self.assertNotIn(p, self.o.procs)                                     # incl. the v4 outer loop (no relaunch)
        self.assertIn((100, int(signal.SIGSTOP)), self.o.kills)

    def test_relaunch_after_rc241_needs_clean_gpu_and_new_handoff(self):
        g = self.g(); self.adopt(g)
        g.c = {**g.c, 'follower': None}
        self.act_on_foreign(g)
        pid2 = self.launch(200, R2)                                          # outer loop retried (rc 241) while foreign present
        ok, why = self.req(g, pid2, R2, timeout=5)
        self.assertFalse(ok); self.assertIn('not clean', why)
        self.assertFalse(list((self.sd / 'handoff').glob('request.*')))    # gave up -> request withdrawn
        self.o.remove(500); g.step()                                         # foreign left
        ok, info = self.req(g, pid2, R2)
        self.assertTrue(ok, info); self.assertEqual(g.adopted['receipt'], R2)

    def test_unadopted_consumer_registration_sets_hold(self):
        g = self.g(); self.adopt(g)
        pid2 = self.launch(200, R2); self.exec_consumer(pid2, R2)           # e.g. an old v3 loop: no handoff
        g.step()
        self.assertIsNotNone(g.hold); self.assertIn('HOLD', self.ready())
        self.assertNotIn(pid2, {p for (p, _) in g.owned})

    def test_old_receipt_returned_is_refused(self):
        g = self.g(); self.adopt(g)
        self.act_on_foreign(g); self.o.remove(500); g.step()
        pid2 = self.launch(200, R1)                                          # same (old) receipt comes back
        ok, why = self.req(g, pid2, R1)
        self.assertFalse(ok); self.assertIn('retired', why)

    def test_receipt_must_be_in_ledger_with_matching_consumer_pid(self):
        g = self.g(); g.step()
        pid = self.launch(100, R1); self.o.ledger[R1]['consumer_pid'] = 999
        ok, why = self.req(g, pid, R1); self.assertFalse(ok); self.assertIn('consumer_pid', why)
        del self.o.ledger[R1]
        ok, why = self.req(g, pid, R1); self.assertFalse(ok); self.assertIn('not in harness ledger', why)
        self.o.ledger[R1] = {'machine': 'srv50', 'gpus': [1, 2], 'session': S, 'consumer_pid': pid}
        ok, why = self.req(g, pid, R1); self.assertFalse(ok); self.assertIn('GPU [1]', why)

    def test_pid_reuse(self):
        g = self.g(); g.step()
        pid = self.launch(100, R1)
        start_seen = self.o.start(pid)
        self.o.procs[pid].start = start_seen + 1                              # pid reused between request and judge
        ok, why = HG.request(self.sd, 'srv50', 1, S, R1, pid, 5, now_fn=lambda: self.o.t,
                             sleep_fn=lambda s: (self.o.sleep(1), g.step()),
                             start_fn=lambda p: start_seen if p == pid else self.o.start(p), boot_fn=lambda: 'BOOT')
        self.assertFalse(ok); self.assertIn('PID reuse', why)
        # adopted consumer exits, its pid is reused by a foreign process -> foreign, never owned
        pid = self.launch(300, R2); self.assertTrue(self.req(g, pid, R2)[0])
        self.o.remove(pid); g.step(); self.assertIsNone(g.adopted)
        self.o.add(pid, 777, 1, 'python someone_elses_job.py', uid=0); self.o.gpu.append(pid)
        g.step(); g.step()
        self.assertNotIn(pid, self.kills()); self.assertIn((pid, 777), g.seen)

    def test_new_foreign_blocks_adoption(self):
        g = self.g(); g.step(); self.foreign(); g.step()
        pid = self.launch(100, R1)
        ok, why = self.req(g, pid, R1, timeout=5); self.assertFalse(ok); self.assertIn('not clean', why)

    def test_requester_must_be_v4_inner_under_harness(self):
        g = self.g(); g.step()
        pid = self.launch(100, R1); self.o.procs[pid].cmd = 'bash eval_gpu_queue_v3.sh --inner'
        self.assertIn('not eval_gpu_queue_v4.sh', self.req(g, pid, R1)[1])
        self.o.procs[pid].cmd = INNER; self.o.procs[pid].ppid = 100
        self.assertIn('harness', self.req(g, pid, R1)[1])

    def test_duplicate_launcher_refused(self):
        g = self.g(); self.adopt(g)
        pid2 = self.launch(200, R2)
        ok, why = self.req(g, pid2, R2); self.assertFalse(ok); self.assertIn('duplicate launcher', why)

    def test_duplicate_guardian_refused(self):
        g = self.g()
        with self.assertRaises(SystemExit):
            G4.GuardianV4(self.cfg, self.o, self.t / 'other_state', take_lock=True)
        g.close(); self.g()                                                   # lock released with the process

    def test_stale_state_files_not_trusted(self):
        (self.sd / 'handoff').mkdir(parents=True)
        (self.sd / 'OWNED.json').write_text(json.dumps({'consumer': {'pid': 500, 'start': 50, 'receipt': R1},
                                                         'owned': [[500, 50, 'consumer']]}))
        (self.sd / 'handoff/ack.old.json').write_text(json.dumps({'accepted': True, 'pid': 500}))
        self.foreign()
        g = self.g(); g.step(); g.step()
        self.assertIsNone(g.adopted); self.assertIn((500, 50), g.seen); self.assertNotIn(500, self.kills())
        # a request left behind by a crashed launcher (pid gone) is refused, not adopted
        self.o.remove(500); g.step()
        (self.sd / 'handoff/request.dead.json').write_text(json.dumps({'nonce': 'dead', 'host': 'srv50', 'gpu': 1, 'session': S,
                                                                       'receipt': R1, 'pid': 4242, 'start': 1, 'boot_id': 'BOOT'}))
        g.step()
        self.assertIsNone(g.adopted)
        self.assertIn('not alive', json.loads((self.sd / 'handoff/nack.dead.json').read_text())['reason'])

    def test_kept_sets_persistent_hold(self):
        g = self.g(); self.adopt(g)
        self.o.wrapper_stays = True                                           # harness wrapper does not exit
        self.assertEqual(self.act_on_foreign(g), ['kept'])
        self.assertIn('HOLD', self.ready())
        g.close(); g2 = self.g()
        self.assertIsNotNone(g2.hold)                                         # survives a guardian restart
        g2.step(); self.assertIn('HOLD', self.ready())

    def test_restart_adopts_registered_consumer_only_with_matching_receipt(self):
        pid = self.launch(100, R1); self.exec_consumer(pid, R1)               # consumer running before guardian start
        g = self.g(); g.step()
        self.assertEqual(g.adopted['pid'], pid)
        g.close()
        self.o.ledger[R1]['consumer_pid'] = 1
        g2 = self.g(); g2.step()
        self.assertIsNone(g2.adopted)
        self.assertNotIn(pid, {p for (p, _) in g2.owned})


class LowVramTest(Base):
    def setUp(self):
        super().setUp()
        self.o.add(600, 60, 1, T2.APPROVED_CMD, uid=0); self.o.gpu.append(600)
        auth = {'kind': 'groot_gpu_sharing_exception_v1', 'machine': 'srv50', 'hostname': 'worker2', 'gpu': 1, 'gpu_uuid': T2.UUID,
                'session': S, 'experiment': 'fold27', 'allowed_kind': 'eval', 'max_serves': 2, 'per_serve_mib': 6118,
                'reserve_mib': 16384, 'expires_at': self.o.t + 3600, 'boot_id': 'BOOT', 'prepared_at': self.o.t,
                'approval': 'test', 'foreign': [{'pid': 600, 'uid': 0, 'start_ticks': 60, 'cmdline': T2.APPROVED_CMD,
                                                 'cmdline_sha256': hashlib.sha256(T2.APPROVED_CMD.encode()).hexdigest(),
                                                 'max_used_mib': 22000}]}
        self.ap = self.t / 'auth.json'; self.ap.write_text(json.dumps(auth)); self.files[self.ap] = self.ap.read_bytes()
        self.probe = T2.GProbe(self.o)

    def gg(self):
        return self.g(probe=self.probe, sharing_authorization=str(self.ap))

    def test_approved_process_tolerated_and_low_headroom(self):
        g = self.gg(); pid = self.adopt(g)
        self.assertEqual(g.adopted['pid'], pid)
        self.probe.free = 16383                                               # below reserve -> stop our jobs only
        res = [r for r in (g.step() for _ in range(3)) if r]
        self.assertEqual(res, ['released']); self.assertNotIn(600, self.kills()); self.assertIn(600, self.o.procs)
        pid2 = self.launch(200, R2)
        ok, why = self.req(g, pid2, R2, timeout=5)
        self.assertFalse(ok); self.assertIn('not clean', why)                 # still low: no new adoption
        self.probe.free = 40000; g.step()
        self.assertTrue(self.req(g, pid2, R2)[0])

    def test_pinned_pid_reused_is_foreign(self):
        g = self.gg(); g.step()
        self.o.procs[600].start = 61
        g.step()
        pid = self.launch(100, R1)
        ok, why = self.req(g, pid, R1, timeout=5); self.assertFalse(ok)
        self.assertNotIn(600, self.kills())


class GateTest(Base):
    def test_ready_requires_live_fresh_guardian(self):
        self.assertEqual(self.ready(), 'no guardian heartbeat')
        g = self.g(); g.step(); self.assertIsNone(self.ready())
        self.o.t += 60; self.assertIn('stale', self.ready())
        g.step(); self.o.remove(990); self.assertIn('not alive', self.ready())   # guardian crashed

    def test_ack_must_echo_request_and_guardian(self):
        g = self.g(); g.step()
        pid = self.launch(100, R1)
        hd = self.sd / 'handoff'

        def forge(_s):
            self.o.sleep(1); g.heartbeat()
            (hd / 'ack.n1.json').write_text(json.dumps({'accepted': True, 'nonce': 'n1', 'receipt': R1, 'pid': pid + 1,
                                                        'start': self.o.start(pid), 'boot_id': 'BOOT', 'guardian': g.me}))
        ok, why = HG.request(self.sd, 'srv50', 1, S, R1, pid, 5, now_fn=lambda: self.o.t, sleep_fn=forge,
                             start_fn=self.o.start, boot_fn=lambda: 'BOOT', nonce='n1')
        self.assertFalse(ok); self.assertIn('does not echo', why)
        self.assertFalse((hd / 'request.n1.json').exists())

    def test_no_answer_times_out_and_withdraws(self):
        g = self.g(); g.step()
        pid = self.launch(100, R1)
        ok, why = HG.request(self.sd, 'srv50', 1, S, R1, pid, 3, now_fn=lambda: self.o.t,
                             sleep_fn=lambda s: (self.o.sleep(1), g.heartbeat()), start_fn=self.o.start, boot_fn=lambda: 'BOOT')
        self.assertFalse(ok); self.assertIn('no guardian answer', why)
        self.assertFalse(list((self.sd / 'handoff').glob('request.*')))
        g.step(); self.assertIsNone(g.adopted)                                # late guardian cannot adopt it


class WatchdogTest(Base):
    def setUp(self):
        super().setUp()
        self.ws = []

    def tearDown(self):
        for w in self.ws:
            w.close()
        super().tearDown()

    def armed(self, **cfg):
        g = self.g(**cfg); pid = self.adopt(g)
        w = CW.Watchdog({**self.cfg, **cfg}, self.o, self.sd, pid, R1, self.last_nonce); self.ws.append(w)
        self.assertIsNone(w.check()); self.assertTrue(w.armed)
        return g, pid, w

    def crash(self, g):
        g.close(); self.o.remove(990)                    # kernel drops the flock when the guardian dies

    def drive(self, w, n=10, step=None):
        for _ in range(n):
            r = w.check()
            if r:
                return r
            if step:
                step()
            self.o.sleep(5)
        return None

    def gone(self, pid):
        return all(p not in self.o.procs for p in (100, 110, pid, pid + 10, pid + 20))

    def test_ack_then_guardian_crash_stops_only_our_consumer(self):
        g, pid, w = self.armed()
        self.crash(g); self.foreign()                            # foreign appears after the crash: never signalled
        self.assertEqual(self.drive(w), 'watchdog_released')
        self.assertTrue(self.gone(pid)); self.assertNotIn(500, self.kills()); self.assertIn(500, self.o.procs)
        self.assertNotIn(R1, self.o.ledger)
        self.assertTrue((self.sd / 'HOLD.json').exists())
        self.assertIn((100, int(signal.SIGSTOP)), self.o.kills)          # follower frozen first (v1 order)

    def test_crash_between_ack_and_heartbeat(self):
        g, pid, w = self.armed()
        (self.sd / 'handoff/guardian.json').unlink()             # guardian died right after writing the ack
        self.crash(g)
        self.assertEqual(self.drive(w), 'watchdog_released'); self.assertTrue(self.gone(pid))

    def test_long_act_release_is_not_a_hang_but_bounded(self):
        g, pid, w = self.armed(release_wait_s=600)
        g.heartbeat('acting')                                    # guardian blocked inside act/release
        for _ in range(140):                                     # 700 s with a live, silent guardian
            self.assertIsNone(w.check()); self.o.sleep(5)
        self.assertEqual(self.o.kills, [])
        self.assertEqual(w.act_s, 20 + 2 * 10 + 600 + 360 + 60 + 60)
        self.o.t = json.loads((self.sd / 'handoff/guardian.json').read_text())['time'] + w.act_s + 1
        for _ in range(3):
            w.check(); self.o.sleep(5)
        self.assertIn((990, int(signal.SIGTERM)), self.o.kills)            # our hung guardian only
        self.assertIn(pid, self.o.procs)
        g.close()                                                # kernel drops its flock
        self.assertEqual(self.drive(w), 'watchdog_released'); self.assertTrue(self.gone(pid))

    def test_watch_phase_hang_is_bounded(self):
        g, pid, w = self.armed()
        t0 = self.o.t
        while self.o.t - t0 < w.hang_s - 10:
            self.assertIsNone(w.check()); self.o.sleep(5)
        self.assertEqual(self.o.kills, [])
        for _ in range(6):
            w.check(); self.o.sleep(5)
        self.assertIn((990, int(signal.SIGTERM)), self.o.kills)

    def test_guardian_pid_reuse_counts_as_lost(self):
        g, pid, w = self.armed()
        g.close(); self.o.procs[990].start = 100                 # same pid, different process
        self.assertEqual(self.drive(w), 'watchdog_released')
        self.assertNotIn((990, int(signal.SIGTERM)), self.o.kills)

    def test_dead_identity_but_lock_held_defers(self):
        g, pid, w = self.armed()
        self.o.remove(990)                                       # lock still held -> someone owns the GPU guard
        self.assertIsNone(self.drive(w)); self.assertEqual(self.o.kills, [])
        self.assertIn('stop_deferred_lock_busy', w.log.read_text())

    def test_restarted_guardian_takes_over(self):
        g, pid, w = self.armed()
        self.crash(g)
        self.o.add(991, 98, 1, 'python guardian_v4.py --config c.json'); self.o.me = {'pid': 991, 'start': 98}
        g2 = self.g()
        self.assertIsNone(self.drive(w, n=40, step=g2.step))
        self.assertEqual(g2.adopted['pid'], pid); self.assertEqual(self.o.kills, [])

    def test_orphan_needs_fence_ack_then_stops(self):
        g, pid, w = self.armed()
        self.crash(g)
        (self.sd / 'HOLD.json').write_text(json.dumps({'reason': 'operator hold'}))
        self.o.add(991, 98, 1, 'python guardian_v4.py --config c.json'); self.o.me = {'pid': 991, 'start': 98}
        g2 = self.g(); self.foreign(600, 60)
        r = self.drive(w, n=80, step=g2.step)
        self.assertEqual(r, 'watchdog_released')
        self.assertIsNone(g2.adopted); self.assertTrue(self.gone(pid))
        self.assertIn('fence_written', w.log.read_text()); self.assertNotIn(600, self.kills())
        self.assertNotIn(991, self.kills())

    def test_fenced_consumer_never_adopted(self):
        g, pid, w = self.armed()
        self.crash(g)
        (self.sd / 'HOLD.json').write_text(json.dumps({'reason': 'operator hold'}))
        self.o.add(991, 98, 1, 'python guardian_v4.py --config c.json'); self.o.me = {'pid': 991, 'start': 98}
        g2 = self.g()
        for _ in range(30):                                      # until the fence exists
            g2.step(); w.check(); self.o.sleep(5)
            if w.fence_t is not None:
                break
        self.assertIsNotNone(w.fence_t)
        (self.sd / 'HOLD.json').unlink(); g2.hold = None         # operator clears hold: fence still wins
        g2.step(); self.assertIsNone(g2.adopted)

    def test_consumer_normal_end_no_action(self):
        g, pid, w = self.armed()
        self.o.remove(pid)
        self.assertEqual(w.check(), 'consumer_ended')
        self.assertEqual(self.o.kills, []); self.assertFalse([c for c in self.o.harness_calls if c[0] == 'release'])

    def test_duplicate_watchdog_refused(self):
        g, pid, w = self.armed()
        with self.assertRaises(SystemExit):
            CW.Watchdog(self.cfg, self.o, self.sd, pid, R1, self.last_nonce)

    def test_ownership_unverified_sends_no_signal(self):
        g, pid, w = self.armed()
        self.crash(g); self.o.ledger[R1]['consumer_pid'] = 1
        self.assertEqual(self.drive(w), 'watchdog_refused'); self.assertEqual(self.o.kills, [])

    def test_unarmed_watchdog_never_acts(self):
        g = self.g(); g.step()
        pid = self.launch(100, R1)
        w = CW.Watchdog(self.cfg, self.o, self.sd, pid, R1, 'never-acked'); self.ws.append(w)
        self.crash(g)
        self.assertIsNone(self.drive(w)); self.assertEqual(self.o.kills, [])

    def test_watchdog_crash_sets_guardian_hold_no_kill(self):
        g, pid, w = self.armed()
        self.o.remove(8000 + pid); g.step()
        self.assertIsNotNone(g.hold); self.assertIn(pid, self.o.procs); self.assertEqual(self.o.kills, [])

    def test_handoff_requires_detached_watchdog_for_this_nonce(self):
        g = self.g(); g.step()
        pid = self.launch(100, R1)

        def raw(nonce, wp):
            return HG.request(self.sd, 'srv50', 1, S, R1, pid, 5, now_fn=lambda: self.o.t,
                              sleep_fn=lambda s: (self.o.sleep(1), g.step()), start_fn=self.o.start,
                              boot_fn=lambda: 'BOOT', nonce=nonce, watchdog_pid=wp)
        self.assertIn('watchdog not alive', raw('n0', None)[1])
        wp = self.watchdog_proc(pid, R1, 'nA')
        self.assertIn('does not name this consumer/nonce', raw('nB', wp)[1])          # someone else's watchdog
        self.o.procs[wp].ppid = pid                                                      # inside the consumer tree
        self.assertIn('inside the consumer tree', raw('nA', wp)[1])
        self.assertIn('refused', raw('nA', wp)[1])                                     # a nonce is single-use
        self.o.remove(wp)
        wp = self.watchdog_proc(pid, R1, 'nC')                                           # fresh, detached
        ok, info = raw('nC', wp)
        self.assertTrue(ok, info); self.assertEqual(g.adopted['watchdog'], {'pid': wp, 'start': self.o.start(wp)})


class ShellTest(unittest.TestCase):
    SH = HD / 'eval_gpu_queue_v4.sh'

    def run_sh(self, args=(), gate_ready=0, gate_req=0, **env):
        t = Path(tempfile.mkdtemp()); log = t / 'calls'
        stub = t / 'py'
        stub.write_text(f'#!/bin/bash\necho "$*" >> {log}\ncase "$*" in *"handoff_gate.py ready"*) exit {gate_ready};; '
                        f'*"handoff_gate.py request"*) exit {gate_req};; *may-start*) exit 3;; esac\nexit 0\n')
        stub.chmod(0o755)
        bin_ = t / 'bin'; bin_.mkdir()
        (bin_ / 'python3').write_text(f'#!/bin/bash\necho "python3 $*" >> {log}\nexit 0\n'); (bin_ / 'python3').chmod(0o755)
        env = {k: (str(t / 'st') if v == '/x' else v) for k, v in env.items()}
        e = dict(os.environ, EVAL_GPU_PY=str(stub), HARNESS_MACHINE='kanu', HARNESS_SESSION=S, FOLLOW='0', GPU='5',
                 GUARDIAN_CONFIG=str(t / 'cfg.json'), PATH=f'{bin_}:{os.environ["PATH"]}', **env)
        r = subprocess.run(['bash', str(self.SH), *args], env=e, capture_output=True, text=True, timeout=60)
        out = log.read_text().splitlines() if log.exists() else []
        shutil.rmtree(t)
        return r, out

    def test_state_dir_required(self):
        r, calls = self.run_sh()
        self.assertNotEqual(r.returncode, 0); self.assertEqual(calls, [])

    def test_no_reservation_without_ready_guardian(self):
        r, calls = self.run_sh(gate_ready=3, GUARDIAN_STATE_DIR='/x')
        self.assertEqual(r.returncode, 75)
        self.assertEqual(len(calls), 1); self.assertIn('handoff_gate.py ready', calls[0])

    def test_ready_guardian_then_v3_flow(self):
        r, calls = self.run_sh(GUARDIAN_STATE_DIR='/x')
        self.assertEqual(r.returncode, 0); self.assertIn('may-start', calls[1])

    def test_inner_refused_handoff_never_starts_consumer(self):
        r, calls = self.run_sh(['--inner'], gate_req=75, GUARDIAN_STATE_DIR='/x', HARNESS_RECEIPT=R1)
        self.assertEqual(r.returncode, 75)
        self.assertTrue(any('groot_harness.py verify' in c for c in calls))
        self.assertFalse(any(' run --min-jobs' in c for c in calls))

    def test_inner_adopted_then_exec_consumer(self):
        r, calls = self.run_sh(['--inner'], GUARDIAN_STATE_DIR='/x', HARNESS_RECEIPT=R1)
        self.assertEqual(r.returncode, 0)
        i = [k for k, c in enumerate(calls) if 'handoff_gate.py request' in c]
        j = [k for k, c in enumerate(calls) if ' run --min-jobs' in c]
        self.assertTrue(i and j and i[0] < j[0])
        self.assertIn(f'--receipt {R1} --pid', calls[i[0]])
        self.assertRegex(calls[i[0]], r'--nonce [0-9a-f]{32} --watchdog-pid [0-9]+ ')


if __name__ == '__main__':
    unittest.main()

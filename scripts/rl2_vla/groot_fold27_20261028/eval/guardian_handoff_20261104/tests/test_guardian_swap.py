"""guardian_swap on the v1/v2 fake fixture (no real signal: pidfd calls are routed to the fake OS). tearDown asserts that
progress/claims/registry/contract files are byte-identical afterwards. Covers: dry-run = checks only, swap ends ONLY the
old guardian (consumer/follower/serves/approved/new guardian never signalled), config/contract diff refusals, wrong
expiry, new guardian identity/cmdline/observe-only refusals, replaced consumer, untolerated GPU process, too close to the
old expiry, old guardian identity change at signal time."""
import json, signal, sys, unittest
from pathlib import Path
from unittest.mock import patch

HD = Path(__file__).resolve().parents[1]
F27 = HD.parents[1]
sys.path[:0] = [str(HD), str(F27 / 'tests'), str(F27 / 'eval/runtime'), str(F27.parents[2] / 'scripts/utils')]
import test_foreign_gpu_guardian as T     # noqa: E402  (unchanged)
import test_foreign_gpu_guardian_v2 as T2  # noqa: E402  (unchanged)
import guardian_swap as GS                 # noqa: E402

V3 = 'python scripts/rl2_vla/groot_fold27_20261028/eval/runtime/foreign_gpu_guardian_v3.py'


class SwapTest(T2.SharingTest):
    def setUp(self):
        super().setUp()
        self.c0 = self.t / 'cfg_old.json'; self.c1 = self.t / 'cfg_new.json'; self.a1 = self.t / 'auth_new.json'
        self.c0.write_text(json.dumps({**self.cfg, 'sharing_authorization': str(self.ap)}))
        self.new_exp = self.auth['expires_at'] + 86400
        self.a1.write_text(json.dumps({**self.auth, 'expires_at': self.new_exp, 'extensions': [{'approval': 'x'}]}))
        self.c1.write_text(json.dumps({**self.cfg, 'sharing_authorization': str(self.a1)}))
        for p in (self.c0, self.c1, self.a1):
            self.files[p] = p.read_bytes()
        self.o.add(980, 98, 1, f'{V3} --config {self.c0} --state-dir st_old watch')
        self.o.add(990, 99, 1, f'{V3} --config {self.c1} --state-dir st_new watch')
        self.pins = {'consumer_pid': 120, 'consumer_start': 12, 'receipt': T.RECEIPT, 'old_guardian_pid': 980,
                     'old_guardian_start': 98, 'new_guardian_pid': 990, 'new_guardian_start': 99}
        self.n = 0

    def chk(self, **kw):
        self.n += 1
        pins = {**self.pins, **{k: v for k, v in kw.items() if k in self.pins}}
        return GS.check(kw.get('c0', self.c0), kw.get('c1', self.c1), kw.get('expect', self.new_exp), pins, self.o,
                        self.t / f'chk{self.n}', probe=self.probe, now=kw.get('now', self.o.t))

    def refused(self, msg, **kw):
        with self.assertRaisesRegex((GS.Refused, SystemExit), msg):
            self.chk(**kw)
        self.assertEqual(self.o.kills, [])

    def swap(self):
        with patch.object(GS.os, 'pidfd_open', lambda pid: pid, create=True), patch.object(GS.os, 'close', lambda fd: None), \
             patch.object(GS.signal, 'pidfd_send_signal', lambda fd, s: self.o.kill(fd, s), create=True):
            return GS.term_old(self.pins, self.o, self.c0, wait_s=1, sleep=lambda s: None)

    def test_fallback_without_pidfd(self):
        with patch.object(GS.os, 'kill', self.o.kill), patch.object(GS, 'hasattr', lambda o, n: False, create=True):
            self.assertTrue(GS.term_old(self.pins, self.o, self.c0, wait_s=1, sleep=lambda s: None))
        self.assertEqual(self.o.kills, [(980, int(signal.SIGTERM))])

    def test_check_then_swap_ends_only_old_guardian(self):
        ev = self.chk()
        self.assertEqual(ev['adopted'], {'pid': 120, 'start': 12, 'receipt': T.RECEIPT})
        self.assertEqual(ev['contract_diff'], ['expires_at', 'extensions']); self.assertEqual(self.o.kills, [])
        self.assertTrue(self.swap())
        self.assertEqual(self.o.kills, [(980, int(signal.SIGTERM))])
        self.never(100, 110, 120, 130, 140, 150, 160, 600, 990)

    def test_config_and_contract_diff_refused(self):
        bad = self.t / 'cfg_bad.json'; bad.write_text(json.dumps({**self.cfg, 'sharing_authorization': str(self.a1), 'poll_s': 1}))
        self.refused('config diff', c1=bad)
        a2 = self.t / 'auth_bad.json'; a2.write_text(json.dumps({**self.auth, 'expires_at': self.new_exp, 'max_serves': 3}))
        bad2 = self.t / 'cfg_bad2.json'; bad2.write_text(json.dumps({**self.cfg, 'sharing_authorization': str(a2)}))
        self.o.procs[990].cmd = f'{V3} --config {bad2} --state-dir st_new watch'
        self.refused('contract diff', c1=bad2)

    def test_wrong_expiry_or_too_late_refused(self):
        self.refused('expires_at', expect=self.new_exp + 1)
        self.refused('OLD expiry', now=self.auth['expires_at'] - 60)

    def test_new_guardian_identity_refused(self):
        self.refused('start', new_guardian_start=7)
        self.o.procs[990].cmd = f'{V3} --config {self.c1} --state-dir st_new --observe-only watch'
        self.refused('cmdline')
        self.o.procs[990].cmd = f'{V3} --config {self.c0} --state-dir st_new watch'
        self.refused('cmdline')

    def test_old_guardian_identity_refused(self):
        self.refused('old guardian', old_guardian_start=1)
        self.o.procs[980].cmd = f'{V3} --config other.json --state-dir st watch'
        self.refused('old guardian')

    def test_replaced_consumer_refused(self):
        self.refused('pinned consumer', consumer_start=13)

    def test_untolerated_gpu_process_refused(self):
        self.o.add(777, 77, 1, 'python stranger.py', uid=1000); self.o.gpu.append(777)
        self.refused('untolerated')

    def test_old_guardian_changed_at_signal_time(self):
        self.chk()
        self.o.procs[980].start = 55                              # PID reused between check and signal
        with self.assertRaisesRegex(GS.Refused, 'changed'):
            self.swap()
        self.assertEqual(self.o.kills, [])

    def test_cli_refuses_existing_paths(self):
        rc = GS.main(['--old-config', str(self.c0), '--new-config', str(self.c1), '--check-dir', str(self.t),
                      '--out', str(self.t / 'o.json'), '--expect-expires-at', '1', '--receipt', 'r',
                      *sum([[f'--{k}', '1'] for k in ('consumer-pid', 'consumer-start', 'old-guardian-pid', 'old-guardian-start',
                                                        'new-guardian-pid', 'new-guardian-start')], [])])
        self.assertEqual(rc, 2); self.assertEqual(self.o.kills, [])


for _n in [n for n in dir(T2.SharingTest) if n.startswith('test_') and n not in SwapTest.__dict__]:
    setattr(SwapTest, _n, None)

if __name__ == '__main__':
    unittest.main()

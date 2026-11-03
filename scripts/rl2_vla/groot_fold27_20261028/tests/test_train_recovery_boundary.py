"""train_recovery_boundary (read-only) + gpu_sharing_derive_v1 on fake hosts: READY only when eval is gone, ledger empty
on the GPU, GPU holds only the pinned identity, old coordinator idle without live claims, no replacement running; derive
copies the pinned identity verbatim, never approves other processes, never weakens budget/reserve/expiry."""
import argparse, hashlib, json, shutil, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'trqam'), str(ROOT / 'eval/runtime'), str(ROOT.parents[2] / 'scripts/utils')]
import train_recovery_boundary as B       # noqa: E402
import gpu_sharing_derive_v1 as D         # noqa: E402
import gpu_sharing_authorization as SA    # noqa: E402

CMD = 'python sj_train.py --x'
SHA = hashlib.sha256(CMD.replace(' ', '\0').encode()).hexdigest()
UUID, BOOT = 'GPU-9fdab52c-2f3e-38e1-1fcd-855c290ef8c3', '521dd1f3-71bf-437d-8fdb-11505e397d7f'


class Probe:
    def __init__(self):
        self.t, self.bootv, self.apps, self.free, self.procs, self.fail = 1000.0, BOOT, [{'pid': 2795788, 'used_mib': 24348}], 40000, {
            2795788: {'uid': 1019, 'start_ticks': 1377893575, 'cmdline_sha256': SHA, 'cmdline': CMD}}, False

    def now(self): return self.t
    def boot_id(self): return self.bootv
    def hostname(self): return 'worker2'
    def proc(self, pid): return self.procs.get(pid)

    def gpu(self, i):
        if self.fail:
            raise SA.AuthError('nvidia-smi failed')
        return {'uuid': UUID, 'free_mib': self.free, 'total_mib': 81920, 'apps': [dict(x) for x in self.apps]}


def contract(t, path, session='fold27-v2-eval', reserve=16384):
    body = {'kind': SA.KIND, 'machine': 'srv50', 'hostname': 'worker2', 'gpu': 3, 'gpu_uuid': UUID, 'session': session,
            'experiment': 'fold27', 'allowed_kind': 'eval', 'max_serves': 2, 'per_serve_mib': 6118, 'reserve_mib': reserve,
            'expires_at': t + 86400, 'boot_id': BOOT, 'prepared_at': t, 'approval': 'user',
            'foreign': [{'pid': 2795788, 'uid': 1019, 'start_ticks': 1377893575, 'cmdline_sha256': SHA, 'cmdline': CMD,
                         'max_used_mib': 25600}]}
    path.write_text(json.dumps(body)); return body


class Host:
    def __init__(self, probe):
        self.probe, self.starts, self.cmds, self.led, self.files, self.ledger_fail = probe, {3088084: 11, 631152: 22, 627282: 33}, {}, {}, {}, False

    def boot(self): return self.probe.bootv
    def start(self, pid): return self.starts.get(pid)
    def cmdline(self, pid): return self.cmds.get(pid, 'x' if pid in self.starts else None)
    def pids(self): return list(self.starts)

    def ledger(self):
        if self.ledger_fail:
            raise RuntimeError('ssh failed')
        return self.led

    def read_json(self, p):                                                    # same contract as HostView.read_json
        try:
            return json.loads(Path(p).read_text())
        except (OSError, ValueError):
            return None


class Boundary(unittest.TestCase):
    def setUp(self):
        self.t = Path(tempfile.mkdtemp()); self.p = Probe(); self.h = Host(self.p)
        self.ap = self.t / 'eval.json'; contract(self.p.t, self.ap)
        self.out = self.t / 'train_out'; (self.out / 'queue' / 'train_claims').mkdir(parents=True)
        (self.out / 'queue' / 'host_srv50_COORD_V5_STATE.json').write_text(json.dumps({'status': 'training', 'running': {}}))
        self.h.led = {'E' * 32: {'machine': 'srv50', 'gpus': [3], 'session': 'fold27-v2-eval', 'kind': 'eval'}}
        self.p.apps.append({'pid': 4000, 'used_mib': 6000})                   # eval serve
        self.argv = ['--machine', 'srv50', '--gpu', '3', '--boot-id', BOOT, '--session', 'fold27-t49-srv50-candle-v2',
                     '--contract', str(self.ap), '--eval-consumer', '3088084:11', '--old-coordinator', '631152:22',
                     '--old-guardian', '627282:33', '--old-out', str(self.out)]

    def tearDown(self):
        shutil.rmtree(self.t)

    def run_(self):
        ap = argparse.Namespace(machine='srv50', gpu=3, boot_id=BOOT, session='fold27-t49-srv50-candle-v2', contract=str(self.ap),
                                eval_consumer='3088084:11', old_coordinator='631152:22', old_guardian='627282:33', old_out=str(self.out))
        return B.check(ap, self.h)

    def eval_done(self):
        del self.h.starts[3088084]; self.h.led = {}; self.p.apps = [a for a in self.p.apps if a['pid'] != 4000]

    def test_not_ready_while_eval_runs(self):
        r = self.run_()
        self.assertFalse(r['ready'])
        self.assertTrue(any('eval consumer' in x for x in r['reasons']) and any('receipt' in x for x in r['reasons'])
                        and any('other than the verified pinned' in x for x in r['reasons']))

    def test_ready_after_eval_boundary_with_idle_old_coordinator(self):
        self.eval_done()
        r = self.run_()
        self.assertTrue(r['ready'], r['reasons'])
        self.assertEqual(r['snapshot']['gpu']['pinned_ok'], [2795788])
        self.assertTrue(r['snapshot']['old_coordinator']['alive'])

    def test_running_arm_or_live_claim_blocks(self):
        self.eval_done()
        (self.out / 'queue' / 'host_srv50_COORD_V5_STATE.json').write_text(json.dumps({'running': {'basic': 3}}))
        self.assertFalse(self.run_()['ready'])
        (self.out / 'queue' / 'host_srv50_COORD_V5_STATE.json').write_text(json.dumps({'running': {}}))
        (self.out / 'queue' / 'train_claims' / 'x.claim').write_text(json.dumps({'boot_id': BOOT, 'consumer': {'pid': 631152, 'start': 22}}))
        self.assertFalse(self.run_()['ready'])
        (self.out / 'queue' / 'train_claims' / 'x.claim').write_text('{broken')
        self.assertFalse(self.run_()['ready'])                                 # unreadable claim is not "free"

    def test_unknown_or_reused_gpu_process_blocks(self):
        self.eval_done(); self.p.apps.append({'pid': 5000, 'used_mib': 10})
        self.assertFalse(self.run_()['ready'])
        self.p.apps.pop(); self.p.procs[2795788] = dict(self.p.procs[2795788], start_ticks=1)
        self.assertFalse(self.run_()['ready'])

    def test_query_failures_block(self):
        self.eval_done(); self.h.ledger_fail = True
        self.assertFalse(self.run_()['ready'])
        self.h.ledger_fail = False; self.p.fail = True
        self.assertFalse(self.run_()['ready'])

    def test_boot_change_and_duplicate_replacement_block(self):
        self.eval_done(); self.p.bootv = 'other'
        self.assertFalse(self.run_()['ready'])
        self.p.bootv = BOOT; self.h.starts[7000] = 1; self.h.cmds[7000] = 'python trqam/train_guardian_v3.py --config c'
        self.assertFalse(self.run_()['ready'])

    def test_cli_once_exit_codes_and_no_writes(self):
        before = {p: p.read_bytes() for p in self.t.rglob('*') if p.is_file()}
        self.assertEqual(B.main(self.argv, host=self.h), 3)
        self.eval_done()
        self.assertEqual(B.main(self.argv, host=self.h), 0)
        self.assertEqual(before, {p: p.read_bytes() for p in self.t.rglob('*') if p.is_file()})


class Derive(unittest.TestCase):
    def setUp(self):
        self.t = Path(tempfile.mkdtemp()); self.p = Probe()
        self.src = self.t / 'SHARING_PID_1735.json'; self.body = contract(self.p.t, self.src)
        self.p.apps.append({'pid': 4000, 'used_mib': 6000})                   # our eval serve is present while deriving

    def tearDown(self):
        shutil.rmtree(self.t)

    def a(self, **kw):
        d = dict(source=self.src, session='fold27-t49-srv50-candle-v2', max_serves=1, per_serve_mib=7168,
                 per_serve_basis='measured_4584_x1.5_rounded_7168', measured_peak_mib=4584, reserve_mib=16384, expires_in_h=24.0,
                 expect_pid=2795788, expect_uid=1019, expect_start_ticks=1377893575, expect_cmdline_sha256=SHA, expect_boot_id=BOOT,
                 expect_gpu_uuid=UUID, approval='user: use GPU3 overlapping foreign job', out=self.t / 'train.json')
        d.update(kw); return argparse.Namespace(**d)

    def test_derive_copies_pinned_identity_only(self):
        b = D.derive(self.a(), self.p)
        self.assertEqual(b['foreign'], self.body['foreign'])                    # verbatim; eval serve 4000 NOT approved
        self.assertEqual((b['session'], b['max_serves'], b['per_serve_mib'], b['reserve_mib']), ('fold27-t49-srv50-candle-v2', 1, 7168, 16384))
        self.assertLessEqual(b['expires_at'], self.body['expires_at'])
        self.assertEqual(b['derived_from']['sha256'], hashlib.sha256(self.src.read_bytes()).hexdigest())
        got = SA.load(self.t / 'train.json')
        self.p.apps = [x for x in self.p.apps if x['pid'] != 4000]
        SA.authorized_processes(got, 'srv50', [3], 'fold27-t49-srv50-candle-v2', 'eval', 1, self.p)   # admits after eval
        self.p.apps.append({'pid': 4000, 'used_mib': 6000})
        with self.assertRaises(SA.AuthError):                                   # never while an eval serve is on the GPU
            SA.authorized_processes(got, 'srv50', [3], 'fold27-t49-srv50-candle-v2', 'eval', 1, self.p)

    def test_expiry_never_beyond_source(self):
        b = D.derive(self.a(expires_in_h=72.0), self.p)
        self.assertEqual(b['expires_at'], self.body['expires_at'])

    def refused(self, **kw):
        with self.assertRaises((SA.AuthError, ValueError, KeyError)):
            D.derive(self.a(**kw), self.p)
        self.assertFalse((self.t / 'train.json').exists())

    def test_refusals(self):
        self.refused(expect_start_ticks=1)                                      # expectations != source pin
        self.refused(expect_cmdline_sha256='0' * 64)
        self.refused(expect_boot_id='other')
        self.refused(expect_gpu_uuid='GPU-other')
        self.refused(per_serve_mib=4000)                                        # below measured peak
        self.refused(per_serve_basis=' ')
        self.refused(reserve_mib=8192)                                          # weaker than source
        self.refused(session='fold27-v2-eval')                                  # same session as source
        self.refused(expires_in_h=100)
        self.refused(max_serves=99)

    def test_live_identity_refusals(self):
        for mut in (lambda: self.p.procs.__setitem__(2795788, dict(self.p.procs[2795788], start_ticks=2)),   # PID reuse
                    lambda: setattr(self.p, 'apps', [{'pid': 4000, 'used_mib': 6000}]),                     # pinned gone
                    lambda: setattr(self.p, 'apps', [{'pid': 2795788, 'used_mib': 26000}]),                 # over max_used
                    lambda: setattr(self.p, 'bootv', 'rebooted'),
                    lambda: setattr(self.p, 't', self.body['expires_at'] + 1),                                # source expired
                    lambda: setattr(self.p, 'fail', True)):                                                  # NVML failure
            self.tearDown(); self.setUp(); mut()
            self.refused()

    def test_v2_or_multi_pin_source_refused(self):
        b = dict(self.body, foreign=self.body['foreign'] * 2); self.src.write_text(json.dumps(b))
        self.refused()

    def test_existing_out_never_overwritten(self):
        (self.t / 'train.json').write_text('{}')
        with self.assertRaises(FileExistsError):
            D.derive(self.a(), self.p)


if __name__ == '__main__':
    unittest.main()

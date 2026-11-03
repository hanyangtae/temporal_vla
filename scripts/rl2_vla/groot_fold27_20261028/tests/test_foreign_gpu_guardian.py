"""foreign_gpu_guardian on a fake process table (CPU only; no signal ever reaches a real process, no nvidia-smi, no harness).
Covers: no foreign -> nothing; admission race (foreign root + foreign same-uid appear after admission) -> freeze order,
graceful arm stop, owned serve/collector stop (root collector via container exec), consumer/follower stop, release only
after own absence; foreign never signalled; transient unknown not confirmed; PID reuse of an owned PID; refusal on
unverified ownership (receipt missing, consumer pid reused, follower mismatch); observe-only; no premature release
(GPU query failure, harness wrapper alive); unverified container -> its processes never signalled; partial outputs untouched."""
import json, os, shutil, signal, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'eval/runtime'))
import foreign_gpu_guardian as G   # noqa: E402

SESSION, RECEIPT = 'fold27-v2-eval', 'R' * 32


class P:
    def __init__(self, pid, start, ppid, cmd, uid=1000, cgroup='', pgid=None, nspid=None, ignore_term=False):
        self.pid, self.start, self.ppid, self.cmd, self.uid, self.cgroup = pid, start, ppid, cmd, uid, cgroup
        self.pgid, self.nspid, self.ignore_term = pgid or pid, nspid, ignore_term
        self.stopped, self.pending = False, []


class FakeOS:
    def __init__(self, deploy):
        self.procs, self.bootv, self.t, self.gpu, self.gpu_fail = {}, 'BOOT', 1000.0, [], False
        self.kills, self.cexec, self.harness_calls, self.myuid = [], [], [], 1000
        self.ledger = {RECEIPT: {'machine': 'srv50', 'gpus': [1], 'serves': 6, 'session': SESSION, 'kind': 'eval'}}
        self.container = {'Id': 'CID123', 'Name': '/fold27-iso-env-gpu1', 'Image': 'sha256:img', 'State': {'Running': True},
                          'Mounts': [{'Destination': '/temporal_vla', 'Source': deploy, 'RW': True}],
                          'Config': {'Env': ['NVIDIA_VISIBLE_DEVICES=1'], 'Labels': {'fold27.deploy': deploy, 'fold27.gpu': '1'}}}
        self.wrapper_stays = False

    def add(self, *a, **k):
        p = P(*a, **k); self.procs[p.pid] = p; return p

    # interface
    def pids(self): return list(self.procs)
    def start(self, pid): return self.procs[pid].start if pid in self.procs else None
    def boot(self): return self.bootv
    def ppid(self, pid): return self.procs[pid].ppid if pid in self.procs else None
    def uid(self, pid): return self.procs[pid].uid if pid in self.procs else None
    def nspid(self, pid): return self.procs[pid].nspid if pid in self.procs else None
    def cmdline(self, pid): return self.procs[pid].cmd if pid in self.procs else None
    def cgroup(self, pid): return self.procs[pid].cgroup if pid in self.procs else ''

    def pgid(self, pid):
        if pid not in self.procs:
            raise ProcessLookupError
        return self.procs[pid].pgid

    def gpu_pids(self, gpu):
        if self.gpu_fail:
            raise RuntimeError('nvidia-smi failed')
        return [p for p in self.gpu if p in self.procs]

    def remove(self, pid):
        p = self.procs.pop(pid, None)
        if p is None:
            return
        for q in self.procs.values():
            if q.ppid == pid:
                q.ppid = 1
        if 'eval_gpu_queue_v2.py' in p.cmd and not self.wrapper_stays:       # harness run wrapper finishes with its child
            for q in list(self.procs.values()):
                if 'groot_harness.py' in q.cmd:
                    self.remove(q.pid)

    def deliver(self, p, sig):
        if sig == signal.SIGKILL:
            self.remove(p.pid); return
        if sig == signal.SIGSTOP:
            p.stopped = True; return
        if sig == signal.SIGCONT:
            p.stopped = False
            for s in p.pending:
                self.deliver(p, s)
            p.pending = []; return
        if p.stopped:
            p.pending.append(sig); return
        if sig == signal.SIGINT and 'run_eval_arm' in p.cmd:                 # run_eval_arm finally: serve pg + exec client
            for q in list(self.procs.values()):
                if q.ppid == p.pid and ('serve_mil_qam' in q.cmd or q.cmd.startswith('docker exec')):
                    self.remove(q.pid)
            self.remove(p.pid); return
        if sig in (signal.SIGTERM, signal.SIGINT) and not p.ignore_term:
            self.remove(p.pid)

    def kill(self, pid, sig):
        if pid not in self.procs:
            raise ProcessLookupError
        if self.procs[pid].uid != self.myuid:
            raise PermissionError
        self.kills.append((pid, int(sig)))
        self.deliver(self.procs[pid], sig)

    def killpg(self, pg, sig):
        self.kills.append((('pg', pg), int(sig)))
        for p in [q for q in self.procs.values() if q.pgid == pg]:
            self.deliver(p, sig)

    def docker_inspect(self, name):
        if self.container is None:
            raise RuntimeError('no such container')
        return self.container

    def container_kill(self, name, nspid, sig):
        self.cexec.append((name, nspid, int(sig)))
        for p in list(self.procs.values()):
            if p.nspid == nspid and 'CID123' in p.cgroup:
                self.deliver(p, sig)
        return True

    def harness(self, *args):
        self.harness_calls.append(args)
        if args[0] == 'status':
            return json.loads(json.dumps(self.ledger))
        if args[0] == 'release':
            r = args[args.index('--receipt') + 1]
            assert self.ledger[r]['session'] == args[args.index('--session') + 1]
            del self.ledger[r]
            return {'released': r}
        raise AssertionError(args)

    def now(self): return self.t
    def sleep(self, s): self.t += s


class GuardianTest(unittest.TestCase):
    def setUp(self):
        self.t = Path(tempfile.mkdtemp())
        self.deploy = os.path.realpath(str(self.t / 'deploy')); os.makedirs(self.deploy)
        self.q = self.t / 'queue' / 'srv50'; (self.q / 'consumers').mkdir(parents=True); (self.q / 'claims').mkdir()
        (self.q / 'consumers' / 'gpu1.json').write_text(json.dumps({'gpu': 1, 'receipt': RECEIPT, 'boot_id': 'BOOT',
                                                                    'consumer': {'pid': 120, 'start': 12}}))
        (self.q / 'claims' / 'fold.arm.claim').write_text('{"claim": 1}')
        self.arm_dir = self.t / 'arm'; (self.arm_dir / 'rows/n03/a1').mkdir(parents=True)
        (self.arm_dir / 'progress.jsonl').write_text('{"n": 1}\n'); (self.arm_dir / 'rows/n03/a1/collect.log').write_text('partial')
        o = self.o = FakeOS(self.deploy)
        o.add(100, 10, 1, 'bash scripts/rl2_vla/groot_fold27_20261028/eval/runtime/eval_gpu_queue_v2.sh')
        o.add(110, 11, 100, 'python3 scripts/utils/groot_harness.py run --machine srv50 --gpus 1 -- bash eval_gpu_queue_v2.sh --inner')
        o.add(120, 12, 110, '.venvs/rl2_iid/bin/python -u eval/runtime/eval_gpu_queue_v2.py --pool 1 run --min-jobs 1')
        o.add(130, 13, 120, 'python -u eval/runtime/run_eval_arm_v2.py --manifest m.json --arm mil_qam65 --gpu 1')
        o.add(140, 14, 130, '/home/u/miniconda3/envs/lerobot_050_groot/bin/python eval/runtime/serve_mil_qam.py --port 8996')
        o.add(150, 15, 130, 'docker exec -w /temporal_vla fold27-iso-env-gpu1 timeout --signal=TERM 2400 python collect_task.py')
        o.add(160, 16, 900, 'timeout --signal=TERM 2400 python /temporal_vla/eval/runtime/collect_task.py', uid=0,
              cgroup='0::/system.slice/docker-CID123.scope', nspid=42)
        o.gpu = [140, 160]
        self.cfg = {'kind': G.KIND, 'host': 'srv50', 'gpu': 1, 'session': SESSION, 'queue_dir': str(self.q),
                    'env_container': 'fold27-iso-env-gpu1', 'deploy': self.deploy, 'expect_image': 'sha256:img',
                    'poll_s': 5, 'confirm_polls': 2, 'grace_s': 20, 'kill_wait_s': 10, 'release_wait_s': 30}
        self.files = {p: p.read_bytes() for p in self.t.rglob('*') if p.is_file()}

    def tearDown(self):
        for p, b in self.files.items():
            assert p.read_bytes() == b, f'guardian modified {p}'                    # progress/attempt/claims/registry untouched
        shutil.rmtree(self.t)

    def g(self, **kw):
        return G.Guardian({**self.cfg, **kw}, self.o, self.t / 'state')

    def foreign(self):
        self.o.add(500, 50, 1, 'python track_cotracker3.py', uid=0)
        self.o.add(501, 51, 1, 'python someone_elses_job.py', uid=1000)                # same uid, still foreign
        self.o.gpu += [500, 501]

    def signalled(self):
        return {k[0] if not isinstance(k[0], tuple) else k[0][1] for k in self.o.kills} | {42 for _ in self.o.cexec}

    def test_no_foreign_no_action(self):
        self.assertEqual(self.g().run(max_polls=3), 'no_action')
        self.assertEqual(self.o.kills, []); self.assertEqual([c for c in self.o.harness_calls if c[0] == 'release'], [])
        owned = json.loads((self.t / 'state/OWNED.json').read_text())['owned']
        self.assertEqual(sorted(r for _, _, r in owned), ['arm', 'consumer', 'docker_exec', 'env_collector', 'serve'])

    def test_admission_race_full_stop_and_release(self):
        g = self.g()
        self.assertIsNone(g.poll())                         # admitted, only our processes
        self.foreign()
        self.assertIsNone(g.poll())                         # first sighting: not yet confirmed
        conf = g.poll()
        self.assertEqual(sorted(p for p, _ in conf), [500, 501])
        self.assertEqual(g.act(conf), 'released')
        k = self.o.kills
        self.assertEqual(k[:3], [(100, int(signal.SIGSTOP)), (120, int(signal.SIGSTOP)), (130, int(signal.SIGSTOP))])
        self.assertLess(k.index((130, int(signal.SIGINT))), k.index((120, int(signal.SIGTERM))))
        self.assertIn(('fold27-iso-env-gpu1', 42, int(signal.SIGTERM)), self.o.cexec)    # root collector via container exec
        self.assertTrue({500, 501}.isdisjoint(self.signalled()))
        self.assertTrue(all(p in self.o.procs for p in (500, 501)))
        self.assertTrue(all(p not in self.o.procs for p in (100, 110, 120, 130, 140, 150, 160)))
        rel = [c for c in self.o.harness_calls if c[0] == 'release']
        self.assertEqual(rel, [('release', '--receipt', RECEIPT, '--session', SESSION)]); self.assertNotIn(RECEIPT, self.o.ledger)
        self.assertEqual(len(list((self.t / 'state').glob('ACTION.*.json'))), 1)

    def test_transient_unknown_not_confirmed(self):
        g = self.g()
        self.o.add(500, 50, 1, 'python probe.py', uid=0); self.o.gpu.append(500)
        self.assertIsNone(g.poll())
        self.o.remove(500)
        self.assertEqual(g.run(max_polls=3), 'no_action'); self.assertEqual(self.o.kills, [])

    def test_pid_reuse_of_owned_pid_is_foreign_and_untouched(self):
        g = self.g()
        g.poll()                                            # serve (140, start 14) recorded as owned
        self.o.remove(140)
        self.o.add(140, 77, 1, 'python track_cotracker3.py', uid=0)          # same pid, other process
        self.assertIsNone(g.poll())
        conf = g.poll()
        self.assertEqual(conf, [(140, 77)])
        g.act(conf)
        self.assertNotIn(140, self.signalled()); self.assertNotIn(('pg', 140), [x[0] for x in self.o.kills])
        self.assertIn(140, self.o.procs)

    def test_refusals_never_signal(self):
        cases = {'receipt missing': lambda: self.o.ledger.clear(),
                 'consumer pid reused': lambda: setattr(self.o.procs[120], 'start', 99),
                 'session mismatch': lambda: self.o.ledger[RECEIPT].update(session='other'),
                 'boot changed': lambda: setattr(self.o, 'bootv', 'OTHER')}
        for name, mutate in cases.items():
            self.setUp_fresh()
            g = self.g(); g.poll(); self.foreign(); g.poll(); conf = g.poll() or [(500, 50)]
            mutate()
            self.assertEqual(g.act(conf), 'refused', name)
            self.assertEqual(self.o.kills, [], name); self.assertEqual(self.o.cexec, [], name)
            self.assertTrue((self.t / 'state/ALERT.json').exists(), name)

    def setUp_fresh(self):
        self.tearDown(); self.setUp()

    def test_follower_mismatch_refused(self):
        g = self.g(follower={'pid': 100, 'start': 10})
        self.o.add(105, 9, 1, 'bash eval_gpu_queue_v2.sh')
        g.c['follower'] = {'pid': 105, 'start': 9}
        g.poll(); self.foreign(); g.poll(); conf = g.poll()
        self.assertEqual(g.act(conf), 'refused'); self.assertEqual(self.o.kills, [])

    def test_observe_only(self):
        g = G.Guardian(self.cfg, self.o, self.t / 'state', observe_only=True)
        g.poll(); self.foreign(); g.poll(); conf = g.poll()
        self.assertEqual(g.act(conf), 'observed')
        self.assertEqual(self.o.kills, []); self.assertIn(RECEIPT, self.o.ledger)

    def test_no_release_when_gpu_query_fails_after_stop(self):
        g = self.g(); g.poll(); self.foreign(); g.poll(); conf = g.poll()
        orig = self.o.kill

        def kill(pid, sig):
            orig(pid, sig)
            if pid == 100 and sig == signal.SIGTERM:
                self.o.gpu_fail = True                      # NVML breaks right after the stop
        self.o.kill = kill
        self.assertEqual(g.act(conf), 'kept'); self.assertIn(RECEIPT, self.o.ledger)
        self.assertIn('not proven absent', json.loads((self.t / 'state/ALERT.json').read_text())['reason'])

    def test_no_release_while_harness_wrapper_alive(self):
        self.o.wrapper_stays = True
        g = self.g(); g.poll(); self.foreign(); g.poll(); conf = g.poll()
        self.assertEqual(g.act(conf), 'kept'); self.assertIn(RECEIPT, self.o.ledger)
        self.assertIn(110, self.o.procs); self.assertNotIn(110, self.signalled())          # wrapper never signalled

    def test_gpu_query_failure_means_no_action(self):
        self.foreign(); self.o.gpu_fail = True
        self.assertEqual(self.g().run(max_polls=4), 'no_action'); self.assertEqual(self.o.kills, [])

    def test_foreign_while_idle_no_action(self):
        for p in (100, 110, 120, 130, 140, 150, 160):
            self.o.remove(p)
        self.foreign()
        self.assertEqual(self.g().run(max_polls=4), 'no_action')
        self.assertEqual(self.o.kills, []); self.assertEqual(self.o.harness_calls, [])

    def test_unverified_container_processes_never_signalled(self):
        self.o.container['Config']['Labels']['fold27.gpu'] = '2'
        g = self.g()
        self.assertIsNone(g.poll()); conf = g.poll()                  # its collector is unrecognised -> treated foreign
        self.assertEqual(conf, [(160, 16)])
        g.act(conf)
        self.assertNotIn(42, [c[1] for c in self.o.cexec]); self.assertIn(160, self.o.procs)
        self.assertNotIn(110, self.o.procs)                           # own eval stopped (safe direction)


if __name__ == '__main__':
    unittest.main()

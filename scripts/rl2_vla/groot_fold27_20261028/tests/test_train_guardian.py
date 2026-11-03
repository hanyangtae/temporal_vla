"""train_guardian on a fake host (no signal reaches a real process; no NVML, harness or docker).
Chain: coordinator -> wait_launch -> groot_harness run (receipt) -> train_queue_v2.py (consumer) -> finetune_fold_v2.py on
the GPU, next to the approved pinned root and a verified member of the approved container instance."""
import hashlib, json, shutil, signal, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tests'), str(ROOT / 'trqam'), str(ROOT / 'eval/runtime'), str(ROOT.parents[2] / 'scripts/utils')]
import test_foreign_gpu_guardian as T       # noqa: E402  (FakeOS)
import test_foreign_gpu_guardian_v2 as T2   # noqa: E402
import test_foreign_gpu_guardian_v3 as T3   # noqa: E402  (GProbe3)
import train_guardian as TG                 # noqa: E402

SESSION, RID = 'fold27-trqam-v2', 'T' * 32
CID = T3.CID
HARNESS_CMD = ('python3 scripts/utils/groot_harness.py run --machine srv50 --gpus 1 --serves 1 --session fold27-trqam-v2 '
               '--kind eval -- bash scripts/rl2_vla/groot_fold27_20261028/trqam/train_queue_v2.sh --sources S --folds F')


class HostFake(T.FakeOS):
    def __init__(self, deploy):
        super().__init__(deploy)
        self.wrapper_stays = False

    def start_epoch(self, pid):
        return None if pid not in self.procs else float(self.procs[pid].start)

    def own_uid(self):
        return 1000

    def remove(self, pid):
        p = self.procs.pop(pid, None)
        if p is None:
            return
        for q in self.procs.values():
            if q.ppid == pid:
                q.ppid = 1
        if pid == 230 and not self.wrapper_stays:          # the harness launcher exits with its consumer
            self.remove(220)


class TrainGuardianTest(unittest.TestCase):
    def setUp(self):
        self.t = Path(tempfile.mkdtemp())
        o = self.o = HostFake(str(self.t))
        o.add(200, 20, 1, 'python scripts/rl2_vla/groot_fold27_20261028/trqam/host_fold_coordinator_v2.py --machine srv50 --gpus 1')
        o.add(210, 21, 200, 'python scripts/rl2_vla/groot_fold27_20261028/trqam/wait_launch.py --max-wait-h 72 -- env GPUS=1 bash train_queue_v2.sh')
        o.add(220, 22, 210, HARNESS_CMD)
        o.add(230, 23, 220, '.fold27_env/bin/python scripts/rl2_vla/groot_fold27_20261028/trqam/train_queue_v2.py --python py --folds F')
        o.add(240, 24, 230, 'py scripts/rl2_vla/groot_fold27_20261028/trqam/finetune_fold_v2.py --fold F --arm basic')
        o.add(600, 60, 1, T2.APPROVED_CMD, uid=0, cgroup='0::/system.slice/isaac.scope')
        dock = f'0::/docker/{CID}'
        i = o.add(700, 70, 1, 'bash', uid=1019, cgroup=dock); i.gid, i.nsl = 1018, [700, 1]
        k = o.add(701, 71, 700, 'python -B -', uid=1019, cgroup=dock); k.gid, k.nsl = 1018, [701, 9]
        o.gpu = [240, 600, 701]
        o.ledger = {RID: {'machine': 'srv50', 'gpus': [1], 'serves': 1, 'session': SESSION, 'kind': 'eval', 'created': 25.0,
                          'launcher_host': 'worker2', 'launcher_pid': 220, 'consumer_pid': 230}}
        self.auth = {'kind': 'groot_gpu_sharing_exception_v2', 'machine': 'srv50', 'hostname': 'worker2', 'gpu': 1, 'gpu_uuid': T2.UUID,
                     'experiment': 'fold27', 'sessions': {'fold27-v2-eval': {'allowed_kind': 'eval', 'max_serves': 2, 'per_serve_mib': 6118},
                                                          SESSION: {'allowed_kind': 'eval', 'max_serves': 1, 'per_serve_mib': 12000}},
                     'reserve_mib': 8192, 'expires_at': o.t + 3600, 'boot_id': o.bootv, 'prepared_at': o.t, 'approval': 'user one-time',
                     'foreign': [{'pid': 600, 'uid': 0, 'start_ticks': 60, 'cmdline': T2.APPROVED_CMD, 'max_used_mib': 22000,
                                  'cmdline_sha256': hashlib.sha256(T2.APPROVED_CMD.encode()).hexdigest()}],
                     'containers': [{'id': CID, 'name': '/jolly_wright', 'image_id': 'sha256:img', 'started_at': 'x', 'cgroup': dock,
                                     'devices': [T2.UUID], 'member_uid': 1019, 'member_gid': 1018, 'max_used_mib': 8000,
                                     'init': {'pid': 700, 'start_ticks': 70, 'uid': 1019, 'gid': 1018}, 'launcher_evidence': ''}]}
        self.ap = self.t / 'auth.json'; self.ap.write_text(json.dumps(self.auth))
        self.cfg = {'kind': TG.KIND, 'host': 'srv50', 'gpu': 1, 'session': SESSION, 'sharing_authorization': str(self.ap),
                    'confirm_polls': 2, 'kill_wait_s': 10, 'release_wait_s': 30, 'poll_s': 5, 'clock_skew_s': 900}
        self.probe = T3.GProbe3(o); self.probe.used.update({701: 1500, 240: 9000})
        ck = self.t / 'run' / 'resume_v2'; ck.mkdir(parents=True)
        (ck / 'state_5000.pkl').write_bytes(b'state5000'); (ck / 'state_10000.pkl').write_bytes(b'state10000')
        self.files = {p: p.read_bytes() for p in self.t.rglob('*') if p.is_file()}

    def tearDown(self):
        for p, b in self.files.items():
            assert p.read_bytes() == b, f'guardian modified {p}'                    # checkpoints / contract untouched
        shutil.rmtree(self.t)

    def g(self, **kw):
        return TG.TrainGuardian({**self.cfg, **kw}, self.o, self.t / 'state', probe=self.probe)

    def signalled(self):
        return {k[0] for k in self.o.kills if not isinstance(k[0], tuple)} | {k[0][1] for k in self.o.kills if isinstance(k[0], tuple)}

    def never(self, *pids):
        for p in pids:
            self.assertNotIn(p, self.signalled(), p); self.assertIn(p, self.o.procs, p)

    def act(self, g):
        self.assertIsNone(g.poll()); r = g.poll(); self.assertTrue(r, r)
        return g.act(r)

    def test_no_action_with_approved_neighbours(self):
        g = self.g()
        self.assertEqual(g.run(max_polls=4), 'no_action'); self.assertEqual(self.o.kills, [])
        roles = sorted(r for _, _, r in json.loads(json.dumps([[p, s, r] for (p, s), r in g.owned.items()])))
        self.assertEqual(roles, ['queue', 'trainer'])

    def test_new_foreign_stops_only_our_training(self):
        self.o.add(900, 90, 1, 'python other.py', uid=1019, cgroup='0::/user.slice'); self.o.gpu.append(900)
        self.assertEqual(self.act(self.g()), 'released')
        k = self.o.kills
        self.assertEqual(k[:3], [(200, int(signal.SIGSTOP)), (210, int(signal.SIGSTOP)), (230, int(signal.SIGSTOP))])
        self.assertLess(k.index((240, int(signal.SIGTERM))), k.index((230, int(signal.SIGTERM))))
        self.never(600, 700, 701, 900)
        self.assertNotIn(220, self.signalled())                                        # harness launcher never signalled
        self.assertNotIn(RID, self.o.ledger)
        self.assertTrue(all(p not in self.o.procs for p in (200, 210, 220, 230, 240)))

    def test_own_memory_bound(self):
        self.probe.used[240] = 12001
        self.act(self.g()); self.never(600, 701)

    def test_headroom_bound(self):
        self.probe.free = 8191
        self.act(self.g()); self.never(600, 701)

    def test_restarted_container_is_foreign(self):
        self.o.procs[700].start = 99
        self.act(self.g()); self.never(600, 700, 701)

    def test_unproven_ownership_never_signals(self):
        muts = [lambda: setattr(self.o.procs[220], 'cmd', HARNESS_CMD.replace(SESSION, 'fold27-other')),
                lambda: setattr(self.o.procs[230], 'ppid', 1),
                lambda: self.o.ledger.update({'U' * 32: dict(self.o.ledger[RID])}),
                lambda: self.o.ledger[RID].update(created=-10000.0),
                lambda: setattr(self.o.procs[220], 'uid', 1019),
                lambda: self.o.ledger[RID].pop('consumer_pid'),
                lambda: self.o.ledger[RID].update(launcher_host='kanu')]
        for m in muts:
            self.tearDown(); self.setUp()
            m()
            self.o.add(900, 90, 1, 'python other.py', uid=1019); self.o.gpu.append(900)
            self.assertEqual(self.g().run(max_polls=4), 'no_action')
            self.assertEqual(self.o.kills, []); self.assertTrue((self.t / 'state' / 'ALERT.json').exists())

    def test_idle_foreign_no_action(self):
        self.o.ledger.clear()
        self.o.add(900, 90, 1, 'python other.py', uid=1019); self.o.gpu.append(900)
        self.assertEqual(self.g().run(max_polls=4), 'no_action'); self.assertEqual(self.o.kills, [])

    def test_reused_trainer_pid_is_not_ours(self):
        g = self.g(); g.poll()
        self.o.remove(240); self.o.add(240, 99, 1, 'python unrelated.py', uid=1000); self.probe.used[240] = 100
        r = g.poll(); self.assertIsNone(r); r = g.poll(); self.assertTrue(r)
        g.act(r)
        self.assertNotIn((240, int(signal.SIGTERM)), self.o.kills); self.assertIn(240, self.o.procs)

    def test_observe_only(self):
        self.o.add(900, 90, 1, 'python other.py', uid=1019); self.o.gpu.append(900)
        self.assertEqual(self.act(self.g(observe_only=True)), 'observed')
        self.assertEqual(self.o.kills, []); self.assertIn(RID, self.o.ledger)

    def test_release_kept_if_gpu_unverifiable_after_stop(self):
        self.o.add(900, 90, 1, 'python other.py', uid=1019); self.o.gpu.append(900)
        orig = self.o.kill

        def kill(pid, sig):
            orig(pid, sig)
            if pid == 240 and sig == signal.SIGTERM:
                self.probe.fail = True
        self.o.kill = kill
        g = self.g(); g.poll(); r = g.poll()
        self.probe.fail = False; self.assertEqual(g.act(r), 'kept')
        self.assertIn(RID, self.o.ledger)

    def test_probe_failure_no_action(self):
        self.probe.fail = True
        self.assertEqual(self.g().run(max_polls=4), 'no_action'); self.assertEqual(self.o.kills, [])

    def test_contract_scope_required(self):
        with self.assertRaises(SystemExit):
            self.g(session='fold27-other')
        v1 = dict(self.auth); v1['kind'] = 'groot_gpu_sharing_exception_v1'
        p = self.t / 'v1.json'; p.write_text(json.dumps(v1)); self.files[p] = p.read_bytes()
        with self.assertRaises(Exception):
            TG.TrainGuardian({**self.cfg, 'sharing_authorization': str(p)}, self.o, self.t / 'state', probe=self.probe)


class SamplerTest(unittest.TestCase):
    def test_peak_and_completion(self):
        import train_mem_sampler as S
        seq = iter([{'uuid': 'u', 'used_mib': 3000, 'total_mib': 81920, 'apps': {5: 2900}}, None,
                    {'uuid': 'u', 'used_mib': 9100, 'total_mib': 81920, 'apps': {5: 9000, 6: 50}}])
        alive = iter([True, True, True, False, False])
        t = [0.0]
        r = S.measure(3, 5, 0.5, 100, sample_fn=lambda g: next(seq), alive_fn=lambda p: next(alive),
                      now=lambda: t[0], sleep=lambda s: t.__setitem__(0, t[0] + s))
        self.assertEqual((r['peak_gpu_used_mib'], r['peak_per_pid_mib'], r['failed_samples'], r['complete']), (9100, {5: 9000, 6: 50}, 1, True))


if __name__ == '__main__':
    unittest.main()

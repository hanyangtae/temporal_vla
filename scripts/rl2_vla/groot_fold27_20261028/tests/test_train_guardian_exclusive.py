"""train_guardian_exclusive on the train_guardian fake host (no real signal / NVML / harness / process spawn).
Chain (fixture of test_train_guardian): coordinator 200 -> wait_launch 210 -> harness launcher 220 (receipt RID, serves 1)
-> train_queue_v2.py consumer 230 -> finetune_fold_v2.py trainer 240 on GPU1; the guardian itself is pid 800.
Covers: exclusive no-action + measurement; persist: foreign arrival never stops us, only memory protection does (and
only our identities are signalled); stop policy; own cap per slot; fail-closed GPU failure; normal end (harness released
/ receipt retained with a foreign process -> official release by the guardian / consumer exits before the launcher ->
no false alarm); orphan trainer after launcher death still protected via the cached proof; dead launcher pid reused ->
never walked; config validation; lock; admission wrapper (admit/arm/exec, wait cases, refusals); watchdog (healthy,
lost -> verified stop, hung -> stop guardian then verified stop, lock busy defers, completion exit)."""
import json, signal, sys, types, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tests'), str(ROOT / 'trqam'), str(ROOT / 'eval/runtime'), str(ROOT.parents[2] / 'scripts/utils')]
import test_train_guardian as TT          # noqa: E402  (HostFake, chain, GProbe3)
import test_foreign_gpu_guardian_v2 as T2  # noqa: E402  (UUID)
import train_guardian_exclusive as GX     # noqa: E402

SESSION, RID = TT.SESSION, TT.RID
APPROVAL = '남은 jug·marshmallow 모델 18개 학습을 50번으로 옮겨 진행 (test statement)'


class Exec(Exception):
    pass


class ExclusiveTest(TT.TrainGuardianTest):
    def setUp(self):
        super().setUp()
        o = self.o
        o.gpu = [240]                                                          # exclusive: only our trainer
        self.probe.used[240] = 5000; self.probe.free = 70000
        self.sd = self.t / 'gstate'
        self.cfgp = self.t / 'gx.json'
        self.ecfg = {'kind': GX.KIND, 'host': 'srv50', 'hostname': 'worker2', 'gpu': 1, 'gpu_uuid': T2.UUID, 'session': SESSION,
                     'per_job_mib': 7168, 'per_job_basis': 'measured_test', 'max_jobs': 6, 'reserve_mib': 16384,
                     'foreign_policy': 'persist', 'persist_approval': APPROVAL, 'persist_scope': 'training',
                     'confirm_polls': 2, 'poll_s': 5, 'kill_wait_s': 10, 'release_wait_s': 30, 'clock_skew_s': 900}
        self.cfgp.write_text(json.dumps(self.ecfg))
        o.add(800, 80, 1, f'python3 /r/{GX.SELF} watch --config {self.cfgp} --state-dir {self.sd}')
        self.open = []

    def tearDown(self):
        for g in self.open:
            g.close()
        super().tearDown()

    def gx(self, **kw):
        g = GX.ExclusiveTrainGuardian({**self.ecfg, **kw}, self.o, self.sd, probe=self.probe, me=(800, 80))
        self.open.append(g); return g

    def foreign(self, pid=900, used=6000):
        self.o.add(pid, 90, 1, 'python other_user_job.py', uid=1019); self.o.gpu.append(pid); self.probe.used[pid] = used

    def state(self, name):
        p = self.sd / name
        return json.loads(p.read_text()) if p.exists() else None

    def events(self):
        p = self.sd / 'events.jsonl'
        return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []

    def end_launch(self, keep_receipt):
        self.o.remove(240); self.o.remove(230)                                  # HostFake: launcher 220 exits with 230
        if not keep_receipt:
            self.o.ledger.clear()

    # ------------------------------------------------------------------ runtime
    def test_exclusive_no_action_and_measure(self):
        g = self.gx()
        self.assertEqual(g.run(max_polls=4), 'no_action'); self.assertEqual(self.o.kills, [])
        m = self.state('MEASURE.json')
        self.assertEqual(m['budget'], {'cap_mib': 7168, 'basis': 'measured_test', 'session': SESSION})
        self.assertEqual(m['measured']['own_used_peak_mib'], 5000)
        hb = self.state('heartbeat.json')
        self.assertEqual((hb['receipt'], hb['launcher'], hb['phase']), (RID, [220, 22], 'watch')); self.assertTrue(hb['last_ok'])

    def test_persist_foreign_arrival_keeps_training(self):
        g = self.gx(); g.poll()
        self.foreign()
        for _ in range(6):
            self.assertIsNone(g.poll())
        self.assertEqual(self.o.kills, [])
        self.assertTrue(any(e['kind'] == 'foreign_present' for e in self.events()))
        self.assertFalse(self.state('heartbeat.json')['last_ok'])              # admission of a new launch would wait

    def test_persist_headroom_stops_only_ours(self):
        g = self.gx(); g.poll()
        self.foreign(); self.probe.free = 16383
        self.assertIsNone(g.poll()); r = g.poll(); self.assertEqual(r, ['headroom breach'])
        self.assertEqual(g.act(r), 'released')                                  # official release after own absence
        self.assertNotIn(RID, self.o.ledger)
        self.never(900); self.assertNotIn(220, self.signalled())
        self.assertIn((240, int(signal.SIGTERM)), self.o.kills)
        self.assertEqual(self.state('HOLD.json')['reasons'], ['headroom breach'])

    def test_own_memory_cap_per_slot(self):
        g = self.gx(); self.probe.used[240] = 7169
        self.assertIsNone(g.poll()); self.assertEqual(g.poll(), ['own_mem breach'])
        self.tearDown(); self.setUp()
        self.o.ledger[RID]['serves'] = 6; self.probe.used[240] = 6 * 7168
        g = self.gx(); self.assertIsNone(g.poll()); self.assertIsNone(g.poll())  # 6 slots x 7168 = cap
        self.o.ledger[RID]['serves'] = 6
        self.tearDown(); self.setUp()
        self.ecfg['max_jobs'] = 3; self.o.ledger[RID]['serves'] = 6; self.probe.used[240] = 3 * 7168 + 1
        g = self.gx(); g.poll(); self.assertEqual(g.poll(), ['own_mem breach'])  # cap = min(serves, max_jobs)
        self.assertIn('exceed max_jobs', self.state('ADMISSION_UNSAFE.json')['reason'])

    def test_stop_policy_foreign_stops_ours(self):
        g = self.gx(foreign_policy='stop'); g.poll()
        self.foreign()
        self.assertIsNone(g.poll()); r = g.poll(); self.assertTrue(any(x.startswith('foreign (900') for x in r), r)
        g.act(r); self.never(900); self.assertNotIn(220, self.signalled())

    def test_gpu_failure_while_training_stops_ours_and_keeps_receipt(self):
        self.probe.fail = True
        self.assertEqual(self.gx().run(max_polls=4), 'kept')
        self.assertIn((240, int(signal.SIGTERM)), self.o.kills); self.assertIn(RID, self.o.ledger)
        self.assertTrue(self.state('ADMISSION_UNSAFE.json')); self.assertTrue(self.state('HOLD.json'))

    def test_normal_end_released_by_harness(self):
        g = self.gx(); g.poll()
        self.end_launch(keep_receipt=False)
        self.assertIsNone(g.poll())
        c = self.state(f'COMPLETED.{RID}.json'); self.assertEqual(c['result'], 'released_by_harness')
        self.assertIsNone(g.owner); self.assertEqual(self.o.kills, [])
        self.assertNotIn(('release', '--receipt', RID, '--session', SESSION), self.o.harness_calls)
        self.assertIsNone(self.state('ALERT.json'))
        g.poll(); hb = self.state('heartbeat.json')
        self.assertTrue(hb['last_ok']); self.assertIsNone(hb['receipt']); self.assertIn(RID, hb['completed'])

    def test_normal_end_receipt_retained_by_foreign_released_by_guardian(self):
        g = self.gx(); g.poll()
        self.foreign(); g.poll()
        self.end_launch(keep_receipt=True)                                      # harness kept it: GPU not empty at exit
        g.poll()
        self.assertIn(('release', '--receipt', RID, '--session', SESSION), self.o.harness_calls)
        self.assertNotIn(RID, self.o.ledger)
        self.assertEqual(self.state(f'COMPLETED.{RID}.json')['result'], 'released_by_guardian')
        self.never(900); self.assertEqual(self.o.kills, [])

    def test_consumer_exits_before_launcher_no_false_alarm(self):
        self.o.wrapper_stays = True
        g = self.gx(); g.poll()
        self.o.remove(240); self.o.remove(230)                                  # harness still in its release step
        self.assertIsNone(g.poll()); self.assertIsNone(self.state('ALERT.json'))
        self.o.remove(220); self.o.ledger.clear()
        g.poll(); self.assertEqual(self.state(f'COMPLETED.{RID}.json')['result'], 'released_by_harness')
        self.assertIsNone(self.state('ALERT.json')); self.assertIsNone(self.state('ADMISSION_UNSAFE.json'))

    def test_end_not_returned_while_own_training_on_gpu_or_entry_changed(self):
        g = self.gx(); g.poll()
        self.end_launch(keep_receipt=True)
        self.o.add(950, 95, 1, 'py scripts/rl2_vla/groot_fold27_20261028/trqam/finetune_fold_v2.py --fold X', uid=1000)
        self.o.gpu.append(950)                                                  # own-uid trainer never seen as owned
        g.poll(); g.poll()
        self.assertIn(RID, self.o.ledger); self.assertIn('not provable', self.state('ADMISSION_UNSAFE.json')['reason'])
        self.assertNotIn(950, self.signalled())
        self.tearDown(); self.setUp()
        g = self.gx(); g.poll()
        self.end_launch(keep_receipt=True); self.o.ledger[RID]['launcher_pid'] = 999
        g.poll(); g.poll(); self.assertIn(RID, self.o.ledger)
        self.assertIsNone(self.state(f'COMPLETED.{RID}.json'))

    def test_orphan_trainer_after_launcher_death_still_protected(self):
        g = self.gx(); g.poll()
        self.o.wrapper_stays = True; self.o.remove(220)                         # launcher killed; receipt retained
        self.probe.used[240] = 8000
        self.assertIsNone(g.poll()); r = g.poll(); self.assertEqual(r, ['own_mem breach'])
        self.assertEqual(g.act(r), 'released')
        self.assertIn((240, int(signal.SIGTERM)), self.o.kills); self.assertNotIn(RID, self.o.ledger)

    def test_dead_launcher_pid_reuse_never_walked(self):
        g = self.gx(); g.poll()
        self.o.wrapper_stays = True; self.o.remove(220)
        self.o.add(220, 99, 1, 'python unrelated.py', uid=1000)
        self.o.add(960, 96, 220, 'python unrelated_child.py', uid=1000); self.o.gpu.append(960)
        for _ in range(4):
            g.poll()
        self.assertFalse(any(p in (220, 960) for p, _ in g.owned)); self.assertEqual(self.o.kills, [])

    def test_config_validation(self):
        bad = [{'kind': 'fold27_train_guardian_v3'}, {'foreign_policy': None}, {'persist_approval': ''},
               {'persist_scope': 'eval'}, {'sharing_authorization': 'x.json'}, {'max_jobs': 7}, {'per_job_mib': 0},
               {'confirm_polls': 1}, {'gpu_uuid': ''}]
        for kw in bad:
            with self.assertRaises(SystemExit, msg=kw):
                GX.ExclusiveTrainGuardian({**self.ecfg, **kw}, self.o, self.sd, probe=self.probe, me=(800, 80), lock=False)

    def test_single_actor_lock(self):
        self.gx()
        with self.assertRaises(SystemExit):
            self.gx()

    def test_unproven_ownership_never_signals(self):
        self.o.ledger[RID]['launcher_host'] = 'kanu'
        self.probe.used[240] = 9000
        self.assertEqual(self.gx().run(max_polls=4), 'no_action'); self.assertEqual(self.o.kills, [])
        self.assertTrue(self.state('ADMISSION_UNSAFE.json'))

    # ------------------------------------------------------------------ admission
    def idle(self):
        self.o.ledger.clear(); self.o.gpu = []; self.probe.free = 81000
        for p in (240, 230, 220, 210, 200):
            self.o.remove(p)
        self.o.add(850, 85, 1, f'python3 /r/{GX.SELF} launch --config {self.cfgp}', uid=1000)
        g = self.gx(); g.poll(); g.poll()
        return g

    ENV = {'GPUS': '1', 'HARNESS_MACHINE': 'srv50', 'HARNESS_SESSION': SESSION, 'SERVES_PER_GPU': '6',
           'FOLD27_OUT': 'outputs/rl2_vla/x'}
    CMD = ['bash', 'scripts/rl2_vla/groot_fold27_20261028/trqam/train_queue_v2.sh', '--sources', 'S', '--out', 'O',
           '--folds', 'F1', 'F2', '--per-gpu', '6']

    def launch(self, env=None, cmd=None, arm=True):
        self.spawned, self.execd = [], []

        def spawn(c, **kw):
            self.spawned.append((c, kw))
            if arm:
                key = f'{c[c.index("--launch-pid") + 1]}.{c[c.index("--launch-start") + 1]}'
                GX.write_json(self.sd / 'watchdogs' / f'armed.{key}.json',
                              {'launch': {'pid': 850, 'start': 85}, 'watchdog': {'pid': 870, 'start': 87},
                               'guardian': {'pid': 800, 'start': 80}})
            return types.SimpleNamespace(pid=870)

        def execvpe(f, a, e):
            self.execd.append((f, a, e)); raise Exec()
        try:
            return GX.launch(self.cfgp, self.sd, ['--'] + (cmd or self.CMD), self.o, self.probe, env=env or dict(self.ENV),
                             spawn=spawn, execvpe=execvpe, me=(850, 85), python='python3')
        except Exec:
            return 'exec'

    def test_launch_admits_arms_watchdog_then_execs(self):
        self.idle()
        self.assertEqual(self.launch(), 'exec')
        c, kw = self.spawned[0]
        self.assertTrue(kw['start_new_session'])
        self.assertEqual(c[2:4], ['watchdog', '--config'])
        self.assertEqual(c[c.index('--guardian-pid') + 1:c.index('--guardian-pid') + 2], ['800'])
        self.assertEqual(self.execd[0][1], self.CMD)
        adm = json.loads((self.sd / 'watchdogs' / 'ADMIT.850.85.json').read_text())
        self.assertEqual(adm['evidence']['need_mib'], 6 * 7168 + 16384)

    def test_launch_waits_transient(self):
        cases = [lambda: (self.o.gpu.append(900), self.o.add(900, 90, 1, 'python x.py', uid=1019)),
                 lambda: self.o.ledger.update({'Z' * 32: {'machine': 'srv50', 'gpus': [1], 'serves': 1, 'session': 'other', 'kind': 'eval'}}),
                 lambda: setattr(self.probe, 'free', 6 * 7168 + 16383),
                 lambda: setattr(self.probe, 'fail', True)]
        for mut in cases:
            self.tearDown(); self.setUp(); self.idle()
            mut()
            self.assertEqual(self.launch(), 2); self.assertEqual((self.spawned, self.execd), ([], []))
        self.tearDown(); self.setUp(); g = self.idle()
        self.foreign(); g.poll()                                                # guardian saw a foreign process
        self.o.gpu = []; self.assertEqual(self.launch(), 2)

    def test_launch_refusals(self):
        env_bad = [{'GPUS': '2'}, {'HARNESS_SESSION': 'other'}, {'SERVES_PER_GPU': '7'}, {'SERVES_PER_GPU': ''},
                   {'HARNESS_SHARING_AUTH': '/c.json'}, {'HARNESS_RECEIPT': 'r'}]
        cmd_bad = [['python', 'x.py'], self.CMD[:-2], self.CMD[:-1] + ['7'], self.CMD + ['--gpus', '1']]
        for kw in env_bad:
            self.tearDown(); self.setUp(); self.idle()
            self.assertEqual(self.launch(env={**self.ENV, **kw}), 3, kw); self.assertEqual(self.execd, [])
        for cmd in cmd_bad:
            self.tearDown(); self.setUp(); self.idle()
            self.assertEqual(self.launch(cmd=cmd), 3, cmd); self.assertEqual(self.execd, [])
        for f in ('ALERT.json', 'HOLD.json', 'ADMISSION_UNSAFE.json'):
            self.tearDown(); self.setUp(); self.idle()
            (self.sd / f).write_text('{}'); self.assertEqual(self.launch(), 3, f)
        self.tearDown(); self.setUp(); self.idle()
        self.o.procs[800].cmd += ' --observe-only'; self.assertEqual(self.launch(), 3)
        self.tearDown(); self.setUp(); self.idle()
        self.o.procs[800].start = 81; self.assertEqual(self.launch(), 3)          # guardian pid reused
        self.tearDown(); self.setUp(); self.idle()
        self.o.t += 10000; self.assertEqual(self.launch(), 3)                   # stale heartbeat
        self.tearDown(); self.setUp(); self.idle()
        self.probe.uuid = 'GPU-other'; self.assertEqual(self.launch(), 3)
        self.tearDown(); self.setUp(); self.idle()
        self.assertEqual(self.launch(arm=False), 3); self.assertEqual(self.execd, [])   # watchdog never armed
        self.tearDown(); self.setUp(); self.idle()
        self.assertEqual(self.launch(cmd=['bash', 'x/smoke_v2b.sh', 'F', 'basic', 'E', 'L'],
                                     env={**self.ENV, 'SERVES_PER_GPU': '1'}), 'exec')

    # ------------------------------------------------------------------ watchdog
    def wdog(self):
        self.o.add(870, 87, 1, f'python3 /r/{GX.SELF} watchdog', uid=1000)
        w = GX.Watchdog(self.cfgp, self.sd, self.o, self.probe, (220, 22), (800, 80), me=(870, 87))
        self.open.append(w); return w

    def test_watchdog_healthy_no_action(self):
        g = self.gx(); w = self.wdog()
        for _ in range(6):
            g.poll(); self.assertIsNone(w.check()); self.o.t += 5
        self.assertEqual(self.o.kills, [])

    def test_watchdog_guardian_lost_stops_ours_only(self):
        g = self.gx(); g.poll(); w = self.wdog()
        self.foreign()
        self.o.remove(800); g.close()                                           # guardian process died (lock dropped)
        self.assertEqual(w.run(max_polls=8), 'watchdog_released')
        self.assertIn((240, int(signal.SIGTERM)), self.o.kills); self.never(900)
        self.assertNotIn(220, self.signalled()); self.assertTrue(self.state('HOLD.json'))

    def test_watchdog_hung_guardian_stopped_then_verified_stop(self):
        g = self.gx(); g.poll(); w = self.wdog()
        orig = self.o.kill

        def kill(pid, sig):
            orig(pid, sig)
            if pid == 800:
                g.close()                                                       # kernel drops the dead guardian's flock
        self.o.kill = kill
        for _ in range(12):
            self.o.t += 200
            r = w.check()
            if r:
                break
        self.assertEqual(self.o.kills[0], (800, int(signal.SIGTERM)))
        self.assertEqual(r, 'watchdog_released'); self.assertIn((240, int(signal.SIGTERM)), self.o.kills)

    def test_watchdog_lock_busy_defers(self):
        g = self.gx(); g.poll(); w = self.wdog()
        self.o.remove(800)                                                      # identity gone but lock still held
        self.assertIsNone(w.run(max_polls=8)); self.assertEqual(self.o.kills, [])

    def test_watchdog_exits_after_completion(self):
        g = self.gx(); g.poll(); w = self.wdog()
        self.end_launch(keep_receipt=False)
        self.assertIsNone(w.check())
        self.o.t += 1; g.poll()
        self.assertEqual(w.check(), 'launch_completed'); self.assertEqual(self.o.kills, [])


for _n in [n for n in dir(TT.TrainGuardianTest) if n.startswith('test')]:
    if getattr(ExclusiveTest, _n) is getattr(TT.TrainGuardianTest, _n):
        setattr(ExclusiveTest, _n, None)                                     # contract fixtures run in their own module


if __name__ == '__main__':
    unittest.main()

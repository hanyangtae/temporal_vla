"""job_claims + eval_gpu_queue (per-GPU consumers sharing work): race / stale / crash / fair-share / failure / port / disk."""
import json, os, shutil, sys, tempfile, threading, time, types, unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'eval/runtime'))
import job_claims as JC        # noqa: E402
import eval_gpu_queue as Q     # noqa: E402

EP = Q.EP
ARMS = ('mil_qam65', 'mil_trqam_basic', 'mil_trqam_pbrs', 'mil_trqam_mix')


class FakePipeline:
    def __init__(self, root, folds):
        self.root = Path(root); self.folds = folds
        self.complete, self.verified, self.lk = set(), Counter(), threading.Lock()

    def fdir(self, fid):
        return self.root / fid

    def failed(self, host):
        d = self.root / 'queue' / host / 'failed'
        return {tuple(p.name.split('.')[:2]) for p in d.glob('*.json')} if d.exists() else set()

    def jobs(self, host, retry_failed=False, exclude=()):
        bad = set() if retry_failed else self.failed(host)
        return [(f, a) for f in self.folds for a in ARMS if (f, a) not in self.complete and (f, a) not in bad and (f, a) not in exclude]

    def arm_complete(self, fid, arm):
        return (fid, arm) in self.complete

    def maybe_verify(self, fid):
        with self.lk:
            if all((fid, a) in self.complete for a in ARMS) and not self.verified[fid]:
                self.verified[fid] += 1
                return True
            if all((fid, a) in self.complete for a in ARMS):
                self.verified[fid] += 1                 # a second terminal verify would show up as 2
            return None


class Proc:
    def __init__(self, rc, on_done, delay=0.02):
        self.pid, self.rc, self.t0, self.delay, self.on_done, self.fired = os.getpid(), rc, time.time(), delay, on_done, False

    def poll(self):
        if time.time() - self.t0 < self.delay:
            return None
        if not self.fired:
            self.fired = True; self.on_done()
        return self.rc

    def wait(self):
        while self.poll() is None:
            time.sleep(0.005)
        return self.rc


class EvalGpuQueueTest(unittest.TestCase):
    def setUp(self):
        self.t = Path(tempfile.mkdtemp())
        self.p = FakePipeline(self.t, ['F1', 'F2'])
        self.runs, self.ports, self.lk = Counter(), {}, threading.Lock()
        self.fail = set()
        self._rel = Q.C.rel; Q.C.rel = lambda p: str(p)          # tmp dirs are outside the repo

    def tearDown(self):
        Q.C.rel = self._rel
        shutil.rmtree(self.t)

    def popen(self, gpu, delay=0.02):
        def f(cmd, stdout=None, stderr=None, cwd=None):
            fid, arm = cmd[cmd.index('--manifest') + 1].split('/')[-2], cmd[cmd.index('--arm') + 1]
            self.assertTrue(cmd[2].endswith('run_eval_arm_v2.py'))
            self.assertEqual(int(cmd[cmd.index('--gpu') + 1]), gpu)
            port = int(cmd[cmd.index('--port') + 1])
            flag = Path(cmd[cmd.index('--ready-flag') + 1]); flag.touch()
            with self.lk:
                self.runs[(fid, arm)] += 1; self.ports.setdefault(gpu, set()).add(port)
            rc = 1 if (fid, arm) in self.fail else 0

            def done():
                if rc == 0:
                    self.p.complete.add((fid, arm))
            return Proc(rc, done, delay)
        return f

    def consume(self, gpu, pool=(5, 6, 7), **kw):
        return Q.consume(self.p, 'kanu', gpu, 2, list(pool), receipt=f'r{gpu}', session='fold27-v2-eval', popen=self.popen(gpu, kw.pop('delay', 0.02)),
                         port_free=lambda port: kw.get('port_ok', True), sleep=lambda s: time.sleep(0.001), py='py',
                         **{k: v for k, v in kw.items() if k != 'port_ok'})

    def qdir(self):
        return self.t / 'queue' / 'kanu'

    # ----------------------------------------------------------------------------------------------------------------
    def test_two_gpu_consumers_race_no_duplicate_and_disjoint_ports(self):
        rcs = {}
        ts = [threading.Thread(target=lambda g=g: rcs.__setitem__(g, self.consume(g))) for g in (5, 6)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        self.assertEqual(rcs, {5: 0, 6: 0})
        self.assertEqual(set(self.runs), {(f, a) for f in ('F1', 'F2') for a in ARMS})
        self.assertTrue(all(v == 1 for v in self.runs.values()), self.runs)            # each (fold, arm) exactly once
        self.assertEqual(self.p.verified, Counter({'F1': 1, 'F2': 1}))                  # verify once per fold
        self.assertTrue(self.ports[5].isdisjoint(self.ports.get(6, set())))
        self.assertEqual(Q.gpu_ports('kanu', 5, 2), [8970, 8971]); self.assertEqual(Q.gpu_ports('kanu', 6, 2), [8972, 8973])
        self.assertEqual(list((self.qdir() / 'claims').glob('*.claim')), [])            # all released
        done = [json.loads(p.read_text()) for p in (self.qdir() / 'claims/done').glob('*.json')]
        self.assertEqual(sum(1 for d in done if d['key'][1] != '__verify__' and d['outcome'] == 'complete'), 8)
        term = Counter(d['key'][0] for d in done if d['key'][1] == '__verify__' and d['outcome'] in ('verified', 'verify_failed'))
        self.assertEqual(term, Counter({'F1': 1, 'F2': 1}))
        self.assertFalse((self.qdir() / 'QUEUE_STATE.json').exists())                   # frozen queue's file untouched

    def test_live_claim_blocks_and_dead_claim_is_reclaimed_not_deleted(self):
        st = JC.ClaimStore(self.t / 'c')
        self.assertTrue(st.try_claim(('F1', 'mil_qam65'), gpu=5))
        self.assertFalse(st.try_claim(('F1', 'mil_qam65'), gpu=6))                       # own pid alive -> blocks
        dead = JC.ClaimStore(self.t / 'c', start_fn=lambda pid: None)                    # every pid looks dead
        self.assertTrue(dead.try_claim(('F1', 'mil_qam65'), gpu=6))
        self.assertEqual(len(list((self.t / 'c/stale').glob('*.json'))), 1)
        # consumer dead but child (run_eval_arm) alive -> still live (no duplicate of an orphaned running arm)
        st2 = JC.ClaimStore(self.t / 'd')
        st2.try_claim(('F2', 'mil_qam65'), pid=999999999); st2.set_child(('F2', 'mil_qam65'), os.getpid())
        self.assertFalse(st2.try_claim(('F2', 'mil_qam65')))
        # other boot -> stale
        other = JC.ClaimStore(self.t / 'd', boot_fn=lambda: 'other-boot')
        self.assertTrue(other.try_claim(('F2', 'mil_qam65')))

    def test_crash_after_result_before_marker_reruns_once(self):
        st = JC.ClaimStore(self.qdir() / 'claims')
        st.try_claim(('F1', 'mil_trqam_mix'), pid=999999999)                             # crashed consumer, no child
        self.assertEqual(self.consume(5), 0)
        self.assertEqual(self.runs[('F1', 'mil_trqam_mix')], 1)                          # reclaimed and run (resume = run_eval_arm's job)
        self.assertEqual(len(list((self.qdir() / 'claims/stale').glob('*.json'))), 1)

    def test_failed_job_marker_not_done_not_retried(self):
        self.fail = {('F1', 'mil_trqam_pbrs')}
        self.assertEqual(self.consume(5), 4)
        self.assertEqual(self.runs[('F1', 'mil_trqam_pbrs')], 1)
        self.assertNotIn(('F1', 'mil_trqam_pbrs'), self.p.complete)
        self.assertEqual(self.p.verified['F1'], 0); self.assertEqual(self.p.verified['F2'], 1)
        self.assertTrue(list((self.qdir() / 'failed').glob('F1.mil_trqam_pbrs.*')))
        self.assertEqual(self.consume(6), 0); self.assertEqual(self.runs[('F1', 'mil_trqam_pbrs')], 1)   # other consumers skip it too

    def test_port_busy_and_disk_gate(self):
        self.assertEqual(self.consume(5, port_ok=False), 4)
        self.assertEqual(sum(self.runs.values()), 0)
        orig = EP.shutil.disk_usage
        EP.shutil.disk_usage = lambda p: types.SimpleNamespace(free=0)
        try:
            p2 = FakePipeline(self.t / 'x', ['F3']); self.p = p2
            self.assertEqual(self.consume(5), 0); self.assertEqual(sum(self.runs.values()), 0)
        finally:
            EP.shutil.disk_usage = orig

    def request(self, want=1):
        q = self.qdir(); q.mkdir(parents=True, exist_ok=True)
        (q / 'TRAIN_REQUEST.json').write_text(json.dumps({'trainer': JC.ident(os.getpid()), 'boot_id': JC.boot_id(), 'want': want}))

    def test_fair_share_yield_at_arm_boundary(self):
        self.p = FakePipeline(self.t, ['F1'])                                            # demand 4 -> share 2 of 3
        st = JC.ClaimStore(self.qdir() / 'claims'); co = Q.Coordination(self.qdir(), st)
        co.register(6, 'r6'); co.register(7, 'r7')                                       # two other live eval consumers
        self.request()
        self.assertEqual(self.consume(5), 0)
        self.assertLessEqual(sum(self.runs.values()), 2)                                 # only arms taken before the first completion
        self.assertTrue(all(self.p.arm_complete(*k) for k in self.runs))                # running arms finished, none interrupted
        st5 = json.loads((self.qdir() / 'QUEUE_STATE.gpu5.json').read_text())
        self.assertEqual(st5['exit'], 'yielded to trainer')
        self.assertTrue(list((self.qdir() / 'consumers').glob('gpu5.exited.*.json')))

    def test_within_share_keeps_working_and_stale_request_ignored(self):
        self.p = FakePipeline(self.t, ['F1'])
        st = JC.ClaimStore(self.qdir() / 'claims'); co = Q.Coordination(self.qdir(), st)
        self.request()                                                                  # only this consumer: always within share
        self.assertEqual(self.consume(5), 0); self.assertEqual(sum(self.runs.values()), 4)
        q = self.qdir() / 'TRAIN_REQUEST.json'
        q.write_text(json.dumps({'trainer': {'pid': 999999999, 'start': 1}, 'boot_id': JC.boot_id(), 'want': 3}))
        self.assertIsNone(co.train_request())                                           # dead trainer -> no yielding

    def test_may_start(self):
        q = self.qdir(); st = JC.ClaimStore(q / 'claims'); co = Q.Coordination(q, st)
        self.assertTrue(Q.may_start(self.p, 'kanu', 5, [5, 6, 7], 2, co, harness=lambda: {})[0])
        for f in ('F1', 'F2'):
            for a in ARMS:
                st.try_claim((f, a))
        self.assertFalse(Q.may_start(self.p, 'kanu', 5, [5, 6, 7], 2, co, harness=lambda: {})[0])            # everything claimed by live owner
        shutil.rmtree(q / 'claims'); st = JC.ClaimStore(q / 'claims'); co = Q.Coordination(q, st)
        co.register(6, 'r6'); co.register(7, 'r7'); self.request()
        self.assertFalse(Q.may_start(self.p, 'kanu', 5, [5, 6, 7], 2, co, harness=lambda: {})[0])            # 8 jobs -> share 2, already 2

    def test_worker_exception_is_marked_released_nonzero(self):
        def boom(cmd, **kw):
            raise OSError('exec failed')
        rc = Q.consume(self.p, 'kanu', 5, 2, [5, 6, 7], receipt='r5', session='s', popen=boom, port_free=lambda p: True,
                       sleep=lambda s: None, py='py')
        self.assertEqual(rc, 4)
        self.assertEqual(list((self.qdir() / 'claims').glob('*.claim')), [])
        self.assertTrue(list((self.qdir() / 'failed').glob('*.json')))

    def test_legacy_multi_gpu_receipt_blocks(self):
        q = self.qdir(); co = Q.Coordination(q, JC.ClaimStore(q / 'claims'))
        legacy = {'e7e3': {'machine': 'kanu', 'gpus': [5, 6, 7], 'serves': 2, 'session': 'fold27-v2-eval', 'kind': 'eval'}}
        ok, why = Q.may_start(self.p, 'kanu', 7, [5, 6, 7], 2, co, harness=lambda: legacy)
        self.assertFalse(ok); self.assertIn('legacy', why)
        single = {'x': {'machine': 'kanu', 'gpus': [5], 'serves': 2, 'session': 'fold27-v2-eval', 'kind': 'eval'}}
        self.assertTrue(Q.may_start(self.p, 'kanu', 7, [5, 6, 7], 2, co, harness=lambda: single)[0])

    def test_share_rule(self):
        self.assertEqual(JC.eval_share(0, 3, 2, True), 0); self.assertEqual(JC.eval_share(8, 3, 2, False), 3)
        self.assertEqual(JC.eval_share(8, 3, 2, True), 2); self.assertEqual(JC.eval_share(4, 3, 2, True), 2)
        self.assertEqual(JC.eval_share(2, 3, 2, True), 1); self.assertEqual(JC.eval_share(1, 1, 6, True), 1)
        self.assertEqual(JC.train_share(8, 3, 2), 1); self.assertEqual(JC.train_share(0, 3, 2), 3)


if __name__ == '__main__':
    unittest.main()

"""host_fold_coordinator_v5: arm-level single-GPU units under a fair share with eval (fakes: harness ledger, nvidia-smi,
eval demand, process spawn). Fixture reused from the v4 tests (real fold spec, real v2 queue freeze/DONE verification)."""
import json, os, sys, types, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tests'), str(ROOT / 'trqam'), str(ROOT)]
import test_host_coordinator_v4 as T4     # noqa: E402
import host_fold_coordinator_v5 as H5     # noqa: E402
import job_claims as JC                    # noqa: E402

TQ, Q2 = H5.TQ, H5.Q2


class CoordV5Test(T4.CoordV4Test):
    def setUp(self):
        super().setUp()
        self.ledger, self.units, self.conc = {}, [], []
        self.fail_arm, self.refuse_once = None, set()
        self.dur = 5.0

    def spawn(self):
        def sp(cmd, stdout=None, stderr=None, env=None, cwd=None):
            i = cmd.index('--')
            self.assertTrue(cmd[i + 2].endswith('train_arm_v5.sh'))
            inner = cmd[i + 3:]
            fid, arm, gpu = inner[inner.index('--fold') + 1], inner[inner.index('--arm') + 1], int(env['GPUS'])
            self.assertEqual((env['CUDA_VISIBLE_DEVICES'], env['JAX_PLATFORMS'], env['SERVES_PER_GPU'], env['OMP_NUM_THREADS']),
                             (str(gpu), 'cuda', '1', '2'))
            ref = Q2.TrainQueueV2(Q2.parse(['--spec', str(T4.SPEC), '--sources', str(self.t / 'sources.json'), '--out', str(self.out),
                                            '--init', str(self.init), '--gpus', '5'])[0], reuse_ae_from=Path('/v1root'))
            f = next(x for x in ref.spec['folds'] if x['fold_id'] == fid)
            self.assertEqual(inner, ref.cmd(f, arm, 'prod'))                 # exactly the queue's production job
            q = ref
            self.assertTrue(all(q.done(fid, a, 'dry') for a in TQ.ARMS))      # CPU dry done before any GPU claim
            key = (fid, arm)
            refused = key in self.refuse_once
            self.refuse_once.discard(key)
            if not refused:
                self.ledger[f'{fid}.{arm}'] = {'machine': 'kanu', 'gpus': [gpu], 'serves': 1, 'session': 'fold27-trqam-v2', 'kind': 'eval'}
            self.units.append((fid, arm, gpu, self.clock[0]))
            t_end = self.clock[0] + (0 if refused else self.dur)
            this = self

            class P:
                pid = os.getpid()

                def poll(s):
                    if this.clock[0] < t_end:
                        return None
                    this.ledger.pop(f'{fid}.{arm}', None)
                    if refused:
                        return 75
                    if this.fail_arm == key:
                        return 1
                    if not this.queue().done(fid, arm, 'prod'):
                        this.write(fid, arm, 'prod')
                    return 0
            return P()
        return sp

    def v5(self, demand, harness=None, extra=()):
        self.clock = clock = [0.0]

        def sleep(s):
            clock[0] += s
            live = [k for k, v in self.ledger.items() if v['session'] == 'fold27-trqam-v2']
            self.conc.append((clock[0], len(live)))
        argv = self.argv(*extra) + ['--eval-root', str(self.t / 'eval')]
        co = H5.HostCoordinatorV5(H5.parse(argv), popen=self.fake_popen(), run=None, spawn=self.spawn(), demand=demand,
                                  harness=harness or (lambda: self.ledger), busy=lambda gs: [], pending=lambda: (3, ''),
                                  sleep=sleep, now=lambda: clock[0])
        orig = co.run

        def go():
            try:
                return orig()
            finally:
                if co._lock:
                    co._lock.close(); co._lock = None
        co.run = go
        return co

    def req(self):
        return self.t / 'eval/queue/kanu/TRAIN_REQUEST.json'

    # ----------------------------------------------------------------------------------------------------------------
    def test_no_eval_demand_three_arms_parallel_each_fold(self):
        res = self.v5(lambda: 0).run()
        todo = [f for f in self.folds if f != T4.SMOKE]
        self.assertEqual(res['done_folds'][1:], todo)
        self.assertEqual(len(self.units), 3 * len(todo))
        first = [u for u in self.units if u[0] == todo[0]]
        self.assertEqual(sorted(u[2] for u in first), [5, 6, 7]); self.assertEqual(len({u[3] for u in first}), 1)
        self.assertLessEqual(max(c for _, c in self.conc), 3)
        self.assertFalse(self.req().exists()); self.assertEqual(self.violations, [])
        self.assertFalse((self.q / 'QUEUE_STATE.json').exists()); self.assertFalse((self.q / 'FAILED.json').exists())

    def test_fair_share_with_eval_then_full_after_drain(self):
        evalg = {'e5': {'machine': 'kanu', 'gpus': [5], 'serves': 2, 'session': 'fold27-v2-eval', 'kind': 'eval'},
                 'e6': {'machine': 'kanu', 'gpus': [6], 'serves': 2, 'session': 'fold27-v2-eval', 'kind': 'eval'}}
        reqs = []

        def demand():
            if self.req().exists():
                reqs.append(json.loads(self.req().read_text())['want'])
            return 4 if self.clock[0] < 12 else 0
        self.ledger.update(evalg)

        def harness():
            if self.clock[0] >= 12:
                self.ledger.pop('e5', None); self.ledger.pop('e6', None)
            return self.ledger
        self.v5(demand, harness=harness, extra=('--poll-s', '1')).run()
        todo = [f for f in self.folds if f != T4.SMOKE]
        first = sorted((u for u in self.units if u[0] == todo[0]), key=lambda u: u[3])
        self.assertEqual(first[0][2], 7); self.assertLess(first[0][3], 12)          # trainer used the idle GPU at once
        early = [c for t, c in self.conc if t < 12]
        self.assertLessEqual(max(early), 1)                                           # share 1 while eval has demand 4
        self.assertEqual(reqs, [])                                                     # trainer at its share: eval not asked to yield
        later = [u for u in self.units if u[3] >= 12]
        self.assertTrue({u[2] for u in later} >= {5, 6})                              # freed eval GPUs used for training
        self.assertFalse(self.req().exists()); self.assertEqual(self.violations, [])

    def test_request_published_when_below_share(self):
        full = {f'e{g}': {'machine': 'kanu', 'gpus': [g], 'serves': 2, 'session': 'fold27-v2-eval', 'kind': 'eval'} for g in (5, 6, 7)}
        seen = []

        def demand():
            if self.req().exists():
                seen.append((self.clock[0], json.loads(self.req().read_text())['want']))
            return 8
        self.ledger.update(full)

        def harness():
            if self.clock[0] >= 4:
                self.ledger.pop('e7', None)                                          # GPU7 consumer yielded at its arm boundary
            if self.clock[0] >= 40:
                self.ledger.pop('e5', None); self.ledger.pop('e6', None)
            return self.ledger
        self.v5(demand, harness=harness, extra=('--poll-s', '1')).run()
        self.assertTrue(seen and seen[0][1] == 1)                                     # share 1 of 3 with demand 8
        todo = [f for f in self.folds if f != T4.SMOKE]
        self.assertEqual(min(u[3] for u in self.units if u[0] == todo[0]), 4.0)      # launched as soon as GPU7 was handed back
        self.assertLessEqual(max(c for t, c in self.conc), 1)                         # never above share while demand persists
        self.assertFalse(self.req().exists())

    def test_harness_refusal_retries_and_failure_stops(self):
        todo = [f for f in self.folds if f != T4.SMOKE]
        self.refuse_once = {(todo[0], 'pbrs')}
        self.fail_arm = (todo[1], 'mix')
        with self.assertRaises(H5.HC.CoordError) as e:
            self.v5(lambda: 0).run()
        self.assertIn(f'train:{todo[1]}', str(e.exception))
        self.assertEqual(sum(1 for u in self.units if u[:2] == (todo[0], 'pbrs')), 2)     # refused once, then ran
        self.assertTrue(all(self.queue().done(todo[0], a, 'prod') for a in TQ.ARMS))
        self.assertFalse(any(u[0] == todo[2] for u in self.units))                     # nothing after the failed fold
        self.assertTrue((self.q / 'host_kanu_COORD_V5_FAILED.json').exists())
        self.assertFalse(self.req().exists())

    def test_live_orphan_arm_is_never_duplicated(self):
        todo = [f for f in self.folds if f != T4.SMOKE]
        st = JC.ClaimStore(self.q / 'train_claims')
        st.try_claim((todo[0], 'basic', 'prod'), pid=999999999); st.set_child((todo[0], 'basic', 'prod'), os.getpid())

        calls = [0]
        def demand():
            calls[0] += 1
            if calls[0] == 6:                                                          # orphan finishes its run
                self.write(todo[0], 'basic', 'prod')
                c = json.loads(st.path((todo[0], 'basic', 'prod')).read_text()); c['child'] = {'pid': 999999999, 'start': 1}
                st.path((todo[0], 'basic', 'prod')).write_text(json.dumps(c))
            return 0
        self.v5(demand).run()
        self.assertFalse(any(u[:2] == (todo[0], 'basic') for u in self.units))       # never relaunched while alive / once DONE

    def test_resume_skips_done_arms(self):
        todo = [f for f in self.folds if f != T4.SMOKE]
        for a in TQ.ARMS:
            self.write(todo[0], a, 'dry')
        self.write(todo[0], 'basic', 'prod')
        self.v5(lambda: 0).run()
        self.assertEqual(sorted(u[1] for u in self.units if u[0] == todo[0]), ['mix', 'pbrs'])


for name in [n for n in dir(T4.CoordV4Test) if n.startswith('test_')]:
    if not hasattr(CoordV5Test, name) or getattr(CoordV5Test, name) is getattr(T4.CoordV4Test, name):
        setattr(CoordV5Test, name, None)


if __name__ == '__main__':
    unittest.main()

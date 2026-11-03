"""v5 = host_fold_coordinator_v4 (CPU dry/AE prep one fold ahead, takeover, smoke/inputs/verify/freeze unchanged) with the
fold-level GPU launch replaced by ARM-level GPU units under a bounded fair share with eval.

Why: v4 launches train_queue_v2 once per fold on a fixed GPUS set and (via v3) waits until ALL READY eval drained. With
per-GPU eval consumers (eval/runtime/eval_gpu_queue.py) a fold-level launch on a partial set would serialize its arms.
v5 instead runs each arm of the current fold as its own unit, exactly the queue's production job:
  command = TrainQueueV2(...).cmd(fold, arm, 'prod') (+ the same --reuse-ae-from), env = TrainQueue.env(gpu)
  (JAX cuda, CUDA_VISIBLE_DEVICES=gpu, threads), launched as wait_launch.py -> train_arm_v5.sh (groot_launch_guard, kind
  eval, SERVES_PER_GPU=1, own single-GPU receipt) -> finetune_fold_v2.py. Recipe/seeds/fingerprints identical; resume is
  finetune_fold_v2's bitwise resume_v2; DONE = frozen TrainQueue.done (receipt, 50k, code pins, init sha, params sha).
Scheduling, per fold (folds stay sequential, CPU dry of the fold ensured before any GPU claim, next fold prepped on CPU):
  demand = incomplete non-failed READY eval jobs on this host;  cap = |pool| if demand == 0 else train_share(demand)
  while arms of this fold are not all DONE:
    launch an unlaunched arm on a free pool GPU (no harness reservation, no compute process) while running < cap;
    publish TRAIN_REQUEST.json(want = arms still waiting for a GPU, capped by share) so eval consumers hand GPUs back at
    their next completed arm; withdraw when nothing waits. A harness refusal (race) returns the arm to waiting (short
    wait_launch window), never a forced claim. Arm failure: no new arms, running arms finish, then the inherited fail path.
  Arm ownership = job_claims claim (fold, arm, 'prod') in <out>/queue/train_claims with the wait_launch pid as child, so a
  restarted coordinator never duplicates an orphaned running arm (its claim stays live until that process ends).
The train side writes no train_queue files (QUEUE_STATE/FAILED/FREEZE untouched; FREEZE still verified on every queue view).
State host_<m>_COORD_V5_STATE.json / host_<m>_COORD_V5_FAILED.json; takeover from v2/v3/v4 boundary stops as in v4.

    .venvs/rl2_iid/bin/python host_fold_coordinator_v5.py <same args as v4> [--eval-serves 2] [--arm-wait-s 180]
"""
import json, os, subprocess, sys, types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import host_fold_coordinator_v4 as H4    # noqa: E402  (frozen; subclassed)
sys.path.insert(0, str(HERE.parent / 'eval/runtime'))
import eval_pipeline as EP               # noqa: E402  (frozen; harness_state / gpu_busy reused read-only)
import job_claims as JC                  # noqa: E402

HC, TQ, Q2 = H4.HC, H4.TQ, H4.Q2


class HostCoordinatorV5(H4.HostCoordinatorV4):
    def __init__(self, a, harness=None, busy=None, demand=None, spawn=subprocess.Popen, **kw):
        super().__init__(a, **kw)
        v4 = (self.fail_p, self.state_p)
        self.state_p = self.qdir / f'host_{a.machine}_COORD_V5_STATE.json'
        self.fail_p = self.qdir / f'host_{a.machine}_COORD_V5_FAILED.json'
        self.stopper_files = [*self.stopper_files[:2], v4, (self.fail_p, self.state_p)]
        self.state = {'machine': a.machine, 'done_folds': [], 'version': 'v5', 'events': [], 'prep': {}, 'arms': []}
        self.harness, self.busy, self.spawn = harness or EP.harness_state, busy or EP.gpu_busy, spawn
        self.demand = demand or self._demand_cli
        root = Path(a.eval_root) if a.eval_root else EP.C.R / EP.PROD_ROOT
        self.eval_qdir = root / 'queue' / a.machine
        self.req_p = self.eval_qdir / 'TRAIN_REQUEST.json'
        self.claims = None                      # created after the single-writer lock is held

    # -- read-only views
    def _demand_cli(self):
        p = subprocess.run([self.a.python, str(EP.HERE / 'eval_pipeline.py'), '--trqam-root', str(self.a.out), 'pending',
                            '--host', self.a.machine], capture_output=True, text=True)
        if p.returncode == 3:
            return 0
        if p.returncode != 0:
            raise HC.CoordError(f'eval_pipeline pending rc={p.returncode}: {p.stderr.strip()[-300:]}')
        return len(json.loads(p.stdout)['pending'])

    def free_gpus(self, exclude=()):
        held = set(exclude)
        for it in self.harness().values():
            if it.get('machine') == self.a.machine:
                held.update(int(g) for g in it['gpus'])
        cand = [g for g in self.a.gpus if g not in held]
        bad = set(self.busy(cand)) if cand else set()
        return [g for g in cand if g not in bad]

    # -- train request (this process is its only writer)
    def publish(self, want, fid):
        if want <= 0:
            return self.withdraw()
        self.eval_qdir.mkdir(parents=True, exist_ok=True)
        body = {'trainer': JC.ident(os.getpid()), 'boot_id': JC.boot_id(), 'want': want, 'fold': fid, 'time': self.now()}
        tmp = self.req_p.with_suffix('.tmp'); tmp.write_text(json.dumps(body, indent=1)); os.replace(tmp, self.req_p)

    def withdraw(self):
        if self.req_p.exists():
            os.replace(self.req_p, self.req_p.with_name('TRAIN_REQUEST.last.json'))

    # -- one arm = one single-GPU unit
    def arm_cmd(self, q, f, arm, gpu):
        env = {**q.env(gpu), 'GPUS': str(gpu), 'HARNESS_MACHINE': self.a.machine, 'HARNESS_SESSION': self.a.session,
               'SERVES_PER_GPU': '1'}
        cmd = [self.a.python, str(HERE / 'wait_launch.py'), '--max-wait-h', str(self.a.arm_wait_s / 3600), '--poll-s', '30',
               '--state', str(self.qdir / 'wait_v5' / f'{f["fold_id"]}.{arm}'), '--', 'bash', str(HERE / 'train_arm_v5.sh'),
               *q.cmd(f, arm, 'prod')]
        return cmd, env

    def run_fold_arms(self, fid):
        q = self.dry_queue()
        f = next(x for x in self.spec['folds'] if x['fold_id'] == fid)
        logs = self.qdir / 'logs_v5'; logs.mkdir(parents=True, exist_ok=True)
        running, failed, deadline = {}, None, self.now() + self.a.max_wait_h * 3600
        while True:
            for arm, r in list(running.items()):
                rc = r['proc'].poll()
                if rc is None:
                    continue
                del running[arm]
                done = q.done(fid, arm, 'prod')
                self.claims.release((fid, arm, 'prod'), 'done' if done else f'rc={rc}')
                rec = {'fold': fid, 'arm': arm, 'gpu': r['gpu'], 'rc': rc, 'done': done, 'end': self.now(), 'start': r['start']}
                self.state['arms'].append(rec)
                if rc == 75 and not done:
                    continue                                   # harness refused (race): arm goes back to waiting
                if not done and failed is None:
                    failed = rec
            todo = [a for a in TQ.ARMS if a not in running and not q.done(fid, a, 'prod')]
            if failed is not None or not todo:
                if not running:
                    self.withdraw()
                    return types.SimpleNamespace(returncode=0 if failed is None else 2, failed=failed)
                self.publish(0, fid)
            else:
                try:
                    demand = self.demand()
                except HC.CoordError as e:
                    self.withdraw(); self.fail('fair_share', str(e))
                cap = len(self.a.gpus) if demand == 0 else JC.train_share(demand, len(self.a.gpus), self.a.eval_serves)
                slots = max(0, cap - len(running))
                free = self.free_gpus(exclude=[r['gpu'] for r in running.values()])
                for arm, gpu in zip(todo[:slots], free):
                    if not self.claims.try_claim((fid, arm, 'prod'), gpu=gpu, session=self.a.session):
                        continue                               # an earlier run's arm is still alive: never duplicate
                    if q.done(fid, arm, 'prod'):               # finished between the scan and the claim
                        self.claims.release((fid, arm, 'prod'), 'already_done'); continue
                    cmd, env = self.arm_cmd(q, f, arm, gpu)
                    log = logs / f'{fid}.{arm}.{len(list(logs.glob(f"{fid}.{arm}.*.log"))) + 1}.log'
                    with open(log, 'x') as fh:
                        fh.write('# ' + ' '.join(cmd) + '\n'); fh.flush()
                        proc = self.spawn(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(TQ.REPO))
                    self.claims.set_child((fid, arm, 'prod'), proc.pid)
                    running[arm] = {'proc': proc, 'gpu': gpu, 'start': self.now(), 'log': log.name}
                waiting = [a for a in TQ.ARMS if a not in running and not q.done(fid, a, 'prod')]
                self.publish(min(len(waiting), max(0, cap - len(running))), fid)
                if not running and self.now() > deadline:
                    self.withdraw(); self.fail(f'train:{fid}', f'no pool GPU for {self.a.max_wait_h} h (eval demand {demand})')
            if running:
                deadline = self.now() + self.a.max_wait_h * 3600     # bound only the time with NO arm making progress
            self.save(status='training', running={a: r['gpu'] for a, r in running.items()})
            self.sleep(self.a.poll_s)

    def _yield_then_launch(self, cmd, **kw):          # replaces v4 -> v3 eval drain + fold-level launch
        fid = cmd[cmd.index('--folds') + 1]
        if self.claims is None:
            self.claims = JC.ClaimStore(self.qdir / 'train_claims')
        self.ensure_dry(fid)                          # CPU, before any GPU claim
        self.start_prep(self.next_prep_fold(fid))
        try:
            res = self.run_fold_arms(fid)
        except BaseException:
            self.withdraw()
            if self._prep is not None:
                self.join_prep()
            raise
        if res.returncode:
            self.fail(f'train:{fid}', f'arm failed: {res.failed}')
        return res


def parse(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = {'--eval-serves': 2, '--arm-wait-s': 180.0, '--eval-root': None}
    for k in list(extra):
        if k in argv:
            i = argv.index(k); extra[k] = argv[i + 1]; del argv[i:i + 2]
    a = H4.parse(argv)
    a.eval_serves, a.arm_wait_s, a.eval_root = int(extra['--eval-serves']), float(extra['--arm-wait-s']), extra['--eval-root']
    return a


def main():
    a = parse()
    try:
        print(json.dumps(HostCoordinatorV5(a).run()))
    except (HC.CoordError, TQ.QueueError) as e:
        print('HOST_COORD_V5_ERROR', e, file=sys.stderr); sys.exit(2)


if __name__ == '__main__':
    main()

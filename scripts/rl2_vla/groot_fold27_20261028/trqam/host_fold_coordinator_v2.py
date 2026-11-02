"""v2 per-host coordinator: ONE train_queue_v2 process at a time per out root (exclusive lock), one fold per launch with
its 3 arms in parallel (GPUS=5,6,7 --per-gpu 1), all GPUs released at each fold boundary for the eval queue.

Why: train_queue's QUEUE_STATE/FAILED/QUEUE_FREEZE live in one <out>/queue dir, so independent fold queues on the same
out root must never run concurrently. Recipe/freeze/done-verification = frozen train_queue (+ v2 pins) unchanged.

Phases (state <out>/queue/host_<m>_COORD_V2_STATE.json, failure host_<m>_COORD_V2_FAILED.json, lock host_coord_v2.lock):
  smoke   CPU wait until --smoke-fold has all 3 arms SMOKE_DONE verified (train_queue done('smoke')) and
          <out>/<smoke-fold>/resume_check_v2/RESUME_CHECK.json passed all gates; any <out>/<smoke-fold>/<tag>_smoke/FAILED.json
          or a failed resume check -> stop. This coordinator never launches a smoke itself.
  fold    for each host fold in order (skip 3/3 DONE): wait inputs (wait_ready.not_ready) -> wait_launch.py ->
          train_queue_v2.sh --folds <fold> --per-gpu 1 [--reuse-ae-from v1] -> verify 3 DONE.
Failure handling as v1 coordinator (queue FAILED.json, --fail-marker, harness rc 75, deadlines; --retry-failed keeps old).

    .venvs/rl2_iid/bin/python host_fold_coordinator_v2.py --machine kanu --sources <kanu sources> --gpus 5 6 7 \
        --smoke-fold OpenDrawer-left_s2_j2 --reuse-ae-from outputs/rl2_vla/fold27_trqam_20261102
"""
import fcntl, json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import host_fold_coordinator as HC   # noqa: E402  (frozen v1: helpers reused)
import train_queue_v2 as Q2          # noqa: E402  (patches pins for v2)
TQ = Q2.TQ

RESUME_GATES = ('restore_exact', 'minibatch_indices_equal', 'np_rng_state_equal', 'trajectory_bitwise_equal')


class HostCoordinatorV2(HC.HostCoordinator):
    def __init__(self, a, **kw):
        super().__init__(a, **kw)
        self.state_p = self.qdir / f'host_{a.machine}_COORD_V2_STATE.json'
        self.fail_p = self.qdir / f'host_{a.machine}_COORD_V2_FAILED.json'
        self.state = {'machine': a.machine, 'done_folds': [], 'version': 'v2'}
        self._lock = None

    def queue(self):
        args, _ = Q2.parse(['--spec', str(self.a.spec), '--sources', str(self.a.sources), '--out', str(self.a.out),
                            '--init', str(self.a.init), '--dry-only'])
        return Q2.TrainQueueV2(args)

    def fold_done(self, fid):
        q = self.queue()
        return all(q.done(fid, arm, 'prod') for arm in TQ.ARMS)

    def acquire_lock(self):
        self._lock = open(self.qdir / 'host_coord_v2.lock', 'a+')
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise HC.CoordError(f'another v2 coordinator holds {self.qdir}/host_coord_v2.lock (one queue per out root)')
        self._lock.seek(0); self._lock.truncate(); self._lock.write(str(os.getpid())); self._lock.flush()

    def smoke_ready(self):
        f = self.a.smoke_fold
        for arm, tag in TQ.TAGS.items():
            if (self.a.out / f / f'{tag}_smoke' / 'FAILED.json').exists():
                self.fail('smoke', f'{arm} smoke FAILED.json present')
        rc = self.a.out / f / 'resume_check_v2' / 'RESUME_CHECK.json'
        if rc.exists():
            r = json.loads(rc.read_text())
            bad = [g for g in RESUME_GATES if r.get(g) is not True] + ([] if r.get('backend') == 'gpu' else ['backend'])
            if bad:
                self.fail('smoke', f'GPU resume check failed gates {bad} (max_abs {r.get("trajectory_max_abs_diff")})')
        q = self.queue()      # creates/validates the v2 QUEUE_FREEZE (same pins smoke_v2 runs were fingerprinted with)
        missing = [arm for arm in TQ.ARMS if not q.done(f, arm, 'smoke')]
        if missing or not rc.exists():
            return False, {'smoke_missing': missing, 'resume_check': rc.exists()}
        return True, 'smokes + resume check passed'

    def run(self):
        self.acquire_lock()
        if self.fail_p.exists():
            if not self.a.retry_failed:
                raise HC.CoordError(f'{self.fail_p} present; inspect, then --retry-failed')
            os.replace(self.fail_p, self.fail_p.with_name(f'{self.fail_p.stem}.prev.{int(self.now())}.json'))
        folds = self.host_folds()
        if not folds:
            raise HC.CoordError(f'no folds for {self.a.machine}')
        self.save(status='started', plan=folds, smoke_fold=self.a.smoke_fold)
        self.wait_until('smoke', self.smoke_ready, 'smoke gate')
        for fid in folds:
            if self.fold_done(fid):
                self.state['done_folds'].append(fid); continue
            self.wait_until(f'inputs:{fid}', lambda: self.inputs_ready(fid), 'inputs not ready')
            self.save(status='training', phase=f'train:{fid}')
            cmd = [self.a.python, str(HERE / 'wait_launch.py'), '--max-wait-h', str(self.a.max_wait_h), '--poll-s', str(self.a.harness_poll_s),
                   '--state', str(self.qdir / 'wait'), '--', 'env', f'GPUS={",".join(map(str, self.a.gpus))}',
                   f'HARNESS_MACHINE={self.a.machine}', f'HARNESS_SESSION={self.a.session}',
                   *([f'FOLD27_PY={self.a.python}'] if self.a.machine != 'kanu' else []),
                   'bash', str(HERE / 'train_queue_v2.sh'), '--sources', str(self.a.sources), '--out', str(self.a.out),
                   '--init', str(self.a.init), '--folds', fid, '--per-gpu', '1',
                   *(['--reuse-ae-from', str(self.a.reuse_ae_from)] if self.a.reuse_ae_from else [])]
            rc = self.run_cmd(cmd).returncode
            if rc == 75:
                self.fail(f'train:{fid}', 'harness still BLOCKED at the wait deadline')
            if rc != 0:
                self.check_failures(f'train:{fid}')
                self.fail(f'train:{fid}', f'train_queue_v2 rc={rc}')
            if not self.fold_done(fid):
                self.fail(f'verify:{fid}', 'train_queue_v2 returned 0 but not all 3 arms are verified DONE')
            self.state['done_folds'].append(fid)
            self.save(status='fold_done', phase=f'done:{fid}')
        self.save(status='done')
        return {'machine': self.a.machine, 'done_folds': self.state['done_folds']}


def parse(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = {}
    for k in ('--smoke-fold', '--reuse-ae-from'):
        if k in argv:
            i = argv.index(k); extra[k] = argv[i + 1]; del argv[i:i + 2]
    if '--smoke-fold' not in extra:
        raise SystemExit('--smoke-fold required (fold whose 3 v2 smokes + resume check gate production)')
    if '--out' not in argv:
        argv += ['--out', str(Q2.V2_OUT)]
    a = HC.parse(argv)
    a.smoke_fold, a.reuse_ae_from = extra['--smoke-fold'], (Path(extra['--reuse-ae-from']) if '--reuse-ae-from' in extra else None)
    a.after_fold = None
    if a.session == 'fold27-trqam':
        a.session = 'fold27-trqam-v2'
    return a


def main():
    a = parse()
    try:
        print(json.dumps(HostCoordinatorV2(a).run()))
    except (HC.CoordError, TQ.QueueError) as e:
        print('HOST_COORD_V2_ERROR', e, file=sys.stderr); sys.exit(2)


if __name__ == '__main__':
    main()

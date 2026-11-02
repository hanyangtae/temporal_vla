"""v3 = host_fold_coordinator_v2 (unchanged fold logic) + eval priority at fold boundaries + lock-wait takeover from a live v2.

Differences from v2 only:
  * own files: state host_<m>_COORD_V3_STATE.json, failure host_<m>_COORD_V3_FAILED.json. The live v2's
    COORD_V2_STATE / COORD_V2_FAILED are only READ (kept as evidence), never written.
  * lock: WAITS (bounded --lock-wait-h) for the same single-writer lock queue/host_coord_v2.lock, writing NOTHING while
    waiting (no state, no queue file) -> safe to start while v2 still runs; proceeds only after v2 has exited.
  * handover (after the lock is held): a <out>/queue/FAILED.json is moved aside to HANDOVER_STOP.<ts>.json (kept) ONLY if
      - it is a handover_boundary.py sentinel (handover_sentinel: true; a real train_queue failure overwrites it -> no key),
      - the fold named in expected_stop_after ('train:<fold>') is 3/3 verified DONE,
      - v2 stopped at a boundary: COORD_V2_FAILED phase 'inputs:*' / 'smoke' citing the sentinel, or v2 state 'done',
      - no queue job is running (QUEUE_STATE running == []).
    Otherwise it is left untouched and v3 stops (same as v2 would with a real FAILED.json).
  * eval yield: right before EACH fold launch (the moment GPUs would be claimed), wait while
    `eval_pipeline.py --trqam-root <out> pending --host <machine>` exits 0 (READY eval work on this host). Exit 3 -> launch;
    any other exit -> stop (fail-closed). Bounded by --yield-max-h per boundary (then COORD_V3_FAILED; never starves silently).
    The eval pipeline is read-only here (pending = jobs() scan); v3 never launches eval or touches its files.
Everything else (smoke gate, inputs wait, train_queue_v2 --per-gpu 1 on --gpus, DONE verification, failure markers,
QUEUE_FREEZE, recipe) is inherited from v2 unchanged.

    .venvs/rl2_iid/bin/python host_fold_coordinator_v3.py --machine kanu --sources <kanu sources> --gpus 5 6 7 \
        --smoke-fold OpenDrawer-left_s2_j2 --reuse-ae-from outputs/rl2_vla/fold27_trqam_20261102 --session fold27-trqam-v2
"""
import fcntl, json, os, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import host_fold_coordinator_v2 as H2   # noqa: E402  (frozen; subclassed)

HC, TQ = H2.HC, H2.TQ
EVAL_PIPELINE = HERE.parent / 'eval/runtime/eval_pipeline.py'
SENTINEL_KEY = 'handover_sentinel'


class HostCoordinatorV3(H2.HostCoordinatorV2):
    def __init__(self, a, pending=None, **kw):
        super().__init__(a, **kw)
        self.v2_state_p, self.v2_fail_p = self.state_p, self.fail_p          # read-only evidence
        self.state_p = self.qdir / f'host_{a.machine}_COORD_V3_STATE.json'
        self.fail_p = self.qdir / f'host_{a.machine}_COORD_V3_FAILED.json'
        self.state = {'machine': a.machine, 'done_folds': [], 'version': 'v3', 'events': []}
        self._launch, self.run_cmd = self.run_cmd, self._yield_then_launch
        self.pending = pending or self._pending_cli

    # -- single writer: wait silently for the live v2 to exit, then (holding the lock) consume its handover sentinel
    def acquire_lock(self):
        lock = open(self.qdir / 'host_coord_v2.lock', 'a+')
        deadline = self.now() + self.a.lock_wait_h * 3600
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if self.now() > deadline:
                    lock.close()
                    raise HC.CoordError(f'single-writer lock still held after {self.a.lock_wait_h} h (no file written)')
                self.sleep(self.a.poll_s)
        self._lock = lock
        lock.seek(0); lock.truncate(); lock.write(str(os.getpid())); lock.flush()
        self.consume_handover()

    @staticmethod
    def _read(p):
        return json.loads(p.read_text()) if p.exists() else None

    def consume_handover(self):
        f = self.qdir / 'FAILED.json'
        body = self._read(f)
        if body is None:
            return None
        if body.get(SENTINEL_KEY) is not True:
            return None                                   # real failure: untouched; inherited check_failures stops v3
        why = []
        stop = str(body.get('expected_stop_after') or '')
        if not stop.startswith('train:') or not self.fold_done(stop.split(':', 1)[1]):
            why.append(f'fold of expected_stop_after={stop!r} is not 3/3 verified DONE')
        v2f, v2s = self._read(self.v2_fail_p), self._read(self.v2_state_p) or {}
        at_boundary = (v2f is not None and str(v2f.get('phase', '')).split(':')[0] in ('inputs', 'smoke')
                       and SENTINEL_KEY in str(v2f.get('error', ''))) or (v2f is None and v2s.get('status') == 'done')
        if not at_boundary:
            why.append(f'v2 did not stop at a fold boundary (v2 failed phase={None if v2f is None else v2f.get("phase")}, '
                       f'v2 status={v2s.get("status")})')
        qs = self._read(self.qdir / 'QUEUE_STATE.json') or {}
        if qs.get('running'):
            why.append(f'queue jobs still running: {qs["running"]}')
        if why:
            raise HC.CoordError('handover sentinel NOT consumed: ' + '; '.join(why))
        dst = self.qdir / f'HANDOVER_STOP.{int(self.now())}.json'
        os.replace(f, dst)
        self.state['events'].append({'handover_consumed': body, 'moved_to': dst.name, 'time': self.now()})
        return dst

    # -- eval priority right before claiming GPUs
    def _pending_cli(self):
        p = subprocess.run([self.a.python, str(EVAL_PIPELINE), '--trqam-root', str(self.a.out), 'pending', '--host', self.a.machine],
                           capture_output=True, text=True)
        return p.returncode, (p.stdout.strip() or p.stderr.strip())[-400:]

    def _yield_then_launch(self, cmd, **kw):
        deadline = self.now() + self.a.yield_max_h * 3600
        while True:
            rc, info = self.pending()
            if rc == 3:
                break                                          # nothing READY to evaluate on this host
            if rc != 0:
                self.fail('eval_yield', f'eval_pipeline pending rc={rc}: {info}')
            if self.now() > deadline:
                self.fail('eval_yield', f'eval still pending after {self.a.yield_max_h} h (is the eval follower running?): {info}')
            self.save(status='yielding_to_eval', pending=info)
            self.sleep(self.a.poll_s)
        self.save(status='training', pending=None)
        return self._launch(cmd, **kw)


def parse(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = {'--lock-wait-h': 24.0, '--yield-max-h': 12.0}
    for k in list(extra):
        if k in argv:
            i = argv.index(k); extra[k] = float(argv[i + 1]); del argv[i:i + 2]
    a = H2.parse(argv)
    a.lock_wait_h, a.yield_max_h = extra['--lock-wait-h'], extra['--yield-max-h']
    return a


def main():
    a = parse()
    try:
        print(json.dumps(HostCoordinatorV3(a).run()))
    except (HC.CoordError, TQ.QueueError) as e:
        print('HOST_COORD_V3_ERROR', e, file=sys.stderr); sys.exit(2)


if __name__ == '__main__':
    main()

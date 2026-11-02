"""v4 = host_fold_coordinator_v3 + CPU dry/AE preparation OUTSIDE the GPU reservation, one fold ahead.

Why: train_queue_v2.sh runs a fold's 3 CPU dry-runs (AE16/K48 potential fit, ~hours at 2 threads) INSIDE the harness
reservation, so GPUs 5,6,7 sit idle while the AE fits. v4 runs exactly those dry jobs itself on CPU, before claiming GPUs:
  * per fold, right before the launch (after the inherited inputs wait, BEFORE the eval yield / GPU claim): ensure the
    fold's 3 DRYRUN.json exist — join the background prep of that fold, or run it now (CPU). The launched train_queue_v2
    then sees dry done (train_queue.done('dry') = DRYRUN.json exists) and goes straight to GPU prod.
  * then start background prep of the NEXT host fold (inputs ready, not 3/3 DONE) while this fold trains / eval runs.
Dry command/env are the queue's own: TrainQueueV2(...).cmd(fold, arm, 'dry') (+ the same --reuse-ae-from) with
TrainQueue.env(None) (JAX_PLATFORMS=cpu, CUDA_VISIBLE_DEVICES='', OMP/MKL/OPENBLAS = --threads, default 2 = queue default)
-> identical recipe, identical AE (fit once in the pbrs dry, reused by mix and prod), identical fingerprints.
Safety:
  * same-fold AE race impossible: at most ONE prep fold at a time, its arms serial (basic, pbrs, mix = queue order); the
    prep fold is never the fold being launched (a launch first joins its prep), and a launched train_queue_v2 is
    --folds <that fold> only.
  * single writer: prep writes only <out>/<prep fold>/<tag>/ (finetune_fold's own outputs) and logs under
    queue/prep_v4/; never QUEUE_STATE/FAILED/QUEUE_FREEZE. Only the main thread writes the V4 state/failure files.
  * bounded resources: 1 dry process at a time x --threads; CPU only; disk = the AE/DRYRUN files the queue would write.
  * fail-closed: a prep failure is recorded and stops v4 (COORD_V4_FAILED) when that fold would launch, before any GPU
    claim — same outcome as the queue's dry failure, minus the idle reservation.
  * takeover: like v3 (silent lock wait; nothing is written or run before the lock), consuming a handover sentinel only
    when the stopping coordinator (v2 or v3) recorded the boundary stop AFTER the sentinel was armed.
State host_<m>_COORD_V4_STATE.json, failure host_<m>_COORD_V4_FAILED.json, prep logs queue/prep_v4/.

    .venvs/rl2_iid/bin/python host_fold_coordinator_v4.py <same args as v3> [--threads 2] [--no-prep-ahead]
"""
import json, os, subprocess, sys, threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import host_fold_coordinator_v3 as H3   # noqa: E402  (frozen; subclassed)
import wait_ready as WR                 # noqa: E402  (frozen; read-only use)

H2, HC, TQ, Q2 = H3.H2, H3.HC, H3.TQ, H3.H2.Q2
SENTINEL_KEY = H3.SENTINEL_KEY


class HostCoordinatorV4(H3.HostCoordinatorV3):
    def __init__(self, a, popen=subprocess.Popen, **kw):
        super().__init__(a, **kw)
        v3 = (self.fail_p, self.state_p)                                       # set by V3.__init__
        self.state_p = self.qdir / f'host_{a.machine}_COORD_V4_STATE.json'
        self.fail_p = self.qdir / f'host_{a.machine}_COORD_V4_FAILED.json'
        # coordinators whose boundary stop may release a sentinel (read-only; v4's own = an earlier v4 run)
        self.stopper_files = [(self.v2_fail_p, self.v2_state_p), v3, (self.fail_p, self.state_p)]
        self.state = {'machine': a.machine, 'done_folds': [], 'version': 'v4', 'events': [], 'prep': {}}
        self.popen = popen
        self.prep_dir = self.qdir / 'prep_v4'
        self._prep = None                         # {'fid', 'thread', 'result'} — at most one fold in prep

    # -- takeover: sentinel consumed only with a boundary stop recorded by v2 or v3 after it was armed
    def consume_handover(self):
        f = self.qdir / 'FAILED.json'
        body = self._read(f)
        if body is None or body.get(SENTINEL_KEY) is not True:
            return None                                   # none / real failure: untouched (inherited checks stop v4)
        why = []
        stop = str(body.get('expected_stop_after') or '')
        if not stop.startswith('train:') or not self.fold_done(stop.split(':', 1)[1]):
            why.append(f'fold of expected_stop_after={stop!r} is not 3/3 verified DONE')
        t0 = float(body.get('time', 0))
        stopped = []
        for fp, sp in self.stopper_files:
            ff, ss = self._read(fp), self._read(sp) or {}
            if ff is not None and float(ff.get('time', -1)) >= t0 and str(ff.get('phase', '')).split(':')[0] in ('inputs', 'smoke') \
                    and SENTINEL_KEY in str(ff.get('error', '')):
                stopped.append(fp.name)
            elif ff is None and ss.get('status') == 'done' and float(ss.get('time', -1)) >= t0:
                stopped.append(sp.name)
        if not stopped:
            why.append('no coordinator recorded a fold-boundary stop after the sentinel was armed')
        qs = self._read(self.qdir / 'QUEUE_STATE.json') or {}
        if qs.get('running'):
            why.append(f'queue jobs still running: {qs["running"]}')
        if why:
            raise HC.CoordError('handover sentinel NOT consumed: ' + '; '.join(why))
        dst = self.qdir / f'HANDOVER_STOP.{int(self.now())}.json'
        os.replace(f, dst)
        self.state['events'].append({'handover_consumed': body, 'stopped_by': stopped, 'moved_to': dst.name, 'time': self.now()})
        return dst

    # -- CPU dry/AE prep (identical command/env to the queue's dry stage)
    def dry_queue(self):
        args, _ = Q2.parse(['--spec', str(self.a.spec), '--sources', str(self.a.sources), '--out', str(self.a.out),
                            '--init', str(self.a.init), '--dry-only', '--python', str(self.a.python), '--threads', str(self.a.threads)])
        return Q2.TrainQueueV2(args, reuse_ae_from=self.a.reuse_ae_from)

    def dry_done(self, fid):
        q = self.dry_queue()
        return all(q.done(fid, arm, 'dry') for arm in TQ.ARMS)

    def run_dry(self, fid):
        """Runs in the prep thread or the main thread; never touches coordinator state files. Returns a result dict."""
        q = self.dry_queue()
        f = next(x for x in self.spec['folds'] if x['fold_id'] == fid)
        res = {'fold': fid, 'arms': {}}
        self.prep_dir.mkdir(parents=True, exist_ok=True)
        for arm in TQ.ARMS:                                   # serial: pbrs fits the fold AE once, mix reuses it
            if q.done(fid, arm, 'dry'):
                res['arms'][arm] = 'already'; continue
            log = self.prep_dir / f'{fid}.{arm}.{1 + len(list(self.prep_dir.glob(f"{fid}.{arm}.*.log")))}.log'
            with open(log, 'x') as fh:
                cmd = q.cmd(f, arm, 'dry')
                fh.write('# ' + ' '.join(cmd) + '\n'); fh.flush()
                rc = self.popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=q.env(None)).wait()
            if rc != 0 or not q.done(fid, arm, 'dry'):
                res.update(error=f'{arm} dry rc={rc}', log=log.name); return res
            res['arms'][arm] = log.name
        res['ok'] = True
        return res

    def start_prep(self, fid):
        if fid is None or self.a.no_prep_ahead or self._prep is not None:
            return

        def work(box):
            try:
                box['result'] = self.run_dry(fid)
            except Exception as e:                            # recorded; acted on (fail) by the main thread
                box['result'] = {'fold': fid, 'error': repr(e)}
        box = {'fid': fid}
        box['thread'] = threading.Thread(target=work, args=(box,), daemon=False, name=f'prep-{fid}')
        self._prep = box; box['thread'].start()
        self.save(status='training', prep_running=fid)

    def join_prep(self):
        box, self._prep = self._prep, None
        if box is None:
            return None
        while box['thread'].is_alive():
            box['thread'].join(timeout=self.a.poll_s)
        self.state['prep'][box['fid']] = box['result']
        self.save(prep_running=None)
        return box

    def next_prep_fold(self, after):
        folds = self.host_folds()
        for fid in folds[folds.index(after) + 1:]:
            if self.fold_done(fid) or self.dry_done(fid):
                continue
            nr = WR.not_ready(self.spec, self.sources, [fid], self.a.expect_records)
            return fid if not nr else None                    # next fold not ready yet -> it is prepared at its launch
        return None

    def ensure_dry(self, fid):
        box = self.join_prep()                                # any running prep finishes first: never two dry folds
        if box and box['fid'] == fid and box['result'].get('error'):
            self.fail(f'prep:{fid}', f'CPU dry/AE prep failed: {box["result"]}')
        if not self.dry_done(fid):
            self.save(status='prep', prep_fold=fid)          # phase stays train:<fid> (sentinel expected_stop_after)
            res = self.run_dry(fid)
            self.state['prep'][fid] = res
            if res.get('error'):
                self.fail(f'prep:{fid}', f'CPU dry/AE prep failed: {res}')

    def _yield_then_launch(self, cmd, **kw):
        fid = cmd[cmd.index('--folds') + 1]
        self.ensure_dry(fid)                                  # CPU, before any GPU claim
        self.start_prep(self.next_prep_fold(fid))             # overlaps with eval yield + this fold's GPU training
        try:
            return super()._yield_then_launch(cmd, **kw)
        except BaseException:
            if self._prep is not None:
                self.join_prep()                              # never leave an orphan dry job behind a failed v4
            raise

    def run(self):
        try:
            return super().run()
        finally:
            if self._prep is not None:
                self.join_prep()


def parse(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    threads, no_prep = 2, False
    if '--threads' in argv:
        i = argv.index('--threads'); threads = int(argv[i + 1]); del argv[i:i + 2]
    if '--no-prep-ahead' in argv:
        argv.remove('--no-prep-ahead'); no_prep = True
    a = H3.parse(argv)
    a.threads, a.no_prep_ahead = threads, no_prep
    return a


def main():
    a = parse()
    try:
        print(json.dumps(HostCoordinatorV4(a).run()))
    except (HC.CoordError, TQ.QueueError) as e:
        print('HOST_COORD_V4_ERROR', e, file=sys.stderr); sys.exit(2)


if __name__ == '__main__':
    main()

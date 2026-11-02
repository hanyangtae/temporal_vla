"""Per-host coordinator for the remaining folds: CPU waits, then ONE bounded 3-arm launch per fold (GPU released between
folds so the eval queue can run). Reuses the frozen train_queue.py / wait_ready.py / wait_launch.py / train_queue.sh
read-only; recipe and freeze are theirs (QUEUE_FREEZE.json) — this file adds only ordering and failure handling.

Phases (state: <out>/queue/host_<machine>_COORD_STATE.json; failure: <out>/queue/host_<machine>_COORD_FAILED.json):
  1 first   wait (CPU) until --after-fold has all 3 arms verified DONE (train_queue.TrainQueue.done: fingerprint code ==
            freeze, init sha, params sha, receipt). The first fold is never re-launched here.
  2 per fold, in order, skipping folds already 3/3 DONE:
      inputs  wait_ready.not_ready == {} (latents COMPLETE == expect, compact contract, every --require file)
      train   wait_launch.py -> train_queue.sh --sources S --folds <fold>   (harness reserve; BLOCKED = wait)
      verify  3 arms DONE, else fail
Never waits forever: any <out>/queue/FAILED.json (train_queue failure), any --fail-marker file, wait_launch rc 75
(harness still blocked at deadline) or the per-phase deadline -> COORD_FAILED.json + exit 2. A previous COORD_FAILED is
kept (renamed *.prev.<ts>) only when rerun with --retry-failed.

    .venvs/rl2_iid/bin/python host_fold_coordinator.py --machine kanu --sources <sources.json> \
        --after-fold OpenDrawer-left_s2_j2 [--folds ...] [--gpus 5] [--require <gate>] [--fail-marker <file>]
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
REPO = PKG.parents[2]
sys.path[:0] = [str(HERE), str(PKG)]
import train_queue as TQ   # noqa: E402  (frozen; read-only use)
import wait_ready as WR    # noqa: E402  (frozen; read-only use)
from folds import load_spec   # noqa: E402

HOST_TO_ARCHIVE = {'kanu': 'kanu', 'srv48': 'worker1', 'srv50': 'worker2'}


class CoordError(RuntimeError):
    pass


class HostCoordinator:
    def __init__(self, a, run=subprocess.run, sleep=time.sleep, now=time.time):
        self.a, self.run_cmd, self.sleep, self.now = a, run, sleep, now
        self.spec = load_spec(a.spec)
        self.sources = json.loads(a.sources.read_text())
        self.qdir = a.out / 'queue'; self.qdir.mkdir(parents=True, exist_ok=True)
        self.state_p = self.qdir / f'host_{a.machine}_COORD_STATE.json'
        self.fail_p = self.qdir / f'host_{a.machine}_COORD_FAILED.json'
        self.state = {'machine': a.machine, 'done_folds': [], 'events': []}

    # -- helpers
    def host_folds(self):
        want = HOST_TO_ARCHIVE[self.a.machine]
        fs = [f['fold_id'] for f in self.spec['folds'] if f['archive_machine'] == want
              and f'{f["instruction"]}|s{f["scene"]}' in self.sources]
        if self.a.folds:
            bad = set(self.a.folds) - set(fs)
            if bad:
                raise CoordError(f'folds not on {self.a.machine} or not in sources: {sorted(bad)}')
            fs = [f for f in fs if f in self.a.folds]
        return [f for f in fs if f != self.a.after_fold]

    def queue(self):
        """Fresh read-only TrainQueue view (verifies QUEUE_FREEZE; never launches)."""
        args = TQ.parse(['--spec', str(self.a.spec), '--sources', str(self.a.sources), '--out', str(self.a.out),
                         '--init', str(self.a.init), '--dry-only'])
        return TQ.TrainQueue(args)

    def fold_done(self, fid):
        if not (self.qdir / 'QUEUE_FREEZE.json').exists():
            return False                                  # no train_queue launch has happened yet under this --out
        q = self.queue()
        return all(q.done(fid, arm, 'prod') for arm in TQ.ARMS)

    def save(self, **kw):
        self.state.update(kw, time=self.now())
        tmp = self.state_p.with_suffix('.tmp'); tmp.write_text(json.dumps(self.state, indent=1)); os.replace(tmp, self.state_p)

    def fail(self, phase, msg):
        self.fail_p.write_text(json.dumps({'phase': phase, 'error': msg, 'time': self.now(), 'state': self.state}, indent=1))
        self.save(status='failed', phase=phase)
        raise CoordError(f'{phase}: {msg}')

    def check_failures(self, phase):
        if (self.qdir / 'FAILED.json').exists():
            self.fail(phase, f'train_queue FAILED.json present: {(self.qdir / "FAILED.json").read_text()[:300]}')
        hit = [str(p) for p in self.a.fail_marker if p.exists()]
        if hit:
            self.fail(phase, f'failure marker present: {hit}')

    def wait_until(self, phase, cond, detail):
        deadline = self.now() + self.a.max_wait_h * 3600
        while True:
            self.check_failures(phase)
            ok, info = cond()
            if ok:
                return
            if self.now() + self.a.poll_s > deadline:
                self.fail(phase, f'deadline {self.a.max_wait_h} h: {detail}: {info}')
            self.save(status='waiting', phase=phase, waiting_for=info)
            self.sleep(self.a.poll_s)

    # -- run
    def run(self):
        if self.fail_p.exists():
            if not self.a.retry_failed:
                raise CoordError(f'{self.fail_p} present; inspect, then --retry-failed')
            os.replace(self.fail_p, self.fail_p.with_name(f'{self.fail_p.stem}.prev.{int(self.now())}.json'))
        folds = self.host_folds()
        if not folds:
            raise CoordError(f'no remaining folds for {self.a.machine}')
        self.save(status='started', plan=folds, after_fold=self.a.after_fold)
        if self.a.after_fold:
            self.wait_until('first', lambda: (self.fold_done(self.a.after_fold), f'{self.a.after_fold} 3 arms DONE'),
                            'first fold not finished')
        for fid in folds:
            if self.fold_done(fid):
                self.state['done_folds'].append(fid); continue
            self.wait_until(f'inputs:{fid}', lambda: self.inputs_ready(fid), 'inputs not ready')
            self.save(status='training', phase=f'train:{fid}')
            cmd = [self.a.python, str(HERE / 'wait_launch.py'), '--max-wait-h', str(self.a.max_wait_h), '--poll-s', str(self.a.harness_poll_s),
                   '--state', str(self.qdir / 'wait'), '--', 'env', f'GPUS={",".join(map(str, self.a.gpus))}',
                   f'HARNESS_MACHINE={self.a.machine}', f'HARNESS_SESSION={self.a.session}',
                   *([f'FOLD27_PY={self.a.python}'] if self.a.machine != 'kanu' else []),
                   'bash', str(HERE / 'train_queue.sh'), '--sources', str(self.a.sources), '--out', str(self.a.out),
                   '--init', str(self.a.init), '--folds', fid]
            rc = self.run_cmd(cmd).returncode
            if rc == 75:
                self.fail(f'train:{fid}', 'harness still BLOCKED at the wait deadline')
            if rc != 0:
                self.check_failures(f'train:{fid}')
                self.fail(f'train:{fid}', f'train_queue rc={rc}')
            if not self.fold_done(fid):
                self.fail(f'verify:{fid}', 'train_queue returned 0 but not all 3 arms are verified DONE')
            self.state['done_folds'].append(fid)
            self.save(status='fold_done', phase=f'done:{fid}')
        self.save(status='done')
        return {'machine': self.a.machine, 'done_folds': self.state['done_folds']}

    def inputs_ready(self, fid):
        nr = WR.not_ready(self.spec, self.sources, [fid], self.a.expect_records)
        nr.update({str(p): 'required gate file missing' for p in self.a.require if not p.exists()})
        fatal = {k: v for k, v in nr.items() if v.startswith('latents COMPLETE episodes') or v in ('unknown fold', 'scene missing from sources')}
        if fatal:
            self.fail(f'inputs:{fid}', f'inputs can never become ready: {fatal}')
        return (not nr), nr


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--machine', choices=sorted(HOST_TO_ARCHIVE), required=True)
    ap.add_argument('--spec', type=Path, default=REPO / 'outputs/analysis/groot_fold27_20261028/folds/fold_spec.json')
    ap.add_argument('--sources', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=REPO / 'outputs/rl2_vla/fold27_trqam_20261102')
    ap.add_argument('--init', type=Path, default=REPO / 'outputs/rl2_vla/atomic65_common_qam_work_20261029/scratch_run/checkpoints/step_0500000')
    ap.add_argument('--after-fold', help='fold that must be 3/3 DONE first (launched elsewhere; never re-launched here)')
    ap.add_argument('--folds', nargs='+'); ap.add_argument('--gpus', nargs='+', type=int, default=[5])
    ap.add_argument('--session', default='fold27-trqam'); ap.add_argument('--python', default=str(REPO / '.venvs/rl2_iid/bin/python'))
    ap.add_argument('--require', type=Path, action='append', default=[]); ap.add_argument('--fail-marker', type=Path, action='append', default=[])
    ap.add_argument('--expect-records', type=int, default=200)
    ap.add_argument('--max-wait-h', type=float, default=48); ap.add_argument('--poll-s', type=float, default=120)
    ap.add_argument('--harness-poll-s', type=float, default=300); ap.add_argument('--retry-failed', action='store_true')
    return ap.parse_args(argv)


def main():
    a = parse()
    try:
        print(json.dumps(HostCoordinator(a).run()))
    except (CoordError, TQ.QueueError) as e:
        print('HOST_COORD_ERROR', e, file=sys.stderr); sys.exit(2)


if __name__ == '__main__':
    main()

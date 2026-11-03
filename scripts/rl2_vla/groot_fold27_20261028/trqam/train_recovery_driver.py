"""One-shot boundary recovery driver (new file): wait read-only for the eval->train boundary, then replace the IDLE blocked
training coordinator/guardian (ours only, identity-pinned) by train_guardian_v3 + host_fold_coordinator_v5_shared_g3
under a LIVE-derived session contract. Never touches eval processes/receipts, never edits contracts/manifests/ledger.

  0 lock     <state>/INTENT.json created O_EXCL at start (a second driver or a re-run refuses); every step appends to
             <state>/steps.jsonl; terminal outcome = DONE.json or REFUSED.json (both O_EXCL).
  1 wait     train_recovery_boundary.check (read-only) until READY. Additionally the eval consumer's own exit record
             (<eval_root>/queue/<host>/QUEUE_STATE.gpu<g>.json) must say exit 'no runnable job' with no failures: an eval
             that ended by failure -> REFUSED (main inspects), nothing stopped.
  2 verify   old coordinator and old guardian: pid + start ticks + uid + cmdline sha256 + boot == pinned at install; any
             mismatch (gone, PID reuse) for the COORDINATOR -> REFUSED; a guardian already gone is recorded and allowed.
             Boundary re-checked immediately before signalling.
  3 stop     SIGTERM to exactly those two identities (re-verified per signal); wait until gone (term_wait_s); still alive
             -> REFUSED (no SIGKILL, main decides). No other process is signalled.
  4 contract gpu_sharing_derive_v1.derive (live re-proof of the pinned foreign identity) -> <state>/SHARING_TRAIN.json
  5 guardian write <state>/guardian_v3.json (template + this contract), spawn train_guardian_v3 (setsid) with a NEW
             state dir; after settle_s it must be alive with the same start ticks and no ALERT/ADMISSION_UNSAFE
  6 coord    spawn host_fold_coordinator_v5_shared_g3 with the OLD coordinator's exact argv (pinned at install) except
             the script and --sharing-auth/--train-guardian-* (new values); after settle_s it must be alive -> DONE.json
Any refusal after step 3 leaves training stopped (it was blocked anyway) and records why; nothing is restarted
automatically.

    python3 train_recovery_driver.py --config <driver.json>    # kind fold27_train_recovery_driver_v1
"""
import argparse, hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path[:0] = [str(HERE), str(REPO / 'scripts/utils')]
import train_recovery_boundary as B     # noqa: E402
import gpu_sharing_derive_v1 as D       # noqa: E402


def audited_eval_snapshot(root, host):
    """Existing read-only aggregator: run_eval_arm.load_progress (row hashes) + verify_fold.audit_decisions per arm."""
    sys.path.insert(0, str(HERE.parent / 'eval/runtime'))
    import aggregate_eval_report as AG     # noqa: E402  (unchanged, read-only)
    return AG.snapshot(root, host)

KIND = 'fold27_train_recovery_driver_v1'
SHARED_FLAGS = ('--sharing-auth', '--train-guardian-pid', '--train-guardian-start', '--train-guardian-config')


class Refused(RuntimeError):
    pass


class HostOS(B.HostView):
    def uid(self, pid):
        try:
            return next(int(l.split()[1]) for l in Path(f'/proc/{int(pid)}/status').read_text().splitlines() if l.startswith('Uid:'))
        except (OSError, ValueError, StopIteration):
            return None

    def cmd_sha(self, pid):
        try:
            return hashlib.sha256(Path(f'/proc/{int(pid)}/cmdline').read_bytes()).hexdigest()
        except (OSError, ValueError):
            return None

    def kill(self, pid, sig):
        os.kill(pid, sig)

    def spawn(self, argv, log, env):
        with open(log, 'x') as fh:
            return subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=str(REPO),
                                    env={**os.environ, **env}, start_new_session=True).pid

    def exists(self, p): return Path(p).exists()
    def now(self): return time.time()


def replace_flags(argv, new):
    out, i = [], 0
    while i < len(argv):
        if argv[i] in new:
            out += [argv[i], str(new[argv[i]])]; i += 2
        else:
            out.append(argv[i]); i += 1
    missing = [k for k in new if k not in argv]
    if missing:
        raise Refused(f'old coordinator argv lacks {missing}')
    return out


class Driver:
    def __init__(self, cfg, h, sleep=time.sleep, derive=D.derive, eval_snapshot=audited_eval_snapshot):
        if cfg.get('kind') != KIND:
            raise Refused(f'config kind must be {KIND}')
        self.c, self.h, self.sleep, self.derive, self.eval_snapshot = cfg, h, sleep, derive, eval_snapshot
        self.sd = Path(cfg['state_dir'])

    def step(self, name, **kw):
        with open(self.sd / 'steps.jsonl', 'a') as f:
            f.write(json.dumps({'step': name, 'time': self.h.now(), **kw}, default=str) + '\n')

    def final(self, name, body):
        with open(self.sd / name, 'x') as f:
            f.write(json.dumps({'time': self.h.now(), **body}, indent=1, default=str))

    def same(self, ident):
        return (self.h.boot() == self.c['boot_id'] and self.h.start(ident['pid']) == ident['start']
                and self.h.uid(ident['pid']) == ident['uid'] and self.h.cmd_sha(ident['pid']) == ident['cmdline_sha256'])

    def boundary(self):
        return B.check(argparse.Namespace(**self.c['boundary']), self.h)

    def eval_ended_cleanly(self):
        """(complete, why). Incomplete -> keep waiting; a FAILED eval -> Refused (main inspects, no takeover)."""
        st = self.h.read_json(self.c['eval_queue_state'])
        if st is not None and st.get('failures_this_run'):
            raise Refused(f'eval consumer recorded failures {st["failures_this_run"][:3]}')
        ev = self.c['eval_complete']                     # {'root','host','expected': {fold_id: [arms]}, 'rows_per_arm': 40}
        want = {(f, a) for f, arms in ev['expected'].items() for a in arms}
        snap = self.eval_snapshot(ev['root'], ev['host'])
        got = {(f['fold_id'], arm): (f, a) for f in snap['folds'] for arm, a in f['arms'].items()}
        if set(got) != want:
            raise Refused(f'eval root coordinates differ from the pinned set: missing {sorted(want - set(got))[:3]} '
                          f'extra {sorted(set(got) - want)[:3]}')
        bad, incomplete, rows = [], [], 0
        n = int(ev['rows_per_arm'])
        for (fid, arm), (f, a) in sorted(got.items()):
            if a['load_error'] or a['audit_errors'] or any(not r['audit_ok'] for r in a['rows']):
                bad.append(f'{fid}/{arm}: {a["load_error"] or a["audit_errors"][:2]}')
            if f['planned_rows'] != n:
                bad.append(f'{fid}: planned rows {f["planned_rows"]} != pinned {n}')
            if not a['run_complete'] or len(a['rows']) != n:
                incomplete.append(f'{fid}/{arm} {len(a["rows"])}/{n}')
            else:
                rc = self.h.read_json(Path(ev['root']) / fid / arm / 'RUN_COMPLETE.json')
                prog = Path(ev['root']) / fid / arm / 'progress.jsonl'
                if rc is None or rc.get('fold_id') != fid or rc.get('arm') != arm or rc.get('rows') != n \
                        or rc.get('progress_sha256') != self.file_sha(prog):
                    bad.append(f'{fid}/{arm}: RUN_COMPLETE does not bind fold/arm/rows/progress sha')
            rows += len(a['rows'])
        if bad:
            raise Refused(f'eval failed load/audit/completion binding: {bad[:3]}')
        if incomplete:
            return False, f'eval incomplete: {rows}/{len(want) * n} rows; {incomplete[:3]}'
        if st is None or st.get('exit') != 'no runnable job':
            return False, f'eval consumer exit record {st and st.get("exit")!r}'
        return True, f'{rows} audited complete rows over {len(want)} pinned (fold, arm)'

    @staticmethod
    def file_sha(p):
        try:
            return hashlib.sha256(Path(p).read_bytes()).hexdigest()
        except OSError:
            return None

    def run_spawned(self, argv, log, env):
        pid = self.h.spawn(argv, log, env)
        start = self.h.start(pid)                                             # identity captured immediately
        self.sleep(float(self.c.get('settle_s', 30)))
        if start is None or self.h.start(pid) != start or self.h.boot() != self.c['boot_id']:
            raise Refused(f'spawned {Path(argv[1]).name} pid {pid} exited or was replaced during settle (see {log})')
        return pid, start

    def run(self):
        self.sd.mkdir(parents=True, exist_ok=True)
        with open(self.sd / 'INTENT.json', 'x') as f:
            f.write(json.dumps({'config': self.c, 'pid': os.getpid(), 'time': self.h.now()}, indent=1, default=str))
        try:
            return self._run()
        except Exception as e:   # noqa: BLE001 - every failure (derive AuthError, I/O, query) is a recorded refusal
            why = f'{type(e).__name__}: {e}'
            self.step('refused', reason=why); self.final('REFUSED.json', {'reason': why}); return 2

    def _run(self):
        while True:                                                            # 1 wait (read-only)
            r = self.boundary()
            ok, why = (self.eval_ended_cleanly() if r['ready'] else (False, 'boundary not ready'))
            if r['ready'] and ok:
                self.step('boundary_ready', snapshot=r['snapshot']); break
            self.step('waiting', reasons=r['reasons'] + ([] if ok else [why]))
            self.sleep(float(self.c.get('poll_s', 300)))
        oc, og = self.c['old_coordinator'], self.c['old_guardian']             # 2 verify
        if not self.same(oc):
            raise Refused('old coordinator identity changed or gone before stop (PID reuse / restart)')
        g_alive = self.same(og)
        if not g_alive and self.h.start(og['pid']) is not None:                # only a really GONE guardian is accepted
            raise Refused(f'old guardian pid {og["pid"]} is alive with another identity (PID reuse?): not ours, refuse')
        self.step('identities_verified', coordinator=oc['pid'], guardian_alive=g_alive)
        r = self.boundary()
        if not r['ready']:
            raise Refused(f'boundary lost before stop: {r["reasons"]}')
        for ident in [oc] + ([og] if g_alive else []):                        # 3 stop (ours only)
            if not self.same(ident):
                raise Refused(f'identity of {ident["pid"]} changed just before signal')
            self.h.kill(ident['pid'], signal.SIGTERM); self.step('sigterm', pid=ident['pid'], start=ident['start'])
        t_end = self.h.now() + float(self.c.get('term_wait_s', 120))
        while any(self.h.start(i['pid']) == i['start'] for i in (oc, og)):
            if self.h.now() > t_end:
                raise Refused('old coordinator/guardian still alive after SIGTERM (no SIGKILL; main decides)')
            self.sleep(2)
        self.step('old_stopped')
        contract = self.sd / 'SHARING_TRAIN.json'                             # 4 live-derived contract
        self.derive(argparse.Namespace(**self.c['derive'], out=contract), self.h.probe)
        self.step('contract_derived', path=str(contract), sha256=hashlib.sha256(contract.read_bytes()).hexdigest())
        gc = self.sd / 'guardian_v3.json'                                     # 5 guardian
        gcfg = {**self.c['guardian']['config'], 'kind': 'fold27_train_guardian_v3', 'sharing_authorization': str(contract)}
        with open(gc, 'x') as f:
            f.write(json.dumps(gcfg, indent=1))
        gsd = self.sd / 'guardian_state'
        gpid, gstart = self.run_spawned([self.c['guardian']['python'], str(HERE / 'train_guardian_v3.py'), '--config', str(gc),
                                         '--state-dir', str(gsd)], str(self.sd / 'guardian.log'), {})
        if any(self.h.exists(gsd / f) for f in ('ALERT.json', 'ADMISSION_UNSAFE.json')):
            raise Refused(f'new guardian {gpid} raised ALERT/UNSAFE (see {gsd}); it is left running for main')
        self.step('guardian_started', pid=gpid, start=gstart, config=str(gc))
        argv = replace_flags(list(oc['argv']), {'--sharing-auth': contract, '--train-guardian-pid': gpid,   # 6 coordinator
                                                '--train-guardian-start': gstart, '--train-guardian-config': gc})
        script = [i for i, x in enumerate(argv) if x.endswith('host_fold_coordinator_v5_shared.py')]
        if len(script) != 1:
            raise Refused('old coordinator argv does not name host_fold_coordinator_v5_shared.py exactly once')
        argv[script[0]] = str(HERE / 'host_fold_coordinator_v5_shared_g3.py')
        cpid, cstart = self.run_spawned(argv, str(self.sd / 'coordinator.log'), {'HARNESS_SHARING_AUTH': str(contract)})
        self.step('coordinator_started', pid=cpid, start=cstart, argv=argv)
        self.final('DONE.json', {'contract': str(contract), 'guardian': [gpid, gstart], 'coordinator': [cpid, cstart], 'argv': argv})
        return 0


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--config', type=Path, required=True)
    a = ap.parse_args(argv)
    try:
        return Driver(json.loads(a.config.read_text()), HostOS()).run()
    except (Refused, FileExistsError) as e:
        print('TRAIN_RECOVERY_REFUSED', e, file=sys.stderr); return 2


if __name__ == '__main__':
    sys.exit(main())

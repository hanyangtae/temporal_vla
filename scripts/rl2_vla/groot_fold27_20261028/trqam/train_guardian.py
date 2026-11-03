"""Runtime foreign-compute guardian for fold27 TRAINING on one shared GPU (new; no deployed file changed).

Ownership proof (official harness only; anything unproven -> no signal at all, ALERT):
  receipt   `groot_harness.py status`: EXACTLY ONE receipt with our training session, this machine, gpus == [gpu],
            kind eval, launcher_host == this host, consumer_pid recorded (the harness `verify` step ran)
  launcher  receipt.launcher_pid alive, uid == ours, cmdline = groot_harness.py run with --session <S> --gpus <gpu>
            --kind eval, and it started no later than receipt.created + clock_skew_s (coordinator/host clocks differ)
  consumer  receipt.consumer_pid alive, uid == ours, a descendant of the launcher by PPid chain, cmdline names a fold27
            training entry (train_queue_v2.sh/.py, train_queue.sh/.py, train_arm_v5.sh, smoke_v2b.sh, finetune_fold*.py)
  owned     every descendant of the launcher, recorded as (pid, start ticks), sticky (re-parented orphans stay owned)
  freeze    ancestors of the launcher with our uid whose cmdline is wait_launch.py / host_fold_coordinator*.py /
            t49_host_chain*.py (they would relaunch); never the harness launcher itself
Observation each poll (one NVML probe: UUID, free, per-PID used):
  ours       owned identities
  approved   gpu_sharing_authorization.approved_now (pinned root 1101996, verified members of the pinned container
             instance, aggregate bounds) -- never signalled
  foreign    everything else on the GPU (new PID, reused PID, other/restarted container, over-limit container)
  breaches   foreign seen confirm_polls times; our own GPU memory > receipt serves x session per_serve_mib (measured);
             free < contract reserve_mib -- each for confirm_polls consecutive polls
  NVML/status failure: observe nothing, act on nothing.
Action (once; skipped with --observe-only): re-verify ownership -> SIGSTOP freeze ancestors -> SIGSTOP queue processes
(no new job, no FAILED.json written) -> SIGTERM trainers (finetune_fold_v2 resume states are written atomically every
5000 steps, current + previous kept: at most one interval is recomputed on resume) -> SIGKILL after kill_wait_s ->
TERM+CONT queue processes and frozen ancestors -> wait for every owned identity to be gone, no owned PID on the GPU and
the harness launcher exited -> release the receipt through `groot_harness.py release` only if it is still ours (the
launcher may already have released it; then nothing). No file is touched except <state-dir>/.

    python train_guardian.py --config <cfg.json> --state-dir <new dir> [--observe-only] [--once] [snapshot]
"""
import argparse, json, os, signal, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path[:0] = [str(HERE), str(PKG / 'eval/runtime'), str(HERE.parents[3] / 'scripts/utils')]
import foreign_gpu_guardian as G1          # noqa: E402  (OS layer reused; unchanged)
import gpu_sharing_authorization as SA     # noqa: E402

KIND = 'fold27_train_guardian_v1'
TRAIN_MARKS = ('train_queue_v2.sh', 'train_queue_v2.py', 'train_queue.sh', 'train_queue.py', 'train_arm_v5.sh', 'smoke_v2b.sh',
               'finetune_fold_v2.py', 'finetune_fold.py')
QUEUE_MARKS = ('train_queue_v2.py', 'train_queue.py')
TRAINER_MARKS = ('finetune_fold_v2.py', 'finetune_fold.py', 'gpu_resume_check.py')
FREEZE_MARKS = ('wait_launch.py', 'host_fold_coordinator', 't49_host_chain')


class Unproven(RuntimeError):
    pass


class HostOS(G1.OS):
    def start_epoch(self, pid):
        st = self.start(pid)
        if st is None:
            return None
        btime = next(int(l.split()[1]) for l in Path('/proc/stat').read_text().splitlines() if l.startswith('btime'))
        return btime + st / os.sysconf('SC_CLK_TCK')

    def own_uid(self):
        return os.getuid()


class TrainGuardian:
    def __init__(self, cfg, osi, state_dir, probe=None, observe_only=False):
        if cfg.get('kind') != KIND:
            raise SystemExit(f'config kind must be {KIND}')
        self.c, self.o, self.sd = cfg, osi, Path(state_dir)
        self.sd.mkdir(parents=True, exist_ok=True)
        self.gpu, self.session, self.boot0 = int(cfg['gpu']), cfg['session'], osi.boot()
        self.auth = SA.load(cfg['sharing_authorization'])
        if self.auth.get('kind') != SA.KIND_V2 or self.auth['machine'] != cfg['host'] or int(self.auth['gpu']) != self.gpu \
                or self.session not in self.auth['sessions']:
            raise SystemExit('training guardian needs a v2 sharing contract covering this host/gpu/session')
        self.probe = probe or SA.Probe()
        self.observe_only = observe_only or bool(cfg.get('observe_only'))
        self.owner, self.owned, self.seen, self.breach = None, {}, {}, {'own_mem': 0, 'headroom': 0}

    # ---------------- records ----------------
    def event(self, kind, **kw):
        with open(self.sd / 'events.jsonl', 'a') as f:
            f.write(json.dumps({'kind': kind, 'time': self.o.now(), **kw}, default=str) + '\n')

    def alert(self, why, **kw):
        tmp = self.sd / f'ALERT.{os.getpid()}.tmp'
        tmp.write_text(json.dumps({'time': self.o.now(), 'reason': why, **kw}, indent=1, default=str)); tmp.replace(self.sd / 'ALERT.json')
        self.event('alert', reason=why, **kw)

    # ---------------- identity ----------------
    def same(self, pid, start):
        return start is not None and self.o.boot() == self.boot0 and self.o.start(pid) == start

    def has(self, pid, marks):
        c = self.o.cmdline(pid) or ''
        return any(m in c for m in marks)

    def chain_up(self, pid, depth=12):
        out, cur = [], pid
        for _ in range(depth):
            cur = self.o.ppid(cur)
            if not cur or cur <= 1:
                break
            out.append(cur)
        return out

    def prove_owner(self):
        """-> owner dict, or None if no receipt of ours exists; raises Unproven on any ambiguity."""
        st = self.o.harness('status')
        host = self.probe.hostname()
        mine = {r: it for r, it in st.items() if it.get('session') == self.session and it.get('machine') == self.c['host']
                and it.get('gpus') == [self.gpu]}
        if not mine:
            return None
        if len(mine) != 1:
            raise Unproven(f'{len(mine)} receipts of session {self.session} on GPU{self.gpu}')
        rid, it = next(iter(mine.items()))
        if it.get('kind') != 'eval' or str(it.get('launcher_host', '')).split('.')[0] != host:
            raise Unproven(f'receipt {rid[:8]} kind/launcher_host {it.get("kind")}/{it.get("launcher_host")} not ours here')
        lp, cp = it.get('launcher_pid'), it.get('consumer_pid')
        if not lp or not cp:
            raise Unproven(f'receipt {rid[:8]} lacks launcher_pid/consumer_pid (verify not run)')
        lc = self.o.cmdline(lp) or ''
        if self.o.uid(lp) != self.o.own_uid() or not all(s in f' {lc} ' for s in
                                                       ('groot_harness.py', ' run ', f'--session {self.session}', f'--gpus {self.gpu}', '--kind eval')):
            raise Unproven(f'launcher {lp} uid/cmdline is not our harness run for this session/gpu')
        lse = self.o.start_epoch(lp)
        if lse is None or lse > float(it.get('created', 0)) + float(self.c.get('clock_skew_s', 900)):
            raise Unproven(f'launcher {lp} started after its receipt (PID reuse?)')
        if self.o.uid(cp) != self.o.own_uid() or lp not in self.chain_up(cp) or not self.has(cp, TRAIN_MARKS):
            raise Unproven(f'consumer {cp} is not our training process under launcher {lp}')
        freeze = [(a, self.o.start(a)) for a in reversed(self.chain_up(lp))          # outermost (coordinator) first
                  if self.o.uid(a) == self.o.own_uid() and self.has(a, FREEZE_MARKS)]
        return {'receipt': rid, 'launcher': (lp, self.o.start(lp)), 'consumer': (cp, self.o.start(cp)), 'freeze': freeze,
                'serves': int(it.get('serves', 1))}

    def refresh(self):
        if self.owner is None or not self.same(*self.owner['launcher']):
            self.owner = self.prove_owner()
        if self.owner:
            kids = {}
            for p in self.o.pids():
                pp = self.o.ppid(p)
                if pp:
                    kids.setdefault(pp, []).append(p)
            stack = list(kids.get(self.owner['launcher'][0], []))
            while stack:
                p = stack.pop()
                st = self.o.start(p)
                if st is None:
                    continue
                role = 'trainer' if self.has(p, TRAINER_MARKS) else 'queue' if self.has(p, QUEUE_MARKS) else 'desc'
                self.owned.setdefault((p, st), role)
                stack += kids.get(p, [])
        self.owned = {k: v for k, v in self.owned.items() if self.same(*k)}

    def owned_pid(self, pid):
        st = self.o.start(pid)
        return st is not None and (pid, st) in self.owned

    # ---------------- poll ----------------
    def poll(self):
        try:
            self.refresh()
            g = self.probe.gpu(self.gpu)
        except Unproven as e:
            self.alert('ownership unproven: no signal', detail=str(e)); return None
        except Exception as e:   # noqa: BLE001 - NVML / harness failure: observe nothing
            self.event('probe_failed', error=str(e)[:300]); self.seen.clear(); return None
        if g['uuid'] != self.auth['gpu_uuid']:
            self.alert('GPU UUID differs from contract: no signal', uuid=g['uuid']); return None
        ours = [a for a in g['apps'] if self.owned_pid(a['pid'])]
        others = [a for a in g['apps'] if not self.owned_pid(a['pid'])]
        ok, viol = SA.approved_now(self.auth, others, self.probe)
        unknown = {}
        for a in others:
            if a['pid'] in ok:
                continue
            st = self.o.start(a['pid'])
            if st is not None:
                unknown[(a['pid'], st)] = self.seen.get((a['pid'], st), 0) + 1
        self.seen = unknown
        need = int(self.c.get('confirm_polls', 2))
        reasons = [f'foreign {k}' for k, n in unknown.items() if n >= need]
        if unknown or viol:
            self.event('unrecognized_or_violation', pids=[list(k) for k in unknown], violations=viol[:10])
        if self.owner:
            sc = self.auth['sessions'][self.session]
            own_used, limit = sum(a['used_mib'] for a in ours), self.owner['serves'] * int(sc['per_serve_mib'])
            self.breach['own_mem'] = self.breach['own_mem'] + 1 if own_used > limit else 0
            self.breach['headroom'] = self.breach['headroom'] + 1 if g['free_mib'] < int(self.auth['reserve_mib']) else 0
            if self.breach['own_mem']:
                self.event('own_memory_over_bound', used_mib=own_used, limit_mib=limit)
            if self.breach['headroom']:
                self.event('headroom_low', free_mib=g['free_mib'], reserve_mib=self.auth['reserve_mib'])
            reasons += [f'{k} breach' for k, n in self.breach.items() if n >= need]
        elif reasons:
            self.event('foreign_while_idle', reasons=reasons); return None
        return reasons or None

    # ---------------- action ----------------
    def sig(self, pid, start, s, why):
        if not self.same(pid, start):
            self.event('signal_skipped_identity_changed', pid=pid, sig=int(s), why=why); return False
        try:
            self.o.kill(pid, s)
        except (ProcessLookupError, PermissionError) as e:
            self.event('signal_failed', pid=pid, sig=int(s), why=why, error=type(e).__name__); return False
        self.event('signal', pid=pid, start=start, sig=int(s), why=why)
        return True

    def wait(self, pred, timeout):
        t_end = self.o.now() + timeout
        while not pred():
            if self.o.now() >= t_end:
                return False
            self.o.sleep(1)
        return True

    def alive(self, *roles):
        return [(p, s) for (p, s), r in self.owned.items() if r in roles and self.same(p, s)]

    def act(self, reasons):
        try:
            owner = self.prove_owner()
        except Unproven as e:
            self.alert('ownership unproven at action time: no signal', detail=str(e), reasons=reasons); return 'refused'
        if owner is None or owner['receipt'] != self.owner['receipt']:
            self.alert('receipt changed/gone at action time: no signal', reasons=reasons); return 'refused'
        self.refresh()
        rec = {'reasons': reasons, 'owner': owner, 'owned': [[p, s, r] for (p, s), r in self.owned.items()], 'observe_only': self.observe_only}
        with open(self.sd / f'ACTION.{int(self.o.now() * 1000)}.json', 'x') as f:
            f.write(json.dumps(rec, indent=1, default=str))
        if self.observe_only:
            self.event('observe_only_would_act', **rec); return 'observed'
        for p, s in owner['freeze']:
            self.sig(p, s, signal.SIGSTOP, 'freeze launcher ancestor (no relaunch)')
        for p, s in self.alive('queue'):
            self.sig(p, s, signal.SIGSTOP, 'freeze training queue (no new job / no FAILED.json)')
        self.refresh()
        for p, s in self.alive('trainer'):
            self.sig(p, s, signal.SIGTERM, 'stop trainer (atomic resume states kept)')
        kw = float(self.c.get('kill_wait_s', 60))
        self.wait(lambda: not self.alive('trainer'), kw)
        for p, s in self.alive('trainer', 'desc'):
            self.sig(p, s, signal.SIGKILL, 'owned process still alive')
        for p, s in self.alive('queue'):
            self.sig(p, s, signal.SIGTERM, 'stop queue'); self.sig(p, s, signal.SIGCONT, 'deliver TERM to stopped queue')
        self.wait(lambda: not self.alive('queue', 'trainer', 'desc'), kw)
        for p, s in self.alive('queue'):
            self.sig(p, s, signal.SIGKILL, 'queue still alive')
        return self.release(owner)

    def release(self, owner):
        t_end = self.o.now() + float(self.c.get('release_wait_s', 600))
        why = ''
        while self.o.now() < t_end:
            try:
                apps = self.probe.gpu(self.gpu)['apps']
                mine = [a['pid'] for a in apps if self.owned_pid(a['pid'])]
            except Exception as e:   # noqa: BLE001
                mine, why = ['?'], f'gpu query failed: {e}'
            gone = not any(self.same(*k) for k in self.owned) and not mine
            if gone and not self.same(*owner['launcher']):
                break
            why = why or ('owned alive' if not gone else 'harness launcher still running')
            self.o.sleep(2)
        else:
            self.alert('receipt kept: own processes not proven absent', receipt=owner['receipt'], detail=why)
            self._unfreeze_note(owner); return 'kept'
        for p, s in owner['freeze']:
            self.sig(p, s, signal.SIGTERM, 'stop frozen launcher ancestor'); self.sig(p, s, signal.SIGCONT, 'deliver TERM')
        try:
            it = self.o.harness('status').get(owner['receipt'])
        except Exception as e:   # noqa: BLE001
            self.alert('receipt status unavailable after stop', detail=str(e)[:300]); return 'kept'
        if it is None:
            self.event('receipt_released_by_launcher', receipt=owner['receipt']); return 'released_elsewhere'
        if it.get('session') != self.session or it.get('gpus') != [self.gpu]:
            self.alert('ledger entry no longer ours', receipt=owner['receipt']); return 'kept'
        try:
            self.o.harness('release', '--receipt', owner['receipt'], '--session', self.session)
        except Exception as e:   # noqa: BLE001
            self.alert('receipt release failed', detail=str(e)[:300]); return 'kept'
        self.event('receipt_released', receipt=owner['receipt']); return 'released'

    def _unfreeze_note(self, owner):
        self.event('ancestors_left_frozen', freeze=owner['freeze'], note='main decides: inspect, then SIGCONT or stop')

    def run(self, once=False, max_polls=None):
        n = 0
        while True:
            n += 1
            r = self.poll()
            if r:
                return self.act(r)
            if once or (max_polls and n >= max_polls):
                return 'no_action'
            self.o.sleep(float(self.c.get('poll_s', 10)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True); ap.add_argument('--state-dir', type=Path, required=True)
    ap.add_argument('--observe-only', action='store_true'); ap.add_argument('--once', action='store_true')
    ap.add_argument('mode', nargs='?', default='watch', choices=('watch', 'snapshot'))
    a = ap.parse_args(argv)
    cfg = json.loads(a.config.read_text())
    g = TrainGuardian(cfg, HostOS(cfg.get('harness', ['python3', 'scripts/utils/groot_harness.py'])), a.state_dir, observe_only=a.observe_only)
    if a.mode == 'snapshot':
        try:
            g.refresh()
        except Unproven as e:
            print(json.dumps({'unproven': str(e)})); return 6
        print(json.dumps({'owner': g.owner, 'owned': [[p, s, r] for (p, s), r in g.owned.items()]}, default=str))
        return 0
    r = g.run(once=a.once)
    print(json.dumps({'result': r}))
    return {'no_action': 0, 'observed': 0, 'released': 0, 'released_elsewhere': 0, 'kept': 5, 'refused': 6}.get(r, 1)


if __name__ == '__main__':
    sys.exit(main())

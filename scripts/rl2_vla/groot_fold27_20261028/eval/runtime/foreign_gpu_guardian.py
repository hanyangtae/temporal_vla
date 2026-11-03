"""Runtime foreign-compute guardian for ONE host GPU running the fold27 per-GPU EVAL consumer (new; nothing existing changed).

SCOPE (explicit): eval only = eval_gpu_queue(_v2).sh follower -> groot_harness run -> eval_gpu_queue(_v2).py run consumer ->
run_eval_arm(_v2).py arms -> host-conda serve sessions + docker-exec collectors in the label-verified isolated env container.
NOT covered: native latents (run_latents_v2 / t49 chains), trainers (train_queue / finetune), kanu docker_run eval. For those
it does nothing at all.

Observe: every poll, nvidia-smi compute PIDs of the configured GPU vs the OWNED set:
  consumer   consumers/gpu<G>.json in the queue dir (receipt + consumer pid/starttime/boot_id); alive with the same start
             time and boot id, cmdline names eval_gpu_queue ... run
  tree       all descendants of the consumer (/proc PPid walk; run_eval_arm, serve, docker exec) recorded as
             (pid, starttime) and kept STICKY (a serve re-parented to init after its runner died stays owned)
  container  processes whose cgroup carries the full id of the configured env container, only if iso_env_container.gate
             passes for (gpu, deploy[, image]) AND the cmdline names the collector (collect_task / http_feature_collect)
  follower   the ancestor bash loop running eval_gpu_queue(_v2).sh (not --inner), or the configured pid+starttime
A GPU PID outside the owned set, seen in >= confirm_polls consecutive polls with the same start time, is FOREIGN.
Action (once; skipped in --observe-only):
  0 ownership re-verified: consumer identity + its receipt in `groot_harness.py status` with our session/machine/[gpu]
    -> otherwise ALERT ownership_unverified and NO signal at all (fail closed)
  1 freeze: SIGSTOP follower, SIGSTOP consumer (no new launch), SIGSTOP arms (no new episode child), then the final
    owned-tree scan (every launcher frozen -> complete)
  2 arms: SIGINT + SIGCONT each owned run_eval_arm (its finally stops its serve session); wait grace_s
  3 leftovers: owned serve sessions killpg TERM then KILL; owned docker-exec clients TERM; owned in-container collectors
    TERM (PermissionError -> `docker exec <container> kill -s TERM <ns pid>`), then KILL
  4 consumer TERM+CONT, follower TERM+CONT (verified identities only)
  5 release: only when every owned identity is gone, no owned PID is on the GPU (query must succeed), the harness run
    wrapper of that receipt has exited, and the receipt is still ours -> `groot_harness.py release --receipt --session`
    (the harness ledger is never edited directly). Otherwise ALERT and keep the receipt.
Every signal re-checks pid + starttime + boot id immediately before os.kill. Unknown / foreign / unverifiable processes
are never signalled. Files: only <state-dir>/ (events.jsonl, OWNED.json, ACTION.<t>.json, ALERT.json); progress, attempts,
claims, logs are never touched (claims of dead owners become stale and are preserved by ClaimStore later).

    python foreign_gpu_guardian.py --config <cfg.json> --state-dir <dir> [--observe-only] [--once]
    python foreign_gpu_guardian.py --config <cfg.json> --state-dir <dir> snapshot      # read-only ownership dump
"""
import argparse, json, os, signal, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import job_claims as JC           # noqa: E402  (boot_id / proc_start)
import iso_env_container as ISO   # noqa: E402  (docker_inspect / gate)

KIND = 'fold27_foreign_gpu_guardian_v1'
CONSUMER_MARK = ('eval_gpu_queue',)
FOLLOWER_MARK = ('eval_gpu_queue_v2.sh', 'eval_gpu_queue.sh')
ARM_MARK = 'run_eval_arm'
SERVE_MARK = 'serve_mil_qam'
COLLECT_MARK = ('collect_task', 'http_feature_collect')
HARNESS_MARK = 'groot_harness.py'


class Unverified(RuntimeError):
    pass


class OS:
    """Real host interface (tests substitute a fake with the same methods)."""

    def __init__(self, harness_cmd, env=None):
        self.harness_cmd, self.env = list(harness_cmd), env

    def pids(self):
        return [int(p.name) for p in Path('/proc').iterdir() if p.name.isdigit()]

    def start(self, pid):
        return JC.proc_start(pid)

    def boot(self):
        return JC.boot_id()

    def _status(self, pid):
        try:
            return dict(l.split(':', 1) for l in Path(f'/proc/{pid}/status').read_text().splitlines() if ':' in l)
        except OSError:
            return {}

    def ppid(self, pid):
        v = self._status(pid).get('PPid')
        return int(v) if v else None

    def uid(self, pid):
        v = self._status(pid).get('Uid')
        return int(v.split()[0]) if v else None

    def nspid(self, pid):
        v = self._status(pid).get('NSpid')
        return int(v.split()[-1]) if v else None

    def cmdline(self, pid):
        try:
            return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace').strip()
        except OSError:
            return None

    def cgroup(self, pid):
        try:
            return Path(f'/proc/{pid}/cgroup').read_text()
        except OSError:
            return ''

    def pgid(self, pid):
        return os.getpgid(pid)

    def gpu_pids(self, gpu):
        r = subprocess.run(['nvidia-smi', f'--id={gpu}', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            raise RuntimeError(f'nvidia-smi failed: {r.stderr.strip()[:200]}')
        return [int(x) for x in r.stdout.split() if x.strip().isdigit()]

    def kill(self, pid, sig):
        os.kill(pid, sig)

    def killpg(self, pgid, sig):
        os.killpg(pgid, sig)

    def docker_inspect(self, name):
        return ISO.docker_inspect(name)

    def container_kill(self, name, nspid, sig):
        r = subprocess.run(['docker', 'exec', name, 'kill', '-s', signal.Signals(sig).name.replace('SIG', ''), str(nspid)],
                           capture_output=True, text=True, timeout=30)
        return r.returncode == 0

    def harness(self, *args):
        r = subprocess.run([*self.harness_cmd, *args], capture_output=True, text=True, timeout=120, env=self.env)
        if r.returncode != 0:
            raise RuntimeError(f'harness {args[0]} failed: {r.stderr.strip()[:300]}')
        return json.loads(r.stdout)

    def now(self):
        return time.time()

    def sleep(self, s):
        time.sleep(s)


class Guardian:
    def __init__(self, cfg, osi, state_dir, observe_only=False):
        if cfg.get('kind') != KIND:
            raise SystemExit(f'config kind must be {KIND}')
        self.c, self.o, self.sd = cfg, osi, Path(state_dir)
        self.sd.mkdir(parents=True, exist_ok=True)
        self.observe_only = observe_only or bool(cfg.get('observe_only'))
        self.gpu, self.boot0 = int(cfg['gpu']), osi.boot()
        self.owned = {}          # (pid, start) -> role ; sticky
        self.seen = {}           # (pid, start) -> consecutive polls unrecognised
        self.consumer = None     # {'pid','start','receipt'}
        self.acted = False

    # ---------------- records ----------------
    def event(self, kind, **kw):
        with open(self.sd / 'events.jsonl', 'a') as f:
            f.write(json.dumps({'kind': kind, 'time': self.o.now(), **kw}, default=str) + '\n')

    def alert(self, why, **kw):
        tmp = self.sd / f'ALERT.{os.getpid()}.tmp'
        tmp.write_text(json.dumps({'time': self.o.now(), 'reason': why, **kw}, indent=1, default=str)); os.replace(tmp, self.sd / 'ALERT.json')
        self.event('alert', reason=why, **kw)

    # ---------------- identity ----------------
    def same(self, pid, start):
        return start is not None and self.o.boot() == self.boot0 and self.o.start(pid) == start

    def cmd_has(self, pid, marks):
        c = self.o.cmdline(pid) or ''
        return any(m in c for m in ((marks,) if isinstance(marks, str) else marks))

    def read_consumer(self):
        p = Path(self.c['queue_dir']) / 'consumers' / f'gpu{self.gpu}.json'
        if not p.exists():
            return None
        r = json.loads(p.read_text())
        ident = r.get('consumer') or {}
        pid, start = ident.get('pid'), ident.get('start')
        if r.get('boot_id') != self.o.boot() or not pid or not self.same(pid, start):
            return None
        cmd = self.o.cmdline(pid) or ''
        if not (any(m in cmd for m in CONSUMER_MARK) and ' run' in f' {cmd}'):
            return None
        return {'pid': pid, 'start': start, 'receipt': r.get('receipt')}

    def ancestors(self, pid, depth=6):
        out, cur = [], pid
        for _ in range(depth):
            cur = self.o.ppid(cur)
            if not cur or cur <= 1:
                break
            out.append(cur)
        return out

    def follower(self, consumer):
        cand = []
        if consumer:
            for a in self.ancestors(consumer['pid']):
                c = self.o.cmdline(a) or ''
                if any(m in c for m in FOLLOWER_MARK) and '--inner' not in c:
                    cand.append((a, self.o.start(a)))
                    break
        f = self.c.get('follower')
        if f:
            if not self.same(f['pid'], f['start']) or not self.cmd_has(f['pid'], FOLLOWER_MARK):
                f = None
            elif cand and cand[0][0] != f['pid']:
                raise Unverified(f'configured follower {f["pid"]} != ancestor follower {cand[0][0]}')
            else:
                cand = [(f['pid'], f['start'])]
        return cand[0] if cand else None

    def harness_wrappers(self, consumer):
        return [(a, self.o.start(a)) for a in self.ancestors(consumer['pid']) if self.cmd_has(a, HARNESS_MARK)] if consumer else []

    def container_id(self):
        name = self.c.get('env_container')
        if not name:
            return None
        try:
            info = self.o.docker_inspect(name)
            ISO.gate(info, self.gpu, self.c['deploy'], self.c.get('expect_image'))
            return info['Id']
        except Exception as e:   # noqa: BLE001 - unverifiable container: its processes are NOT owned
            self.event('container_unverified', error=str(e)[:300])
            return None

    def refresh(self):
        """Recompute the owned set (sticky) from the live consumer tree + verified container."""
        self.consumer = self.read_consumer() or self.consumer
        kids = {}
        for p in self.o.pids():
            pp = self.o.ppid(p)
            if pp:
                kids.setdefault(pp, []).append(p)
        if self.consumer and self.same(self.consumer['pid'], self.consumer['start']):
            self.owned[(self.consumer['pid'], self.consumer['start'])] = 'consumer'
            stack = list(kids.get(self.consumer['pid'], []))
            while stack:
                p = stack.pop()
                st = self.o.start(p)
                if st is None:
                    continue
                c = self.o.cmdline(p) or ''
                role = 'arm' if ARM_MARK in c else 'serve' if SERVE_MARK in c else 'docker_exec' if c.startswith('docker exec') else 'desc'
                self.owned.setdefault((p, st), role)
                stack += kids.get(p, [])
        cid = self.container_id()
        if cid:
            for p in self.o.pids():
                if cid in self.o.cgroup(p) and self.cmd_has(p, COLLECT_MARK):
                    st = self.o.start(p)
                    if st is not None:
                        self.owned.setdefault((p, st), 'env_collector')
        self.owned = {k: v for k, v in self.owned.items() if self.same(*k)}      # drop dead identities
        tmp = self.sd / 'OWNED.tmp'
        tmp.write_text(json.dumps({'consumer': self.consumer, 'owned': [[p, s, r] for (p, s), r in sorted(self.owned.items())]}, indent=1))
        os.replace(tmp, self.sd / 'OWNED.json')

    def owned_pid(self, pid):
        st = self.o.start(pid)
        return st is not None and (pid, st) in self.owned

    # ---------------- poll ----------------
    def poll(self):
        try:
            gp = self.o.gpu_pids(self.gpu)
        except Exception as e:   # noqa: BLE001 - uncertainty: observe nothing, act on nothing
            self.event('gpu_query_failed', error=str(e)[:300]); self.seen.clear(); return None
        self.refresh()
        unknown = {}
        for p in gp:
            if self.owned_pid(p):
                continue
            st = self.o.start(p)
            if st is None:                                   # vanished between nvidia-smi and /proc: not present
                continue
            unknown[(p, st)] = self.seen.get((p, st), 0) + 1
        self.seen = unknown
        confirmed = [k for k, n in unknown.items() if n >= int(self.c.get('confirm_polls', 2))]
        if unknown:
            self.event('unrecognized_gpu_process', gpu=self.gpu, pids=[[p, s, n, self.o.uid(p), (self.o.cmdline(p) or '')[:160]]
                                                                      for (p, s), n in unknown.items()])
        if confirmed and self.read_consumer() is None:
            # nothing of ours is running on this GPU (harness admission already refuses while foreign compute exists)
            self.event('foreign_while_idle', pids=[list(k) for k in confirmed]); return None
        return confirmed or None

    # ---------------- action ----------------
    def verify_ownership(self):
        cons = self.read_consumer()
        if cons is None:
            raise Unverified('no live, identity-verified consumer registered for this GPU')
        st = self.o.harness('status')
        it = st.get(cons['receipt'])
        if not it:
            raise Unverified(f'receipt {cons["receipt"]} not in harness ledger')
        if it.get('session') != self.c['session'] or it.get('machine') != self.c['host'] or it.get('gpus') != [self.gpu]:
            raise Unverified(f'receipt {cons["receipt"]} is not session {self.c["session"]} on {self.c["host"]} GPU [{self.gpu}]')
        return cons

    def sig(self, pid, start, s, why, container=False):
        if not self.same(pid, start):
            self.event('signal_skipped_identity_changed', pid=pid, start=start, sig=int(s), why=why); return False
        try:
            self.o.kill(pid, s)
        except ProcessLookupError:
            return False
        except PermissionError:
            if container and self.c.get('env_container'):
                ns = self.o.nspid(pid)
                ok = bool(ns) and self.same(pid, start) and self.o.container_kill(self.c['env_container'], ns, s)
                self.event('signal_via_container', pid=pid, nspid=ns, sig=int(s), ok=ok, why=why); return ok
            self.event('signal_permission_denied', pid=pid, sig=int(s), why=why); return False
        self.event('signal', pid=pid, start=start, sig=int(s), why=why)
        return True

    def sigpg(self, pid, start, s, why):
        if not self.same(pid, start):
            self.event('signal_skipped_identity_changed', pid=pid, start=start, sig=int(s), why=why); return False
        try:
            if self.o.pgid(pid) != pid:
                return self.sig(pid, start, s, why + ' (not a group leader: pid only)')
            self.o.killpg(pid, s)
        except ProcessLookupError:
            return False
        self.event('signal_group', pgid=pid, start=start, sig=int(s), why=why)
        return True

    def wait(self, pred, timeout):
        t_end = self.o.now() + timeout
        while not pred():
            if self.o.now() >= t_end:
                return False
            self.o.sleep(1)
        return True

    def alive_roles(self, *roles):
        return [(p, s) for (p, s), r in self.owned.items() if r in roles and self.same(p, s)]

    def act(self, confirmed):
        rec = {'gpu': self.gpu, 'foreign': [[p, s, self.o.uid(p), (self.o.cmdline(p) or '')[:200]] for p, s in confirmed],
               'observe_only': self.observe_only}
        try:
            cons = self.verify_ownership()
        except Unverified as e:
            self.alert('ownership_unverified: no signal sent', detail=str(e), **rec); return 'refused'
        self.refresh()
        try:
            fol = self.follower(cons)
        except Unverified as e:
            self.alert('ownership_unverified: no signal sent', detail=str(e), **rec); return 'refused'
        wrappers = self.harness_wrappers(cons)
        rec.update(consumer=cons, follower=fol, harness_wrappers=wrappers, owned=[[p, s, r] for (p, s), r in self.owned.items()])
        with open(self.sd / f'ACTION.{int(self.o.now() * 1000)}.json', 'x') as f:
            f.write(json.dumps(rec, indent=1, default=str))
        if self.observe_only:
            self.event('observe_only_would_act', **rec); return 'observed'
        self.acted = True
        cpid, cst = cons['pid'], cons['start']
        if fol:
            self.sig(fol[0], fol[1], signal.SIGSTOP, 'freeze follower (no new admission)')
        self.sig(cpid, cst, signal.SIGSTOP, 'freeze consumer (no new arm launch)')
        self.refresh()
        for p, s in self.alive_roles('arm'):
            self.sig(p, s, signal.SIGSTOP, 'freeze arm (no new episode child) before the final tree scan')
        self.refresh()                                       # complete tree while every launcher is frozen
        rec['owned_at_stop'] = [[p, s, r] for (p, s), r in self.owned.items()]
        self.event('owned_at_stop', owned=rec['owned_at_stop'])
        for p, s in self.alive_roles('arm'):
            self.sig(p, s, signal.SIGINT, 'arm graceful stop (finally stops its serve)')
            self.sig(p, s, signal.SIGCONT, 'resume arm to handle SIGINT')
        self.wait(lambda: not self.alive_roles('arm'), float(self.c.get('grace_s', 90)))
        for p, s in self.alive_roles('arm'):
            self.sig(p, s, signal.SIGTERM, 'arm did not stop after SIGINT')
        for p, s in self.alive_roles('serve'):
            self.sigpg(p, s, signal.SIGTERM, 'owned serve session')
        for p, s in self.alive_roles('docker_exec'):
            self.sig(p, s, signal.SIGTERM, 'owned docker exec client')
        for p, s in self.alive_roles('env_collector'):
            self.sig(p, s, signal.SIGTERM, 'owned collector in verified env container', container=True)
        kw = float(self.c.get('kill_wait_s', 30))
        self.wait(lambda: not self.alive_roles('arm', 'serve', 'docker_exec', 'env_collector', 'desc'), kw)
        for p, s in self.alive_roles('serve'):
            self.sigpg(p, s, signal.SIGKILL, 'owned serve session still alive')
        for p, s in self.alive_roles('arm', 'docker_exec', 'desc'):
            self.sig(p, s, signal.SIGKILL, 'owned process still alive')
        for p, s in self.alive_roles('env_collector'):
            self.sig(p, s, signal.SIGKILL, 'owned collector still alive', container=True)
        for p, s, why in ((cpid, cst, 'consumer'), *(((fol[0], fol[1], 'follower'),) if fol else ())):
            self.sig(p, s, signal.SIGTERM, f'stop {why}'); self.sig(p, s, signal.SIGCONT, f'deliver TERM to stopped {why}')
        self.wait(lambda: not self.same(cpid, cst) and not (fol and self.same(*fol)), kw)
        for p, s in ((cpid, cst), *((fol,) if fol else ())):
            if self.same(p, s):
                self.sig(p, s, signal.SIGKILL, 'consumer/follower still alive')
        return self.release(cons, wrappers)

    def own_absent(self):
        if any(self.same(p, s) for (p, s) in self.owned):
            return False, 'owned process alive'
        try:
            gp = self.o.gpu_pids(self.gpu)
        except Exception as e:   # noqa: BLE001
            return False, f'gpu query failed: {e}'
        mine = [p for p in gp if self.owned_pid(p)]
        return (not mine), (f'owned pids on GPU {mine}' if mine else 'ok')

    def release(self, cons, wrappers):
        t_end = self.o.now() + float(self.c.get('release_wait_s', 600))
        why = ''
        while self.o.now() < t_end:
            ok, why = self.own_absent()
            wr = [w for w in wrappers if self.same(*w)]
            if ok and not wr:
                break
            why = why if not ok else f'harness run wrapper alive {wr}'
            self.o.sleep(2)
        else:
            self.alert('receipt kept: own processes not proven absent', receipt=cons['receipt'], detail=why); return 'kept'
        try:
            it = self.o.harness('status').get(cons['receipt'])
        except Exception as e:   # noqa: BLE001
            self.alert('receipt kept: harness status unavailable', receipt=cons['receipt'], detail=str(e)[:300]); return 'kept'
        if it is None:
            self.event('receipt_already_released', receipt=cons['receipt']); return 'released_elsewhere'
        if it.get('session') != self.c['session'] or it.get('gpus') != [self.gpu]:
            self.alert('receipt kept: ledger entry no longer ours', receipt=cons['receipt']); return 'kept'
        try:
            self.o.harness('release', '--receipt', cons['receipt'], '--session', self.c['session'])
        except Exception as e:   # noqa: BLE001
            self.alert('receipt release failed', receipt=cons['receipt'], detail=str(e)[:300]); return 'kept'
        self.event('receipt_released', receipt=cons['receipt'])
        return 'released'

    def run(self, once=False, max_polls=None):
        n = 0
        while True:
            n += 1
            confirmed = self.poll()
            if confirmed:
                return self.act(confirmed)
            if once or (max_polls and n >= max_polls):
                return 'no_action'
            self.o.sleep(float(self.c.get('poll_s', 5)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True); ap.add_argument('--state-dir', type=Path, required=True)
    ap.add_argument('--observe-only', action='store_true'); ap.add_argument('--once', action='store_true')
    ap.add_argument('mode', nargs='?', default='watch', choices=('watch', 'snapshot'))
    a = ap.parse_args(argv)
    cfg = json.loads(a.config.read_text())
    g = Guardian(cfg, OS(cfg.get('harness', ['python3', 'scripts/utils/groot_harness.py'])), a.state_dir, a.observe_only)
    if a.mode == 'snapshot':
        g.refresh()
        print(json.dumps({'consumer': g.consumer, 'follower': g.follower(g.consumer) if g.consumer else None,
                          'owned': [[p, s, r] for (p, s), r in g.owned.items()]}, default=str))
        return 0
    r = g.run(once=a.once)
    print(json.dumps({'result': r}))
    return {'no_action': 0, 'observed': 0, 'released': 0, 'released_elsewhere': 0, 'kept': 5, 'refused': 6}.get(r, 1)


if __name__ == '__main__':
    sys.exit(main())

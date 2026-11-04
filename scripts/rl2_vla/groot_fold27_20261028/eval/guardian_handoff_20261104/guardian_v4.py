"""foreign_gpu_guardian v4 = v3 (unchanged; v1/v2/v3 files frozen) + guardian<->launcher HANDOFF, so that no eval consumer
can start GPU work unless THIS guardian process has adopted exactly that consumer and its protection loop is running.

Why (srv50 GPU3, 2026-11-04; see RECEIPT.md):
  1. v1 run() is one-shot: `if confirmed: return act(confirmed)` -> after one stop/release the guardian process exits.
  2. v1 FOLLOWER_MARK = ('eval_gpu_queue_v2.sh', 'eval_gpu_queue.sh') does not match eval_gpu_queue_v3.sh, so the v3 outer
     loop was neither frozen nor stopped. The consumer got SIGTERM -> `groot_harness.py run` returns child.wait() == -15
     -> sys.exit(-15) -> exit status 241 -> the v3 loop's `*)` branch sleeps POLL_S and launches a NEW consumer.
  3. That new consumer ran with no guardian: the old one was still bound to the old OWNED identities / waiting up to
     release_wait_s (600 s) for the old harness wrapper, and then exited.

v4 changes (only these):
  * persistent: after act() the guardian keeps polling (never returns on its own). 'kept' / 'refused' (ownership or
    absence not proven) latch HOLD.json -> no new consumer is adopted until an operator removes it (fail closed).
  * follower marks also match eval_gpu_queue_v3.sh / eval_gpu_queue_v4.sh (v1 marks unchanged for v1-v3).
  * adoption only by handshake (handoff/request.<nonce>.json from handoff_gate.py run by eval_gpu_queue_v4.sh --inner
    BEFORE it execs the consumer, i.e. before any GPU work). Accepted iff: same boot; requester pid+start alive; its
    cmdline is `eval_gpu_queue_v4.sh --inner`; its parent is the `groot_harness.py run` wrapper; host/gpu/session equal
    the config; receipt in `groot_harness.py status` with this session/machine/[gpu] AND consumer_pid == requester pid
    (set by the harness verify); receipt never seen before by this guardian (an old/returned receipt is refused); no
    other live consumer adopted or registered for this GPU (duplicate launcher); the last poll succeeded with no
    unrecognised GPU process and no low-headroom streak; not on HOLD. Otherwise a NACK with the reason.
    The ack carries nonce/receipt/pid/start/boot + this guardian's pid/start; handoff_gate.py checks all of them.
  * a consumer registration (consumers/gpu<G>.json) that was NOT adopted while an adopted consumer lives -> HOLD.
    With no adopted consumer (guardian restarted under a running consumer) the v1 registration is adopted only after
    the same harness receipt check (consumer_pid must match); otherwise it is not owned.
  * single instance per (queue, GPU): flock on <queue_dir>/guardians/gpu<G>.lock (new file; nothing existing touched).
  * (2nd round) the request must name a live consumer_watchdog.py for this consumer+nonce running OUTSIDE the consumer
    tree; heartbeat phase 'acting' is written before act(); a lost watchdog while its consumer runs -> HOLD (no signal).
  * heartbeat handoff/guardian.json every loop (pid/start/boot/time/hold/adopted); handoff_gate.py `ready` requires it.
Sharing contract handling, foreign detection, signal procedure and release are v3/v2/v1 code, unchanged.

    python guardian_v4.py --config <cfg.json> --state-dir <dir> [--observe-only]
"""
import argparse, fcntl, json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RT = HERE.parent / 'runtime'
sys.path[:0] = [str(HERE), str(RT), str(HERE.parents[4] / 'scripts/utils')]
import foreign_gpu_guardian as G1          # noqa: E402  (unchanged)
import foreign_gpu_guardian_v3 as G3       # noqa: E402  (unchanged)
import job_claims as JC                    # noqa: E402

FOLLOWER_MARK4 = G1.FOLLOWER_MARK + ('eval_gpu_queue_v3.sh', 'eval_gpu_queue_v4.sh')
INNER_MARK = 'eval_gpu_queue_v4.sh'


class OS4(G1.OS):
    def self_ident(self):
        return {'pid': os.getpid(), 'start': JC.proc_start(os.getpid())}


def lock_path(cfg, gpu):
    d = Path(cfg['queue_dir']) / 'guardians'; d.mkdir(parents=True, exist_ok=True)
    return d / f'gpu{int(gpu)}.lock'


def write_json(path, obj):
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, indent=1, default=str)); os.replace(tmp, path)


class GuardianV4(G3.GuardianV3):
    def __init__(self, cfg, osi, state_dir, observe_only=False, probe=None, take_lock=True):
        super().__init__(cfg, osi, state_dir, observe_only, probe)
        self._lock = None
        if take_lock:                                    # False only for consumer_watchdog's orphan stop (see there)
            self._lock = open(lock_path(cfg, self.gpu), 'a')
            try:
                fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                self._lock.close()
                raise SystemExit(f'another guardian holds {lock_path(cfg, self.gpu)} (duplicate guardian refused)')
        self.hd = self.sd / 'handoff'; self.hd.mkdir(exist_ok=True)
        self.me = osi.self_ident()
        self.adopted = None              # {'pid','start','receipt'}
        self.retired = set()             # receipts this guardian has already seen end / acted on
        self.last_ok = False             # last poll: GPU query succeeded and nothing unrecognised
        self.observed = set()
        hp = self.sd / 'HOLD.json'
        self.hold = json.loads(hp.read_text()).get('reason', 'hold') if hp.exists() else None

    def close(self):
        if self._lock and not self._lock.closed:
            self._lock.close()

    # ---------------- state ----------------
    def set_hold(self, why, **kw):
        if self.hold is None:
            self.hold = why
            write_json(self.sd / 'HOLD.json', {'time': self.o.now(), 'reason': why, **kw})
            self.alert('HOLD: no new consumer will be adopted until HOLD.json is removed by an operator', reason2=why, **kw)

    def heartbeat(self, phase='watch'):
        write_json(self.hd / 'guardian.json', {'guardian': self.me, 'boot_id': self.boot0, 'time': self.o.now(), 'phase': phase,
                                               'host': self.c['host'], 'gpu': self.gpu, 'session': self.c['session'],
                                               'hold': self.hold, 'adopted': self.adopted, 'fenced': sorted(self.fenced()),
                                               'poll_s': float(self.c.get('poll_s', 5))})

    def fenced(self):
        """consumer identities a consumer_watchdog has fenced: this guardian never adopts them (see consumer_watchdog)."""
        out = set()
        for p in self.hd.glob('fence.*.json'):
            try:
                f = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            if f.get('boot_id') == self.boot0:
                out.add((f['consumer']['pid'], f['consumer']['start']))
        return out

    def alive_adopted(self):
        a = self.adopted
        return a if a and self.same(a['pid'], a['start']) else None

    def receipt_problem(self, receipt, pid):
        try:
            it = self.o.harness('status').get(receipt)
        except Exception as e:   # noqa: BLE001
            return f'harness status unavailable: {str(e)[:200]}'
        if not it:
            return f'receipt {receipt} not in harness ledger'
        if it.get('session') != self.c['session'] or it.get('machine') != self.c['host'] or it.get('gpus') != [self.gpu]:
            return f'receipt {receipt} is not session {self.c["session"]} on {self.c["host"]} GPU [{self.gpu}]'
        if it.get('consumer_pid') != pid:
            return f'receipt {receipt} consumer_pid {it.get("consumer_pid")} != {pid}'
        return None

    # ---------------- identity (v1 hooks) ----------------
    def read_consumer(self):
        reg = super().read_consumer()                # v1: registration alive, same start/boot, consumer cmdline
        a = self.alive_adopted()
        if a:
            if reg and ((reg['pid'], reg['start']) != (a['pid'], a['start']) or reg['receipt'] != a['receipt']):
                self.set_hold('consumer registered for this GPU that was not adopted (launched outside the handoff)',
                              registered=reg, adopted=a)
            return dict(a)
        if reg:                                      # guardian (re)started under a running consumer
            why = 'fenced by its watchdog' if (reg['pid'], reg['start']) in self.fenced() else \
                'receipt already retired' if reg['receipt'] in self.retired else (
                'on hold' if self.hold else self.receipt_problem(reg['receipt'], reg['pid']))
            if why:
                self.event('registration_not_adopted', consumer=reg, why=why); return None
            self.adopted = dict(reg); self.event('adopted_registered_consumer', consumer=reg)
            return dict(reg)
        return None

    def verify_ownership(self):
        cons = super().verify_ownership()
        why = self.receipt_problem(cons['receipt'], cons['pid'])
        if why:
            raise G1.Unverified(why)
        return cons

    def follower(self, consumer):
        cand = []
        if consumer:
            for a in self.ancestors(consumer['pid']):
                c = self.o.cmdline(a) or ''
                if any(m in c for m in FOLLOWER_MARK4) and '--inner' not in c:
                    cand.append((a, self.o.start(a))); break
        f = self.c.get('follower')
        if f:
            if not self.same(f['pid'], f['start']) or not self.cmd_has(f['pid'], FOLLOWER_MARK4):
                f = None
            elif cand and cand[0][0] != f['pid']:
                raise G1.Unverified(f'configured follower {f["pid"]} != ancestor follower {cand[0][0]}')
            else:
                cand = [(f['pid'], f['start'])]
        return cand[0] if cand else None

    def event(self, kind, **kw):
        if kind == 'gpu_query_failed':
            self._qfail = True
        super().event(kind, **kw)

    def poll(self):
        self.last_ok, self._qfail = False, False
        r = super().poll()
        self.last_ok = not self._qfail and not self.seen and not getattr(self, 'low_mem', 0) and r is None
        return r

    # ---------------- handoff ----------------
    def judge(self, req):
        pid, start = req.get('pid'), req.get('start')
        if (req.get('host'), req.get('gpu'), req.get('session')) != (self.c['host'], self.gpu, self.c['session']):
            return 'host/gpu/session differ from this guardian'
        if req.get('boot_id') != self.o.boot() or not isinstance(pid, int) or not self.same(pid, start):
            return 'requester not alive with that pid/start/boot (PID reuse or exited)'
        cmd = self.o.cmdline(pid) or ''
        if INNER_MARK not in cmd or '--inner' not in cmd:
            return 'requester is not eval_gpu_queue_v4.sh --inner'
        if not self.cmd_has(self.o.ppid(pid) or 0, G1.HARNESS_MARK):
            return 'requester parent is not the groot_harness.py run wrapper'
        w = req.get('watchdog') or {}
        if not isinstance(w.get('pid'), int) or not self.same(w['pid'], w.get('start')):
            return 'consumer watchdog not alive with that pid/start'
        wc = self.o.cmdline(w['pid']) or ''
        if 'consumer_watchdog.py' not in wc or f'--consumer-pid {pid}' not in wc or f'--nonce {req.get("nonce")}' not in wc:
            return 'watchdog cmdline does not name this consumer/nonce'
        if pid in self.ancestors(w['pid']):
            return 'watchdog is inside the consumer tree (would be stopped with it)'
        if self.hold:
            return f'guardian on HOLD: {self.hold}'
        if (pid, start) in self.fenced():
            return 'consumer identity fenced by a watchdog'
        if req.get('receipt') in self.retired:
            return 'receipt already used/retired (old receipt returned)'
        if self.alive_adopted():
            return f'another consumer is adopted on GPU {self.gpu} (duplicate launcher)'
        if G1.Guardian.read_consumer(self):
            return 'a live consumer is registered for this GPU (not adoptable here)'
        if not self.last_ok:
            return 'last poll not clean (GPU query failed, unrecognised GPU process or low headroom)'
        return self.receipt_problem(req.get('receipt'), pid)

    def serve_requests(self):
        for p in sorted(self.hd.glob('request.*.json')):
            nonce = p.name[len('request.'):-len('.json')]
            if (self.hd / f'ack.{nonce}.json').exists() or (self.hd / f'nack.{nonce}.json').exists():
                continue
            try:
                req = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            if req.get('nonce') != nonce:
                continue
            why = self.judge(req)
            base = {'nonce': nonce, 'receipt': req.get('receipt'), 'pid': req.get('pid'), 'start': req.get('start'),
                    'boot_id': self.boot0, 'guardian': self.me, 'time': self.o.now()}
            if why:
                write_json(self.hd / f'nack.{nonce}.json', {**base, 'accepted': False, 'reason': why})
                self.event('handoff_refused', request=req, why=why); continue
            self.adopted = {'pid': req['pid'], 'start': req['start'], 'receipt': req['receipt'],
                            'watchdog': {'pid': req['watchdog']['pid'], 'start': req['watchdog']['start']}}
            self.consumer = dict(self.adopted); self.seen.clear()
            self.refresh()                                   # owned now contains the adopted identity
            write_json(self.hd / f'ack.{nonce}.json', {**base, 'accepted': True})
            self.event('handoff_adopted', consumer=self.adopted)

    def after_action(self, result, confirmed):
        if result == 'observed':
            self.observed |= set(confirmed); return
        for c in (self.adopted, self.consumer):
            if c and c.get('receipt'):
                self.retired.add(c['receipt'])
        self.adopted = None; self.seen.clear()
        if result in ('kept', 'refused'):
            self.set_hold(f'guardian action ended {result!r}: ownership or absence not proven', result=result)
        self.event('action_finished_guarding_continues', result=result)

    def step(self):
        hp = self.sd / 'HOLD.json'
        if self.hold is None and hp.exists():                # e.g. written by a consumer_watchdog stop
            self.hold = (json.loads(hp.read_text()) or {}).get('reason', 'hold')
        confirmed = self.poll()
        result = None
        if confirmed and not (self.observe_only and set(confirmed) <= self.observed):
            self.heartbeat('acting')                         # act/release may block for minutes: watchdog must not
            result = self.act(confirmed)                     # read a stale heartbeat as a lost guardian
            self.after_action(result, confirmed)
        a = self.adopted
        if a and not self.same(a['pid'], a['start']):        # adopted consumer ended normally
            self.retired.add(a['receipt']); self.adopted = None; self.event('adopted_consumer_ended', consumer=a)
        elif a and a.get('watchdog') and not self.same(a['watchdog']['pid'], a['watchdog']['start']):
            self.set_hold('consumer watchdog lost while its consumer runs (guardian keeps protecting it)', adopted=a)
        self.serve_requests()
        self.heartbeat()
        return result

    def run(self, once=False, max_polls=None):
        n, results = 0, []
        while True:
            n += 1
            r = self.step()
            if r:
                results.append(r)
            if once or (max_polls and n >= max_polls):
                return results
            self.o.sleep(float(self.c.get('poll_s', 5)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True); ap.add_argument('--state-dir', type=Path, required=True)
    ap.add_argument('--observe-only', action='store_true')
    a = ap.parse_args(argv)
    cfg = json.loads(a.config.read_text())
    g = GuardianV4(cfg, OS4(cfg.get('harness', ['python3', 'scripts/utils/groot_harness.py'])), a.state_dir, a.observe_only)
    g.run()                                                  # runs until killed
    return 0


if __name__ == '__main__':
    sys.exit(main())

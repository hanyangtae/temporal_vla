"""Per-consumer watchdog for guardian_v4: when the guardian that adopted THIS consumer is lost or hung, stop THIS consumer
(only its verified tree) with the unchanged v1 safe procedure, so a consumer never keeps running unprotected.

Started by eval_gpu_queue_v4.sh --inner BEFORE the handoff request, detached outside the consumer tree (subshell + setsid,
re-parented), so a guardian stop never kills it as a consumer descendant; the guardian refuses the handoff otherwise.

Every poll (watchdog_poll_s, default poll_s):
  ended     consumer identity (pid+start+boot) gone -> exit, no action (normal end; the harness run wrapper releases
            the receipt on its own path, untouched).
  armed     only after ack.<nonce>.json accepted THIS consumer (pid+start+receipt+boot). Before that the launcher has not
            exec'd the consumer, so nothing to protect.
  healthy   the heartbeat guardian is alive (pid+start+boot), adopted THIS consumer and its heartbeat is younger than the
            phase deadline -> nothing (fence withdrawn if any).
  hung      the heartbeat guardian is alive but its heartbeat is older than the BOUNDED phase deadline:
              watch : hang_s      = 3*poll_s + nvidia-smi timeout (60) + 2 harness calls (2*120) + 60      (default)
              acting: act_deadline = grace_s + 2*kill_wait_s + release_wait_s + 3*120 + 60 + 60           (v1 defaults:
                      90+60+600+480 = 1230 s) -- so a normal act/release (<= ~750 s) is never misread as a hang.
            The deadline must hold for confirm_polls+1 polls; then the watchdog stops THAT guardian process (our own:
            identity re-checked, cmdline guardian_v4.py; TERM, then KILL after kill_wait_s). Its flock is dropped by the
            kernel, and the 'lost' path below applies. A guardian is never trusted forever just because its PID lives.
  lost      no known guardian identity alive for >= confirm_polls+1 polls spanning >= confirm_s, AND the per-GPU guardian
            lock can be taken (held only by a live guardian). Holding that lock the watchdog is the only actor: it builds
            a GuardianV4 on the same state dir, re-verifies ownership exactly like the guardian (identity + harness ledger
            receipt session/machine/[gpu] + consumer_pid) and runs v1 act(): freeze follower/consumer/arms, arm SIGINT,
            owned serve/exec/collector TERM->KILL, consumer TERM, harness release only after own absence (else 'kept').
            Every signal re-checks pid+start+boot right before os.kill (v1 sig()). Foreign / approved shared processes are
            never owned -> never signalled. Then HOLD.json. Lock busy (a NEW guardian started) -> follow it (orphan path).
  orphan    a live, non-hung guardian holds the lock but has not adopted this consumer for orphan_grace_s (e.g. restarted
            on HOLD or refused): the watchdog writes handoff/fence.<pid>.<start>.json; the guardian then never adopts that
            identity and lists it in heartbeat 'fenced'. Only after a heartbeat NEWER than the fence that lists it and still
            does not adopt it, the watchdog stops the consumer (same procedure, without the lock -- the guardian has
            acknowledged it will not touch it). If the guardian adopted it meanwhile, the fence is withdrawn, no action.
A refused ownership check sends no signal at all (fail closed; ALERT.json). One watchdog per consumer identity
(flock handoff/watchdog.<pid>.<start>.lock). Limits: see RECEIPT.md.

    consumer_watchdog.py --config <guardian cfg.json> --state-dir <guardian state dir> --consumer-pid P --receipt R --nonce N
"""
import argparse, fcntl, json, signal, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import guardian_v4 as G4   # noqa: E402

NVSMI_S, HARNESS_S = 60, 120          # subprocess timeouts in foreign_gpu_guardian.OS (gpu_pids / harness)


def deadlines(c):
    poll = float(c.get('poll_s', 5))
    hang = float(c.get('hang_s', 3 * poll + NVSMI_S + 2 * HARNESS_S + 60))
    act = float(c.get('act_deadline_s', float(c.get('grace_s', 90)) + 2 * float(c.get('kill_wait_s', 30))
                      + float(c.get('release_wait_s', 600)) + 3 * HARNESS_S + NVSMI_S + 60))
    return hang, act


class Watchdog:
    def __init__(self, cfg, osi, state_dir, consumer_pid, receipt, nonce, probe=None):
        self.c, self.o, self.sd, self.probe = cfg, osi, Path(state_dir), probe
        self.hd = self.sd / 'handoff'; self.hd.mkdir(parents=True, exist_ok=True)
        self.boot0 = osi.boot()
        self.cons = {'pid': int(consumer_pid), 'start': osi.start(int(consumer_pid)), 'receipt': receipt}
        if self.cons['start'] is None:
            raise SystemExit('consumer pid not alive')
        self.nonce, self.key = nonce, f'{self.cons["pid"]}.{self.cons["start"]}'
        self._lock = open(self.hd / f'watchdog.{self.key}.lock', 'a')
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lock.close()
            raise SystemExit('another watchdog already guards this consumer (duplicate watchdog refused)')
        self.armed, self.guardians = False, []
        self.lost_n, self.lost_t0, self.hang_n, self.orphan_t0, self.fence_t = 0, None, 0, None, None
        self.log = self.hd / f'watchdog.{self.key}.jsonl'
        self.fence = self.hd / f'fence.{self.key}.json'
        self.hang_s, self.act_s = deadlines(cfg)

    def close(self):
        if not self._lock.closed:
            self._lock.close()

    def event(self, kind, **kw):
        with open(self.log, 'a') as f:
            f.write(json.dumps({'kind': kind, 'time': self.o.now(), **kw}, default=str) + '\n')

    def same(self, ident):
        return bool(ident) and ident.get('start') is not None and self.o.boot() == self.boot0 \
            and self.o.start(ident.get('pid')) == ident['start']

    def mine(self, d):
        return bool(d) and (d.get('pid'), d.get('start')) == (self.cons['pid'], self.cons['start'])

    def _json(self, p):
        try:
            return json.loads(p.read_text())
        except (OSError, ValueError):
            return None

    def learn(self):
        a = self._json(self.hd / f'ack.{self.nonce}.json')
        if a and a.get('accepted') and self.mine(a) and a.get('receipt') == self.cons['receipt'] \
                and a.get('boot_id') == self.boot0 and a.get('guardian') not in self.guardians:
            self.guardians.append(a['guardian']); self.armed = True; self.event('armed', guardian=a['guardian'])
        hb = self._json(self.hd / 'guardian.json')
        if hb and hb.get('boot_id') != self.boot0:
            hb = None
        if hb and self.armed and self.mine(hb.get('adopted')) and hb.get('guardian') not in self.guardians:
            self.guardians.append(hb['guardian']); self.event('guardian_followed', guardian=hb['guardian'])
        return hb

    def withdraw_fence(self):
        if self.fence_t is not None:
            self.fence.unlink(missing_ok=True); self.fence_t = None; self.event('fence_withdrawn')

    def check(self):
        """One poll. Returns None (keep watching) or a final result string."""
        if not self.same(self.cons):
            self.withdraw_fence(); self.event('consumer_ended'); return 'consumer_ended'
        hb = self.learn()
        if not self.armed:
            return None
        now = self.o.now()
        hbg = hb.get('guardian') if hb else None
        if hb and self.same(hbg):                                       # a live heartbeat guardian
            self.lost_n, self.lost_t0 = 0, None
            age, phase = now - float(hb.get('time', 0)), hb.get('phase')
            limit = self.act_s if phase == 'acting' else self.hang_s
            if age > limit:
                self.hang_n += 1
                self.event('guardian_hung', phase=phase, age=age, limit=limit, polls=self.hang_n)
                if self.hang_n >= int(self.c.get('confirm_polls', 2)) + 1:
                    self.stop_guardian(hbg, f'heartbeat {age:.0f}s > {phase} deadline {limit:.0f}s')
                return None
            self.hang_n = 0
            if self.mine(hb.get('adopted')) or phase == 'acting':
                self.orphan_t0 = None; self.withdraw_fence(); return None
            self.orphan_t0 = self.orphan_t0 or now                      # live, healthy, but not guarding us
            if now - self.orphan_t0 < float(self.c.get('orphan_grace_s', 120)):
                return None
            if self.fence_t is None:
                G4.write_json(self.fence, {'consumer': {'pid': self.cons['pid'], 'start': self.cons['start']},
                                           'receipt': self.cons['receipt'], 'boot_id': self.boot0, 'time': now})
                self.fence_t = now; self.event('fence_written'); return None
            acked = [self.cons['pid'], self.cons['start']] in (hb.get('fenced') or [])
            if acked and float(hb.get('time', 0)) > self.fence_t:
                return self.stop('live guardian does not guard this consumer (fence acknowledged)', take_lock=False)
            return None
        live = [g for g in self.guardians if self.same(g)]
        if live:                                                        # alive, but no heartbeat of its own
            self.hang_n += 1; self.lost_n, self.lost_t0 = 0, None
            self.event('guardian_alive_without_heartbeat', polls=self.hang_n)
            if self.hang_n * float(self.c.get('watchdog_poll_s', self.c.get('poll_s', 5))) > self.hang_s:
                self.stop_guardian(live[0], 'no heartbeat from a live guardian beyond hang_s')
            return None
        self.hang_n = 0
        self.lost_n += 1; self.lost_t0 = self.lost_t0 or now
        self.event('guardian_identity_missing', polls=self.lost_n, since=self.lost_t0)
        if self.lost_n >= int(self.c.get('confirm_polls', 2)) + 1 and now - self.lost_t0 >= float(self.c.get('confirm_s', 10)):
            return self.stop('guardian identity lost (pid/start/boot)', take_lock=True)
        return None

    def stop_guardian(self, g, why):
        """Stop OUR hung guardian process (identity + cmdline re-checked before each signal)."""
        for sig in (signal.SIGTERM, signal.SIGKILL):
            if not self.same(g) or 'guardian_v4.py' not in (self.o.cmdline(g['pid']) or ''):
                break
            try:
                self.o.kill(g['pid'], sig)
            except (ProcessLookupError, PermissionError) as e:
                self.event('guardian_signal_failed', guardian=g, sig=int(sig), error=type(e).__name__); return
            self.event('guardian_signalled', guardian=g, sig=int(sig), why=why)
            t_end = self.o.now() + float(self.c.get('kill_wait_s', 30))
            while self.same(g) and self.o.now() < t_end:
                self.o.sleep(1)
        self.hang_n = 0

    def stop(self, why, take_lock):
        try:
            g = G4.GuardianV4(self.c, self.o, self.sd, probe=self.probe, take_lock=take_lock)
        except SystemExit as e:                                  # lock busy: a guardian lives -> do not act
            self.event('stop_deferred_lock_busy', why=why, detail=str(e)); self.lost_n, self.lost_t0 = 0, None; return None
        except Exception as e:   # noqa: BLE001 - cannot build the verified stopper: keep watching, report
            self.event('stop_unavailable', why=why, error=str(e)[:300]); return None
        try:
            if not self.same(self.cons):
                self.event('consumer_ended'); return 'consumer_ended'
            if not take_lock:                                    # last re-check of the fence acknowledgement
                hb = self._json(self.hd / 'guardian.json') or {}
                if self.mine(hb.get('adopted')) or [self.cons['pid'], self.cons['start']] not in (hb.get('fenced') or []):
                    self.event('stop_aborted_guardian_state_changed'); self.orphan_t0 = None; return None
            g.adopted = dict(self.cons); g.consumer = dict(self.cons)
            g.refresh()
            g.event('watchdog_stop', why=why, consumer=self.cons, locked=take_lock, watchdog_log=self.log.name)
            r = g.act([])                                        # v1 procedure; verify_ownership first (fail closed)
            g.set_hold(f'watchdog stopped its consumer ({why}); result {r}', consumer=self.cons)
            self.event('stopped', why=why, result=r)
            return 'watchdog_' + r
        finally:
            g.close()

    def run(self, max_polls=None):
        n = 0
        while True:
            n += 1
            r = self.check()
            if r:
                return r
            if max_polls and n >= max_polls:
                return None
            self.o.sleep(float(self.c.get('watchdog_poll_s', self.c.get('poll_s', 5))))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True); ap.add_argument('--state-dir', type=Path, required=True)
    ap.add_argument('--consumer-pid', type=int, required=True); ap.add_argument('--receipt', required=True)
    ap.add_argument('--nonce', required=True)
    a = ap.parse_args(argv)
    cfg = json.loads(a.config.read_text())
    w = Watchdog(cfg, G4.OS4(cfg.get('harness', ['python3', 'scripts/utils/groot_harness.py'])), a.state_dir,
                 a.consumer_pid, a.receipt, a.nonce)
    print(json.dumps({'result': w.run(), 'time': time.time()}))
    return 0


if __name__ == '__main__':
    sys.exit(main())

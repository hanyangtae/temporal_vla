"""Training runtime guardian v2 = train_guardian (v1, deployed d5b9dc5c, unchanged) with two changes.

1. A failed query is never treated as safe.
   * NVML failure while OUR training is proven running: ALERT + ADMISSION_UNSAFE.json every time; after confirm_polls
     consecutive failures our training is stopped (own identity is proven from the ledger receipt + /proc, so this
     is the conservative direction). The receipt is then KEPT (release requires a successful GPU check), so no other
     session can be admitted on this GPU by the harness, and our own launcher ancestors stay frozen (no relaunch).
   * Harness ledger unreachable: if ownership was proven earlier and the launcher and consumer still have exactly the
     same pid + start ticks + boot, the cached proof is used (recorded as ledger_unreachable) for monitoring and for
     stopping OUR processes; release is impossible without the ledger -> receipt kept + ALERT. Without an earlier
     proof: ALERT + ADMISSION_UNSAFE, no signal.
   * Ownership unproven (ambiguous/mismatching receipt, launcher, consumer): ALERT + ADMISSION_UNSAFE, no signal.
   ADMISSION_UNSAFE.json (state dir) is an explicit hand-off to main: the guardian cannot block other sessions'
   admission except by keeping our receipt, which it does whenever it cannot prove that releasing is safe.
2. Own-PID-only memory measurement on the SHARED GPU (no exclusive GPU needed, no ledger bypass): each successful poll
   records the NVML used_mib of OUR proven processes only (launcher descendants by pid + start ticks); approved
   foreign usage (pinned root, verified container members) and unknown processes are never counted. MEASURE.json
   keeps the measured statistics SEPARATE from the contract's per-job budget:
     measured: n_samples, own_used_peak_mib, own_used_last_mib, interval_s, first/last time, samples (bounded)
     budget:   cap_mib = receipt serves x session per_serve_mib, basis = session 'per_serve_basis' as written in the
               contract ('provisional_cap' for the first smoke, 'measured' only after a measurement) -- never relabelled
   The own-memory bound is enforced exactly as in v1 (stop OUR training when exceeded), so a provisional cap is a hard
   runtime limit, not a claimed peak. Sampling is the guardian poll (default 2 s for a measurement run): transient
   peaks between samples are not seen and are stated as such.

    python train_guardian_v2.py --config <cfg.json> --state-dir <new dir> [--observe-only] [--once] [snapshot]
"""
import argparse, json, os, signal, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_guardian as TG1      # noqa: E402  (unchanged)

SA = TG1.SA
KIND = 'fold27_train_guardian_v2'
MAX_SAMPLES = 5000


class TrainGuardianV2(TG1.TrainGuardian):
    def __init__(self, cfg, osi, state_dir, probe=None, observe_only=False):
        if cfg.get('kind') != KIND:
            raise SystemExit(f'config kind must be {KIND}')
        super().__init__({**cfg, 'kind': TG1.KIND}, osi, state_dir, probe, observe_only)
        self.c = cfg
        self.fails = {'gpu': 0, 'ledger': 0}
        self.meas = {'n_samples': 0, 'own_used_peak_mib': None, 'own_used_last_mib': None, 'first_t': None, 'last_t': None, 'samples': []}

    # ---------------- unsafe marking ----------------
    def unsafe(self, why, **kw):
        self.alert(why, **kw)
        tmp = self.sd / f'ADMISSION_UNSAFE.{os.getpid()}.tmp'
        tmp.write_text(json.dumps({'time': self.o.now(), 'reason': why, 'receipt': (self.owner or {}).get('receipt'), **kw},
                                  indent=1, default=str))
        tmp.replace(self.sd / 'ADMISSION_UNSAFE.json')

    def cached_owner_valid(self):
        o = self.owner
        return bool(o) and self.same(*o['launcher']) and self.same(*o['consumer'])

    def owner_now(self):
        """Fresh proof from the ledger; if the ledger is unreachable, the cached proof iff identities are unchanged."""
        try:
            o = self.prove_owner()
            self.fails['ledger'] = 0
            return o, 'ledger'
        except TG1.Unproven:
            raise
        except Exception as e:   # noqa: BLE001 - ledger unreachable
            self.fails['ledger'] += 1
            if self.cached_owner_valid():
                self.alert('harness ledger unreachable: using cached ownership proof (identities unchanged)', error=str(e)[:300])
                return self.owner, 'cached'
            raise TG1.Unproven(f'ledger unreachable and no valid cached proof: {e}')

    def refresh(self):
        if self.owner is None or not self.cached_owner_valid() or self.fails['ledger']:
            self.owner, _ = self.owner_now()
        self._tree()

    def _tree(self):
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
                role = 'trainer' if self.has(p, TG1.TRAINER_MARKS) else 'queue' if self.has(p, TG1.QUEUE_MARKS) else 'desc'
                self.owned.setdefault((p, st), role)
                stack += kids.get(p, [])
        self.owned = {k: v for k, v in self.owned.items() if self.same(*k)}

    # ---------------- measurement ----------------
    def record(self, own_used, g, cap, basis):
        m = self.meas
        t = self.o.now()
        m['n_samples'] += 1
        m['own_used_peak_mib'] = own_used if m['own_used_peak_mib'] is None else max(m['own_used_peak_mib'], own_used)
        m['own_used_last_mib'] = own_used
        m['first_t'] = m['first_t'] if m['first_t'] is not None else t
        m['last_t'] = t
        if len(m['samples']) < MAX_SAMPLES:
            m['samples'].append([t, own_used, g['free_mib']])
        body = {'measured': {**m, 'interval_s': float(self.c.get('poll_s', 2)), 'source': 'NVML used_mib of proven own PIDs only'},
                'budget': {'cap_mib': cap, 'basis': basis, 'session': self.session},
                'receipt': (self.owner or {}).get('receipt'),
                'caveat': 'samples every interval_s; transient peaks between samples are not observed'}
        tmp = self.sd / f'MEASURE.{os.getpid()}.tmp'
        tmp.write_text(json.dumps(body, indent=1, default=str)); tmp.replace(self.sd / 'MEASURE.json')

    # ---------------- poll ----------------
    def poll(self):
        try:
            self.refresh()
        except TG1.Unproven as e:
            self.unsafe('ownership unproven: no signal', detail=str(e)); return None
        try:
            g = self.probe.gpu(self.gpu)
            self.fails['gpu'] = 0
        except Exception as e:   # noqa: BLE001 - NOT a pass
            self.fails['gpu'] += 1
            self.seen.clear()
            if not self.owner:
                self.unsafe('GPU query failed while idle (state unverifiable)', error=str(e)[:300]); return None
            self.unsafe('GPU query failed while our training runs (state unverifiable)', error=str(e)[:300], failures=self.fails['gpu'])
            return ['gpu unverifiable'] if self.fails['gpu'] >= int(self.c.get('confirm_polls', 2)) else None
        if g['uuid'] != self.auth['gpu_uuid']:
            self.unsafe('GPU UUID differs from contract', uuid=g['uuid'])
            return ['gpu uuid mismatch'] if self.owner else None
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
        if not self.owner:
            if reasons:
                self.event('foreign_while_idle', reasons=reasons)
            return None
        sc = self.auth['sessions'][self.session]
        own_used, cap = sum(a['used_mib'] for a in ours), self.owner['serves'] * int(sc['per_serve_mib'])
        self.record(own_used, g, cap, sc.get('per_serve_basis', 'unspecified'))
        self.breach['own_mem'] = self.breach['own_mem'] + 1 if own_used > cap else 0
        self.breach['headroom'] = self.breach['headroom'] + 1 if g['free_mib'] < int(self.auth['reserve_mib']) else 0
        if self.breach['own_mem']:
            self.event('own_memory_over_bound', used_mib=own_used, cap_mib=cap)
        if self.breach['headroom']:
            self.event('headroom_low', free_mib=g['free_mib'], reserve_mib=self.auth['reserve_mib'])
        reasons += [f'{k} breach' for k, n in self.breach.items() if n >= need]
        return reasons or None

    # ---------------- action ----------------
    def act(self, reasons):
        try:
            owner, src = self.owner_now()
        except TG1.Unproven as e:
            self.unsafe('ownership unproven at action time: no signal', detail=str(e), reasons=reasons); return 'refused'
        if owner is None or (self.owner and owner['receipt'] != self.owner['receipt']):
            self.unsafe('receipt changed/gone at action time: no signal', reasons=reasons); return 'refused'
        self.owner = owner
        self._tree()
        rec = {'reasons': reasons, 'owner': owner, 'proof': src, 'owned': [[p, s, r] for (p, s), r in self.owned.items()],
               'observe_only': self.observe_only}
        with open(self.sd / f'ACTION.{int(self.o.now() * 1000)}.json', 'x') as f:
            f.write(json.dumps(rec, indent=1, default=str))
        if self.observe_only:
            self.event('observe_only_would_act', **rec); return 'observed'
        for p, s in owner['freeze']:
            self.sig(p, s, signal.SIGSTOP, 'freeze launcher ancestor (no relaunch)')
        for p, s in self.alive('queue'):
            self.sig(p, s, signal.SIGSTOP, 'freeze training queue (no new job / no FAILED.json)')
        self._tree()
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
        r = self.release(owner)
        if r == 'kept':
            self.unsafe('own training stopped but receipt kept: release not provably safe', reasons=reasons, receipt=owner['receipt'])
        return r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True); ap.add_argument('--state-dir', type=Path, required=True)
    ap.add_argument('--observe-only', action='store_true'); ap.add_argument('--once', action='store_true')
    ap.add_argument('mode', nargs='?', default='watch', choices=('watch', 'snapshot'))
    a = ap.parse_args(argv)
    cfg = json.loads(a.config.read_text())
    g = TrainGuardianV2(cfg, TG1.HostOS(cfg.get('harness', ['python3', 'scripts/utils/groot_harness.py'])), a.state_dir,
                        observe_only=a.observe_only)
    if a.mode == 'snapshot':
        try:
            g.refresh()
        except TG1.Unproven as e:
            print(json.dumps({'unproven': str(e)})); return 6
        print(json.dumps({'owner': g.owner, 'owned': [[p, s, r] for (p, s), r in g.owned.items()]}, default=str))
        return 0
    r = g.run(once=a.once)
    print(json.dumps({'result': r}))
    return {'no_action': 0, 'observed': 0, 'released': 0, 'released_elsewhere': 0, 'kept': 5, 'refused': 6}.get(r, 1)


if __name__ == '__main__':
    sys.exit(main())

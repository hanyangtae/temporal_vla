"""Training runtime guardian v3 = train_guardian_v2 (deployed b6957d56, unchanged) + support for a v1 sharing contract.

Why: v1/v2 refuse at init unless the contract is kind v2 with a sessions map. The srv50 GPU3 training contract that can
be prepared officially now is a kind-v1 contract (one pinned foreign process, one session): the v2 contract pinned a
container instance that no longer exists, and the current foreign job is a single pinned process.
Changes, nothing else:
  * init accepts kind v1 (session == contract session, allowed_kind eval, machine/gpu equal) or kind v2 exactly as v2.
  * poll = v2 poll with two lookups abstracted:
      scope     v1: the contract's own per_serve_mib / max_serves / per_serve_basis;  v2: sessions[session] (as v2)
      approved  v1: pinned identities only (pid + uid + start ticks + cmdline sha, max_used_mib), expiry/boot -> none;
                v2: gpu_sharing_authorization.approved_now (as v2)
    Ownership proof, own-PID-only measurement (MEASURE.json), own cap = receipt serves x per_serve_mib, reserve check,
    NVML/ledger failure handling (ADMISSION_UNSAFE, never a pass), action (freeze ancestors, stop only proven owned
    descendants, release only if provably ours) are the unchanged v1/v2 code.
Config kind 'fold27_train_guardian_v3'.

    python train_guardian_v3.py --config <cfg.json> --state-dir <new dir> [--observe-only] [--once] [snapshot]
"""
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_guardian_v2 as TG2      # noqa: E402  (unchanged)

TG1, SA = TG2.TG1, TG2.SA
KIND = 'fold27_train_guardian_v3'


def approved_v1(auth, apps, probe):
    """(approved pid set, violations) for a v1 contract: pinned identities only; expired/other boot -> nothing approved."""
    if probe.now() >= float(auth['expires_at']) or probe.boot_id() != auth['boot_id']:
        return set(), ['contract expired or boot changed']
    cl = SA.classify({'foreign': auth['foreign'], 'containers': []}, apps, probe)   # pinned branch of the shared checker
    return set(cl['pinned']), cl['violations']


class TrainGuardianV3(TG2.TrainGuardianV2):
    def __init__(self, cfg, osi, state_dir, probe=None, observe_only=False):
        if cfg.get('kind') != KIND:
            raise SystemExit(f'config kind must be {KIND}')
        auth = SA.load(cfg['sharing_authorization'])
        if auth.get('kind') == SA.KIND_V2:
            super().__init__({**cfg, 'kind': TG2.KIND}, osi, state_dir, probe, observe_only)
            self.c = cfg
            return
        if auth.get('kind') != SA.KIND or auth['machine'] != cfg['host'] or int(auth['gpu']) != int(cfg['gpu']) \
                or auth['session'] != cfg['session'] or auth['allowed_kind'] != 'eval' or int(auth['max_serves']) < 1:
            raise SystemExit('training guardian v3 needs a v1/v2 sharing contract covering exactly this host/gpu/session')
        # same attributes as TrainGuardian.__init__ + TrainGuardianV2.__init__ (their contract-kind check is the only
        # difference; both files are unchanged)
        self.c, self.o, self.sd = cfg, osi, Path(state_dir)
        self.sd.mkdir(parents=True, exist_ok=True)
        self.gpu, self.session, self.boot0 = int(cfg['gpu']), cfg['session'], osi.boot()
        self.auth = auth
        self.probe = probe or SA.Probe()
        self.observe_only = observe_only or bool(cfg.get('observe_only'))
        self.owner, self.owned, self.seen, self.breach = None, {}, {}, {'own_mem': 0, 'headroom': 0}
        self.fails = {'gpu': 0, 'ledger': 0}
        self.meas = {'n_samples': 0, 'own_used_peak_mib': None, 'own_used_last_mib': None, 'first_t': None, 'last_t': None, 'samples': []}

    def v1(self):
        return self.auth.get('kind') == SA.KIND

    def scope(self):
        if self.v1():
            return {'per_serve_mib': self.auth['per_serve_mib'], 'max_serves': self.auth['max_serves'],
                    'per_serve_basis': self.auth.get('per_serve_basis', 'unspecified')}
        return self.auth['sessions'][self.session]

    def approved(self, apps):
        return approved_v1(self.auth, apps, self.probe) if self.v1() else SA.approved_now(self.auth, apps, self.probe)

    def poll(self):                                   # TrainGuardianV2.poll with scope()/approved() lookups
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
        # v3: any exception while proving identities / contract validity (approvals, /proc, boot, host, clock) must not
        # kill the guardian and leave our training unprotected: UNSAFE, and after confirm_polls stop OUR training
        try:
            r = self._evaluate(g)
            self.fails['identity'] = 0
            return r
        except Exception as e:   # noqa: BLE001 - NOT a pass
            self.fails['identity'] = self.fails.get('identity', 0) + 1
            self.seen.clear()
            self.unsafe('identity/contract verification raised (state unverifiable)', error=f'{type(e).__name__}: {e}'[:300],
                        failures=self.fails['identity'])
            if self.owner and self.fails['identity'] >= int(self.c.get('confirm_polls', 2)):
                return ['identity unverifiable']
            return None

    def _evaluate(self, g):
        ours = [a for a in g['apps'] if self.owned_pid(a['pid'])]
        others = [a for a in g['apps'] if not self.owned_pid(a['pid'])]
        ok, viol = self.approved(others)
        unknown = {}
        for a in others:
            if a['pid'] in ok:
                continue
            # v3: an NVML-listed PID whose /proc identity is unreadable (start None) is UNVERIFIABLE and counts as foreign
            # (v1/v2 skipped it); a PID that really exited disappears from NVML and its counter resets next poll
            key = (a['pid'], self.o.start(a['pid']))
            unknown[key] = self.seen.get(key, 0) + 1
        self.seen = unknown
        need = int(self.c.get('confirm_polls', 2))
        reasons = [f'foreign {k}' for k, n in unknown.items() if n >= need]
        if unknown or viol:
            self.event('unrecognized_or_violation', pids=[list(k) for k in unknown], violations=viol[:10])
        # v3: contract validity is checked directly every poll, also when no foreign process is present (v1/v2 only saw
        # it indirectly through approvals): expired / other boot / other host -> UNSAFE, confirmed -> stop OUR training
        valid = (self.probe.now() < float(self.auth['expires_at']) and self.probe.boot_id() == self.auth['boot_id']
                 and self.probe.hostname() == self.auth['hostname'])
        self.breach['contract'] = 0 if valid else self.breach.get('contract', 0) + 1
        if not valid:
            self.unsafe('sharing contract no longer valid (expired / boot / host)', expires_at=self.auth['expires_at'])
        if not self.owner:
            if reasons:
                self.event('foreign_while_idle', reasons=reasons)
            return None
        sc = self.scope()
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True); ap.add_argument('--state-dir', type=Path, required=True)
    ap.add_argument('--observe-only', action='store_true'); ap.add_argument('--once', action='store_true')
    ap.add_argument('mode', nargs='?', default='watch', choices=('watch', 'snapshot'))
    a = ap.parse_args(argv)
    cfg = json.loads(a.config.read_text())
    g = TrainGuardianV3(cfg, TG1.HostOS(cfg.get('harness', ['python3', 'scripts/utils/groot_harness.py'])), a.state_dir,
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

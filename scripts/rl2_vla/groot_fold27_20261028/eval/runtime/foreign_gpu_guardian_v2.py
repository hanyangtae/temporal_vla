"""foreign_gpu_guardian v2 = v1 (unchanged, 3f099499) + the SAME one-time sharing contract the harness honours
(scripts/utils/gpu_sharing_authorization.py). Only difference: in poll(), a GPU process that is NOT ours is tolerated iff
it is one of the contract's pinned approved processes with identical pid + uid + start_ticks + cmdline + boot id, the
contract is unexpired, the GPU UUID equals the contract's, and its GPU memory is within max_used_mib. It is never added to
the owned set and therefore is never signalled (v1 only ever signals owned identities). Everything else is unchanged:
any other unrecognised process (incl. a reused PID or a replacement task) is foreign -> v1 stop/release procedure.
Additionally, while an approved process shares the GPU and our consumer runs, free GPU memory below the contract's
reserve_mib for confirm_polls polls triggers the same v1 procedure (stop only OUR verified jobs; protects the approved
task from an OOM caused by us). A failed NVML query still means: observe nothing, act on nothing.
Config: v1 config + 'sharing_authorization': <contract path>; contract machine/gpu/session must equal the config's.
Without that key v2 behaves exactly like v1.

    python foreign_gpu_guardian_v2.py --config <cfg.json> --state-dir <dir> [--observe-only] [--once] [snapshot]
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[4] / 'scripts/utils')]
import foreign_gpu_guardian as G1          # noqa: E402  (unchanged)
import gpu_sharing_authorization as SA     # noqa: E402


class GuardianV2(G1.Guardian):
    def __init__(self, cfg, osi, state_dir, observe_only=False, probe=None):
        super().__init__(cfg, osi, state_dir, observe_only)
        self.auth, self.probe, self.low_mem = None, probe or SA.Probe(), 0
        p = cfg.get('sharing_authorization')
        if p:
            a = SA.load(p)
            if a['machine'] != cfg['host'] or int(a['gpu']) != self.gpu or a['session'] != cfg['session']:
                raise SystemExit('sharing authorization machine/gpu/session differ from the guardian config')
            self.auth = a

    def tolerated(self, unknown):
        """Subset of unknown (pid, start) that are the exactly pinned approved processes right now."""
        if not self.auth:
            return set(), None
        g = self.probe.gpu(self.gpu)                 # raises -> caller treats as query failure
        if g['uuid'] != self.auth['gpu_uuid']:
            self.event('sharing_uuid_mismatch', uuid=g['uuid']); return set(), g
        used = {a['pid']: a['used_mib'] for a in g['apps']}
        lim = {int(f['pid']): int(f['max_used_mib']) for f in self.auth['foreign']}
        ok = set()
        for p, s in unknown:
            if SA.matches_identity(self.auth, p, self.probe) and used.get(p, 0) <= lim.get(p, -1):
                ok.add((p, s))
            elif p in lim:
                self.event('sharing_identity_or_memory_mismatch', pid=p, start=s, used_mib=used.get(p))
        return ok, g

    def poll(self):
        try:
            gp = self.o.gpu_pids(self.gpu)
        except Exception as e:   # noqa: BLE001
            self.event('gpu_query_failed', error=str(e)[:300]); self.seen.clear(); return None
        self.refresh()
        cand = {}
        for p in gp:
            if self.owned_pid(p):
                continue
            st = self.o.start(p)
            if st is not None:
                cand[(p, st)] = True
        try:
            ok, g = self.tolerated(cand)
        except Exception as e:   # noqa: BLE001 - cannot verify the contract now: treat like any query failure
            self.event('gpu_query_failed', error=f'sharing probe: {e}'[:300]); self.seen.clear(); return None
        if ok:
            self.event('sharing_tolerated', pids=[list(k) for k in sorted(ok)])
        unknown = {k: self.seen.get(k, 0) + 1 for k in cand if k not in ok}
        self.seen = unknown
        need = int(self.c.get('confirm_polls', 2))
        confirmed = [k for k, n in unknown.items() if n >= need]
        if unknown:
            self.event('unrecognized_gpu_process', gpu=self.gpu, pids=[[p, s, n, self.o.uid(p), (self.o.cmdline(p) or '')[:160]]
                                                                      for (p, s), n in unknown.items()])
        if ok and g is not None and g['free_mib'] < int(self.auth['reserve_mib']):
            self.low_mem += 1
            self.event('sharing_headroom_low', free_mib=g['free_mib'], reserve_mib=self.auth['reserve_mib'], polls=self.low_mem)
            if self.low_mem >= need and not confirmed:
                confirmed = sorted(ok)               # recorded as the reason; act() signals only OUR owned identities
        else:
            self.low_mem = 0
        if confirmed and self.read_consumer() is None:
            self.event('foreign_while_idle', pids=[list(k) for k in confirmed]); return None
        return confirmed or None


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True); ap.add_argument('--state-dir', type=Path, required=True)
    ap.add_argument('--observe-only', action='store_true'); ap.add_argument('--once', action='store_true')
    ap.add_argument('mode', nargs='?', default='watch', choices=('watch', 'snapshot'))
    a = ap.parse_args(argv)
    cfg = json.loads(a.config.read_text())
    g = GuardianV2(cfg, G1.OS(cfg.get('harness', ['python3', 'scripts/utils/groot_harness.py'])), a.state_dir, a.observe_only)
    if a.mode == 'snapshot':
        g.refresh()
        print(json.dumps({'consumer': g.consumer, 'owned': [[p, s, r] for (p, s), r in g.owned.items()],
                          'sharing_authorization': cfg.get('sharing_authorization')}, default=str))
        return 0
    r = g.run(once=a.once)
    print(json.dumps({'result': r}))
    return {'no_action': 0, 'observed': 0, 'released': 0, 'released_elsewhere': 0, 'kept': 5, 'refused': 6}.get(r, 1)


if __name__ == '__main__':
    sys.exit(main())

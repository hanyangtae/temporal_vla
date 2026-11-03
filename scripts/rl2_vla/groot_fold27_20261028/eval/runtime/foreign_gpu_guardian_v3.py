"""foreign_gpu_guardian v3 = v2 (unchanged, c14d8a15) able to read a v2 sharing contract (kind
groot_gpu_sharing_exception_v2): pinned processes as before PLUS processes that are verified members of a pinned foreign
container INSTANCE (gpu_sharing_authorization.container_member: exact cgroup line, pinned init pid+start ticks with
NSpid==1, exact real uid/gid, nested NSpid, unbroken same-cgroup PPid lineage with consistent start times; aggregate
container VRAM <= its limit). Tolerated processes are never added to the owned set, so they are never signalled.
Everything not proven that way -- a new container, a restarted one, a host process, a same-UID or same-command process
outside the pinned cgroup, an over-limit container -- is foreign and v1's stop-our-jobs procedure applies.
Config: v1 config + 'sharing_authorization' (v1 or v2 contract). The guardian config's session must be one of the
contract's sessions (v2) or equal it (v1). With a v1 contract v3 behaves exactly like v2.
Scope stays the eval consumer path (v1): fold27 training is protected by the harness contract check at admission /
post-admission / release only (see OPUS_RECEIPT_SRV50_GPU3_CONTAINER_TRAIN_20261104.md).
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[4] / 'scripts/utils')]
import foreign_gpu_guardian as G1          # noqa: E402  (unchanged)
import foreign_gpu_guardian_v2 as G2       # noqa: E402  (unchanged)
import gpu_sharing_authorization as SA     # noqa: E402


class GuardianV3(G2.GuardianV2):
    def __init__(self, cfg, osi, state_dir, observe_only=False, probe=None):
        p = cfg.get('sharing_authorization')
        a = SA.load(p) if p else None
        if a is None or a['kind'] == SA.KIND:
            super().__init__(cfg, osi, state_dir, observe_only, probe)   # v1 contract / none: exactly v2
            return
        G1.Guardian.__init__(self, cfg, osi, state_dir, observe_only)
        if a['machine'] != cfg['host'] or int(a['gpu']) != self.gpu or cfg['session'] not in a['sessions']:
            raise SystemExit('sharing authorization machine/gpu/sessions do not cover the guardian config')
        self.auth, self.probe, self.low_mem = a, probe or SA.Probe(), 0

    def tolerated(self, unknown):
        if not self.auth or self.auth['kind'] == SA.KIND:
            return super().tolerated(unknown)
        g = self.probe.gpu(self.gpu)
        if g['uuid'] != self.auth['gpu_uuid']:
            self.event('sharing_uuid_mismatch', uuid=g['uuid']); return set(), g
        ok_pids, violations = SA.approved_now(self.auth, g['apps'], self.probe)
        if violations:
            self.event('sharing_violations', violations=violations[:20])
        return {(p, s) for p, s in unknown if p in ok_pids}, g


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True); ap.add_argument('--state-dir', type=Path, required=True)
    ap.add_argument('--observe-only', action='store_true'); ap.add_argument('--once', action='store_true')
    ap.add_argument('mode', nargs='?', default='watch', choices=('watch', 'snapshot'))
    a = ap.parse_args(argv)
    cfg = json.loads(a.config.read_text())
    g = GuardianV3(cfg, G1.OS(cfg.get('harness', ['python3', 'scripts/utils/groot_harness.py'])), a.state_dir, a.observe_only)
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

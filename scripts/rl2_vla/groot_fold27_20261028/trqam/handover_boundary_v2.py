"""handover_boundary for whichever coordinator (v2 / v3 / v4) currently holds queue/host_coord_v2.lock.

Same sentinel as handover_boundary.py (that file is frozen and only checks the v2 state file). The live coordinator is
the one whose state file is newest; arming requires it to be mid-training and the single-writer lock to be held.

    python handover_boundary_v2.py status --out outputs/rl2_vla/fold27_trqam_v2_20261102
    python handover_boundary_v2.py arm    --out outputs/rl2_vla/fold27_trqam_v2_20261102 --reason "..."
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import handover_boundary as HB   # noqa: E402  (frozen; lock_holder / SENTINEL_KEY reused)

VERSIONS = ('V2', 'V3', 'V4')


def status(out, machine='kanu'):
    q = Path(out) / 'queue'
    states = {}
    for v in VERSIONS:
        p = q / f'host_{machine}_COORD_{v}_STATE.json'
        if p.exists():
            states[v] = json.loads(p.read_text())
    live = max(states, key=lambda v: float(states[v].get('time', 0)), default=None)
    failed = q / 'FAILED.json'
    fj = json.loads(failed.read_text()) if failed.exists() else None
    return {'live_version': live, 'coordinator_state': states.get(live), 'lock_holder_pid': HB.lock_holder(q),
            'failed_json': None if fj is None else ('handover_sentinel' if fj.get(HB.SENTINEL_KEY) else 'REAL_FAILURE')}


def arm(out, reason, machine='kanu', now=time.time):
    s = status(out, machine)
    if s['failed_json'] is not None:
        raise SystemExit(f'FAILED.json already present ({s["failed_json"]}); not arming')
    cs = s['coordinator_state'] or {}
    if s['lock_holder_pid'] is None or cs.get('status') not in ('training', 'yielding_to_eval', 'prep') \
            or not str(cs.get('phase', '')).startswith('train:'):
        raise SystemExit(f'{s["live_version"]} not mid-fold (lock={s["lock_holder_pid"]}, status={cs.get("status")}, '
                         f'phase={cs.get("phase")}); nothing to hand over')
    body = {HB.SENTINEL_KEY: True, 'reason': reason, 'time': now(), 'expected_stop_after': cs.get('phase'),
            'stopping_version': s['live_version'],
            'note': 'boundary stop for the lock-holding coordinator; consumed (renamed) by host_fold_coordinator_v4'}
    fd = os.open(Path(out) / 'queue' / 'FAILED.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, 'w') as f:
        f.write(json.dumps(body, indent=1))
    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('status', 'arm')); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--machine', default='kanu'); ap.add_argument('--reason', default='fold-boundary handover')
    a = ap.parse_args()
    print(json.dumps(status(a.out, a.machine) if a.cmd == 'status' else arm(a.out, a.reason, a.machine), indent=1))


if __name__ == '__main__':
    main()

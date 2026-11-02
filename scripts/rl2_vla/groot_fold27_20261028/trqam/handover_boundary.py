"""Fold-boundary handover for the LIVE v2 coordinator, without signals or source edits.

Mechanism (host_fold_coordinator_v2 as running, unchanged): before launching each next fold it calls check_failures(), which
stops the coordinator if <out>/queue/FAILED.json exists. The running train_queue_v2 never re-reads that file during a run.
So writing a marked sentinel there lets the CURRENT fold finish normally (3 arms DONE, GPUs released by the harness when its
queue exits) and makes the coordinator exit cleanly at the boundary BEFORE it re-claims GPUs. host_fold_coordinator_v3 then
takes the single-writer lock, moves the sentinel aside (HANDOVER_STOP.<ts>.json, never deleted) and continues with an
eval-yield gate.

    python handover_boundary.py status --out outputs/rl2_vla/fold27_trqam_v2_20261102 [--machine kanu]
    python handover_boundary.py arm    --out outputs/rl2_vla/fold27_trqam_v2_20261102 --reason "eval READY priority"
`arm` refuses if a real FAILED.json already exists or the coordinator is not mid-training (nothing to hand over).
"""
import argparse, fcntl, json, os, time
from pathlib import Path

SENTINEL_KEY = 'handover_sentinel'


def lock_holder(qdir):
    p = qdir / 'host_coord_v2.lock'
    if not p.exists():
        return None
    with open(p, 'a+') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(f, fcntl.LOCK_UN)
            return None                                    # nobody holds it
        except OSError:
            f.seek(0)
            return (f.read().strip() or '?')


def status(out, machine='kanu'):
    q = Path(out) / 'queue'
    st = q / f'host_{machine}_COORD_V2_STATE.json'
    failed = q / 'FAILED.json'
    fj = json.loads(failed.read_text()) if failed.exists() else None
    return {'coordinator_state': json.loads(st.read_text()) if st.exists() else None,
            'lock_holder_pid': lock_holder(q),
            'failed_json': None if fj is None else ('handover_sentinel' if fj.get(SENTINEL_KEY) else 'REAL_FAILURE'),
            'queue_state': json.loads((q / 'QUEUE_STATE.json').read_text()) if (q / 'QUEUE_STATE.json').exists() else None}


def arm(out, reason, machine='kanu', now=time.time):
    s = status(out, machine)
    if s['failed_json'] is not None:
        raise SystemExit(f'FAILED.json already present ({s["failed_json"]}); not arming')
    cs = s['coordinator_state'] or {}
    if s['lock_holder_pid'] is None or cs.get('status') != 'training':
        raise SystemExit(f'coordinator not mid-training (lock={s["lock_holder_pid"]}, status={cs.get("status")}); nothing to hand over')
    body = {SENTINEL_KEY: True, 'reason': reason, 'time': now(), 'expected_stop_after': cs.get('phase'),
            'note': 'boundary stop for host_fold_coordinator_v2; consumed (renamed) by host_fold_coordinator_v3'}
    p = Path(out) / 'queue' / 'FAILED.json'
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)    # never overwrite a real failure
    with os.fdopen(fd, 'w') as f:
        f.write(json.dumps(body, indent=1))
    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('status', 'arm')); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--machine', default='kanu'); ap.add_argument('--reason', default='eval READY priority at fold boundary')
    a = ap.parse_args()
    print(json.dumps(status(a.out, a.machine) if a.cmd == 'status' else arm(a.out, a.reason, a.machine), indent=1))


if __name__ == '__main__':
    main()

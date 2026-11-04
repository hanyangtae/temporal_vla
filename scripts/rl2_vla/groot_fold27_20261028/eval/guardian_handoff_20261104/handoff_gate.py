"""Launcher side of the guardian_v4 handoff (used only by eval_gpu_queue_v4.sh). Fail closed: any doubt -> non-zero.

    handoff_gate.py ready   --state-dir D --host H --gpu G --session S              # rc 0 ready / 3 not ready
    handoff_gate.py request --state-dir D --host H --gpu G --session S --receipt R --pid P --nonce N --watchdog-pid W
                            [--timeout 120]
                                                                                       # rc 0 adopted / 75 refused|timeout
ready: <D>/handoff/guardian.json is fresh (age <= 3*poll_s, at most --max-age), its guardian pid+start is alive on this
boot, host/gpu/session equal, and it is not on HOLD.
request (run by `eval_gpu_queue_v4.sh --inner` after the harness verify, BEFORE exec of the consumer, so P is the
consumer-to-be): writes handoff/request.<nonce>.json, waits for ack/nack.<nonce>.json. Accepted only if the ack echoes
nonce/receipt/pid/start/boot exactly and names the guardian of a still-ready heartbeat. On refusal or timeout the request
file is removed (a late guardian cannot adopt a launcher that gave up) and rc 75 -> no consumer, no GPU work.
"""
import argparse, json, os, sys, time, uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'runtime'))
import job_claims as JC   # noqa: E402


def check_ready(sd, host, gpu, session, now, start_fn=JC.proc_start, boot_fn=JC.boot_id, max_age=60.0):
    """None if the guardian is ready, else the reason."""
    p = Path(sd) / 'handoff' / 'guardian.json'
    try:
        hb = json.loads(p.read_text())
    except (OSError, ValueError):
        return 'no guardian heartbeat'
    if (hb.get('host'), hb.get('gpu'), hb.get('session')) != (host, gpu, session):
        return 'heartbeat host/gpu/session differ'
    if hb.get('boot_id') != boot_fn():
        return 'heartbeat from another boot'
    age = now - float(hb.get('time', 0))
    if not 0 <= age <= min(max_age, 3 * float(hb.get('poll_s', 5)) + 5):
        return f'heartbeat stale ({age:.0f}s)'
    g = hb.get('guardian') or {}
    if g.get('start') is None or start_fn(g.get('pid')) != g.get('start'):
        return 'guardian process not alive (crash?)'
    if hb.get('hold'):
        return f'guardian on HOLD: {hb["hold"]}'
    return None


def request(sd, host, gpu, session, receipt, pid, timeout, now_fn=time.time, sleep_fn=time.sleep,
            start_fn=JC.proc_start, boot_fn=JC.boot_id, nonce=None, watchdog_pid=None):
    """(True, ack) or (False, reason)."""
    why = check_ready(sd, host, gpu, session, now_fn(), start_fn, boot_fn)
    if why:
        return False, why
    hd = Path(sd) / 'handoff'
    nonce = nonce or uuid.uuid4().hex
    req = {'nonce': nonce, 'host': host, 'gpu': gpu, 'session': session, 'receipt': receipt, 'pid': pid,
           'start': start_fn(pid), 'boot_id': boot_fn(), 'time': now_fn(),
           'watchdog': {'pid': watchdog_pid, 'start': start_fn(watchdog_pid)} if watchdog_pid else None}
    if req['start'] is None:
        return False, 'own pid not found'
    rp = hd / f'request.{nonce}.json'
    tmp = hd / f'.request.{nonce}.tmp'; tmp.write_text(json.dumps(req)); os.replace(tmp, rp)
    t_end = now_fn() + timeout
    try:
        while now_fn() < t_end:
            for kind in ('ack', 'nack'):
                ap = hd / f'{kind}.{nonce}.json'
                if not ap.exists():
                    continue
                a = json.loads(ap.read_text())
                if kind == 'nack' or not a.get('accepted'):
                    return False, 'refused: ' + str(a.get('reason'))
                if any(a.get(k) != req[k] for k in ('nonce', 'receipt', 'pid', 'start', 'boot_id')):
                    return False, 'ack does not echo this request'
                hb = json.loads((hd / 'guardian.json').read_text())
                if a.get('guardian') != hb.get('guardian'):
                    return False, 'ack from a guardian other than the heartbeat guardian'
                why = check_ready(sd, host, gpu, session, now_fn(), start_fn, boot_fn)
                return (True, a) if why is None else (False, 'after ack: ' + why)
            sleep_fn(0.5)
        return False, f'no guardian answer within {timeout}s'
    finally:                                        # always withdrawn: a late guardian cannot adopt a launcher
        rp.unlink(missing_ok=True)                  # that gave up; the ack/nack file stays as the record


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('ready', 'request'))
    ap.add_argument('--state-dir', required=True); ap.add_argument('--host', required=True)
    ap.add_argument('--gpu', type=int, required=True); ap.add_argument('--session', required=True)
    ap.add_argument('--receipt'); ap.add_argument('--pid', type=int); ap.add_argument('--timeout', type=float, default=120)
    ap.add_argument('--nonce'); ap.add_argument('--watchdog-pid', type=int)
    a = ap.parse_args(argv)
    if a.cmd == 'ready':
        why = check_ready(a.state_dir, a.host, a.gpu, a.session, time.time())
        print(json.dumps({'ready': why is None, 'why': why})); return 0 if why is None else 3
    if not a.receipt or not a.pid:
        print('--receipt and --pid required', file=sys.stderr); return 75
    ok, info = request(a.state_dir, a.host, a.gpu, a.session, a.receipt, a.pid, a.timeout, nonce=a.nonce,
                       watchdog_pid=a.watchdog_pid)
    print(json.dumps({'adopted': ok, 'detail': info}, default=str), file=sys.stderr)
    return 0 if ok else 75


if __name__ == '__main__':
    sys.exit(main())

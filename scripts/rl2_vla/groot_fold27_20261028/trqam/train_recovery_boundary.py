"""READ-ONLY boundary check for replacing an idle blocked training coordinator/guardian on one shared GPU (new file).
Never signals, stops, launches or writes anything except stdout. Main performs every action after a READY result.

READY only if ALL hold (each failure is listed; nothing is assumed):
  boot         live boot id == --boot-id
  eval         the eval consumer identity (--eval-consumer PID:START) is gone (its receipt/serves are checked below)
  ledger       `groot_harness.py status` has NO receipt for --machine on --gpu (any session) and the query succeeds
  gpu          every NVML compute process on the GPU is a pinned identity of --contract (pid + uid + start ticks + cmdline sha,
               used <= max_used_mib); query must succeed -> no eval serve / our trainer / unknown process left
  coordinator  old coordinator (--old-coordinator PID:START, --old-out): if still alive, its v5 state shows no running arm,
               and <old-out>/queue/train_claims has no live claim (job_claims liveness: same boot, pid+start alive)
  duplicates   no live process whose cmdline runs host_fold_coordinator_v5_shared_g3.py or train_guardian_v3.py (any
               session; a replacement must not already be running when this boundary is evaluated)
Snapshot (for the receipt): boot, ledger size, GPU apps with identities, old coordinator/guardian alive flags + cmdlines.

    python3 train_recovery_boundary.py --machine srv50 --gpu 3 --boot-id B --session S --contract <eval v1 json> \
        --eval-consumer 3088084:<start> --old-coordinator 631152:<start> --old-guardian 627282:<start> --old-out <train out> \
        [--wait --poll-s 300]          # exit 0 READY, 3 not ready (once)
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path[:0] = [str(HERE.parent / 'eval/runtime'), str(REPO / 'scripts/utils')]
import job_claims as JC                     # noqa: E402  (read-only helpers)
import gpu_sharing_authorization as SA      # noqa: E402

DUP_MARKS = ('host_fold_coordinator_v5_shared_g3.py', 'train_guardian_v3.py')


class HostView:
    """Read-only host access (injectable for tests)."""
    def __init__(self, harness=('python3', str(REPO / 'scripts/utils/groot_harness.py'))):
        self.harness_cmd, self.probe = list(harness), SA.Probe()

    def boot(self): return JC.boot_id()
    def start(self, pid): return JC.proc_start(pid)

    def cmdline(self, pid):
        try:
            return Path(f'/proc/{int(pid)}/cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace').strip()
        except (OSError, ValueError):
            return None

    def pids(self):
        return [int(p.name) for p in Path('/proc').iterdir() if p.name.isdigit()]

    def ledger(self):
        r = subprocess.run([*self.harness_cmd, 'status'], capture_output=True, text=True, timeout=120, cwd=str(REPO))
        if r.returncode != 0:
            raise RuntimeError(f'harness status rc={r.returncode}: {r.stderr.strip()[-200:]}')
        return json.loads(r.stdout)

    def read_json(self, p):
        try:
            return json.loads(Path(p).read_text())
        except (OSError, ValueError):
            return None


def ident(s):
    pid, start = s.split(':')
    return int(pid), int(start)


def check(a, h):
    reasons, snap = [], {}
    boot = h.boot(); snap['boot'] = boot
    if boot != a.boot_id:
        reasons.append(f'boot {boot} != expected {a.boot_id}')
    ep, es = ident(a.eval_consumer)
    if h.start(ep) == es:
        reasons.append(f'eval consumer {ep} still alive')
    try:
        led = h.ledger()
        mine = {r: it for r, it in led.items() if it.get('machine') == a.machine and a.gpu in [int(g) for g in it.get('gpus', [])]}
        snap['ledger_receipts_on_gpu'] = {r[:8]: [it.get('session'), it.get('kind')] for r, it in mine.items()}
        if mine:
            reasons.append(f'{len(mine)} harness receipt(s) still on GPU{a.gpu}')
    except Exception as e:   # noqa: BLE001 - unknown is not ready
        reasons.append(f'ledger unreadable: {e}')
    try:
        auth = SA.load(a.contract)
        g = h.probe.gpu(a.gpu)
        cl = SA.classify({'foreign': auth['foreign'], 'containers': []}, g['apps'], h.probe)
        snap['gpu'] = {'uuid': g['uuid'], 'free_mib': g['free_mib'], 'apps': g['apps'], 'pinned_ok': cl['pinned']}
        if g['uuid'] != auth['gpu_uuid']:
            reasons.append('GPU UUID differs from the contract')
        if cl['violations']:
            reasons.append('GPU holds processes other than the verified pinned identity: ' + ' | '.join(cl['violations'][:3]))
    except Exception as e:   # noqa: BLE001
        reasons.append(f'GPU/contract unverifiable: {e}')
    cp, cs = ident(a.old_coordinator)
    alive = h.start(cp) == cs
    snap['old_coordinator'] = {'pid': cp, 'start': cs, 'alive': alive, 'cmdline': h.cmdline(cp) if alive else None}
    if alive:
        st = h.read_json(Path(a.old_out) / 'queue' / f'host_{a.machine}_COORD_V5_STATE.json')
        if st is None:
            reasons.append('old coordinator alive but its v5 state is unreadable')
        elif st.get('running'):
            reasons.append(f'old coordinator has running arms {st.get("running")}')
    claims = Path(a.old_out) / 'queue' / 'train_claims'
    live = []
    for c in sorted(claims.glob('*.claim')) if claims.exists() else []:
        b = h.read_json(c)
        if b is None or (b.get('boot_id') == boot and (JC.alive(b.get('consumer'), h.start) or JC.alive(b.get('child'), h.start))):
            live.append(c.name)
    if live:
        reasons.append(f'live/unreadable train claims {live}')
    if a.old_guardian:
        gp, gs = ident(a.old_guardian)
        snap['old_guardian'] = {'pid': gp, 'start': gs, 'alive': h.start(gp) == gs}
    # any live v3 guardian / g3 coordinator (any session) means a replacement step already happened: not a fresh boundary
    dup = [p for p in h.pids() if any(m in (h.cmdline(p) or '') for m in DUP_MARKS)]
    if dup:
        reasons.append(f'replacement already running: pids {dup}')
    return {'ready': not reasons, 'reasons': reasons, 'snapshot': snap, 'time': time.time()}


def main(argv=None, host=None, sleep=time.sleep):
    ap = argparse.ArgumentParser()
    for k in ('machine', 'boot-id', 'session', 'contract', 'eval-consumer', 'old-coordinator', 'old-out'):
        ap.add_argument('--' + k, required=True)
    ap.add_argument('--gpu', type=int, required=True); ap.add_argument('--old-guardian')
    ap.add_argument('--wait', action='store_true'); ap.add_argument('--poll-s', type=float, default=300)
    a = ap.parse_args(argv)
    h = host or HostView()
    while True:
        r = check(a, h)
        print(json.dumps(r, default=str), flush=True)
        if r['ready']:
            return 0
        if not a.wait:
            return 3
        sleep(a.poll_s)


if __name__ == '__main__':
    sys.exit(main())

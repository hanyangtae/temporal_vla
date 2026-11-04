"""Replace a RUNNING fold27 guardian v3 (which holds its sharing contract in memory) by an already-launched NEW guardian v3
that loaded the extended contract, then end ONLY the old guardian. New file; v1-v4 files unchanged. No other process is
ever signalled (consumer, follower, serves, foreign: 0 signals).

Step 0 (main, outside this tool): gpu_sharing_extend.py -> NEW contract file; copy the old guardian config to a NEW
config file whose only change is 'sharing_authorization' -> the new contract; launch the new guardian with the SAME
command as the old one but --config <new cfg> --state-dir <NEW dir> (watch, no --observe-only), detached.

    python guardian_swap.py --old-config C0 --new-config C1 --expect-expires-at E1 --check-dir <NEW scratch dir> \
        --consumer-pid P --consumer-start S --receipt R --old-guardian-pid GP --old-guardian-start GS \
        --new-guardian-pid NP --new-guardian-start NS --out <NEW evidence.json> [--dry-run]

Checks (any failure -> rc 6, no signal):
  config  C1 == C0 except 'sharing_authorization'; contracts: new loads (official load), expires_at == E1 exactly, and
          new == old except expires_at/extensions; still before the OLD expiry (with --margin-s, default 120 s).
  new     NP alive with start NS, cmdline foreign_gpu_guardian_v3.py + `--config C1` + ` watch`, no --observe-only.
  same    an observe-only GuardianV3 built from C1 (scratch state dir) adopts exactly the pinned consumer pid/start/receipt
          and, after one poll, sees no untolerated GPU process (seen == {}), i.e. the new guardian would not act now.
  old     GP alive with start GS, cmdline foreign_gpu_guardian + `--config C0`.
Then (not --dry-run): pidfd_open(GP) -> re-verify start+cmdline -> pidfd_send_signal(SIGTERM) (no PID-reuse race;
without pidfd support: re-verify immediately before os.kill);
wait up to --wait-s for exit. A v1-v3 guardian has no TERM handler: it just exits, it releases/stops nothing.
rc 0 ok / 6 refused / 8 old guardian did not exit (reported; nothing else done).
"""
import argparse, json, os, signal, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import guardian_v4 as G4   # noqa: E402  (path setup; unchanged)

G1, G3 = G4.G1, G4.G3
SA = G3.SA


class Refused(Exception):
    pass


def need(c, msg):
    if not c:
        raise Refused(msg)


def has_arg(cmd, flag, val):
    t = (cmd or '').split()
    return any(t[i] == flag and t[i + 1] == str(val) for i in range(len(t) - 1))


def check(old_cfg_p, new_cfg_p, expect, pins, osi, check_dir, probe=None, margin_s=120.0, now=None):
    c0, c1 = json.loads(Path(old_cfg_p).read_text()), json.loads(Path(new_cfg_p).read_text())
    diff = sorted(k for k in set(c0) | set(c1) if c0.get(k) != c1.get(k))
    need(diff == ['sharing_authorization'], f'config diff {diff} != [sharing_authorization]')
    a0, a1 = SA.load(c0['sharing_authorization']), SA.load(c1['sharing_authorization'])
    need(repr(float(a1['expires_at'])) == repr(float(expect)), f'new contract expires_at {a1["expires_at"]!r} != {expect!r}')
    cd = sorted(k for k in set(a0) | set(a1) if a0.get(k) != a1.get(k))
    need(set(cd) <= {'expires_at', 'extensions'} and 'expires_at' in cd, f'contract diff {cd}')
    now = time.time() if now is None else now
    need(now < float(a0['expires_at']) - margin_s, 'too close to / past the OLD expiry: old guardian may be acting')
    np_, ns = pins['new_guardian_pid'], pins['new_guardian_start']
    cmd = osi.cmdline(np_) or ''
    need(osi.start(np_) == ns, f'new guardian {np_} start {osi.start(np_)} != {ns}')
    need('foreign_gpu_guardian_v3.py' in cmd and has_arg(cmd, '--config', new_cfg_p) and cmd.split()[-1] == 'watch'
         and '--observe-only' not in cmd, f'new guardian cmdline mismatch: {cmd[:300]}')
    gp, gs = pins['old_guardian_pid'], pins['old_guardian_start']
    ocmd = osi.cmdline(gp) or ''
    need(osi.start(gp) == gs and 'foreign_gpu_guardian' in ocmd and has_arg(ocmd, '--config', old_cfg_p),
         f'old guardian {gp} identity/cmdline mismatch: {ocmd[:300]}')
    need(gp != np_, 'old == new guardian pid')
    g = G3.GuardianV3(c1, osi, check_dir, observe_only=True, probe=probe)
    need(float(g.auth['expires_at']) == float(expect), 'guardian built from new config does not hold the new expiry')
    g.refresh()
    cons = g.consumer or {}
    need((cons.get('pid'), cons.get('start'), cons.get('receipt')) ==
         (pins['consumer_pid'], pins['consumer_start'], pins['receipt']), f'new-config guardian adopts {cons} != pinned consumer')
    confirmed = g.poll()
    need(not confirmed and not g.seen, f'untolerated GPU processes under the new contract: {sorted(g.seen)}')
    return {'config_diff': diff, 'contract_diff': cd, 'old_expires_at': a0['expires_at'], 'new_expires_at': a1['expires_at'],
            'adopted': cons, 'owned': [[p, s, r] for (p, s), r in sorted(g.owned.items())], 'new_guardian_cmd': cmd,
            'old_guardian_cmd': ocmd}


def term_old(pins, osi, old_cfg_p, wait_s=30.0, sleep=time.sleep):
    gp, gs = pins['old_guardian_pid'], pins['old_guardian_start']
    def verify():
        ocmd = osi.cmdline(gp) or ''
        need(osi.start(gp) == gs and 'foreign_gpu_guardian' in ocmd and has_arg(ocmd, '--config', old_cfg_p),
             'old guardian identity changed before signal')
    if hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'):
        fd = os.pidfd_open(gp)                    # pins the process: no PID-reuse race between verify and signal
        try:
            verify(); signal.pidfd_send_signal(fd, signal.SIGTERM)
        finally:
            os.close(fd)
    else:                                         # python without pidfd: verify immediately before the kill
        verify(); os.kill(gp, signal.SIGTERM)
    t = 0.0
    while t < wait_s:
        if osi.start(gp) != gs:
            return True
        sleep(0.5); t += 0.5
    return osi.start(gp) != gs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('old-config', 'new-config', 'check-dir', 'out'):
        ap.add_argument('--' + k, type=Path, required=True)
    ap.add_argument('--expect-expires-at', type=float, required=True); ap.add_argument('--receipt', required=True)
    for k in ('consumer-pid', 'consumer-start', 'old-guardian-pid', 'old-guardian-start', 'new-guardian-pid', 'new-guardian-start'):
        ap.add_argument('--' + k, type=int, required=True)
    ap.add_argument('--margin-s', type=float, default=120.0); ap.add_argument('--wait-s', type=float, default=30.0)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(argv)
    if a.out.exists() or a.check_dir.exists():
        print(f'refusing: {a.out} or {a.check_dir} exists (new paths only)', file=sys.stderr); return 2
    pins = {k: getattr(a, k) for k in ('consumer_pid', 'consumer_start', 'receipt', 'old_guardian_pid', 'old_guardian_start',
                                       'new_guardian_pid', 'new_guardian_start')}
    cfg = json.loads(a.new_config.read_text())
    osi = G1.OS(cfg.get('harness', ['python3', 'scripts/utils/groot_harness.py']))
    ev, rc = {'pins': pins, 'dry_run': a.dry_run}, 0
    try:
        ev['check'] = check(a.old_config, a.new_config, a.expect_expires_at, pins, osi, a.check_dir, margin_s=a.margin_s)
        if a.dry_run:
            ev['result'] = 'dry_run_ok'
        else:
            gone = term_old(pins, osi, a.old_config, a.wait_s)
            ev['result'], rc = ('old_guardian_ended', 0) if gone else ('old_guardian_still_alive', 8)
            ev['post'] = {'new_guardian_alive': osi.start(a.new_guardian_pid) == a.new_guardian_start,
                          'consumer_alive': osi.start(a.consumer_pid) == a.consumer_start}
    except (Refused, SA.AuthError, SystemExit, OSError, KeyError, ValueError) as e:
        ev['result'], ev['reason'], rc = 'refused', str(e), 6
    ev['rc'] = rc
    with open(a.out, 'x') as f:
        f.write(json.dumps(ev, indent=1, default=str))
    print(json.dumps({'result': ev['result'], 'rc': rc, 'evidence': str(a.out)}))
    return rc


if __name__ == '__main__':
    sys.exit(main())

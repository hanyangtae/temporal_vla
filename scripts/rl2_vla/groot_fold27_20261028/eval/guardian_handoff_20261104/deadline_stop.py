"""One-shot deadline stop of OUR pinned fold27 eval consumer on one GPU (before a sharing contract expires).
New file; guardian_v4 / v1-v3 files unchanged. No new guardian/watchdog is started.

    python deadline_stop.py --config <the running guardian's cfg.json> --state-dir <NEW dir> --out <NEW evidence.json> \
        --consumer-pid P --consumer-start S --receipt R --old-guardian-pid GP --old-guardian-start GS \
        [--follower-pid FP --follower-start FS] [--dry-run]

Pins (all exact; any mismatch -> rc 6, NO signal at all):
  consumer   pid+start alive on this boot, cmdline `eval_gpu_queue... run`, registration consumers/gpu<G>.json names the
             same pid/start/receipt, harness ledger receipt = config session/machine/[gpu] and consumer_pid == pid.
             A replaced consumer (other pid/start/receipt) is NOT retargeted: rc 6, no signal (re-run with new pins after
             verifying the new receipt yourself).
  follower   detected as the consumer's ancestor outer loop (eval_gpu_queue_v2/v3/v4.sh, not --inner); if --follower-* is
             given it must be that same identity.
  old guard  --old-guardian-pid/start: if alive with that start and cmdline foreign_gpu_guardian, it is SIGSTOPped first
             (so it cannot act concurrently) and TERM+CONT after the stop; any other identity there is never signalled.
Consumer already gone -> rc 7, no signal (evidence only).
Stop = GuardianV4 (take_lock: <queue_dir>/guardians/gpu<G>.lock, so two deadline_stops cannot race) running the unchanged
v1 act(): freeze follower/consumer/arms, arm SIGINT, owned serve pg / docker-exec / verified-container collectors TERM->KILL,
consumer+follower TERM, identity (pid+start+boot) re-checked before every signal, foreign and contract-approved processes
are never owned so never signalled, then `groot_harness.py release` only after own absence + harness wrapper exit
(else receipt kept, rc 5). Progress / claims / partial outputs are never touched.
--dry-run: pins + the real act() in observe-only mode (ownership verify, follower/wrapper detection, ACTION file); no signal.
Evidence JSON (--out, must not exist): pins, plan, result, signals, post-state (consumer/follower/old guardian alive?,
receipt in ledger?, our pids on GPU, approved pids still alive).
"""
import argparse, json, signal, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import guardian_v4 as G4   # noqa: E402

G1 = G4.G1
OLD_GUARD_MARK = 'foreign_gpu_guardian'


class Pinned(Exception):
    pass


def run(cfg, osi, state_dir, pins, dry_run=False, probe=None):
    ev = {'pins': pins, 'dry_run': dry_run, 'signals': [], 'result': None}
    g = None
    try:
        try:
            g = G4.GuardianV4(cfg, osi, state_dir, probe=probe, take_lock=True)
        except SystemExit as e:
            ev.update(result='refused', reason=str(e)); return ev, 6
        o = g.o
        cons = {'pid': pins['consumer_pid'], 'start': pins['consumer_start'], 'receipt': pins['receipt']}
        if not g.same(cons['pid'], cons['start']):
            reg = G1.Guardian.read_consumer(g)
            ev.update(result='consumer_not_running', registered=reg,
                      receipt_in_ledger=pins['receipt'] in (o.harness('status') or {}))
            return ev, 7
        reg = G1.Guardian.read_consumer(g)
        if not reg or (reg['pid'], reg['start'], reg['receipt']) != (cons['pid'], cons['start'], cons['receipt']):
            raise Pinned(f'registration {reg} != pinned consumer (replaced consumer is never retargeted)')
        why = g.receipt_problem(cons['receipt'], cons['pid'])
        if why:
            raise Pinned(why)
        g.adopted = dict(cons); g.consumer = dict(cons); g.refresh()
        fol = g.follower(cons)
        if pins.get('follower_pid') is not None and fol != (pins['follower_pid'], pins['follower_start']):
            raise Pinned(f'detected follower {fol} != pinned follower')
        og = (pins['old_guardian_pid'], pins['old_guardian_start'])
        og_ok = g.same(*og) and OLD_GUARD_MARK in (o.cmdline(og[0]) or '')
        ev.update(follower=fol, old_guardian_verified=og_ok,
                  owned=[[p, s, r] for (p, s), r in sorted(g.owned.items())])
        if dry_run:                                          # real v1 act() in observe-only mode: ownership verify,
            g.observe_only = True                            # follower / wrapper detection, ACTION record; no signal
            r = g.act([])
            ev['result'] = 'dry_run_ok' if r == 'observed' else 'dry_run_' + r
            return ev, 0 if r == 'observed' else 6
        if og_ok:
            g.sig(og[0], og[1], signal.SIGSTOP, 'deadline stop: freeze old guardian (no concurrent act)')
        g.event('deadline_stop', pins=pins)
        r = g.act([])                                        # v1 procedure (verify_ownership first: fail closed)
        if og_ok:
            g.sig(og[0], og[1], signal.SIGTERM, 'deadline stop: end old guardian')
            g.sig(og[0], og[1], signal.SIGCONT, 'deliver TERM to stopped old guardian')
        ev['result'] = r
        try:
            gp = o.gpu_pids(g.gpu)
        except Exception as e:   # noqa: BLE001
            gp = f'query failed: {e}'
        try:
            led = o.harness('status')
            in_ledger = pins['receipt'] in led
        except Exception as e:   # noqa: BLE001
            in_ledger = f'status failed: {e}'
        ev['post'] = {'consumer_alive': g.same(cons['pid'], cons['start']),
                      'follower_alive': bool(fol) and g.same(*fol),
                      'old_guardian_alive': g.same(*og),
                      'receipt_in_ledger': in_ledger, 'gpu_pids': gp,
                      'our_pids_on_gpu': [p for p in gp if g.owned_pid(p)] if isinstance(gp, list) else None,
                      'approved_alive': [[f['pid'], g.same(f['pid'], f.get('start_ticks'))]
                                         for f in ((g.auth or {}).get('foreign') or [])]}
        return ev, {'released': 0, 'released_elsewhere': 0, 'kept': 5, 'refused': 6}.get(r, 1)
    except Pinned as e:
        ev.update(result='refused_pin_mismatch', reason=str(e)); return ev, 6
    finally:
        if g is not None:
            ev['events_file'] = str(g.sd / 'events.jsonl')
            g.close()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True); ap.add_argument('--state-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    for k in ('consumer-pid', 'consumer-start', 'old-guardian-pid', 'old-guardian-start'):
        ap.add_argument('--' + k, type=int, required=True)
    ap.add_argument('--receipt', required=True)
    ap.add_argument('--follower-pid', type=int); ap.add_argument('--follower-start', type=int)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(argv)
    if a.out.exists() or a.state_dir.exists():
        print(f'refusing: {a.out} or {a.state_dir} exists (new paths only)', file=sys.stderr); return 2
    if (a.follower_pid is None) != (a.follower_start is None):
        print('--follower-pid and --follower-start go together', file=sys.stderr); return 2
    cfg = json.loads(a.config.read_text())
    pins = {'consumer_pid': a.consumer_pid, 'consumer_start': a.consumer_start, 'receipt': a.receipt,
            'old_guardian_pid': a.old_guardian_pid, 'old_guardian_start': a.old_guardian_start,
            'follower_pid': a.follower_pid, 'follower_start': a.follower_start}
    osi = G4.OS4(cfg.get('harness', ['python3', 'scripts/utils/groot_harness.py']))
    ev, rc = run(cfg, osi, a.state_dir, pins, a.dry_run)
    ev['rc'] = rc
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'x') as f:
        f.write(json.dumps(ev, indent=1, default=str))
    print(json.dumps({'result': ev['result'], 'rc': rc, 'evidence': str(a.out)}))
    return rc


if __name__ == '__main__':
    sys.exit(main())

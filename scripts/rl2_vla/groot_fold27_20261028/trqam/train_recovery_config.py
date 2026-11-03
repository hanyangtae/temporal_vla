"""Build the train_recovery_driver config from LIVE read-only /proc evidence (new file; writes only --out, O_EXCL).

Captured and verified (any mismatch -> refuse, nothing written):
  old coordinator  PID:START given by main; live start ticks + boot equal before AND after the capture; uid == --expect-owner-uid
                   (operating user; other users' processes refused); uid / raw cmdline sha256 / argv readable (None ->
                   refuse); cwd == this repo root (the driver spawns the replacement with cwd = repo root); argv names
                   host_fold_coordinator_v5_shared.py exactly once and carries --machine/--session/--gpus equal to the
                   arguments and --out/--eval-root resolving to --old-out/--eval-root
  old guardian     same identity rules; cmdline names train_guardian_v2.py
  eval consumer    PID:START equal live (it must be running now; the driver waits for it to end)
  boot id          live == --boot-id
Everything else (contract source/derive budget, eval pins, guardian template) is copied from the arguments verbatim.

    python3 train_recovery_config.py --out <new driver.json> --state-dir <new dir> --machine srv50 --gpu 3 \
        --session <S> --boot-id <B> --old-coordinator 631152:1376533122 --old-guardian 627282:1376530452 \
        --eval-consumer 3088084:<start> --old-out <candle train out> --eval-root <bread eval root> --eval-queue-state <json> \
        --eval-folds F1 F2 F3 --eval-arms A1 A2 A3 A4 --rows-per-arm 40 --contract-source <SHARING_PID_1735.json> \
        --foreign-pid P --foreign-uid U --foreign-start T --foreign-cmdline-sha256 H --gpu-uuid GPU-... \
        --per-serve-mib 7168 --per-serve-basis measured_4584_x1.5_rounded_7168 --measured-peak-mib 4584 --reserve-mib 16384 \
        --expires-in-h 24 --approval '<user statement>' --guardian-python .fold27_env/bin/python
"""
import argparse, json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_recovery_driver as RD     # noqa: E402


class ProcView(RD.HostOS):
    def argv(self, pid):
        try:
            return [x.decode(errors='replace') for x in Path(f'/proc/{int(pid)}/cmdline').read_bytes().split(b'\0') if x]
        except (OSError, ValueError):
            return None

    def cwd(self, pid):
        try:
            return os.path.realpath(f'/proc/{int(pid)}/cwd')
        except OSError:
            return None


def ident(h, spec, marks, what, owner_uid, boot):
    pid, start = (int(x) for x in spec.split(':'))
    if h.boot() != boot or h.start(pid) != start:
        raise RD.Refused(f'{what} {pid}: live start {h.start(pid)} != given {start} (or boot changed)')
    argv, uid, sha = h.argv(pid) or [], h.uid(pid), h.cmd_sha(pid)
    if h.boot() != boot or h.start(pid) != start:                             # identity unchanged across the capture
        raise RD.Refused(f'{what} {pid}: identity changed during capture')
    if uid is None or sha is None or not argv:
        raise RD.Refused(f'{what} {pid}: uid/cmdline unreadable')
    if uid != owner_uid:
        raise RD.Refused(f'{what} {pid}: uid {uid} != operating owner uid {owner_uid} (not our process)')
    if not any(any(x.endswith(m) for x in argv) for m in marks):
        raise RD.Refused(f'{what} {pid}: argv does not name {marks}')
    return {'pid': pid, 'start': start, 'uid': uid, 'cmdline_sha256': sha}, argv


def _abs(p):
    p = Path(p)
    return str((p if p.is_absolute() else RD.REPO / p).resolve())


def build(a, h):
    if h.boot() != a.boot_id:
        raise RD.Refused('live boot id differs from --boot-id')
    oc, oargv = ident(h, a.old_coordinator, ('host_fold_coordinator_v5_shared.py',), 'old coordinator', a.expect_owner_uid, a.boot_id)
    if sum(x.endswith('host_fold_coordinator_v5_shared.py') for x in oargv) != 1:
        raise RD.Refused('old coordinator argv must name host_fold_coordinator_v5_shared.py exactly once')
    for k, v in (('--session', a.session), ('--gpus', str(a.gpu)), ('--machine', a.machine)):
        if k not in oargv or oargv[oargv.index(k) + 1] != v:
            raise RD.Refused(f'old coordinator argv {k} != {v}')
    for k, v in (('--out', a.old_out), ('--eval-root', a.eval_root)):
        if k not in oargv or _abs(oargv[oargv.index(k) + 1]) != _abs(v):
            raise RD.Refused(f'old coordinator argv {k} does not equal the given {k} {v}')
    if h.cwd(oc['pid']) != str(RD.REPO.resolve()):
        raise RD.Refused(f'old coordinator cwd {h.cwd(oc["pid"])} != this repo root {RD.REPO.resolve()}')
    og, _ = ident(h, a.old_guardian, ('train_guardian_v2.py',), 'old guardian', a.expect_owner_uid, a.boot_id)
    ep, es = (int(x) for x in a.eval_consumer.split(':'))
    if h.start(ep) != es:
        raise RD.Refused(f'eval consumer {ep}: live start {h.start(ep)} != given {es}')
    cfg = {'kind': RD.KIND, 'state_dir': str(a.state_dir), 'boot_id': a.boot_id, 'eval_queue_state': str(a.eval_queue_state),
           'boundary': dict(machine=a.machine, gpu=a.gpu, boot_id=a.boot_id, session=a.session, contract=str(a.contract_source),
                            eval_consumer=a.eval_consumer, old_coordinator=a.old_coordinator, old_guardian=a.old_guardian,
                            old_out=str(a.old_out)),
           'eval_complete': {'root': str(a.eval_root), 'host': a.machine, 'rows_per_arm': a.rows_per_arm,
                             'expected': {f: list(a.eval_arms) for f in a.eval_folds}},
           'old_coordinator': {**oc, 'argv': oargv}, 'old_guardian': og,
           'derive': dict(source=str(a.contract_source), session=a.session, max_serves=1, per_serve_mib=a.per_serve_mib,
                          per_serve_basis=a.per_serve_basis, measured_peak_mib=a.measured_peak_mib, reserve_mib=a.reserve_mib,
                          expires_in_h=a.expires_in_h, expect_pid=a.foreign_pid, expect_uid=a.foreign_uid,
                          expect_start_ticks=a.foreign_start, expect_cmdline_sha256=a.foreign_cmdline_sha256,
                          expect_boot_id=a.boot_id, expect_gpu_uuid=a.gpu_uuid, approval=a.approval),
           'guardian': {'python': a.guardian_python, 'config': {'host': a.machine, 'gpu': a.gpu, 'session': a.session,
                                                                 'confirm_polls': 2, 'poll_s': a.guardian_poll_s,
                                                                 'kill_wait_s': 60, 'release_wait_s': 600, 'clock_skew_s': 900}},
           'poll_s': a.poll_s, 'term_wait_s': a.term_wait_s, 'settle_s': a.settle_s}
    with open(a.out, 'x') as f:
        f.write(json.dumps(cfg, indent=1))
    return cfg


def main(argv=None, host=None):
    ap = argparse.ArgumentParser()
    for k, t in (('out', Path), ('state-dir', Path), ('machine', str), ('gpu', int), ('session', str), ('boot-id', str),
                 ('old-coordinator', str), ('old-guardian', str), ('eval-consumer', str), ('old-out', Path), ('eval-root', Path),
                 ('eval-queue-state', Path), ('rows-per-arm', int), ('contract-source', Path), ('foreign-pid', int),
                 ('foreign-uid', int), ('foreign-start', int), ('foreign-cmdline-sha256', str), ('gpu-uuid', str),
                 ('per-serve-mib', int), ('per-serve-basis', str), ('measured-peak-mib', int), ('reserve-mib', int),
                 ('expires-in-h', float), ('approval', str), ('guardian-python', str), ('expect-owner-uid', int)):
        ap.add_argument('--' + k, type=t, required=True)
    ap.add_argument('--eval-folds', nargs='+', required=True); ap.add_argument('--eval-arms', nargs='+', required=True)
    ap.add_argument('--guardian-poll-s', type=float, default=10); ap.add_argument('--poll-s', type=float, default=300)
    ap.add_argument('--term-wait-s', type=float, default=120); ap.add_argument('--settle-s', type=float, default=30)
    a = ap.parse_args(argv)
    try:
        cfg = build(a, host or ProcView())
    except (RD.Refused, FileExistsError, ValueError) as e:
        print('TRAIN_RECOVERY_CONFIG_REFUSED', e, file=sys.stderr); return 2
    print(json.dumps({'written': str(a.out), 'old_coordinator': cfg['old_coordinator']['pid'], 'old_guardian': cfg['old_guardian']['pid']}))
    return 0


if __name__ == '__main__':
    sys.exit(main())

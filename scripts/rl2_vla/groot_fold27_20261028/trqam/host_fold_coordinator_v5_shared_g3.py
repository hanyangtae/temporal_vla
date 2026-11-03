"""host_fold_coordinator_v5_shared (deployed, unchanged) requiring train_guardian_v3 instead of v2. Nothing else differs:
same v5 scheduling/fair share/TRAIN_REQUEST/claims/DONE/resume, same contract scope check (v1 or v2; for v1 the contract
session must equal --session), same per-round sharing check (gpu_sharing_authorization.authorized_processes, which
supports v1), same HARNESS_SHARING_AUTH in each arm env. The demand-root behaviour of v5 (_demand_cli without --root)
is deliberately NOT changed here (host_fold_coordinator_v5_shared_er.py is a separate, undeployed fix).
Guardian requirement (every round and at start): pid + start ticks on the recorded boot, argv = train_guardian_v3.py with
--config <the given file> and no --observe-only, config kind fold27_train_guardian_v3 with host/gpu/session/contract equal
to this coordinator's and not observe_only, argv identity re-checked after the cmdline read, no pending ALERT.json /
ADMISSION_UNSAFE.json in its --state-dir.

    .fold27_env/bin/python host_fold_coordinator_v5_shared_g3.py <v5_shared args> (guardian pid/start/config = the v3 guardian)
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import host_fold_coordinator_v5_shared as HS   # noqa: E402  (unchanged)

H5, HC, SA, TQ = HS.H5, HS.HC, HS.SA, HS.TQ
GUARD_SCRIPT, GUARD_KIND = 'train_guardian_v3.py', 'fold27_train_guardian_v3'


class HostCoordinatorV5SharedG3(HS.HostCoordinatorV5Shared):
    def guardian_check(self):
        pid, start, cfg = self.gid

        def same():
            return self.gprobe.boot() == self.gboot and self.gprobe.start(pid) == start
        if not same():
            return False, 'guardian pid/start/boot changed or gone'
        argv = self.gprobe.argv(pid)
        if not same():
            return False, 'guardian pid/start/boot changed during check'
        if not any(x.endswith(GUARD_SCRIPT) for x in argv) or '--observe-only' in argv:
            return False, f'not an enforcing {GUARD_SCRIPT} process'
        if HS._real(HS._arg(argv, '--config')) != HS._real(cfg):
            return False, 'guardian runs a different config'
        c = self.gprobe.read_json(cfg)
        if not c or c.get('kind') != GUARD_KIND or c.get('host') != self.a.machine \
                or int(c.get('gpu', -1)) != int(self.a.gpus[0]) or c.get('session') != self.a.session \
                or HS._real(c.get('sharing_authorization')) != HS._real(self.auth_path) or c.get('observe_only'):
            return False, 'guardian config kind/host/gpu/session/contract/enforcing mismatch'
        sd = HS._arg(argv, '--state-dir')
        if not sd:
            return False, 'guardian state dir unknown'
        for f in ('ALERT.json', 'ADMISSION_UNSAFE.json'):
            if self.gprobe.exists(str(Path(sd) / f)):
                return False, f'guardian has pending {f} (main must inspect and clear before training resumes)'
        return True, 'ok'


def main():
    a = HS.parse()
    try:
        print(json.dumps(HostCoordinatorV5SharedG3(a).run()))
    except (HC.CoordError, TQ.QueueError, SA.AuthError) as e:
        print('HOST_COORD_V5_SHARED_G3_ERROR', e, file=sys.stderr); sys.exit(2)


if __name__ == '__main__':
    main()

"""v5 coordinator for ONE user-approved shared GPU (new; host_fold_coordinator_v5.py and everything below it unchanged).

The ONLY behavioural difference from HostCoordinatorV5: the GPU probe injected through v5's own constructor argument
`busy=` (used only in free_gpus). v5 defaults it to eval_pipeline.gpu_busy, which treats ANY compute process as busy, so
the user-approved foreign processes on the shared GPU keep the coordinator waiting forever before the official harness
is even asked. Here a pool GPU counts as free only if ALL of the following hold, re-checked at every scheduling round:
  * the sharing contract (HARNESS_SHARING_AUTH, gpu_sharing_authorization v1/v2) passes authorized_processes for
    (machine, [gpu], this coordinator's --session, kind eval, serves 1): every compute process on that GPU is a pinned
    approved process or a verified member of the pinned container instance, bounds hold, free >= per_serve + reserve,
    unexpired, same boot / UUID -- any refusal or query failure = busy (fail closed)
  * a train_guardian_v2 runtime guardian is running in ENFORCING mode for this session/GPU, identified by pid + start
    ticks (+ boot) and its cmdline (train_guardian_v2.py, --config <the configured file>, no --observe-only); otherwise
    busy -- no training arm is launched on the shared GPU without runtime protection
Everything else is v5 unchanged: fair share (eval demand, train_share), TRAIN_REQUEST publish/withdraw (eval consumers
yield at their next completed arm), harness-ledger exclusion (a GPU with ANY receipt, e.g. the eval consumer's, is never
free -> eval and training never overlap on it), arm claims, DONE skip, resume, failure path, folds sequential. Each arm's
env additionally carries HARNESS_SHARING_AUTH so the official harness (groot_harness.probe_gpus) re-verifies the SAME
contract at admission, right after admission and before release. --gpus must be exactly the contract's single GPU.

    .fold27_env/bin/python host_fold_coordinator_v5_shared.py <v5 args> --gpus 3 --sharing-auth <contract.json> \
        --train-guardian-pid <pid> --train-guardian-start <ticks> --train-guardian-config <guardian cfg.json>
"""
import json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[3] / 'scripts/utils')]
import host_fold_coordinator_v5 as H5      # noqa: E402  (unchanged)
import gpu_sharing_authorization as SA     # noqa: E402

HC, TQ, JC = H5.HC, H5.TQ, H5.JC


class GuardianProbe:
    """Read-only view of the required runtime guardian (/proc + its config + its state dir)."""

    def start(self, pid):
        return JC.proc_start(pid)

    def boot(self):
        return JC.boot_id()

    def argv(self, pid):
        try:
            return [x.decode(errors='replace') for x in Path(f'/proc/{int(pid)}/cmdline').read_bytes().split(b'\0') if x]
        except (OSError, ValueError):
            return []

    def read_json(self, path):
        try:
            return json.loads(Path(path).read_text())
        except (OSError, ValueError):
            return None

    def exists(self, path):
        return Path(path).exists()


def _arg(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else None


def _real(p):
    return str(Path(p).resolve()) if p else None


class HostCoordinatorV5Shared(H5.HostCoordinatorV5):
    def __init__(self, a, harness=None, busy=None, demand=None, spawn=subprocess.Popen, probe=None, gprobe=None, **kw):
        self.auth_path = str(a.sharing_auth)
        auth = SA.load(self.auth_path)
        scope = auth['sessions'] if auth['kind'] == SA.KIND_V2 else {auth['session']: None}
        if auth['machine'] != a.machine or [int(auth['gpu'])] != [int(g) for g in a.gpus] or a.session not in scope:
            raise HC.CoordError('sharing contract does not cover this machine / --gpus (exactly its one GPU) / --session')
        self.sprobe, self.gprobe = probe, gprobe or GuardianProbe()
        self.gid = (int(a.train_guardian_pid), int(a.train_guardian_start), str(a.train_guardian_config))
        # reference boot = the contract's pinned boot (not whatever the probe says at start: a fresh boot would otherwise
        # become its own reference and a stale pid/start pair could match a different process)
        self.gboot = auth.get('boot_id') or self.gprobe.boot()
        super().__init__(a, harness=harness, busy=busy or self.shared_busy, demand=demand, spawn=spawn, **kw)
        ok, why = self.guardian_check()
        if not ok:
            raise HC.CoordError(f'required train_guardian_v2 not valid at start: {why}')

    def guardian_check(self):
        """(ok, reason). The guardian must be THIS session's enforcing train_guardian_v2 on this machine/GPU with the SAME
        contract, alive with the recorded start ticks on the same boot, and have no pending ALERT / ADMISSION_UNSAFE."""
        pid, start, cfg = self.gid

        def same():
            return self.gprobe.boot() == self.gboot and self.gprobe.start(pid) == start
        if not same():
            return False, 'guardian pid/start/boot changed or gone'
        argv = self.gprobe.argv(pid)
        # identity re-checked AFTER the cmdline read (bracketing): the argv judged below must belong to the same live
        # process on the same boot, not to a pid reused / a boot change between the two /proc reads
        if not same():
            return False, 'guardian pid/start/boot changed during check'
        if not any(x.endswith('train_guardian_v2.py') for x in argv) or '--observe-only' in argv:
            return False, 'not an enforcing train_guardian_v2 process'
        if _real(_arg(argv, '--config')) != _real(cfg):
            return False, 'guardian runs a different config'
        c = self.gprobe.read_json(cfg)
        if not c or c.get('kind') != 'fold27_train_guardian_v2' or c.get('host') != self.a.machine \
                or int(c.get('gpu', -1)) != int(self.a.gpus[0]) or c.get('session') != self.a.session \
                or _real(c.get('sharing_authorization')) != _real(self.auth_path) or c.get('observe_only'):
            return False, 'guardian config kind/host/gpu/session/contract/enforcing mismatch'
        sd = _arg(argv, '--state-dir')
        if not sd:
            return False, 'guardian state dir unknown'
        for f in ('ALERT.json', 'ADMISSION_UNSAFE.json'):
            if self.gprobe.exists(str(Path(sd) / f)):
                return False, f'guardian has pending {f} (main must inspect and clear before training resumes)'
        return True, 'ok'

    def guardian_ok(self):
        return self.guardian_check()[0]

    def shared_busy(self, gpus):
        busy = []
        ok, why = self.guardian_check()
        if not ok:
            self.state.setdefault('events', []).append({'time': self.now(), 'shared_gpu_busy': f'guardian: {why}'})
            return list(gpus)
        for g in gpus:
            try:
                SA.authorized_processes(SA.load(self.auth_path), self.a.machine, [g], self.a.session, 'eval', 1, self.sprobe)
            except (SA.AuthError, OSError, ValueError, KeyError) as e:
                busy.append(g)
                self.state.setdefault('events', []).append({'time': self.now(), 'shared_gpu_busy': str(e)[:300]})
        return busy

    def arm_cmd(self, q, f, arm, gpu):
        cmd, env = super().arm_cmd(q, f, arm, gpu)
        env['HARNESS_SHARING_AUTH'] = self.auth_path
        return cmd, env


def parse(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = {}
    for k in ('--sharing-auth', '--train-guardian-pid', '--train-guardian-start', '--train-guardian-config'):
        if k not in argv:
            raise SystemExit(f'{k} required')
        i = argv.index(k); extra[k] = argv[i + 1]; del argv[i:i + 2]
    a = H5.parse(argv)
    a.sharing_auth, a.train_guardian_pid = Path(extra['--sharing-auth']), extra['--train-guardian-pid']
    a.train_guardian_start, a.train_guardian_config = extra['--train-guardian-start'], extra['--train-guardian-config']
    return a


def main():
    a = parse()
    try:
        print(json.dumps(HostCoordinatorV5Shared(a).run()))
    except (HC.CoordError, TQ.QueueError, SA.AuthError) as e:
        print('HOST_COORD_V5_SHARED_ERROR', e, file=sys.stderr); sys.exit(2)


if __name__ == '__main__':
    main()

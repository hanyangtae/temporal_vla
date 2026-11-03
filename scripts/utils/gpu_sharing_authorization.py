#!/usr/bin/env python3
"""ONE-TIME, single-experiment GPU sharing authorization (stdlib only; used by groot_harness.py and the fold27 guardian v2).

Default everywhere: a GPU with ANY compute process is refused (groot_harness.check_gpu_processes). This module only
changes that when the caller explicitly passes a contract file (env HARNESS_SHARING_AUTH=<path>) AND every field of the
request and of the live GPU matches the contract. It is not a general override and authorizes nothing else.

Contract (kind groot_gpu_sharing_exception_v1, written by `prepare` from LIVE state + main-supplied expectations):
  machine / hostname / gpu index / gpu_uuid / session / experiment / kind / max_serves / expires_at (epoch, required)
  boot_id                       kernel boot id at preparation (a reboot voids the contract)
  per_serve_mib, reserve_mib    required free memory at every check = serves * per_serve_mib + reserve_mib
  foreign: exactly the approved process(es), each pinned by pid + uid + start_ticks + cmdline (raw bytes sha256 and text)
           + max_used_mib (its GPU memory may not grow beyond this)
Check (`authorized_processes`): host/machine/GPU/UUID/session/kind/experiment/serves/expiry/boot id equal; GPU query
succeeds; every compute PID on that GPU is one of the pinned identities (pid AND start_ticks AND uid AND cmdline AND boot
id all equal -- PID alone never matches; a reused PID or a different/new process fails); its used memory <= max_used_mib;
free memory >= the requirement. If the approved process has ENDED the GPU simply is empty (normal path); the contract
never extends to a replacement process. Any doubt -> AuthError (callers fail closed).

    python3 gpu_sharing_authorization.py prepare --machine srv50 --gpu 3 --gpu-uuid GPU-... --session S --experiment fold27 \
        --pid 1101996 --expect-uid 0 --expect-cmd '<exact cmdline>' --max-used-mib 22000 --max-serves 4 \
        --per-serve-mib 6118 --reserve-mib 8192 --expires-in-h 24 --approval '<user statement>' --out <new path>
    python3 gpu_sharing_authorization.py check --auth <path> --machine srv50 --gpus 3 --session S --kind eval --serves 4
"""
import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

KIND = 'groot_gpu_sharing_exception_v1'
POLICY = Path(__file__).resolve().parents[2] / 'configs/harness/gpu_policy.json'


class AuthError(ValueError):
    pass


def need(ok, msg):
    if not ok:
        raise AuthError(msg)


# ---------------- host probes (injectable) ----------------
class Probe:
    def boot_id(self):
        return Path('/proc/sys/kernel/random/boot_id').read_text().strip()

    def hostname(self):
        return socket.gethostname().split('.')[0]

    def now(self):
        return time.time()

    def proc(self, pid):
        """{'uid','start_ticks','cmdline_sha256','cmdline'} or None if absent."""
        try:
            stat = Path(f'/proc/{int(pid)}/stat').read_text()
            status = Path(f'/proc/{int(pid)}/status').read_text()
            raw = Path(f'/proc/{int(pid)}/cmdline').read_bytes()
        except (OSError, ValueError):
            return None
        uid = next(int(l.split()[1]) for l in status.splitlines() if l.startswith('Uid:'))
        return {'uid': uid, 'start_ticks': int(stat.rsplit(')', 1)[1].split()[19]),
                'cmdline_sha256': hashlib.sha256(raw).hexdigest(), 'cmdline': raw.replace(b'\0', b' ').decode(errors='replace').strip()}

    def gpu(self, index):
        """{'uuid','free_mib','total_mib','apps': [{'pid','used_mib'}]}; raises on any query failure."""
        r = subprocess.run(['nvidia-smi', f'--id={index}', '--query-gpu=uuid,memory.free,memory.total', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=60)
        need(r.returncode == 0 and r.stdout.strip(), f'GPU{index}: nvidia-smi query failed: {r.stderr.strip()}')
        uuid, free, total = [x.strip() for x in r.stdout.strip().splitlines()[0].split(',')]
        a = subprocess.run(['nvidia-smi', f'--id={index}', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=60)
        need(a.returncode == 0, f'GPU{index}: nvidia-smi compute-apps failed: {a.stderr.strip()}')
        apps = []
        for line in a.stdout.strip().splitlines():
            if line.strip():
                pid, used = [x.strip() for x in line.split(',')]
                need(pid.isdigit() and used.isdigit(), f'GPU{index}: unparsable compute app line {line!r}')
                apps.append({'pid': int(pid), 'used_mib': int(used)})
        return {'uuid': uuid, 'free_mib': int(free), 'total_mib': int(total), 'apps': apps}


# ---------------- contract ----------------
FIELDS = ('kind', 'machine', 'hostname', 'gpu', 'gpu_uuid', 'session', 'experiment', 'allowed_kind', 'max_serves',
          'per_serve_mib', 'reserve_mib', 'expires_at', 'boot_id', 'foreign', 'approval', 'prepared_at')


def load(path):
    a = json.loads(Path(path).read_text())
    need(a.get('kind') == KIND, f'authorization kind must be {KIND}')
    missing = [f for f in FIELDS if f not in a]
    need(not missing, f'authorization missing {missing}')
    need(isinstance(a['foreign'], list) and a['foreign'], 'authorization must pin at least one approved process')
    for f in a['foreign']:
        need(all(k in f for k in ('pid', 'uid', 'start_ticks', 'cmdline_sha256', 'cmdline', 'max_used_mib')),
             'every approved process needs pid, uid, start_ticks, cmdline_sha256, cmdline, max_used_mib (PID alone is never enough)')
    pol = json.loads(POLICY.read_text())['machines']
    need(a['machine'] in pol and pol[a['machine']]['hostname'] == a['hostname'], 'machine/hostname not in GPU policy')
    need(1 <= int(a['max_serves']) <= pol[a['machine']]['max_serves_per_gpu'], 'max_serves outside the GPU policy (never raised)')
    return a


def authorized_processes(auth, machine, gpus, session, kind, serves, probe=None):
    """Raise AuthError unless the request and the live GPU match the contract exactly. Returns the verified summary."""
    probe = probe or Probe()
    need(probe.now() < float(auth['expires_at']), 'sharing authorization expired')
    need(probe.hostname() == auth['hostname'], f'host {probe.hostname()} != authorized {auth["hostname"]}')
    need(machine == auth['machine'], f'machine {machine} != authorized {auth["machine"]}')
    need(list(gpus) == [int(auth['gpu'])], f'GPUs {list(gpus)} != authorized [{auth["gpu"]}] (single GPU only)')
    need(session == auth['session'], f'session {session} != authorized {auth["session"]}')
    need(kind == auth['allowed_kind'], f'kind {kind} != authorized {auth["allowed_kind"]}')
    need(1 <= int(serves) <= int(auth['max_serves']), f'serves {serves} exceeds authorized {auth["max_serves"]}')
    need(probe.boot_id() == auth['boot_id'], 'boot id changed since authorization (reboot): contract void')
    g = probe.gpu(int(auth['gpu']))
    need(g['uuid'] == auth['gpu_uuid'], f'GPU{auth["gpu"]} uuid {g["uuid"]} != authorized {auth["gpu_uuid"]}')
    pinned = {int(f['pid']): f for f in auth['foreign']}
    seen = []
    for app in g['apps']:
        f = pinned.get(app['pid'])
        need(f is not None, f'GPU{auth["gpu"]}: unapproved compute process PID {app["pid"]}')
        live = probe.proc(app['pid'])
        need(live is not None, f'PID {app["pid"]} listed by NVML but /proc unreadable: identity unverifiable')
        for k in ('uid', 'start_ticks', 'cmdline_sha256'):
            need(live[k] == f[k], f'PID {app["pid"]} {k} {live[k]!r} != authorized {f[k]!r} (PID reuse / different process)')
        need(app['used_mib'] <= int(f['max_used_mib']), f'approved PID {app["pid"]} uses {app["used_mib"]} MiB > authorized {f["max_used_mib"]}')
        seen.append(app['pid'])
    required = int(serves) * int(auth['per_serve_mib']) + int(auth['reserve_mib'])
    need(g['free_mib'] >= required, f'GPU{auth["gpu"]} free {g["free_mib"]} MiB < required {required} MiB '
                                    f'({serves} x {auth["per_serve_mib"]} + {auth["reserve_mib"]})')
    return {'gpu': auth['gpu'], 'uuid': g['uuid'], 'approved_present': seen, 'free_mib': g['free_mib'], 'required_mib': required}


def matches_identity(auth, pid, probe=None):
    """True only if pid is a pinned approved process with identical uid/start_ticks/cmdline and the same boot."""
    probe = probe or Probe()
    f = {int(x['pid']): x for x in auth['foreign']}.get(int(pid))
    if f is None or probe.boot_id() != auth['boot_id'] or probe.now() >= float(auth['expires_at']):
        return False
    live = probe.proc(pid)
    return live is not None and all(live[k] == f[k] for k in ('uid', 'start_ticks', 'cmdline_sha256'))


def prepare(a, probe=None):
    """Write a contract from LIVE identity; main's expectations (uid, exact cmdline, uuid) must match what is observed."""
    probe = probe or Probe()
    pol = json.loads(POLICY.read_text())['machines']
    need(a.machine in pol and probe.hostname() == pol[a.machine]['hostname'], 'prepare must run on the authorized host itself')
    g = probe.gpu(a.gpu)
    need(g['uuid'] == a.gpu_uuid, f'GPU{a.gpu} uuid {g["uuid"]} != expected {a.gpu_uuid}')
    pids = [x['pid'] for x in g['apps']]
    need(pids == [a.pid], f'GPU{a.gpu} compute processes {pids} != exactly the approved PID [{a.pid}]')
    live = probe.proc(a.pid)
    need(live is not None, f'PID {a.pid} not readable')
    need(live['uid'] == a.expect_uid, f'PID {a.pid} uid {live["uid"]} != expected {a.expect_uid}')
    need(live['cmdline'] == a.expect_cmd, f'PID {a.pid} cmdline {live["cmdline"]!r} != expected {a.expect_cmd!r}')
    used = g['apps'][0]['used_mib']
    need(used <= a.max_used_mib, f'PID {a.pid} already uses {used} MiB > max_used_mib {a.max_used_mib}')
    now = probe.now()
    body = {'kind': KIND, 'machine': a.machine, 'hostname': probe.hostname(), 'gpu': a.gpu, 'gpu_uuid': g['uuid'],
            'session': a.session, 'experiment': a.experiment, 'allowed_kind': a.kind, 'max_serves': a.max_serves,
            'per_serve_mib': a.per_serve_mib, 'reserve_mib': a.reserve_mib, 'expires_at': now + a.expires_in_h * 3600,
            'boot_id': probe.boot_id(), 'prepared_at': now, 'approval': a.approval,
            'observed': {'free_mib': g['free_mib'], 'total_mib': g['total_mib'], 'approved_used_mib': used},
            'foreign': [{'pid': a.pid, **{k: live[k] for k in ('uid', 'start_ticks', 'cmdline_sha256', 'cmdline')},
                         'max_used_mib': a.max_used_mib}]}
    need(1 <= a.max_serves <= pol[a.machine]['max_serves_per_gpu'], 'max_serves outside the GPU policy')
    with open(a.out, 'x') as f:
        f.write(json.dumps(body, indent=1))
    load(a.out)
    return body


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prepare')
    for k, t in (('machine', str), ('gpu', int), ('gpu-uuid', str), ('session', str), ('experiment', str), ('pid', int),
                 ('expect-uid', int), ('expect-cmd', str), ('max-used-mib', int), ('max-serves', int), ('approval', str)):
        p.add_argument('--' + k, type=t, required=True)
    p.add_argument('--kind', default='eval', choices=('eval',)); p.add_argument('--per-serve-mib', type=int, default=6118)
    p.add_argument('--reserve-mib', type=int, default=8192); p.add_argument('--expires-in-h', type=float, required=True)
    p.add_argument('--out', type=Path, required=True)
    c = sub.add_parser('check')
    c.add_argument('--auth', type=Path, required=True); c.add_argument('--machine', required=True); c.add_argument('--gpus', required=True)
    c.add_argument('--session', required=True); c.add_argument('--kind', default='eval'); c.add_argument('--serves', type=int, required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == 'prepare':
            need(a.expires_in_h <= 72, 'expiry must be <= 72 h (one experiment)')
            r = prepare(a)
        else:
            r = authorized_processes(load(a.auth), a.machine, [int(g) for g in a.gpus.split(',')], a.session, a.kind, a.serves)
    except AuthError as e:
        print(f'[sharing] REFUSED: {e}', file=sys.stderr)
        return 2
    print(json.dumps(r, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())

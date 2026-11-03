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

    def status(self, pid):
        """/proc/<pid>/status as {key: value} (world-readable: PPid, Uid, Gid, NSpid) or None."""
        try:
            return dict(l.split(':', 1) for l in Path(f'/proc/{int(pid)}/status').read_text().splitlines() if ':' in l)
        except (OSError, ValueError):
            return None

    def cgroup(self, pid):
        """/proc/<pid>/cgroup text (world-readable) or None."""
        try:
            return Path(f'/proc/{int(pid)}/cgroup').read_text()
        except (OSError, ValueError):
            return None

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
    if a.get('kind') == KIND_V2:
        return load_v2_body(a)
    need(a.get('kind') == KIND, f'authorization kind must be {KIND} or {KIND_V2}')
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
    if auth.get('kind') == KIND_V2:
        return authorized_processes_v2(auth, machine, gpus, session, kind, serves, probe)
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


# ======================================================================================================================
# kind v2: pinned processes (as v1) + ONE OR MORE pinned foreign CONTAINER INSTANCES whose changing children are admitted
# only by verified membership, + several sessions (eval and the existing fold27 training) with explicit per-job memory.
# Container instance evidence (all world-readable; nothing bypasses permissions -- /proc/<pid>/ns/* links are NOT read,
# they are unreadable for another uid without privileges):
#   init   host pid + start_ticks (+ boot id) of the container's init (docker inspect State.Pid), its exact cgroup line
#          (cgroup v2 '0::/docker/<64-hex id>'), its NSpid innermost value == 1 (it is PID 1 of a nested namespace),
#          its real uid/gid
#   member every GPU process admitted through a container must: have EXACTLY that cgroup content; real uid/gid ==
#          the pinned member uid/gid; a nested NSpid (>= 2 levels); a PPid chain (<= MAX_HOPS) up to the pinned init
#          where every hop is alive, has the same exact cgroup, a nested NSpid and a start time <= its child's; and
#          start no earlier than init. A restarted/replaced container (new init start/pid), another container, a host
#          process, or a same-uid/same-command process outside the cgroup never qualifies.
#   memory aggregate NVML used_mib of the container's processes on this GPU <= the container's max_used_mib
# ======================================================================================================================
KIND_V2 = 'groot_gpu_sharing_exception_v2'
FIELDS_V2 = ('kind', 'machine', 'hostname', 'gpu', 'gpu_uuid', 'experiment', 'sessions', 'reserve_mib', 'expires_at',
             'boot_id', 'foreign', 'containers', 'approval', 'prepared_at')
MAX_HOPS = 64
HEX64 = set('0123456789abcdef')


def load_v2_body(a):
    missing = [f for f in FIELDS_V2 if f not in a]
    need(not missing, f'authorization missing {missing}')
    pol = json.loads(POLICY.read_text())['machines']
    need(a['machine'] in pol and pol[a['machine']]['hostname'] == a['hostname'], 'machine/hostname not in GPU policy')
    need(isinstance(a['sessions'], dict) and a['sessions'], 'v2 needs an explicit sessions map')
    for s, v in a['sessions'].items():
        need(v.get('allowed_kind') == 'eval', f'session {s}: allowed_kind must be eval (the harness kind of eval and fold27 training)')
        need(1 <= int(v.get('max_serves', 0)) <= pol[a['machine']]['max_serves_per_gpu'], f'session {s}: max_serves outside the GPU policy')
        need(int(v.get('per_serve_mib', 0)) >= 1, f'session {s}: explicit per_serve_mib (measured) required')
    need(isinstance(a['foreign'], list) and isinstance(a['containers'], list) and (a['foreign'] or a['containers']),
         'v2 must pin at least one approved process or container')
    for f in a['foreign']:
        need(all(k in f for k in ('pid', 'uid', 'start_ticks', 'cmdline_sha256', 'cmdline', 'max_used_mib')),
             'every approved process needs pid, uid, start_ticks, cmdline_sha256, cmdline, max_used_mib (PID alone is never enough)')
    for c in a['containers']:
        cid = str(c.get('id', ''))
        need(len(cid) == 64 and set(cid) <= HEX64, 'container id must be the full 64-hex docker id')
        need(c.get('cgroup') == f'0::/docker/{cid}', 'container cgroup must be exactly 0::/docker/<id>')
        init = c.get('init') or {}
        need(all(k in init for k in ('pid', 'start_ticks', 'uid', 'gid')), 'container init needs pid, start_ticks, uid, gid')
        need(all(k in c for k in ('member_uid', 'member_gid', 'max_used_mib', 'image_id', 'started_at', 'devices')),
             'container needs member_uid, member_gid, max_used_mib, image_id, started_at, devices')
        need(c['devices'] == [a['gpu_uuid']], 'container must be bound to exactly the authorized GPU UUID')
    return a


def session_scope(auth, session):
    need(session in auth['sessions'], f'session {session} not in authorized sessions {sorted(auth["sessions"])}')
    return auth['sessions'][session]


def _status(probe, pid):
    st = probe.status(pid)
    if st is None:
        return None
    try:
        return {'ppid': int(st['PPid'].split()[0]), 'uid': int(st['Uid'].split()[0]), 'gid': int(st['Gid'].split()[0]),
                'nspid': [int(x) for x in st['NSpid'].split()]}
    except (KeyError, ValueError, IndexError):
        return None


def _cg(probe, pid):
    t = probe.cgroup(pid)
    return None if t is None else [l for l in t.splitlines() if l.strip()]


def container_member(c, pid, probe):
    """None if pid is a verified member of container instance c, else the reason (string)."""
    init = c['init']
    ist, ips, icg = probe.proc(init['pid']), _status(probe, init['pid']), _cg(probe, init['pid'])
    if ist is None or ist['start_ticks'] != init['start_ticks'] or ips is None or icg != [c['cgroup']] \
            or ips['nspid'][-1] != 1 or len(ips['nspid']) < 2 or (ips['uid'], ips['gid']) != (init['uid'], init['gid']):
        return 'container init identity changed or gone (restart/replacement)'
    me, ms, mcg = probe.proc(pid), _status(probe, pid), _cg(probe, pid)
    if me is None or ms is None or mcg is None:
        return 'process identity unreadable'
    if mcg != [c['cgroup']]:
        return 'not in the pinned container cgroup'
    if (ms['uid'], ms['gid']) != (int(c['member_uid']), int(c['member_gid'])):
        return f'uid/gid {ms["uid"]}/{ms["gid"]} != pinned {c["member_uid"]}/{c["member_gid"]}'
    if len(ms['nspid']) < 2:
        return 'not in a nested pid namespace'
    if me['start_ticks'] < init['start_ticks']:
        return 'started before the container init'
    if pid == init['pid']:
        return None
    cur, cur_start = pid, me['start_ticks']
    for _ in range(MAX_HOPS):
        s = _status(probe, cur)
        par = s['ppid'] if s else 0
        if par == init['pid']:
            return None if cur_start >= init['start_ticks'] else 'lineage start order broken'
        pst, pss, pcg = probe.proc(par) if par > 1 else None, _status(probe, par) if par > 1 else None, _cg(probe, par) if par > 1 else None
        if pst is None or pss is None or pcg != [c['cgroup']] or len(pss['nspid']) < 2 or pst['start_ticks'] > cur_start:
            return 'lineage to the pinned init broken (left the container / reused PID / unreadable)'
        cur, cur_start = par, pst['start_ticks']
    return 'lineage deeper than MAX_HOPS'


def classify(auth, apps, probe):
    """{'pinned': [pid], 'containers': {id: [pid]}, 'violations': [str]} for the GPU's compute apps (v2 contract)."""
    pinned = {int(f['pid']): f for f in auth['foreign']}
    out = {'pinned': [], 'containers': {c['id']: [] for c in auth['containers']}, 'violations': []}
    used = {c['id']: 0 for c in auth['containers']}
    for app in apps:
        p = app['pid']
        f = pinned.get(p)
        if f is not None:
            live = probe.proc(p)
            bad = [k for k in ('uid', 'start_ticks', 'cmdline_sha256') if live is None or live[k] != f[k]]
            if bad:
                out['violations'].append(f'PID {p}: pinned identity {bad} differ (PID reuse / different process)')
            elif app['used_mib'] > int(f['max_used_mib']):
                out['violations'].append(f'approved PID {p} uses {app["used_mib"]} MiB > authorized {f["max_used_mib"]}')
            else:
                out['pinned'].append(p)
            continue
        reasons = []
        for c in auth['containers']:
            r = container_member(c, p, probe)
            if r is None:
                out['containers'][c['id']].append(p); used[c['id']] += app['used_mib']
                break
            reasons.append(f'{c["id"][:12]}: {r}')
        else:
            out['violations'].append(f'unapproved compute process PID {p} ({"; ".join(reasons) or "no approved container"})')
    for c in auth['containers']:
        if used[c['id']] > int(c['max_used_mib']):
            out['violations'].append(f'container {c["id"][:12]} uses {used[c["id"]]} MiB > authorized {c["max_used_mib"]}')
    return out


def authorized_processes_v2(auth, machine, gpus, session, kind, serves, probe):
    need(probe.now() < float(auth['expires_at']), 'sharing authorization expired')
    need(probe.hostname() == auth['hostname'], f'host {probe.hostname()} != authorized {auth["hostname"]}')
    need(machine == auth['machine'], f'machine {machine} != authorized {auth["machine"]}')
    need(list(gpus) == [int(auth['gpu'])], f'GPUs {list(gpus)} != authorized [{auth["gpu"]}] (single GPU only)')
    sc = session_scope(auth, session)
    need(kind == sc['allowed_kind'], f'kind {kind} != authorized {sc["allowed_kind"]}')
    need(1 <= int(serves) <= int(sc['max_serves']), f'serves {serves} exceeds authorized {sc["max_serves"]} for {session}')
    need(probe.boot_id() == auth['boot_id'], 'boot id changed since authorization (reboot): contract void')
    g = probe.gpu(int(auth['gpu']))
    need(g['uuid'] == auth['gpu_uuid'], f'GPU{auth["gpu"]} uuid {g["uuid"]} != authorized {auth["gpu_uuid"]}')
    cl = classify(auth, g['apps'], probe)
    need(not cl['violations'], f'{len(cl["violations"])} violation(s): ' + ' | '.join(cl['violations'][:5]))
    required = int(serves) * int(sc['per_serve_mib']) + int(auth['reserve_mib'])
    need(g['free_mib'] >= required, f'GPU{auth["gpu"]} free {g["free_mib"]} MiB < required {required} MiB '
                                    f'({serves} x {sc["per_serve_mib"]} + {auth["reserve_mib"]})')
    return {'gpu': auth['gpu'], 'uuid': g['uuid'], 'approved_present': cl['pinned'], 'container_members': cl['containers'],
            'free_mib': g['free_mib'], 'required_mib': required, 'session': session}


def approved_now(auth, apps, probe):
    """Guardian helper (v2 contract): set of pids currently approved (pinned or verified container members), with the
    whole container excluded if its aggregate memory is over its limit. Expired / other boot -> empty set."""
    if probe.now() >= float(auth['expires_at']) or probe.boot_id() != auth['boot_id']:
        return set(), ['contract expired or boot changed']
    cl = classify(auth, apps, probe)
    over = {c['id'] for c in auth['containers'] if any(v.startswith(f'container {c["id"][:12]} uses') for v in cl['violations'])}
    ok = set(cl['pinned']) | {p for cid, ps in cl['containers'].items() if cid not in over for p in ps}
    return ok, cl['violations']


def prepare_v2(a, probe=None):
    """Write a v2 contract from LIVE state + main-supplied docker inspect of the approved container."""
    probe = probe or Probe()
    pol = json.loads(POLICY.read_text())['machines']
    need(a.machine in pol and probe.hostname() == pol[a.machine]['hostname'], 'prepare must run on the authorized host itself')
    g = probe.gpu(a.gpu)
    need(g['uuid'] == a.gpu_uuid, f'GPU{a.gpu} uuid {g["uuid"]} != expected {a.gpu_uuid}')
    sessions = json.loads(a.sessions)
    foreign = []
    if a.pinned_from:
        old = load(a.pinned_from)
        need(old['gpu_uuid'] == a.gpu_uuid and old['boot_id'] == probe.boot_id(), 'pinned-from contract is for another GPU/boot')
        foreign = old['foreign']
    insp = json.loads(Path(a.container_inspect).read_text())
    insp = insp[0] if isinstance(insp, list) else insp
    cid = insp['Id']
    need(cid == a.expect_container_id, f'inspect Id {cid} != expected {a.expect_container_id}')
    need(insp['Image'] == a.expect_image, f'inspect Image {insp["Image"]} != expected {a.expect_image}')
    need(insp['State']['Running'] is True, 'container not running')
    reqs = insp['HostConfig'].get('DeviceRequests') or []
    devs = [d for r in reqs for d in (r.get('DeviceIDs') or [])]
    need(devs == [a.gpu_uuid] and all((r.get('Count') or 0) != -1 for r in reqs), f'container GPU devices {devs} != exactly [{a.gpu_uuid}]')
    ipid = int(insp['State']['Pid'])
    ist, ips, icg = probe.proc(ipid), _status(probe, ipid), _cg(probe, ipid)
    need(ist is not None and ips is not None, f'container init PID {ipid} not readable')
    need(icg == [f'0::/docker/{cid}'], f'init cgroup {icg} != exactly 0::/docker/<id>')
    need(len(ips['nspid']) >= 2 and ips['nspid'][-1] == 1, f'init NSpid {ips["nspid"]} is not PID 1 of a nested namespace')
    need(f'{a.expect_member_uid}:{a.expect_member_gid}' == str(insp['Config'].get('User', '')),
         f'Config.User {insp["Config"].get("User")!r} != {a.expect_member_uid}:{a.expect_member_gid}')
    now = probe.now()
    body = {'kind': KIND_V2, 'machine': a.machine, 'hostname': probe.hostname(), 'gpu': a.gpu, 'gpu_uuid': g['uuid'],
            'experiment': a.experiment, 'sessions': sessions, 'reserve_mib': a.reserve_mib, 'expires_at': now + a.expires_in_h * 3600,
            'boot_id': probe.boot_id(), 'prepared_at': now, 'approval': a.approval, 'foreign': foreign,
            'containers': [{'id': cid, 'name': insp.get('Name'), 'image_id': insp['Image'], 'started_at': insp['State']['StartedAt'],
                            'cgroup': f'0::/docker/{cid}', 'devices': [a.gpu_uuid], 'member_uid': a.expect_member_uid,
                            'member_gid': a.expect_member_gid, 'max_used_mib': a.container_max_used_mib,
                            'init': {'pid': ipid, 'start_ticks': ist['start_ticks'], 'uid': ips['uid'], 'gid': ips['gid']},
                            'launcher_evidence': a.launcher_evidence}],
            'observed': {'free_mib': g['free_mib'], 'total_mib': g['total_mib'], 'apps': g['apps']}}
    load_v2_body(json.loads(json.dumps(body)))
    cl = classify(body, g['apps'], probe)
    need(not cl['violations'], f'live GPU has unapproved processes now: {cl["violations"][:3]}')
    with open(a.out, 'x') as f:
        f.write(json.dumps(body, indent=1))
    return body


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
    p2 = sub.add_parser('prepare-v2', help='pinned processes (from a v1 contract) + one approved container instance + sessions')
    for k, t in (('machine', str), ('gpu', int), ('gpu-uuid', str), ('experiment', str), ('sessions', str),
                 ('container-inspect', Path), ('expect-container-id', str), ('expect-image', str), ('expect-member-uid', int),
                 ('expect-member-gid', int), ('container-max-used-mib', int), ('approval', str)):
        p2.add_argument('--' + k, type=t, required=True)
    p2.add_argument('--pinned-from', type=Path); p2.add_argument('--launcher-evidence', default='')
    p2.add_argument('--reserve-mib', type=int, default=8192); p2.add_argument('--expires-in-h', type=float, required=True)
    p2.add_argument('--out', type=Path, required=True)
    c = sub.add_parser('check')
    c.add_argument('--auth', type=Path, required=True); c.add_argument('--machine', required=True); c.add_argument('--gpus', required=True)
    c.add_argument('--session', required=True); c.add_argument('--kind', default='eval'); c.add_argument('--serves', type=int, required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == 'prepare':
            need(a.expires_in_h <= 72, 'expiry must be <= 72 h (one experiment)')
            r = prepare(a)
        elif a.cmd == 'prepare-v2':
            need(a.expires_in_h <= 72, 'expiry must be <= 72 h (one experiment)')
            r = prepare_v2(a)
        else:
            r = authorized_processes(load(a.auth), a.machine, [int(g) for g in a.gpus.split(',')], a.session, a.kind, a.serves)
    except AuthError as e:
        print(f'[sharing] REFUSED: {e}', file=sys.stderr)
        return 2
    print(json.dumps(r, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""GR00T launch admission. Stdlib only; one coordinator, shared worktree ledger.

Reservations deliberately do not expire: a dead launcher can leave live servers.
Release requires the session AND receipt. Never infer completion from elapsed time.
"""
import argparse
import contextlib
import fcntl
import hashlib
import math
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
import uuid

REPO = Path(__file__).resolve().parents[2]


def common_repo():
    common = subprocess.check_output(
        ['git', '-C', str(REPO), 'rev-parse', '--path-format=absolute', '--git-common-dir'],
        text=True).strip()
    return Path(common).parent


def policy():
    # Always read the common checkout, not a stale worktree copy.
    return json.loads((common_repo() / 'configs/harness/gpu_policy.json').read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def resources(machine, gpus, serves):
    rules = policy()['machines']
    require(machine in rules, f'unknown machine: {machine}')
    require(gpus and all(type(g) is int and g >= 0 for g in gpus)
            and len(set(gpus)) == len(gpus), 'GPU ids must be unique nonnegative integers')
    require(type(serves) is int and 1 <= serves <= rules[machine]['max_serves_per_gpu'],
            f'{machine}: serves/GPU limit {rules[machine]["max_serves_per_gpu"]}')
    require(len(gpus) <= rules[machine]['max_gpus_all_sessions'],
            f'{machine}: all-session GPU limit {rules[machine]["max_gpus_all_sessions"]}')


def read_meta(path):
    return dict(line.split('=', 1) for line in path.read_text().splitlines() if '=' in line)


def legacy_live(meta):
    # Only coordinator-local PIDs are meaningful; unknown metadata fails closed.
    pid = meta.get('pid', 'none')
    if pid != 'none':
        try:
            os.kill(int(pid), 0)
            return True
        except ProcessLookupError:
            return False
        except (PermissionError, ValueError):
            return True
    try:
        return time.time() - int(meta['start']) <= int(meta['ttl_s'])
    except (KeyError, ValueError):
        return True


@contextlib.contextmanager
def ledger(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / 'harness_slots.json'
        state = json.loads(path.read_text()) if path.exists() else {}
        yield state
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        os.replace(tmp, path)


def transact(root, request):
    action = request['action']
    with ledger(root) as state:
        if action == 'status':
            return state
        if action in ('release', 'verify'):
            item = state.get(request['receipt'])
            require(item is not None, 'unknown reservation receipt')
            require(item['session'] == request['session'], 'reservation owner mismatch')
            if action == 'verify':
                for k in ('machine', 'gpus', 'serves', 'kind'):
                    require(item[k] == request[k], f'reservation {k} mismatch')
                consumer = request.get('consumer_pid')
                require(type(consumer) is int and consumer > 0, 'consumer PID required')
                require('consumer_pid' not in item or item['consumer_pid'] == consumer,
                        'receipt already consumed by another runner')
                item['consumer_pid'] = consumer
                return item
            del state[request['receipt']]
            return {'released': request['receipt']}
        require(action == 'reserve', 'unknown action')
        machine, gpus, serves = request['machine'], request['gpus'], request['serves']
        resources(machine, gpus, serves)
        require(request.get('session') and request.get('kind') in ('collect', 'eval'),
                'session and kind (collect/eval) required')
        occupied = set()
        relevant = [x for x in state.values() if x['machine'] == machine]
        for item in relevant:
            occupied.update(item['gpus'])
            if set(gpus) & set(item['gpus']):
                require(item['session'] == request['session'], 'GPU owned by another session')
                require(item['kind'] == request['kind'], 'GPU reserved for a different task kind')
        # Legacy reservations remain authoritative; do not delete or migrate them here.
        for path in root.glob(f'{machine}_gpu*/meta'):
            meta = read_meta(path)
            if not legacy_live(meta):
                continue
            g = int(path.parent.name.split('_gpu')[1])
            if meta.get('model', 'groot') == 'groot':
                occupied.add(g)
            if g in gpus:
                require(meta.get('session') == request['session'], 'legacy lease owned by another session')
                # A legacy lease alone cannot establish how many servers already exist.
                # The launch-side process probe also checks for untracked compute PIDs.
        cap = policy()['machines'][machine]
        require(len(occupied | set(gpus)) <= cap['max_gpus_all_sessions'],
                f'{machine}: aggregate GPU limit exceeded across sessions')
        for g in gpus:
            used = sum(x['serves'] for x in relevant if g in x['gpus'])
            require(used + serves <= cap['max_serves_per_gpu'],
                    f'{machine} GPU{g}: {used}+{serves} exceeds slot limit')
        receipt = uuid.uuid4().hex
        state[receipt] = {k: request[k] for k in ('machine', 'gpus', 'serves', 'session', 'kind')}
        state[receipt].update(created=time.time(), launcher_host=request.get('launcher_host'),
                              launcher_pid=request.get('launcher_pid'))
        return {'receipt': receipt}


def legacy_check(root, machine, gpu, session, model):
    require(machine in policy()['machines'] and gpu >= 0, 'invalid lease machine/GPU')
    path = root / 'harness_slots.json'
    state = json.loads(path.read_text()) if path.exists() else {}
    occupied = set()
    for item in state.values():
        if item['machine'] != machine:
            continue
        occupied.update(item['gpus'])
        require(gpu not in item['gpus'], 'GPU has active harness reservations; do not acquire/release its legacy lease')
    for path in root.glob(f'{machine}_gpu*/meta'):
        meta = read_meta(path)
        if legacy_live(meta) and meta.get('model', 'groot') == 'groot':
            occupied.add(int(path.parent.name.split('_gpu')[1]))
    if model == 'groot':
        require(len(occupied | {gpu}) <= policy()['machines'][machine]['max_gpus_all_sessions'],
                f'{machine}: all-session GPU limit exceeded')
    return {'lease_admissible': True}


def broker(request):
    if socket.gethostname().split('.')[0] == policy()['coordinator_host']:
        return transact(common_repo() / 'outputs/gpu_leases', request)
    target = os.environ.get('HARNESS_COORDINATOR_SSH')
    script = os.environ.get('HARNESS_COORDINATOR_SCRIPT')
    require(target and script and script.startswith('/'),
            'remote launch requires HARNESS_COORDINATOR_SSH and absolute HARNESS_COORDINATOR_SCRIPT; no local-ledger fallback')
    cmd = shlex.join(['python3', script, 'broker'])
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', target, cmd],
                            input=json.dumps(request), text=True, capture_output=True)
    require(result.returncode == 0, f'coordinator rejected/unreachable: {result.stderr.strip()}')
    return json.loads(result.stdout)


def collection_check(contract_path, plan_path, instructions, noise_limit, done_list, grid_root):
    contract = json.loads(Path(contract_path).read_text())
    require(contract.get('kind') == 'collect' and contract.get('model') == 'groot', 'collection contract required')
    parent_path = Path(contract['parent_plan'])
    if not parent_path.is_absolute():
        parent_path = Path(contract_path).resolve().parent / parent_path
    require(hashlib.sha256(parent_path.read_bytes()).hexdigest() == contract['parent_plan_sha256'],
            'parent plan digest mismatch')
    output = Path(grid_root).resolve()
    parent_grid = Path(contract['parent_grid_root']).resolve()
    require(output != parent_grid and parent_grid not in output.parents,
            'additional collection must use a separate grid root')
    old = json.loads(parent_path.read_text())
    new = json.loads(Path(plan_path).read_text())
    require(new['model'] == 'groot', 'GR00T policy only')
    for field in ('model', 'version', 'ckpt', 'capture_layers', 'denoise_k', 'token_mode', 'noise_seeds'):
        require(old[field] == new[field], f'collection contract changed: {field}')
    for field in ('env_kwargs', 'env_names', 'instruction_text'):
        require(old['extra'].get(field) == new['extra'].get(field), f'collection environment changed: {field}')
    require(set(old['instructions']) == set(new['instructions']), 'instruction keys changed')
    wanted = set(instructions.split(','))
    require(wanted and wanted <= set(contract['allowed_instructions']), 'instruction outside allowed scope')
    require(not wanted & set(contract.get('excluded_instructions', [])), 'excluded instruction requested')
    require(noise_limit == contract['noise_count'] == 10 and len(new['noise_seeds']) == 10,
            'expected exactly ten policy noise seeds')
    require(len(set(new['noise_seeds'])) == 10, 'duplicate policy noise seed')
    done = set(Path(done_list).read_text().splitlines())
    new_count = 0
    for key, seeds in old['instructions'].items():
        n = len(seeds)
        for field in ('instructions', 'scenes', 'jitters'):
            require(new[field][key][:n] == old[field][key], f'existing coordinates changed: {field}/{key}')
        if key in wanted:
            expected_old = {f'{key}|s{sid}|j{j}|n{noise}' for sid in range(n)
                            for j in range(len(old['jitters'][key][sid])) for noise in range(10)}
            require(expected_old <= done, f'DONE_LIST missing {len(expected_old - done)} parent cells: {key}')
        count = len(new['instructions'][key])
        require(count == len(new['scenes'][key]) == len(new['jitters'][key]), 'scene/seed/jitter lengths differ')
        for sid in range(n, count):
            scene, jitters = new['scenes'][key][sid], new['jitters'][key][sid]
            require('layout' in scene and 'style' in scene, 'new scene missing layout/style')
            require(len(jitters) == contract['jitter_count'] == 5, 'expected five jitters per new scene')
            require(len({json.dumps(j, sort_keys=True) for j in jitters}) == 5, 'duplicate jitter definition')
            if key.startswith('OpenDrawer/'):
                require(len({j.get('reset_idx') for j in jitters}) == 5, 'drawer requires five distinct reset indices')
            for jitter in jitters:
                require(all(f in jitter for f in ('reset_idx', 'lat', 'back')), 'missing jitter replay coordinates')
                require(type(jitter['reset_idx']) is int and jitter['reset_idx'] >= 0,
                        'invalid jitter reset index')
                require(all(isinstance(jitter[f], (int, float)) and math.isfinite(jitter[f]) for f in ('lat', 'back')),
                        'non-finite jitter offset')
            if key in wanted:
                new_count += 1
    expected_plan = contract.get('plan_sha256')
    require(expected_plan and hashlib.sha256(Path(plan_path).read_bytes()).hexdigest() == expected_plan,
            'collection plan differs from frozen contract')
    require(contract['machine'] in policy()['machines'], 'collection machine required')
    require(new_count > 0, 'no additional scenes in selected scope')
    return {'new_scenes': new_count, 'new_episodes': new_count * 50}


def request_for(args, action):
    return dict(action=action, machine=args.machine, gpus=[int(g) for g in args.gpus.split(',')],
                serves=args.serves, session=args.session, kind=args.kind,
                launcher_host=socket.gethostname(), launcher_pid=os.getpid())


def check_gpu_processes(gpus):
    # Fail closed on NVML/query errors. Unknown existing processes must not be silently adopted.
    for g in gpus:
        result = subprocess.run(['nvidia-smi', f'--id={g}', '--query-compute-apps=pid',
                                 '--format=csv,noheader,nounits'], text=True, capture_output=True)
        require(result.returncode == 0, f'GPU{g}: nvidia-smi failed: {result.stderr.strip()}')
        require(not result.stdout.strip(), f'GPU{g}: existing compute processes; use one grouped launch, do not adopt untracked servers')


def run(args):
    req = request_for(args, 'reserve')
    resources(req['machine'], req['gpus'], req['serves'])
    require(socket.gethostname().split('.')[0] == policy()['machines'][args.machine]['hostname'],
            'machine does not match actual launcher host')
    command = args.command
    if command and command[0] == '--':
        command = command[1:]
    require(command, 'command required')
    check_gpu_processes(req['gpus'])
    receipt = broker(req)['receipt']
    child = None
    try:
        # Check again after admission to reduce races with launches outside this harness.
        check_gpu_processes(req['gpus'])
        env = dict(os.environ, HARNESS_RECEIPT=receipt, HARNESS_MACHINE=args.machine,
                   HARNESS_SESSION=args.session, GPUS=args.gpus, SERVES_PER_GPU=str(args.serves))
        child = subprocess.Popen(command, env=env, start_new_session=True)
        def forward(sig, _frame):
            if child.poll() is None:
                os.killpg(child.pid, sig)
        signal.signal(signal.SIGTERM, forward)
        signal.signal(signal.SIGINT, forward)
        rc = child.wait()
    finally:
        if child is None or child.poll() is not None:
            # A runner may exit while detached model servers remain. Keep the receipt then.
            try:
                check_gpu_processes(req['gpus'])
            except (ValueError, FileNotFoundError) as exc:
                print(f'[harness] reservation retained {receipt}: {exc}', file=sys.stderr)
            else:
                broker(dict(action='release', receipt=receipt, session=args.session))
    return rc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='cmd', required=True)
    sub.add_parser('broker')
    sub.add_parser('status')
    legacy = sub.add_parser('check-legacy-lease')
    legacy.add_argument('--machine', required=True)
    legacy.add_argument('--gpu', type=int, required=True)
    legacy.add_argument('--session', required=True)
    legacy.add_argument('--model', default='groot')
    release = sub.add_parser('release')
    release.add_argument('--receipt', required=True)
    release.add_argument('--session', required=True)
    collect = sub.add_parser('check-collection')
    for name in ('contract', 'plan', 'instructions', 'done-list', 'grid-root', 'machine'):
        collect.add_argument('--' + name, required=True)
    collect.add_argument('--noise-limit', type=int, default=10)
    for name in ('run', 'verify', 'check-resources'):
        p = sub.add_parser(name)
        p.add_argument('--machine', required=True)
        p.add_argument('--gpus', required=True)
        p.add_argument('--serves', required=True, type=int)
        p.add_argument('--session', required=True)
        p.add_argument('--kind', choices=('eval', 'collect'), required=True)
        if name == 'run':
            p.add_argument('command', nargs=argparse.REMAINDER)
        elif name == 'verify':
            p.add_argument('--receipt', required=True)
            p.add_argument('--consumer-pid', required=True, type=int)
    args = parser.parse_args()
    if args.cmd == 'broker':
        require(socket.gethostname().split('.')[0] == policy()['coordinator_host'], 'broker must run on coordinator')
        result = transact(common_repo() / 'outputs/gpu_leases', json.load(sys.stdin))
    elif args.cmd == 'check-legacy-lease':
        require(socket.gethostname().split('.')[0] == policy()['coordinator_host'], 'legacy leases must run on coordinator')
        result = legacy_check(common_repo() / 'outputs/gpu_leases', args.machine, args.gpu, args.session, args.model)
    elif args.cmd == 'status':
        result = broker({'action': 'status'})
    elif args.cmd == 'release':
        result = broker(dict(action='release', receipt=args.receipt, session=args.session))
    elif args.cmd == 'check-collection':
        result = collection_check(args.contract, args.plan, args.instructions, args.noise_limit, args.done_list, args.grid_root)
        require(json.loads(Path(args.contract).read_text())['machine'] == args.machine, 'collection machine mismatch')
    elif args.cmd == 'run':
        return run(args)
    elif args.cmd == 'verify':
        req = request_for(args, 'verify')
        req['receipt'] = args.receipt
        req['consumer_pid'] = args.consumer_pid
        result = broker(req)
    else:
        req = request_for(args, 'reserve')
        resources(req['machine'], req['gpus'], req['serves'])
        result = {'resource_shape_valid': True, 'aggregate_reservation_checked': False}
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as exc:
        print(f'[harness] BLOCKED: {exc}', file=sys.stderr)
        sys.exit(2)

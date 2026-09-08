#!/usr/bin/env python3
"""Two same-session lanes on kanu; exact resume and target replay gates retained.

One GPU lease covers both lanes. Batches start from an empty GPU, load servers
in sequence, and allow only the first lane's exact process contract when starting
the second. Existing runner artifacts and completed coordinate sets are reused.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

try:
    from .eval import build_jobs, completed, check_idle, check_port_available
    from .remote_stage import verify_inputs
except ImportError:
    from eval import build_jobs, completed, check_idle, check_port_available
    from remote_stage import verify_inputs


def target(task):
    return tuple(task['env'][k] for k in ('SLUGS', 'EVAL_SCENES', 'EVAL_JITTERS'))


def next_batch(pending, passed, lanes):
    return [j for j in pending if j['gate'] or target(j) in passed][:lanes]


def select_target(jobs, selected):
    if selected is None:
        return jobs
    chosen = [j for j in jobs if target(j) == tuple(selected)]
    if not chosen or not any(j['gate'] for j in chosen):
        raise ValueError('target shard must contain its replay gate')
    return chosen


def option(argv, name):
    try:
        return argv[argv.index(name) + 1]
    except (ValueError, IndexError):
        return None


def owns_gpu_process(argv, task, repo):
    """Exact serving/collector contract; same UID alone is never sufficient."""
    port = task['env']['PORT_BASE']
    detector = task['env']['DETECTOR_CKPT']
    inside = '/temporal_vla/' + str(Path(detector).relative_to(repo))
    serve = any(a.endswith('scripts/serve/lerobot.py') for a in argv)
    if serve:
        return (option(argv, '--port') == port and
                option(argv, '--failure-detector') in (detector, inside) and
                option(argv, '--failure-task') == task['pair_id'])
    collector = any(a.endswith('collect/http_feature_collect.py') for a in argv)
    raw = str(Path(task['result']).parent / 'raw_rollouts')
    raw_inside = '/temporal_vla/' + str(Path(raw).relative_to(repo))
    return (collector and option(argv, '--vla-server') == f'http://127.0.0.1:{port}' and
            option(argv, '--output-dir') in (raw, raw_inside))


def check_owned_gpu(gpu, tasks, repo):
    result = subprocess.run(['nvidia-smi', f'--id={gpu}', '--query-compute-apps=pid',
                             '--format=csv,noheader'], check=True, capture_output=True, text=True)
    for value in result.stdout.splitlines():
        if not value.strip():
            continue
        proc = Path('/proc') / str(int(value.strip()))
        try:
            uid = proc.stat().st_uid
            argv = proc.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')
        except FileNotFoundError:
            continue  # exited after NVML snapshot
        if uid != os.getuid() or not any(owns_gpu_process(argv, t, repo) for t in tasks):
            raise RuntimeError(f'GPU {gpu}: unowned process {value.strip()}; refusing shared launch')


def check_port(port):
    check_port_available(port)
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            argv = p.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(a.endswith('scripts/serve/lerobot.py') for a in argv) and option(argv, '--port') == str(port):
            raise RuntimeError(f'port {port}: preexisting serve PID {p.name}')


def prepare(task):
    if task['condition'] not in ('plain', 'lr_jfair'):
        return
    source, link = Path(task['operator_source']), Path(task['operator_link'])
    if not list(source.rglob('*.npz')):
        raise RuntimeError(f'no registered operator NPZ: {source}')
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() != source.resolve():
            raise RuntimeError(f'wrong operator link: {link}')
    elif link.exists():
        raise RuntimeError(f'refuse replacing path: {link}')
    else:
        link.symlink_to(os.path.relpath(source, link.parent), target_is_directory=True)


def wait_loaded(proc, port):
    """Avoid simultaneous model-load peaks; inference may run while lane 2 loads."""
    deadline = time.monotonic() + 780
    while time.monotonic() < deadline:
        code = proc.poll()
        if code is not None:
            if code:
                raise RuntimeError(f'first lane failed before second launch: exit {code}')
            return
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as response:
                if json.load(response).get('status') == 'ok':
                    return
        except (OSError, ValueError):
            pass
        time.sleep(2)
    raise RuntimeError('first lane model startup timed out')


def run_batch(batch, repo, gpu, ports, assert_lease):
    active = []
    try:
        assert_lease()
        check_idle(gpu, ports[0])
        for port in ports:
            check_port(port)
        for i, source in enumerate(batch):
            task = {**source, 'env': {**source['env'], 'PORT_BASE': str(ports[i])}}
            if active:
                wait_loaded(active[0][0], ports[0])
            assert_lease()
            check_owned_gpu(gpu, [a[1] for a in active], repo)
            check_port(ports[i])
            prepare(task)
            # Only this single lease holder can set the bypass after the exact
            # allowlist above. Never expose it as a general share-GPU option.
            task['env']['ALLOW_BUSY_GPU'] = '1' if active else '0'
            log = Path(task['log'])
            log.parent.mkdir(parents=True, exist_ok=True)
            stream = log.open('a')
            try:
                proc = subprocess.Popen(task['command'], cwd=repo, env={**os.environ, **task['env']},
                                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            except BaseException:
                stream.close()
                raise
            active.append((proc, task, stream))
            print(f"START port={ports[i]} pid={proc.pid} {task['pair_id']}/{task['model_arm']}/{task['condition']}", flush=True)
        # Finish the peer's current condition even if one lane fails. No new
        # batch is launched until both complete and all exact audits pass.
        for proc, task, stream in active:
            proc.wait()
        for proc, task, stream in active:
            if proc.returncode:
                raise RuntimeError(f"runner exit {proc.returncode}: {task['log']}")
            if not completed(task['result'], task['expected'], require_failure=task['gate']):
                raise RuntimeError(f"missing completed results: {task['result']}")
            print(f"DONE {task['pair_id']}/{task['model_arm']}/{task['condition']}", flush=True)
        for port in ports:
            check_idle(gpu, port)
    finally:
        for proc, _, _ in active:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
        for proc, _, stream in active:
            try:
                proc.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
            finally:
                stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--ports', type=int, nargs=2, default=[8866, 8867])
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--target', nargs=3, metavar=('SLUG', 'SCENE', 'JITTER'))
    args = parser.parse_args()
    if args.gpu not in range(8) or len(set(args.ports)) != 2 or any(p < 8860 or p > 65535 for p in args.ports):
        parser.error('kanu GPU 0..7 and two distinct ports >=8860 required')
    args.root, args.repo = args.root.resolve(), args.repo.resolve()
    verify_inputs(args.root)
    jobs, coverage = build_jobs(args.root, args.repo, 'kanu', args.gpu, port=args.ports[0])
    all_jobs = jobs
    jobs = select_target(jobs, args.target)
    if not jobs or any(c['status'] == 'awaiting_fit' for c in coverage):
        raise RuntimeError('no jobs or missing fits')
    passed, pending = set(), []
    for task in jobs:
        for required in (task['env']['DETECTOR_CKPT'], task['fit_report'], task['pair_report'],
                         task['env']['PLAN_JSON'], task['env']['INDEX_TSV']):
            if not Path(required).is_file():
                raise FileNotFoundError(required)
        if json.loads(Path(task['pair_report']).read_text())['status'] != 'complete':
            raise RuntimeError('fit pair not complete')
        if completed(task['result'], task['expected'], require_failure=task['gate']):
            if task['gate']:
                passed.add(target(task))
        else:
            pending.append(task)
    global_output = args.root / 'eval/kanu'
    output = global_output if args.target is None else global_output / 'shards' / ('__'.join(args.target))
    output.mkdir(parents=True, exist_ok=True)
    schedule = dict(jobs=jobs, operator_coverage=coverage, ports=args.ports, lanes=2,
                    n_jobs=len(jobs), n_pending=len(pending), n_episode_runs=sum(len(j['expected']) for j in jobs))
    (output / ('parallel_dry_run.json' if args.dry_run else 'parallel_schedule.json')).write_text(json.dumps(schedule, indent=2)+'\n')
    print(json.dumps({k: v for k, v in schedule.items() if k not in ('jobs', 'operator_coverage')}), flush=True)
    if args.dry_run:
        return
    def interrupted(signum, frame):
        raise InterruptedError(f'parallel eval interrupted: {signum}')
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    lease = args.repo / 'scripts/utils/gpu_lease.sh'
    meta_path = args.repo / f'outputs/gpu_leases/kanu_gpu{args.gpu}/meta'
    session = f'lr-equal-parallel-{os.getpid()}'
    def assert_lease():
        meta = dict(line.split('=', 1) for line in meta_path.read_text().splitlines())
        if meta.get('session') != session or meta.get('pid') != str(os.getpid()):
            raise RuntimeError('GPU lease ownership changed')
    subprocess.run(['bash', str(lease), 'claim', 'kanu', str(args.gpu), session, 'lr_equal_budget_two_lanes'],
                   env={**os.environ, 'LEASE_PID': str(os.getpid())}, check=True)
    try:
        while pending:
            batch = next_batch(pending, passed, 2)
            if not batch:
                raise RuntimeError('unresolved target replay gates')
            run_batch(batch, args.repo, args.gpu, args.ports, assert_lease)
            for task in batch:
                pending.remove(task)
                if task['gate']:
                    passed.add(target(task))
        for task in jobs:
            if not completed(task['result'], task['expected'], require_failure=task['gate']):
                raise RuntimeError('final exact completion audit failed')
        (output / 'EVAL_DONE').write_text('EVAL_DONE_kanu\n')
        if args.target is not None:
            # Another shard may still be running. Only the last fully audited
            # shard writes the global completion sentinel.
            if all(completed(j['result'], j['expected'], require_failure=j['gate']) for j in all_jobs):
                (global_output / 'EVAL_DONE').write_text('EVAL_DONE_kanu\n')
    finally:
        subprocess.run(['bash', str(lease), 'release', 'kanu', str(args.gpu), session], check=True)


if __name__ == '__main__':
    main()

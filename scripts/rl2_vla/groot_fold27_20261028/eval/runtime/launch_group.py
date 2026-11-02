"""Grouped launch on one host: (manifest, arm) jobs over GPUS x SERVES_PER_GPU slots (one serve per slot at a time).

Entered only through launch.sh (groot_launch_guard eval -> harness receipt). Caps come from configs/harness/gpu_policy.json
(kanu <=3 GPU x 2 serve, srv48/srv50 <=1 GPU x 6 serve, all sessions; the harness ledger enforces the all-session sum).

    launch.sh --jobs <jobs.txt> [--port0 8860] [--rows 0]      # jobs.txt: "<manifest.json> <arm>" per line
"""
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402


def read_jobs(path):
    jobs = []
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            mp, arm = line.split()
            jobs.append((str(Path(mp).resolve()), arm))
    C.need(jobs and len(set(jobs)) == len(jobs), 'empty or duplicate jobs')
    return jobs


def plan_slots(host, gpus, serves, port0):
    max_g, max_s = C.caps(host)
    C.need(1 <= len(gpus) <= max_g, f'{host}: {len(gpus)} GPUs > cap {max_g}')
    C.need(1 <= serves <= max_s, f'{host}: {serves} serves/GPU > cap {max_s}')
    return [(g, port0 + i * serves + k) for i, g in enumerate(gpus) for k in range(serves)]


def check_jobs(jobs, host):
    smoke = set()
    for mp, arm in jobs:
        m = json.loads(Path(mp).read_text())
        C.need(m['host'] == host, f'{mp}: fold host {m["host"]} != launch host {host} (eval only on the archive machine)')
        C.need(arm in m['arms'], f'{mp}: arm {arm} absent')
        smoke.add(bool(m['smoke']))
    C.need(len(smoke) == 1, 'do not mix smoke and production jobs in one launch')
    return smoke.pop()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=Path, required=True); ap.add_argument('--port0', type=int)
    ap.add_argument('--rows', help='smoke only, comma list'); ap.add_argument('--serve-py', default=C.SRV_SERVE_PY)
    a = ap.parse_args(argv)
    host = os.environ['HARNESS_MACHINE']
    C.check_host(host)
    C.need(os.environ.get('HARNESS_RECEIPT'), 'HARNESS_RECEIPT missing')
    receipt = os.environ['HARNESS_RECEIPT']
    gpus = [int(g) for g in os.environ['GPUS'].split(',')]
    serves = int(os.environ['SERVES_PER_GPU'])
    subprocess.run(['python3', str(C.R / 'scripts/utils/groot_harness.py'), 'verify', '--machine', host, '--gpus', os.environ['GPUS'],
                    '--serves', str(serves), '--session', os.environ['HARNESS_SESSION'], '--kind', 'eval', '--receipt', receipt,
                    '--consumer-pid', str(os.getpid())], check=True)
    jobs = read_jobs(a.jobs)
    smoke = check_jobs(jobs, host)
    C.need(a.rows is None or smoke, '--rows only with smoke jobs')
    slots = plan_slots(host, gpus, serves, a.port0 or C.HOSTS[host]['port0'])
    log_dir = C.R / (C.SMOKE_ROOT if smoke else C.EVAL_ROOT) / 'launch_logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    q = queue.Queue()
    for j in jobs:
        q.put(j)
    results, lock = [], threading.Lock()
    boot = {g: threading.Lock() for g in gpus}     # one serve booting per GPU at a time (load peak ~2x steady VRAM)

    def worker(gpu, port):
        while True:
            try:
                mp, arm = q.get_nowait()
            except queue.Empty:
                return
            fold = Path(mp).parent.name
            flag = log_dir / f'{stamp}.{fold}.{arm}.g{gpu}p{port}.ready'
            cmd = [sys.executable, '-u', str(HERE / 'run_eval_arm.py'), '--manifest', mp, '--arm', arm, '--gpu', str(gpu),
                   '--port', str(port), '--serve-py', a.serve_py, '--ready-flag', str(flag)] + (['--rows', a.rows] if a.rows else [])
            with (log_dir / f'{stamp}.{fold}.{arm}.g{gpu}p{port}.log').open('x') as f:
                with boot[gpu]:
                    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
                    while p.poll() is None and not flag.exists():
                        time.sleep(5)
                rc = p.wait()
            with lock:
                results.append({'manifest': mp, 'arm': arm, 'gpu': gpu, 'port': port, 'rc': rc})
    threads = [threading.Thread(target=worker, args=s) for s in slots]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    (log_dir / f'{stamp}.group_exit.json').write_text(json.dumps({'host': host, 'slots': slots, 'results': results}, indent=1))
    return int(any(r['rc'] for r in results))


if __name__ == '__main__':
    try:
        sys.exit(main())
    except C.RuntimeContractError as e:
        print('LAUNCH_REFUSED', e, file=sys.stderr); sys.exit(2)

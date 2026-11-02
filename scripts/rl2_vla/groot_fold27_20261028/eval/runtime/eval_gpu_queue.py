"""Per-GPU eval consumer (new version of eval_pipeline.queue; the frozen queue is unchanged and must not run concurrently).

One harness receipt per GPU (GPUS=<g>, SERVES_PER_GPU slots). Several consumers (one per GPU) share work safely:
  * job ownership = job_claims.ClaimStore at <root>/queue/<host>/claims (O_EXCL + flock, pid+starttime liveness, stale ->
    stale/, finished -> done/); fold verification is claimed too ((fold, '__verify__')), so verify_fold runs once.
  * ports are a function of the GPU id, not of slot order: port0 + 100 + g*serves + k (disjoint from the frozen queue's
    port0 + i*serves + k, and two GPUs can never share a port); the existing bind probe (wait_port_free) still applies.
  * failures use the frozen marker dir <root>/queue/<host>/failed, so Pipeline.jobs() excludes them for every consumer.
  * fair share: if the train coordinator has an active TRAIN_REQUEST.json and live non-yielding eval consumers exceed
    job_claims.eval_share(...), this consumer stops taking new jobs after >= --min-jobs completed arms (never interrupts a
    running arm/episode), lets its running arms finish, exits -> the harness releases the whole GPU (no partial receipt).
  * the job itself is run_eval_arm_v2.py (identical to run_eval_arm.py on kanu; explicit isolated env container on srv).
Rows/seeds/manifests/actors are untouched; run_eval_arm resumes finished rows by hash, so a re-run after a crash only does
missing rows.

    may-start --host H --gpu G --pool 5,6,7 : exit 0 if eval may (re)take GPU G now, 3 otherwise (read-only)
    run (inside harness via eval_gpu_queue.sh): consumer loop for the single GPU in $GPUS
"""
import argparse, json, os, subprocess, sys, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C              # noqa: E402
import eval_pipeline as EP      # noqa: E402  (frozen; helpers reused read-only)
import job_claims as JC         # noqa: E402

PORT_OFFSET = 100


def gpu_ports(host, gpu, serves):
    base = C.HOSTS[host]['port0'] + PORT_OFFSET
    return [base + gpu * serves + k for k in range(serves)]


class Coordination:
    """Registry of live consumers + the trainer's request, all under the claim lock."""

    def __init__(self, qdir, store):
        self.qdir, self.store = Path(qdir), store
        self.reg = self.qdir / 'consumers'; self.reg.mkdir(parents=True, exist_ok=True)

    def train_request(self):
        p = self.qdir / 'TRAIN_REQUEST.json'
        if not p.exists():
            return None
        r = json.loads(p.read_text())
        return r if r.get('boot_id') == self.store.boot_fn() and JC.alive(r.get('trainer'), self.store.start_fn) else None

    def consumers(self):
        out = {}
        for p in self.reg.glob('gpu*.json'):
            c = json.loads(p.read_text())
            if c.get('boot_id') == self.store.boot_fn() and JC.alive(c.get('consumer'), self.store.start_fn):
                out[int(c['gpu'])] = c
        return out

    def register(self, gpu, receipt, pid=None):
        body = {'gpu': gpu, 'receipt': receipt, 'boot_id': self.store.boot_fn(), 'consumer': JC.ident(pid or os.getpid()),
                'yielding': False, 'time': time.time()}
        self._write(gpu, body)
        return body

    def _write(self, gpu, body):
        p = self.reg / f'gpu{gpu}.json'; tmp = p.with_suffix('.tmp')
        tmp.write_text(json.dumps(body, indent=1)); os.replace(tmp, p)

    def over_share(self, demand, pool, serves, me=None):
        req = self.train_request()
        if req is None:
            return False
        live = [g for g, c in self.consumers().items() if not c.get('yielding') and g != me]
        share = JC.eval_share(demand, len(pool), serves, True)
        return len(live) + (1 if me is not None else 0) > share

    def mark_yield(self, gpu):
        p = self.reg / f'gpu{gpu}.json'
        c = json.loads(p.read_text()); c['yielding'] = True; c['yield_time'] = time.time(); self._write(gpu, c)

    def unregister(self, gpu):
        p = self.reg / f'gpu{gpu}.json'
        if p.exists():
            os.replace(p, self.reg / f'gpu{gpu}.exited.{time.time_ns()}.json')


def legacy_eval_receipts(state, host, pool):
    """Frozen eval_pipeline.queue runs under ONE multi-GPU eval receipt and does not know these claims: while such a
    receipt exists on the pool, per-GPU consumers must not take jobs (its running arms would look unclaimed)."""
    return [rid for rid, it in state.items() if it.get('machine') == host and it.get('kind') == 'eval'
            and len(it.get('gpus', [])) > 1 and set(it['gpus']) & set(pool)]


def may_start(p, host, gpu, pool, serves, coord, retry_failed=False, harness=None):
    legacy = legacy_eval_receipts((harness or EP.harness_state)(), host, pool)
    if legacy:
        return False, f'legacy multi-GPU eval receipt active {[r[:8] for r in legacy]}; wait for it to end'
    with coord.store.locked():
        js = p.jobs(host, retry_failed)
        live = set()
        for c in coord.store.d.glob('*.claim'):
            b = json.loads(c.read_text())
            if coord.store.live(b):
                live.add(tuple(b['key']))
        free_jobs = [j for j in js if tuple(j) not in live]
        if not free_jobs:
            return False, 'no unclaimed READY eval job'
        if coord.over_share(len(js), pool, serves, me=gpu):
            return False, 'trainer waiting and eval already at its fair share'
        return True, f'{len(free_jobs)} unclaimed jobs'


def consume(p, host, gpu, serves, pool, *, receipt, session, retry_failed=False, min_jobs=1, env_args=(),
            popen=subprocess.Popen, port_free=EP.wait_port_free, sleep=time.sleep, py=sys.executable):
    qdir = p.root / 'queue' / host
    (qdir / 'logs').mkdir(parents=True, exist_ok=True)
    store = JC.ClaimStore(qdir / 'claims')
    coord = Coordination(qdir, store)
    with store.locked():
        coord.register(gpu, receipt)
    lock, boot = threading.Lock(), threading.Lock()
    st = {'done': [], 'failures': [], 'yielding': False, 'jobs_done': 0}

    def take(port):
        with store.locked():
            if st['yielding']:
                return None
            if EP.shutil.disk_usage(C.R).free < EP.MIN_FREE_GB * 1024 ** 3:
                st['failures'].append({'error': f'disk free < {EP.MIN_FREE_GB}GiB; consumer stops taking jobs'})
                return None
            try:
                js = p.jobs(host, retry_failed, exclude={tuple(f.get('job', ())) for f in st['failures']})
            except Exception as e:  # noqa: BLE001
                m = EP.fail_marker(qdir / 'failed', f'QUEUE.gpu{gpu}.scan', e)
                st['failures'].append({'error': repr(e), 'marker': C.rel(m)}); return None
            if st['jobs_done'] >= min_jobs and coord.over_share(len(js), pool, serves, me=gpu):
                st['yielding'] = True; coord.mark_yield(gpu); return None
            for job in js:
                if store._try_claim(tuple(job), gpu=gpu, port=port, receipt=receipt, session=session):
                    return tuple(job)
            return None

    def verify(fid):
        """verify_fold once per fold across consumers: a terminal outcome (True/False from maybe_verify) is recorded in
        claims/done; None (= not all arms complete yet / already VERIFIED) is not terminal."""
        key = (fid, '__verify__')
        with store.locked():
            for d in (store.d / 'done').glob(f'{JC.key_name(key)}.*.json'):
                if json.loads(d.read_text()).get('outcome') in ('verified', 'verify_failed'):
                    return
            if not store._try_claim(key, gpu=gpu, receipt=receipt, session=session):
                return
        res = 'error'
        try:
            r = p.maybe_verify(fid)
            res = 'not_ready' if r is None else ('verified' if r else 'verify_failed')
        finally:
            store.release(key, res)

    def run_one(port, job):
        if True:
            fid, arm = job
            stamp = EP.now_ns()
            flag = qdir / 'logs' / f'{fid}.{arm}.{stamp}.ready'
            cmd = [py, '-u', str(HERE / 'run_eval_arm_v2.py'), '--manifest', str(p.fdir(fid) / 'manifest.json'), '--arm', arm,
                   '--gpu', str(gpu), '--port', str(port), '--ready-flag', str(flag), *env_args]
            if not port_free(port):
                m = EP.fail_marker(qdir / 'failed', f'{fid}.{arm}', RuntimeError(f'port {port} not free after 300s'))
                with lock:
                    st['failures'].append({'job': job, 'marker': C.rel(m)})
                store.release(job, 'port_busy'); return
            with (qdir / 'logs' / f'{fid}.{arm}.{stamp}.g{gpu}p{port}.log').open('x') as fh:
                with boot:
                    proc = popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(C.R))
                    store.set_child(job, proc.pid)
                    while proc.poll() is None and not flag.exists():
                        sleep(5)
                rc = proc.wait()
            ok = rc == 0 and p.arm_complete(fid, arm)
            with lock:
                if ok:
                    st['done'].append(job)
                else:
                    m = EP.fail_marker(qdir / 'failed', f'{fid}.{arm}', RuntimeError(f'run_eval_arm rc={rc} gpu={gpu} port={port}'))
                    st['failures'].append({'job': job, 'marker': C.rel(m)})
                st['jobs_done'] += 1
            store.release(job, 'complete' if ok else f'failed rc={rc}')
            if ok:
                verify(fid)
            with lock:
                EP.atomic_json(qdir / f'QUEUE_STATE.gpu{gpu}.json', {'time': time.time(), 'gpu': gpu, 'done_this_run': st['done'],
                                                                    'failures_this_run': st['failures'], 'yielding': st['yielding']})
    def worker(port):
        while True:
            job = take(port)
            if job is None:
                return
            try:
                run_one(port, job)
            except Exception as e:  # noqa: BLE001 - never leave a claim held by a dead worker thread, never exit 0 silently
                m = EP.fail_marker(qdir / 'failed', f'{job[0]}.{job[1]}', RuntimeError(f'consumer worker error: {e!r}'))
                with lock:
                    st['failures'].append({'job': job, 'marker': str(m), 'worker_error': repr(e)})
                if store.read(job) is not None:
                    store.release(job, f'worker_error {e!r}')
                return

    threads = [threading.Thread(target=worker, args=(port,)) for port in gpu_ports(host, gpu, serves)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    with store.locked():
        coord.unregister(gpu)
    EP.atomic_json(qdir / f'QUEUE_STATE.gpu{gpu}.json', {'time': time.time(), 'gpu': gpu, 'done_this_run': st['done'],
                                                        'failures_this_run': st['failures'], 'yielding': st['yielding'],
                                                        'exit': 'yielded to trainer' if st['yielding'] else 'no runnable job'})
    return 4 if any('job' in f for f in st['failures']) else 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path); ap.add_argument('--trqam-root', type=Path); ap.add_argument('--mil-root', type=Path)
    ap.add_argument('--pool', default='5,6,7', help='GPUs shared by eval and train on this host (fair-share denominator)')
    ap.add_argument('--retry-failed', action='store_true')
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('may-start'); s.add_argument('--host', required=True); s.add_argument('--gpu', type=int, required=True)
    s.add_argument('--serves', type=int, required=True)
    s = sub.add_parser('run'); s.add_argument('--min-jobs', type=int, default=1)
    s.add_argument('--env-container-prefix', help='srv: isolated env container name prefix, GPU id appended (fold27-iso-env-gpu)')
    s.add_argument('--expect-image')
    a = ap.parse_args(argv)
    p = EP.Pipeline(root=a.root, mil_root=a.mil_root, trqam_root=a.trqam_root)
    pool = [int(g) for g in a.pool.split(',')]
    if a.cmd == 'may-start':
        qdir = p.root / 'queue' / a.host
        ok, why = may_start(p, a.host, a.gpu, pool, a.serves, Coordination(qdir, JC.ClaimStore(qdir / 'claims')), a.retry_failed)
        print(json.dumps({'gpu': a.gpu, 'may_start': ok, 'why': why})); return 0 if ok else 3
    host = os.environ['HARNESS_MACHINE']; C.check_host(host)
    legacy = legacy_eval_receipts(EP.harness_state(), host, pool)
    C.need(not legacy, f'legacy multi-GPU eval receipt active {legacy}; per-GPU consumers must not run beside it')
    C.need(os.environ.get('HARNESS_RECEIPT'), 'HARNESS_RECEIPT missing (start through eval_gpu_queue.sh)')
    gpus = [int(g) for g in os.environ['GPUS'].split(',')]
    C.need(len(gpus) == 1 and gpus[0] in pool, f'one pool GPU per consumer, got GPUS={os.environ["GPUS"]}')
    serves = int(os.environ['SERVES_PER_GPU'])
    subprocess.run(['python3', str(C.R / 'scripts/utils/groot_harness.py'), 'verify', '--machine', host, '--gpus', os.environ['GPUS'],
                    '--serves', str(serves), '--session', os.environ['HARNESS_SESSION'], '--kind', 'eval',
                    '--receipt', os.environ['HARNESS_RECEIPT'], '--consumer-pid', str(os.getpid())], check=True)
    env_args = []
    if C.HOSTS[host]['env'] == 'docker_exec':
        C.need(a.env_container_prefix, 'srv hosts need --env-container-prefix (isolated env container per GPU)')
        env_args = ['--env-container', f'{a.env_container_prefix}{gpus[0]}', '--expect-repo', str(C.R),
                    *(['--expect-image', a.expect_image] if a.expect_image else [])]
    return consume(p, host, gpus[0], serves, pool, receipt=os.environ['HARNESS_RECEIPT'], session=os.environ['HARNESS_SESSION'],
                   retry_failed=a.retry_failed, min_jobs=a.min_jobs, env_args=env_args)


if __name__ == '__main__':
    sys.exit(main())

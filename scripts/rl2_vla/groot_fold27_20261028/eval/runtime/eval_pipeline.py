"""fold27 production eval connector: TRQAM DONE -> actors -> immutable manifest -> per-host eval queue (resumable).

Separate orchestrator; reuses restore_actor / build_manifest / run_eval_arm / verify_fold unchanged. No new hyperparameters:
arms = mil_qam65 (common atomic65 500k) + mil_trqam_{basic,pbrs,mix} at trqam/train_queue.py's fixed run tags.

  prepare  (kanu, CPU, .venvs/rl2_iid) per fold, as soon as THAT fold is ready (never waits for all 27):
           fold MIL FOLD_CONTRACT + 3 TRQAM DONE.json (50k, not smoke, init = verified common 500k)
           -> export missing TRQAM actors (write-once) -> build_manifest (same 40 rows, write-once) -> <fold>/READY.json
           failures -> <fold>/PREPARE_FAILED.<ns>.json (skipped later unless --retry-failed). --watch S repeats.
  status   per-host table from markers (no side effects except PIPELINE_STATE.json)
  pending  --host H: exit 0 if runnable (fold, arm) eval jobs exist on H, 3 if none
  wait-resources --host H --gpus G --serves N --session S: block until the harness ledger can admit the request
           (all-session GPU cap, per-GPU serves, foreign ownership) and the GPUs have no compute processes. Never forces.
  queue    (inside the harness; started by eval_queue.sh) slots = GPUS x SERVES_PER_GPU; each slot pulls the next
           READY (fold, arm) of this host lacking RUN_COMPLETE; run_eval_arm resumes completed rows by hash; a failing
           (fold, arm) gets queue/<host>/failed/<fold>.<arm>.<ns>.json and is not retried in that queue run
           (--retry-failed on the next run). All 4 arms complete -> verify_fold -> <fold>/VERIFIED.json or VERIFY_FAILED.<ns>.json.
           Exits when nothing is runnable (GPUs released); exit 4 if any job failed.
Layout: <root>=outputs/eval/robocasa/groot_n15/fold27_mil_qam_20261102 ; actors/<fold>/<arm>/ ; <fold>/manifest.json ; <fold>/<arm>/...
"""
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

PROD_ROOT = 'outputs/eval/robocasa/groot_n15/fold27_mil_qam_20261102'
MIL_ROOT = 'outputs/rl2_vla/fold27_mil_20261030'
TRQAM_ROOT = 'outputs/rl2_vla/fold27_trqam_20261102'
COMMON_RUN = C.COMMON_QAM_WORK + '/scratch_run'
COMMON_ACTOR_SHA = 'f79ac933f119648810cdd9f339714c5a2e981343eba2274c6f2328cd78de65b7'
TRQAM_ARMS = {'mil_trqam_basic': 'basic', 'mil_trqam_pbrs': 'pbrs', 'mil_trqam_mix': 'mix'}
ARM_ORDER = ('mil_qam65', 'mil_trqam_basic', 'mil_trqam_pbrs', 'mil_trqam_mix')
HOST_FOLDS = {'kanu': 6, 'srv48': 9, 'srv50': 12}
MIN_FREE_GB = 10          # queue refuses a new (fold, arm) below this; run_eval_arm keeps its own 2GiB per-episode guard
EPISODE_MB = 0.6          # smoke s1 measured ~0.35MB rows + ~0.1MB logs per episode -> 4 arms x 40 x 27 ~ 2.6GB total


def trqam_tags():
    sys.path.insert(0, str(FOLD27_TRQAM))
    from train_queue import TAGS          # single source of the TRQAM run tags (read-only)
    return TAGS


FOLD27_TRQAM = C.FOLD27 / 'trqam'


def now_ns():
    return time.time_ns()


def atomic_json(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True) + '\n')
    os.replace(tmp, p)


def fail_marker(d, prefix, err):
    d = Path(d); d.mkdir(parents=True, exist_ok=True)
    p = d / f'{prefix}.{now_ns()}.json'
    C.write_json_new(p, {'error': repr(err), 'type': type(err).__name__, 'time': time.time(), 'host': socket.gethostname()})
    return p


class Pipeline:
    def __init__(self, root=None, mil_root=None, trqam_root=None, common_run=None, spec=None, python=None):
        self.root = Path(root or C.R / PROD_ROOT)
        self.mil_root = Path(mil_root or C.R / MIL_ROOT)
        self.trqam_root = Path(trqam_root or C.R / TRQAM_ROOT)
        self.common_run = Path(common_run or C.R / COMMON_RUN)
        self.spec_path = Path(spec or C.R / C.FOLD_SPEC)
        self.python = python or str(C.R / '.venvs/rl2_iid/bin/python')
        self.spec = json.loads(self.spec_path.read_text())

    # ------------------------------------------------------------------ paths / markers
    def folds(self, host=None, only=None):
        return [f for f in self.spec['folds'] if (host is None or f['eval_host'] == host) and (not only or f['fold_id'] in only)]

    def fdir(self, fid):
        return self.root / fid

    def mil_run(self, fid):
        return self.mil_root / fid / 'attn/raw/s0'

    def actor_dir(self, fid, arm):
        return self.root / 'actors' / ('common_qam65' if arm == 'mil_qam65' else f'{fid}/{TRQAM_ARMS[arm]}')

    def trqam_run(self, fid, arm):
        return self.trqam_root / fid / trqam_tags()[TRQAM_ARMS[arm]]

    def ready(self, fid):
        p = self.fdir(fid) / 'READY.json'
        if not p.exists():
            return None
        r = json.loads(p.read_text())
        C.need(C.sha256_path(self.fdir(fid) / 'manifest.json') == r['manifest_sha256'], f'{fid}: manifest changed after READY')
        return r

    def prepare_failed(self, fid):
        return sorted(self.fdir(fid).glob('PREPARE_FAILED.*.json')) if self.fdir(fid).exists() else []

    def arm_complete(self, fid, arm):
        return (self.fdir(fid) / arm / 'RUN_COMPLETE.json').exists()

    # ------------------------------------------------------------------ prepare (CPU)
    def check_common_actor(self):
        d = self.actor_dir(None, 'mil_qam65')
        prov = json.loads((d / 'runtime_provenance.json').read_text())
        C.need(prov['production'] is True and prov['kind'] == 'common' and int(prov['common_steps']) == C.COMMON_STEPS,
               'common actor provenance not production 500k')
        C.need(C.sha256_path(d / 'actor.npz') == prov['actor_sha256'] == COMMON_ACTOR_SHA, 'common actor sha differs from canonical f79ac933')
        return d

    def fold_inputs(self, fid):
        """None-valued entries mean 'not ready yet' (normal waiting), exceptions mean contract violations."""
        import restore_actor as RS
        st = {'mil': (self.mil_run(fid) / 'FOLD_CONTRACT.json').exists(), 'trqam': {}}
        for arm in TRQAM_ARMS:
            run = self.trqam_run(fid, arm)
            if not (run / 'DONE.json').exists():
                st['trqam'][arm] = None
                continue
            done = json.loads((run / 'DONE.json').read_text())
            C.need(done.get('smoke') is not True, f'{run}: smoke DONE in production slot')
            RS.check_trqam_run(run, fid, TRQAM_ARMS[arm], self.common_run)
            st['trqam'][arm] = run
        return st

    def export_actor(self, fid, arm, run):
        out = self.actor_dir(fid, arm)
        if out.exists():
            C.need((out / 'runtime_provenance.json').is_file(), f'{out}: partial actor export (no provenance); inspect, no auto-delete')
            prov = json.loads((out / 'runtime_provenance.json').read_text())
            C.need(prov['fold_id'] == fid and prov['kind'] == f'trqam_{TRQAM_ARMS[arm]}', f'{out}: provenance mismatch')
            C.need(prov['params_sha256'] == json.loads((run / 'DONE.json').read_text())['params_sha256'], f'{out}: exported from other params')
            return out
        out.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                   XLA_PYTHON_CLIENT_PREALLOCATE='false')
        log = self.fdir(fid) / 'logs' / f'export.{arm}.{now_ns()}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('x') as fh:
            rc = subprocess.run([self.python, str(HERE / 'restore_actor.py'), 'trqam', '--run', str(run), '--fold', fid,
                                 '--arm', TRQAM_ARMS[arm], '--common-run', str(self.common_run), '--out', str(out)],
                                stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(C.R)).returncode
        C.need(rc == 0, f'actor export failed rc={rc}; see {C.rel(log)}')
        return out

    def prepare_fold(self, fid, retry_failed=False):
        if self.ready(fid):
            return 'ready'
        if self.prepare_failed(fid) and not retry_failed:
            return 'failed'
        try:
            st = self.fold_inputs(fid)
            if not st['mil'] or any(v is None for v in st['trqam'].values()):
                return 'waiting'
            actors = {'mil_qam65': self.check_common_actor()}
            for arm, run in st['trqam'].items():
                actors[arm] = self.export_actor(fid, arm, run)
            import build_manifest as B
            path, m = B.build(fid, self.mil_run(fid), actors, self.root, spec_path=self.spec_path)
            C.write_json_new(self.fdir(fid) / 'READY.json', {'fold_id': fid, 'host': m['host'], 'manifest_sha256': C.sha256_path(path),
                                                            'arms': sorted(m['arms']), 'rows': len(m['rows'])})
            return 'ready'
        except Exception as e:  # noqa: BLE001 - every failure becomes an explicit marker
            fail_marker(self.fdir(fid), 'PREPARE_FAILED', e)
            return 'failed'

    def prepare(self, only=None, retry_failed=False):
        out = {f['fold_id']: self.prepare_fold(f['fold_id'], retry_failed) for f in self.folds(only=only)}
        self.write_state()
        return out

    # ------------------------------------------------------------------ status / jobs
    def fold_status(self, f):
        fid = f['fold_id']
        if (self.fdir(fid) / 'VERIFIED.json').exists():
            return 'verified'
        if self.ready(fid):
            done = [a for a in ARM_ORDER if self.arm_complete(fid, a)]
            return f'evaluating {len(done)}/4' if len(done) < 4 else 'eval_complete_unverified'
        return 'prepare_failed' if self.prepare_failed(fid) else 'waiting_inputs'

    def write_state(self):
        st = {'time': time.time(), 'hosts': {}}
        for h in HOST_FOLDS:
            st['hosts'][h] = {f['fold_id']: self.fold_status(f) for f in self.folds(host=h)}
        atomic_json(self.root / 'PIPELINE_STATE.json', st)
        return st

    def failed_jobs(self, host):
        d = self.root / 'queue' / host / 'failed'
        return {tuple(p.name.split('.')[:2]) for p in d.glob('*.json')} if d.exists() else set()

    def jobs(self, host, retry_failed=False, exclude=()):
        failed = set() if retry_failed else self.failed_jobs(host)
        out = []
        for f in self.folds(host=host):
            fid = f['fold_id']
            if not self.ready(fid):
                continue
            for arm in ARM_ORDER:
                if not self.arm_complete(fid, arm) and (fid, arm) not in failed and (fid, arm) not in exclude:
                    out.append((fid, arm))
        return out

    def maybe_verify(self, fid):
        if (self.fdir(fid) / 'VERIFIED.json').exists() or not all(self.arm_complete(fid, a) for a in ARM_ORDER):
            return None
        log = self.fdir(fid) / 'logs' / f'verify.{now_ns()}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('x') as fh:
            rc = subprocess.run([sys.executable, str(HERE / 'verify_fold.py'), '--manifest', str(self.fdir(fid) / 'manifest.json')],
                                stdout=fh, stderr=subprocess.STDOUT, cwd=str(C.R)).returncode
        if rc == 0:
            C.write_json_new(self.fdir(fid) / 'VERIFIED.json', {'fold_id': fid, 'log': C.rel(log), 'per_episode_tsv_sha256':
                                                                C.sha256_path(self.fdir(fid) / 'per_episode.tsv')})
            return True
        fail_marker(self.fdir(fid), 'VERIFY_FAILED', RuntimeError(f'verify_fold rc={rc}; see {C.rel(log)}'))
        return False


# ---------------------------------------------------------------------- resource wait (never forces)
def admissible(state, host, gpus, serves, session):
    cap_g, cap_s = C.caps(host)
    occupied, reasons = set(), []
    for rid, it in state.items():
        if it.get('machine') != host:
            continue
        occupied.update(it['gpus'])
        for g in set(gpus) & set(it['gpus']):
            if it['session'] != session or it['kind'] != 'eval':
                reasons.append(f'GPU{g} held by {it["session"]}/{it["kind"]} ({rid[:8]})')
    if len(occupied | set(gpus)) > cap_g:
        reasons.append(f'{host} all-session GPU cap {cap_g}: held {sorted(occupied)} + want {sorted(gpus)}')
    for g in gpus:
        used = sum(it['serves'] for it in state.values() if it.get('machine') == host and g in it['gpus'])
        if used + serves > cap_s:
            reasons.append(f'GPU{g}: {used}+{serves} serves > {cap_s}')
    return reasons


def harness_state():
    r = subprocess.run(['python3', str(C.R / 'scripts/utils/groot_harness.py'), 'status'], capture_output=True, text=True, timeout=120)
    C.need(r.returncode == 0, f'harness status failed: {r.stderr.strip()}')
    return json.loads(r.stdout)


def gpu_busy(gpus):
    busy = []
    for g in gpus:
        r = subprocess.run(['nvidia-smi', f'--id={g}', '--query-compute-apps=pid', '--format=csv,noheader'], capture_output=True, text=True)
        if r.returncode != 0 or r.stdout.strip():
            busy.append(g)
    return busy


def wait_resources(host, gpus, serves, session, poll=120, max_wait=None, state_fn=harness_state, busy_fn=gpu_busy, sleep=time.sleep):
    t0 = time.time()
    while True:
        reasons = admissible(state_fn(), host, gpus, serves, session)
        busy = [] if reasons else busy_fn(gpus)
        if not reasons and not busy:
            return True
        print(json.dumps({'time': time.time(), 'waiting': reasons + [f'GPU{g} has compute processes' for g in busy]}), flush=True)
        if max_wait is not None and time.time() - t0 > max_wait:
            return False
        sleep(poll)


# ---------------------------------------------------------------------- queue (inside harness)
def wait_port_free(port, timeout=300, sleep=time.sleep):
    """run_eval_arm probes its port with a plain bind (no SO_REUSEADDR). Right after the previous serve on the same slot
    exits, server-side TIME_WAIT sockets (~60s) make that probe fail with EADDRINUSE (smoke s1 smoke_gated, 2026-11-02).
    Wait for the same probe to pass instead of failing the job."""
    t0 = time.time()
    while True:
        try:
            with socket.socket() as s:
                s.bind(('127.0.0.1', port))
            return True
        except OSError:
            if time.time() - t0 > timeout:
                return False
            sleep(5)


def queue(p, retry_failed=False, port0=None, serve_py=C.SRV_SERVE_PY):
    host = os.environ['HARNESS_MACHINE']
    C.check_host(host)
    C.need(os.environ.get('HARNESS_RECEIPT'), 'HARNESS_RECEIPT missing (start through eval_queue.sh)')
    gpus = [int(g) for g in os.environ['GPUS'].split(',')]
    serves = int(os.environ['SERVES_PER_GPU'])
    subprocess.run(['python3', str(C.R / 'scripts/utils/groot_harness.py'), 'verify', '--machine', host, '--gpus', os.environ['GPUS'],
                    '--serves', str(serves), '--session', os.environ['HARNESS_SESSION'], '--kind', 'eval',
                    '--receipt', os.environ['HARNESS_RECEIPT'], '--consumer-pid', str(os.getpid())], check=True)
    import launch_group as LG
    slots = LG.plan_slots(host, gpus, serves, port0 or C.HOSTS[host]['port0'])
    qdir = p.root / 'queue' / host
    (qdir / 'logs').mkdir(parents=True, exist_ok=True)
    lock, boot = threading.Lock(), {g: threading.Lock() for g in gpus}
    running, failures, done = set(), [], []

    def take():
        with lock:
            if shutil.disk_usage(C.R).free < MIN_FREE_GB * 1024 ** 3:
                failures.append({'error': f'disk free < {MIN_FREE_GB}GiB; queue stops taking jobs (no deletion)'})
                return None
            try:
                js = p.jobs(host, retry_failed, exclude=running | {tuple(f.get('job', ())) for f in failures})
            except Exception as e:  # noqa: BLE001 - e.g. manifest changed after READY: stop taking, keep marker
                m = fail_marker(qdir / 'failed', 'QUEUE.scan', e)
                failures.append({'error': repr(e), 'marker': C.rel(m)})
                return None
            if not js:
                return None
            running.add(js[0])
            return js[0]

    def worker(gpu, port):
        while True:
            job = take()
            if job is None:
                return
            fid, arm = job
            stamp = now_ns()
            flag = qdir / 'logs' / f'{fid}.{arm}.{stamp}.ready'
            cmd = [sys.executable, '-u', str(HERE / 'run_eval_arm.py'), '--manifest', str(p.fdir(fid) / 'manifest.json'), '--arm', arm,
                   '--gpu', str(gpu), '--port', str(port), '--serve-py', serve_py, '--ready-flag', str(flag)]
            if not wait_port_free(port):
                with lock:
                    running.discard(job)
                    m = fail_marker(qdir / 'failed', f'{fid}.{arm}', RuntimeError(f'port {port} not free after 300s'))
                    failures.append({'job': job, 'marker': C.rel(m)})
                continue
            with (qdir / 'logs' / f'{fid}.{arm}.{stamp}.g{gpu}p{port}.log').open('x') as fh:
                with boot[gpu]:
                    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(C.R))
                    while proc.poll() is None and not flag.exists():
                        time.sleep(5)
                rc = proc.wait()
            with lock:
                running.discard(job)
                if rc == 0 and p.arm_complete(fid, arm):
                    done.append(job)
                else:
                    m = fail_marker(qdir / 'failed', f'{fid}.{arm}', RuntimeError(f'run_eval_arm rc={rc} gpu={gpu} port={port}'))
                    failures.append({'job': job, 'marker': C.rel(m)})
            if rc == 0:
                p.maybe_verify(fid)
            with lock:
                atomic_json(qdir / 'QUEUE_STATE.json', {'time': time.time(), 'slots': slots, 'running': sorted(running),
                                                        'done_this_run': done, 'failures_this_run': failures})
    threads = [threading.Thread(target=worker, args=s) for s in slots]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    p.write_state()
    atomic_json(qdir / 'QUEUE_STATE.json', {'time': time.time(), 'slots': slots, 'running': [], 'done_this_run': done,
                                            'failures_this_run': failures, 'exit': 'no runnable job'})
    return 4 if failures else 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path); ap.add_argument('--trqam-root', type=Path); ap.add_argument('--mil-root', type=Path)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('prepare'); s.add_argument('--folds', nargs='+'); s.add_argument('--retry-failed', action='store_true')
    s.add_argument('--watch', type=float, help='repeat every N seconds until all 27 folds are READY or failed')
    sub.add_parser('status')
    s = sub.add_parser('pending'); s.add_argument('--host', required=True); s.add_argument('--retry-failed', action='store_true')
    s = sub.add_parser('wait-resources'); s.add_argument('--host', required=True); s.add_argument('--gpus', required=True)
    s.add_argument('--serves', type=int, required=True); s.add_argument('--session', required=True); s.add_argument('--poll', type=float, default=120)
    s = sub.add_parser('queue'); s.add_argument('--retry-failed', action='store_true'); s.add_argument('--port0', type=int)
    s.add_argument('--serve-py', default=C.SRV_SERVE_PY)
    a = ap.parse_args(argv)
    p = Pipeline(root=a.root, mil_root=a.mil_root, trqam_root=a.trqam_root)
    if a.cmd == 'prepare':
        while True:
            res = p.prepare(a.folds, a.retry_failed)
            print(json.dumps({'time': time.time(), 'counts': {k: list(res.values()).count(k) for k in set(res.values())}}), flush=True)
            if not a.watch or all(v != 'waiting' for v in res.values()):
                return 0 if 'failed' not in res.values() else 4
            time.sleep(a.watch)
    if a.cmd == 'status':
        print(json.dumps(p.write_state(), indent=1)); return 0
    if a.cmd == 'pending':
        js = p.jobs(a.host, a.retry_failed)
        print(json.dumps({'host': a.host, 'pending': js}))
        return 0 if js else 3
    if a.cmd == 'wait-resources':
        wait_resources(a.host, [int(g) for g in a.gpus.split(',')], a.serves, a.session, a.poll); return 0
    return queue(p, a.retry_failed, a.port0, a.serve_py)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except C.RuntimeContractError as e:
        print('PIPELINE_REFUSED', e, file=sys.stderr); sys.exit(2)

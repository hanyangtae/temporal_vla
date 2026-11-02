"""Run one (fold manifest, arm) on one GPU slot: one serve, the manifest rows that are still missing, sequentially.

Generalizes groot_rescue9_eval_20261022/run_cell.py (same container/serve/collector protocol) to every fold host.
Must be a child of the GR00T harness (HARNESS_RECEIPT, GPUS) — launch through launch.sh / launch_group.py.

    run_eval_arm.py --manifest <out>/<fold>/manifest.json --arm mil_qam65 --gpu 6 --port 8860

Output <manifest dir>/<arm>/: progress.jsonl (one line per finished row), rows/n<NN>/a<k>/ (one dir per attempt;
an unfinished attempt is kept and never reused), serve logs, decisions logs, RUN_COMPLETE.json, failures/*.json.
Resume = same command: rows already in progress.jsonl are re-hashed (meta/traj sha) and skipped; only missing rows run.
"""
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

SERVE = HERE / 'serve_mil_qam.py'
COLLECT = HERE / 'collect_task.py'
READY_TIMEOUT = 900
EPISODE_TIMEOUT = 1800


def get_json(port, route):
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/{route}', timeout=5) as r:
        return json.load(r)


def lines(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()] if Path(p).exists() else []


# ------------------------------------------------------------------ manifest / resume (pure, CPU-tested)
def verify_manifest(mpath, arm, *, check_files=True):
    mpath = Path(mpath).resolve()
    m = json.loads(mpath.read_text())
    C.need(arm in m['arms'], f'arm {arm} not in manifest')
    smoke_root = str((C.R / C.SMOKE_ROOT).resolve())
    if m['smoke']:
        C.need(str(mpath).startswith(smoke_root) and arm in C.SMOKE_ARMS, 'smoke manifest must live under SMOKE_ROOT with smoke arms')
    else:
        C.need(not str(mpath).startswith(smoke_root) and arm in C.PRODUCTION_ARMS, 'production manifest with smoke arm/location')
        C.need(len(m['rows']) == 40 and [r['n'] for r in m['rows']] == list(range(40)), 'production needs the same 40 rows n0..39')
        C.need(m['arms'][arm]['mode'] == 'gated', 'production arms are MIL-gated')
    for r in m['rows']:
        for k in ('env_seed', 'inference_seed', 'plan_id', 'j', 'n', 'base_success'):
            C.need(r.get(k) is not None, f'row n{r.get("n")} missing {k}')
    if not check_files:
        return m
    from build_manifest import check_code_hashes
    check_code_hashes(m['code'])
    spec = C.R / m['fold_spec']['path']
    C.need(C.sha256_path(spec) == m['fold_spec']['sha256'], 'fold spec changed')
    for pid, p in m['plans'].items():
        C.need((C.R / p['path']).is_file(), f'plan {pid} missing on this host: {p["path"]}')
        C.need(C.sha256_path(C.R / p['path']) == p['sha256'], f'plan {pid} sha mismatch')
        _, plan = C.load_plan(pid)
        for r in m['rows']:
            if r['plan_id'] == pid:
                C.check_row_against_plan(plan, m['instruction'], m['scene'], r)
                C.need(C.task_identity(plan, m['instruction'], m['scene']) == m['identity'], 'plan identity drift')
    C.need((C.R / m['mil']['model']).is_file(), f'MIL model missing on this host: {m["mil"]["model"]}')
    C.need(C.sha256_path(C.R / m['mil']['model']) == m['mil']['sha256'], 'MIL model sha mismatch')
    if m['mil']['fold_contract']:
        C.need(C.sha256_path(C.R / m['mil']['fold_contract']['path']) == m['mil']['fold_contract']['sha256'], 'FOLD_CONTRACT sha mismatch')
    pin = m['arms'][arm]
    if pin['actor']:
        a = C.R / pin['actor']
        C.need(a.is_dir(), f'actor missing on this host: {pin["actor"]}')
        C.need(C.sha256_path(a / 'actor.npz') == pin['actor_sha256'], 'actor.npz sha mismatch')
        C.need(C.sha256_path(a / 'actor.json') == pin['actor_json_sha256'], 'actor.json sha mismatch')
    C.need(C.tc_commit() == C.TC_COMMIT, f'task_classification HEAD {C.tc_commit()} != {C.TC_COMMIT}')
    return m


def row_files_sha(meta_path):
    d = Path(meta_path).parent
    out = {'meta.json': C.sha256_path(meta_path)}
    for name in ('traj.csv',):
        if (d / name).exists():
            out[name] = C.sha256_path(d / name)
    return out


def load_progress(out, m, arm):
    """Return {n: item}; re-hash every finished row; any inconsistency refuses to resume."""
    done = {}
    for it in lines(out / 'progress.jsonl'):
        C.need(it['fold_id'] == m['fold_id'] and it['arm'] == arm, 'foreign progress line')
        C.need(it['n'] not in done, f'duplicate progress n{it["n"]}')
        row = next((r for r in m['rows'] if r['n'] == it['n']), None)
        C.need(row is not None, f'progress n{it["n"]} not in manifest rows')
        C.need(it['env_seed'] == row['env_seed'] and it['inference_seed'] == row['inference_seed'], f'n{it["n"]}: seed drift')
        C.need(row_files_sha(out / it['meta']) == it['files_sha256'], f'n{it["n"]}: finished row files changed (hash)')
        done[it['n']] = it
    for adir in sorted((out / 'rows').glob('n*/a*')):
        n = int(adir.parent.name[1:])
        metas = list(adir.rglob('meta.json'))
        if metas and (n not in done or done[n]['attempt'] != adir.name):
            raise C.RuntimeContractError(f'{adir}: finished episode without progress line (orphan); inspect before resuming')
    return done


def pending_rows(m, done):
    return [r for r in m['rows'] if r['n'] not in done]


def next_attempt(out, n):
    d = out / 'rows' / f'n{n:02d}'
    k = len(list(d.glob('a*'))) if d.exists() else 0
    return d / f'a{k}'


# ------------------------------------------------------------------ host commands
def kanu_docker(out, gpu, name, image):
    sys.path.insert(0, str(C.ADAPTER))
    from paired_eval_audited import docker_args
    argv = docker_args(out, gpu, name, image)
    return argv[:2] + ['--log-driver', 'none', '--tmpfs', '/tmp:rw,size=1g'] + argv[2:]   # rescue9 run_cell.py


def inside(p, _host=None):
    return C.CONTAINER_REPO + '/' + C.rel(p)


def serve_command(m, arm, out, gpu, port, name, host, serve_py):
    pin = m['arms'][arm]
    args = ['--mode', pin['mode'], '--mil-model', None, '--mil-sha256', m['mil']['sha256'], '--target-j', str(m['target_j'])]
    if m['smoke']:
        args.append('--smoke')
    tail = ['--profile', 'configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml', '--host', '127.0.0.1', '--port', str(port)]
    if C.HOSTS[host]['serve'] == 'docker_run':
        sys.path.insert(0, str(C.ADAPTER))
        from paired_eval_audited import MODEL_IMAGE
        args[3] = inside(C.R / m['mil']['model'], host)
        if pin['actor']:
            args += ['--qam-actor', inside(C.R / pin['actor'], host), '--actor-sha256', pin['actor_sha256']]
        return kanu_docker(out, gpu, name, MODEL_IMAGE) + [inside(SERVE, host), *args, '--', *tail], None
    args[3] = str(C.R / m['mil']['model'])
    if pin['actor']:
        args += ['--qam-actor', str(C.R / pin['actor']), '--actor-sha256', pin['actor_sha256']]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PDK_LOG_FILE=str(out / f'{name}.module.log'),
               PYTHONPATH=f'{C.R}/lerobot/src:{C.R}:{C.R}/src/policies/Isaac-GR00T', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1')
    return [os.path.expanduser(serve_py), str(SERVE), *args, '--', *tail], env


def collect_command(m, row, out, attempt, gpu, port, name, host, manifest_sha):
    ident = m['identity']
    plan = C.R / m['plans'][row['plan_id']]['path']
    args = [inside(COLLECT, host), '--vla-server', f'http://127.0.0.1:{port}', '--task', ident['task'], '--env-name', ident['env_name'],
            '--split', 'target', '--output-dir', inside(attempt / 'raw_rollouts', host), '--episode-start-idx', '0', '--n-episodes', '1',
            '--seed', str(row['env_seed']), '--inference-seed', str(row['inference_seed']), '--n-action-steps', str(C.N_ACTION_STEPS),
            '--max-episode-steps', str(C.MAX_STEPS), '--video-fps', '20', '--steps-per-render', '2', '--wait-ready', '--no-features',
            '--expect-chunk-len', str(C.CHUNK), '--plan-json', inside(plan, host), '--scene-idx', str(m['scene']),
            '--jitter-idx', str(row['j']), '--noise-idx', str(row['n']), '--grid-instruction', m['instruction'],
            '--canonical-instruction', ident['lang'], '--grid-root', inside(attempt / 'grid', host), '--arm-dir', m['_arm'],
            '--run-tag', manifest_sha]
    if C.HOSTS[host]['env'] == 'docker_run':
        sys.path.insert(0, str(C.ADAPTER))
        from paired_eval_audited import ENV_IMAGE
        return kanu_docker(out, gpu, name, ENV_IMAGE) + args
    pypath = '/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla'
    return ['docker', 'exec', '-w', C.CONTAINER_REPO, '-e', 'MUJOCO_GL=egl', '-e', f'PYTHONPATH={pypath}',
            '-e', 'OMP_NUM_THREADS=4', '-e', 'OPENBLAS_NUM_THREADS=4', '-e', 'PYTHONDONTWRITEBYTECODE=1',
            '-e', f'PDK_LOG_FILE={inside(out / (name + ".module.log"), host)}', C.SRV_ENV_CONTAINER,
            'timeout', '--signal=TERM', str(EPISODE_TIMEOUT), 'python', *args]


def srv_env_preflight(out):
    """docker exec target must see this repo at /temporal_vla read-write (outputs are written through it)."""
    r = subprocess.run(['docker', 'inspect', C.SRV_ENV_CONTAINER, '--format', '{{json .Mounts}}|{{.Image}}|{{.State.Running}}'],
                       capture_output=True, text=True, timeout=20)
    C.need(r.returncode == 0, f'robocasa container missing: {r.stderr.strip()}')
    mounts, image, running = r.stdout.strip().split('|')
    C.need(running == 'true', 'robocasa container not running')
    ok = any(x['Destination'] == C.CONTAINER_REPO and Path(x['Source']).resolve() == C.R and x.get('RW') for x in json.loads(mounts))
    C.need(ok, f'robocasa container does not mount {C.R} rw at {C.CONTAINER_REPO}')
    return image


def storage_guard():
    C.need(shutil.disk_usage(C.R).free >= 2 * 1024 ** 3, 'disk reserve below 2GiB; paused before next episode (no deletion)')


# ------------------------------------------------------------------ audit of one finished row
def audit_row(out, attempt, row, arm, m, q0, q1, trace_path):
    metas = list((attempt / 'grid').rglob('meta.json'))
    C.need(len(metas) == 1, f'expected one meta.json, got {metas}')
    meta = json.loads(metas[0].read_text())
    C.need(metas[0].parent.name == arm, f'arm dir {metas[0].parent.name} != {arm}')
    C.need(f'/n{row["n"]}/' in str(metas[0]) and f'/j{row["j"]}/' in str(metas[0]) and f'/s{m["scene"]}/' in str(metas[0]),
           f'grid coordinate mismatch {metas[0]}')
    for k, want in (('inference_seed', row['inference_seed']), ('scenario_seed', row['env_seed'])):
        if k in meta:
            C.need(int(meta[k]) == int(want), f'meta {k} {meta[k]} != {want}')
    tr = lines(trace_path)
    C.need(tr and tr[-1]['event'] == 'complete', 'collector trace incomplete')
    sends = sum(x['event'] == 'http_send' for x in tr)
    reqs = q1['requests'] - q0['requests']
    C.need(sends == reqs, f'HTTP sends {sends} != serve inferences {reqs}')
    C.need(q1['gate_calls'] - q0['gate_calls'] == reqs, 'MIL gate not evaluated on every inference')
    C.need(q1['off_identity_verified'], 'gate-off identity not verified')
    C.need(q1['episode_resets'] > q0['episode_resets'], 'policy reset (MIL reset) not observed for this episode')
    success = meta.get('episode_success', meta.get('success'))
    C.need(success is not None, 'meta lacks success')
    return {'success': int(bool(success)), 'steps': meta.get('steps'), 'meta': str(metas[0].relative_to(out)),
            'files_sha256': row_files_sha(metas[0]), 'requests': [q0['requests'] + 1, q1['requests']],
            'composed': q1['composed_calls'] - q0['composed_calls'], 'fired': q1['fired_requests'] - q0['fired_requests'],
            'first_fire_t': q1['first_fire_t'], 'http_sends': sends}


# ------------------------------------------------------------------ main loop
def run(a):
    C.need(os.environ.get('HARNESS_RECEIPT'), 'launch through the GR00T harness (HARNESS_RECEIPT missing)')
    C.need(str(a.gpu) in os.environ.get('GPUS', '').split(','), f'GPU {a.gpu} not in harness GPUS {os.environ.get("GPUS")}')
    m = verify_manifest(a.manifest, a.arm)
    m['_arm'] = a.arm
    host = m['host']
    C.check_host(host)
    C.need(os.environ.get('HARNESS_MACHINE', host) == host, 'HARNESS_MACHINE differs from manifest host')
    out = Path(a.manifest).resolve().parent / a.arm
    (out / 'failures').mkdir(parents=True, exist_ok=True)
    msha = C.sha256_path(a.manifest)
    done = load_progress(out, m, a.arm)
    todo = pending_rows(m, done)
    if a.rows is not None:
        C.need(m['smoke'], '--rows subset only for smoke manifests')
        todo = [r for r in todo if r['n'] in a.rows]
    env_image = srv_env_preflight(out) if C.HOSTS[host]['env'] == 'docker_exec' else None
    print(json.dumps({'fold': m['fold_id'], 'arm': a.arm, 'done': len(done), 'pending': [r['n'] for r in todo]}), flush=True)
    if not todo:
        if a.ready_flag:
            Path(a.ready_flag).touch()
        return finish(out, m, a.arm)
    with socket.socket() as s:
        s.bind(('127.0.0.1', a.port))
    name = f'f27-{m["fold_id"].lower().replace("/", "-")[:40]}-{a.arm}-{os.getpid()}'.replace('_', '-')
    argv, env = serve_command(m, a.arm, out, a.gpu, a.port, name, host, a.serve_py)
    if env is None:
        _ensure_absent(name)
    serve_log = (out / f'{name}.serve.log').open('x')
    proc = subprocess.Popen(argv, stdout=serve_log, stderr=subprocess.STDOUT, env=env, cwd=str(C.R), start_new_session=env is not None)
    progress = (out / 'progress.jsonl').open('a', buffering=1)
    try:
        deadline = time.monotonic() + READY_TIMEOUT
        while True:
            C.need(proc.poll() is None, f'serve exited {proc.returncode}; see {serve_log.name}')
            try:
                state = get_json(a.port, 'qam_contract')
                if state.get('ready'):
                    break
            except (OSError, ValueError):
                pass
            C.need(time.monotonic() < deadline, 'serve readiness timeout')
            time.sleep(3)
        C.need(state['mode'] == m['arms'][a.arm]['mode'] and state['smoke'] == m['smoke'], 'serve identity differs from manifest')
        C.need(state['mil']['model_sha256'] == m['mil']['sha256'] and state['actor_sha256'] == m['arms'][a.arm].get('actor_sha256'),
               'serve pins differ from manifest')
        (out / f'{name}.identity.json').write_text(json.dumps(state, indent=1))
        if a.ready_flag:
            Path(a.ready_flag).touch()
        for row in todo:
            storage_guard()
            if env_image is not None:
                C.need(srv_env_preflight(out) == env_image, 'robocasa container image changed mid-run')
            attempt = next_attempt(out, row['n'])
            attempt.mkdir(parents=True)
            cn = f'{name}-c-n{row["n"]}-{attempt.name}'
            cmd = collect_command(m, row, out, attempt, a.gpu, a.port, cn, host, msha)
            q0 = get_json(a.port, 'qam_contract')
            t0 = time.time()
            with (attempt / 'collect.log').open('x') as f:
                if env is None:
                    _ensure_absent(cn)
                try:
                    subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=EPISODE_TIMEOUT + 60, check=True, cwd=str(C.R))
                finally:
                    if env is None:
                        _stop_owned(cn)
            q1 = get_json(a.port, 'qam_contract')
            item = audit_row(out, attempt, row, a.arm, m, q0, q1, out / f'{cn}.module.task_trace.jsonl')
            item.update({'fold_id': m['fold_id'], 'arm': a.arm, 'n': row['n'], 'j': row['j'], 'attempt': attempt.name,
                         'env_seed': row['env_seed'], 'inference_seed': row['inference_seed'], 'plan_id': row['plan_id'],
                         'base_success': row['base_success'], 'seconds': time.time() - t0, 'time': time.time(),
                         'manifest_sha256': msha, 'decisions_log': f'{name}.module.decisions.jsonl', 'trace': f'{cn}.module.task_trace.jsonl',
                         'harness_receipt': os.environ['HARNESS_RECEIPT'], 'host': host, 'gpu': a.gpu})
            progress.write(json.dumps(item, sort_keys=True) + '\n')
            print(json.dumps({k: item[k] for k in ('n', 'success', 'base_success', 'composed', 'first_fire_t')}), flush=True)
    finally:
        progress.close()
        if env is None:
            _stop_owned(name)
        else:
            try:
                os.killpg(proc.pid, 15)
            except ProcessLookupError:
                pass
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            if env is not None:
                os.killpg(proc.pid, 9)
            raise
        serve_log.close()
    return finish(out, m, a.arm)


def finish(out, m, arm):
    done = load_progress(out, m, arm)
    if m['smoke'] or len(done) != len(m['rows']):
        print(json.dumps({'status': 'partial' if not m['smoke'] else 'smoke', 'done': sorted(done)}))
        return 0
    C.write_json_new(out / 'RUN_COMPLETE.json', {'fold_id': m['fold_id'], 'arm': arm, 'rows': len(done),
                                                 'progress_sha256': C.sha256_path(out / 'progress.jsonl'),
                                                 'validation': 'pending verify_fold.py'})
    return 0


def _ensure_absent(name):
    r = subprocess.run(['docker', 'inspect', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
    C.need(r.returncode != 0, f'refusing pre-existing container name {name}')


def _stop_owned(name):
    subprocess.run(['docker', 'stop', '--time', '10', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True); ap.add_argument('--arm', required=True)
    ap.add_argument('--gpu', type=int, required=True); ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--rows', type=lambda s: [int(x) for x in s.split(',')], help='smoke only: row subset')
    ap.add_argument('--serve-py', default=C.SRV_SERVE_PY, help='srv48/50 host conda python')
    ap.add_argument('--ready-flag', type=Path, help='touched once the serve is ready (launcher boot serialization)')
    a = ap.parse_args(argv)
    try:
        return run(a)
    except Exception as e:
        out = Path(a.manifest).resolve().parent / a.arm / 'failures'
        if out.parent.parent.is_dir():
            out.mkdir(parents=True, exist_ok=True)
            (out / f'{time.time_ns()}.json').write_text(json.dumps({'error': repr(e), 'type': type(e).__name__, 'time': time.time()}, indent=1))
        print('RUN_FAILED', repr(e), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())

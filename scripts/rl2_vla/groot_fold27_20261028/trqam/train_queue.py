"""Durable fold x arm TRQAM scheduler (JAX training on the fold's collection host, GPUs reserved by the groot harness).

    GPUS=5 HARNESS_MACHINE=kanu HARNESS_SESSION=fold27-trqam bash scripts/rl2_vla/groot_fold27_20261028/trqam/train_queue.sh \
        --sources <sources.json> [--folds <fold_id> ...] [--per-gpu 2] [--cpu-jobs 4] [--smoke-steps 200]
State for watchers (eval queue): <out>/queue/QUEUE_STATE.json (running, prod_done), <out>/<fold>/<tag>/DONE.json.

sources.json: {"<instruction>|s<scene>": {"episodes": <compact dir>, "latents": <rollout_latents dir>}, ...}
A fold is READY when its scene has a latents dir with COMPLETE.json (host-matched conversion done) and a compact dir.
Per (fold, arm) stages, each skipped when its marker verifies (resume = rerun the same command):
  dry   finetune_fold --dry-run (CPU, JAX_PLATFORMS=cpu)                      -> <out>/<fold>/<tag>/DRYRUN.json
  smoke finetune_fold --smoke-steps N on GPU, ONE fold per arm, before any production of that arm
                                                                              -> <out>/<fold>/<tag>_smoke/SMOKE_DONE.json
  prod  finetune_fold production 50k on GPU (recipe fixed in finetune_fold) -> <out>/<fold>/<tag>/DONE.json
Arms/settings are finetune_fold's fixed recipe: basic | pbrs (alpha .1, lambda 5) | mix (alpha .1, beta .0396), AE16/K48.
First failure: no new launches, running jobs finish, <out>/queue/FAILED.json, exit 2 (rerun with --retry-failed).
"""
import argparse, hashlib, json, os, shlex, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
REPO = PKG.parents[2]
sys.path[:0] = [str(PKG)]
from folds import load_spec   # noqa: E402

ARMS = ('basic', 'pbrs', 'mix')
TAGS = {'basic': 'basic', 'pbrs': 'pbrs_a0.1', 'mix': 'mix_a0.1_b0.0396'}   # finetune_fold RECIPE_DEFAULTS run tags
COMMON_STEP = 500000


class QueueError(RuntimeError):
    pass


def scene_key(fold):
    return f'{fold["instruction"]}|s{fold["scene"]}'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


# exactly the files finetune_fold.fingerprint() hashes into every run's fingerprint['code']
FINGERPRINT_CODE = {'finetune_fold.py': HERE / 'finetune_fold.py', 'rewards.py': HERE / 'rewards.py',
                    'trqam_prefix.py': HERE / 'trqam_prefix.py', 'folds.py': PKG / 'folds.py'}
EXTRA_CODE = {'ae_potential.py': HERE / 'ae_potential.py', 'train_queue.py': Path(__file__).resolve(),
              'qam_prefix.py': REPO / 'scripts/rl2_vla/groot_robocasa_adapter_20261021/qam_prefix.py'}

HARNESS_LEDGER = REPO / 'outputs/gpu_leases/harness_slots.json'
KANU_GPU_CAP = 3          # docs/05: all sessions together, kanu


def harness_preflight(gpus, ledger=HARNESS_LEDGER, cap=KANU_GPU_CAP):
    """with_gpu_lease.sh only sees shell leases; GR00T harness reservations live in harness_slots.json. Never take a
    GPU reserved there, and keep (harness-reserved kanu GPUs + ours) within the all-session cap. Stale-looking
    reservations are NOT reclaimed here — their owner (or the harness) must release them."""
    slots = json.loads(Path(ledger).read_text()) if Path(ledger).exists() else {}
    reserved = {}
    for rid, s in slots.items():
        if s.get('machine') == 'kanu':
            for g in s.get('gpus', []):
                reserved[int(g)] = f'{s.get("session")}({s.get("kind")}, receipt {rid[:8]})'
    clash = {g: reserved[g] for g in gpus if g in reserved}
    if clash:
        raise QueueError(f'requested kanu GPUs reserved in the groot harness: {clash}')
    if len(set(reserved) | set(gpus)) > cap:
        raise QueueError(f'kanu all-session GPU cap {cap}: harness holds {sorted(reserved)} -> cannot add {sorted(gpus)}')
    return reserved


class TrainQueue:
    def __init__(self, a, popen=subprocess.Popen, sleep=time.sleep, now=time.time):
        self.a, self.popen, self.sleep, self.now = a, popen, sleep, now
        self.spec = load_spec(a.spec)
        self.sources = json.loads(a.sources.read_text())
        self.qdir = a.out / 'queue'; self.qdir.mkdir(parents=True, exist_ok=True)
        self.logs = self.qdir / 'logs'
        self.pins = self.freeze()

    def freeze(self):
        """Cross-run freeze: first launch pins code/spec/sources/init; later launches must match byte-for-byte."""
        pins = {'fingerprint_code': {k: sha(p) for k, p in FINGERPRINT_CODE.items()},
                'extra_code': {k: sha(p) for k, p in EXTRA_CODE.items()},
                'fold_spec_sha256': sha(self.a.spec), 'sources_sha256': sha(self.a.sources),
                'init_dir_name': self.a.init.name, 'init_params_sha256': sha(self.a.init / f'params_{COMMON_STEP}.pkl'),
                'smoke_steps': self.a.smoke_steps, 'arms_tags': TAGS}
        p = self.qdir / 'QUEUE_FREEZE.json'
        if p.exists():
            old = json.loads(p.read_text())
            if old != pins:
                diff = sorted(k for k in pins if old.get(k) != pins[k])
                raise QueueError(f'frozen queue inputs changed {diff}; use a new --out (no hyperparameter/code drift across runs)')
        else:
            p.write_text(json.dumps(pins, indent=1))
        return pins

    # -- readiness / markers
    def ready_folds(self):
        out = []
        for f in self.spec['folds']:
            if self.a.folds and f['fold_id'] not in self.a.folds:
                continue
            s = self.sources.get(scene_key(f))
            if s and (Path(s['latents']) / 'COMPLETE.json').exists() and (Path(s['episodes']) / 'contract.json').exists():
                out.append(f)
        return out

    def marker(self, fid, arm, stage):
        tag = TAGS[arm]
        return {'dry': self.a.out / fid / tag / 'DRYRUN.json', 'smoke': self.a.out / fid / f'{tag}_smoke' / 'SMOKE_DONE.json',
                'prod': self.a.out / fid / tag / 'DONE.json'}[stage]

    def done(self, fid, arm, stage):
        m = self.marker(fid, arm, stage)
        if not m.exists():
            return False
        if stage == 'dry':
            return True
        d = json.loads(m.read_text())
        steps = self.a.smoke_steps if stage == 'smoke' else 50000
        fp, ck = d.get('fingerprint') or {}, m.parent / 'checkpoint' / f'params_{steps}.pkl'
        init = fp.get('init') or {}
        return (bool(d.get('harness_receipt')) and int(fp.get('steps', -1)) == steps
                and fp.get('code') == self.pins['fingerprint_code']
                and int(init.get('step', -1)) == COMMON_STEP and init.get('params_sha256') == self.pins['init_params_sha256']
                and ck.exists() and sha(ck) == d.get('params_sha256'))

    # -- commands
    def cmd(self, f, arm, stage):
        s = self.sources[scene_key(f)]
        c = [self.a.python, str(HERE / 'finetune_fold.py'), '--spec', str(self.a.spec), '--fold', f['fold_id'], '--arm', arm,
             '--episodes', str(s['episodes']), '--latents', str(s['latents']), '--out', str(self.a.out)]
        if stage == 'dry':
            return c + ['--dry-run']
        c += ['--init', str(self.a.init), '--init-step', str(COMMON_STEP)]
        return c + (['--smoke-steps', str(self.a.smoke_steps)] if stage == 'smoke' else [])

    def env(self, gpu=None):
        e = dict(os.environ, OMP_NUM_THREADS=str(self.a.threads), MKL_NUM_THREADS=str(self.a.threads),
                 OPENBLAS_NUM_THREADS=str(self.a.threads), XLA_PYTHON_CLIENT_PREALLOCATE='false')
        if gpu is None:
            e.update(JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='')
        else:
            e.update(JAX_PLATFORMS='cuda', CUDA_VISIBLE_DEVICES=str(gpu))
        return e

    def pending(self, folds):
        """Next runnable jobs in dependency order: dry before smoke/prod; smoke for one fold per arm before any prod."""
        jobs = []
        for arm in self.a.arms:
            # a passed smoke anywhere under this --out counts (bounded --folds launches must not redo it)
            smoke_ok = any(self.done(f['fold_id'], arm, 'smoke') for f in self.spec['folds'])
            for i, f in enumerate(folds):
                fid = f['fold_id']
                if not self.done(fid, arm, 'dry'):
                    jobs.append((f, arm, 'dry')); continue
                if not smoke_ok:
                    if i == 0 and not self.a.skip_smoke:
                        jobs.append((f, arm, 'smoke'))
                    if not self.a.skip_smoke:
                        continue
                if not self.done(fid, arm, 'prod'):
                    jobs.append((f, arm, 'prod'))
        return jobs

    def run(self):
        if (self.qdir / 'FAILED.json').exists() and not self.a.retry_failed:
            raise QueueError('previous FAILED.json present; inspect, then --retry-failed')
        if not self.a.dry_only:
            if os.environ.get('HARNESS_RECEIPT'):
                pass                                   # groot harness reserved exactly these GPUs (central caps applied)
            elif os.environ.get('FOLD27_GPU_LEASE'):
                harness_preflight(self.a.gpus, self.a.harness_ledger)   # shell-lease path: never take harness GPUs
            else:
                raise QueueError('GPU stages need a reservation: launch via train_queue.sh (groot harness guard)')
        ready = {f['fold_id'] for f in self.ready_folds()}
        if not ready:
            raise QueueError('no READY fold (latents COMPLETE + compact contract); nothing to train')
        if self.a.folds and set(self.a.folds) - ready:
            raise QueueError(f'requested folds not READY (latents/compact missing): {sorted(set(self.a.folds) - ready)}')
        gpu_slots = [g for g in self.a.gpus for _ in range(self.a.per_gpu)]
        running, failed = {}, None
        while True:
            folds = self.ready_folds()
            todo = [j for j in self.pending(folds) if (j[0]['fold_id'], j[1], j[2]) not in running]
            if self.a.dry_only:
                todo = [j for j in todo if j[2] == 'dry']
            if failed:
                todo = []
            busy_gpu = [r['gpu'] for r in running.values() if r['gpu'] is not None]
            cpu_busy = sum(1 for r in running.values() if r['gpu'] is None)
            for f, arm, stage in todo:
                key = (f['fold_id'], arm, stage)
                if any(k[:2] == key[:2] for k in running):
                    continue                               # one stage per (fold, arm) at a time
                if stage == 'dry' and any(k[0] == key[0] and k[2] == 'dry' for k in running):
                    continue                               # same-fold dry-runs serialized: the shared AE potential is fit once
                if stage == 'dry':
                    if cpu_busy >= self.a.cpu_jobs:
                        continue
                    gpu = None; cpu_busy += 1
                else:
                    free = list(gpu_slots)
                    for g in busy_gpu:
                        free.remove(g)
                    if not free:
                        continue
                    gpu = free[0]; busy_gpu.append(gpu)
                d = self.logs / stage; d.mkdir(parents=True, exist_ok=True)
                log = d / f'{key[0]}.{arm}.{1 + sum(1 for _ in d.glob(f"{key[0]}.{arm}.*.log"))}.log'
                with open(log, 'x') as fh:
                    fh.write('# ' + ' '.join(map(shlex.quote, self.cmd(f, arm, stage))) + '\n'); fh.flush()
                    running[key] = {'proc': self.popen(self.cmd(f, arm, stage), stdout=fh, stderr=subprocess.STDOUT, env=self.env(gpu)),
                                    'gpu': gpu, 'log': str(log)}
            for key, r in list(running.items()):
                rc = r['proc'].poll()
                if rc is None:
                    continue
                del running[key]
                if rc != 0 or not self.done(*key):
                    failed = failed or {'job': list(key), 'rc': rc, 'log': r['log']}
            self.state(running, failed, folds)
            if not running and (failed or not [j for j in self.pending(folds) if not self.a.dry_only or j[2] == 'dry']):
                break
            self.sleep(self.a.poll_s)
        if failed:
            (self.qdir / 'FAILED.json').write_text(json.dumps({**failed, 'time': self.now()}, indent=1))
            raise QueueError(f'job failed: {failed}')
        folds = self.ready_folds()
        total = len(folds) * len(self.a.arms)
        prod = sum(self.done(f['fold_id'], arm, 'prod') for f in folds for arm in self.a.arms)
        return {'ready_folds': len(folds), 'prod_done': prod, 'prod_total_ready': total,
                'not_ready': sorted({scene_key(f) for f in self.spec['folds']} - {scene_key(f) for f in folds})}

    def state(self, running, failed, folds):
        st = {'time': self.now(), 'running': [list(k) + [r['gpu']] for k, r in running.items()], 'failed': failed,
              'ready_folds': [f['fold_id'] for f in folds],
              'prod_done': sorted(f'{f["fold_id"]}/{arm}' for f in folds for arm in self.a.arms if self.done(f['fold_id'], arm, 'prod'))}
        tmp = self.qdir / 'QUEUE_STATE.tmp'; tmp.write_text(json.dumps(st, indent=1)); os.replace(tmp, self.qdir / 'QUEUE_STATE.json')


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', type=Path, default=REPO / 'outputs/analysis/groot_fold27_20261028/folds/fold_spec.json')
    ap.add_argument('--sources', type=Path, required=True)
    ap.add_argument('--init', type=Path, default=REPO / 'outputs/rl2_vla/atomic65_common_qam_work_20261029/scratch_run/checkpoints/step_0500000')
    ap.add_argument('--out', type=Path, default=REPO / 'outputs/rl2_vla/fold27_trqam_20261102')
    ap.add_argument('--gpus', nargs='+', type=int, default=[])
    ap.add_argument('--per-gpu', type=int, default=2); ap.add_argument('--cpu-jobs', type=int, default=4)
    ap.add_argument('--threads', type=int, default=2); ap.add_argument('--smoke-steps', type=int, default=200)
    ap.add_argument('--arms', nargs='+', default=list(ARMS), choices=ARMS); ap.add_argument('--folds', nargs='+')
    ap.add_argument('--python', default=str(REPO / '.venvs/rl2_iid/bin/python'))
    ap.add_argument('--poll-s', type=float, default=30); ap.add_argument('--retry-failed', action='store_true')
    ap.add_argument('--dry-only', action='store_true', help='CPU dry-runs only (no GPU, no lease needed)')
    ap.add_argument('--skip-smoke', action='store_true', help='only if an equivalent smoke already passed elsewhere')
    ap.add_argument('--harness-ledger', type=Path, default=HARNESS_LEDGER)
    a = ap.parse_args(argv)
    if not a.gpus and os.environ.get('GPUS'):
        a.gpus = [int(g) for g in os.environ['GPUS'].replace(' ', ',').split(',') if g]   # set by the harness
    if not a.dry_only and not a.gpus:
        ap.error('--gpus (or harness GPUS) required unless --dry-only')
    if not (a.init / f'params_{COMMON_STEP}.pkl').exists():
        ap.error(f'common checkpoint params_{COMMON_STEP}.pkl missing in {a.init}')
    cap = {'kanu': 3, 'srv48': 1, 'srv50': 1}.get(os.environ.get('HARNESS_MACHINE', 'kanu'), 1)
    if len(a.gpus) > cap or a.per_gpu < 1:
        ap.error(f'machine cap: at most {cap} GPUs (docs/05), per-gpu >= 1')
    return a


def main():
    a = parse()
    try:
        print(json.dumps(TrainQueue(a).run()))
    except QueueError as e:
        print('QUEUE_ERROR', e, file=sys.stderr); sys.exit(2)


if __name__ == '__main__':
    main()

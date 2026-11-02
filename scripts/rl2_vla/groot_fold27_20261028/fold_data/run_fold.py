"""Per-fold orchestration (CPU stages runnable now; GPU training only with --production under the groot harness).

    python run_fold.py --spec fold_spec.json --fold ID --work <root> --episodes <compact or full extract dir> \
        [--remote-dir <승준 extract dir>]  [--latents <rollout_latents dir>] [--arms basic pbrs mix] \
        [--py-mil <python with torch+pandas>] [--py-trqam <.venvs/rl2_iid/bin/python>] [--production --init DIR]

Stages (in order, each skipped when its completion marker verifies):
  stage            stage_episodes.stage (only with --remote-dir): fold's 160 fit + 40 eval records, sha-verified
  mil_shards       mil/fold_mil.py shards        -> <work>/mil_shards/<fold>/SHARD_COMPLETE.json
  mil_train        mil/fold_mil.py train seed 0  -> <work>/mil_runs/<fold>/attn/raw/s0/{model.pt,FOLD_CONTRACT.json}
  trqam_dry_<arm>  trqam/finetune_fold.py --dry-run -> <work>/trqam/<fold>/<tag>/DRYRUN.json   (needs --latents COMPLETE)
  trqam_<arm>      trqam/finetune_fold.py production -> DONE.json (only --production + HARNESS_RECEIPT + --init)
Exit codes: 0 done, 2 failed (ORCH_FAILED.json, later stages not run), 3 waiting for GPU latents (ORCH_STATUS state=waiting).
Logs: <logs>/<stage>/<fold>.<attempt>.log, attempt = 1 + existing attempts (never overwritten).
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path[:0] = [str(PKG), str(HERE)]
from folds import load_spec   # noqa: E402

ARMS = ('basic', 'pbrs', 'mix')
RECIPE_TAGS = {'basic': 'basic', 'pbrs': 'pbrs_a0.1', 'mix': 'mix_a0.1_b0.0396'}   # finetune_fold RECIPE_DEFAULTS tags
EXIT_WAITING = 3


class OrchError(RuntimeError):
    pass


PROOF_KEYS = ('train_val_equals_fit160', 'train_val_disjoint_eval40', 'target_j_absent_from_train_val')   # fold_mil.heldout_proof


def mil_committed(run):
    """Same commit rule as fold_mil.verify_committed (contract last, model sha, held-out proof present)."""
    import hashlib
    cp, mp = run / 'FOLD_CONTRACT.json', run / 'model.pt'
    if not cp.exists() or not mp.exists():
        return False
    c = json.loads(cp.read_text())
    proof = c.get('heldout_proof') or {}
    return all(proof.get(k) is True for k in PROOF_KEYS) and hashlib.sha256(mp.read_bytes()).hexdigest() == c.get('model_sha256')


def next_log(logs, stage, fold):
    d = logs / stage
    d.mkdir(parents=True, exist_ok=True)
    n = 1 + sum(1 for _ in d.glob(f'{fold}.*.log'))
    return d / f'{fold}.{n}.log'


def subprocess_runner(cmd, log_path, env=None):
    with open(log_path, 'x') as f:
        f.write(json.dumps({'cmd': cmd, 'time': time.time()}) + '\n'); f.flush()
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode


class FoldRun:
    def __init__(self, a, runner=subprocess_runner, stager=None):
        self.a, self.runner, self.stager = a, runner, stager
        self.fold = a.fold
        self.status_dir = a.work / 'orch' / self.fold
        self.status_dir.mkdir(parents=True, exist_ok=True)
        self.logs = a.logs

    def status(self, **kw):
        (self.status_dir / 'ORCH_STATUS.json').write_text(json.dumps({'fold_id': self.fold, 'time': time.time(), **kw}, indent=1))

    def run_cmd(self, stage, cmd, env=None):
        log = next_log(self.logs, stage, self.fold)
        self.status(stage=stage, state='running', log=str(log))
        rc = self.runner(cmd, log, env)
        if rc != 0:
            (self.status_dir / 'ORCH_FAILED.json').write_text(json.dumps({'stage': stage, 'rc': rc, 'log': str(log), 'time': time.time()}, indent=1))
            self.status(stage=stage, state='failed', rc=rc, log=str(log))
            raise OrchError(f'{stage} rc={rc} (log {log})')

    def stages(self):
        a, f = self.a, self.fold
        mil_shards = a.work / 'mil_shards'; mil_runs = a.work / 'mil_runs'; trq = a.work / 'trqam'
        st = []
        if a.remote_dir:
            st.append(('stage', None, lambda: self.stager(f), lambda: False))
        st.append(('mil_shards', [a.py_mil, str(PKG / 'mil/fold_mil.py'), 'shards', '--spec', str(a.spec), '--episodes', str(a.episodes),
                                  '--out', str(mil_shards), '--fold', f], None, lambda: (mil_shards / f / 'SHARD_COMPLETE.json').exists()))
        run = mil_runs / f / 'attn/raw/s0'
        st.append(('mil_train', [a.py_mil, str(PKG / 'mil/fold_mil.py'), 'train', '--spec', str(a.spec), '--shards', str(mil_shards),
                                 '--out', str(mil_runs), '--fold', f, '--seed', '0', '--device', 'cpu'], None,
                   lambda: mil_committed(run)))
        base = [str(PKG / 'trqam/finetune_fold.py'), '--spec', str(a.spec), '--fold', f, '--episodes', str(a.episodes),
                '--latents', str(a.latents), '--out', str(trq)]
        for arm in a.arms:
            tag = trq / f / RECIPE_TAGS[arm]
            st.append((f'trqam_dry_{arm}', [a.py_trqam, *base, '--arm', arm, '--dry-run'], None, lambda t=tag: (t / 'DRYRUN.json').exists()))
        if a.production:
            for arm in a.arms:
                tag = trq / f / RECIPE_TAGS[arm]
                st.append((f'trqam_{arm}', [a.py_trqam, *base, '--arm', arm, '--init', str(a.init), '--init-step', str(a.init_step)], None,
                           lambda t=tag: (t / 'DONE.json').exists()))
        return st

    def latents_ready(self):
        lat = self.a.latents
        return lat is not None and (lat / 'COMPLETE.json').exists()

    def run(self):
        if (self.status_dir / 'ORCH_FAILED.json').exists() and not self.a.retry_failed:
            raise OrchError('previous attempt failed (ORCH_FAILED.json); inspect, then rerun with --retry-failed')
        if self.a.production and not os.environ.get('HARNESS_RECEIPT'):
            raise OrchError('--production requires HARNESS_RECEIPT (groot harness); GPU launch is the main session decision')
        if self.a.production and self.a.init is None:
            raise OrchError('--production requires --init (common atomic65 checkpoint)')
        done = []
        for name, cmd, fn, is_done in self.stages():
            if is_done():
                done.append(name); continue
            if name.startswith('trqam') and not self.latents_ready():
                self.status(stage=name, state='waiting', reason='GPU rollout latents not COMPLETE', done=done)
                return EXIT_WAITING
            if fn is not None:
                self.status(stage=name, state='running')
                try:
                    fn()
                except Exception as e:
                    (self.status_dir / 'ORCH_FAILED.json').write_text(json.dumps({'stage': name, 'error': repr(e), 'time': time.time()}, indent=1))
                    raise OrchError(f'{name}: {e!r}')
            else:
                self.run_cmd(name, cmd)
            if not is_done():
                raise OrchError(f'{name} returned 0 but its completion marker is missing')
            done.append(name)
        self.status(stage='all', state='done', done=done)
        if (self.status_dir / 'ORCH_FAILED.json').exists():
            os.replace(self.status_dir / 'ORCH_FAILED.json', self.status_dir / f'ORCH_FAILED.resolved.{int(time.time())}.json')
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', type=Path, required=True); ap.add_argument('--fold', required=True)
    ap.add_argument('--work', type=Path, required=True); ap.add_argument('--episodes', type=Path, required=True)
    ap.add_argument('--logs', type=Path, default=PKG.parents[2] / 'outputs/analysis/groot_fold27_20261028/logs')
    ap.add_argument('--remote-dir'); ap.add_argument('--latents', type=Path)
    ap.add_argument('--arms', nargs='+', default=list(ARMS), choices=ARMS)
    ap.add_argument('--py-mil', default=sys.executable); ap.add_argument('--py-trqam', default=sys.executable)
    ap.add_argument('--production', action='store_true'); ap.add_argument('--init', type=Path); ap.add_argument('--init-step', type=int, default=500000)
    ap.add_argument('--retry-failed', action='store_true'); ap.add_argument('--reserve-gib', type=float, default=20.0)
    a = ap.parse_args()
    spec = load_spec(a.spec)
    if a.fold not in {f['fold_id'] for f in spec['folds']}:
        raise OrchError(f'unknown fold {a.fold}')
    stager = None
    if a.remote_dir:
        import stage_episodes as SE
        stager = lambda fold: print(json.dumps(SE.stage(spec, SE.WrapperFetcher(), a.remote_dir, a.episodes, folds=[fold], reserve_gib=a.reserve_gib)))
    sys.exit(FoldRun(a, stager=stager).run())


if __name__ == '__main__':
    try:
        main()
    except OrchError as e:
        print('ORCH_ERROR', e, file=sys.stderr); sys.exit(2)

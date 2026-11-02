"""Bounded, durable continuation for the remaining fold MIL runs (main-local coordinator; no GPU, no TRQAM).

Phases (state in <root>/coord/COORD_STATE.json; every phase verifies its own completion, so a rerun resumes):
  freeze   pin sha256 of folds.py, stage_episodes.py, compact_export.py, mil/fold_mil.py, fold spec; a rerun with
           different bytes refuses (new --root needed). Verifies the expected number of already-committed MIL runs
           (default 15) and the expected remaining count (default 12); committed runs are never touched.
  wait     poll 승준 via scripts/utils/remote_compute.sh only: COMPLETE.json present with episodes == --expect-episodes
           -> continue; no extract_scene process for --remote-extract and no COMPLETE -> ALERT.json + exit 2;
           bounded by --max-wait-h.
  compact  remote compact_export.py (deployed copy, sha must equal the pinned local frozen file) unless the remote
           compact dir already has all records; then record count re-checked remotely.
  pull     one tar stream of the compact dir via the wrapper -> <root>/compact_raw (safe extraction, no links).
  stage    stage_episodes.stage(LocalFetcher) raw -> <root>/compact_verified for the remaining folds (hash gate).
  mil      per remaining fold: fold_mil.py shards, then train --seed 0 --device cpu; <= --jobs (default/cap 12) processes,
           OMP/MKL/OPENBLAS threads = --threads, CUDA_VISIBLE_DEVICES=''. First failure: no new launches, running jobs
           finish, FAILED.json, exit 2. Done = fold_data/run_fold.mil_committed (contract + model sha + held-out proof).

    .venvs/rl2_iid/bin/python mil_continuation.py --spec <fold_spec.json> --root outputs/rl2_vla/fold27_mil_20261030 \
        --remote-code <deployed scripts/rl2_vla/groot_fold27_20261028> --remote-extract <abs remaining_a_v4> \
        --remote-compact <abs remaining_a_v4_compact> [--dry-run]
"""
import argparse, hashlib, json, os, shlex, subprocess, sys, tarfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
REPO = PKG.parents[2]
sys.path[:0] = [str(PKG), str(HERE)]
from folds import load_spec, validate_fold   # noqa: E402
import stage_episodes as SE                  # noqa: E402
from run_fold import mil_committed, next_log  # noqa: E402

WRAPPER = REPO / 'scripts/utils/remote_compute.sh'
FROZEN = {'folds.py': PKG / 'folds.py', 'stage_episodes.py': HERE / 'stage_episodes.py',
          'compact_export.py': HERE / 'compact_export.py', 'fold_mil.py': PKG / 'mil/fold_mil.py',
          'run_fold.py': HERE / 'run_fold.py', 'mil_continuation.py': Path(__file__).resolve(),
          'remote_compute.sh': WRAPPER}
EXIT_FAILED, EXIT_ALERT = 2, 2


class CoordError(RuntimeError):
    pass


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


class WrapperRemote:
    """All remote access through remote_compute.sh `run` (stdout byte-clean). Paths must be absolute."""
    def _run(self, cmd, timeout=600, stdout=None):
        p = subprocess.run(['bash', str(WRAPPER), 'run', cmd], stdout=stdout or subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        return p.returncode, (p.stdout if stdout is None else b''), p.stderr

    def read(self, path):
        rc, out, err = self._run('cat -- ' + shlex.quote(path))
        if rc:
            raise CoordError(f'remote read {path}: {err.decode(errors="replace")[-200:]}')
        return out

    def exists(self, path):
        return self._run('test -e ' + shlex.quote(path))[0] == 0

    def count_records(self, d):
        rc, out, _ = self._run(f'ls -1 {shlex.quote(d)} | grep -c "\\.json$"')
        return int(out.strip() or 0) - (1 if self.exists(d + '/contract.json') else 0) - (1 if self.exists(d + '/COMPLETE.json') else 0)

    def extract_running(self, out_dir):
        # bracket trick: the pattern never matches pgrep's own command line (ssh pkill/pgrep self-match pitfall)
        rc, out, _ = self._run('pgrep -f ' + shlex.quote('[e]xtract_scene.py.*' + out_dir) + ' | wc -l')
        return int(out.strip() or 0)

    def sha256(self, path):
        rc, out, err = self._run('sha256sum -- ' + shlex.quote(path))
        if rc:
            raise CoordError(f'remote sha {path}: {err.decode(errors="replace")[-200:]}')
        return out.split()[0].decode()

    def run_py(self, args, log):
        """Runs `"$REMOTE_PYTHON" <args>` (the wrapper exports REMOTE_PYTHON = anaconda python with torch)."""
        with open(log, 'x') as f:
            p = subprocess.run(['bash', str(WRAPPER), 'run', '"$REMOTE_PYTHON" ' + ' '.join(map(shlex.quote, args))],
                               stdout=f, stderr=subprocess.STDOUT)
        return p.returncode

    def tar_to(self, d, dst):
        with open(dst, 'xb') as f:
            rc, _, err = self._run('tar -C ' + shlex.quote(d) + ' -cf - .', timeout=3600, stdout=f)
        if rc:
            raise CoordError(f'remote tar {d}: {err.decode(errors="replace")[-200:]}')


def safe_extract(tar_path, dst):
    dst.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        for m in tf.getmembers():
            t = (dst / m.name).resolve()
            if not (m.isfile() or m.isdir()) or not str(t).startswith(str(dst.resolve())):
                raise CoordError(f'unsafe tar member {m.name}')
        tf.extractall(dst)


class Coordinator:
    def __init__(self, a, remote, popen=subprocess.Popen, sleep=time.sleep, now=time.time):
        self.a, self.remote, self.popen, self.sleep, self.now = a, remote, popen, sleep, now
        self.root = a.root
        self.coord = self.root / 'coord'; self.coord.mkdir(parents=True, exist_ok=True)
        self.state_p = self.coord / 'COORD_STATE.json'
        self.state = json.loads(self.state_p.read_text()) if self.state_p.exists() else {'phases': {}}
        self.logs = self.root / 'logs'
        self.spec = load_spec(a.spec)
        self.runs, self.shards = a.mil_runs, self.root / 'mil_shards'   # runs root holds the 15 committed folds
        self.raw, self.verified = self.root / 'compact_raw', self.root / 'compact_verified'

    def save(self, **kw):
        self.state.update(kw, time=self.now())
        tmp = self.state_p.with_suffix('.tmp'); tmp.write_text(json.dumps(self.state, indent=1)); os.replace(tmp, self.state_p)

    def mark(self, phase, **info):
        self.state['phases'][phase] = {'done': True, **info}; self.save()

    def done(self, phase):
        return self.state['phases'].get(phase, {}).get('done', False)

    def fail(self, phase, err, name='FAILED.json'):
        (self.coord / name).write_text(json.dumps({'phase': phase, 'error': str(err), 'time': self.now()}, indent=1))
        self.save(status='failed', failed_phase=phase)
        raise CoordError(f'{phase}: {err}')

    # -- phases
    def remaining_folds(self):
        committed = [f['fold_id'] for f in self.spec['folds'] if mil_committed(self.runs / f['fold_id'] / 'attn/raw/s0')]
        rem = [f for f in self.spec['folds'] if f['fold_id'] not in committed]
        return committed, rem

    def phase_freeze(self):
        pins = {k: sha(p) for k, p in FROZEN.items()}
        pins['fold_spec'] = sha(self.a.spec)
        committed, rem = self.remaining_folds()
        if 'pins' in self.state and self.state['pins'] != pins:
            self.fail('freeze', f'frozen sources changed since this root started: {self.state["pins"]} vs {pins}')
        if not self.done('freeze'):
            if len(committed) != self.a.expect_committed or len(rem) != self.a.expect_remaining:
                self.fail('freeze', f'committed {len(committed)} / remaining {len(rem)} != expected '
                                    f'{self.a.expect_committed} / {self.a.expect_remaining}')
            for f in rem:
                validate_fold(f)
            self.state['pins'] = pins
            self.mark('freeze', committed=sorted(committed), remaining=sorted(f['fold_id'] for f in rem),
                      committed_fingerprints=self.fingerprints(committed))

    def fingerprints(self, folds):
        out = {}
        for fid in folds:
            run = self.runs / fid / 'attn/raw/s0'
            out[fid] = {'contract_sha256': sha(run / 'FOLD_CONTRACT.json'), 'model_sha256': sha(run / 'model.pt')}
        return out

    def check_committed_untouched(self):
        st = self.state['phases']['freeze']
        if self.fingerprints(st['committed']) != st['committed_fingerprints']:
            self.fail('mil', 'a pre-existing committed MIL run changed during the continuation')

    def phase_wait(self):
        if self.done('wait'):
            return
        deadline = self.now() + self.a.max_wait_h * 3600
        x = self.a.remote_extract
        while True:
            if self.remote.exists(x + '/COMPLETE.json'):
                c = json.loads(self.remote.read(x + '/COMPLETE.json'))
                if c.get('episodes') != self.a.expect_episodes:
                    self.fail('wait', f'remote COMPLETE episodes {c.get("episodes")} != {self.a.expect_episodes}')
                self.mark('wait', remote_complete=c, contract_sha256=hashlib.sha256(self.remote.read(x + '/contract.json')).hexdigest())
                return
            if self.remote.extract_running(x) == 0:
                self.fail('wait', 'no extract_scene process for the remote out dir and no COMPLETE.json', 'ALERT.json')
            if self.now() > deadline:
                self.fail('wait', f'remote extraction not COMPLETE within {self.a.max_wait_h} h', 'ALERT.json')
            self.save(status='waiting_remote')
            self.sleep(self.a.poll_s)

    def phase_compact(self):
        if self.done('compact'):
            return
        code = self.a.remote_code + '/fold_data/compact_export.py'
        if self.remote.sha256(code) != self.state['pins']['compact_export.py']:
            self.fail('compact', 'remote compact_export.py differs from the pinned frozen file')
        n = self.a.expect_episodes
        if not (self.remote.exists(self.a.remote_compact + '/contract.json') and self.remote.count_records(self.a.remote_compact) == n):
            log = next_log(self.logs, 'compact', 'remote'); self.save(status='compact_running', log=str(log))
            if self.remote.run_py([code, '--src', self.a.remote_extract, '--out', self.a.remote_compact], log):
                self.fail('compact', f'remote compact_export failed (log {log})')
        got = self.remote.count_records(self.a.remote_compact)
        cc = json.loads(self.remote.read(self.a.remote_compact + '/contract.json'))
        if got != n or cc.get('source_contract_sha256') != self.state['phases']['wait']['contract_sha256']:
            self.fail('compact', f'compact records {got}/{n} or source contract mismatch')
        self.mark('compact', records=got)

    def phase_pull(self):
        if self.done('pull'):
            return
        tar = self.coord / f'compact_{int(self.now())}.tar'
        self.remote.tar_to(self.a.remote_compact, tar)
        tmp = self.raw.with_name(self.raw.name + '.partial')
        if tmp.exists():
            self.fail('pull', f'{tmp} exists from an interrupted pull; inspect and remove manually')
        if self.raw.exists():
            self.fail('pull', f'{self.raw} already exists without a recorded pull; inspect manually')
        safe_extract(tar, tmp)
        os.replace(tmp, self.raw)
        tar.unlink()
        self.mark('pull', files=sum(1 for _ in self.raw.iterdir()))

    def phase_stage(self):
        if self.done('stage'):
            return
        rem = self.state['phases']['freeze']['remaining']
        s = SE.stage(self.spec, SE.LocalFetcher(), str(self.raw), self.verified, folds=rem, reserve_gib=self.a.reserve_gib,
                     log=lambda *_: None)
        self.mark('stage', **s)

    def jobs(self):
        py, fm = self.a.python, str(PKG / 'mil/fold_mil.py')
        out = []
        for fid in self.state['phases']['freeze']['remaining']:
            run = self.runs / fid / 'attn/raw/s0'
            shard_done = (self.shards / fid / 'SHARD_COMPLETE.json').exists()
            out.append((fid, [py, fm, 'shards', '--spec', str(self.a.spec), '--episodes', str(self.verified), '--out', str(self.shards), '--fold', fid],
                        [py, fm, 'train', '--spec', str(self.a.spec), '--shards', str(self.shards), '--out', str(self.runs),
                         '--fold', fid, '--seed', '0', '--device', 'cpu'], shard_done, mil_committed(run)))
        return out

    def phase_mil(self):
        if self.done('mil'):
            return
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS=str(self.a.threads),
                   MKL_NUM_THREADS=str(self.a.threads), OPENBLAS_NUM_THREADS=str(self.a.threads))
        queue = [(fid, sh, tr, sd) for fid, sh, tr, sd, committed in self.jobs() if not committed]
        running, failed = {}, None
        while queue or running:
            while queue and len(running) < self.a.jobs and failed is None:
                fid, sh, tr, sd = queue.pop(0)
                log = next_log(self.logs, 'mil_continuation', fid)
                cmd = ['bash', '-c', ('' if sd else ' '.join(map(shlex.quote, sh)) + ' && ') + ' '.join(map(shlex.quote, tr))]
                with open(log, 'x') as f:
                    running[fid] = (self.popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env), log)
                self.save(status='mil_running', running=sorted(running))
            for fid, (p, log) in list(running.items()):
                rc = p.poll()
                if rc is None:
                    continue
                del running[fid]
                if rc != 0 or not mil_committed(self.runs / fid / 'attn/raw/s0'):
                    failed = failed or (fid, rc, str(log))
            if failed and not running:
                queue.clear()
            if running:
                self.sleep(self.a.poll_job_s)
        self.check_committed_untouched()
        if failed:
            self.fail('mil', f'fold {failed[0]} rc={failed[1]} or not committed (log {failed[2]})')
        committed, rem = self.remaining_folds()
        if rem:
            self.fail('mil', f'still uncommitted: {[f["fold_id"] for f in rem]}')
        self.mark('mil', committed=len(committed))

    def run(self):
        if (self.coord / 'FAILED.json').exists() and not self.a.retry_failed:
            raise CoordError('previous FAILED.json present; inspect, then --retry-failed')
        for name in ('freeze', 'wait', 'compact', 'pull', 'stage', 'mil'):
            if self.a.dry_run and name != 'freeze':
                break
            getattr(self, f'phase_{name}')()
        self.save(status='dry_run_ok' if self.a.dry_run else 'done')
        return 0


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', type=Path, required=True)
    ap.add_argument('--root', type=Path, default=REPO / 'outputs/rl2_vla/fold27_mil_20261030_continuation',
                    help='coordinator work dir (state, compact raw/verified, shards, logs)')
    ap.add_argument('--mil-runs', type=Path, default=REPO / 'outputs/rl2_vla/fold27_mil_20261030',
                    help='MIL run root: <fold>/attn/raw/s0 (15 committed runs live here and are never modified)')
    ap.add_argument('--remote-code', required=True); ap.add_argument('--remote-extract', required=True)
    ap.add_argument('--remote-compact', required=True)
    ap.add_argument('--expect-episodes', type=int, default=800); ap.add_argument('--expect-committed', type=int, default=15)
    ap.add_argument('--expect-remaining', type=int, default=12)
    ap.add_argument('--jobs', type=int, default=12); ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--python', default=str(REPO / '.venvs/rl2_iid/bin/python'))
    ap.add_argument('--poll-s', type=float, default=300); ap.add_argument('--poll-job-s', type=float, default=30)
    ap.add_argument('--max-wait-h', type=float, default=24); ap.add_argument('--reserve-gib', type=float, default=20)
    ap.add_argument('--retry-failed', action='store_true'); ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(argv)
    for p in (a.remote_code, a.remote_extract, a.remote_compact):
        if not p.startswith('/'):
            ap.error(f'remote path must be absolute: {p}')
    if not 1 <= a.jobs <= 12 or a.threads < 1:   # main-measured: 12 x 2 threads + baseline < 64 cores, ~0.7 GB/proc
        ap.error('--jobs must be 1..12 and --threads >= 1')
    return a


def main():
    a = parse()
    try:
        sys.exit(Coordinator(a, WrapperRemote()).run())
    except CoordError as e:
        print('COORD_ERROR', e, file=sys.stderr); sys.exit(EXIT_FAILED)


if __name__ == '__main__':
    main()

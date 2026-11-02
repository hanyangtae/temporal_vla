"""Episode-by-episode staging of extract_scene derived data from 승준 to a compute host (resumable, sha-verified).

Only the episodes a fold (or scene) needs are pulled, one record at a time (json, then npz streamed to tmp, hashed,
renamed) — no bulk copy, no change to the remote originals or remote derived files. Per record:
  json.episode_id == fold coordinate id, json.pkl_sha256 == fold-spec pkl_sha256 (archive meta), sha(npz) == json.npz_sha256.
The local dir carries the remote extraction contract.json verbatim; a different contract aborts (new local dir).
Disk guard: stop before an episode if free space < --reserve-gib.

    python stage_episodes.py --spec fold_spec.json --remote-dir <승준 extract dir> --local-dir <dir> \
        (--fold ID ... | --scene 'PPCC/bread|s4' ...) [--roles fit eval]
"""
import argparse, hashlib, json, os, shlex, shutil, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from folds import load_spec   # noqa: E402


class StageError(RuntimeError):
    pass


def episode_id(plan, machine, instr, s, j, n):
    return f'{plan}/{machine}/{instr}/s{s}/j{j}/n{n}'


def stem(eid):
    return hashlib.sha256(eid.encode()).hexdigest()[:24]      # == extract_scene naming


class WrapperFetcher:
    """Reads remote files ONLY through scripts/utils/remote_compute.sh `run` (project rule: no hand-rolled ssh).
    `run` does `cd $REMOTE_REPO && export <thread caps> && <cmd>` — stdout is the command's bytes unchanged.
    Paths must be absolute (the wrapper cds into the remote repo)."""
    WRAPPER = Path(__file__).resolve().parents[4] / 'scripts/utils/remote_compute.sh'

    def _cmd(self, path):
        if not str(path).startswith('/'):
            raise StageError(f'remote path must be absolute: {path}')
        return ['bash', str(self.WRAPPER), 'run', 'cat -- ' + shlex.quote(str(path))]

    def read(self, path):
        p = subprocess.run(self._cmd(path), capture_output=True, timeout=300)
        if p.returncode:
            raise StageError(f'remote cat {path}: rc={p.returncode} {p.stderr.decode(errors="replace")[-300:]}')
        return p.stdout

    def stream(self, path, dst):
        with open(dst, 'xb') as f:
            p = subprocess.run(self._cmd(path), stdout=f, stderr=subprocess.PIPE, timeout=3600)
        if p.returncode:
            raise StageError(f'remote cat {path}: rc={p.returncode} {p.stderr.decode(errors="replace")[-300:]}')


class LocalFetcher:
    """Same interface over a local directory (tests / same-host staging)."""
    def read(self, path):
        return Path(path).read_bytes()

    def stream(self, path, dst):
        with open(path, 'rb') as s, open(dst, 'xb') as d:
            shutil.copyfileobj(s, d, 1 << 22)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def needed(spec, folds=None, scenes=None, roles=('fit', 'eval')):
    out = {}
    for f in spec['folds']:
        tag = f'{f["instruction"]}|s{f["scene"]}'
        if (folds and f['fold_id'] not in folds) or (scenes and tag not in scenes):
            continue
        for role in roles:
            for r in f[role]:
                eid = episode_id(r['plan_id'], f['archive_machine'], f['instruction'], f['scene'], r['j'], r['n'])
                out[eid] = r
    if not out:
        raise StageError('no episodes selected')
    return out


def stage(spec, fetcher, remote_dir, local_dir, *, folds=None, scenes=None, roles=('fit', 'eval'), reserve_gib=20.0, log=print):
    local_dir.mkdir(parents=True, exist_ok=True)
    rc = fetcher.read(f'{remote_dir}/contract.json')
    lc = local_dir / 'contract.json'
    if lc.exists():
        if lc.read_bytes() != rc:
            raise StageError('local dir holds a different extraction contract; use a new --local-dir')
    else:
        lc.write_bytes(rc)
    eps, done, pulled = needed(spec, folds, scenes, roles), 0, 0
    for eid, r in sorted(eps.items()):
        st = stem(eid)
        lj, ln = local_dir / f'{st}.json', None
        if lj.exists():
            rec = json.loads(lj.read_text())
            ln = local_dir / rec['npz']
            if rec['episode_id'] == eid and rec['pkl_sha256'] == r['pkl_sha256'] and ln.exists() and sha_file(ln) == rec['npz_sha256']:
                done += 1
                continue
            raise StageError(f'local record inconsistent {eid}; refuse to overwrite')
        if shutil.disk_usage(local_dir).free < reserve_gib * 1024 ** 3:
            raise StageError(f'free space below {reserve_gib} GiB; stopped before {eid}')
        rec = json.loads(fetcher.read(f'{remote_dir}/{st}.json'))
        if rec['episode_id'] != eid or rec['pkl_sha256'] != r['pkl_sha256']:
            raise StageError(f'remote record linkage {eid}')
        if int(rec['success']) != int(r['success']):
            raise StageError(f'remote success != spec {eid}')
        tmp = local_dir / f'.{rec["npz"]}.part'
        if tmp.exists():
            tmp.unlink()                      # our own partial from a killed attempt
        fetcher.stream(f'{remote_dir}/{rec["npz"]}', tmp)
        if sha_file(tmp) != rec['npz_sha256']:
            tmp.unlink()
            raise StageError(f'npz sha mismatch after transfer {eid}')
        os.replace(tmp, local_dir / rec['npz'])
        with open(lj, 'x') as f:
            f.write(json.dumps(rec, indent=1))   # json last = commit marker
        done += 1; pulled += 1
        log(json.dumps({'staged': done, 'total': len(eps), 'pulled': pulled, 'episode': eid}))
    summary = {'episodes': len(eps), 'pulled_this_run': pulled, 'contract_sha256': hashlib.sha256(rc).hexdigest()}
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', type=Path, required=True); ap.add_argument('--remote-dir', required=True)
    ap.add_argument('--local-dir', type=Path, required=True)
    ap.add_argument('--fold', action='append'); ap.add_argument('--scene', action='append')
    ap.add_argument('--roles', nargs='+', default=['fit', 'eval'], choices=['fit', 'eval'])
    ap.add_argument('--reserve-gib', type=float, default=20.0); ap.add_argument('--local-source', action='store_true')
    a = ap.parse_args()
    fetcher = LocalFetcher() if a.local_source else WrapperFetcher()
    print(json.dumps(stage(load_spec(a.spec), fetcher, a.remote_dir, a.local_dir, folds=a.fold, scenes=a.scene,
                           roles=tuple(a.roles), reserve_gib=a.reserve_gib)))


if __name__ == '__main__':
    try:
        main()
    except StageError as e:
        print('STAGE_ERROR', e, file=sys.stderr); sys.exit(2)

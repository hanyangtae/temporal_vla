"""Git-only transport for exact pins the srv hosts cannot fetch from GitHub (local-only commits / private repo on srv48).

Builds verified git bundles LOCALLY (no transfer, no change to any local repo: a throwaway bare repo borrows objects via
objects/info/alternates and holds the single bundle ref refs/fold27/pin/<name>). The receiving side applies them with
plain git (fetch from the bundle / worktree add / submodule checkout of the pinned sha) — see srv_prepare_v2.sh.

  name                 local repo (objects)                      pin (deploy 04092cbf gitlink / AE recipe)  prerequisite
  trqam_retention      main repo                                 8070d4c5 (feat/trqam-success-retention)   merge-base with deploy 04092cbf
  task_classification  task_classification                       33e5e049 (gitlink)                         none (full; private repo)
  robocasa             .git/modules/robocasa                     3a6b40d0 (gitlink; not on GitHub)          none (full, 48 MB)
  isaac_gr00t          .git/modules/src/policies/Isaac-GR00T     108f3381 (gitlink; not on GitHub)          merge-base with origin/main

    python git_bundles.py build  [--out outputs/analysis/groot_fold27_20261028/git_bundles] [--only NAME ...]
    python git_bundles.py check  [--out ...]      # re-verify sha256 + `git bundle verify` + heads == pins
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
DEPLOY_COMMIT = '04092cbf'
OUT = REPO / 'outputs/analysis/groot_fold27_20261028/git_bundles'
# closure the AE potential imports under <deploy>/.worktrees/trqam-retention (ae_potential.PRIOR, tc_import.REPO_ROOT,
# cache_io.STAGE2): must be byte-identical between the bundle commit and the kanu working tree that built v1/v2 potentials
AE_CLOSURE = ('scripts/rl2_vla/ae_cluster_reward', 'scripts/rl2_vla/stage2_cluster_reward',
              'scripts/analysis/grid_phase/ae_cluster.py', 'manuscript/numbers.md')
PINS = {
    'trqam_retention': {'repo': '.', 'objects': '.git/objects', 'pin': '8070d4c5a581e5734304c763a1c1684de5d957cd',
                        'prereq': ('merge-base', DEPLOY_COMMIT), 'closure_check': '.worktrees/trqam-retention'},
    'task_classification': {'repo': 'task_classification', 'objects': 'task_classification/.git/objects', 'gitlink': 'task_classification',
                            'pin': '33e5e0495d8918b19e87da105bff735d2d80461d', 'prereq': None},
    'robocasa': {'repo': 'src/benchmarks/robocasa', 'objects': '.git/modules/robocasa/objects', 'gitlink': 'src/benchmarks/robocasa',
                 'pin': '3a6b40d01be7acd7511c893c6ea034f8ab940d69', 'prereq': None},
    'isaac_gr00t': {'repo': 'src/policies/Isaac-GR00T', 'objects': '.git/modules/src/policies/Isaac-GR00T/objects',
                    'gitlink': 'src/policies/Isaac-GR00T', 'pin': '108f3381d0b4ac6dd79dfd1c699734b230d03931',
                    'prereq': ('merge-base', 'origin/main')},
}


def git(*a, cwd=REPO, check=True):
    p = subprocess.run(['git', *a], cwd=cwd, capture_output=True, text=True)
    if check and p.returncode:
        raise RuntimeError(f'git {" ".join(a)} (cwd {cwd}): {p.stderr.strip()}')
    return p.stdout.strip()


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def gitlink(path):
    out = git('ls-tree', DEPLOY_COMMIT, '--', path)
    return out.split()[2] if out else None


def closure_clean(spec):
    """AE closure files at the pin == the working tree that ran (tracked, unmodified, no untracked .py in the closure)."""
    wt = REPO / spec['closure_check']
    bad = git('status', '--porcelain', '--', *AE_CLOSURE, cwd=wt).splitlines()
    bad = [l for l in bad if '__pycache__' not in l]
    head = git('rev-parse', 'HEAD', cwd=wt)
    if head != spec['pin']:
        bad.append(f'worktree HEAD {head} != pin')
    files = git('ls-tree', '-r', '--name-only', spec['pin'], '--', *AE_CLOSURE).splitlines()
    for need in ('scripts/rl2_vla/ae_cluster_reward/tc_import.py', 'scripts/rl2_vla/ae_cluster_reward/potential.py',
                 'scripts/rl2_vla/ae_cluster_reward/cache_io.py', 'scripts/rl2_vla/stage2_cluster_reward/data.py',
                 'scripts/rl2_vla/stage2_cluster_reward/clusters.py', 'scripts/analysis/grid_phase/ae_cluster.py', 'manuscript/numbers.md'):
        if need not in files:
            bad.append(f'{need} missing at pin')
    return bad, {f: git('rev-parse', f'{spec["pin"]}:{f}') for f in files if f.endswith(('.py', '.md'))}


def build_one(name, spec, out):
    repo = REPO / spec['repo']
    if spec.get('gitlink') and gitlink(spec['gitlink']) != spec['pin']:
        raise RuntimeError(f'{name}: deploy {DEPLOY_COMMIT} gitlink {gitlink(spec["gitlink"])} != pin {spec["pin"]}')
    if git('cat-file', '-t', spec['pin'], cwd=repo) != 'commit':
        raise RuntimeError(f'{name}: pin {spec["pin"]} not present locally')
    rec = {'name': name, 'pin': spec['pin'], 'ref': f'refs/fold27/pin/{name}'}
    if spec.get('closure_check'):
        bad, blobs = closure_clean(spec)
        if bad:
            raise RuntimeError(f'{name}: AE closure not clean: {bad}')
        rec['closure_blobs'] = blobs
    prereq = []
    if spec['prereq']:
        prereq = [git('merge-base', spec['pin'], spec['prereq'][1], cwd=repo)]
    tmp = Path(tempfile.mkdtemp(prefix=f'fold27_bundle_{name}_'))
    try:
        git('init', '-q', '--bare', str(tmp), cwd=tmp)
        (tmp / 'objects/info/alternates').write_text(str((REPO / spec['objects']).resolve()) + '\n')
        git('update-ref', rec['ref'], spec['pin'], cwd=tmp)
        dst = out / f'{name}.bundle'
        part = dst.with_suffix('.bundle.part')
        git('bundle', 'create', str(part), rec['ref'], *[f'^{p}' for p in prereq], cwd=tmp)
        os.replace(part, dst)
    finally:
        shutil.rmtree(tmp)
    rec.update(prerequisites=prereq, file=dst.name, bytes=dst.stat().st_size, sha256=sha256(dst))
    check_one(rec, out)
    return rec


def check_one(rec, out):
    f = out / rec['file']
    if sha256(f) != rec['sha256']:
        raise RuntimeError(f'{rec["name"]}: bundle sha256 mismatch')
    heads = git('bundle', 'list-heads', str(f)).split()
    if heads != [rec['pin'], rec['ref']]:
        raise RuntimeError(f'{rec["name"]}: bundle heads {heads} != {[rec["pin"], rec["ref"]]}')
    # verify against an empty repo borrowing the source objects: proves prerequisites are exactly the recorded ones
    spec = PINS[rec['name']]
    tmp = Path(tempfile.mkdtemp(prefix='fold27_bverify_'))
    try:
        git('init', '-q', '--bare', str(tmp), cwd=tmp)
        if rec['prerequisites']:
            (tmp / 'objects/info/alternates').write_text(str((REPO / spec['objects']).resolve()) + '\n')
        v = subprocess.run(['git', 'bundle', 'verify', str(f)], cwd=tmp, capture_output=True, text=True)
        if v.returncode:
            raise RuntimeError(f'{rec["name"]}: git bundle verify failed: {v.stderr.strip()}')
        listed = [l.split()[0] for l in v.stdout.splitlines() if len(l.split()[0]) == 40 and l.startswith(tuple('0123456789abcdef'))]
        req = sorted(set(listed) - {rec['pin']})
        if req != sorted(rec['prerequisites']):
            raise RuntimeError(f'{rec["name"]}: verify prerequisites {req} != recorded {rec["prerequisites"]}')
    finally:
        shutil.rmtree(tmp)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('build', 'check')); ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--only', nargs='+', choices=sorted(PINS))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    man_p = a.out / 'BUNDLES.json'
    man = json.loads(man_p.read_text()) if man_p.exists() else {'deploy_commit': DEPLOY_COMMIT, 'bundles': {}}
    if a.cmd == 'build':
        for n in (a.only or sorted(PINS)):
            man['bundles'][n] = build_one(n, PINS[n], a.out)
            print(n, man['bundles'][n]['bytes'], man['bundles'][n]['sha256'][:16], flush=True)
        tmp = man_p.with_suffix('.tmp'); tmp.write_text(json.dumps(man, indent=1)); os.replace(tmp, man_p)
    else:
        for n, rec in sorted(man['bundles'].items()):
            check_one(rec, a.out); print(n, 'OK', rec['sha256'][:16])


if __name__ == '__main__':
    sys.exit(main())

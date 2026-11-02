"""CPU integration check of finetune_fold --dry-run on REAL fold spec + REAL compact episodes, with SYNTHETIC latents.

Synthetic latents (random, marked synthetic_test_only) stand in for GPU rollout latents only to exercise loading,
linkage, codec normalization, rewards, potential and the prefix-QAM batch contract. They live in a scratch dir and can
never be used for training (finetune production also needs the real latents contract / COMPLETE from rollout_latents).

    .venvs/rl2_iid/bin/python integration_dryrun.py --fold OpenDrawer-left_s2_j2 --episodes <compact dir> \
        --scratch outputs/tmp/fold27_integration [--arms basic pbrs mix]
"""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SPEC = REPO / 'outputs/analysis/groot_fold27_20261028/folds/fold_spec.json'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def synth_latents(spec, fold_id, episodes, out):
    out.mkdir(parents=True, exist_ok=False)
    ec = json.loads((episodes / 'contract.json').read_text())
    (out / 'contract.json').write_text(json.dumps({'synthetic_test_only': True,
                                                   'episodes_contract_sha256': ec['source_contract_sha256']}))
    fold = next(f for f in json.loads(spec.read_text())['folds'] if f['fold_id'] == fold_id)
    rng = np.random.default_rng(0)
    for r in fold['fit']:
        eid = f'{r["plan_id"]}/{fold["archive_machine"]}/{fold["instruction"]}/s{fold["scene"]}/j{r["j"]}/n{r["n"]}'
        st = hashlib.sha256(eid.encode()).hexdigest()[:24]
        rec = json.loads((episodes / f'{st}.json').read_text())
        n = rec['n_inferences']
        np.savez(out / f'{st}.npz', latents=rng.normal(size=(n, 1024)).astype(np.float32))
        (out / f'{st}.json').write_text(json.dumps({'episode_id': eid, 'pkl_sha256': rec['pkl_sha256'], 'npz': f'{st}.npz',
                                                     'npz_sha256': sha(out / f'{st}.npz'),
                                                     'source_npz_sha256': rec.get('source_npz_sha256'), 'synthetic_test_only': True}))
    return len(fold['fit'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', required=True); ap.add_argument('--episodes', type=Path, required=True)
    ap.add_argument('--scratch', type=Path, required=True); ap.add_argument('--arms', nargs='+', default=['basic'])
    a = ap.parse_args()
    lat = a.scratch / 'synthetic_latents'
    n = synth_latents(SPEC, a.fold, a.episodes, lat)
    res = {'synthetic_fit_latents': n}
    for arm in a.arms:
        p = subprocess.run([sys.executable, str(HERE / 'finetune_fold.py'), '--spec', str(SPEC), '--fold', a.fold, '--arm', arm,
                            '--episodes', str(a.episodes), '--latents', str(lat), '--out', str(a.scratch / 'runs'), '--dry-run'],
                           capture_output=True, text=True)
        res[arm] = {'rc': p.returncode, 'tail': (p.stdout + p.stderr)[-600:]}
    print(json.dumps(res, indent=1))
    sys.exit(0 if all(v['rc'] == 0 for k, v in res.items() if k in a.arms) else 2)


if __name__ == '__main__':
    main()

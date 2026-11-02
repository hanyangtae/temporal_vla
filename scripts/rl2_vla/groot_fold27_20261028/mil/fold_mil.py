"""Per-fold MIL detector retraining on the fold's 160 non-target rollouts (task_classification PR11 trainer, unmodified).

    python fold_mil.py preflight --spec fold_spec.json
    python fold_mil.py shards    --spec fold_spec.json --episodes <extract_scene out> --out <shard root> [--fold ID]
    python fold_mil.py train     --spec fold_spec.json --shards <shard root> --out <run root> --fold ID [--seed 0]

Recipe = output/share/grid_n10_n19 runs (attn, raw, SGD lr 0.3 momentum .9 nesterov, wd 1e-4, 400 epochs /
patience 60 / min 60, grad clip 1, fire_threshold 0.5, val_ratio .2 DataSplitter split_seed 0) — only the data
changes (n0..39, 160/40 instead of n0..9 40/10). Shared n10/n19 checkpoints are NEVER reused for folds; they only
validate loading/inference (verify_shared48.py).
Leakage: train shard = 160 fit episodes only; test shard = the 40 target-j episodes, scored by the trainer as an
offline detector diagnostic (fires.tsv) and never used for weights, early stopping, transform fit or threshold
(transform is fit on train-split frames after the val split; val comes out of the 160).
Shard feature = stored f16 token mean as [N,1,1536]: the trainer's _token_mean on it reproduces
f32 mean(49) -> f16 -> f32 bit-exactly (tests/test_mil_gate.py).
"""
import argparse, copy, hashlib, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from folds import load_spec, validate_fold   # noqa: E402

R = HERE.parents[3]
TC = R / 'task_classification'
RECIPE_TRAIN = {'optimizer': 'sgd', 'lr': 0.3, 'momentum': 0.9, 'nesterov': True, 'weight_decay': 1.0e-4,
                'epochs': 400, 'patience': 60, 'min_epochs': 60, 'grad_clip': 1.0}
LAYER, DENOISE = 12, 3


class FoldMilError(RuntimeError):
    pass


# Main-session implementation exception (2026-10-29, routine choice within the user's "run all 27" scope — NOT an
# explicit user decision): only this fold (fit S3/F157) cannot be split by
# DataSplitter(min_per_side=2). Successes: train 2 / val 1 (episode-disjoint, fixed seed); failures: unchanged
# DataSplitter draw. All within the 160 non-target rollouts; no target data, no re-selection, threshold .5 kept.
# Consequence: val has 1 success episode, so the val bag sampler's min_side_episodes must be 1 (train sides all
# have >= 2 episodes, so train sampling is unaffected). Other 26 folds use the unmodified recipe.
SPLIT_EXCEPTIONS = {'DishwasherRack-out-right_s1_j0': {'success_val': 1, 'min_side_episodes': 1}}


class ExceptionSplitter:
    """DataSplitter mirror: same rng order (failure side first, then success), success-side val count overridden."""

    def __init__(self, val_ratio, seed, success_val, min_per_side=2):
        self.val_ratio, self.seed, self.success_val, self.min_per_side = float(val_ratio), int(seed), int(success_val), min_per_side

    def split_rows(self, shard):
        episode_idx = shard.data_meta['episode_idx'].to_numpy()
        success = np.asarray(shard.data_success)
        eps, first = np.unique(episode_idx, return_index=True)
        ep_s = success[first]
        rng = np.random.default_rng(self.seed)
        val = []
        for ok in (0, 1):
            ids = eps[ep_s == ok]
            k = self.success_val if ok == 1 else max(self.min_per_side, int(round(len(ids) * self.val_ratio)))
            if len(ids) - k < (2 if ok == 1 else self.min_per_side) or k < 1:
                raise FoldMilError(f'exception split infeasible: side {ok} n={len(ids)} k={k}')
            val += list(rng.choice(ids, k, replace=False))
        is_val = np.isin(episode_idx, np.asarray(val, dtype=eps.dtype))
        return np.flatnonzero(~is_val), np.flatnonzero(is_val)


def _tc_imports():
    for p in (TC, TC / 'libs/dataset_utils/src'):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from dataset_utils.data_shard import DataShard
    from failure_detection_multi_instance_learning.train.train_mil_pooling_cell import CFG, MilPoolingTrainer
    return DataShard, CFG, MilPoolingTrainer


def splitter_feasible(n_succ, n_fail, val_ratio=0.2, min_per_side=2):
    """Mirror of dataset_utils.DataSplitter._val_episodes feasibility (single-cell training raises otherwise)."""
    for n in (n_succ, n_fail):
        if n == 0:
            continue
        k = max(min_per_side, int(round(n * val_ratio)))
        if n - k < min_per_side:
            return False
    return n_succ >= 2 * min_per_side and n_fail >= 2 * min_per_side   # sampler also needs both sides


def preflight(spec):
    rows = []
    for f in spec['folds']:
        e = f['expected']
        ok = splitter_feasible(e['fit_success'], e['fit_failure'])
        exc = SPLIT_EXCEPTIONS.get(f['fold_id'])
        rows.append({'fold_id': f['fold_id'], 'fit_success': e['fit_success'], 'fit_failure': e['fit_failure'],
                     'eval_success': e['eval_success'], 'mil_trainable_unmodified': ok,
                     'split_exception': exc if (not ok and exc) else None,
                     'blocker': None if (ok or exc) else 'DataSplitter(val_ratio=.2,min_per_side=2) cannot split this class count; '
                                                         'needs an approved SPLIT_EXCEPTIONS entry'})
    return rows


def build_shard(fold, episodes_dir, role):
    """fold role ('fit'|'eval') -> DataShard (feature [N,1,1536] f16, success int8, phase int8 zeros, meta)."""
    import pandas as pd
    DataShard, _, _ = _tc_imports()
    feats, succ, metas = [], [], []
    for e_idx, r in enumerate(sorted(fold[role], key=lambda r: (r['j'], r['n']))):
        eid = f'{r["plan_id"]}/{fold["archive_machine"]}/{fold["instruction"]}/s{fold["scene"]}/j{r["j"]}/n{r["n"]}'
        stem = hashlib.sha256(eid.encode()).hexdigest()[:24]
        rec = json.loads((episodes_dir / f'{stem}.json').read_text())
        if rec['episode_id'] != eid or rec['pkl_sha256'] != r['pkl_sha256']:
            raise FoldMilError(f'episode record mismatch {eid}')
        npz = episodes_dir / rec['npz']
        if hashlib.sha256(npz.read_bytes()).hexdigest() != rec['npz_sha256']:
            raise FoldMilError(f'npz sha mismatch {eid}')
        with np.load(npz) as z:
            tm, s = z['tokmean_l12d3'], int(z['success'])
        if s != int(r['success']):
            raise FoldMilError(f'success mismatch {eid}')
        T = len(tm)
        feats.append(tm[:, None, :])
        succ.append(np.full(T, s, np.int8))
        metas.append(pd.DataFrame({'episode_idx': e_idx, 't': np.arange(T), 'T': T, 'episode_id': eid,
                                   'task': fold['instruction'], 'scene': fold['scene'], 'jitter': r['j'],
                                   'noise': r['n'], 'layer': LAYER, 'denoise': DENOISE}))
    sh = DataShard()
    sh.set_data_feature(np.concatenate(feats).astype(np.float16))
    sh.set_data_success(np.concatenate(succ))
    sh.set_data_phase(np.zeros(len(sh.data_success), np.int8))
    sh.set_data_meta(pd.concat(metas, ignore_index=True))
    return sh


def cmd_shards(a):
    spec = load_spec(a.spec)
    for f in spec['folds']:
        if a.fold and f['fold_id'] not in a.fold:
            continue
        validate_fold(f)
        out = a.out / f['fold_id']
        if (out / 'SHARD_COMPLETE.json').exists():
            continue
        for role, split in (('fit', 'train'), ('eval', 'test')):
            build_shard(f, a.episodes, role).save(out / split)
        (out / 'SHARD_COMPLETE.json').write_text(json.dumps({'fold_id': f['fold_id'], 'train_episodes': 160, 'test_episodes': 40,
                                                            'feature': 'tokmean_l12d3 f16 [N,1,1536]'}, indent=1))
        print('shard', f['fold_id'], flush=True)


def fold_episode_ids(fold, role):
    return {f'{r["plan_id"]}/{fold["archive_machine"]}/{fold["instruction"]}/s{fold["scene"]}/j{r["j"]}/n{r["n"]}' for r in fold[role]}


def heldout_proof(trainer, fold):
    """Computed from what the trainer actually loaded (before fit): train+val == the 160 fit ids, disjoint from the 40
    eval ids, no target-j step anywhere in train/val; test == exactly the 40 eval ids."""
    tr = set(trainer.data_meta['train'].episode_id.unique()); va = set(trainer.data_meta['val'].episode_id.unique())
    te = set(trainer.data_meta['test'].episode_id.unique())
    fit, ev = fold_episode_ids(fold, 'fit'), fold_episode_ids(fold, 'eval')
    tj = int(fold['target_j'])
    js = set(trainer.data_meta['train'].jitter.astype(int)) | set(trainer.data_meta['val'].jitter.astype(int))
    ok = (tr | va) == fit and not (tr & va) and not ((tr | va) & ev) and te == ev and tj not in js
    if not ok:
        raise FoldMilError(f'held-out violation: train{len(tr)} val{len(va)} test{len(te)} fit{len(fit)} eval{len(ev)} jitters{sorted(js)}')
    return {'train_episodes': len(tr), 'val_episodes': len(va), 'test_episodes': len(te), 'train_val_equals_fit160': True,
            'train_val_disjoint_eval40': True, 'target_j_absent_from_train_val': True, 'train_val_jitters': sorted(js)}


def verify_committed(run, fold_id, seed, shard_sha, fold):
    """Skip only a complete, matching run: FOLD_CONTRACT (commit marker, written last) + model sha + shard sha + recipe."""
    cp = run / 'FOLD_CONTRACT.json'
    if not cp.exists() or not (run / 'model.pt').exists():
        raise FoldMilError(f'{run}: partial run (model.pt/FOLD_CONTRACT.json missing); move it aside and retrain')
    c = json.loads(cp.read_text())
    want = {'fold_id': fold_id, 'seed': seed, 'recipe': RECIPE_TRAIN, 'shard_complete_sha256': shard_sha, 'target_j': fold['target_j']}
    bad = {k: (c.get(k), v) for k, v in want.items() if c.get(k) != v}
    if hashlib.sha256((run / 'model.pt').read_bytes()).hexdigest() != c.get('model_sha256'):
        bad['model_sha256'] = 'mismatch'
    if not (c.get('heldout_proof') or {}).get('train_val_disjoint_eval40'):
        bad['heldout_proof'] = 'missing'
    if bad:
        raise FoldMilError(f'{run}: existing run does not match ({bad}); refuse to reuse')
    return c


def recipe_cfg(CFG):
    cfg = copy.deepcopy(CFG)
    cfg['train'].update(RECIPE_TRAIN)
    if cfg['fire_threshold'] != 0.5 or cfg['val_ratio'] != 0.2:
        raise FoldMilError('task_classification CFG drifted from PR11 (fire_threshold/val_ratio)')
    return cfg


def cmd_train(a):
    spec = load_spec(a.spec)
    fold = next((f for f in spec['folds'] if f['fold_id'] == a.fold), None)
    if fold is None:
        raise FoldMilError(f'unknown fold {a.fold}')
    pf = {r['fold_id']: r for r in preflight(spec)}[a.fold]
    run = a.out / a.fold / 'attn/raw' / f's{a.seed}'
    shard_sha = hashlib.sha256((a.shards / a.fold / 'SHARD_COMPLETE.json').read_bytes()).hexdigest()
    code_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if (run / 'model.pt').exists() or (run / 'FOLD_CONTRACT.json').exists():
        verify_committed(run, a.fold, a.seed, shard_sha, fold)
        print('verified existing', run); return
    if pf['blocker']:
        (a.out / a.fold).mkdir(parents=True, exist_ok=True)
        (a.out / a.fold / 'BLOCKED.json').write_text(json.dumps(pf, indent=1))
        raise FoldMilError(f'{a.fold}: {pf["blocker"]}')
    _, CFG, MilPoolingTrainer = _tc_imports()
    cfg, exc = recipe_cfg(CFG), pf['split_exception']
    if exc:
        cfg['sampler']['min_side_episodes'] = exc['min_side_episodes']
    trainer = MilPoolingTrainer(pooling='attn', input_kind='raw', seed=a.seed, split_seed=0, device=a.device, cfg=cfg)
    if exc:
        trainer.splitter = ExceptionSplitter(cfg['val_ratio'], 0, exc['success_val'])
    trainer.load_shards([a.shards / a.fold])
    if trainer.cell['target_jitter'] != [fold['target_j']] or fold['target_j'] in trainer.cell['train_jitter']:
        raise FoldMilError('shard jitter roles do not match fold spec')
    proof = heldout_proof(trainer, fold)
    trainer.fit()
    trainer.save(run)
    (run / 'FOLD_CONTRACT.json').write_text(json.dumps({'fold_id': a.fold, 'target_j': fold['target_j'],
        'recipe': RECIPE_TRAIN, 'seed': a.seed, 'split_seed': 0, 'split_exception': exc,
        'shard_complete_sha256': shard_sha, 'fold_mil_sha256': code_sha, 'heldout_proof': proof,
        'selection': 'best epoch by val bag BCE (val drawn from the 160 fit episodes); target-j test scores are written '
                     'after fit for diagnostics only and never used for epoch/threshold/checkpoint selection',
        'threshold': 'fire_threshold 0.5 fixed by recipe (not tuned)',
        'val_episodes': sorted(trainer.data_meta['val'].episode_id.unique().tolist()), 'tc_commit_expected': '33e5e0495d8918b19e87da105bff735d2d80461d',
        'model_sha256': hashlib.sha256((run / 'model.pt').read_bytes()).hexdigest(),
        'shared_checkpoint_reused': False}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('preflight'); p.add_argument('--spec', type=Path, required=True)
    p = sub.add_parser('shards'); p.add_argument('--spec', type=Path, required=True); p.add_argument('--episodes', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--fold', action='append')
    p = sub.add_parser('train'); p.add_argument('--spec', type=Path, required=True); p.add_argument('--shards', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--fold', required=True); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default=None)
    a = ap.parse_args()
    if a.cmd == 'preflight':
        rows = preflight(load_spec(a.spec))
        print(json.dumps({'blocked': [r for r in rows if not r['mil_trainable_unmodified']], 'n_ok': sum(r['mil_trainable_unmodified'] for r in rows)}, indent=1))
    elif a.cmd == 'shards':
        cmd_shards(a)
    else:
        cmd_train(a)


if __name__ == '__main__':
    try:
        main()
    except FoldMilError as e:
        print('FOLD_MIL_ERROR', e, file=sys.stderr); sys.exit(2)

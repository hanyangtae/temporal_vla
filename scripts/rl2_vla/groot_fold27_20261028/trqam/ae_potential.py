"""AE16/K48 cluster potential for a fold (D-POT resolved: user's earlier instruction = task_classification AE clustering).

Reuses — does not reimplement — the prior recipe .worktrees/trqam-retention/scripts/rl2_vla/ae_cluster_reward/:
  tc_import.TaskClassification(root)  original task_classification AE (phase.models.factory.build_model conf/model/ae.yaml),
                                      optimizer/trainer (phase.train._loop), GPUKMeans (phase.clustering.gpu), manuscript
                                      scalar standardizer (scripts/analysis/grid_phase/ae_cluster.py, AST-extracted)
  potential.fit_ae_potential          joint AE on train chunks, val = reconstruction early stopping only, per-task K48,
                                      stage2 occupancy scores with prior 5
Fold differences (recorded in metadata): input = GR00T decoder-input latent 1024 (prior: pi0 context 1024); one task =
the fold instruction; AE validation split = stratified 20% of the fold's 160 fit episodes (no target-j data); task_classification
root pinned to the submodule checkout 33e5e04 (prior run used 88543a2 — the 12 TC_SOURCES files are re-hashed and recorded,
so any change between the two commits is visible in provenance; equality is NOT assumed).
"""
import hashlib, importlib.util, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
PRIOR = ROOT / '.worktrees/trqam-retention/scripts/rl2_vla/ae_cluster_reward'
TC_ROOT = ROOT / 'task_classification'
EXPECT_TC = '33e5e0495d8918b19e87da105bff735d2d80461d'
PRIOR_TC = '88543a2a00e37a1de409ccdaaaa865ab787c7196'
VAL_RATIO, SPLIT_SEED, K, SEED = 0.2, 0, 48, 42


def _load(name):
    spec = importlib.util.spec_from_file_location(f'ae_cluster_reward_{name}', PRIOR / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def prior_modules():
    if str(PRIOR) not in sys.path:
        sys.path.insert(0, str(PRIOR))     # potential.py does `from cache_io import sha256`
    return _load('tc_import'), _load('potential')


def tc_head():
    g = TC_ROOT / '.git'
    if g.is_file():
        g = (TC_ROOT / g.read_text().split(':', 1)[1].strip()).resolve()
    return (g / 'HEAD').read_text().strip()


def ae_val_split(episodes):
    """Episode-level stratified split inside the fold's fit set: per outcome k = round(.2 n), train keeps >= 1."""
    rng = np.random.default_rng(SPLIT_SEED)
    val = set()
    for ok in (False, True):
        ids = [e['episode_id'] for e in episodes if bool(e['success']) == ok]
        k = int(round(len(ids) * VAL_RATIO))
        k = min(k, len(ids) - 1) if len(ids) > 1 else 0
        val |= set(rng.choice(ids, k, replace=False).tolist()) if k else set()
    return [e for e in episodes if e['episode_id'] not in val], [e for e in episodes if e['episode_id'] in val]


def fit_fold_ae(episodes, instruction, *, device='cpu', train_overrides=None):
    """episodes: [{'episode_id','latents'[n,1024],'success'}] = the fold's 160 fit rollouts only."""
    head = tc_head()
    if head != EXPECT_TC:
        raise RuntimeError(f'task_classification HEAD {head} != pinned {EXPECT_TC}')
    tci, pot = prior_modules()
    tc = tci.TaskClassification(TC_ROOT)
    tr, va = ae_val_split(episodes)
    conv = lambda es: [{'episode_id': e['episode_id'], 'task': instruction, 'success': bool(e['success']),
                        'contexts': np.asarray(e['latents'], np.float32)} for e in es]
    manifest = hashlib.sha256('\n'.join(sorted(e['episode_id'] for e in episodes)).encode()).hexdigest()
    p = pot.fit_ae_potential(conv(tr), conv(va), tc, manifest, device=device, seed=SEED, k=K, train_overrides=train_overrides)
    p.metadata.update(fold_input='GR00T decoder-input last16 mean 1024 (QAM observation)',
                      ae_val_split=dict(ratio=VAL_RATIO, seed=SPLIT_SEED, stratified='outcome', within='fold fit 160 only'),
                      task_classification_head=head, prior_recipe_task_classification=PRIOR_TC,
                      prior_recipe_sources={n: hashlib.sha256((PRIOR / f'{n}.py').read_bytes()).hexdigest()
                                            for n in ('tc_import', 'potential', 'cache_io')})
    return p


def load_fold_ae(path):
    """Reload a saved fold potential (prior recipe AEClusterPotential.load verifies the encoder weights sha)."""
    _, pot = prior_modules()
    return pot.AEClusterPotential.load(path)


def potential_fn(p, instruction):
    return lambda lat: p.phi_batch(instruction, np.asarray(lat, np.float32))

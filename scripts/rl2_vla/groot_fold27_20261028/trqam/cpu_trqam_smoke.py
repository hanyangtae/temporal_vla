"""CPU execution check of the TRQAM prefix agent on the REAL common 500k checkpoint (no GPU, no production output).

Restores the common atomic65 checkpoint, migrates it into TrqamPrefixAgent (frozen actor_fast reference), runs N updates
on rows built by finetune_fold from real compact episodes + the given latents (synthetic allowed: this is a code-path
check, not training), and asserts: finite losses, dual lambda/kl_ema move, frozen reference + target actor slots
bit-unchanged, trained actor_fast changed, inference_agent save/load round-trip. Writes only under --scratch.

    JAX_PLATFORMS=cpu .venvs/rl2_iid/bin/python cpu_trqam_smoke.py --fold F --episodes <compact> --latents <dir> --scratch <dir>
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path[:0] = [str(HERE), str(HERE.parent), str(REPO / 'scripts/rl2_vla/groot_robocasa_adapter_20261021'), str(REPO)]
import finetune_fold as FF   # noqa: E402

SPEC = REPO / 'outputs/analysis/groot_fold27_20261028/folds/fold_spec.json'
INIT = REPO / 'outputs/rl2_vla/atomic65_common_qam_work_20261029/scratch_run/checkpoints/step_0500000'


def leaves_equal(a, b):
    import jax
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return len(la) == len(lb) and all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', required=True); ap.add_argument('--episodes', type=Path, required=True)
    ap.add_argument('--latents', type=Path, required=True); ap.add_argument('--scratch', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=3); ap.add_argument('--batch', type=int, default=32)
    a = ap.parse_args()
    import jax
    from qam_prefix import PrefixQAMTrainer, to_qam_batch
    from trqam_prefix import RETENTION_DEFAULTS, migrate, inference_agent
    from contract import load_checkpoint_contract, resolve_cached_checkpoint_dir
    from folds import load_spec
    fold = next(f for f in load_spec(SPEC)['folds'] if f['fold_id'] == a.fold)
    codec = load_checkpoint_contract(resolve_cached_checkpoint_dir()).codec
    eps = FF.load_fit_episodes(fold, a.episodes, a.latents, codec)
    rows, _, _ = FF.build_rows(fold, eps, 'basic', None, None)
    base = PrefixQAMTrainer.load(INIT, 500000, feature_signature=FF.FEATURE_SIGNATURE)
    agent = migrate(base.agent, 42, RETENTION_DEFAULTS)
    p0 = agent.network.params
    ref0 = {k: p0[k] for k in ('modules_actor_slow', 'modules_target_actor_slow')}
    assert leaves_equal(p0['modules_actor_slow'], p0['modules_actor_fast']), 'reference != deployed actor_fast after migrate'
    rng = np.random.default_rng(0)
    infos = []
    for _ in range(a.steps):
        idx = rng.integers(len(rows['rewards']), size=a.batch)
        agent, info = agent.update(to_qam_batch({k: v[idx] for k, v in rows.items()}, execute_steps=5, gamma=0.99))
        infos.append({k: float(v) for k, v in info.items()})
    p1 = agent.network.params
    checks = {
        'finite': all(np.isfinite(list(i.values())).all() for i in infos),
        'reference_frozen': all(leaves_equal(ref0[k], p1[k]) for k in ref0),
        'actor_fast_changed': not leaves_equal(p0['modules_actor_fast'], p1['modules_actor_fast']),
        'critic_changed': not leaves_equal(p0['modules_critic'], p1['modules_critic']),
        'dual_moved': float(agent.kl_ema) != 0.0,
        'backend': jax.default_backend(),
    }
    out = a.scratch / 'cpu_trqam_ckpt'
    tr = PrefixQAMTrainer(inference_agent(agent, base.agent.config), gamma=0.99, data_label=f'cpu_smoke_{a.fold}')
    tr.steps_done = a.steps
    tr.save(out, feature_signature=FF.FEATURE_SIGNATURE)
    back = PrefixQAMTrainer.load(out, a.steps, feature_signature=FF.FEATURE_SIGNATURE)
    checks['save_load_roundtrip'] = leaves_equal(back.agent.network.params['modules_actor_fast'], p1['modules_actor_fast'])
    res = {'checks': checks, 'last_info': {k: infos[-1][k] for k in sorted(infos[-1]) if 'dual' in k or 'path_kl' in k or 'loss' in k}}
    (a.scratch / 'CPU_TRQAM_SMOKE.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    sys.exit(0 if all(v for k, v in checks.items() if k != 'backend') else 2)


if __name__ == '__main__':
    main()

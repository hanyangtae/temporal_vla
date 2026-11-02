"""GPU resume check on REAL fold data + REAL common 500k checkpoint (diagnostic only; writes no DONE/SMOKE_DONE).

continuous N+M updates vs N -> resume_state.save_state -> load_state into a fresh template -> M updates, batch 256.
Gate (exit 0) requires ALL: exact restore, identical minibatch indices + NumPy generator state, GPU backend, and a BITWISE
identical continued trajectory. No tolerance is applied; max-abs leaf difference is recorded only to diagnose a failure.
Output: <out>/<fold>/resume_check_v2/RESUME_CHECK.json.

    (under harness) python gpu_resume_check.py --fold F --episodes <compact> --latents <latents> --out <v2 root> [--n 20 --m 20]
"""
import argparse, json, sys, tempfile, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parents[1] / 'groot_robocasa_adapter_20261021'), str(HERE.parents[3])]
import finetune_fold as FF   # noqa: E402
import resume_state as RS    # noqa: E402

SPEC = HERE.parents[3] / 'outputs/analysis/groot_fold27_20261028/folds/fold_spec.json'
INIT = HERE.parents[3] / 'outputs/rl2_vla/atomic65_common_qam_work_20261029/scratch_run/checkpoints/step_0500000'


def max_abs_diff(a, b):
    import flax, jax
    la = jax.tree_util.tree_leaves(jax.device_get(flax.serialization.to_state_dict(a)))
    lb = jax.tree_util.tree_leaves(jax.device_get(flax.serialization.to_state_dict(b)))
    return max(float(np.max(np.abs(np.asarray(x, np.float64) - np.asarray(y, np.float64)))) if np.size(x) else 0.0
               for x, y in zip(la, lb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', required=True); ap.add_argument('--episodes', type=Path, required=True)
    ap.add_argument('--latents', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--n', type=int, default=20); ap.add_argument('--m', type=int, default=20)
    a = ap.parse_args()
    import jax
    from qam_prefix import PrefixQAMTrainer, to_qam_batch
    from trqam_prefix import RETENTION_DEFAULTS, migrate
    from contract import load_checkpoint_contract, resolve_cached_checkpoint_dir
    from folds import load_spec
    fold = next(f for f in load_spec(SPEC)['folds'] if f['fold_id'] == a.fold)
    eps = FF.load_fit_episodes(fold, a.episodes, a.latents, load_checkpoint_contract(resolve_cached_checkpoint_dir()).codec)
    rows, _, _ = FF.build_rows(fold, eps, 'basic', None, None)
    base = PrefixQAMTrainer.load(INIT, 500000, feature_signature=FF.FEATURE_SIGNATURE).agent
    tmpl = lambda: migrate(base, FF.SEED, RETENTION_DEFAULTS)
    to_batch = lambda b: to_qam_batch(b, execute_steps=5, gamma=FF.GAMMA)
    ia, ib = [], []
    ra = np.random.default_rng(FF.SEED)
    cont, _ = RS.update_loop(tmpl(), rows, ra, 0, a.n + a.m, FF.BATCH, to_batch, lambda s, g, i, x: ia.append(x.copy()))
    rb = np.random.default_rng(FF.SEED)
    part, _ = RS.update_loop(tmpl(), rows, rb, 0, a.n, FF.BATCH, to_batch, lambda s, g, i, x: ib.append(x.copy()))
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'state.pkl'
        RS.save_state(p, part, rb, a.n, {'check': True})
        rest, rr, _, _ = RS.load_state(p, tmpl())
    restore_exact = RS.state_equal(rest, part)
    res, _ = RS.update_loop(rest, rows, rr, a.n, a.n + a.m, FF.BATCH, to_batch, lambda s, g, i, x: ib.append(x.copy()))
    out = {'fold': a.fold, 'backend': jax.default_backend(), 'device': str(jax.devices()[0]), 'n': a.n, 'm': a.m, 'batch': FF.BATCH,
           'restore_exact': restore_exact, 'minibatch_indices_equal': all(np.array_equal(x, y) for x, y in zip(ia, ib)) and len(ia) == len(ib),
           'np_rng_state_equal': ra.bit_generator.state == rr.bit_generator.state,
           'trajectory_bitwise_equal': RS.state_equal(cont, res), 'trajectory_max_abs_diff': max_abs_diff(cont, res),
           'lam': [float(cont.lam), float(res.lam)], 'kl_ema': [float(cont.kl_ema), float(res.kl_ema)], 'time': time.time()}
    d = a.out / a.fold / 'resume_check_v2'; d.mkdir(parents=True, exist_ok=True)
    (d / 'RESUME_CHECK.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    ok = (out['restore_exact'] and out['minibatch_indices_equal'] and out['np_rng_state_equal'] and out['backend'] == 'gpu'
          and out['trajectory_bitwise_equal'])
    # No tolerance: if the GPU continuation is not bitwise identical, stop and report (max_abs_diff is for diagnosis only).
    sys.exit(0 if ok else 2)


if __name__ == '__main__':
    main()

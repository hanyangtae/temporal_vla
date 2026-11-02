"""Full numerical TRQAM training state save/restore + the update loop shared by finetune_fold_v2 and its tests.

Fix for the 2026-11-02 smoke failure (finetune_fold.py:232 pickled agent.network -> optax closure
"Can't pickle local object 'chain.<locals>.init_fn'"). Pattern = trqam_retention/train.py save_resume/load_resume:
only numbers are stored — flax.serialization.to_state_dict(agent) (actor/critic/target/reference params, optimizer
moments + count, network.step, JAX rng, dual lambda, KL EMA) and the NumPy minibatch Generator.bit_generator.state —
and restored into a freshly built template agent (static fns / optimizer transformation / config come from code).
Also fixes the minibatch stream: v1 restarted np.random.default_rng(SEED + start) on resume; here the generator state
itself is saved, so N + M updates == N -> save -> restore -> M updates (same minibatches, same state).
"""
import hashlib, os, pickle
from pathlib import Path

import numpy as np

FORMAT = 'fold27_trqam_resume_v2'


def save_state(path, agent, np_rng, step, fingerprint):
    import flax, jax
    state = {'format': FORMAT, 'agent': jax.device_get(flax.serialization.to_state_dict(agent)),
             'np_rng': np_rng.bit_generator.state, 'step': int(step), 'fingerprint': fingerprint}
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'wb') as f:
        pickle.dump(state, f)
    os.replace(tmp, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_state(path, template):
    import flax
    with open(path, 'rb') as f:
        state = pickle.load(f)
    if state.get('format') != FORMAT:
        raise ValueError(f'{path}: not a {FORMAT} state')
    agent = flax.serialization.from_state_dict(template, state['agent'])
    rng = np.random.default_rng()
    rng.bit_generator.state = state['np_rng']
    return agent, rng, state['step'], state['fingerprint']


def update_loop(agent, rows, np_rng, start, stop, batch, to_batch, on_step=None):
    """Updates start+1..stop. Minibatch indices come only from np_rng (continuous stream across resumes)."""
    n = len(rows['rewards'])
    info = None
    for step in range(start + 1, stop + 1):
        idx = np_rng.integers(n, size=batch)
        agent, info = agent.update(to_batch({k: v[idx] for k, v in rows.items()}))
        if on_step is not None:
            on_step(step, agent, info, idx)
    return agent, info


def state_equal(a, b):
    """Bitwise equality of every numeric leaf of two agents' state dicts (tests / resume audits)."""
    import flax, jax
    la = jax.tree_util.tree_leaves(jax.device_get(flax.serialization.to_state_dict(a)))
    lb = jax.tree_util.tree_leaves(jax.device_get(flax.serialization.to_state_dict(b)))
    return len(la) == len(lb) and all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb))

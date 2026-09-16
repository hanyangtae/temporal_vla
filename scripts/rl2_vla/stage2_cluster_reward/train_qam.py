"""Paired offline QAM continuation. Run in the RL2/JAX environment."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

import numpy as np

from data import load_manifest, sha256, transitions
from clusters import ClusterPotential


def agent_class(qam_root):
    sys.path.insert(0, str(Path(qam_root).resolve()))
    from agents.qam import QAMAgent
    import jax
    import jax.numpy as jnp

    class PrefixQAM(QAMAgent):
        def critic_loss(self, batch, grad_params, rng):
            action = (batch['actions'] * batch['action_mask']).reshape((len(batch['actions']), -1))
            nxt = batch['next_observations'][..., -1, :]
            next_actions = jnp.clip(self.sample_actions(nxt, rng), -1, 1)
            qs = self.network.select('target_critic')(nxt, next_actions)
            next_q = qs.mean(0) - self.config['rho'] * qs.std(0)
            target = batch['rewards'][..., -1] + batch['discounts'] * batch['masks'][..., -1] * next_q
            q = self.network.select('critic')(batch['observations'], action, params=grad_params)
            loss = jnp.mean((q-target)**2)
            return loss, dict(critic_loss=loss, q_mean=q.mean(), q_max=q.max(), q_min=q.min())

        def actor_loss(self, batch, grad_params, rng):
            # Keep QAM adjoint matching exactly; replace only BC's padded-tail loss.
            batch = {**batch, 'actions': batch['actions'] * batch['action_mask']}
            total, info = super().actor_loss(batch, grad_params, rng)
            _, x_rng, t_rng, _, _ = jax.random.split(rng, 5)
            actions = batch['actions'].reshape((len(batch['actions']), -1))
            mask = batch['action_mask'].reshape(actions.shape)
            x0 = jax.random.normal(x_rng, actions.shape)
            t = jax.random.uniform(t_rng, (len(actions), 1))
            pred = self.network.select('actor_slow')(
                batch['observations'], (1-t)*x0 + t*actions, t, params=grad_params)
            squared = (pred - (actions-x0))**2
            old = squared.mean()
            new = ((squared*mask).sum(-1) / mask.sum(-1)).mean()
            return total-old+new, {**info, 'flow_loss': new}

        @jax.jit
        def warmup(self, batch):
            new_rng, rng = jax.random.split(self.rng)
            net, info = self.network.apply_loss_fn(
                lambda params: self.critic_loss(batch, params, rng))
            self.target_update(net, 'critic')
            return self.replace(network=net, rng=new_rng), info

    return PrefixQAM


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--qam-root', required=True)
    p.add_argument('--training-cache', help='derived cache.json exported beside archived source')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--reward-mode', choices=['base', 'cluster_potential'], required=True)
    p.add_argument('--cluster-bundle')
    p.add_argument('--shaping-scale', type=float, default=.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--warmup-steps', type=int, default=5000)
    p.add_argument('--steps', type=int, default=50000)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    cache = None
    if args.training_cache:
        cache = json.loads(Path(args.training_cache).read_text())
        if cache['manifest_sha256'] != sha256(args.manifest):
            raise ValueError('derived cache/manifest mismatch')
        manifest_rows = json.loads(Path(args.manifest).read_text())['episodes']
        episodes = [r for r in manifest_rows if r['split'] == 'train']
    else:
        episodes = load_manifest(args.manifest, 'train')
    bundle = None
    if args.reward_mode == 'cluster_potential':
        if not args.cluster_bundle:
            p.error('--cluster-bundle required for shaped arm')
        bundle = ClusterPotential.load(args.cluster_bundle)
        if (bundle.metadata['manifest_sha256'] != sha256(args.manifest) or
                set(bundle.metadata['episode_ids']) != {e['episode_id'] for e in episodes} or
                bundle.metadata['split'] != 'train'):
            raise ValueError('cluster bundle split/manifest mismatch')
    flags = json.loads(Path(args.checkpoint).with_name('flags.json').read_text())
    config = copy.deepcopy(flags['agent'])
    if config['horizon_length'] != 4 or not config['action_chunking'] or config['best_of_n'] != 1:
        raise ValueError('checkpoint must be 4-action chunk QAM with N=1')
    if config['ob_dims'] != [1024] or config['action_dim'] != 7 or config['discount'] != .99:
        raise ValueError('checkpoint context/action/discount contract mismatch')
    if cache is None:
        data = transitions(episodes, bundle, args.shaping_scale if bundle else 0., config['discount'])
    else:
        meta = cache['qam_transitions'][args.reward_mode]
        path = Path(args.training_cache).parent / meta['path']
        if meta['sha256'] != sha256(path) or meta['manifest_sha256'] != sha256(args.manifest):
            raise ValueError('derived transitions hash mismatch')
        if set(meta['episode_ids']) != {e['episode_id'] for e in episodes}:
            raise ValueError('derived transitions train split mismatch')
        if meta['discount'] != config['discount'] or meta['shaping_scale'] != (args.shaping_scale if bundle else 0.):
            raise ValueError('derived reward parameters mismatch')
        if bundle and meta['cluster_sha256'] != sha256(args.cluster_bundle):
            raise ValueError('derived transitions cluster mismatch')
        with np.load(path, allow_pickle=False) as archive:
            data = {k: archive[k] for k in archive.files}
        if not all(np.isfinite(v).all() for v in data.values()):
            raise ValueError('nonfinite derived transitions')
    klass = agent_class(args.qam_root)
    import jax
    from utils.flax_utils import restore_agent_with_file, save_agent
    obs, action = np.zeros(1024, np.float32), np.zeros(7, np.float32)
    fresh = klass.create(args.seed, obs, action, copy.deepcopy(config))
    restored = restore_agent_with_file(
        klass.create(flags['seed'], obs, action, copy.deepcopy(config)), args.checkpoint)
    params = copy.deepcopy(fresh.network.params)
    for name in params:
        if 'critic' not in name:
            params[name] = restored.network.params[name]
    agent = fresh.replace(network=fresh.network.replace(
        params=params, opt_state=fresh.network.tx.init(params)))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    flags.update(seed=args.seed, agent=dict(agent.config), stage2=vars(args),
                 manifest_sha256=sha256(args.manifest), initial_checkpoint_sha256=sha256(args.checkpoint))
    if bundle:
        flags['cluster_sha256'] = sha256(args.cluster_bundle)
    (out/'flags.json').write_text(json.dumps(flags, indent=2))
    print('TRAIN_READY', args.reward_mode, len(data['observations']), jax.devices(), flush=True)
    rng = np.random.default_rng(args.seed)
    with (out/'metrics.jsonl').open('w') as log:
        for step in range(args.warmup_steps + args.steps):
            indices = rng.integers(len(data['observations']), size=args.batch_size)
            batch = {k: v[indices] for k, v in data.items()}
            agent, info = agent.warmup(batch) if step < args.warmup_steps else agent.update(batch)
            if step % 100 == 0 or step+1 == args.warmup_steps + args.steps:
                info = {k: float(v) for k, v in jax.device_get(info).items()}
                if not all(np.isfinite(v) for v in info.values()):
                    raise FloatingPointError(f'nonfinite update at step {step}')
                log.write(json.dumps(dict(step=step+1, **info))+'\n')
                log.flush()
    path = save_agent(agent, str(out), args.steps)
    (out/'DONE.json').write_text(json.dumps({'checkpoint': path, 'sha256': sha256(path)}))


if __name__ == '__main__':
    main()

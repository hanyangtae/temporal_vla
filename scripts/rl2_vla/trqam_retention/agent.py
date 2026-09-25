"""Frozen deployed-policy adaptation of QAM/TRQAM for RL2 Bridge.

TRQAM equations audited against yonghdong/trqam commit
6f74d36baf552565fcd373198be25541db10abe3 (agents/trqam.py).
Unlike upstream, the reference is a frozen original actor_fast; no BC update.
"""
import copy
from typing import Any

import jax
import jax.numpy as jnp
from agents.qam import QAMAgent


class RetentionAgent(QAMAgent):
    lam: Any = 1.0
    kl_ema: Any = 0.0

    def critic_loss(self, batch, grad_params, rng):
        actions = (batch['actions'] * batch['action_mask']).reshape((len(batch['actions']), -1))
        nxt = batch['next_observations'][..., -1, :]
        next_actions = self.sample_actions(nxt, rng)
        qs = self.network.select('target_critic')(nxt, next_actions)
        next_q = qs.mean(0) - self.config['rho'] * qs.std(0)
        target = batch['rewards'][..., -1] + batch['discounts'] * batch['masks'][..., -1] * next_q
        q = self.network.select('critic')(batch['observations'], actions, params=grad_params)
        loss = jnp.mean((q - target)**2)
        return loss, dict(critic_loss=loss, q_mean=q.mean(), target_mean=target.mean(),
                          q_std=q.std(), td_abs=jnp.abs(q-target).mean())

    def actor_loss(self, batch, grad_params, rng):
        # Use the same stochastic trajectory and adjoint construction in both arms.
        # Upstream QAM multiplies the terminal adjoint by inv_temp. The adjoint
        # equation is linear, so divide it back out for TRQAM's unscaled -grad Q.
        xs, adjs, ts, pre = self.adj_matching(batch['observations'], rng)
        obs = jnp.repeat(batch['observations'][None], self.config['flow_steps'], axis=0)
        fine = self.network.select('actor_fast')(obs, xs, ts, params=grad_params)
        reference = self.network.select('target_actor_slow')(obs, xs, ts)
        diff = fine-reference
        h = 1/self.config['flow_steps']
        g_sq = 2*(1-ts+h)/(ts+h)
        sigma = jnp.sqrt(g_sq)
        if self.config['retention_method'] == 'trqam':
            adjs = adjs/self.config['inv_temp']
            sigma = sigma/jnp.sqrt(self.config['lam_scale']*self.lam)
        loss = jnp.square(2*diff/sigma + sigma*adjs).sum(-1).sum(0).mean()
        kl_each = (2*h/g_sq[..., 0]*jnp.square(diff).sum(-1)).sum(0)/self.config['horizon_length']
        return loss, dict(adj_loss=loss, path_kl=kl_each.mean(), path_kl_max=kl_each.max(),
                          path_kl_p95=jnp.quantile(kl_each,.95), velocity_mse=jnp.square(diff).mean(), **pre)

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        net, info = self.network.apply_loss_fn(lambda p: self.total_loss(batch, p, rng))
        params = dict(net.params)
        # Explicitly restore nontrainable modules, even if an optimizer were to
        # carry momentum. This also makes the frozen-reference contract auditable.
        for name in params:
            if name not in ('modules_actor_fast', 'modules_critic'):
                params[name] = self.network.params[name]
        params['modules_target_critic'] = jax.tree_util.tree_map(
            lambda p,t: self.config['tau']*p+(1-self.config['tau'])*t,
            params['modules_critic'], self.network.params['modules_target_critic'])
        net = net.replace(params=params)
        raw = info['actor/path_kl']
        clipped = jnp.minimum(raw, self.config['kl_clip_coef']*self.config['kl_budget'])
        ema = (1-self.config['kl_ema_coef'])*self.kl_ema+self.config['kl_ema_coef']*clipped
        lam = self.lam
        if self.config['retention_method'] == 'trqam':
            lam = jnp.clip(lam+self.config['eta_lambda']*(ema-self.config['kl_budget']),
                           self.config['lambda_min'], self.config['lambda_max'])
        info.update({'dual/lambda':lam*self.config['lam_scale'], 'dual/kl_raw':raw,
                     'dual/kl_clipped':clipped, 'dual/kl_ema':ema})
        return self.replace(network=net, rng=new_rng, lam=lam, kl_ema=ema), info


def migrate(restored, seed, config):
    """Restore every weight, copying deployed fast into frozen reference slots."""
    params = copy.deepcopy(restored.network.params)
    for name in ('modules_actor_slow','modules_target_actor_slow'):
        params[name] = copy.deepcopy(params['modules_actor_fast'])
    from flax.core import FrozenDict
    network = restored.network.replace(params=params, step=0,
                                       opt_state=restored.network.tx.init(params))
    return RetentionAgent(rng=jax.random.PRNGKey(seed), network=network,
                          config=FrozenDict(config), lam=jnp.asarray(config['lambda_init']),
                          kl_ema=jnp.asarray(0.,jnp.float32))


def inference_agent(agent):
    """Export standard QAM state so existing VLA denoise mixing loads unchanged."""
    return QAMAgent(rng=agent.rng, network=agent.network, config=agent.config)

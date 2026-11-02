"""TRQAM (trust-region QAM) on the GR00T prefix-critic QAM. JAX; host env .venvs/rl2_iid.

Port of .worktrees/trqam-retention/scripts/rl2_vla/trqam_retention/agent.py (RetentionAgent; audited against
yonghdong/trqam 6f74d36 agents/trqam.py) onto groot_robocasa_adapter_20261021/qam_prefix.PrefixCriticQAMAgent:
  * reference = frozen copy of the atomic65 actor_fast in actor_slow/target_actor_slow (migrate), no BC update
  * actor loss = sum_k ||2(v_fast - v_ref)/sigma_k + sigma_k * adj_k||^2 with adj / inv_temp and
    sigma / sqrt(lam_scale*lam)  (agent.py:37-48); adjoint from the PREFIX critic (tail-zero terminal adjoint)
  * critic = prefix-masked TD of PrefixCriticQAMAgent.critic_loss (row discounts; NOT the SIMPLER masks[-1] form)
  * update: only actor_fast and critic trained; target_critic Polyak tau; dual lambda step (agent.py:68-107)
Dimension note (open decision D-KL): path_cost divides by horizon_length (=16 here, 4 in SIMPLER) and sums over
12 action dims (7 in SIMPLER); kl_budget .1 is inherited unchanged but is not dimension-invariant.
"""
import copy
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

ADAPTER = Path(__file__).resolve().parents[2] / 'groot_robocasa_adapter_20261021'
if str(ADAPTER) not in sys.path:
    sys.path.insert(0, str(ADAPTER))
from qam_prefix import PrefixCriticQAMAgent  # noqa: E402

# trqam_retention README / train.py recipe (agent.py dual_step keys). Inherited, not re-tuned.
RETENTION_DEFAULTS = {'retention_method': 'trqam', 'kl_budget': 0.1, 'lambda_init': 1.0, 'lam_scale': 3.0,
                      'eta_lambda': 0.01, 'kl_ema_coef': 0.1, 'kl_clip_coef': 2.0, 'lambda_min': 0.01,
                      'lambda_max': 100.0}


def path_cost(diff, g_sq, h, horizon):
    """agent.py:95-97 unchanged: sum_k 2h/g_k^2 ||v_fine-v_ref||^2 / horizon (per row)."""
    return (2 * h / g_sq[..., 0] * jnp.square(diff).sum(-1)).sum(0) / horizon


def dual_step(lam, kl_ema, raw, config):
    """agent.py:100-107 unchanged."""
    clipped = jnp.minimum(raw, config['kl_clip_coef'] * config['kl_budget'])
    ema = (1 - config['kl_ema_coef']) * kl_ema + config['kl_ema_coef'] * clipped
    if config['retention_method'] == 'trqam':
        lam = jnp.clip(lam + config['eta_lambda'] * (ema - config['kl_budget']), config['lambda_min'], config['lambda_max'])
    return lam, ema, clipped


class TrqamPrefixAgent(PrefixCriticQAMAgent):
    lam: Any = 1.0
    kl_ema: Any = 0.0

    def actor_loss(self, batch, grad_params, rng):
        obs = batch['observations']
        xs, adjs, ts, pre = self.adj_matching(obs, rng)
        o = jnp.repeat(obs[None], self.config['flow_steps'], axis=0)
        fine = self.network.select('actor_fast')(o, xs, ts, params=grad_params)
        reference = self.network.select('target_actor_slow')(o, xs, ts)
        diff = fine - reference
        h = 1 / self.config['flow_steps']
        g_sq = 2 * (1 - ts + h) / (ts + h)
        sigma = jnp.sqrt(g_sq)
        if self.config['retention_method'] == 'trqam':
            adjs = adjs / self.config['inv_temp']
            sigma = sigma / jnp.sqrt(self.config['lam_scale'] * self.lam)
        loss = jnp.square(2 * diff / sigma + sigma * adjs).sum(-1).sum(0).mean()
        kl = path_cost(diff, g_sq, h, self.config['horizon_length'])
        return loss, dict(adj_loss=loss, path_kl=kl.mean(), path_kl_max=kl.max(), velocity_mse=jnp.square(diff).mean(),
                          adj_used_abs_mean=jnp.abs(adjs).mean(), **{f'adj0_qam_scaled_{k[4:]}': v for k, v in pre.items()})

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        net, info = self.network.apply_loss_fn(lambda p: self.total_loss(batch, p, rng))
        params = dict(net.params)
        for name in params:
            if name not in ('modules_actor_fast', 'modules_critic'):
                params[name] = self.network.params[name]     # frozen reference / targets restored explicitly
        params['modules_target_critic'] = jax.tree_util.tree_map(
            lambda p, t: self.config['tau'] * p + (1 - self.config['tau']) * t,
            params['modules_critic'], self.network.params['modules_target_critic'])
        net = net.replace(params=params)
        raw = info['actor/path_kl']
        lam, ema, clipped = dual_step(self.lam, self.kl_ema, raw, self.config)
        info.update({'dual/lambda': lam * self.config['lam_scale'], 'dual/lambda_internal': lam, 'dual/kl_raw': raw,
                     'dual/kl_clipped': clipped, 'dual/kl_ema': ema})
        return self.replace(network=net, rng=new_rng, lam=lam, kl_ema=ema), info


def migrate(restored, seed, retention):
    """agent.py:110-120: restore every weight, copy deployed actor_fast into the frozen reference slots, fresh opt state."""
    from flax.core import FrozenDict
    cfg = dict(restored.config)
    clash = set(cfg) & set(retention)
    if clash:
        raise ValueError(f'retention keys clash with QAM config: {sorted(clash)}')
    params = copy.deepcopy(restored.network.params)
    for name in ('modules_actor_slow', 'modules_target_actor_slow'):
        params[name] = copy.deepcopy(params['modules_actor_fast'])
    network = restored.network.replace(params=params, step=0, opt_state=restored.network.tx.init(params))
    return TrqamPrefixAgent(rng=jax.random.PRNGKey(seed), network=network, config=FrozenDict({**cfg, **retention}),
                            lam=jnp.asarray(retention['lambda_init']), kl_ema=jnp.asarray(0., jnp.float32))


def inference_agent(agent, base_config):
    """agent.py:123-125: plain prefix-QAM state (original config, retention keys dropped) for save/export."""
    return PrefixCriticQAMAgent(rng=agent.rng, network=agent.network, config=base_config)

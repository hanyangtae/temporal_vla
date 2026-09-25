"""Small CPU contract tests; real checkpoint smoke runs on the remote GPU."""
import copy
import sys
import unittest
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from agents.qam import QAMAgent,get_config
from agent import migrate
from train import tree_equal


class Contracts(unittest.TestCase):
    def test_restore_freeze_and_update(self):
        cfg=dict(get_config())
        cfg.update(horizon_length=4,action_chunking=True,best_of_n=1,actor_hidden_dims=(16,16),
                   value_hidden_dims=(16,16),num_qs=2,ob_dims=[8],action_dim=7,
                   inv_temp=.1,edit_target_entropy=None)
        original=QAMAgent.create(42,np.zeros(8,np.float32),np.zeros(7,np.float32),cfg)
        cfg=dict(original.config)
        cfg.update(retention_method='trqam',lam_scale=3.,lambda_init=1.,kl_budget=1.,
                   eta_lambda=.01,kl_ema_coef=.1,kl_clip_coef=2.,lambda_min=.01,lambda_max=100.)
        batch=dict(observations=jnp.ones((4,8)),next_observations=jnp.ones((4,1,8)),
                   actions=jnp.ones((4,4,7))*.1,action_mask=jnp.ones((4,4,7)),
                   rewards=-jnp.ones((4,1)),discounts=jnp.ones(4)*.99**4,
                   masks=jnp.ones((4,1)),valid=jnp.ones((4,1)))
        for method in ('qam','trqam'):
            agent=migrate(original,42,{**cfg,'retention_method':method})
            self.assertTrue(tree_equal(original.network.params['modules_critic'],agent.network.params['modules_critic']))
            self.assertTrue(tree_equal(original.network.params['modules_actor_fast'],agent.network.params['modules_target_actor_slow']))
            noises=jnp.ones((4,28))
            np.testing.assert_array_equal(original.compute_flow_actions(batch['observations'],noises,'fast'),agent.compute_flow_actions(batch['observations'],noises,'fast'))
            updated,info=agent.update(batch)
            self.assertTrue(all(np.isfinite(float(v)) for v in info.values()))
            self.assertEqual(float(info['actor/path_kl']),0.)
            self.assertTrue(tree_equal(agent.network.params['modules_target_actor_slow'],updated.network.params['modules_target_actor_slow']))
            self.assertTrue(tree_equal(agent.network.params['modules_actor_slow'],updated.network.params['modules_actor_slow']))
            self.assertFalse(tree_equal(agent.network.params['modules_actor_fast'],updated.network.params['modules_actor_fast']))
            self.assertFalse(tree_equal(agent.network.params['modules_critic'],updated.network.params['modules_critic']))
            if method=='trqam':self.assertLess(float(updated.lam),float(agent.lam))
            else:self.assertEqual(float(updated.lam),float(agent.lam))


if __name__=='__main__':unittest.main()

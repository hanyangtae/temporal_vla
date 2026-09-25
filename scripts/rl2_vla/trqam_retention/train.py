"""Success-only QAM/TRQAM continuation with a frozen original-policy reference."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from dataset import load_success_cache, sha256


def tree_equal(a,b):
    import jax
    la,ta=jax.tree_util.tree_flatten(a); lb,tb=jax.tree_util.tree_flatten(b)
    return ta==tb and all(np.array_equal(np.asarray(x),np.asarray(y)) for x,y in zip(la,lb))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qam-root',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--cache',required=True);p.add_argument('--output',required=True)
    p.add_argument('--method',choices=['qam','trqam'],required=True)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--steps',type=int,default=50000)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--kl-budget',type=float,default=1.)
    p.add_argument('--smoke',action='store_true',help='migration, update, frozen-reference and standard-loader tests only')
    args=p.parse_args()
    if args.steps<1 or args.batch_size<1 or args.kl_budget<=0: p.error('positive steps/batch/budget required')
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    sys.path.insert(0,str(Path(args.qam_root).resolve()))
    import jax
    import jax.numpy as jnp
    from agents.qam import QAMAgent
    from utils.flax_utils import restore_agent_with_file,save_agent
    from agent import migrate,inference_agent
    if not any(d.platform=='gpu' for d in jax.devices()):
        raise RuntimeError('Remote GPU required for this experiment')
    data,provenance=load_success_cache(args.cache)
    source_flags=json.loads(Path(args.checkpoint).with_name('flags.json').read_text())
    cfg=copy.deepcopy(source_flags['agent'])
    expected={'horizon_length':4,'action_chunking':True,'best_of_n':1,'ob_dims':[1024],
              'action_dim':7,'discount':.99,'residual':False,'target_actor':True,
              'fql_alpha':0.,'edit_scale':0.}
    for k,v in expected.items():
        if cfg[k]!=v: raise ValueError(f'unsupported checkpoint {k}: {cfg[k]}')
    if cfg['inv_temp']<=0: raise ValueError('positive QAM temperature required')
    obs=np.zeros(1024,np.float32); action=np.zeros(7,np.float32)
    template=QAMAgent.create(source_flags['seed'],obs,action,copy.deepcopy(cfg))
    original=restore_agent_with_file(template,args.checkpoint)
    expected_tree=jax.tree_util.tree_map(lambda x:(x.shape,str(x.dtype)),template.network.params)
    restored_tree=jax.tree_util.tree_map(lambda x:(x.shape,str(x.dtype)),original.network.params)
    if expected_tree!=restored_tree: raise ValueError('checkpoint parameter shape/dtype mismatch')
    cfg=dict(original.config)
    cfg.update(retention_method=args.method, lam_scale=3.,lambda_init=1.,kl_budget=args.kl_budget,
               eta_lambda=.01,kl_ema_coef=.1,kl_clip_coef=2.,lambda_min=.01,lambda_max=100.)
    agent=migrate(original,args.seed,cfg)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    flags=copy.deepcopy(source_flags)
    flags.update(seed=args.seed,agent=dict(agent.config),retention=vars(args),data=provenance,
                 initial_checkpoint_sha256=sha256(args.checkpoint),
                 reference='frozen original deployed actor_fast',optimizer='reset identically in both arms',
                 critic='restored critic AND target_critic; no warmup/reset',
                 upstream_trqam_commit='6f74d36baf552565fcd373198be25541db10abe3')
    (out/'flags.json').write_text(json.dumps(flags,indent=2))
    # Test both direct inference and every mixed-denoising vector at migration.
    probes=jnp.asarray(data['observations'][:32])
    noises=jax.random.normal(jax.random.PRNGKey(123),(len(probes),28))
    baseline=original.compute_flow_actions(probes,noises,model='fast')
    candidate=agent.compute_flow_actions(probes,noises,model='fast')
    if not np.array_equal(np.asarray(baseline),np.asarray(candidate)):
        raise AssertionError('step-0 action identity failed')
    for i in range(cfg['flow_steps']):
        t=jnp.full((len(probes),1),i/cfg['flow_steps'])
        a=original.network.select('actor_fast')(probes,noises,t)
        b=agent.network.select('actor_fast')(probes,noises,t)
        if not np.array_equal(np.asarray(a),np.asarray(b)):
            raise AssertionError(f'step-0 vector identity failed at denoise {i}')
    for key in ('modules_actor_fast','modules_critic','modules_target_critic'):
        if not tree_equal(original.network.params[key],agent.network.params[key]):
            raise AssertionError(f'weight migration failed: {key}')
    reference=copy.deepcopy(agent.network.params['modules_target_actor_slow'])
    (out/'migration.json').write_text(json.dumps(dict(strict_restore=True,action_identity=True,
                denoise_vectors_identical=cfg['flow_steps'],critic_restored=True,devices=list(map(str,jax.devices())))))
    print('MIGRATION_PASS',provenance['episodes'],provenance['transitions'],jax.devices(),flush=True)
    rng=np.random.default_rng(args.seed)
    checkpoints={1,10,100,1000,5000,10000,25000,args.steps}
    start=time.monotonic()
    steps=3 if args.smoke else args.steps
    with (out/'metrics.jsonl').open('w') as log:
        for step in range(1,steps+1):
            idx=rng.integers(len(data['observations']),size=args.batch_size)
            batch={k:v[idx] for k,v in data.items()}
            agent,info=agent.update(batch)
            # Check finiteness every update, not only reporting intervals.
            metrics={k:float(v) for k,v in jax.device_get(info).items()}
            if not all(np.isfinite(v) for v in metrics.values()):
                raise FloatingPointError(f'nonfinite update {step}: {metrics}')
            if step in checkpoints or step%100==0 or step==steps:
                if not tree_equal(reference,agent.network.params['modules_target_actor_slow']):
                    raise AssertionError('reference drift')
                if not tree_equal(reference,agent.network.params['modules_actor_slow']):
                    raise AssertionError('reference slow drift')
                actions=agent.compute_flow_actions(probes,noises,model='fast')
                metrics.update(action_mse_from_original=float(jnp.square(actions-baseline).mean()),
                               action_max_delta=float(jnp.abs(actions-baseline).max()),
                               elapsed_seconds=time.monotonic()-start)
                record=dict(step=step,**metrics)
                log.write(json.dumps(record)+'\n');log.flush()
                print(json.dumps(record),flush=True)
            if step in checkpoints or step==steps:
                export=inference_agent(agent)
                path=save_agent(export,str(out),step)
                # Sidecar keeps dual continuation state without breaking QAM inference loader.
                (out/f'dual_{step}.json').write_text(json.dumps(dict(lam=float(agent.lam),kl_ema=float(agent.kl_ema))))
                if args.smoke:
                    reloaded=restore_agent_with_file(QAMAgent.create(args.seed,obs,action,copy.deepcopy(cfg)),path)
                    if not tree_equal(export.network.params,reloaded.network.params):
                        raise AssertionError('standard QAM loader roundtrip failed')
        changed=not tree_equal(original.network.params['modules_actor_fast'],agent.network.params['modules_actor_fast'])
        if not changed: raise AssertionError('actor never updated')
    (out/'DONE.json').write_text(json.dumps(dict(steps=steps,smoke=args.smoke,checkpoint=path,
        sha256=sha256(path),actor_changed=changed,reference_frozen=True)))


if __name__=='__main__':
    main()

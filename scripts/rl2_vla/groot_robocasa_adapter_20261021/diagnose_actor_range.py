import argparse,json
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
from qam_prefix import PrefixQAMTrainer
from composition import SPEC
from baseline_data import write_json_exclusive,sha256_path
ap=argparse.ArgumentParser();ap.add_argument('--step',type=int,required=True);a=ap.parse_args()
r=Path(__file__).resolve().parents[3]
f=next((r/'outputs/rl2_vla/groot_opendrawer_102_latents_20261021/episodes').glob('episode_000001__*.npz'))
with np.load(f,allow_pickle=False) as z:
 obs=jnp.asarray(z['latents'][:16]);target=z['expert_chunks'][:16].reshape(16,192);mask=np.repeat(z['chunk_valid'][:16],12,axis=-1)
ck=r/'outputs/rl2_vla/groot_opendrawer_rl2_baseline_500k_20261021/checkpoints'/f'step_{a.step:07d}'
tr=PrefixQAMTrainer.load(ck,a.step,feature_signature=SPEC.signature());result={}
for model in ['slow','fast']:
 x=jax.random.normal(jax.random.PRNGKey(42),(len(obs),192));vel=[]
 for i in range(4):
  v=tr.agent.flow_velocity(obs,x,i,model=model);vel.append(float(jnp.max(jnp.abs(v))));x=x+v/4
 raw=np.asarray(x);clipped=np.clip(raw,-1,1);execmask=mask.copy();execmask[:,60:]=False
 result[model]={'velocity_absmax_per_denoise':vel,'raw_action_absmax':float(np.abs(raw).max()),
  'fraction_outside_unit':float((np.abs(raw)>1).mean()),'finite':bool(np.isfinite(raw).all()),
  'raw_prefix_mse':float(((raw-target)**2*execmask).sum()/execmask.sum()),
  'clipped_prefix_mse':float(((clipped-target)**2*execmask).sum()/execmask.sum())}
out=r/'outputs/analysis/groot_robocasa_lineage_20261021'/f'actor_range_step_{a.step:07d}.json'
write_json_exclusive(out,{'step':a.step,'validation_episode':1,'decisions':16,'latent_sha256':sha256_path(f),
 'checkpoint_sha256':sha256_path(ck/f'params_{a.step}.pkl'),'results':result,
 'use':'diagnostic only; not checkpoint selection or hyperparameter tuning'})
print(json.dumps(result))

"""Reset-only preflight: no policy actions or success-rate evaluation."""
import importlib.util,json
from readonly_assets import install
install()
from pathlib import Path
p=Path('/temporal_vla/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py')
s=importlib.util.spec_from_file_location('qam_eval_collector_preflight',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
out=Path('/temporal_vla/outputs/analysis/groot_robocasa_lineage_20261021/env_reset_smoke_v2')
for seed in [100000,100001]:
 env=m.make_env('OpenDrawer','pretrain',env_name='robocasa_panda_omron/OpenDrawer_PandaOmron_Env',
  scenario_seed=seed,n_action_steps=5,env_kwargs={'layout_and_style_ids':[[16,17]],'obj_instance_split':'pretrain'})
 try:
  obs,info=env.reset();meta=m.get_robocasa_ep_meta(env)
  assert meta['layout_id']==16 and meta['style_id']==17
  result={'seed':seed,'ep_meta':m.json_safe(meta),'observation_keys':list(obs),'actions_executed':0}
  with open(out/f'seed_{seed}.json','x') as f:json.dump(result,f,indent=2)
  print('RESET_PASS',seed,meta.get('lang'),flush=True)
 finally:env.close()

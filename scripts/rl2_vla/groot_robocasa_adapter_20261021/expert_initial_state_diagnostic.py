"""One held-out expert replay and fixed-GR00T diagnostic from saved simulator state."""
import argparse,copy,gzip,hashlib,importlib.util,json,time
from pathlib import Path
import numpy as np
from readonly_assets import install
install()
ROOT=Path('/temporal_vla');DATA=Path('/cache/datasets/robocasa/v1.0/pretrain/atomic/OpenDrawer/20250819/lerobot')

def dump(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['expert','groot'],required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--port',type=int,default=8896);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    import cv2,robosuite,robocasa,mujoco
    import robocasa.utils.lerobot_utils as LU
    from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
    from robocasa.utils.gym_utils.gymnasium_groot import GrootRoboCasaEnv
    from robocasa.models.robots import PandaOmronKeyConverter
    meta=json.loads((DATA/'extras/dataset_meta.json').read_text());ep=json.loads((DATA/'extras/episode_000001/ep_meta.json').read_text())
    with np.load(DATA/'extras/episode_000001/states.npz',allow_pickle=False) as z:states=z['states'].copy()
    import pandas as pd
    table=pd.read_parquet(DATA/'data/chunk-000/episode_000001.parquet')
    raw_actions=np.stack(table['action'].to_list());ref_obs=np.stack(table['observation.state'].to_list())
    native=LU.get_episode_actions(DATA,1);modality=json.loads((DATA/'meta/modality.json').read_text())
    split=json.loads((ROOT/'outputs/analysis/groot_robocasa_lineage_20261021/OpenDrawer_aux_split_v2.json').read_text())
    entry=next(x for x in split['episodes'] if x['episode_index']==1);assert entry['split']=='val'
    kw=copy.deepcopy(meta['env_args']['env_kwargs']);kw.update(has_renderer=False,has_offscreen_renderer=True,
      use_camera_obs=True,camera_names=['robot0_agentview_left','robot0_agentview_right','robot0_eye_in_hand'],
      camera_widths=512,camera_heights=512,ignore_done=True,control_freq=20,seed=0)
    env=robosuite.make(**kw)
    xml=gzip.decompress((DATA/'extras/episode_000001/model.xml.gz').read_bytes()).decode()
    initial={'model':xml,'states':states[0],'ep_meta':json.dumps(ep)}
    started=time.time()
    try:
        reset_to(env,initial)
        got=np.asarray(env.sim.get_state().flatten())
        initial_error=float(np.max(np.abs(got-states[0]))) if got.shape==states[0].shape else None
        if initial_error is None or initial_error>1e-10:raise RuntimeError(f'Saved state restoration mismatch: {got.shape}, maxerr={initial_error}')
        bridge=object.__new__(GrootRoboCasaEnv);bridge.env=env;bridge.key_converter=PandaOmronKeyConverter;bridge.verbose=False
        bridge.enable_render=True;bridge.camera_names=kw['camera_names'];bridge.camera_widths=[512]*3;bridge.camera_heights=[512]*3
        bridge.render_obs_key='robot0_agentview_left_image';bridge.render_cache=None;bridge.dump_rollout_dataset_dir=None
        def observation():
            raw=env._get_observations(force_update=True)
            return bridge.get_groot_observation(bridge.get_basic_observation(raw))
        def state_obs(obs):
            x=np.zeros(16)
            for k,v in modality['state'].items():x[v['start']:v['end']]=obs['state.'+k]
            return x
        obs=observation();initial_obs=state_obs(obs);obs_error=float(np.max(np.abs(initial_obs-ref_obs[0])))
        initial_hash=hashlib.sha256(got.tobytes()).hexdigest()
        current_ctrl=env.robots[0].composite_controller.part_controllers['right']
        initial_report={'mode':a.mode,'episode':1,'auxiliary_split':'validation','selection_reason':'Lowest indexed validation episode with complete saved XML, states and successful-expert reward metadata; not selected by new outcome.',
          'not_claimed':'Held out from the frozen GR00T original training data.',
          'source_versions':{k:meta.get(k) for k in ['robocasa_version','robosuite_version','mujoco_version']},
          'runtime_versions':{'robocasa':robocasa.__version__,'robosuite':robosuite.__version__,'mujoco':mujoco.__version__},
          'robot':'PandaOmron','layout':ep['layout_id'],'style':ep['style_id'],'instruction':ep['lang'],
          'source_files':{str(f.relative_to(DATA)):sha(f) for f in [DATA/'extras/episode_000001/states.npz',DATA/'extras/episode_000001/model.xml.gz',DATA/'extras/episode_000001/ep_meta.json',DATA/'extras/dataset_meta.json',DATA/'data/chunk-000/episode_000001.parquet']},
          'initial_state_shape':list(got.shape),'initial_state_max_abs_error':initial_error,'initial_state_sha256':initial_hash,
          'initial_observation_max_abs_error':obs_error,'initial_observation':initial_obs.tolist(),'source_initial_observation':ref_obs[0].tolist(),
          'controller_input_type':getattr(current_ctrl,'input_type',None),'controller_input_ref_frame':getattr(current_ctrl,'input_ref_frame',None),
          'expert_actions':'Raw LeRobot values reordered with official LU.get_episode_actions; no learned normalization; delta OSC input in base frame.',
          'policy_actions':'Existing native GR00T denormalization and existing PandaOmron key converter; no QAM.',
          'prediction_horizon':16,'execution_prefix':5,'inference_seed':1400000}
        dump(a.out/f'{a.mode}_initial.json',initial_report)
        np.savez(a.out/f'{a.mode}_initial_arrays.npz',sim_state=got,observation=initial_obs)
        # Stored final-state success is a separate check from free-running action replay.
        env.sim.set_state_from_flattened(states[-1]);env.sim.forward();stored_final_success=bool(env._check_success())
        reset_to(env,initial);obs=observation()
        if not np.allclose(env.sim.get_state().flatten(),states[0],rtol=0,atol=1e-10):raise RuntimeError('Second initial restoration differs')
        keys=['video.res256_image_side_0','video.res256_image_side_1','video.res256_image_wrist_0']
        frame=np.concatenate([obs[k] for k in keys],axis=1)
        writer=cv2.VideoWriter(str(a.out/f'{a.mode}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(frame.shape[1],frame.shape[0]));assert writer.isOpened()
        writer.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
        observations=[];actions=[];successes=[];state_errors=[];first_success=None;calls=0
        client=None
        if a.mode=='groot':
            q=ROOT/'scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py';spec=importlib.util.spec_from_file_location('expert_diag_collector',q);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
            client=m.N15LerobotHttpFeatureClient(f'http://127.0.0.1:{a.port}',inference_seed=1400000,no_features=True,expect_chunk_len=16)
            client.reset()
        limit=len(raw_actions) if a.mode=='expert' else 720
        chunk=None
        for step in range(limit):
            observations.append(state_obs(obs))
            if a.mode=='expert':
                act={f'action.{k}':raw_actions[step,v['start']:v['end']].copy() for k,v in modality['action'].items()}
            else:
                if step%5==0:chunk,_=client.get_action(obs);calls+=1
                act={k:np.asarray(v[step%5]).copy() for k,v in chunk.items()}
            mapped=PandaOmronKeyConverter.unmap_action(copy.deepcopy(act));cc=env.robots[0].composite_controller
            vec=np.zeros(cc.action_limits[0].shape)
            for part,(lo,hi) in cc._action_split_indexes.items():vec[lo:hi]=mapped['robot0_'+part]
            vec[-1]=mapped['robot0_base_mode']
            if a.mode=='expert' and not np.array_equal(vec,native[step]):raise RuntimeError('Official raw action vs bridge mapping mismatch')
            if not np.isfinite(vec).all():raise RuntimeError('Nonfinite native action')
            actions.append(vec)
            obs,reward,terminated,truncated,info=bridge.step(copy.deepcopy(act))
            success=bool(env._check_success());successes.append(success)
            if success and first_success is None:first_success=step+1
            if a.mode=='expert' and step+1<len(states):state_errors.append(float(np.max(np.abs(env.sim.get_state().flatten()-states[step+1]))))
            if step%2==0:writer.write(cv2.cvtColor(np.concatenate([obs[k] for k in keys],axis=1),cv2.COLOR_RGB2BGR))
            if step%50==0:print(a.mode,'step',step+1,'success',success,flush=True)
            if a.mode=='groot' and success:break
        writer.release()
        np.savez_compressed(a.out/f'{a.mode}_trajectory.npz',observations=np.asarray(observations),native_actions=np.asarray(actions),success=np.asarray(successes),state_max_abs_errors=np.asarray(state_errors))
        result={'mode':a.mode,'status':'completed','steps':len(actions),'success':any(successes),'first_success_step':first_success,
          'stored_final_state_success':stored_final_success,'initial_state_sha256':initial_hash,'initial_state_max_abs_error':initial_error,
          'initial_observation_max_abs_error':obs_error,'native_actions_finite':True,'http_predictions':calls,
          'replay_state_error_first':state_errors[0] if state_errors else None,'replay_state_error_max':max(state_errors) if state_errors else None,
          'seconds':time.time()-started,'checkpoint_selection_changed':False}
        dump(a.out/f'{a.mode}_result.json',result);print(json.dumps(result),flush=True)
    finally:env.close()
if __name__=='__main__':main()

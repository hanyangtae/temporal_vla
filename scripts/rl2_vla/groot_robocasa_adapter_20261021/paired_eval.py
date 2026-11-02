"""Paired OpenDrawer evaluation through the existing collector and GPU harness.
No fitting or selection from evaluation outcomes. Run only after selection_final.json.
"""
import argparse,csv,hashlib,json,os,socket,subprocess,time,urllib.request
from pathlib import Path
from baseline_data import sha256_path,write_json_exclusive
ROOT=Path(__file__).resolve().parents[3]
ADAPTER=Path(__file__).resolve().parent
MODEL_IMAGE='sha256:55008acf7d21c1ec2aefc72593b2248caec8094c6aa1829cf907ba98d9e373c9'
ENV_IMAGE='sha256:e6e62329e8768a9a30d76a08d64322ee48a608971b9a67db7afe57e1116bef3a'
PROFILE='configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml'
COLLECTOR='scripts/rl2_vla/groot_robocasa_adapter_20261021/collect_qam.py'
ENV_SEEDS=[100000,100001]
NOISE_SEED=1400000

def inside(p):return '/temporal_vla/'+str(Path(p).resolve().relative_to(ROOT))
def get_json(port,route):
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/{route}',timeout=5) as r:return json.load(r)
def docker_args(out,gpu,name,image,network='host'):
    cache=Path('/home/dongkyu/.cache')
    return ['docker','run','--rm','--pull=never','--name',name,'--hostname','kanu',
      '--gpus',f'device={gpu}','--network',network,'--shm-size','1g','--entrypoint','python',
      '-v',f'{ROOT}:/temporal_vla:ro','-v',f'{out}:{inside(out)}:rw',
      '-v',f'{cache}/temporal_vla:/cache:ro','-v',f'{cache}/huggingface:{cache}/huggingface:ro',
      '-e','HF_HOME=/cache/datasets/huggingface','-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1',
      '-e','HF_MODULES_CACHE=/tmp/hf_modules','-e','PYTHONDONTWRITEBYTECODE=1',
      '-e','OMP_NUM_THREADS=4','-e','OPENBLAS_NUM_THREADS=4','-e','TOKENIZERS_PARALLELISM=false',
      '-e','MPLCONFIGDIR=/tmp/mpl','-e',f'PDK_LOG_FILE={inside(out)}/{name}.module.log',
      '-e','MUJOCO_GL=egl','-e','PYTHONPATH=/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla',
      '-w','/temporal_vla',image]

def ensure_absent(name):
    r=subprocess.run(['docker','inspect',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10)
    if r.returncode==0:raise RuntimeError('Refusing pre-existing container name: '+name)

def stop_owned(name):
    # Names are created by this invocation; never operate on a pre-existing name.
    subprocess.run(['docker','stop','--time','10',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=20)

def run(a):
    if not os.environ.get('HARNESS_RECEIPT'):raise RuntimeError('Use run_paired_eval.sh GPU harness')
    if a.gpu!=int(os.environ['GPUS']):raise RuntimeError('GPU differs from harness')
    a.out=a.out.resolve();a.actor=a.actor.resolve();a.selection=a.selection.resolve()
    meta=json.loads((a.actor/'actor.json').read_text());sel=json.loads(a.selection.read_text())
    if meta['diagnostic_only'] or not sel['production_run']:raise RuntimeError('Production selected actor required')
    if sel['evaluations'][-1]['step']!=500000:raise RuntimeError('500k completion required')
    if meta['source_checkpoint']['step']!=sel['selected_step']:raise RuntimeError('Selected step mismatch')
    if a.out.exists():raise RuntimeError('Fresh output directory required; preserve earlier artifacts')
    # Verify selected checkpoint files against their recorded hashes before model use.
    selected=a.selection.parent/'checkpoints'/sel['selected_checkpoint']['dir']
    for name,digest in sel['selected_checkpoint']['files'].items():
        if sha256_path(selected/name)!=digest['sha256']:raise RuntimeError('Selected checkpoint hash mismatch')
    with socket.socket() as sock:sock.bind(('127.0.0.1',a.port))
    a.out.mkdir(parents=True)
    import sys
    sys.path.insert(0,str(ROOT))
    from src.collect.plan import CollectionPlan
    plan=CollectionPlan(name='OpenDrawer QAM pilot paired evaluation',model='groot_n15',version='20261021',
      ckpt='robocasa/robocasa365_checkpoints@c484448aba1a9b60a04c9b0ca117241518ea69f3/gr00t_n1-5/multitask_learning/checkpoint-120000',
      capture_layers=[],denoise_k=4,token_mode='none',instructions={'OpenDrawer':ENV_SEEDS},noise_seeds=[NOISE_SEED],
      note='Same expert-seen layout16/style17; fresh reset seeds; baseline pilot only, no TRQAM adaptation.',
      extra={'env_kwargs':{'layout_and_style_ids':[[16,17]],'obj_instance_split':'pretrain'}})
    plan_path=plan.save(a.out/'plan')
    contract={'scope':'OpenDrawer92/10 auxiliary pilot; two fresh rollouts per arm; not 300task or TRQAM',
      'arms':['base','gate0','gate1'],'qam_weight':.5,'plan_id':plan.plan_id,'env_seeds':ENV_SEEDS,
      'inference_seed':NOISE_SEED,'max_steps':720,'predict_horizon':16,'execute_steps':5,
      'gpu':a.gpu,'machine':'kanu','actor_metadata_sha256':sha256_path(a.actor/'actor.json'),
      'selection_sha256':sha256_path(a.selection),'selected_step':sel['selected_step'],
      'evaluation_use':'never training or checkpoint selection',
      'historical_demo_reset_seeds':'not published; cannot assert numeric seed disjointness with original demos',
      'new_rollouts':'generated after frozen validation selection; no reuse of expert trajectories',
      'images':{'model':MODEL_IMAGE,'environment':ENV_IMAGE},
      'code_sha256':{f:sha256_path(ADAPTER/f) for f in ['paired_eval.py','serve_qam.py','qam_torch_actor.py','composition.py','readonly_assets.py','collect_qam.py']}}
    write_json_exclusive(a.out/'eval_contract.json',contract)
    rows=[]
    for arm in contract['arms']:
        name=f'groot-qam-{arm}-{os.getpid()}'
        argv=docker_args(a.out,a.gpu,name,MODEL_IMAGE)+[
          'scripts/rl2_vla/groot_robocasa_adapter_20261021/serve_qam.py','--qam-arm',arm,
          '--qam-actor',inside(a.actor),'--qam-weight','.5','--profile',PROFILE,'--host','127.0.0.1','--port',str(a.port)]
        log=open(a.out/f'{arm}.serve.log','x')
        ensure_absent(name)
        proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+240
            while True:
                if proc.poll() is not None:raise RuntimeError(f'{arm} server exited {proc.returncode}')
                try:
                    q=get_json(a.port,'qam_contract');health=get_json(a.port,'health')
                    if q['ready'] and q['arm']==arm:break
                except (OSError,ValueError,KeyError):pass
                if time.monotonic()>deadline:raise RuntimeError(f'{arm} readiness timeout')
                time.sleep(2)
            write_json_exclusive(a.out/f'{arm}.serve_identity.json',{'qam':q,'health':health})
            for i,seed in enumerate(ENV_SEEDS):
                collector_name=f'groot-qam-collect-{arm}-{i}-{os.getpid()}'
                cmd=docker_args(a.out,a.gpu,collector_name,ENV_IMAGE)+[COLLECTOR,
                  '--vla-server',f'http://127.0.0.1:{a.port}','--task','OpenDrawer',
                  '--env-name','robocasa_panda_omron/OpenDrawer_PandaOmron_Env',
                  '--split','pretrain','--output-dir',inside(a.out/'raw_rollouts'),
                  '--episode-start-idx',str(i),'--n-episodes','1','--seed',str(seed),
                  '--inference-seed',str(NOISE_SEED),'--n-action-steps','5','--max-episode-steps','720',
                  '--video-fps','20','--steps-per-render','2','--wait-ready','--no-features','--expect-chunk-len','16',
                  '--plan-json',inside(plan_path),'--scene-idx',str(i),'--noise-idx','0',
                  '--grid-instruction','OpenDrawer','--grid-root',inside(a.out/'grid'),
                  '--arm-dir',f'qam_{arm}','--run-tag',sha256_path(a.out/'eval_contract.json')]
                ensure_absent(collector_name)
                with open(a.out/f'{arm}.episode{i}.log','x') as f:
                    try:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=1800,check=True)
                    finally:stop_owned(collector_name)
            write_json_exclusive(a.out/f'{arm}.after.json',get_json(a.port,'qam_contract'))
        finally:
            stop_owned(name)
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:raise RuntimeError('Owned server Docker client did not exit after stop')
            log.close()
    for f in (a.out/'grid').rglob('meta.json'):
        d=json.loads(f.read_text());arm=f.parent.name
        if arm not in {'qam_base','qam_gate0','qam_gate1'}:raise RuntimeError('Unexpected arm path')
        rows.append({'arm':arm,'seed':d['scenario_seed'],'inference_seed':d['inference_seed'],
          'success':d['episode_success'],'steps':d['steps'],'task_description':d['task_description'],
          'meta':str(f.relative_to(a.out)),'trajectory_sha256':sha256_path(f.parent/'traj.csv')})
    if len(rows)!=6:raise RuntimeError(f'Expected 6 completed episodes, got {len(rows)}')
    lookup={(r['arm'],r['seed']):r for r in rows}
    for seed in ENV_SEEDS:
        b,z=lookup['qam_base',seed],lookup['qam_gate0',seed]
        if b['trajectory_sha256']!=z['trajectory_sha256']:raise RuntimeError('base/gate0 trajectory mismatch')
        if len({lookup[arm,seed]['task_description'] for arm in ['qam_base','qam_gate0','qam_gate1']})!=1:
            raise RuntimeError('Paired instruction mismatch')
    with open(a.out/'per_episode.tsv','x') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
    write_json_exclusive(a.out/'evaluation_complete.json',{'completed':6,'base_gate0_trajectory_equal':True,
      'success_counts':{arm:sum(r['success'] for r in rows if r['arm']==arm) for arm in ['qam_base','qam_gate0','qam_gate1']},
      'n_per_arm':2,'claim':'diagnostic evidence only; confound audit required before interpretation'})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--actor',type=Path,required=True);ap.add_argument('--selection',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--gpu',type=int,default=4);ap.add_argument('--port',type=int,default=8896)
    a=ap.parse_args()
    try:run(a)
    except Exception as exc:
        if a.out.is_dir():write_json_exclusive(a.out/'FAILED.json',{'error':str(exc),'type':type(exc).__name__,'time':time.time()})
        raise
if __name__=='__main__':main()

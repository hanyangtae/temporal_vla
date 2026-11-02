"""Continue an existing training run; never starts or stops training.
Audit each saved milestone; after completion export selected actor and run existing
harness-controlled smoke/evaluation. Resource denial stops this continuation.
"""
import fcntl,json,math,os,subprocess,time
from pathlib import Path
from baseline_data import sha256_path,write_json_exclusive
ROOT=Path(__file__).resolve().parents[3];ADAPTER=Path(__file__).resolve().parent
RUN=ROOT/'outputs/rl2_vla/groot_opendrawer_rl2_baseline_500k_20261021'
OUT=ROOT/'outputs/analysis/groot_robocasa_lineage_20261021/baseline_continuation_v1'
PY=ROOT/'.venvs/rl2_iid/bin/python'
ACTOR=ROOT/'outputs/rl2_vla/groot_opendrawer_selected_actor_20261021'
SMOKE=OUT/'selected_gate1_smoke';EVAL=ROOT/'outputs/eval/robocasa/groot_n15/qam_opendrawer_pilot_20261021'
TRAIN_PID=3342388

def command(argv,timeout,env=None):
    print('COMMAND',argv,flush=True)
    subprocess.run([str(x) for x in argv],cwd=ROOT,env=env,check=True,timeout=timeout)

def gpu_env():
    gpu_lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    candidates=[]
    for line in gpu_lines:
        i,u,mem=[x.strip() for x in line.split(',')]
        if int(i) in [4,5,6,7] and u not in apps and int(mem)<100:candidates.append(int(i))
    if not candidates:raise RuntimeError('No idle GPU among4..7; no preemption or reservation bypass')
    gpu=candidates[0];env=os.environ.copy()
    env.update(HARNESS_MACHINE='kanu',HARNESS_SESSION='groot-qam-20261021',GPUS=str(gpu),SERVES_PER_GPU='1',PYTHONDONTWRITEBYTECODE='1')
    return gpu,env

def monitor():
    OUT.mkdir(exist_ok=True)
    lock=open(OUT/'monitor.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write_json_exclusive(OUT/'started.json',{'pid':os.getpid(),'training_pid':TRAIN_PID,'time':time.time(),
      'scope':'Monitor existing92/10pilot then selected checkpoint smoke and paired evaluation; no TRQAM',
      'code_sha256':{f:sha256_path(ADAPTER/f) for f in ['continue_baseline.py','paired_eval.py','gpu_final_smoke.py']}})
    seen=set();deadline=time.monotonic()+4*3600
    while True:
        lines=(RUN/'progress.jsonl').read_text().splitlines();records=[]
        for line in lines:
            try:records.append(json.loads(line))
            except json.JSONDecodeError:
                if line!=lines[-1]:raise
        for rec in records:
            for key,val in rec.items():
                if isinstance(val,float) and not math.isfinite(val):raise RuntimeError(f'Nonfinite progress metric {key}')
            if rec.get('event')=='eval' and rec['step'] not in seen:
                step=rec['step'];selection=json.loads((RUN/'selection'/f'after_step_{step:07d}.json').read_text())
                checkpoint=RUN/'checkpoints'/rec['checkpoint_dir']
                files={f.name:{'sha256':sha256_path(f),'bytes':f.stat().st_size} for f in checkpoint.iterdir()}
                if sum(x['bytes'] for x in files.values())!=rec['checkpoint_bytes']:raise RuntimeError('Checkpoint byte mismatch')
                if selection['best_step_so_far']==step and files!=selection['best_checkpoint_sha256']:raise RuntimeError('Best checkpoint hash mismatch')
                receipt={'eval':rec,'selection':selection,'checkpoint_files':files,'checked_time':time.time()}
                write_json_exclusive(OUT/f'checkpoint_{step:07d}.json',receipt);seen.add(step)
                print('MILESTONE',step,rec['val/action_mse_exec_prefix'],flush=True)
        if (RUN/'FAILED.json').exists():raise RuntimeError('Training FAILED.json present')
        if (RUN/'selection_final.json').exists() and any(r.get('event')=='done' for r in records):break
        cmdline=Path(f'/proc/{TRAIN_PID}/cmdline')
        if not cmdline.exists() or b'train_qam_baseline.py' not in cmdline.read_bytes():raise RuntimeError('Training exited without completion')
        if time.monotonic()>deadline:raise RuntimeError('Monitoring4h limit reached; training not stopped')
        last=max((r.get('time',0) for r in records))
        if time.time()-last>300:raise RuntimeError('No training progress for5min; training not stopped')
        time.sleep(30)
    final=json.loads((RUN/'selection_final.json').read_text())
    if [r['step'] for r in final['evaluations']]!=list(range(0,500001,50000)):raise RuntimeError('Missing validation milestone')
    best=min(final['evaluations'],key=lambda r:(r['metrics']['val/action_mse_exec_prefix'],r['step']))
    if best['step']!=final['selected_step']:raise RuntimeError('Validation selection mismatch')
    for rec in final['evaluations']:
        c=rec['checkpoint']
        for name,digest in c['files'].items():
            if sha256_path(RUN/'checkpoints'/c['dir']/name)!=digest['sha256']:raise RuntimeError('Final checkpoint hash mismatch')
    write_json_exclusive(OUT/'training_verified.json',{'step':500000,'selected_step':best['step'],
      'selected_value':final['selected_value'],'selection_sha256':sha256_path(RUN/'selection_final.json'),
      'all11_checkpoints_hash_verified':True,'walltime_s':final['walltime_s']})
    cpu=os.environ.copy();cpu.update(JAX_PLATFORMS='cpu',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
    command([PY,ADAPTER/'qam_torch_actor.py','--checkpoint-dir',RUN/'checkpoints'/best['checkpoint']['dir'],
      '--step',best['step'],'--selection',RUN/'selection_final.json','--out',ACTOR],180,cpu)
    gpu,env=gpu_env()
    command(['bash',ADAPTER/'run_gpu_final_smoke.sh','--actor',ACTOR,'--out',SMOKE,'--gpu',gpu],360,env)
    result=json.loads((SMOKE/'result.json').read_text())
    if result['status']!='passed' or result['diagnostic_only']:raise RuntimeError('Selected actor smoke invalid')
    gpu,env=gpu_env()
    command(['bash',ADAPTER/'run_paired_eval.sh','--actor',ACTOR,'--selection',RUN/'selection_final.json',
      '--out',EVAL,'--gpu',gpu],12000,env)
    summary=json.loads((EVAL/'evaluation_complete.json').read_text())
    write_json_exclusive(OUT/'COMPLETE.json',{'training':500000,'selected_step':best['step'],'evaluation':summary,
      'TRQAM':'pending scenario answer; not run','full300':'not run','time':time.time()})
    print('CONTINUATION_COMPLETE',flush=True)
if __name__=='__main__':
    try:monitor()
    except Exception as e:
        OUT.mkdir(exist_ok=True)
        write_json_exclusive(OUT/'BLOCKED.json',{'error':str(e),'type':type(e).__name__,'time':time.time(),
          'training_stopped_by_this_script':False})
        raise

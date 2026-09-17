#!/usr/bin/env python3
"""Reproduce an explicit baseline manifest through the shared GR00T harness."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / 'configs/experiments/kanu_base_recovery_mesa_20260917'
OUT = ROOT / 'outputs/collect/kanu_base_recovery_mesa_20260917'
CONT = Path('/temporal_vla')
PORTS = [8860, 8861]
GPU = 2
SESSION = 'codex-kanu-base-mesa-20260917'

def cp(path):
    return str(CONT / Path(path).resolve().relative_to(ROOT))

def save(name, value):
    dest = OUT / name
    temp = dest.with_suffix(dest.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(dest)

def checked(args, **kw):
    return subprocess.run(args, check=True, **kw)

def verify_receipt():
    assert os.environ['HARNESS_SESSION'] == SESSION
    assert os.environ['GPUS'] == '2'
    checked(['python3', str(ROOT/'scripts/utils/groot_harness.py'), 'verify',
             '--machine','kanu','--gpus','2','--serves','2','--kind','eval',
             '--session',SESSION,'--receipt',os.environ['HARNESS_RECEIPT'],
             '--consumer-pid',str(os.getpid())])

def stop_container_pid(container, pidfile, required):
    code = ('import os,signal; from pathlib import Path; '
            f'p=Path({cp(pidfile)!r}); '
            'pid=int(p.read_text()); q=Path(f"/proc/{pid}/cmdline"); '
            f'assert not q.exists() or {required.encode()!r} in q.read_bytes(); '
            'os.kill(pid,signal.SIGTERM) if q.exists() else None')
    if pidfile.exists():
        subprocess.run(['docker','exec',container,'python','-c',code],check=False)

def start_server(port):
    log = open(OUT/'logs'/f'serve_{port}.log','ab')
    pidfile=OUT/f'serve_{port}.pid'
    args=['python','scripts/serve/lerobot.py','--profile',
          'configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml',
          '--host','127.0.0.1','--port',str(port),'--device','cuda','--collect','--capture-vl',
          '--groot-dit-token-pool','all_token_full','--groot-dit-capture-layers','0,2,4,8,10,12,15']
    command='cd /temporal_vla && echo $$ > '+shlex.quote(cp(pidfile))+' && exec '+shlex.join(args)
    proc=subprocess.Popen(['docker','exec','-e','CUDA_VISIBLE_DEVICES=2',
        '-e','OMP_NUM_THREADS=4','-e','OPENBLAS_NUM_THREADS=4','-e','MKL_NUM_THREADS=4',
        'lerobot','bash','-lc',command],stdout=log,stderr=subprocess.STDOUT)
    for _ in range(180):
        if proc.poll() is not None:
            raise RuntimeError(f'server {port} exited; see log')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as r:
                health=json.load(r)
            if health.get('status')=='ok':
                save(f'health_{port}.json',health)
                print('HEALTH',port,flush=True)
                return proc
        except (OSError,ValueError):
            pass
        time.sleep(3)
    raise RuntimeError(f'server {port} startup timeout')

def collect(row,port):
    dest=OUT/'grid'/'4fa6496cd684'/'kanu'/row['instruction']/f"s{row['scene']}"/f"j{row['jitter']}"/f"n{row['noise']}"/'base'
    if (dest/'meta.json').exists() and (dest/'rollout.pkl').exists():
        return audit(row,dest,True)
    prefix='/temporal_vla/.claude/worktrees/collect-topup'
    pypath=prefix+':/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla/src/benchmarks/robocasa:/temporal_vla/src/benchmarks/robosuite:/temporal_vla'
    args=['python',prefix+'/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py',
          '--vla-server',f'http://127.0.0.1:{port}','--task','PickPlaceCounterToCabinet',
          '--env-name',row['env_name'],'--output-dir',cp(OUT/'work'/row['id']),
          '--grid-root',cp(OUT/'grid'),'--plan-json',cp(CONFIG/'collection_plan.json'),
          '--scene-idx',str(row['scene']),'--jitter-idx',str(row['jitter']),
          '--noise-idx',str(row['noise']),'--grid-instruction',row['instruction'],
          '--canonical-instruction',row['lang'],'--episode-start-idx','0','--n-episodes','1',
          '--seed',str(row['env_seed']),'--inference-seed',str(row['inference_seed']),
          '--n-action-steps','5','--max-episode-steps','720','--video-fps','20',
          '--steps-per-render','2','--wait-ready','--ep-meta-dir',cp(OUT/'ep_meta')]
    pidfile=OUT/f"collector_{port}.pid"
    command='echo $$ > '+shlex.quote(cp(pidfile))+' && exec '+shlex.join(args)
    logpath=OUT/'logs'/f"{row['id']}.log"
    print('START',row['id'],'port',port,flush=True)
    with open(logpath,'ab') as log:
        proc=subprocess.Popen(['docker','exec','-e','MUJOCO_GL=egl','-e','CUDA_VISIBLE_DEVICES=',
            '-e','MUJOCO_EGL_DEVICE_ID=0','-e','LP_NUM_THREADS=2',
            '-e','__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json',
            '-e','OMP_NUM_THREADS=2','-e','OPENBLAS_NUM_THREADS=2',
            '-e','PYTHONPATH='+pypath,'robocasa','bash','-lc',command],stdout=log,stderr=subprocess.STDOUT)
        rc=proc.wait()
    if rc:
        raise RuntimeError(f"{row['id']}: collector exit {rc}; {logpath}")
    return audit(row,dest,False)

def audit(row,dest,reused):
    m=json.loads((dest/'meta.json').read_text())
    for k,expected in [('env_seed',row['env_seed']),('inference_seed',row['inference_seed']),
        ('layout_id',row['layout']),('style_id',row['style'])]:
        assert str(m[k])==str(expected),(row['id'],k,m[k],expected)
    assert m['machine']=='kanu'
    assert (dest/'rollout.pkl').stat().st_size>0 and (dest/'video.mp4').stat().st_size>0
    result={'id':row['id'],'output':str(dest.relative_to(ROOT)),'original_success':row['original_success'],
            'reproduced_success':int(m['success']),'outcome_match':int(m['success'])==row['original_success'],
            'original_sig':row['sig'],'reproduced_sig':m['sig'],'pkl_sha_match':m['pkl_sha256']==row['pkl_sha256'],
            'full_action_state_match':'not_yet_verified','reused':reused}
    print('DONE',row['id'],'success',m['success'],'original',row['original_success'],flush=True)
    return result

def main():
    verify_receipt()
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
    rows=json.loads((CONFIG/'targets.json').read_text());assert len(rows)==22
    contract=json.loads((CONFIG/'contract.json').read_text())
    assert hashlib.sha256((CONFIG/'collection_plan.json').read_bytes()).hexdigest()==contract['plan_sha256']
    assert socket.gethostname().split('.')[0]=='kanu'
    for port in PORTS:
        with socket.socket() as s:assert s.connect_ex(('127.0.0.1',port))!=0,f'port {port} busy'
    results=[];failures=[];servers=[]
    save('state.json',{'phase':'starting','pid':os.getpid(),'total':22,'completed':0,'gpu':2,'ports':PORTS})
    def interrupted(sig,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    try:
        for port in PORTS:servers.append(start_server(port))
        save('state.json',{'phase':'running','pid':os.getpid(),'total':22,'completed':0,'gpu':2,'ports':PORTS})
        def lane(w):
            for row in rows[w::2]:
                try:yield collect(row,PORTS[w])
                except Exception as e:raise RuntimeError(str(e)) from e
        def run_lane(w):
            local=[]
            for result in lane(w):
                local.append(result)
                save(f'results_lane{w}.json',local)
            return local
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            future={pool.submit(run_lane,w):w for w in range(2)}
            for f in concurrent.futures.as_completed(future):
                try:results.extend(f.result())
                except Exception as e:failures.append({'lane':future[f],'error':str(e)})
        # Lane files preserve partial results even when a later episode fails.
        results=[r for w in range(2) for r in json.loads((OUT/f'results_lane{w}.json').read_text())] if all((OUT/f'results_lane{w}.json').exists() for w in range(2)) else results
        save('results.json',results)
        save('state.json',{'phase':'complete' if len(results)==22 and not failures else 'failed',
             'total':22,'completed':len(results),'failures':failures,'gpu':2})
        if failures:raise RuntimeError(str(failures))
    finally:
        for port in PORTS:stop_container_pid('robocasa',OUT/f'collector_{port}.pid','http_feature_collect.py')
        for port in PORTS:stop_container_pid('lerobot',OUT/f'serve_{port}.pid',f'--port\x00{port}')
        for proc in servers:
            try:proc.wait(timeout=30)
            except subprocess.TimeoutExpired:print('WARNING server client still alive',proc.pid,flush=True)
        print('CLEANUP completed',flush=True)

if __name__=='__main__':main()

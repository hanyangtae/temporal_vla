"""Complete the four-arm IID evaluation, reusing only verified fresh-env episodes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from analyze import summarize, validate_paired_rng
from data import TASKS, load_episode, sha256

ARMS = ('vanilla', 'qam_original', 'qam_base', 'qam_shaped')
SEEDS = (42, 0, 7)


def missing_batches():
    # 296 reused (288 trained-QAM + 8 natural-gate smoke vanilla); 904 new.
    return [
        ('original', ['qam_original'], list(SEEDS), 10000, 25),
        ('vanilla42', ['vanilla'], [42], 10002, 23),
        ('vanilla07', ['vanilla'], [0, 7], 10000, 25),
        ('trained_remaining', ['qam_base', 'qam_shaped'], list(SEEDS), 10012, 13),
    ]


def expected_keys(arm, reused=False):
    if reused and arm == 'qam_original':
        return set()
    seeds = (42,) if reused and arm == 'vanilla' else SEEDS
    trials = (2 if arm == 'vanilla' else 12) if reused else 25
    return {f'{t}/env{e}/policy{s}' for t in TASKS for s in seeds for e in range(10000, 10000 + trials)}


def atomic(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[3])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reuse-root', type=Path, required=True)
    p.add_argument('--run', action='store_true')
    args = p.parse_args()
    root, out, reuse = args.repo.resolve(), args.output.resolve(), args.reuse_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (out/'pipeline.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    train = root/'outputs/rl2_iid_20260916/training'
    cache = root/'outputs/rl2_iid_training_cache_20260916'
    py = root/'.venvs/rl2_iid/bin/python'
    source = root/'scripts/rl2_vla/stage2_cluster_reward'
    policy = root/'outputs/rl2_assets/pi0_bridge'
    safe = train/'safe_sweep/selected'
    checkpoints = dict(qam_original=root/'outputs/rl2_assets/qam_bridge/rl2_vla_qam_bridge_500k.pkl',
                       qam_base=train/'qam_base/params_50000.pkl', qam_shaped=train/'qam_cluster_potential/params_50000.pkl')
    assert json.loads((reuse/'status.json').read_text())['stage'] == 'complete'
    assert (reuse/'smoke_gateoff.PASS.json').is_file() and (reuse/'smoke_natural.PASS.json').is_file()
    # Freeze exact dev code and loaded upstream patch; no checkpoint data in git.
    commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip()
    files = list(source.glob('*.py')) + [root/'scripts/rl2_vla/patches/rl2_vla_stage2_single_candidate.patch']
    upstream = ['README.md','RL2_CoVer_VLA/simpler/eval_utils.py','RL2_CoVer_VLA/simpler/rl2_utils.py','RL2_CoVer_VLA/simpler/run_simpler_eval_with_openpi.py']
    files += [root/'RL2-VLA'/f for f in upstream]
    contract = dict(dev_commit=commit, rng='paired_rng_v2', environment='fresh_env_per_episode_v1',
                    sha256={str(f.relative_to(root)):sha256(f) for f in files},
                    checkpoint_sha256={a:sha256(f) for a,f in checkpoints.items()},
                    policy_sha256=sha256(policy/'model.safetensors'), safe_sha256=sha256(safe/'provenance.json'),
                    reuse_root=str(reuse), total=1200, reused=296, new=904)
    old_contract=out/'contract.json'
    if old_contract.exists() and json.loads(old_contract.read_text()) != contract:
        raise ValueError('full evaluation restart contract changed')
    atomic(old_contract, contract)
    atomic(out/'plan.json',dict(batches=missing_batches(),episodes_per_arm=300,reused=296,new=904))
    if not args.run:
        return
    # Verify checked-in source equals the executing source (other worktree edits are allowed).
    for f in [p for p in files if not p.is_relative_to(root/'RL2-VLA')]:
        blob = subprocess.check_output(['git','show',f'HEAD:{f.relative_to(root)}'],cwd=root)
        if blob != f.read_bytes():
            raise ValueError(f'executing file differs from dev commit: {f}')
    subprocess.run(['git','-C',str(root/'RL2-VLA'),'apply','--reverse','--check','--unidiff-zero',str(root/'scripts/rl2_vla/patches/rl2_vla_stage2_single_candidate.patch')],check=True)
    arms={a:{} for a in ARMS}; records=[]
    for arm in ('vanilla','qam_base','qam_shaped'):
        base = reuse/('smoke_natural' if arm == 'vanilla' else 'eval')/arm
        for path in sorted(base.rglob('episode_*.json')):
            ep=load_episode(path); request=json.loads((path.parent/'request.json').read_text())
            if ep.get('rng_contract')!='paired_rng_v2' or ep.get('environment_contract')!='fresh_env_per_episode_v1' or ep.get('force_gate_off') is not False:
                raise ValueError(f'legacy/diagnostic result rejected: {path}')
            if request['policy']!=str(policy) or ep['policy_checkpoint']!=str(policy):
                raise ValueError('reused policy mismatch')
            if arm!='vanilla' and (request['safe_sha256']!=contract['safe_sha256'] or request['qam_sha256']!=contract['checkpoint_sha256'][arm]):
                raise ValueError('reused SAFE/QAM mismatch')
            if ep['episode_id'] in arms[arm]: raise ValueError('duplicate reuse')
            arms[arm][ep['episode_id']]=ep
            records.append(dict(arm=arm,episode_id=ep['episode_id'],source=str(path),sha256=sha256(path)))
        if set(arms[arm])!=expected_keys(arm,reused=True):raise ValueError(f'reuse coverage mismatch {arm}')
    atomic(out/'reuse_manifest.json',records)
    claimed=[]; owner='codex-rl2-full-paired-20260917'
    def status(stage,**kw):atomic(out/'status.json',dict(stage=stage,pid=os.getpid(),time=time.time(),gpus=claimed,**kw))
    def slots():
        while True:
            rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
            for row in rows:
                gpu,mem=map(int,row.split(','))
                if gpu not in (2,3,4) or gpu in claimed or mem>64:continue
                rc=subprocess.run(['bash','scripts/utils/gpu_lease.sh','claim','kanu',str(gpu),owner,'full paired RL2 evaluation','24'],cwd=root,env=dict(os.environ,LEASE_MODEL='rl2',LEASE_PID=str(os.getpid()))).returncode
                if rc==0:claimed.append(gpu)
            if claimed:return list(map(str,claimed))
            status('waiting_free_gpu');time.sleep(30)
    try:
        for name, selected, seeds, start, trials in missing_batches():
            for rel,digest in contract['sha256'].items():
                if sha256(root/rel)!=digest:raise ValueError(f'source changed mid-evaluation: {rel}')
            cmd=[str(py),str(source/'run_experiment.py'),'--mode','eval','--rl2-root',str(root/'RL2-VLA'),'--python',str(py),'--gpus',*slots(),'--output',str(out/name),'--policy',str(policy),'--training-cache',str(cache/'cache.json'),'--manifest',str(cache/'manifest.json'),'--safe-dir',str(safe),'--qam-original',str(checkpoints['qam_original']),'--qam-base',str(checkpoints['qam_base']),'--qam-shaped',str(checkpoints['qam_shaped']),'--eval-arms',*selected,'--trials',str(trials),'--seeds',*map(str,seeds),'--env-seed-start',str(start),'--run']
            atomic(out/(name+'.command.json'),cmd);status(name)
            with (out/(name+'.log')).open('a') as log:subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
            for arm in selected:
                for path in (out/name/arm).rglob('episode_*.json'):
                    ep=load_episode(path)
                    if ep.get('force_gate_off') is not False or ep.get('environment_contract')!='fresh_env_per_episode_v1':raise ValueError('wrong evaluation contract')
                    if ep['episode_id'] in arms[arm]:raise ValueError('duplicate full evaluation episode')
                    arms[arm][ep['episode_id']]=ep
            status(name+'_complete',counts={a:len(v) for a,v in arms.items()})
        for arm in ARMS:
            if set(arms[arm])!=expected_keys(arm):raise ValueError(f'incomplete {arm}')
        validate_paired_rng(arms)
        atomic(out/'summary.json',summarize(arms));status('complete',episodes=1200)
    except BaseException as exc:
        status('failed',error=repr(exc));raise
    finally:
        for gpu in claimed:subprocess.run(['bash','scripts/utils/gpu_lease.sh','release','kanu',str(gpu),owner],cwd=root)


if __name__=='__main__':main()

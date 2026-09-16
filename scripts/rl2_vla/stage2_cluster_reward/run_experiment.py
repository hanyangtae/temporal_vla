"""Serial IID collection/evaluation launcher. Dry run unless --run is supplied."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from data import TASKS, load_episode, load_manifest, sha256
from safe_provenance import validate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['collect', 'eval'], required=True)
    p.add_argument('--rl2-root', required=True)
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--gpu', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--policy', default='juexzz/INTACT-pi0-finetune-bridge')
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 0, 7])
    p.add_argument('--trials', type=int)
    p.add_argument('--env-seed-start', type=int)
    p.add_argument('--safe-dir')
    p.add_argument('--qam-original')
    p.add_argument('--qam-base')
    p.add_argument('--qam-shaped')
    p.add_argument('--manifest')
    p.add_argument('--run', action='store_true')
    args = p.parse_args()
    rl2, out = Path(args.rl2_root).resolve(), Path(args.output).resolve()
    trials = args.trials if args.trials is not None else (100 if args.mode == 'collect' else 25)
    start = args.env_seed_start if args.env_seed_start is not None else (1000 if args.mode == 'collect' else 10000)
    if trials <= 0 or len(set(args.seeds)) != len(args.seeds):
        p.error('positive trials and unique policy seeds required')
    if args.mode == 'eval':
        if not all([args.safe_dir, args.qam_original, args.qam_base, args.qam_shaped, args.manifest]):
            p.error('eval requires our SAFE, three QAM checkpoints, and training --manifest')
        safe = Path(args.safe_dir).resolve()
        validate(safe, safe/'provenance.json')
        episodes = load_manifest(args.manifest)
        used = {(e['task_id'], e['env_seed']) for e in episodes}
        if any((task, seed) in used for task in TASKS for seed in range(start,start+trials)):
            raise ValueError('evaluation reset overlaps offline data')
        flags = [json.loads(Path(c).with_name('flags.json').read_text()) for c in (args.qam_base,args.qam_shaped)]
        if any(f['manifest_sha256'] != sha256(args.manifest) for f in flags):
            raise ValueError('QAM training manifest mismatch')
        if [f['stage2']['reward_mode'] for f in flags] != ['base','cluster_potential']:
            raise ValueError('base/shaped checkpoints swapped or missing training provenance')
        for field in ('seed','batch_size','warmup_steps','steps'):
            if flags[0]['stage2'][field] != flags[1]['stage2'][field]:
                raise ValueError(f'unpaired QAM training {field}')
        if flags[0]['initial_checkpoint_sha256'] != flags[1]['initial_checkpoint_sha256']:
            raise ValueError('different QAM initializations')
        if flags[0]['initial_checkpoint_sha256'] != sha256(args.qam_original):
            raise ValueError('original QAM differs from C/D initialization')
        arms = [('vanilla',None),('qam_original',args.qam_original),('qam_base',args.qam_base),('qam_shaped',args.qam_shaped)]
    else:
        arms = [('collect',None)]
    for seed in args.seeds:
        for arm, checkpoint in arms:
            for task in TASKS:
                lane = out/arm/f'seed{seed}'/task
                cmd = [args.python, str(rl2/'RL2_CoVer_VLA/simpler/run_simpler_eval_with_openpi.py'),
                    '--task_suite_name',task,'--pretrained_checkpoint',args.policy,
                    '--num_trials_per_task',str(trials),'--env_seed_start',str(start),'--seed',str(seed),
                    '--stage2_single_candidate','True','--stage2_rollout_dir',str(lane),
                    '--local_log_dir',str(lane/'logs'),'--use_verifier','False','--use_verifier_always','False',
                    '--lang_transform_type','no_transform','--lang_rephrase_num_prefail','1',
                    '--lang_rephrase_num','1','--action_samples_prefail','1','--action_samples','1',
                    '--composed_samples_prefail','0','--composed_samples','1' if checkpoint else '0',
                    '--use_failure_prediction','True' if checkpoint else 'False',
                    '--use_rephrased_latents_for_qam','False','--merge_rel_weight','0.5',
                    '--num_steps_wait','0','--n_action_steps','4']
                if checkpoint:
                    cmd += ['--qam_ckpt',str(Path(checkpoint).resolve()),'--failure_checkpoint_dir',str(safe),
                            '--failure_cp_alpha','0.2','--use_taskwise_cp_band','False']
                print(json.dumps({'arm':arm,'task':task,'seed':seed,'command':cmd}), flush=True)
                if not args.run:
                    continue
                request = dict(command=cmd, policy=args.policy, safe_sha256=sha256(safe/'provenance.json') if checkpoint else None,
                               qam_sha256=sha256(checkpoint) if checkpoint else None)
                request_path = lane/'request.json'
                if request_path.exists():
                    if json.loads(request_path.read_text()) != request:
                        raise ValueError(f'output lane belongs to different config: {lane}')
                    existing = [load_episode(p) for p in lane.glob('episode_*.json')]
                    identities = {(e['env_seed'],e['policy_seed']) for e in existing}
                    if len(existing)==trials and identities=={(s,seed) for s in range(start,start+trials)}:
                        print(f'skip verified completed lane: {lane}')
                        continue
                    raise ValueError(f'partial lane preserved; select missing reset range and a new output directory: {lane}')
                lane.mkdir(parents=True, exist_ok=False)
                request_path.write_text(json.dumps(request, indent=2))
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=args.gpu, MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa',WANDB_MODE='offline',
                           PYTHONPATH=f'{rl2}:{rl2}/RL2_CoVer_VLA:'+env.get('PYTHONPATH',''))
                with (lane/'run.log').open('w') as log:
                    subprocess.run(cmd,cwd=rl2/'RL2_CoVer_VLA/simpler',env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                records = [load_episode(p) for p in lane.glob('episode_*.json')]
                if len(records)!=trials or {(e['env_seed'],e['policy_seed']) for e in records}!={(s,seed) for s in range(start,start+trials)}:
                    raise ValueError(f'incomplete/duplicate episode output: {lane}')
                (lane/'DONE.json').write_text(json.dumps({'episodes':len(records),'request_sha256':sha256(request_path)}))


if __name__ == '__main__':
    main()

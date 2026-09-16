"""Serial IID collection/evaluation launcher. Dry run unless --run is supplied."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from data import TASKS, load_episode, load_manifest, sha256
from safe_provenance import validate


def _load_receipt_episode(path):
    receipt = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(receipt, dict):
        raise ValueError(f'{path}: malformed archive receipt')
    payload = receipt.get('episode', receipt.get('episode_payload'))
    if not isinstance(payload, dict):
        raise ValueError(f'{path}: missing original episode payload')
    remote = receipt.get('verified_remote')
    files = receipt.get('files')
    if not isinstance(remote, list) or not isinstance(files, list):
        raise ValueError(f'{path}: receipt lacks verified remote proof')
    episode_file = next((entry for entry in files if isinstance(entry, dict) and entry.get('role') == 'episode'), None)
    if not isinstance(episode_file, dict):
        raise ValueError(f'{path}: receipt lacks episode file proof')
    expected_sha = receipt.get('episode_sha256', episode_file.get('sha256'))
    expected_size = receipt.get('episode_size_bytes', episode_file.get('size_bytes'))
    if not isinstance(expected_sha, str) or not isinstance(expected_size, int):
        raise ValueError(f'{path}: receipt lacks episode identity proof')
    if episode_file.get('sha256') != expected_sha or episode_file.get('size_bytes') != expected_size:
        raise ValueError(f'{path}: episode hash/size proof mismatch')
    remote_match = any(isinstance(entry, dict) and entry.get('path') == episode_file.get('path')
                       and entry.get('sha256') == expected_sha and entry.get('size_bytes') == expected_size
                       for entry in remote)
    if not remote_match:
        raise ValueError(f'{path}: remote episode proof mismatch')
    # Compact receipts intentionally omit chunks. Validate the identity and
    # outcome contract needed by lane completion without reconstructing data.
    required = ('schema_version', 'feature_contract', 'action_contract', 'episode_id',
                'reset_id', 'task_id', 'env_seed', 'policy_seed', 'success', 'end_reason')
    if all(key in payload for key in required) and 'chunks' not in payload:
        if payload['schema_version'] != 1 or payload['feature_contract'] != 'pi0_denoise0_action_mean_1024':
            raise ValueError(f'{path}: unsupported compact rollout contract')
        if payload['action_contract'] != 'pi0_normalize_targets_executed_v1':
            raise ValueError(f'{path}: unsupported compact action contract')
        if not payload['episode_id'] or not payload['reset_id'] or payload['task_id'] not in TASKS:
            raise ValueError(f'{path}: missing compact episode identity')
        if payload['end_reason'] not in ('success', 'horizon', 'terminated', 'truncated') or type(payload['success']) is not bool:
            raise ValueError(f'{path}: invalid compact episode outcome')
        if payload['success'] != (payload['end_reason'] == 'success'):
            raise ValueError(f'{path}: inconsistent compact outcome')
        payload = dict(payload)
        payload['_sha256'] = expected_sha
        payload['_size_bytes'] = expected_size
        return payload
    # Backward-compatible full receipts still go through the canonical loader.
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.json', encoding='utf-8') as stream:
        json.dump(payload, stream)
        stream.flush()
        result = load_episode(stream.name)
    result['_sha256'] = expected_sha
    result['_size_bytes'] = expected_size
    return result


def _completed_episodes(lane):
    """Validate local episodes and receipts, deduping only identical data."""
    for attempt in range(3):
        try:
            local = {p.name: p for p in lane.glob('episode_*.json')}
            receipts = {p.name: p for p in (lane / '.archive_receipts').glob('episode_*.json')}
            records = {name: _load_receipt_episode(path) for name, path in receipts.items()}
            for name, path in local.items():
                try:
                    record = load_episode(path)
                except FileNotFoundError:
                    if name in records:
                        continue
                    raise
                if name in records and (record.get('_sha256') != records[name].get('_sha256') or
                                        (record.get('env_seed'), record.get('policy_seed')) !=
                                        (records[name].get('env_seed'), records[name].get('policy_seed'))):
                    raise ValueError(f'local/receipt episode conflict: {lane / name}')
                records[name] = record
            identities = {}
            for name, record in records.items():
                identity = (record.get('env_seed'), record.get('policy_seed'))
                if identity in identities:
                    raise ValueError(f'duplicate episode identity: {lane} {identity}')
                identities[identity] = record
            return list(records.values())
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            if attempt == 2:
                raise
            time.sleep(0.05)
    return []


def _check_lane(lane, trials, start, policy_seed, request):
    request_path = lane / 'request.json'
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise ValueError(f'output lane belongs to different config: {lane}')
        existing = _completed_episodes(lane)
        identities = {(e.get('env_seed'), e.get('policy_seed')) for e in existing}
        if len(existing) == trials and identities == {(s, policy_seed) for s in range(start, start + trials)}:
            return True
        raise ValueError(f'partial lane preserved; select missing reset range and a new output directory: {lane}')
    lane.mkdir(parents=True, exist_ok=False)
    request_path.write_text(json.dumps(request, indent=2))
    return False


def _run_collection_lane(rl2, lane, cmd, request, gpu, trials, start, policy_seed):
    if _check_lane(lane, trials, start, policy_seed, request):
        print(f'skip verified completed lane: {lane}', flush=True)
        return
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), SAPIEN_RENDER_DEVICE='cuda:0',
               TF_FORCE_GPU_ALLOW_GROWTH='true', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               XLA_PYTHON_CLIENT_PREALLOCATE='false', MUJOCO_GL='osmesa', PYOPENGL_PLATFORM='osmesa', WANDB_MODE='offline',
               PYTHONPATH=f'{rl2}:{rl2}/RL2_CoVer_VLA:' + env.get('PYTHONPATH', ''))
    if '--stage2_save_activations' in cmd:
        env['JAX_PLATFORMS'] = 'cpu'
    with (lane / 'run.log').open('w') as log:
        subprocess.run(cmd, cwd=rl2 / 'RL2_CoVer_VLA/simpler', env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    records = _completed_episodes(lane)
    identities = {(e.get('env_seed'), e.get('policy_seed')) for e in records}
    expected = {(s, policy_seed) for s in range(start, start + trials)}
    if len(records) != trials or identities != expected:
        raise ValueError(f'incomplete/duplicate episode output: {lane}')
    (lane / 'DONE.json').write_text(json.dumps({'episodes': len(records), 'request_sha256': sha256(lane / 'request.json')}))


def _run_gpu_worker(gpu, jobs, rl2, trials, start):
    """Own one GPU for the lifetime of the worker and run its lanes serially."""
    for lane, cmd, request, seed in jobs:
        _run_collection_lane(rl2, lane, cmd, request, gpu, trials, start, seed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['collect', 'eval'], required=True)
    p.add_argument('--rl2-root', required=True)
    p.add_argument('--python', default=sys.executable)
    gpu_group = p.add_mutually_exclusive_group(required=True)
    gpu_group.add_argument('--gpu')
    gpu_group.add_argument('--gpus', nargs='+')
    p.add_argument('--output', required=True)
    p.add_argument('--models-per-gpu', type=int, choices=(1,2), default=1)
    p.add_argument('--tasks', nargs='+', choices=TASKS, default=list(TASKS))
    p.add_argument('--policy', default='juexzz/INTACT-pi0-finetune-bridge')
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 0, 7])
    p.add_argument('--trials', type=int)
    p.add_argument('--env-seed-start', type=int)
    p.add_argument('--safe-dir')
    p.add_argument('--qam-original')
    p.add_argument('--qam-base')
    p.add_argument('--qam-shaped')
    p.add_argument('--manifest')
    p.add_argument('--training-cache')
    p.add_argument('--run', action='store_true')
    args = p.parse_args()
    if len(set(args.tasks)) != len(args.tasks):
        p.error('--tasks must not contain duplicates')
    if args.gpus and len(set(args.gpus)) != len(args.gpus):
        p.error('--gpus must not contain duplicates')
    rl2, out = Path(args.rl2_root).resolve(), Path(args.output).resolve()
    trials = args.trials if args.trials is not None else (100 if args.mode == 'collect' else 25)
    start = args.env_seed_start if args.env_seed_start is not None else (1000 if args.mode == 'collect' else 10000)
    if trials <= 0 or len(set(args.seeds)) != len(args.seeds):
        p.error('positive trials and unique policy seeds required')
    if args.mode == 'eval':
        if args.models_per_gpu != 1:
            p.error('multiple models per GPU supported for collection only')
        if not all([args.safe_dir, args.qam_original, args.qam_base, args.qam_shaped, args.manifest]):
            p.error('eval requires our SAFE, three QAM checkpoints, and training --manifest')
        safe = Path(args.safe_dir).resolve()
        validate(safe, safe/'provenance.json')
        if args.training_cache:
            cache = json.loads(Path(args.training_cache).read_text())
            if cache['manifest_sha256'] != sha256(args.manifest):
                raise ValueError('training cache/manifest mismatch')
            episodes = json.loads(Path(args.manifest).read_text())['episodes']
        else:
            episodes = load_manifest(args.manifest)
        used = {(e['task_id'], e['env_seed']) for e in episodes}
        if any((task, seed) in used for task in args.tasks for seed in range(start,start+trials)):
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
    jobs = []
    for seed in args.seeds:
        for arm, checkpoint in arms:
            for task in args.tasks:
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
                if args.mode == 'collect':
                    cmd += ['--stage2_save_activations', 'True']
                if checkpoint:
                    cmd += ['--qam_ckpt',str(Path(checkpoint).resolve()),'--failure_checkpoint_dir',str(safe),
                            '--failure_cp_alpha','0.2','--use_taskwise_cp_band','True']
                print(json.dumps({'arm':arm,'task':task,'seed':seed,'command':cmd}), flush=True)
                request = dict(command=cmd, policy=args.policy,
                               safe_sha256=sha256(safe/'provenance.json') if checkpoint else None,
                               qam_sha256=sha256(checkpoint) if checkpoint else None)
                jobs.append((lane, cmd, request, seed))
    if not args.run:
        return
    if args.gpus or args.models_per_gpu > 1:
        slots = [gpu for gpu in (args.gpus or [args.gpu]) for _ in range(args.models_per_gpu)]
        with ThreadPoolExecutor(max_workers=len(slots)) as pool:
            grouped = [[] for _ in slots]
            for i, job in enumerate(jobs):
                grouped[i % len(slots)].append(job)
            futures = [pool.submit(_run_gpu_worker, gpu, group, rl2, trials, start)
                       for gpu, group in zip(slots, grouped) if group]
            for future in futures:
                future.result()
    else:
        for lane, cmd, request, seed in jobs:
            _run_collection_lane(rl2, lane, cmd, request, args.gpu, trials, start, seed)


if __name__ == '__main__':
    main()

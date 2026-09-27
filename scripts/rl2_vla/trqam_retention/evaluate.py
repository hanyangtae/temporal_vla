"""Paired fresh-env evaluation: original QAM vs success-only QAM/TRQAM continuations.

Dry run (validate + write contract/plan) unless --run. GPU lease is owned by the
parent coordinator; this launcher runs every lane serially on the given --gpu.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
STAGE2 = HERE.parent / 'stage2_cluster_reward'
sys.path.insert(0, str(STAGE2))
from analyze import validate_paired_rng  # noqa: E402
from data import TASKS, load_episode, sha256  # noqa: E402
from run_experiment import _run_collection_lane  # noqa: E402
from safe_provenance import validate  # noqa: E402

METHODS = ('qam', 'trqam')
RNG_CONTRACT = 'paired_rng_v2'
ENV_CONTRACT = 'fresh_env_per_episode_v1'
PATCH = HERE.parent / 'patches/rl2_vla_stage2_single_candidate.patch'
UPSTREAM = ('README.md', 'RL2_CoVer_VLA/simpler/eval_utils.py', 'RL2_CoVer_VLA/simpler/rl2_utils.py',
            'RL2_CoVer_VLA/simpler/run_simpler_eval_with_openpi.py')


def arm_names(train_seeds):
    return ['original'] + [f'{m}_trainseed{s}' for m in METHODS for s in train_seeds]


def success_train_ids(cache):
    """Expected training set: train split AND success, in manifest order."""
    cache = Path(cache)
    digest = sha256(cache / 'manifest.json')
    meta = json.loads((cache / 'cache.json').read_text())
    if meta['manifest_sha256'] != digest:
        raise ValueError('training cache/manifest mismatch')
    rows = json.loads((cache / 'manifest.json').read_text())['episodes']
    ids = [r['episode_id'] for r in rows if r['split'] == 'train' and r['success'] is True]
    return digest, meta['qam_transitions']['cluster_potential']['sha256'], ids, rows


def load_training_run(run_dir, method, seed):
    flags = json.loads((run_dir / 'flags.json').read_text())
    done = json.loads((run_dir / 'DONE.json').read_text())
    migration = json.loads((run_dir / 'migration.json').read_text())
    r = flags['retention']
    if r['method'] != method or flags['agent'].get('retention_method') != method:
        raise ValueError(f'{run_dir}: retention method is not {method}')
    if r['seed'] != seed or flags['seed'] != seed:
        raise ValueError(f'{run_dir}: training seed is not {seed}')
    if r.get('smoke') is not False or done.get('smoke') is not False:
        raise ValueError(f'{run_dir}: smoke training run')
    if done['steps'] != r['steps'] or done.get('actor_changed') is not True or done.get('reference_frozen') is not True:
        raise ValueError(f'{run_dir}: incomplete training run')
    if migration.get('action_identity') is not True or migration.get('strict_restore') is not True:
        raise ValueError(f'{run_dir}: migration identity not proven')
    ckpt = run_dir / f'params_{r["steps"]}.pkl'
    if Path(done['checkpoint']).name != ckpt.name or sha256(ckpt) != done['sha256']:
        raise ValueError(f'{run_dir}: final checkpoint missing or hash mismatch')
    return flags, ckpt


def validate_training(training_root, original, cache, train_seeds, tasks, start, trials):
    """Return arm -> metadata; fail closed on any provenance/budget mismatch."""
    manifest_sha, transitions_sha, ids, rows = success_train_ids(cache)
    used = set()
    for r in rows:
        if 'task_id' not in r or 'env_seed' not in r:
            raise ValueError('manifest row lacks task_id/env_seed; cannot prove held-out resets')
        used.add((r['task_id'], r['env_seed']))
    if any((t, s) in used for t in tasks for s in range(start, start + trials)):
        raise ValueError('evaluation reset overlaps offline data')
    original_sha = sha256(original)
    arms = {'original': dict(method=None, train_seed=None, checkpoint=str(Path(original).resolve()),
                             checkpoint_sha256=original_sha)}
    reference = None
    for method in METHODS:
        for seed in train_seeds:
            run_dir = Path(training_root) / f'{method}_seed{seed}'
            flags, ckpt = load_training_run(run_dir, method, seed)
            data = flags['data']
            if flags['initial_checkpoint_sha256'] != original_sha:
                raise ValueError(f'{run_dir}: not initialized from the original checkpoint')
            if data['manifest_sha256'] != manifest_sha or data['transitions_sha256'] != transitions_sha:
                raise ValueError(f'{run_dir}: training cache mismatch')
            if data['episode_ids'] != ids or data['episodes'] != len(ids):
                raise ValueError(f'{run_dir}: training set is not exactly train-split successes')
            budget = {k: flags['retention'][k] for k in ('steps', 'batch_size', 'kl_budget')}
            agent = {k: v for k, v in flags['agent'].items() if k != 'retention_method'}
            if reference is None:
                reference = dict(budget=budget, data=data, agent=agent)
            for key, value in (('budget', budget), ('data', data), ('agent', agent)):
                if value != reference[key]:
                    raise ValueError(f'{run_dir}: unmatched {key} across retention arms')
            arms[f'{method}_trainseed{seed}'] = dict(
                method=method, train_seed=seed, checkpoint=str(ckpt.resolve()), checkpoint_sha256=sha256(ckpt),
                flags_sha256=sha256(run_dir / 'flags.json'), done_sha256=sha256(run_dir / 'DONE.json'))
    training = dict(reference['budget'], manifest_sha256=manifest_sha, transitions_sha256=transitions_sha,
                    success_train_episodes=len(ids), selection=reference['data'].get('selection'))
    return arms, training


def lane_command(python, rl2, policy, task, seed, trials, start, lane, checkpoint, safe, force_gate_off):
    """Same eval flags as stage2 run_experiment (checkpoint arms): single candidate,
    SAFE gate with task-wise CP band, 0.5 denoise mix, chunk 4, no verifier/best-of-N."""
    cmd = [python, str(rl2 / 'RL2_CoVer_VLA/simpler/run_simpler_eval_with_openpi.py'),
           '--task_suite_name', task, '--pretrained_checkpoint', policy,
           '--num_trials_per_task', str(trials), '--env_seed_start', str(start), '--seed', str(seed),
           '--stage2_single_candidate', 'True', '--stage2_rollout_dir', str(lane),
           '--local_log_dir', str(lane / 'logs'), '--use_verifier', 'False', '--use_verifier_always', 'False',
           '--lang_transform_type', 'no_transform', '--lang_rephrase_num_prefail', '1',
           '--lang_rephrase_num', '1', '--action_samples_prefail', '1', '--action_samples', '1',
           '--composed_samples_prefail', '0', '--composed_samples', '1',
           '--use_failure_prediction', 'True',
           '--use_rephrased_latents_for_qam', 'False', '--merge_rel_weight', '0.5',
           '--num_steps_wait', '0', '--n_action_steps', '4']
    if force_gate_off:
        cmd += ['--stage2_force_gate_off', 'True']
    return cmd + ['--qam_ckpt', checkpoint, '--failure_checkpoint_dir', str(safe),
                  '--failure_cp_alpha', '0.2', '--use_taskwise_cp_band', 'True']


def code_hashes(rl2):
    files = {str(p.relative_to(REPO)): p for p in
             [Path(__file__).resolve(), *(STAGE2 / f for f in ('run_experiment.py', 'analyze.py', 'data.py',
                                                              'safe_provenance.py')), PATCH]}
    files.update({f'RL2-VLA/{f}': rl2 / f for f in UPSTREAM})
    return {name: sha256(path) for name, path in files.items()}


def first_trigger(ep):
    return next((i for i, c in enumerate(ep['chunks']) if c['safe_trigger']), None)


def validate_pairs(episodes):
    """Each arm vs original: identical inputs/actions up to the first SAFE trigger.

    Same VLA, SAFE and RNG contract means the first trigger must coincide; after it
    the QAM actions differ, so post-trigger histories are never compared. Episodes
    without any trigger must match the original in length and outcome.
    """
    base = episodes['original']
    for arm, eps in episodes.items():
        if arm == 'original':
            continue
        if set(eps) != set(base):
            raise ValueError(f'unpaired/missing episodes in {arm}')
        for key, ep in eps.items():
            if first_trigger(ep) != first_trigger(base[key]):
                raise ValueError(f'first SAFE trigger differs from original: {arm}/{key}')
        validate_paired_rng({'vanilla': base, arm: eps})


def collect(out, arms, tasks, seeds, start, trials, policy, force_gate_off):
    """Load full local episodes for every lane; require exact coverage."""
    episodes = {}
    for arm in arms:
        episodes[arm] = {}
        for seed in seeds:
            for task in tasks:
                lane = out / arm / f'seed{seed}' / task
                if not (lane / 'DONE.json').is_file():
                    raise ValueError(f'incomplete lane: {lane}')
                if (lane / '.archive_receipts').exists():
                    raise ValueError(f'archived lane lacks local chunks for paired validation: {lane}')
                for path in sorted(lane.glob('episode_*.json')):
                    ep = load_episode(path)
                    if (ep['task_id'] != task or ep['policy_seed'] != seed or ep['policy_checkpoint'] != policy
                            or ep.get('force_gate_off') is not force_gate_off
                            or ep.get('rng_contract') != RNG_CONTRACT or ep.get('environment_contract') != ENV_CONTRACT):
                        raise ValueError(f'episode contract mismatch: {path}')
                    if ep['episode_id'] in episodes[arm]:
                        raise ValueError(f'duplicate {arm}/{ep["episode_id"]}')
                    episodes[arm][ep['episode_id']] = ep
        expected = {f'{t}/env{e}/policy{s}' for t in tasks for s in seeds for e in range(start, start + trials)}
        if set(episodes[arm]) != expected:
            raise ValueError(f'coverage mismatch in {arm}: {len(episodes[arm])}/{len(expected)}')
    return episodes


def summarize(episodes, tasks):
    """Descriptive counts only; no significance test or effect claim."""
    base = episodes['original']
    result = {}
    for arm, eps in episodes.items():
        result[arm] = {}
        for task in ('all', *tasks):
            ids = sorted(k for k in eps if task == 'all' or eps[k]['task_id'] == task)
            won = [eps[k]['success'] for k in ids]
            ref = [base[k]['success'] for k in ids]
            result[arm][task] = dict(
                episodes=len(ids), successes=sum(won), success_rate=sum(won) / len(ids),
                triggered_episodes=sum(first_trigger(eps[k]) is not None for k in ids),
                triggered_chunks=sum(c['safe_trigger'] for k in ids for c in eps[k]['chunks']),
                total_chunks=sum(len(eps[k]['chunks']) for k in ids),
                vs_original=dict(both_success=sum(a and b for a, b in zip(won, ref)),
                                 original_fail_arm_success=sum(a and not b for a, b in zip(won, ref)),
                                 original_success_arm_fail=sum(b and not a for a, b in zip(won, ref))))
    return result


def atomic(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rl2-root', required=True)
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--gpu', required=True)
    p.add_argument('--policy', required=True, help='local pi0 directory containing model.safetensors')
    p.add_argument('--safe-dir', required=True)
    p.add_argument('--original', required=True, help='original QAM checkpoint (retention initialization)')
    p.add_argument('--training-root', required=True, help='contains {qam,trqam}_seed{N}/ from train.py')
    p.add_argument('--cache', required=True, help='training cache root with manifest.json and cache.json')
    p.add_argument('--output', required=True)
    p.add_argument('--trials', type=int, default=25)
    p.add_argument('--env-seed-start', type=int, default=11000)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 0, 7], help='policy seeds')
    p.add_argument('--tasks', nargs='+', choices=TASKS, default=list(TASKS))
    p.add_argument('--train-seeds', type=int, nargs='+', default=[42, 0, 7])
    p.add_argument('--force-gate-off', action='store_true', help='migration smoke only; never a performance result')
    p.add_argument('--run', action='store_true')
    args = p.parse_args(argv)
    for name in ('seeds', 'tasks', 'train_seeds'):
        if len(set(getattr(args, name))) != len(getattr(args, name)):
            p.error(f'--{name.replace("_", "-")} must not contain duplicates')
    if args.trials <= 0:
        p.error('positive --trials required')
    rl2, out, safe = Path(args.rl2_root).resolve(), Path(args.output).resolve(), Path(args.safe_dir).resolve()
    start, trials = args.env_seed_start, args.trials
    validate(safe, safe / 'provenance.json')
    arms, training = validate_training(args.training_root, args.original, args.cache, args.train_seeds,
                                       args.tasks, start, trials)
    hashes = code_hashes(rl2)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    safe_sha = sha256(safe / 'provenance.json')
    contract = dict(schema='trqam_retention_eval_v1', repo_commit=commit, code_sha256=hashes,
                    policy=args.policy, policy_sha256=sha256(Path(args.policy) / 'model.safetensors'),
                    safe_dir=str(safe), safe_sha256=safe_sha, arms=arms, training=training,
                    tasks=args.tasks, policy_seeds=args.seeds, env_seeds=[start, start + trials], trials=trials,
                    force_gate_off=args.force_gate_off, rng_contract=RNG_CONTRACT, environment_contract=ENV_CONTRACT,
                    episodes_per_arm=len(args.tasks) * len(args.seeds) * trials,
                    total_episodes=len(arms) * len(args.tasks) * len(args.seeds) * trials)
    jobs = []
    for seed in args.seeds:
        for task in args.tasks:
            for arm, meta in arms.items():
                lane = out / arm / f'seed{seed}' / task
                cmd = lane_command(args.python, rl2, args.policy, task, seed, trials, start, lane,
                                   meta['checkpoint'], safe, args.force_gate_off)
                request = dict(command=cmd, policy=args.policy, rng_contract=RNG_CONTRACT,
                               environment_contract=ENV_CONTRACT, safe_sha256=safe_sha,
                               qam_sha256=meta['checkpoint_sha256'], arm=arm, force_gate_off=args.force_gate_off)
                jobs.append((arm, task, seed, lane, cmd, request))
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'contract.json').exists() and json.loads((out / 'contract.json').read_text()) != contract:
        raise ValueError('output directory belongs to a different evaluation contract')
    atomic(out / 'contract.json', contract)
    atomic(out / 'plan.json', [dict(arm=a, task=t, policy_seed=s, lane=str(l.relative_to(out)), command=c)
                               for a, t, s, l, c, _ in jobs])
    print(json.dumps(dict(arms=list(arms), lanes=len(jobs), total_episodes=contract['total_episodes'])), flush=True)
    if not args.run:
        return contract
    subprocess.run(['git', '-C', str(rl2), 'apply', '--reverse', '--check', '--unidiff-zero', str(PATCH)], check=True)

    def status(stage, **kw):
        atomic(out / 'status.json', dict(stage=stage, time=time.time(), **kw))
    try:
        for i, (arm, task, seed, lane, cmd, request) in enumerate(jobs):
            if code_hashes(rl2) != hashes:
                raise ValueError('source changed mid-evaluation')
            status('running', lane=i, of=len(jobs), arm=arm, task=task, policy_seed=seed)
            _run_collection_lane(rl2, lane, cmd, request, args.gpu, trials, start, seed)
        episodes = collect(out, arms, args.tasks, args.seeds, start, trials, args.policy, args.force_gate_off)
        validate_pairs(episodes)
        atomic(out / 'summary.json', dict(contract_sha256=sha256(out / 'contract.json'),
                                          arms=summarize(episodes, args.tasks)))
        status('complete', episodes=sum(map(len, episodes.values())))
    except BaseException as exc:
        status('failed', error=repr(exc))
        raise
    return contract


if __name__ == '__main__':
    main()

"""Strict paired IID evaluation; no silent truncation to common subsets."""
import argparse
import json
from pathlib import Path

import numpy as np

from data import TASKS, load_episode


def validate_paired_rng(arms):
    """Fail closed before interpreting outcome flips as intervention effects."""
    baseline = arms['vanilla']
    for name, episodes in arms.items():
        if set(episodes) != set(baseline):
            raise ValueError(f'unpaired/missing episodes in {name}')
        for key, ep in episodes.items():
            base = baseline[key]
            if ep.get('rng_contract') != 'paired_rng_v2' or base.get('rng_contract') != 'paired_rng_v2':
                raise ValueError(f'legacy/unverified RNG contract: {name}/{key}')
            for field in ('task_id', 'reset_id', 'env_seed', 'policy_seed', 'episode_rng_seed',
                          'policy_checkpoint', 'action_contract', 'feature_contract'):
                if field not in ep or field not in base or ep[field] != base[field]:
                    raise ValueError(f'paired metadata mismatch {field}: {name}/{key}')
            if ep.get('environment_contract') != base.get('environment_contract'):
                raise ValueError(f'environment contract mismatch: {name}/{key}')
            # Require identical observations' latents/actions until first intervention.
            triggered = False
            for a, b in zip(ep['chunks'], base['chunks']):
                if ep.get('environment_contract') == 'fresh_env_per_episode_v1':
                    for field in ('input_hashes', 'policy_noise_sha256'):
                        if field not in a or field not in b or a[field] != b[field]:
                            raise ValueError(f'pre-intervention {field} mismatch: {name}/{key}/step{a["start_step"]}')
                if a['start_step'] != b['start_step'] or not np.array_equal(a['context'], b['context']):
                    raise ValueError(f'pre-intervention context mismatch: {name}/{key}')
                if a['safe_trigger']:
                    triggered = True
                    break
                for field in ('context', 'proposed_actions', 'executed_actions'):
                    if a['start_step'] != b['start_step'] or not np.array_equal(a[field], b[field]):
                        raise ValueError(f'pre-intervention {field} mismatch: {name}/{key}')
            if not triggered:
                if any(c['safe_trigger'] for c in ep['chunks']):
                    raise ValueError(f'baseline ended before intervention: {name}/{key}')
                if len(ep['chunks']) != len(base['chunks']) or ep['success'] != base['success']:
                    raise ValueError(f'nonintervention outcome/length mismatch: {name}/{key}')


def summarize(arms, bootstraps=2000):
    names = ('vanilla','qam_original','qam_base','qam_shaped')
    baseline = arms['vanilla']
    if not baseline:
        raise ValueError('empty evaluation')
    keys = set(baseline)
    for name in names:
        if set(arms[name]) != keys:
            raise ValueError(f'unpaired/missing episodes in {name}')
    rng = np.random.default_rng(42)
    result = {}
    for task in ('all',)+TASKS:
        ids = sorted(k for k in keys if task=='all' or baseline[k]['task_id']==task)
        if not ids:
            raise ValueError(f'missing task {task}')
        task_result = {}
        for arm in names:
            failed = [k for k in ids if not baseline[k]['success']]
            passed = [k for k in ids if baseline[k]['success']]
            detected = lambda k: any(c['safe_trigger'] for c in arms[arm][k]['chunks'])
            task_result[arm] = dict(episodes=len(ids), successes=sum(arms[arm][k]['success'] for k in ids),
                rescued=sum(arms[arm][k]['success'] for k in failed), baseline_failures=len(failed),
                detected_baseline_failures=sum(detected(k) for k in failed),
                destroyed=sum(not arms[arm][k]['success'] for k in passed), baseline_successes=len(passed),
                detected_baseline_successes=sum(detected(k) for k in passed),
                triggered_chunks=sum(c['safe_trigger'] for k in ids for c in arms[arm][k]['chunks']),
                total_chunks=sum(len(arms[arm][k]['chunks']) for k in ids),
                elapsed_seconds=sum(arms[arm][k].get('elapsed_seconds',0.) for k in ids),
                elapsed_records=sum('elapsed_seconds' in arms[arm][k] for k in ids),
                first_trigger_steps=[next((c['start_step'] for c in arms[arm][k]['chunks'] if c['safe_trigger']),None) for k in ids])
        # Resample reset groups, keeping sibling policy seeds paired together.
        group_keys = sorted({(baseline[k]['task_id'],baseline[k]['reset_id']) for k in ids})
        groups = [[k for k in ids if (baseline[k]['task_id'],baseline[k]['reset_id'])==g] for g in group_keys]
        samples = []
        delta = lambda k: int(arms['qam_shaped'][k]['success'])-int(arms['qam_base'][k]['success'])
        for _ in range(bootstraps):
            resampled = [k for i in rng.integers(len(groups),size=len(groups)) for k in groups[i]]
            samples.append(np.mean([delta(k) for k in resampled]))
        task_result['shaped_minus_base'] = dict(delta_sr=float(np.mean([delta(k) for k in ids])),
            reset_bootstrap_ci95=np.quantile(samples,[.025,.975]).tolist())
        result[task] = task_result
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eval-root',required=True)
    p.add_argument('--output',required=True)
    args = p.parse_args()
    arms = {}
    for name in ('vanilla','qam_original','qam_base','qam_shaped'):
        arms[name] = {}
        for path in (Path(args.eval_root)/name).rglob('episode_*.json'):
            ep = load_episode(path)
            if ep['episode_id'] in arms[name]:
                raise ValueError(f'duplicate {name}/{ep["episode_id"]}')
            arms[name][ep['episode_id']] = ep
    validate_paired_rng(arms)
    Path(args.output).write_text(json.dumps(summarize(arms),indent=2))


if __name__ == '__main__':
    main()

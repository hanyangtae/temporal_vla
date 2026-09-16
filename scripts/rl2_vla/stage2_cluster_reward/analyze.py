"""Strict paired IID evaluation; no silent truncation to common subsets."""
import argparse
import json
from pathlib import Path

import numpy as np

from data import TASKS, load_episode


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
    Path(args.output).write_text(json.dumps(summarize(arms),indent=2))


if __name__ == '__main__':
    main()

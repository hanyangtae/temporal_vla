"""Validate complete episode records, freeze reset-grouped splits, fit potentials."""
import argparse
import json
from pathlib import Path

from data import TASKS, assign_splits, load_episode, load_manifest, sha256
from clusters import fit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rollouts', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    paths = sorted(Path(args.rollouts).rglob('episode_*.json'))
    if not paths:
        raise ValueError('no complete episode JSON; legacy SAFE boundary-only pickle is not sufficient')
    eps = [load_episode(path) for path in paths]
    if len({e['episode_id'] for e in eps}) != len(eps):
        raise ValueError('duplicate episode identity')
    policies = {e.get('policy_checkpoint') for e in eps}
    if None in policies or len(policies) != 1:
        raise ValueError('missing/mixed policy checkpoint identity')
    splits = assign_splits(eps, args.seed)
    for task in TASKS:
        if not any(e['task_id'] == task for e in eps):
            raise ValueError(f'missing IID task: {task}')
        for split in ('train', 'validation', 'holdout'):
            subset = [e for e in eps if e['task_id'] == task and splits[(task,e['reset_id'])] == split]
            if not subset:
                raise ValueError(f'{task}: empty {split}')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    rows = [dict(path=e['_path'], sha256=e['_sha256'], episode_id=e['episode_id'],
                 reset_id=e['reset_id'], task_id=e['task_id'], success=e['success'],
                 split=splits[(e['task_id'],e['reset_id'])]) for e in eps]
    manifest = out/'manifest.json'
    manifest.write_text(json.dumps(dict(schema_version=1, seed=args.seed, episodes=rows), indent=2))
    for split in ('train', 'validation', 'holdout'):
        (out/f'{split}.json').write_text(json.dumps(dict(episodes=[r for r in rows if r['split']==split]), indent=2))
    bundle = fit(load_manifest(manifest, 'train'), sha256(manifest), seed=args.seed)
    bundle.save(out/'clusters.json')
    summary = {task: {split: {
        'episodes': sum(r['task_id']==task and r['split']==split for r in rows),
        'successes': sum(r['task_id']==task and r['split']==split and r['success'] for r in rows),
    } for split in ('train','validation','holdout')} for task in TASKS}
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

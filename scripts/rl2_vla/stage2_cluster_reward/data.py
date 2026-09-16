"""Audited single-candidate transitions; never infer step outcomes from old SAFE logs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

TASKS = (
    'simpler_put_eggplant_in_basket', 'simpler_spoon_on_towel',
    'simpler_stack_cube', 'simpler_carrot_on_plate',
)
FEATURE_CONTRACT = 'pi0_denoise0_action_mean_1024'
ACTION_CONTRACT = 'pi0_normalize_targets_executed_v1'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array(value, shape, name):
    x = np.asarray(value, dtype=np.float32)
    if x.shape != shape or not np.isfinite(x).all():
        raise ValueError(f'{name}: expected finite {shape}, got {x.shape}')
    return x


def load_episode(path):
    ep = json.loads(Path(path).read_text())
    if ep.get('schema_version') != 1 or ep.get('feature_contract') != FEATURE_CONTRACT:
        raise ValueError(f'{path}: unsupported/missing complete rollout contract')
    if ep.get('action_contract') != ACTION_CONTRACT:
        raise ValueError(f'{path}: unverified executed-action normalization')
    if ep.get('task_id') not in TASKS or not ep.get('episode_id') or not ep.get('reset_id'):
        raise ValueError(f'{path}: missing identity or non-IID task')
    reason = ep.get('end_reason')
    if reason not in ('success', 'horizon', 'terminated', 'truncated'):
        raise ValueError(f'{path}: interrupted/unknown termination')
    if type(ep.get('success')) is not bool or ep['success'] != (reason == 'success'):
        raise ValueError(f'{path}: inconsistent outcome')
    chunks = ep['chunks']
    if not chunks:
        raise ValueError(f'{path}: no executed chunks')
    cursor = 0
    for i, c in enumerate(chunks):
        array(c['context'], (1024,), 'context')
        m = len(c['executed_actions'])
        if not 1 <= m <= 4 or c['start_step'] != cursor:
            raise ValueError(f'{path}: chunk length/gap at {i}')
        if i < len(chunks) - 1 and m != 4:
            raise ValueError(f'{path}: incomplete nonterminal chunk')
        executed = array(c['executed_actions'], (m, 7), 'executed_actions')
        expected = array(c['postprocessed_actions'], (4, 7), 'postprocessed_actions')
        if not np.allclose(executed, expected[:m], atol=1e-6, rtol=1e-5):
            raise ValueError(f'{path}: proposed/actually executed action mismatch')
        array(c['normalized_actions'], (4, 7), 'normalized_actions')
        for field in ('step_success', 'step_reward', 'step_truncated'):
            if len(c[field]) != m:
                raise ValueError(f'{path}: missing per-step {field}')
        array(c['step_reward'], (m,), 'step_reward')
        if any(type(v) is not bool for k in ('step_success', 'step_truncated') for v in c[k]):
            raise ValueError(f'{path}: nonboolean step labels')
        cursor += m
    outcomes = [v for c in chunks for v in c['step_success']]
    truncs = [v for c in chunks for v in c['step_truncated']]
    if any(outcomes[:-1]) or outcomes[-1] != ep['success'] or any(truncs[:-1]):
        raise ValueError(f'{path}: post-success/post-truncation steps or inconsistent label')
    ep['_path'] = str(Path(path).resolve())
    ep['_sha256'] = sha256(path)
    return ep


def assign_splits(episodes, seed=42):
    """Exact 60/20/20 by task/reset, keeping all policy seeds together."""
    result = {}
    for task in TASKS:
        ids = sorted({e['reset_id'] for e in episodes if e['task_id'] == task})
        ids.sort(key=lambda x: hashlib.sha256(f'{seed}:{x}'.encode()).hexdigest())
        n = len(ids)
        for i, reset in enumerate(ids):
            result[(task, reset)] = 'train' if i < int(.6*n) else 'validation' if i < int(.8*n) else 'holdout'
    return result


def load_manifest(path, split=None):
    manifest = json.loads(Path(path).read_text())
    episodes, seen, reset_splits = [], set(), {}
    for row in manifest['episodes']:
        p = Path(row['path'])
        if not p.is_absolute():
            p = Path(path).parent / p
        if sha256(p) != row['sha256']:
            raise ValueError(f'rollout hash mismatch: {p}')
        e = load_episode(p)
        for field in ('episode_id', 'reset_id', 'task_id', 'success'):
            if field in row and row[field] != e[field]:
                raise ValueError(f'manifest metadata mismatch: {field} in {p}')
        if e['episode_id'] in seen:
            raise ValueError(f'duplicate episode: {e["episode_id"]}')
        seen.add(e['episode_id'])
        key = (e['task_id'], e['reset_id'])
        if row['split'] not in ('train', 'validation', 'holdout'):
            raise ValueError('unknown split')
        if key in reset_splits and reset_splits[key] != row['split']:
            raise ValueError(f'reset leakage: {key}')
        reset_splits[key] = row['split']
        if split is None or split == row['split']:
            episodes.append(e)
    return episodes


def base_rewards(length, success):
    """RL2 Bridge convention: -1 until final three successful steps, then 0."""
    if length <= 0:
        raise ValueError('empty episode')
    rewards = -np.ones(length, dtype=np.float32)
    masks = np.ones(length, dtype=np.float32)
    if success:
        rewards[-min(3, length):] = 0
        masks[-min(3, length):] = 0
    else:
        masks[-1] = 0  # finite benchmark horizon, not a collection interruption
    return rewards, masks


def transitions(episodes, potential=None, scale=0., gamma=.99):
    """Build chunk transitions, retaining a separate action-prefix loss mask.

    Partial final chunks have no bootstrap. Unexecuted actions are zero padded
    and excluded from both BC loss and critic action input gradients.
    """
    rows = []
    for ep in episodes:
        chunks = ep['chunks']
        length = sum(len(c['executed_actions']) for c in chunks)
        rewards, masks = base_rewards(length, ep['success'])
        for i, c in enumerate(chunks):
            t, m = c['start_step'], len(c['executed_actions'])
            obs = np.asarray(c['context'], dtype=np.float32)
            nxt = np.asarray(chunks[i+1]['context'], dtype=np.float32) if i+1 < len(chunks) else np.zeros_like(obs)
            b = float(np.prod(masks[t:t+m]))
            r = float(np.dot(gamma ** np.arange(m), rewards[t:t+m]))
            if scale:
                if potential is None:
                    raise ValueError('cluster shaping requires a frozen train-only bundle')
                r += scale * (gamma**m * b * potential(ep['task_id'], nxt) - potential(ep['task_id'], obs))
            actions = np.zeros((4, 7), np.float32)
            actions[:m] = np.asarray(c['normalized_actions'], np.float32)[:m]
            action_mask = np.zeros((4, 7), np.float32)
            action_mask[:m] = 1
            rows.append(dict(observations=obs, next_observations=nxt[None], actions=actions,
                             action_mask=action_mask, rewards=np.array([r], np.float32),
                             masks=np.array([b], np.float32), valid=np.ones(1, np.float32),
                             discounts=np.float32(gamma**m), actual_steps=np.int32(m)))
    if not rows:
        raise ValueError('no transitions in split')
    return {k: np.stack([r[k] for r in rows]) for k in rows[0]}

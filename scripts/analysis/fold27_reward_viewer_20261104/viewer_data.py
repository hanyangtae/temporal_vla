"""Data layer of the Fold27 reward viewer (stdlib + numpy only; Python >= 3.8, numpy >= 1.21).

Input = ONLY the SHA-verified reconstruct.py outputs (scripts/analysis/fold27_reward_arrays_20261104):
  <derived root>/sha_matched/<fold>/{reward_arrays.npz, receipt.json}
A fold is served only if its receipt says status sha_matched, every gate is true and the npz sha256 equals the receipt.
npz are opened with allow_pickle=False. No rollout pkl, no recomputation of rewards, no cluster inference:
cluster IDs were not stored by reconstruct.py, so they are reported as not stored (never derived from phi).
Data provenance: RoboCasa GR00T Fold27 production 18fold, each fold's 160 FIT rollouts (training data, not eval).
"""
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

FOLD_RE = re.compile(r'^[A-Za-z0-9-]+_s\d+_j\d+$')
ROW_KEYS = ('basic', 'mix', 'alpha_term', 'beta_cost', 'delta', 'phi', 'anchor', 'C', 'episode_index', 'step',
            'env_step_start', 'm', 'terminal', 'final', 'success')
EP_KEYS = ('episode_id', 'ep_j', 'ep_n', 'ep_success', 'ep_steps', 'ep_max_steps', 'ep_n_decisions', 'ep_row_start')
REQUIRED = ('basic', 'mix', 'alpha_term', 'beta_cost', 'delta', 'episode_index', 'step', 'env_step_start', 'm',
            'terminal', 'final', 'success') + EP_KEYS
PROVENANCE = ('RoboCasa GR00T Fold27 production 18fold: 각 fold의 160개 fit rollout = 훈련(fit) 데이터, 평가 아님. '
              '보상 배열은 학습 DONE fingerprint의 rewards_sha256/observations_sha256과 일치하도록 재구성된 값.')


class DataError(ValueError):
    pass


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def split_episode_id(eid):
    """'<plan>/<machine>/<instruction with '/'>/s<scene>/j<j>/n<n>' -> (instruction, scene, j, n)."""
    p = eid.split('/')
    if len(p) < 6 or not (p[-3][:1] == 's' and p[-2][:1] == 'j' and p[-1][:1] == 'n'):
        raise DataError('bad episode id ' + eid)
    return '/'.join(p[2:-3]), int(p[-3][1:]), int(p[-2][1:]), int(p[-1][1:])


def load_fold(d):
    """One sha_matched/<fold> directory -> dict, or raise DataError."""
    d = Path(d)
    fold = d.name
    if not FOLD_RE.match(fold):
        raise DataError('bad fold dir name ' + fold)
    r = json.loads((d / 'receipt.json').read_text())
    gates = (r.get('sha') or {}).get('gates') or {}
    if r.get('status') != 'sha_matched' or not gates or not all(gates.values()) or r.get('fold') != fold:
        raise DataError(fold + ': receipt not sha_matched with all gates true')
    npz = d / 'reward_arrays.npz'
    want = ((r.get('outputs') or {}).get('reward_arrays.npz') or {}).get('sha256')
    if not want or sha256_file(npz) != want:
        raise DataError(fold + ': reward_arrays.npz sha256 != receipt')
    with np.load(npz, allow_pickle=False) as z:
        missing = [k for k in REQUIRED if k not in z.files]
        if missing:
            raise DataError(fold + ': npz lacks ' + ','.join(missing))
        a = {k: z[k] for k in z.files if k in ROW_KEYS + EP_KEYS}
    n_rows, n_ep = len(a['basic']), len(a['episode_id'])
    if any(len(a[k]) != n_rows for k in ROW_KEYS if k in a) or any(len(a[k]) != n_ep for k in EP_KEYS):
        raise DataError(fold + ': array length mismatch')
    starts = np.concatenate([a['ep_row_start'], [n_rows]])
    if not np.array_equal(np.diff(starts), a['ep_n_decisions']):
        raise DataError(fold + ': episode row index mismatch')
    instr = {split_episode_id(str(e))[0] for e in a['episode_id']}
    if len(instr) != 1:
        raise DataError(fold + ': more than one instruction')
    return {'fold': fold, 'instruction': instr.pop(), 'arrays': a, 'receipt': r, 'starts': starts,
            'has_phi': 'phi' in a}


def load_all(roots):
    """All sha_matched folds of the given derived roots (+ listed failed attempts, never served as data)."""
    folds, failures = {}, {}
    for root in roots:
        root = Path(root)
        for d in sorted((root / 'sha_matched').glob('*')):
            if d.is_dir():
                f = load_fold(d)
                if f['fold'] in folds:
                    raise DataError('duplicate fold across roots: ' + f['fold'])
                folds[f['fold']] = f
        for d in sorted((root / 'contract_failure').glob('*')):
            if d.is_dir() and (d / 'receipt.json').is_file():
                failures.setdefault(d.name, []).append(json.loads((d / 'receipt.json').read_text()).get('error'))
    if not folds:
        raise DataError('no sha_matched fold found')
    return folds, failures


def duplicates(folds):
    """episode_id -> [(fold, episode index)] across all folds (each rollout is in the fit set of several folds)."""
    m = {}
    for name in sorted(folds):
        for i, e in enumerate(folds[name]['arrays']['episode_id']):
            m.setdefault(str(e), []).append((name, i))
    return m


def _f(x):
    return [float(v) for v in np.asarray(x, np.float64)]


def sha_panel(f):
    s = f['receipt']['sha']
    o, r = s['done_fingerprint'], s['reconstructed']
    return {'원본기록_DONE_fingerprint': {'rows': o.get('rows'), 'basic_rewards_sha256': o.get('basic_rewards_sha256'),
                                       'mix_rewards_sha256': o.get('mix_rewards_sha256'),
                                       'basic_observations_sha256': o.get('basic_observations_sha256'),
                                       'mix_observations_sha256': o.get('mix_observations_sha256')},
            '재구성값': {'rows': r.get('rows'), 'basic_rewards_sha256': r.get('basic_rewards_sha256'),
                      'mix_rewards_sha256': r.get('mix_rewards_sha256'),
                      'observations_sha256': r.get('observations_sha256'),
                      'mix_observations_sha256': r.get('mix_observations_sha256')},
            'gates': s.get('gates'), 'episode_scalars': f['receipt'].get('episode_scalars', 'npz'),
            'host': f['receipt'].get('host'), 'coefficients': coefficients(f),
            'npz_sha256': f['receipt']['outputs']['reward_arrays.npz']['sha256']}


def meta(folds, failures, videos):
    tasks = {}
    for name in sorted(folds):
        f = folds[name]
        a = f['arrays']
        instr = f['instruction']
        tasks.setdefault(instr.split('/')[0], {}).setdefault(instr, []).append(
            {'fold': name, 'target_j': int(name.rsplit('_j', 1)[1]), 'episodes': int(len(a['episode_id'])),
             'success': int(a['ep_success'].sum()), 'rows': int(len(a['basic']))})
    return {'provenance': PROVENANCE, 'tasks': tasks, 'folds': {k: sha_panel(v) for k, v in folds.items()},
            'failed_attempts_not_served': failures, 'videos_validated': len(videos),
            'cluster_id': '미저장 (reconstruct.py가 저장하지 않음; Φ에서 역추론하지 않음)',
            'x_axis': '기본 inference step(저장 0-based decision 번호), 토글로 env step(decision마다 5 env step, 마지막 chunk m<5 가능); 시간(초) 추정 없음'}


def fold_episodes(f):
    a = f['arrays']
    out = []
    for i, e in enumerate(a['episode_id']):
        out.append({'ep': i, 'episode_id': str(e), 'j': int(a['ep_j'][i]), 'n': int(a['ep_n'][i]),
                    'success': bool(a['ep_success'][i]), 'env_steps': int(a['ep_steps'][i]),
                    'decisions': int(a['ep_n_decisions'][i])})
    return out


def episode_payload(f, ep, dup, videos):
    a, s = f['arrays'], f['starts']
    lo, hi = int(s[ep]), int(s[ep + 1])
    sl = slice(lo, hi)
    eid = str(a['episode_id'][ep])
    instr, scene, j, n = split_episode_id(eid)
    final = a['final'][sl].astype(bool)
    succ = bool(a['ep_success'][ep])
    out = {'fold': f['fold'], 'ep': ep, 'episode_id': eid, 'instruction': instr, 'scene': scene, 'j': j, 'n': n,
           'success': succ, 'env_steps': int(a['ep_steps'][ep]), 'max_episode_steps': int(a['ep_max_steps'][ep]),
           'decisions': hi - lo, 'last_chunk_m': int(a['m'][hi - 1]),
           'terminal_anchor': 1.0 if succ else -1.0,
           'chunks': {'step': [int(v) for v in a['step'][sl]], 'env_step_start': [int(v) for v in a['env_step_start'][sl]],
                      'm': [int(v) for v in a['m'][sl]], 'terminal': [bool(v) for v in a['terminal'][sl]],
                      'final': [bool(v) for v in final],
                      'basic': _f(a['basic'][sl]), 'mix': _f(a['mix'][sl]), 'alpha_term': _f(a['alpha_term'][sl]),
                      'minus_beta_cost': _f(-np.asarray(a['beta_cost'][sl], np.float64)), 'delta': _f(a['delta'][sl])},
           'phi': _f(a['phi'][sl]) if f['has_phi'] else None,
           'anchor': _f(a['anchor'][sl]) if 'anchor' in a else None,
           'cluster_id': None, 'cluster_id_note': '미저장',
           'duplicates': [{'fold': fo, 'ep': i} for fo, i in dup.get(eid, [])],
           'video': eid in videos}
    return out


def load_video_index(index_path, root):
    """Optional {episode_id: {"path": <relative to root>, "sha256": hex}} -> {episode_id: absolute real path}.
    A video is accepted only if: relative path, resolves (symlinks followed) inside root, regular .mp4 file, sha256
    matches. Everything else is rejected and reported as missing. Rollout pkl are never opened."""
    if index_path is None:
        return {}, {}
    root = Path(root).resolve()
    idx = json.loads(Path(index_path).read_text())
    ok, rejected = {}, {}
    for eid, rec in sorted(idx.items()):
        try:
            split_episode_id(eid)
            rel = str(rec['path'])
            if rel.startswith('/') or '..' in Path(rel).parts or not rel.endswith('.mp4'):
                raise DataError('path not a relative .mp4 without ..')
            p = (root / rel).resolve()
            if os.path.commonpath([str(p), str(root)]) != str(root) or not p.is_file():
                raise DataError('outside root or not a regular file')
            if sha256_file(p) != rec['sha256']:
                raise DataError('sha256 mismatch')
            ok[eid] = str(p)
        except (DataError, KeyError, TypeError, OSError) as e:
            rejected[eid] = str(e)
    return ok, rejected


MEAN_KEYS = ('basic', 'mix', 'alpha_term', 'minus_beta_cost', 'delta')


def coefficients(f):
    """alpha/beta exactly as recorded in the reconstruct receipt (reconstruct.py refused to run unless the mix DONE
    fingerprint had the same alpha/beta). Missing -> None (displayed as unconfirmed, never guessed)."""
    d = f['receipt'].get('decomposition') or {}
    a, b = d.get('alpha'), d.get('beta')
    return {'alpha': a, 'beta': b, 'gamma': d.get('gamma'),
            'source': 'reconstruct receipt.decomposition (mix DONE fingerprint alpha/beta와 일치 확인 후 실행)'
            if a is not None and b is not None else None}


def outcome_means(f):
    """Per outcome group (success / failure) and inference index t (stored 0-based step, original time alignment):
    mean over the rollouts of that group that ACTUALLY HAVE decision t (no zero padding, no last-value extension).
    n_t = number of such rollouts. 'cum' = each rollout's own cumulative sum up to t, then the same existing-sample
    mean. Late t are averaged over fewer, longer rollouts (length/outcome confound) -> always shown with n_t."""
    a, s = f['arrays'], f['starts']
    series = {'basic': np.asarray(a['basic'], np.float64), 'mix': np.asarray(a['mix'], np.float64),
              'alpha_term': np.asarray(a['alpha_term'], np.float64),
              'minus_beta_cost': -np.asarray(a['beta_cost'], np.float64), 'delta': np.asarray(a['delta'], np.float64)}
    out = {}
    for name, flag in (('success', True), ('failure', False)):
        eps = [i for i in range(len(a['episode_id'])) if bool(a['ep_success'][i]) == flag]
        if not eps:
            out[name] = {'total_n': 0, 't': [], 'n_t': [], 'mean': None, 'cum_mean': None}
            continue
        L = max(int(s[i + 1] - s[i]) for i in eps)
        n_t = np.zeros(L, np.int64)
        sums = {k: np.zeros(L) for k in MEAN_KEYS}
        csums = {k: np.zeros(L) for k in MEAN_KEYS}
        for i in eps:
            lo, hi = int(s[i]), int(s[i + 1])
            if not np.array_equal(a['step'][lo:hi], np.arange(hi - lo)):
                raise DataError(f['fold'] + ': stored step is not 0..n-1 within an episode')
            n_t[:hi - lo] += 1
            for k in MEAN_KEYS:
                v = series[k][lo:hi]
                sums[k][:hi - lo] += v
                csums[k][:hi - lo] += np.cumsum(v)
        out[name] = {'total_n': len(eps), 't': list(range(L)), 'n_t': [int(x) for x in n_t],
                     'mean': {k: _f(sums[k] / n_t) for k in MEAN_KEYS},
                     'cum_mean': {k: _f(csums[k] / n_t) for k in MEAN_KEYS},
                     'length': {'min': int(min(s[i + 1] - s[i] for i in eps)), 'max': L}}
    return out

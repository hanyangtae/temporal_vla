"""Fold rollout -> prefix-QAM chunk transitions with basic / PBRS / MIX rewards (numpy only).

Ports of the validated SIMPLER recipes (sources in .worktrees/trqam-retention, cited per function), adapted to
GR00T decisions (16 predicted / m=5 executed, gamma .99, prefix-critic row semantics of
groot_robocasa_adapter_20261021/transitions.py + groot_target13 multitask_data.build_multitask_rl2_split).

Episode inputs (fold fit rollouts only; eval-j rollouts are rejected by the caller via folds.validate_fold):
  latents   [n,1024]  decoder-input last16 mean at denoise 0 (rollout_latents.py, contract amendment)
  actions   [n,16,12] GR00T predicted chunk, model order, NORMALIZED with the checkpoint codec
  success   bool
  steps     env steps actually executed = len(env_step_success) (extract_scene.episode_steps), never estimated.
            Success episodes run to the end of the chunk where success first fired (up to 4 steps after it); the
            base reward keeps the existing "last 3 EXECUTED steps" convention (data.py) — it is NOT re-aligned to
            the first success trigger. Failure -> steps == max_episode_steps (from the pkl).
"""
import numpy as np

GAMMA, EXECUTE, HORIZON, MAX_STEPS = 0.99, 5, 16, 720
ARMS = ('basic', 'pbrs', 'mix')
TERMINAL_ANCHOR = {True: 1.0, False: -1.0}     # build_riskmix_cache.TERMINAL_ANCHOR


class RewardContractError(ValueError):
    pass


def base_rewards(length, success):
    """stage2_cluster_reward/data.py:117-128 (RL2 Bridge convention), unchanged."""
    if length <= 0:
        raise RewardContractError('empty episode')
    r = -np.ones(length, np.float32); mask = np.ones(length, np.float32)
    if success:
        r[-min(3, length):] = 0; mask[-min(3, length):] = 0
    else:
        mask[-1] = 0     # finite benchmark horizon, not a collection interruption
    return r, mask


def check_episode(n, steps, success, max_steps):
    """steps = len(env_step_success) from the pkl (extract_scene.episode_steps), never estimated."""
    if steps is None or max_steps is None:
        raise RewardContractError('steps/max_episode_steps unknown: not estimated')
    if not (EXECUTE * (n - 1) < steps <= EXECUTE * n):
        raise RewardContractError(f'steps {steps} inconsistent with {n} decisions')
    if not success and steps != max_steps:
        raise RewardContractError(f'failure must be a timeout: steps {steps} != max_episode_steps {max_steps}')


def chunk_terms(steps, success, n, max_steps=MAX_STEPS):
    """Per decision i: executed m_i (cut at the first masked step, as multitask_data), base reward R_i, bootstrap b_i,
    terminal flag. Mirrors data.py transitions(): b = prod(mask over executed steps)."""
    check_episode(n, steps, success, max_steps)
    r, mask = base_rewards(steps, success)
    m, R, b = np.zeros(n, np.int32), np.zeros(n, np.float64), np.zeros(n, np.float64)
    for i in range(n):
        t = EXECUTE * i
        observed = min(EXECUTE, steps - t)
        zeros = np.flatnonzero(mask[t:t + observed] == 0)      # groot_target13 multitask_data.py:58-61
        mi = int(zeros[0]) + 1 if len(zeros) else observed
        m[i] = mi
        R[i] = float(np.dot(GAMMA ** np.arange(mi), r[t:t + mi]))
        b[i] = 0.0 if len(zeros) else 1.0
    final = np.zeros(n, bool); final[-1] = True
    return m, R, b, final


def shaped(arm, R, m, b, final, success, phi, phi_next_actual, *, alpha=None, beta=None):
    """basic: R.  pbrs (data.py:151): R + alpha*(g^m*b*phi(next) - phi); on final rows b=0 so the next term
    vanishes (same as SIMPLER).  mix (build_riskmix_cache.py:16,113,135):
    R + alpha*(phi_next - phi) - beta*C, phi_next = actual next (incl. early-cut b=0 rows), final rows = z_end anchor,
    C = max(0,-phi)*sum_{i<m} g^i."""
    if arm == 'basic':
        if alpha is not None or beta is not None:
            raise RewardContractError('basic takes no alpha/beta')
        return R.copy()
    if alpha is None:
        raise RewardContractError(f'{arm} requires explicit alpha')
    if arm == 'pbrs':
        if beta is not None:
            raise RewardContractError('pbrs takes no beta')
        if np.any(b[final] != 0):
            raise RewardContractError('final rows must have b=0')
        return R + alpha * (GAMMA ** m * b * phi_next_actual - phi)
    if arm == 'mix':
        if beta is None:
            raise RewardContractError('mix requires explicit beta (no default; SIMPLER winner choice is selection-on-eval)')
        anchor = np.where(final, TERMINAL_ANCHOR[bool(success)], phi_next_actual)
        C = np.maximum(0.0, -phi) * (1 - GAMMA ** m) / (1 - GAMMA)
        return R + alpha * (anchor - phi) - beta * C
    raise RewardContractError(f'unknown arm {arm}')


def episode_transitions(ep, arm, potential=None, *, alpha=None, beta=None):
    """ep: dict(episode_id, latents, actions, success, steps). Returns prefix-QAM rows (transitions.py §7 fields)."""
    lat, act = np.asarray(ep['latents'], np.float32), np.asarray(ep['actions'], np.float32)
    n = len(lat)
    if lat.shape != (n, 1024) or act.shape != (n, HORIZON, 12):
        raise RewardContractError(f'{ep["episode_id"]}: shapes {lat.shape} {act.shape}')
    if not np.isfinite(lat).all() or not np.isfinite(act).all():
        raise RewardContractError('non-finite input')
    m, R, b, final = chunk_terms(int(ep['steps']) if ep.get('steps') is not None else None, bool(ep['success']), n,
                                 int(ep['max_episode_steps']) if ep.get('max_episode_steps') is not None else None)
    if arm == 'basic':
        phi = phi_next = np.zeros(n)
    else:
        if potential is None:
            raise RewardContractError(f'{arm} requires a fold-fit potential')
        phi = potential(lat)
        phi_next = np.append(phi[1:], 0.0)     # actual next decision latent; final row overwritten by b=0 / anchor
    rew = shaped(arm, R, m, b, final, ep['success'], phi, phi_next, alpha=alpha, beta=beta)
    terminal = b == 0          # cut at first masked step => no bootstrap (multitask_data: discount 0 when terminal)
    nxt = np.zeros_like(lat); nxt[:-1] = lat[1:]
    nxt[terminal] = 0
    exec_mask = np.arange(HORIZON)[None] < m[:, None]
    return {'observations': lat, 'actions': act, 'rewards': rew.astype(np.float32),
            'discounts': np.where(terminal, 0.0, GAMMA ** m).astype(np.float32), 'terminals': terminal,
            'has_next': ~terminal, 'next_observations': nxt, 'executed_m': m, 'executed_mask': exec_mask,
            'next_executed_m': np.where(terminal, 0, EXECUTE).astype(np.int32),
            'chunk_valid': np.ones((n, HORIZON), bool), 'episode_id': np.asarray([ep['episode_id']] * n),
            'phi': phi.astype(np.float32), 'base_reward': R.astype(np.float32)}


def concat(rows):
    keys = rows[0].keys()
    return {k: np.concatenate([r[k] for r in rows]) for k in keys}


def recompute_check(trans, arm, alpha=None, beta=None, tol=1e-5):
    """Independent re-derivation of the reward column from stored (base, phi, m, discounts) — cache validator."""
    R, phi, m = trans['base_reward'].astype(np.float64), trans['phi'].astype(np.float64), trans['executed_m']
    if arm == 'basic':
        want = R
    else:
        ids = trans['episode_id']
        last = np.append(ids[1:] != ids[:-1], True)
        phin = np.append(phi[1:], 0.0); phin[last] = 0.0
        b = (trans['discounts'] > 0).astype(np.float64)
        if arm == 'pbrs':
            want = R + alpha * (GAMMA ** m * b * phin - phi)
        else:
            raise RewardContractError('mix recompute needs per-row success; use episode_transitions twice instead')
    resid = np.abs(want - trans['rewards']).max()
    if resid > tol:
        raise RewardContractError(f'reward recompute residual {resid}')
    return float(resid)


# ---- potential (stage2_cluster_reward/clusters.py:40-76 port; one task = the fold's instruction) ----

def fit_potential(episodes, *, k=48, dim=16, seed=42, prior=5.0):
    """episodes: [(latents [n,1024], success bool)] from the fold's 160 fit rollouts ONLY.
    Episode-weighted standardization -> weighted PCA dim -> KMeans(k, n_init=10, random_state=seed) with sample
    weights -> score (s-f)/(s+f)*support/(support+prior). One outcome class -> potential fixed at 0 (clusters.py:44)."""
    from sklearn.cluster import KMeans
    outcomes = np.asarray([bool(s) for _, s in episodes])
    if len(set(outcomes.tolist())) < 2:
        return {'disabled': 'one outcome class; potential fixed at zero'}
    xs = [np.asarray(x, np.float64) for x, _ in episodes]
    x = np.concatenate(xs)
    if len(x) < max(k, dim):
        raise RewardContractError(f'fewer than {max(k, dim)} contexts; do not silently change K')
    w = np.concatenate([np.full(len(a), 1 / len(a)) for a in xs]); w /= w.sum()
    mean = np.sum(x * w[:, None], 0)
    std = np.sqrt(np.sum((x - mean) ** 2 * w[:, None], 0)).clip(1e-6)
    z = (x - mean) / std
    _, _, vt = np.linalg.svd(z * np.sqrt(w[:, None]), full_matrices=False)
    comps = vt[:dim]
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(z @ comps.T, sample_weight=w)
    counts, support, cur = [], np.zeros(k), 0
    for a in xs:
        lab = km.labels_[cur:cur + len(a)]
        nn = np.bincount(lab, minlength=k)
        counts.append(nn / len(a)); support += nn > 0; cur += len(a)
    counts = np.asarray(counts)
    s, f = counts[outcomes].mean(0), counts[~outcomes].mean(0)
    scores = (s - f) / (s + f + 1e-12) * support / (support + prior)
    return {'mean': mean, 'std': std, 'components': comps, 'centers': km.cluster_centers_, 'scores': scores,
            'support': support, 'k': k, 'dim': dim, 'seed': seed, 'prior': prior}


def potential_fn(bundle):
    if bundle.get('disabled'):
        return lambda lat: np.zeros(len(lat))
    def phi(lat):
        z = ((np.asarray(lat, np.float64) - bundle['mean']) / bundle['std']) @ bundle['components'].T
        k = np.argmin(((bundle['centers'][None] - z[:, None]) ** 2).sum(-1), axis=1)
        return bundle['scores'][k]
    return phi

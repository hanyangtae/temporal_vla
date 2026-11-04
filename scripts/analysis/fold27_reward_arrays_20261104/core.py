"""Pure-numpy core: exact per-inference basic / MIX reward decomposition + distribution statistics.

The reward values themselves are never re-derived here: `rows_*['rewards']` come from the pinned
scripts/rl2_vla/groot_fold27_20261028/trqam/rewards.py (episode_transitions) called exactly as
finetune_fold.build_rows does. This module only re-expresses the SAME float64 expression of
rewards.shaped('mix') term by term so that each row can be attributed:

    mix = R + alpha*(anchor - phi) - beta*C        (rewards.py shaped, evaluation order kept)
    anchor = +1/-1 on the final row (success/failure), else phi(actual next decision latent)
    C = max(0, -phi) * (1 - g**m)/(1 - g)  == max(0,-phi) * sum_{i<m} .99**i
    no gamma**m and no bootstrap b on the potential difference.

`decompose` asserts float32(R + alpha_term - beta_cost) is bitwise equal to the trained mix rewards.
"""
import hashlib

import numpy as np

GAMMA = 0.99
PCTS = (0, 50, 90, 95, 99, 100)


class ContractError(ValueError):
    pass


def sha_bytes(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def parse_jn(episode_id):
    """'<plan>/<machine>/<instruction (may contain '/')>/s<scene>/j<j>/n<n>' -> (j, n) from the tail."""
    parts = episode_id.split('/')
    if not (parts[-2].startswith('j') and parts[-1].startswith('n') and parts[-3].startswith('s')):
        raise ContractError(f'bad episode id {episode_id}')
    return int(parts[-2][1:]), int(parts[-1][1:])


def episode_terms(RW, ep, phi):
    """Per decision: m, R, b, final (rewards.chunk_terms, same inputs as episode_transitions) + anchor, C."""
    n = len(ep['latents'])
    m, R, b, final = RW.chunk_terms(int(ep['steps']), bool(ep['success']), n, int(ep['max_episode_steps']))
    phi = np.asarray(phi, np.float64)
    phi_next = np.append(phi[1:], 0.0)
    anchor = np.where(final, RW.TERMINAL_ANCHOR[bool(ep['success'])], phi_next)
    C = np.maximum(0.0, -phi) * (1 - GAMMA ** m) / (1 - GAMMA)
    return dict(m=m, R=R, b=b, final=final, phi=phi, anchor=anchor, C=C)


def decompose(terms, basic32, mix32, alpha, beta):
    """terms: concatenated episode_terms. Returns attribution arrays; raises on any non-bitwise match."""
    R, phi, anchor, C = terms['R'], terms['phi'], terms['anchor'], terms['C']
    alpha_term = alpha * (anchor - phi)
    beta_cost = beta * C
    mix64 = R + alpha * (anchor - phi) - beta * C          # literal rewards.shaped expression
    if not np.array_equal(mix64.astype(np.float32), mix32):
        raise ContractError('float32(R + alpha*(anchor-phi) - beta*C) != trained mix rewards')
    if not np.array_equal(R.astype(np.float32), basic32):
        raise ContractError('float32(R) != trained basic rewards')
    delta = mix32.astype(np.float64) - basic32.astype(np.float64)
    delta_exact = alpha_term - beta_cost
    return dict(alpha_term=alpha_term, beta_cost=beta_cost, delta=delta, delta_exact=delta_exact,
                delta_float32_rounding_max_abs=float(np.abs(delta - delta_exact).max()) if len(delta) else 0.0)


# ---------------------------------------------------------------- statistics

def _pct(x):
    if len(x) == 0:
        return {f'p{p}': None for p in PCTS}
    q = np.percentile(x, PCTS)
    return {f'p{p}': float(v) for p, v in zip(PCTS, q)}


def summarize(basic, mix, alpha_term, beta_cost, delta):
    """basic/mix float32 trained rewards; others float64. All counts exact; ratios exclude basic == 0."""
    basic = np.asarray(basic, np.float64)
    mix = np.asarray(mix, np.float64)
    ad = np.abs(delta)
    sum_abs = float(ad.sum())
    zero_base = basic == 0
    nz = ~zero_base
    ratio = ad[nz] / np.abs(basic[nz])
    out = dict(
        rows=int(len(delta)),
        abs_delta=_pct(ad),
        signed_delta=_pct(delta),
        sums=dict(basic=float(basic.sum()), mix=float(mix.sum()), delta=float(delta.sum()), abs_delta=sum_abs,
                  alpha_term=float(alpha_term.sum()), abs_alpha_term=float(np.abs(alpha_term).sum()),
                  beta_cost=float(beta_cost.sum()), abs_basic=float(np.abs(basic).sum())),
        sign_counts=dict(pos=int((delta > 0).sum()), neg=int((delta < 0).sum()), zero=int((delta == 0).sum())),
        alpha_term_sign_counts=dict(pos=int((alpha_term > 0).sum()), neg=int((alpha_term < 0).sum()),
                                    zero=int((alpha_term == 0).sum())),
        beta_cost_nonzero=int((beta_cost != 0).sum()),
        # 1 = positive and negative deltas cancel completely, 0 = all deltas share one sign
        cancellation=None if sum_abs == 0 else float(1 - abs(delta.sum()) / sum_abs),
        ratio_nonzero_basic=dict(rows=int(nz.sum()), abs_delta_over_abs_basic=_pct(ratio),
                                 sum_abs_delta_over_sum_abs_basic=None if not nz.any() or np.abs(basic[nz]).sum() == 0
                                 else float(ad[nz].sum() / np.abs(basic[nz]).sum())),
        zero_basic=dict(rows=int(zero_base.sum()), abs_delta=_pct(ad[zero_base]),
                        sum_delta=float(delta[zero_base].sum()), sum_abs_delta=float(ad[zero_base].sum())),
    )
    return out


def grouped(arrs, key_mask):
    """summarize on a boolean row subset."""
    k = key_mask
    return summarize(arrs['basic'][k], arrs['mix'][k], arrs['alpha_term'][k], arrs['beta_cost'][k], arrs['delta'][k])


def fold_report(arrs):
    """Full report for one fold (or a pooled concatenation of SHA-matched folds)."""
    succ, final = arrs['success'].astype(bool), arrs['final'].astype(bool)
    term = arrs['terminal'].astype(bool)
    rep = dict(all=grouped(arrs, np.ones(len(succ), bool)),
               by_outcome=dict(success=grouped(arrs, succ), failure=grouped(arrs, ~succ)),
               by_row=dict(final=grouped(arrs, final), nonfinal=grouped(arrs, ~final),
                           terminal=grouped(arrs, term), nonterminal=grouped(arrs, ~term)))
    ep = arrs['episode_index']
    if len(ep):
        nep = int(ep.max()) + 1
        ep_sum = np.bincount(ep, weights=arrs['delta'], minlength=nep)
        ep_abs = np.bincount(ep, weights=np.abs(arrs['delta']), minlength=nep)
        ep_basic = np.bincount(ep, weights=arrs['basic'].astype(np.float64), minlength=nep)
        canc = np.where(ep_abs > 0, 1 - np.abs(ep_sum) / np.where(ep_abs > 0, ep_abs, 1), np.nan)
        rep['per_episode'] = dict(episodes=nep, sum_delta=_pct(ep_sum), sum_abs_delta=_pct(ep_abs),
                                  sum_basic=_pct(ep_basic), cancellation=_pct(canc[np.isfinite(canc)]))
    return rep


ROW_KEYS = ('basic', 'mix', 'alpha_term', 'beta_cost', 'delta', 'delta_exact', 'phi', 'anchor', 'C',
            'episode_index', 'step', 'env_step_start', 'm', 'terminal', 'final', 'success', 'j', 'n')
EP_KEYS = ('episode_id', 'ep_j', 'ep_n', 'ep_success', 'ep_steps', 'ep_max_steps', 'ep_n_decisions', 'ep_row_start')

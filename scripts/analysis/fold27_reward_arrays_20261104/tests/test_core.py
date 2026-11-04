"""Unit tests: bitwise MIX recomposition against the pinned rewards.py, statistics, segregation, no-overwrite."""
import json, sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
REPO = PKG.parents[2]
sys.path[:0] = [str(PKG), str(REPO / 'scripts/rl2_vla/groot_fold27_20261028/trqam')]
import core  # noqa: E402
import rewards as RW  # noqa: E402
import analyze  # noqa: E402


def _episode(rng, n, success, eid):
    steps = 5 * (n - 1) + int(rng.integers(1, 6)) if success else 720
    if not success:
        n = 144
    return {'episode_id': eid, 'latents': rng.normal(size=(n, 1024)).astype(np.float32),
            'actions': np.zeros((n, 16, 12), np.float32), 'success': success, 'steps': steps, 'max_episode_steps': 720}


def _pot(lat):     # deterministic phi in [-1, 1] with exact zeros, positives, negatives
    v = np.round(np.tanh(lat[:, 0].astype(np.float64)), 2)
    v[::7] = 0.0
    return v


@pytest.fixture
def eps():
    rng = np.random.default_rng(0)
    return [_episode(rng, int(rng.integers(20, 100)), s, f'p/kanu/PPCC/jug/s1/j{j}/n{k}')
            for j, s in ((0, True), (1, False), (2, True)) for k in range(3)]


def _build(eps):
    rb = RW.concat([RW.episode_transitions(e, 'basic') for e in eps])
    rm = RW.concat([RW.episode_transitions(e, 'mix', _pot, alpha=0.1, beta=0.0396) for e in eps])
    T = [core.episode_terms(RW, e, _pot(e['latents'])) for e in eps]
    T = {k: np.concatenate([t[k] for t in T]) for k in T[0]}
    return rb, rm, T


def test_parse_jn_handles_slash_j_instruction():
    assert core.parse_jn('abc/kanu/PPCC/jug/s1/j3/n17') == (3, 17)
    with pytest.raises(core.ContractError):
        core.parse_jn('abc/kanu/PPCC/jug/s1/n17')


def test_mix_recomposition_bitwise(eps):
    rb, rm, T = _build(eps)
    d = core.decompose(T, rb['rewards'], rm['rewards'], 0.1, 0.0396)
    assert np.array_equal(T['phi'].astype(np.float32), rm['phi'])
    assert np.array_equal(T['m'], rm['executed_m'])
    assert d['delta_float32_rounding_max_abs'] < 1e-5
    # final rows use the +/-1 anchor, not phi_next; C uses sum_{i<m} .99**i
    fin = T['final']
    assert set(np.unique(T['anchor'][fin])) <= {-1.0, 1.0}
    m = T['m']
    ref_c = np.maximum(0, -T['phi']) * np.array([sum(.99 ** i for i in range(k)) for k in m])
    assert np.allclose(T['C'], ref_c, rtol=0, atol=1e-12)
    # potential difference carries no gamma**m and no bootstrap b
    nf = ~fin
    assert np.allclose(d['alpha_term'][nf], 0.1 * (np.append(T['phi'][1:], 0)[nf] - T['phi'][nf]))


def test_decompose_rejects_perturbed_rewards(eps):
    rb, rm, T = _build(eps)
    bad = rm['rewards'].copy(); bad[3] = np.nextafter(bad[3], np.float32(10))
    with pytest.raises(core.ContractError):
        core.decompose(T, rb['rewards'], bad, 0.1, 0.0396)
    with pytest.raises(core.ContractError):     # wrong beta
        core.decompose(T, rb['rewards'], rm['rewards'], 0.1, 0.04)


def test_summary_counts_and_zero_baseline():
    basic = np.array([-1, -2, 0, 0, -1], np.float32)
    delta = np.array([0.5, -0.5, 0.25, 0.0, -0.25])
    s = core.summarize(basic, basic + delta, delta, np.zeros(5), delta)
    assert s['sign_counts'] == {'pos': 2, 'neg': 2, 'zero': 1}
    assert s['zero_basic']['rows'] == 2 and s['zero_basic']['sum_abs_delta'] == 0.25
    assert s['ratio_nonzero_basic']['rows'] == 3
    assert s['ratio_nonzero_basic']['sum_abs_delta_over_sum_abs_basic'] == pytest.approx(1.25 / 4)
    assert s['cancellation'] == pytest.approx(1 - 0 / 1.5)
    assert s['abs_delta']['p0'] == 0 and s['abs_delta']['p100'] == 0.5


def _fake_fold(root, status, name, gates_ok=True, n=10):
    d = root / status / name; d.mkdir(parents=True)
    arr = dict(basic=-np.ones(n, np.float32), mix=-np.ones(n, np.float32) + .1, alpha_term=np.full(n, .1),
               beta_cost=np.zeros(n), delta=np.full(n, .1), episode_index=np.repeat([0, 1], n // 2).astype(np.int32),
               terminal=np.zeros(n, bool), final=np.zeros(n, bool), success=np.zeros(n, bool))
    if status == 'sha_matched':
        np.savez(d / 'reward_arrays.npz', **arr)
    (d / 'receipt.json').write_text(json.dumps({'status': status, 'sha': {'gates': {'rewards_mix': gates_ok}},
                                                'error': None if status == 'sha_matched' else 'x'}))


def test_analyze_segregates_failures(tmp_path):
    der = tmp_path / 'der'
    _fake_fold(der, 'sha_matched', 'A'); _fake_fold(der, 'sha_matched', 'B')
    _fake_fold(der, 'sha_matched', 'C', gates_ok=False)
    _fake_fold(der, 'contract_failure', 'D')
    out = tmp_path / 'ana'
    assert analyze.main(['--derived', str(der), '--out', str(out), '--expect-folds', '4']) == 0
    res = json.loads((out / 'analysis.json').read_text())
    assert sorted(res['folds']) == ['A', 'B'] and 'C' in res['rejected'] and 'D' in res['contract_failure']
    assert res['pooled']['all']['rows'] == 20 and res['pooled']['per_episode']['episodes'] == 4
    assert analyze.main(['--derived', str(der), '--out', str(out)]) == 3      # no overwrite


def test_reconstruct_refuses_existing(tmp_path, monkeypatch):
    import reconstruct
    (tmp_path / 'sha_matched' / 'F').mkdir(parents=True)
    rc = reconstruct.main(['--spec', 'x', '--fold', 'F', '--episodes', 'x', '--latents', 'x', '--fold-dir', 'x',
                           '--out-root', str(tmp_path)])
    assert rc == 3


def test_reconstruct_missing_input_goes_to_contract_failure(tmp_path):
    import reconstruct
    rc = reconstruct.main(['--spec', 'x', '--fold', 'F', '--episodes', 'x', '--latents', 'x',
                           '--fold-dir', str(tmp_path / 'nofold'), '--out-root', str(tmp_path / 'o')])
    assert rc == 2
    r = json.loads((tmp_path / 'o/contract_failure/F/receipt.json').read_text())
    assert r['status'] == 'contract_failure' and 'missing input' in r['error']
    assert not (tmp_path / 'o/sha_matched').exists()


def test_analyze_supersedes_failed_attempt_from_other_root(tmp_path):
    r1, r2 = tmp_path / 'r1', tmp_path / 'r2'
    _fake_fold(r1, 'sha_matched', 'A'); _fake_fold(r1, 'contract_failure', 'B'); _fake_fold(r2, 'sha_matched', 'B')
    out = tmp_path / 'ana'
    assert analyze.main(['--derived', str(r1), str(r2), '--out', str(out), '--expect-folds', '2']) == 0
    res = json.loads((out / 'analysis.json').read_text())
    assert sorted(res['folds']) == ['A', 'B'] and res['contract_failure'] == {}
    assert list(res['superseded_failed_attempts']) == ['B'] and not res['missing']

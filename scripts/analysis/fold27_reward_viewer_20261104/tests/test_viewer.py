"""Viewer tests: data validation, routing/traversal/input validation, payload semantics, video index, HTTP smoke."""
import hashlib, json, os, sys, threading, urllib.error, urllib.request
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
import viewer_data as VD  # noqa: E402
import server as SV  # noqa: E402


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def make_fold(root, fold, episodes, status='sha_matched', gates_ok=True, corrupt=False):
    """episodes: [(episode_id, success, n_decisions)]"""
    d = root / 'sha_matched' / fold; d.mkdir(parents=True)
    cols = {k: [] for k in ('basic', 'mix', 'alpha_term', 'beta_cost', 'delta', 'phi', 'anchor', 'C', 'episode_index',
                            'step', 'env_step_start', 'm', 'terminal', 'final', 'success')}
    starts, cur = [], 0
    for i, (eid, succ, n) in enumerate(episodes):
        starts.append(cur); cur += n
        m = [5] * (n - 1) + [3 if succ else 5]
        phi = np.linspace(-0.5, 0.5, n)
        anchor = np.append(phi[1:], 1.0 if succ else -1.0)
        alpha = 0.1 * (anchor - phi)
        C = np.maximum(0, -phi) * np.array([sum(.99 ** k for k in range(mm)) for mm in m])
        basic = -np.array([sum(.99 ** k for k in range(mm)) for mm in m])
        cols['basic'] += list(basic); cols['alpha_term'] += list(alpha); cols['beta_cost'] += list(0.0396 * C)
        cols['mix'] += list(basic + alpha - 0.0396 * C); cols['delta'] += list(alpha - 0.0396 * C)
        cols['phi'] += list(phi); cols['anchor'] += list(anchor); cols['C'] += list(C)
        cols['episode_index'] += [i] * n; cols['step'] += list(range(n)); cols['env_step_start'] += [5 * k for k in range(n)]
        cols['m'] += m; cols['terminal'] += [False] * (n - 1) + [True]; cols['final'] += [False] * (n - 1) + [True]
        cols['success'] += [succ] * n
    arr = {k: np.asarray(v, np.float32 if k in ('basic', 'mix') else None) for k, v in cols.items()}
    arr.update(episode_id=np.asarray([e for e, _, _ in episodes]),
               ep_j=np.asarray([VD.split_episode_id(e)[2] for e, _, _ in episodes], np.int16),
               ep_n=np.asarray([VD.split_episode_id(e)[3] for e, _, _ in episodes], np.int16),
               ep_success=np.asarray([s for _, s, _ in episodes]),
               ep_steps=np.asarray([5 * (n - 1) + (3 if s else 5) for _, s, n in episodes], np.int32),
               ep_max_steps=np.full(len(episodes), 720, np.int32),
               ep_n_decisions=np.asarray([n for _, _, n in episodes], np.int32),
               ep_row_start=np.asarray(starts, np.int64))
    np.savez(d / 'reward_arrays.npz', **arr)
    sha = _sha(d / 'reward_arrays.npz')
    rec = {'fold': fold, 'status': status, 'host': 'test', 'episode_scalars': 'npz',
           'sha': {'gates': {'rewards_mix': gates_ok}, 'done_fingerprint': {'rows': cur, 'mix_rewards_sha256': 'a'},
                   'reconstructed': {'rows': cur, 'mix_rewards_sha256': 'a'}},
           'outputs': {'reward_arrays.npz': {'sha256': 'f' * 64 if corrupt else sha}}}
    (d / 'receipt.json').write_text(json.dumps(rec))
    return d


E1 = 'p1/kanu/PPCC/jug/s1/j0/n3'
E2 = 'p1/kanu/PPCC/jug/s1/j2/n4'


@pytest.fixture
def app(tmp_path):
    r = tmp_path / 'der'
    make_fold(r, 'PPCC-jug_s1_j1', [(E1, True, 4), (E2, False, 6)])
    make_fold(r, 'PPCC-jug_s1_j3', [(E1, True, 4)])
    (r / 'contract_failure' / 'PPCC-jug_s1_j4').mkdir(parents=True)
    (r / 'contract_failure' / 'PPCC-jug_s1_j4' / 'receipt.json').write_text(json.dumps({'error': 'boom'}))
    return SV.App([r])


def test_split_episode_id_slash_instruction():
    assert VD.split_episode_id('a/b/DishwasherRack/out-left/s3/j1/n2') == ('DishwasherRack/out-left', 3, 1, 2)


@pytest.mark.parametrize('kw', [dict(corrupt=True), dict(gates_ok=False), dict(status='contract_failure')])
def test_rejects_unverified(tmp_path, kw):
    make_fold(tmp_path / 'r', 'PPCC-jug_s1_j1', [(E1, True, 3)], **kw)
    with pytest.raises(VD.DataError):
        VD.load_all([tmp_path / 'r'])


def test_meta_and_duplicates(app):
    m = app.meta
    assert set(m['tasks']) == {'PPCC'} and len(m['tasks']['PPCC']['PPCC/jug']) == 2
    assert m['failed_attempts_not_served'] == {'PPCC-jug_s1_j4': ['boom']}
    assert '미저장' in m['cluster_id'] and '평가 아님' in m['provenance']
    assert app.dup[E1] == [('PPCC-jug_s1_j1', 0), ('PPCC-jug_s1_j3', 0)]


def test_episode_payload(app):
    code, _, body = app.route('/api/episode?fold=PPCC-jug_s1_j1&ep=1')
    p = json.loads(body)
    assert code == 200 and p['episode_id'] == E2 and not p['success'] and p['terminal_anchor'] == -1.0
    c = p['chunks']
    f = app.folds['PPCC-jug_s1_j1']['arrays']
    assert c['minus_beta_cost'] == [-float(x) for x in f['beta_cost'][4:10]]          # actual subtraction, signed
    assert np.allclose(np.add(c['alpha_term'], c['minus_beta_cost']), c['delta'])
    assert c['final'][-1] and c['env_step_start'] == [0, 5, 10, 15, 20, 25]
    assert p['cluster_id'] is None and p['cluster_id_note'] == '미저장' and p['phi'] is not None
    assert [d['fold'] for d in p['duplicates']] == ['PPCC-jug_s1_j1'] and p['video'] is False
    p0 = json.loads(app.route('/api/episode?fold=PPCC-jug_s1_j1&ep=0')[2])
    assert p0['last_chunk_m'] == 3 and len(p0['duplicates']) == 2


@pytest.mark.parametrize('path,code', [
    ('/static/../server.py', 404), ('/static/%2e%2e/server.py', 404), ('/static/..%2fviewer_data.py', 404),
    ('/static/index.html', 404), ('/static/app.js', 200), ('/', 200), ('/api/meta', 200),
    ('/api/fold?fold=../../etc', 404), ('/api/fold?fold=PPCC-jug_s1_j1', 200),
    ('/api/episode?fold=PPCC-jug_s1_j1&ep=-1', 400), ('/api/episode?fold=PPCC-jug_s1_j1&ep=abc', 400),
    ('/api/episode?fold=PPCC-jug_s1_j1&ep=2', 400), ('/api/episode?fold=PPCC-jug_s1_j1&ep=1&ep=0', 400),
    ('/api/episode?fold=PPCC-jug_s1_j1&ep=1e1', 400), ('/video?id=0', 404), ('/video?id=../x', 404),
    ('/etc/passwd', 404), ('//etc/passwd', 404)])
def test_routes(app, path, code):
    assert app.route(path)[0] == code


def test_video_index_validation(tmp_path):
    root = tmp_path / 'vids'; root.mkdir()
    good = root / 'a.mp4'; good.write_bytes(b'\x00\x00\x00\x18ftypmp42' + b'x' * 100)
    outside = tmp_path / 'secret.mp4'; outside.write_bytes(b'secret')
    os.symlink(outside, root / 'link.mp4')
    (root / 'b.txt').write_bytes(b't')
    idx = {E1: {'path': 'a.mp4', 'sha256': _sha(good)},
           E2: {'path': '../secret.mp4', 'sha256': _sha(outside)},
           'p/m/PPCC/jug/s1/j0/n9': {'path': 'link.mp4', 'sha256': _sha(outside)},
           'p/m/PPCC/jug/s1/j0/n8': {'path': 'b.txt', 'sha256': _sha(root / 'b.txt')},
           'p/m/PPCC/jug/s1/j0/n7': {'path': 'a.mp4', 'sha256': '0' * 64},
           'p/m/PPCC/jug/s1/j0/n6': {'path': str(good), 'sha256': _sha(good)}}
    (tmp_path / 'idx.json').write_text(json.dumps(idx))
    ok, rej = VD.load_video_index(tmp_path / 'idx.json', root)
    assert list(ok) == [E1] and len(rej) == 5


def test_reserved_port_refused(tmp_path):
    with pytest.raises(SystemExit):
        SV.main(['--derived', str(tmp_path), '--port', '8782'])


def test_http_smoke_with_video_range(tmp_path):
    r = tmp_path / 'der'
    make_fold(r, 'PPCC-jug_s1_j1', [(E1, True, 4)])
    root = tmp_path / 'v'; root.mkdir(); data = bytes(range(256)) * 4; (root / 'a.mp4').write_bytes(data)
    (tmp_path / 'idx.json').write_text(json.dumps({E1: {'path': 'a.mp4', 'sha256': hashlib.sha256(data).hexdigest()}}))
    app = SV.App([r], tmp_path / 'idx.json', root)
    srv = SV.ThreadingHTTPServer(('127.0.0.1', 0), SV.make_handler(app))
    th = threading.Thread(target=srv.serve_forever, daemon=True); th.start()
    base = 'http://127.0.0.1:%d' % srv.server_address[1]
    try:
        with urllib.request.urlopen(base + '/api/meta') as resp:
            assert resp.status == 200 and resp.headers['X-Content-Type-Options'] == 'nosniff'
            assert "default-src 'self'" in resp.headers['Content-Security-Policy']
        p = json.loads(urllib.request.urlopen(base + '/api/episode?fold=PPCC-jug_s1_j1&ep=0').read())
        assert p['video'] and p['video_id'] == 0
        req = urllib.request.Request(base + '/video?id=0', headers={'Range': 'bytes=10-19'})
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 206 and resp.read() == data[10:20]
            assert resp.headers['Content-Range'] == 'bytes 10-19/1024'
        assert urllib.request.urlopen(base + '/video?id=0').read() == data
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(urllib.request.Request(base + '/api/meta', data=b'x', method='POST'))
        assert e.value.code == 405
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(base + '/static/../server.py')
        assert e.value.code == 404
    finally:
        srv.shutdown(); srv.server_close()

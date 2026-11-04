"""Viewer tests: data validation, routing/traversal/input validation, payload semantics, video index, HTTP smoke."""
import hashlib, json, os, subprocess, sys, threading, urllib.error, urllib.request
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
import viewer_data as VD  # noqa: E402
import server as SV  # noqa: E402


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def make_fold(root, fold, episodes, status='sha_matched', gates_ok=True, corrupt=False, coef=True):
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
    if coef:
        rec['decomposition'] = {'alpha': 0.1, 'beta': 0.0396, 'gamma': 0.99}
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


G5 = -sum(.99 ** k for k in range(5))
G3 = -sum(.99 ** k for k in range(3))


def test_means_unequal_length_known_values(tmp_path):
    r = tmp_path / 'r'
    # success: lengths 3 and 1 (m = [5,5,3] and [3]); failure: one rollout of length 2 (m = [5,5])
    make_fold(r, 'PPCC-jug_s1_j1', [(E1, True, 3), ('p1/kanu/PPCC/jug/s1/j0/n5', True, 1), (E2, False, 2)])
    app = SV.App([r])
    g = app.means['PPCC-jug_s1_j1']
    s, f = g['success'], g['failure']
    assert s['total_n'] == 2 and s['n_t'] == [2, 1, 1] and s['length'] == {'min': 1, 'max': 3}
    want = [(G5 + G3) / 2, G5, G3]                       # t>=1 averages ONLY the rollout that exists there
    assert np.allclose(s['mean']['basic'], want, atol=1e-6)
    assert np.allclose(s['cum_mean']['basic'], [(G5 + G3) / 2, 2 * G5, 2 * G5 + G3], atol=1e-6)
    assert f['total_n'] == 1 and f['n_t'] == [1, 1] and np.allclose(f['mean']['basic'], [G5, G5], atol=1e-6)
    a = app.folds['PPCC-jug_s1_j1']['arrays']
    assert np.allclose(s['mean']['minus_beta_cost'][1:], -a['beta_cost'][1:3])
    assert np.allclose(np.add(s['mean']['alpha_term'], s['mean']['minus_beta_cost']), s['mean']['delta'])


def test_means_match_independent_existing_only_average(app):
    a = app.folds['PPCC-jug_s1_j1']['arrays']; st = app.folds['PPCC-jug_s1_j1']['starts']
    for name, flag in (('success', True), ('failure', False)):
        rows = [list(a['delta'][st[i]:st[i + 1]]) for i in range(len(st) - 1) if bool(a['ep_success'][i]) == flag]
        L = max(map(len, rows))
        ref = [np.mean([r[t] for r in rows if len(r) > t]) for t in range(L)]
        assert np.allclose(app.means['PPCC-jug_s1_j1'][name]['mean']['delta'], ref)


def test_empty_outcome_group(tmp_path):
    make_fold(tmp_path / 'r', 'PPCC-jug_s1_j1', [(E1, True, 3)])
    g = SV.App([tmp_path / 'r']).means['PPCC-jug_s1_j1']['failure']
    assert g == {'total_n': 0, 't': [], 'n_t': [], 'mean': None, 'cum_mean': None}


def test_coefficients_from_receipt_or_unconfirmed(tmp_path):
    make_fold(tmp_path / 'a', 'PPCC-jug_s1_j1', [(E1, True, 3)])
    make_fold(tmp_path / 'b', 'PPCC-jug_s1_j1', [(E1, True, 3)], coef=False)
    ca = json.loads(SV.App([tmp_path / 'a']).route('/api/fold_means?fold=PPCC-jug_s1_j1')[2])['coefficients']
    cb = json.loads(SV.App([tmp_path / 'b']).route('/api/fold_means?fold=PPCC-jug_s1_j1')[2])['coefficients']
    assert (ca['alpha'], ca['beta']) == (0.1, 0.0396) and ca['source']
    assert cb['alpha'] is None and cb['beta'] is None and cb['source'] is None
    assert SV.App([tmp_path / 'a']).route('/api/fold_means?fold=../x')[0] == 404


CHROMIUM = Path.home() / '.cache/ms-playwright/chromium_headless_shell-1181/chrome-linux/headless_shell'


@pytest.mark.skipif(not CHROMIUM.exists(), reason='headless chromium not available')
def test_browser_fixture_means_coef_and_toggles(app):
    srv = SV.ThreadingHTTPServer(('127.0.0.1', 0), SV.make_handler(app))
    th = threading.Thread(target=srv.serve_forever, daemon=True); th.start()
    base = 'http://127.0.0.1:%d/' % srv.server_address[1]

    def dom(url):
        return subprocess.run([str(CHROMIUM), '--headless', '--no-sandbox', '--disable-gpu',
                               '--virtual-time-budget=6000', '--dump-dom', url],
                              capture_output=True, timeout=60).stdout.decode('utf-8', 'replace')
    try:
        d0, d1 = dom(base), dom(base + '#hide=alpha_term,delta')
    finally:
        srv.shutdown(); srv.server_close()
    for d in (d0, d1):
        assert 'α = 0.1' in d and 'β = 0.0396' in d and 'receipt' in d
        assert '성공 평균 · 집단 total N=1' in d and '실패 평균 · 집단 total N=1' in d and 'N_t' in d
        assert '0 채움' in d and '길이 혼입' in d
    # 2 decomposition charts (success mean, failure mean) + individual rollout: each key drawn in all of them
    assert d0.count('data-key="alpha_term" d=') == 3 and d0.count('data-key="delta" d=') == 3
    assert d1.count('data-key="alpha_term" d=') == 0 and d1.count('data-key="delta" d=') == 0
    assert d1.count('data-key="minus_beta_cost" d=') == 3
    assert d1.count('aria-pressed="false"') == 6 and d0.count('aria-pressed="false"') == 0


def test_build_video_index_split_and_roundtrip(tmp_path):
    import build_video_index as BV
    der = tmp_path / 'der'
    e3, e4, e5 = 'p1/kanu/PPCC/jug/s1/j0/n7', 'p1/kanu/PPCC/jug/s1/j0/n8', 'p1/kanu/PPCC/jug/s1/j0/n9'
    make_fold(der, 'PPCC-jug_s1_j1', [(E1, True, 3), (E2, False, 2), (e3, True, 2), (e4, False, 2), (e5, True, 2)])
    root = tmp_path / 'grid'
    mp4 = b'\x00\x00\x00\x18ftypisom' + bytes(200)

    def put(eid, data=mp4, meta=None):
        d = root / eid / 'base'; d.mkdir(parents=True)
        (d / 'video.mp4').write_bytes(data)
        if meta is not None:
            (d / 'meta.json').write_text(json.dumps(meta))
    put(E1, meta={'success': 1, 'jitter': 0, 'noise': 3})      # ok, identity checked
    put(e3)                                                    # ok, path only
    put(e4, meta={'success': 1})                               # arrays say failure -> mismatch
    put(e5, data=b'not a video at all')                        # no ftyp -> mismatch
    outside = tmp_path / 'outside.mp4'; outside.write_bytes(mp4)
    (root / E2 / 'base').mkdir(parents=True); os.symlink(outside, root / E2 / 'base' / 'video.mp4')   # escapes root
    assert BV.main(['--derived', str(der), '--video-root', str(root), '--out', str(tmp_path / 'vi')]) == 0
    rec = json.loads((tmp_path / 'vi/video_index_receipt.json').read_text())
    assert sorted(rec['ok']) == [E1, e3] and rec['ok'][E1]['identity'] == 'meta_checked'
    assert rec['ok'][e3]['identity'] == 'path_only' and sorted(rec['mismatch']) == sorted([E2, e4, e5])
    assert rec['missing'] == {} and rec['counts']['ok'] == 2
    with pytest.raises(SystemExit):     # never overwrites
        BV.main(['--derived', str(der), '--video-root', str(root), '--out', str(tmp_path / 'vi')])
    app = SV.App([der], tmp_path / 'vi/video_index.json', root)       # viewer re-verifies sha256 at startup
    assert sorted(app.videos) == [E1, e3] and app.video_rejected == {}
    p = json.loads(app.route('/api/episode?fold=PPCC-jug_s1_j1&ep=0')[2])
    assert p['video'] and p['video_id'] == 0
    q = json.loads(app.route('/api/episode?fold=PPCC-jug_s1_j1&ep=1')[2])
    assert not q['video'] and q['video_id'] is None
    shutil_missing = tmp_path / 'grid2'; shutil_missing.mkdir()
    assert BV.main(['--derived', str(der), '--video-root', str(shutil_missing), '--out', str(tmp_path / 'vi2')]) == 0
    assert json.loads((tmp_path / 'vi2/video_index_receipt.json').read_text())['counts']['missing'] == 5


# ---------------------------------------------------------------- video <-> chart sync (probe_video_sync + front end)
def make_mp4(path, frames, codec='libx264', fps=20):
    """Real encoded MP4 (PyAV) so the probe parses an actual muxer's moov (stts/ctts/elst), not a hand-made box."""
    av = pytest.importorskip('av')
    from fractions import Fraction
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), 'w', format='mp4') as c:
        st = c.add_stream(codec, rate=fps)
        st.width, st.height, st.pix_fmt = 64, 48, 'yuv420p'
        for i in range(frames):
            fr = av.VideoFrame.from_ndarray(np.full((48, 64, 3), (i * 23) % 255, np.uint8), format='rgb24')
            fr.pts, fr.time_base = i, Fraction(1, fps)
            for pk in st.encode(fr):
                c.mux(pk)
        for pk in st.encode():
            c.mux(pk)
    return path


def simulate_recorder(K, s):
    """Independent re-statement of upstream VideoRecordingWrapper: reset -> step_count=1, no frame;
    each env step: step_count += 1, frame iff step_count % s == 0. Returns the 0-based env step after each frame."""
    after, sc = [], 1
    for e in range(K):
        sc += 1
        if sc % s == 0:
            after.append(e)
    return after


@pytest.mark.parametrize('s', [1, 2, 3, 4, 5])
def test_frame_rule_matches_wrapper_simulation(s):
    import probe_video_sync as PV
    for K in range(0, 60):
        after = simulate_recorder(K, s)
        assert PV.predicted_frames(K, s) == len(after)
        e0 = s * -(-2 // s) - 2
        assert after == [e0 + s * f for f in range(len(after))]
    assert PV.predicted_frames(720, 2) == 360


def test_probe_real_mp4_verified_and_rejections(tmp_path):
    import probe_video_sync as PV
    root = tmp_path / 'grid'
    good = make_mp4(root / 'a/base/video.mp4', 9)               # K=18, s=2 -> 9 frames
    tm = PV.video_timing(good)
    assert tm['frames'] == 9 and tm['fps'] == 20 and tm['pts_regular_from_zero']
    r = PV.probe_one(root, {'path': 'a/base/video.mp4', 'sha256': _sha(good)}, 18)
    assert r['status'] == 'verified', r
    assert r['sync'] == {**r['sync'], 'fps': 20, 'frames': 9, 'steps_per_render': 2, 'first_frame_env_step': 0, 'env_steps': 18}
    short = make_mp4(root / 'b/base/video.mp4', 8)              # off-by-one frame count -> never synced
    r = PV.probe_one(root, {'path': 'b/base/video.mp4', 'sha256': _sha(short)}, 18)
    assert r['status'] == 'unverified' and 'frame count 8' in r['reason']
    fast = make_mp4(root / 'c/base/video.mp4', 9, fps=25)       # right count, wrong rate -> never synced
    r = PV.probe_one(root, {'path': 'c/base/video.mp4', 'sha256': _sha(fast)}, 18)
    assert r['status'] == 'unverified' and 'fps' in r['reason']
    r = PV.probe_one(root, {'path': 'a/base/video.mp4', 'sha256': '0' * 64}, 18)
    assert r['status'] == 'unverified' and 'sha256' in r['reason']
    (root / 'a/base/meta.json').write_text(json.dumps({'steps_per_render': 1, 'video_fps': 20}))   # meta overrides default
    r = PV.probe_one(root, {'path': 'a/base/video.mp4', 'sha256': _sha(good)}, 18)
    assert r['status'] == 'unverified' and 'expected exactly s=1' in r['reason']
    bad = root / 'd/base/video.mp4'; bad.parent.mkdir(parents=True)
    bad.write_bytes(b'\x00\x00\x00\x18ftypisom' + bytes(64))   # no moov
    r = PV.probe_one(root, {'path': 'd/base/video.mp4', 'sha256': _sha(bad)}, 18)
    assert r['status'] == 'unverified' and r['reason'].startswith('container')


def _sync_fixture(tmp_path, codec='libx264'):
    import build_video_index as BV
    import probe_video_sync as PV
    der, root = tmp_path / 'der', tmp_path / 'grid'
    make_fold(der, 'PPCC-jug_s1_j1', [(E1, True, 4), (E2, False, 6)])   # K: E1=18, E2=30
    make_mp4(root / E1 / 'base/video.mp4', 9, codec)                   # matches s=2
    make_mp4(root / E2 / 'base/video.mp4', 14, codec)                  # 30 steps needs 15 -> unverified
    assert BV.main(['--derived', str(der), '--video-root', str(root), '--out', str(tmp_path / 'vi')]) == 0
    assert PV.main(['--derived', str(der), '--video-index', str(tmp_path / 'vi/video_index.json'),
                    '--video-root', str(root), '--out', str(tmp_path / 'vs')]) == 0
    with pytest.raises(SystemExit):                                    # never overwrites
        PV.main(['--derived', str(der), '--video-index', str(tmp_path / 'vi/video_index.json'),
                 '--video-root', str(root), '--out', str(tmp_path / 'vs')])
    return der, root


def test_probe_cli_load_sync_and_payload(tmp_path):
    der, root = _sync_fixture(tmp_path)
    sync = json.loads((tmp_path / 'vs/video_sync.json').read_text())
    rec = json.loads((tmp_path / 'vs/video_sync_receipt.json').read_text())
    assert list(sync) == [E1] and rec['verified'] == 1 and list(rec['unverified']) == [E2]
    app = SV.App([der], tmp_path / 'vi/video_index.json', root, tmp_path / 'vs/video_sync.json')
    p = json.loads(app.route('/api/episode?fold=PPCC-jug_s1_j1&ep=0')[2])
    assert p['video_sync']['frames'] == 9 and p['video_sync_reason'] is None
    q = json.loads(app.route('/api/episode?fold=PPCC-jug_s1_j1&ep=1')[2])
    assert q['video'] and q['video_sync'] is None and q['video_sync_reason'].startswith('probe 미검증: frame count 14')
    # tampered sync file: sha differs from the video index / record incomplete -> rejected, not synced
    bad = dict(sync); bad[E1] = {**sync[E1], 'video_sha256': '0' * 64}
    (tmp_path / 'vs2').mkdir(); (tmp_path / 'vs2/video_sync.json').write_text(json.dumps(bad))
    ok, rej = VD.load_video_sync(tmp_path / 'vs2/video_sync.json', tmp_path / 'vi/video_index.json')
    assert ok == {} and 'sha256' in rej[E1]
    bad[E1] = {k: v for k, v in sync[E1].items() if k != 'first_frame_env_step'}
    (tmp_path / 'vs2/video_sync.json').write_text(json.dumps(bad))
    ok, rej = VD.load_video_sync(tmp_path / 'vs2/video_sync.json', tmp_path / 'vi/video_index.json')
    assert ok == {} and rej[E1] == 'incomplete sync record'
    # sync recorded for a different env-step count than the arrays -> dropped at serve time
    bad[E1] = {**sync[E1], 'env_steps': 17}
    (tmp_path / 'vs2/video_sync.json').write_text(json.dumps(bad))
    app2 = SV.App([der], tmp_path / 'vi/video_index.json', root, tmp_path / 'vs2/video_sync.json')
    assert json.loads(app2.route('/api/episode?fold=PPCC-jug_s1_j1&ep=0')[2])['video_sync'] is None


@pytest.mark.skipif(not CHROMIUM.exists(), reason='headless chromium not available')
def test_browser_fixture_video_cursor_and_click_seek(tmp_path, monkeypatch):
    # VP9-in-MP4: open-source Chromium has no H.264 decoder; the moov timing the probe checks is codec-independent
    der, root = _sync_fixture(tmp_path, codec='libvpx-vp9')
    app = SV.App([der], tmp_path / 'vi/video_index.json', root, tmp_path / 'vs/video_sync.json')
    monkeypatch.setitem(SV.STATIC, 'sync_harness.js', 'text/javascript; charset=utf-8')   # test-only static entry
    app.static['sync_harness.js'] = (PKG / 'tests/sync_harness.js').read_bytes()
    app.static['index.html'] = app.static['index.html'].replace(
        b'</body>', b'<script src="/static/sync_harness.js" defer></script></body>')
    import urllib.parse
    results, done = [], threading.Event()
    real_route = app.route

    def route(raw):     # test-only sink for the harness reports; everything else is the real router
        if raw.startswith('/harness-result?'):
            o = json.loads(urllib.parse.parse_qs(raw.split('?', 1)[1])['d'][0]); results.append(o)
            if o.get('done'):
                done.set()
            return 200, {'Content-Type': 'text/plain'}, b'ok'
        return real_route(raw)
    app.route = route
    srv = SV.ThreadingHTTPServer(('127.0.0.1', 0), SV.make_handler(app))
    th = threading.Thread(target=srv.serve_forever, daemon=True); th.start()
    prof = tmp_path / 'chrome-profile'
    br = subprocess.Popen([str(CHROMIUM), '--headless', '--no-sandbox', '--disable-gpu', '--user-data-dir=' + str(prof),
                           '--autoplay-policy=no-user-gesture-required', '--remote-debugging-port=0',
                           'http://127.0.0.1:%d/' % srv.server_address[1]],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        finished = done.wait(90)
    finally:
        br.kill(); br.wait(10); srv.shutdown(); srv.server_close()
    o = results[-1] if results else {}
    assert finished and 'error' not in o, o
    Xi = lambda x: 56 + 774 * x / 4      # inference axis: 4 decisions (E1)
    Xe = lambda x: 56 + 774 * x / 18     # env axis: 18 env steps

    def at(curs, x):
        return len(curs) == 3 and all(v == 'visible' and abs(px - x) < 0.05 for v, px in curs)
    assert o['verified_text'] and o['means_cursors'] == 0 and o['n_cursors'] == 3
    assert abs(o['duration'] - 9 / 20) < 1e-6
    # seek to frame 4 -> observation after env step 8 -> inference 1 (env 5..9), 4/5 through it
    assert at(o['seek_frame4']['curs'], Xi(1.8)), o['seek_frame4']
    assert 'frame 5/9 = env step 8' in o['seek_frame4']['lab'] and 'inference step 1' in o['seek_frame4']['lab']
    # click inside inference 2 (env 10..14) -> first frame at/after env step 10 = frame 5, mid-frame t
    assert abs(o['click_inf2']['t'] - 5.5 / 20) < 1e-3 and at(o['click_inf2']['curs'], Xi(2.2))
    assert 'inference step 2' in o['click_inf2']['lab']
    # last inference (3, env 15..17, m=3) -> frame 8 (last), within duration
    assert abs(o['click_last']['t'] - 8.5 / 20) < 1e-3 and o['click_last']['t'] < o['duration']
    assert 'frame 9/9 = env step 16' in o['click_last']['lab'] and 'inference step 3' in o['click_last']['lab']
    assert at(o['env_axis']['curs'], Xe(17))                  # same frame, env x-axis
    assert abs(o['env_click']['t'] - 3.5 / 20) < 1e-3 and at(o['env_click']['curs'], Xe(7))
    assert at(o['cum']['curs'], Xe(7)) and at(o['hidden_series']['curs'], Xe(7))
    assert o['hidden_series']['hash'].startswith('#hide=') and at(o['paused']['curs'], Xe(7))
    assert o['ended_fired'] and at(o['ended']['curs'], Xe(17))
    # rollout change -> E2 (video present but probe-unverified): no cursor, explicit reason, click does not seek
    assert all(v == 'hidden' for v, _ in o['changed']['curs']) and len(o['changed']['curs']) == 3
    assert '동기화 미검증' in o['changed']['unverified_text'] and 'frame count 14' in o['changed']['unverified_text']
    assert o['changed']['lab'] == '' and o['unverified_click']['t'] == 0
    assert all(v == 'hidden' for v, _ in o['unverified_click']['curs'])

"""CPU tests for fold_data/mil_continuation.py with a directory-backed fake remote and a fake process launcher.

    .venvs/rl2_iid/bin/python -m pytest -q scripts/rl2_vla/groot_fold27_20261028/tests/test_mil_continuation.py
"""
import hashlib, json, shutil, sys, tarfile, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'fold_data')]
import mil_continuation as MC   # noqa: E402
import compact_export as CE     # noqa: E402
import run_fold as RF           # noqa: E402

SPEC = ROOT.parents[2] / 'outputs/analysis/groot_fold27_20261028/folds/fold_spec.json'
DONE15 = ['OpenDrawer-left_s2_j2', 'OpenDrawer-left_s2_j3', 'OpenDrawer-left_s2_j4', 'PPCC-bread_s4_j1', 'PPCC-bread_s4_j3',
          'PPCC-bread_s4_j4', 'PPCC-candle_s0_j0', 'PPCC-candle_s0_j1', 'PPCC-candle_s0_j2', 'PPCC-jug_s1_j1', 'PPCC-jug_s1_j3',
          'PPCC-jug_s1_j4', 'PPCC-marshmallow_s4_j0', 'PPCC-marshmallow_s4_j2', 'PPCC-marshmallow_s4_j3']


def commit(runs, fid):
    p = runs / fid / 'attn/raw/s0'; p.mkdir(parents=True, exist_ok=True)
    (p / 'model.pt').write_bytes(b'model-' + fid.encode())
    (p / 'FOLD_CONTRACT.json').write_text(json.dumps({'model_sha256': hashlib.sha256((p / 'model.pt').read_bytes()).hexdigest(),
                                                      'heldout_proof': {k: True for k in RF.PROOF_KEYS}}))


class FakeRemote:
    def __init__(self, running=(1,)):
        self.running, self.calls = list(running), []

    def read(self, p): return Path(p).read_bytes()
    def exists(self, p): return Path(p).exists()
    def count_records(self, d): return sum(1 for q in Path(d).glob('*.json') if q.name not in ('contract.json', 'COMPLETE.json'))
    def extract_running(self, d): return self.running.pop(0) if len(self.running) > 1 else self.running[0]
    def sha256(self, p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    def run_py(self, args, log):
        self.calls.append(args); Path(log).write_text('fake')
        CE.export(Path(args[2]), Path(args[4]), log=lambda *_: None)
        return 0

    def tar_to(self, d, dst):
        with tarfile.open(dst, 'x') as tf:
            tf.add(d, arcname='.')


class FakeProc:
    def __init__(self, rc, on_done):
        self.rc, self.on_done, self.polled = rc, on_done, False

    def poll(self):
        if not self.polled:
            self.polled = True
            return None
        self.on_done(); return self.rc


@unittest.skipUnless(SPEC.exists(), 'production fold_spec.json absent')
class ContinuationTest(unittest.TestCase):
    def setUp(self):
        self.t = Path(tempfile.mkdtemp()); self.runs = self.t / 'runs'
        for f in DONE15:
            commit(self.runs, f)
        self.remote_x = self.t / 'remote/remaining_a_v4'; self.remote_x.mkdir(parents=True)
        self.remote_c = self.t / 'remote/remaining_a_v4_compact'
        code = self.t / 'remote/code/fold_data'; code.mkdir(parents=True)
        shutil.copy(MC.FROZEN['compact_export.py'], code / 'compact_export.py')
        self.clock = [0.0]

    def tearDown(self):
        shutil.rmtree(self.t)

    def args(self, *extra):
        return MC.parse(['--spec', str(SPEC), '--root', str(self.t / 'coord_root'), '--mil-runs', str(self.runs),
                         '--remote-code', str(self.t / 'remote/code'), '--remote-extract', str(self.remote_x),
                         '--remote-compact', str(self.remote_c), '--poll-s', '1', '--poll-job-s', '0', '--max-wait-h', '1', *extra])

    def coord(self, remote=None, popen=None, *extra):
        def sleep(s): self.clock[0] += s
        return MC.Coordinator(self.args(*extra), remote or FakeRemote(), popen=popen or (lambda *a, **k: None),
                              sleep=sleep, now=lambda: self.clock[0])

    def test_freeze_counts_and_pins(self):
        c = self.coord(); c.phase_freeze()
        st = c.state['phases']['freeze']
        self.assertEqual((len(st['committed']), len(st['remaining'])), (15, 12))
        self.assertFalse(set(st['remaining']) & set(DONE15))
        c2 = self.coord(); c2.state['pins']['fold_mil.py'] = 'x'
        with self.assertRaises(MC.CoordError):
            c2.phase_freeze()                                      # source changed since this root started
        commit(self.runs, st['remaining'][0])
        with self.assertRaises(MC.CoordError):
            MC.Coordinator(MC.parse(self.args_list(root=self.t / 'r2')), FakeRemote()).phase_freeze()   # 16 committed != 15

    def args_list(self, root):
        return ['--spec', str(SPEC), '--root', str(root), '--mil-runs', str(self.runs), '--remote-code', '/x',
                '--remote-extract', '/y', '--remote-compact', '/z']

    def test_wait_alerts_when_extractors_gone(self):
        c = self.coord(FakeRemote(running=(2, 1, 0))); c.phase_freeze()
        with self.assertRaises(MC.CoordError):
            c.phase_wait()
        self.assertTrue((c.coord / 'ALERT.json').exists())
        self.assertEqual(self.clock[0], 2)                         # polled twice before the alert

    def test_wait_complete_and_bounded(self):
        (self.remote_x / 'contract.json').write_text('{"v":4}')
        (self.remote_x / 'COMPLETE.json').write_text(json.dumps({'episodes': 799}))
        c = self.coord(); c.phase_freeze()
        with self.assertRaises(MC.CoordError):
            c.phase_wait()                                         # wrong episode count
        (self.remote_x / 'COMPLETE.json').write_text(json.dumps({'episodes': 800}))
        c = self.coord(); c.phase_freeze(); c.state['phases'].pop('wait', None); c.phase_wait()
        self.assertTrue(c.done('wait'))
        (self.remote_x / 'COMPLETE.json').unlink()
        c3 = MC.Coordinator(MC.parse(self.args_list(self.t / 'r3')[:6] + ['--remote-code', '/x', '--remote-extract', str(self.remote_x), '--remote-compact', '/z',
                                     '--max-wait-h', '0.001', '--poll-s', '10']), FakeRemote(running=(1,)),
                            sleep=lambda s: self.clock.__setitem__(0, self.clock[0] + s), now=lambda: self.clock[0])
        c3.phase_freeze()
        with self.assertRaises(MC.CoordError):
            c3.phase_wait()                                        # timeout -> ALERT

    def make_full_extract(self, n):
        (self.remote_x / 'contract.json').write_text(json.dumps({'with_last_block': True}))
        import numpy as np
        for i in range(n):
            np.savez(self.remote_x / f'{i:03d}.npz', tokmean_l12d3=np.zeros((2, 4), np.float16), actions_model12=np.zeros((2, 16, 12)),
                     success=np.int8(0), n_inferences=np.int32(2), env_steps=np.int32(10), first_success_step=np.int32(-1),
                     max_episode_steps=np.int32(10), env_step_success=np.zeros(10, np.int8), last_block_d0=np.zeros((2, 16, 4), np.float16))
            (self.remote_x / f'{i:03d}.json').write_text(json.dumps({'episode_id': f'e{i}', 'pkl_sha256': 'p', 'npz': f'{i:03d}.npz',
                                                                    'npz_sha256': hashlib.sha256((self.remote_x / f'{i:03d}.npz').read_bytes()).hexdigest()}))
        (self.remote_x / 'COMPLETE.json').write_text(json.dumps({'episodes': n}))

    def test_compact_frozen_sha_reuse_and_pull(self):
        self.make_full_extract(3)
        r = FakeRemote()
        c = self.coord(r, None, '--expect-episodes', '3'); c.phase_freeze(); c.phase_wait(); c.phase_compact()
        self.assertEqual(len(r.calls), 1)
        self.assertEqual(r.calls[0][0], str(self.t / 'remote/code/fold_data/compact_export.py'))
        c.state['phases'].pop('compact'); c.phase_compact()
        self.assertEqual(len(r.calls), 1)                          # already complete remotely -> reused, not rerun
        c.phase_pull()
        self.assertEqual(len(list((c.raw).glob('*.json'))), 4)     # 3 records + contract
        self.assertTrue(c.done('pull'))
        c2 = self.coord(r, None, '--expect-episodes', '3')
        c2.state['phases'].pop('compact')
        (self.t / 'remote/code/fold_data/compact_export.py').write_text('# modified\n')
        with self.assertRaises(MC.CoordError):
            c2.phase_compact()                                     # remote code differs from pinned frozen file

    def test_safe_extract_rejects_links(self):
        tar = self.t / 'bad.tar'
        with tarfile.open(tar, 'w') as tf:
            info = tarfile.TarInfo('link'); info.type = tarfile.SYMTYPE; info.linkname = '/etc/passwd'; tf.addfile(info)
        with self.assertRaises(MC.CoordError):
            MC.safe_extract(tar, self.t / 'out')

    def test_mil_only_remaining_cap_and_failure_stop(self):
        launched, peak = [], [0]
        live = set()

        def popen(cmd, **kw):
            fid = cmd[2].split('--fold ')[-1].split()[0]
            launched.append(fid); live.add(fid); peak[0] = max(peak[0], len(live))
            self.assertEqual(kw['env']['CUDA_VISIBLE_DEVICES'], '')
            self.assertEqual(kw['env']['OMP_NUM_THREADS'], '2')
            fail = fid == self.fail_fold

            def done():
                live.discard(fid)
                if not fail:
                    commit(self.runs, fid)
            return FakeProc(1 if fail else 0, done)

        self.fail_fold = None
        c = self.coord(None, popen, '--jobs', '5'); c.phase_freeze()
        c.state['phases'].update(wait={'done': True}, compact={'done': True}, pull={'done': True}, stage={'done': True})
        c.phase_mil()
        self.assertEqual(sorted(launched), sorted(c.state['phases']['freeze']['remaining']))
        self.assertLessEqual(peak[0], 5)
        self.assertFalse(set(launched) & set(DONE15))
        # failure path: fresh runs dir without the 12, one fold fails -> no new launches after, FAILED.json, then resume
        shutil.rmtree(self.runs)
        for f in DONE15:
            commit(self.runs, f)
        launched.clear()
        c = self.coord(None, popen, '--jobs', '1'); c.state['phases']['mil'] = {}
        self.fail_fold = c.state['phases']['freeze']['remaining'][0]
        with self.assertRaises(MC.CoordError):
            c.phase_mil()
        self.assertEqual(launched, [self.fail_fold])               # stopped launching after the first failure
        self.assertTrue((c.coord / 'FAILED.json').exists())
        with self.assertRaises(MC.CoordError):
            self.coord(None, popen).run()                          # refuses until --retry-failed
        self.fail_fold = None; launched.clear()
        self.coord(None, popen, '--retry-failed').phase_mil()
        self.assertEqual(len(launched), 12)

    def test_committed_runs_must_stay_untouched(self):
        c = self.coord(); c.phase_freeze()
        self.assertEqual(set(c.state['phases']['freeze']['committed_fingerprints']), set(DONE15))
        self.assertIn('mil_continuation.py', c.state['pins']); self.assertIn('remote_compute.sh', c.state['pins'])
        (self.runs / DONE15[0] / 'attn/raw/s0/model.pt').write_bytes(b'tampered')
        with self.assertRaises(MC.CoordError):
            c.check_committed_untouched()

    def test_parse_guards(self):
        with self.assertRaises(SystemExit):
            MC.parse(self.args_list(self.t / 'x')[:-1] + ['relative/compact'])
        with self.assertRaises(SystemExit):
            MC.parse(self.args_list(self.t / 'x') + ['--jobs', '13'])
        self.assertEqual(MC.parse(self.args_list(self.t / 'x')).jobs, 12)


if __name__ == '__main__':
    unittest.main()

"""t49_host_chain on CPU with a tiny fake scene (10 records, 3 folds 8/2): archive gate, adopt/own staging (live-stager
and quiescence gates, partial preservation, space guard), staged-dir contract (PROGRESS.json / wrong shape refused),
latents via the harness wrapper (parity stop), smokes + resume gate, coordinator handoff, freeze and restart.
No GPU, no docker, no remote: frozen stage_episodes.stage runs in-process with LocalFetcher; tool launches are faked."""
import hashlib, json, os, shutil, sys, tempfile, types, unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'trqam'), str(ROOT), str(ROOT / 'fold_data')]
import t49_host_chain as CH       # noqa: E402
import stage_episodes as SE       # noqa: E402

KEY, NAME, INSTR = 'DishwasherRack/out-left|s3', 'srv48_dish_left_s3', 'DishwasherRack/out-left'
N_INF = 2


def sha(b):
    return hashlib.sha256(b).hexdigest()


class Env:
    def __init__(self, t, tokens=49, stage='own'):
        self.t, self.repo, self.arch = t, t / 'repo', t / 'archive'
        rows = [{'plan_id': 'p', 'j': j, 'n': n, 'pkl_sha256': sha(f'{j}{n}'.encode()), 'success': (j + n) % 2} for j in range(5) for n in range(2)]
        folds = [{'fold_id': f'DishwasherRack-out-left_s3_j{tj}', 'instruction': INSTR, 'scene': 3, 'target_j': tj, 'archive_machine': 'worker1',
                  'fit': [r for r in rows if r['j'] != tj], 'eval': [r for r in rows if r['j'] == tj]} for tj in (2, 0, 1)]
        self.spec = {'production_ready': True, 'folds': folds}
        o = self.repo / 'o'; o.mkdir(parents=True)
        (o / 'spec.json').write_text(json.dumps(self.spec))
        self.cfg = {'kind': CH.KIND, 'machine': 'srv48', 'gpus': [3], 'session': 's48', 'python': 'py', 'stage_python': 'python3',
                    'spec': 'o/spec.json', 'source49_root': str(self.arch), 'stage_root': 'o/full49', 'latents_root': 'o/lat49',
                    'out_root': 'out_t49', 'sources': 'o/sources.json', 'smoke_fold': 'DishwasherRack-out-left_s3_j2',
                    'scenes': [{'key': KEY, 'name': NAME, 'stage': stage}], 'expect_records': 10, 'fold_fit': 8, 'fold_eval': 2,
                    'max_wait_h': 1, 'poll_s': 1, 'harness_poll_s': 1}
        (o / 'sources.json').write_text(json.dumps({KEY: {'episodes': f'o/full49/{NAME}', 'latents': f'o/lat49/{NAME}'}}))
        self.write_cfg()
        a = self.arch / NAME; a.mkdir(parents=True)
        contract = {'kind': CH.EXTRACT_KIND, 'scene': KEY, 'expect_records': 10, 'spec_sha256': sha((o / 'spec.json').read_bytes()),
                    'last_block_layer': 15, 'last_block_denoise': 0, 'last_block_tokens': [0, 49], 'last_block_dtype': 'float16',
                    'with_last_block': True}
        cb = json.dumps(contract).encode(); (a / 'contract.json').write_bytes(cb)
        for r in rows:
            eid = SE.episode_id('p', 'worker1', INSTR, 3, r['j'], r['n']); st = SE.stem(eid)
            np.savez(a / f'{st}.npz', last_block_d0=np.zeros((N_INF, tokens, 1536), np.float16), actions_model12=np.zeros((N_INF, 12), np.float32))
            (a / f'{st}.json').write_text(json.dumps({'episode_id': eid, 'pkl_sha256': r['pkl_sha256'], 'success': r['success'], 'npz': f'{st}.npz',
                                                      'npz_sha256': sha((a / f'{st}.npz').read_bytes()), 'n_inferences': N_INF}))
        (a / 'PROGRESS.json').write_text('{"status": "complete"}')          # present in the archive, must never be staged
        self.complete = {'contract': contract, 'contract_sha256': sha(cb), 'episodes': 10}
        self.calls, self.parity_ok, self.resume_ok, self.latents_rc = [], True, True, 0

    def write_cfg(self):
        (self.repo / 'o' / 'cfg.json').write_text(json.dumps(self.cfg))

    def finish_archive(self):
        (self.arch / NAME / 'COMPLETE.json').write_text(json.dumps(self.complete))

    def run(self, cmd, **kw):
        s = ' '.join(cmd); self.calls.append(cmd)
        if 'stage_episodes.py' in s:
            local = self.repo / cmd[cmd.index('--local-dir') + 1]
            SE.stage(self.spec, SE.LocalFetcher(), cmd[cmd.index('--remote-dir') + 1], local, scenes=[KEY], reserve_gib=0, log=lambda *_: None)
        elif 'run_latents_v2.sh' in s:
            ep, lat = self.repo / cmd[cmd.index('--episodes') + 1], self.repo / cmd[cmd.index('--out') + 1]
            lat.mkdir(parents=True, exist_ok=True)
            (lat / 'PARITY_SMOKE_V2.json').write_text(json.dumps({'provenance': {'token_contract': 49}, 'verdict': {'passed': self.parity_ok}}))
            if self.parity_ok:
                (lat / 'contract.json').write_text(json.dumps({'version': 'rollout_latents_v2', 'token_contract': 49,
                                                               'episodes_contract_sha256': sha((ep / 'contract.json').read_bytes())}))
                n = 0
                for jp in ep.glob('*.json'):
                    if jp.name == 'contract.json':
                        continue
                    rec = json.loads(jp.read_text())
                    np.savez(lat / rec['npz'], latents=np.zeros((N_INF, 1024), np.float32))
                    (lat / jp.name).write_text(json.dumps({'episode_id': rec['episode_id'], 'pkl_sha256': rec['pkl_sha256'], 'token_contract': 49,
                                                           'source_npz_sha256': rec['npz_sha256'], 'npz': rec['npz'], 'n': N_INF,
                                                           'npz_sha256': sha((lat / rec['npz']).read_bytes())}))
                    n += 1
                (lat / 'COMPLETE.json').write_text(json.dumps({'episodes': n}))
            return types.SimpleNamespace(returncode=self.latents_rc if self.parity_ok else 1)
        elif 'smoke_v2b.sh' in s:
            i = [k for k, c in enumerate(cmd) if c.endswith('smoke_v2b.sh')][0]
            fold, arm = cmd[i + 1], cmd[i + 2]
            out = self.repo / [c for c in cmd if c.startswith('FOLD27_OUT=')][0].split('=', 1)[1]
            d = out / fold / f'{CH.TAGS[arm]}_smoke'; d.mkdir(parents=True, exist_ok=True)
            (d / 'SMOKE_DONE.json').write_text('{}')
            if arm == 'basic':
                rc = out / fold / 'resume_check_v2'; rc.mkdir(parents=True, exist_ok=True)
                (rc / 'RESUME_CHECK.json').write_text(json.dumps({**{g: self.resume_ok for g in CH.RESUME_GATES}, 'backend': 'gpu'}))
        return types.SimpleNamespace(returncode=0)

    def chain(self, stagers=lambda d: [], free=10 ** 15, **kw):
        return CH.Chain(self.repo / 'o' / 'cfg.json', repo=self.repo, fetcher=SE.LocalFetcher(), run=self.run, sleep=lambda s: None,
                        stagers=stagers, disk=lambda p: types.SimpleNamespace(free=free), poll_s=1, **kw)

    def kinds(self):
        out = []
        for c in self.calls:
            s = ' '.join(c)
            out.append('stage' if 'stage_episodes.py' in s else 'latents' if 'run_latents_v2.sh' in s else
                       f'smoke_{c[[k for k, x in enumerate(c) if x.endswith("smoke_v2b.sh")][0] + 2]}' if 'smoke_v2b.sh' in s else
                       'coord' if 'host_fold_coordinator_v2.py' in s else s)
        return out


class ChainTest(unittest.TestCase):
    def setUp(self):
        self.t = Path(tempfile.mkdtemp()); self.e = Env(self.t)

    def tearDown(self):
        shutil.rmtree(self.t)

    def close(self, ch):
        ch._lock.close()

    def test_config_refusals(self):
        for mut in (lambda c: c['scenes'].append({'key': 'PPCC/bread|s4', 'name': 'srv50_bread_s4', 'stage': 'own'}),
                    lambda c: c.update(gpus=[3, 4]), lambda c: c.update(out_root='outputs/rl2_vla/fold27_trqam_v2_20261102'),
                    lambda c: c.update(sources='o/spec.json'), lambda c: c.update(machine='kanu'),
                    lambda c: c.update(smoke_fold='PPCC-bread_s4_j1'), lambda c: c.update(extra_env=['GPUS=0']),
                    lambda c: c.update(extra_env=['no_equals'])):
            e = Env(Path(tempfile.mkdtemp(dir=self.t))); mut(e.cfg); e.write_cfg()
            with self.assertRaises(CH.ChainError):
                CH.load_config(e.repo / 'o' / 'cfg.json', e.repo)

    def test_full_chain_order_commands_and_restart(self):
        self.e.finish_archive()
        ch = self.e.chain(); st = ch.run(); self.close(ch)
        self.assertEqual(st['status'], 'done')
        self.assertEqual(self.e.kinds(), ['stage', 'latents', 'smoke_basic', 'smoke_pbrs', 'smoke_mix', 'coord'])
        lat = self.e.calls[1]
        self.assertIn('wait_launch.py', lat[1]); self.assertIn('GPUS=3', lat); self.assertIn('HARNESS_MACHINE=srv48', lat)
        i = lat.index('--token-contract'); self.assertEqual(lat[i + 1], '49')
        self.assertEqual(lat[lat.index('--expect-records') + 1], '10')
        self.assertTrue(all('FOLD27_OUT=out_t49' in c for c in self.e.calls[2:5]))
        co = self.e.calls[5]
        self.assertEqual(co[co.index('--sources') + 1], 'o/sources.json'); self.assertEqual(co[co.index('--gpus') + 1], '3')
        self.assertEqual(co[co.index('--smoke-fold') + 1], 'DishwasherRack-out-left_s3_j2'); self.assertEqual(co[co.index('--out') + 1], 'out_t49')
        staged = self.e.repo / 'o/full49' / NAME
        self.assertFalse((staged / 'PROGRESS.json').exists())
        self.assertTrue((self.e.arch / NAME / 'PROGRESS.json').exists())          # source untouched
        self.assertEqual(st['scenes'][NAME], {'archive_contract_sha256': self.e.complete['contract_sha256'], 'stage': 'staged', 'latents': 'converted'})
        # restart: everything verified/adopted, only the (idempotent) coordinator is re-invoked
        self.e.calls.clear()
        ch = self.e.chain(); st = ch.run(); self.close(ch)
        self.assertEqual(self.e.kinds(), ['coord']); self.assertEqual(st['scenes'][NAME]['stage'], 'adopted')
        # frozen inputs changed -> refuse
        self.e.cfg['poll_s'] = 2; self.e.write_cfg()
        with self.assertRaises(CH.ChainError):
            self.e.chain().run()

    def test_archive_waits_until_complete_then_strict_contract(self):
        ch = self.e.chain(); s = ch.cfg['scenes'][0]; ch.sd.mkdir(parents=True)
        with self.assertRaises(CH.Wait):
            ch.archive_gate(s)
        self.e.complete['contract']['last_block_tokens'] = [33, 49]; self.e.finish_archive()
        with self.assertRaises(CH.ChainError):
            ch.archive_gate(s)
        self.assertTrue((ch.sd / 'CHAIN_FAILED.json').exists())
        e2 = Env(Path(tempfile.mkdtemp(dir=self.t))); e2.cfg['extractor49_sha256'] = 'x' * 64; e2.write_cfg(); e2.finish_archive()
        ch2 = e2.chain(); ch2.sd.mkdir(parents=True)
        with self.assertRaisesRegex(CH.ChainError, 'extractor49_sha256'):
            ch2.archive_gate(ch2.cfg['scenes'][0])

    def test_staged_dir_contract_progress_json_and_shape(self):
        self.e.finish_archive()
        ch = self.e.chain(); s = ch.cfg['scenes'][0]; ch.sd.mkdir(parents=True)
        cb = ch.archive_gate(s); ch.stage(s, cb)
        staged = self.e.repo / 'o/full49' / NAME
        self.assertEqual(ch.verify_staged(s, cb)[0], 'ok')
        (staged / 'PROGRESS.json').write_text('{}')
        st, pr = ch.verify_staged(s, cb)
        self.assertEqual(st, 'corrupt'); self.assertIn('PROGRESS.json', ' '.join(pr))
        e3 = Env(Path(tempfile.mkdtemp(dir=self.t)), tokens=16); e3.finish_archive()
        ch3 = e3.chain(); s3 = ch3.cfg['scenes'][0]; ch3.sd.mkdir(parents=True)
        with self.assertRaisesRegex(CH.ChainError, 'last_block_d0'):
            ch3.stage(s3, ch3.archive_gate(s3))

    def test_in_flight_npz_before_json_commit_is_incomplete_not_corrupt(self):
        self.e.finish_archive()
        ch = self.e.chain(); s = ch.cfg['scenes'][0]; ch.sd.mkdir(parents=True)
        cb = ch.archive_gate(s)
        staged = self.e.repo / 'o/full49' / NAME; staged.mkdir(parents=True); (staged / 'contract.json').write_bytes(cb)
        st0 = SE.stem(SE.episode_id('p', 'worker1', INSTR, 3, 0, 0))
        shutil.copyfile(self.e.arch / NAME / f'{st0}.npz', staged / f'{st0}.npz')
        self.assertEqual(ch.verify_staged(s, cb)[0], 'incomplete')
        (staged / 'stray.npz').write_bytes(b'x')
        self.assertEqual(ch.verify_staged(s, cb)[0], 'corrupt')

    def test_fold_target_leak_refused(self):
        f = self.e.spec['folds'][0]; f['fit'].append(f['eval'][0])
        (self.e.repo / 'o/spec.json').write_text(json.dumps(self.e.spec))
        self.e.complete['contract']['spec_sha256'] = sha((self.e.repo / 'o/spec.json').read_bytes())
        c = self.e.complete['contract']; cb = json.dumps(c).encode()
        (self.e.arch / NAME / 'contract.json').write_bytes(cb); self.e.complete['contract_sha256'] = sha(cb); self.e.finish_archive()
        ch = self.e.chain(); s = ch.cfg['scenes'][0]; ch.sd.mkdir(parents=True)
        with self.assertRaisesRegex(CH.ChainError, 'fit'):
            ch.stage(s, ch.archive_gate(s))

    def test_adopt_never_duplicates_and_needs_quiescence(self):
        e = Env(Path(tempfile.mkdtemp(dir=self.t)), stage='adopt'); e.finish_archive()
        ch = e.chain(stagers=lambda d: [4105042]); s = ch.cfg['scenes'][0]; ch.sd.mkdir(parents=True)
        cb = ch.archive_gate(s)
        with self.assertRaisesRegex(CH.Wait, 'already running'):
            ch.stage(s, cb)
        ch.stagers = lambda d: []
        with self.assertRaisesRegex(CH.Wait, 'quiescence'):
            ch.stage(s, cb)
        self.assertEqual(e.calls, [])
        (ch.sd / f'QUIESCENT_{NAME}').write_text('main confirmed')
        self.assertEqual(ch.stage(s, cb), 'staged'); self.assertEqual(e.kinds(), ['stage'])
        e.calls.clear()
        self.assertEqual(ch.stage(s, cb), 'adopted'); self.assertEqual(e.calls, [])   # completed by another owner -> adopt

    def test_own_partial_preserved_and_space_guard(self):
        self.e.finish_archive()
        ch = self.e.chain(free=10); s = ch.cfg['scenes'][0]; ch.sd.mkdir(parents=True)
        cb = ch.archive_gate(s)
        with self.assertRaisesRegex(CH.ChainError, 'space guard'):
            ch.stage(s, cb)
        self.assertEqual(self.e.calls, [])
        (ch.sd / 'CHAIN_FAILED.json').unlink()                     # test-only reset of the stop marker
        staged = self.e.repo / 'o/full49' / NAME; staged.mkdir(parents=True)
        (staged / 'contract.json').write_bytes(cb); (staged / '.abc.npz.part').write_bytes(b'partial-bytes')
        ch.disk = lambda p: types.SimpleNamespace(free=10 ** 15)
        self.assertEqual(ch.stage(s, cb), 'staged')
        kept = list((self.e.repo / 'o/full49/.preserved_partials' / NAME).rglob('.abc.npz.part'))
        self.assertEqual(len(kept), 1); self.assertEqual(kept[0].read_bytes(), b'partial-bytes')

    def test_parity_failure_stops_before_smokes(self):
        self.e.finish_archive(); self.e.parity_ok = False
        ch = self.e.chain()
        with self.assertRaisesRegex(CH.ChainError, 'parity'):
            ch.run()
        self.assertEqual(self.e.kinds(), ['stage', 'latents'])
        self.close(ch)
        with self.assertRaisesRegex(CH.ChainError, 'retry-failed'):
            self.e.chain().run()

    def test_resume_gate_failure_stops_before_training(self):
        self.e.finish_archive(); self.e.resume_ok = False
        ch = self.e.chain()
        with self.assertRaisesRegex(CH.ChainError, 'exact-resume'):
            ch.run()
        self.assertEqual(self.e.kinds(), ['stage', 'latents', 'smoke_basic'])

    def test_harness_deadline_rc75(self):
        self.e.finish_archive(); self.e.latents_rc = 75
        with self.assertRaisesRegex(CH.ChainError, 'BLOCKED'):
            self.e.chain().run()


class LiveStagersTest(unittest.TestCase):
    def test_proc_scan(self):
        t = Path(tempfile.mkdtemp())
        try:
            proc, cwd = t / 'proc', t / 'deploy'; (cwd / 'o/full49/X').mkdir(parents=True)
            def mk(pid, argv, link=True):
                d = proc / str(pid); d.mkdir(parents=True)
                (d / 'cmdline').write_bytes(b'\0'.join(a.encode() for a in argv) + b'\0')
                if link:
                    os.symlink(cwd, d / 'cwd')
            mk(11, ['python3', 'scripts/x/stage_episodes.py', '--spec', 's', '--local-dir', 'o/full49/X'])
            mk(12, ['python3', 'scripts/x/stage_episodes.py', '--local-dir', 'o/full49/Y'])
            mk(13, ['bash', '-c', 'echo stage_episodes.py --local-dir o/full49/X'])
            mk(14, ['python3', '/abs/stage_episodes.py', '--local-dir', str(cwd / 'o/full49/X')], link=False)
            mk(15, ['python3', 'stage_episodes.py', '--local-dir', 'o/full49/X'], link=False)      # unreadable cwd -> conservative
            self.assertEqual(sorted(CH.live_stagers(cwd / 'o/full49/X', proc_root=proc, self_pid=1)), [11, 14, 15])
        finally:
            shutil.rmtree(t)


if __name__ == '__main__':
    unittest.main()

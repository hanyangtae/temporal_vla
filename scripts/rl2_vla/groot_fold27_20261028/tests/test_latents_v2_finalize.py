"""latents_v2_audit / rollout_latents_v2.convert_all / finalize_latents_v2 / bread_t49_readiness on CPU with synthetic
episode + latent dirs (no model, no GPU). Regression for the srv50 61a66541 run: DIAG_V2.json counted as a record."""
import hashlib, json, shutil, sys, tempfile, unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'trqam'))
import latents_v2_audit as AU        # noqa: E402
import rollout_latents_v2 as R2      # noqa: E402
import finalize_latents_v2 as FZ     # noqa: E402
import bread_t49_readiness as BR     # noqa: E402

ROW = {'episode': 0, 'block_shape': [1, 49, 1536], 'decoder_input_shape': [1, 49, 1536], 'exact_49': True, 'exact_16': False}


def sha_b(b):
    return hashlib.sha256(b).hexdigest()


class Dirs:
    """episodes dir with k records (49-token f16 blocks) + an out dir converted by convert_all."""

    def __init__(self, t, k=3, n_inf=4, eids=None):
        self.ep, self.out = t / 'episodes', t / 'out'
        self.ep.mkdir(); self.out.mkdir()
        (self.ep / 'contract.json').write_text(json.dumps({'kind': 'extract_scene', 'with_last_block': True}))
        self.eids = eids or [f'p/worker2/PPCC/bread/s4/j{i % 5}/n{i}' for i in range(k)]
        for i, e in enumerate(self.eids):
            np.savez(self.ep / f'e{i}.npz', last_block_d0=np.full((n_inf, 49, 1536), i, np.float16))
            (self.ep / f'e{i}.json').write_text(json.dumps({'episode_id': e, 'pkl_sha256': f'pkl{i}', 'npz': f'e{i}.npz',
                                                            'npz_sha256': AU.sha(self.ep / f'e{i}.npz'), 'n_inferences': n_inf}))
        self.calls = []

    def convert(self):
        def one(lb):
            self.calls.append(lb.shape)
            return np.full((len(lb), 1024), float(lb[0, 0, 0]), np.float32)
        return R2.convert_all(AU.expected_records(self.ep), self.ep, self.out, 49, one, gate=lambda rec: None, log=lambda s: None)


class AuditTest(unittest.TestCase):
    def setUp(self):
        self.t = Path(tempfile.mkdtemp()); self.d = Dirs(self.t); self.d.convert()

    def tearDown(self):
        shutil.rmtree(self.t)

    def test_diag_and_known_files_do_not_count(self):
        for n in ('DIAG_V2.json', 'PARITY_SMOKE_V2.json', 'contract.json'):
            (self.d.out / n).write_text('{}')
        r = AU.audit(self.d.ep, self.d.out, 49, expect=3)
        self.assertTrue(r['ok'], r); self.assertEqual(r['verified_records'], 3)
        self.assertIn('DIAG_V2.json', r['non_record_present'])

    def test_missing_unexpected_duplicate_unreferenced(self):
        (self.d.out / 'e1.json').rename(self.d.out / 'e1.json.bak')     # (test fixture only)
        r = AU.audit(self.d.ep, self.d.out, 49, expect=3)
        self.assertFalse(r['ok']); self.assertEqual(r['problems']['missing'], ['e1.json'])
        self.assertIn('e1.npz', r['problems']['unreferenced_npz'])
        (self.d.out / 'e1.json.bak').rename(self.d.out / 'e1.json')
        (self.d.out / 'stray.json').write_text('{}')
        r = AU.audit(self.d.ep, self.d.out, 49, expect=3)
        self.assertEqual(r['problems']['unexpected_json'], ['stray.json']); self.assertFalse(r['ok'])
        (self.d.out / 'stray.json').unlink()
        j = json.loads((self.d.out / 'e2.json').read_text()); j['episode_id'] = self.d.eids[0]
        (self.d.out / 'e2.json').write_text(json.dumps(j))
        r = AU.audit(self.d.ep, self.d.out, 49, expect=3)
        self.assertFalse(r['ok']); self.assertTrue(r['problems']['duplicate_episode_id'])
        self.assertIn('episode_id', r['problems']['mismatch'][0]['fields'])

    def test_contract_expect_and_tmp(self):
        self.assertFalse(AU.audit(self.d.ep, self.d.out, 16, expect=3)['ok'])          # token contract mismatch
        self.assertFalse(AU.audit(self.d.ep, self.d.out, 49, expect=200)['ok'])
        (self.d.out / 'x.tmp.npz').write_bytes(b'')
        r = AU.audit(self.d.ep, self.d.out, 49, expect=3)
        self.assertTrue(r['ok']); self.assertEqual(r['problems']['tmp_leftover'], ['x.tmp.npz'])

    def test_npz_bytes_or_array_change_detected(self):
        # e0's committed latents are all 0.0 (source block filled with 0), so corrupt with a DIFFERENT value
        before = AU.sha(self.d.out / 'e0.npz')
        np.savez(self.d.out / 'e0.npz', latents=np.ones((4, 1024), np.float32))
        self.assertNotEqual(AU.sha(self.d.out / 'e0.npz'), before)                      # the corruption is real
        r = AU.audit(self.d.ep, self.d.out, 49)
        self.assertFalse(r['ok'])
        self.assertEqual(r['problems']['mismatch'][0]['record'], 'e0.json')
        self.assertIn('npz sha', r['problems']['mismatch'][0]['fields'])
        # record re-pointed at a wrong-shaped array with a consistent sha: the array check itself must reject it
        np.savez(self.d.out / 'e0.npz', latents=np.zeros((3, 1024), np.float32))
        j = json.loads((self.d.out / 'e0.json').read_text()); j['npz_sha256'] = AU.sha(self.d.out / 'e0.npz')
        (self.d.out / 'e0.json').write_text(json.dumps(j))
        r = AU.audit(self.d.ep, self.d.out, 49)
        self.assertFalse(r['ok'])
        self.assertTrue(any(f.startswith('latents') for f in r['problems']['mismatch'][0]['fields']), r['problems'])


class ConvertAllTest(unittest.TestCase):
    def setUp(self):
        self.t = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.t)

    def test_resume_never_recomputes_committed(self):
        d = Dirs(self.t, k=3)
        first = d.convert()
        self.assertEqual(first, {'computed': 3, 'skipped': 0})
        before = {p.name: p.read_bytes() for p in d.out.iterdir()}
        (d.out / 'e1.json').unlink(); (d.out / 'e1.npz').unlink()                      # simulate an interrupted run
        d.calls.clear()
        second = d.convert()
        self.assertEqual(second, {'computed': 1, 'skipped': 2}); self.assertEqual(len(d.calls), 1)
        self.assertEqual((d.out / 'e0.json').read_bytes(), before['e0.json'])
        self.assertEqual((d.out / 'e0.npz').read_bytes(), before['e0.npz'])

    def test_committed_mismatch_and_wrong_shape_refuse(self):
        d = Dirs(self.t, k=2); d.convert()
        np.savez(d.out / 'e0.npz', latents=np.ones((4, 1024), np.float32))
        with self.assertRaises(SystemExit):
            d.convert()
        t2 = Path(tempfile.mkdtemp())
        try:
            d2 = Dirs(t2, k=1)
            np.savez(d2.ep / 'e0.npz', last_block_d0=np.zeros((4, 16, 1536), np.float16))
            j = json.loads((d2.ep / 'e0.json').read_text()); j['npz_sha256'] = AU.sha(d2.ep / 'e0.npz')
            (d2.ep / 'e0.json').write_text(json.dumps(j))
            with self.assertRaises(SystemExit):
                d2.convert()                                                        # 16-token data under the 49 contract
        finally:
            shutil.rmtree(t2)


class FinalizeTest(unittest.TestCase):
    def setUp(self):
        self.t = Path(tempfile.mkdtemp())
        self.d = Dirs(self.t, k=3); self.d.convert()
        self.src = self.t / 'converter_61a66541.py'; self.src.write_bytes(b'# converter bytes at 61a66541\n')
        prov = {'version': R2.VERSION, 'token_contract': 49, 'script_sha256': AU.sha(self.src), 'kernel': {'allow_tf32_matmul': False}}
        smoke = {'provenance': prov, 'rows': [ROW, dict(ROW, episode=1)], 'verdict': {'passed': True}}
        (self.d.out / 'PARITY_SMOKE_V2.json').write_text(json.dumps(smoke, indent=1))
        self.contract = {**prov, 'parity_smoke_v2_sha256': AU.sha(self.d.out / 'PARITY_SMOKE_V2.json'),
                         'episodes_contract_sha256': AU.sha(self.d.ep / 'contract.json'), 'cross_device_validated_sha256': None}
        (self.d.out / 'contract.json').write_text(json.dumps(self.contract, indent=1))
        (self.d.out / 'DIAG_V2.json').write_text(json.dumps({'diag': {}}))                # the file that broke the count

    def tearDown(self):
        shutil.rmtree(self.t)

    def snapshot(self):
        return {p.name: sha_b(p.read_bytes()) for p in self.d.out.iterdir()}

    def fin(self, write=True, src=None):
        return FZ.finalize(self.d.ep, self.d.out, 49, 3, src or self.src, '61a66541', write=write)

    def test_completes_in_place_without_touching_existing_bytes(self):
        before = self.snapshot()
        self.assertTrue(self.fin(write=False)['would_complete']); self.assertEqual(self.snapshot(), before)   # report-only writes nothing
        r = self.fin()
        self.assertEqual(r['refused'], []); self.assertEqual(r['written'], ['COMPLETE_AUDIT_V2.json', 'COMPLETE.json'])
        after = self.snapshot()
        self.assertEqual({k: v for k, v in after.items() if k in before}, before)          # nothing existing changed
        c = json.loads((self.d.out / 'COMPLETE.json').read_text())
        self.assertEqual(c['episodes'], 3); self.assertEqual(c['contract'], self.contract)
        self.assertEqual(c['completed_by']['converter_script_sha256'], AU.sha(self.src))
        self.assertEqual(c['completed_by']['converter_commit'], '61a66541')
        self.assertEqual(c['audit_report_sha256'], AU.sha(self.d.out / 'COMPLETE_AUDIT_V2.json'))
        self.assertTrue(self.fin()['refused'])                                             # never overwritten

    def test_refusals(self):
        bad = self.t / 'other.py'; bad.write_bytes(b'# fixed converter\n')
        self.assertTrue(any('converter-source' in w for w in self.fin(write=False, src=bad)['refused']))
        p = self.d.out / 'PARITY_SMOKE_V2.json'; keep = p.read_bytes()
        p.write_bytes(keep.replace(b'"exact_49": true', b'"exact_49": false', 1))
        r = self.fin(write=False)
        self.assertTrue(any('bytes differ' in w for w in r['refused'])); self.assertTrue(any('smoke rows' in w for w in r['refused']))
        p.write_bytes(keep)
        (self.d.out / 'e2.json').rename(self.d.out / 'e2.hold')
        self.assertTrue(any(w.startswith('audit') for w in self.fin(write=False)['refused']))
        (self.d.out / 'e2.hold').rename(self.d.out / 'e2.json')
        self.assertEqual(self.fin(write=False)['refused'], [])
        self.assertFalse((self.d.out / 'COMPLETE.json').exists())


class BreadReadinessTest(unittest.TestCase):
    def setUp(self):
        self.t = Path(tempfile.mkdtemp())
        folds, eids = [], {}
        for tj in (1, 3, 4):
            js = [j for j in range(5) if j != tj]
            fit = [{'plan_id': 'p', 'j': j, 'n': n, 'pkl_sha256': f'k{j}_{n}'} for j in js for n in range(40)]
            ev = [{'plan_id': 'p', 'j': tj, 'n': n, 'pkl_sha256': f'k{tj}_{n}'} for n in range(40)]
            folds.append({'fold_id': f'PPCC-bread_s4_j{tj}', 'instruction': 'PPCC/bread', 'scene': 4, 'archive_machine': 'worker2',
                          'target_j': tj, 'fit': fit, 'eval': ev})
        folds.append({'fold_id': 'PPCC-jug_s1_j0', 'instruction': 'PPCC/jug', 'scene': 1, 'archive_machine': 'worker2',
                      'target_j': 0, 'fit': [], 'eval': []})
        self.spec = self.t / 'spec.json'; self.spec.write_text(json.dumps({'folds': folds}))
        ep, lat = self.t / 'compact', self.t / 'lat'; ep.mkdir(); lat.mkdir()
        (ep / 'contract.json').write_text(json.dumps({'kind': 'fold27_compact_v1', 'source_contract_sha256': 'S'}))
        (lat / 'contract.json').write_text(json.dumps({'version': 'rollout_latents_v2', 'token_contract': 49, 'episodes_contract_sha256': 'S'}))
        (lat / 'COMPLETE.json').write_text(json.dumps({'episodes': 200}))
        for j in range(5):
            for n in range(40):
                e = f'p/worker2/PPCC/bread/s4/j{j}/n{n}'
                (ep / f'j{j}n{n}.json').write_text(json.dumps({'episode_id': e, 'pkl_sha256': f'k{j}_{n}'}))
                (lat / f'j{j}n{n}.json').write_text(json.dumps({'episode_id': e, 'pkl_sha256': f'k{j}_{n}', 'token_contract': 49}))
        self.ep, self.lat = ep, lat
        self.src = self.t / 'src.json'
        self.src.write_text(json.dumps({'PPCC/bread|s4': {'episodes': str(ep), 'latents': str(lat)}}))

    def tearDown(self):
        shutil.rmtree(self.t)

    def test_ready_and_heldout(self):
        r = BR.check(self.spec, self.src, repo=Path('/'))
        self.assertTrue(r['ready'], r['problems']); self.assertEqual(r['trainings'], 9)
        self.assertEqual(sorted(f['fold'] for f in r['folds']), ['PPCC-bread_s4_j1', 'PPCC-bread_s4_j3', 'PPCC-bread_s4_j4'])

    def test_failures(self):
        s = json.loads(self.spec.read_text())
        s['folds'][0]['fit'][0]['j'] = 1                                   # a target-j episode leaks into fit
        self.spec.write_text(json.dumps(s))
        self.assertFalse(BR.check(self.spec, self.src, repo=Path('/'))['ready'])
        s['folds'][0]['fit'][0]['j'] = 0; self.spec.write_text(json.dumps(s))
        (self.lat / 'j0n5.json').unlink()
        r = BR.check(self.spec, self.src, repo=Path('/'))
        self.assertFalse(r['ready']); self.assertTrue(any('latent False' in p for p in r['problems']))
        src = json.loads(self.src.read_text()); src['PPCC/jug|s1'] = src['PPCC/bread|s4']
        self.src.write_text(json.dumps(src))
        self.assertTrue(any('only' in p for p in BR.check(self.spec, self.src, repo=Path('/'))['problems']))


if __name__ == '__main__':
    unittest.main()

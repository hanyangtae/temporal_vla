"""aggregate_eval_report on synthetic eval roots (real run_eval_arm.load_progress + verify_fold.audit_decisions): per
host x policy stats, archived-baseline pairing, rescue/damage, common subset, failed audit rows excluded AND listed,
load errors recorded, smoke skipped, partial arms counted row-wise, inputs never modified, outputs exclusive."""
import json, shutil, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'eval/runtime'))
import aggregate_eval_report as A      # noqa: E402
import run_eval_arm as RA              # noqa: E402

ARMS = ('mil_qam65', 'mil_trqam_basic')


def dec(i, fired):
    return {'request': i, 'mode': 'gated', 'mil': {'score': 0.9 if fired else 0.1, 'fired': fired}, 'gate': fired,
            'qam_delta_max_normalized': 0.3 if fired else 0, 'noise_sha256_composed': 'x' if fired else None,
            'noise_sha256_first_pass': 'x', 'qam_steps': [0, 1, 2, 3], 'action_finite': True, 'off_identity_verified': True,
            'rng_restored_equal': True}


def make_fold(root, fid, base, results, smoke=False, upto=None):
    fd = root / fid; fd.mkdir(parents=True)
    rows = [{'n': n, 'j': 2, 'env_seed': 100 + n, 'inference_seed': 1000 + n, 'plan_id': 'p', 'base_success': b} for n, b in enumerate(base)]
    m = {'fold_id': fid, 'smoke': smoke, 'host': 'kanu', 'target_j': 2, 'instruction': 'X', 'scene': 1, 'rows': rows,
         'mil': {'model': f'mil/{fid}/model.pt', 'sha256': 'mil' + fid, 'fold_contract': {'path': 'fc.json', 'sha256': 'fc' + fid}},
         'policy': {'gate': 'MIL > 0.5'},
         'arms': {a: {'mode': 'gated', 'actor': f'actors/{a}', 'actor_sha256': f'{a}-{fid}', 'actor_json_sha256': 'j' + a,
                      'provenance': {'kind': a, 'fold_id': fid}} for a in ARMS}}   # same pin keys as build_manifest
    (fd / 'manifest.json').write_text(json.dumps(m))
    for a in ARMS:
        out = fd / a; out.mkdir()
        prog, decs, req = [], [], 0
        for n, s in enumerate(results[a][:upto]):
            md = out / 'rows' / f'n{n:02d}' / 'a0' / 'grid' / a; md.mkdir(parents=True)
            (md / 'meta.json').write_text(json.dumps({'episode_success': s}))
            fired = n % 2 == 0
            decs += [dec(req + 1, False), dec(req + 2, fired)]
            prog.append({'fold_id': fid, 'arm': a, 'n': n, 'env_seed': 100 + n, 'inference_seed': 1000 + n, 'attempt': 'a0',
                         'meta': str((md / 'meta.json').relative_to(out)), 'files_sha256': RA.row_files_sha(md / 'meta.json'),
                         'success': s, 'composed': int(fired), 'fired': int(fired), 'http_sends': 2, 'requests': [req + 1, req + 2],
                         'first_fire_t': 1 if fired else None, 'decisions_log': 'd.jsonl'})
            req += 2
        (out / 'progress.jsonl').write_text('\n'.join(json.dumps(p) for p in prog) + '\n')
        (out / 'd.jsonl').write_text('\n'.join(json.dumps(d) for d in decs) + '\n')
    return fd


class AggTest(unittest.TestCase):
    def setUp(self):
        self.t = Path(tempfile.mkdtemp())
        self.r1 = self.t / 'kanu_root'; self.r2 = self.t / 'srv48_root'
        make_fold(self.r1, 'F1', [0, 1, 0, 1], {'mil_qam65': [1, 1, 0, 0], 'mil_trqam_basic': [0, 1, 1, 1]})
        make_fold(self.r2, 'F2', [1, 0], {'mil_qam65': [1, 1], 'mil_trqam_basic': [0, 0]}, upto=None)
        make_fold(self.r2, 'S1', [1], {'mil_qam65': [1], 'mil_trqam_basic': [1]}, smoke=True)
        self.before = {p: p.read_bytes() for p in self.t.rglob('*') if p.is_file()}

    def tearDown(self):
        for p, b in self.before.items():
            assert p.read_bytes() == b, f'modified {p}'
        shutil.rmtree(self.t)

    def rep(self, policies=ARMS):
        s1, s2 = A.snapshot(self.r1, 'kanu'), A.snapshot(self.r2, 'srv48')
        return A.merge([s1, s2], {'kanu': 240, 'srv48': 360}, 1080, policies), s2

    def row(self, rep, host, pol):
        return next(r for r in rep['per_host_policy'] if r['host'] == host and r['policy'] == pol)

    def test_stats_pairing_and_common_subset(self):
        rep, s2 = self.rep()
        k = self.row(rep, 'kanu', 'mil_qam65')
        self.assertEqual((k['N'], k['success'], k['base_success_archived'], k['rescue'], k['damage'], k['plannedN']), (4, 2, 2, 1, 1, 240))
        self.assertEqual((k['fired_inferences'], k['inferences'], k['composed_rows']), (2, 8, 2))
        self.assertEqual((k['base_success_N'], k['base_fail_N'], k['rescue_detected_den'], k['damage_detected_den']), (2, 2, 2, 0))
        self.assertEqual(rep['common_subset']['status'], 'complete over all 2 policies')
        a = next(x for x in rep['arms'] if x['host'] == 'kanu' and x['policy'] == 'mil_qam65')
        self.assertEqual((a['actor_sha256'], a['mil_sha256'], a['mil_fold_contract_sha256'], a['actor_kind'], a['pin_error']),
                         ('mil_qam65-F1', 'milF1', 'fcF1', 'mil_qam65', None))
        allq = self.row(rep, 'ALL', 'mil_qam65')
        self.assertEqual((allq['N'], allq['success'], allq['plannedN']), (6, 4, 1080)); self.assertAlmostEqual(allq['coverage'], 6 / 1080)
        self.assertEqual(rep['common_subset']['n_keys'], 6)
        self.assertEqual(s2['skipped_smoke'], [str(self.r2 / 'S1' / 'manifest.json')])

    def test_failed_audit_rows_excluded_and_listed(self):
        p = self.r1 / 'F1' / 'mil_trqam_basic' / 'd.jsonl'
        ds = [json.loads(x) for x in p.read_text().splitlines()]
        ds[1]['gate'] = not ds[1]['gate']                                            # n0 gate rule broken
        p.write_text('\n'.join(json.dumps(d) for d in ds) + '\n'); self.before[p] = p.read_bytes()
        rep, _ = self.rep()
        b = self.row(rep, 'kanu', 'mil_trqam_basic')
        self.assertEqual((b['N'], b['failed_audit_rows']), (3, 1))
        self.assertTrue(any('n0' in e.get('audit_error', '') for e in rep['errors']))
        self.assertEqual(rep['common_subset']['n_keys'], 5)                          # n0 of F1 no longer common

    def test_load_error_recorded_not_hidden(self):
        # previous version rewrote an arbitrary meta with possibly IDENTICAL bytes (n2/n3 already 0) -> no mismatch, flaky
        meta = self.r1 / 'F1' / 'mil_qam65' / 'rows' / 'n00' / 'a0' / 'grid' / 'mil_qam65' / 'meta.json'
        old = meta.read_bytes()
        meta.write_text(json.dumps({'episode_success': 0, 'tampered': True})); self.before[meta] = meta.read_bytes()
        self.assertNotEqual(old, meta.read_bytes())
        rep, _ = self.rep()
        self.assertEqual(self.row(rep, 'kanu', 'mil_qam65')['N'], 0)
        self.assertTrue(any('load_error' in e and 'F1' == e['fold_id'] for e in rep['errors']))

    def test_missing_policy_makes_common_subset_incomplete(self):
        rep, _ = self.rep(policies=None)                                  # default = the 4 production arms
        self.assertEqual(rep['policies'], sorted(A.C.PRODUCTION_ARMS))
        cs = rep['common_subset']
        self.assertEqual((cs['n_keys'], cs['missing_policies']), (0, ['mil_trqam_mix', 'mil_trqam_pbrs']))
        self.assertTrue(cs['status'].startswith('INCOMPLETE'))
        self.assertEqual(self.row(rep, 'ALL', 'mil_trqam_pbrs')['N'], 0)  # still listed with N=0

    def test_duplicates_refused(self):
        s1 = A.snapshot(self.r1, 'kanu')
        with self.assertRaises(A.MergeError):
            A.merge([s1, s1], {}, None, ARMS)                            # same snapshot twice
        r3 = self.t / 'copy_root'; shutil.copytree(self.r1, r3)
        self.before.update({p: p.read_bytes() for p in r3.rglob('*') if p.is_file()})
        with self.assertRaises(A.MergeError):
            A.merge([s1, A.snapshot(r3, 'srv50')], {}, None, ARMS)        # same fold rows under two hosts

    def test_missing_pin_excluded_and_listed(self):
        mp = self.r1 / 'F1' / 'manifest.json'
        m = json.loads(mp.read_text()); del m['arms']['mil_qam65']['actor_sha256']
        mp.write_text(json.dumps(m)); self.before[mp] = mp.read_bytes()
        rep, _ = self.rep()
        k = self.row(rep, 'kanu', 'mil_qam65')
        self.assertEqual((k['N'], k['failed_audit_rows']), (0, 4))
        self.assertTrue(any('actor_sha256' in (e.get('pin_error') or '') for e in rep['errors']))

    def test_cli_outputs_new_dir_only(self):
        s1, s2, out = self.t / 's1.json', self.t / 's2.json', self.t / 'report'
        self.assertEqual(A.main(['snapshot', '--root', str(self.r1), '--host', 'kanu', '--out', str(s1)]), 0)
        self.assertEqual(A.main(['snapshot', '--root', str(self.r2), '--host', 'srv48', '--out', str(s2)]), 0)
        with self.assertRaises(FileExistsError):
            A.main(['snapshot', '--root', str(self.r1), '--host', 'kanu', '--out', str(s1)])
        self.assertEqual(A.main(['merge', '--snapshot', str(s1), str(s2), '--planned', 'kanu=240', 'srv48=360', '--planned-total', '1080',
                                 '--policies', *ARMS, '--out-dir', str(out)]), 0)
        for f in ('report.json', 'per_policy.csv', 'per_episode.csv', 'report.md'):
            self.assertTrue((out / f).exists())
        self.assertIn('Limitations', (out / 'report.md').read_text())
        with self.assertRaises(FileExistsError):
            A.main(['merge', '--snapshot', str(s1), '--out-dir', str(out)])
        with self.assertRaises(A.MergeError):
            A.main(['merge', '--snapshot', str(s1), str(s1), '--out-dir', str(self.t / 'report2')])


if __name__ == '__main__':
    unittest.main()

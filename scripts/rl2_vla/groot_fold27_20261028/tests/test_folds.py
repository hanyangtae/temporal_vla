"""CPU tests for folds.py. Run: python3 -m unittest scripts/rl2_vla/groot_fold27_20261028/tests/test_folds.py"""
import copy, sys, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import folds as F


class FoldSpecTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec, cls.sources = F.build(F.R)

    def test_counts(self):
        s = self.spec
        self.assertEqual(len(s['folds']), 27)
        self.assertEqual(s['unique_sources'], 1800)
        self.assertEqual(sum(len(f['fit']) for f in s['folds']), 27 * 160)
        self.assertEqual(sum(len(f['eval']) for f in s['folds']), 27 * 40)

    def test_no_leak_no_dup(self):
        for f in self.spec['folds']:
            fit = {(r['j'], r['n']) for r in f['fit']}; ev = {(r['j'], r['n']) for r in f['eval']}
            self.assertEqual(len(fit), 160); self.assertEqual(len(ev), 40)
            self.assertFalse(fit & ev)
            self.assertNotIn(f['target_j'], {j for j, _ in fit})

    def test_source_plans(self):
        sp = self.spec['source_plans_n0_9']
        self.assertEqual(sp['DishwasherRack/out-left|s3'], '4ab360df2b71')
        self.assertEqual(sp['PPCC/bread|s4'], '4fa6496cd684')
        self.assertEqual(sp['PPCC/marshmallow|s4'], '4fa6496cd684')
        self.assertEqual(sp['OpenDrawer/left|s2'], '08f1c9df8207')
        for f in self.spec['folds']:
            for r in f['fit'] + f['eval']:
                want = {'n10_19': F.N20_PLAN, 'n20_39': F.N40_PLAN}.get(F.noise_range(r['n']), f['source_plan_n0_9'])
                self.assertEqual(r['plan_id'], want)

    def test_expected_counts_vs_selection(self):
        by = {f['fold_id']: f for f in self.spec['folds']}
        oven = by['OvenRack-out-left_s2_j4']
        self.assertEqual(oven['expected']['eval_success'], 0)       # success-0 target kept
        self.assertEqual(oven['expected']['fit_success'], 34 + 20 + 25 + 25)
        self.assertEqual(by['OpenDrawer-left_s2_j2']['expected']['eval_success'], 6)
        self.assertEqual({f['eval_host'] for f in self.spec['folds'] if f['instruction'].startswith('PPCC')}, {'srv50'})
        self.assertEqual({f['archive_machine'] for f in self.spec['folds'] if f['instruction'].startswith('Dish')}, {'worker1'})

    def test_aliases(self):
        self.assertEqual(F.canonical_machine('srv48'), 'worker1')
        self.assertEqual(F.canonical_machine('worker2'), 'worker2')
        self.assertEqual(F.canonical_machine('srv50'), 'worker2')
        with self.assertRaises(F.FoldError):
            F.canonical_machine('pdk')

    def test_validate_rejects_leak(self):
        f = copy.deepcopy(self.spec['folds'][0])
        f['fit'][0] = dict(f['eval'][0])
        with self.assertRaises(F.FoldError):
            F.validate_fold(f)
        g = copy.deepcopy(self.spec['folds'][0])
        g['eval'][0]['plan_id'] = 'ffffffffffff'
        with self.assertRaises(F.FoldError):
            F.validate_fold(g)

    def test_archive_index_gate(self):
        # synthetic archive index matching selection counts must pass; one flipped success must fail
        sel = {t['instruction']: t for t in __import__('json').loads((F.R / F.SEL).read_text())['targets']}
        idx = {}
        for (plan, m, instr, s, j, n), r in self.sources.items():
            succ = int(n < sel[instr]['counts'][j][0])
            idx[(plan, m, instr, s, j, n)] = {'env_seed': r['env_seed'], 'inference_seed': r['inference_seed'], 'success': succ,
                                              'steps': 100, 'sig': 'x', 'pkl_sha256': 'y', 'has_pkl': 1}
        spec, _ = F.build(F.R, idx)
        self.assertTrue(spec['production_ready'])
        k = next(iter(idx)); idx[k] = dict(idx[k], success=1 - idx[k]['success'])
        with self.assertRaises(F.FoldError):
            F.build(F.R, idx)


if __name__ == '__main__':
    unittest.main()

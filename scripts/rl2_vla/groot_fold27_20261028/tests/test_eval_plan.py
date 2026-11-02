"""CPU tests for eval/plan_fold_eval.py on a synthetic archive-confirmed spec."""
import json, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'eval')]
import folds as F            # noqa: E402
import plan_fold_eval as P   # noqa: E402


def confirmed_spec():
    _, sources = F.build(F.R)
    sel = {t['instruction']: t for t in json.loads((F.R / F.SEL).read_text())['targets']}
    idx = {k: {'env_seed': r['env_seed'], 'inference_seed': r['inference_seed'], 'success': int(k[5] < sel[k[2]]['counts'][k[4]][0]),
               'steps': '', 'n_inferences': '', 'sig': 's', 'pkl_sha256': 'p', 'has_pkl': 1} for k, r in sources.items()}
    spec, _ = F.build(F.R, idx)
    return spec


class EvalPlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = confirmed_spec()

    def test_manifests(self):
        ms = [P.fold_manifest(f, P.DEFAULT_ARMS) for f in self.spec['folds']]
        self.assertEqual(len(ms), 27)
        for m in ms:
            self.assertEqual(len(m['test_rows']), 40)
            self.assertEqual(sorted(r['n'] for r in m['test_rows']), list(range(40)))
            self.assertEqual({r['j'] for r in m['test_rows']}, {m['target_j']})
            self.assertEqual(m['new_episodes'], 40 * 4 + 3)
        hosts = {m['fold_id']: m['host'] for m in ms}
        self.assertEqual(hosts['OpenDrawer-left_s2_j2'], 'kanu')
        self.assertEqual(hosts['OvenRack-out-left_s2_j4'], 'srv48')
        self.assertEqual(hosts['PPCC-jug_s1_j1'], 'srv50')
        sch = P.schedule(ms)
        self.assertEqual(sch['kanu']['max_parallel_serves'], 6)
        self.assertEqual(sch['srv48']['max_parallel_serves'], 6)
        self.assertEqual(sum(len(v['folds']) for v in sch.values()), 27)

    def test_unconfirmed_rows_rejected(self):
        spec, _ = F.build(F.R)       # draft: n0-9 derived seeds
        with self.assertRaises(P.PlanError):
            P.fold_manifest(spec['folds'][0], P.DEFAULT_ARMS)

    def test_main_requires_base(self):
        with tempfile.TemporaryDirectory() as d:
            sp = Path(d) / 'spec.json'; sp.write_text(json.dumps(self.spec))
            sys.argv = ['x', '--spec', str(sp), '--out', str(Path(d) / 'o'), '--arms', 'mil_qam65']
            with self.assertRaises(P.PlanError):
                P.main()
            sys.argv = ['x', '--spec', str(sp), '--out', str(Path(d) / 'o')]
            P.main()
            self.assertEqual(len(list((Path(d) / 'o').glob('*.manifest.json'))), 27)


if __name__ == '__main__':
    unittest.main()

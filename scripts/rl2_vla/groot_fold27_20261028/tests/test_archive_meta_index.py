"""CPU tests for archive_meta_index.py (meta.json primary, optional stale-index cross-check)."""
import json, sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import archive_meta_index as A

COORD = ('4fa6496cd684', 'worker2', 'PPCC/marshmallow', 4, 0, 3)
SHA = 'ab' * 32


def meta(**kw):
    m = {'sig': SHA[:16], 'pkl_sha256': SHA, 'success': 1, 'env_seed': 100245, 'inference_seed': 1300003,
         'plan_id': '4fa6496cd684', 'machine': 'worker2', 'grid_instruction': 'PPCC/marshmallow', 'scene_idx': 4,
         'jitter_idx': 0, 'noise_idx': 3, 'capture_layers': [0, 2, 4, 8, 10, 12, 15], 'record_shape': [7, 4, 49, 1536]}
    m.update(kw)
    return m


def irow(rel, **kw):
    r = {'rel_path': rel, 'armsig': rel.rsplit('/', 1)[-1], 'sig': SHA[:16], 'pkl_sha256': SHA, 'success': '1',
         'steps': '', 'n_inferences': '63', 'machine': 'worker2'}
    r.update(kw)
    return r


class ArchiveIndexTest(unittest.TestCase):
    def test_meta_only_row(self):
        r = A.from_meta(COORD, meta(), 1)
        self.assertEqual(r[6:9], [100245, 1300003, 1])
        self.assertEqual(r[9], '')                 # steps never estimated
        self.assertEqual(r[12], SHA)
        self.assertEqual(r[15], 'absent')

    def test_meta_contract_rejections(self):
        for bad in (dict(jitter_idx=1), dict(plan_id='08f1c9df8207'), dict(machine='kanu:gpu6'),
                    dict(capture_layers=[12]), dict(record_shape=[7, 4, 49, 1024]), dict(sig='ffff')):
            with self.assertRaises(A.IndexError_, msg=str(bad)):
                A.from_meta(COORD, meta(**bad), 1)
        with self.assertRaises(A.IndexError_):
            A.from_meta(COORD, meta(pkl_sha256='', sig=''), 1)      # pkl present but no sha
        self.assertEqual(A.from_meta(COORD, meta(machine='srv50:gpu1'), 1)[1], 'worker2')

    def test_pkl_hash_gate(self):
        self.assertEqual(A.from_meta(COORD, meta(), 1, pkl_sha=SHA)[14], 1)
        with self.assertRaises(A.IndexError_):
            A.from_meta(COORD, meta(), 1, pkl_sha='cd' * 32)

    def test_index_crosscheck(self):
        ix = irow('4fa6496cd684/worker2/PPCC/marshmallow/s4/j0/n3/base')
        r = A.from_meta(COORD, meta(), 1, ix)
        self.assertEqual((r[10], r[15]), (63, 'agree'))
        with self.assertRaises(A.IndexError_):
            A.from_meta(COORD, meta(), 1, dict(ix, success='0'))

    def test_rel_path_parse(self):
        c, arm = A.parse_rel('2f26fb046fc4/srv50/PPCC/bread/s4/j1/n25/base')
        self.assertEqual(c, ('2f26fb046fc4', 'worker2', 'PPCC/bread', 4, 1, 25))

    def test_index_filter_skips_unrelated_rows(self):
        wanted = {('4fa6496cd684', 'worker2', 'PPCC/marshmallow')}
        rows = [{'rel_path': '979d4833a7db/dongkyu-MS-7D43/PPCC/marshmallow/s0/n0/base'},
                {'rel_path': '4fa6496cd684/worker2/PPCC/bread/s4/j0/n0/base'},
                irow('4fa6496cd684/worker2/PPCC/marshmallow/s4/j0/n3/base')]
        self.assertEqual(list(A.load_index(rows, wanted)), [COORD])
        with self.assertRaises(A.IndexError_):
            A.load_index(rows + [{'rel_path': '4fa6496cd684/worker2/PPCC/marshmallow/s4/n1/base'}], wanted)
        with self.assertRaises(A.IndexError_):
            A.load_index(rows)

    def test_main_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            base = d / 'store/groot/n15/grid/4fa6496cd684/worker2/PPCC/marshmallow/s4/j0/n3/base'
            base.mkdir(parents=True)
            (base / 'rollout.pkl').write_bytes(b'x')
            import hashlib
            sha = hashlib.sha256(b'x').hexdigest()
            (base / 'meta.json').write_text(json.dumps(meta(pkl_sha256=sha, sig=sha[:16])))
            (d / 'coords.tsv').write_text('plan_id\tmachine\tinstruction\tscene\tjitter\tnoise\n'
                                          '4fa6496cd684\tsrv50\tPPCC/marshmallow\t4\t0\t3\n')
            sys.argv = ['x', '--coords', str(d / 'coords.tsv'), '--store', str(d / 'store'), '--out', str(d / 'o.tsv'), '--hash-pkl']
            A.main()
            lines = (d / 'o.tsv').read_text().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertIn('\t1\tabsent', lines[1])


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3
"""watch_target9_n40 순수 함수 (ssh·GPU·docker 없음, stdlib unittest).

Run: python3 scripts/collect/tests/test_watch_target9_n40.py
"""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import watch_target9_n40 as w  # noqa: E402

N40 = w.N40_DIR
N20 = w.N20_DIR


class FrozenConfigTest(unittest.TestCase):
    def test_requested_counts(self):
        rows = w.read_tsv(N40 / 'requests.tsv')
        self.assertEqual(len(w.requested_n40(rows, 'srv48')), 300)
        self.assertEqual(len(w.requested_n40(rows, 'srv50')), 400)
        bf = w.requested_backfill(w.read_tsv(N40 / 'backfill_development.tsv'), 'worker1')
        self.assertEqual(len(bf), 108)
        self.assertEqual(w.requested_backfill(w.read_tsv(N40 / 'backfill_development.tsv'), 'worker2'), set())

    def test_requested_inside_contract_and_harness_universe(self):
        for machine in ('srv48', 'srv50'):
            c = json.loads((N40 / f'{machine}_contract.json').read_text())
            req = w.requested_n40(w.read_tsv(N40 / 'requests.tsv'), machine)
            allowed = w.allowed_cells(c, 20, 40)
            self.assertEqual(req, allowed)
            skip = w.read_lines(N40 / f'{machine}_skip.txt')
            self.assertFalse(skip & allowed, 'frozen skip must not hide requested cells')
        c20 = json.loads((N20 / 'srv48_contract.json').read_text())
        bf = w.requested_backfill(w.read_tsv(N40 / 'backfill_development.tsv'), 'worker1')
        self.assertTrue(bf <= w.allowed_cells(c20, 10, 20))
        self.assertFalse(bf & w.read_lines(N20 / 'srv48_skip.txt'))

    def test_bad_rows_rejected(self):
        rows = [{'machine': 'srv48', 'plan_id': 'zzz', 'instruction': 'A', 'scene': '0', 'jitter': '0', 'noise': '20'}]
        with self.assertRaises(ValueError):
            w.requested_n40(rows, 'srv48')
        bad = [{'machine': 'worker1', 'source_plan': w.N20_PLAN_ID, 'instruction': 'A', 'scene': '0',
                'jitter': '0', 'noise': '25'}]
        with self.assertRaises(ValueError):
            w.requested_backfill(bad, 'worker1')

    def test_runtime_contract_paths(self):
        common = '/home/junhyeong/pkt_ws/temporal_vla'
        c40 = json.loads((N40 / 'srv50_contract.json').read_text())
        rt = w.runtime_contract(c40, str(N40), common, common + '/outputs/collect/grid_n20_srv50_20261014')
        self.assertEqual(Path(rt['parent_plan']), N20 / 'collection_plan.json')
        self.assertEqual(rt['parent_grid_root'], common + '/outputs/collect/grid_n20_srv50_20261014')
        self.assertEqual(c40['parent_plan'], '../n15_target9_n20_20261013/collection_plan.json', 'frozen untouched')
        c20 = json.loads((N20 / 'srv48_contract.json').read_text())
        rt = w.runtime_contract(c20, str(N20), common, 'x')
        self.assertTrue(rt['parent_plan'].startswith(common + '/.claude/worktrees/collect-topup/configs/'))


class InventoryTest(unittest.TestCase):
    def test_cell_key_from_rel(self):
        self.assertEqual(w.cell_key_from_rel('DishwasherRack/out-left/s3/j0/n11/base'),
                         'DishwasherRack/out-left|s3|j0|n11')
        self.assertEqual(w.cell_key_from_rel('PPCC/jug/s1/j4/n39/arm_x/'), 'PPCC/jug|s1|j4|n39')
        self.assertIsNone(w.cell_key_from_rel('PPCC/jug/s1/n39/base'))
        self.assertIsNone(w.cell_key_from_rel('ep_meta/x.json'))

    def test_phase_counts(self):
        req = {'a', 'b', 'c', 'd', 'e'}
        c = w.phase_counts(req, archive_complete={'a', 'z'}, archive_incomplete={'b'},
                           shipped={'a', 'c'}, local_meta={'d'})
        self.assertEqual(c['archive_complete'], 1)
        self.assertEqual(c['archive_incomplete'], 1)
        self.assertEqual(c['shipped_not_archived'], 1)       # c: shipped but archive missing → not done
        self.assertEqual(c['local_pending_ship'], 1)
        self.assertEqual(c['remaining_to_launch'], 2)        # b (incomplete) + e
        self.assertFalse(c['complete'])
        self.assertTrue(w.phase_counts({'a'}, {'a'}, set(), set(), set())['complete'])

    def test_done_list_never_includes_remaining(self):
        req = {'x|s0|j0|n20', 'x|s0|j0|n21', 'x|s0|j0|n22'}
        allowed = req | {'x|s0|j1|n20'}
        done = w.build_done_list({'x|s1|j0|n0'}, {'x|s0|j0|n20'}, {'x|s0|j0|n21'}, allowed, req)
        self.assertEqual(set(done), {'x|s1|j0|n0', 'x|s0|j0|n20', 'x|s0|j0|n21', 'x|s0|j1|n20'})
        self.assertNotIn('x|s0|j0|n22', done)
        self.assertEqual(done, sorted(done))

    def test_local_meta_keys(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            cell = Path(d) / 'P' / 'worker1' / 'PPCC/jug/s1/j0/n20/base'
            cell.mkdir(parents=True)
            (cell / 'meta.json').write_text('{}')
            (Path(d) / 'P/worker2/PPCC/jug/s1/j0/n21/base').mkdir(parents=True)
            (Path(d) / 'P/worker2/PPCC/jug/s1/j0/n21/base/meta.json').write_text('{}')
            self.assertEqual(w.local_meta_keys(d, 'P', 'worker1'), {'PPCC/jug|s1|j0|n20'})
            self.assertEqual(w.local_meta_keys(d, 'Q', 'worker1'), set())


class GpuProcTest(unittest.TestCase):
    GPUS = '0, GPU-a, 40000\n1, GPU-b, 3\n2, GPU-c, 0\n3, GPU-d, [N/A]\n'

    def test_choose_gpu(self):
        gpus = w.parse_gpu_csv(self.GPUS)
        apps = w.parse_apps_csv('GPU-a, 111\nGPU-b, 222\n')
        self.assertEqual(w.choose_gpu(gpus, apps), 2)       # b has a process despite low memory
        self.assertIsNone(w.choose_gpu(gpus, w.parse_apps_csv('GPU-c, 9\nGPU-a, 1\nGPU-b, 2')))
        self.assertEqual(w.choose_gpu(gpus, w.parse_apps_csv('')), 1)
        hot = w.parse_gpu_csv('0, GPU-a, 100\n1, GPU-b, 150\n')
        self.assertIsNone(w.choose_gpu(hot, []))             # <100MiB strict
        self.assertIsNone(w.choose_gpu(w.parse_gpu_csv('3, GPU-d, [N/A]'), []))

    def test_matching_procs(self):
        procs = [(1, ['bash', '/r/scripts/safe/x/collect_grid.sh']),
                 (2, ['python', '/temporal_vla/x/http_feature_collect.py', '--task', 'T']),
                 (3, ['python3', 'watch_target9_n40.py', '--machine', 'srv48']),
                 (4, ['grep', 'collect_grid.sh.bak']),
                 (5, ['bash', 'collect_grid.sh'])]
        self.assertEqual(w.matching_procs(procs, exclude={5}), [1, 2])

    def test_shipper_root(self):
        argv = ['bash', 'scripts/safe/groot_n15/robocasa/collect/ship_to_archive.sh']
        hit = w.shipper_root(argv, {'GRID_ROOT': 'outputs/collect/grid_n20_srv48_20261014'}, '/c')
        self.assertEqual(hit, ('/c/outputs/collect/grid_n20_srv48_20261014',
                               '/c/scripts/safe/groot_n15/robocasa/collect/ship_to_archive.sh'))
        self.assertIsNone(w.shipper_root(argv, {}, '/c'))
        self.assertIsNone(w.shipper_root(['bash', 'collect_grid.sh'], {'GRID_ROOT': '/g'}, '/c'))

    def test_container_helpers(self):
        info = {'Mounts': [{'Destination': '/cache', 'Source': '/h/.cache/temporal_vla'},
                           {'Destination': '/temporal_vla', 'Source': '/h/pkt_ws/temporal_vla'}],
                'HostConfig': {'DeviceRequests': [{'DeviceIDs': ['3']}]}}
        self.assertEqual(w.mount_source(info, '/cache'), '/h/.cache/temporal_vla')
        self.assertIsNone(w.mount_source(info, '/x'))
        self.assertEqual(w.container_gpu_ids(info), ['3'])
        self.assertEqual(w.container_gpu_ids({'HostConfig': {}}), [])


class HarnessReceiptTest(unittest.TestCase):
    S = w.HARNESS_ENV['HARNESS_SESSION']
    BASE = {'machine': 'srv48', 'gpus': [3], 'serves': 6, 'session': S, 'kind': 'collect',
            'launcher_host': 'worker1.lab', 'launcher_pid': 1111, 'consumer_pid': 2222}

    def judge(self, alive=(), idle=(3,), host='worker1', **over):
        item = dict(self.BASE, **over)
        return w.receipt_release_eligible(item, 'srv48', self.S, host, lambda p: p in alive, set(idle))

    def test_stale_receipt_eligible(self):
        self.assertEqual(self.judge(), (True, 'stale'))
        self.assertTrue(self.judge(host='worker1.example.org', gpus=[3, 4], idle=(3, 4, 5))[0])

    def test_conservative_rejections(self):
        cases = [dict(machine='srv50'), dict(session='other'), dict(kind='eval'),
                 dict(launcher_host='worker2'), dict(launcher_host=None),
                 dict(launcher_pid=None), dict(consumer_pid=None), dict(consumer_pid='2222'),
                 dict(launcher_pid=0), dict(consumer_pid=-5), dict(consumer_pid=True),
                 dict(gpus=[]), dict(gpus=None), dict(gpus=['3']), dict(gpus=[3, 4])]
        for over in cases:
            self.assertFalse(self.judge(**over)[0], over)
        item = dict(self.BASE)
        del item['consumer_pid']                                       # verify never ran
        self.assertFalse(w.receipt_release_eligible(item, 'srv48', self.S, 'worker1', lambda p: False, {3})[0])
        self.assertFalse(self.judge(alive=(1111,))[0])                  # harness run still alive
        self.assertFalse(self.judge(alive=(2222,))[0])                  # consumer still alive
        self.assertFalse(self.judge(idle=())[0])                        # detached servers still on GPU
        self.assertFalse(self.judge(host='worker2')[0])

    def test_idle_gpu_indices(self):
        gpus = w.parse_gpu_csv('3, GPU-d, 0\n4, GPU-e, 99\n5, GPU-f, 100\n6, GPU-g, [N/A]\n')
        self.assertEqual(w.idle_gpu_indices(gpus, w.parse_apps_csv('GPU-e, 7')), {3})
        self.assertEqual(w.idle_gpu_indices(gpus, []), {3, 4})

    def test_own_receipts_filter(self):
        state = {'a': dict(self.BASE), 'b': dict(self.BASE, machine='srv50'),
                 'c': dict(self.BASE, session='x'), 'd': dict(self.BASE, kind='eval'), 'e': 'junk'}
        self.assertEqual(set(w.own_receipts(state, 'srv48', self.S)), {'a', 'd'})

    def test_admission_classifier(self):
        blocked = 'x\n[harness] BLOCKED: srv48 GPU3: 6+6 exceeds slot limit\n'
        self.assertTrue(w.harness_admission_blocked(2, blocked))
        self.assertFalse(w.harness_admission_blocked(1, blocked))       # genuine collector error
        self.assertFalse(w.harness_admission_blocked(2, 'Traceback ... RuntimeError: CUDA OOM\n'))
        self.assertFalse(w.harness_admission_blocked(0, blocked))
        self.assertFalse(w.harness_admission_blocked(2, None))

    def test_read_attempt_log_includes_rotated(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'a.log'
            self.assertEqual(w.read_attempt_log(p), '')
            Path(str(p) + '.1').write_text('[harness] BLOCKED: x\n')
            p.write_text('tail\n')
            self.assertTrue(w.harness_admission_blocked(2, w.read_attempt_log(p)))


if __name__ == '__main__':
    unittest.main()

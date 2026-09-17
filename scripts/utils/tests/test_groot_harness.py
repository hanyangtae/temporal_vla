"""No live GPU, SSH, production ledger, or model launches in this suite."""
import argparse
import concurrent.futures
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import sys
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / 'groot_harness.py'
spec = importlib.util.spec_from_file_location('groot_harness', MODULE)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def reserve(root, machine, gpu, session, serves=1):
    return h.transact(Path(root), dict(action='reserve', machine=machine, gpus=[gpu],
                                      serves=serves, session=session, kind='eval'))


def concurrent_reserve(args):
    try:
        reserve(*args)
        return True
    except ValueError:
        return False


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_a100_limit_across_sessions_and_independent_machines(self):
        reserve(self.root, 'srv48', 1, 'a', 6)
        with self.assertRaisesRegex(ValueError, 'aggregate GPU'):
            reserve(self.root, 'srv48', 2, 'b')
        reserve(self.root, 'srv50', 2, 'b', 6)

    def test_kanu_three_across_sessions(self):
        for g in range(3):
            reserve(self.root, 'kanu', g, str(g), 2)
        with self.assertRaisesRegex(ValueError, 'aggregate GPU'):
            reserve(self.root, 'kanu', 3, 'extra')

    def test_aggregate_slots_and_foreign_session(self):
        reserve(self.root, 'srv50', 1, 'a', 4)
        reserve(self.root, 'srv50', 1, 'a', 2)
        with self.assertRaisesRegex(ValueError, 'slot limit'):
            reserve(self.root, 'srv50', 1, 'a')
        with self.assertRaisesRegex(ValueError, 'another session'):
            reserve(self.root, 'srv50', 1, 'b')

    def test_race_between_different_gpus(self):
        args = [(str(self.root), 'srv48', i, str(i), 6) for i in range(4)]
        with concurrent.futures.ProcessPoolExecutor(4) as pool:
            self.assertEqual(sum(pool.map(concurrent_reserve, args)), 1)

    def test_existing_legacy_lease_counts(self):
        p = self.root / 'srv48_gpu2'
        p.mkdir()
        (p / 'meta').write_text(f'session=old\npid={os.getpid()}\n')
        with self.assertRaisesRegex(ValueError, 'aggregate GPU'):
            reserve(self.root, 'srv48', 1, 'new')

    def test_non_groot_legacy_not_counted_but_exclusive(self):
        p = self.root / 'srv48_gpu2'
        p.mkdir()
        (p / 'meta').write_text(f'session=other\npid={os.getpid()}\nmodel=other\n')
        reserve(self.root, 'srv48', 1, 'new')
        with self.assertRaises(ValueError):
            reserve(self.root, 'srv48', 2, 'new')

    def test_legacy_claim_cannot_bypass_harness(self):
        reserve(self.root, 'srv48', 1, 'a')
        with self.assertRaises(ValueError):
            h.legacy_check(self.root, 'srv48', 2, 'b', 'groot')
        with self.assertRaises(ValueError):
            h.legacy_check(self.root, 'srv48', 1, 'a', 'groot')

    def test_invalid_resources(self):
        for m, gs, n in [('kanu', [1], 3), ('srv48', [0, 1], 1), ('kanu', [1, 1], 1), ('kanu', [-1], 1)]:
            with self.assertRaises(ValueError):
                h.resources(m, gs, n)

    def test_release_and_single_consumer(self):
        receipt = reserve(self.root, 'kanu', 1, 'a')['receipt']
        req = dict(action='verify', receipt=receipt, session='a', machine='kanu',
                   gpus=[1], serves=1, kind='eval', consumer_pid=123)
        h.transact(self.root, req)
        with self.assertRaisesRegex(ValueError, 'already consumed'):
            h.transact(self.root, dict(req, consumer_pid=124))
        with self.assertRaisesRegex(ValueError, 'owner mismatch'):
            h.transact(self.root, dict(action='release', receipt=receipt, session='b'))
        h.transact(self.root, dict(action='release', receipt=receipt, session='a'))
        reserve(self.root, 'kanu', 1, 'b')

    def test_remote_no_private_ledger_fallback(self):
        with patch.object(h.socket, 'gethostname', return_value='worker1'), patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'remote launch requires'):
                h.broker({'action': 'status'})

    def test_run_smoke_child_exit_and_reservation_release(self):
        args = argparse.Namespace(machine='kanu', gpus='5', serves=2, session='smoke', kind='eval',
                                  command=[sys.executable, '-c',
                                           "import os; assert os.environ['SERVES_PER_GPU']=='2'; assert os.environ['HARNESS_RECEIPT']; raise SystemExit(7)"])
        def local_broker(request):
            return h.transact(self.root, request)
        with patch.object(h, 'broker', side_effect=local_broker), \
                patch.object(h, 'check_gpu_processes'), \
                patch.object(h.socket, 'gethostname', return_value='kanu'), \
                patch.object(h.signal, 'signal'):
            self.assertEqual(h.run(args), 7)
        self.assertEqual(h.transact(self.root, {'action': 'status'}), {})

    def test_run_retains_reservation_on_residual_server(self):
        args = argparse.Namespace(machine='kanu', gpus='5', serves=2, session='smoke', kind='eval',
                                  command=[sys.executable, '-c', 'pass'])
        with patch.object(h, 'broker', side_effect=lambda r: h.transact(self.root, r)), \
                patch.object(h, 'check_gpu_processes', side_effect=[None, None, ValueError('residual GPU PID')]), \
                patch.object(h.socket, 'gethostname', return_value='kanu'), \
                patch.object(h.signal, 'signal'):
            self.assertEqual(h.run(args), 0)
        self.assertEqual(len(h.transact(self.root, {'action': 'status'})), 1)

    def test_host_identity_mismatch_blocks_before_probe(self):
        args = argparse.Namespace(machine='srv48', gpus='0', serves=6, session='a', kind='eval', command=['true'])
        with patch.object(h.socket, 'gethostname', return_value='kanu'), \
                patch.object(h, 'check_gpu_processes') as probe:
            with self.assertRaisesRegex(ValueError, 'actual launcher host'):
                h.run(args)
            probe.assert_not_called()

    def test_nvml_failure_and_existing_process_block(self):
        for result in [subprocess.CompletedProcess([], 1, '', 'NVML error'),
                       subprocess.CompletedProcess([], 0, '123\n', '')]:
            with patch.object(h.subprocess, 'run', return_value=result), self.assertRaises(ValueError):
                h.check_gpu_processes([0])


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.old = json.loads((h.REPO / 'configs/collect/n15_grid_v6_scene_jitter/collection_plan.json').read_text())
        self.new = copy.deepcopy(self.old)
        self.key = 'PPCC/bread'
        self.new['instructions'][self.key].append(99999)
        self.new['scenes'][self.key].append(dict(layout=2, style=2))
        self.new['jitters'][self.key].append(copy.deepcopy(self.old['jitters'][self.key][0]))
        self.parent = self.root / 'parent.json'
        self.parent.write_text(json.dumps(self.old))
        self.plan = self.root / 'plan.json'
        self.contract = self.root / 'contract.json'
        self.done = self.root / 'done.txt'
        self.done.write_text('\n'.join(f'{self.key}|s{s}|j{j}|n{n}' for s in range(3) for j in range(5) for n in range(10)))
        self.spec = dict(kind='collect', model='groot', machine='kanu', parent_plan='parent.json',
                         parent_plan_sha256=hashlib.sha256(self.parent.read_bytes()).hexdigest(),
                         parent_grid_root=str(self.root / 'original'), allowed_instructions=[self.key],
                         excluded_instructions=['PPCC/apple', 'CoffeeSetupMug'], noise_count=10, jitter_count=5)

    def check(self, **kwargs):
        self.plan.write_text(json.dumps(self.new))
        self.spec['plan_sha256'] = hashlib.sha256(self.plan.read_bytes()).hexdigest()
        self.contract.write_text(json.dumps(self.spec))
        args = dict(contract_path=self.contract, plan_path=self.plan, instructions=self.key,
                    noise_limit=10, done_list=self.done, grid_root=self.root / 'new')
        args.update(kwargs)
        return h.collection_check(**args)

    def test_superset(self):
        self.assertEqual(self.check()['new_episodes'], 50)

    def test_existing_scene_changed(self):
        self.new['scenes'][self.key][0]['style'] = 99
        with self.assertRaisesRegex(ValueError, 'existing coordinates'):
            self.check()

    def test_excluded_instruction(self):
        with self.assertRaises(ValueError):
            self.check(instructions='CoffeeSetupMug')

    def test_missing_old_skip(self):
        self.done.write_text('')
        with self.assertRaisesRegex(ValueError, 'DONE_LIST'):
            self.check()

    def test_parent_root_reuse(self):
        with self.assertRaisesRegex(ValueError, 'separate grid root'):
            self.check(grid_root=self.root / 'original')

    def test_noise_and_jitter_counts(self):
        with self.assertRaises(ValueError):
            self.check(noise_limit=5)
        self.new['jitters'][self.key][-1].pop()
        with self.assertRaisesRegex(ValueError, 'five jitters'):
            self.check()

    def test_environment_and_parent_hash(self):
        self.new['extra']['env_kwargs'] = {}
        with self.assertRaisesRegex(ValueError, 'environment changed'):
            self.check()
        self.spec['parent_plan_sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'digest mismatch'):
            self.check()


if __name__ == '__main__':
    unittest.main()

import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    from . import parallel_eval as module
except ImportError:
    import parallel_eval as module


class ParallelTests(unittest.TestCase):
    def test_shards_are_disjoint_and_each_keeps_its_gate(self):
        jobs = []
        for name in ('a', 'b', 'c'):
            jobs.extend([self.task(name, True), self.task(name)])
        shards = [module.select_target(jobs, [name, '0', '1']) for name in ('a', 'b', 'c')]
        self.assertEqual(sum(map(len, shards)), len(jobs))
        self.assertEqual(len({id(j) for shard in shards for j in shard}), len(jobs))
        for shard in shards:
            self.assertEqual(module.next_batch(shard, set(), 2), [shard[0]])
        with self.assertRaisesRegex(ValueError, 'gate'):
            module.select_target(jobs, ['missing', '0', '1'])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def task(self, name, gate=False):
        return dict(pair_id=name, model_arm='L_only', condition='ps_base' if gate else 'ps_reseed',
                    gate=gate, expected=[], result=str(self.root/name/'per_episode.tsv'),
                    log=str(self.root/name/'wrapper.log'),
                    env=dict(SLUGS=name, EVAL_SCENES='0', EVAL_JITTERS='1', PORT_BASE='8866',
                             DETECTOR_CKPT=str(self.root/name/'detector.pt')))

    def test_target_gate_blocks_comparison_but_allows_other_gate(self):
        ga, a, gb, b = self.task('a', True), self.task('a'), self.task('b', True), self.task('b')
        self.assertEqual(module.next_batch([a, ga, b, gb], set(), 2), [ga, gb])
        self.assertEqual(module.next_batch([a, b, gb], {module.target(ga)}, 2), [a, gb])
        self.assertEqual(module.next_batch([b], {module.target(ga)}, 2), [])

    def test_ownership_requires_exact_port_and_artifact_not_only_process_name(self):
        task = self.task('a')
        argv = ['python', 'scripts/serve/lerobot.py', '--port', '8866', '--failure-detector',
                '/temporal_vla/a/detector.pt', '--failure-task', 'a']
        self.assertTrue(module.owns_gpu_process(argv, task, self.root))
        for index, value in [(3, '8892'), (5, '/temporal_vla/other/detector.pt'), (7, 'b')]:
            wrong = argv[:]; wrong[index] = value
            self.assertFalse(module.owns_gpu_process(wrong, task, self.root))
        collector = ['python', '/temporal_vla/scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py',
                     '--vla-server', 'http://127.0.0.1:8866', '--output-dir', '/temporal_vla/a/raw_rollouts']
        self.assertTrue(module.owns_gpu_process(collector, task, self.root))
        collector[-1] = '/temporal_vla/other/raw_rollouts'
        self.assertFalse(module.owns_gpu_process(collector, task, self.root))

    def test_two_real_children_overlap_and_results_remain_separate(self):
        # Exercise real subprocesses, polling/cleanup and existing output paths;
        # only GPU/HTTP dependencies are replaced for this CPU test.
        tasks = [self.task('a'), self.task('b')]
        for task in tasks:
            task['command'] = [sys.executable, '-c',
                'import os,time,pathlib,json; p=pathlib.Path(os.environ["OUT"]); '
                'p.write_text(json.dumps({"start":time.time(),"port":os.environ["PORT_BASE"]})); '
                'time.sleep(0.3); p.with_suffix(".done").write_text(str(time.time()))']
            task['env']['OUT'] = str(self.root/(task['pair_id']+'.json'))
        with patch.object(module, 'check_idle'), patch.object(module, 'check_port'), \
             patch.object(module, 'check_owned_gpu') as guard, patch.object(module, 'wait_loaded'), \
             patch.object(module, 'completed', return_value=True):
            module.run_batch(tasks, self.root, 6, [8866, 8867], lambda: None)
        a, b = [json.loads((self.root/(n+'.json')).read_text()) for n in ('a','b')]
        ends = [float((self.root/(n+'.done')).read_text()) for n in ('a','b')]
        self.assertLess(max(a['start'], b['start']), min(ends))
        self.assertEqual([a['port'], b['port']], ['8866','8867'])
        self.assertEqual(len(guard.call_args_list[0].args[1]), 0)
        self.assertEqual(len(guard.call_args_list[1].args[1]), 1)
        self.assertNotIn('ALLOW_BUSY_GPU', tasks[0]['env'])  # immutable original task

    def test_failed_first_child_never_starts_peer(self):
        a, b = self.task('a'), self.task('b')
        a['command'] = [sys.executable, '-c', 'raise SystemExit(4)']
        marker = self.root/'unexpected_peer'
        b['command'] = [sys.executable, '-c', f'from pathlib import Path; Path({str(marker)!r}).touch()']
        # Do not query the real evaluation server's health endpoint in this test.
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        with patch.object(module, 'check_idle'), patch.object(module, 'check_port'), \
             patch.object(module, 'check_owned_gpu'):
            with self.assertRaisesRegex(RuntimeError, 'first lane failed'):
                module.run_batch([a, b], self.root, 6, [port,port+1], lambda: None)
        self.assertFalse(marker.exists())


if __name__ == '__main__':
    unittest.main()

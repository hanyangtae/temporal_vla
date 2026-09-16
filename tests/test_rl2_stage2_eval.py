"""Dependency-free contract tests for the Stage 2 evaluator helpers."""
import ast
from collections import deque
from pathlib import Path
import unittest


SOURCE = Path(__file__).parents[1] / "RL2-VLA/RL2_CoVer_VLA/simpler/run_simpler_eval_with_openpi.py"


def _extract_helpers():
    tree = ast.parse(SOURCE.read_text())
    names = {"_stage2_should_compose", "_stage2_next_action", "_stage2_end_reason"}
    module = ast.Module(
        body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names],
        type_ignores=[],
    )
    namespace = {"deque": deque}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace


class Stage2ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helpers = _extract_helpers()
        cls.source = SOURCE.read_text()

    def test_queue_progression_keeps_tail_and_stops_mid_chunk(self):
        queue = deque(["a", "b", "c", "d"])
        self.assertEqual(self.helpers["_stage2_next_action"](queue), "a")
        self.assertEqual(self.helpers["_stage2_next_action"](queue), "b")
        self.assertEqual(list(queue), ["c", "d"])
        self.assertEqual(self.helpers["_stage2_end_reason"](True, False, 2, 150), "success")
        self.assertEqual(list(queue), ["c", "d"])

    def test_horizon_has_no_missing_tail_requirement(self):
        self.assertEqual(self.helpers["_stage2_end_reason"](False, False, 150, 150), "horizon")
        self.assertEqual(self.helpers["_stage2_end_reason"](False, True, 12, 150), "truncated")

    def test_safe_and_composed_gate(self):
        compose = self.helpers["_stage2_should_compose"]
        self.assertFalse(compose(False, True, 1))
        self.assertFalse(compose(True, False, 1))
        self.assertFalse(compose(True, True, 0))
        self.assertTrue(compose(True, True, 1))
        self.assertIn("hidden_states_np=np.asarray(context, dtype=np.float32)[None, :]", self.source)
        self.assertIn("device=hidden_states_device", self.source)
        self.assertIn("_stage2_should_compose(cfg.use_failure_prediction, safe_trigger, cfg.composed_samples)", self.source)

    def test_output_and_atomic_contract_are_present(self):
        self.assertIn('episode_task{task_idx}_env{env_seed}_policy{cfg.seed}.json', self.source)
        self.assertIn('os.link(tmp_path, path)', self.source)
        self.assertIn('env_queue = deque(list(postprocessed))', self.source)
        self.assertIn('env_queue.popleft', self.source)
        self.assertIn('elapsed_seconds', self.source)
        self.assertIn('cp_band_time_unit', Path(__file__).parents[1].joinpath('RL2-VLA/RL2_CoVer_VLA/simpler/rl2_utils.py').read_text())


if __name__ == "__main__":
    unittest.main()

"""eval_gpu_queue_v2.sh control flow: same three cases as test_eval_gpu_queue_sh, against the v2 script."""
import sys, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_eval_gpu_queue_sh as T   # noqa: E402

V2 = Path(__file__).resolve().parents[1] / 'eval/runtime/eval_gpu_queue_v2.sh'


class ShV2Test(T.ShTest):
    def setUp(self):
        self._sh, T.SH = T.SH, V2

    def tearDown(self):
        T.SH = self._sh

    def test_calls_v2_python(self):
        r, calls = self.run_sh(3, GPU='5')
        self.assertTrue(calls and 'may-start' in calls[0])
        self.assertIn('eval_gpu_queue_v2.py', V2.read_text()); self.assertNotIn('/eval_gpu_queue.py"', V2.read_text())


if __name__ == '__main__':
    unittest.main()

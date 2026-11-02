"""eval_gpu_queue.sh control flow with a stub python (no harness, no GPU): no reservation unless may-start says yes."""
import os, subprocess, tempfile, unittest
from pathlib import Path

SH = Path(__file__).resolve().parents[1] / 'eval/runtime/eval_gpu_queue.sh'


class ShTest(unittest.TestCase):
    def run_sh(self, may_rc, **env):
        t = Path(tempfile.mkdtemp()); log = t / 'calls'
        stub = t / 'py'
        stub.write_text(f'#!/bin/bash\necho "$*" >> {log}\ncase "$*" in *may-start*) exit {may_rc};; *wait-resources*) exit 0;; esac\nexit 0\n')
        stub.chmod(0o755)
        e = dict(os.environ, EVAL_GPU_PY=str(stub), HARNESS_MACHINE='kanu', HARNESS_SESSION='fold27-v2-eval', FOLLOW='0', **env)
        r = subprocess.run(['bash', str(SH)], env=e, capture_output=True, text=True, timeout=60)
        return r, (log.read_text().splitlines() if log.exists() else [])

    def test_no_reservation_when_may_start_says_no(self):
        r, calls = self.run_sh(3, GPU='6')
        self.assertEqual(r.returncode, 0)
        self.assertEqual(len(calls), 1); self.assertIn('--pool 5,6,7 may-start --host kanu --gpu 6 --serves 2', calls[0])
        self.assertFalse(any('wait-resources' in c for c in calls))

    def test_gpu_outside_pool_refused(self):
        r, calls = self.run_sh(0, GPU='2')
        self.assertEqual(r.returncode, 2); self.assertEqual(calls, [])

    def test_scan_error_propagates(self):
        r, calls = self.run_sh(5, GPU='5')
        self.assertEqual(r.returncode, 5); self.assertFalse(any('wait-resources' in c for c in calls))


if __name__ == '__main__':
    unittest.main()

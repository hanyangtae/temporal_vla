"""Stdlib integration checks for durable queue failure and resume behavior."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


QUEUE = Path(__file__).with_name('run_queue.py')


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / 'jobs.json'
        self.state = self.root / 'state'

    def python(self, code):
        return [sys.executable, '-c', code]

    def job(self, **changes):
        job = {'id': 'fit', 'cwd': str(self.root),
               'command': self.python("from pathlib import Path; Path('result').write_text('ok')"),
               'outputs': ['result']}
        job.update(changes)
        return job

    def write(self, jobs):
        self.config.write_text(json.dumps({'jobs': jobs}))

    def command(self, once=True):
        command = [sys.executable, str(QUEUE), '--config', str(self.config),
                   '--state-dir', str(self.state), '--poll-seconds', '0.05']
        return command + (['--once'] if once else [])

    def run_queue(self):
        return subprocess.run(self.command(), capture_output=True, text=True, timeout=10)

    def status(self, key='fit'):
        return json.loads((self.state / 'status.json').read_text())['jobs'][key]

    def test_failed_readiness_never_executes(self):
        self.write([self.job(readiness=[{'argv': self.python('raise SystemExit(4)'), 'timeout_s': 2}])])
        result = self.run_queue()
        self.assertEqual(result.returncode, 75, result.stderr)
        self.assertFalse((self.root / 'result').exists())
        self.assertEqual(self.status()['status'], 'PENDING')
        self.assertIn('exit 4', self.status()['reason'])

    def test_readiness_timeout_stays_pending(self):
        self.write([self.job(readiness=[{'argv': self.python('import time; time.sleep(10)'), 'timeout_s': 0.05}])])
        result = self.run_queue()
        self.assertEqual(result.returncode, 75, result.stderr)
        self.assertFalse((self.root / 'result').exists())
        self.assertEqual(self.status()['status'], 'PENDING')
        self.assertIn('timed out', self.status()['reason'])

    def test_missing_sentinel_prevents_completion_and_retry(self):
        self.write([self.job(sentinel={'path': 'done', 'text': 'FIT_DONE'})])
        first = self.run_queue()
        self.assertEqual(first.returncode, 1, first.stderr)
        self.assertEqual(self.status()['status'], 'FAILED')
        self.assertIn('missing sentinel', self.status()['reason'])
        (self.root / 'result').unlink()
        second = self.run_queue()
        self.assertEqual(second.returncode, 1)
        self.assertFalse((self.root / 'result').exists())

    def test_config_change_refuses_resume(self):
        self.write([self.job()])
        self.assertEqual(self.run_queue().returncode, 0)
        old_status = (self.state / 'status.json').read_bytes()
        self.write([self.job(command=self.python("raise RuntimeError('must not run')"))])
        result = self.run_queue()
        self.assertEqual(result.returncode, 2)
        self.assertIn('SHA mismatch', result.stderr)
        self.assertEqual((self.state / 'status.json').read_bytes(), old_status)

    def test_missing_output_stops_downstream(self):
        self.write([self.job(command=self.python('pass')),
                    self.job(id='eval', outputs=['second'],
                             command=self.python("from pathlib import Path; Path('second').touch()"))])
        self.assertEqual(self.run_queue().returncode, 1)
        self.assertEqual(self.status()['status'], 'FAILED')
        self.assertEqual(self.status('eval')['status'], 'PENDING')
        self.assertFalse((self.root / 'second').exists())

    def test_exact_sentinel_and_completed_resume(self):
        code = "from pathlib import Path; p=Path('result'); p.write_text(p.read_text()+'x' if p.exists() else 'x'); Path('done').write_text('FIT_DONE\\n')"
        self.write([self.job(command=self.python(code), sentinel={'path': 'done', 'text': 'FIT_DONE'})])
        self.assertEqual(self.run_queue().returncode, 0)
        self.assertEqual(self.run_queue().returncode, 0)
        self.assertEqual((self.root / 'result').read_text(), 'x')
        self.assertEqual(self.status()['status'], 'COMPLETED')
        (self.root / 'done').write_text('NOT_FIT_DONE\n')
        self.assertEqual(self.run_queue().returncode, 1)
        self.assertEqual(self.status()['status'], 'FAILED')

    def wait_state(self, state):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                if self.status()['status'] == state:
                    return
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            time.sleep(0.025)
        self.fail(f'queue never reached {state}')

    def test_singleton_and_signal_during_readiness(self):
        self.write([self.job(readiness=[{'argv': self.python('raise SystemExit(1)'), 'timeout_s': 1}])])
        first = subprocess.Popen(self.command(once=False), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            self.wait_state('PENDING')
            second = self.run_queue()
            self.assertEqual(second.returncode, 2)
            self.assertIn('another queue', second.stderr)
            first.terminate()
            self.assertEqual(first.wait(timeout=5), 143)
            self.assertFalse((self.root / 'result').exists())
        finally:
            if first.poll() is None:
                first.kill()
                first.wait()
            first.stderr.close()

    def test_signal_stops_own_running_command(self):
        code = "import time; from pathlib import Path; Path('started').touch(); time.sleep(20); Path('result').touch()"
        self.write([self.job(command=self.python(code))])
        process = subprocess.Popen(self.command(), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            self.wait_state('RUNNING')
            process.terminate()
            self.assertEqual(process.wait(timeout=8), 143)
            self.assertEqual(self.status()['status'], 'FAILED')
            self.assertFalse((self.root / 'result').exists())
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stderr.close()


if __name__ == '__main__':
    unittest.main()
